use std::collections::{HashMap, HashSet};
use std::fmt;

use oxide_ml_core::{BufferId, DType, Dim, LifetimePoint, MemoryPlan, Shape, SymbolId, ValueId};
use oxide_ml_graph::{CanonicalGraph, CanonicalShape, CanonicalValueRef, Graph, GraphHashError};

/// Structural version of `MemoryPlan` identity.
pub const MEMORY_PLAN_VERSION: u32 = 1;

/// Version of the canonical plan representation.
pub const CANONICAL_PLAN_VERSION: u32 = 1;

const PLAN_HASH_DOMAIN: &[u8] = b"oxide-ml.memory-plan.v1";
const RAW_PLAN_HASH_DOMAIN: &[u8] = b"oxide-ml.memory-plan.raw.v1";

/// Structural hash of a memory plan.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PlanHash(pub [u8; 32]);

impl PlanHash {
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    pub fn to_hex(self) -> String {
        let mut out = String::with_capacity(64);
        for byte in self.0 {
            use fmt::Write;
            let _ = write!(out, "{byte:02x}");
        }
        out
    }
}

impl fmt::Debug for PlanHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PlanHash({})", self.to_hex())
    }
}

impl fmt::Display for PlanHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_hex())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PlanHashError {
    InvalidPlan(String),
    Graph(GraphHashError),
    UnknownValue(ValueId),
}

impl fmt::Display for PlanHashError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PlanHashError::InvalidPlan(message) => {
                write!(f, "invalid memory plan: {message}")
            }
            PlanHashError::Graph(error) => write!(f, "{error}"),
            PlanHashError::UnknownValue(value) => {
                write!(f, "memory plan references unknown value {value}")
            }
        }
    }
}

impl std::error::Error for PlanHashError {}

impl From<GraphHashError> for PlanHashError {
    fn from(value: GraphHashError) -> Self {
        PlanHashError::Graph(value)
    }
}

impl From<PlanHashError> for oxide_ml_core::Error {
    fn from(value: PlanHashError) -> Self {
        oxide_ml_core::Error::invalid_graph(value.to_string())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CanonicalSlot {
    pub id: BufferId,
    pub size_bytes: usize,
    pub alignment: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CanonicalAssignment {
    pub value: CanonicalValueRef,
    pub slot: BufferId,
    pub offset: usize,
    pub size_bytes: usize,
    pub dtype: DType,
    pub shape: CanonicalShape,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum CanonicalLifetimePoint {
    Node(u32),
    End,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CanonicalLifetime {
    pub value: CanonicalValueRef,
    pub defined_at: CanonicalLifetimePoint,
    pub last_used_at: CanonicalLifetimePoint,
    pub size_bytes: Option<usize>,
    pub dtype: DType,
    pub shape: CanonicalShape,
}

/// Graph-relative canonical memory plan.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CanonicalPlan {
    pub version: u32,
    pub slots: Vec<CanonicalSlot>,
    pub assignments: Vec<CanonicalAssignment>,
    pub lifetimes: Vec<CanonicalLifetime>,
    pub unplanned: Vec<CanonicalValueRef>,
    pub total_value_bytes: usize,
}

impl CanonicalPlan {
    pub fn hash(&self) -> PlanHash {
        let mut encoder = PlanEncoder::new(PLAN_HASH_DOMAIN);
        encoder.u32(self.version);
        encoder.usize(self.slots.len());
        for slot in &self.slots {
            encoder.u32(slot.id.0);
            encoder.usize(slot.size_bytes);
            encoder.usize(slot.alignment);
        }
        encoder.usize(self.assignments.len());
        for assignment in &self.assignments {
            encoder.value_ref(assignment.value);
            encoder.u32(assignment.slot.0);
            encoder.usize(assignment.offset);
            encoder.usize(assignment.size_bytes);
            encoder.u8(dtype_tag(assignment.dtype));
            encoder.shape(&assignment.shape);
        }
        encoder.usize(self.lifetimes.len());
        for lifetime in &self.lifetimes {
            encoder.value_ref(lifetime.value);
            encoder.lifetime_point(lifetime.defined_at);
            encoder.lifetime_point(lifetime.last_used_at);
            match lifetime.size_bytes {
                Some(bytes) => {
                    encoder.u8(1);
                    encoder.usize(bytes);
                }
                None => encoder.u8(0),
            }
            encoder.u8(dtype_tag(lifetime.dtype));
            encoder.shape(&lifetime.shape);
        }
        encoder.usize(self.unplanned.len());
        for value in &self.unplanned {
            encoder.value_ref(*value);
        }
        encoder.usize(self.total_value_bytes);
        PlanHash(encoder.finish())
    }
}

/// Canonicalizes a plan relative to a graph.
///
/// Assignments and lifetimes are keyed by [`CanonicalValueRef`] and lifetime
/// points by canonical node indices, so the result does not depend on the
/// graph's original ValueId/NodeId numbering.
pub fn canonicalize_memory_plan(
    graph: &Graph,
    plan: &MemoryPlan,
) -> Result<CanonicalPlan, PlanHashError> {
    let canonical_graph = graph.canonicalize()?;
    validate_memory_plan_with_graph(graph, plan, &canonical_graph)?;

    let mut symbol_names: HashMap<SymbolId, String> = HashMap::new();
    for symbol in &graph.symbols {
        symbol_names.insert(symbol.id, symbol.name.clone());
    }

    let mut assignments = plan
        .assignments
        .iter()
        .map(|(&value, assignment)| {
            Ok(CanonicalAssignment {
                value: canonical_graph
                    .value_ref(value)
                    .ok_or(PlanHashError::UnknownValue(value))?,
                slot: assignment.slot,
                offset: assignment.offset,
                size_bytes: assignment.size_bytes,
                dtype: assignment.dtype,
                shape: CanonicalShape::from_shape(&assignment.shape, &symbol_names)?,
            })
        })
        .collect::<Result<Vec<_>, PlanHashError>>()?;
    assignments.sort_by_key(|assignment| assignment.value);

    let mut lifetimes = plan
        .lifetimes
        .iter()
        .map(|lifetime| {
            Ok(CanonicalLifetime {
                value: canonical_graph
                    .value_ref(lifetime.value)
                    .ok_or(PlanHashError::UnknownValue(lifetime.value))?,
                defined_at: canonical_lifetime_point(&canonical_graph, lifetime.defined_at)?,
                last_used_at: canonical_lifetime_point(&canonical_graph, lifetime.last_used_at)?,
                size_bytes: lifetime.size_bytes,
                dtype: lifetime.dtype,
                shape: CanonicalShape::from_shape(&lifetime.shape, &symbol_names)?,
            })
        })
        .collect::<Result<Vec<_>, PlanHashError>>()?;
    lifetimes.sort_by_key(|lifetime| lifetime.value);

    let mut unplanned = plan
        .unplanned_values
        .iter()
        .map(|&value| {
            canonical_graph
                .value_ref(value)
                .ok_or(PlanHashError::UnknownValue(value))
        })
        .collect::<Result<Vec<_>, PlanHashError>>()?;
    unplanned.sort();

    let slots = plan
        .slots
        .iter()
        .map(|slot| CanonicalSlot {
            id: slot.id,
            size_bytes: slot.size_bytes,
            alignment: slot.alignment,
        })
        .collect();

    Ok(CanonicalPlan {
        version: MEMORY_PLAN_VERSION,
        slots,
        assignments,
        lifetimes,
        unplanned,
        total_value_bytes: plan.total_value_bytes,
    })
}

/// Graph-aware plan hash used by arena identity.
pub fn canonical_memory_plan_hash(
    graph: &Graph,
    plan: &MemoryPlan,
) -> Result<PlanHash, PlanHashError> {
    Ok(canonicalize_memory_plan(graph, plan)?.hash())
}

/// Graph-local plan hash.
///
/// This uses the raw ValueId/NodeId numbering. It is deterministic for one
/// graph and is useful for validating plan structure in isolation. Arena
/// identity uses [`canonical_memory_plan_hash`].
pub fn memory_plan_hash(plan: &MemoryPlan) -> Result<PlanHash, PlanHashError> {
    validate_memory_plan_graph_independent(plan)?;

    let mut encoder = PlanEncoder::new(RAW_PLAN_HASH_DOMAIN);
    encoder.u32(MEMORY_PLAN_VERSION);
    encoder.usize(plan.slots.len());
    for slot in &plan.slots {
        encoder.u32(slot.id.0);
        encoder.usize(slot.size_bytes);
        encoder.usize(slot.alignment);
    }

    let mut assignments = plan.assignments.iter().collect::<Vec<_>>();
    assignments.sort_by_key(|(value, _)| value.0);
    encoder.usize(assignments.len());
    for (value, assignment) in assignments {
        encoder.u32(value.0);
        encoder.u32(assignment.slot.0);
        encoder.usize(assignment.offset);
        encoder.usize(assignment.size_bytes);
        encoder.u8(dtype_tag(assignment.dtype));
        encoder.raw_shape(&assignment.shape);
    }

    let mut lifetimes = plan.lifetimes.iter().collect::<Vec<_>>();
    lifetimes.sort_by_key(|lifetime| lifetime.value.0);
    encoder.usize(lifetimes.len());
    for lifetime in lifetimes {
        encoder.u32(lifetime.value.0);
        encoder.raw_lifetime_point(lifetime.defined_at);
        encoder.raw_lifetime_point(lifetime.last_used_at);
        match lifetime.size_bytes {
            Some(bytes) => {
                encoder.u8(1);
                encoder.usize(bytes);
            }
            None => encoder.u8(0),
        }
        encoder.u8(dtype_tag(lifetime.dtype));
        encoder.raw_shape(&lifetime.shape);
    }

    let mut unplanned = plan.unplanned_values.clone();
    unplanned.sort_by_key(|value| value.0);
    encoder.usize(unplanned.len());
    for value in unplanned {
        encoder.u32(value.0);
    }
    encoder.usize(plan.total_value_bytes);
    Ok(PlanHash(encoder.finish()))
}

/// Validates graph-independent structure. Slot capacity, alignments, lifetime
/// ordering, assignment capacity and dtype/shape size consistency are checked.
pub fn validate_memory_plan_graph_independent(plan: &MemoryPlan) -> Result<(), PlanHashError> {
    if plan.slots.len() > u32::MAX as usize {
        return Err(PlanHashError::InvalidPlan(
            "slot count exceeds u32 range".into(),
        ));
    }
    for (index, slot) in plan.slots.iter().enumerate() {
        if slot.id.0 as usize != index {
            return Err(PlanHashError::InvalidPlan(format!(
                "slot table is not dense at {index}: {:?}",
                slot.id
            )));
        }
        if slot.alignment == 0 || !slot.alignment.is_power_of_two() {
            return Err(PlanHashError::InvalidPlan(format!(
                "slot {:?} has invalid alignment {}",
                slot.id, slot.alignment
            )));
        }
        if slot.size_bytes == 0 {
            return Err(PlanHashError::InvalidPlan(format!(
                "slot {:?} has zero capacity",
                slot.id
            )));
        }
        if slot.size_bytes % slot.alignment != 0 {
            return Err(PlanHashError::InvalidPlan(format!(
                "slot {:?} capacity {} is not aligned to {}",
                slot.id, slot.size_bytes, slot.alignment
            )));
        }
    }

    let mut assignment_bytes = 0usize;
    for (&value, assignment) in &plan.assignments {
        let slot = plan.slot(assignment.slot).ok_or_else(|| {
            PlanHashError::InvalidPlan(format!(
                "value {value} references missing slot {:?}",
                assignment.slot
            ))
        })?;
        let end = assignment
            .offset
            .checked_add(assignment.size_bytes)
            .ok_or_else(|| {
                PlanHashError::InvalidPlan(format!("value {value} assignment overflows usize"))
            })?;
        if end > slot.size_bytes {
            return Err(PlanHashError::InvalidPlan(format!(
                "value {value} needs {}..{} but slot {:?} has {}",
                assignment.offset, end, slot.id, slot.size_bytes
            )));
        }
        if assignment.offset % slot.alignment != 0 {
            return Err(PlanHashError::InvalidPlan(format!(
                "value {value} offset {} is not aligned to {}",
                assignment.offset, slot.alignment
            )));
        }
        let elements = assignment.shape.numel_static().ok_or_else(|| {
            PlanHashError::InvalidPlan(format!(
                "value {value} has a dynamic shape but is assigned a slot"
            ))
        })?;
        let expected = elements
            .checked_mul(assignment.dtype.size_in_bytes())
            .ok_or_else(|| {
                PlanHashError::InvalidPlan(format!("value {value} byte size overflows"))
            })?;
        if assignment.size_bytes != expected {
            return Err(PlanHashError::InvalidPlan(format!(
                "value {value} assignment size {} does not match {}x{} = {expected}",
                assignment.size_bytes,
                elements,
                assignment.dtype.size_in_bytes()
            )));
        }
        assignment_bytes = assignment_bytes.saturating_add(assignment.size_bytes);
    }

    let mut lifetime_values = HashSet::new();
    for lifetime in &plan.lifetimes {
        if !lifetime_values.insert(lifetime.value) {
            return Err(PlanHashError::InvalidPlan(format!(
                "duplicate lifetime for value {}",
                lifetime.value
            )));
        }
        if lifetime.defined_at > lifetime.last_used_at {
            return Err(PlanHashError::InvalidPlan(format!(
                "value {} lifetime starts {:?} after it ends {:?}",
                lifetime.value, lifetime.defined_at, lifetime.last_used_at
            )));
        }
        let elements = lifetime.shape.numel_static();
        match (elements, lifetime.size_bytes) {
            (Some(elements), Some(bytes)) => {
                let expected = elements.saturating_mul(lifetime.dtype.size_in_bytes());
                if bytes != expected {
                    return Err(PlanHashError::InvalidPlan(format!(
                        "value {} lifetime size {bytes} does not match {expected}",
                        lifetime.value
                    )));
                }
            }
            (None, None) => {}
            (Some(_), None) => {
                return Err(PlanHashError::InvalidPlan(format!(
                    "value {} has a static shape but no lifetime size",
                    lifetime.value
                )));
            }
            (None, Some(_)) => {
                return Err(PlanHashError::InvalidPlan(format!(
                    "value {} has a dynamic shape but a concrete lifetime size",
                    lifetime.value
                )));
            }
        }
    }

    let mut unplanned = HashSet::new();
    for &value in &plan.unplanned_values {
        if !unplanned.insert(value) {
            return Err(PlanHashError::InvalidPlan(format!(
                "value {value} is listed as unplanned more than once"
            )));
        }
        if plan.assignments.contains_key(&value) {
            return Err(PlanHashError::InvalidPlan(format!(
                "value {value} is both assigned and unplanned"
            )));
        }
    }

    if plan.total_value_bytes != assignment_bytes {
        return Err(PlanHashError::InvalidPlan(format!(
            "total_value_bytes {} does not match assigned bytes {assignment_bytes}",
            plan.total_value_bytes
        )));
    }

    Ok(())
}

/// Full validation against a graph and its canonical form.
pub fn validate_memory_plan_with_graph(
    graph: &Graph,
    plan: &MemoryPlan,
    canonical_graph: &CanonicalGraph,
) -> Result<(), PlanHashError> {
    validate_memory_plan_graph_independent(plan)?;

    let mut lifetime_by_value = HashMap::new();
    for lifetime in &plan.lifetimes {
        lifetime_by_value.insert(lifetime.value, lifetime);
    }

    for (&value, assignment) in &plan.assignments {
        let info = graph
            .value(value)
            .map_err(|_| PlanHashError::UnknownValue(value))?;
        if info.dtype != assignment.dtype || info.shape != assignment.shape {
            return Err(PlanHashError::InvalidPlan(format!(
                "value {value} assignment metadata disagrees with graph value"
            )));
        }
        if canonical_graph.value_ref(value).is_none() {
            return Err(PlanHashError::UnknownValue(value));
        }
        if !lifetime_by_value.contains_key(&value) {
            return Err(PlanHashError::InvalidPlan(format!(
                "assigned value {value} has no lifetime"
            )));
        }
    }

    for lifetime in &plan.lifetimes {
        let info = graph
            .value(lifetime.value)
            .map_err(|_| PlanHashError::UnknownValue(lifetime.value))?;
        if info.dtype != lifetime.dtype || info.shape != lifetime.shape {
            return Err(PlanHashError::InvalidPlan(format!(
                "value {} lifetime metadata disagrees with graph value",
                lifetime.value
            )));
        }
        if canonical_graph.value_ref(lifetime.value).is_none() {
            return Err(PlanHashError::UnknownValue(lifetime.value));
        }
        match (
            lifetime.size_bytes,
            plan.assignments.contains_key(&lifetime.value),
        ) {
            (Some(_), false) => {
                return Err(PlanHashError::InvalidPlan(format!(
                    "value {} has a concrete lifetime but no assignment",
                    lifetime.value
                )));
            }
            (None, true) => {
                return Err(PlanHashError::InvalidPlan(format!(
                    "value {} is assigned but has no concrete size",
                    lifetime.value
                )));
            }
            _ => {}
        }
    }

    for node in graph.nodes() {
        for &output in &node.outputs {
            if !lifetime_by_value.contains_key(&output) {
                return Err(PlanHashError::InvalidPlan(format!(
                    "node output {output} has no lifetime"
                )));
            }
        }
    }

    Ok(())
}

fn canonical_lifetime_point(
    graph: &CanonicalGraph,
    point: LifetimePoint,
) -> Result<CanonicalLifetimePoint, PlanHashError> {
    Ok(match point {
        LifetimePoint::End => CanonicalLifetimePoint::End,
        LifetimePoint::Node(node) => CanonicalLifetimePoint::Node(
            graph.node_ref(oxide_ml_core::NodeId(node)).ok_or_else(|| {
                PlanHashError::InvalidPlan(format!(
                    "lifetime references node {node} outside the graph"
                ))
            })?,
        ),
    })
}

struct PlanEncoder {
    hasher: blake3::Hasher,
}

impl PlanEncoder {
    fn new(domain: &[u8]) -> Self {
        let mut hasher = blake3::Hasher::new();
        hasher.update(domain);
        PlanEncoder { hasher }
    }

    fn u8(&mut self, value: u8) {
        self.hasher.update(&[value]);
    }

    fn u32(&mut self, value: u32) {
        self.hasher.update(&value.to_le_bytes());
    }

    fn u64(&mut self, value: u64) {
        self.hasher.update(&value.to_le_bytes());
    }

    fn usize(&mut self, value: usize) {
        self.u64(value as u64);
    }

    fn value_ref(&mut self, value: CanonicalValueRef) {
        match value {
            CanonicalValueRef::Input(index) => {
                self.u8(0);
                self.u32(index);
            }
            CanonicalValueRef::Parameter(index) => {
                self.u8(1);
                self.u32(index);
            }
            CanonicalValueRef::Node { node, output } => {
                self.u8(2);
                self.u32(node);
                self.u32(output);
            }
        }
    }

    fn lifetime_point(&mut self, point: CanonicalLifetimePoint) {
        match point {
            CanonicalLifetimePoint::Node(node) => {
                self.u8(0);
                self.u32(node);
            }
            CanonicalLifetimePoint::End => self.u8(1),
        }
    }

    fn shape(&mut self, shape: &CanonicalShape) {
        self.usize(shape.dims.len());
        for dim in &shape.dims {
            match dim {
                oxide_ml_graph::CanonicalDim::Static(value) => {
                    self.u8(0);
                    self.u64(*value);
                }
                oxide_ml_graph::CanonicalDim::Symbol(name) => {
                    self.u8(1);
                    self.usize(name.len());
                    self.hasher.update(name.as_bytes());
                }
                oxide_ml_graph::CanonicalDim::Dynamic => self.u8(2),
            }
        }
    }

    fn raw_shape(&mut self, shape: &Shape) {
        self.usize(shape.rank());
        for dim in shape.dims() {
            match *dim {
                Dim::Static(value) => {
                    self.u8(0);
                    self.u64(value as u64);
                }
                Dim::Symbol(id) => {
                    self.u8(1);
                    self.u32(id.0);
                }
                Dim::Dynamic => self.u8(2),
            }
        }
    }

    fn raw_lifetime_point(&mut self, point: LifetimePoint) {
        match point {
            LifetimePoint::Node(node) => {
                self.u8(0);
                self.u32(node);
            }
            LifetimePoint::End => self.u8(1),
        }
    }

    fn finish(self) -> [u8; 32] {
        *self.hasher.finalize().as_bytes()
    }
}

fn dtype_tag(dtype: DType) -> u8 {
    match dtype {
        DType::F32 => 0,
        DType::F16 => 1,
        DType::BF16 => 2,
        DType::F64 => 3,
        DType::I8 => 4,
        DType::I16 => 5,
        DType::I32 => 6,
        DType::I64 => 7,
        DType::U8 => 8,
        DType::Bool => 9,
    }
}
