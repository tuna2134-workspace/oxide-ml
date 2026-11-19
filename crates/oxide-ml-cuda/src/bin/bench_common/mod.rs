//! Shared benchmark plumbing for the Phase 7F-19 closure binaries.
//!
//! This module is included from each `src/bin/*_check.rs` binary with
//! `#[path = "bench_common/mod.rs"]`. It deliberately depends only on the
//! public oxide-ml-cuda / std APIs so the benchmark binaries remain thin.
//!
//! Timing definitions
//! ------------------
//! * `gpu_ms`            CUDA event elapsed time for the measured region.
//! * `host_submit_ms`    host wall-clock time spent submitting work only.
//! * `wait_ms`           explicit host wait (`CudaEvent::synchronize`).
//! * `d2h_ms`            device-to-host copy time, when measured separately.
//! * `e2e_ms`            host wall-clock time for submit + wait (+ d2h).
//!
//! For synchronous executor APIs (`run`, `run_planned`, `run_arena`) the final
//! stream synchronisation is inside the API call. Those cases report
//! `host_call_ms` (submit + implicit wait) and leave `wait_ms` empty; the GPU
//! event span is still recorded but includes the host tail between kernel
//! completion and the end-event record. Async cases never synchronise inside
//! the submit region.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use oxide_ml_cuda::{CudaDevice, CudaEvent};

pub const WARMUP: usize = 10;
pub const ROUNDS: usize = 10;
pub const ITERATIONS: usize = 10;
pub const PERCENTILE_METHOD: &str = "nearest-rank: sorted[ceil(p/100*n)-1]";

#[derive(Clone, Debug, Default)]
pub struct Stats {
    pub n: usize,
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    pub median: f64,
    pub p90: f64,
    pub p95: f64,
    pub p99: f64,
    pub stddev: f64,
}

impl Stats {
    pub fn new(values: &[f64]) -> Self {
        if values.is_empty() {
            return Stats::default();
        }
        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = sorted.len();
        let mean = sorted.iter().sum::<f64>() / n as f64;
        let variance = sorted
            .iter()
            .map(|value| {
                let delta = value - mean;
                delta * delta
            })
            .sum::<f64>()
            / n as f64;
        Stats {
            n,
            min: sorted[0],
            max: sorted[n - 1],
            mean,
            median: percentile(&sorted, 50.0),
            p90: percentile(&sorted, 90.0),
            p95: percentile(&sorted, 95.0),
            p99: percentile(&sorted, 99.0),
            stddev: variance.sqrt(),
        }
    }

    pub fn p99_reliable(&self) -> bool {
        self.n >= 100
    }

    pub fn json(&self) -> String {
        format!(
            "{{\"n\":{},\"min\":{:.9},\"max\":{:.9},\"mean\":{:.9},\"median\":{:.9},\
             \"p90\":{:.9},\"p95\":{:.9},\"p99\":{:.9},\"stddev\":{:.9},\
             \"p99_reliable\":{},\"percentile_method\":\"{}\"}}",
            self.n,
            self.min,
            self.max,
            self.mean,
            self.median,
            self.p90,
            self.p95,
            self.p99,
            self.stddev,
            self.p99_reliable(),
            PERCENTILE_METHOD
        )
    }
}

/// Nearest-rank percentile. `p` is in (0, 100].
pub fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let rank = ((p / 100.0) * sorted.len() as f64).ceil() as usize;
    sorted[rank.saturating_sub(1).min(sorted.len() - 1)]
}

#[derive(Clone, Debug, Default)]
pub struct Sample {
    pub round: usize,
    #[allow(dead_code)]
    pub iteration: usize,
    pub gpu_ms: f64,
    pub host_submit_ms: f64,
    pub wait_ms: f64,
    pub d2h_ms: f64,
    pub e2e_ms: f64,
    /// True when the API call itself blocked (synchronous executor API).
    pub host_call_includes_wait: bool,
}

#[derive(Clone, Debug, Default)]
pub struct MemorySnapshot {
    pub logical_bytes: Option<usize>,
    pub planned_logical_bytes: Option<usize>,
    pub runtime_required_bytes: Option<usize>,
    pub pool_reserved_bytes: Option<usize>,
    pub pool_active_bytes: Option<usize>,
    pub pool_pending_release_bytes: Option<usize>,
    pub pool_quarantined_bytes: Option<usize>,
    pub pool_allocation_count: Option<usize>,
    pub pool_reuse_count: Option<usize>,
    pub arena_capacity_bytes: Option<usize>,
    pub arena_used_bytes: Option<usize>,
    pub arena_physical_reserved_bytes: Option<usize>,
    pub arena_slot_allocation_count: Option<u64>,
    pub arena_slot_reuse_count: Option<u64>,
    pub arena_active_lease_count: Option<u64>,
    pub event_allocation_count: Option<u64>,
    pub event_reuse_count: Option<u64>,
    pub event_pending_count: Option<u64>,
    pub event_quarantine_count: Option<u64>,
    pub access_reads: Option<u64>,
    pub access_writes: Option<u64>,
    pub access_read_write: Option<u64>,
    pub access_same_stream_elided: Option<u64>,
    pub access_cross_stream_waits: Option<u64>,
    pub access_read_read_elision: Option<u64>,
    pub driver_used_bytes: Option<usize>,
}

impl MemorySnapshot {
    pub fn json(&self) -> String {
        fn opt<T: std::fmt::Display>(v: &Option<T>) -> String {
            match v {
                Some(value) => value.to_string(),
                None => "null".to_string(),
            }
        }
        format!(
            "{{\"logical_bytes\":{},\"planned_logical_bytes\":{},\
             \"runtime_required_bytes\":{},\"pool\":{{\"reserved_bytes\":{},\
             \"active_bytes\":{},\"pending_release_bytes\":{},\"quarantined_bytes\":{},\
             \"allocation_count\":{},\"reuse_count\":{}}},\
             \"arena\":{{\"capacity_bytes\":{},\"used_bytes\":{},\
             \"physical_reserved_bytes\":{},\"slot_allocation_count\":{},\
             \"slot_reuse_count\":{},\"active_lease_count\":{}}},\
             \"event\":{{\"allocation_count\":{},\"reuse_count\":{},\
             \"pending_count\":{},\"quarantine_count\":{}}},\
             \"access\":{{\"read_count\":{},\"write_count\":{},\
             \"read_write_count\":{},\"same_stream_elided_count\":{},\
             \"cross_stream_wait_count\":{},\"read_read_elision_count\":{}}},\
             \"driver_used_bytes\":{}}}",
            opt(&self.logical_bytes),
            opt(&self.planned_logical_bytes),
            opt(&self.runtime_required_bytes),
            opt(&self.pool_reserved_bytes),
            opt(&self.pool_active_bytes),
            opt(&self.pool_pending_release_bytes),
            opt(&self.pool_quarantined_bytes),
            opt(&self.pool_allocation_count),
            opt(&self.pool_reuse_count),
            opt(&self.arena_capacity_bytes),
            opt(&self.arena_used_bytes),
            opt(&self.arena_physical_reserved_bytes),
            opt(&self.arena_slot_allocation_count),
            opt(&self.arena_slot_reuse_count),
            opt(&self.arena_active_lease_count),
            opt(&self.event_allocation_count),
            opt(&self.event_reuse_count),
            opt(&self.event_pending_count),
            opt(&self.event_quarantine_count),
            opt(&self.access_reads),
            opt(&self.access_writes),
            opt(&self.access_read_write),
            opt(&self.access_same_stream_elided),
            opt(&self.access_cross_stream_waits),
            opt(&self.access_read_read_elision),
            opt(&self.driver_used_bytes),
        )
    }
}

#[derive(Clone, Debug, Default)]
pub struct Meta {
    pub gpu_name: String,
    pub gpu_uuid: String,
    pub compute_capability: String,
    pub cuda_driver_version: String,
    pub cuda_toolkit_version: String,
    pub rust_version: String,
    pub cuda_oxide_revision: String,
    pub timestamp_utc: String,
    pub gpu_clocks: String,
    pub gpu_temperature: String,
    pub gpu_power: String,
    pub total_memory_bytes: usize,
}

fn command_first_line(program: &str, args: &[&str]) -> String {
    Command::new(program)
        .args(args)
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_string())
        .filter(|text| !text.is_empty())
        .unwrap_or_else(|| "unavailable".to_string())
}

fn nvidia_smi_field(field: &str) -> String {
    command_first_line(
        "nvidia-smi",
        &[&format!("--query-gpu={field}"), "--format=csv,noheader"],
    )
    .lines()
    .next()
    .unwrap_or("unavailable")
    .trim()
    .to_string()
}

pub fn collect_meta(device: &CudaDevice) -> Meta {
    let info = device.info();
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0);
    let (cc_major, cc_minor) = info.compute_capability;
    let toolkit = {
        let nvcc = command_first_line("nvcc", &["--version"]);
        if nvcc == "unavailable" {
            command_first_line("cuda-oxide", &["--version"])
        } else {
            nvcc
        }
    };
    Meta {
        gpu_name: info.name.clone(),
        gpu_uuid: nvidia_smi_field("uuid"),
        compute_capability: format!("{cc_major}.{cc_minor}"),
        cuda_driver_version: nvidia_smi_field("driver_version"),
        cuda_toolkit_version: toolkit,
        rust_version: command_first_line("rustc", &["--version"]),
        cuda_oxide_revision: command_first_line("cargo", &["oxide", "--version"]),
        timestamp_utc: format!("unix:{timestamp}"),
        gpu_clocks: format!(
            "sm={} mem={}",
            nvidia_smi_field("clocks.sm"),
            nvidia_smi_field("clocks.mem")
        ),
        gpu_temperature: nvidia_smi_field("temperature.gpu"),
        gpu_power: nvidia_smi_field("power.draw"),
        total_memory_bytes: info.total_memory,
    }
}

pub fn graph_hash(graph: &oxide_ml_graph::Graph) -> String {
    graph
        .canonicalize()
        .map(|canonical| canonical.hash().to_string())
        .unwrap_or_else(|_| "unavailable".to_string())
}

pub fn plan_hash(plan: &oxide_ml_cuda::CudaMemoryPlan) -> String {
    plan.plan_hash()
        .map(|hash| hash.to_string())
        .unwrap_or_else(|| "unavailable".to_string())
}

pub fn escape_json(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 2);
    for character in text.chars() {
        match character {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

pub struct Case {
    pub benchmark: String,
    pub graph: String,
    pub mode: String,
    pub shape: String,
    pub dtype: String,
    pub stream_count: usize,
    pub graph_hash: String,
    pub plan_hash: String,
    pub samples: Vec<Sample>,
    pub memory: MemorySnapshot,
    pub note: String,
    pub correctness: String,
    pub compile_ms: f64,
}

impl Case {
    pub fn new(
        benchmark: impl Into<String>,
        graph: impl Into<String>,
        mode: impl Into<String>,
        shape: impl Into<String>,
    ) -> Self {
        Case {
            benchmark: benchmark.into(),
            graph: graph.into(),
            mode: mode.into(),
            shape: shape.into(),
            dtype: "F32".to_string(),
            stream_count: 1,
            graph_hash: "unavailable".to_string(),
            plan_hash: "unavailable".to_string(),
            samples: Vec::new(),
            memory: MemorySnapshot::default(),
            note: String::new(),
            correctness: "NOT CHECKED".to_string(),
            compile_ms: 0.0,
        }
    }

    pub fn gpu_stats(&self) -> Stats {
        Stats::new(
            &self
                .samples
                .iter()
                .map(|sample| sample.gpu_ms)
                .collect::<Vec<_>>(),
        )
    }

    pub fn host_stats(&self) -> Stats {
        Stats::new(
            &self
                .samples
                .iter()
                .map(|sample| {
                    if sample.host_call_includes_wait {
                        sample.host_submit_ms + sample.wait_ms
                    } else {
                        sample.host_submit_ms
                    }
                })
                .collect::<Vec<_>>(),
        )
    }

    pub fn wait_stats(&self) -> Stats {
        Stats::new(
            &self
                .samples
                .iter()
                .map(|sample| sample.wait_ms)
                .collect::<Vec<_>>(),
        )
    }

    pub fn e2e_stats(&self) -> Stats {
        Stats::new(
            &self
                .samples
                .iter()
                .map(|sample| sample.e2e_ms)
                .collect::<Vec<_>>(),
        )
    }

    pub fn d2h_stats(&self) -> Stats {
        Stats::new(
            &self
                .samples
                .iter()
                .map(|sample| sample.d2h_ms)
                .collect::<Vec<_>>(),
        )
    }

    pub fn round_means_gpu(&self) -> Vec<(usize, f64)> {
        let mut means = Vec::new();
        for round in 0..=self.samples.iter().map(|s| s.round).max().unwrap_or(0) {
            let values = self
                .samples
                .iter()
                .filter(|sample| sample.round == round)
                .map(|sample| sample.gpu_ms)
                .collect::<Vec<_>>();
            if !values.is_empty() {
                means.push((round, values.iter().sum::<f64>() / values.len() as f64));
            }
        }
        means
    }

    fn json(&self) -> String {
        let rounds = self
            .round_means_gpu()
            .iter()
            .map(|(round, mean)| format!("{{\"round\":{round},\"gpu_mean_ms\":{mean:.9}}}"))
            .collect::<Vec<_>>()
            .join(",");
        format!(
            "{{\"benchmark\":\"{}\",\"graph\":\"{}\",\"mode\":\"{}\",\"shape\":\"{}\",\
             \"dtype\":\"{}\",\"stream_count\":{},\"graph_hash\":\"{}\",\"plan_hash\":\"{}\",\
             \"warmup\":{},\"rounds\":{},\"iterations\":{},\"sample_count\":{},\
             \"compile_ms\":{:.6},
             \"timing\":{{\"gpu\":{},\"host_submit\":{},\"wait\":{},\"d2h\":{},\"e2e\":{}}},\
             \"round_gpu_means\":[{}],\"memory\":{},\"correctness\":\"{}\",\"note\":\"{}\"}}",
            escape_json(&self.benchmark),
            escape_json(&self.graph),
            escape_json(&self.mode),
            escape_json(&self.shape),
            escape_json(&self.dtype),
            self.stream_count,
            escape_json(&self.graph_hash),
            escape_json(&self.plan_hash),
            WARMUP,
            ROUNDS,
            ITERATIONS,
            self.samples.len(),
            self.compile_ms,
            self.gpu_stats().json(),
            self.host_stats().json(),
            self.wait_stats().json(),
            self.d2h_stats().json(),
            self.e2e_stats().json(),
            rounds,
            self.memory.json(),
            escape_json(&self.correctness),
            escape_json(&self.note),
        )
    }
}

pub fn write_results(
    path: &Path,
    benchmark: &str,
    meta: &Meta,
    cases: &[Case],
) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut json = String::new();
    json.push_str("{\"benchmark\":\"");
    json.push_str(&escape_json(benchmark));
    json.push_str("\",\"meta\":{");
    json.push_str(&format!(
        "\"gpu_name\":\"{}\",\"gpu_uuid\":\"{}\",\"compute_capability\":\"{}\",\
         \"cuda_driver_version\":\"{}\",\"cuda_toolkit_version\":\"{}\",\
         \"rust_version\":\"{}\",\"cuda_oxide_revision\":\"{}\",\
         \"timestamp_utc\":\"{}\",\"gpu_clocks\":\"{}\",\"gpu_temperature\":\"{}\",\
         \"gpu_power\":\"{}\",\"total_memory_bytes\":{},\
         \"warmup\":{},\"rounds\":{},\"iterations\":{},\"percentile_method\":\"{}\"",
        escape_json(&meta.gpu_name),
        escape_json(&meta.gpu_uuid),
        escape_json(&meta.compute_capability),
        escape_json(&meta.cuda_driver_version),
        escape_json(&meta.cuda_toolkit_version),
        escape_json(&meta.rust_version),
        escape_json(&meta.cuda_oxide_revision),
        escape_json(&meta.timestamp_utc),
        escape_json(&meta.gpu_clocks),
        escape_json(&meta.gpu_temperature),
        escape_json(&meta.gpu_power),
        meta.total_memory_bytes,
        WARMUP,
        ROUNDS,
        ITERATIONS,
        PERCENTILE_METHOD
    ));
    json.push_str("},\"cases\":[");
    for (index, case) in cases.iter().enumerate() {
        if index > 0 {
            json.push(',');
        }
        json.push_str(&case.json());
    }
    json.push_str("]}");
    let mut file = fs::File::create(path)?;
    file.write_all(json.as_bytes())?;

    let csv_path = path.with_extension("csv");
    let mut csv = String::from(
        "benchmark,graph,mode,shape,dtype,stream_count,compile_ms,metric,n,min_ms,max_ms,mean_ms,\
         median_ms,p90_ms,p95_ms,p99_ms,stddev_ms,note\n",
    );
    for case in cases {
        for (metric, stats) in [
            ("gpu", case.gpu_stats()),
            ("host_submit", case.host_stats()),
            ("wait", case.wait_stats()),
            ("d2h", case.d2h_stats()),
            ("e2e", case.e2e_stats()),
        ] {
            csv.push_str(&format!(
                "{},{},{},{},{},{},{:.6},{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{}\n",
                csv_escape(&case.benchmark),
                csv_escape(&case.graph),
                csv_escape(&case.mode),
                csv_escape(&case.shape),
                case.dtype,
                case.stream_count,
                case.compile_ms,
                metric,
                stats.n,
                stats.min,
                stats.max,
                stats.mean,
                stats.median,
                stats.p90,
                stats.p95,
                stats.p99,
                stats.stddev,
                csv_escape(&case.note)
            ));
        }
    }
    fs::File::create(csv_path)?.write_all(csv.as_bytes())?;
    Ok(())
}

fn csv_escape(text: &str) -> String {
    if text.contains(',') || text.contains('"') || text.contains('\n') {
        format!("\"{}\"", text.replace('"', "\"\""))
    } else {
        text.to_string()
    }
}

pub fn out_dir(default: &str) -> PathBuf {
    std::env::var("BENCH_OUT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(default))
}

/// Synchronous measurement. The final wait is inside the closure.
#[allow(dead_code)]
pub fn measure_sync<F>(
    stream: &std::sync::Arc<cuda_core::CudaStream>,
    rounds: usize,
    iterations: usize,
    warmup: usize,
    mut call: F,
) -> Result<Vec<Sample>, Box<dyn std::error::Error>>
where
    F: FnMut() -> Result<(), Box<dyn std::error::Error>>,
{
    for _ in 0..warmup {
        call()?;
    }
    let start_event = CudaEvent::new(stream.context())?;
    let end_event = CudaEvent::new(stream.context())?;
    let mut samples = Vec::with_capacity(rounds * iterations);
    for round in 0..rounds {
        for iteration in 0..iterations {
            start_event.record(stream)?;
            let begin = Instant::now();
            call()?;
            let host_ms = begin.elapsed().as_secs_f64() * 1000.0;
            end_event.record(stream)?;
            end_event.synchronize()?;
            let gpu_ms = f64::from(start_event.elapsed_ms(&end_event)?);
            let e2e_ms = begin.elapsed().as_secs_f64() * 1000.0;
            samples.push(Sample {
                round,
                iteration,
                gpu_ms,
                host_submit_ms: host_ms,
                wait_ms: 0.0,
                d2h_ms: 0.0,
                e2e_ms,
                host_call_includes_wait: true,
            });
        }
    }
    Ok(samples)
}

/// Asynchronous measurement. The submit closure must not synchronize.
#[allow(dead_code)]
pub fn measure_async<F>(
    stream: &std::sync::Arc<cuda_core::CudaStream>,
    rounds: usize,
    iterations: usize,
    warmup: usize,
    mut submit: F,
) -> Result<Vec<Sample>, Box<dyn std::error::Error>>
where
    F: FnMut() -> Result<oxide_ml_cuda::CudaExecution, Box<dyn std::error::Error>>,
{
    for _ in 0..warmup {
        let mut execution = submit()?;
        execution.wait()?;
    }
    let start_event = CudaEvent::new(stream.context())?;
    let end_event = CudaEvent::new(stream.context())?;
    let mut samples = Vec::with_capacity(rounds * iterations);
    for round in 0..rounds {
        for iteration in 0..iterations {
            start_event.record(stream)?;
            let begin = Instant::now();
            let execution = submit()?;
            let host_ms = begin.elapsed().as_secs_f64() * 1000.0;
            end_event.record(stream)?;
            let wait_begin = Instant::now();
            end_event.synchronize()?;
            let wait_ms = wait_begin.elapsed().as_secs_f64() * 1000.0;
            let gpu_ms = f64::from(start_event.elapsed_ms(&end_event)?);
            let e2e_ms = begin.elapsed().as_secs_f64() * 1000.0;
            samples.push(Sample {
                round,
                iteration,
                gpu_ms,
                host_submit_ms: host_ms,
                wait_ms,
                d2h_ms: 0.0,
                e2e_ms,
                host_call_includes_wait: false,
            });
            drop(execution);
        }
    }
    Ok(samples)
}

/// Errors before any kernel launch are the only ones treated as capacity
/// rejections. This helper keeps that assertion explicit in the binaries.
#[allow(dead_code)]
pub fn is_capacity_error(error: &oxide_ml_cuda::CudaError) -> bool {
    matches!(error, oxide_ml_cuda::CudaError::Capacity(_))
}

#[allow(dead_code)]
pub fn max_abs_diff(left: &[f32], right: &[f32]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(&a, &b)| (a - b).abs() as f64)
        .fold(0.0, f64::max)
}

#[allow(dead_code)]
pub fn all_finite(values: &[f32]) -> bool {
    values.iter().all(|value| value.is_finite())
}

/// Absolute-or-relative tolerance used by the closure reports:
/// `|a-b| <= atol + rtol * |b|`.
pub fn within_tolerance(
    actual: &[f32],
    expected: &[f32],
    atol: f64,
    rtol: f64,
) -> Result<(), String> {
    if actual.len() != expected.len() {
        return Err(format!(
            "length mismatch: actual {} expected {}",
            actual.len(),
            expected.len()
        ));
    }
    for (index, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        if !a.is_finite() || !e.is_finite() {
            return Err(format!(
                "non-finite value at index {index}: actual={a} expected={e}"
            ));
        }
        let tolerance = atol + rtol * (e.abs() as f64);
        if ((a - e).abs() as f64) > tolerance {
            return Err(format!(
                "tolerance exceeded at index {index}: actual={a} expected={e} \
                 diff={} tolerance={tolerance}",
                (a - e).abs()
            ));
        }
    }
    Ok(())
}
