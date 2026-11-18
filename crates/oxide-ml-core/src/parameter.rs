use std::collections::HashMap;

use crate::{DenseTensor, Error, ParameterId, Result, Scalar, Shape};

/// One materialized parameter.
#[derive(Clone, Debug)]
pub struct Parameter<T: Scalar = f32> {
    pub id: ParameterId,
    pub name: String,
    pub data: DenseTensor<T>,
    pub trainable: bool,
}

impl<T: Scalar> Parameter<T> {
    pub fn shape(&self) -> &Shape {
        self.data.shape()
    }
}

/// Parameter storage owned by the training loop, not by the graph.
///
/// The graph only refers to `ValueId`s whose `ValueKind` is `Parameter`. This
/// keeps graph compilation independent from checkpoint/optimizer state.
#[derive(Clone, Debug, Default)]
pub struct ParameterStore<T: Scalar = f32> {
    params: Vec<Parameter<T>>,
    index: HashMap<ParameterId, usize>,
}

impl<T: Scalar> ParameterStore<T> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, parameter: Parameter<T>) -> Result<()> {
        if self.index.contains_key(&parameter.id) {
            return Err(Error::invalid_graph(format!(
                "parameter {} was inserted twice",
                parameter.id
            )));
        }
        let position = self.params.len();
        self.index.insert(parameter.id, position);
        self.params.push(parameter);
        Ok(())
    }

    pub fn add(
        &mut self,
        id: ParameterId,
        name: impl Into<String>,
        data: DenseTensor<T>,
        trainable: bool,
    ) -> Result<()> {
        self.insert(Parameter {
            id,
            name: name.into(),
            data,
            trainable,
        })
    }

    pub fn get(&self, id: ParameterId) -> Result<&Parameter<T>> {
        self.index
            .get(&id)
            .and_then(|&index| self.params.get(index))
            .ok_or(Error::UnknownParameter(id))
    }

    pub fn get_mut(&mut self, id: ParameterId) -> Result<&mut Parameter<T>> {
        let index = self
            .index
            .get(&id)
            .copied()
            .ok_or(Error::UnknownParameter(id))?;
        Ok(&mut self.params[index])
    }

    pub fn get_by_name(&self, name: &str) -> Option<&Parameter<T>> {
        self.params.iter().find(|parameter| parameter.name == name)
    }

    pub fn iter(&self) -> impl Iterator<Item = &Parameter<T>> {
        self.params.iter()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut Parameter<T>> {
        self.params.iter_mut()
    }

    pub fn len(&self) -> usize {
        self.params.len()
    }

    pub fn is_empty(&self) -> bool {
        self.params.is_empty()
    }

    pub fn trainable(&self) -> impl Iterator<Item = &Parameter<T>> {
        self.params.iter().filter(|parameter| parameter.trainable)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn store_roundtrip() {
        let mut store = ParameterStore::new();
        store
            .add(
                ParameterId(0),
                "w",
                DenseTensor::from_vec([2, 2], vec![0.0; 4]).unwrap(),
                true,
            )
            .unwrap();
        assert_eq!(store.len(), 1);
        assert_eq!(store.get(ParameterId(0)).unwrap().name, "w");
    }
}
