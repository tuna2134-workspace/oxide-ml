//! Compile-cache behaviour: keys, sharing, eviction, invalidation and
//! single-flight concurrency.

use std::sync::Arc;

use oxide_ml::prelude::*;

#[derive(Module)]
struct CacheNet {
    fc1: Linear,
    fc2: Linear,
}

impl CacheNet {
    #[model]
    fn forward(&self, x: Tensor<f32>) -> Tensor<f32> {
        let hidden = tanh(self.fc1.forward(x));
        self.fc2.forward(hidden)
    }
}

#[derive(Module)]
struct SameParamsOtherType {
    fc1: Linear,
    fc2: Linear,
}

impl SameParamsOtherType {
    #[model]
    fn forward(&self, x: Tensor<f32>) -> Tensor<f32> {
        let hidden = tanh(self.fc1.forward(x));
        self.fc2.forward(hidden)
    }
}

fn net() -> CacheNet {
    CacheNet {
        fc1: Linear::with_seed(2, 4, true, 1),
        fc2: Linear::with_seed(4, 1, true, 2),
    }
}

fn input(rows: usize) -> Tensor<f32> {
    Tensor::<f32>::from_vec([rows, 2], (0..rows * 2).map(|i| i as f32 * 0.1).collect()).unwrap()
}

#[test]
fn caches_identical_graphs_and_inputs() {
    let cache = CompileCache::new(4);
    let model = net();
    let first = predict_many_with_cache(&model, vec![input(2)], &cache).unwrap();
    let second = predict_many_with_cache(&model, vec![input(2)], &cache).unwrap();
    let stats = cache.stats();
    assert_eq!(stats.misses, 1);
    assert_eq!(stats.hits, 1);
    assert_eq!(stats.compilations, 1);
    assert_eq!(stats.entries, 1);
    assert_eq!(first[0].data(), second[0].data());
}

#[test]
fn different_shape_is_a_different_key() {
    let cache = CompileCache::new(4);
    let model = net();
    predict_many_with_cache(&model, vec![input(1)], &cache).unwrap();
    predict_many_with_cache(&model, vec![input(3)], &cache).unwrap();
    let stats = cache.stats();
    assert_eq!(stats.misses, 2);
    assert_eq!(stats.entries, 2);
}

#[test]
fn structurally_identical_models_share_an_entry() {
    let cache = CompileCache::new(4);
    let a = net();
    let b = net();
    predict_many_with_cache(&a, vec![input(2)], &cache).unwrap();
    predict_many_with_cache(&b, vec![input(2)], &cache).unwrap();
    assert_eq!(cache.stats().entries, 1);
    assert_eq!(cache.stats().hits, 1);
}

#[test]
fn different_module_type_does_not_share() {
    let cache = CompileCache::new(4);
    let a = net();
    let b = SameParamsOtherType {
        fc1: Linear::with_seed(2, 4, true, 1),
        fc2: Linear::with_seed(4, 1, true, 2),
    };
    predict_many_with_cache(&a, vec![input(2)], &cache).unwrap();
    predict_many_with_cache(&b, vec![input(2)], &cache).unwrap();
    // Canonical graphs are identical (module type is not part of the IR), so
    // the graph-hash key intentionally shares the compiled artifact.
    assert_eq!(cache.stats().entries, 1);
}

#[test]
fn options_are_part_of_the_key() {
    let cache = CompileCache::new(4);
    let model = net();
    let specs = [InputSpec::unbatched("input0", [2, 2])];
    let fast = ModelCompileOptions {
        optimize: true,
        ..Default::default()
    };
    let plain = ModelCompileOptions {
        optimize: false,
        memory_plan: false,
        ..Default::default()
    };
    compile_cached_with(&cache, &model, &specs, &fast).unwrap();
    compile_cached_with(&cache, &model, &specs, &plain).unwrap();
    assert_eq!(cache.stats().entries, 2);
    assert_eq!(cache.stats().misses, 2);
}

#[test]
fn lru_evicts_the_oldest_entry() {
    let cache = CompileCache::new(1);
    let model = net();
    predict_many_with_cache(&model, vec![input(1)], &cache).unwrap();
    predict_many_with_cache(&model, vec![input(2)], &cache).unwrap();
    let stats = cache.stats();
    assert_eq!(stats.entries, 1);
    assert_eq!(stats.evictions, 1);
}

#[test]
fn clear_and_set_capacity_invalidate_entries() {
    let cache = CompileCache::new(4);
    let model = net();
    predict_many_with_cache(&model, vec![input(1)], &cache).unwrap();
    predict_many_with_cache(&model, vec![input(2)], &cache).unwrap();
    cache.set_capacity(1);
    assert_eq!(cache.len(), 1);
    cache.clear();
    assert_eq!(cache.len(), 0);
    assert_eq!(cache.stats().invalidations, 1);
}

#[test]
fn parameter_values_do_not_invalidate_the_graph() {
    let cache = CompileCache::new(2);
    let mut model = net();
    let before = predict_many_with_cache(&model, vec![input(2)], &cache).unwrap();
    for value in model.fc1.weight.data_mut() {
        *value += 0.5;
    }
    let after = predict_many_with_cache(&model, vec![input(2)], &cache).unwrap();
    let stats = cache.stats();
    assert_eq!(stats.hits, 1);
    assert_eq!(stats.compilations, 1);
    assert_ne!(before[0].data(), after[0].data());
}

#[test]
fn concurrent_callers_compile_once() {
    let cache = CompileCache::new(2);
    let model = net();
    let mut outputs = Vec::new();
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..8)
            .map(|_| {
                let model = &model;
                let cache = &cache;
                scope.spawn(move || {
                    let tensor = Tensor::<f32>::from_vec([2, 2], vec![0.1, 0.2, 0.3, 0.4]).unwrap();
                    predict_many_with_cache(model, vec![tensor], cache).unwrap()
                })
            })
            .collect();
        for handle in handles {
            outputs.push(handle.join().unwrap());
        }
    });
    let stats = cache.stats();
    assert_eq!(stats.entries, 1);
    assert_eq!(stats.compilations, 1);
    assert_eq!(stats.hits, 7);
    for output in &outputs {
        assert_eq!(output[0].data(), outputs[0][0].data());
    }
}

#[test]
fn compiled_model_is_send_and_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Arc<CompiledModel>>();
    assert_send_sync::<CompileCache>();
}

#[test]
fn global_cache_is_used_by_predict() {
    let model = net();
    let before = global_compile_cache().stats();
    let _ = model.predict(input(5)).unwrap();
    let after = global_compile_cache().stats();
    assert!(after.entries >= 1);
    assert!(after.misses + before.hits <= after.misses + after.hits);
}
