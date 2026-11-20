//! Reusable KV-cache state: capacity, growth, sharing, reset and sessions.

use oxide_ml::prelude::*;

fn uniform_cache(layers: usize, capacity: usize) -> KvCache {
    KvCache::new(KvCacheConfig::uniform(layers, 2, 4, capacity)).unwrap()
}

#[test]
fn append_and_read_round_trip() {
    let mut cache = uniform_cache(2, 4);
    assert_eq!(cache.sequence_length(), 0);
    assert!(cache.is_empty());
    assert_eq!(cache.len(0).unwrap(), 0);

    let keys: Vec<f32> = (0..8).map(|value| value as f32).collect();
    let values: Vec<f32> = (0..8).map(|value| 100.0 + value as f32).collect();
    cache.append(0, &keys, &values).unwrap();
    assert_eq!(cache.len(0).unwrap(), 1);
    assert_eq!(cache.len(1).unwrap(), 0);
    assert_eq!(cache.keys(0).unwrap(), keys.as_slice());
    assert_eq!(cache.values(0).unwrap(), values.as_slice());

    let repeated_keys: Vec<f32> = keys.iter().chain(keys.iter()).copied().collect();
    let repeated_values: Vec<f32> = values.iter().chain(values.iter()).copied().collect();
    cache
        .append_many(0, &repeated_keys, &repeated_values, 2)
        .unwrap();
    assert_eq!(cache.len(0).unwrap(), 3);
    assert_eq!(cache.keys(0).unwrap().len(), 3 * 8);
    assert_eq!(cache.sequence_length(), 3);
}

#[test]
fn capacity_growth_preserves_rows() {
    let mut cache = uniform_cache(1, 2);
    let first: Vec<f32> = (0..8).map(|value| value as f32).collect();
    let second: Vec<f32> = (0..8).map(|value| 50.0 + value as f32).collect();
    cache.append(0, &first, &first).unwrap();
    cache.append(0, &first, &first).unwrap();
    cache.append(0, &second, &second).unwrap();
    assert_eq!(cache.len(0).unwrap(), 3);
    assert_eq!(cache.capacity(), 4);

    cache.reserve(64).unwrap();
    assert_eq!(cache.capacity(), 64);
    assert_eq!(cache.keys(0).unwrap()[..8], first[..]);
    assert_eq!(cache.keys(0).unwrap()[16..24], second[..]);
}

#[test]
fn overflow_and_shape_errors_are_explicit() {
    let mut cache = uniform_cache(1, 1);
    assert!(cache.append(0, &[1.0], &[1.0, 2.0]).is_err());
    assert!(cache.append(4, &[1.0], &[1.0]).is_err());
    cache.append(0, &[1.0; 8], &[1.0; 8]).unwrap();
    // Growth is automatic; a manual reserve below the current capacity is a no-op.
    cache.reserve(0).unwrap();
    assert_eq!(cache.capacity(), 1);
}

#[test]
fn shared_layers_read_the_owner_and_cannot_append() {
    let config = KvCacheConfig::new(
        4,
        vec![
            KvLayerConfig::new(2, 4),
            KvLayerConfig::new(2, 4).shared_with(0),
            KvLayerConfig::new(4, 8),
            KvLayerConfig::new(4, 8).shared_with(2),
        ],
    );
    let mut cache = KvCache::new(config).unwrap();
    assert_eq!(cache.owner(1).unwrap(), 0);
    assert_eq!(cache.owner(2).unwrap(), 2);

    let row_a = vec![0.5f32; 8];
    cache.append(0, &row_a, &row_a).unwrap();
    assert_eq!(cache.keys(1).unwrap(), row_a.as_slice());
    assert!(cache.append(1, &row_a, &row_a).is_err());

    let row_b = vec![0.25f32; 32];
    cache.append(2, &row_b, &row_b).unwrap();
    assert_eq!(cache.len(3).unwrap(), 1);
    assert_eq!(cache.keys(3).unwrap().len(), 32);
}

#[test]
fn invalid_shared_configurations_are_rejected() {
    let forward = KvCacheConfig::new(
        2,
        vec![
            KvLayerConfig::new(1, 2).shared_with(1),
            KvLayerConfig::new(1, 2),
        ],
    );
    assert!(KvCache::new(forward).is_err());

    let chained = KvCacheConfig::new(
        3,
        vec![
            KvLayerConfig::new(1, 2),
            KvLayerConfig::new(1, 2).shared_with(0),
            KvLayerConfig::new(1, 2).shared_with(1),
        ],
    );
    assert!(KvCache::new(chained).is_err());
}

#[test]
fn reset_and_truncate_keep_capacity() {
    let mut cache = uniform_cache(2, 8);
    for _ in 0..3 {
        let row = vec![1.0f32; 8];
        cache.append(0, &row, &row).unwrap();
        cache.append(1, &row, &row).unwrap();
    }
    assert_eq!(cache.sequence_length(), 3);
    cache.truncate(1);
    assert_eq!(cache.sequence_length(), 1);
    cache.reset();
    assert_eq!(cache.sequence_length(), 0);
    assert_eq!(cache.capacity(), 8);
    assert_eq!(ModelState::sequence_length(&cache), 0);
}

#[test]
fn sessions_are_independent() {
    let mut first = uniform_cache(1, 4);
    let mut second = uniform_cache(1, 4);
    let row = vec![7.0f32; 8];
    first.append(0, &row, &row).unwrap();
    assert_eq!(first.sequence_length(), 1);
    assert_eq!(second.sequence_length(), 0);
    second.append(0, &[3.0; 8], &[3.0; 8]).unwrap();
    assert_eq!(first.keys(0).unwrap(), row.as_slice());
    assert_eq!(second.keys(0).unwrap(), [3.0f32; 8]);

    fn assert_send<T: Send>(_: &T) {}
    assert_send(&first);
}
