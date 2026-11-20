//! Paged KV cache: block tables, gather, capacity, sharing and truncation.

use oxide_ml::prelude::*;

fn cache(layers: usize, block_size: usize, blocks: usize) -> PagedKvCache {
    PagedKvCache::new(PagedKvConfig::uniform(block_size, blocks, layers, 2, 2)).unwrap()
}

fn row(seed: f32) -> Vec<f32> {
    (0..4).map(|index| seed + index as f32 * 0.1).collect()
}

#[test]
fn append_across_block_boundaries_and_gather_in_order() {
    let mut cache = cache(1, 2, 3);
    assert_eq!(cache.block_size(), 2);
    assert_eq!(cache.capacity_tokens(), 6);
    let tokens: Vec<Vec<f32>> = (0..5).map(|index| row(index as f32)).collect();
    for token in &tokens {
        cache.append(0, token, token).unwrap();
    }
    assert_eq!(cache.len(0).unwrap(), 5);
    assert_eq!(cache.block_table(0).unwrap(), &[0, 1, 2]);

    let keys = cache.gather_keys(0).unwrap();
    let values = cache.gather_values(0).unwrap();
    let expected: Vec<f32> = tokens.iter().flatten().copied().collect();
    assert_eq!(keys, expected);
    assert_eq!(values, expected);
    // Physical pool holds the block layout, not logical order.
    assert_eq!(cache.physical_keys(0).unwrap().dims(), vec![6, 2, 2]);
}

#[test]
fn capacity_is_explicit_and_reset_reuses_blocks() {
    let mut cache = cache(1, 2, 1);
    cache.append(0, &row(0.0), &row(0.0)).unwrap();
    cache.append(0, &row(1.0), &row(1.0)).unwrap();
    let error = cache.append(0, &row(2.0), &row(2.0)).unwrap_err();
    assert!(error.to_string().contains("capacity"), "{error}");

    cache.reset();
    assert!(cache.is_empty());
    assert!(cache.block_table(0).unwrap().is_empty());
    cache.append(0, &row(5.0), &row(5.0)).unwrap();
    assert_eq!(cache.len(0).unwrap(), 1);
    assert_eq!(cache.block_table(0).unwrap(), &[0]);
    assert_eq!(cache.gather_keys(0).unwrap(), row(5.0));
}

#[test]
fn truncate_releases_trailing_blocks_and_keeps_prefix() {
    let mut cache = cache(1, 2, 4);
    for index in 0..6 {
        cache
            .append(0, &row(index as f32), &row(index as f32))
            .unwrap();
    }
    cache.truncate(3);
    assert_eq!(cache.len(0).unwrap(), 3);
    assert_eq!(cache.block_table(0).unwrap().len(), 2);
    assert_eq!(cache.gather_keys(0).unwrap()[..4], row(0.0)[..]);
    // Appending after truncation reuses the freed physical block.
    cache.append(0, &row(9.0), &row(9.0)).unwrap();
    assert_eq!(cache.len(0).unwrap(), 4);
    assert_eq!(cache.block_table(0).unwrap().len(), 2);
}

#[test]
fn shared_layers_read_the_owner_and_reject_appends() {
    let config = PagedKvConfig::new(
        4,
        2,
        vec![
            KvLayerConfig::new(2, 2),
            KvLayerConfig::new(2, 2).shared_with(0),
            KvLayerConfig::new(1, 2),
            KvLayerConfig::new(1, 2).shared_with(2),
        ],
    );
    let mut cache = PagedKvCache::new(config).unwrap();
    let token = row(1.0);
    cache.append(0, &token, &token).unwrap();
    assert_eq!(cache.len(1).unwrap(), 1);
    assert_eq!(cache.gather_keys(1).unwrap(), token);
    assert!(cache.append(1, &token, &token).is_err());

    let small = vec![1.0f32, 2.0];
    cache.append(2, &small, &small).unwrap();
    assert_eq!(cache.gather_values(3).unwrap(), small);
}

#[test]
fn invalid_layouts_and_shapes_are_rejected() {
    let zero_block = PagedKvConfig::uniform(0, 1, 1, 1, 1);
    assert!(PagedKvCache::new(zero_block).is_err());
    let forward_share = PagedKvConfig::new(
        2,
        1,
        vec![
            KvLayerConfig::new(1, 1).shared_with(1),
            KvLayerConfig::new(1, 1),
        ],
    );
    assert!(PagedKvCache::new(forward_share).is_err());

    let mut cache = cache(1, 2, 1);
    assert!(cache.append(0, &[1.0], &[1.0]).is_err());
    assert!(cache.append(3, &row(0.0), &row(0.0)).is_err());
}

#[test]
fn sessions_are_send_and_independent() {
    let mut first = cache(1, 2, 2);
    let mut second = cache(1, 2, 2);
    first.append(0, &row(1.0), &row(1.0)).unwrap();
    second.append(0, &row(2.0), &row(2.0)).unwrap();
    assert_eq!(first.gather_keys(0).unwrap(), row(1.0));
    assert_eq!(second.gather_keys(0).unwrap(), row(2.0));
    first.reset();
    assert!(first.is_empty());
    assert_eq!(second.len(0).unwrap(), 1);

    fn assert_send<T: Send>(_: &T) {}
    assert_send(&first);
    assert_eq!(ModelState::sequence_length(&second), 1);
}
