//! Streaming session tests: prefill/decode equivalence and state isolation.
//!
//! They use the tiny E2B-feature fixture when it is present in `.refs/` and skip
//! otherwise, so the suite still runs in a clean checkout.

use std::path::{Path, PathBuf};

use oxide_ml_gemma::StreamingGemma;

fn fixture() -> Option<PathBuf> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.refs/gemma4-e2b-tiny");
    dir.join("config.json").exists().then_some(dir)
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn argmax(values: &[f32]) -> u32 {
    values
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(index, _)| index as u32)
        .expect("non-empty logits")
}

#[test]
fn prefill_plus_decode_matches_a_full_forward() {
    let Some(dir) = fixture() else {
        eprintln!("skipping: .refs/gemma4-e2b-tiny is absent");
        return;
    };
    let model = StreamingGemma::open(&dir).unwrap();
    let prompt = [5u32, 17, 42];
    let next = 224u32;

    let full = model.forward(&[5, 17, 42, 224]).unwrap();
    let mut session = model.session().unwrap();
    let _ = model.prefill(&mut session, &prompt).unwrap();
    assert_eq!(session.sequence_length(), prompt.len());
    let decoded = model.decode(&mut session, next).unwrap();

    let diff = max_diff(&full, &decoded);
    assert!(
        diff < 1e-5,
        "prefill/decode differs from full forward: {diff:e}"
    );
    assert_eq!(session.sequence_length(), prompt.len() + 1);
}

#[test]
fn independent_sessions_do_not_share_state() {
    let Some(dir) = fixture() else {
        eprintln!("skipping: .refs/gemma4-e2b-tiny is absent");
        return;
    };
    let model = StreamingGemma::open(&dir).unwrap();
    let prompt_a = [5u32, 17, 42];
    let prompt_b = [9u32, 1, 2];

    let mut a = model.session().unwrap();
    let mut b = model.session().unwrap();
    let logits_a = model.prefill(&mut a, &prompt_a).unwrap();
    let logits_b = model.prefill(&mut b, &prompt_b).unwrap();
    assert_eq!(a.sequence_length(), prompt_a.len());
    assert_eq!(b.sequence_length(), prompt_b.len());
    assert_ne!(argmax(&logits_a), argmax(&logits_b));

    let token_a = argmax(&logits_a);
    let token_b = argmax(&logits_b);
    let step_a = model.decode(&mut a, token_a).unwrap();
    let step_b = model.decode(&mut b, token_b).unwrap();

    // Replaying each prompt in a fresh session reproduces both streams exactly:
    // no state leaked between `a` and `b`.
    let mut replay = model.session().unwrap();
    let replay_a = model.prefill(&mut replay, &prompt_a).unwrap();
    assert!(max_diff(&replay_a, &logits_a) < 1e-6);
    let replay_step_a = model.decode(&mut replay, token_a).unwrap();
    assert!(max_diff(&replay_step_a, &step_a) < 1e-6);

    let mut replay = model.session().unwrap();
    let replay_b = model.prefill(&mut replay, &prompt_b).unwrap();
    assert!(max_diff(&replay_b, &logits_b) < 1e-6);
    let replay_step_b = model.decode(&mut replay, token_b).unwrap();
    assert!(max_diff(&replay_step_b, &step_b) < 1e-6);

    // Resetting one session leaves the other untouched.
    a.reset();
    assert_eq!(a.sequence_length(), 0);
    assert_eq!(b.sequence_length(), prompt_b.len() + 1);
}

#[test]
fn concurrent_sessions_share_the_model_but_not_state() {
    let Some(dir) = fixture() else {
        eprintln!("skipping: .refs/gemma4-e2b-tiny is absent");
        return;
    };
    let model = std::sync::Arc::new(StreamingGemma::open(&dir).unwrap());
    let prompts = [[5u32, 17, 42], [9u32, 1, 2]];

    // Sequential reference.
    let mut expected = Vec::new();
    for prompt in &prompts {
        let mut session = model.session().unwrap();
        let logits = model.prefill(&mut session, prompt).unwrap();
        let token = argmax(&logits);
        let step = model.decode(&mut session, token).unwrap();
        expected.push((logits, step));
    }

    // Concurrent sessions on one shared model value.
    let handles: Vec<_> = prompts
        .iter()
        .map(|prompt| {
            let model = std::sync::Arc::clone(&model);
            let prompt = *prompt;
            std::thread::spawn(move || {
                let mut session = model.session().unwrap();
                let logits = model.prefill(&mut session, &prompt).unwrap();
                let token = argmax(&logits);
                let step = model.decode(&mut session, token).unwrap();
                (logits, step)
            })
        })
        .collect();
    for (handle, (expected_logits, expected_step)) in handles.into_iter().zip(expected) {
        let (logits, step) = handle.join().expect("worker thread panicked");
        assert!(max_diff(&logits, &expected_logits) < 1e-6);
        assert!(max_diff(&step, &expected_step) < 1e-6);
    }
}
