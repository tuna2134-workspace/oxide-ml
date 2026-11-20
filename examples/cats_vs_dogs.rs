//! Train MobileNetV4 on a cats-vs-dogs subset and export safetensors.
//!
//! Prepare data with `scripts/fetch_cats_dogs.py`, then:
//!
//! ```bash
//! cargo run -p oxide-ml --example cats_vs_dogs --release -- \
//!   --data data/cats_dogs --epochs 8 --batch 16 --export cats_dogs.safetensors
//! ```
//!
//! The loop is explicit (rather than `Trainer`) because CNN kernels require a
//! static batch and because BatchNorm buffers need a post-training calibration
//! pass, which the framework exposes as `calibrate_batch_norm`.

use std::path::{Path, PathBuf};
use std::time::Instant;

use oxide_ml::prelude::*;
use oxide_ml::{calibrate_batch_norm, save_module_safetensors_with_metadata};

const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];

struct Args {
    data: PathBuf,
    epochs: usize,
    batch: usize,
    learning_rate: f32,
    export: Option<PathBuf>,
    calibration_batches: usize,
}

fn parse_args() -> std::result::Result<Args, String> {
    let mut args = Args {
        data: PathBuf::from("data/cats_dogs"),
        epochs: 8,
        batch: 16,
        learning_rate: 3e-4,
        export: Some(PathBuf::from("cats_dogs.safetensors")),
        calibration_batches: 20,
    };
    let mut iter = std::env::args().skip(1);
    while let Some(flag) = iter.next() {
        let mut value = || iter.next().ok_or(format!("{flag} needs a value"));
        match flag.as_str() {
            "--data" => args.data = PathBuf::from(value()?),
            "--epochs" => args.epochs = value()?.parse().map_err(|e| format!("{e}"))?,
            "--batch" => args.batch = value()?.parse().map_err(|e| format!("{e}"))?,
            "--lr" => args.learning_rate = value()?.parse().map_err(|e| format!("{e}"))?,
            "--export" => args.export = Some(PathBuf::from(value()?)),
            "--no-export" => args.export = None,
            "--calibration-batches" => {
                args.calibration_batches = value()?.parse().map_err(|e| format!("{e}"))?
            }
            other => return Err(format!("unknown argument {other:?}")),
        }
    }
    Ok(args)
}

struct Split {
    images: Vec<u8>,
    labels: Vec<u8>,
}

struct Data {
    size: usize,
    train: Split,
    val: Split,
}

fn load_split(dir: &Path, name: &str, size: usize) -> Result<Split> {
    let images = std::fs::read(dir.join(format!("{name}_images.u8")))
        .map_err(|error| Error::Io(format!("read {name}_images.u8: {error}")))?;
    let labels = std::fs::read(dir.join(format!("{name}_labels.u8")))
        .map_err(|error| Error::Io(format!("read {name}_labels.u8: {error}")))?;
    let per_image = 3 * size * size;
    if images.len() != labels.len() * per_image {
        return Err(Error::invalid_shape(
            "cats_dogs dataset",
            format!(
                "{name}: {} images of {per_image} bytes do not match {} labels",
                images.len(),
                labels.len()
            ),
        ));
    }
    Ok(Split { images, labels })
}

fn load_data(dir: &Path) -> Result<Data> {
    let meta = std::fs::read_to_string(dir.join("meta.json"))
        .map_err(|error| Error::Io(format!("read meta.json: {error}")))?;
    let value: serde_json::Value = serde_json::from_str(&meta)
        .map_err(|error| Error::Serialization(format!("meta.json: {error}")))?;
    let size = value
        .get("size")
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| Error::Serialization("meta.json has no size".into()))?
        as usize;
    Ok(Data {
        size,
        train: load_split(dir, "train", size)?,
        val: load_split(dir, "val", size)?,
    })
}

fn batch_tensor(
    split: &Split,
    indices: &[usize],
    size: usize,
    augment: bool,
    flip_seed: u64,
) -> Tensor<f32> {
    let per_image = 3 * size * size;
    let mut values = vec![0.0f32; indices.len() * per_image];
    for (row, index) in indices.iter().enumerate() {
        let flip = augment && ((flip_seed.wrapping_add(*index as u64) & 1) == 1);
        let source = &split.images[index * per_image..(index + 1) * per_image];
        for channel in 0..3 {
            for y in 0..size {
                for x in 0..size {
                    let source_x = if flip { size - 1 - x } else { x };
                    let pixel = source[channel * size * size + y * size + source_x] as f32 / 255.0;
                    values[row * per_image + channel * size * size + y * size + x] =
                        (pixel - MEAN[channel]) / STD[channel];
                }
            }
        }
    }
    Tensor::<f32>::from_vec([indices.len(), 3, size, size], values).unwrap()
}

fn labels_tensor(split: &Split, indices: &[usize]) -> Tensor<f32> {
    Tensor::<f32>::from_vec(
        [indices.len()],
        indices
            .iter()
            .map(|index| split.labels[*index] as f32)
            .collect(),
    )
    .unwrap()
}

fn accuracy(logits: &Tensor<f32>, labels: &Tensor<f32>) -> f32 {
    let classes = logits.dims()[1];
    let mut correct = 0usize;
    for (row, target) in labels.data().iter().enumerate() {
        let values = &logits.data()[row * classes..(row + 1) * classes];
        let predicted = values
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(index, _)| index)
            .unwrap_or(0);
        if (predicted as f32 - target).abs() < 0.5 {
            correct += 1;
        }
    }
    correct as f32 / labels.data().len().max(1) as f32
}

fn evaluate(model: &MobileNetV4, split: &Split, size: usize, batch: usize) -> Result<f32> {
    let compiled = CompiledModel::compile(
        model,
        &[InputSpec::unbatched("input0", [batch, 3, size, size])],
        ModelCompileOptions::default(),
    )?;
    let store = compiled.parameter_store(model)?;
    let mut correct = 0usize;
    let mut seen = 0usize;
    for start in (0..split.labels.len()).step_by(batch) {
        let end = (start + batch).min(split.labels.len());
        let mut indices: Vec<usize> = (start..end).collect();
        while indices.len() < batch {
            indices.push(indices[start % (end - start)]);
        }
        let images = batch_tensor(split, &indices, size, false, 0);
        let expected = labels_tensor(split, &indices);
        let outputs = compiled.run(&store, &[("input0", &images)])?;
        // Count only the real samples in the padded batch.
        let real = end - start;
        let classes = outputs[0].dims()[1];
        for row in 0..real {
            let values = &outputs[0].data()[row * classes..(row + 1) * classes];
            let predicted = values
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(index, _)| index)
                .unwrap_or(0);
            if (predicted as f32 - expected.data()[row]).abs() < 0.5 {
                correct += 1;
            }
        }
        seen += real;
    }
    Ok(correct as f32 / seen.max(1) as f32)
}

fn main() -> Result<()> {
    let args = parse_args().map_err(Error::invalid_graph)?;
    let data = load_data(&args.data)?;
    let size = data.size;
    let classes = 2usize;
    println!(
        "cats_vs_dogs: train={} val={} size={size} batch={} epochs={}",
        data.train.labels.len(),
        data.val.labels.len(),
        args.batch,
        args.epochs
    );

    let config = MobileNetV4Config::nano(classes);
    let mut model = MobileNetV4::new(&config)?;
    println!("parameters: {}", model.parameter_count()?);

    let target = InputSpec::unbatched("target0", [args.batch]);
    let compiled = CompiledModel::compile_training(
        &model,
        &[InputSpec::unbatched("input0", [args.batch, 3, size, size])],
        target,
        &CrossEntropyLoss::new(),
        ModelCompileOptions::training(),
    )?;
    let mut store = compiled.parameter_store(&model)?;
    let mut optimizer = AdamW::new(args.learning_rate)?;
    let mut indices: Vec<usize> = (0..data.train.labels.len()).collect();
    let mut rng = 0x9E37_79B9_7F4A_7C15u64;
    let start = Instant::now();

    for epoch in 0..args.epochs {
        // Deterministic Fisher-Yates shuffle.
        for index in (1..indices.len()).rev() {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let other = (rng as usize) % (index + 1);
            indices.swap(index, other);
        }
        let mut epoch_loss = 0.0f64;
        let mut epoch_correct = 0usize;
        let mut epoch_seen = 0usize;
        let mut batches = 0usize;
        for chunk in indices.chunks(args.batch) {
            if chunk.len() < args.batch {
                break;
            }
            let images = batch_tensor(&data.train, chunk, size, true, epoch as u64);
            let targets = labels_tensor(&data.train, chunk);
            let output =
                compiled.run_training(&store, &[("input0", &images), ("target0", &targets)])?;
            optimizer.step(&mut store, &output.gradients)?;
            epoch_loss += output.loss as f64;
            epoch_correct += (accuracy(&output.outputs[0], &targets) * chunk.len() as f32) as usize;
            epoch_seen += chunk.len();
            batches += 1;
        }
        let train_accuracy = epoch_correct as f32 / epoch_seen.max(1) as f32;
        println!(
            "epoch {}/{}: loss {:.4} train_acc {:.3} ({:.1}s)",
            epoch + 1,
            args.epochs,
            epoch_loss / batches.max(1) as f64,
            train_accuracy,
            start.elapsed().as_secs_f32()
        );
    }

    compiled.write_back(&mut model, &store)?;
    // Calibrate BatchNorm buffers on real training batches before evaluation.
    let calibration: Vec<Vec<Tensor<f32>>> = (0..args
        .calibration_batches
        .min(data.train.labels.len().div_ceil(args.batch)))
        .map(|index| {
            let begin = index * args.batch;
            let slice: Vec<usize> = (begin..begin + args.batch).collect();
            vec![batch_tensor(&data.train, &slice, size, false, 0)]
        })
        .collect();
    if !calibration.is_empty() {
        let report = calibrate_batch_norm(
            &mut model,
            &[InputSpec::unbatched("input0", [3, size, size])],
            &calibration,
            0.3,
        )?;
        println!(
            "batch-norm calibration: {} layers, {} samples",
            report.layers, report.samples
        );
    }
    model.set_training(false);
    let train_accuracy = evaluate(&model, &data.train, size, args.batch)?;
    let val_accuracy = evaluate(&model, &data.val, size, args.batch)?;
    println!("train accuracy: {train_accuracy:.4}");
    println!("val accuracy:   {val_accuracy:.4}");

    if let Some(path) = &args.export {
        let metadata = [
            ("model.architecture", "MobileNetV4-nano"),
            ("model.classes", "cat,dog"),
            ("model.input_size", if size == 96 { "96" } else { "custom" }),
        ];
        let report = save_module_safetensors_with_metadata(&model, path, &metadata)?;
        println!(
            "exported {} tensors ({:.1} MiB, {} aliases) to {}",
            report.tensors,
            report.bytes as f64 / (1024.0 * 1024.0),
            report.aliases.len(),
            path.display()
        );
    }
    Ok(())
}
