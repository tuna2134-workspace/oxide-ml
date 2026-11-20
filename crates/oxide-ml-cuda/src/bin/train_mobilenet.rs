//! Train MobileNetV4 (cats vs dogs) on CUDA and export safetensors.
//!
//! ```bash
//! cargo oxide run --arch sm_75 --bin train_mobilenet -- \
//!   --data data/cats_dogs --arch conv_small --size 128 --epochs 20 --batch 64
//! ```

use std::path::PathBuf;
use std::time::Instant;

use oxide_ml_core::DenseTensor;
use oxide_ml_model::{InputSpec, Module, ModulePath, ParameterRegistry, calibrate_batch_norm};
use oxide_ml_nn::vision::{MobileNetV4, MobileNetV4Config};
use oxide_ml_nn::CrossEntropyLoss;
use oxide_ml_safetensors::SafeTensorsWriter;

use oxide_ml_cuda::{AdamWParams, CudaAdamW, CudaDevice, CudaModel, CudaTrainingModel};

const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];

struct Args {
    data: PathBuf,
    arch: String,
    size: usize,
    epochs: usize,
    batch: usize,
    lr: f32,
    export: PathBuf,
    calibration_batches: usize,
}

fn parse() -> std::result::Result<Args, String> {
    let mut args = Args {
        data: PathBuf::from("data/cats_dogs"),
        arch: "conv_small".into(),
        size: 128,
        epochs: 20,
        batch: 64,
        lr: 1.5e-3,
        export: PathBuf::from("data/cats_dogs/mobilenetv4_cuda.safetensors"),
        calibration_batches: 2,
    };
    let mut iter = std::env::args().skip(1);
    while let Some(flag) = iter.next() {
        let mut value = || iter.next().ok_or(format!("{flag} needs a value"));
        match flag.as_str() {
            "--data" => args.data = PathBuf::from(value()?),
            "--arch" => args.arch = value()?,
            "--size" => args.size = value()?.parse().map_err(|e| format!("{e}"))?,
            "--epochs" => args.epochs = value()?.parse().map_err(|e| format!("{e}"))?,
            "--batch" => args.batch = value()?.parse().map_err(|e| format!("{e}"))?,
            "--lr" => args.lr = value()?.parse().map_err(|e| format!("{e}"))?,
            "--export" => args.export = PathBuf::from(value()?),
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

fn load_split(dir: &PathBuf, name: &str, size: usize) -> Result<Split, String> {
    let images = std::fs::read(dir.join(format!("{name}_images.u8")))
        .map_err(|error| format!("{name}_images.u8: {error}"))?;
    let labels = std::fs::read(dir.join(format!("{name}_labels.u8")))
        .map_err(|error| format!("{name}_labels.u8: {error}"))?;
    let per_image = 3 * size * size;
    if images.len() != labels.len() * per_image {
        return Err(format!(
            "{name}: {} bytes for {} labels, expected {}",
            images.len(),
            labels.len(),
            labels.len() * per_image
        ));
    }
    Ok(Split { images, labels })
}

fn batch(split: &Split, indices: &[usize], size: usize) -> DenseTensor<f32> {
    let per = 3 * size * size;
    let mut values = vec![0.0f32; indices.len() * per];
    for (row, index) in indices.iter().enumerate() {
        for channel in 0..3 {
            for pixel in 0..size * size {
                let raw = split.images[index * per + channel * size * size + pixel] as f32 / 255.0;
                values[row * per + channel * size * size + pixel] =
                    (raw - MEAN[channel]) / STD[channel];
            }
        }
    }
    DenseTensor::from_vec([indices.len(), 3, size, size], values).expect("shape agrees")
}

fn targets(labels: &[u8]) -> DenseTensor<f32> {
    DenseTensor::from_vec(
        [labels.len()],
        labels.iter().map(|value| *value as f32).collect(),
    )
    .expect("shape agrees")
}

fn evaluate(
    device: CudaDevice,
    model: &MobileNetV4,
    split: &Split,
    size: usize,
    batch_size: usize,
) -> Result<f32, String> {
    let compiled = CudaModel::compile(
        device,
        model,
        &[InputSpec::unbatched(
            "input0",
            [batch_size, 3, size, size],
        )],
        oxide_ml_model::ModelCompileOptions::default(),
    )
    .map_err(|error| error.to_string())?;
    let mut correct = 0usize;
    let mut seen = 0usize;
    for start in (0..split.labels.len()).step_by(batch_size) {
        let end = (start + batch_size).min(split.labels.len());
        let mut indices: Vec<usize> = (start..end).collect();
        while indices.len() < batch_size {
            indices.push(indices[start % (end - start)]);
        }
        let images = batch(split, &indices, size);
        let outputs = compiled
            .run(&[("input0", images)])
            .map_err(|error| error.to_string())?;
        let real = end - start;
        for row in 0..real {
            let logits = outputs[0].data();
            let predicted = logits[row * 2..(row + 1) * 2]
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(index, _)| index)
                .unwrap_or(0);
            if predicted == split.labels[start + row] as usize {
                correct += 1;
            }
        }
        seen += real;
    }
    Ok(correct as f32 / seen.max(1) as f32)
}

fn export(model: &MobileNetV4, path: &PathBuf, metadata: &[(&str, &str)]) -> Result<(), String> {
    let mut writer = SafeTensorsWriter::new();
    for (key, value) in metadata {
        writer.set_metadata(*key, *value);
    }
    writer.set_metadata("format", "oxide-ml.safetensors");
    writer.set_metadata("module.type", model.module_type());
    let mut registry = ParameterRegistry::new();
    model
        .visit_parameters(&ModulePath::root(), &mut registry)
        .map_err(|error| error.to_string())?;
    let mut count = 0usize;
    model
        .visit_tensors(&ModulePath::root(), &mut |path, tensor, role| {
            writer
                .set_metadata(format!("role.{}", path.as_string()), role.as_str().to_string());
            writer
                .add_f32(path.as_string(), tensor.dims(), tensor.data())
                .map_err(|error| oxide_ml_core::Error::Serialization(error.to_string()))?;
            count += 1;
            Ok(())
        })
        .map_err(|error| error.to_string())?;
    writer
        .write(path)
        .map_err(|error| error.to_string())?;
    println!("exported {count} tensors to {}", path.display());
    Ok(())
}

fn main() -> Result<(), String> {
    let args = parse()?;
    let train = load_split(&args.data, "train", args.size)?;
    let val = load_split(&args.data, "val", args.size)?;
    println!(
        "data: train={} val={} size={} batch={}",
        train.labels.len(),
        val.labels.len(),
        args.size,
        args.batch
    );
    let config = if args.arch == "nano" {
        MobileNetV4Config::nano(2)
    } else {
        MobileNetV4Config::conv_small(2)
    };
    let mut model = MobileNetV4::new(&config).map_err(|error| error.to_string())?;
    println!("parameters: {}", model.parameter_count().map_err(|e| e.to_string())?);

    let device = CudaDevice::new(0).map_err(|error| error.to_string())?;
    let mut training = CudaTrainingModel::new(
        device.clone(),
        &model,
        &[InputSpec::unbatched(
            "input0",
            [args.batch, 3, args.size, args.size],
        )],
        InputSpec::unbatched("target0", [args.batch]),
        &CrossEntropyLoss::new(),
        Box::new(
            CudaAdamW::new(AdamWParams {
                learning_rate: args.lr,
                beta1: 0.9,
                beta2: 0.999,
                eps: 1e-8,
                weight_decay: 0.01,
            })
            .map_err(|error| error.to_string())?,
        ),
    )
    .map_err(|error| error.to_string())?;

    let mut indices: Vec<usize> = (0..train.labels.len()).collect();
    let mut rng = 0x9E37_79B9_7F4A_7C15u64;
    let started = Instant::now();
    for epoch in 0..args.epochs {
        for index in (1..indices.len()).rev() {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let other = (rng as usize) % (index + 1);
            indices.swap(index, other);
        }
        let epoch_start = Instant::now();
        let mut loss_sum = 0.0f64;
        let mut batches = 0usize;
        for chunk in indices.chunks(args.batch) {
            if chunk.len() < args.batch {
                break;
            }
            let images = batch(&train, chunk, args.size);
            let labels: Vec<u8> = chunk.iter().map(|index| train.labels[*index]).collect();
            let target = targets(&labels);
            let loss = training
                .step(&[("input0", images.clone())], &target)
                .map_err(|error| error.to_string())?;
            loss_sum += loss as f64;
            batches += 1;
        }
        println!(
            "epoch {}/{}: loss {:.4} ({:.2}s, total {:.1}s)",
            epoch + 1,
            args.epochs,
            loss_sum / batches.max(1) as f64,
            epoch_start.elapsed().as_secs_f32(),
            started.elapsed().as_secs_f32()
        );
    }

    training
        .sync_to_model(&mut model)
        .map_err(|error| error.to_string())?;

    // The reference executor does not mutate BatchNorm running buffers; fold
    // batch statistics into the host module, then upload for evaluation.
    let calibration: Vec<Vec<oxide_ml_model::Tensor<f32>>> = (0..args
        .calibration_batches
        .min(train.labels.len().div_ceil(args.batch)))
        .map(|index| {
            let begin = index * args.batch;
            let indices: Vec<usize> = (begin..begin + args.batch).collect();
            vec![oxide_ml_model::Tensor::from_dense(batch(&train, &indices, args.size))]
        })
        .collect();
    if !calibration.is_empty() {
        let report = calibrate_batch_norm(
            &mut model,
            &[InputSpec::unbatched("input0", [3, args.size, args.size])],
            &calibration,
            0.3,
        )
        .map_err(|error| error.to_string())?;
        println!(
            "batch-norm calibration: {} layers, {} samples",
            report.layers, report.samples
        );
    }
    model.set_training(false);
    let val_accuracy = evaluate(device.clone(), &model, &val, args.size, args.batch)?;
    println!("val accuracy: {val_accuracy:.4}");
    export(
        &model,
        &args.export,
        &[
            ("model.architecture", &format!("MobileNetV4-{}", args.arch)),
            ("model.classes", "cat,dog"),
            ("model.input_size", &args.size.to_string()),
            ("training.backend", "oxide-ml-cuda"),
        ],
    )?;
    Ok(())
}
