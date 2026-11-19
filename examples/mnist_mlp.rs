//! MNIST MLP training example.
//!
//! This intentionally uses the raw IDX files (no image crate, no Python
//! runtime) so the example exercises the same Rust graph executor as a test.
//! Run `python3 scripts/fetch_mnist.py` once, then:
//!
//! ```text
//! cargo run --release --example mnist_mlp -- data/mnist 3
//! ```

use std::path::Path;

use oxide_ml::legacy::{Mlp, Trainer};
use oxide_ml::prelude::*;

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let data_dir = args.next().unwrap_or_else(|| "data/mnist".to_string());
    let epochs: usize = args
        .next()
        .and_then(|value| value.parse().ok())
        .unwrap_or(3);
    let limit: usize = args
        .next()
        .and_then(|value| value.parse().ok())
        .unwrap_or(10_000);

    let train_images = Path::new(&data_dir).join("train-images-idx3-ubyte");
    let train_labels = Path::new(&data_dir).join("train-labels-idx1-ubyte");
    let test_images = Path::new(&data_dir).join("t10k-images-idx3-ubyte");
    let test_labels = Path::new(&data_dir).join("t10k-labels-idx1-ubyte");
    if !train_images.exists() || !train_labels.exists() {
        eprintln!(
            "MNIST files not found in {data_dir}. Run:\n  python3 scripts/fetch_mnist.py {data_dir}"
        );
        return Ok(());
    }
    let (train_count, rows, cols, train_x) = read_images(&train_images)?;
    let (_, train_y) = read_labels(&train_labels)?;
    let (test_count, _, _, test_x) = read_images(&test_images)?;
    let (_, test_y) = read_labels(&test_labels)?;
    assert_eq!(rows * cols, 28 * 28);
    let train_count = train_count.min(limit).min(train_y.len());
    let test_count = test_count.min(2_000).min(test_y.len());
    println!(
        "MNIST: {} train / {} test images, {}x{}",
        train_count, test_count, rows, cols
    );

    let mut builder = GraphBuilder::new();
    let batch = builder.symbol("batch");
    let x = builder.input(
        "x",
        Shape::new(vec![Dim::Symbol(batch), Dim::Static(rows * cols)]),
        DType::F32,
    );
    let target = builder.input("target", Shape::new(vec![Dim::Symbol(batch)]), DType::F32);
    let mlp = Mlp::new(
        &mut builder,
        "mlp",
        &[rows * cols, 128, 10],
        ActivationKind::ReLU,
        true,
    )?;
    let logits = mlp.forward(&mut builder, x)?;
    let loss = builder.cross_entropy_loss(logits, target, Reduction::Mean)?;
    let forward = builder.finish(&[logits, loss])?;
    let training = build_training_graph(&forward, loss)?;
    let parameters = init_parameters::<f32>(&forward, 0xC0FFEE)?;
    let optimizer = AdamW::with_config(1e-3, 0.9, 0.999, 1e-8, 1e-4)?;
    let mut trainer = Trainer::new(training, parameters, Box::new(optimizer))?;
    let mut rng = Rng::new(0xBADC0DE);

    let batch_size = 64;
    for epoch in 0..epochs {
        let mut order: Vec<usize> = (0..train_count).collect();
        rng.shuffle(&mut order);
        let mut total_loss = 0.0;
        let mut batches = 0;
        for chunk in order.chunks(batch_size) {
            let mut xs = Vec::with_capacity(chunk.len() * rows * cols);
            let mut ys = Vec::with_capacity(chunk.len());
            for &index in chunk {
                xs.extend_from_slice(&train_x[index * rows * cols..(index + 1) * rows * cols]);
                ys.push(train_y[index] as f32);
            }
            let mut inputs = ExecutionInputs::<f32>::new();
            inputs.insert(
                forward.inputs()[0],
                DenseTensor::from_vec([chunk.len(), rows * cols], xs)?,
            );
            inputs.insert(
                forward.inputs()[1],
                DenseTensor::from_vec([chunk.len()], ys)?,
            );
            total_loss += trainer.step(&inputs)?;
            batches += 1;
        }
        println!(
            "epoch {}: mean loss = {:.4}",
            epoch + 1,
            total_loss / batches as f32
        );
    }

    let mut correct = 0usize;
    for start in (0..test_count).step_by(512) {
        let end = (start + 512).min(test_count);
        let count = end - start;
        let xs = test_x[start * rows * cols..end * rows * cols].to_vec();
        let ys: Vec<f32> = test_y[start..end]
            .iter()
            .map(|&value| value as f32)
            .collect();
        let mut inputs = ExecutionInputs::<f32>::new();
        inputs.insert(
            forward.inputs()[0],
            DenseTensor::from_vec([count, rows * cols], xs)?,
        );
        inputs.insert(forward.inputs()[1], DenseTensor::from_vec([count], ys)?);
        let result = trainer.run(&inputs)?;
        let logits = result.output(0)?;
        for row in 0..count {
            let mut best = 0usize;
            for class in 1..10 {
                if logits.data()[row * 10 + class] > logits.data()[row * 10 + best] {
                    best = class;
                }
            }
            if best == test_y[start + row] as usize {
                correct += 1;
            }
        }
    }
    println!(
        "test accuracy: {:.2}% ({}/{})",
        correct as f32 / test_count as f32 * 100.0,
        correct,
        test_count
    );
    Ok(())
}

fn read_u32(data: &[u8], offset: &mut usize) -> Result<u32> {
    if *offset + 4 > data.len() {
        return Err(Error::execution("truncated IDX header"));
    }
    let value = u32::from_be_bytes(data[*offset..*offset + 4].try_into().unwrap());
    *offset += 4;
    Ok(value)
}

fn read_images(path: &Path) -> Result<(usize, usize, usize, Vec<f32>)> {
    let data = std::fs::read(path)?;
    let mut offset = 0;
    let magic = read_u32(&data, &mut offset)?;
    if magic != 0x0000_0803 {
        return Err(Error::execution(format!(
            "{} is not an IDX image file (magic {magic:#x})",
            path.display()
        )));
    }
    let count = read_u32(&data, &mut offset)? as usize;
    let rows = read_u32(&data, &mut offset)? as usize;
    let cols = read_u32(&data, &mut offset)? as usize;
    let expected = count * rows * cols;
    if data.len() < offset + expected {
        return Err(Error::execution("truncated IDX image payload"));
    }
    let pixels = data[offset..offset + expected]
        .iter()
        .map(|&value| value as f32 / 255.0)
        .collect();
    Ok((count, rows, cols, pixels))
}

fn read_labels(path: &Path) -> Result<(usize, Vec<u8>)> {
    let data = std::fs::read(path)?;
    let mut offset = 0;
    let magic = read_u32(&data, &mut offset)?;
    if magic != 0x0000_0801 {
        return Err(Error::execution(format!(
            "{} is not an IDX label file (magic {magic:#x})",
            path.display()
        )));
    }
    let count = read_u32(&data, &mut offset)? as usize;
    if data.len() < offset + count {
        return Err(Error::execution("truncated IDX label payload"));
    }
    Ok((count, data[offset..offset + count].to_vec()))
}
