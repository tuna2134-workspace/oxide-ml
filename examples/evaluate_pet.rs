//! Evaluate a MobileNetV4 cats/dogs safetensors checkpoint.
//!
//! ```bash
//! cargo run -p oxide-ml --release --example evaluate_pet -- \
//!   --data data/cats_dogs --model data/cats_dogs/mobilenetv4_nano.safetensors
//! ```

use std::path::PathBuf;

use oxide_ml::load_module_safetensors;
use oxide_ml::prelude::*;
use oxide_ml::safetensors::SafeTensorsFile;

const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];

fn main() -> Result<()> {
    let mut data = PathBuf::from("data/cats_dogs");
    let mut model_path = data.join("mobilenetv4_nano.safetensors");
    let mut reference = None;
    let mut split = "val".to_string();
    let mut limit: Option<usize> = None;
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--data" => {
                data = PathBuf::from(
                    args.next()
                        .ok_or_else(|| Error::invalid_graph("--data needs a path"))?,
                )
            }
            "--model" => {
                model_path = PathBuf::from(
                    args.next()
                        .ok_or_else(|| Error::invalid_graph("--model needs a path"))?,
                )
            }
            "--reference" => {
                reference =
                    Some(PathBuf::from(args.next().ok_or_else(|| {
                        Error::invalid_graph("--reference needs a path")
                    })?))
            }
            "--split" => {
                split = args
                    .next()
                    .ok_or_else(|| Error::invalid_graph("--split needs a name"))?
            }
            "--limit" => {
                limit = Some(
                    args.next()
                        .ok_or_else(|| Error::invalid_graph("--limit needs a count"))?
                        .parse()
                        .map_err(|_| Error::invalid_graph("invalid --limit"))?,
                )
            }
            other => return Err(Error::invalid_graph(format!("unknown argument {other:?}"))),
        }
    }

    let meta: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(data.join("meta.json"))
            .map_err(|error| Error::Io(format!("meta.json: {error}")))?,
    )?;
    let size = meta["size"].as_u64().unwrap_or(96) as usize;
    let images = std::fs::read(data.join(format!("{split}_images.u8")))
        .map_err(|error| Error::Io(format!("val_images.u8: {error}")))?;
    let labels = std::fs::read(data.join(format!("{split}_labels.u8")))
        .map_err(|error| Error::Io(format!("val_labels.u8: {error}")))?;
    let per_image = 3 * size * size;
    let count = limit.unwrap_or(labels.len()).min(labels.len());
    let images = &images[..count * per_image];
    if images.len() != count * per_image {
        return Err(Error::invalid_shape(
            "evaluate_pet",
            "dataset sizes disagree",
        ));
    }
    let mut values = vec![0.0f32; count * per_image];
    for sample in 0..count {
        for channel in 0..3 {
            for pixel in 0..size * size {
                let raw = images[sample * per_image + channel * size * size + pixel] as f32 / 255.0;
                values[sample * per_image + channel * size * size + pixel] =
                    (raw - MEAN[channel]) / STD[channel];
            }
        }
    }
    let input = Tensor::<f32>::from_vec([count, 3, size, size], values)?;

    let file = SafeTensorsFile::open(&model_path)?;
    let classes = file
        .metadata()
        .get("model.classes")
        .map(|value| value.split(',').count())
        .unwrap_or(2);
    let architecture = file
        .metadata()
        .get("model.architecture")
        .map(String::as_str)
        .unwrap_or("MobileNetV4-nano");
    let config = if architecture.contains("conv_small") {
        MobileNetV4Config::conv_small(classes)
    } else {
        MobileNetV4Config::nano(classes)
    };
    let mut model = MobileNetV4::new(&config)?;
    let load = load_module_safetensors(&mut model, &model_path)?;
    if !load.missing.is_empty() {
        return Err(Error::invalid_graph(format!(
            "checkpoint misses {} tensors",
            load.missing.len()
        )));
    }
    model.set_training(false);
    let logits = model.predict(input)?;
    let mut correct = 0usize;
    for (sample, target) in labels.iter().take(count).enumerate() {
        let row = &logits.data()[sample * classes..(sample + 1) * classes];
        let predicted = row
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(index, _)| index)
            .unwrap_or(0);
        if predicted == *target as usize {
            correct += 1;
        }
    }
    let accuracy = correct as f32 / count.max(1) as f32;
    println!(
        "checkpoint: {} ({} tensors)",
        model_path.display(),
        load.loaded
    );
    println!("{split} accuracy: {accuracy:.4} ({correct}/{count})");

    if let Some(reference) = reference {
        let reference: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(&reference)
                .map_err(|error| Error::Io(format!("reference: {error}")))?,
        )?;
        let expected = reference["logits"]
            .as_array()
            .ok_or_else(|| Error::Serialization("reference has no logits".into()))?;
        let mut max_diff = 0.0f32;
        for (sample, row) in expected.iter().enumerate() {
            if sample >= count {
                break;
            }
            let Some(row) = row.as_array() else { continue };
            for (index, expected) in row.iter().enumerate() {
                let expected = expected.as_f64().unwrap_or(0.0) as f32;
                let actual = logits.data()[sample * classes + index];
                max_diff = max_diff.max((actual - expected).abs());
            }
        }
        println!("reference max |diff| (8 samples): {max_diff:.3e}");
    }
    Ok(())
}
