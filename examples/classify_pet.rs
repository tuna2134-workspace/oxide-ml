//! Classify a cat/dog image with a MobileNetV4 safetensors checkpoint.
//!
//! ```bash
//! cargo run -p oxide-ml --example classify_pet -- \
//!   --model cats_dogs.safetensors --image cat.jpg
//! ```

use std::path::PathBuf;

use oxide_ml::load_module_safetensors;
use oxide_ml::prelude::*;
use oxide_ml::safetensors::SafeTensorsFile;

const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];

fn parse() -> std::result::Result<(PathBuf, PathBuf), String> {
    let mut model = None;
    let mut image = None;
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--model" => model = Some(PathBuf::from(args.next().ok_or("--model needs a path")?)),
            "--image" => image = Some(PathBuf::from(args.next().ok_or("--image needs a path")?)),
            other => return Err(format!("unknown argument {other:?}")),
        }
    }
    Ok((
        model.ok_or("--model is required")?,
        image.ok_or("--image is required")?,
    ))
}

fn main() -> Result<()> {
    let (model_path, image_path) = parse().map_err(Error::invalid_graph)?;
    let file = SafeTensorsFile::open(&model_path)?;
    let metadata = file.metadata();
    let classes_meta = metadata
        .get("model.classes")
        .map(String::as_str)
        .unwrap_or("cat,dog");
    let classes: Vec<&str> = classes_meta.split(',').collect();
    let size = metadata
        .get("model.input_size")
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(96);

    let architecture = metadata
        .get("model.architecture")
        .map(String::as_str)
        .unwrap_or("MobileNetV4-nano");
    let config = if architecture.contains("conv_small") {
        MobileNetV4Config::conv_small(classes.len())
    } else {
        MobileNetV4Config::nano(classes.len())
    };
    let mut model = MobileNetV4::new(&config)?;
    let report = load_module_safetensors(&mut model, &model_path)?;
    if !report.missing.is_empty() {
        return Err(Error::invalid_graph(format!(
            "checkpoint misses {} parameters: {:?}",
            report.missing.len(),
            &report.missing[..report.missing.len().min(5)]
        )));
    }
    model.set_training(false);

    let source = image::open(&image_path)
        .map_err(|error| Error::Io(format!("open {}: {error}", image_path.display())))?
        .resize_exact(
            size as u32,
            size as u32,
            image::imageops::FilterType::Lanczos3,
        )
        .to_rgb8();
    let mut values = vec![0.0f32; 3 * size * size];
    for (x, y, pixel) in source.enumerate_pixels() {
        for channel in 0..3 {
            let value = pixel[channel] as f32 / 255.0;
            values[channel * size * size + y as usize * size + x as usize] =
                (value - MEAN[channel]) / STD[channel];
        }
    }
    let input = Tensor::<f32>::from_vec([1, 3, size, size], values)?;
    let logits = model.predict(input)?;
    let row = logits.data();
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = row.iter().map(|value| (value - max).exp()).collect();
    let sum: f32 = exp.iter().sum();
    let mut ranked: Vec<(usize, f32)> = exp
        .iter()
        .enumerate()
        .map(|(index, value)| (index, value / sum))
        .collect();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
    println!("image: {}", image_path.display());
    for (index, probability) in ranked {
        let name = classes.get(index).copied().unwrap_or("class");
        println!("  {name:<8} {probability:.4}");
    }
    Ok(())
}
