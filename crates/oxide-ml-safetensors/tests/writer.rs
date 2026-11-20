//! Safetensors writer: layout, dtypes, metadata and round trips.

use oxide_ml_safetensors::{SafeTensorsFile, SafeTensorsWriter, StoredDType, TensorRecord};

#[test]
fn writes_and_rereads_f32_tensors_with_metadata() {
    let path = std::env::temp_dir().join(format!(
        "oxide-writer-{}-{}.safetensors",
        std::process::id(),
        1
    ));
    let mut writer = SafeTensorsWriter::new();
    writer.set_metadata("format", "oxide-ml.safetensors");
    writer.set_metadata("note", "hello");
    writer
        .add_f32("a.weight", vec![2, 2], &[1.0, 2.0, 3.0, 4.0])
        .unwrap();
    writer
        .add_tensor(TensorRecord::f16("b", vec![2], &[0.5, -0.25]).unwrap())
        .unwrap();
    writer.write(&path).unwrap();

    let file = SafeTensorsFile::open(&path).unwrap();
    assert_eq!(file.len(), 2);
    assert_eq!(
        file.metadata().get("format").unwrap(),
        "oxide-ml.safetensors"
    );
    assert_eq!(file.metadata().get("note").unwrap(), "hello");
    assert_eq!(file.dtype("a.weight"), Some(StoredDType::F32));
    assert_eq!(file.dtype("b"), Some(StoredDType::F16));
    assert_eq!(
        file.tensor("a.weight").unwrap().data,
        vec![1.0, 2.0, 3.0, 4.0]
    );
    assert_eq!(file.tensor("b").unwrap().data, vec![0.5, -0.25]);
    std::fs::remove_file(&path).unwrap();
}

#[test]
fn duplicate_names_are_rejected() {
    let mut writer = SafeTensorsWriter::new();
    writer.add_f32("w", vec![1], &[1.0]).unwrap();
    let error = writer.add_f32("w", vec![1], &[2.0]).unwrap_err();
    assert!(error.to_string().contains("duplicate"), "{error}");
}

#[test]
fn shape_and_byte_count_are_validated() {
    let record = TensorRecord::f32("bad", vec![3, 3], &[1.0, 2.0]);
    assert!(record.is_err());
    let writer = SafeTensorsWriter::new();
    assert!(writer.to_bytes().is_err(), "empty files are rejected");
}

#[test]
fn bf16_round_trip() {
    let path = std::env::temp_dir().join(format!(
        "oxide-writer-{}-{}.safetensors",
        std::process::id(),
        2
    ));
    let mut writer = SafeTensorsWriter::new();
    writer
        .add_tensor(TensorRecord::bf16("x", vec![3], &[0.5, 1.5, -2.0]).unwrap())
        .unwrap();
    writer.write(&path).unwrap();
    let file = SafeTensorsFile::open(&path).unwrap();
    assert_eq!(file.dtype("x"), Some(StoredDType::BF16));
    assert_eq!(file.tensor("x").unwrap().data, vec![0.5, 1.5, -2.0]);
    std::fs::remove_file(&path).unwrap();
}
