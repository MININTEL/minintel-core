//! Machine Intelligence Node - Rust AI Core Tests
//!
//! This module verifies AI inference, multi-threading, and memory efficiency.
//!
//! Author: Machine Intelligence Node Development Team

use minintel_core::model::AIModel;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Instant;

/// Tests AI model inference for correct execution
#[test]
fn test_model_inference() {
    let model = AIModel::new("models/test_model.onnx").expect("Failed to load model");

    let input_data = vec![0.5; 512];
    let result = model.predict(&input_data);

    assert!(result.is_ok(), "Model inference failed");
    assert_eq!(result.unwrap().len(), 10, "Unexpected output size");
}

/// Tests parallel inference using multiple threads
#[test]
fn test_parallel_inference() {
    let model = Arc::new(AIModel::new("models/test_model.onnx").expect("Failed to load model"));
    let mut handles = vec![];

    for _ in 0..8 {
        let model_clone = Arc::clone(&model);
        let handle = thread::spawn(move || {
            let input_data = vec![0.3; 512];
            let result = model_clone.predict(&input_data);
            assert!(result.is_ok(), "Parallel inference failed");
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().expect("Thread panicked");
    }
}

/// Tests inference latency
#[test]
fn test_inference_latency() {
    let model = AIModel::new("models/test_model.onnx").expect("Failed to load model");
    let input_data = vec![0.1; 512];

    let start = Instant::now();
    let result = model.predict(&input_data);
    let duration = start.elapsed();

    assert!(result.is_ok(), "Inference failed");
    assert!(duration.as_millis() < 50, "Inference latency too high: {:?}", duration);
}

/// Tests memory efficiency by running multiple inferences
#[test]
fn test_memory_usage() {
    let model = AIModel::new("models/test_model.onnx").expect("Failed to load model");
    let input_data = vec![0.7; 512];

    let mem_before = sys_info::mem_info().unwrap().free;
    for _ in 0..100 {
        let _ = model.predict(&input_data);
    }
    let mem_after = sys_info::mem_info().unwrap().free;

    assert!(mem_after >= mem_before - 5, "Memory usage increased unexpectedly");
}

/// Tests handling of invalid input
#[test]
#[should_panic]
fn test_invalid_input() {
    let model = AIModel::new("models/test_model.onnx").expect("Failed to load model");
    let input_data = vec![9999.0; 512]; // Out-of-range values
    let _ = model.predict(&input_data);
}
