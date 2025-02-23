//! Machine Intelligence Node - Model Exporter
//!
//! Provides functionality for exporting trained models to ONNX and TorchScript formats,
//! allowing deployment across multiple platforms including cloud, edge, and web environments.
//!
//! Author: Machine Intelligence Node Development Team

use tch::{CModule, nn, Device, Tensor};
use onnxruntime::{environment::Environment, session::Session, tensor::OrtTensor};
use std::path::Path;
use std::error::Error;

/// ModelExporter struct for handling various model export formats.
pub struct ModelExporter {
    model: nn::Module,
}

impl ModelExporter {
    /// Creates a new ModelExporter instance.
    ///
    /// # Arguments
    /// * `model` - A trained PyTorch model instance to be exported.
    pub fn new(model: nn::Module) -> Self {
        ModelExporter { model }
    }

    /// Exports the model to TorchScript format (.pt).
    ///
    /// # Arguments
    /// * `output_path` - The file path where the TorchScript model will be saved.
    ///
    /// # Returns
    /// * `Result<(), Box<dyn Error>>` - Returns Ok if successful, otherwise an error.
    pub fn export_torchscript(&self, output_path: &str) -> Result<(), Box<dyn Error>> {
        println!("Exporting model to TorchScript format...");

        let script_module = CModule::from(&self.model);
        script_module.save(output_path)?;

        println!("Model successfully exported to TorchScript at {}", output_path);
        Ok(())
    }

    /// Exports the model to ONNX format (.onnx).
    ///
    /// # Arguments
    /// * `output_path` - The file path where the ONNX model will be saved.
    /// * `dummy_input` - A tensor representing the expected input shape.
    ///
    /// # Returns
    /// * `Result<(), Box<dyn Error>>` - Returns Ok if successful, otherwise an error.
    pub fn export_onnx(&self, output_path: &str, dummy_input: Tensor) -> Result<(), Box<dyn Error>> {
        println!("Exporting model to ONNX format...");

        let device = Device::cuda_if_available();
        let input_tensor = dummy_input.to(device);
        
        self.model.trace(&input_tensor)?.save(output_path)?;

        println!("Model successfully exported to ONNX at {}", output_path);
        Ok(())
    }

    /// Loads and verifies an ONNX model to ensure successful export.
    ///
    /// # Arguments
    /// * `onnx_path` - The path to the exported ONNX model.
    ///
    /// # Returns
    /// * `Result<Session, Box<dyn Error>>` - Returns an ONNX session if the model loads successfully.
    pub fn verify_onnx_model(onnx_path: &str) -> Result<Session, Box<dyn Error>> {
        println!("Verifying exported ONNX model...");

        let environment = Environment::builder()
            .with_name("Machine Intelligence Node")
            .build()?;
        let session = Session::new(&environment, onnx_path)?;

        println!("ONNX model successfully loaded and verified.");
        Ok(session)
    }
}

/// Example usage of the ModelExporter
fn main() -> Result<(), Box<dyn Error>> {
    use tch::nn::{self, Module, OptimizerConfig};
    
    let device = Device::cuda_if_available();
    let vs = nn::VarStore::new(device);
    let model = nn::seq()
        .add(nn::linear(vs.root() / "layer1", 512, 256, Default::default()))
        .add(nn::relu())
        .add(nn::linear(vs.root() / "layer2", 256, 10, Default::default()));

    let exporter = ModelExporter::new(model);

    // Export model to TorchScript
    exporter.export_torchscript("exported_model.pt")?;

    // Export model to ONNX
    let dummy_input = Tensor::randn(&[1, 512], (tch::Kind::Float, device));
    exporter.export_onnx("exported_model.onnx", dummy_input)?;

    // Verify ONNX model
    ModelExporter::verify_onnx_model("exported_model.onnx")?;

    Ok(())
}
