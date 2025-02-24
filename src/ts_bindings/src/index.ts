/**
 * Machine Intelligence Node - TypeScript Bindings
 *
 * Provides an entry point for TypeScript-based AI execution.
 * Supports WebAssembly (WASM) integration for efficient AI processing.
 *
 * Author: Machine Intelligence Node Development Team
 */

import { loadModel, infer } from "./wasm_loader";
import { Config, ModelResult } from "./types";

/**
 * Loads the AI model into memory.
 *
 * @param modelPath - The path to the AI model file.
 * @returns A Promise resolving to a boolean indicating success.
 */
export async function initializeModel(modelPath: string): Promise<boolean> {
  try {
    await loadModel(modelPath);
    console.log("Model successfully loaded.");
    return true;
  } catch (error) {
    console.error("Failed to load model:", error);
    return false;
  }
}

/**
 * Runs inference on the given input data.
 *
 * @param inputData - The input data for the AI model.
 * @param config - Optional inference configuration settings.
 * @returns A Promise resolving to the model's prediction result.
 */
export async function runInference(inputData: Float32Array, config?: Config): Promise<ModelResult> {
  try {
    const result = await infer(inputData, config);
    return result;
  } catch (error) {
    console.error("Inference failed:", error);
    throw error;
  }
}

// Export types for external usage
export * from "./types";

// Default export for streamlined usage
export default {
  initializeModel,
  runInference
};
