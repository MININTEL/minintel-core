/**
 * Machine Intelligence Node - API Wrapper
 *
 * Provides a high-level interface for interacting with AI models via WebAssembly (WASM).
 *
 * Author: Machine Intelligence Node Development Team
 */

import { initializeModel, runInference } from "./index";
import { Config, ModelResult } from "./types";

/**
 * Loads an AI model from the specified path.
 *
 * @param modelPath - The path to the AI model file.
 * @returns A Promise resolving to a success status.
 */
export async function loadAIModel(modelPath: string): Promise<boolean> {
  console.log(`Loading model from: ${modelPath}`);
  return await initializeModel(modelPath);
}

/**
 * Runs AI inference with the given input data.
 *
 * @param inputData - The input tensor data.
 * @param config - Optional configuration parameters for inference.
 * @returns A Promise resolving to the model's output.
 */
export async function inferAI(inputData: Float32Array, config?: Config): Promise<ModelResult> {
  console.log("Running inference...");
  
  try {
    const result = await runInference(inputData, config);
    console.log("Inference result:", result);
    return result;
  } catch (error) {
    console.error("Inference failed:", error);
    throw error;
  }
}

/**
 * Exposes the AI API functions for external use.
 */
export default {
  loadAIModel,
  inferAI
};
