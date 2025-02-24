/**
 * Machine Intelligence Node - Model Utilities
 *
 * Provides tensor conversion, input preprocessing, and normalization functions
 * for AI inference.
 *
 * Author: Machine Intelligence Node Development Team
 */

import * as tf from "@tensorflow/tfjs";

/**
 * Normalizes input data to the range [0, 1].
 *
 * @param input - The input data array.
 * @returns A normalized Float32Array.
 */
export function normalizeInput(input: Float32Array): Float32Array {
  const maxVal = Math.max(...input);
  const minVal = Math.min(...input);
  return input.map(val => (val - minVal) / (maxVal - minVal));
}

/**
 * Converts an input array into a TensorFlow.js tensor.
 *
 * @param input - The input data array.
 * @param shape - The expected shape of the tensor.
 * @returns A TensorFlow.js tensor.
 */
export function toTensor(input: Float32Array, shape: number[]): tf.Tensor {
  return tf.tensor(input, shape);
}

/**
 * Converts a batch of inputs into tensors.
 *
 * @param batchInputs - Array of input Float32Arrays.
 * @param shape - Expected shape for each input.
 * @returns An array of TensorFlow.js tensors.
 */
export function batchToTensor(batchInputs: Float32Array[], shape: number[]): tf.Tensor[] {
  return batchInputs.map(input => toTensor(input, shape));
}

/**
 * Converts a TensorFlow.js tensor back to a Float32Array.
 *
 * @param tensor - The tensor to convert.
 * @returns A Float32Array representation of the tensor.
 */
export function tensorToArray(tensor: tf.Tensor): Float32Array {
  return new Float32Array(tensor.dataSync());
}

// Export utilities for external usage
export default {
  normalizeInput,
  toTensor,
  batchToTensor,
  tensorToArray
};
