/**
 * Machine Intelligence Node - TypeScript AI Tests
 *
 * Verifies AI model inference, WebSocket communication, and async execution.
 *
 * Author: Machine Intelligence Node Development Team
 */

import { initializeModel, runInference } from "../src/tsbindings/src/api";
import { WebSocketHandler } from "../src/tsbindings/src/websocket_handler";

const MODEL_PATH = "models/test_model.onnx";
const INPUT_DATA = new Float32Array(512).fill(0.5);
const WS_URL = "ws://localhost:8080";

/**
 * Tests AI model initialization.
 */
test("Model should initialize successfully", async () => {
  const success = await initializeModel(MODEL_PATH);
  expect(success).toBe(true);
});

/**
 * Tests AI model inference execution.
 */
test("Model should return valid inference output", async () => {
  const output = await runInference(INPUT_DATA);
  expect(output).toBeDefined();
  expect(output.length).toBe(10);
});

/**
 * Tests asynchronous inference execution.
 */
test("Inference should run asynchronously without blocking", async () => {
  const start = performance.now();
  const promise = runInference(INPUT_DATA);
  const mid = performance.now();
  
  expect(mid - start).toBeLessThan(5); // Should return quickly without blocking
  
  const output = await promise;
  expect(output).toBeDefined();
});

/**
 * Tests WebSocket AI inference request.
 */
test("WebSocket should send and receive inference requests", (done) => {
  const ws = new WebSocketHandler(WS_URL);

  ws.sendMessage({ type: "inference", payload: INPUT_DATA });

  ws.socket.onmessage = (event) => {
    const response = JSON.parse(event.data);
    expect(response).toHaveProperty("output");
    expect(response.output.length).toBe(10);
    done();
  };

  ws.socket.onerror = (error) => {
    done.fail(`WebSocket error: ${error}`);
  };
});

/**
 * Tests error handling for invalid input.
 */
test("Model should handle invalid input gracefully", async () => {
  await expect(runInference(new Float32Array(0))).rejects.toThrow();
});
