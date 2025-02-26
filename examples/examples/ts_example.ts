/**
 * Machine Intelligence Node - TypeScript Example Inference Script
 *
 * This script demonstrates how to send inference requests to an AI model
 * running on Machine Intelligence Node using WebSockets.
 *
 * Author: Machine Intelligence Node Development Team
 */

import { WebSocketHandler } from "../src/tsbindings/src/websocket_handler";

const WS_URL = "ws://localhost:8080";
const ws = new WebSocketHandler(WS_URL);

// Sample input data for inference
const inputData = {
    type: "inference",
    payload: new Float32Array(512).fill(0.5) // Example feature vector
};

// Function to handle AI inference
async function runInference() {
    try {
        console.log("[INFO] Connecting to WebSocket server...");

        ws.sendMessage(inputData);

        ws.socket.onmessage = (event: MessageEvent) => {
            const response = JSON.parse(event.data);
            console.log("[INFO] Inference result:", response.output);
        };

        ws.socket.onerror = (error) => {
            console.error("[ERROR] WebSocket error:", error);
        };

    } catch (error) {
        console.error("[ERROR] Inference failed:", error);
    }
}

// Start inference
runInference();
