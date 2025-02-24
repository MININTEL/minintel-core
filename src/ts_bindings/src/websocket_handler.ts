/**
 * Machine Intelligence Node - WebSocket Handler
 *
 * Provides real-time AI inference via WebSocket communication.
 *
 * Author: Machine Intelligence Node Development Team
 */

export class WebSocketHandler {
  private socket: WebSocket;
  private url: string;
  private reconnectInterval: number;
  private isConnected: boolean = false;

  /**
   * Initializes a WebSocket connection for AI inference streaming.
   *
   * @param url - The WebSocket server URL.
   * @param reconnectInterval - Interval (ms) for reconnection attempts.
   */
  constructor(url: string, reconnectInterval: number = 5000) {
    this.url = url;
    this.reconnectInterval = reconnectInterval;
    this.connect();
  }

  /**
   * Establishes the WebSocket connection.
   */
  private connect() {
    this.socket = new WebSocket(this.url);

    this.socket.onopen = () => {
      console.log("Connected to WebSocket server:", this.url);
      this.isConnected = true;
    };

    this.socket.onmessage = (event) => {
      console.log("Received message:", event.data);
    };

    this.socket.onerror = (error) => {
      console.error("WebSocket error:", error);
    };

    this.socket.onclose = () => {
      console.warn("WebSocket connection closed. Reconnecting...");
      this.isConnected = false;
      setTimeout(() => this.connect(), this.reconnectInterval);
    };
  }

  /**
   * Sends a message to the WebSocket server.
   *
   * @param data - Data to send.
   */
  public sendMessage(data: any) {
    if (this.isConnected) {
      this.socket.send(JSON.stringify(data));
    } else {
      console.warn("WebSocket not connected. Message not sent.");
    }
  }

  /**
   * Closes the WebSocket connection.
   */
  public closeConnection() {
    this.socket.close();
  }
}

// Example Usage
if (typeof window !== "undefined") {
  const wsHandler = new WebSocketHandler("ws://localhost:8080");

  // Example: Sending an AI inference request
  setTimeout(() => {
    wsHandler.sendMessage({ type: "inference", payload: [1.2, 3.4, 5.6] });
  }, 1000);
}
