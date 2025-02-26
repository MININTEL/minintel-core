#!/bin/bash

# Machine Intelligence Node - Rust Build Script
# Automates compilation and optimization of Rust AI core.
#
# Author: Machine Intelligence Node Development Team

RUST_PROJECT_DIR="src/rust"
BUILD_MODE="release"
WASM_TARGET="wasm32-unknown-unknown"
LOG_DIR="logs"

# Create log directory if it doesn't exist
mkdir -p $LOG_DIR

# Check if Rust is installed
if ! command -v cargo &> /dev/null; then
    echo "[ERROR] Rust toolchain not found. Please install Rust (https://rustup.rs/) and try again."
    exit 1
fi

echo "[INFO] Compiling Rust AI core..."
cd $RUST_PROJECT_DIR

# Build Rust project with optimization flags
cargo build --$BUILD_MODE 2>&1 | tee ../../$LOG_DIR/rust_build.log

if [ $? -ne 0 ]; then
    echo "[ERROR] Rust build failed. Check logs for details."
    exit 1
fi

echo "[INFO] Rust build complete. Optimized binary available in target/$BUILD_MODE."

# Build WebAssembly target if available
if rustup show active-toolchain | grep -q "wasm"; then
    echo "[INFO] Compiling for WebAssembly..."
    cargo build --target $WASM_TARGET --$BUILD_MODE 2>&1 | tee ../../$LOG_DIR/wasm_build.log

    if [ $? -ne 0 ]; then
        echo "[ERROR] WebAssembly build failed. Check logs for details."
        exit 1
    fi

    echo "[INFO] WebAssembly build complete. Output in target/wasm32-unknown-unknown/$BUILD_MODE."
else
    echo "[INFO] WebAssembly target not detected. Skipping WASM build."
fi

echo "[INFO] Rust AI core build process completed successfully."
