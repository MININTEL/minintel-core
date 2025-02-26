#!/bin/bash

# Machine Intelligence Node - TypeScript Build Script
# Automates compilation and bundling of TypeScript AI bindings.
#
# Author: Machine Intelligence Node Development Team

TS_PROJECT_DIR="src/tsbindings"
BUILD_OUTPUT_DIR="dist"
LOG_DIR="logs"

# Create log directory if it doesn't exist
mkdir -p $LOG_DIR

# Check if Node.js and npm are installed
if ! command -v node &> /dev/null; then
    echo "[ERROR] Node.js not found. Please install Node.js (https://nodejs.org/) and try again."
    exit 1
fi

if ! command -v npm &> /dev/null; then
    echo "[ERROR] npm not found. Please install npm and try again."
    exit 1
fi

# Navigate to TypeScript project directory
cd $TS_PROJECT_DIR

# Install dependencies
echo "[INFO] Installing dependencies..."
npm install 2>&1 | tee ../../$LOG_DIR/ts_npm_install.log

if [ $? -ne 0 ]; then
    echo "[ERROR] Dependency installation failed. Check logs for details."
    exit 1
fi

# Compile TypeScript
echo "[INFO] Compiling TypeScript..."
npx tsc --project tsconfig.json 2>&1 | tee ../../$LOG_DIR/ts_compile.log

if [ $? -ne 0 ]; then
    echo "[ERROR] TypeScript compilation failed. Check logs for details."
    exit 1
fi

# Bundle using esbuild
echo "[INFO] Bundling JavaScript output..."
npx esbuild $BUILD_OUTPUT_DIR/index.js --bundle --minify --outfile=$BUILD_OUTPUT_DIR/bundle.js 2>&1 | tee ../../$LOG_DIR/ts_bundle.log

if [ $? -ne 0 ]; then
    echo "[ERROR] Bundling failed. Check logs for details."
    exit 1
fi

echo "[INFO] TypeScript build complete. Output in $BUILD_OUTPUT_DIR."
