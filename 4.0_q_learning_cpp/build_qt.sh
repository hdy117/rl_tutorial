#!/bin/bash

set -e

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$SCRIPT_DIR"

# Create separate build directory to avoid conflicts with main project
BUILD_DIR="build_qt"

# Setup build directory
# rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Copy source files
cp ../CMakeLists_qt.txt CMakeLists.txt
cp ../qt_visualizer.cc .

# Configure
echo "Configuring Qt6 + Protobuf project..."
cmake .

# Build
echo "Building..."
make -j$(nproc)

# Copy json file for testing
cp ../q_learning.json . 2>/dev/null || true

echo ""
echo "========================================"
echo "Build complete!"
echo "Run: cd $BUILD_DIR && ./qt_visualizer [json_file]"
echo "========================================"
