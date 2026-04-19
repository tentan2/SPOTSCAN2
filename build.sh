#!/bin/bash

# Build script for Spotscan C++ project

echo "Building Spotscan C++ project..."

# Create build directory
mkdir -p build
cd build

# Configure with CMake
echo "Configuring with CMake..."
cmake .. -DCMAKE_BUILD_TYPE=Release

if [ $? -ne 0 ]; then
    echo "CMake configuration failed!"
    exit 1
fi

# Build the project
echo "Building project..."
make -j$(nproc)

if [ $? -eq 0 ]; then
    echo "Build successful!"
    echo "Executable: build/spotscan_cpp"
    echo ""
    echo "To run: ./build/spotscan_cpp"
else
    echo "Build failed!"
    exit 1
fi
