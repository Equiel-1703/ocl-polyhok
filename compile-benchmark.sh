#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <PATH TO OPENCL/CUDA BENCHMARK>"
    exit 1
fi

BENCHMARK_FILE="$1"
BENCHMARK_NAME="$(basename "$1")"
BENCHMARK_EXT="${BENCHMARK_NAME##*.}"
BENCHMARK_DIR=$(dirname "$1")
OUTPUT_FILE="${BENCHMARK_DIR}/${BENCHMARK_NAME%.*}.out"

case "$BENCHMARK_EXT" in
    "cu")
        echo "Compiling CUDA benchmark: $BENCHMARK_NAME"
        nvcc "$BENCHMARK_FILE" -o "$OUTPUT_FILE"
        ;;
    "cpp")
        echo "Compiling OpenCL benchmark: $BENCHMARK_NAME"
        g++ "$BENCHMARK_FILE" -o "$OUTPUT_FILE" -lOpenCL
        ;;
    *)
        echo "Unsupported file extension: $BENCHMARK_EXT"
        exit 1
        ;;
esac

if [ $? -ne 0 ]; then
    echo "Compilation failed."
    exit 1
fi

echo "Done."
echo "Executable created at: $OUTPUT_FILE"