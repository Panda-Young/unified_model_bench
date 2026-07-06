# Unified Benchmark

Multi-framework inference benchmark tool for comparing **ONNX Runtime**, **TensorFlow Lite**, **NCNN**, and **MNN** on the same model across Windows and Android (arm64-v8a).

## Architecture

```
unified_bench/
├── src/                    # C++ benchmark & backends
│   ├── main.cpp            # Entry point
│   ├── benchmark_runner.cpp/.hpp  # Orchestrator
│   ├── backend_interface.hpp      # Abstract IBackend
│   ├── onnx_backend.cpp    # ONNX Runtime (CPU/DML/oneDNN/OpenVINO/QNN)
│   ├── tflite_backend.cpp  # TFLite (CPU/XNNPACK/NNAPI/GPU)
│   ├── ncnn_backend.cpp    # NCNN (CPU/Vulkan)
│   ├── mnn_backend.cpp     # MNN (CPU/OpenCL)
│   ├── backend_registry.cpp
│   ├── result_collector.cpp/.hpp
│   ├── input_provider.cpp/.hpp
│   ├── model_loader.cpp/.hpp
│   ├── cmd_args.cpp/.hpp
│   ├── device_info.cpp/.hpp
│   ├── file_ops.cpp/.hpp
│   └── log.cpp/.hpp
├── include/                # Shared headers
├── deps/                   # Third-party libraries
│   ├── onnxruntime/        # ONNX Runtime (win-x64/win-x86/android)
│   ├── tflite/             # TensorFlow Lite C API
│   ├── ncnn/               # ncnn framework
│   └── mnn/                # MNN framework
├── tools/                  # Build & conversion scripts
│   ├── onnx_convert.py     # ONNX → NCNN/MNN/TFLite converter
│   ├── NDK_build_Android_auto.bat  # Android NDK build + deploy
│   ├── VS_build_win_x64.bat       # Windows x64 MSVC build
│   ├── VS_build_win_x86.bat       # Windows x86 MSVC build
│   └── csv_to_excel.py     # CSV → Excel converter
└── test_model.onnx         # Sample model
```

## Features

- **Four inference frameworks**: ONNX Runtime, TensorFlow Lite, NCNN, MNN
- **Multiple backends per framework**:
  - ONNX: CPU, oneDNN, DirectML, OpenVINO (CPU/GPU), NNAPI, XNNPACK, QNN (CPU/GPU/HTP)
  - TFLite: CPU, XNNPACK, NNAPI, GPU (OpenCL)
  - NCNN: CPU, Vulkan, Vulkan FP16
  - MNN: CPU, OpenCL, OpenCL FP16
- **Cross-platform**: Windows (x86/x64) and Android (arm64-v8a)
- **Deterministic shared inputs**: Same random seed (42) for all backends
- **Numerical accuracy verification**: Compares each backend's output against ONNX CPU baseline
- **CSV output**: Append-mode CSV for incremental results
- **NCHW↔NHWC auto-handling**: TFLite models (NHWC) are transparently fed NCHW data via runtime transpose

## Prerequisites

### Windows

| Dependency | Purpose |
|---|---|
| Visual Studio 2019+ | MSVC compiler (`vswhere` detection) |
| Python 3.10+ | Model conversion tool |
| Android NDK (optional) | Cross-compile for Android |

### Android (device)

| Library | Location |
|---|---|
| `libonnxruntime.so` | `deps/onnxruntime/lib/android/arm64-v8a/` |
| `libtensorflowlite_c.so` | `deps/tflite/lib/android/arm64-v8a/` |
| `libncnn.so` | `deps/ncnn/lib/android/arm64-v8a/` |
| `libMNN.so` | `deps/mnn/lib/arm64-v8a/` |

> **Note**: Native `.so` files for Android must be downloaded separately from each project's releases. The directory structure exists but may be empty.

## Quick Start

### 1. Model Conversion

Convert an ONNX model to all target formats:

```bash
python tools/onnx_convert.py model.onnx --to all
```

This generates:
- `model.ncnn.param` + `model.ncnn.bin` (NCNN FP32)
- `model_fp16.ncnn.bin` (NCNN FP16 weights, optional)
- `model.mnn` (MNN)
- `model.tflite` (TFLite)

> Missing Python dependencies are auto-installed via pip (with mirror fallback).

### 2. Build & Run (Windows x64)

```bash
tools\VS_build_win_x64.bat
```

The built binary `unified_bench_win_x64.exe` auto-discovers model variants (`.onnx`, `.tflite`, `.ncnn.param`, `.mnn`) and benchmarks all available backends.

### 3. Build & Run (Android)

```bash
set ANDROID_NDK_ROOT=D:\path\to\ndk\version
tools\NDK_build_Android_auto.bat
```

This script:
1. Compiles with NDK clang
2. Pushes binary + models + `.so` libraries to device via ADB
3. Runs the full benchmark with warmup=5, repeat=100, threads=4
4. Pulls results CSV back to host

### 4. CLI Options

```bash
unified_bench <model_path> [options]

Options:
  --model <path>      Model file path
  --input <path>      Input data file (binary float32)
  --repeat <N>        Benchmark repeat count (default: 100)
  --warmup <N>        Warmup runs (default: 1)
  --threads <N>       Number of threads (default: 4)
  --csv <path>        CSV output path (default: summary.csv)
  --no-csv            Don't write CSV
  --no-output-print   Don't print output summary
```

Results are appended to `summary.csv` to preserve history across runs.

## Output Format

Each CSV row contains:

| Column | Description |
|---|---|
| `model_name` | Model file path |
| `backend_name` | Backend identifier (e.g., `CPU`, `DML`, `NCNN_Vulkan`) |
| `avg_run_ms` | Average inference time (ms) |
| `max_output_diff` | Max element-wise difference vs ONNX CPU baseline |
| `avg_output_diff` | Average element-wise difference vs baseline |
| `acceleration_vs_cpu` | Speedup ratio relative to ONNX CPU baseline |

## Numerical Accuracy

The benchmark reports `max_output_diff` by comparing each backend's **first output tensor** against the ONNX CPU baseline:

| max_output_diff | Meaning |
|---|---|
| `< 1e-5` | **PASS** — fp32 equivalent |
| `< 0.01` | **MARGINAL** — minor fp32 rounding |
| `< 0.5` | **ACCEPTABLE** — moderate differences, validate on real data |
| `>= 0.5` | **LARGE** — conversion completed, numerical results differ |

### Known Accuracy Notes

- **TFLite**: Model conversion (ONNX → TFLite via onnx2tf) changes data layout from NCHW to NHWC. The C++ benchmark auto-transposes inputs and treats output[0] (kept in NCHW by onnx2tf) for fair comparison.
- **QNN_HTP / NCNN_Vulkan_FP16 / MNN_OpenCL_FP16**: FP16 inference introduces quantization noise (typical max_diff < 0.1).
- **NNAPI (TFLite)**: May fall back to CPU if the device driver doesn't support all model ops. A runtime warning is printed when timing is within 10% of CPU.

## Project Structure Notes

- `tools/onnx_convert.py` uses **onnx2tf** (PINTO0309) for TFLite conversion with automatic NCHW↔NHWC parameter replacement (PRF) generation.
- NCNN `.param` files are identical for FP32 and FP16 (structure only, no weights). Only `.bin` differs.
- TFLite C API is used instead of the C++ API for ABI stability across TFLite versions.
