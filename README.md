# Unified Benchmark Tool v2.0 (C++)

跨平台多框架推理基准测试工具，覆盖 **ONNX Runtime / TensorFlow Lite / LiteRT / NCNN / MNN** 五大框架，支持 **CPU / GPU / NPU / DSP** 多种加速后端。同一模型（从同一 ONNX 转换）在多个后端上运行，使用**共享确定性输入**对比推理延迟与输出精度。

> **目的**：尽可能真实地测试每一种 backend 的推理能力——不支持的 backend 直接报错标记失败（CSV 行 `avg/accel` 及内存列显示 `-`，`notes` 写原因），绝不静默降级。

---

## 1. 目录结构

```
unified_model_bench/
├── CMakeLists.txt              # 主构建脚本（VS / NDK / Ninja 通用）
├── include/                    # 公共头文件
│   ├── backend_interface.hpp   # IBackend 抽象接口 + BackendId/BackendType 枚举 + Registry
│   ├── benchmark_runner.hpp    # 基准测试编排（worker 流程 + RunPerProcess 声明）
│   ├── scheduler.hpp           # 每 backend 独立进程调度器的辅助函数声明
│   ├── csv_utils.hpp           # CSV 行级工具声明（列索引 + 行解析/回查）
│   ├── cmd_args.hpp            # CLI 参数结构
│   ├── model_format.hpp        # 模型格式枚举（ONNX/TFLite/NCNN/MNN/QNN）
│   ├── platform.hpp            # 平台宏（ARCH_STR 等）
│   └── ...                     # log / input_provider / result_collector / qnn_soc 等
├── src/                        # 实现
│   ├── main.cpp                # 入口
│   ├── benchmark_runner.cpp    # worker 编排：模型发现 → 输入生成 → 逐 backend 测试 → CSV
│   ├── scheduler.cpp           # 每 backend 独立进程调度（spawn worker + 跨进程 baseline）
│   ├── csv_utils.cpp           # CSV 行级工具（解析/列索引/worker 记录回查）
│   ├── backend_registry.cpp    # 各平台 backend 注册表
│   ├── onnx_backend.cpp        # ONNX Runtime EP（DML/OpenVINO/oneDNN/QNN/NNAPI/XNNPACK）
│   ├── tflite_backend.cpp      # TFLite Delegate（XNNPACK/NNAPI/GPU/QNN-NPU）
│   ├── litert_backend.cpp      # LiteRT（CPU/GPU/NPU，QNN dispatch）
│   ├── litert_qualcomm_stubs.cpp # Android 专用：QNN options API 的 stub 实现
│   ├── qnn_backend.cpp         # QNN SDK 原生后端（context binary，直接调 QNN C API）
│   ├── ncnn_backend.cpp        # NCNN（CPU/Vulkan，FP32/FP16/BF16）
│   ├── mnn_backend.cpp         # MNN（CPU/OpenCL/Vulkan）
│   └── ...                     # cmd_args / device_info / file_ops / input_provider / result_collector / log / qnn_soc
├── deps/                       # 第三方依赖（离线 vendored）
│   ├── onnxruntime/            # ORT 1.22.0 + DML/OpenVINO/oneDNN/QNN 各 EP 变体
│   ├── tflite/                 # TFLite 2.18.0（include + 各平台 .so/.dll）
│   ├── litert/                 # LiteRT runtime + litert_cc_sdk + NPU dispatch 库
│   ├── ncnn/                   # NCNN（桌面 + Android 头文件/库）
│   └── mnn/                    # MNN
├── tools/                      # 构建与转换脚本
│   ├── NDK_build_Android_auto.bat  # Android NDK 构建 + adb 部署 + 自动推 QNN/NCNN/TFLite/MNN/LiteRT 库
│   ├── VS_build_win_x64.bat        # Windows x64 VS 构建
│   ├── VS_build_win_x86.bat        # Windows x86 VS 构建
│   ├── onnx_convert.py             # ONNX → TFLite/NCNN/MNN 转换
│   ├── gen_test_model.py           # 生成测试模型
│   └── csv_to_excel.py             # CSV → Excel
├── docs/                       # 各模块调试记录（详见 §8）
│   ├── ONNX_DEBUG_LOG.md
│   ├── TFLITE_NPU_DEBUG_LOG.md
│   ├── NCNN_DEBUG_LOG.md
│   ├── LiteRT_GPU_DEBUG_LOG.md
│   ├── QNN_SDK_DEBUG_LOG.md
│   ├── QNN_SDK_HTP_OPTIMIZATIONS.md
│   └── PROJECT_SCORECARD.md
└── summary.csv                 # 历史测试结果（追加式）
```

---

## 2. 支持的后端矩阵

### 2.1 BackendId 编号规则

| 范围 | 框架 | 说明 |
|------|------|------|
| 0–17 | ONNX | Execution Provider |
| 100–107 | TFLite | Delegate |
| 200–206 | NCNN | CPU/Vulkan |
| 300–309 | MNN | CPU/OpenCL/Vulkan |
| 400–405 | LiteRT | CPU/GPU/NPU |
| 500–503 | QNN SDK | HTP/GPU/CPU（context binary） |

### 2.2 各框架后端

**ONNX Runtime（0–17）**

| Backend | 名称 | 桌面 | Android | 说明 |
|---------|------|------|---------|------|
| ONNX_CPU | `ONNX_CPU` | ✅ | ✅ | 默认 CPU EP（基准） |
| ONNX_ONEDNN | `ONNX_oneDNN` | ✅ | ❌ | Intel oneDNN |
| ONNX_DML_GPU | `ONNX_DML_GPU` | ✅ | ❌ | DirectML GPU |
| ONNX_DML_NPU | `ONNX_DML_NPU` | ✅ | ❌ | DirectML NPU（V2 API） |
| ONNX_OPENVINO_CPU | `ONNX_OpenVINO_CPU` | ✅ | ❌ | OpenVINO CPU |
| ONNX_OPENVINO_GPU | `ONNX_OpenVINO_GPU` | ✅ | ❌ | OpenVINO GPU FP32 |
| ONNX_OPENVINO_GPU_FP16 | `ONNX_OpenVINO_GPU_FP16` | ✅ | ❌ | OpenVINO GPU FP16 |
| ONNX_OPENVINO_NPU | `ONNX_OpenVINO_NPU` | ✅ | ❌ | OpenVINO NPU |
| ONNX_NNAPI | `ONNX_NNAPI` | ❌ | ✅ | Android NNAPI |
| ONNX_XNNPACK | `ONNX_XNNPACK` | ❌ | ✅ | XNNPACK EP |
| ONNX_QNN_CPU / _GPU / _HTP | `ONNX_QNN_*` | ❌ | ✅ | Qualcomm QNN |

> **已删除**：`ONNX_OpenVINO_GPU_BF16`（id=8）——OpenVINO GPU 只支持 FP16/FP32/ACCURACY，BF16 在框架层面无此能力，所有硬件均不支持，故移除。
> **已删除**：`ONNX_DML_GPU_FP16`（id=3）——DML 无独立 FP16 开关，自动按硬件能力选择精度，与 `ONNX_DML_GPU` 等价，故移除。

**TFLite（100–107）**

| Backend | 名称 | 桌面 | Android | 说明 |
|---------|------|------|---------|------|
| TFLITE_CPU | `TFLITE_CPU` | ✅ | ✅ | 默认（基准） |
| TFLITE_XNNPACK | `TFLITE_XNNPACK` | ✅ | ✅ | XNNPACK delegate |
| TFLITE_XNNPACK_FP16 | `TFLITE_XNNPACK_FP16` | ✅* | ✅ | FP16（桌面需硬件支持） |
| TFLITE_NNAPI | `TFLITE_NNAPI` | ❌ | ✅ | NNAPI |
| TFLITE_GPU / _FP16 | `TFLITE_GPU` | ❌ | ✅ | GPU delegate |
| TFLITE_NPU | `TFLITE_NPU` | ❌ | ✅ | QNN HTP delegate |

**LiteRT（400–405）**

| Backend | 名称 | 桌面 | Android | 说明 |
|---------|------|------|---------|------|
| LITERT_CPU | `LiteRT_CPU` | ✅ | ✅ | LiteRT CPU（基准） |
| LITERT_GPU | `LiteRT_GPU` | ✅ | ✅ | WebGPU/D3D12（桌面）/ CL/GL（Android） |
| LITERT_GPU_FP16 | `LiteRT_GPU_FP16` | ✅ | ✅ | GPU FP16 |
| LITERT_NPU | `LiteRT_NPU` | ❌ | ✅ | Qualcomm HTP（dispatch） |
| LITERT_NPU_FP16 | `LiteRT_NPU_FP16` | ❌ | ✅ | HTP FP16 |

**NCNN（200–206）**

| Backend | 名称 | 桌面 | Android | 说明 |
|---------|------|------|---------|------|
| NCNN_CPU | `NCNN_CPU` | ✅ | ✅ | CPU（基准） |
| NCNN_CPU_FP16 | `NCNN_CPU_FP16` | ✅ | ✅ | CPU FP16（NEON） |
| NCNN_CPU_BF16 | `NCNN_CPU_BF16` | ✅* | ✅* | CPU BF16（需 CPUID 支持 + trial 验证） |
| NCNN_VK / _FP16 / _BF16 | `NCNN_Vulkan*` | ✅ | ❌* | Vulkan（Android NCNN 编译时无 Vulkan，报错不降级） |

**MNN（300–309）**：CPU / OpenCL / Vulkan / OpenGL，桌面与 Android 均支持（桌面默认关闭 `HAVE_MNN_BACKEND=OFF`）。

**QNN SDK（500–503，Android）**

| Backend | 名称 | 桌面 | Android | 说明 |
|---------|------|------|---------|------|
| QNN_HTP | `QNN_HTP` | ❌ | ✅ | 直接加载 context binary 到 Hexagon（零拷贝共享内存） |
| QNN_GPU | `QNN_GPU` | ❌ | ✅ | 同一 context binary 在 Adreno GPU 执行 |
| QNN_CPU | `QNN_CPU` | ❌ | ✅ | 同一 context binary 在 CPU 执行 |

> 模型格式为 **QNN context binary**（`.dlc`/`.serialized.bin`/`.bin`/`.so`），由 `qnn-context-binary-generator` 从 DLC 离线生成。仅 Android 且 `HAVE_QNN_SDK_BACKEND=ON` 时编译。

---

## 3. 构建系统

### 3.1 依赖与版本

| 依赖 | 版本 | 桌面路径 | Android 路径 |
|------|------|----------|--------------|
| ONNX Runtime | 1.22.0 | `deps/onnxruntime/lib/win-x64/{cpu,dml,onednn,openvino}/` | `deps/onnxruntime/lib/android/arm64-v8a/` |
| TensorFlow Lite | 2.18.0 | `deps/tflite/lib/win-x64/tensorflowlite_c.dll` | `deps/tflite/lib/android/arm64-v8a/*.so` |
| LiteRT | — | `deps/litert/liteRT_runtime/windows_x86_64/` | `deps/litert/liteRT_runtime/android_arm64/` |
| NCNN | 1.0.x | `deps/ncnn/lib/win-x64/ncnn.dll` | `deps/ncnn/lib/android/arm64-v8a/libncnn.so` |
| MNN | — | `deps/mnn/lib/win-x64/MNN.dll` | `deps/mnn/lib/arm64-v8a/libMNN.so` |
| QNN SDK | 2.48.40.260702 | — | `C:\Qualcomm\AIStack\QAIRT\2.48.40.260702` |

### 3.2 CMake 选项

| 选项 | 默认 | 说明 |
|------|------|------|
| `HAVE_ONNX_BACKEND` | ON | ONNX Runtime |
| `HAVE_TFLITE_BACKEND` | OFF | TFLite |
| `HAVE_NCNN_BACKEND` | ON | NCNN |
| `HAVE_MNN_BACKEND` | OFF | MNN |
| `HAVE_LITERT_BACKEND` | OFF | LiteRT |
| `HAVE_QNN_SDK_BACKEND` | OFF | 原生 QNN SDK 后端（仅 Android） |
| `QNN_SDK_ROOT` | — | QNN SDK 路径（Android） |

### 3.3 Windows x64 构建

```bat
tools\VS_build_win_x64.bat
:: 或手动：
cmake -B build/win-x64 -DHAVE_TFLITE_BACKEND=ON -DHAVE_LITERT_BACKEND=ON -DHAVE_ONNX_BACKEND=ON -DHAVE_NCNN_BACKEND=ON
cmake --build build/win-x64 --config Debug
```

### 3.4 Android NDK 构建 + 部署

```bat
set ANDROID_NDK_ROOT=C:\Users\...\Android\Sdk\ndk\27.0.12077973
tools\NDK_build_Android_auto.bat
```

脚本自动完成：配置 → 编译 → `adb push` 可执行文件与全部 `.so` → 按 `ro.soc.model` 自动选 Hexagon 版本并推送 Stub/Skel → 运行基准 → 回拉 CSV。

---

## 4. CLI 用法

```
unified_bench <model_path> [选项]

选项：
  --model <path>       模型路径（也支持位置参数）
  --input-list <path>  输入列表文件（见下），由 tools/generate_test_data_for_onnx.py 生成
  --input-format <fmt> 输入数据格式：auto|float32|uint8（默认 auto，按文件大小探测）
  --backend <name,...> 指定 backend（逗号分隔，缺省=全部可用）
  --no-backend <name,...> 排除 backend（黑名单，可与 --backend 组合，先白名单后排除）
  --repeat <N>         推理次数（默认 100）
  --warmup <N>         预热次数（默认 1）
  --threads <N>        线程数（默认 4）
  --csv <path>         CSV 输出路径（默认 summary.csv）
  --log-level <0-4>    日志级别（0=OFF..4=ERR，默认 2=INFO）
  --output-dir <path>  输出目录（存放基准临时文件）
  --no-csv             不写 CSV
  --help / --version
```

### 运行模式：每 backend 独立进程（唯一模式）

工具**只有一种运行模式**——调度器为每个（模型变体 × backend）启动一个**独立子进程（worker）**
执行单后端测试，无同进程顺序跑：

- **内存测量干净**：每个 worker 是全新进程，`peak_mem_mb` / `resident_mem_mb` 不含其他框架的
  残留（同进程顺序跑时前面 backend 的页会污染后面的读数，这正是采用本架构的原因）
- **崩溃隔离**：某个 backend 崩溃（如 ncnn Vulkan 在部分驱动上的析构崩溃）只影响自己的 worker，
  其余 backend 照常出结果；异常退出码合并进该行 `notes` 列
- **同批同时间戳**：同一调度器运行产生的所有 CSV 行共享同一个 `time` 值
- **全局基准**：以**入口模型格式的 CPU backend** 为精度基准（`test_model.onnx`→ONNX_CPU、
  `test_model.ncnn.bin`/`.ncnn.param`→NCNN_CPU、`test_model.mnn`→MNN_CPU，以此类推），其输出
  dump 给所有其他 worker 做 diff/accel 对比；**基准行本身 diff/accel 显示 `-`**

```bash
# 直接运行即调度全部可用 backend（每个一个独立进程）
unified_bench.exe test_model.onnx --repeat 10 --warmup 1

# 限制范围（仍每 backend 独立进程）
unified_bench.exe test_model.ncnn.bin --backend NCNN_CPU,NCNN_CPU_BF16 --repeat 10
```

### 使用外部输入（--input-list）

通过 `tools/generate_test_data_for_onnx.py` 生成模型测试数据与输入列表：

```bash
# 生成 float32（+可选 uint8）输入与 input_list 文件
python tools/generate_test_data_for_onnx.py
#   交互式输入模型路径；或直接用函数：
python -c "import sys; sys.path.insert(0,'tools'); \
  from generate_test_data_for_onnx import generate_test_data; \
  generate_test_data('test_model.onnx','data','relative',gen_uint8=True)"
```

生成的 `input_list_<model>_float32.txt` 格式（每行一个 .bin，`#` 为注释）：

```
# float32 inputs for test_model (2 input(s))
# one input .bin file per line (relative to this file's dir):

input_data_test_model_float32/input_a_1x3x224x224.bin
input_data_test_model_float32/input_b_1x48x1x1.bin
```

运行基准时通过 `--input-list` 加载（**路径按列表文件所在目录解析**，可从任意工作目录运行）：

```bash
# float32（auto 自动探测）
unified_bench test_model.onnx --input-list data/input_list_test_model_float32.txt

# uint8（auto 探测，或显式指定）
unified_bench test_model.onnx --input-list data/input_list_test_model_uint8.txt --input-format uint8
```

- **auto 探测**：文件大小 = 元素数×4 → float32；= 元素数 → uint8（uint8 按 /255 归一化到 float）
- **格式不匹配** / 文件缺失 / 输入数量不符 → 直接报错并跳过该变体，不静默兜底
- 输入数据对**所有 backend 共享**，保证公平对比

### 示例

```bash
# 桌面：全后端
unified_bench.exe test_model.onnx --repeat 10 --warmup 1

# 桌面：指定 TFLite 后端
unified_bench.exe test_model.tflite --backend TFLITE_CPU,TFLITE_XNNPACK --repeat 10

# Android：QNN HTP（ONNX）与 TFLite NPU 对比
adb shell "cd /data/local/tmp/bench_test && LD_LIBRARY_PATH=.:./qnn ADSP_LIBRARY_PATH=./qnn \
  ./unified_bench test_model.onnx --backend ONNX_QNN_HTP,TFLITE_NPU --repeat 10"

# Android：LiteRT NPU
adb shell "cd /data/local/tmp/bench_test && LD_LIBRARY_PATH=.:./qnn ADSP_LIBRARY_PATH=./qnn \
  ./unified_bench test_model.onnx --backend litert_npu --repeat 1 --warmup 0"
```

### backend 命名（不区分大小写）

`ONNX_CPU`, `ONNX_DML_GPU`, `ONNX_OpenVINO_GPU_FP16`, `ONNX_QNN_Htp`, `TFLITE_XNNPACK`, `TFLITE_NPU`, `LiteRT_CPU`, `LiteRT_GPU`, `LiteRT_NPU`, `NCNN_CPU`, `NCNN_Vulkan`, `NCNN_CPU_BF16`, `MNN_OpenCL` 等。

---

## 5. 架构设计

### 5.1 核心流程（`benchmark_runner.cpp`）

```mermaid
flowchart TD
    A[模型路径] --> B[search_model_variants]
    B --> C{发现变体}
    C --> D[入口格式 CPU backend = 全局基准]
    D --> E[spawn 基准 worker: --dump-output 落盘输出 + .avg 计时]
    E --> F[spawn 每个 variant × backend 一个 worker]
    F --> G[worker: 单后端 Initialize → 共享确定性输入 → RunBenchmark]
    G --> H[读 baseline 文件 → 算 max_diff / avg_diff / accel]
    H --> I[各自追加写共享 CSV 崩溃安全, 同批同 time 戳]
    I --> J[调度器回收退出码: 非零合并进 notes, 删除 baseline 临时文件]
```

- **调度器**（`RunPerProcess`，非 worker 进程）：发现模型变体 → 先跑入口格式 CPU baseline worker
  （`--dump-output` 把输出与 avg 落盘）→ 再为每个 (variant × backend) spawn 一个 worker
  （`--worker --backend <name> --batch-time <t>`，带 `--baseline-file`/`--baseline-ms` 做跨进程对比）
- **worker**（`--worker` 标记的子进程）：普通单后端流程（`TestVariant`→`TestBackend`），
  写共享 CSV（append），日志直通终端；父进程按退出码区分：`0`=成功、`1`=预期失败（worker 已自写
  失败行）、其他=异常崩溃（退出码合并进该行 `notes`）
- 每个 variant 跑完后调度器删除 baseline 临时文件（`<base>_<backend>.out` + `.avg`）

### 5.2 后端抽象（`backend_interface.hpp`）

所有 backend 实现同一 `IBackend` 接口，通过 `BackendRegistry` 注册与工厂创建：

```
IBackend
├── Initialize(model_path, num_threads)
├── QueryIOInfo(input_shape, output_shape)
├── SetSharedInput / PrepareInputs
├── RunBenchmark(warmup, repeat, ...)
├── GetTiming(...)
└── SaveOutputs(...)
```

### 5.3 共享输入机制

- 所有 backend 使用**同一份确定性输入**（`InputProvider`，seed=42）保证公平对比
- per-process 模式下每个 worker 是独立进程，各自用**相同 seed** 生成数值完全一致的输入
  （无进程间内存共享，但数据相同）
- 临时 CPU backend 先查询模型 IO shape → 解析成元素个数 → 生成输入
- 各 backend 内部处理自己的布局需求（TFLite NCHW→NHWC 转置、NCNN [N,C,H,W]→[w,h,c] 等）

### 5.4 失败语义

- **初始化失败**（backend 不可用、精度不支持等）→ CSV 行 `avg/accel` 及内存 4 列均显示 `-`，
  `notes` 列写具体原因（`last_error_`，如 "DML_NPU append: No devices detected"），不崩溃
- **worker 崩溃**（进程非零退出，如 0xC0000409）→ 数据已写则把退出码合并进该行 `notes`
  （`worker process exited abnormally with code ...`）；仅在崩溃发生在写记录**之前**才补独立失败行
- **不支持即报错**，绝不静默降级（NCNN BF16 无指令 → 报错；Vulkan 无设备 → 报错；FP16 权重缺失 → 报错）
- 模型类型不匹配提前拦截：输入非 FLOAT（如 int64 的 `input_ids`）或含动态维（`-1`）时，
  `QueryIOMetadata` 给出明确原因并标记失败（本工具基准为 float32，不做 int64/动态 shape 支持）

### 5.5 CSV 输出与内存测量

CSV 共 26 列：模型信息（`model_name` → `output_elements`）之后紧跟 `weight_mem_mb`，
然后是运行配置与 timing/精度列（`warmup_runs` → `acceleration_vs_cpu`），
内存实测列（`peak_mem_mb` / `resident_mem_mb`）与元数据（`backend_name` → `notes`）收尾。
部署侧关键列：

| 列 | 含义 |
|----|------|
| `max_output_diff` / `avg_output_diff` | 与基准输出逐元素差异（`max\|a-b\|` / `avg\|a-b\|`） |
| `acceleration_vs_cpu` | 基准 avg / 本 backend avg（`baseline_ms / avg_ms`）；无基准时 `-` |
| `weight_mem_mb` | 模型权重内存（ONNX 解析 initializer / NCNN 取 `.ncnn.bin` 大小 / TFLite 解析 flatbuffer Buffer 精确统计 / MNN·QNN 按文件大小近似），与 backend 无关 |
| `peak_mem_mb` | 进程峰值工作集（Windows `PeakWorkingSetSize` / Linux `VmHWM`），单调递增 |
| `resident_mem_mb` | run 结束后的常驻工作集（`WorkingSetSize` / `VmRSS`） |
| `notes` | 初始化耗时明细、失败原因、worker 异常退出码等 |

测量语义：`peak`/`resident` 是**进程级**统计（含框架运行时/arena 开销，不含 GPU 显存）；
per-process 架构下每 backend 独占进程，读数即该 backend 的部署值，无跨 backend 污染。
失败行内存列与 `avg/accel` 一致显示 `-`。

---

## 6. 关键平台适配（踩坑记录摘要）

### 6.1 Windows：TFLite + LiteRT 共存（2026-07-21 解决）

**问题**：`libLiteRt.dll` 与 `tensorflowlite_c.dll` 都导出完整 TFLite C API（21 个 `TfLite*` 符号）。linker 解析到 `libLiteRt.dll` 的 XNNPACK 实现，其在 Intel Iris Xe 上初始化时 `0xC0000005` 访问违规崩溃。

**方案**：
1. Windows 上**不链接** `tensorflowlite_c.dll.if.lib`（避免符号冲突），基础 TFLite 函数取自 `libLiteRt.lib`（验证正常）
2. **仅 XNNPACK delegate** 通过 `LoadLibrary("tensorflowlite_c.dll")` + `GetProcAddress` 动态加载（避免 `libLiteRt` 的 buggy 副本）
3. CMake 中两者可同时 `ON`，运行互不冲突

**验证**：TFLITE_CPU 29.6ms / TFLITE_XNNPACK 17.2ms / LiteRT_CPU 37.9ms 同时运行正常。

### 6.2 Android：NCNN Vulkan 宏冲突（2026-07-21 解决）

**问题**：`ncnn/platform.h` 定义 `#define NCNN_VULKAN 0`（Android 无 Vulkan），与 `BackendId::NCNN_VULKAN` 枚举名冲突，导致编译错误。

**方案**：枚举改名 `NCNN_VULKAN → NCNN_VK`（`NCNN_VK` / `NCNN_VK_FP16` / `NCNN_VK_BF16`），保留 `NCNN_VULKAN` 宏用于 `#if` 平台检测。Android 无 Vulkan 时初始化报错而非降级。

### 6.3 Android：NCNN BF16 / FP16 硬失败（2026-07-21）

- CPU BF16：无 `AVX512-BF16`/`ARM-BF16` 指令 → 直接 `return false`（不再 fallback FP32）
- CPU BF16：trial 前向崩溃 → `return false`
- Vulkan BF16：GPU 不支持 → `return false`
- Vulkan FP16：`_fp16.ncnn.bin` 权重缺失 → `return false`

### 6.4 Windows：LiteRT GPU 需要新版 DXC（2026-07-21）

`libLiteRtWebGpuAccelerator.dll` 运行时 `LoadLibrary` 加载 `dxcompiler.dll`，旧版（10.0.19041）无 Dawn 需要的 CLSID → `E_NOINTERFACE`。CMake post-build 自动搜索 Windows SDK 最新 `dxcompiler.dll`/`dxil.dll` 复制到输出目录。

### 6.5 Android：QNN 版本严格匹配

| 场景 | 版本要求 |
|------|----------|
| TFLITE_NPU（QNN TFLite Delegate） | Stub（aarch64-android）与 Skel（hexagon）必须同 SDK，否则 DSP 固件加载失败 error 1008 |
| ONNX_QNN（QNN EP） | `libonnxruntime.so` 必须是含 QNN EP 的定制构建 |
| LiteRT_NPU（dispatch） | dispatch 编译时的 QNN 版本需与设备库匹配（2.36.x vs 2.37.x 会导致 504） |

---

## 7. QNN 集成细节

### 7.1 TFLITE_NPU（QNN TFLite Delegate）

- 通过 `dlopen("libQnnTFLiteDelegate.so")` 动态加载（**无需 LD_PRELOAD**）
- `backend_type` 必须显式设 `kHtpBackend`（默认是 `kUndefinedBackend`）
- ARM64 上通过 dlsym 调用返回 struct 的函数，须声明为 >16 字节（触发 hidden-pointer ABI，x8 传递）
- 详见 `docs/TFLITE_NPU_DEBUG_LOG.md`

### 7.2 LiteRT_NPU（Qualcomm dispatch）

- `LiteRtCreateEnvironment` 传 `kLiteRtEnvOptionTagDispatchLibraryDir` = `/data/local/tmp/bench_test/qnn`
- 需配置 `LrtQualcommOptions`（backend=HTP、performance=burst、optimization=O3）并 attach 到编译选项，否则 dispatch 报 "Null Qualcomm options"
- Android `libLiteRt.so` 不导出 `Lrt*QualcommOptions*` API → 由 `litert_qualcomm_stubs.cpp` 提供 stub（序列化为 TOML payload）
- **已知问题**：设备上 QNN 库（2.37.0/5.48.0）与 dispatch 期望（2.36.0/5.47.0）不匹配时 `CreateCompiledModel` 返回 **504**，需匹配版本的 QNN 库

### 7.3 ONNX QNN EP

- 使用含 QNN EP 的定制 `libonnxruntime.so`（`deps/onnxruntime/lib/android/qnn/`）
- HTP 完整配置：`backend_type=htp`、`htp_performance_mode=burst`、`htp_graph_finalization_optimization_mode=3`、`enable_htp_fp16_precision=1` 等
- 支持 EP Context 缓存：首次编译生成 `*_epContext.onnx`，后续复用（`kOrtSessionOptionsDisableModelCompile`）

### 7.4 设备端 QNN 目录布局

```
/data/local/tmp/bench_test/qnn/
├── libQnnTFLiteDelegate.so      # TFLite NPU delegate
├── libLiteRtDispatch_Qualcomm.so # LiteRT NPU dispatch（SoC 对应版本）
├── libQnnHtp.so / libQnnSystem.so / libQnnHtpPrepare.so / libQnnSaver.so / libQnnCpu.so / libQnnGpu.so
├── libQnnHtpV73Stub.so / CalculatorStub.so   # 必须与 Skel 同 SDK
├── libQnnHtpV73Skel.so          # Hexagon DSP 固件（与 Stub 同 SDK）
├── libc++.so.1 / libc++abi.so.1 # Hexagon C++ 运行时（v73/G0/pic）
└── (qtld-net-run)               # QNN 官方验证工具
```

### 7.5 QNN_SDK（原生 QNN C API backend）

- 模型格式：**QNN context binary**（`.dlc`/`.serialized.bin`/`.bin`/`.so`），由 `qnn-context-binary-generator` 从 DLC 离线生成；检测到上述后缀自动路由到 QNN SDK backend
- 执行流程（仿 `SampleAppSharedBuffer`）：`dlopen(libQnnHtp.so)` → `QnnInterface_getProviders` → 版本匹配选 `QNN_INTERFACE_VER_TYPE` → `backendCreate` → `deviceCreate` → `dlopen(libQnnSystem.so)` → `systemContextCreate` + `systemContextGetBinaryInfo` → `contextCreateFromBinary` → `graphRetrieve` → `graphExecute`
- **输入/输出缓冲**：Android 上优先用 `rpc_mem`（dlopen `libcdsprpc.so`，无需头文件）分配共享内存并经 `QnnMem_register` 注册为 DMA-BUF 实现零拷贝；失败时回退普通 client buffer
- **量化**：根据 context binary 的 `Qnn_ScaleOffset_t` 做 float↔uint8 量化/反量化，自动适配量化模型
- **QNN 2.48 API 差异**（已在实现中踩坑）：`Qnn_Tensor_t` 无 `QNN_TENSOR_GET/SET_*` 宏，需直接访问 `tensor.v1.*`；`QnnMem_register` 需 `Qnn_MemDescriptor_t`（含 memType=QNN_MEM_TYPE_DMA_BUF、fd）；`QnnSystemContext_getBinaryInfo` 首参为 `sysCtxHandle`；`QnnSystemContext_free` 仅 1 参数；BinaryInfo 用 `contextBinaryInfoV1/V2`（非 `binaryInfoV1`），GraphInfo 用 `graphInfoV1.graphName/numGraphInputs/graphInputs/numGraphOutputs/graphOutputs`
- 编译开关：`-DHAVE_QNN_SDK_BACKEND=ON -DQNN_SDK_ROOT=...`（自动检测 `QnnInterface.h`）；仅 Android

---

## 8. 调试文档索引

| 文档 | 内容 |
|------|------|
| `docs/ONNX_DEBUG_LOG.md` | ORT_API_VERSION 运行时解析、DML/OpenVINO 各 EP 配置 |
| `docs/TFLITE_NPU_DEBUG_LOG.md` | QNN TFLite Delegate 全链路（dlopen、struct ABI、版本匹配）+ 桌面 XNNPACK 崩溃与 TFLite/LiteRT 共存方案 |
| `docs/NCNN_DEBUG_LOG.md` | NaN（clone + packing）、BF16 崩溃（buf_pool）、版本输出、BF16 检测、Vulkan extract 4D 崩溃、退出期 0xC0000409 |
| `docs/MNN_DEBUG_LOG.md` | Vulkan 后端 NaN 分析（BF16 官方不支持、FP32 混合精度路径不稳定、FP16 全链路自洽） |
| `docs/LiteRT_GPU_DEBUG_LOG.md` | DXC 版本过旧导致 D3D12 shader 编译失败 |
| `docs/QNN_SDK_DEBUG_LOG.md` | QNN SDK 后端全链路（2.48 API 差异、BinaryInfo V3、tensor 悬空指针、model.so composeGraphs、libQnnHtpPrepare.so 版本匹配、QNN log callback） |
| `docs/QNN_SDK_HTP_OPTIMIZATIONS.md` | QNN SDK HTP 优化选项汇总（术语表 / 选项速查） |
| `docs/PROJECT_SCORECARD.md` | 项目综合评估与打分机制（D1–D7 评分卡 + 基线统计） |

---

## 9. 已知问题 / 待办

| 项目 | 状态 | 说明 |
|------|------|------|
| LiteRT_NPU `CreateCompiledModel` 504 | 🔄 待解决 | QNN 库版本（2.37/5.48）与 dispatch 期望（2.36/5.47）不匹配；需匹配版本的 QNN 库或新版 dispatch |
| ONNX QNN HTP Android 构建 | 🔄 待解决 | 需 QNN 专用 ORT 构建（onnxruntime-qnn，受 pip/uv 镜像限制） |
| QNN_SDK 真机验证 | 🔄 待验证 | 需用 `qnn-context-binary-generator` 生成 context binary（需 DLC + hexagon SDK），推送 `libQnnHtp.so/libQnnSystem.so/libQnnHtpV73Stub.so/libQnnHtpV73Skel.so` 后实机跑 `QNN_HTP` |
| Android NCNN Vulkan | ⬜ 不可用 | Android NCNN 库编译时未启用 Vulkan（NCNN_VULKAN=0），Vulkan backend 报错不降级 |
| 桌面 TFLITE_XNNPACK_FP16 | ⚠️ 条件可用 | 依赖 FP16 硬件（Intel Iris Xe 支持） |
| MNN 桌面 | ⚠️ 默认关闭 | `HAVE_MNN_BACKEND=OFF`，需显式开启 |
