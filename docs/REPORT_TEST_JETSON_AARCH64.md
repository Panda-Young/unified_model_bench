# Jetson Orin Nano — unified_bench 推理测试报告
**模型**: `test_model` 多框架基准模型（输入 `[1,3,224,224]` + `[1,48,1,1]`，输出 `[1,10]` softmax 概率）
**设备**: Jetson Orin Nano（JetPack 6.0 / R36.4.4，Ubuntu 22.04 aarch64，内核 5.15.148-tegra，arm Cortex-A78AE `CPU part 0xd42`，6 核，7.62 GB RAM）
**构建配置**: CPU-only（GPU/OpenCL/Vulkan/OpenVINO 关闭）；threads=4，repeat=100，warmup=5

---

## 1. Linux aarch64 依赖库落位方式（全网预构建库调研结论）

按 aarch64 类型逐一核对了网上可用的预构建库，最终**没有直接可用的 C++ 链接库**，均改为板上源码编译：

| 框架 | 落位结论 | 详细 |
|---|---|---|
| **ONNX Runtime** | ✅ 复用板载预编译 | CPU-only `1.30.0` + **GPU 版 `1.19.0`（CUDA/TensorRT）**，glibc≥2.28，取自 Jetson 官方 Linux aarch64 包，直接链接运行 |
| **ncnn** | ❌ 预构建不可用 → 源码编译 | Arch Linux ARM `ncnn-*.pkg.tar.xz` 的 `libncnn.so` 需要 `libglslang.so.16` 且 GLIBC≥2.38，板载 glibc 2.35 无法加载；pip 轮子只导出 Python C-API（`nm` 确认无 `_ZN4ncnn*`）。源码编译 ncnn（`1.0.20260526`）生成静态 `libncnn.a`，关 Vulkan，无 glslang 依赖 |
| **MNN** | ❌ 预构建 ABI 不匹配 → 源码编译 | aarch64 pip 轮子的 `_mnncengine.so` 用旧 `_GLIBCXX_USE_CXX11_ABI=0` 编译，`getSessionInputAll` 等 ABI 不兼容；源码编译 MNN **3.6.1**（导出 `B5cxx11` 符号）生成 `libMNN.so`+`libMNN_Express.so` |

> 关键经验：网上 aarch64 的 ncnn/MNN「预构建库」多为 Python 扩展或为过新 glibc 编译，**无法直接链接进 C++ 基准工具**，源码编译是可靠路径。

---

## 2. 工具自身性能（推理延迟，repeat=100, threads=4）

| 后端 | 平均延迟 | 最大延迟 | 相对 ONNX CPU | 初始化 | 结果说明 |
|---|---|---|---|---|---|
| **ONNX_CPU** | **54.36 ms** | 55.75 ms | 1.000x（基准） | 7.13 ms | 复用板载 ORT 1.30.0 |
| **NCNN_CPU** | **37.50 ms** | 38.82 ms | **1.449x** | 2.97 ms | 源码编译 ncnn |
| **NCNN_CPU_FP16** | **37.49 ms** | 39.79 ms | **1.450x** | 2.65 ms | 与 FP32 基本一致 |
| **MNN_CPU** | **21.73 ms** | 22.10 ms | **2.501x** | 4.66 ms | 源码编译 MNN，最快 |

**结论**：同模型 4 线程下，MNN_CPU 最快（约 46 FPS），是 ONNX CPU 的 2.5 倍；ncnn 其次（约 26.7 FPS，比 ONNX 快 45%）。

### 2b. ONNX Runtime GPU（CUDA / TensorRT）加速

接入 Onyx Jetson 版 **ONNX Runtime 1.19.0 GPU 包**（`onnxruntime-linux-aarch64-gpu-1.19.0`，通过实验验证 CUDA 与 TensorRT EP 均可用后并入 deps）：

| 后端 | 平均延迟 | 相对 ONNX CPU | 初始化 | diff_max | peak 内存 |
|---|---|---|---|---|---|
| ONNX_CPU | 56.1 ms | 1.000x（基准） | 5.1 ms | 0 | 73.6 MB |
| **ONNX_CUDA** | **6.9 ms** | **8.08x** | 179.7 ms | 1.9e-7 | 543.8 MB |
| **ONNX_TensorRT** | **3.6 ms** | **15.47x** | 22699 ms(首次建引擎) | 5.3e-7 | 1612 MB |

**要点**
- 纯 CPU 精度与 GPU 两 EP 输出**一致**（diff ≤ 5e-7），分类判定不变。
- **TensorRT 最快**：3.6 ms（~278 FPS/277 FPS），比 ONNX_CPU 快 15.5x，比 MNN_CPU(21.7ms) 快 6x。
- CUDA EP 约 6.9 ms，8x。
- 代价：GPU 驱动上下文要大得多——CUDA 占用 RMS ~544 MB、TensorRT 峰值 ~1.6 GB（预分配引擎缓存），且首次建引擎慢（22.7s，TensorRT 引擎固化到磁盘后可复用）。

**无法运行的变体**（本机构建不含）：
- `ONNX_oneDNN`/全系 `ONNX_OpenVINO_*` — OpenVINO 共享库缺失（ConfigureEP/加载失败）
- `NCNN_CPU_BF16` — Orin CPU 未暴露 BF16 指令集（ncnn 检测不支持）
- `NCNN_VK*` — 构建时 Vulkan 关闭
- `MNN_OpenCL/VULKAN/OPENGL/NN` — 请求的 GPU/加速后端本机构建未启用（不静默回退 CPU）

---

## 3. 精度/结果准确性分析（对照 ONNX CPU 参考输出）

参考 dump 实测：**argmax 索引 = 5（val 0.1020），前三大 [5,4,2]**，10 类概率约 [0.0985…0.102]。

各后端全程峰值输出偏差（`max_output_diff`）：

| 后端 | max_output_diff | 相对参考**|
|---|---|---|
| ONNX_CPU（基准自比） | 0.00000000 | 完全一致 |
| NCNN_CPU | 0.00003560 | 顶部1e-5 量级 |
| NCNN_CPU_FP16 | 0.00003560 | 同上（FP16 计算，偏差可忽略） |
| MNN_CPU | 0.00000001 | 一致到 1e-8 |
| ONNX_CUDA | 0.00000019 | 1e-7 量级 |
| ONNX_TensorRT | 0.00000053 | 5e-7 量级 |

**结论**：ncnn / MNN / ONNX(CUDA·TensorRT) 的最终分类结果（argmax=5、概率分布、top3 顺序）与 ONNX CPU 参考**逐一吻合**，偏差最大仅 ~3.6e-5（CPU）与 ~5e-7（GPU），均不影响分类判定。六后端部署数值等价。

---

## 4. 资源占用统计（推理进行中采样 worker 进程 + tegrastats）

**进程采样**（worker PID，`/proc/<pid>/stat` utime+stime 差分 + VmRSS）：

| 后端 | 平均 CPU% | 峰值 CPU% | 常驻 Rss 峰值 | 推理延迟 |
|---|---|---|---|---|
| ONNX_CPU | 399% | 405% | 79.3 MB | 54.4 ms |
| NCNN_CPU | 346% | 352% | 97.1 MB | 37.5 ms |
| NCNN_CPU_FP16 | 348% | 354% | 96.9 MB | 37.5 ms |
| MNN_CPU | 398% | 402% | **28.7 MB** | **21.7 ms** |
| **ONNX_CUDA** | 399% | 406% | 73.7 MB | **6.9 ms** |
| **ONNX_TensorRT** | 399% | 404% | 73.4 MB | **3.6 ms** |

**整体平台资源**（`tegrastats`，`GR3D_FREQ` / `VDD_IN`）：

| 项 | 空闲 | CPU 推理 | GPU(CUDA/TRT)推理 | 说明 |
|---|---|---|---|---|
| GPU GR3D 频率 | 0% | **0%** | **~0–99%（有负载时 GR3D 起转）** | 纯 CPU 后端未用 GPU；GPU EP 真正拉 GPGPU |
| 板级总功率 VDD_IN | ~4.6 W | ~7.0–7.8 W | ~6.8 W | 推理负载 |
| RAM 总使用 | 2352/7620 MB | ~2611 MB | ~3.0+ GB（TRT 预分配） | GPU 上下文占大量内存 |
| CPU 时钟 | idle 729 MHz | 满载 1344 MHz | 满载 1344 MHz | GPU 推理 I/O 仍吃 CPU |

**数据要点**
- **MNN_CPU 纯 CPU 最省**：内存仅 28.9 MB，21.7 ms。
- **TensorRT GPU 最快**：3.6 ms（CUDA 6.9ms），比 CPU 快 15.5x，代价是单进程 ~1.6 GB 内存 + 首次建引擎 22.7s。
- 纯 CPU 三后端 GR3D=0%；真正 GPU 加速须用 ONNX_CUDA / ONNX_TensorRT，二者已在 deps 连同 GPU 版 ORT 一并提供。

---

## 4. 交付物

- 板上构建目录：`/home/nvidia/ub/build-linux-aarch64/unified_bench`
- 报告及背景调研：本文件 `docs/REPORT_TEST_JETSON_AARCH64.md`、`docs/libncnn-so-aarch64-glibc-research.md`
- 源码改动（已提交）：`CMakeLists.txt`、`cmake/backends.cmake`、`src/backend_registry.cpp`、`src/onnx_backend.cpp`
- 依赖 staging 脚本（已提交）：`tools/setup_linux_aarch64_deps.sh` —— 从 `ORT_GPU_LIBDIR`/`MNN_BUILD_DIR`/`NCNN_*` 还原 `deps/{onnxruntime,mnn,ncnn}/lib/linux-aarch64/`
- 已提交依赖库：onnxruntime GPU 版 `libonnxruntime.so`1.19.0 + providers_{tensorrt,shared}.so、MNN 3.6.1（linux-aarch64）。**`libonnxruntime_providers_cuda.so`（~291 MB）超出 GitHub 100 MB 单文件上限，不入 git**，需用脚本从自备 GPU 包拷贝