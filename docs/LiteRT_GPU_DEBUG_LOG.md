# LiteRT_GPU (WebGPU/D3D12) x64 集成与调试完整记录

> **日期**: 2026-07-21
> **环境**: Windows x64, Intel i5-1240P (Iris Xe Graphics), LiteRT SDK, MSVC 14.51
> **结论**: ✅ LiteRT_GPU 已完全可用，根因为 dxcompiler.dll 版本过旧导致 D3D12 shader 编译失败

---

## 1. 背景

在 unified_model_bench 中实现 `LiteRT_GPU` backend，通过 `libLiteRtWebGpuAccelerator.dll` 将 TFLite 模型推理卸载到 GPU。
该 accelerator 内部使用 Google Dawn 库通过 WebGPU API 访问 D3D12（Windows 平台）。

初始状态：运行 `--backend LiteRT_GPU` 时 `LiteRtCreateCompiledModel` 失败，无法初始化 GPU 推理。

---

## 2. 问题现象

### 2.1 运行命令

```powershell
build\win-x64\Release\unified_bench.exe test_model.onnx --repeat 1 --backend LiteRT_GPU --log-level 1
```

### 2.2 关键日志（LiteRT 内部日志，由 glog/gRPC 输出到 stderr）

```
INFO: [gpu_registry.cc:101] Attempting to load GPU accelerator(l).
INFO: [accelerator_registry.cc:54] RegisterAccelerator: ptr=..., name=GPU WebGPU
INFO: [gpu_registry.cc:109] Dynamically loaded GPU accelerator(
    libLiteRtWebGpuAccelerator.dll) registered.

I0000 delegate_webgpu.cc:214] Create WebGPU environment
I0000 environment.cc:522] Selected adapter: Intel(R) Iris(R) Xe Graphics,
    arch=gen-12lp, vendor=intel, backend=Direct3D 12, adapterType=Integrated GPU

E0000 environment.cc:104] Failed creation: Failed to create device:
DXC create compiler failed with <Unknown HRESULT> (0x80004002)
    at CheckHRESULTImpl (third_party/dawn/src/dawn/native/d3d/D3DError.cpp:121)

E0000 delegate_webgpu.cc:238] Failed to initialize WebGPU environment:
    INTERNAL: Device could not be created
E0000 delegate_webgpu.cc:651] Failed to get WebGPU environment:
    kLiteRtStatusErrorRuntimeFailure: Failed to create WebGPU environment

### 应用层日志
litert_backend.cpp:319 LiteRT: CreateCompiledModel failed: 3
benchmark_runner.cpp:384 Init failed: LiteRT_GPU
```

### 2.3 关键观察

1. GPU accelerator DLL 加载成功 — 不是 DLL 缺失问题
2. GPU adapter 识别成功 — Intel Iris Xe 被正确识别为 D3D12 后端
3. **DXC `E_NOINTERFACE` 错误** — 发生在创建 D3D12 设备时，Dawn 调用 DXC 编译 shader 失败
4. `0x80004002` = `E_NOINTERFACE` — COM 接口不支持，通常是 DLL 版本不匹配

---

## 3. 排查过程

### 3.1 初步假设：缺少 dxcompiler.dll

Dawn 在 Windows 上使用 D3D12 后端，shader 编译依赖 DirectX Shader Compiler (DXC) 运行时库。
首先检查构建输出目录是否包含 `dxcompiler.dll`。

```
build\win-x64\Release\
├── dxcompiler.dll     ← 存在！（12 MB）
├── dxil.dll           ← 也存在（1.5 MB）
├── libLiteRt.dll
└── libLiteRtWebGpuAccelerator.dll
```

DLL 已存在，排除缺失假设。

### 3.2 第二阶段：检查 DXC DLL 版本

使用 PowerShell 查询 DLL 文件版本信息：

```powershell
$ver = [System.Diagnostics.FileVersionInfo]::GetVersionInfo("dxcompiler.dll")
# 构建输出中的版本: 10.0.19041.5609 (WinBuild.160101.0800)
```

**对比 Windows SDK 中的版本**：

```powershell
# C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x64\dxcompiler.dll
# 版本: 1.8.2502.11
```

| 来源 | 版本 | 大小 |
|------|------|------|
| 构建输出 | 10.0.19041 (旧版) | 12.6 MB |
| Windows SDK | 1.8.2502.11 (新版) | 14.3 MB |

### 3.3 根因确认

Dawn 项目在运行时通过 `LoadLibrary` 动态加载 `dxcompiler.dll`，调用 `DxcCreateInstance(CLSID_DxcCompiler, ...)` 创建 shader 编译器。
旧版 DXC (v10.0.19041) 不包含 Dawn 需要的 COM CLSID，返回 `E_NOINTERFACE`。

版本号 10.0.19041 对应 Windows 10 20H1 SDK 时代的 DXC，而 Dawn 需要 2024+ 的 DXC redist（语义版本 1.8.x 系列）。

### 3.4 验证：用 dumpbin 确认加载方式

```powershell
& "path\to\dumpbin.exe" /dependents libLiteRtWebGpuAccelerator.dll
```

结果：导入表中**没有** `dxcompiler.dll` — 确认为运行时 `LoadLibrary` 动态加载，无法通过 linker 错误发现。

---

## 4. 解决方案

### 4.1 方案选择

| 方案 | 优点 | 缺点 | 采纳 |
|------|------|------|------|
| 替换为新版 DXC DLL | 简单直接 | 需确定正确版本 | ✅ |
| 添加 PATH 指向 SDK | 无需复制文件 | 用户环境不可控 | ❌ |
| 使用 Vulkan 后端 | 无需 DXC | Dawn 不支持运行时切换 | ❌ |

### 4.2 实施

**步骤 1：手动替换验证**

```powershell
Copy-Item "C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x64\dxcompiler.dll" `
          "D:\WorkSpace\unified_bench\build\win-x64\Release\" -Force
Copy-Item "C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x64\dxil.dll" `
          "D:\WorkSpace\unified_bench\build\win-x64\Release\" -Force
```

验证成功：D3D12 设备创建成功，GPU 推理正常。

**步骤 2：CMake 自动化（CMakeLists.txt）**

在 `HAVE_LITERT_BACKEND` 的 post-build 步骤中添加 DXC DLL 自动搜索和复制：

```cmake
if(HAVE_LITERT_BACKEND)
    # ... existing LiteRT DLL copy ...

    # LiteRT GPU accelerator uses Dawn/WebGPU which requires DXC
    # (DirectX Shader Compiler) at runtime for D3D12 shader compilation.
    # Search Windows SDK for matching DLLs.
    set(DXC_ARCH "x64")
    foreach(sdk_ver "10.0.26100.0" "10.0.22621.0" "10.0.22000.0" "10.0.19041.0")
        set(DXC_DIR "C:/Program Files (x86)/Windows Kits/10/bin/${sdk_ver}/${DXC_ARCH}")
        if(EXISTS "${DXC_DIR}/dxcompiler.dll")
            add_custom_command(TARGET unified_bench_${TARGET_SUFFIX} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${DXC_DIR}/dxcompiler.dll"
                    "${DXC_DIR}/dxil.dll"
                    "${OUT_DIR}/"
                COMMENT "Copying DXC DLLs (${sdk_ver})")
            break()
        endif()
    endforeach()
endif()
```

### 4.3 验证结果

```
LiteRT_GPU avg=12.0 ms  (vs TFLITE_CPU 23.4 ms, 加速 ~2x)
```

完整日志显示 D3D12 路径成功：

```
I0000 delegate_webgpu.cc:243] Created a WebGPU environment.    ← ✅
I0000 delegate_kernel.cc:845] Initializing WebGPU-based API from graph.
LiteRT: init complete (336.4 ms), 2 in, 1 out
LiteRT_GPU avg=12.029 ms
```

---

## 5. 经验总结

### 5.1 DXC (DirectX Shader Compiler) 关键约束

1. **运行时动态加载**：Dawn 通过 `LoadLibrary("dxcompiler.dll")` 加载，导入表无法体现依赖
2. **版本严格匹配**：DXC 采用 COM 接口，`DxcCreateInstance` 根据 CLSID 创建对象；旧版 DLL 不包含新版 CLSID 则返回 `E_NOINTERFACE`
3. **版本号混淆**：旧 DXC 用 Windows SDK 版本号（如 10.0.19041），新 DXC 用语义版本（如 1.8.2502）
4. **需要配套 dxil.dll**：DXC 编译的 shader 使用 DXIL（DirectX Intermediate Language），需要 `dxil.dll` 做运行时签名验证

### 5.2 调试方法论

1. **区分"缺失"与"版本不匹配"**：先确认 DLL 是否存在，再检查版本
2. **理解 COM 错误码**：`E_NOINTERFACE` (0x80004002) ≠ 缺少 DLL → 接口不支持 → 版本问题
3. **工具使用**：
   - `dumpbin /dependents` — 查看静态导入依赖
   - `[System.Diagnostics.FileVersionInfo]` — PowerShell 查询 DLL 版本
   - 对比系统 SDK 版本 — 找最新可用的 DLL

### 5.3 代码设计要点

- 自动化优于手动：通过 CMake 自动搜索 SDK DXC DLL，无需用户干预
- 多版本 SDK 容错：`foreach` 遍历多个 SDK 版本，找到任意一个即可
- LiteRT 内部日志限制：LiteRT C API 无 error reporter，详细错误仅输出到 stderr（glog），CSV 记录的是 `LiteRtStatusStr` 枚举状态码

---

## 6. 关于 "Failed to create OpenCL context" 日志

LiteRT GPU 加速器内部按优先级尝试多个后端：
1. **WebGPU (D3D12)** ← Intel Iris Xe 优先走此路径 ✅
2. **OpenCL** ← Intel 核显不支持，打印 INFO 日志但无影响
3. GPU Environment 创建完成

OpenCL 失败是**无害的 INFO 级别日志**，推理实际使用 D3D12 路径。

---

## 7. 相关文件

| 文件 | 说明 |
|------|------|
| `CMakeLists.txt` | DXC DLL 自动复制逻辑 |
| `src/litert_backend.cpp` | LiteRT backend 实现 |
| `deps/litert/liteRT_runtime/windows_x86_64/` | LiteRT 运行时 DLL |
| `build/win-x64/Release/dxcompiler.dll` | DXC 运行时（由 CMake 自动复制） |
| `build/win-x64/Release/dxil.dll` | DXIL 运行时（由 CMake 自动复制） |


---

## 9. LiteRT_CPU 数值错误：缺失 NCHW -> NHWC 转置（2026-08-31）

> 本节记录 `LiteRT_CPU`（非 GPU）的一个数值正确性 bug。虽然本篇以 GPU 为主，
> 但同属 `src/litert_backend.cpp`，故记在此处。

### 9.1 现象

同一个 `.tflite` 文件（22-in/22-out 的 `tfc_tdf_...` 模型），两个后端差异巨大：

| 后端 | max_output_diff | avg_output_diff |
|---|---|---|
| `TFLITE_CPU` | 0.00002742 | 0.00000035 |
| `LiteRT_CPU` | **3.66572148** | **0.11408538** |

相差约 13 万倍，且 **max/avg ≈ 32**——误差集中在部分元素，不是均匀的精度损失
（后者 max/avg 通常 1~3）。这个分布特征说明"部分数据错了"，而非"整体精度低"。

同时 `transfer_in_ms` 也异常：LiteRT 20.9 ms vs TFLite 43.5 ms。

### 9.2 根因

**LiteRT 后端缺少 NCHW -> NHWC 转置。**

- `InputProvider` 生成的共享输入按 **ONNX 模型的 NCHW** 顺序排列；
- `.tflite` 模型被 LiteRT 消费时按 **NHWC** 解释；
- `tflite_backend.cpp` 在 `feed()` 里处理了这一点（注释：
  *"Shared inputs are in NCHW; TFLite expects NHWC"*）；
- `LiteRTBackend::SetSharedInput()` 只保存了指针，`create_input_buffers()`
  直接 `memcpy` 到输入 buffer——**元素个数对，但顺序错**。

因为元素总数一致，模型能正常跑完、不报任何错误，只是结果错。这正是最难发现的一类
bug：无崩溃、无警告、输出看起来合理。

### 9.3 修复

在拷贝进输入 buffer 时对 4D 输入做与 TFLite 后端**完全一致**的转置
（`input_shapes_[i]` 是模型侧的 NHWC `[N,H,W,C]`，源是 NCHW `[N,C,H,W]`）：

```cpp
dst[n*H*W*C + h*W*C + w*C + c] = src[n*C*H*W + c*H*W + h*W + w];
```

非 4D 输入（rank 1/2/3）保持原样 `memcpy`。

### 9.4 顺带修复：输入 buffer 不足时不再静默截断

原代码：

```cpp
size_t copy_bytes = input_elems_[i] * sizeof(float);
if (copy_bytes > buffer_size) { copy_bytes = buffer_size; }   // 静默截断
```

截断后尾部是 `_aligned_malloc` 的**未初始化内存**，模型会在垃圾数据上运算。
本次排查中该分支**没有触发**（不是本次 bug 的成因），但按项目"绝不静默降级"的
规则，已改为显式报错：

```
LiteRT: input N needs X bytes but the compiled model only provides Y
```

### 9.5 验证

修复后 `max_output_diff` **3.66572148 -> 0.00002919**，与 TFLite 的 0.00002742
同一量级。

### 9.6 为什么转置逻辑保持重复（未提取公共函数）

项目里有三处布局转换，语义**并不相同**，强行统一会引入错误：

| 位置 | 变换 | 说明 |
|---|---|---|
| `tflite_backend.cpp` | NCHW -> NHWC | 4D 逐元素置换 |
| `litert_backend.cpp` | NCHW -> NHWC | 与上者**逐字符相同** |
| `ncnn_backend.cpp` | `[1,C,H,W]` -> `Mat(w,h,c)` | **折叠 batch + 按通道分块 memcpy**，是 CHW 平面布局，**不是 NHWC** |

因此只有 TFLite / LiteRT 两处理论上可合并，NCNN 必须独立。考虑到仅两处、且
转置只有 12 行，当前选择保持重复 + 明确注释，避免"看起来统一实则语义不同"的坑。

另：`tools/onnx_convert.py` 有第四处独立的 NCHW<->NHWC 逻辑（`L = [0,3,1,2]`、
`nhwc_perm()`、`nhwc_axis()`），用于给 onnx2tf 生成参数替换文件。它与 C++ 侧是
同一套数学的独立实现——**改一处不会自动同步另一处**，修改时需两边都看。
