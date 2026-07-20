# TFLITE_NPU (QNN TFLite Delegate) 集成与调试完整记录

> **日期**: 2026-07-19  
> **环境**: SM8550 (Snapdragon 8 Gen 2), Android 16, QNN SDK 2.48.40.260702, TFLite 2.18.0  
> **结论**: ✅ TFLITE_NPU 已完全可用，通过 dlopen 动态加载 + 匹配 SDK 版本库实现 HTP 推理，无需 LD_PRELOAD

---

## 1. 背景

在 unified_model_bench 中实现 `TFLITE_NPU` backend，通过 Qualcomm QNN TFLite Delegate (`libQnnTFLiteDelegate.so`) 将 TFLite 模型推理卸载到 Hexagon DSP (HTP) 上。

初始状态：`CreateDelegate` 中只有一条 `LOGW` 占位日志，返回 `nullptr`，模型实际在 CPU 上执行。

---

## 2. 问题演进与解决

### 2.1 阶段一：实现 Delegate 加载（→ Segfault）

**做法**: 通过 `dlopen("libQnnTFLiteDelegate.so")` + `dlsym()` 动态加载 delegate 创建函数。

**现象**: `Segmentation fault`，崩溃在 `dlopen` 阶段。

**排查**: tombstone 显示 `strlen(NULL)`：
```
signal 11 (SIGSEGV), fault addr 0x0000000000000000
#00 __strlen_aarch64+16 (libc.so)
#01 libQnnTFLiteDelegate.so
```

**根因**: `libQnnTFLiteDelegate.so` 的构造函数在 dlopen 时运行，内部触发 QNN 初始化并崩溃。

**解决**: 放弃 `dlopen`，改用 `dlsym(RTLD_DEFAULT, ...)` ——只在库已被预加载时尝试使用。

---

### 2.2 阶段二：LD_PRELOAD + dlsym（→ 仍然 Segfault）

**做法**: 在 benchmark 启动前通过 `LD_PRELOAD=libQnnTFLiteDelegate.so` 预加载库，代码中通过 `dlsym(RTLD_DEFAULT, "TfLiteQnnDelegateCreate")` 获取函数指针。

**现象**: 仍然 Segfault，日志输出 `step 3b: dlopen → step 3c: dlsym → step 3d → step 3e: calling TfLiteQnnDelegateOptionsDefault → CRASH`。

**根因**: `TfLiteQnnDelegateOptionsDefault` 函数返回一个 value-type struct，在 ARM64 ABI 上：
- ≤16 字节 struct：通过 x0:x1 寄存器返回
- \>16 字节 struct：通过 hidden pointer (x8) 返回

我们最初声明 struct 为 16 字节（`int32_t data[4]`），但实际 struct **>16 字节**，导致 ABI 不匹配——函数通过 x8 写数据，但我们从 x0:x1 读，读取到的是垃圾指针。

**解决**: 声明 struct 为 256 字节（`int64_t data[32]`），触发 ARM64 hidden-pointer ABI，编译器正确设置 x8 传递目标地址。

> **关键知识点**: 在 ARM64 上通过 dlsym 调用返回 value-type struct 的 C 函数时，struct 大小决定 ABI 路径（≤16B vs >16B），必须正确声明大小。

---

### 2.3 阶段三：Delegate 创建成功，但 Interpreter 创建失败

**做法**: 256 字节 struct 正确获取 `TfLiteQnnDelegateOptionsDefault` 返回值，传递给 `TfLiteQnnDelegateCreate`。

**现象**:
```
TFLite: TfLiteQnnDelegate created    ← delegate 创建成功
ERROR: Restored original execution plan after delegate application failure.
```

**排查**: 
- 尝试强制设置 `backend_type=2` (HTP) → 无效
- 尝试 `ADSP_LIBRARY_PATH=./qnn` (替代 `./qnn/hexagon`) → 无效
- 尝试不同模型（Conv+ReLU, mix_precision_sample）→ 全部失败
- TFLite 版本检查：我们使用 2.18.0，QLIRT 文档兼容 2.10.1/2.17.0

**初步怀疑**: `libQnnTFLiteDelegate.so` 与 TFLite 版本不兼容，或 QNN runtime 版本不匹配。

---

### 2.4 阶段四：qtld-net-run 验证（→ 定位真正根因）

**做法**: 使用 QNN SDK 自带的 `qtld-net-run` 测试工具验证 delegate 可用性。

**命令**:
```bash
cd /data/local/tmp/bench_test
LD_LIBRARY_PATH=./qnn ADSP_LIBRARY_PATH=./qnn \
  ./qnn/qtld-net-run \
  --model mix_precision_sample.tflite \
  --input input_list.txt \
  --backend htp --log_level 5
```

**输出关键信息**:
```
VERBOSE: Replacing 3 out of 3 node(s) with delegate  ← delegate 能替换 ops

ERROR: [Qnn] QnnDsp <E> Stub lib id mismatch:
  expected (v2.48.40.260702151143)
  detected (v2.46.0.260424121129)
ERROR: [Qnn Delegate] Failed to create device_handle, error=1008
```

**根因确定**: **Stub library 版本不匹配**！设备上的 `libQnnHtpV73Stub.so` 是 v2.46，而 SDK 的 `libQnnHtpV73Skel.so` 是 v2.48。DSP 固件加载时版本校验失败。

---

### 2.5 阶段五：匹配 SDK 版本（→ ✅ 成功）

**做法**: 从 QNN SDK 2.48.40.260702 推送所有匹配版本的库到设备：

```bash
# HTP Stub (aarch64-android)
adb push SDK/lib/aarch64-android/libQnnHtpV73Stub.so ./qnn/
adb push SDK/lib/aarch64-android/libQnnHtpV73CalculatorStub.so ./qnn/

# HTP backend + system
adb push SDK/lib/aarch64-android/libQnnHtp.so ./qnn/
adb push SDK/lib/aarch64-android/libQnnHtpPrepare.so ./qnn/
adb push SDK/lib/aarch64-android/libQnnSystem.so ./qnn/

# HTP Skel (hexagon DSP firmware)
adb push SDK/lib/hexagon-v73/unsigned/libQnnHtpV73Skel.so ./qnn/
adb push SDK/lib/hexagon-v73/unsigned/libQnnHtpV73.so ./qnn/

# TFLite Delegate
adb push SDK/lib/aarch64-android/libQnnTFLiteDelegate.so ./qnn/
```

**结果**: 
```
TFLite: TfLiteQnnDelegate created
VERBOSE: Replacing 1 out of 1 node(s) with delegate
TFLITE_NPU avg=0.292 ms  ← HTP 推理成功！
```

---

### 2.6 阶段六：去除 LD_PRELOAD 依赖（→ ✅ dlopen 方案）

**背景**: 原先方案依赖 `LD_PRELOAD=./qnn/libQnnTFLiteDelegate.so` 预加载库，再用 `dlsym(RTLD_DEFAULT, ...)` 解析符号。这要求用户设置自定义环境变量，不够优雅。

**做法**: 改用 `dlopen` 直接加载 delegate 库，用返回的 handle 做 `dlsym`：

```cpp
qnn_lib_ = dlopen("libQnnTFLiteDelegate.so", RTLD_NOW | RTLD_GLOBAL);
auto delegate_create = dlsym(qnn_lib_, "TfLiteQnnDelegateCreate");
// ... 创建 delegate，Cleanup 时 dlclose(qnn_lib_)
```

**遇到的问题与解决**:

1. **"Restored original execution plan" 再次出现**: 即使 ld_preload 也失败。原以为是 dlopen vs ld_preload 差异，但用 `qtld-net-run` 验证 QNN 后端正常。

2. **根因定位**: `TfLiteQnnDelegateOptionsDefault()` 将 `backend_type` 初始化为 `kUndefinedBackend` (0)，导致 delegate 不知道使用哪个 QNN 后端。必须手动设置为 `kHtpBackend` (2)：
   ```cpp
   ((int32_t *)&opts)[0] = 2;  /* kHtpBackend, offset 0 */
   ```

3. **struct 缓冲区扩容**: 原始 256 字节可能不够，扩大到 2048 字节（`int64_t[256]`）。

**最终结果**: ✅ 无需 `LD_PRELOAD`，纯 `dlopen` 方案完美工作：
```
TFLITE_NPU avg=3.7 ms  (vs CPU 42.1 ms, 加速 11.4x)
VERBOSE: Replacing 13 out of 13 node(s) with delegate
```

---

## 3. 最终架构

### 3.1 设备端库文件结构

```
/data/local/tmp/bench_test/qnn/
├── libQnnTFLiteDelegate.so     # QNN TFLite Delegate (SDK 2.48)
├── libLiteRtDispatch_Qualcomm.so  # LiteRT dispatch
├── libQnnHtp.so                # QNN HTP backend (SDK 2.48)
├── libQnnHtpPrepare.so         # QNN HTP prepare (SDK 2.48)
├── libQnnSystem.so             # QNN system (SDK 2.48)
├── libQnnHtpV73Stub.so         # HTP v73 stub (SDK 2.48) ★ 必须匹配
├── libQnnHtpV73Skel.so         # HTP v73 DSP firmware (SDK 2.48) ★ 必须匹配
├── libQnnHtpV73.so             # HTP v73 backend
├── libQnnHtpV73CalculatorStub.so
├── libQnnGpu.so                # GPU backend
├── libQnnCpu.so                # CPU backend
└── libonnxruntime.so           # ONNX Runtime (QNN EP)
```

### 3.2 运行时环境变量

```bash
LD_LIBRARY_PATH=.:./qnn                # 库搜索路径（含 libQnnTFLiteDelegate.so）
ADSP_LIBRARY_PATH=./qnn                # DSP 固件搜索路径（含 Skel.so）
# 无需 LD_PRELOAD！通过 dlopen 动态加载
```

### 3.3 代码流程 (`tflite_backend.cpp`)

```
TFLITE_NPU 初始化:
  ├── dlopen("libQnnTFLiteDelegate.so", RTLD_NOW | RTLD_GLOBAL)
  │    └── 失败则 fallback CPU（无需 LD_PRELOAD）
  ├── dlsym(qnn_lib_, "TfLiteQnnDelegateCreate")
  ├── dlsym(qnn_lib_, "TfLiteQnnDelegateOptionsDefault")
  ├── opts_fn() → 2048 字节 struct → ARM64 hidden-pointer ABI
  ├── 设置 backend_type = kHtpBackend (2)
  ├── delegate_create(&opts) → TfLiteDelegate*
  ├── TfLiteInterpreterOptionsAddDelegate()
  ├── TfLiteInterpreterCreate() → 内部触发 delegate->Prepare()
  │    └── QNN graph 编译 → HTP 固件加载 → 推理就绪
  └── Cleanup: TfLiteQnnDelegateDelete() + dlclose(qnn_lib_)
```

### 3.4 降级策略

| 条件 | 行为 |
|------|------|
| dlopen 失败（库未找到） | LOGW → CPU fallback |
| dlsym 符号未找到 | LOGW → CPU fallback |
| delegate_create 失败 | LOGW → CPU fallback |
| TfLiteInterpreterCreate 失败 | LOGE → 记录失败到 CSV |

---

## 4. 经验总结

### 4.1 QNN TFLite Delegate 关键约束

1. **库版本必须严格匹配**: Stub.so (aarch64-android) 与 Skel.so (hexagon DSP) 必须是同一 SDK 版本，否则 DSP 固件加载失败（error 1008）
2. **backend_type 必须显式设置**: `TfLiteQnnDelegateOptionsDefault()` 初始化为 `kUndefinedBackend`，需手动设为 `kHtpBackend` (2) 才能启用 HTP 推理
3. **ARM64 struct 返回 ABI**: 通过 dlsym 调用返回 value-type struct 的函数时，struct 大小决定 ABI 路径，必须正确声明
4. **ADSP_LIBRARY_PATH**: 应指向包含 Skel.so 的目录（与 QNN .so 同目录），而非子目录

### 4.2 调试方法论

1. **tombstone 分析**: `adb logcat -b crash` 获取精确的崩溃调用栈和寄存器状态
2. **官方工具验证**: 先用 `qtld-net-run` 排除代码问题，确认 delegate 本身可用
3. **逐层排查**: dlopen → dlsym → struct ABI → DLL 版本 → SDK 匹配
4. **verbose 日志**: `--log_level 5` 在 qtld-net-run 中输出 QNN 内部详细信息

### 4.3 代码设计要点

- 动态加载优于静态链接：无需 QNN SDK 头文件，编译不依赖 QNN
- 优雅降级：delegate 不可用时自动 fallback CPU，不影响其他 backend
- 零自定义环境变量：使用 `dlopen` 动态加载 delegate，无需 `LD_PRELOAD`；库路径通过标准 `LD_LIBRARY_PATH` 控制
- ARM64 struct ABI 兼容：2048 字节不透明缓冲区确保 struct 不溢出；`backend_type` 通过偏移 0 设置

---

## 5. 相关文件

| 文件 | 说明 |
|------|------|
| `src/tflite_backend.cpp` | TFLITE_NPU 实现 |
| `include/backend_interface.hpp` | BackendId 枚举定义 |
| `tools/NDK_build_Android_auto.bat` | Android NDK 构建脚本 |
| `deps/onnxruntime/lib/android/qnn/` | QNN 运行时库（需从 SDK 更新） |
| `memories/repo/qnn_delegate_symbols.md` | QNN delegate 符号表备忘 |

---

## 6. Desktop TFLITE_XNNPACK Crash (2026-07-21)

### 问题现象
Windows x64 Desktop (Intel Iris Xe GPU) 运行 TFLITE_XNNPACK 时，程序打印：
```
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
```
后立即崩溃（无任何错误日志），退出码 1。

### 调试过程

1. **定位崩溃点**：在 `TfLiteXNNPackDelegateCreate` 调用前后添加 `LOGI`+`fflush` 日志，
   发现 TFLite 内部日志打印后，我们的日志从未出现 → 崩溃在 `TfLiteXNNPackDelegateCreate` 内部。

2. **SEH 捕获异常**：用 `__try/__except` 包裹调用，捕获到：
   - 异常码：`0xC0000005` (STATUS_ACCESS_VIOLATION)
   - 崩溃地址：`0x00007FF8471785F0`
   - 崩溃模块：`libLiteRt.dll`

3. **根因分析**：
   - `libLiteRt.dll` 是 Google LiteRT 运行时库（`deps/litert/liteRT_runtime/windows_x86_64/`）
   - `libLiteRt.dll` 是 Google LiteRT 运行时（也导出完整 TFLite C API）
   - `tensorflowlite_c.dll` 是标准 TFLite C API DLL
   - 两个 DLL 都导出 21 个 `TfLite*` 符号（`TfLiteModelCreate`, `TfLiteInterpreterCreate`, `TfLiteXNNPackDelegateCreate` 等）
   - `libLiteRt.lib` 是**导入库**（不是静态库），但导出 1053 个符号（LiteRt* + TfLite*）
   - Linker 将 `TfLite*` 符号解析到 `libLiteRt.dll` → XNNPACK delegate 在该 DLL 中有 bug（ACCESS_VIOLATION）
   - 基础 TFLite 操作（模型加载、解释器创建、CPU 推理）在 `libLiteRt.dll` 中**正常工作**

4. **最终解决方案（两者共存）**：
   - **不链接** `tensorflowlite_c.dll.if.lib`（避免符号冲突）
   - 所有 TFLite C API 函数从 `libLiteRt.lib` 获取（→ `libLiteRt.dll`），经验证完全正常
   - **仅 XNNPACK delegate 创建** 使用 `LoadLibrary("tensorflowlite_c.dll")` + `GetProcAddress` 动态加载
   - LiteRT 后端正常链接 `libLiteRt.lib`，调用 `LiteRt*` 函数
   - 两者在同一进程中和平共存

### 验证结果（三者同时运行）
```
TFLITE_CPU         avg=31.6 ms   ← 使用 libLiteRt.dll 的 TFLite C API
TFLITE_XNNPACK     avg=16.7 ms   ← XNNPACK 从 tensorflowlite_c.dll 动态加载
LiteRT_CPU         avg=39.9 ms   ← LiteRT 原生 API
```

### 关键教训
- **同名 DLL 导出冲突**：两个 DLL 都导出相同的 `TfLite*` 符号，linker 选其一；无法通过链接顺序控制
- **按需动态加载**：仅对有 bug 的函数使用 `GetProcAddress`，无需重构整个后端
- **`libLiteRt.dll` 可用作 TFLite 运行时**：其 TFLite C API 实现完整可用，仅有 XNNPACK delegate 有 bug
- **跨平台差异**：Android 上两者各用独立 .so（ELF 无冲突），Windows 上导入库重叠需要特殊处理
