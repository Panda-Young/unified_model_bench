# ONNX Runtime Backend 修复记录

> **日期**: 2026-07-20
> **范围**: `src/onnx_backend.cpp`, `src/benchmark_runner.cpp`

---

## 1. ORT_API_VERSION 硬编码 → 运行时动态获取

### 问题

`onnx_backend.cpp` 中 `base->GetApi(ORT_API_VERSION)` 硬编码编译时头文件的 API 版本号（如 22）。当运行时链接不同版本的 `libonnxruntime.so` 时，API 版本不匹配 → `GetApi` 返回 NULL 或行为异常。

### 修复

通过 `GetVersionString()` 获取运行时版本字符串（如 `"1.22.0"`），解析小版本号作为 API 版本：

```cpp
// 旧: 硬编码编译时版本号
ort_ = base->GetApi(ORT_API_VERSION);

// 新: 运行时动态获取
const char *ort_ver_str = base->GetVersionString(); // "1.22.0"
ort_api_ver_ = 1;  // fallback
if (ort_ver_str) {
    LOGI("ONNX Runtime version: %s", ort_ver_str);
    const char *dot = strchr(ort_ver_str, '.');
    if (dot) ort_api_ver_ = (uint32_t)atoi(dot + 1);  // 22
}
ort_ = base->GetApi(ort_api_ver_);
```

同时将两处 DML `GetExecutionProviderApi("DML", ...)` 中的 `ORT_API_VERSION` 替换为成员变量 `ort_api_ver_`。

### 关键改动

- 添加成员变量 `uint32_t ort_api_ver_ = 1`
- 三处调用全部改用运行时版本

---

## 2. 经验总结

| 问题 | 教训 |
|------|------|
| ORT_API_VERSION 硬编码 | 任何 SDK API 版本号都不应硬编码编译时常量，必须运行时获取 |
| 版本兼容性 | `GetVersionString()` 返回格式 `major.minor.patch`，minor == API version |

---

## 3. ONNX_QNN_HTP 性能回退（~30ms ≈ CPU → 1.7ms / 17.5x）

> **日期**: 2026-08-05
> **范围**: `src/onnx_backend.cpp`（QNN EP HTP 配置）、设备 `/data/local/tmp/bench_test/qnn/` 库部署
> **结论先行**: 不是代码逻辑问题，是**设备上 ORT/QNN 库版本错配** + **libcdsprpc 直连被 SELinux 拦截**。

### 3.1 问题现象

- `ONNX_QNN_HTP` 实测 ~28-55 ms（≈CPU），accel 仅 1.0-1.3x（期望 ~3ms / 13x+）
- 每次运行都打印 `EP Context will be created`（epContext 缓存从不落盘/复用）
- QNN profiling CSV（`/data/local/tmp/qnn_htp_profiling.csv`）从不生成
- GPU backend 正常（`ONNX_QNN_GPU` 6.9ms / 4.2x，epContext 正常生成）

### 3.2 排查过程（时间线）

1. **确认 HTP 未真正执行**：profiling CSV 未生成 + 推理时间≈CPU → 图静默回退到 CPU EP。
2. **logcat 定位**（关键）：
   ```
   QNNExecutionProvider ... SetupBackend failed ... Failed to create device. Error:
   QNN_DEVICE_ERROR_INVALID_CONFIG: Invalid config values
   open_device_node failed ... /dev/adsprpc-smd-secure, /dev/adsprpc-smd. (errno 13, Permission denied)
   avc: denied { search } ... scontext=u:r:shell:s0 tcontext=u:object_r:adsprpcd_file:s0
   ```
   → HTP backend **CreateDevice 失败**，根因是 fastrpc（libcdsprpc）连不上 DSP。
3. **为什么之前要自定义 libcdsprpc**：设备 `/vendor/lib64/libcdsprpc.so` 缺 `remote_register_buf_attr2`，而 **1.28 版 ORT**（21.6MB，7-25 被误换）的 QNN EP 在 `enable_htp_shared_memory_allocator=1` + **epContext 缓存命中**时需要该符号 → 最初报 `undefined symbol: remote_register_buf_attr2`。
4. **fastrpc 源码编译 libcdsprpc.so 失败**：从 GitHub `quic/fastrpc` 编译的库能提供 attr2，但它是**直连模式**——`open("/dev/adsprpc-smd")` 被 **SELinux 拒绝**（shell 域无权访问 `vendor_qdsp_device`），而设备上根本没有 `/dev/fastrpc-cdsp`（SM8550 是老的 `adsprpc-smd` 命名，已改 `inc/fastrpc_ioctl.h` 宏验证过，仍 errno 13）。
5. **strace 揭示系统库真实机制**：
   ```
   openat("/dev/adsprpc-smd", O_RDONLY|O_NONBLOCK) = -1 EACCES   # 系统库也 open 失败！
   openat("/vendor/lib64/vendor.qti.hardware.dsp@1.0.so", ...) = 5  # 但它加载了 HIDL
   ```
   → **系统 libcdsprpc.so 通过 HIDL（`vendor.qti.hardware.dsp@1.0` binder 服务）连 DSP**，后台服务进程（system 域）才有权 open 设备节点。**shell 域用户态直连永远被拒**。
6. **onnx_test 工程对比实验**（`Script_and_config/C_CPP/onnx_test`，QNN_HTP 1.16ms）：用的是 **14.2MB 1.22 ORT**（`onnxruntime_1.22.0-QNN_2.46.0`）+ **系统 libcdsprpc** + `ADSP_LIBRARY_PATH=/data/local/tmp/qnn/hexagon` → **不需要 attr2**，HIDL 正常。

### 3.3 根因

| # | 根因 | 说明 |
|---|------|------|
| 1 | **`enable_htp_shared_memory_allocator=1` + epContext 缓存命中** | QNN EP 加载缓存时初始化 rpcmem shared memory，需要 `remote_register_buf_attr2`（系统 libcdsprpc 没有）→ CreateSession 失败（avg=0ms）；**首次生成不需要 attr2** |
| 2 | **自定义 libcdsprpc 死路** | fastrpc 源码版是直连模式（无 HIDL 支持），shell 域 open `/dev/adsprpc-smd` 被 SELinux 拒，**无法替代系统库** |
| 3 | （早期误判）设备 ORT 被换 1.28 | 实测 **1.28 在 `shared_allocator=0` 下同样能跑 HTP**（1.86ms/21.8x），attr2 与 ORT 版本无关 |

### 3.4 解决方案

1. **`enable_htp_shared_memory_allocator=0`**（`src/onnx_backend.cpp`）：=1 时缓存命中需 attr2；=0 则首次生成与缓存命中都正常。**这是修复的关键**。
2. **使用 deps 的 qnn 版 ORT**：`deps/onnxruntime/lib/android/qnn/arm64-v8a/libonnxruntime.so`（20.7MB，1.26）push 到设备 `qnn/libonnxruntime.so`（1.28 亦可，但保持与 deps 一致）。
3. **不使用自定义 libcdsprpc**：删除设备 `qnn/libcdsprpc.so`（如有），走系统库 HIDL。
4. **`ADSP_LIBRARY_PATH` 指向 skel 所在目录**（`./qnn` 或 `…/qnn/hexagon`，与部署布局一致）。
5. 清理旧 epContext 缓存后重新生成（配置变更后必须删缓存，否则 0ms）。

### 3.5 修复后结果（SM8550 / test_model.onnx）

| 后端 | avg (ms) | accel |
|------|---------|-------|
| ONNX_CPU | 28.9 | 1.00x |
| ONNX_QNN_HTP（首次生成） | 1.51 | 19.2x |
| **ONNX_QNN_HTP（epContext 缓存命中）** | **1.64** | **17.6x** |
| ONNX_QNN_GPU（缓存命中） | 6.9 | 4.2x |

### 3.6 关键教训

| 问题 | 教训 |
|------|------|
| shared_allocator | `enable_htp_shared_memory_allocator=1` 在 **epContext 缓存命中**时需要 `remote_register_buf_attr2`（系统 libcdsprpc 无）→ 缓存加载失败 avg=0ms；**=0 则无需 attr2，缓存命中正常**。**与 ORT 版本无关**（1.26/1.28 均可） |
| 直连 vs HIDL | Qualcomm fastrpc：用户态库走 HIDL（`vendor.qti.hardware.dsp@1.0`），**shell 域无法直接 open `/dev/adsprpc-smd`**（SELinux）。编译 fastrpc 源码替代系统库是死路 |
| 静默回退 | QNN EP backend 初始化失败会**静默回退 CPU**（session 创建成功、推理正常但慢）。判断依据：profiling CSV 不生成 + 耗时≈CPU。**已实现检测：生成模式下 epContext 未生成 → 报错标记失败**（见 3.7） |
| epContext 缓存 | 配置项（如 shared_allocator）变更后旧缓存不兼容 → 加载后 avg=0ms；**改配置必须删缓存** |
| 调试手段 | `logcat` 能看到 ORT QNN 内部日志（`qnn_backend_manager.cc`）；`strace -f -e trace=openat` 能看库实际 open 哪些节点/加载哪些 HIDL |

### 3.7 静默回退 CPU 检测（设计文档：绝不静默降级）

README 设计原则：后端初始化失败必须**报错标记失败**（`avg=0.0ms, accel=-1.00x`），**绝不静默降级**。ORT QNN EP 在 backend setup 失败时静默回退 CPU（session 创建成功、推理正常但慢、无错误），此前会把 CPU 耗时误记成 QNN 结果。

`src/onnx_backend.cpp` 的 `Initialize` 在 CreateSession 成功后增加检测：

```cpp
/* QNN EP 生成模式下：若 epContext 文件未生成 => QNN EP 未 partition 任何节点
 * （静默回退 CPU）=> 报错 return false，runner 标记 avg=0, accel=-1
 * 注：仅 GPU/HTP 会写 epContext；QNN CPU backend 不写（每次重编译、本就近 CPU 速度），
 *     故排除在检测外 */
if ((id_ == ONNX_QNN_GPU || id_ == ONNX_QNN_HTP) &&
    !ep_cache_hit && !file_readable_nonzero(ep_context_path_.c_str())) {
    LOGE("ONNX: QNN EP did NOT activate - no epContext was generated: %s. "
         "The QNN backend silently fell back to CPU; marking this backend FAILED ...");
    last_error_ = "ONNX: QNN EP inactive (silent CPU fallback) - no epContext generated";
    ort_->ReleaseSession(session_);
    session_ = nullptr;
    return false;
}
```

验证结果：
- **正常**：epContext 生成 → 通过（ONNX_QNN_HTP 1.45ms / 20x；ONNX_QNN_GPU 7.1ms / 4.0x）；
- **回退**（移走 `libQnnHtp.so`）：报 `QNN EP did NOT activate` → `Init failed: ONNX_QNN_HTP` → `avg=0.000 ms, accel=-1.00x`（不再记录误导性的 CPU 耗时）；
- **排除**：`ONNX_QNN_CPU` 不写 epContext，不参与检测（正常记录，≈CPU 速度）。

---

## 4. ONNX_DML_GPU 初始化失败：DML 图融合 + LayerNormalization（DirectML 版本）

> **日期**: 2026-08-05
> **范围**: `src/onnx_backend.cpp`（DML EP 配置）、`deps/onnxruntime/lib/win-x86/dml/`（DirectML.dll）
> **结论先行**: 两个独立问题叠加——① DML 图融合编译崩溃（`DmlGraphFusionHelper` E_INVALIDARG）；② ORT 1.22 的 LayerNormalization DML 内核在 DirectML < 1.15 上创建失败。前者用 `ep.dml.disable_graph_fusion=1` 绕过，后者靠升级 **DirectML.dll 1.15.4**（x86，NuGet `Microsoft.AI.DirectML`）修复。

### 4.1 问题现象

- `ONNX_DML_GPU` 初始化报：
  - `DmlGraphFusionHelper.cpp(365) ... 80070057 参数错误`（E_INVALIDARG）
  - 或（禁用图融合后）`MLOperatorAuthorImpl.cpp(2410) ... 80070057`
- `ONNX_DML_NPU` 报 `No devices detected that match the filter criteria`（本机无 NPU，属正常）

### 4.2 排查过程（时间线）

1. **确认正确的 DirectML.dll**：ORT 1.22 DML 构建自带
   `deps/onnxruntime/lib/win-x86/dml/DirectML.dll`（**1.12.1**），由 `onnxruntime.dll`
   delay-load，从 onnxruntime.dll **自身所在目录**加载（实测：exe 目录放损坏 DLL 无效；
   dml 目录放损坏 DLL → `DML GPU V2 append failed`）。
   - `C:\Windows\System32\directml.dll` / `SysWOW64\directml.dll`（v1.0.200713，Win10 内建）太旧，
     ORT 1.22 要求 DML feature level 5.0 → 旧版连 DML 设备都建不了（`dml_provider_factory.cc:520`）。
2. **第一层失败 = DML 图融合**：ORT 1.17+ 默认把分区内所有节点编译成一个 DML 图
   （`DmlGraphFusionHelper`），任一个算子不被图编译器接受就整体失败。
   → 增加会话配置 `ep.dml.disable_graph_fusion=1`，退回**逐算子 kernel + CPU 回退**。
3. **第二层失败 = LayerNormalization 内核**：用 `tools/test_dml_ops.py` 逐个算子最小模型测试，
   其余 17 种算子（LSTM/Conv/Slice/Split/Tile/Erf…）全部通过，**单独一个 LayerNorm 必现**
   `MLOperatorAuthorImpl.cpp(2410)`。ORT 的 DML LayerNorm 内核用
   `DML_OPERATOR_MEAN_VARIANCE_NORMALIZATION2` + DML 图算子实现。
4. **版本相关性验证**：1.12.1 / 1.13.1 均崩溃 → 一度误判"与版本无关"；实测升级 **DirectML 1.15.4**
   后 LayerNorm 内核恢复正常。**教训：版本区间测试要覆盖到已修复版本，不要用"抽样版本都失败"推断"所有版本都失败"。**

### 4.3 当前解决方案

- **win-x86**：`deps/onnxruntime/lib/win-x86/dml/DirectML.dll` 升级为 **1.15.4**（x86，16,666,680 字节，哈希 `96503335...2450D8C7`）。
- **win-x64**：`deps/onnxruntime/lib/win-x64/dml/DirectML.dll` 升级为 **1.15.4**（x64，18,527,776 字节，哈希 `9C9E6D82...B4E92DA1`，取自 NuGet `Microsoft.AI.DirectML` 1.15.4 的 `bin/x64-win/`）。注意 Python `onnxruntime\capi\DirectML.dll` 同为 1.15.4 但哈希不同（多 16 字节签名），未采用。
- `src/onnx_backend.cpp`：DML_GPU / DML_NPU 始终设置 `ep.dml.disable_graph_fusion=1`（原因见 4.5）。

### 4.4 模型层面（备选兜底方案）

若在旧 DirectML 上仍要跑：模型含 **60 个 LayerNormalization**，仅两族配置——
- `axis=-1`（48 个，1D scale `(32/8/64/16/128)`，ε=1e-5）；
- `axis=-2`（12 个，scale `(108,128)`，ε=1e-9）。

可用 `ReduceMean→Sub→Pow→ReduceMean→Add(ε)→Sqrt→Div→Mul(scale)→Add(bias)` 替换，
让 DML 按基础算子执行（均为 DML 确定支持的算子）。注意：ORT `LayerNormFusion`
图优化器可能把该模式**重新融合回 LayerNorm**，需实测验证是否生效（与 `ep.dml.disable_graph_fusion`
是两回事——前者是 ORT CPU 侧融合，后者是 DML 图编译）。

### 4.5 精度注意（重要）——融合 ON 是"算错"，不是"精度损失"

受控对比实验（win-x64，ONNX_DML_GPU，repeat=100，DirectML 1.15.4；diff 只统计
第一个输出 `output[1,4,2049,1]` 的 8196 个元素，见 `result_collector.cpp::CompareWithBaseline`）：

| 配置 | avg (ms) | max_output_diff | avg_output_diff | accel vs CPU |
|---|---|---|---|---|
| 图融合**启用** | **36.7** | **2.311517** | **1.064580** | 1.06x |
| 图融合**禁用**（当前） | 43.8 | 0.000005 | 0.0000005 | 1.00x |

关键结论：
- 融合 ON 的 **avg_output_diff=1.065**（非个别离群点）：整个音频输出平均绝对误差 >1.0，
  是 DML 融合图的**真实数值错误**，不是可接受的精度噪声。融合只换来 ~16% 提速，
  **不值得**产出错误结果 → 必须保持 `ep.dml.disable_graph_fusion=1`。
- 融合 ON 不再崩溃（DirectML 1.15.4 修好了图编译崩溃），但图编译后的数值不正确。

### 4.6 为什么"关掉图融合后 DML 还在 GPU 上但没加速"？

**根因：加速本来就来自图融合；逐算子执行在 Intel iGPU 上 ≈ CPU 速度。**

- 该模型是流式语音分离模型：1026 节点，全是小算子（237 Slice / 180 Transpose /
  108 Concat / 60 LayerNorm / 12 LSTM）。每个算子的 GPU kernel 极小，无可并行性，
  而**每次 DML 算子派发都有固定开销**（上传/下载 + 派发，几十 µs 级）。
  逐算子模式下 ~1026 次派发开销 ≈ CPU 完成全部计算的时间。
- 图融合把整个分区编译成**一个** DML 图 = 一次派发，消除逐算子开销 → 这才是之前
  看到的 ~2x 加速的来源（但那套融合图输出是错的，见 4.5）。
- 实测（x64）：CPU ≈39ms、DML 融合关 ≈43.8ms、DML 融合开（错误）≈36.7ms。
  Intel iGPU 上 DML 对本模型**本质没有加速空间**（x86 融合关也只有 ~1.1-1.6x，
  且 x86 CPU 本身偏慢 ~70ms）。
- **结论**：本模型在 Intel iGPU 上 DML 不值得作为加速路径。真正的 DML 收益在
  独显（dGPU）/ NPU 上；本机若需加速，优先走 QNN NPU（见第 3 节，1.7ms）或
  直接用 x64 CPU（39ms）。

### 4.7 排查工具

- `tools/test_dml_ops.py`：生成单算子最小 ONNX（`dml_op_tests/`），逐个在 DML 上定位失败算子。
- `tools/bisect_dml.py`：按拓扑前缀二分定位首个失败节点（注意：其成功判定有缺陷，`avg=0.000`
  也可能是失败记录，最终以 `test_dml_ops.py` 结论为准）。
- `tools/analyze_layernorm.py`：盘点模型全部 LayerNorm 的 axis / scale / epsilon 分布。

---

## 5. ORT C API 返回值检查规范（杜绝 `(void)` 丢弃）

> **日期**: 2026-08-05
> **范围**: `src/onnx_backend.cpp`

### 5.1 问题

ORT C API 的函数几乎都返回 `OrtStatus*`，非空即失败。此前大量调用写成 `(void)ort_->...`
直接丢弃返回值，配置项设置失败、IO 元数据查询失败都会被静默吞掉。

### 5.2 修复

新增文件级辅助函数 `OrOk`，统一"记录日志 + 写入 `last_error_` + 释放 status"：

```cpp
static bool OrOk(const OrtApi *api, OrtStatus *st, const char *what, std::string &err)
{
    if (!st) return true;
    const char *msg = api ? api->GetErrorMessage(st) : "unknown error";
    LOGE("ONNX: %s failed: %s", what, msg);
    err = std::string("ONNX: ") + what + ": " + msg;
    api->ReleaseStatus(st);
    return false;
}
```

调用点：**全部 ORT C API 调用**（所有返回 `OrtStatus*` 的调用都已统一走 `OrOk`），例如：

```cpp
if (!OrOk(ort_, ort_->AddSessionConfigEntry(opts_, "ep.dml.disable_graph_fusion", "1"),
          "AddSessionConfigEntry(ep.dml.disable_graph_fusion)", last_error_)) return false;
```

覆盖范围：
- **会话配置/选项**：`CreateEnv` / `CreateSessionOptions` / `SetIntraOpNumThreads` /
  `SetSessionGraphOptimizationLevel` / `EnableCpuMemArena` / `AddSessionConfigEntry`
  （XNNPACK、DML graph fusion、QNN epContext）/ `CreateSession` / `GetAllocatorWithDefaultOptions`；
- **EP 挂载**：oneDNN / DML GPU(V2/V1) / DML NPU / OpenVINO(CPU/GPU/GPU_FP16/NPU, V2/V1) /
  NNAPI / XNNPACK / QNN(htp/gpu/cpu) 的 `SessionOptionsAppendExecutionProvider*`；
- **IO 元数据**：`SessionGet[In|Out]putCount/Name/TypeInfo` / `CastTypeInfoToTensorInfo` /
  `GetDimensions(Count)`；
- **运行**：`CreateCpuMemoryInfo` / `CreateTensorWithDataAsOrtValue`（失败释放已建 values）/
  `Run`（warmup + repeat，失败释放 input/output）/ `GetTensorMutableData`
  （运行循环内失败则跳过拷贝并记录，不中断整轮）。

### 5.3 对 `last_error_` 的影响（重要）

`OrOk` 会把失败统一写成 `"ONNX: <what>: <ort 错误信息>"`，与手工写法的差异：

| 场景 | 旧行为 | 新行为（OrOk） |
|------|--------|----------------|
| 失败路径 | 部分失败**不写** `last_error_`（XNNPACK append、`SessionGetInputCount`、`CreateCpuMemoryInfo`、warmup/run Run 等），仅靠调用方兜底写笼统信息 | **每次失败都写**具体信息 |
| 消息格式 | 不统一（有的 `"ONNX: CreateSession failed - X"`，有的 `"ONNX: DML GPU: X"`，有的只 LOGE 不写） | 统一 `"ONNX: <what>: X"` |
| oneDNN | bug：`last_error_` 误写成 `"ONNX: OpenVINO NPU: ..."` | 修复为 `"ONNX: oneDNN append: ..."` |
| DML GPU V1 回退 | device=1 失败不落盘（成功后 last_error_ 干净） | device=1 探测失败会写 `last_error_`，**回退成功后需 `last_error_.clear()`**（已处理） |

### 5.4 经验总结

| 问题 | 教训 |
|------|------|
| `(void)` 丢弃返回 | ORT C API 返回 `OrtStatus*`，失败必须记录并 abort，禁止 `(void)` 吞掉 |
| 统一辅助函数 | `OrOk` 模式：失败 → LOGE + `last_error_` + `ReleaseStatus` + `return false` |
| 成功路径残留错误 | 探测式失败（如 DML V1 device=1→0 回退）成功后必须 `last_error_.clear()`，避免污染成功记录 |

---

## 6. ONNX_OpenVINO_GPU 编译失败排查（2026-08-27，scnet_tfc_tdf_v3_20260821.onnx）

### 6.1 现象

``
.\build\win-x64\Release\unified_bench.exe "...\scnet_tfc_tdf_v3_20260821.onnx" --backend ONNX_OpenVINO_GPU --repeat 1
``

失败，日志：

``
[GPU] ProgramBuilder build failed!
[GPU] can't get group dimension for data layout
ONNX: CreateSession failed: Exception during initialization: ... backend_manager.cc:188 ...
``

### 6.2 排查过程（排除法）

| 步骤 | 测试 | 结果 |
|------|------|------|
| 1 | ONNX_CPU | 正常（avg≈18ms） |
| 2 | ONNX_OpenVINO_CPU | 正常（avg≈11ms, accel≈1.3x） |
| 3 | ONNX_OpenVINO_GPU（scnet 模型） | **失败** can't get group dimension for data layout |
| 4 | ONNX_OpenVINO_GPU（test_model.onnx） | 正常（avg=5.7ms, accel=1.91x） |
| 5 | ONNX_OpenVINO_GPU（online_convert_1frame.onnx） | 正常（avg=13.3ms） |
| 6 | OpenVINO 官方 enchmark_app -d GPU（scnet 模型） | **同样失败**，复现同一错误 |
| 7 | 删除/恢复 model_cache 的 blob | 均失败，排除缓存问题 |

### 6.3 根因

**OpenVINO GPU 插件（openvino_intel_gpu_plugin.dll 2025.1.0）无法编译 scnet 模型的特定 group conv 布局。**

对比模型结构：

| 模型 | GPU 结果 | group conv 种类 |
|------|---------|----------------|
| online_convert_1frame | 成功 | 仅 kernel=(3,3) group=128 |
| test_model | 成功 | 无 group conv |
| **scnet_tfc_tdf_v3_20260821** | **失败** | **kernel=(1,3) group=2/40/80（9 个）+ kernel=(3,3) group=40/80/160/320（30 个）** |

scnet 是 **1D 时序卷积用 4D 张量表示**（H=1），其中 9 个 down_convs.* 的 kernel_shape=[1,3] group conv（H 维 kernel=1）在 Intel GPU 插件的 program_builder.cpp:165 解析 data layout 时无法推断 group 维度 → 编译崩溃。

### 6.4 关于"之前可以运行"与 15:45 的 blob 缓存

model_cache/3678230654899558819.blob（4.5MB）看似是 scnet 的 GPU 缓存，经分析：

- blob 内容**无任何 GPU 内核标记**（无 ocl::、_gpu_、cldnn）；
- 删除后跑 ONNX_OpenVINO_CPU **重新生成了完全同名的 blob**（同大小 4.5MB）；
- **结论：该 blob 是 OpenVINO CPU 的编译缓存**，scnet 模型在 GPU 上**从未成功编译过**；
- 用户记忆中的"之前可以运行"实为 **OpenVINO_CPU**（scnet 在 CPU 上一直正常）或**其他模型在 GPU 上**（online_convert/test_model 均可）。

### 6.5 结论与建议

1. **该模型当前无法在 OpenVINO GPU 上运行**，是 OpenVINO GPU 插件对"4D 表示的 1D group conv（kernel=(1,3)）"的编译限制，非 unified_bench 代码问题；
2. 可用 ONNX_OpenVINO_CPU 获得 OpenVINO 加速（accel≈1.3x）；或改用 DML/其他 GPU 后端；
3. 如需 GPU，可尝试：onnxsim 折叠动态 shape 后重试、把 kernel=(1,3) group conv 展开为普通卷积、或升级/降级 OpenVINO GPU 驱动与插件。

### 6.6 二次深入定位（2026-08-27 补充，精确到插件源码）

上一轮已确认是 OpenVINO GPU 插件问题。本轮用 OpenVINO Python API 对 IR 做**逐节点二分编译**，把失败点精确定位到单个 op：

| 步骤 | 方法 | 结果 |
|------|------|------|
| 1 | ovc 把 onnxsim 折叠模型转成 IR（1011 个 op） | 转换成功（ONNX 前端 OK，说明模型本身合法） |
| 2 | OpenVINO Python 逐个 GroupConvolution 前缀子图编译 GPU | **全部 OK**（含 down_convs H=1 group conv、conv/depthwise H=3） |
| 3 | 细粒度扫描 op[800] 之后每 5 个 op | 首个失败在 op[845] 附近 |
| 4 | 精确扫描 op[830]~op[850] | **首个失败 op[843] Concat_80** |
| 5 | 测试 Concat_80 的 3 个输入分支 | **分支[0]（su_layers.0，含 H=1 group ConvTranspose）单独编译即 FAIL** |
| 6 | 测试 su_layers.0 的 GroupConvolutionBackpropData | **全部 FAIL**（op[821]/[868]/[980]） |

**决定性证据**：su_layers.0/deconv/deconv.0/ConvTranspose（GroupConvolutionBackpropData，输入 [1,160,1,122] H=1，权重 5D [160,1,1,1,3]）**单独子图编译 GPU 就失败**，报：

`
[GPU] ProgramBuilder build failed!
[GPU] can't get group dimension for data layout
`

**插件源码根因**（openvino/src/plugins/intel_gpu/src/runtime/layout.cpp:84-86）：

`cpp
tensor::value_type layout::group() const {
    const auto& dims = get_dims();
    if (!format::is_weights_format(format)) {
        throw std::logic_error("[GPU] can't get group dimension for data layout");
    }
    ...
}
`

即 GPU 插件把 **data layout（输入张量）误当成 weights format 调用了 group()**，抛 can't get group dimension for data layout。触发条件是 **H=1（spatial=1）的 group 转置卷积（GroupConvolutionBackpropData，kernel=(1,3)）**，在解码器（su_layers）多分支结构中 layout 推断出错。

### 6.7 修复可行性评估

| 方案 | 可行性 | 说明 |
|------|--------|------|
| 修 OpenVINO 插件源码 | ❌ | 第三方插件，unified_bench 无法修改 |
| onnxsim 折叠/固定 shape | ❌ | 已测试仍失败，非动态 shape 问题 |
| 模型重写（H 维 Squeeze 成 3D） | ⚠️ 理论可行 | 需转换 18 个 H=1 group conv/ConvTranspose 及全部中间张量，风险高、未验证 GPU 能通过 |
| HETERO/配置绕过 | ❌ | 已测试失败（GPU 子图编译本身崩） |
| **换替代后端** | ✅ **推荐** | 见下 |

**推荐替代后端**（均已在 summary.csv 验证成功）：

| 后端 | avg | 说明 |
|------|-----|------|
| ONNX_OpenVINO_CPU | 11ms / 1.3x | 最快，OpenVINO CPU 加速 |
| ONNX_CPU | 18ms / 1.0x | 基准 |
| ONNX_DML_GPU | 17ms / 0.62x | DML GPU，可用 |
| MNN_OpenCL | 11ms / 0.94x | MNN GPU |

**结论**：ONNX_OpenVINO_GPU/GPU_FP16 对本模型不可用，是 OpenVINO GPU 插件对 H=1 group 转置卷积的编译 bug。用 ONNX_OpenVINO_CPU 或 ONNX_DML_GPU/MNN_OpenCL 替代。
