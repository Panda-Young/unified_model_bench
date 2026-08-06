# QNN SDK Backend 开发排障记录

> **日期**: 2026-08-05
> **范围**: `src/qnn_backend.cpp`、`src/backend_registry.cpp`、`src/model_loader.cpp`、`src/benchmark_runner.cpp`、`include/model_format.hpp`、`include/backend_interface.hpp`、`CMakeLists.txt`、`tools/NDK_build_Android_auto.bat`
> **SDK**: QNN 2.48.40.260702（QAIRT）· 设备 SM8550（hexagon-v73）

---

## 0. 架构总览

QNN SDK backend（`QNN_SDK_CPU/GPU/HTP`，id 500-502）支持 **两种模型形态**：

| 形态 | 文件特征 | 加载方式 | 特点 |
|------|---------|---------|------|
| **context binary** | 纯二进制（`.serialized.bin`/`.bin`/`.dlc`） | 读文件 → `contextCreateFromBinary` → `graphRetrieve` | 离线编译，backend 专用（HTP binary 不能跑 CPU） |
| **model library（model.so）** | ELF 共享库（`.so`），导出 `QnnModel_composeGraphs` | `dlopen` → `composeGraphs` → `graphFinalize` | 运行时编译，**同一 .so 可用于 CPU/GPU/HTP** |

**自动识别**：文件头 `\x7fELF`（64 位）→ model.so；否则 context binary。

**model.so 流程**（仿 SampleApp）：
```
dlopen(model.so) → dlsym(QnnModel_composeGraphs / QnnModel_freeGraphsInfo)
→ backendCreate → deviceCreate → contextCreate（新建）
→ composeGraphs(backend, *qnn_, context, nullptr, 0, &gi, &ng, false, nullptr, level)
→ graphFinalize(graph, nullptr, nullptr)   ← 必须！
→ 从 gi[0]->graph / inputTensors / outputTensors 提取元数据
```

---

## 1. 编译期：QNN 2.48 API 差异（20 个错误）

### 1.1 handle 类型带下划线

| 错误写法 | 正确写法 |
|---------|---------|
| `QnnBackend_Handle_t` | `Qnn_BackendHandle_t` |
| `QnnContext_Handle_t` | `Qnn_ContextHandle_t` |
| `QnnGraph_Handle_t` | `Qnn_GraphHandle_t` |
| `QnnLog_Handle_t` | `Qnn_LogHandle_t` |
| `QnnMem_Handle_t` | `Qnn_MemHandle_t` |

### 1.2 `Qnn_Tensor_t` 无 GET/SET 宏

2.48 移除了 `QNN_TENSOR_GET_*` / `QNN_TENSOR_SET_*` 宏，需**直接访问字段**：

```cpp
// Qnn_Tensor_t = { version, union { v1, v2 } }
tensor.v1.name / .dataType / .rank / .dimensions / .quantizeParams
tensor.v1.memType / .clientBuf.data / .clientBuf.dataSize / .memHandle
```

v1/v2 **公共前缀布局完全相同**（id/name/type/dataFormat/dataType/quantizeParams/rank/dimensions/memType/clientBuf 字段一致，v2 尾部多 isDynamicDimensions/sparseParams/isProduced），故统一按 v1 读写对 v2 同样有效，**执行 tensor 强制 `version = QNN_TENSOR_VERSION_1`**。

### 1.3 接口版本宏

```cpp
// QnnInterface_t { backendId; providerName; apiVersion; union { QNN_INTERFACE_VER_NAME }; }
providers[i]->apiVersion.coreApiVersion.major == QNN_API_VERSION_MAJOR   // 2
providers[i]->apiVersion.coreApiVersion.minor >= QNN_API_VERSION_MINOR   // 37
qnn_ = &providers[i]->QNN_INTERFACE_VER_NAME;
```

系统接口同理：`QNN_SYSTEM_INTERFACE_VER_TYPE` / `QNN_SYSTEM_INTERFACE_VER_NAME`，需 include `<QNN/System/QnnSystemInterface.h>`。

### 1.4 函数签名（易错点）

```cpp
QnnSystemContext_getBinaryInfo(sysCtxHandle, buf, size, &info, &infoSize); // 首参是 sysCtxHandle！5 参
QnnSystemContext_free(sysCtxHandle);                                        // 仅 1 参，无 profile
QnnContext_create(backend, device, config, &context);                      // 4 参（含 device）
QnnContext_createFromBinary(backend, device, config, buf, size, &ctx, profile);
QnnGraph_retrieve(context, graphName, &graph);                             // 3 参
QnnMem_register(context, &desc, 1, &handle);                               // desc 用 Qnn_MemDescriptor_t
```

### 1.5 `QnnMem_register` 需要 descriptor（非 int 数组）

```cpp
Qnn_MemDescriptor_t desc = QNN_MEM_DESCRIPTOR_INIT;
desc.dataType = dt;
desc.memType = QNN_MEM_TYPE_DMA_BUF;
desc.dmaBufInfo.fd = fd;          // rpc_mem_to_fd 得到的 dma-buf fd
desc.dmaBufInfo.data = p;
qnn_->memRegister(context_, &desc, 1, &h);
```

### 1.6 BinaryInfo / GraphInfo 结构（2.48 用 V3！）

```cpp
// 2.48 的 context binary，systemContextGetBinaryInfo 返回 version=3！
bin_info->contextBinaryInfoV3.numGraphs / .graphs   // V1→contextBinaryInfoV1, V2→contextBinaryInfoV2
graphs[i].graphInfoV3.graphName / .numGraphInputs / .graphInputs / .numGraphOutputs / .graphOutputs
// GraphInfoV1/V2/V3 字段名相同，只需按 version 选 union 成员
```

> **注意**：`graphRetrieve` 用 null name 返回 **6003**。未处理 V3 时 `num_graphs` 读到 0。

### 1.7 void* → 函数指针转换（Android clang）

- `reinterpret_cast<void* → 函数指针>` **非法**（不同对象类别）→ 用 `memcpy`：
  ```cpp
  void *sym = dlsym(lib, "xxx");
  memcpy(&fn_ptr, &sym, sizeof(sym));
  ```
- **链式赋值陷阱**：`rpc_alloc_ = rpc_free_ = nullptr;` 报
  `incompatible function pointer types assigning to 'rpc_mem_alloc_fn_t' from 'rpc_mem_free_fn_t'`
  （`(rpc_free_=nullptr)` 类型是 `rpc_mem_free_fn_t`，无法赋给不同函数指针类型）→ **分行赋值**。

### 1.8 枚举比较 warning

```cpp
// -Wenum-compare: Qnn_Definition_t vs Qnn_QuantizationEncoding_t
s->quantizeParams.encodingDefinition == (Qnn_Definition_t)QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
```

### 1.9 头文件包含

`QnnSystemContext.h` 内部包含兄弟头 → 需同时 `-I ${QNN_SDK_ROOT}/include` 与 `-I ${QNN_SDK_ROOT}/include/QNN`。

---

## 2. 构建 / 注册问题

### 2.1 `--backend QNN_SDK_HTP` 报 "Unknown backend name"

**现象**：bat 重新配置后（删除 CMakeCache），QNN_SDK_* 全部无法识别。

**根因**：`NDK_build_Android_auto.bat` 的 cmake 配置**缺少 `-DHAVE_QNN_SDK_BACKEND=ON`**。该脚本每次删除 CMakeCache 重新配置，宏回到默认 OFF → `qnn_backend.cpp` 编译为空、`backend_registry.cpp` 中注册块（`#ifdef HAVE_QNN_SDK_BACKEND`）被跳过。

**修复**：bat cmake 配置加：
```bat
-DHAVE_QNN_SDK_BACKEND=ON ^
-DQNN_SDK_ROOT="!QNN_SDK_ROOT!"
```

### 2.2 `is_qnn_sdk_backend` 范围 bug → GPU/CPU 被漏掉

**现象**：QNN 变体只显示 1 个 backend（只有 HTP）。

**根因**：
```cpp
// 错误：起点是 HTP(502)，CPU(500)/GPU(501) 被排除
return v >= bid(BackendId::QNN_SDK_HTP) && v <= bid(BackendId::QNN_SDK_LAST);
```
**枚举**：`QNN_SDK_CPU=500, QNN_SDK_GPU=501, QNN_SDK_HTP=502`（注意 HTP 不是 500！）

**修复**：
```cpp
return v >= bid(BackendId::QNN_SDK_CPU) && v <= bid(BackendId::QNN_SDK_LAST);
```

### 2.3 命名统一 `HAVE_QNN_SDK_BACKEND`

CMake 选项名（`HAVE_QNN_SDK_BACKEND` / `QNN_SDK_ROOT`）必须与源码宏、bat 参数、vscode 配置**保持一致**。曾出现 CMake 改为 `HAVE_QNN_BACKEND`/`QNN_ROOT` 但源码宏未同步，导致宏对不上、后端未编译。改名需全量同步：
`CMakeLists.txt` / `src/*.cpp` / `tools/*.bat` / `.vscode/c_cpp_properties.json` / `README.md`。

---

## 3. 运行期：context binary 路径

### 3.1 temp backend 用 ONNX 查询 QNN shapes → Protobuf 失败

**现象**：
```
Creating temp backend (id=0) to query shapes...
ONNX: CreateSession failed: Load model from libtest_model.so failed:Protobuf parsing failed.
Cannot determine input shapes for QNN
```

**根因**：`benchmark_runner.cpp` 对非 NCNN 变体一律用默认 CPU（ONNX_CPU）临时后端查输入形状；QNN 模型不是 ONNX，解析失败。

**修复**：对 `ModelFormat::QNN` 变体，candidate 改为 QNN SDK 后端（跳过 ONNX CPU）：
```cpp
if (fmt == ModelFormat::QNN) {
    for (auto &bc : backends)
        if (is_qnn_sdk_backend(bc.id)) candidate_ids.push_back(bc.id);
} else {
    candidate_ids.push_back(cpu_id);
}
```

### 3.2 "no compatible interface provider in libQnnHtp.so"

**现象**：`dlopen` 成功但版本匹配失败。

**根因**：设备上 QNN 库是**旧版本**（API < 2.37）。对比哈希：
```
本地 2.48: 734865C1...   设备: 983bb2c2...   ← 不一致！
```
（设备库 7/22 推送，来源旧 SDK）

**修复**：推送 2.48 SDK 全套库（见 §5.1）。

### 3.3 `graphRetrieve` rc=6003

**根因**：BinaryInfo 只处理了 V1/V2，2.48 实际返回 **V3** → `num_graphs=0` → graph name 取不到 → `graphRetrieve(context, nullptr, ...)` 失败。

**修复**：新增 `qnn_binary_graphs()` / `qnn_graph_view()` 统一支持 V1/V2/V3（§1.6）。

### 3.4 `graphExecute` rc=6004（INVALID_TENSOR）

**现象**：context 恢复、graph 检索都成功，执行时报 6004。

**根因**：`*dst = *src` 复制 tensor 时，`dimensions`/`name` **指针直接指向 binary info 内部**。`systemContextFree` 释放 binary info 后指针**悬空**，`graphExecute` 读维度访问无效内存。

**修复**：`qnn_setup_tensor()` 把 dimensions/name **复制到自有存储**（`std::vector<uint32_t>` / `std::string`），并强制 `version=V1`。

---

## 4. 运行期：model.so 路径

### 4.1 识别：model.so 是 ELF 不是 context binary

`libtest_model.so` 是**真正的 ELF 共享库**（NDK r26c 构建，导出 `QnnModel_composeGraphs`/`QnnModel_freeGraphsInfo`），不能 `fread` 后 `contextCreateFromBinary`。

**识别**：文件头 ELF64 → 走 model.so 路径（`is_elf_shared_object()`）。

### 4.2 `graphExecute` rc=6023（GRAPH_NOT_FINALIZED）

**根因**：`composeGraphs` 返回的 graph **未 finalized**，直接执行报 6023。

**修复**：composeGraphs 后必须：
```cpp
qnn_->graphFinalize(graph_, nullptr, nullptr);
```

### 4.3 HTP composeGraphs rc=4（MODEL_GRAPH_ERROR）—— 经典案例

**现象**：
```
QNN: composeGraphs rc=4 graphs=0
[ERROR] QnnModel::initialize() not able to create graph in given context.
```
CPU/GPU 正常，仅 HTP 失败。**用 qnn-net-run 同款命令却能成功**（`./qnn-net-run --backend libQnnHtp.so --model libtest_model.so ...` 正常 3.16ms）。

**排查思路**：
1. 加 **QNN log callback**（见 §5.2）捕获 backend 内部日志 → 定位到：
   ```
   QnnDsp <E> PrepareLibLoader Failed dlsym libQnnHtpPrepare.so
   QnnDsp <E> HTP Prepare backend loading failed. Aborting
   QnnGraph_create done. status 0x3f0
   ```
2. 确认设备上 `libQnnHtpPrepare.so` 是旧版本（7/22，未随新库更新）。

**根因**：HTP 的 `graphCreate`（model.so 现场编译图）需要动态加载 **`libQnnHtpPrepare.so`**（prepare 库）；旧版本与新 `libQnnHtp.so` 不匹配，dlsym 失败 → 图创建中止。

**修复**：推送 2.48 SDK 的 `libQnnHtpPrepare.so`（87.9MB）。

**修复后**：CPU 16.8ms / GPU 7.0ms / HTP 3.4ms（repeat=5），与 qnn-net-run 一致。

### 4.4 model.so wrapper 类型需自建

SDK 的 `include/` 下**没有** `QnnWrapperUtils.hpp`（SampleApp 用自带副本）。`GraphInfo_t`/`GraphConfigInfo_t`/`ModelError_t` 及 `ComposeGraphsFnHandleType_t` 需在工程内自行定义（与 SampleApp 一致）。`composeGraphs` 第 2 参 `QNN_INTERFACE_VER_TYPE` **按值**传（非指针）。

---

## 5. 关键工具 / 技巧

### 5.1 设备 QNN 库清单（必须全套同版本！）

来自 `C:\Qualcomm\AIStack\QAIRT\2.48.40.260702`：
- `lib\aarch64-android\`：libQnnHtp.so、libQnnSystem.so、libQnnCpu.so、libQnnGpu.so、**libQnnHtpPrepare.so**、libQnnHtpV73Stub.so、libQnnHtpV73CalculatorStub.so
- `lib\hexagon-v73\unsigned\`：libQnnHtpV73Skel.so

> 版本不匹配症状：`no compatible interface provider`（libQnnHtp.so 旧）、`PrepareLibLoader Failed`（libQnnHtpPrepare.so 旧）。判断用 `Get-FileHash` vs `adb shell md5sum` 对比。

### 5.2 QNN log callback（诊断 backend 内部错误）

```cpp
// 第 4 参是 va_list（不是 void*）！
static void qnn_log_callback(const char *fmt, QnnLog_Level_t level,
                             uint64_t /*ts*/, va_list args) {
    fprintf(stderr, "[QNN_LOG:%d] ", (int)level);
    vfprintf(stderr, fmt ? fmt : "", args);
    fprintf(stderr, "\n");
}
// 级别设 ERROR 防刷屏（DEBUG 会刷屏淹没 unified_bench 输出）
qnn_->logCreate(qnn_log_callback, QNN_LOG_LEVEL_ERROR, &log_);
```

### 5.3 QNN 错误码速查

| 错误码 | 含义 | 触发场景 |
|-------|------|---------|
| 6003 | `QNN_GRAPH_ERROR_INVALID_NAME` | graphRetrieve 用 null name（V3 未处理） |
| 6004 | `QNN_GRAPH_ERROR_INVALID_TENSOR` | tensor dimensions/name 指针悬空 |
| 6023 | `QNN_GRAPH_ERROR_GRAPH_NOT_FINALIZED` | composeGraphs 后未 graphFinalize |
| 0x3f0 (1008) | HTP 图创建失败 | libQnnHtpPrepare.so 版本不匹配 |

### 5.4 共享内存（零拷贝）

> **⚠️ 已废弃的实现（勿再使用）**：旧代码 `dlopen("libcdsprpc.so")` 取 `rpc_mem_alloc`（老 API，`RPCMEM_HEAP_ID_SYSTEM=9`）经 `QnnMem_register` 注册 DMA-BUF。该实现**从未真正生效**（符号名 `rpc_mem_*` 系统库没有），且直调系统库 `rpcmem_alloc` 会在 SM8550 shell 域崩溃。**正确实现见 5.5（dma-heap 零拷贝）**。

> **⚠️ libcdsprpc 连 DSP 机制（2026-08-05 实测确认）**：
> - 设备 `/vendor/lib64/libcdsprpc.so` 走 **HIDL**（`vendor.qti.hardware.dsp@1.0.so` binder 服务）连 DSP，后台服务进程才有权 open `/dev/adsprpc-smd`，**shell 域用户态也能用**。
> - **不要用 fastrpc 源码自行编译 libcdsprpc.so 替代系统库**：它是直连模式（`open("/dev/adsprpc-smd")`），shell 域被 **SELinux 拒绝**（`errno 13 EACCES`，`avc: denied`），无法连接 DSP。
> - SM8550 设备节点是老的 `adsprpc-smd`（无 `/dev/fastrpc-cdsp`），即使把 `inc/fastrpc_ioctl.h` 节点宏改成 `adsprpc-smd` 也因 SELinux 失败。
> - 系统库缺 `remote_register_buf_attr2`（仅 ORT 1.28 QNN EP 的 shared_memory_allocator 需要；1.22/1.26 不需要）。详见 `docs/ONNX_DEBUG_LOG.md` 第 3 章。

### 5.5 共享内存真正实现：dma-heap 零拷贝（2026-08-05 修复）

**背景**：`qnn_backend.cpp` 的 `AllocateBuffers()` 原本设计走共享内存，但**从未真正生效**——`dlsym` 用的老符号名 `rpc_mem_alloc/rpc_mem_free/rpc_mem_to_fd` 在系统库中不存在（系统库导出的是 `rpcmem_*`），导致一直静默回退 client buffer。

**两次失败的尝试**：
1. **改符号名为 `rpcmem_*` + heap id 25 + `rpcmem_init()`**：`rpcmem_init` 成功（`/dev/dma_heap/system` 打开），但 `rpcmem_alloc` 内部 `remote_register_buf` 直连 DSP 被 SELinux 拦 → **库内段错误**（`si_addr=0x1b`）。
2. **直接 dma-heap 分配 + `QNN_MEM_TYPE_DMA_BUF` 注册**：dma-heap 分配成功（open→ioctl→mmap 全 OK），但 **HTP backend 不接受 `QNN_MEM_TYPE_DMA_BUF`**（它把 union 当 custom descriptor 解析）→ `QnnMem_register` 内段错误（`si_addr=0x1b`）。

**正确实现**（当前 `qnn_backend.cpp`）：
```cpp
/* 1) dma-heap 分配 DMA-BUF（不经过 libcdsprpc） */
int dmafd = open("/dev/dma_heap/system", O_RDONLY | O_CLOEXEC);
struct dma_heap_allocation_data dmabuf = {.len = alloc_size, .fd_flags = O_RDWR | O_CLOEXEC};
ioctl(dmafd, DMA_HEAP_IOCTL_ALLOC, &dmabuf);          // → fd
void *addr = mmap(nullptr, alloc_size, PROT_READ|PROT_WRITE, MAP_SHARED, dmabuf.fd, 0);

/* 2) HTP 要求 CUSTOM descriptor + QnnMemHtp_Descriptor_t（QNN_HTP_MEM_SHARED_BUFFER） */
QnnMemHtp_Descriptor_t htp = {};
htp.type = QNN_HTP_MEM_SHARED_BUFFER;
htp.size = alloc_size;
htp.sharedBufferConfig.fd = dmabuf.fd;
htp.sharedBufferConfig.offset = 0;
Qnn_MemDescriptor_t desc = QNN_MEM_DESCRIPTOR_INIT;
desc.dataType = dt;
desc.memType = QNN_MEM_TYPE_CUSTOM;   // 不是 DMA_BUF！
desc.customInfo = &htp;
qnn_->memRegister(context_, &desc, 1, &h);

/* 3) 每帧 cache 同步（dma_heap/system 是 cached heap） */
dma_buf_sync(fd, DMA_BUF_SYNC_START | DMA_BUF_SYNC_WRITE);  // CPU 写输入前
/* memcpy 输入 */
dma_buf_sync(fd, DMA_BUF_SYNC_END | DMA_BUF_SYNC_WRITE);    // CPU 写输入后
/* graphExecute */
dma_buf_sync(fd, DMA_BUF_SYNC_START | DMA_BUF_SYNC_READ);   // CPU 读输出前
/* 读输出 */
dma_buf_sync(fd, DMA_BUF_SYNC_END | DMA_BUF_SYNC_READ);     // CPU 读输出后
```

**验证结果**（SM8550 / libtest_model.so）：
- HTP：`buffers: 3/3 shared (dma-heap zero-copy)`，avg **1.67ms / 17.2x**，`max_diff=0.00018434`（fp16 正常误差，**输出正确**）
- CPU/GPU backend：`deviceCreate failed, continuing without device` → `memRegister` 失败 → 自动回退 client buffer（**正常降级**）
- 与 client buffer 相比小模型性能持平；**大 I/O 模型（视频帧等）才体现零拷贝收益**

**排查手段**：`strace -f -e trace=openat,ioctl,mmap` 确认 dma-heap 分配成功；`si_addr=0x1b`（偏移 27）定位到 `QnnMem_register` 内部崩溃。

---

### 5.6 性能模式（DCVS_V3 功率投票）+ 零拷贝优化（2026-08-05）

> **勘误（2026-08-06）**：本节当时的 DCVS_V3 配置（`dcvsEnable=1`、电压角 NOM→TURBO、`sleepDisable=1`）**不能持续锁频**——开着 DCVS 动态调压，系统电源管理会在 ~200ms 后把 HTP 降到低频档（实测 6ms↔16ms 阶梯波动，详见 5.9）。等效 ORT burst 的正确配置见 5.9。

**背景**：同一模型（`tfc_tdf_..._8frame_state.onnx`）在 ORT QNN EP HTP 上 avg 9.9ms，而 QNN SDK HTP 只有 40ms 且抖动大。根因：ORT 配了 `enable_htp_fp16_precision=1` + `htp_performance_mode=burst`（锁 TURBO 高频 + 禁睡眠），而 QNN SDK 后端此前**未设置任何性能模式**（HTP 默认低频随负载波动）+ 每轮对每个 tensor 单独 dma_buf_sync + 每轮 memcpy 全部输出。本次在 `qnn_backend.cpp` 落地两项：

**1) HTP 性能模式（等效 ORT burst）**——QNN 2.48 的 DCVS v3 功率投票 API：

```cpp
/* deviceGetInfrastructure 输出的是「指针」（generic QnnDevice_Infrastructure_t* 别名） */
QnnHtpDevice_Infrastructure_t *infra = nullptr;
qnn_->deviceGetInfrastructure(&infra);   // &infra = _QnnDevice_Infrastructure_t**
infra->perfInfra.createPowerConfigId(0, 0, &power_config_id_);   // deviceId=0, coreId=0

QnnHtpPerfInfrastructure_PowerConfig_t cfg = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIG_INIT;
cfg.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3;
cfg.dcvsV3Config.contextId = power_config_id_;
cfg.dcvsV3Config.setDcvsEnable = 1;  cfg.dcvsV3Config.dcvsEnable = 1;
cfg.dcvsV3Config.powerMode = QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE;
cfg.dcvsV3Config.setSleepDisable = 1;  cfg.dcvsV3Config.sleepDisable = 1;  /* 禁睡防抖 */
cfg.dcvsV3Config.setBusParams = 1;
cfg.dcvsV3Config.busVoltageCornerMin/Target/Max = NOM / TURBO / TURBO;
cfg.dcvsV3Config.setCoreParams = 1;
cfg.dcvsV3Config.coreVoltageCornerMin/Target/Max = NOM / TURBO / TURBO;
infra->perfInfra.setPowerConfig(power_config_id_, (const ...*)configs);
/* Cleanup：infra->perfInfra.destroyPowerConfigId(power_config_id_) */
```

要点：`QnnHtpPerfInfrastructure_PerfMode_t`（BURST 等）在 QNN 2.48 已移除，改用 DCVS v3 的**电压角 + powerMode**；TURBO 电压角 + PERFORMANCE_MODE + sleepDisable=1 即等效 burst 锁频（消频率波动 → 平均与峰值同时下降）。

**2) 零拷贝 / 低开销优化**：
- **dma-buf 每方向合并为 1 个 fd**：全部输入放一个 dma-buf、全部输出放一个（每个 tensor 4K 对齐 offset + 各自 `QnnMem_register`，`sharedBufferConfig.offset` 指向其区域）→ 每帧 `dma_buf_sync` 从「每 tensor 一次」降为「每方向一次」（44 输入 + 44 输出 → 2 次 ioctl），注册/分配次数也集中。
- **输出仅最后一轮读取**：中间轮次的输出快照从未被消费，改为只在 `r==repeat-1` 读一次，省掉每帧 ~10MB 输出 memcpy + 相应 sync。

**构建验证**：`cmake --build build/android-arm64`（NDK 29 / HAVE_QNN_SDK_BACKEND=ON）编译链接通过。

**实测建议**：性能模式收益需在 SM8850 上复测 `QNN_SDK_HTP`（预期 avg 从 ~40ms 降至接近 ORT 的 ~10-15ms 区间，峰值显著收窄）。若仍偏慢，下一步是**转换侧**改用 `--float_bitwidth 16`（FP16，HTP 硬件加速）或补 `--input_list` 做 INT8 量化——当前 model.so 是纯 FP32（cpp 里 `dataType=FLOAT_32`、`quantizeParams=UNDEFINED`），FP32 在 HTP 上无硬件加速是慢的主因。

### 5.7 FP16 模型崩溃修复（2026-08-05）

**现象**：用 `qnn-onnx-converter --debug --preserve_io_layout --float_bitwidth 16` 转换的 FP16 model.so 跑 `QNN_SDK_HTP`，init 成功（22 in / 22 out，1641.7ms）但随后 **Segmentation fault**；日志同时出现 `buffers: 0/44 shared (dma-heap zero-copy), rest client buffer`。

**根因**：`AllocateBuffers()` 里的 `tensor_bytes()` 把「非 FP32」一律按 1 字节/元素计算（本是为 INT8 量化设计）：

```cpp
return elems * (dt == QNN_DATATYPE_FLOAT_32 ? 4u : 1u);
```

FP16（`QNN_DATATYPE_FLOAT_16=0x0216`）应为 2 字节/元素 → 缓冲区只分到一半大小 → `QnnMem_register` 的 size 偏小（dma-heap 注册失败，退化为 client buffer）；`PopulateInput` 又按 `n*sizeof(float)` 把 4 字节 memcpy 写进 1 字节/元素的缓冲 → 堆越界 → Segfault。

**修复**（`src/qnn_backend.cpp`）：
1. **手写 IEEE 754 half 转换**（QNN 2.48 SDK 无 `QnnConvert.h`）：`float_to_half()` / `half_to_float()`，round-to-nearest-even，纯位运算跨平台。已与 numpy.float16 对照验证：仅在「精确 tie」边界差 1 ULP（numpy 此处非 RNE，本实现为标准 RNE），对模型精度无实际影响。
2. **`tensor_bytes()` 按 dtype 计算字节数**：`FLOAT_32=4`、`FLOAT_16/BFLOAT_16=2`、其余（量化 8-bit）=1。
3. **`PopulateInput`**：`in_dtypes_[idx]==QNN_DATATYPE_FLOAT_16` 时逐元素 `float→half` 写入 `uint16_t`（不再 4 字节 memcpy）。
4. **`ReadOutput`**：`out_dtypes_[idx]==QNN_DATATYPE_FLOAT_16` 时逐元素 `half→float` 回填。

**验证**：`cmake --build build/android-arm64` 与 `build/win-x64 --config Release` 均通过。修复后 `tensor_bytes` 尺寸正确 → dma-heap 应恢复 shared；若仍报 `0/44 shared`，需看 `dma-heap ... failed` 的 WARN 日志定位 open/alloc/mmap/register 哪一步。精度校验建议与 FP32 结果对比 max_diff（FP16 预期 ~1e-3 量级小误差）。

### 5.8 dma-heap 共享缓冲全部注册失败（`0/140 shared`，2026-08-06 修复）

**现象**：`online_scnet_tfc_tdf`（70 in / 70 out）跑 `QNN_SDK_HTP`，日志刷屏大量 `[QNN_LOG:1] QnnDsp <E>` 错误后 `buffers: 0/140 shared`（全部回退 client buffer）：

```
QnnDsp <E> fd 27 already mapped with mismatched size: registered 36864, got 122880
QnnDsp <E> calculated buffer size: 159748 is more than the actual buffer size: 32768!
QnnDsp <E> Failed to register memHandles ... Current PD has ~38.02 MB in use
QnnDsp <E> qnnMemCreateHandle failed / Failed to register mem with error 0x1f40
```

**根因**（`src/qnn_backend.cpp` AllocateBuffers）：合并 dma-buf 方案里，每个 tensor 各调用一次 `QnnMem_register`（同一 fd、不同 offset），但 `QnnMemHtp_Descriptor_t.size` 填的是**单个 tensor 的大小**。查 `QnnHtpMem.h` 官方注释：`QNN_HTP_MEM_SHARED_BUFFER` 的 `size` 字段必须是**整个共享 buffer 的总大小**（该 fd 覆盖的全部字节），同一 fd 不同 offset 多次注册会返回不同 handle（官方支持的设计）。填 per-tensor size 导致 QnnDsp 侧：
- 同一 fd 每次注册 size 不同 → `fd ... already mapped with mismatched size`；
- 校验 `offset + tensor 需要大小 <= size` 失败 → `calculated buffer size ... is more than the actual buffer size`；
- 全部注册失败 → 方向回退 client buffer（init 变慢 ~10s+，每帧多 ~5MB 拷贝）。

**修复**：注册时 `htp_desc.size = (uint32_t)alloc_size`（整个方向合并 buffer 的 4K 对齐总大小），`offset` 仍为各 tensor 偏移；`tensor_bytes`/`sizes[]` 只用于 layout 与 CPU 侧缓冲，不再直接作为注册 size。`dma_buf_sync` 仍按方向对单个 fd 一次 ioctl，逻辑不变。

**验证**：`cmake --build build/android-arm64` 与 `build/win-x64 --config Release` 均通过，代码全 ASCII。**实测（SM8850，online_scnet_tfc_tdf 70in/70out，2026-08-06）**：
- 修复后 `buffers: 140/140 shared`，`QnnDsp <E>` 刷屏完全消失；
- serialized.bin init 208.7ms（修复前带错误时 ~460ms+，错误风暴时 10-15s）；
- model.so avg 15.17→11.45ms（零拷贝生效）；serialized.bin avg 14.07→14.34ms（repeat=1 单次噪声）；
- `max_diff=0.000000`：共享缓冲下输入/输出读写、offset 布局正确。
**结论性经验**：QNN HTP 共享缓冲注册，`size` 必须填整个 buffer 大小（含 offset 尾部），不是该 tensor 的大小（2026-08-06）。

### 5.9 HTP 持续快档：学 ORT burst = 关闭 DCVS + 锁定最大电压角（2026-08-06）

**现象**：`--repeat 1000` 逐帧日志显示——ONNX_QNN_HTP（ORT）**996/1000 帧 <8ms（全程稳定 6.55ms）**；QNN SDK model.so 仅开头 ~35 帧 <8ms（≈200ms），之后 900+ 帧全在 16ms 慢档；serialized 0 帧快。CPU 后端 45→63ms 为热节流（正常）。

**根因（学 ORT 源码 `qnn_htp_power_config_manager.cc` + `qnn_backend_manager.cc`）**：
- ORT burst 的 DCVS_V3 配置：`dcvsEnable = 0`（**关闭 DCVS 动态调压**）+ bus/core 电压角 Min/Target/Max 全部 = `DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER`（0xA0，最高角）+ `setSleepLatency=1, sleepLatency=40us`（kSleepMinLatency）+ powerMode=PERFORMANCE_MODE。
- ORT 在 `OnRunStart()` 调 `SetHtpPowerConfigs`，但 `AddHtpPerformanceMode` 有去重（模式不变不重复设置）——**所以 ORT 实际只在首次 Run 设置一次**。它能持续快档不是因为反复投票，而是**关闭 DCVS + 锁定最大电压角 = 硬件锁频，系统（perfd/DCVS 仲裁）无法再调低**。
- 我们此前 5.6 的配置开 `dcvsEnable=1` 且电压角只到 TURBO → DCVS 仍可动态调压，系统电源管理约 200ms 后把 HTP 降回低频 → 阶梯波动。

**修复**（`src/qnn_backend.cpp` `ConfigureHtpPerformance()`，与 ORT burst 逐字段对齐）：

```cpp
cfg.dcvsV3Config.setDcvsEnable = 1;
cfg.dcvsV3Config.dcvsEnable = 0; /* kDcvsDisable: hardware-lock the corners */
cfg.dcvsV3Config.powerMode = QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE;
cfg.dcvsV3Config.setSleepLatency = 1;
cfg.dcvsV3Config.sleepLatency = 40; /* kSleepMinLatency (us) */
cfg.dcvsV3Config.setBusParams = 1;
cfg.dcvsV3Config.busVoltageCornerMin/Target/Max = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
cfg.dcvsV3Config.setCoreParams = 1;
cfg.dcvsV3Config.coreVoltageCornerMin/Target/Max = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
```

**验证**：`cmake --build build/android-arm64` 与 `build/win-x64 --config Release` 均通过，ASCII 干净。复测预期：QNN SDK 全程 ~6ms（与 ORT 持平）、无 16ms 慢档、avg 从 15.8ms 降至 ~6.5ms。**结论性经验**：要让 HTP 持续稳定在快档，必须**关闭 DCVS（`dcvsEnable=0`）并锁最高电压角**，一次性投票即持久；开着 DCVS 的“目标档位”投票会被系统覆盖（2026-08-06）。

### 5.10 稀有 ~100ms 尖峰 = HTP 睡眠唤醒，`sleepDisable=1` 对照实验证实（2026-08-06）

**现象**：5.9 锁频后（`sleepLatency=40us` 与 ORT 一致），`--repeat 1000` 的 model.so 段仍偶发 ~100ms 单帧尖峰（run 796=119.9ms、run 806=100.7ms），serialized 段无；0.2% 占比，与基线 11.5ms 相差 ~10 倍。

**对照实验（用户指定）**：把睡眠策略从 `setSleepLatency=1/sleepLatency=40` 改为 `setSleepDisable=1/sleepDisable=1`（强制 HTP 保持唤醒），**保留** DCVS off + MAX 角锁频。仅改 `ConfigureHtpPerformance()` 的 sleep 两行 + 注释 + LOGI 文案。

**实测（SM8850，tfc_tdf_epoch_127，`--repeat 1000` 双变体）**：

| 指标 | 旧版 sleepLatency=40us | 新版 sleepDisable=1 |
|---|---|---|
| model.so avg | 11.35 ms | 11.53 ms |
| model.so max | **119.9 ms（run796）** | **22.1 ms（run497）** |
| serialized avg | 11.66 ms | 11.56 ms |
| serialized max | **100.7 ms（run806）** | **19.9 ms（run268）** |
| fast(<8ms) | 57/1000（so 段开头） | 0/1000 |
| p99 | — | 13.6 / 14.5 ms |

**结论**：
1. **~100ms 尖峰确实来自 HTP 睡眠/唤醒**（`sleepLatency=40us` 允许空闲即睡，唤醒要重新拉起 HTP 管线）。`sleepDisable=1` 后 max 从 119.9/100.7ms → 22.1/19.9ms，长尾完全消除，p99 收紧到 13.6/14.5ms。
2. **avg 几乎不变**（11.53/11.56 vs 11.35/11.66）：尖峰仅 0.2%，对均值无影响，收益全在**尾延迟稳定性**（实时场景关键）。
3. 剩余 10/2000 帧 15~22ms（约 1.5~2x 基线）为**温和抖动**，非睡眠唤醒（无 100ms 级），属 CPU 抢占/调度或 HTP 队列竞争的系统噪声，可接受。
4. 仍达不到 ORT 6.5ms：图本身编译质量（epContext finalization 模式 3）是瓶颈，与睡眠策略无关（见 5.9 结论）。

**最终取舍**：保留 `sleepDisable=1`（强制唤醒）。代价是 HTP 常驻供电、功耗略高，但对基准/实时工具而言尾延迟稳定性优先。**结论性经验**：HTP 偶发百毫秒级单帧尖峰先怀疑睡眠唤醒——`sleepDisable=1` 可消除，且不影响均值与峰值档位（2026-08-06）。

### 5.11 5000 次长跑验证 + 重大发现：model.so 启动瞬态 4.83ms（2026-08-06）

**现象**：保留 `sleepDisable=1`，`--repeat 5000` 双变体长跑。此前"图是瓶颈 ~11.9ms"的结论需要**纠正**——model.so 在**进程启动后前 ~42 帧稳定跑在 4.83ms（比 ORT 6.5ms 还快！）**，之后在 4.8ms↔10~11ms 之间抖动约 24 帧，第 66 帧起永久沉降到 11.5ms。

**实测数据（SM8850，tfc_tdf_epoch_127，repeat=5000）**：

| 指标 | model.so | serialized.bin |
|---|---|---|
| avg | 11.560 ms | 11.670 ms |
| min | **4.819 ms** | 9.169 ms |
| max | 18.889 ms | 14.876 ms |
| p50 / p90 / p99 | 11.657 / 12.697 / 13.709 | 11.673 / 12.711 / 13.660 |
| fast(<8ms) | **55/5000，全部在 run0-65** | 0/5000 |

**model.so 前 70 帧逐帧（ms）**：
```
r0-41: 稳定 ~4.83（4.825~4.943，零抖动）
r42-46: 10.0 / 10.1 / 10.3 / 9.5 / 11.1
r47-49: 4.89 / 4.85 / 5.72
r50: 9.29
r51-57: 6.5 / 4.83 / 4.82 / 4.88 / 4.83 / 4.84 / 4.89
r58-62: 9.9 / 9.9 / 9.8 / 9.9 / 11.5
r63-65: 4.85 / 4.83 / 5.49
r66+: 8.3 → 8.9 → 9.6 → 10.6 → 永久 ~11.5
```

**关键证据**：
1. **投票在 run0 之前已下发**（日志 line 71 `HTP perf configured` < line 87 `run 0`），但启动瞬态依然存在 → 4.83ms 启动态**高于** DCVS-off MAX 角（0xA0）稳态，或投票物理落地有 ~200ms 延迟。
2. serialized 段紧跟 model.so 段（同进程，HTP 已沉降），**0 帧快** → 快档是"冷启动/启动时钟"行为，**与 ORT 前置无关**（5.9 把快帧归因 ORT 余温是错的，run 11 standalone 0 快帧是因为设备已热/进程启动间隔长，瞬态已过）。
3. 抖动模式（4.8↔10~11 来回 3 次、每次快窗变短）像**时钟/仲裁协商**：固件逐步把频率降到锁定角，非单调热节流。

**结论修正**：
- `sleepDisable=1` 在 5000 次下依然**零 100ms 尖峰**（max 18.9/14.9ms，p99 13.7ms）——尾延迟彻底解决，保留。
- **11.5ms 稳态是时钟/固件钳制，不是图计算上限**。model.so 物理可跑 4.83ms（< ORT 6.5ms）→ 追 ORT 甚至反超**有头空间**。
- 下一步实验方向（追 4.83ms 稳态）：
  1. **稳态周期重发投票/每帧 kick**：在 `RunBenchmark` 里周期性 `setPowerConfig`，看能否把 4.83ms 拉回来（验证是否是投票落地延迟）；
  2. `qnn-net-run --perf_profile burst --shared_buffer` 跑 5000 次，看 Qualcomm 官方工具能否保持快档（能 → 图没问题，是我们投票没到位；不能 → 固件行为，仅记录）；
  3. 读 HTP 实际时钟（debugfs/HTP profiling）确认快/慢态是否纯频率差异。

### 5.12 生成更好的 .bin（离线准备，O=3 / P 点 / soc_id，2026-08-06）

**目标**：把 `model.so` 离线编译成上下文二进制（serialized context binary），加载/推理快于运行时在线 prepare，并可用 O=3 优化 + P 点 + 指定 SoC 进一步加速。依据 HTP 官方文档《HTP Optimization (Auto)》与《QNN HTP Backend Extensions》。

**核心结论（文档要点）**：
- 优化级别 `O`：有效值 **2**（默认）和 **3**。O=3 通常更优，且**指定匹配目标的 `soc_id` 时会启用额外算法进一步提速**（但可能在某些图上更差、二进制更大、加载更慢，需实测）。`O=3` 对应 `QNN_HTP_GRAPH_OPTIMIZATION_TYPE_FINALIZE_OPTIMIZATION_FLAG = 3`。
- **P 点**（仅 O=3 生效，`finalize_config: {"P": n}`）：编译器内部配置空间的不同点，调整**延迟 vs DRAM 带宽**等权衡。合法值 `0~23`，**排除 7,9,10,11,12,14,18**。不同网络最佳 P 不同，需逐一扫描实测；同一 P 下输出与无 P 位精确一致。**一次只能指定一个 P**。
- 目标配置：SM8850 → `soc_id = 87`（`QNN_SOC_MODEL_SM8850=87`，见 `include/QNN/QnnTypes.h`）、`dsp_arch = "v81"`（设备加载 libQnnHtpV81* 系列）。
- 其它图级选项：`vtcm_mb`（默认 4，可设更大；0 = 设备最大需配合 soc）、`hvx_threads`（默认 4）、`num_cores`（多核编译，SM8850 双 NSP，潜在最大收益项）、`dlbc`（带宽压缩）、`advanced_activation_fusion`（FP 模型默认开）、`monolithic_lstm`。

**标准流程（x86 主机离线准备，文档推荐）**：
```bash
# 1) ONNX → 主机可加载的 model.so（需要 Python 3.10 或 3.12，见下方踩坑）
python <SDK>/bin/x86_64-windows-msvc/qnn-onnx-converter \
  --input_model tfc_tdf_epoch_127_val_loss_0_05071_causal_8frame_state.onnx \
  --output_model tfc_tdf_host.so --target x86_64-windows-msvc

# 2) 编写 HTP 后端配置 tools/htp_context_sm8850.json（O=3 + soc_id=87 + v81，可加 finalize_config P 点）
# 3) 生成 context binary（离线准备）
<SDK>/bin/x86_64-windows-msvc/qnn-context-binary-generator.exe \
  --backend <SDK>/lib/x86_64-windows-msvc/QnnHtp.dll \
  --model tfc_tdf_host.so \
  --binary_file tfc_tdf_o3.serialized.bin \
  --profiling_level basic \
  --config_file tools/htp_context_sm8850.json

# 4) 性能估算（不跑真机，先看 Simulated cycles + Bandwidth stats）
<SDK>/bin/x86_64-windows-msvc/qnn-profile-viewer.exe \
  --input_log output/qnn-profiling-data.log \
  --reader <SDK>/lib/x86_64-windows-msvc/QnnHtpProfilingReader.dll

# 5) P 点扫描：改 htp_context_sm8850.json 里 finalize_config {"P": n}，重复 3-4，对比 Simulated cycles
```

**配置文件（已入库 tools/config/，全部实测可用）**：
- `tools/config/backend_ext_sm8850.json`：**设备端生成器实际要传的配置**（`backend_extensions` 包装：`shared_library_path=./libQnnHtpNetRunExtensions.so` + `config_file_path=./htp_config_sm8850.json`）。
- `tools/config/htp_config_sm8850.json`：真正的图/设备配置（`graphs[].graph_names` 必须用**图名 = 模型名**、`O:3`、`vtcm_mb`、`hvx_threads`、`advanced_activation_fusion`；`devices[]` 的 `soc_id:87`、`dsp_arch:"v81"`）。
- `tools/config/htp_context_sm8850.json`：同内容的平铺版（x86 主机生成器若支持平铺可直接传；设备端会报 Unknown Key，须用包装版）。

**设备端生成（实测可用，2026-08-06）**：
```bash
# 设备 /data/local/tmp/qnn-2.48.40 下执行（注意命令格式！）
./qnn-context-binary-generator \
  --model ../bench_test/libtfc_tdf_epoch_127_val_loss_0_05071_causal_8frame_state.so \
  --backend ./libQnnHtp.so \
  --binary_file tfc_tdf_o3_opt.serialized \
  --config_file ./backend_ext_sm8850.json
# 产物在 ./output/tfc_tdf_o3_opt.serialized.bin
```
**实测结果（O=3 + soc_id=87 + vtcm 8MB，`--repeat 1000`）**：基线（默认 O=2）p50=12.26ms；O=3 p50=12.46ms——**无明显提升**（avg 被低电量尖峰污染到 20-22ms，p50 对比有效）。与文档"O=3 不一定更优、需实测"一致；TDF 类逐元素/小卷积图对 finalize 优化不敏感，真正的性能上限是时钟（见 5.11 的 4.83ms 启动态），图优化空间有限。下一步可试 **P 点扫描** 与 **`num_cores: 2`（多核编译）**。

**踩坑记录（2026-08-06 实测）**：
1. **设备端生成器命令格式**：`--binary_file` 用**裸文件名**（不要 `output/` 前缀、不要手动加 `.bin`）；`--model` 用相对路径更稳。之前"静默失败"是命令格式问题（绝对路径 + `output/xxx.bin`），改用用户命令格式后正常。
2. **设备端生成器只认 `backend_extensions` 包装配置**：直接传平铺 graphs/devices 会全部报 `Unknown Key = ...` 且优化不生效（仍按默认编译）；必须传 `backend_ext_sm8850.json` 包装，让 `libQnnHtpNetRunExtensions.so` 解析图配置。
3. **图名 = 模型名**（`tfc_tdf_epoch_127_val_loss_0_05071_causal_8frame_state`），不是 ONNX 图名 `main_graph`；写错报 `getQnnGraphConfigFromInfo() unable to find graphName:xxx`。
4. **QNN 2.48 的 `qnn-net-run` 已移除 `--save_context_binary`**（`Invalid Argument Consumption`），只有 `--retrieve_context`；`qairt-net-run` 在设备上 `Permission denied`。→ 落盘靠生成器。
5. **Windows 主机生成器无法加载 aarch64-android 的 model.so**（`dlerror(): load library failed`，架构不匹配），必须先转出 x86 主机版 model.so（需 Python 3.10/3.12，见 6）。
6. **SDK 转换器需要 Python 3.10 或 3.12**：`lib/python/qti/aisw/converters/common/windows-x86_64/libPyIrGraph{310,312}.pyd`；本机仅 3.11/3.7，`qnn-onnx-converter` 报 `ImportError: DLL load failed ... libPyIrGraph312`。→ 走设备端生成可完全绕过主机转换。
7. **低电量（12% 放电）会周期性折叠 HTP 电源**：每 ~120ms 出现一次 ~100-130ms 尖峰（avg 从 11.5 → 20-22ms），且 `sleepDisable=1` 也压不住（低电量下固件强制电源管理优先）。**基准测试必须在插电状态下进行**。

**结论性经验**：设备端 `qnn-context-binary-generator` + `backend_extensions` 包装配置即可完成 O=3/soc_id/P 点优化编译，无需 x86 主机与 Python。本模型 O=3 无收益（TDF 图对 finalize 优化不敏感）；进一步可试 P 点与 `num_cores: 2`；所有基准务必插电测试（2026-08-06）。

---

## 6. 验证命令速查

```bash
# context binary → HTP（离线编译产物）
adb shell "cd /data/local/tmp/bench_test && LD_LIBRARY_PATH=.:./qnn ADSP_LIBRARY_PATH=./qnn \
  ./unified_bench test_model.serialized.bin --backend QNN_SDK_HTP --repeat 5 --warmup 1"

# model.so → CPU/GPU/HTP（现场编译）
adb shell "cd /data/local/tmp/bench_test && LD_LIBRARY_PATH=.:./qnn ADSP_LIBRARY_PATH=./qnn \
  ./unified_bench libtest_model.so --backend QNN_SDK_CPU,QNN_SDK_GPU,QNN_SDK_HTP --repeat 5 --warmup 1"

# 官方对照（证明 model.so 支持 HTP）
./qnn-net-run --backend libQnnHtp.so --model libtest_model.so \
  --input_list input_list_test_model_float32.txt --perf_profile burst --shared_buffer
```

### 5.13 TDF 模型优化机制清单（2026-08-06，基于 ONNX 结构分析 + omen Linux 主机）

**模型结构事实**（tfc_tdf_epoch_127，从 .onnx.text 解析）：735 节点 / 14 种算子；常量 361、Slice 63、Conv 62、Reshape 46、Relu 43、Concat 43、Transpose 37、ConstantOfShape/Cast/Pad 各 21、Add 7、Mul 4、ConvTranspose 3、BatchNormalization 3；141 个 initializer 全 FP32，权重 ~20MB；22 in / 22 out FP32（state 类 21 组 + 音频 input）；7 组 sub-band 重复展开结构（H.0/H.1/H.2 + tdf + ds/us）。

**优化机制分层（按预期收益排序）**：
1. **FP16 转换（最大机制）**：`qnn-onnx-converter --preserve_io_layout --input_network x.onnx --float_bitwidth 16`。权重+激活+IO 全 FP16 → 内存带宽减半 + HTP FP16 硬件加速。**ORT 6.5ms 的核心秘密就是 enable_htp_fp16_precision**；当前 QNN SDK FP32 稳态 11.5ms，FP16 后有望对标/反超 ORT（5.11 已证 FP32 冷启动 4.83ms，FP16 在锁频下应更低）。
2. **量化 A8W8 / A16W8**：HTP 原生算力为 INT8；音频模型建议先试 **A16W8**（`--act_bitwidth 16 --weights_bitwidth 8`）保精度、再试 A8W8（`--act_bitwidth 8 --weights_bitwidth 8`）。已有校准数据：`/data/local/tmp/mss/calibration_data{,_2000}/`、`/data/local/tmp/qnn/tfc_tdf_epoch_127_cal_loss_5071.json`；配合 `--input_list` + `--use_native_input_files`。量化器可选 `--param_quantizer`（percentile/mse 等）调精度。**【已实测 2026-08-06：A16W8 图执行崩 DSP，方向关闭，见 5.14B】**
3. **ONNX 预化简（转换前）**：本图冗余极多（361 个 Constant 常量折叠、7 组 sub-band 重复结构、大量 Reshape/Transpose/Slice）。先跑 `onnx-simplifier`/`onnxoptimizer`（fuse_consecutive_transposes、eliminate_nop_*、常量折叠）再转换，节点更少、转换质量更高、bin 更小。**【已实测 2026-08-06：735→351 节点但慢 16%，HTP O3 已自行优化，方向关闭，见 5.14A】**
4. **编译期**：O=3 + P 点扫描 + `num_cores:2`（双 NSP 多核，3 路 sub-band 可并行）+ `vtcm_mb` 调大（FP16 后 TCM 占用减半）+ `dlbc`（带宽压缩，FP16 后仍带宽受限时）。
5. **IO 优化**：`--preserve_io_layout`（已有）；若 `--float_bitwidth 16` 未把 IO 转 FP16 则显式 `--io_bitwidth 16`（22 in/22 out 的 IO 带宽占比高）。
6. **运行时**：已做 dma-heap 零拷贝 + DCVS off/MAX corner/sleepDisable；多核需按 core 逐个 createPowerConfigId 投票；可试周期重发投票拉回 4.83ms 稳态。
7. **验证纪律**：FP16/量化后必须对比 max_diff（FP16 预期 ~1e-3 量级）；量化还要做主观/感知听感验证；**基准必须插电**（低电量周期尖峰见 5.12）。

**注意**：`--preserve_io_layout` 是一个参数（`--preserve_io_layout`），不是 `--preserve_io layout`。

### 5.14 实测：onnx-simplifier 无收益 + A16W8 量化崩 DSP（2026-08-06，SM8850）

**A. onnx-simplifier 预化简（方向 3 实测 = 无效，甚至更慢）**
- 操作：omen 上 `python -m onnxsim tfc_....onnx simplified/tfc_simplified.onnx`。
- 结果：735 → 351 节点（Constant 502→147、Reshape 46→2、Cast/Pad/ConstantOfShape 21→0），bin 大小不变（~20MB，权重为主）。
- FP16 转换 + host 生成 `tfc_simplified_o3.bin`（11,056,704B），真机同会话 A/B：
  - 基线 `tfc_tdf_o3_opt`：**avg=13.502 ms**
  - 简化 `tfc_simplified_o3`：**avg=15.658 ms（慢 ~16%）**
- DDR 统计（host generator 输出）几乎不变：read 47.6MB vs 47MB、spill/fill 29/32MB 原样。
- **结论**：HTP 编译器 O3 在离线编译时已自行完成图融合/折叠，ONNX 层的冗余（Constant/Reshape/Pad 等）不转化为 HTP 层冗余；强行化简反而破坏 HTP 已有融合模式。**此方向关闭，后续不要再对 TDF 跑 onnx-simplifier**。

**B. A16W8 量化（方向 2 实测 = 图本身崩 DSP，方向关闭）**
- 操作：`qnn-onnx-converter --act_bitwidth 16 --weights_bitwidth 8 --input_list ...` → model.so → host/device 生成 context binary（6,074,240B）。
- 图结构：IO 全 `QNN_DATATYPE_UFIXED_POINT_16`（scale≈1.526e-5≈2^-16, off=0）；内部 224 个 QU16 + 65 个 QU8 + 65 个 QS32（bias）。
- 三次运行三次崩（SM8850 + QNN 2.48.40）：
  1. context binary + 旧后端（只处理 8-bit）→ 推理中 **DSP 崩溃 → 手机 crashDump**；
  2. context binary + 新后端 + 客户端缓冲 → **App 段错误**（CPU 侧，见 C）；
  3. model.so + 新后端（CPU 侧缓冲已正确）→ 再次 **DSP 崩溃 → 手机 crashDump**。
- **结论**：即使 CPU 侧输入/缓冲完全正确，A16W8 量化图在 SM8850 HTP（QNN 2.48.40）上执行即崩 DSP，属于该 SDK/该模型量化路径缺陷。**A16W8 方向关闭**（可后续升 QNN 版本再试；A8W8 未测，但音频模型 8-bit 激活精度风险高，优先级低）。

**C. 根因：QNN 2.48 无法从 context binary 拿到 IO 量化编码 + 旧代码 float memcpy 越界**
- 现象：A16W8 跑 context binary 时 `PopulateInput` 落到 `memcpy(dst,data,n*4)`，把 4 字节/elem 拷进 2 字节/elem 的缓冲 → 2 倍越界 → `__memmove_aarch64_nt` SEGV_ACCERR（scudo 保护区）。
- 根因链：
  1. `QnnSystemContext` 的 binary info 张量 **不携带 quantizeParams**（`encodingDefinition=QNN_DEFINITION_UNDEFINED`）→ `is_quant=false`；
  2. 旧代码按 `is_quant` 标志分派 → 定点 dtype 落入 float memcpy 分支；
  3. QNN 2.48 已移除 `QnnGraph_getTensors`，`graphGetProperty` 仅支持 CUSTOM，**无标准 API 可查图 IO 量化编码**；model.so 的 GraphInfo 张量实测同样 `quant=0`。
- 修复（保留）：
  - `PopulateInput`/`ReadOutput` 改为 **按 dtype 优先分派**（QU16/QS16→int16 量化路径、QU8/QS8→uint8、F16→half、其余→memcpy），定点 dtype 永不落入 float memcpy；
  - 新增调试开关 `QNN_NO_DMA_HEAP=1` 强制客户端缓冲（隔离 DMA/量化问题）。
- 遗留：若未来要正经跑量化模型，需解决量化编码来源（model.so GraphInfo 的 quantizeParams 丢失问题待查），否则 scale/offset 恒为 1/0、数值错误。

**D. 其他**：手机 crashDump 模式恢复后需重新 `su 0 setenforce 0` + 确认 `ADSP_LIBRARY_PATH=/data/local/tmp/qnn-2.48.40`；设备持续有 `qsap_mpamsvc` 崩溃（缺 seccomp policy，userdebug ROM 问题，与本次无关）。

### 5.15 运行时周期重发功率投票：无法拉回 4.83ms 冷启动态（2026-08-06，SM8850）

- **动机**：model.so 冷启动前 ~42 帧稳定 4.83ms（快于 ORT 6.5ms），之后衰减到 ~11.5ms；5.11 假设是固件/仲裁器丢弃我们的 DCVS 投票所致。本次尝试在 RunBenchmark 中**周期重发 setPowerConfig** 看能否维持 4.83ms。
- **实现**（**已回退 2026-08-06**）：`ConfigureHtpPerformance()` 保存 `setPowerConfig` 函数指针 + DCVS_V3 配置副本，新增 `RefreshPowerVote()` 在 `execute_once` 每 N 帧调用，周期由 `QNN_REVOTE_FRAMES=N` 控制。实验后**全部回退**（`src/qnn_backend.cpp` 恢复原状），不保留任何相关代码。
- **实测**（model.so `libtfc_fp16_aarch64.so`，SM8850 充电 37%，1000 次，max_diff 全 0）：
  - 无重发基线：**avg=11.617 ms**
  - 每帧重发（N=1）：**avg=12.840 ms**（+1.2ms——每帧 fastrpc 调用开销）
  - 每 100 帧重发：**avg=11.999 ms**（≈基线，**无收益**）
- **per-run 日志（无重发）**：4.83ms 态**反复出现**（run 0-15、20、24-27 均 ~4.8ms），之后在 4.8↔11.5ms 间振荡并最终稳定 ~11.5-12.5ms——是**双模态时钟仲裁**，非单调衰减。
- **结论**：**周期重发无法拉回 4.83ms**。4.83ms 是固件/仲裁器临时授予的高频窗口，不受我方 DCVS 投票控制（投票本就 MAX corner 硬件锁定）；每帧重发因 fastrpc 开销反而更差。**方向关闭**；相关代码已全部回退（见上），如需复现实验可参考本节数据与思路。
- 说明：4.83ms 快速态在本次会话（充电、37%）中比 5.11 记录（前 42 帧一次性）出现得更频繁，可能与设备温度/负载状态有关。
