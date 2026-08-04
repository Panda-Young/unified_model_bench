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
