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
