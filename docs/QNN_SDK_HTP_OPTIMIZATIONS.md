# QNN SDK HTP 优化选项汇总

> 适用范围：`unified_bench` 的 `QNN_SDK_HTP` 后端（原生 QNN SDK，`src/qnn_backend.cpp`），
> 并附 ORT QNN EP（`ONNX_QNN_HTP`，`src/onnx_backend.cpp`）对照。
> 设备：SM8850（soc_id=87，Hexagon v81），QNN SDK 2.48.40.260702。
> 更新日期：2026-08-12。排查细节见 `docs/QNN_SDK_DEBUG_LOG.md` 5.24~5.29。

---

## 1. 总览：两条执行路径

| 路径 | 模型格式 | 优化施加点 |
|---|---|---|
| **model.so 运行时路径** | QNN model（`lib*.so`） | 图级配置在**运行时 `composeGraphs`** 注入（`GraphConfigInfo_t` + `QnnHtpGraph_CustomConfig_t`）；context 级配置在 `contextCreate` |
| **context binary 路径** | 离线编译的 `.serialized.bin` | 图级优化（O3/soc/hvx/P 点等）在**离线 `qnn-context-binary-generator` 编译期**固化，运行时不可改；运行时只施加 context 级 + 性能投票 |

---

## 2. 图级配置（model.so 路径，运行时 composeGraphs）

经 `QnnModel_composeGraphs` 的 `GraphConfigInfo_t`（`graphConfigs` = `QnnGraph_Config_t[]{ QNN_GRAPH_CONFIG_OPTION_CUSTOM → QnnHtpGraph_CustomConfig_t }`）注入。
**按架构门控**：完整调优集（O3+vtcm+hvx+AAF）仅当检测到 `htp_arch >= v73`（HTP v2+，SM8550/SM8650/SM8750/SM8850）或任意 `QNN_HTP_*` 环境变量被显式设置时应用；旧架构（如 SM8450=v69）只投 VTCM 一项（见 5.29）。

| # | 配置项 | 默认 | 说明 | env 覆盖 |
|---|---|---|---|---|
| 1 | `QNN_HTP_GRAPH_CONFIG_OPTION_OPTIMIZATION` = `FINALIZE_OPTIMIZATION_FLAG` | **O3** | 图终优化级别（0~3）。O3 最强：算子融合、张量布局/生命周期重排、最大化 VTCM 利用。SM8850 实测 O3 是关键收益来源之一 | `QNN_HTP_O` |
| 2 | `QNN_HTP_GRAPH_CONFIG_OPTION_VTCM_SIZE`（MB） | **0 = `QNN_HTP_GRAPH_CONFIG_OPTION_MAX`**（SoC 最大） | HTP Vector TCM 保留量。**0 表示自动取该 SoC 上限**（官方跨 SoC 机制）：SM8850→8MB（默认 4，16 会 `composeGraphs` rc=14 失败）；SM8450/v69→其更小上限。不硬编码（见 5.29） | `QNN_HTP_VTCM_MB` |
| 3 | `QNN_HTP_GRAPH_CONFIG_OPTION_NUM_HVX_THREADS` | **8** | 每图 HVX 线程数。SM8850 实测 hvx8 最优（hvx4=8.55ms、hvx8=7.2ms）；hvx 运行时确认生效 | `QNN_HTP_HVX_THREADS` |
| 4 | `QNN_HTP_GRAPH_CONFIG_OPTION_ADVANCED_ACTIVATION_FUSION` | **true** | 高级激活融合：把更多激活融合进前驱算子内部执行，减少中间激活的 VTCM/DDR 往返。v81 实测 aaf=1 安全无副作用 | `QNN_HTP_AAF` |
| — | `QNN_HTP_GRAPH_CONFIG_OPTION_NUM_CORES` | **不设置** | 运行时多核**无收益**：unsigned PD 只暴露 1 个 NSP（5.24/5.25 实测）。离线 `num_cores:2` 才有效 | — |

> 注：GPU/CPU 后端不施加任何 HTP 图配置。

---

## 3. Context 级配置（两条路径都施加）

| 配置项 | 默认 | 说明 | env 覆盖 |
|---|---|---|---|
| `QnnContext_Config_t{ QNN_CONTEXT_CONFIG_OPTION_PRIORITY }` | **`QNN_PRIORITY_HIGH`(200)** | Context 调度优先级（LOW=0/NORMAL=100/NORMAL_HIGH=150/HIGH=200）。单会话基准无调度竞争，主要意义是与 ORT 对齐；被后端拒绝时**自动重试一次无配置**（防御，保护 context binary 路径） | `QNN_HTP_CONTEXT_PRIORITY`（low/normal/normal_high/high） |

两条路径：model.so 走 `contextCreate`、context binary 走 `contextCreateFromBinary`，均传入上述配置。

---

## 4. 性能基础设施（`QnnDevice_getInfrastructure` + `setPowerConfig`，两条路径都施加）

一次性投票（`QnnHtpPerfInfrastructure_PowerConfig_t configs[]`，可同时投多条），等价 ORT 的 `kHtpBurst` 模式。

| 配置项 | 值 | 说明 |
|---|---|---|
| `QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3` | `dcvsEnable=0`（关 DCVS）、`powerMode=PERFORMANCE_MODE`、`setSleepDisable=1/sleepDisable=1`、bus/core 电压角 `Min/Target/Max = MAX_VOLTAGE_CORNER` | 关掉 DCVS 动态调压、把 HTP 锁在最高电压角、强制不睡眠——避免 perfd/DCVS arbiter 在 ~100-300ms 后把 HTP 降到低档（此前出现的 6ms↔16ms 阶梯抖动根因） |
| `QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_CONTROL_LATENCY` | **100 µs** | RPC 控制延迟。独立于 DCVS 的 power config 项（option=2，字段 `rpcControlLatencyConfig`），与 DCVS 同一次 `setPowerConfig` 一并投票。非关键项，主要对齐 ORT | `QNN_HTP_RPC_LATENCY` |

---

## 5. 内存 / IO 优化

| 项 | 说明 |
|---|---|
| **DMA-BUF 共享缓冲（零拷贝 I/O）** | Android 上经 `/dev/dma_heap/system` 分配 DMA-BUF，`QnnMem_register` + `QNN_HTP_MEM_SHARED_BUFFER` 注册，QNN 直接把 fd 交给 DSP，避免 HTP 与 CPU 间的整帧内存拷贝；非 Android 回退普通 client buffer |
| **FP16 快速转换** | `__aarch64__` 下用 NEON `vcvt_f16_f32` / `vcvt_f32_f16` 加速 CPU 端 float→half 输入转换（FP16 大 IO 图的隐藏瓶颈在 CPU 逐帧转换，见 5.18 实测） |

---

## 6. ORT QNN EP 对照（`ONNX_QNN_HTP`，供对齐参考）

| ORT provider option | 值 | 说明 |
|---|---|---|
| `backend_type` | `htp` | HTP NPU 后端 |
| `soc_model` / `htp_arch` | 自动探测（`qnn_soc_detect`） | 如 SM8850→`87`/`81`（裸数字，ORT 不接受 `v81`） |
| `htp_performance_mode` | `burst` | 对应第 4 节的 burst 电压投票（含 `rpc_polling_time=9999`） |
| `htp_graph_finalization_optimization_mode` | `3` | 同 O3 |
| `enable_htp_fp16_precision` | `1` | HTP 内部 FP16 |
| `enable_htp_shared_memory_allocator` | `0` | **保持 0**：=1 需要 rpcmem attr2（仅系统 libcdsprpc 有），会导致 epContext 缓存命中失败 |
| `vtcm_mb` | v81+→`8`，其余→`0`(=SoC max) | 按架构自适应（见 5.29）；v81 保持 8 以复用已生成的 epContext、避免重新生成 |
| `rpc_control_latency` | `100` | 同第 4 节 |
| `qnn_context_priority` | `HIGH` | 同第 3 节 |
| 会话配置 `session.disable_cpu_ep_fallback` | `1` | 不允许不支持节点静默回退 CPU（暴露 CPU 混跑）；原生 QNN SDK 无此机制（图加载失败即报错） |

---

## 7. 一页速查表（全部运行时可控项）

| 项 | 生效路径 | 默认 | 环境变量 |
|---|---|---|---|
| 优化级别 O3 | model.so 图配置 | 3 | `QNN_HTP_O` |
| VTCM 大小 | model.so 图配置 | 0(=SoC max) | `QNN_HTP_VTCM_MB` |
| HVX 线程数 | model.so 图配置 | 8 | `QNN_HTP_HVX_THREADS` |
| 高级激活融合 AAF | model.so 图配置 | 1 | `QNN_HTP_AAF` |
| Context 优先级 | 两条路径 | HIGH(200) | `QNN_HTP_CONTEXT_PRIORITY` |
| RPC 控制延迟 | 两条路径 | 100 µs | `QNN_HTP_RPC_LATENCY` |
| DCVS burst 电压锁定 | 两条路径 | 开 | — |
| DMA-BUF 零拷贝 | 两条路径 | 开（Android） | — |
| FP16 快速转换 | 两条路径 | 开（aarch64） | — |

---

## 8. 关键结论（避免重复排查）

1. **运行时多核无收益**：unsigned PD 只暴露 1 NSP，`num_cores:2` 运行时无效；离线编译 `num_cores:2` 才有效（5.24/5.25）。
2. **vtcm 别硬编码**：用 `0`(=SoC max) 让 QNN 自动适配，SM8850→8MB、SM8450(v69)→其上限（5.29）。
3. **DCVS 必须关**：否则系统会在数百毫秒后把 HTP 降频，出现 6ms↔16ms 阶梯抖动（5.9 实测）。
4. **`enable_htp_shared_memory_allocator` 保持 0**：=1 会破坏 ORT epContext 缓存复用。
5. **context binary 路径**：图级优化是编译期固化的，运行时只叠加 context 优先级 + 性能投票；生产/基准推荐 serialized.bin（init 223ms，远快于 model.so 的 ~28s compose）。
