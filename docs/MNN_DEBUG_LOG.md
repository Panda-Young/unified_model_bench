# MNN Backend 修复记录

> **日期**: 2026-08-20（§1）；**2026-08-29 追加 §2 / §3**（静默降级与计时错位，Windows 桌面实测）
> **范围**: `src/mnn_backend.cpp`
> **环境**: Android（MNN 3.6.0，vendored 于 `deps/mnn/`）；§2/§3 为 Windows x64 + Intel Iris Xe

---

## 2. 静默降级：6 个后端挂着 GPU 名字跑 CPU（2026-08-29 修复）

### 2.1 现象

Windows 桌面跑 `tfc_tdf_...`（22 in / 22 out）全后端，9 个 MNN 后端的
`max_output_diff` 出现**逐位相同**的分组（不是近似，是完全相等）：

```
0.00002384 : MNN_OpenCL, MNN_OpenCL_BF16              <- 真实 GPU
0.00002640 : MNN_CPU, MNN_VULKAN, MNN_VULKAN_FP16,
             MNN_VULKAN_BF16, MNN_OPENGL, MNN_NN      <- 全部与 CPU bit-identical
0.16594648 : MNN_OpenCL_FP16                          <- 真实 GPU（FP16 精度损失）
```

Vulkan FP16、Vulkan BF16、OpenGL、NN 四个后端不可能与 CPU 产生 bit-identical 结果，
**唯一解释是它们都没在各自设备上执行**。三条独立证据互相印证：

| 证据 | 降级的 6 个 | 真实的 3 个 OpenCL |
|---|---|---|
| `load_lib` | 88~110 ms（无 GPU 上下文建立） | **37164~44087 ms**（真在编译 kernel） |
| `peak_mem_mb` | 412 MB（≈ MNN_CPU 的 412.586） | **498~611 MB**（GPU 额外内存） |
| `avg_run_ms` | 417~480 ms（≈ CPU 的 401） | 修复后 197~269 ms |

### 2.2 根因

`deps/mnn/include/MNN/Interpreter.hpp:63`：

```cpp
/** backup backend used to create execution when desinated backend do NOT support any op */
MNNForwardType backupType = MNN_FORWARD_CPU;
```

MNN 在请求的后端不可用时**自动退到 `backupType`**，`createSession()` 依然成功返回。
而 `mnn_backend.cpp` 只设置了 `sched_.type` 和 `backendConfig`，**从未碰 `backupType`**——
降级通道一直敞开。

这违反项目核心信条。对比同一项目的正确做法：

| 后端 | 修复前行为 |
|---|---|
| `NCNN_CPU_BF16` | ✅ 报错 "NCNN: CPU lacks BF16 support"，CSV 打 `-` |
| `ONNX_DML_NPU` / `OpenVINO_NPU` | ✅ 报错，CSV 打 `-` |
| `MNN_VULKAN/OPENGL/NN` 等 6 个 | ❌ **报 CPU 的数字，挂 GPU 的名字** |

危害是具体的：这 6 行看起来"成功且正常"，会被直接用于跨后端对比。若据此得出
"MNN_OPENGL 比 MNN_OpenCL 慢 100 倍"就完全错了——实际 OpenCL 快得多，那 6 个根本没跑起来。

### 2.3 修复

**(a) 堵住降级通道**（`Initialize`）：

```cpp
sched_.backupType = sched_.type;   // 请求什么就必须用什么，不可用即失败
```

`createSession()` 因此失败 → 走既有 `return false` 路径 → CSV 打 `-` + notes 写明原因。

**(b) 加"实际执行后端"验证**（`VerifyActualForwardType()`）：

仅堵 `backupType` 只防这一种路径。补充检查输出张量是否真在设备上。

判据（**MNN 3.6.0 实测标定，勿凭直觉改**）：输出张量 `halide_buffer_t::device` 字段：

| session | `device` 值 | `host` 指针 |
|---|---|---|
| CPU | `1`（哨兵） | 有效 |
| OpenCL | 真实设备地址（>1，如 `2577890888992`） | `nullptr` |

故判据为 **`device > 1`**。两个错误尝试（均已实测否决，勿重犯）：
- `device != 0` —— CPU 张量带哨兵 `1`，会全部误判为设备
- 比较 `getBackend()` 返回的 Backend **指针相等性** —— MNN 为**每个 session 创建独立
  Backend 实例**，CPU 参考 session 与另一 CPU session 的指针也不同

> `Backend::type()` 不可用：`deps/mnn/include/MNN/` 只 vendored 了 5 个头文件
> （AutoTime/ErrorCode/ImageProcess/Interpreter/Tensor），**没有 `Backend.hpp`**。

### 2.4 结果

修复后（repeat=20，同一模型）：

| Backend | 修复前 | 修复后 |
|---|---|---|
| MNN_VULKAN / _FP16 / _BF16 | 误报 417~430 ms | **失败**（`createSession failed - requested backend unavailable`） |
| MNN_OPENGL | 误报 452.590 ms | **失败**（同上） |
| MNN_NN | 误报 480.248 ms | **失败**（同上） |
| MNN_OpenCL 系列 | 见 §3 | ✅ 真实 GPU，数值修正 |

即：**这台机器上 MNN 只有 CPU + OpenCL 可用，Vulkan/OpenGL/NN 不可用**——
修复前这个事实被完全掩盖了。

---

## 3. 计时错位：avg_run_ms 测的是"命令提交"，不是推理（2026-08-29 修复）

### 3.1 现象

`MNN_OpenCL` 报 `avg_run_ms=4.137`，而同 GPU 经 OpenVINO/DML 是 91~190 ms。
集成显卡（Iris Xe）不可能比同 GPU 的其他路径快 22 倍。

反算搬运带宽即可证伪：输出 14,155,776 元素 = **54.0 MB**，而 `t_out=217.8 ms`
意味着有效带宽仅 **0.26 GB/s**（正常 10~20 GB/s）。所以 `t_out` 里装的不是搬运。

### 3.2 根因

MNN 设备后端的 `runSession()` 是**异步提交**，立即返回。`copyToHostTensor()` 才是
真正阻塞等待 GPU 完成的调用。原代码的计时窗口：

```
t0 = now()
runSession()            <- 只入队，微秒级
t1 = now()
avg_run_ms += t1 - t0   <- 测的是"提交"，不是推理

t_out0 = now()
copyToHostTensor()      <- GPU 计算在这里才被等待
t_out1 = now()
transfer_out_ms += ...  <- 整个 GPU 计算时间混进来
```

总量守恒：`4.1 + 4.5 + 217.8 ≈ 226 ms` 才是真实单次成本，其中 ~222 ms 的 GPU 计算
被错误地记入了 `transfer_out_ms`。

### 3.3 修复

把同步移进计时窗口，让 `avg_run_ms` 是真实推理时间、`transfer_out_ms` 是纯搬运：

```cpp
t0 = now()
runSession()
if (is_device) {                       // 强制同步，计入 avg_run_ms
    for (each output) output_tensors_[i]->copyToHostTensor(gpu_out_tensors[i]);
}
t1 = now()
avg_run_ms += t1 - t0                  // 真推理时间（含 GPU 计算）

t_out0 = now()
/* 快照 memcpy（copyToHostTensor 结果已缓存，此处只剩 memcpy） */
t_out1 = now()
transfer_out_ms += t1 - t0             // 真搬运时间
```

### 3.4 结果

| Backend | avg_before | avg_after | t_out_before | t_out_after |
|---|---|---|---|---|
| MNN_OpenCL | 4.137 | **259.908** | 217.806 | **11.833** |
| MNN_OpenCL_FP16 | 3.415 | **197.559** | 164.420 | **20.452** |
| MNN_OpenCL_BF16 | 4.633 | **269.089** | 227.463 | **11.919** |

`max_diff` 全部不变（0.00002384 / 0.16594648 / 0.00002384）——**精度数据本来就正确**，
错的只是时间归位。修复后 `t_out` 落在 11~20 ms，对应 54 MB 约 2.7~4.9 GB/s，
符合集成显卡 D2H 的真实带宽。

### 3.5 结论性经验

> **GPU/异步后端的 `run()` 返回 ≠ 计算完成。** 计时前必须确认同步点在哪；
> 否则"延迟"会退化成"命令提交延迟"，而真实的设备时间会跑到搬运列里去。
> 判据：若 `transfer_out_ms` 远高于「输出字节数 / 内存带宽」，几乎一定是同步点错位。
>
> 本工具中 MNN 是唯一有此问题的后端：ONNX/TFLite/LiteRT 的同步推理调用返回时
> 数据已在 CPU；NCNN 的 `extract()` 本身即同步（其 D2H 无法拆分，已在 README §5.5 说明）。

---

## 4. 其他

- `kernel.errors.txt` 是 MNN OpenCL 后端的运行产物（Intel Graphics Compiler 的 CISA
  kernel 诊断：寄存器区间重叠、隐式/显式参数顺序）。它是 **`load_lib` 高达 40 秒**的
  伴随现象（kernel 反复编译重试），但**不影响正确性**——§3.4 显示 OpenCL 的
  `max_diff` 正常且是真实 GPU 执行。已加入 `.gitignore`。
- `notes` 列不再重复 `t_in=`/`t_out=`：`transfer_in_ms`/`transfer_out_ms`/
  `transfer_total_ms` 已是独立的 CSV 列（16-18，6 位小数），notes 里的 3 位小数摘要纯属冗余。

---
