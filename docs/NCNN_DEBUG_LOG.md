# NCNN Backend 修复记录

> **日期**: 2026-07-20
> **范围**: `src/ncnn_backend.cpp`

---

## 1. 全 NaN 输出 (`NCNN_CPU` / `NCNN_CPU_FP16`)

### 问题

NCNN 推理输出全部为 `-nan(ind)`，模型正常加载但数值全错。

### 根因 (两个独立问题)

#### 1.1 `build_inputs` 丢失 `.clone()`

行 557-558 处，`ncnn::Mat(w, h, c, input_bufs_[i])` 创建引用外部数据的 Mat，`mats.push_back(m)` 浅拷贝存入。当 `use_packing_layout=true` 时，NCNN 的 SIMD 优化会重组通道内存布局——若 Mat 不拥有数据，重组操作会破坏外部 buffer 或读到错乱数据。

在之前的代码编辑中 `.clone()` 被意外移除（原代码为 `tmp.clone()`）。

#### 1.2 `use_packing_layout` 对多输入模型不安全

测试模型有 2 个输入（`input_a` [1,3,224,224] + `input_b` [1,48,1,1]），PNNX 转换后通道顺序可能与 SIMD packing 布局不兼容。第二个输入的非标准通道数（48ch）尤其容易触发对齐问题。

### 修复

```cpp
// 修复 1: 恢复 clone() — NCNN 拥有数据副本
ncnn::Mat tmp(w, h, c, input_bufs_[i]);
mats.push_back(tmp.clone());

// 修复 2: 多输入模型禁用 packing
net_->opt.use_packing_layout = (num_inputs_ <= 1);
```

---

## 2. BF16 TryBf16Trial 崩溃 (use-after-free)

### 问题

`NCNN_CPU_BF16` backend 在 `model loaded successfully` 后立即崩溃，崩溃点在 `TryBf16Trial()`。

### 根因

`TryBf16Trial()` 的 for 循环中 `buf` 是局部变量：

```cpp
// 崩溃代码:
for (size_t i = 0; i < num_inputs_; ++i) {
    std::vector<float> buf(n, 0.0f);       // ① 循环内声明
    ncnn::Mat m(w, h, c, buf.data());      // ② Mat 仅引用 buf 指针，不拷贝数据
    mats.push_back(m);                      // ③ 浅拷贝存入 mats
    ex.input(name, mats.back());
}  // ④ 循环结束 → buf 销毁 → mats 中 Mat 持有悬空指针

ex.extract(...);  // ⑤ 访问悬空指针 → 崩溃
```

`ncnn::Mat(w, h, c, data)` **不拥有**数据，`extract()` 时才真正读取 → use-after-free。

### 修复

```cpp
// buf_pool 在循环外，数据一直存活；clone() 让 NCNN 拥有副本
std::vector<std::vector<float>> buf_pool(num_inputs_);
std::vector<ncnn::Mat> mats;
for (size_t i = 0; i < num_inputs_; ++i) {
    buf_pool[i].resize(n, 0.0f);
    ncnn::Mat m(w, h, c, buf_pool[i].data());
    mats.push_back(m.clone());  // NCNN 拥有数据，不再依赖 buf_pool
    ex.input(input_names_[i].c_str(), mats.back());
}
```

与 `RunBenchmark::build_inputs` 的 clone 模式一致。

---

## 3. 版本输出

### 问题

NCNN 初始化日志缺少版本信息。

### 修复

```cpp
#include <ncnn/c_api.h>   // ncnn_version() 在此声明

LOGI("NCNN: init complete (%.1f ms), version %s", init_ms_, ncnn_version());
```

---

## 4. BF16 支持检测

CPU 和 GPU 的 BF16 支持检测均在对应分支中正确使用：

| 平台 | 检测接口 |
|------|---------|
| CPU ARM | `ncnn::cpu_support_arm_bf16()` |
| CPU x86 | `safe_check_bf16()` → `ncnn::cpu_support_x86_avx512_bf16()` + SEH 保护 |
| GPU Vulkan | `gpu.support_bf16_packed()` / `gpu.support_bf16_storage()` |

检测不通过时会自动 fallback 到 FP32。

---

## 5. 经验总结

| 问题 | 教训 |
|------|------|
| 全 NaN | `ncnn::Mat(w,h,c,data)` 不拥有数据，必须 `.clone()`；多输入模型禁用 packing |
| TryBf16Trial 崩溃 | 循环内局部变量 + Mat 引用外部数据 = use-after-free，必须 pool 化 + clone |
| 版本输出 | `ncnn_version()` 在 C API 头文件 `c_api.h` 中，需单独 include |
| BF16 检测 | 接口已存在且正确使用，问题不在检测而在内存管理 |

---

## 6. 多输入模型 init 后崩溃 + 前向 -100（2026-08-05）

### 6.1 现象

`online_scnet_tfc_tdf.onnx`（70 输入 / 70 输出，流式语音模型）：
- 原代码在 `init complete` 后**立即崩溃**（debug 无任何日志）。
- 根因：`RunBenchmark::build_inputs` 与 `TryBf16Trial` 对所有输入无条件读
  `sh[3]/sh[2]/sh[1]`，而该模型 **48/70 个输入是 3D**（如 `in1=472,32,2`）→
  `sh[3]` 越界（UB）→ 崩溃。

### 6.2 修复（已合入）

新增秩无关辅助函数 `MakeMatFromShape()`，按 rank 1..4 分别构造 `ncnn::Mat`
（不再越界）；`build_inputs` / `TryBf16Trial` 均改用它，并补充 `ex.input()`
返回值检查 + `mats.size() != num_inputs_` 校验。x86 构建验证：**不再硬崩溃**，
改为优雅失败（`extract -100` → `Benchmark failed` → 进程正常退出，exit=1）。

### 6.3 模型有效性验证（python ncnn）

用 ncnn Python 绑定（`pip install ncnn`，v1.0.20240820）直接加载：
`load_param=0, load_model=0` → **模型文件本身有效**（结构与权重可正常加载）。
前向失败是**输入 Mat 维度映射问题**，不是模型文件损坏。

### 6.4 维度映射排查结论（重要）

用 `tools/ncnn_probe.py`（race 模式）对多种映射实测，结论：

| 映射 | 4D 输入 `[a,b,c,d]` | 3D 输入 `[a,b,c]` | 结果 |
|------|---------------------|-------------------|------|
| reversed+batch-drop | `(w=d,h=c,d=b,c=a)` | `(w=c,h=b,c=a)` | 前向到 blob 86（conv1d）失败 |
| **direct（无 batch-drop）** | `(w=a,h=b,d=c,c=d)` | `(w=a,h=b,c=c)` | **跑通整个编码器**（blob 70~92），末段 blob 1195 失败/硬崩溃 |
| mix（4D h/d 交换 + 3D 直序） | `(w=d,h=b,d=c,c=a)` | `(w=a,h=b,c=c)` | 编码器可跑，末段崩溃 |

- `Convolution1D` 源码：`const int h = bottom_blob.h;` **把 h 当通道数**（权重
  `num_output*kernel_w*h`），所以 3D 状态张量必须满足 **h=中间维（通道）**。
- **结论**：ncnn 输入布局不是简单反转；`direct`/`mix` 最接近正确，但末段仍
  失败并伴随 ncnn 内存池错误（`pool allocator destroyed too early`，错误维度
  会破坏堆 → 硬崩溃 0xC0000005）。**当前 C++ 保持安全的反转映射**（优雅失败
  不崩溃），等模型重新转换后再验证。

### 6.5 转换脚本 Bug：`verify_ncnn` 静默跳过（已修复 2026-08-05）

`tools/onnx_convert.py::verify_ncnn()` 原实现：
- 对每个输入无条件 `arr.squeeze(0)` 去 batch；对 3D 输入（如 `[472,32,2]`，
  dim0≠1）numpy `squeeze(0)` **抛异常** → 被 `except` 捕获 → **打印 SKIP 并
  return True（视为通过）** → 该模型**从未被真正验证**，坏转换静默放行。

**修复内容**：
1. 仅当 `len(shape)>=2 and shape[0]==1` 时才去掉 batch 维（3D 状态张量原样传入）。
2. 去掉"推理异常静默跳过"逻辑：load 失败 / `ex.input` 异常 / `ex.extract` 返回
   非 0 或空输出，一律打印 `FAIL` 并 `return False` → 整体 `return 2`（转换流程
   会删除产物并报错），**不再放行坏模型**。
3. 输出数量不一致也判 FAIL。

### 6.6 重新转换结论（2026-08-05，重要）

用修复后的 `verify_ncnn` 对 `online_scnet_tfc_tdf.onnx` 重新转换：

- `pnnx`（`C:\Python311\Scripts\pnnx.exe`）重新生成 `.ncnn.param/.bin`，与旧文件
  **SHA256 完全一致**（param/bin 均 identical）→ 模型**不是陈旧**，是 pnnx 对当前
  ONNX 的确定性输出。
- 修复后的 verify **如实报告 `FAIL - extract out0 failed (ret=-100)`**（不再静默
  跳过），转换流程按设计删除了产物（已从备份恢复 = 同一文件）。
- **pnnx 生成的官方测试脚本 `*_ncnn.py` 本身就是坏的**：对全部 70 个输入无条件
  `inN.squeeze(0)`，而 3D 输入（如 `in1=torch.rand(472,32,2)`）dim0≠1 → 直接抛
  异常。即安装的旧版 pnnx（`squeeze(0)` 风格）**无法正确处理带非 batch 3D 输入的
  模型**；新版 pnnx 改用 `ncnn.Mat(x.numpy(), batch_index=N)`，但本机 ncnn python
  1.0.20240820 的 `Mat` 构造不支持 `batch_index` 参数，无法按新约定喂数。
- **结论**：该模型在当前 pnnx + ncnn 组合下无法被正确转换/验证（首个
  `Convolution1D` 即失败，`-100`）。要让 NCNN 后端跑通此模型，需升级 pnnx 到支持
  `batch_index` 的版本（并升级 ncnn python/DLL 到配套版本）后重新转换验证；
  或排查 ncnn `Convolution1D` 对 c≠1 输入的处理（该层是首个失败点，blob 86）。

### 6.7 排查工具

- `tools/ncnn_probe.py`：python ncnn 加载/喂数/提取中间 blob（`walk`/`race`）。
- `tools/probe_blob86.py`：针对本模型快速定位 blob 86（conv1d_0）归属层并对比
  `rev_bd` / `dir_nbd` / `mix` 映射，输出首个失败 blob。
- 关键经验：ncnn 维度错误不会报错而是**静默破坏堆**，最终表现为
  `pool allocator destroyed too early` 或随机访问违例——排查时优先用
  python 绑定 + 中间 blob 定位首个失败层；转换脚本的验证绝不能静默跳过。

### 6.8 工具链升级尝试：pnnx/ncnn 升到 batch_index 版本（2026-08-05，结论）

按计划升级 pip 工具链到最新并重新转换验证，结论如下：

| 项 | 旧 | 新（最新 pip） | batch_index 支持 |
|----|----|----------------|------------------|
| pip ncnn python | 1.0.20240820 | **1.0.20260526**（已装） | **不支持**：wheel 未编译 NCNN_BATCH，`ncnn.Mat(arr, batch_index=0)` 抛 `TypeError: incompatible constructor arguments`（构造签名里根本没有 batch_index 参数） |
| pip pnnx | 20251119 | **20260526**（已装） | **不支持**：生成的 `*_ncnn.py` 仍是旧的 `ex.input("in0", ncnn.Mat(in0.squeeze(0).numpy()).clone())` 风格；GitHub master 才改为 `batch_index=`（说明 pip 包落后于 master） |
| 工程 C++ DLL | 1.0.20260113 | 未升级 | 未验证（需自行编译 NCNN_BATCH） |

重新转换结果（pnnx 20260526）：
- 新 `.ncnn.param` 变为 **78 KB**（旧 77 KB，说明新版 pnnx 确实产出了不同模型），
  `.ncnn.bin` 8507 KB。文件格式从 v2（`Layer { ... }`）变为 v1
  （`type name nb nt bottom... top...`，`1011 1206`）。
- 修复后的 `verify_ncnn` 验证：**仍然 `FAIL - extract out0 failed (ret=-100)`**。
- 按 blob 遍历定位：**首个失败点与旧模型完全一致——blob 86 = `conv1d_0`**。
- 映射对比（`tools/probe_blob86.py`）：
  - 默认（reversed+batch-drop）喂数 → blob 86 `ret=-100`（失败）。
  - `dir_nbd`（3D 直序 `(w=a,h=b,c=c)`）喂数 → blob 86 **通过**（`470x16`，
    w=470 正确），但继续前向到 blob 1195 时**硬崩溃**（进程直接退出，code 1）——
    与旧模型行为一致。

**结论（重要）**：
1. **pip ncnn / pnnx 最新版均不带 batch_index 支持**（ncnn wheel 无 NCNN_BATCH；
   pnnx 生成代码仍为 squeeze(0) 风格）。要真正拿到 batch_index，只能：
   - 从 GitHub 源码自行编译 ncnn（CMake 开 `NCNN_BATCH=ON`，python 绑定 + C++ DLL），
   - 从 GitHub master 源码自行编译 pnnx。
2. **即便拿到 batch_index 也救不了此模型**：batch_index 喂数 ≈ squeeze+反转
   （batch=1 时等价），而该喂法在 conv1d_0 处必然失败（`h=32` 通道对、但
   `w=2` 序列过短 → 输出宽度 0 → -100）。真正能跑通编码器的是 **3D 直序喂数**，
   说明模型 3D 张量语义是 `[seq, ch, feat]`，与 pnnx 默认的 `[ch, seq, feat]`
   反转约定冲突——这是模型作者定义与 ncnn 3D 约定的**根本性不匹配**，换版本无法
   解决。
3. **当前状态**：C++ 后端保持安全的反转映射（优雅失败 -100，不崩溃）；模型若要在
   NCNN 上真正跑通，需对转换后的 ncnn 图做输入侧 Permute/Reshape 手术（把 3D 输入
   布局掰成 conv1d 需要的直序），并排查 blob 1195 处崩溃（疑似后续某 4D 输入或层
   的布局问题）。此问题已超出"升级工具链"范畴。

---

## 7. Vulkan extract(out0) 硬崩溃 0xC0000005（2026-08-20，已修复）

### 7.1 现象

`unified_bench.exe test_model.onnx --repeat 1`，三个 Vulkan 后端（NCNN_VK /
NCNN_VK_BF16 / NCNN_VK_FP16）均在 warmup 阶段崩溃：

```
ncnn_backend.cpp:93 @safe_extract  NCNN: HW exception (0xC0000005) in extract(out0)
ncnn_backend.cpp:670 @NCNNBackend::RunBenchmark  NCNN: warmup extract out0 crashed (SEH) (ret=-1)
Benchmark failed: NCNN_VK / NCNN_VK_BF16 / NCNN_VK_FP16
```

CPU 后端（NCNN_CPU / _FP16 / _BF16）完全正常 → 问题锁定在 Vulkan 路径。

### 7.2 模型

`test_model.onnx`（gen_test_model.py，2 输入 1 输出，全部 **rank-4**）：

```
in0=1,3,224,224   (图像)
in1=1,48,1,1      (bias 注入，进 BinaryOp add)
out0=1,10
```

### 7.3 根因（python ncnn 逐步定位）

用 `tools/probe_vk_walk.py` 遍历中间 blob，精确锁定崩溃层：

| blob | 层 | 结果 |
|------|----|------|
| 2~7 | convrelu_0..cat_0 | 正常 |
| **8** | **BinaryOp add_0（cat 输出 + in1）** | **段错误** |

- **输入 Mat 是 dims=4**（`MakeMatFromShape` 用 `Mat(shape[3],shape[2],shape[1],shape[0],data)`
  构造，如 in1 → `Mat(w=1,h=1,d=48,c=1)`）。
- cat_0 输出是 dims=3（`Mat(w=224,h=224,c=48)`）。
- **ncnn Vulkan 的 BinaryOp shader 只支持 dims<=3 的广播**，4D 输入参与广播时
  越界访问 → 0xC0000005。CPU 路径的 BinaryOp 广播逻辑健壮，故 CPU 正常。
- 次要问题：旧代码用外部 flat buffer 直接包装 Mat，依赖外部 stride 恰好等于
  ncnn 对齐后的 cstep（如 `Mat(w=1,h=1,c=48)` 时 cstep=4，外部 stride=1，会错位）。

### 7.4 修复（`src/ncnn_backend.cpp` MakeMatFromShape 重写）

1. **返回自有 Mat**（数据拷入 ncnn 分配缓冲区，逐 channel memcpy），彻底摆脱
   外部 stride 与 cstep 对齐的耦合，消除潜在的 use-after-free / 错位读写。
2. **rank-4 且 N==1 的输入折叠为 3D**：`[1,C,H,W]` → `Mat(w=W,h=H,c=C)`。
   batch=1 时逻辑布局完全等价，但避开了 Vulkan BinaryOp 的 4D 崩溃路径。
   `[1,48,1,1]` → `Mat(w=1,h=1,c=48)`，在 add 中按 w/h=1 正确广播到 48 通道。
3. rank-4 且 N>1 保留旧 4D 映射（罕见，打 LOGW 警告）。

### 7.5 验证

- python ncnn（1.0.20260526）：CPU / VK FP32 / FP16 / BF16 全部 extract 成功，
  softmax 输出 sum≈1.0；与 CPU 参考最大偏差 < 4e-4。
- C++ 全量构建（cl.exe 手工编译+链接，x86）：9 个后端全部出结果：

| 后端 | avg(ms) | max_diff | 修复前 |
|------|---------|----------|--------|
| NCNN_CPU | 53.1 | 1.9e-7 | 正常 |
| NCNN_CPU_FP16 | 52.8 | 1.9e-7 | 正常 |
| NCNN_CPU_BF16 | 79.7 | 2.2e-7 | 正常 |
| **NCNN_VK** | 48.4 | 1e-8 | **崩溃** |
| **NCNN_VK_BF16** | 53.4 | 3.5e-4 | **崩溃** |
| **NCNN_VK_FP16** | 28.2 | 4.2e-5 | **崩溃** |

ONNX_DML_NPU 报 "No devices detected" 是硬件无 NPU 的预期跳过，与本次修复无关。

### 7.6 排查工具（tools/utils/）

- `probe_vk_crash.py` / `probe_vk_steps.py`：逐步复现（load→input→extract）。
- `probe_vk_walk.py`：遍历中间 blob，二分定位首个崩溃层。
- `probe_vk_fix.py` / `probe_vk_verify.py`：验证 3D 折叠修复方案 + 数值一致性。
- 关键经验：**ncnn Vulkan 路径遇到 4D blob 参与 BinaryOp 广播会硬崩**；排查
  Vulkan 崩溃优先用 python 绑定走中间 blob，先把崩溃层缩小到具体算子。
- 补充（2026-08-20）：`safe_extract()` 的 minidump 已从 `MiniDumpWithDataSegs`
  改为 `MiniDumpWithFullMemory`（完整堆/栈内容，便于在崩溃现场查 Mat 等数据；
  文件会显著变大，仅在崩溃时写入）。
