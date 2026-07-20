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
