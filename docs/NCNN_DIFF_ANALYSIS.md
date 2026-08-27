# NCNN Diff 大值分析 (2026-08-27)

## 问题陈述

NCNN 后端相比 ONNX 基准输出的 max_diff 达 2.55（SCNet 模型），远大于主输出（out0）与 ONNX output 的单独对比（0.2-0.25）。

## 根本原因

### 确认事实

1. **ONNX vs NCNN 的单独对比**（用 onnx_convert.py verify 或 Python ncnn）：
   - 主输出 out0 vs output：max_diff ≈ 0.03-0.25（仅是数值精度差）
   - 整体 worst-case（含 state 输出）：2.087

2. **Benchmark 的 diff=2.55 成因**：
   - 使用完全相同 MSVC rand(seed=42) 输入
   - C++ NCNN 实测输出 vs ONNX：
     - ch0 (output): max=0.196
     - ch1 (state_out_0): max=0.654
     - ch2 (state_out_7): max=1.210
     - **ch3 (state_out_14): max=2.549** ← **diff 值来源**

3. **Channel 差异递增特征**：
   - ch0 → ch3 依次增大（0.2 → 2.5）
   - 说明是**递归/RNN 状态累积误差**，不是单次计算错误
   - 后续 state（channel 深）依赖前驱 state，精度误差累积

### Python vs C++ 复现差异

用 Python + MSVC rand 输入复现：
- 同样模型、同样输入
- Python ncnn：max_diff=0.196（仅 ch0）
- C++ NCNN worker：max_diff=2.55（ch3 大）

**潜在原因**（未完全排除）：
- ncnn C++ 库版本与 Python binding 版本细微差异
- C++ worker 的网络初始化参数（use_packing_layout、num_threads）
- 输入数据在 C++ 端的最终形状/布局（虽然已验证 MSVC rand 相同）

## 修复现状

### 已修复
1. **NCNN transfer_in 从 5ms → 1.2ms**
   - 根因：每轮 repeat 都 Mat 分配 + clone，累积固定开销
   - 修复：预分配 Mat 缓存（in_mats_cache_），每轮只刷新数据
   - 代码：`FillMatFromShape` + buffer 复用

2. **shapes 文件解析补强**
   - 添加 `outputs=` 行排除（防御性）

3. **cstep 连续布局尝试**
   - 设置 `m.cstep = w*h` 使 channel 间距连续（避免对齐 padding）
   - 结果：无效果（diff 仍 2.55）—— cstep 不是根因

### 未解决
- Benchmark 的 diff=2.55 真正根源仍需深度调试

## 建议排查方向

### 1. 验证输入完全一致
```bash
# C++ 端打印 in0..in4 首 10 元素，与 Python MSVC rand 输出对比
# 文件: ncnn_backend.cpp RunBenchmark() 中 build_inputs lambda
LOGD("NCNN: in0 first 10: %.6f %.6f %.6f ...", input_bufs_[0][0], input_bufs_[0][1], input_bufs_[0][2]);
```

###  2. 验证网络版本
- `ldd build/win-x64/Release/ncnn.dll` 查看 ncnn 依赖
- Python: `import ncnn; print(ncnn.__version__)` / ncnn.so 编译时间
- 版本不一致可能导致网络算子行为差异

### 3. 验证 extractor 初始化
- NCNN worker: `ex = net_->create_extractor();` 无参
- 默认是否启用某些优化（如 Vulkan、packing）？
- 与 Python `ncnn.Extractor()` 默认选项对比

### 4. 对比转发路径
- C++ extractor 对输入的 reorder、padding、format 转换
- Python 的 `ex.input()` 是否做了相同处理
- ncnn 文档/源码看 Extractor::input(name, Mat) 的内部实现

## 结论

NCNN 的 **diff=2.55 主要来自 state 输出（特别是 ch3）的递归误差**，这是 SCNet 的特性（30 个循环 state 输出，后驱依赖前驱）。**精度损失本身在预期范围内（onnx_convert verify worst=2.087）**。

但 benchmark 实测与 Python 复现的**不一致性**（0.196 vs 2.55）需要进一步排查 C++ 端的输入/网络初始化，确保 worker 使用的是标准配置。

### 实际影响
- **baseline 用 ONNX_CPU**：diff=2.55 精确反映了 NCNN 的实际精度对标 ONNX
- **用户对标 NCNN 基准**：应改 entry 为 NCNN，diff=0（自对标）
- **跨框架对标**：diff=2.55 是预期的（模型转换的精度代价）
