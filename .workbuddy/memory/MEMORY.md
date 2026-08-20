# 项目长期约定 (Unified Benchmark)

## 推理类型定位（重要决策）
- 该工程**仅用于内部音频模型的推理测试**，基准类型为 **float32（f32）**，覆盖绝大多数场景。
- **不做 int64 / 动态 shape / 非 f32 输入支持**（2026-08-20 评估后明确不加）。
  - 理由：音频模型输入输出几乎都是 f32；int64（如 LLM 的 input_ids）不在内部音频模型范围内。
  - 遇非 FLOAT 输入或动态 shape：`QueryIOMetadata` 提前明确报错并 return false，不静默/不崩溃。
- 因此 onnx_backend.cpp 中的两道检测（非 FLOAT / 动态维）是**预期行为**，不是待修复的 bug。

## 平台构建约束
- win-x86 构建仅启用 ONNX + NCNN 后端（TFLite/MNN/LiteRT/QNN 均 OFF）。
- 本机 `cmake --build` 调用的 VS2026 MSBuild 会 Access violation（环境问题），需在 VS IDE 内构建；
  Git Bash 下 MSBuild.exe/vswhere.exe 被安全策略拦截，可用 `cl.exe /Zs` 做单文件语法检查。
