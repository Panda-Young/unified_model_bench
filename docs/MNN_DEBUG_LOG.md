# MNN Backend 修复记录

> **日期**: 2026-08-20
> **范围**: `src/mnn_backend.cpp`
> **环境**: Android（MNN 3.6.0，vendored 于 `deps/mnn/`）

---

## 1. Vulkan 后端 NaN：MNN_VULKAN / MNN_VULKAN_BF16 输出 NaN，MNN_VULKAN_FP16 正常

### 1.1 现象

同一模型（`test_model.mnn`，2 输入 1 输出，含 `BinaryOp add` + `Softmax`）在 Android 真机上：

| Backend | 配置（`mnn_backend.cpp:99-113`） | 结果 |
|---------|----------------------------------|------|
| `MNN_VULKAN` | `MNN_FORWARD_VULKAN` + `Precision_High`（fp32） | **NaN** |
| `MNN_VULKAN_FP16` | `MNN_FORWARD_VULKAN` + `Precision_Low`（fp16） | ✅ 正常 |
| `MNN_VULKAN_BF16` | `MNN_FORWARD_VULKAN` + `Precision_Low_BF16`（bf16） | **NaN** |

输入输出转换路径（`RunBenchmark` 的 `copyFromHostTensor`/`copyToHostTensor`）对三个
后端完全一致——NaN 不在 IO 层，而在 MNN 算子计算层，由精度配置差异引起。

### 1.2 根因（MNN 官方硬件/精度支持矩阵）

MNN 官方 README 的「Architecture / Precision」支持矩阵（当前 master 持续维护）：

```
Architecture / Precision   Normal(FP32)  FP16  BF16  Int8
GPU Vulkan                 A            A     C     A
GPU OpenCL                 A            S     C     S
CPU ARMv8                  S            S(ARMv8.2) S(ARMv8.6)  S

S = 深度优化推荐 ｜ A = 可用但未深度优化（known issues）｜ C = 不支持
```

#### `MNN_VULKAN_BF16` → NaN（决定性根因）

**MNN 的 Vulkan 后端不支持 BF16（矩阵标 `C`）**。Vulkan shader 原生无 bf16
（`VK_KHR_shader_float16_int8` 只覆盖 fp16），`Precision_Low_BF16` 在 Vulkan 上
没有可用内核 → 调度走错误映射/fallback 路径 → 中间张量格式错乱 → NaN。
**这是"不支持却硬跑"，不是精度损失**。同理 `MNN_OPENCL_BF16`（OpenCL 的 BF16 也是
`C`）预期同样 NaN。

#### `MNN_VULKAN`（Precision_High/fp32）→ NaN

Vulkan 的 FP32 是 `A` 级——可用但已知有 computation 问题（MNN 3.4.0 release notes
明确写 *"fixed multiple OpenCL/Vulkan/Metal computation and stability issues"*，
且 FP32 只给 A 不给 S）。机制：MNN Vulkan 后端大量用 **fp16 纹理存中间张量**
（image 模式 `VK_FORMAT_R16G16B16A16_SFLOAT`），`Precision_High` 名义 fp32 但中间
结果仍落 fp16 → **混合精度路径**在数值敏感算子（`BinaryOp add`、`Softmax` 的
exp/除法）上中间值于 fp16 边界溢出 → inf → NaN；fp32↔fp16 转换点也是精度重灾区。

#### `MNN_VULKAN_FP16` → 正常

FP16 是 Vulkan **全链路原生自洽**：输入先转 fp16、所有算子走 fp16 内核（Softmax
等有 fp16 安全实现，max-subtraction 防溢出）、输出转回 fp32。数值路径从头到尾一致，
无"fp32 名义 + fp16 中间"的错配，故不出 NaN。

### 1.3 结论与建议

- **Android 上 MNN 取正确数值**：用 `MNN_CPU`（官方 S 级）或 `MNN_VULKAN_FP16`；
  **避免 `MNN_VULKAN`（FP32，A 级已知坑）与 `MNN_VULKAN_BF16`（官方 C = 不支持）**
- 若要在工具层拦截（参照 NCNN BF16 的提前报错模式），可在 `MNN_VULKAN_BF16`
  初始化时直接 `return false` 并给明确原因（"MNN Vulkan does not support BF16"），
  避免 NaN 结果污染 CSV 对比
- 升级 MNN 版本大概率救不了 BF16（支持矩阵是架构级限制，非版本 bug）

### 1.4 排查工具 / 验证方式

- MNN 官方 README 支持矩阵：`github.com/alibaba/MNN`（README 中 Architecture /
  Precision 表）
- 真机 A/B：同一 `.mnn` 模型分别以 `--backend MNN_VULKAN` /
  `MNN_VULKAN_FP16` / `MNN_VULKAN_BF16` 单跑，对比 `max_diff`（NaN 行 diff 会异常）

---

## 2. 经验总结

| 问题 | 教训 |
|------|------|
| Vulkan BF16 NaN | 先查框架官方精度支持矩阵再启用低精度后端；BF16 在 MNN 仅 CPU（ARMv8.2+）与 x86-AVX512 支持 |
| Vulkan FP32 NaN | GPU 后端"名义 FP32"可能含 fp16 中间纹理，混合精度路径对数值敏感模型不稳；要稳定数值用全链路 fp16 或 CPU |
| FP16 反而稳定 | 全链路单一精度（输入/算子/输出一致）比"高精度名义 + 低精度中间"的混合路径更可靠 |
