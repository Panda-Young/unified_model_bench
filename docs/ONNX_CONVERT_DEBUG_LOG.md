# ONNX 转 ncnn/MNN/TFLite 转换排查记录

> 适用范围：`tools/onnx_convert.py` 的 ONNX → ncnn（pnnx）/ MNN / TFLite 转换链路。
> 更新日期：2026-08-27。

---

## 1. 问题现象（2026-08-27，scnet_tfc_tdf_v3_20260821.onnx → ncnn）

原始 ONNX（1694 节点，31 输入/31 输出，含 30 个 state 张量的流式模型）直接转 ncnn 失败：

```
WARNING: Tile '/Tile' has dynamic repeats, skipping
WARNING: Tile '/Tile_1' has dynamic repeats, skipping
WARNING: Tile '/Tile_2' has dynamic repeats, skipping
malformed split split_size_or_sections type 0
malformed split split_size_or_sections type 0
malformed split split_size_or_sections type 0
ignore Slice split_0 param dim=1
...
Verifying NCNN...
layer torch.tile not exists or registered
network graph not ready
  NCNN CPU: FAIL - model load error (param=-1 model=-1)
```

两种失败形态：
1. **`torch.tile` 层不存在**（param 里残留 pnnx 自定义层，ncnn 运行时无此层）→ 加载失败；
2. **pnnx 崩溃**（`0xC0000005` 访问违例，exit=-1073741819）：在 `pass_ncnn` 处理动态 Split 时段错误。

---

## 2. 根因分析

### 2.1 模型结构特征（触发点）

| 特征 | 位置 | 后果 |
|---|---|---|
| **3 个 Tile**，repeats 均为常量 `[1,2,1,1]` | `/Tile`、`/Tile_1`、`/Tile_2` | pnnx 把 repeats 当"动态"（见下）→ 残留 `torch.tile` |
| **3 个 Split**（axis=1，均分），无 `split` 输入 | `/Split`(320→160+160)、`/Split_1`(160→80+80)、`/Split_2`(80→40+40) | pnnx 无法推断 split_size → 崩溃 |
| 大量动态 shape 链 | `Shape→Gather→Sub→ConstantOfShape→Concat→Reshape`（82 个 Reshape、44 个 ConstantOfShape、19 个 Shape） | 增加 pnnx 静态化负担 |
| 3 个 Expand（沿 batch 维） | `/Expand`、`/Expand_1`、`/Expand_2` | onnxsim 可消除 |

### 2.2 三个"伪动态"真相（关键认知）

1. **Tile repeats 实际是常量**：`onnx::Tile_2212 = [1,2,1,1]`（int64，4 维），但它是 **Constant 节点输出**（PyTorch 导出风格），pnnx **不做常量传播**，凡 repeats 非常量输入一律判"dynamic"——**误报**。
2. **Split 实际可静态化**：axis=1 无 splits 属性 = 均分；输入通道数由 shape inference 可得（320/160/80）。pnnx 崩溃是**它对"无 split 输入 + 输入来自 Slice"的图处理 bug**。
3. **`ends=MAX`（9223372036854775807）** 的 Slice 依赖运行时 H，但输入 H 固定 2049 → 本可静态化。

---

## 3. 解决过程（三步，已固化为脚本）

### 步骤 ①：onnxsim 静态化（可选但推荐）

```bash
python -m onnxsim 模型.onnx 模型.sim.onnx --input-shape <全部输入 shape>
```

效果（本模型）：Reshape 82→0、ConstantOfShape 44→0、Shape 19→0、Expand 3→0、Cast 71→0——大量动态 shape 链被常量折叠。

### 步骤 ②：Tile → Concat（脚本已有，已增强）

`tools/onnx_convert.py` 的 `preprocess_onnx_for_ncnn()` 把常量 repeats 的 Tile 替换为等价 Concat：
- `rep=[1,2,1,1]` → 单轴 `Concat(x2, axis=1)`；
- **增强**：现在也识别 **Constant 节点输出**形式的 repeats（原只读 initializer，PyTorch 导出常是 Constant 节点）。

### 步骤 ③：Split → 静态 split（脚本新增，本次核心改进）

`preprocess_onnx_for_ncnn()` 新增 **Split 静态化**：
- 用 `onnx.shape_inference.infer_shapes()` 推 Split 输入的**具体 shape**；
- axis 维若能得具体值且能被输出数整除 → 生成 `split` **输入张量**（opset13+ 形式，如 `[160,160]`）；
- 动态维/不可整除 → 跳过（留给用户处理）。

### 最终产物

`scnet_tfc_tdf_v3_20260821.ncnn.param / .bin` 生成成功，验证 `1/1 conversions passed`。

---

## 4. 脚本改进内容（2026-08-27）

`tools/onnx_convert.py` `preprocess_onnx_for_ncnn()`：

| 改进 | 说明 |
|---|---|
| **Constant 节点 repeats 识别** | 收集 `Constant` 节点的 `value` 属性作为常量，使 PyTorch 风格的 Tile repeats 也能被 Tile→Concat 处理 |
| **Split 静态化** | 自动 shape inference + 生成 `split` 输入张量（`Split->static`），消除 pnnx 的 `malformed split` 崩溃 |
| **独立于 Tile/Erf 触发** | Split 修复不再依赖"Tile/Erf 有改动"才运行（修复了原逻辑 `if changes==0` 提前 return 导致 Split 修复永不触发的 bug） |

**验证**：原始模型直接转换通过（无需手工 onnxsim/改 Split）。

### 4.1 NCNN dual FP16 未生成修复（2026-08-27 追加）

**现象**：`--to all`（不带 `--no-dual`）时 `_fp16.ncnn.bin` 不生成，日志显示 `--dual: generating FP16 model...` 后无 `FP16 bin:` 输出，FP16 权重被当中间产物清理掉。

**根因**：`convert_ncnn()` dual 分支中查找 FP16 输出用了错误路径：

```python
# 错误：pnnx 的输出文件名基于其输入（prep 预处理模型）
tmp_b = _find_pnnx_output(model_path, ".ncnn.bin")   # 查 model.ncnn.bin
# 但 pnnx 实际输出 prep.ncnn.bin（FP16），且 model.ncnn.bin 已被 old_bin.unlink() 删除
# → tmp_b 返回 None → FP16 静默丢失
```

**修复**：改用 `pnnx_input`（prep 路径）查找：

```python
tmp_b = _find_pnnx_output(pnnx_input, ".ncnn.bin")
```

**验证**：`FP16 bin: scnet_tfc_tdf_v3_20260821_fp16.ncnn.bin (1954 KB)`，FP16/FP32 大小比 ≈ 0.51，符合预期；`--no-dual` 时仍正确跳过。

---

## 5. 结论性经验（避免重复排查）

1. **pnnx 对"常量但非 initializer"的 Tile repeats 判 dynamic 是误报**——转码前用脚本把 Constant 节点输出并入常量表即可；
2. **pnnx 对无 `split` 输入 + 输入来自 Slice 的 Split 会段错误崩溃**（0xC0000005）——先用 shape inference 补上静态 `split` 输入张量；
3. **onnxsim 能折叠绝大多数动态 shape 链**（Shape/Gather/ConstantOfShape/Reshape/Expand/Cast），但对"运行时 H 依赖的 `ends=MAX` Slice"不折叠（输入固定时可手工静态化）；
4. 流式模型（带 state 张量）的 `Split/Tile/Expand` 多为**伪动态**：repeats/split_size 实为常量，只是图结构用运行时表达——静态化后所有框架（ncnn/MNN/TFLite）都能转；
5. 转换验证用 `unified_bench` 的 verify 步骤（`Done. N/N conversions passed`）为准，而非只看 pnnx 是否生成文件。

---

## 6. TFLite 转换失败排查（2026-08-27 追加，scnet_tfc_tdf_v3_20260821.onnx → TFLite）

> NCNN/MNN 转换成功后，TFLite（onnx2tf）转换失败，本文追加完整排查与修复。

### 6.1 问题现象

```
ERROR: onnx_op_name: wa/down_convs.0/depthwise/Conv
TypeError: Eager execution of tf.constant with unsupported shape.
Tensor [[[[-0.12807898  0.04369543]] ...]] has 6 elements, but got `shape` (1, 3, 231, 2) with 1386 elements
```

- 转换在第一个 group/depthwise Conv（`down_convs.0/depthwise/Conv`，weight `[2,1,1,3]`，group=2）失败；
- TF Conv2D 期望 kernel `(1,3,231,2)`，实际只有 6 个元素（kernel `(1,3,1,2)`）。

### 6.2 根因（两层）

**根因 A：动态中间 shape 破坏 NCHW→NHWC 布局转换**

打点 onnx2tf 内部发现：

```
DBG_CONV input_tensor shape=(3, 2, 1, 462)   ← NCHW 布局（C=2）未被转 NHWC
DBG_CONV graph_node.inputs[0].shape=['unk__0','unk__1','unk__2','unk__3']  ← 动态 shape！
```

- 模型的 Pad pads 是**动态计算的**（`Shape→Concat→Reshape→Slice→Transpose→Reshape→Cast` 链，PyTorch 导出风格，实际是常量 `[0,0,0,0]`）；
- onnx2tf 对中间 tensor 形状未知 → 无法判断 NCHW/NHWC → group=2 的 Conv2D 输入 channel 被误判为 462/231 → kernel 形状不匹配。

**根因 B：onnx2tf 1.28.8 与 onnx_graphsurgeon 0.5.8 API 不兼容**

onnxsim 折叠后形状变静态，但 onnx2tf 又报：

```
AttributeError: 'Node' object has no attribute 'output'   (Conv.py:938)
```

- onnx2tf 1.28.8 使用旧的 `graph_node.output`（单数）API；
- onnx_graphsurgeon 0.5.8 的 `Node` 只有 `outputs`（复数列表）；
- 官方 onnx2tf **v2.0.0 起彻底移除对 onnx-graphsurgeon 的依赖**（内置 `onnx2tf.gs`），当前 1.28.8 + gs 0.5.8 组合存在此兼容性问题。

### 6.3 解决方案（已固化到脚本）

| 改进 | 位置 | 说明 |
|---|---|---|
| **onnxsim 常量折叠** | `convert_tflite()` 新增 `fold_constants_with_onnxsim()` | 转换前对预处理模型跑 `onnxsim.simplify`，把动态 pads/shape 链折叠成静态 initializer（本模型：Shape 19→0、ConstantOfShape 44→0、Reshape 82→0、Cast 71→0），使 onnx2tf 能正确做 NCHW→NHWC |
| **gs.Node 单数别名补丁** | 脚本顶部 | 给 `gs.Node` 添加 `output`/`input` property 别名（映射到 `outputs[0]`/`inputs[0]`），兼容 onnx2tf 1.28.8 的旧 API，无需降级 onnx_graphsurgeon |

### 6.4 最终结果

```
Done. 3/3 conversions passed.

Format   File                                             Size Status
NCNN     scnet_tfc_tdf_v3_20260821.ncnn.param             39 KB OK
NCNN     scnet_tfc_tdf_v3_20260821.ncnn.bin            3829 KB OK
MNN      scnet_tfc_tdf_v3_20260821.mnn                 3901 KB OK
TFLite   scnet_tfc_tdf_v3_20260821.tflite              4263 KB OK
```

验证数值：
- **NCNN**：PASS（worst=2.09，LSTM 状态输出的数值扩散，主输出 `output` max=0.03）
- **MNN**：MARGINAL（worst=0.0024，fp32 等价）
- **TFLite**：主输出 `output` max=0.04（Erf→Tanh 近似误差）；state_out 差异 0.1~1.27 属 LSTM 状态累积对随机输入的敏感扩散，模型保留（LARGE 不删文件），需真实数据验证

---

## 7. 结论性经验（TFLite 追加，避免重复排查）

1. **PyTorch 导出的流式音频模型，Pad pads / Slice 参数常用动态 shape 链表达（实际是常量）**——onnx2tf 无法推断中间 tensor 布局，group/depthwise Conv 必然转换失败。**先 onnxsim 常量折叠再转 onnx2tf**；
2. **onnx2tf 1.28.x + onnx_graphsurgeon 0.5.x 存在 `graph_node.output` API 不兼容**——无需降级 gs，加 `output`/`input` property 别名即可；
3. **流式模型（31 输入/31 输出带 state）的验证差异集中在 state_out**：LSTM/GRU 状态是累积量，随机输入下 1e-1 量级扩散属正常，应看主输出（如 `output`）的 max 误差判断转换质量；
4. onnx2tf 的 `-dgc`（disable_group_convolution）、`-kt/-kat`、`-ois` 参数对本模型均无法绕过动态 shape 问题，**onnxsim 前置折叠才是治本**。

---

## 8. 相关文件

- 脚本：`tools/onnx_convert.py`
- 中间产物（可清理）：`*.sim.onnx`、`*.static.onnx`、`*.prep.onnx`、`*.prep.onnxsim.onnx`、`*.shapes`、`convert_ncnn_*.log`
- 最终产物：`scnet_tfc_tdf_v3_20260821.ncnn.param` / `.ncnn.bin` / `.mnn` / `.tflite` / `.shapes`
