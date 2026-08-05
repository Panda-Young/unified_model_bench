#!/usr/bin/env python3
"""Analyze all LayerNormalization nodes in the model to assess whether a
model-level rewrite (LN -> basic ops) is feasible and what configs exist."""
import collections
import onnx

MODEL = r"d:\WorkSpace\MSS\models\online_scnet_epoch_43_val_loss_0_67697_val_sdr_2_527_stream_1frame.onnx"

m = onnx.load(MODEL, load_external_data=False)

vi = {}
for v in list(m.graph.input) + list(m.graph.value_info) + list(m.graph.output):
    t = v.type.tensor_type
    if t.HasField("shape"):
        vi[v.name] = [d.dim_value if d.HasField("dim_value") else "?" for d in t.shape.dim]
for init in m.graph.initializer:
    vi[init.name] = list(init.dims)

ln = [n for n in m.graph.node if n.op_type == "LayerNormalization"]
print("total LayerNormalization nodes:", len(ln))

axes = collections.Counter()
sc_shapes = collections.Counter()
for n in ln:
    attrs = {a.name: onnx.helper.get_attribute_value(a) for a in n.attribute}
    axes[attrs.get("axis", -1)] += 1
    sc = n.input[1]
    sc_shapes[tuple(vi.get(sc, ["?"]))] += 1
print("axis distribution:", dict(axes))
print("scale shape distribution:", dict(sc_shapes))
print("all have 3 inputs:", all(len(n.input) == 3 for n in ln))
print("all single output:", all(len(n.output) == 1 for n in ln))
print(
    "epsilon:",
    collections.Counter(
        onnx.helper.get_attribute_value(a) for n in ln for a in n.attribute if a.name == "epsilon"
    ),
)
print(
    "stash_type:",
    collections.Counter(
        onnx.helper.get_attribute_value(a) for n in ln for a in n.attribute if a.name == "stash_type"
    ),
)

shown = set()
for n in ln:
    sc = n.input[1]
    key = tuple(vi.get(sc, ["?"]))
    if key in shown:
        continue
    shown.add(key)
    attrs = {a.name: onnx.helper.get_attribute_value(a) for a in n.attribute}
    xshp = vi.get(n.input[0], "?")
    yshp = vi.get(n.output[0], "?")
    print(
        "  LN: X=%s scale=%s (%s) out=%s axis=%s  name=%s"
        % (xshp, key, sc, yshp, attrs.get("axis", "-1"), n.name)
    )
