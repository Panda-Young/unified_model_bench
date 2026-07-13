#!/usr/bin/env python3
"""
Generate a multi-input / single-output ONNX test model with widely-supported ops.
Compatible with NCNN (PNNX), MNN (MNNConvert), TFLite (onnx2tf).

Usage:
  python gen_test_model.py                              # → test_model.onnx
  python gen_test_model.py --size 96                    # smaller feature map
  python gen_test_model.py -o custom.onnx               # custom name

Model structure (multi-input):
  input_a(1,3,224,224) ──conv─bn─relu──┬──branch_a─┬──cat─Add─conv─pool─conv─gap──fc─softmax─→ output(1,10)
  input_b(1,48,1,1)    ────────────────┴──branch_b─┘          ↑
                                                          input_b(bias)
"""

import argparse
import onnx
import numpy as np
from onnx import helper, TensorProto
from pathlib import Path

np.random.seed(42)


def make_initializer(name: str, shape, dtype=TensorProto.FLOAT):
    """Create an initializer tensor from random or zero data."""
    data = np.random.randn(*shape).astype(np.float32) * 0.02
    if "weight" in name.lower() or "conv" in name.lower() or "fc" in name.lower():
        pass  # random
    elif "bias" in name.lower():
        data = np.zeros(shape, dtype=np.float32)
    elif "running_mean" in name.lower() or "running_var" in name.lower():
        data = np.zeros(shape, dtype=np.float32) if "mean" in name else np.ones(shape, dtype=np.float32)
    elif "gamma" in name.lower() or "beta" in name.lower() or "scale" in name.lower() or "shift" in name.lower():
        data = np.ones(shape, dtype=np.float32) if "scale" in name or "gamma" in name else np.zeros(shape, dtype=np.float32)
    return helper.make_tensor(name, dtype, list(shape), data.flatten().tolist())


def bn_init(name: str, c: int):
    return [
        helper.make_tensor(f"{name}_gamma", TensorProto.FLOAT, [c], np.ones(c, dtype=np.float32).tolist()),
        helper.make_tensor(f"{name}_beta", TensorProto.FLOAT, [c], np.zeros(c, dtype=np.float32).tolist()),
        helper.make_tensor(f"{name}_mean", TensorProto.FLOAT, [c], np.zeros(c, dtype=np.float32).tolist()),
        helper.make_tensor(f"{name}_var", TensorProto.FLOAT, [c], np.ones(c, dtype=np.float32).tolist()),
    ]


def build_model(H=224, W=224):
    """
    Multi-input CNN (single output for TFLite compatibility):
      input_a(1,3,H,W) → conv1 → bn1 → relu → r1
      input_b(1,48,1,1)─────────────────────→ Add (as bias injection)
          r1 ──conv2a──bn2a──relu──→ branch_a
          r1 ──conv2b──bn2b──relu──→ branch_b
          cat(branch_a, branch_b) ──Add(input_b)──conv3──bn3──relu──pool──conv4──bn4──relu──gap
            └──→ reshape ──fc ──softmax ──→ output(1,10)
    """
    C_in = 3
    output_c = 10

    # Second input: bias injection tensor (1 x 48 x 1 x 1)
    # After concat, 48 channels → injected into conv3's input
    nodes, inits = [], []

    # ── conv1: 3→16 k3s1p1 ──
    nodes.append(helper.make_node("Conv", ["input_a", "conv1_w", "conv1_b"],
                                   ["conv1"], kernel_shape=[3, 3], strides=[1, 1], pads=[1, 1, 1, 1]))
    inits.append(make_initializer("conv1_w", [16, 3, 3, 3]))
    inits.append(make_initializer("conv1_b", [16]))
    nodes.append(helper.make_node("BatchNormalization", ["conv1", "bn1_gamma","bn1_beta","bn1_mean","bn1_var"],
                                   ["bn1"], epsilon=1e-5))
    inits.extend(bn_init("bn1", 16))
    nodes.append(helper.make_node("Relu", ["bn1"], ["r1"]))

    # ── Branch A: conv2a 16→24 k3s1p1 ──
    nodes.append(helper.make_node("Conv", ["r1", "conv2a_w", "conv2a_b"],
                                   ["conv2a"], kernel_shape=[3, 3], strides=[1, 1], pads=[1, 1, 1, 1]))
    inits.append(make_initializer("conv2a_w", [24, 16, 3, 3]))
    inits.append(make_initializer("conv2a_b", [24]))
    nodes.append(helper.make_node("BatchNormalization", ["conv2a","bn2a_gamma","bn2a_beta","bn2a_mean","bn2a_var"],
                                   ["bn2a"], epsilon=1e-5))
    inits.extend(bn_init("bn2a", 24))
    nodes.append(helper.make_node("Relu", ["bn2a"], ["r2a"]))

    # ── Branch B: conv2b 16→24 k5s1p2 ──
    nodes.append(helper.make_node("Conv", ["r1", "conv2b_w", "conv2b_b"],
                                   ["conv2b"], kernel_shape=[5, 5], strides=[1, 1], pads=[2, 2, 2, 2]))
    inits.append(make_initializer("conv2b_w", [24, 16, 5, 5]))
    inits.append(make_initializer("conv2b_b", [24]))
    nodes.append(helper.make_node("BatchNormalization", ["conv2b","bn2b_gamma","bn2b_beta","bn2b_mean","bn2b_var"],
                                   ["bn2b"], epsilon=1e-5))
    inits.extend(bn_init("bn2b", 24))
    nodes.append(helper.make_node("Relu", ["bn2b"], ["r2b"]))

    # ── Concat: (branch_a, branch_b) -> 48 channels ──
    nodes.append(helper.make_node("Concat", ["r2a", "r2b"], ["cat1"], axis=1))

    # ── Inject input_b as a bias (Add after concat, 48ch) ──
    nodes.append(helper.make_node("Add", ["cat1", "input_b"], ["cat_bias"]))

    # ── conv3: 48→32 k3s1p1 ──
    nodes.append(helper.make_node("Conv", ["cat_bias", "conv3_w", "conv3_b"],
                                   ["conv3"], kernel_shape=[3, 3], strides=[1, 1], pads=[1, 1, 1, 1]))
    inits.append(make_initializer("conv3_w", [32, 48, 3, 3]))
    inits.append(make_initializer("conv3_b", [32]))
    nodes.append(helper.make_node("BatchNormalization", ["conv3","bn3_gamma","bn3_beta","bn3_mean","bn3_var"],
                                   ["bn3"], epsilon=1e-5))
    inits.extend(bn_init("bn3", 32))
    nodes.append(helper.make_node("Relu", ["bn3"], ["r3"]))

    # ── MaxPool k2s2 ──
    nodes.append(helper.make_node("MaxPool", ["r3"], ["pool1"],
                                   kernel_shape=[2, 2], strides=[2, 2], pads=[0, 0, 0, 0]))

    # ── conv4: 32→64 k3s1p1 ──
    nodes.append(helper.make_node("Conv", ["pool1", "conv4_w", "conv4_b"],
                                   ["conv4"], kernel_shape=[3, 3], strides=[1, 1], pads=[1, 1, 1, 1]))
    inits.append(make_initializer("conv4_w", [64, 32, 3, 3]))
    inits.append(make_initializer("conv4_b", [64]))
    nodes.append(helper.make_node("BatchNormalization", ["conv4","bn4_gamma","bn4_beta","bn4_mean","bn4_var"],
                                   ["bn4"], epsilon=1e-5))
    inits.extend(bn_init("bn4", 64))
    nodes.append(helper.make_node("Relu", ["bn4"], ["r4"]))

    # ── GlobalAveragePool ──
    nodes.append(helper.make_node("GlobalAveragePool", ["r4"], ["gap"]))

    # ── Output 1: Reshape + FC + Softmax → (1,10) ──
    nodes.append(helper.make_node("Reshape", ["gap", "reshape_shape"], ["flat"]))
    inits.append(helper.make_tensor("reshape_shape", TensorProto.INT64, [2], [1, 64]))
    nodes.append(helper.make_node("MatMul", ["flat", "fc_w"], ["fc_mm"]))
    inits.append(make_initializer("fc_w", [64, output_c]))
    nodes.append(helper.make_node("Add", ["fc_mm", "fc_b"], ["logits"]))
    inits.append(make_initializer("fc_b", [output_c]))
    nodes.append(helper.make_node("Softmax", ["logits"], ["output"], axis=1))

    # ── Graph (single output for TFLite compatibility) ──
    graph_def = helper.make_graph(
        nodes, "test_model",
        [
            helper.make_tensor_value_info("input_a", TensorProto.FLOAT, [1, C_in, H, W]),
            helper.make_tensor_value_info("input_b", TensorProto.FLOAT, [1, 48, 1, 1]),
        ],
        [
            helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, output_c]),
        ],
        inits,
    )
    model = helper.make_model(graph_def, opset_imports=[helper.make_opsetid("", 15)])
    model.ir_version = 8
    return model


def main():
    parser = argparse.ArgumentParser(description="Generate multi-IO test ONNX model")
    parser.add_argument("--size", type=int, default=224, help="Input H/W (default: 224)")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output path (default: test_model.onnx)")
    args = parser.parse_args()

    model = build_model(H=args.size, W=args.size)
    model.producer_name = "gen_test_model"
    model.producer_version = "1.0"

    out_path = args.output or "test_model.onnx"
    onnx.save(model, out_path)

    # Print I/O shapes
    g = model.graph
    print(f"Model: {out_path}")
    for v in g.input:
        shape = [d.dim_value for d in v.type.tensor_type.shape.dim]
        print(f"  Input:  {v.name} {shape}")
    for v in g.output:
        shape = [d.dim_value for d in v.type.tensor_type.shape.dim]
        print(f"  Output: {v.name} {shape}")
    for v in g.value_info:
        if v.type.tensor_type.shape.dim:
            shape = [d.dim_value for d in v.type.tensor_type.shape.dim]
            print(f"  Value:  {v.name} {shape}")

    # Verify
    try:
        import onnxruntime as ort
        data_a = np.random.randn(1, 3, args.size, args.size).astype(np.float32)
        data_b = np.random.randn(1, 48, 1, 1).astype(np.float32)
        sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
        outputs = sess.run(None, {"input_a": data_a, "input_b": data_b})
        print(f"\n  ONNX Runtime inference OK: output shape={outputs[0].shape}")
        sz = Path(out_path).stat().st_size
        print(f"  File size: {sz / 1024:.1f} KB")
    except Exception as e:
        print(f"  Verification error: {e}")


if __name__ == "__main__":
    main()
