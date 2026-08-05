"""Plan-C step 3: verify the 'upgrade 3D -> 4D' path (方案 B) is handled
correctly by pnnx. Build a tiny ONNX using ONLY 4D tensors:
  in_a: [3,32,2,1], in_b: [3,32,1,1]
  Concat(axis=2) -> [3,32,3,1]
  Conv2d(kernel=[3,1]) -> [3,16,1,1]
Run pnnx and inspect the generated ncnn param + feeding script. This mirrors
the real model's cat_0 -> conv1d_0 structure but in 4D, which pnnx is known to
handle reliably.
"""
import os
import subprocess

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

WORK = r"D:\WorkSpace\unified_bench\tools\_pnnx4d"


def build():
    os.makedirs(WORK, exist_ok=True)
    w = np.random.randn(16, 32, 3, 1).astype(np.float32)  # Conv2d [out, in, kh, kw]
    b = np.random.randn(16).astype(np.float32)
    nodes = [
        helper.make_node("Concat", ["in_a", "in_b"], ["cat"], name="cat0", axis=2),
        helper.make_node("Conv", ["cat", "W", "B"], ["out0"],
                         name="conv0", kernel_shape=[3, 1], strides=[1, 1],
                         pads=[0, 0, 0, 0]),
    ]
    graph = helper.make_graph(
        nodes, "g",
        [
            helper.make_tensor_value_info("in_a", TensorProto.FLOAT, [3, 32, 2, 1]),
            helper.make_tensor_value_info("in_b", TensorProto.FLOAT, [3, 32, 1, 1]),
        ],
        [helper.make_tensor_value_info("out0", TensorProto.FLOAT, [3, 16, 1, 1])],
        initializer=[numpy_helper.from_array(w, "W"), numpy_helper.from_array(b, "B")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    p = os.path.join(WORK, "mini4d.onnx")
    onnx.save(model, p)
    return p


def main():
    p = build()
    print("built", p)
    r = subprocess.run(
        ["pnnx", os.path.basename(p), "inputshape=[3,32,2,1],[3,32,1,1]"],
        cwd=WORK, capture_output=True, text=True, errors="replace")
    print(r.stdout[-1200:])
    ncnn_py = os.path.join(WORK, "mini4d_ncnn.py")
    if os.path.exists(ncnn_py):
        print("=== _ncnn.py feeding lines ===")
        for line in open(ncnn_py, encoding="utf-8", errors="replace"):
            if "ex.input" in line or "ncnn.Mat" in line:
                print("  " + line.strip())
    param = os.path.join(WORK, "mini4d.ncnn.param")
    if os.path.exists(param):
        print("=== ncnn param ===")
        for line in open(param, encoding="utf-8", errors="replace"):
            print("  " + line.rstrip())


if __name__ == "__main__":
    main()
