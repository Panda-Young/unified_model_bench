"""Minimal experiment: confirm pnnx's 3D-input layout convention for Conv1d.

Build a tiny ONNX with a single 3D input [N,C,L] = [3,32,5] and one Conv1d
(weight [16,32,3]) -> output [3,16,3]. Run pnnx and inspect the generated
ncnn param to see what layout the converted graph expects (w=dim0 or w=dim2).
"""
import os
import subprocess
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

WORK = r"D:\WorkSpace\unified_bench\tools\_pnnx3d"


def build():
    os.makedirs(WORK, exist_ok=True)
    w = np.random.randn(16, 32, 3).astype(np.float32)
    b = np.random.randn(16).astype(np.float32)
    nodes = [
        helper.make_node("Conv", ["in0", "W", "B"], ["out0"],
                         name="conv1d", kernel_shape=[3], strides=[1], pads=[0, 0]),
    ]
    graph = helper.make_graph(
        nodes, "g",
        [helper.make_tensor_value_info("in0", TensorProto.FLOAT, [3, 32, 5])],
        [helper.make_tensor_value_info("out0", TensorProto.FLOAT, [3, 16, 3])],
        initializer=[numpy_helper.from_array(w, "W"), numpy_helper.from_array(b, "B")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    p = os.path.join(WORK, "mini3d.onnx")
    onnx.save(model, p)
    return p


def main():
    p = build()
    print("built", p)
    r = subprocess.run(
        ["pnnx", os.path.basename(p), "inputshape=[3,32,5]"],
        cwd=WORK, capture_output=True, text=True)
    print(r.stdout[-1500:])
    if r.stderr:
        print("STDERR:", r.stderr[-800:])
    ncnn_py = os.path.join(WORK, "mini3d_ncnn.py")
    if os.path.exists(ncnn_py):
        print("=== _ncnn.py feeding lines ===")
        for line in open(ncnn_py):
            if "ex.input" in line or "ncnn.Mat" in line:
                print("  " + line.strip())
    param = os.path.join(WORK, "mini3d.ncnn.param")
    if os.path.exists(param):
        print("=== ncnn param (first 8 lines) ===")
        for line in open(param):
            print("  " + line.rstrip())
            if line.startswith("Convolution"):
                break


if __name__ == "__main__":
    main()
