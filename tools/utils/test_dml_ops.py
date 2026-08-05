#!/usr/bin/env python3
"""Build minimal single-operator ONNX models to isolate which op the DML EP
rejects at kernel creation (MLOperatorAuthorImpl.cpp:2410, E_INVALIDARG).

Note: ir_version is pinned to 8 because the onnx 1.22 default (13) is rejected
by ORT 1.22 (max supported IR version is 10).
"""
import os
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dml_op_tests")
os.makedirs(OUT, exist_ok=True)

OPSET = helper.make_opsetid("", 18)


def save(graph, inputs, outputs, path, initializer=None):
    m = helper.make_model(
        helper.make_graph(graph, "g", inputs, outputs, initializer=initializer or []),
        opset_imports=[OPSET],
    )
    m.ir_version = 8
    onnx.save(m, path)
    print("saved", path)


def make_lstm(path, bidirectional=False):
    seq, batch, inp, hidden = 108, 1, 128, 128
    num_dir = 2 if bidirectional else 1
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [seq, batch, inp])
    W = helper.make_tensor_value_info("W", TensorProto.FLOAT, [num_dir, 4 * hidden, inp])
    R = helper.make_tensor_value_info("R", TensorProto.FLOAT, [num_dir, 4 * hidden, hidden])
    B = helper.make_tensor_value_info("B", TensorProto.FLOAT, [num_dir, 8 * hidden])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [seq, num_dir, batch, hidden])
    node = helper.make_node(
        "LSTM", ["X", "W", "R", "B"], ["Y"],
        hidden_size=hidden,
        direction="bidirectional" if bidirectional else "forward",
        input_forget=0, layout=0,
    )
    save([node], [X, W, R, B], [Y], path, initializer=[
        numpy_helper.from_array(np.zeros((num_dir, 4 * hidden, inp), np.float32), "W"),
        numpy_helper.from_array(np.zeros((num_dir, 4 * hidden, hidden), np.float32), "R"),
        numpy_helper.from_array(np.zeros((num_dir, 8 * hidden), np.float32), "B"),
    ])


def make_layernorm(path):
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [108, 128])
    S = helper.make_tensor_value_info("S", TensorProto.FLOAT, [128])
    Bv = helper.make_tensor_value_info("B", TensorProto.FLOAT, [128])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [108, 128])
    node = helper.make_node("LayerNormalization", ["X", "S", "Bv"], ["Y"], axis=-1, epsilon=1e-5, stash_type=1)
    save([node], [X, S, Bv], [Y], path, initializer=[
        numpy_helper.from_array(np.ones(128, np.float32), "S"),
        numpy_helper.from_array(np.zeros(128, np.float32), "Bv"),
    ])


def make_slice(path):
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4, 2049, 1])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4, 2049, 1])
    node = helper.make_node("Slice", ["X", "starts", "ends", "axes", "steps"], ["Y"])
    save([node], [X], [Y], path, initializer=[
        numpy_helper.from_array(np.array([0], np.int64), "starts"),
        numpy_helper.from_array(np.array([2049], np.int64), "ends"),
        numpy_helper.from_array(np.array([2], np.int64), "axes"),
        numpy_helper.from_array(np.array([1], np.int64), "steps"),
    ])


def make_slice_neg(path):
    # Real model config: starts=[-2], ends=[INT64_MAX], axes=[2], steps=[1]
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4, 2049, 1])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4, 2, 1])
    node = helper.make_node("Slice", ["X", "starts", "ends", "axes", "steps"], ["Y"])
    save([node], [X], [Y], path, initializer=[
        numpy_helper.from_array(np.array([-2], np.int64), "starts"),
        numpy_helper.from_array(np.array([9223372036854775807], np.int64), "ends"),
        numpy_helper.from_array(np.array([2], np.int64), "axes"),
        numpy_helper.from_array(np.array([1], np.int64), "steps"),
    ])


def make_split(path):
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4, 2049, 1])
    Y1 = helper.make_tensor_value_info("Y1", TensorProto.FLOAT, [1, 4, 683, 1])
    Y2 = helper.make_tensor_value_info("Y2", TensorProto.FLOAT, [1, 4, 683, 1])
    Y3 = helper.make_tensor_value_info("Y3", TensorProto.FLOAT, [1, 4, 683, 1])
    node = helper.make_node("Split", ["X", "split"], ["Y1", "Y2", "Y3"], axis=2)
    save([node], [X], [Y1, Y2, Y3], path, initializer=[
        numpy_helper.from_array(np.array([683, 683, 683], np.int64), "split"),
    ])


def make_tile(path):
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4, 2, 1])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4, 4, 1])
    node = helper.make_node("Tile", ["X", "repeats"], ["Y"])
    save([node], [X], [Y], path, initializer=[
        numpy_helper.from_array(np.array([1, 1, 2, 1], np.int64), "repeats"),
    ])


def make_expand(path):
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4, 2, 1])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4, 2, 1])
    node = helper.make_node("Expand", ["X", "shape"], ["Y"])
    save([node], [X], [Y], path, initializer=[
        numpy_helper.from_array(np.array([1, 4, 2, 1], np.int64), "shape"),
    ])


def make_erf(path):
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4, 2049, 1])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4, 2049, 1])
    node = helper.make_node("Erf", ["X"], ["Y"])
    save([node], [X], [Y], path)


def make_conv(path):
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 32, 765, 2])
    W = helper.make_tensor_value_info("W", TensorProto.FLOAT, [64, 32, 5, 1])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 64, 765, 2])
    node = helper.make_node("Conv", ["X", "W"], ["Y"], pads=[2, 0, 2, 0])
    save([node], [X, W], [Y], path, initializer=[
        numpy_helper.from_array(np.zeros((64, 32, 5, 1), np.float32), "W"),
    ])


if __name__ == "__main__":
    make_lstm(os.path.join(OUT, "lstm_fwd.onnx"), False)
    make_lstm(os.path.join(OUT, "lstm_bi.onnx"), True)
    make_layernorm(os.path.join(OUT, "layernorm.onnx"))
    make_slice(os.path.join(OUT, "slice.onnx"))
    make_slice_neg(os.path.join(OUT, "slice_neg.onnx"))
    make_split(os.path.join(OUT, "split.onnx"))
    make_tile(os.path.join(OUT, "tile.onnx"))
    make_expand(os.path.join(OUT, "expand.onnx"))
    make_erf(os.path.join(OUT, "erf.onnx"))
    make_conv(os.path.join(OUT, "conv.onnx"))
