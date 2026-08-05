"""Inspect blob sizes around conv1d_0 (blob 85/86) under BOTH reversed and
direct 3D feeding, to determine why reversed feeding returns -100 and whether
the pnnx-generated graph layout matches ONNX semantics.

ONNX semantics (from model report): 3D inputs like in1=[N=472, C=32, L=2] are
[N, C, L] (472 independent 1D signals). ncnn Convolution1D convolves over w with
h=channels, so the correct ncnn layout should be Mat(w=L, h=C, c=N) = reversed.
"""
import sys

import ncnn
import numpy as np

PARAM = r"D:\WorkSpace\MSS\models\online_scnet_tfc_tdf.ncnn.param"
BIN = r"D:\WorkSpace\MSS\models\online_scnet_tfc_tdf.ncnn.bin"
SHAPES = r"D:\WorkSpace\MSS\models\online_scnet_tfc_tdf.shapes"


def read_shapes():
    ins = []
    for line in open(SHAPES, encoding="utf-8", errors="replace"):
        line = line.strip()
        if "=" not in line or line.startswith("inputs=") or line.startswith("outputs="):
            continue
        name, dims = line.split("=", 1)
        if name.startswith("in"):
            ins.append((name, [int(x) for x in dims.split(",") if x != ""]))
    return ins


def probe(mapping, blobs, keep_net=False):
    net = ncnn.Net()
    net.opt.use_vulkan_compute = False
    net.opt.num_threads = 4
    net.opt.use_packing_layout = False
    net.opt.use_winograd_convolution = True
    net.opt.use_sgemm_convolution = True
    net.load_param(PARAM)
    net.load_model(BIN)
    ex = net.create_extractor()
    for name, sh in read_shapes():
        arr = np.zeros(sh, dtype=np.float32)
        if mapping.endswith("_bd") and len(sh) >= 2 and sh[0] == 1:
            arr = arr.reshape(sh[1:])
        arr = np.ascontiguousarray(arr)
        d = arr.shape
        if mapping.startswith("rev"):
            m = ncnn.Mat(arr)
        else:  # direct
            if len(d) == 3:
                m = ncnn.Mat(d[0], d[1], d[2])
            elif len(d) == 4:
                m = ncnn.Mat(d[3], d[1], d[2], d[0])
            elif len(d) == 2:
                m = ncnn.Mat(d[0], d[1])
            else:
                m = ncnn.Mat(d[0])
        r = ex.input(name, m.clone())
        if r != 0:
            print("  [%s] input %s ret=%d" % (mapping, name, r))
            return None
    for b in blobs:
        o = ncnn.Mat()
        r = ex.extract(b, o)
        if r != 0 or o.empty():
            print("  [%s] blob %s ret=%d EMPTY" % (mapping, b, r))
            return None
        print("  [%s] blob %s OK dims=%d  %dx%dx%dx%d" % (mapping, b, o.dims, o.w, o.h, o.d, o.c))
    return None


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "85,86"
    blobs = [x.strip() for x in target.split(",")]
    print("== reversed (standard pnnx script style) feeding ==")
    probe("rev_bd", blobs)
    print("== direct (w=s0,h=s1,c=s2) feeding ==")
    probe("dir", blobs)
