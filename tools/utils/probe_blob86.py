"""Find which layer produces blob 85/86 in the new (pnnx 20260526) ncnn param,
and test alternative input mappings on the new model."""
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


def layer_of_blob(target):
    out = []
    for line in open(PARAM, encoding="utf-8", errors="replace"):
        t = line.split()
        if len(t) < 4:
            continue
        typ, name = t[0], t[1]
        nb, nk = int(t[2]), int(t[3])
        tops = t[4 + nb: 4 + nb + nk]
        if target in tops:
            out.append("%s %s bottom=[%s] top=[%s]" % (typ, name, ",".join(t[4:4 + nb]), ",".join(tops)))
    return out


def race(mapping, probe_blob):
    """Feed all inputs with mapping, then extract probe_blob.
    mapping: 'rev_bd' reversed+batch-drop, 'dir_nbd' direct no batch-drop, 'mix'."""
    net = ncnn.Net()
    net.opt.use_vulkan_compute = False
    net.opt.num_threads = 4
    net.opt.use_packing_layout = False
    net.opt.use_winograd_convolution = True
    net.opt.use_sgemm_convolution = True
    lr = net.load_param(PARAM)
    mr = net.load_model(BIN)
    print("  load_param=%d load_model=%d" % (lr, mr))
    ex = net.create_extractor()
    for name, sh in read_shapes():
        arr = np.zeros(sh, dtype=np.float32)
        if mapping.endswith("_bd") and len(sh) >= 2 and sh[0] == 1:
            arr = arr.reshape(sh[1:])
        arr = np.ascontiguousarray(arr)
        d = arr.shape
        if mapping == "rev_bd":
            m = ncnn.Mat(arr)
        elif mapping == "dir_nbd":
            if len(d) == 3:
                m = ncnn.Mat(d[0], d[1], d[2])
            elif len(d) == 4:
                m = ncnn.Mat(d[0], d[1], d[2], d[3])
            elif len(d) == 2:
                m = ncnn.Mat(d[0], d[1])
            else:
                m = ncnn.Mat(d[0])
        else:  # mix
            if len(d) == 4:
                m = ncnn.Mat(d[3], d[1], d[2], d[0])
            elif len(d) == 3:
                m = ncnn.Mat(d[0], d[1], d[2])
            elif len(d) == 2:
                m = ncnn.Mat(d[0], d[1])
            else:
                m = ncnn.Mat(d[0])
        r = ex.input(name, m.clone())
        if r != 0:
            return "input %s ret=%d" % (name, r)
    o = ncnn.Mat()
    r = ex.extract(probe_blob, o)
    if r != 0 or o.empty():
        return "blob %s ret=%d" % (probe_blob, r)
    return "blob %s OK dims=%d %dx%dx%dx%d" % (probe_blob, o.dims, o.w, o.h, o.d, o.c)


if __name__ == "__main__":
    print("== layer producing blob 86 ==")
    for s in layer_of_blob("86"):
        print("  " + s)
    print("== layer producing blob 85 ==")
    for s in layer_of_blob("85"):
        print("  " + s)
    print("== race rev_bd -> blob 86 ==")
    print("  " + race("rev_bd", "86"))
    print("== race dir_nbd -> blob 86 ==")
    print("  " + race("dir_nbd", "86"))
    print("== race mix -> blob 86 ==")
    print("  " + race("mix", "86"))
    print("== race dir_nbd -> blob 1195 ==")
    print("  " + race("dir_nbd", "1195"))
    print("== race dir_nbd -> out0 ==")
    print("  " + race("dir_nbd", "out0"))
