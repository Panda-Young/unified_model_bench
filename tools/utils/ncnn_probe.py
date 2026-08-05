"""Prove the validity of online_scnet_tfc_tdf.ncnn.{param,bin} using the
ncnn python binding, and find the 3D-input Mat mapping that makes the full
graph forward (extract out0) succeed. The 4D mapping is already confirmed:
ONNX [a,b,c,d] -> ncnn Mat(w=d, h=b, d=c, c=a)  (w,h,d,c).
"""
import sys

import ncnn

PARAM = r"D:\WorkSpace\MSS\models\online_scnet_tfc_tdf.ncnn.param"
BIN = r"D:\WorkSpace\MSS\models\online_scnet_tfc_tdf.ncnn.bin"
SHAPES = r"D:\WorkSpace\MSS\models\online_scnet_tfc_tdf.shapes"

# 3D mapping candidates: how [a,b,c] maps to Mat(w,h,c)
MODES = {
    0: lambda s: (s[2], s[1], s[0]),  # reversed
    1: lambda s: (s[2], s[0], s[1]),
    2: lambda s: (s[1], s[2], s[0]),
    3: lambda s: (s[0], s[2], s[1]),
}


def read_shapes():
    in_names, in_shapes, out_names = [], [], []
    for line in open(SHAPES, encoding="utf-8", errors="replace"):
        line = line.strip()
        if line.startswith("inputs=") or line.startswith("outputs="):
            continue
        if "=" not in line:
            continue
        name, dims = line.split("=", 1)
        shape = [int(x) for x in dims.split(",") if x != ""]
        if name.startswith("in"):
            in_names.append(name)
            in_shapes.append(shape)
        elif name.startswith("out"):
            out_names.append(name)
    return in_names, in_shapes, out_names


def run(mode3, in_names, in_shapes, want_out="out0"):
    import numpy as np
    net = ncnn.Net()
    net.opt.use_vulkan_compute = False
    net.opt.num_threads = 4
    net.opt.use_packing_layout = False
    net.opt.use_winograd_convolution = True
    net.opt.use_sgemm_convolution = True
    lr = net.load_param(PARAM)
    mr = net.load_model(BIN)
    if lr != 0 or mr != 0:
        print(f"  load_param={lr} load_model={mr}  <<< MODEL FILE INVALID")
        return False
    print(f"  load_param={lr} load_model={mr}  (model files load OK)")

    ex = net.create_extractor()
    for i, sh in enumerate(in_shapes):
        # mimic onnx_convert.verify_ncnn intent: drop the leading batch dim
        # when it is 1, then ncnn.Mat(numpy) reverses [d0,d1,d2] -> (w,h,c)
        arr = np.zeros(sh, dtype=np.float32)
        if len(sh) >= 2 and sh[0] == 1:
            arr = arr.reshape(sh[1:])
        arr = np.ascontiguousarray(arr)
        m = ncnn.Mat(arr).clone()
        iret = ex.input(in_names[i], m)
        if iret != 0:
            print(f"  input {in_names[i]} ret={iret}")
            return False

    out = ncnn.Mat()
    oret = ex.extract(want_out, out)
    if oret != 0:
        print(f"  extract {want_out} ret={oret}  (empty)  -> forward FAILED")
        return False
    print(f"  extract {want_out} OK  dims={out.dims} {out.w}x{out.h}x{out.d}x{out.c}")
    return True


def walk(in_names, in_shapes, blobs):
    """Feed all inputs, then extract each blob in order; stop at first empty."""
    import numpy as np
    net = ncnn.Net()
    net.opt.use_vulkan_compute = False
    net.opt.num_threads = 4
    net.opt.use_packing_layout = False
    net.opt.use_winograd_convolution = True
    net.opt.use_sgemm_convolution = True
    net.load_param(PARAM)
    net.load_model(BIN)
    ex = net.create_extractor()
    for i, sh in enumerate(in_shapes):
        arr = np.zeros(sh, dtype=np.float32)
        if len(sh) >= 2 and sh[0] == 1:
            arr = arr.reshape(sh[1:])
        m = ncnn.Mat(np.ascontiguousarray(arr)).clone()
        r = ex.input(in_names[i], m)
        if r != 0:
            print(f"input {in_names[i]} ret={r}")
            return
    for b in blobs:
        o = ncnn.Mat()
        r = ex.extract(b, o)
        if r != 0 or o.empty():
            print(f"  blob {b}: ret={r} EMPTY  <-- first failure")
            return
        print(f"  blob {b}: OK dims={o.dims} {o.w}x{o.h}x{o.d}x{o.c}")


def race(mapping, chain_blobs):
    """Feed all inputs with the given mapping, then walk blobs in order.
    mapping: 'rev_bd' reversed+batch-drop, 'dir_bd' direct+batch-drop,
             'rev_nbd' reversed no batch-drop, 'dir_nbd' direct no batch-drop.
    Returns the first failing blob (or 'ALL OK')."""
    import numpy as np
    net = ncnn.Net()
    net.opt.use_vulkan_compute = False
    net.opt.num_threads = 4
    net.opt.use_packing_layout = False
    net.opt.use_winograd_convolution = True
    net.opt.use_sgemm_convolution = True
    net.load_param(PARAM)
    net.load_model(BIN)
    ex = net.create_extractor()
    in_names, in_shapes, _ = read_shapes()
    for i, sh in enumerate(in_shapes):
        arr = np.zeros(sh, dtype=np.float32)
        if mapping.endswith("_bd") and len(sh) >= 2 and sh[0] == 1:
            arr = arr.reshape(sh[1:])
        arr = np.ascontiguousarray(arr)
        # ncnn.Mat(np) always reverses; if direct mapping requested, build manually
        if mapping == "batch":
            # new-pnnx convention: ncnn.Mat(arr, batch_index=0) for 4D inputs
            # (dim0 is the batch), plain Mat for 3D inputs.
            d = arr.shape
            if len(d) >= 2 and d[0] == 1:
                m = ncnn.Mat(np.ascontiguousarray(arr), batch_index=0)
            else:
                m = ncnn.Mat(np.ascontiguousarray(arr))
        elif mapping == "mix":
            d = arr.shape
            # 4D: (w=s3,h=s1,d=s2,c=s0) - empirically confirmed for in0 slice;
            # 3D: direct (w=s0,h=s1,c=s2) - required by Convolution1D (h=channels)
            if len(d) == 4:
                m = ncnn.Mat(d[3], d[1], d[2], d[0])
            elif len(d) == 3:
                m = ncnn.Mat(d[0], d[1], d[2])
            elif len(d) == 2:
                m = ncnn.Mat(d[0], d[1])
            else:
                m = ncnn.Mat(d[0])
        elif mapping.startswith("rev"):
            m = ncnn.Mat(arr)
        else:
            d = arr.shape
            if len(d) == 3:
                m = ncnn.Mat(d[0], d[1], d[2])
            elif len(d) == 4:
                m = ncnn.Mat(d[0], d[1], d[2], d[3])
            else:
                m = ncnn.Mat(d[0]) if len(d) == 1 else ncnn.Mat(d[0], d[1])
        r = ex.input(in_names[i], m.clone())
        if r != 0:
            return f"input {in_names[i]} ret={r}"
    for b in chain_blobs:
        o = ncnn.Mat()
        r = ex.extract(b, o)
        if r != 0 or o.empty():
            return f"blob {b} ret={r}"
    return "ALL OK"


def main():
    in_names, in_shapes, out_names = read_shapes()
    print(f"shapes: {len(in_shapes)} in, {len(out_names)} out")
    if len(sys.argv) > 1 and sys.argv[1] == "walk":
        blobs = sys.argv[2].split(",") if len(sys.argv) > 2 else []
        walk(in_names, in_shapes, blobs)
        return
    if len(sys.argv) > 1 and sys.argv[1] == "out0":
        # just full forward with default mode
        for mode in (0, 1, 2, 3):
            print(f"--- 3D mode {mode} ---")
            run(mode, in_names, in_shapes)
        return
    # probe an intermediate blob (e.g. 70 = first slice output)
    mode = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    blob = sys.argv[2] if len(sys.argv) > 2 else "out0"
    print(f"--- 3D mode {mode}, probe blob {blob} ---")
    run(mode, in_names, in_shapes, blob)


if __name__ == "__main__":
    main()
