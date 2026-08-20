#!/usr/bin/env python3
"""Walk intermediate blobs on Vulkan to find the first crashing layer."""
import sys
import numpy as np
import ncnn

PARAM = r"D:\WorkSpace\unified_bench\test_model.ncnn.param"
BIN   = r"D:\WorkSpace\unified_bench\test_model.ncnn.bin"

# blob index -> description (from param dump)
BLOBS = {
    2: "convrelu_0(out)", 3: "splitncnn_0[0]", 4: "splitncnn_0[1]",
    5: "convrelu_1(out)", 6: "convrelu_2(out)", 7: "cat_0(out)",
    8: "add_0(out)", 9: "convrelu_3(out)", 10: "maxpool2d_7(out)",
    11: "convrelu_4(out)", 12: "gap_6(out)", 13: "reshape_14(out)",
    14: "linear_5(out)", 15: "softmax_13(out)=out0",
}

def log(msg):
    print(msg, flush=True)

def main():
    vulkan = "--cpu" not in sys.argv
    net = ncnn.Net()
    net.opt.use_vulkan_compute = vulkan
    net.opt.use_fp16_packed = False
    net.opt.use_fp16_storage = False
    assert net.load_param(PARAM) == 0
    assert net.load_model(BIN) == 0
    log("model loaded")

    ex = net.create_extractor()
    arr_a = np.random.rand(1, 3, 224, 224).astype(np.float32)
    arr_b = np.random.rand(1, 48, 1, 1).astype(np.float32)
    assert ex.input("in0", ncnn.Mat(arr_a)) == 0
    assert ex.input("in1", ncnn.Mat(arr_b)) == 0
    log("inputs fed")

    start = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    for bidx in sorted(BLOBS):
        if bidx < start:
            continue
        m = ncnn.Mat()
        ret = ex.extract(str(bidx), m)
        desc = BLOBS.get(bidx, "?")
        if ret == 0 and not m.empty():
            log(f"blob {bidx} ({desc}): ret={ret} dims={m.dims} "
                f"w={m.w} h={m.h} d={m.d} c={m.c}")
        else:
            log(f"blob {bidx} ({desc}): ret={ret} EMPTY")
    log("ALL BLOBS OK")

if __name__ == "__main__":
    main()
