"""Plan-C step 1: inspect Convolution1D weight params for the layers around
blob 86 (conv1d_0) and blob 1190 (conv_102) in the NEW pnnx param, and compare
the weight in-channels against the actual blob channel dims observed at
runtime under reversed vs direct feeding.

v1 param line format:
    type name nb nt b0.. bNb-1 t0.. tNt-1  0=num_output 1=kernel_w 6=weight_data_size
For Convolution1D weight_data_size = num_output * kernel_w * in_channels.
"""
import re

PARAM = r"D:\WorkSpace\MSS\models\online_scnet_tfc_tdf.ncnn.param"


def find_layers(needles):
    out = []
    for line in open(PARAM, encoding="utf-8", errors="replace"):
        t = line.split()
        if len(t) < 4:
            continue
        typ, name = t[0], t[1]
        if name in needles:
            out.append((typ, name, t[4:]))
    return out


def main():
    for typ, name, rest in find_layers(
            {"conv1d_0", "conv1d_1", "conv_102", "conv_101"}):
        # parse key=value params
        params = {}
        for tok in rest:
            m = re.match(r"(\d+)=([-\d]+)", tok)
            if m:
                params[int(m.group(1))] = int(m.group(2))
        num_out = params.get(0, -1)
        kernel = params.get(1, -1)
        wsize = params.get(6, -1)
        if num_out > 0 and kernel > 0 and wsize > 0:
            in_ch = wsize // (num_out * kernel)
            print("%s %-10s num_output=%d kernel_w=%d weight_size=%d in_channels=%d"
                  % (typ, name, num_out, kernel, wsize, in_ch))
        else:
            print("%s %-10s params=%s" % (typ, name, params))


if __name__ == "__main__":
    main()
