"""Locate the layer producing blob 1195 (and nearby blobs) in the NEW pnnx
(20260526) v1-format ncnn param, and list which inputs feed the subgraph around
it. Used to explain the hard crash when feeding 3D inputs in direct layout."""
import re

PARAM = r"D:\WorkSpace\MSS\models\online_scnet_tfc_tdf.ncnn.param"
SHAPES = r"D:\WorkSpace\MSS\models\online_scnet_tfc_tdf.shapes"


def parse_layers():
    layers = []
    for line in open(PARAM, encoding="utf-8", errors="replace"):
        t = line.split()
        if len(t) < 4:
            continue
        typ, name = t[0], t[1]
        nb, nk = int(t[2]), int(t[3])
        bottoms = t[4:4 + nb]
        tops = t[4 + nb:4 + nb + nk]
        layers.append((typ, name, bottoms, tops))
    return layers


def layer_for_top(target, layers):
    out = []
    for typ, name, bottoms, tops in layers:
        if target in tops:
            out.append((typ, name, bottoms, tops))
    return out


def main():
    layers = parse_layers()
    print("total layers:", len(layers))
    for target in ("1195", "1194", "1193", "1192", "1191", "1190"):
        hits = layer_for_top(target, layers)
        for typ, name, bottoms, tops in hits:
            print("top %s <- %s %s bottom=[%s] top=[%s]" % (target, typ, name,
                                                             ",".join(bottoms), ",".join(tops)))
    # What consumes 1195?
    print("== consumers of 1195 ==")
    for typ, name, bottoms, tops in layers:
        if "1195" in bottoms:
            print("  %s %s bottom=[%s] top=[%s]" % (typ, name, ",".join(bottoms), ",".join(tops)))
    # Which layer types appear near the end (last 40 layers)?
    print("== last 40 layers (type name) ==")
    for typ, name, bottoms, tops in layers[-40:]:
        print("  %-16s %s" % (typ, name))


if __name__ == "__main__":
    main()
