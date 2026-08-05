"""Plan-C step 2: trace the producer chain of blob 85 (conv1d_0 input) and
blob 83 to find where the channel dim becomes 64. List every layer whose top
blob is in range 70..90 with its type/name/bottoms/tops, so we can correlate
the ncnn graph with the ONNX node list.
"""
PARAM = r"D:\WorkSpace\MSS\models\online_scnet_tfc_tdf.ncnn.param"


def parse():
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


def main():
    layers = parse()
    # build blob -> producer
    producer = {}
    for typ, name, bottoms, tops in layers:
        for tb in tops:
            producer.setdefault(tb, (typ, name, bottoms))
    # print layers in graph order whose tops intersect 70..90 (and named blobs)
    print("== layers with tops in blob range 70..92 ==")
    for typ, name, bottoms, tops in layers:
        if any(b.isdigit() and 70 <= int(b) <= 92 for b in tops):
            print("  %-16s %-16s bottom=[%s] top=[%s]" % (typ, name,
                                                           ",".join(bottoms), ",".join(tops)))
    print("== producers of blob 80..90 ==")
    for b in range(80, 91):
        p = producer.get(str(b))
        print("  blob %s <- %s" % (b, p))


if __name__ == "__main__":
    main()
