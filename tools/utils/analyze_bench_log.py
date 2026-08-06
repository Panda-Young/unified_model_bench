#!/usr/bin/env python3
"""Parse unified_bench debug log and report per-(variant, backend) run stats.

Sections:  "=== <variant> ... ==="  then  "--- <backend> ---"
Run lines: "<Backend>: run <n> took <ms> ms"
Outputs count/avg/min/max/p50/p90/p99, per-100-run bucket means and fast-frame
(<8ms) groups to reveal stair-step (power corner) patterns.
"""
import re
import sys
from collections import OrderedDict

RUN_RE = re.compile(r"(ONNX|QNN|MNN|NCNN|TFLite|LiteRT): run (\d+) took ([\d.]+) ms")
VAR_RE = re.compile(r"=== (.+?) ===")
BENCH_RE = re.compile(r"--- (.+?) ---")


def main(path):
    cur_var = "?"
    cur_bench = "?"
    buckets = OrderedDict()  # (var, bench) -> list of ms

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = VAR_RE.search(line)
            if m:
                cur_var = m.group(1)
                continue
            m = BENCH_RE.search(line)
            if m:
                cur_bench = m.group(1)
                continue
            m = RUN_RE.search(line)
            if m:
                key = (cur_var, cur_bench)
                buckets.setdefault(key, []).append(float(m.group(3)))

    if not buckets:
        print("no run lines found")
        return 1

    for (var, bench), vals in buckets.items():
        n = len(vals)
        vals_sorted = sorted(vals)
        avg = sum(vals) / n
        p50 = vals_sorted[int(n * 0.50)]
        p90 = vals_sorted[int(n * 0.90)]
        p99 = vals_sorted[int(n * 0.99)]
        print("=" * 72)
        print("variant : %s" % var)
        print("backend : %s" % bench)
        print("runs    : %d" % n)
        print("avg=%.3f  min=%.3f  max=%.3f  p50=%.3f  p90=%.3f  p99=%.3f"
              % (avg, min(vals), max(vals), p50, p90, p99))
        bsize = max(1, n // 10)
        step = "  "
        for i in range(0, n, bsize):
            seg = vals[i:i + bsize]
            if not seg:
                break
            step += "r%4d-%4d:%.2f | " % (i, i + len(seg) - 1, sum(seg) / len(seg))
        print("buckets : %s" % step)
        fast = [i for i, v in enumerate(vals) if v < 8.0]
        if fast:
            groups = []
            start = prev = fast[0]
            for i in fast[1:]:
                if i == prev + 1:
                    prev = i
                else:
                    groups.append((start, prev))
                    start = prev = i
            groups.append((start, prev))
            print("fast(<8ms) : %d/%d  groups=%s" % (len(fast), n, groups))
        else:
            print("fast(<8ms) : 0/%d" % n)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "qnn_sdk_htp.log"))
