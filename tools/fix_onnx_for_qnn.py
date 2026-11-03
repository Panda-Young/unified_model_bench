#!/usr/bin/env python3
"""
fix_onnx_for_qnn.py — make a streaming/TFC-TDF ONNX model fully QNN-partitionable
by folding all-zero "Pad" nodes into Identity.

WHY
  ONNX Runtime's QNN EP (HTP) splits the graph into partitions and only keeps
  nodes it can compile on QNN. In models produced by the scnet (TFC-TDF)
  exporter, a few Pad nodes have *dynamically computed* pads. ORT constant-
  folds them to all-zero pads, but then keeps a cross-partition Pad node
  instead of folding it to Identity. With the benchmark's session config
  `session.disable_cpu_ep_fallback=1`, that leftover Pad left on CPU fails
  CreateSession with:

      "This session contains graph nodes that are assigned to the default
       CPU EP, but fallback to CPU EP has been explicitly disabled by the
       user."

  (ONNX_QNN_GPU is unaffected because that path does NOT set the
  disable_cpu_ep_fallback flag — it silently mixes; ONNX_QNN_HTP does set it.)

  Folding an all-zero Pad -> Identity is mathematically exact (zero padding is
  a no-op) and removes the blocker so the whole graph can land on the HTP EP.

IMPORTANT
  We must NOT ship an ORT-optimized model file: ORT_ENABLE_ALL bakes in
  desktop/CPU execution-provider operators (com.microsoft.nchwc:Conv,
  ReorderInput/ReorderOutput, ...) that the *device* ORT build cannot load
  ("com.microsoft.nchwc:Conv(-1) is not a registered function/op").

  So this tool only USES ORT optimization to identify which Pad nodes fold to
  all-zero pads (by stable node name), then edits the ORIGINAL source model,
  keeping its original operator set intact.

Usage:
  python tools/fix_onnx_for_qnn.py <model.onnx> [-o out.onnx] [--verify]
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np


def find_zero_pad_names(src: Path, tmp_dir: Path) -> set:
    import onnxruntime as ort
    from onnx import numpy_helper

    opt_path = tmp_dir / (src.stem + "_ort_analysis.onnx")
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.optimized_model_filepath = str(opt_path)
    ort.InferenceSession(str(src), so, providers=["CPUExecutionProvider"])

    import onnx
    m = onnx.load(str(opt_path))
    inits = {i.name: numpy_helper.to_array(i) for i in m.graph.initializer}
    zero_names = set()
    for n in m.graph.node:
        if n.op_type != "Pad":
            continue
        pads_name = n.input[1] if len(n.input) >= 2 and n.input[1] else None
        if pads_name and pads_name in inits:
            p = inits[pads_name]
            if p.size and bool(p.any()) is False:
                zero_names.add(n.name)
    print(f"  ORT analysis: {len(zero_names)} Pad node(s) have all-zero pads"
          f"{' -> ' + repr(sorted(zero_names)) if zero_names else ''}")
    return zero_names


def fold_source_pads(src: Path, zero_names: set):
    import onnx
    from onnx import helper

    m = onnx.load(str(src))
    g = m.graph
    new_nodes = []
    folded = 0
    for n in g.node:
        if n.op_type == "Pad" and n.name in zero_names:
            ident = helper.make_node("Identity", [n.input[0]], list(n.output),
                                     name=f"{n.name}_foldpad")
            new_nodes.append(ident)
            print(f"  fold {n.name}: Pad -> Identity (all-zero pads, no-op)")
            folded += 1
        else:
            new_nodes.append(n)
    if folded:
        g.node.clear()
        g.node.extend(new_nodes)
    return m, folded


def verify(source: Path, fixed: Path, tolerance: float = 1e-4) -> bool:
    import onnxruntime as ort

    s1 = ort.InferenceSession(str(source), providers=["CPUExecutionProvider"])
    s2 = ort.InferenceSession(str(fixed), providers=["CPUExecutionProvider"])

    rng = np.random.default_rng(1234)
    worst = 0.0
    bad = False
    for trial in range(3):
        feed = {}
        for inp in s1.get_inputs():
            shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]
            feed[inp.name] = rng.random(shape).astype(np.float32)
        o1 = s1.run(None, feed)
        o2 = s2.run(None, feed)
        for idx, (a, b) in enumerate(zip(o1, o2)):
            a = np.asarray(a).flatten()
            b = np.asarray(b).flatten()
            if a.shape != b.shape:
                print(f"  trial{trial} out{idx}: SHAPE MISMATCH {a.shape} vs {b.shape}  FAIL")
                bad = True
                continue
            if a.size == 0:
                continue
            worst = max(worst, float(np.abs(a - b).max()))
    ok = (not bad) and (worst < tolerance)
    print(f"  compare: worst max-abs-diff = {worst:.6f}  {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help="input .onnx model")
    ap.add_argument("-o", "--out", default=None, help="output path (default <model>_qnn.onnx)")
    ap.add_argument("--verify", action="store_true", help="CPU-verify equivalence")
    args = ap.parse_args()

    src = Path(args.model)
    if not src.exists():
        print(f"ERROR: model not found: {src}")
        return 1
    out = Path(args.out or f"{src.with_suffix('')}_qnn.onnx")

    with tempfile.TemporaryDirectory(prefix="qnnfix_") as td:
        tmp = Path(td)
        print("== 1/3 Identify all-zero-Pad nodes (ORT constant-fold analysis) ==")
        zeros = find_zero_pad_names(src, tmp)

        print("== 2/3 Fold zero-Pad -> Identity in the ORIGINAL model ==")
        model, folded = fold_source_pads(src, zeros)
        if folded == 0:
            print("  Nothing to fold — model may already be QNN-ready.")
        out.parent.mkdir(parents=True, exist_ok=True)
        import onnx
        onnx.save(model, str(out))
        print(f"  saved {out}")

    if args.verify:
        print("== 3/3 Verify (CPU) ==")
        ok = verify(src, out)
        return 0 if ok else 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())