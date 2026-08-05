#!/usr/bin/env python3
"""Bisect the original model by topological prefix to find the exact node whose
DML kernel creation fails (MLOperatorAuthorImpl.cpp:2410, E_INVALIDARG).

Runs the unified_bench DML backend on prefix sub-models and narrows down the
failing node count.
"""
import os
import subprocess
import sys
from collections import defaultdict, deque

import numpy as np
import onnx
from onnx import helper, numpy_helper
from onnx.shape_inference import infer_shapes

MODEL = r"d:\WorkSpace\MSS\models\online_scnet_epoch_43_val_loss_0_67697_val_sdr_2_527_stream_1frame.onnx"
WORK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dml_op_tests")
EXE = r"d:\WorkSpace\unified_bench\build\win-x86\Release\unified_bench.exe"
os.makedirs(WORK, exist_ok=True)


def topo_sort(nodes):
    name_to_prod = {}
    for i, n in enumerate(nodes):
        for o in n.output:
            if o:
                name_to_prod[o] = i
    indeg = [0] * len(nodes)
    edges = defaultdict(list)
    for i, n in enumerate(nodes):
        for inp in n.input:
            if inp and inp in name_to_prod:
                edges[name_to_prod[inp]].append(i)
                indeg[i] += 1
    q = deque(i for i in range(len(nodes)) if indeg[i] == 0)
    order = []
    while q:
        u = q.popleft()
        order.append(u)
        for v in edges[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    if len(order) != len(nodes):
        raise RuntimeError("cycle detected")
    return [nodes[i] for i in order]


def build_prefix(orig, order, k):
    kept = order[:k]
    kept_set = set(id(n) for n in kept)
    produced = {}
    for n in kept:
        for o in n.output:
            if o:
                produced[o] = n
    # boundary outputs: original graph outputs produced by kept nodes,
    # plus tensors produced by kept nodes and consumed by dropped nodes
    orig_outputs = {o.name for o in orig.graph.output}
    dropped_consumes = set()
    for n in order[k:]:
        dropped_consumes.update(i for i in n.input if i)
    out_names = set()
    for o in produced:
        if o in orig_outputs or o in dropped_consumes:
            out_names.add(o)
    if not out_names:
        # fall back: last produced tensors
        for o in reversed(list(produced.keys())):
            out_names.add(o)
            if len(out_names) >= 8:
                break
    # value_info from shape inference provides types for all tensors
    vi_map = {vi.name: vi for vi in orig.graph.value_info}
    for vi in orig.graph.input:
        vi_map[vi.name] = vi
    for vi in orig.graph.output:
        vi_map[vi.name] = vi

    outputs = []
    for name in sorted(out_names):
        vi = vi_map.get(name)
        if vi is None:
            # try to infer from producer output type
            prod = produced.get(name)
            continue
        outputs.append(helper.make_tensor_value_info(name, vi.type.tensor_type.elem_type,
                                                     [d.dim_value if d.HasField("dim_value") else -1
                                                      for d in vi.type.tensor_type.shape.dim]))
    graph = helper.make_graph(
        list(kept), "bisect",
        list(orig.graph.input),
        outputs,
        initializer=list(orig.graph.initializer),
    )
    m = helper.make_model(graph, opset_imports=orig.opset_import,
                          ir_version=orig.ir_version)
    if orig.ir_version > 10:
        m.ir_version = 8
    return m


def run_prefix(order, k, tag):
    m = build_prefix(orig, order, k)
    path = os.path.join(WORK, f"bisect_{tag}.onnx")
    onnx.save(m, path)
    r = subprocess.run([EXE, path, "--repeat", "1", "--backend", "ONNX_DML_GPU"],
                       capture_output=True, text=True, errors="replace")
    log = r.stdout + r.stderr
    ok = ("Record added: ONNX_DML_GPU" in log)
    return ok, log


if __name__ == "__main__":
    orig = onnx.load(MODEL, load_external_data=False)
    # external data may be referenced; load external too
    orig = onnx.load(MODEL)
    try:
        orig = infer_shapes(orig)
    except Exception as e:
        print("shape inference warning:", e)
    order = topo_sort(list(orig.graph.node))
    N = len(order)
    print("total nodes:", N)

    # quick sanity: full model should fail
    ok, log = run_prefix(order, N, "full")
    print(f"full({N}): ok={ok}")

    lo, hi = 1, N  # lo = works, hi = fails (prefix sizes)
    # find smallest k where prefix fails
    lo_ok = None
    # binary search
    l, r = 1, N
    while l < r:
        mid = (l + r) // 2
        ok, _ = run_prefix(order, mid, f"m{mid}")
        print(f"prefix[{mid}]: ok={ok}")
        if ok:
            l = mid + 1
        else:
            r = mid
    fail_k = l
    print("smallest failing prefix size:", fail_k)
    failing_node = order[fail_k - 1]
    print("FAILING NODE:", failing_node.name, failing_node.op_type, "domain:", failing_node.domain)
    print("  inputs:", list(failing_node.input))
    print("  outputs:", list(failing_node.output))
    for a in failing_node.attribute:
        print("  attr:", a.name, helper.get_attribute_value(a))
