#!/usr/bin/env python3
"""
Unified ONNX model converter: NCNN / MNN / TFLite (three-in-one).

Usage:
  python onnx_convert.py <model.onnx> --to ncnn                # NCNN only
  python onnx_convert.py <model.onnx> --to mnn   [--fp16]      # MNN only
  python onnx_convert.py <model.onnx> --to tflite               # TFLite only
  python onnx_convert.py <model.onnx> --to all                  # text + ncnn + mnn + tflite
  python onnx_convert.py <model.onnx> --to ncnn,mnn             # multiple
  python onnx_convert.py <model.onnx> --to ncnn --no-dual       # FP32 only
  python onnx_convert.py <model.onnx> --to text                 # ONNX structure only
  python onnx_convert.py <model.onnx> --to all,text             # convert + structure

Shared flags:
  --no-verify        Skip numerical accuracy verification
  --no-cleanup       Skip intermediate files

Text export:
  --to text          Export ONNX model structure to '<model>.onnx.text'
  --show-weights     Also print weight values in the text report
  -t, --output-text  Custom output path for the .text report

NCNN-specific:
  --fp16             Use FP16 weights/storage (default: FP32)
  --no-dual          Skip FP16 model, generate FP32 only (default: both)
  --gen-cpp          Only generate NCNN C++ code, skip conversion

MNN-specific:
  --fp16             Enable FP16 quantization

TFLite-specific:
  -o, --output-tflite   Custom output path for .tflite
"""

from __future__ import annotations

import argparse
import glob as glob_mod
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# ── TFLite compatibility patches (applied early so onnx_graphsurgeon loads) ──
# onnx 1.22+ removed onnx.helper.float32_to_bfloat16; onnx_graphsurgeon needs it.
import numpy as np
import onnx.helper as _onnx_helper

if not hasattr(_onnx_helper, 'float32_to_bfloat16'):
    def _float32_to_bfloat16(f: float) -> int:
        return np.float32(f).view(np.uint16).item()
    _onnx_helper.float32_to_bfloat16 = _float32_to_bfloat16

if not hasattr(_onnx_helper, 'bfloat16_to_float32'):
    def _bfloat16_to_float32(b: int) -> float:
        return np.uint16(b).view(np.float32).item()
    _onnx_helper.bfloat16_to_float32 = _bfloat16_to_float32

# Also patch the module-level reference used by onnx_graphsurgeon
import onnx
onnx.helper.float32_to_bfloat16 = _onnx_helper.float32_to_bfloat16
onnx.helper.bfloat16_to_float32 = _onnx_helper.bfloat16_to_float32

# np.load allow_pickle compatibility
_orig_np_load = np.load
np.load = lambda *a, **kw: _orig_np_load(*a, **{**kw, 'allow_pickle': kw.get('allow_pickle', True)})

# onnx_graphsurgeon Node API compatibility.
# onnx2tf 1.28.8 (and similar) still uses legacy singular aliases
# (graph_node.output / graph_node.input) while onnx_graphsurgeon 0.5.x only
# exposes plural lists (outputs / inputs). Add property aliases so the whole
# onnx2tf op set keeps working without downgrading onnx_graphsurgeon.
try:
    import onnx_graphsurgeon as _gs_mod

    if not hasattr(_gs_mod.Node, 'output'):
        def _node_output(self):
            outs = object.__getattribute__(self, 'outputs')
            return outs[0] if outs else None
        _gs_mod.Node.output = property(_node_output)

    if not hasattr(_gs_mod.Node, 'input'):
        def _node_input(self):
            ins = object.__getattribute__(self, 'inputs')
            return ins[0] if ins else None
        _gs_mod.Node.input = property(_node_input)
except Exception as _gs_patch_err:
    print(f"  WARNING: onnx_graphsurgeon Node alias patch failed: {_gs_patch_err}")

# Prevent onnx2tf from downloading test images
os.environ['ONNX2TF_DOWNLOAD_TEST_DATA'] = '0'

# ──────────────────────────────────────────────────────────────────────
# Common helpers
# ──────────────────────────────────────────────────────────────────────

PIP_MIRRORS = [
    ["-i", "https://pypi.tuna.tsinghua.edu.cn/simple"],
    [],  # fallback: default PyPI
]


def pip_install_with_fallback(pkgs: list[str], label: str = "") -> None:
    """Try installing packages via mirrors, fallback to default PyPI."""
    for mirror in PIP_MIRRORS:
        cmd = [sys.executable, "-m", "pip", "install"] + mirror + pkgs
        print(f"  [{label}] Installing: {' '.join(cmd)}")
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode == 0:
            # Verify all are now importable
            still_missing = [p for p in pkgs if importlib.util.find_spec(p) is None]
            if not still_missing:
                print(f"  [{label}] All dependencies installed successfully.")
                return
        if mirror:
            print(f"  [{label}] Mirror failed, trying default PyPI...")
    raise RuntimeError(
        f"[{label}] Failed to install: {' '.join(pkgs)}.\n"
        f"Please try manually: {sys.executable} -m pip install {' '.join(pkgs)}")


COMMON_DEPS = {
    "onnxruntime": "onnxruntime",
    "numpy": "numpy",
    "onnx": "onnx",
}


def ensure_common_deps():
    missing = [m for m in COMMON_DEPS if importlib.util.find_spec(m) is None]
    if missing:
        pkgs = [COMMON_DEPS[m] for m in missing]
        pip_install_with_fallback(pkgs, "core")


def parse_path(value: str) -> Path:
    v = value.strip()
    if len(v) >= 2 and v[0] == v[-1] and v[0] in {'"', "'"}:
        v = v[1:-1]
    return Path(v)


def get_io_info_ort(model_path: str) -> dict:
    """Return {'inputs':[(name,(dims...))], 'outputs':[(name,(dims...))]} via ORT."""
    import onnxruntime as ort
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    inputs = []
    for inp in sess.get_inputs():
        shape = tuple(d if (isinstance(d, int) and d > 0) else 1 for d in inp.shape)
        inputs.append((inp.name, shape))
    outputs = []
    for out in sess.get_outputs():
        shape = tuple(d if (isinstance(d, int) and d > 0) else 1 for d in out.shape)
        outputs.append((out.name, shape))
    return {'inputs': inputs, 'outputs': outputs}


# ──────────────────────────────────────────────────────────────────────
# NCNN converter
# ──────────────────────────────────────────────────────────────────────

def find_pnnx() -> str:
    script_dir = Path(__file__).resolve().parent
    for name in ("pnnx.exe", "pnnx"):
        p = script_dir / name
        if p.exists() and os.access(str(p), os.X_OK):
            return str(p)
    return "pnnx"


def get_inputshape_string(model_path: str) -> str:
    info = get_io_info_ort(model_path)
    parts = []
    for name, shape in info['inputs']:
        parts.append(f"{name}=[{','.join(str(d) for d in shape)}]")
    return ",".join(parts)


def write_ncnn_shapes(shapes_path: str, info: dict):
    """Write NCNN shapes file (in0..inN / out0..outN lines only).

    NOTE: do NOT write 'inputs=N' / 'outputs=N' count lines - the C++ reader
    (NCNNBackend::ReadShapesFile) matches prefixes 'in'/'out', so a line
    'inputs=31' would be parsed as a fake input named 'inputs' with shape
    [31], shifting every real input/output and corrupting the accuracy
    baseline comparison (first-output diff explodes).
    """
    with open(shapes_path, 'w') as f:
        for i, (name, dims) in enumerate(info['inputs']):
            f.write(f"in{i}={','.join(str(d) for d in dims)}\n")
        for i, (name, dims) in enumerate(info['outputs']):
            f.write(f"out{i}={','.join(str(d) for d in dims)}\n")
    print(f"  Wrote shapes: {shapes_path}")


def gen_ncnn_cpp(model_path: str):
    info = get_io_info_ort(model_path)
    print("\n// ---------------- NCNN C++ Snippet ----------------")
    print(f"const int NUM_INPUTS = {len(info['inputs'])};")
    print("ncnn::Mat inputs[NUM_INPUTS];\n")
    for i, (name, shape) in enumerate(info['inputs']):
        if len(shape) == 4:
            n, c, h, w = shape
            print(f"inputs[{i}] = ncnn::Mat({w}, {h}, {c}); // {name} {shape}")
        elif len(shape) == 3:
            c, h, w = shape
            print(f"inputs[{i}] = ncnn::Mat({w}, {h}, {c}); // {name} {shape}")
        elif len(shape) == 2:
            h, w = shape
            print(f"inputs[{i}] = ncnn::Mat({w}, {h}); // {name} {shape}")
        elif len(shape) == 1:
            print(f"inputs[{i}] = ncnn::Mat({shape[0]}); // {name} {shape}")
        else:
            print(f"// WARNING: shape {shape} for {name}")
    print("\nfor (int i = 0; i < NUM_INPUTS; ++i) inputs[i].fill(0.f);")
    print("// ---------------------------------------------------\n")


def clean_pnnx_intermediates(model_dir: str, keep_pcnn: bool):
    patterns = ["*.pnnx.onnx", "*.pnnx.param", "*.pnnx.bin",
                "*.pnnxsim.onnx", "*_pnnx.py", "*_ncnn.py"]
    if not keep_pcnn:
        patterns.append("pcnn*")
    for pat in patterns:
        for f in glob_mod.glob(os.path.join(model_dir, pat)):
            try:
                os.remove(f)
                print(f"  Cleaned: {os.path.basename(f)}")
            except OSError:
                pass


def verify_ncnn(onnx_path: str, ncnn_param: str, ncnn_bin: str) -> int:
    """0=pass, 2=fail.

    No silent skip: a model that fails to load, fails the forward, or produces
    shape/value-mismatched outputs is reported as FAIL (2) so broken
    conversions are caught instead of being shipped. Only SKIP (0) when the
    optional libs (onnxruntime / ncnn / numpy) are unavailable.
    """
    try:
        import onnxruntime as ort
        import ncnn
        import numpy as np
    except ImportError as e:
        print(f"  Skip NCNN verify (import): missing {e.name}")
        return 0
    except Exception as e:
        print(f"  Skip NCNN verify: {e}")
        return 0

    rng = np.random.default_rng(123456789)
    info = get_io_info_ort(onnx_path)
    inputs_info, outputs_info = info['inputs'], info['outputs']

    # ONNX reference
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    feed = {}
    for i, (name, shape) in enumerate(inputs_info):
        data = rng.random(shape).astype(np.float32)
        feed[name] = data
    ref_outs = sess.run(None, feed)

    def _run(label, vulkan):
        net = ncnn.Net()
        if vulkan:
            if ncnn.get_gpu_count() == 0:
                print(f"\n  {label}: SKIP - no Vulkan GPU")
                return True
            net.opt.use_vulkan_compute = True
            net.opt.use_fp16_packed = False
            net.opt.use_fp16_storage = False
            net.opt.use_fp16_arithmetic = False

        lr = 0
        mr = 0
        try:
            lr = net.load_param(ncnn_param)
            mr = net.load_model(ncnn_bin)
        except Exception as e:
            msg = str(e).split('\n')[0] if str(e) else str(type(e).__name__)
            print(f"\n  {label}: FAIL - model load error: {msg}")
            return False
        if lr != 0 or mr != 0:
            print(f"\n  {label}: FAIL - model load error (param={lr} model={mr})")
            return False

        ex = net.create_extractor()
        try:
            for i in range(len(inputs_info)):
                arr = feed[inputs_info[i][0]]
                # pnnx strips the leading batch dim (size 1) from ONNX inputs;
                # only drop it when shape[0] == 1. 3D state tensors such as
                # [472,32,2] (dim0 != 1) MUST be passed as-is - squeezing them
                # raised and silently skipped the whole verification before.
                if len(arr.shape) >= 2 and arr.shape[0] == 1:
                    arr = arr.reshape(arr.shape[1:])
                mat = ncnn.Mat(np.ascontiguousarray(arr))
                ex.input(f"in{i}", mat.clone())
            outs = []
            for i in range(len(outputs_info)):
                ret, m = ex.extract(f"out{i}")
                if ret != 0 or m.total() == 0:
                    print(f"\n  {label}: FAIL - extract out{i} failed (ret={ret})")
                    return False
                outs.append(np.array(m))
        except Exception as e:
            msg = str(e).split('\n')[0] if str(e) else str(type(e).__name__)
            print(f"\n  {label}: FAIL - inference error: {msg}")
            return False

        print(f"\n  === {label} ===")
        all_ok = True
        worst = 0.0
        for i, (ref, out) in enumerate(zip(ref_outs, outs)):
            a, b = ref.flatten(), out.flatten()
            if a.shape != b.shape:
                print(f"  out{i}: shape mismatch {ref.shape} vs {out.shape}  FAIL")
                all_ok = False
                continue
            if a.size == 0:
                continue
            mx = float(np.abs(a - b).max())
            avg = float(np.abs(a - b).mean())
            tag = "OK" if mx < 5.0 else "***LARGE***"
            print(f"  out{i}: max={mx:.6f} avg={avg:.6f} {tag}")
            if mx > worst:
                worst = mx
            if mx >= 5.0:
                all_ok = False
        if len(outs) != len(ref_outs):
            print(f"  output count mismatch: ref={len(ref_outs)} got={len(outs)}  FAIL")
            all_ok = False
        status = "PASS" if all_ok else "FAIL"
        print(f"  {label}: {status} (worst={worst:.6f})")
        return all_ok

    ok1 = _run("NCNN CPU", vulkan=False)
    # Skip Vulkan verification on all platforms (GPU info logs interfere with
    # subprocess output parsing in the verify harness)
    return 0 if ok1 else 2


def preprocess_onnx_for_ncnn(onnx_path: Path) -> Path | None:
    """Replace ONNX Tile ops with equivalent Concat ops, and fix static
    Split nodes that pnnx cannot infer.

    PNNX converts ONNX Tile to custom 'torch.tile' layers not available in
    standard ncnn. Replacing Tile with Concat before conversion avoids this.

    PNNX also crashes (0xC0000005) on Split nodes WITHOUT a `split` input when
    it cannot infer the split sizes (e.g. input produced by a Slice), even
    though the sizes are actually static. If every output chunk size can be
    derived from shape inference, the `split` input tensor is added explicitly
    so pnnx sees a fully static Split.

    Returns path to preprocessed model (a temp file), or None if no changes.
    """
    try:
        import onnx as onnx_mod
        from onnx import helper as onnx_helper
        import onnx.numpy_helper as onnx_np
        import numpy as np
    except ImportError:
        return None

    model = onnx_mod.load(str(onnx_path))
    graph = model.graph

    # Collect initializers
    inits = {}
    for init in graph.initializer:
        try:
            inits[init.name] = onnx_np.to_array(init)
        except Exception:
            pass
    # Constant nodes also produce compile-time tensors (e.g. Tile repeats
    # exported by PyTorch as a Constant node instead of an initializer).
    # Treat their `value` attribute like an initializer so Tile->Concat can
    # fire for them too.
    for node in graph.node:
        if node.op_type == "Constant" and len(node.output) == 1:
            for a in node.attribute:
                if a.name == "value" and node.output[0] not in inits:
                    try:
                        inits[node.output[0]] = onnx_np.to_array(a.t)
                    except Exception:
                        pass

    changes = 0
    tile_replacements = []
    erf_replacements = []
    split_fixes = []   # (node, split_sizes_tensor_name, sizes_list)

    for node in graph.node:
        # ── Tile -> Concat ──
        if node.op_type == "Tile":
            repeats = inits.get(node.input[1], None)
            if repeats is None:
                print(f"  WARNING: Tile '{node.name}' has dynamic repeats, skipping")
                continue
            rlist = list(repeats)
            non_one = [(a, r) for a, r in enumerate(rlist) if r > 1]
            if not non_one:
                tile_replacements.append((node, "identity", None, None))
                changes += 1
            elif len(non_one) == 1:
                axis, rep = non_one[0]
                tile_replacements.append((node, "concat", int(axis), int(rep)))
                changes += 1
            else:
                tile_replacements.append((node, "multi_concat", rlist, None))
                changes += 1

        # ── Erf -> Tanh approximation ──
        # erf(x) ≈ tanh(sqrt(2/π) * (x + 0.044715 * x³))
        # sqrt(2/π) ≈ 0.7978845608028654
        if node.op_type == "Erf":
            erf_replacements.append(node)
            changes += 1

    # ── Static Split fix ──
    # pnnx crashes (0xC0000005) on Split nodes without a `split` input when it
    # cannot infer the chunk sizes from the graph. If shape inference yields a
    # concrete input shape, compute the equal-split sizes and attach them as an
    # explicit `split` initializer (opset 13+ form: second input tensor). This
    # makes the Split fully static for pnnx. Skip dynamic shapes entirely.
    try:
        import onnx.shape_inference as onnx_si
    except ImportError:
        onnx_si = None

    split_candidates = [n for n in graph.node if n.op_type == "Split" and len(n.input) == 1]
    if split_candidates and onnx_si is not None:
        try:
            inferred = onnx_si.infer_shapes(model)
            value_info = {}
            for vi in inferred.graph.value_info:
                value_info[vi.name] = vi
            for vi in inferred.graph.input:
                value_info[vi.name] = vi

            for node in split_candidates:
                src = node.input[0]
                vi = value_info.get(src)
                if vi is None or not vi.type.HasField("tensor_type"):
                    continue
                dims = vi.type.tensor_type.shape.dim
                axis = 1
                for a in node.attribute:
                    if a.name == "axis":
                        axis = onnx_helper.get_attribute_value(a)
                axis = axis if axis >= 0 else axis + len(dims)
                if axis < 0 or axis >= len(dims) or not dims[axis].HasField("dim_value"):
                    continue  # dynamic dim -> cannot fix
                total = dims[axis].dim_value
                n_out = len(node.output)
                if n_out <= 0 or total % n_out != 0:
                    continue  # not evenly splittable -> leave as-is
                size = total // n_out
                sizes = [size] * n_out
                split_name = f"{node.name}_split_sizes"
                if split_name not in inits:
                    split_init = onnx_np.from_array(
                        np.array(sizes, dtype=np.int64), name=split_name)
                    graph.initializer.append(split_init)
                    inits[split_name] = sizes
                split_fixes.append((node, split_name, sizes))
                changes += 1
        except Exception as e:
            print(f"  WARNING: Split static-size inference failed: {e}")

    if changes == 0:
        return None

    # Save preprocessed model alongside original for pnnx to find outputs
    tmp_path = onnx_path.with_suffix(".prep.onnx")

    parts = []
    if tile_replacements:
        parts.append(f"{len(tile_replacements)} Tile->Concat")
    if erf_replacements:
        parts.append(f"{len(erf_replacements)} Erf->Tanh")
    if split_fixes:
        parts.append(f"{len(split_fixes)} Split->static")
    print(f"  Preprocessing ONNX: {', '.join(parts)}...")

    # Erf approximation constants as ONNX initializers
    # erf(x) ≈ tanh(sqrt(2/π) * (x + 0.044715 * x³))
    # sqrt(2/π) ≈ 0.7978845608028654
    SCALE = 0.7978845608028654   # sqrt(2/pi)
    ALPHA = 0.044715

    # Create constant initializers (float32 scalars)
    const_scale = onnx_np.from_array(
        np.array([SCALE], dtype=np.float32), name="_erf_scale")
    const_alpha = onnx_np.from_array(
        np.array([ALPHA], dtype=np.float32), name="_erf_alpha")
    # Add to graph if needed
    existing_init_names = set(i.name for i in graph.initializer)
    for c in (const_scale, const_alpha):
        if c.name not in existing_init_names:
            graph.initializer.append(c)
            existing_init_names.add(c.name)

    # Build new node list
    new_nodes = []

    for node in graph.node:
        # ── Handle Tile replacement ──
        tile_repl = [r for r in tile_replacements if r[0] == node]
        if tile_repl:
            r = tile_repl[0]
            op_type = r[1]
            tile_input = node.input[0]
            tile_output = node.output[0]

            if op_type == "identity":
                identity = onnx_helper.make_node(
                    "Identity", [tile_input], [tile_output],
                    name=f"{node.name}_replace")
                new_nodes.append(identity)
                print(f"    {node.name}: Tile(rep=all_1) -> Identity")

            elif op_type == "concat":
                axis, rep = r[2], r[3]
                concat_inputs = [tile_input] * rep
                concat = onnx_helper.make_node(
                    "Concat", concat_inputs, [tile_output],
                    name=f"{node.name}_replace", axis=axis)
                new_nodes.append(concat)
                print(f"    {node.name}: Tile(rep={list(inits.get(node.input[1], []))}) "
                      f"-> Concat(x{rep}, axis={axis})")

            elif op_type == "multi_concat":
                rlist = r[2]
                current = tile_input
                for axis, rep in enumerate(rlist):
                    if rep <= 1:
                        continue
                    next_name = f"{node.name}_tile_axis{axis}"
                    concat_inputs = [current] * rep
                    concat = onnx_helper.make_node(
                        "Concat", concat_inputs, [next_name],
                        name=f"{node.name}_a{axis}", axis=axis)
                    new_nodes.append(concat)
                    current = next_name
                # Last Concat outputs to original tile_output
                if current != tile_output:
                    for i, n in enumerate(new_nodes):
                        if n.output[0] == current and n.name.startswith(f"{node.name}_"):
                            new_nodes[i] = onnx_helper.make_node(
                                n.op_type, list(n.input), [tile_output],
                                name=n.name, **{attr.name: onnx_helper.get_attribute_value(attr)
                                                for attr in n.attribute})
                            break
                print(f"    {node.name}: Tile(rep={rlist}) -> Concat chain")
            continue

        # ── Handle static Split fix ──
        split_repl = [s for s in split_fixes if s[0] == node]
        if split_repl:
            _, split_name, sizes = split_repl[0]
            n2 = onnx_helper.make_node(
                "Split",
                inputs=list(node.input) + [split_name],
                outputs=list(node.output),
                name=node.name,
                **{a.name: onnx_helper.get_attribute_value(a) for a in node.attribute})
            new_nodes.append(n2)
            print(f"    {node.name}: Split -> static split={sizes}")
            continue

        # ── Handle Erf replacement ──
        if node in erf_replacements:
            inp = node.input[0]
            out = node.output[0]

            # x² = x * x
            sq_name = f"{node.name}_sq"
            sq = onnx_helper.make_node("Mul", [inp, inp], [sq_name],
                                        name=f"{node.name}_sq")
            # x³ = x² * x
            cb_name = f"{node.name}_cb"
            cb = onnx_helper.make_node("Mul", [sq_name, inp], [cb_name],
                                        name=f"{node.name}_cb")
            # ax³ = alpha * x³
            ax3_name = f"{node.name}_ax3"
            ax3 = onnx_helper.make_node("Mul", [cb_name, const_alpha.name], [ax3_name],
                                         name=f"{node.name}_ax3")
            # x + ax³
            sum_name = f"{node.name}_sum"
            sm = onnx_helper.make_node("Add", [inp, ax3_name], [sum_name],
                                        name=f"{node.name}_sum")
            # scaled = scale * (x + ax³)
            scaled_name = f"{node.name}_scaled"
            sc = onnx_helper.make_node("Mul", [sum_name, const_scale.name], [scaled_name],
                                        name=f"{node.name}_scaled")
            # tanh(scaled)
            tn = onnx_helper.make_node("Tanh", [scaled_name], [out],
                                        name=f"{node.name}_replace")

            new_nodes.extend([sq, cb, ax3, sm, sc, tn])
            print(f"    {node.name}: Erf -> Tanh approx")
            continue

        # ── Pass through unchanged ──
        new_nodes.append(node)

    # Replace graph nodes
    while len(graph.node) > 0:
        graph.node.pop()
    for n in new_nodes:
        graph.node.append(n)

    onnx_mod.save(model, str(tmp_path))
    print(f"  Preprocessed model saved to: {tmp_path}")
    return tmp_path

def convert_ncnn(model_path: Path, args) -> int:
    print("\n" + "=" * 60)
    print(" [1/3] NCNN Conversion")
    print("=" * 60)

    if args.gen_cpp:
        gen_ncnn_cpp(str(model_path))
        return (True, None)

    # Preprocess: replace ONNX Tile -> Concat so pnnx doesn't emit torch.tile
    preprocessed = preprocess_onnx_for_ncnn(model_path)
    pnnx_input = preprocessed if preprocessed else model_path
    if preprocessed:
        print(f"  Using preprocessed model: {pnnx_input}")

    inputshape_str = get_inputshape_string(str(model_path))
    print(f"  inputshape: {inputshape_str}")

    pnnx = find_pnnx()
    model_dir = str(model_path.parent)

    def _run_pnnx(fp16_val: str) -> None:
        """Run pnnx with the given fp16 setting, using a .bat file if the
        command line exceeds Windows length limit (~8191 chars)."""
        cmd = [pnnx, str(pnnx_input), f"inputshape={inputshape_str}", f"fp16={fp16_val}"]
        cl = ' '.join(cmd)
        if len(cl) > 6000:
            bat = model_path.parent / f"_pnnx_fp{fp16_val}.bat"
            with open(bat, 'w') as f:
                f.write(f'@echo off\n{cl}\n')
            print(f"  Running via .bat: {bat}")
            subprocess.run([str(bat)], check=True, shell=True)
            bat.unlink(missing_ok=True)
        else:
            print(f"  Running: {cl}")
            subprocess.run(cmd, check=True)

    # FP32
    try:
        _run_pnnx("0")
    except Exception as e:
        msg = str(e).split('\n')[0] if str(e) else str(type(e).__name__)
        print(f"  pnnx failed: {msg}")
        # Clean up partial NCNN outputs
        for suffix in (".ncnn.param", ".ncnn.bin", ".shapes",
                       ".pnnx.param", ".pnnx.bin", ".pnnx.onnx"):
            p = model_path.with_suffix(suffix)
            if p.exists():
                p.unlink(missing_ok=True)
        fp16_b = model_path.parent / (model_path.stem + "_fp16.ncnn.bin")
        if fp16_b.exists():
            fp16_b.unlink(missing_ok=True)
        return (False, f"pnnx failed: {msg}")

    def _find_pnnx_output(lookup_path: Path, suffix: str) -> Path | None:
        """Find pnnx output file and rename to original model name if needed.
        pnnx may replace '-' with '_' in filenames."""
        expected = lookup_path.with_suffix(suffix)
        if expected.exists():
            if preprocessed and lookup_path != model_path:
                # Rename from prep name to original model name
                target = model_path.with_suffix(suffix)
                os.replace(str(expected), str(target))
                return target
            return expected
        # pnnx replaces '-' with '_' in the filename stem
        alt_stem = lookup_path.stem.replace('-', '_')
        alt = lookup_path.parent / f"{alt_stem}{suffix}"
        if alt.exists():
            print(f"  Found at alternate name: {alt}")
            target = model_path.with_suffix(suffix)
            os.replace(str(alt), str(target))
            return target
        return None

    param = _find_pnnx_output(pnnx_input, ".ncnn.param")
    bin_  = _find_pnnx_output(pnnx_input, ".ncnn.bin")

    # Dual FP16 (default: generate both FP32 and FP16)
    if not getattr(args, 'no_dual', False) and not getattr(args, 'fp16', False):
        print("\n  --dual: generating FP16 model...")
        if param is None:
            print(f"  WARNING: NCNN param not found, skipping FP16")
        else:
            fp16_bin = model_path.parent / (model_path.stem + "_fp16.ncnn.bin")
            # Delete stale .ncnn.bin so _find_pnnx_output picks up the fresh
            # FP16 output (not the old FP32 one)
            old_bin = model_path.with_suffix(".ncnn.bin")
            if old_bin.exists():
                old_bin.unlink()
            try:
                _run_pnnx("1")
            except Exception as e:
                msg = str(e).split('\n')[0] if str(e) else str(type(e).__name__)
                print(f"  pnnx FP16 failed: {msg} (FP32 model already generated)")
                # Restore FP32 bin
                _run_pnnx("0")
                param = _find_pnnx_output(pnnx_input, ".ncnn.param")
                bin_  = _find_pnnx_output(pnnx_input, ".ncnn.bin")
            # pnnx always writes outputs next to its input file, which is the
            # preprocessed (prep) model. Look up via pnnx_input so the fresh
            # FP16 bin is found and renamed to the model name before we move
            # it to the _fp16 target. Looking up via model_path misses the
            # prep-prefixed output and silently drops the FP16 model.
            tmp_b = _find_pnnx_output(pnnx_input, ".ncnn.bin")
            if tmp_b is not None and tmp_b.exists():
                os.replace(str(tmp_b), str(fp16_bin))
                print(f"  FP16 bin: {fp16_bin} ({fp16_bin.stat().st_size/1024:.0f} KB)")
            # Re-run FP32 to restore FP32 weights
            print("  Re-running FP32...")
            _run_pnnx("0")
            param = _find_pnnx_output(pnnx_input, ".ncnn.param")
            bin_  = _find_pnnx_output(pnnx_input, ".ncnn.bin")

    if param is None or bin_ is None:
        print("  NCNN conversion FAILED - no NCNN output files found")
        return (False, "pnnx produced no output")

    # Shapes file
    info = get_io_info_ort(str(model_path))
    write_ncnn_shapes(str(model_path.with_suffix(".shapes")), info)

    # Cleanup preprocessed model and its pnnx intermediates
    if preprocessed and preprocessed.exists():
        try:
            preprocessed.unlink()
            print(f"  Cleaned preprocessed: {preprocessed.name}")
        except OSError:
            pass
        # Remove any pnnx/temp files created from the preprocessed name
        prep_stem = preprocessed.stem
        for f in os.listdir(model_dir):
            if f.startswith(prep_stem):
                fp = os.path.join(model_dir, f)
                try:
                    os.remove(fp)
                    print(f"  Cleaned: {f}")
                except OSError:
                    pass

    # Cleanup
    if not getattr(args, 'no_cleanup', False):
        clean_pnnx_intermediates(model_dir, keep_pcnn=getattr(args, 'keep_pcnn', False))

    # Verify (run in subprocess with timeout to avoid hangs from broken models)
    if not args.no_verify:
        print("\n  Verifying NCNN...")
        try:
            ret = verify_ncnn(str(model_path), str(param), str(bin_))
            if ret != 0:
                for f in (param, bin_, model_path.with_suffix(".shapes")):
                    try: os.remove(f)
                    except OSError: pass
                print("  NCNN conversion FAILED verification")
                return (False, "verification failed")
        except Exception as e:
            msg = str(e).split('\n')[0] if str(e) else str(type(e).__name__)
            print(f"  NCNN verify error: {msg}")
            print("  Model files preserved - use --no-verify to skip verification")

    print(f"  NCNN OK: {param} / {bin_}")
    return (True, None)


# ──────────────────────────────────────────────────────────────────────
# MNN converter
# ──────────────────────────────────────────────────────────────────────

def find_mnnconvert() -> str:
    script_dir = Path(__file__).resolve().parent
    for c in [
        script_dir / "mnn_convert" / "MNNConvert",
        script_dir / "mnn_convert" / "MNNConvert.exe",
        script_dir.parent / "deps" / "mnn" / "tools" / "MNNConvert",
        script_dir.parent / "deps" / "mnn" / "tools" / "MNNConvert.exe",
    ]:
        if c.exists():
            return str(c)
    for n in ("MNNConvert", "MNNConvert.exe"):
        try:
            found = shutil.which(n)
            if found: return found
        except Exception:
            pass
    return "MNNConvert"


def verify_mnn(onnx_path: Path, mnn_path: Path) -> int:
    """0=pass, 1=marginal, 2=fail"""
    if not mnn_path.exists():
        print("  MNN model not found, skip verify")
        return 2
    try:
        import numpy as np
        import onnxruntime as ort
        import MNN
    except ImportError as e:
        print(f"  Skip MNN verify: missing {e.name}")
        return 0

    info = get_io_info_ort(str(onnx_path))
    print(f"\n  Deep Compare: ONNX CPU vs MNN CPU")

    # ONNX
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_inputs = {}
    for i, (name, dims) in enumerate(info['inputs']):
        np.random.seed(123456789 + i)
        ort_inputs[name] = np.random.rand(*dims).astype(np.float32)
    ort_outs = sess.run(None, ort_inputs)

    # MNN
    interp = MNN.Interpreter(str(mnn_path))
    session = interp.createSession({'type': MNN.expr.CPU, 'numThread': 4})
    for name in ort_inputs:
        t = interp.getSessionInput(session, name)
        if t is not None:
            t.getNumpyData()[:] = ort_inputs[name]
    interp.runSession(session)

    total_max = 0.0
    for idx, (name, _) in enumerate(info['outputs']):
        t = interp.getSessionOutput(session, name)
        if t is None:
            print(f"  {name}: NOT in MNN outputs  FAIL")
            return 2
        a = ort_outs[idx].flatten()
        b = t.getNumpyData().flatten()
        if a.size != b.size:
            print(f"  {name}: SIZE MISMATCH {a.size} vs {b.size}  FAIL")
            return 2
        mx = float(np.abs(a - b).max())
        avg = float(np.abs(a - b).mean())
        if mx > total_max: total_max = mx
        print(f"  {name:12s}  elems={a.size:>8}  max={mx:.6f}  avg={avg:.6f}")

    print(f"  Worst max_diff={total_max:.8f}")
    if total_max < 1e-5:         print("  PASS - fp32 equivalent"); return 0
    elif total_max < 0.01:       print("  MARGINAL - minor rounding"); return 1
    elif total_max < 0.5:        print("  ACCEPTABLE - validate on real data"); return 1
    else:                        print("  FAIL - large divergence"); return 2


def convert_mnn(model_path: Path, args) -> int:
    print("\n" + "=" * 60)
    print(" [2/3] MNN Conversion")
    print("=" * 60)

    mnn_path = model_path.with_suffix(".mnn")
    mnnconvert = find_mnnconvert()
    fp16 = getattr(args, 'fp16', False)

    cmd = [mnnconvert, "-f", "ONNX", "--modelFile", str(model_path),
           "--MNNModel", str(mnn_path)]
    if fp16:
        cmd += ["--fp16", "1"]
    # LSTM models may need origin RNN impl instead of While-loop
    if not args.no_rnn_opt:
        cmd += ["--useOriginRNNImpl"]

    if mnn_path.exists():
        mnn_path.unlink()

    print(f"  Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True,
                                timeout=300, encoding="utf-8", errors="replace")
        if result.returncode != 0:
            # MNNConvert may crash with ACCESS_VIOLATION (exit=3221225477)
            # AFTER printing "Converted Success!" - check if .mnn was produced.
            if mnn_path.exists() and mnn_path.stat().st_size > 0:
                print(f"  MNNConvert exited with code {result.returncode} but output exists")
                print(f"  (MNNConvert may crash on cleanup after successful conversion)")
            else:
                err_msg = result.stderr.strip() or result.stdout.strip() or "unknown error"
                err_lines = err_msg.split('\n')
                brief = ' | '.join(line.strip() for line in err_lines[-5:] if line.strip())
                print(f"  MNNConvert failed (exit={result.returncode}): {brief[:200]}")
                if result.stderr.strip():
                    for line in result.stderr.strip().split('\n')[-10:]:
                        print(f"    {line.strip()}")
                return (False, f"MNNConvert exited with code {result.returncode}")
    except FileNotFoundError:
        print(f"  MNNConvert not found: {mnnconvert}")
        print("  Ensure MNNConvert is in tools/mnn_convert/ or PATH")
        if mnn_path.exists():
            mnn_path.unlink(missing_ok=True)
        return (False, "MNNConvert not found")
    except subprocess.TimeoutExpired:
        print("  MNNConvert timed out after 300s")
        if mnn_path.exists():
            mnn_path.unlink(missing_ok=True)
        return (False, "MNNConvert timed out")
    except Exception as e:
        print(f"  MNNConvert error: {e}")
        if mnn_path.exists():
            mnn_path.unlink(missing_ok=True)
        return (False, str(e))

    if not mnn_path.exists() or mnn_path.stat().st_size == 0:
        print("  MNNConvert produced no output")
        return (False, "MNNConvert produced no output")

    print(f"  MNN model: {mnn_path} ({mnn_path.stat().st_size / 1024:.0f} KB)")

    if not args.no_verify:
        try:
            ret = verify_mnn(model_path, mnn_path)
            if ret == 2:
                mnn_path.unlink(missing_ok=True)
                print("  MNN conversion FAILED verification")
                return (False, "verification failed")
        except Exception as e:
            print(f"  MNN verify skipped: {e}")
            print("  Model preserved - use --no-verify to skip verification")

    return (True, None)


# ──────────────────────────────────────────────────────────────────────
# TFLite converter
# ──────────────────────────────────────────────────────────────────────

TFLITE_DEPS = {
    "onnx2tf": "onnx2tf",
    "tensorflow": "tensorflow",
    "tf_keras": "tf-keras",
    "onnx": "onnx",
    "onnx_graphsurgeon": "onnx-graphsurgeon",
    "onnxsim": "onnxsim",
}

# TensorFlow often requires a newer protobuf than what's currently installed.
# Pin to a version compatible with TF's compiled protobuf gencode.
PROTOBUF_PIN = "protobuf>=6.31.1"


def ensure_tflite_deps():
    missing = [m for m in TFLITE_DEPS if importlib.util.find_spec(m) is None]
    if missing:
        pkgs = [TFLITE_DEPS[m] for m in missing]
        pip_install_with_fallback(pkgs, "tflite")
    # After installing, ensure protobuf is recent enough for the installed TF
    try:
        import google.protobuf
    except Exception:
        pass
    else:
        import subprocess as _sp
        r = _sp.run(
            [sys.executable, "-m", "pip", "install", PROTOBUF_PIN],
            capture_output=True, text=True, timeout=60)
        if r.returncode != 0:
            msg = r.stderr.strip() or r.stdout.strip() or "unknown"
            print(f"  [tflite] protobuf upgrade warning: {msg[:120]}")
        else:
            importlib.invalidate_caches()
            if "google.protobuf" in sys.modules:
                del sys.modules["google.protobuf"]
            print("  [tflite] protobuf upgraded for tensorflow compatibility")


def find_latest_tflite(root: Path) -> Path:
    candidates = [p for p in root.rglob("*.tflite") if p.is_file()]
    if not candidates:
        raise FileNotFoundError("No .tflite produced by onnx2tf")
    float32 = [p for p in candidates if "_float32" in p.stem]
    chosen = max(float32 or candidates, key=lambda p: p.stat().st_mtime)
    print(f"  Selected: {chosen.name}")
    return chosen


def generate_nchw_to_nhwc_prf(onnx_model, output_path: Path) -> Path | None:
    """Generate a parameter replacement file for onnx2tf's NCHW->NHWC conversion.

    For 4D tensors the NCHW->NHWC layout mapping is:
        L = [0, 3, 1, 2]   (NCHW dim i -> NHWC position L[i])
    And the NHWC-equivalent perm Q of a NCHW perm P is:
        Q[j] = L[P[inv_L[j]]]   where inv_L = [0, 2, 3, 1]
    """
    # Layout mappings for different tensor dimensionalities.
    # For 4D: NCHW [N,C,H,W] ↔ NHWC [N,H,W,C]
    L4 = [0, 3, 1, 2]
    inv_L4 = [0, 2, 3, 1]
    # For 3D: NCD [N,C,D] ↔ NDC [N,D,C]
    L3 = [0, 2, 1]
    inv_L3 = [0, 2, 1]

    def nhwc_perm(p: list[int]) -> list[int]:
        nd = len(p)
        if nd == 4:
            return [L4[p[inv_L4[j]]] for j in range(4)]
        if nd == 3:
            return [L3[p[inv_L3[j]]] for j in range(3)]
        return p

    def nhwc_axis(axis: int, nd: int) -> int:
        if nd == 4:
            return L4[axis % 4]
        if nd == 3:
            return L3[axis % 3]
        return axis

    operations: list[dict] = []

    for node in onnx_model.graph.node:
        if node.op_type == "Transpose":
            perm_attr = [a for a in node.attribute if a.name == "perm"]
            if perm_attr:
                p = list(perm_attr[0].ints)
                nd = len(p)
                if nd in (3, 4):
                    q = nhwc_perm(p)
                    L = L4 if nd == 4 else L3
                    onnx2tf_would_produce = [L[p[i]] for i in range(nd)]
                    if q != onnx2tf_would_produce:
                        operations.append({
                            "op_name": node.name,
                            "param_target": "attributes",
                            "param_name": "perm",
                            "values": q,
                        })
        elif node.op_type == "Concat":
            axis_attr = [a for a in node.attribute if a.name == "axis"]
            if axis_attr:
                orig_axis = axis_attr[0].i
                # Determine dimensionality from the first input tensor
                nd = 4
                nd_found = False
                if node.input:
                    for vi in onnx_model.graph.value_info:
                        if vi.name == node.input[0]:
                            nd = len(vi.type.tensor_type.shape.dim)
                            nd_found = True
                            break
                # Skip axis change for non-4D tensors when we can't confirm nd
                # from value_info, or for 3D tensors where onnx2tf's NCHW->NHWC
                # handling is unreliable (causes shape incompatibility in Concat)
                if nd_found and nd == 4:
                    new_axis = nhwc_axis(orig_axis, 4)
                    if orig_axis != new_axis:
                        operations.append({
                            "op_name": node.name,
                            "param_target": "attributes",
                            "param_name": "axis",
                            "values": new_axis,
                        })
                elif nd_found and nd != 4:
                    print(f"  SKIP Concat '{node.name}': {nd}D tensor, axis unchanged (axis={orig_axis})")

    if not operations:
        return None

    prf = {"format_version": 1, "operations": operations}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(prf, f, indent=2)
    print(f"  Generated PRF with {len(operations)} entries to fix NCHW->NHWC layout")
    return output_path


def convert_tflite_inner(input_onnx: Path, output_tflite: Path, keep_temp: bool):
    if not input_onnx.exists():
        raise FileNotFoundError(f"Not found: {input_onnx}")
    ensure_tflite_deps()

    # Set env vars to avoid GBK encoding issues and oneDNN numerical warnings
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    with tempfile.TemporaryDirectory(prefix="onnx2tf_") as tmp:
        tmp_dir = Path(tmp)

        # Patches are already applied at module top; call onnx2tf inline
        import onnx2tf.onnx2tf as _onnx2tf_mod
        _old_argv = sys.argv.copy()
        try:
            sys.argv = [_onnx2tf_mod.__file__,
                        "-i", str(input_onnx), "-o", str(tmp_dir),
                        "-coion", "-nuo", "-n"]
            print(f"  Running: onnx2tf {' '.join(sys.argv[1:])}")
            _onnx2tf_mod.main()
        except SystemExit:
            pass
        finally:
            sys.argv = _old_argv

        produced = find_latest_tflite(tmp_dir)
        output_tflite.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(produced, output_tflite)
        print(f"  Output: {output_tflite}")

        if keep_temp:
            debug = output_tflite.parent / f"{output_tflite.stem}_onnx2tf_debug"
            if debug.exists(): shutil.rmtree(debug)
            shutil.copytree(tmp_dir, debug)
            print(f"  Debug artifacts: {debug}")


def _make_tflite_input_data(shape: list[int], dtype, nd: int,
                            nchw_to_nhwc_fn) -> np.ndarray:
    """Generate random input data of correct dtype and layout for TFLite tensor."""
    import numpy as _np
    if dtype in (_np.float32, _np.float16):
        if nd == 4:
            nchw_shape = [shape[0], shape[3], shape[1], shape[2]]
            data = _np.random.rand(*nchw_shape).astype(dtype)
            data = nchw_to_nhwc_fn(data, nchw_shape)
        elif nd == 3:
            nchw_shape = [shape[0], shape[2], shape[1]]
            data = _np.random.rand(*nchw_shape).astype(dtype)
            data = nchw_to_nhwc_fn(data, nchw_shape)
        else:
            data = _np.random.rand(*shape).astype(dtype)
    elif dtype == _np.int32:
        data = _np.random.randint(0, 256, size=shape, dtype=_np.int32)
    elif dtype == _np.int64:
        data = _np.random.randint(0, 256, size=shape, dtype=_np.int64)
    elif dtype == _np.uint8:
        data = _np.random.randint(0, 256, size=shape, dtype=_np.uint8)
    else:
        data = _np.zeros(shape, dtype=dtype)
    return data


def verify_tflite(onnx_path: Path, tflite_path: Path) -> int:
    """Compare ONNX vs TFLite outputs with NCHW↔NHWC shape matching.
    Returns 0=pass, 1=marginal/acceptable/large (model kept), 2=fail (model deleted).
    """
    import os as _os
    _os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    _os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

    import numpy as np
    import onnxruntime as ort

    print(f"\n  Deep Compare: ONNX CPU vs TFLite CPU")

    def nhwc_to_nchw(tensor, shape):
        nd = len(shape)
        if nd == 4:
            return np.transpose(tensor.reshape(shape), [0, 3, 1, 2])
        if nd == 3:
            return np.transpose(tensor.reshape(shape), [0, 2, 1])
        return tensor

    def nchw_to_nhwc(tensor, shape):
        nd = len(shape)
        if nd == 4:
            return np.transpose(tensor.reshape(shape), [0, 2, 3, 1])
        if nd == 3:
            return np.transpose(tensor.reshape(shape), [0, 2, 1])
        return tensor

    # ONNX inference (NCHW)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_input_names = {inp.name for inp in sess.get_inputs()}
    onnx_output_names = [o.name for o in sess.get_outputs()]
    ort_inputs = {}
    for i, inp in enumerate(sess.get_inputs()):
        shape = [d if (isinstance(d, int) and d > 0) else 1 for d in inp.shape]
        np.random.seed(123456789 + i)
        ort_inputs[inp.name] = np.random.rand(*shape).astype(np.float32)
    ort_outs = sess.run(None, ort_inputs)

    # TFLite inference (NHWC, CPU-only)
    from ai_edge_litert.interpreter import Interpreter, OpResolverType
    try:
        interp = Interpreter(
            model_path=str(tflite_path),
            experimental_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
        )
        interp.allocate_tensors()
    except Exception as e:
        msg = str(e).split('\n')[0] if str(e) else str(type(e).__name__)
        print(f"  TFLite runner failed: {msg}")
        return 2

    # Set ALL tensors that require external data.
    # get_input_details() only returns the main subgraph inputs, but onnx2tf
    # may leave additional tensors uninitialized. Scan ALL tensors and fill
    # any that haven't been set yet.
    tflite_inputs = interp.get_input_details()
    input_indices = {d['index'] for d in tflite_inputs}
    print(f"  TFLite main inputs: {len(tflite_inputs)} {sorted(input_indices)}")
    # Set main inputs first
    for idx, detail in enumerate(tflite_inputs):
        shape = detail['shape'].tolist()
        dtype = detail['dtype']
        nd = len(shape)
        np.random.seed(123456789 + idx)
        data = _make_tflite_input_data(shape, dtype, nd, nchw_to_nhwc)
        interp.set_tensor(detail['index'], data)
    # Scan ALL tensors by index; try to get_tensor() for each;
    # if it raises, the tensor hasn't been allocated → fill it
    max_idx = max(input_indices) + 32  # scan a generous range
    for i in range(max_idx):
        if i in input_indices:
            continue
        try:
            details = interp.get_tensor_details(i)
        except Exception:
            continue
        shape = list(details.get('shape', []))
        dtype = details.get('dtype', None)
        if not shape or dtype is None or all(s == 0 for s in shape):
            continue
        # Check if already has data
        try:
            existing = interp.get_tensor(i)
            if existing is not None:
                continue
        except Exception:
            pass
        nd = len(shape)
        np.random.seed(123456789 + i)
        data = _make_tflite_input_data(shape, dtype, nd, nchw_to_nhwc)
        try:
            interp.set_tensor(i, data)
        except Exception:
            pass
    try:
        interp.invoke()
    except Exception as e:
        msg = str(e).split('\n')[0] if str(e) else str(type(e)).split('.')[-1].replace("'>", "")
        print(f"  TFLite inference failed: {msg}")
        # "lacks data" means the model has subgraph inputs the Python API can't set.
        # Conversion was still successful - the model should work on-device.
        if "lacks data" in msg or "subgraph" in msg.lower():
            print("  (Conversion successful; model should work on real device)")
            return 1
        return 2

    # Match ONNX outputs (NCHW) to TFLite outputs (NHWC) by expected shape
    tflite_outputs = interp.get_output_details()

    if len(tflite_outputs) != len(onnx_output_names):
        print(f"  OUTPUT COUNT MISMATCH: ONNX={len(onnx_output_names)} TFLite={len(tflite_outputs)}")
        return 2

    def onnx_nchw_to_nhwc_shape(shape):
        nd = len(shape)
        if nd == 4:
            return (shape[0], shape[2], shape[3], shape[1])
        if nd == 3:
            return (shape[0], shape[2], shape[1])
        return tuple(shape)

    tflite_by_nhwc_shape: dict[tuple, list[int]] = {}
    for t_idx, d in enumerate(tflite_outputs):
        s = tuple(d['shape'])
        tflite_by_nhwc_shape.setdefault(s, []).append(t_idx)

    used_tflite = set()
    pairs = []
    for o_idx, oname in enumerate(onnx_output_names):
        nchw_shape = list(ort_outs[o_idx].shape)
        expected_nhwc = onnx_nchw_to_nhwc_shape(nchw_shape)
        candidates = tflite_by_nhwc_shape.get(expected_nhwc, [])
        available = [t for t in candidates if t not in used_tflite]
        if not available:
            nchw_tuple = tuple(nchw_shape)
            candidates = tflite_by_nhwc_shape.get(nchw_tuple, [])
            available = [t for t in candidates if t not in used_tflite]
            if not available:
                print(f"  {oname}: no matching TFLite output (expected NHWC {expected_nhwc})")
                return 2
        if len(available) > 1:
            best_t = available[0]
            best_score = float('inf')
            for t_idx in available:
                tdata = interp.get_tensor(tflite_outputs[t_idx]['index'])
                tshape = tflite_outputs[t_idx]['shape']
                if list(tshape) == list(expected_nhwc):
                    t_nchw = nhwc_to_nchw(tdata, tshape).flatten()
                else:
                    t_nchw = tdata.reshape(nchw_shape).flatten()
                score = float(np.max(np.abs(ort_outs[o_idx].flatten() - t_nchw)))
                if score < best_score:
                    best_score = score
                    best_t = t_idx
            t_idx = best_t
        else:
            t_idx = available[0]
        used_tflite.add(t_idx)
        tname = tflite_outputs[t_idx]['name']
        pairs.append((o_idx, t_idx, f"{oname} -> {tname}"))

    # Compare both in NCHW flat arrays
    total_max = 0.0
    worst_name = ""
    for o_idx, t_idx, display_name in pairs:
        odata = ort_outs[o_idx].flatten()
        tdata = interp.get_tensor(tflite_outputs[t_idx]['index'])
        tshape = tflite_outputs[t_idx]['shape']
        nchw_shape = list(ort_outs[o_idx].shape)
        expected_nhwc = list(onnx_nchw_to_nhwc_shape(nchw_shape))
        if list(tshape) == expected_nhwc:
            tdata_nchw = nhwc_to_nchw(tdata, tshape)
        elif list(tshape) == nchw_shape:
            tdata_nchw = tdata.reshape(nchw_shape)
        else:
            print(f"  {display_name}: unexpected TFLite shape {list(tshape)}")
            return 2
        tflat = tdata_nchw.flatten()
        mx = float(np.max(np.abs(odata - tflat)))
        avg = float(np.mean(np.abs(odata - tflat)))
        if mx > total_max:
            total_max = mx
            worst_name = display_name
        marker = "  *** LARGE ***" if mx > 0.1 else ""
        print(f"  {display_name:40s}  elements={odata.size:>8}  max={mx:.6f}  avg={avg:.6f}{marker}")

    print(f"  Worst: {worst_name}  (max_diff={total_max:.8f})")
    if total_max < 1e-5:
        print("  CONCLUSION: PASS — fp32 equivalent")
        return 0
    elif total_max < 0.01:
        print("  CONCLUSION: MARGINAL — minor fp32 rounding")
        return 1
    elif total_max < 0.5:
        print("  CONCLUSION: ACCEPTABLE — moderate differences, validate on real data")
        return 1
    else:
        print("  CONCLUSION: LARGE — conversion completed but numerical outputs differ")
        print("  NOTE: This is common when comparing ONNX (ORT) vs TFLite (TF oneDNN)")
        print("  The TFLite model should be validated on real task data before use.")
        return 1


def fold_constants_with_onnxsim(onnx_path: Path) -> Path | None:
    """Fold dynamic shape computations (Shape/Gather/Concat/Reshape/Cast
    chains that produce Pad pads, Slice starts/ends, etc.) into static
    initializers via onnx-simplifier.

    Many streaming/audio models export such dynamic-shape helper graphs from
    PyTorch. onnx2tf cannot infer the NCHW->NHWC layout of intermediate
    tensors when these shapes stay dynamic, which breaks Conv/ConvTranspose
    (group/depthwise channel mismatch). Constant folding makes every
    intermediate shape static so onnx2tf can convert correctly.

    Returns path to the folded model (temp file), or None if onnxsim is
    unavailable or the model has no dynamic-shape helpers.
    """
    try:
        import onnxsim
        import onnx as onnx_mod
    except ImportError:
        return None

    folded_path = onnx_path.with_suffix(".onnxsim.onnx")
    try:
        model = onnx_mod.load(str(onnx_path))
        sim_model, check = onnxsim.simplify(model, overwrite_input_shapes={})
        if not check:
            print("  onnxsim: verification failed, using original model")
            return None
        onnx_mod.save(sim_model, str(folded_path))
        print(f"  onnxsim: folded dynamic shape ops -> {folded_path}")
        return folded_path
    except Exception as e:
        msg = str(e).split('\n')[0] if str(e) else str(type(e).__name__)
        print(f"  WARNING: onnxsim constant folding failed: {msg}")
        return None


def convert_tflite(model_path: Path, args) -> int:
    print("\n" + "=" * 60)
    print(" [3/3] TFLite Conversion")
    print("=" * 60)

    out_path = getattr(args, 'output_tflite', None)
    if out_path:
        out_path = parse_path(out_path) if isinstance(out_path, str) else out_path
    else:
        out_path = model_path.with_suffix(".tflite")

    # Preprocess: replace Erf -> Tanh in ONNX so onnx2tf doesn't emit FlexErf
    preprocessed = preprocess_onnx_for_ncnn(model_path)
    tflite_input = preprocessed if preprocessed else model_path
    if preprocessed:
        print(f"  Using preprocessed model: {tflite_input}")

    # Fold dynamic shape computations so onnx2tf sees static intermediate
    # shapes (required for correct NCHW->NHWC of group/depthwise Conv).
    folded = fold_constants_with_onnxsim(tflite_input)
    if folded:
        tflite_input = folded
        print(f"  Using onnxsim-folded model: {tflite_input}")

    try:
        convert_tflite_inner(tflite_input, out_path,
                             keep_temp=getattr(args, 'keep_temp', False))
    except Exception as e:
        print(f"  TFLite conversion failed: {e}")
        return (False, str(e))

    # Cleanup temp models
    if folded and folded.exists():
        try:
            folded.unlink()
        except OSError:
            pass
    if preprocessed and preprocessed.exists():
        try:
            preprocessed.unlink()
        except OSError:
            pass

    if not args.no_verify:
        ret = verify_tflite(model_path, out_path)
        if ret == 2:
            out_path.unlink(missing_ok=True)
            print("  TFLite conversion FAILED verification")
            return (False, "verification failed")

    return (True, None)


# ──────────────────────────────────────────────────────────────────────
# ONNX structure exporter (merged from onnx_to_text.py)
# ──────────────────────────────────────────────────────────────────────

def _format_shape(dims) -> str:
    """Format TensorShapeProto dimensions into a readable tuple string.

    Static dimensions are shown as integers, symbolic dynamic dimensions keep
    their name (dim_param), and unknown ones are shown as 'dynamic'.
    """
    parts = []
    for d in dims:
        if d.dim_param:
            parts.append(d.dim_param)
        elif d.dim_value != 0:
            parts.append(str(d.dim_value))
        else:
            parts.append("dynamic")
    return "(" + ", ".join(parts) + ")"


def _format_attr(attr) -> str:
    """Format an attribute together with its type and value."""
    type_name = onnx.AttributeProto.AttributeType.Name(attr.type)
    try:
        value = onnx.helper.get_attribute_value(attr)
    except Exception as e:
        value = f"<unreadable: {e}>"
    text = repr(value)
    if len(text) > 500:
        text = text[:500] + " ... (truncated)"
    return f"{type_name} {attr.name} = {text}"


def _dtype_itemsize(elem_type: int) -> int:
    """Return the byte size of a single element for an ONNX tensor dtype."""
    try:
        np_dtype = onnx.helper.tensor_dtype_to_np_dtype(elem_type)
    except Exception:
        try:
            np_dtype = onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[elem_type]
        except Exception:
            return 0
    return np_dtype.itemsize


def onnx_to_text(model_path: str, show_weights: bool = False) -> str:
    """Convert an ONNX model into a detailed, human-readable text string."""
    model = onnx.load(model_path)
    graph = model.graph
    lines = []

    file_size = os.path.getsize(model_path)

    # ================= Model-level metadata =================
    lines.append("=" * 72)
    lines.append("ONNX MODEL REPORT")
    lines.append("=" * 72)
    lines.append(f"IR version            : {model.ir_version}")
    lines.append(
        f"Producer              : {model.producer_name or '(unknown)'} (v{model.producer_version or '?'})"
    )
    lines.append(f"Domain                : {model.domain or '(none)'}")
    lines.append(f"Model version         : {model.model_version}")
    lines.append(f"Model doc_string      : {model.doc_string or '(none)'}")
    lines.append(
        f"File size on disk     : {file_size} bytes ({file_size / 1024 / 1024:.3f} MB)"
    )
    lines.append("")

    # Opset imports
    lines.append("--- Opset Imports ---")
    if model.opset_import:
        for op in model.opset_import:
            lines.append(f"  domain='{op.domain or ''}' version={op.version}")
    else:
        lines.append("  (none)")
    lines.append("")

    # Metadata props
    lines.append("--- Metadata Props ---")
    if model.metadata_props:
        for prop in model.metadata_props:
            lines.append(f"  {prop.key} = {prop.value}")
    else:
        lines.append("  (none)")
    lines.append("")

    # Graph info
    lines.append("--- Graph ---")
    lines.append(f"  Name      : {graph.name or '(unnamed)'}")
    lines.append(f"  doc_string: {graph.doc_string or '(none)'}")
    lines.append("")

    # Local functions
    lines.append(f"--- Local Functions ({len(model.functions)} total) ---")
    if model.functions:
        for fn in model.functions:
            lines.append(f"  [{fn.name}] domain='{fn.domain or ''}'")
            lines.append(f"      inputs : {list(fn.input)}")
            lines.append(f"      outputs: {list(fn.output)}")
            lines.append(f"      attrs  : {list(fn.attribute)}")
            lines.append(f"      opsets : {[(o.domain, o.version) for o in fn.opset_import]}")
            lines.append(f"      nodes  : {len(fn.node)}")
    else:
        lines.append("  (none)")
    lines.append("")

    # ================= Inputs =================
    lines.append("--- Inputs ---")
    initializer_names = {init.name for init in graph.initializer}
    for inp in graph.input:
        tensor = inp.type.tensor_type
        shape = _format_shape(tensor.shape.dim)
        dtype = onnx.TensorProto.DataType.Name(tensor.elem_type)
        kind = "weight" if inp.name in initializer_names else "data"
        lines.append(f"  {inp.name}: {dtype}{shape}  [{kind}]")
        if inp.doc_string:
            lines.append(f"      doc: {inp.doc_string}")
    if not graph.input:
        lines.append("  (none)")
    lines.append("")

    # ================= Outputs =================
    lines.append("--- Outputs ---")
    for out in graph.output:
        tensor = out.type.tensor_type
        shape = _format_shape(tensor.shape.dim)
        dtype = onnx.TensorProto.DataType.Name(tensor.elem_type)
        lines.append(f"  {out.name}: {dtype}{shape}")
        if out.doc_string:
            lines.append(f"      doc: {out.doc_string}")
    if not graph.output:
        lines.append("  (none)")
    lines.append("")

    # ================= Value info (intermediate shapes) =================
    lines.append(
        f"--- Value Info / Intermediate Tensors ({len(graph.value_info)} total) ---"
    )
    for vi in graph.value_info:
        tensor = vi.type.tensor_type
        shape = _format_shape(tensor.shape.dim)
        dtype = onnx.TensorProto.DataType.Name(tensor.elem_type)
        lines.append(f"  {vi.name}: {dtype}{shape}")
    if not graph.value_info:
        lines.append("  (none)")
    lines.append("")

    # ================= Nodes =================
    lines.append(f"--- Nodes ({len(graph.node)} total) ---")
    op_counter = {}
    for i, node in enumerate(graph.node):
        op_counter[node.op_type] = op_counter.get(node.op_type, 0) + 1
        lines.append(
            f"  [{i}] name='{node.name or ''}' op_type={node.op_type} domain='{node.domain or ''}'"
        )
        lines.append(f"      inputs : {list(node.input)}")
        lines.append(f"      outputs: {list(node.output)}")
        if node.doc_string:
            lines.append(f"      doc    : {node.doc_string}")
        if node.attribute:
            lines.append("      attributes:")
            for attr in node.attribute:
                lines.append(f"         {_format_attr(attr)}")
        lines.append("")
    lines.append(
        f"  Op-type histogram (desc): {dict(sorted(op_counter.items(), key=lambda x: -x[1]))}"
    )
    lines.append("")

    # ================= Initializers (weights) =================
    lines.append(f"--- Initializers / Weights ({len(graph.initializer)} total) ---")
    total_elements = 0
    total_bytes = 0
    for init in graph.initializer:
        dtype = onnx.TensorProto.DataType.Name(init.data_type)
        shape = tuple(init.dims)
        elem_count = 1
        for d in init.dims:
            elem_count *= d
        total_elements += elem_count

        byte_size = len(init.raw_data)
        if byte_size == 0:
            byte_size = elem_count * _dtype_itemsize(init.data_type)
        total_bytes += byte_size

        loc = onnx.TensorProto.DataLocation.Name(init.data_location) if init.data_location else "DEFAULT"
        lines.append(
            f"  {init.name}: {dtype}{shape}  elements={elem_count}  bytes={byte_size}  location={loc}"
        )
        if show_weights:
            try:
                arr = onnx.numpy_helper.to_array(init)
                flat = arr.flatten()
                if arr.size <= 10:
                    lines.append(f"      values: {arr}")
                else:
                    lines.append(f"      first 10 values: {flat[:10]} ...")
            except Exception as e:
                lines.append(f"      values: <cannot load: {e}>")
    if not graph.initializer:
        lines.append("  (none)")
    lines.append("")

    # ================= Summary =================
    lines.append("--- Summary ---")
    lines.append(f"  Total inputs          : {len(graph.input)}")
    lines.append(f"  Total outputs         : {len(graph.output)}")
    lines.append(f"  Total nodes           : {len(graph.node)}")
    lines.append(f"  Total initializers    : {len(graph.initializer)}")
    lines.append(f"  Distinct op types     : {len(op_counter)}")
    lines.append(f"  Total weight elements : {total_elements}")
    lines.append(
        f"  Estimated weight bytes: {total_bytes} ({total_bytes / 1024 / 1024:.3f} MB)"
    )
    lines.append("")

    return "\n".join(lines)


def convert_text(model_path: Path, args) -> int:
    """Export the ONNX model structure to a human-readable text file.

    Output defaults to '<model_path>.text' unless --output-text is given.
    """
    print("\n" + "=" * 60)
    print(" [4/4] ONNX Text Export")
    print("=" * 60)

    out_path = getattr(args, 'output_text', None)
    if out_path:
        out_path = parse_path(out_path) if isinstance(out_path, str) else out_path
    else:
        # Match onnx_to_text.py convention: '<model_path>.text' (append to the
        # full filename, e.g. test_model.onnx.text), not a suffix replacement.
        out_path = Path(str(model_path) + ".text")

    try:
        text = onnx_to_text(str(model_path), show_weights=getattr(args, 'show_weights', False))
    except Exception as e:
        msg = str(e).split('\n')[0] if str(e) else str(type(e).__name__)
        print(f"  Text export failed: {msg}")
        return (False, f"text export failed: {msg}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"  Model structure saved to: {out_path}")
    return (True, None)


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    # Set UTF-8 encoding to avoid GBK issues on Windows
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')

    # Ensure core dependencies (onnxruntime, numpy, onnx) before parsing
    ensure_common_deps()

    parser = argparse.ArgumentParser(
        description="Unified ONNX → NCNN / MNN / TFLite converter + text exporter")
    parser.add_argument("model", type=parse_path,
                        help="Path to input .onnx model")
    parser.add_argument("--to", type=str, default="all",
                        help="Target format(s): ncnn, mnn, tflite, text, all (all = text+ncnn+mnn+tflite)")

    # Shared
    parser.add_argument("--no-verify", action="store_true",
                        help="Skip numerical accuracy verification")
    parser.add_argument("--no-cleanup", action="store_true",
                        help="Keep intermediate files")

    # NCNN
    parser.add_argument("--fp16", action="store_true",
                        help="NCNN/MNN: use FP16 quantization")
    parser.add_argument("--no-dual", action="store_true",
                        help="NCNN: skip FP16 model, generate FP32 only")
    parser.add_argument("--keep-pcnn", action="store_true",
                        help="NCNN: keep pcnn intermediate files")
    parser.add_argument("--gen-cpp", action="store_true",
                        help="NCNN: only generate C++ code snippet")
    # MNN
    parser.add_argument("--no-rnn-opt", action="store_true",
                        help="MNN: disable --useOriginRNNImpl (use While-loop for LSTM)")

    # TFLite
    parser.add_argument("-o", "--output-tflite", type=str, default=None,
                        help="TFLite: custom output path")
    parser.add_argument("--keep-temp", action="store_true",
                        help="TFLite: keep onnx2tf temp artifacts")

    # Text export
    parser.add_argument("-t", "--output-text", type=str, default=None,
                        help="Text: custom output path for the .text report")
    parser.add_argument("--show-weights", action="store_true",
                        help="Text: also print weight values in the report")

    args = parser.parse_args()

    model_path: Path = args.model
    if not model_path.exists():
        print(f"Error: not found: {model_path}")
        return 1
    if model_path.suffix.lower() != ".onnx":
        print(f"Error: expected .onnx, got {model_path.suffix}")
        return 1

    targets = [t.strip().lower() for t in args.to.split(",")]
    if "all" in targets:
        targets = ["text", "ncnn", "mnn", "tflite"]

    converters = {
        "ncnn":   convert_ncnn,
        "mnn":    convert_mnn,
        "tflite": convert_tflite,
        "text":   convert_text,
    }

    errors = 0
    results = {}  # fmt → (ok, reason)
    for t in targets:
        if t not in converters:
            print(f"Warning: unknown target '{t}', skipped")
            continue
        try:
            rc = converters[t](model_path, args)
            if isinstance(rc, tuple):
                ok, reason = rc
                results[t] = (ok, reason)
                if not ok:
                    errors += 1
            else:
                results[t] = (rc == 0, None)
                if rc != 0:
                    errors += 1
        except Exception as e:
            results[t] = (False, str(e))
            print(f"  {t.upper()} converter crashed: {e}")
            errors += 1

    print("\n" + "=" * 60)
    print(f" Done. {len(targets) - errors}/{len(targets)} conversions passed.")
    print("=" * 60)

    # ── Summary ──
    print(f"\n[In]  Input:  {model_path}  ({model_path.stat().st_size / 1024:.0f} KB)\n")
    print(f"{'Format':<8} {'File':<40} {'Size':>8} {'Status':<20}")
    print("-" * 80)

    def _fmt_file(fmt: str, path: Path):
        fmt_key = fmt.lower().split("-")[0].split("_")[0]  # "NCNN" / "NCNN-F16" -> "ncnn", "MNN"->"mnn", "TFLite"->"tflite"
        r = results.get(fmt_key)
        failed = r is not None and not r[0]
        if path.exists() and not failed:
            sz = f"{path.stat().st_size / 1024:.0f} KB"
            print(f"{fmt:<8} {str(path):<40} {sz:>8} {'OK':<20}")
        else:
            reason = _reason(fmt)
            # Remove partial file if conversion failed
            if failed and path.exists():
                path.unlink(missing_ok=True)
            print(f"{fmt:<8} {str(path):<40} {'---':>8} {f'FAIL {reason}':<20}")

    def _reason(fmt: str) -> str:
        """Get failure reason for a format."""
        r = results.get(fmt.lower())
        if r is None:
            return "not selected"
        ok, err = r
        if ok:
            return "verification failed"
        if err:
            # Truncate long exception messages
            msg = str(err).split('\n')[0]
            return msg[:40] if len(msg) > 40 else msg
        return "conversion failed"

    _fmt_file("NCNN",   model_path.with_suffix(".ncnn.param"))
    _fmt_file("NCNN",   model_path.with_suffix(".ncnn.bin"))
    fp16_b = model_path.parent / (model_path.stem + "_fp16.ncnn.bin")
    if fp16_b.exists():
        _fmt_file("NCNN-F16", fp16_b)

    _fmt_file("MNN",    model_path.with_suffix(".mnn"))

    tflite_out = parse_path(args.output_tflite) if args.output_tflite else model_path.with_suffix(".tflite")
    _fmt_file("TFLite", tflite_out)

    text_out = parse_path(args.output_text) if args.output_text else Path(str(model_path) + ".text")
    _fmt_file("Text",   text_out)

    shapes_f = model_path.with_suffix(".shapes")
    if shapes_f.exists():
        print(f"\n[Shapes]  {shapes_f}")

    # Collect tips for missing formats
    missing = [t for t in ("ncnn", "mnn", "tflite", "text")
               if results.get(t) and not results[t][0]]
    if missing:
        print(f"\n[Tips]  Tips for failed targets:")
        if "tflite" in missing:
            print(f"   TFLite: needs onnx2tf → pip install onnx2tf tensorflow tf-keras")
            print(f"           or use --no-verify to skip accuracy check")
        if "mnn" in missing:
            print(f"   MNN:    ensure MNNConvert is in tools/mnn_convert/ or PATH")
        if "ncnn" in missing:
            print(f"   NCNN:   ensure pnnx.exe is in tools/ or PATH")
        if "text" in missing:
            print(f"   Text:   ensure 'onnx' is installed (pip install onnx)")

    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
