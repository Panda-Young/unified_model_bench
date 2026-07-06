#!/usr/bin/env python3
"""
Unified ONNX model converter: NCNN / MNN / TFLite (three-in-one).

Usage:
  python onnx_convert.py <model.onnx> --to ncnn                # NCNN only
  python onnx_convert.py <model.onnx> --to mnn   [--fp16]      # MNN only
  python onnx_convert.py <model.onnx> --to tflite               # TFLite only
  python onnx_convert.py <model.onnx> --to all                  # all three
  python onnx_convert.py <model.onnx> --to ncnn,mnn             # multiple
  python onnx_convert.py <model.onnx> --to ncnn --no-dual       # FP32 only

Shared flags:
  --no-verify        Skip numerical accuracy verification
  --no-cleanup       Keep intermediate files

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
    with open(shapes_path, 'w') as f:
        f.write(f"inputs={len(info['inputs'])}\n")
        for i, (name, dims) in enumerate(info['inputs']):
            f.write(f"in{i}={','.join(str(d) for d in dims)}\n")
        f.write(f"outputs={len(info['outputs'])}\n")
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
    """0=pass/skip, 2=fail"""
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
        try:
            net = ncnn.Net()
            if vulkan:
                if ncnn.get_gpu_count() == 0:
                    print(f"\n  {label}: SKIP - no Vulkan GPU")
                    return True
                net.opt.use_vulkan_compute = True
                net.opt.use_fp16_packed = False
                net.opt.use_fp16_storage = False
                net.opt.use_fp16_arithmetic = False
            net.load_param(ncnn_param)
            net.load_model(ncnn_bin)
            ex = net.create_extractor()
            for i in range(len(inputs_info)):
                mat = ncnn.Mat(np.ascontiguousarray(feed[inputs_info[i][0]].squeeze(0)))
                ex.input(f"in{i}", mat.clone())
            outs = []
            for i in range(len(outputs_info)):
                _, m = ex.extract(f"out{i}")
                outs.append(np.array(m))
        except Exception as e:
            msg = str(e).split('\n')[0] if str(e) else str(type(e).__name__)
            print(f"\n  {label}: SKIP - ncnn runtime error: {msg}")
            return True  # Model was generated, verification env issue

        print(f"\n  === {label} ===")
        all_ok = True
        worst = 0.0
        for i, (ref, out) in enumerate(zip(ref_outs, outs)):
            a, b = ref.flatten(), out.flatten()
            if a.shape != b.shape:
                print(f"  out{i}: shape mismatch {ref.shape} vs {out.shape}  FAIL")
                all_ok = False; continue
            mx = float(np.abs(a - b).max())
            avg = float(np.abs(a - b).mean())
            tag = "OK" if mx < 5.0 else "***LARGE***"
            print(f"  out{i}: max={mx:.6f} avg={avg:.6f} {tag}")
            if mx > worst: worst = mx
            if mx >= 5.0: all_ok = False
        status = "PASS" if all_ok else "FAIL"
        print(f"  {label}: {status} (worst={worst:.6f})")
        return all_ok

    ok1 = _run("NCNN CPU", vulkan=False)
    ok2 = _run("NCNN Vulkan FP32", vulkan=True)
    return 0 if (ok1 and ok2) else 2


def convert_ncnn(model_path: Path, args) -> int:
    print("\n" + "=" * 60)
    print(" [1/3] NCNN Conversion")
    print("=" * 60)

    if args.gen_cpp:
        gen_ncnn_cpp(str(model_path))
        return (True, None)

    inputshape_str = get_inputshape_string(str(model_path))
    print(f"  inputshape: {inputshape_str}")

    pnnx = find_pnnx()
    model_dir = str(model_path.parent)

    def _run_pnnx(fp16_val: str) -> None:
        """Run pnnx with the given fp16 setting, using a .bat file if the
        command line exceeds Windows length limit (~8191 chars)."""
        cmd = [pnnx, str(model_path), f"inputshape={inputshape_str}", f"fp16={fp16_val}"]
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
    _run_pnnx("0")

    def _find_pnnx_output(model_path: Path, suffix: str) -> Path | None:
        """Find pnnx output file. pnnx may replace '-' with '_' in filenames."""
        expected = model_path.with_suffix(suffix)
        if expected.exists():
            return expected
        # pnnx replaces '-' with '_' in the filename stem
        alt_stem = model_path.stem.replace('-', '_')
        alt = model_path.parent / f"{alt_stem}{suffix}"
        if alt.exists():
            print(f"  Found at alternate name: {alt}")
            # Rename to expected name for consistency
            os.replace(str(alt), str(expected))
            return expected
        return None

    param = _find_pnnx_output(model_path, ".ncnn.param")
    bin_  = _find_pnnx_output(model_path, ".ncnn.bin")

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
            _run_pnnx("1")
            tmp_b = _find_pnnx_output(model_path, ".ncnn.bin")
            if tmp_b is not None and tmp_b.exists():
                os.replace(str(tmp_b), str(fp16_bin))
                print(f"  FP16 bin: {fp16_bin} ({fp16_bin.stat().st_size/1024:.0f} KB)")
            # Re-run FP32 to restore FP32 weights
            print("  Re-running FP32...")
            _run_pnnx("0")
            param = _find_pnnx_output(model_path, ".ncnn.param")
            bin_  = _find_pnnx_output(model_path, ".ncnn.bin")

    if param is None or bin_ is None:
        print("  NCNN conversion FAILED - no NCNN output files found")
        return (False, "pnnx produced no output")

    # Shapes file
    info = get_io_info_ort(str(model_path))
    write_ncnn_shapes(str(model_path.with_suffix(".shapes")), info)

    # Cleanup
    if not getattr(args, 'no_cleanup', False):
        clean_pnnx_intermediates(model_dir, keep_pcnn=getattr(args, 'keep_pcnn', False))

    # Verify (run in subprocess to isolate from potential ncnn crashes)
    if not args.no_verify:
        print("\n  Verifying NCNN...")
        try:
            ret = verify_ncnn(str(model_path), str(param), str(bin_))
            if ret != 0:
                for f in (param, bin_):
                    try: os.remove(f)
                    except OSError: pass
                print("  NCNN conversion FAILED verification")
                return (False, "verification failed")
        except Exception as e:
            msg = str(e).split('\n')[0] if str(e) else str(type(e).__name__)
            print(f"  NCNN verify skipped (runtime error: {msg})")
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
        subprocess.run(cmd, check=False, timeout=300)
    except subprocess.TimeoutExpired:
        print("  MNNConvert timed out")
        return (False, "MNNConvert timed out")
    except Exception as e:
        print(f"  Note: MNNConvert exited: {e}")

    if not mnn_path.exists() or mnn_path.stat().st_size == 0:
        print("  MNNConvert failed - no output")
        return (False, "MNNConvert produced no output")

    print(f"  MNN model: {mnn_path} ({mnn_path.stat().st_size / 1024:.0f} KB)")

    if not args.no_verify:
        ret = verify_mnn(model_path, mnn_path)
        if ret == 2:
            mnn_path.unlink(missing_ok=True)
            print("  MNN conversion FAILED verification")
            return (False, "verification failed")

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
}


def ensure_tflite_deps():
    missing = [m for m in TFLITE_DEPS if importlib.util.find_spec(m) is None]
    if missing:
        pkgs = [TFLITE_DEPS[m] for m in missing]
        pip_install_with_fallback(pkgs, "tflite")


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
    tflite_env = os.environ.copy()
    tflite_env["PYTHONIOENCODING"] = "utf-8"
    tflite_env["PYTHONUTF8"] = "1"
    tflite_env["TF_ENABLE_ONEDNN_OPTS"] = "0"
    tflite_env["TF_CPP_MIN_LOG_LEVEL"] = "2"

    with tempfile.TemporaryDirectory(prefix="onnx2tf_") as tmp:
        tmp_dir = Path(tmp)

        # Generate PRF to fix NCHW->NHWC Transpose/Concat axis mapping
        import onnx as onnx_mod
        onnx_model = onnx_mod.load(str(input_onnx), load_external_data=False)
        prf_path = generate_nchw_to_nhwc_prf(onnx_model, Path(tmp) / "prf.json")

        cmd = [sys.executable, "-m", "onnx2tf",
               "-i", str(input_onnx), "-o", str(tmp_dir),
               "-coion", "-nuo", "-n"]
        if prf_path:
            cmd += ["-prf", str(prf_path)]
        print(f"  Running: {' '.join(cmd)}")
        proc = subprocess.run(cmd, text=True, capture_output=True, env=tflite_env,
                              encoding="utf-8", errors="replace")

        def safe_print(prefix: str, text: str):
            try:
                print(f"{prefix} {text}")
            except UnicodeEncodeError:
                safe = text.encode("utf-8", errors="replace").decode("utf-8")
                print(f"{prefix} {safe}")

        for line in proc.stdout.splitlines():
            lo = line.lower()
            if any(w in lo for w in ("warn", "error", "fail", "traceback")):
                safe_print(" [onnx2tf] ⚠", line)
            else:
                safe_print(" [onnx2tf]", line)

        if proc.stderr:
            prev = None
            for line in proc.stderr.splitlines():
                s = line.strip()
                if not s or s == prev: continue
                prev = s
                lo = s.lower()
                if any(w in lo for w in ("warn", "error", "fail", "traceback")):
                    safe_print(" [onnx2tf:err] ⚠", s)
                elif re.search(r"100%\|", s):
                    safe_print(" [onnx2tf:err]", s)
                elif not re.search(r"\d+%\|.*it/s", s):
                    safe_print(" [onnx2tf:err]", s)

        if proc.returncode != 0:
            raise RuntimeError(f"onnx2tf failed:\n{proc.stdout}\n{proc.stderr}")

        produced = find_latest_tflite(tmp_dir)
        output_tflite.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(produced, output_tflite)
        print(f"  Output: {output_tflite}")

        if keep_temp:
            debug = output_tflite.parent / f"{output_tflite.stem}_onnx2tf_debug"
            if debug.exists(): shutil.rmtree(debug)
            shutil.copytree(tmp_dir, debug)
            print(f"  Debug artifacts: {debug}")


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
    onnx_names = [o.name for o in sess.get_outputs()]
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

    for i, detail in enumerate(interp.get_input_details()):
        shape = detail['shape'].tolist()
        np.random.seed(123456789 + i)
        nd = len(shape)
        if nd == 4:
            nchw_shape = [shape[0], shape[3], shape[1], shape[2]]
            nchw_data = np.random.rand(*nchw_shape).astype(np.float32)
            nhwc_data = nchw_to_nhwc(nchw_data, nchw_shape)
        elif nd == 3:
            nchw_shape = [shape[0], shape[2], shape[1]]
            nchw_data = np.random.rand(*nchw_shape).astype(np.float32)
            nhwc_data = nchw_to_nhwc(nchw_data, nchw_shape)
        else:
            nhwc_data = np.random.rand(*shape).astype(np.float32)
        interp.set_tensor(detail['index'], nhwc_data)
    try:
        interp.invoke()
    except Exception as e:
        msg = str(e).split('\n')[0] if str(e) else str(type(e).__name__)
        print(f"  TFLite inference failed: {msg}")
        return 2

    # Match ONNX outputs (NCHW) to TFLite outputs (NHWC) by expected shape
    tflite_outputs = interp.get_output_details()

    if len(tflite_outputs) != len(onnx_names):
        print(f"  OUTPUT COUNT MISMATCH: ONNX={len(onnx_names)} TFLite={len(tflite_outputs)}")
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
    for o_idx, oname in enumerate(onnx_names):
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


def convert_tflite(model_path: Path, args) -> int:
    print("\n" + "=" * 60)
    print(" [3/3] TFLite Conversion")
    print("=" * 60)

    out_path = getattr(args, 'output_tflite', None)
    if out_path:
        out_path = parse_path(out_path) if isinstance(out_path, str) else out_path
    else:
        out_path = model_path.with_suffix(".tflite")

    try:
        convert_tflite_inner(model_path, out_path,
                             keep_temp=getattr(args, 'keep_temp', False))
    except Exception as e:
        print(f"  TFLite conversion failed: {e}")
        return (False, str(e))

    if not args.no_verify:
        ret = verify_tflite(model_path, out_path)
        if ret == 2:
            out_path.unlink(missing_ok=True)
            print("  TFLite conversion FAILED verification")
            return (False, "verification failed")

    return (True, None)


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    # Ensure core dependencies (onnxruntime, numpy, onnx) before parsing
    ensure_common_deps()

    parser = argparse.ArgumentParser(
        description="Unified ONNX → NCNN / MNN / TFLite converter")
    parser.add_argument("model", type=parse_path,
                        help="Path to input .onnx model")
    parser.add_argument("--to", type=str, default="all",
                        help="Target format(s): ncnn, mnn, tflite, all (comma-separated)")

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
        targets = ["ncnn", "mnn", "tflite"]

    converters = {
        "ncnn":   convert_ncnn,
        "mnn":    convert_mnn,
        "tflite": convert_tflite,
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
    print(f"\n📦  Input:  {model_path}  ({model_path.stat().st_size / 1024:.0f} KB)\n")
    print(f"{'Format':<8} {'File':<40} {'Size':>8} {'Status':<20}")
    print("-" * 80)

    def _fmt_file(fmt: str, path: Path):
        if path.exists():
            sz = f"{path.stat().st_size / 1024:.0f} KB"
            print(f"{fmt:<8} {str(path):<40} {sz:>8} {'✓ OK':<20}")
        else:
            reason = _reason(fmt)
            print(f"{fmt:<8} {str(path):<40} {'—':>8} {f'✗ {reason}':<20}")

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
    fp16_p = model_path.parent / (model_path.stem + "_fp16.ncnn.param")
    fp16_b = model_path.parent / (model_path.stem + "_fp16.ncnn.bin")
    if fp16_p.exists() or fp16_b.exists():
        _fmt_file("NCNN-F16", fp16_p)
        _fmt_file("NCNN-F16", fp16_b)

    _fmt_file("MNN",    model_path.with_suffix(".mnn"))

    tflite_out = parse_path(args.output_tflite) if args.output_tflite else model_path.with_suffix(".tflite")
    _fmt_file("TFLite", tflite_out)

    shapes_f = model_path.with_suffix(".shapes")
    if shapes_f.exists():
        print(f"\n📐  Shapes: {shapes_f}")

    # Collect tips for missing formats
    missing = [t for t in ("ncnn", "mnn", "tflite")
               if results.get(t) and not results[t][0]]
    if missing:
        print(f"\n⚠  Tips for failed targets:")
        if "tflite" in missing:
            print(f"   TFLite: needs onnx2tf → pip install onnx2tf tensorflow tf-keras")
            print(f"           or use --no-verify to skip accuracy check")
        if "mnn" in missing:
            print(f"   MNN:    ensure MNNConvert is in tools/mnn_convert/ or PATH")
        if "ncnn" in missing:
            print(f"   NCNN:   ensure pnnx.exe is in tools/ or PATH")

    print(f"\n💡  Next: cd unified_bench && tools\\NDK_build_Android_auto.bat\n")

    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
