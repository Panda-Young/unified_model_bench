#!/usr/bin/env python3
"""Generate a QNN HTP context binary (.serialized.bin) with the DSP architecture
taken from the official QNN "HTP TARGET CONFIG TABLE".

Docs:
  https://docs.qualcomm.com/nav/home/htp_auto_single_nsp.html
  "Please refer to the following table to select the appropriate DSP
   architecture ('dsp_arch') value."

Table (Target Name / Target dsp_arch / Target soc_id):
  SA8295 v68 39 | SA8540 v68 62 | SA8650 v73 52 | SA8775 v73 52
  SA8255 v73 52 | SA8620 v75 67 | SA7255 v75 67 | SA8797 v81 72
  (Snapdragon phone targets use the same soc_id/dsp_arch pairs, e.g.
   SM8450 -> v69 / 36.)

The generator selects dsp_arch via the HTP backend-extensions JSON:
  {"devices": [{"soc_id": <int>, "dsp_arch": "<vNN>", "cores": [...]}]}
passed with --config_file, plus --backend libQnnHtp.so and the runtime libs
on LD_LIBRARY_PATH / ADSP_LIBRARY_PATH on the device.

Usage (from the host, with the device connected):
  python tools/gen_qnn_context_binary.py --target SM8450 \
      --model libscnet_tfc_tdf_v3_20260821.so \
      --out scnet_tfc_tdf_v3_20260821.serialized.bin
"""
import argparse
import json
import os
import subprocess
import sys

# soc_id / dsp_arch pairs from the docs table. Phone SoCs are included too
# because the generator's JSON accepts the same pairs.
TARGETS = {
    # --- automotive (docs "HTP TARGET CONFIG TABLE") ---
    "SA8295": (39, "v68"),
    "SA8540": (62, "v68"),
    "SA8650": (52, "v73"),
    "SA8775": (52, "v73"),
    "SA8255": (52, "v73"),
    "SA8620": (67, "v75"),
    "SA7255": (67, "v75"),
    "SA8797": (72, "v81"),
    # --- phones / computes (same rule: soc_id + dsp_arch) ---
    "SM8850": (87, "v81"),
    "SM8750": (69, "v79"),
    "SM8650": (57, "v75"),
    "SM8550": (43, "v73"),
    "SM7750": (86, "v73"),
    "SM8475": (42, "v69"),
    "SM8450": (36, "v69"),
    "SM7450": (41, "v69"),
    "SM8350": (30, "v68"),
    "SM7325": (35, "v68"),
    "SC8280X": (37, "v68"),
}

DEFAULT_DEVICE_DIR = "/data/local/tmp/bench_test"


def resolve_target(name):
    key = name.strip().upper()
    for cand in (key, key + "P"):
        if cand in TARGETS:
            return cand, TARGETS[cand]
    # allow the "P" marketing suffix: SA8295P -> SA8295
    if key.endswith("P") and key[:-1] in TARGETS:
        return key[:-1], TARGETS[key[:-1]]
    return None, None


def build_extensions(soc_id, dsp_arch, hvx_threads=None, o_level=None,
                     vtcm_mb=None):
    graph = {}
    if o_level is not None:
        graph["O"] = float(o_level)
    if vtcm_mb is not None:
        graph["vtcm_mb"] = int(vtcm_mb)
    if hvx_threads is not None:
        graph["hvx_threads"] = int(hvx_threads)
    return {
        "graphs": [graph] if graph else [{}],
        "devices": [{
            "soc_id": int(soc_id),
            "dsp_arch": dsp_arch,
            "cores": [{"core_id": 0, "perf_profile": "burst",
                       "rpc_control_latency": 100}],
        }],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True,
                    help="SoC target name, e.g. SM8450 / SA8295 / SA8797")
    ap.add_argument("--model", required=True,
                    help="model.so on the device (QnnModel_composeGraphs)")
    ap.add_argument("--out", required=True,
                    help="output .serialized.bin name")
    ap.add_argument("--serial", default=None, help="adb serial")
    ap.add_argument("--device-dir", default=DEFAULT_DEVICE_DIR)
    ap.add_argument("--generator",
                    default="/data/local/tmp/bench_test/qnn-context-binary-generator")
    ap.add_argument("--hvx-threads", type=int, default=None)
    ap.add_argument("--o-level", type=int, default=None)
    ap.add_argument("--vtcm-mb", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    canon, pair = resolve_target(args.target)
    if pair is None:
        print("ERROR: unknown target %r. Known: %s"
              % (args.target, ", ".join(sorted(TARGETS))), file=sys.stderr)
        return 2
    soc_id, dsp_arch = pair
    print("[table] %s -> soc_id=%d dsp_arch=%s" % (canon, soc_id, dsp_arch))

    ext = build_extensions(soc_id, dsp_arch, args.hvx_threads,
                           args.o_level, args.vtcm_mb)
    cfg_local = os.path.abspath("htp_backend_extensions.json")
    with open(cfg_local, "w", encoding="utf-8") as f:
        json.dump(ext, f, indent=2)
    print("[config]", cfg_local)
    print(json.dumps(ext, indent=2))

    cfg_dev = args.device_dir.rstrip("/") + "/htp_backend_extensions.json"
    adb = ["adb"] + (["-s", args.serial] if args.serial else [])
    cmds = [
        adb + ["push", cfg_local, cfg_dev],
        adb + ["shell",
               "cd %s && LD_LIBRARY_PATH=.:./qnn ADSP_LIBRARY_PATH=./qnn "
               "%s --model %s --backend ./qnn/libQnnHtp.so "
               "--binary_file %s --config_file htp_backend_extensions.json "
               "--output_dir ./ctx_out"
               % (args.device_dir, args.generator, args.model, args.out)],
        adb + ["shell",
               "cd %s && ls -la ctx_out/" % args.device_dir],
    ]
    for c in cmds:
        print("+", " ".join(c))
        if args.dry_run:
            continue
        r = subprocess.run(c, capture_output=True, text=True)
        if r.stdout:
            print(r.stdout.rstrip())
        if r.returncode != 0:
            sys.stderr.write(r.stderr)
            print("FAILED (rc=%d)" % r.returncode, file=sys.stderr)
            return r.returncode
    return 0


if __name__ == "__main__":
    sys.exit(main())
