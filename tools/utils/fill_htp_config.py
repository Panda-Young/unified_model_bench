#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""fill_htp_config.py - Auto-fill soc_id / dsp_arch in htp_config.json.

QNN HTP config (htp_config.json) needs a "devices" entry with:
    "soc_id":   <int>    -- Real QNN SoC model enum (QNN_SOC_MODEL_* from
                          QnnTypes.h). Filled with the actual value from the
                          SoC model (e.g. SM8550 -> 43, SM8650 -> 57,
                          SM8850 -> 87). Some backend extensions / offline
                          context-binary generation enable SoC-specific
                          optimizations only when a matching soc_id is given.
                          0 means QNN_SOC_MODEL_UNKNOWN (auto fallback).
    "dsp_arch": "<str>"  -- Hexagon DSP arch generation (e.g. SM8550 -> "v73",
                          SM8850 -> "v81"), matching the Skel/stub dir under the
                          QNN SDK. "0" would mean auto.

This script resolves the values from the target device:
    * adb shell getprop ro.soc.model   (default, requires a connected device)
    * or --soc SM8550                  (manual override, no device needed)
and writes them back into htp_config.json (the --config path or the one next
to this script by default).

Usage:
    python tools/utils/fill_htp_config.py                 # read device via adb
    python tools/utils/fill_htp_config.py --soc SM8550    # explicit model
    python tools/utils/fill_htp_config.py --config path/to/htp_config.json
    python tools/utils/fill_htp_config.py --dry-run       # print, don't write
"""

import argparse
import json
import os
import subprocess
import sys

# ---------------------------------------------------------------------------
# Platform -> (soc_id, dsp_arch) lookup.
# Keys are matched case-insensitively against `ro.soc.model` (e.g. "SM8550").
#
#   soc_id   : Real QNN SoC model enum (QNN_SOC_MODEL_* in QnnTypes.h). These
#              values are STABLE per SDK release and are REQUIRED for some
#              backend extensions / SoC-specific optimizations (e.g. offline
#              context-binary generation picks different algorithms when a
#              matching soc_id is given). Source of truth:
#              C:\Qualcomm\AIStack\QAIRT\<ver>\include\QNN\QnnTypes.h
#                SM8550=43  SM8650=57  SM8750=69  SM8850=87
#                SM8450=36  SM8475=42  SM8350=30
#                SM7350=32  SM7450=41  SM7550=64  SM7750=86  SM7635=73
#                SC8280X=37 SC8380XP=60
#              Unknown / unmapped -> soc_id 0 (QNN_SOC_MODEL_UNKNOWN, auto).
#
#   dsp_arch : Hexagon DSP architecture generation string ("vXX"), matching the
#              Skel/stub directory under the QNN SDK (hexagon-vXX). Same mapping
#              already used by NDK_build_Android_auto.bat (HEXVER). This is a
#              STABLE, observable mapping from the SoC model.
# ---------------------------------------------------------------------------
PLATFORM_MAP = {
    # SM8xxx flagships
    "SM8850": (87, "v81"),  # Snapdragon 8 Elite (Gen 4)
    "SM8750": (69, "v79"),  # Snapdragon 8 Gen 3
    "SM8650": (57, "v75"),  # Snapdragon 8 Gen 2
    "SM8550": (43, "v73"),  # Snapdragon 8 Gen 1
    "SM8475": (42, "v73"),  # Snapdragon 8+ Gen 1
    "SM8450": (36, "v69"),  # Snapdragon 8 Gen 1 (base)
    "SM8350": (30, "v68"),  # Snapdragon 888
    # SM7xxx / 6xxx mid-tier
    "SM7750": (86, "v73"),  # Snapdragon 8s Gen 3
    "SM7650": (86, "v73"),  # Snapdragon 8s Gen 4 (shared enum with 7750)
    "SM7635": (73, "v73"),  # Snapdragon 7+ Gen 3
    "SM7550": (64, "v73"),  # Snapdragon 7 Gen 4
    "SM7475": (54, "v69"),  # Snapdragon 7 Gen 3
    "SM7450": (41, "v69"),  # Snapdragon 7 Gen 1
    "SM7350": (32, "v68"),  # Snapdragon 780G
    "SM7325": (35, "v68"),  # Snapdragon 7 Gen 1+ / 782G
    "SM7250": (25, "v68"),  # Snapdragon 765G
    "SM7150": (17, "v65"),  # Snapdragon 765G (base)
    # SM6xxx
    "SM6635": (68, "v69"),  # Snapdragon 6 Gen 4
    "SM6550": (74, "v69"),  # Snapdragon 6 Gen 3
    "SM6450": (50, "v69"),  # Snapdragon 6 Gen 1
    "SM6375": (40, "v68"),  # Snapdragon 6 Gen 1 (base)
    "SM6250": (27, "v66"),  # Snapdragon 695
    "SM6125": (19, "v65"),  # Snapdragon 665
    "SM6115": (12, "v65"),  # Snapdragon 662 (SDM6125 enum)
    # SC / QCM (compute / auto / IoT)
    "SC8380XP": (60, "v81"), # Snapdragon 8cx Gen 4 (X Elite)
    "SC8280X": (37, "v68"),  # Snapdragon 8cx Gen 3
    "SC7280X": (38, "v68"),  # Snapdragon 7c+ Gen 3
    "QCS8550": (66, "v79"),  # QRB / IoT (based on SM8550)
    "QCM6490": (33, "v68"),  # QRB / IoT (shared enum with QCS410)
    "QCS6490": (33, "v68"),  # QRB / IoT
    "QCS8250": (51, "v68"),  # QRB / IoT (QCS7230 enum)
}


def resolve_soc_model(manual=None):
    """Return the SoC model string, or None if it cannot be determined."""
    if manual:
        return manual.strip().upper()
    try:
        out = subprocess.run(
            ["adb", "shell", "getprop", "ro.soc.model"],
            capture_output=True, text=True, timeout=30,
        )
        if out.returncode == 0:
            val = out.stdout.strip()
            # strip stray brackets / whitespace some properties carry
            val = val.strip("[] \t\r\n")
            if val:
                return val.upper()
    except FileNotFoundError:
        print("[warn] adb not found; pass --soc MODEL to skip device detection",
              file=sys.stderr)
    except subprocess.TimeoutExpired:
        print("[warn] adb timed out; pass --soc MODEL to skip device detection",
              file=sys.stderr)
    return None


def lookup(soc_model):
    """Return (soc_id, dsp_arch) for a soc model, or None if unknown."""
    if not soc_model:
        return None
    return PLATFORM_MAP.get(soc_model.upper())


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    default_config = os.path.normpath(
        os.path.join(here, "..", "config", "htp_config.json"))

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--soc", help="SoC model override, e.g. SM8550 (skips adb)")
    ap.add_argument("--config", default=default_config,
                    help="path to htp_config.json (default: tools/config/htp_config.json)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the resolved values without writing the file")
    args = ap.parse_args()

    config_path = os.path.abspath(args.config)
    if not os.path.isfile(config_path):
        print("[error] config not found: %s" % config_path, file=sys.stderr)
        return 1

    soc_model = resolve_soc_model(args.soc)
    print("[info] SoC model: %s" % (soc_model if soc_model else "(unknown)"))

    resolved = lookup(soc_model)
    if resolved is None:
        print("[warn] unknown SoC model %r; cannot resolve a real soc_id. "
              "Keeping soc_id=0 (QNN_SOC_MODEL_UNKNOWN / auto) and "
              "dsp_arch=\"0\" (auto). Add the model to PLATFORM_MAP to fill "
              "the actual values." % soc_model, file=sys.stderr)
        soc_id, dsp_arch = 0, "0"
    else:
        soc_id, dsp_arch = resolved
        print("[info] resolved soc_id=%d, dsp_arch=\"%s\""
              % (soc_id, dsp_arch))

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    devices = cfg.get("devices")
    if not isinstance(devices, list) or not devices:
        print("[warn] no 'devices' array in config; appending one", file=sys.stderr)
        devices = [{}]
        cfg["devices"] = devices

    changed = False
    for dev in devices:
        if dev.get("soc_id") != soc_id:
            dev["soc_id"] = soc_id
            changed = True
        if str(dev.get("dsp_arch")) != str(dsp_arch):
            dev["dsp_arch"] = dsp_arch
            changed = True

    if resolved is None and not args.soc:
        # No device detected and no --soc given: we cannot fill a meaningful
        # value, so leave the file untouched instead of writing auto "0/0".
        print("[skip] no device detected and no --soc given; "
              "connect a device or pass --soc MODEL. File not modified.",
              file=sys.stderr)
        return 0

    if args.dry_run:
        print("[dry-run] would write to %s:" % config_path)
        print(json.dumps(cfg, indent=2))
        return 0

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")
    print("[ok] updated %s (changed=%s)" % (config_path, changed))
    return 0


if __name__ == "__main__":
    sys.exit(main())
