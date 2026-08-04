# coding: utf-8
"""
Description: Generate ONNX model test data (float32 + optional quantized uint8) and corresponding list files
Version: 0.2.2
Author: Panda-Young
Date: 2025-11-24
"""

import os
import numpy as np
import onnxruntime as ort
import re
import argparse


def sanitize_name(name):
    """Sanitize name by replacing invalid characters with underscore."""
    return re.sub(r"[^a-zA-Z0-9_]", "_", name)


def generate_test_data(
    model_path, output_dir="data", path_type="relative", gen_uint8=False, use_fixed_value=False, fixed_float_value=0.5, fixed_uint8_value=64
):
    """Generate float32 inputs and optionally uint8 inputs (quantized from float or random) for an ONNX model."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # create output dir if missing
    if not output_dir:
        output_dir = os.path.dirname(os.path.abspath(model_path)) or "."
    os.makedirs(output_dir, exist_ok=True)

    # load ONNX session
    session = ort.InferenceSession(model_path)

    # get model base name
    model_name = sanitize_name(os.path.splitext(os.path.basename(model_path))[0])

    # collect input configurations
    input_configs = []
    rng = np.random.default_rng()

    for idx, input_info in enumerate(session.get_inputs()):
        name = input_info.name
        shape = input_info.shape
        dtype = input_info.type

        # decide whether to include this input based on its declared ONNX dtype and gen_uint8 flag
        if dtype == "tensor(float)":
            data_type = np.float32
            folder = f"input_data_{model_name}_float32"
        elif dtype == "tensor(uint8)":
            # by default skip original uint8 inputs unless user explicitly requests uint8 generation
            if not gen_uint8:
                print(
                    f"Skipping uint8 input '{name}' (use --gen-uint8 to enable uint8 generation)"
                )
                continue
            data_type = np.uint8
            folder = f"input_data_{model_name}_uint8"
        else:
            print(f"Skipping unsupported input type: {dtype}")
            continue

        clean_name = sanitize_name(name)

        # normalize shape: replace dynamic dims (None or str) with 1 and ensure ints
        fixed_shape = []
        for dim in shape:
            if dim is None or isinstance(dim, str):
                fixed_shape.append(1)
            else:
                fixed_shape.append(int(dim))

        shape_str = "x".join(map(str, fixed_shape))
        filename = f"{clean_name}_{shape_str}.bin"

        input_configs.append(
            {
                "name": name,
                "clean_name": clean_name,
                "shape": fixed_shape,
                "dtype": data_type,
                "filename": filename,
                "folder": folder,
            }
        )

    # create output folders for configured inputs
    for config in input_configs:
        os.makedirs(os.path.join(output_dir, config["folder"]), exist_ok=True)

    # ========== Stage 1: generate float32 data ==========
    float_input_list = []
    float_configs = [c for c in input_configs if c["dtype"] == np.float32]

    for config in float_configs:
        # generate random float32 in [0,1) or fixed value
        if use_fixed_value:
            data = np.full(tuple(config["shape"]), fixed_float_value, dtype=np.float32)
        else:
            data = rng.random(size=tuple(config["shape"])).astype(np.float32)

        # save binary file
        filepath = os.path.join(output_dir, config["folder"], config["filename"])
        data.tofile(filepath)

        float_input_list.append(os.path.join(config["folder"], config["filename"]))

    # ========== Stage 2: uint8 generation (optional) ==========
    uint8_input_list = []
    if gen_uint8:
        # ensure a common uint8 folder for quantized-from-float outputs
        uint8_folder_common = f"input_data_{model_name}_uint8"
        os.makedirs(os.path.join(output_dir, uint8_folder_common), exist_ok=True)

        # 1) quantize float-generated inputs -> uint8
        for config in float_configs:
            float_filepath = os.path.join(
                output_dir, config["folder"], config["filename"]
            )
            float_data = np.fromfile(float_filepath, dtype=np.float32).reshape(
                config["shape"]
            )
            # simple quantization: [0,1] -> [0,255] or use fixed uint8 value
            if use_fixed_value:
                uint8_data = np.full(tuple(config["shape"]), fixed_uint8_value, dtype=np.uint8)
            else:
                uint8_data = (
                    (np.clip(float_data, 0.0, 1.0) * 255.0).round().astype(np.uint8)
                )
            uint8_filepath = os.path.join(
                output_dir, uint8_folder_common, config["filename"]
            )
            uint8_data.tofile(uint8_filepath)
            uint8_input_list.append(
                os.path.join(uint8_folder_common, config["filename"])
            )

        # 2) for inputs that are originally uint8 (included only when gen_uint8 is True), generate random uint8
        original_uint8_configs = [c for c in input_configs if c["dtype"] == np.uint8]
        for config in original_uint8_configs:
            if use_fixed_value:
                data = np.full(tuple(config["shape"]), fixed_uint8_value, dtype=np.uint8)
            else:
                data = rng.integers(0, 256, size=tuple(config["shape"])).astype(np.uint8)
            filepath = os.path.join(output_dir, config["folder"], config["filename"])
            data.tofile(filepath)
            relpath = os.path.join(config["folder"], config["filename"])
            if relpath not in uint8_input_list:
                uint8_input_list.append(relpath)

    # ========== Write list files ==========
    def format_paths_for_list(entries):
        """Format list entries as relative (to output_dir) or absolute paths."""
        formatted = []
        for p in entries:
            if path_type == "absolute":
                abs_p = os.path.abspath(os.path.join(output_dir, p))
                formatted.append(abs_p.replace("\\", "/"))
            else:
                formatted.append(p.replace("\\", "/"))
        return formatted

    def write_list_file(filename, entries, header):
        """Write one input path per line; '# ' lines are comments, so the
        file is compatible with unified_bench --input-list parsing."""
        lines = ["# " + header, "# one input .bin file per line (relative to this file's dir):", ""]
        for p in format_paths_for_list(entries):
            lines.append(p)
        with open(os.path.join(output_dir, filename), "w") as f:
            f.write("\n".join(lines) + "\n")

    if float_input_list:
        write_list_file(
            f"input_list_{model_name}_float32.txt",
            float_input_list,
            f"float32 inputs for {model_name} ({len(float_input_list)} input(s))",
        )

    if gen_uint8 and uint8_input_list:
        write_list_file(
            f"input_list_{model_name}_uint8.txt",
            uint8_input_list,
            f"uint8 inputs for {model_name} ({len(uint8_input_list)} input(s))",
        )

    print(f"Generated test data under: {os.path.abspath(output_dir)}")
    print(
        f"float32 files: {len(float_input_list)}, uint8 files: {len(uint8_input_list)}"
    )
    if float_input_list:
        print(f"float32 list: {os.path.join(output_dir, f'input_list_{model_name}_float32.txt')}")
    if gen_uint8 and uint8_input_list:
        print(f"uint8   list: {os.path.join(output_dir, f'input_list_{model_name}_uint8.txt')}")
    print("Usage: unified_bench <model> --input-list <list_file> [--input-format auto|float32|uint8]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate ONNX test data and list files. Optionally choose path type stored in list files."
    )
    parser.add_argument(
        "--path-type",
        "-p",
        choices=["relative", "absolute"],
        help="Specify whether paths in input_list_*.txt are 'relative' (to output dir) or 'absolute'. If omitted, the script will ask interactively.",
    )
    parser.add_argument(
        "--gen-uint8",
        "-q",
        action="store_true",
        help="Generate uint8 inputs. By default uint8 generation is disabled; use this flag to enable quantized/random uint8 outputs.",
    )
    parser.add_argument(
        "--fixed-value",
        "-f",
        action="store_true",
        help="Use fixed values instead of random data. Float inputs will be 0.5f, uint8 inputs will be 64.",
    )
    parser.add_argument(
        "--float-value",
        type=float,
        default=0.5,
        help="Fixed float value to use when --fixed-value is enabled (default: 0.5).",
    )
    parser.add_argument(
        "--uint8-value",
        type=int,
        default=64,
        help="Fixed uint8 value to use when --fixed-value is enabled (default: 64).",
    )
    args = parser.parse_args()

    # get user input for model path
    model_path = input("Enter ONNX model path: ").strip('" ').strip()
    # get optional output directory from user
    output_dir = (
        input("Enter output directory (default: model file directory): ")
        .strip('" ')
        .strip()
    )

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if not output_dir:
        output_dir = os.path.dirname(os.path.abspath(model_path)) or "."

    # determine path_type (interactive if not provided)
    path_type = args.path_type
    if not path_type:
        while True:
            choice = input("Use relative paths in list files? (y/n) ").strip().lower()
            if choice in ("y", "yes"):
                path_type = "relative"
                break
            elif choice in ("n", "no"):
                path_type = "absolute"
                break
            else:
                print("Please enter 'y' (relative) or 'n' (absolute).")

    generate_test_data(model_path, output_dir, path_type, gen_uint8=args.gen_uint8, use_fixed_value=args.fixed_value, fixed_float_value=args.float_value, fixed_uint8_value=args.uint8_value)
