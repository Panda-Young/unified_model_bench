#!/usr/bin/env bash
# ============================================================================
# setup_linux_aarch64_deps.sh - Stage linux-aarch64 ONNX Runtime / MNN / ncnn
# dependency libraries for a Jetson Orin Nano (JetPack) build.
#
# Unified Benchmark builds against vendored prebuilt libs under deps/.  The
# binaries themselves are NOT committed (project stores them via scripts), so
# this script reconstructs the deps/onnxruntime, deps/mnn and deps/ncnn
# linux-aarch64 layout that CMake (CMakeLists.txt / cmake/backends.cmake) and
# the ONNX backend loader (src/onnx_backend.cpp) expect at build/runtime time.
#
# Usage:
#   bash tools/setup_linux_aarch64_deps.sh            # read paths from env
#
# Required sources (point env vars at whatever you have on hand; each is
# pre-verified to work on JetPack 6 / aarch64 glibc 2.35):
#
#   ORT_GPU_LIBDIR   - dir containing the ONNX Runtime 1.19.0 GPU (CUDA+TRT)
#                      aarch64 package libs:
#                        libonnxruntime.so.1.19.0
#                        libonnxruntime_providers_cuda.so
#                        libonnxruntime_providers_tensorrt.so
#                        libonnxruntime_providers_shared.so
#                      NOTE: libonnxruntime_providers_cuda.so is ~291 MB (over
#                      GitHub's 100 MB per-file limit) and is NOT committed to
#                      the repo - ORT_GPU_LIBDIR must point at a staged copy of
#                      the full GPU package you obtained yourself.
#   MNN_LIBDIR       - dir with the source-built MNN 3.6.1 libs:
#                        libMNN.so  libMNN_Express.so
#   NCNN_STATIC      - path to source-built ncnn static lib: libncnn.a
#   NCNN_INCDIR      - ncnn include dir (ncnn/ headers + generated platform.h)
#
# Outputs (written into the repo's deps tree):
#   deps/onnxruntime/lib/linux-aarch64/{libonnxruntime.so*,
#       libonnxruntime_providers_{cuda,tensorrt,shared}.so}
#   deps/mnn/lib/linux-aarch64/{libMNN.so,libMNN_Express.so}
#   deps/ncnn/lib/linux-aarch64/libncnn.a
#   deps/ncnn/include-linux/ncnn/  (headers)
# ============================================================================
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DST="${REPO}/deps"

# ---- resolve inputs ---------------------------------------------------------
: "${ORT_GPU_LIBDIR:?set ORT_GPU_LIBDIR}"
: "${MNN_BUILD_DIR:?set MNN_BUILD_DIR}"
: "${NCNN_STATIC:?set NCNN_STATIC}"
: "${NCNN_INCLUDE:?set NCNN_INCLUDE}"

[ -d "$ORT_GPU_LIBDIR" ]      || { echo "ERR: ort libdir missing: $ORT_GPU_LIBDIR"; return 1 2>/dev/null || exit 1; }
[ -d "$MNN_BUILD_DIR" ]       || { echo "ERR: mnn build dir missing: $MNN_BUILD_DIR"; exit 1; }
[ -f "$NCNN_STATIC" ]         || { echo "ERR: ncnn static missing: $NCNN_STATIC"; exit 1; }
[ -d "$NCNN_INCLUDE/ncnn" ]   || { echo "ERR: ncnn include/ncnn missing under: $NCNN_INCLUDE"; exit 1; }

# ---- ONNX Runtime GPU (CUDA + TensorRT) ----------------------------------------
echo "== onnxruntime (GPU 1.19.0) =="
O_DIR="${DST}/onnxruntime/lib/linux-aarch64"
mkdir -p "$O_DIR"
for f in libonnxruntime.so.1.19.0 \
         libonnxruntime_providers_cuda.so \
         libonnxruntime_providers_tensorrt.so \
         libonnxruntime_providers_shared.so; do
    cp -f "${ORT_GPU_LIBDIR}/$f" "$O_DIR/$f"
done
# versioned symlinks: libonnxruntime.so -> .so.1 -> .so.1.19.0
ln -sf libonnxruntime.so.1 "$O_DIR/libonnxruntime.so"
ln -sf libonnxruntime.so.1.19.0 "$O_DIR/libonnxruntime.so.1"

# ---- MNN (source-built 3.6.1) -------------------------------------------------------
echo "==> MNN (source-built 3.6.1) =="
M_DIR="${DST}/mnn/lib/linux-aarch64"
mkdir -p "$M_DIR"
cp -f "$MNN_BUILD_DIR/libMNN.so"                       "$M_DIR/libMNN.so"
cp -f "$MNN_BUILD_DIR/express/libMNN_Express.so"       "$M_DIR/libMNN_Express.so"

# ---- ncnn (source-built static) ----------------------------------------------------
echo "==> ncnn (source-built static libncnn.a) =="
N_DIR="${DST}/ncnn/lib/linux-aarch64"
mkdir -p "$N_DIR"
cp -f "$NCNN_STATIC" "$N_DIR/libncnn.a"
# Headers copied so the build can #include <ncnn/...>; platform.h must be the
# board-generated one (Vulkan OFF) not the shipped desktop default.
mkdir -p "${DST}/ncnn/include-linux/ncnn"
cp -rf "$NCNN_INCLUDE/ncnn/." "${DST}/ncnn/include-linux/ncnn/"

echo "==> done. Staged linux-aarch64 deps under $DST"
echo "    onnxruntime: $O_DIR"
echo "    mnn        : $M_DIR"
echo "    ncnn       : $N_DIR  (static + headers)"