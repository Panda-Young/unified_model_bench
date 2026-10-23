# Research Report: Prebuilt `libncnn.so` for Linux aarch64 (glibc)

**Target:** Tencent ncnn C++ inference library (`ncnn::Net`, `ncnn::load_param`, `ncnn::Extractor`), compiled as a genuine linkable `libncnn.so` (shared) running on **Linux aarch64 with glibc** (e.g. Jetson Orin Nano, Ubuntu 22.04, glibc 2.35).

## Bottom line

**There is NO prebuilt glibc-aarch64 `libncnn.so` that is readily available as a direct download.** Every official and community candidate investigated is either x86_64-only, an Android/bionic build (wrong ABI — cannot run on Linux), a Python-extension `.so` that does not export the ncnn C++ API, or a source-only repository. **The recommended path is to build ncnn from source on the Jetson device itself.** Concretely vetted below.

---

## 1. Official Tencent/ncnn releases — x86_64 only for Linux

- Latest release: tag **`20260526`**, title "...linux... 预编译库" (published 2026-05-26).
- The Linux assets shipped are `ncnn-20260526-ubuntu-2204.zip` / `ncnn-20260526-ubuntu-2404.zip` (and similar). These are **x86_64**.
- Other platforms shipped prebuilt: android (bionic ABI), apple/ios/macos, windows, webassembly, watchos, tvos, visionos.
- **No Linux aarch64 glibc asset exists in the official release set.** (Your stated premise is correct.)

## 2. `pip` ncnn wheels — NOT linkable C++ libraries

- The `ncnn` PyPI "wheel" ships an `_ncnn` CPython-extension `.so`. It is a Python binding; it does **not** export `ncnn::Net` / `ncnn::load_param` / `ncnn::Extractor` symbols and cannot be linked as a C/C++ library. (Matches your premise.)
- The `sherpa-ncnn` PyPI wheel also ships manylinux `py`-extension wheels (x86_64 and some arm64) but these are the Python binding built by sherpa, not a standalone linkable `libncnn.so` exposing the ncnn C++ ABI.

## 3. Community / third-party candidates — rejected

| Source | Result |
|--------|--------|
| `Qengineering` RPMImage / ncnn-YoloVx-Jetson / YoloV8-ncnn-Raspberry-Pi-4 repos | Build ncnn **from source**; release asset arrays are empty (`[]`); install scripts 404. No prebuilt `libncnn.so`. |
| `dirkarnez/ncnn-prebuilt` | Tag `20231027` exists but repo size 4 KB — near-empty, README missing. No usable aarch64 build. |
| `akemimadoka/ncnn-conan` | Confirmed empty: size 5 KB, no license, no releases, last push 2019. Dead end. |
| GitHub repo search `ncnn+aarch64`, `ncnn+linux+arm64`, `ncnn+glibc*`, `ncnn+vcpkg`, `libncnn+arm`, etc. | `total_count: 0` — no prebuilt aarch64 glibc repos matched. |
| GitHub code search (`/search/code`) | Requires authentication (HTTP 401) unauthenticated — could not enumerate assets by filename globally, but repo-level inspections above cover the known sources. |

---

## 4. Recommended path — build from source (Jetson Orin Nano, Ubuntu 22.04)

Jetson Orin Nano is **aarch64 glibc** (kernel/glibc, not bionic Android), so it needs a native glibc build — the source build on-device or cross-build below is the correct approach.

Environment to target: **Ubuntu 22.04, glibc 2.35, g++ 11.4, cmake 3.22**.

```bash
# Install dependencies
sudo apt-get update
sudo apt-get install -y git build-essential cmake

# Get the source
git clone https://github.com/Tencent/ncnn.git
cd ncnn
# (optional) check out a release tag, e.g. git checkout 20260526

# Configure with examples/tools/benchmark disabled
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Release \
      -DNCNN_BUILD_EXAMPLES=OFF \
      -DNCNN_BUILD_TOOLS=OFF \
      -DNCNN_BUILD_BENCHMARK=OFF \
      ..

# Build (adjust -j to core count)
make -j$(nproc)

# The resulting library is written to ncnn/build/src/libncnn.so
sudo make install   # installs headers + libncnn.a / libncnn.so
```

Notes:
- This produces a genuine `libncnn.so` exporting the ncnn C++ API that you can link with `-lncnn`.
- For Jetson you may additionally want CUDA/Vulkan enabled (`-DNCNN_VULKAN=ON`) if the ncnn-vulkan build is desired; the plain CPU build above needs no CUDA toolchain.
- Building on the Orin itself uses the native aarch64 glibc toolchain, matching glibc 2.35 / g++ 11.4 out of the box.

---

## Conclusion

NO prebuilt glibc-aarch64 `libncnn.so` is readily available from Tencent or known third-party sources. Build from source with the commands above.