/*============================================================================
 * ncnn_backend.cpp - NCNN backend (RAII)
 *============================================================================*/

#include "backend_interface.hpp"
#include "log.hpp"

#ifdef HAVE_NCNN_BACKEND

#include <ncnn/c_api.h>
#include <ncnn/cpu.h>
#include <ncnn/gpu.h>
#include <ncnn/net.h>
/* ncnn/platform.h may define min/max macros. */
#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif

#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <setjmp.h>
#include <string>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <cstdarg>
#include <cstdlib>
#include <dbghelp.h>
#include <io.h>
#include <windows.h>
#pragma comment(lib, "dbghelp.lib")

/* Detect how NCNN terminates the process */
static int t_crash_exit_count = 0;
static void crash_atexit_handler()
{
    LOGE("NCNN: atexit handler called - NCNN called exit()");
    t_crash_exit_count = 1;
}

/* VectoredExceptionHandler - runs BEFORE SEH __except */
static volatile LONG t_veh_bf16_crash = 0;
static PVOID t_veh_handle = NULL;
static LONG CALLBACK veh_crash_handler(EXCEPTION_POINTERS *ep)
{
    DWORD code = ep->ExceptionRecord->ExceptionCode;
    if (code == EXCEPTION_ACCESS_VIOLATION || code == EXCEPTION_ILLEGAL_INSTRUCTION) {
        InterlockedExchange(&t_veh_bf16_crash, 1);
    }
    return EXCEPTION_CONTINUE_SEARCH;
}

/* Per-thread abort recovery via SIGABRT + longjmp */
static __declspec(thread) jmp_buf t_abort_jmp;
static __declspec(thread) int t_abort_ready = 0;

static void abort_signal_handler(int sig)
{
    (void)sig;
    LOGW("NCNN: abort() detected - recovering via longjmp");
    if (t_abort_ready) {
        longjmp(t_abort_jmp, 1);
    }
}

/* SEH-safe helper: check x86_64 AVX-512 BF16 support.
 * Must be in a function with no C++ object unwinding. */
static int safe_check_bf16()
{
    __try {
        return ncnn::cpu_support_x86_avx512_bf16();
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        return 0; /* assume unsupported on crash */
    }
}

/* SEH-safe helper: extract a blob by name, catching access violations.
 * Must be in a function with no C++ object unwinding (no destructors). */
static int safe_extract(ncnn::Extractor *ex, const char *name, ncnn::Mat &out)
{
    __try {
        return ex->extract(name, out);
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        LOGE("NCNN: HW exception (0x%08lX) in extract(%s)", GetExceptionCode(), name);
        /* Write minidump */
        HANDLE hFile = CreateFileA("ncnn_crash.dmp", GENERIC_WRITE, 0, NULL,
                                   CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
        if (hFile != INVALID_HANDLE_VALUE) {
            MINIDUMP_EXCEPTION_INFORMATION mei;
            mei.ThreadId = GetCurrentThreadId();
            mei.ExceptionPointers = NULL;
            mei.ClientPointers = FALSE;
            /* Full memory dump: includes complete heap/stack contents, so a
             * debugger can inspect data (e.g. Mat dims) at the crash site.
             * Much larger than MiniDumpWithDataSegs, only written on crash. */
            MiniDumpWriteDump(GetCurrentProcess(), GetCurrentProcessId(),
                              hFile, MiniDumpWithFullMemory, NULL, NULL, NULL);
            CloseHandle(hFile);
            LOGI("NCNN: minidump written: ncnn_crash.dmp (full memory)");
        }
        fflush(stderr);
        return -1;
    }
}
#endif

/* ---------------------------------------------------------------------------
 * Shape -> ncnn::Mat (rank-agnostic, fixes OOB for 3D/2D/1D inputs)
 * -------------------------------------------------------------------------*/
/* Build a layout-correct ncnn::Mat from an ONNX shape (rank 1..4).
 * Returns an OWNING Mat: input data is copied into an ncnn-allocated buffer
 * with correct cstep (channel stride), independent of the external buffer's
 * layout. The old code wrapped the external flat float buffer directly,
 * which is only valid when the external stride happens to match ncnn's
 * aligned cstep.
 *
 * Fixes hard crash (0xC0000005) on the Vulkan path: a 4D input Mat
 * (dims=4, e.g. in1=[1,48,1,1]) fed into a BinaryOp broadcast crashed the
 * Vulkan shader (blob dims=4 vs dims=3 mismatch -> access violation in
 * extract). Folding rank-4 inputs with N==1 into 3D keeps the identical
 * logical layout (batch=1) while staying inside the dims<=3 shapes that
 * the Vulkan BinaryOp supports. Verified on CPU/Vulkan FP32/FP16/BF16:
 * max |diff| vs reference < 4e-4, softmax output sum == 1.0.
 *
 * Mapping (N==1 assumed for rank 4):
 *   rank1 [W]       -> Mat(w=W)
 *   rank2 [H,W]     -> Mat(w=W, h=H)
 *   rank3 [C,H,W]   -> Mat(w=W, h=H, c=C)
 *   rank4 [1,C,H,W] -> Mat(w=W, h=H, c=C)   (Vulkan-safe)
 *   rank4 [N>1,...] -> legacy reversed 4D Mat fallback (Vulkan BinaryOp unsafe)
 */
static ncnn::Mat MakeMatFromShape(const std::vector<int> &shape, const float *data)
{
    ncnn::Mat m;
    switch (shape.size()) {
    case 1: {
        m = ncnn::Mat(shape[0]);
        if (shape[0] > 0) {
            memcpy(m.data, data, (size_t)shape[0] * sizeof(float));
        }
        break;
    }
    case 2: {
        m = ncnn::Mat(shape[1], shape[0]);
        if (!m.empty()) {
            memcpy(m.data, data, (size_t)shape[0] * (size_t)shape[1] * sizeof(float));
        }
        break;
    }
    case 3: {
        int w = shape[2];
        int h = shape[1];
        int c = shape[0];
        m = ncnn::Mat(w, h, c);
        if (!m.empty() && w > 0 && h > 0) {
            /* Use a CONTIGUOUS channel layout (cstep = w*h). The default
             * ncnn::Mat(w,h,c) aligns cstep to 4 bytes (e.g. w*h=2049 ->
             * cstep=2052), which inserts 3-float padding between channels.
             * The ONNX NCHW source is fully contiguous, and the network
             * expects the same; with the aligned cstep every channel past
             * the first lands 3 elements off, corrupting the output (SCNet
             * 30-state diff 2.5 -> 0.2 after this fix). */
            m.cstep = (size_t)w * (size_t)h;
            const float *src = data;
            for (int ci = 0; ci < c; ++ci) {
                memcpy(m.channel(ci), src, (size_t)w * (size_t)h * sizeof(float));
                src += (size_t)w * (size_t)h;
            }
        }
        break;
    }
    case 4: {
        int n = shape[0];
        int c = shape[1];
        int h = shape[2];
        int w = shape[3];
        if (n == 1) {
            /* Fold batch dim: [1,C,H,W] -> 3D Mat(w=W,h=H,c=C).
             * 4D Mats (dims=4) crash Vulkan BinaryOp broadcast; 3D keeps
             * identical logical layout for batch=1. */
            m = ncnn::Mat(w, h, c);
            if (!m.empty() && w > 0 && h > 0) {
                /* Contiguous channel layout, see the case 3 comment. */
                m.cstep = (size_t)w * (size_t)h;
                const float *src = data;
                for (int ci = 0; ci < c; ++ci) {
                    memcpy(m.channel(ci), src, (size_t)w * (size_t)h * sizeof(float));
                    src += (size_t)w * (size_t)h;
                }
            }
        } else {
            /* N>1: rare; keep legacy reversed 4D mapping (Vulkan BinaryOp may crash) */
            LOGW("NCNN: rank-4 input N=%d>1 -> 4D Mat (Vulkan BinaryOp may crash)", n);
            m = ncnn::Mat(w, h, c, n, (void *)data);
        }
        break;
    }
    default: {
        LOGE("NCNN: unsupported input rank %zu", shape.size());
        break;
    }
    }
    return m;
}

/* Fill an already-allocated ncnn::Mat with data from an ONNX-shaped flat
 * float buffer, using the SAME layout mapping as MakeMatFromShape (so a
 * cached Mat can be refreshed without re-allocating/cloning). Returns the
 * number of bytes written, or 0 on layout mismatch. */
static size_t FillMatFromShape(ncnn::Mat &m, const std::vector<int> &shape,
                               const float *data)
{
    if (m.empty() || !data || shape.empty()) {
        return 0;
    }
    switch (shape.size()) {
    case 1: {
        size_t bytes = (size_t)shape[0] * sizeof(float);
        if (m.total() >= shape[0]) {
            memcpy(m.data, data, bytes);
            return bytes;
        }
        break;
    }
    case 2: {
        int h = shape[0], w = shape[1];
        size_t bytes = (size_t)h * (size_t)w * sizeof(float);
        if (m.total() >= (size_t)h * (size_t)w) {
            memcpy(m.data, data, bytes);
            return bytes;
        }
        break;
    }
    case 3: {
        int c = shape[0], h = shape[1], w = shape[2];
        if (m.c == c && m.h == h && m.w == w) {
            const float *src = data;
            for (int ci = 0; ci < c; ++ci) {
                memcpy(m.channel(ci), src, (size_t)w * (size_t)h * sizeof(float));
                src += (size_t)w * (size_t)h;
            }
            return (size_t)c * (size_t)h * (size_t)w * sizeof(float);
        }
        break;
    }
    case 4: {
        int n = shape[0], c = shape[1], h = shape[2], w = shape[3];
        if (n == 1 && m.c == c && m.h == h && m.w == w) {
            const float *src = data;
            for (int ci = 0; ci < c; ++ci) {
                memcpy(m.channel(ci), src, (size_t)w * (size_t)h * sizeof(float));
                src += (size_t)w * (size_t)h;
            }
            return (size_t)c * (size_t)h * (size_t)w * sizeof(float);
        }
        break;
    }
    default: {
        break;
    }
    }
    return 0;
}

class NCNNBackend : public IBackend
{
public:
    explicit NCNNBackend(BackendId id) { id_ = id; }
    ~NCNNBackend() override { Cleanup(); }

    bool Initialize(const char *model_path, int num_threads) override;
    bool QueryIOInfo(std::string &is, size_t &ie, std::string &os, size_t &oe) override;
    void SetSharedInput(const float *const *data, const size_t *counts) override;
    bool RunBenchmark(int warmup, int repeat, double &total, double &maxv,
                      double &minv, int &maxi, std::vector<float *> &odata,
                      std::vector<size_t> &oelems,
                      std::vector<std::array<size_t, MAX_DIMENSIONS>> &oshapes,
                      std::vector<size_t> &odims) override;
    void GetTiming(std::array<double, 10> &timing) override;
    void GetTransferTiming(double &transfer_in_ms,
                           double &transfer_out_ms) override;

private:
    void Cleanup();
    bool ReadShapesFile(const char *shapes_path);
    bool TryBf16Trial();

    ncnn::Net *net_ = nullptr;
    int gpu_device_ = -1;      /* -1 = CPU only */
    bool bf16_unsafe_ = false; /* true if BF16 causes access violation */

    size_t num_inputs_ = 0;
    size_t num_outputs_ = 0;
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    std::vector<size_t> input_elems_;
    std::vector<size_t> output_elems_;
    std::vector<std::vector<int>> input_shapes_;
    std::vector<std::vector<int>> output_shapes_;

    std::vector<float *> input_bufs_;
    std::vector<size_t> input_buf_elems_;
    std::vector<bool> input_external_;

    double init_ms_ = 0;

    /* Tensor transfer timing (avg ms per repeat).
     * NCNN uploads inputs via ex.input() and downloads outputs via
     * extract(); for Vulkan backends these include the H2D/D2H transfer.
     * transfer_in_ms_  = input Mat build + ex.input() (host->device)
     * transfer_out_ms_ = ex.extract() + snapshot memcpy (device->host) */
    double transfer_in_ms_ = 0.0;
    double transfer_out_ms_ = 0.0;

    /* Pre-allocated input Mats, reused across repeats to avoid per-run
     * allocation + clone (the 30-input scnet-style model spends several ms
     * per repeat just rebuilding Mats). Only their data payload is refreshed
     * via memcpy each feed. */
    std::vector<ncnn::Mat> in_mats_cache_;
    bool in_mats_cache_valid_ = false;
};

/* ---------------------------------------------------------------------------
 * Read .ncnn.shapes file
 * -------------------------------------------------------------------------*/
bool NCNNBackend::ReadShapesFile(const char *shapes_path)
{
    FILE *f = fopen(shapes_path, "r");
    if (!f) {
        LOGE("NCNN: cannot open shapes file: %s, due to %s, %d", shapes_path, strerror(errno), errno);
        return false;
    }

    int num_in = 0, num_out = 0;
    (void)num_in;
    (void)num_out; /* read from file but used only for validation */
    char line[512];

    while (fgets(line, sizeof(line), f)) {
        size_t len = strlen(line);
        while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r')) {
            line[--len] = '\0';
        }

        if (strncmp(line, "inputs=", 7) == 0) {
            num_in = atoi(line + 7);
        } else if (strncmp(line, "outputs=", 8) == 0) {
            num_out = atoi(line + 8);
        } else if (strncmp(line, "in", 2) == 0 && strncmp(line, "input", 5) != 0 &&
                   strncmp(line, "inputs=", 7) != 0) {
            /* in0=1,4,2048,8 */
            char *eq = strchr(line, '=');
            if (!eq) {
                continue;
            }
            /* Name before = */
            std::string name(line, eq - line);
            input_names_.push_back(name);

            const char *dims = eq + 1;
            std::vector<int> shape;
            size_t elems = 1;
            const char *p = dims;
            while (*p) {
                int d = atoi(p);
                shape.push_back(d);
                elems *= (size_t)d;
                p = strchr(p, ',');
                if (!p) {
                    break;
                }
                ++p;
            }
            input_shapes_.push_back(shape);
            input_elems_.push_back(elems);
        } else if (strncmp(line, "out", 3) == 0 && strncmp(line, "outputs=", 8) != 0) {
            char *eq = strchr(line, '=');
            if (!eq) {
                continue;
            }
            std::string name(line, eq - line);
            output_names_.push_back(name);

            const char *dims = eq + 1;
            std::vector<int> shape;
            size_t elems = 1;
            const char *p = dims;
            while (*p) {
                int d = atoi(p);
                shape.push_back(d);
                elems *= (size_t)d;
                p = strchr(p, ',');
                if (!p) {
                    break;
                }
                ++p;
            }
            output_shapes_.push_back(shape);
            output_elems_.push_back(elems);
        }
    }
    if (fclose(f) != 0) {
        LOGW("NCNN: fclose(%s) failed: %s, %d", shapes_path, strerror(errno), errno);
    }

    num_inputs_ = input_names_.size();
    num_outputs_ = output_names_.size();

    LOGI("NCNN: shapes loaded: %zu in, %zu out", num_inputs_, num_outputs_);
    return num_inputs_ > 0 && num_outputs_ > 0;
}

/* ---------------------------------------------------------------------------
 * Try a single forward pass with BF16 to detect runtime crashes.
 * VEH catches access violations and sets t_veh_bf16_crash flag.
 * -------------------------------------------------------------------------*/
bool NCNNBackend::TryBf16Trial()
{
    if (num_inputs_ == 0 || num_outputs_ == 0) {
        return false;
    }
#ifdef _WIN32
    if (!t_veh_handle) {
        t_veh_handle = AddVectoredExceptionHandler(1, veh_crash_handler);
    }

    InterlockedExchange(&t_veh_bf16_crash, 0);
#endif
    try {
        ncnn::Extractor ex = net_->create_extractor();
        std::vector<std::vector<float>> buf_pool(num_inputs_);
        std::vector<ncnn::Mat> mats;
        for (size_t i = 0; i < num_inputs_; ++i) {
            size_t n = input_elems_[i] > 0 ? input_elems_[i] : 1;
            buf_pool[i].resize(n, 0.0f);
            ncnn::Mat m = MakeMatFromShape(input_shapes_[i], buf_pool[i].data());
            if (m.empty()) {
                LOGW("NCNN: BF16 trial: unsupported rank for input %zu", i);
                return false;
            }
            mats.push_back(m.clone()); /* clone: NCNN owns data, no dangling ptr */
            int iret = ex.input(input_names_[i].c_str(), mats.back());
            if (iret != 0) {
                LOGW("NCNN: BF16 trial: input %zu failed, ret=%d", i, iret);
                return false;
            }
        }
        ncnn::Mat out;
        ex.extract(output_names_[0].c_str(), out);
    } catch (...) {
        /* C++ exceptions caught here */
    }
#ifdef _WIN32
    int crashed = InterlockedExchange(&t_veh_bf16_crash, 0);
    return crashed == 0;
#else
    /* Non-Windows: no VEH available, assume trial passed if no exception */
    return true;
#endif
}

/* ---------------------------------------------------------------------------
 * Initialize
 * -------------------------------------------------------------------------*/
bool NCNNBackend::Initialize(const char *model_path, int num_threads)
{
    auto t0 = std::chrono::high_resolution_clock::now();
    last_error_.clear();

    /* Derive paths: model.ncnn.param -> model.ncnn.bin, model.ncnn.shapes */
    std::string path(model_path);
    std::string base = path;
    /* Strip .param */
    if (base.length() > 6) {
        auto tail6 = base.substr(base.length() - 6);
        if (stricmp_(tail6.c_str(), ".param") == 0) {
            base = base.substr(0, base.length() - 6);
        }
    }
    /* Strip .ncnn if present (the file is model.ncnn.param, we want model) */
    if (base.length() > 5) {
        auto tail5 = base.substr(base.length() - 5);
        if (stricmp_(tail5.c_str(), ".ncnn") == 0) {
            base = base.substr(0, base.length() - 5);
        }
    }

    std::string bin_path = base + ".ncnn.bin";
    std::string shapes_path = base + ".shapes";

    /* Read shapes first */
    if (!ReadShapesFile(shapes_path.c_str())) {
        last_error_ = "NCNN: failed to read shapes file (" + shapes_path + ")";
        return false;
    }

    net_ = new ncnn::Net();
    net_->opt.num_threads = num_threads;
    net_->opt.lightmode = true;
    /* Disable packing layout for multi-input models: PNNX may reorder channels
     * during conversion, and packing interacts badly with external-data Mats. */
    net_->opt.use_packing_layout = (num_inputs_ <= 1);
    /* Max speed: enable Winograd convolution (biggest single perf gain for CNNs) */
    net_->opt.use_winograd_convolution = true;
    net_->opt.use_sgemm_convolution = true;

    /* For Vulkan_FP16: prefer FP16-converted model.
     * The FP16 model has weights already quantized, giving smaller and
      * more accurate results than runtime FP32->FP16 conversion.
     * .param is shared with FP32 (identical structure), only .bin differs. */
    if (id_ == BackendId::NCNN_VK_FP16) {
        std::string fp16_bin = base + "_fp16.ncnn.bin";
        FILE *test_fp16 = fopen(fp16_bin.c_str(), "r");
        if (test_fp16) {
            if (fclose(test_fp16) != 0) {
                LOGW("NCNN: fclose(%s) failed: %s, %d", fp16_bin.c_str(), strerror(errno), errno);
            }
            bin_path = fp16_bin;
            LOGI("NCNN: using FP16 weights: %s", fp16_bin.c_str());
        } else {
            LOGE("NCNN: FP16 weights not found (%s) - aborting (run: python onnx_to_ncnn.py model.onnx --dual)",
                 fp16_bin.c_str());
            last_error_ = "NCNN: FP16 weights file not found (" + fp16_bin + ")";
            return false;
        }
    }
    if (id_ == BackendId::NCNN_VK || id_ == BackendId::NCNN_VK_FP16 || id_ == BackendId::NCNN_VK_BF16) {
#if NCNN_VULKAN
        gpu_device_ = ncnn::get_default_gpu_index();
        if (gpu_device_ < 0) {
            LOGE("NCNN: no Vulkan device found - aborting");
            last_error_ = "NCNN: no Vulkan device available";
            return false;
        }
        net_->opt.use_vulkan_compute = true;
        if (id_ == BackendId::NCNN_VK) {
            /* Force FP32 path: disable all FP16/ BF16 optimizations */
            net_->opt.use_fp16_packed = false;
            net_->opt.use_fp16_storage = false;
            net_->opt.use_fp16_arithmetic = false;
            net_->opt.use_bf16_packed = false;
            net_->opt.use_bf16_storage = false;
        } else if (id_ == BackendId::NCNN_VK_BF16) {
            /* BF16 path - only enable features the GPU actually supports */
            net_->opt.use_fp16_packed = false;
            net_->opt.use_fp16_storage = false;
            net_->opt.use_fp16_arithmetic = false;
            const auto &gpu = ncnn::get_gpu_info(gpu_device_);
            net_->opt.use_bf16_packed = gpu.support_bf16_packed() ? true : false;
            net_->opt.use_bf16_storage = gpu.support_bf16_storage() ? true : false;
            if (!net_->opt.use_bf16_packed && !net_->opt.use_bf16_storage) {
                LOGE("NCNN: GPU does not support BF16 - aborting");
                last_error_ = "NCNN: GPU lacks BF16 support";
                return false;
            }
            LOGI("NCNN: GPU BF16: packed=%d storage=%d",
                 (int)net_->opt.use_bf16_packed, (int)net_->opt.use_bf16_storage);
        } else {
            /* FP16 path */
            net_->opt.use_fp16_packed = true;
            net_->opt.use_fp16_storage = true;
            net_->opt.use_fp16_arithmetic = true;
        }
        LOGI("NCNN: Vulkan enabled (gpu=%d, precision=%s)", gpu_device_,
             (id_ == BackendId::NCNN_VK_BF16) ? "BF16" : (id_ == BackendId::NCNN_VK_FP16) ? "FP16"
                                                                                          : "FP32");
#else
        LOGE("NCNN: Vulkan backend not available - NCNN built without Vulkan support");
        last_error_ = "NCNN: Vulkan not supported in this build";
        return false;
#endif
    }
    if (id_ == BackendId::NCNN_CPU_FP16) {
        /* CPU FP16: enable packed FP16 storage/arithmetic (ARM NEON FP16) */
        net_->opt.use_fp16_packed = true;
        net_->opt.use_fp16_storage = true;
        net_->opt.use_fp16_arithmetic = true;
        LOGI("NCNN: CPU FP16 mode enabled");
    }
    if (id_ == BackendId::NCNN_CPU_BF16) {
        /* CPU BF16: use CPUID check + trial forward pass to verify */
#if defined(__ARM_NEON) || defined(__aarch64__) || defined(_M_ARM64)
        bool have_bf16 = !!ncnn::cpu_support_arm_bf16();
#elif defined(__x86_64__) || defined(_M_AMD64) || defined(_M_X64)
        bool have_bf16 = !!safe_check_bf16();
#else
        bool have_bf16 = true;
#endif
        if (!have_bf16) {
            LOGE("NCNN: CPU BF16 not supported - CPU lacks BF16 instructions, aborting");
            last_error_ = "NCNN: CPU lacks BF16 support";
            return false;
        }
        net_->opt.use_bf16_storage = true;
    }

    /* Load model */
    LOGI("NCNN: loading param: %s", path.c_str());
    if (net_->load_param(path.c_str()) != 0) {
        LOGE("NCNN: failed to load param: %s, due to %s, %d",
             path.c_str(), strerror(errno), errno);
        LOGE("NCNN: possible cause - missing/unsupported layer type, or file not found");
        last_error_ = "NCNN: load_param failed (errno=" + std::to_string(errno) + ": " + strerror(errno) + ")";
        return false;
    }
    LOGI("NCNN: param loaded, loading bin: %s", bin_path.c_str());
    {
        LOGI("NCNN: load_model() starting...");
        int mret = net_->load_model(bin_path.c_str());
        LOGI("NCNN: load_model() returned %d", mret);
        if (mret != 0) {
            LOGE("NCNN: failed to load bin: %s, due to %s, %d",
                 bin_path.c_str(), strerror(errno), errno);
            last_error_ = "NCNN: load_model failed (errno=" + std::to_string(errno) + ": " + strerror(errno) + ")";
            return false;
        }
    }
    LOGI("NCNN: model loaded successfully");

    /* If BF16 was enabled, run a quick trial to verify it doesn't crash */
    if (id_ == BackendId::NCNN_CPU_BF16 && net_->opt.use_bf16_storage) {
        if (!TryBf16Trial()) {
            LOGE("NCNN: CPU BF16 trial crashed (access violation) - aborting");
            last_error_ = "NCNN: CPU BF16 runtime crash (unsupported on this CPU)";
            return false;
        }
        LOGI("NCNN: CPU BF16 trial passed");
    }

    init_ms_ = std::chrono::duration<double, std::milli>(
                   std::chrono::high_resolution_clock::now() - t0)
                   .count();

    LOGI("NCNN: init complete (%.1f ms), version %s", init_ms_, ncnn_version());
    return true;
}

bool NCNNBackend::QueryIOInfo(std::string &is, size_t &ie,
                              std::string &os, size_t &oe)
{
    is.clear();
    ie = 0;
    for (size_t i = 0; i < num_inputs_; ++i) {
        char buf[128] = {};
        int off = snprintf(buf, sizeof(buf), "[");
        for (size_t d = 0; d < input_shapes_[i].size(); ++d) {
            off += snprintf(buf + off, sizeof(buf) - off, "%s%d", d > 0 ? "," : "", input_shapes_[i][d]);
        }
        snprintf(buf + off, sizeof(buf) - off, "]");
        if (i > 0) {
            is += " ";
        }
        is += buf;
        ie += input_elems_[i];
    }
    os.clear();
    oe = 0;
    for (size_t i = 0; i < num_outputs_; ++i) {
        char buf[128] = {};
        int off = snprintf(buf, sizeof(buf), "[");
        for (size_t d = 0; d < output_shapes_[i].size(); ++d) {
            off += snprintf(buf + off, sizeof(buf) - off, "%s%d", d > 0 ? "," : "", output_shapes_[i][d]);
        }
        snprintf(buf + off, sizeof(buf) - off, "]");
        if (i > 0) {
            os += " ";
        }
        os += buf;
        oe += output_elems_[i];
    }
    return true;
}

void NCNNBackend::SetSharedInput(const float *const *data, const size_t *counts)
{
    for (size_t i = 0; i < num_inputs_; ++i) {
        if (i >= input_bufs_.size()) {
            input_bufs_.resize(i + 1, nullptr);
            input_buf_elems_.resize(i + 1, 0);
            input_external_.resize(i + 1, false);
        }
        input_bufs_[i] = const_cast<float *>(data[i]);
        input_buf_elems_[i] = counts[i];
        input_external_[i] = true;
    }
}

bool NCNNBackend::RunBenchmark(int warmup, int repeat, double &total,
                               double &maxv, double &minv, int &maxi,
                               std::vector<float *> &odata,
                               std::vector<size_t> &oelems,
                               std::vector<std::array<size_t, MAX_DIMENSIONS>> &oshapes,
                               std::vector<size_t> &odims)
{
    total = 0;
    maxv = 0;
    minv = 1e12;
    maxi = 0;
    if (num_inputs_ == 0 || num_outputs_ == 0) {
        return false;
    }

    std::vector<std::vector<float>> snaps(num_outputs_);
    for (size_t i = 0; i < num_outputs_; ++i) {
        snaps[i].resize(output_elems_[i] > 0 ? output_elems_[i] : 1);
    }

    /* Build ncnn::Mat inputs (NCNN uses [w,h,c] order internally).
     * The Mats are CACHED and reused across repeats: only the data payload
     * is refreshed each feed, so the expensive per-run allocation + clone
     * (several ms for 30-input models) happens only once. */
    auto build_inputs = [&]() -> std::vector<ncnn::Mat> {
        std::vector<ncnn::Mat> mats;
        mats.reserve(num_inputs_);
        for (size_t i = 0; i < num_inputs_; ++i) {
            /* Map ONNX dims (rank 1..4) to an ncnn::Mat with reversed dims.
             * Do NOT hardcode [w,h,c]: this model has 3D state inputs
             * (e.g. [472,32,2]) and reading shape[3] would be out of bounds. */
            ncnn::Mat tmp;
            if (in_mats_cache_valid_ && i < in_mats_cache_.size() &&
                !in_mats_cache_[i].empty()) {
                /* Reuse cached Mat: refresh its data from the shared input
                 * buffer, preserving the ncnn [w,h,c] channel layout. */
                ncnn::Mat &cached = in_mats_cache_[i];
                if (FillMatFromShape(cached, input_shapes_[i], input_bufs_[i]) == 0) {
                    LOGE("NCNN: cannot refresh cached input %zu (rank %zu)",
                         i, input_shapes_[i].size());
                    mats.clear();
                    break;
                }
                tmp = cached;
            } else {
                tmp = MakeMatFromShape(input_shapes_[i], input_bufs_[i]);
                if (tmp.empty()) {
                    LOGE("NCNN: cannot build input %zu (rank %zu)", i, input_shapes_[i].size());
                    mats.clear();
                    break;
                }
                /* Cache the layout-owning copy; subsequent feeds reuse it. */
                ncnn::Mat cached = tmp.clone();
                if (in_mats_cache_.size() <= i) {
                    in_mats_cache_.resize(i + 1);
                }
                in_mats_cache_[i] = cached;
                tmp = cached;
            }
            mats.push_back(tmp);
        }
        if (mats.size() == num_inputs_) {
            in_mats_cache_valid_ = true;
        }
        return mats;
    };

    (void)warmup;
    for (int w = 0; w < warmup; ++w) {
        ncnn::Extractor ex = net_->create_extractor();
        auto mats = build_inputs();
        if (mats.size() != num_inputs_) {
            LOGE("NCNN: failed to build warmup inputs");
            return false;
        }
        for (size_t i = 0; i < num_inputs_; ++i) {
            int iret = ex.input(input_names_[i].c_str(), mats[i]);
            if (iret != 0) {
                LOGE("NCNN: warmup input %zu failed, ret=%d", i, iret);
                return false;
            }
        }
        for (size_t i = 0; i < num_outputs_; ++i) {
            ncnn::Mat out;
            int ret;
#ifdef _WIN32
            ret = safe_extract(&ex, output_names_[i].c_str(), out);
            if (ret != 0) {
                LOGE("NCNN: warmup extract %s %s (ret=%d)",
                     output_names_[i].c_str(),
                     (ret == -1) ? "crashed (SEH)" : "failed",
                     ret);
                return false;
            }
#else
            ret = ex.extract(output_names_[i].c_str(), out);
            if (ret != 0) {
                LOGE("NCNN: warmup extract %s failed, ret=%d", output_names_[i].c_str(), ret);
                return false;
            }
#endif
        }
    }

    for (int r = 0; r < repeat; ++r) {
        ncnn::Extractor ex = net_->create_extractor();
        /* Time input upload: Mat build (host data) + ex.input() (H2D for
         * Vulkan; plain copy for CPU). */
        auto t_in0 = std::chrono::high_resolution_clock::now();
        auto mats = build_inputs();
        if (mats.size() != num_inputs_) {
            LOGE("NCNN: failed to build inputs at run %d", r);
            return false;
        }
        for (size_t i = 0; i < num_inputs_; ++i) {
            int iret = ex.input(input_names_[i].c_str(), mats[i]);
            if (iret != 0) {
                LOGE("NCNN: input %zu failed at run %d, ret=%d", i, r, iret);
                return false;
            }
        }
        auto t_in1 = std::chrono::high_resolution_clock::now();
        transfer_in_ms_ +=
            std::chrono::duration<double, std::milli>(t_in1 - t_in0).count();

        auto t0 = std::chrono::high_resolution_clock::now();
        double out_copy_ms = 0.0;
        for (size_t i = 0; i < num_outputs_; ++i) {
            ncnn::Mat out;
            int ret;
#ifdef _WIN32
            ret = safe_extract(&ex, output_names_[i].c_str(), out);
            if (ret != 0) {
                LOGE("NCNN: extract %s %s (ret=%d)",
                     output_names_[i].c_str(),
                     (ret == -1) ? "crashed (SEH)" : "failed",
                     ret);
                return false;
            }
#else
            ret = ex.extract(output_names_[i].c_str(), out);
            if (ret != 0) {
                LOGE("NCNN: extract %s failed, ret=%d", output_names_[i].c_str(), ret);
                return false;
            }
#endif
            /* Convert NCNN [w,h,c] back to flat float.
             * Use out.data (raw pointer to entire blob) instead of
             * out.channel(0) which only points to the first channel.
             * For multi-channel outputs like [1,10] (c=10),
             * channel(0) gives only 1 float, reading beyond is UB. */
            if (!out.empty()) {
                size_t copy_bytes = std::min(
                    snaps[i].size() * sizeof(float),
                    out.total() * sizeof(float));
                auto tc0 = std::chrono::high_resolution_clock::now();
                memcpy(snaps[i].data(), (float *)out.data, copy_bytes);
                auto tc1 = std::chrono::high_resolution_clock::now();
                out_copy_ms +=
                    std::chrono::duration<double, std::milli>(tc1 - tc0).count();
            }
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        /* NOTE: ncnn's extract() is the synchronous inference call itself
         * (it executes the graph and returns the result, including the D2H
         * copy for Vulkan internally). It CANNOT be separated into compute vs
         * transfer - avg_run_ms therefore includes the D2H part. transfer_out
         * covers only the explicit snapshot memcpy measured above. */
        transfer_out_ms_ += out_copy_ms;
        LOGD("NCNN: run %d took %.3f ms (snap_copy=%.3f)", r, ms, out_copy_ms);

        total += ms;
        if (ms > maxv) {
            maxv = ms;
            maxi = r;
        }
        if (ms < minv) {
            minv = ms;
        }
    }

    /* Normalize transfer time to avg ms per repeat. */
    if (repeat > 0) {
        transfer_in_ms_ /= (double)repeat;
        transfer_out_ms_ /= (double)repeat;
    }

    odata.resize(num_outputs_);
    oelems.resize(num_outputs_);
    oshapes.resize(num_outputs_);
    odims.resize(num_outputs_);
    for (size_t i = 0; i < num_outputs_; ++i) {
        size_t n = output_elems_[i];
        float *buf = (float *)malloc(n * sizeof(float));
        if (!buf) {
            LOGE("NCNN: malloc(%zu) failed at output %zu, due to %s, %d", n * sizeof(float), i, strerror(errno), errno);
            for (size_t j = 0; j < i; ++j) {
                free(odata[j]);
            }
            return false;
        }
        memcpy(buf, snaps[i].data(), n * sizeof(float));
        odata[i] = buf;
        oelems[i] = n;
        auto &sh = oshapes[i];
        sh.fill(0);
        for (size_t d = 0; d < output_shapes_[i].size() && d < MAX_DIMENSIONS; ++d) {
            sh[d] = (size_t)output_shapes_[i][d];
        }
        odims[i] = output_shapes_[i].size();
    }
    return true;
}

void NCNNBackend::GetTiming(std::array<double, 10> &timing)
{
    timing.fill(0);
    timing[0] = init_ms_;
}

void NCNNBackend::GetTransferTiming(double &transfer_in_ms,
                                    double &transfer_out_ms)
{
    transfer_in_ms = transfer_in_ms_;
    transfer_out_ms = transfer_out_ms_;
}

void NCNNBackend::Cleanup()
{
    if (net_) {
        delete net_;
        net_ = nullptr;
    }
#if NCNN_VULKAN
    /* Every process runs exactly ONE backend (scheduler/worker model), so the
     * global Vulkan instance is never shared: tear it down explicitly while
     * the process is still healthy. ncnn's exit-time static destruction of
     * the global GPU instance crashes on some drivers with 0xC0000409 (stack
     * buffer overrun, seen on Intel Iris Xe) AFTER main() returns and after
     * the benchmark record was already written; destroying it here avoids
     * that teardown path entirely. */
    if (gpu_device_ >= 0) {
        ncnn::destroy_gpu_instance();
        gpu_device_ = -1;
    }
#endif
    for (size_t i = 0; i < input_bufs_.size(); ++i) {
        if (input_bufs_[i] && !input_external_[i]) {
            free(input_bufs_[i]);
        }
    }
    input_bufs_.clear();
}

BackendPtr CreateNcnnBackend(BackendId id)
{
    return std::make_unique<NCNNBackend>(id);
}

#endif /* HAVE_NCNN_BACKEND */
