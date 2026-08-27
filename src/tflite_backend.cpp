/*============================================================================
 * tflite_backend.cpp - TensorFlow Lite backend (RAII)
 *============================================================================*/

#include "backend_interface.hpp"
#include "log.hpp"

#ifdef HAVE_TFLITE_BACKEND

#include <tensorflow/lite/c/c_api.h>
#if defined(__ANDROID__) || defined(__android__)
#include <tensorflow/lite/delegates/gpu/delegate.h>
#include <tensorflow/lite/delegates/nnapi/nnapi_delegate.h>
#ifdef HAVE_QNN_DELEGATE
#include "QNN/TFLiteDelegate/QnnTFLiteDelegate.h"
#endif
#endif
#include <tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h>

#include <chrono>
#include <cstdarg>
#include <cstring>
#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

/* ---------------------------------------------------------------------------
 * TFLite error reporter callback -- captures TFLite's internal error messages
 * into a thread-local buffer so Initialize() can read them.
 * -------------------------------------------------------------------------*/
#ifdef _MSC_VER
/* MSVC: use __declspec(thread) for thread-local storage */
static __declspec(thread) char tflite_error_buf[4096] = "";
#else
/* GCC/Clang: use __thread */
static __thread char tflite_error_buf[4096] = "";
#endif

static void TFLiteErrorReporter(void * /*user_data*/,
                                const char *format, va_list args)
{
    vsnprintf(tflite_error_buf, sizeof(tflite_error_buf), format, args);
}

/* Helper: call with TFLite model path, returns model ptr.
 * On failure, tflite_error_buf contains the real TFLite error. */
static TfLiteModel *TFLiteModelCreateWithError(const char *path)
{
    tflite_error_buf[0] = '\0';
    return TfLiteModelCreateFromFileWithErrorReporter(path,
                                                      TFLiteErrorReporter,
                                                      nullptr);
}

/* Flex delegate for Select TensorFlow ops (FlexErf, etc.).
 * Dynamically loaded at runtime so the binary works with or without
 * tensorflow-lite-select-tf-ops library. */

class TFLiteBackend : public IBackend
{
public:
    explicit TFLiteBackend(BackendId id) { id_ = id; }
    ~TFLiteBackend() override { Cleanup(); }

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
    bool CopyOutputToFloat(size_t idx, float *dst, size_t n);

    TfLiteModel *model_ = nullptr;
    TfLiteInterpreterOptions *opts_ = nullptr;
    TfLiteInterpreter *interp_ = nullptr;
    TfLiteDelegate *delegate_ = nullptr;
    TfLiteDelegate *flex_delegate_ = nullptr;

    size_t num_inputs_ = 0;
    size_t num_outputs_ = 0;
    std::vector<size_t> input_elems_;
    std::vector<size_t> output_elems_;
    std::vector<std::vector<int>> input_shapes_;
    std::vector<std::vector<int>> output_shapes_;

    std::vector<float *> input_bufs_;
    std::vector<size_t> input_buf_elems_;
    std::vector<bool> input_external_;

    double init_ms_ = 0;

    /* Tensor transfer timing (avg ms per repeat).
     * transfer_in_ms_  = feed() (NCHW->NHWC transpose + memcpy into input
     *                    tensors; GPU delegates may stage on device)
     * transfer_out_ms_ = CopyOutputToFloat snapshot memcpy (D2H for GPU) */
    double transfer_in_ms_ = 0.0;
    double transfer_out_ms_ = 0.0;
};

static TfLiteDelegate *CreateDelegate(BackendId id, int num_threads,
                                      std::string *error_out)
{
    auto fail = [&](const char *msg) -> TfLiteDelegate * {
        if (error_out) {
            *error_out = msg;
        }
        return nullptr;
    };

    switch (id) {
    case BackendId::TFLITE_XNNPACK:
    case BackendId::TFLITE_XNNPACK_FP16: {
#if defined(__ANDROID__) || defined(__android__)
        TfLiteXNNPackDelegateOptions xo = TfLiteXNNPackDelegateOptionsDefault();
        xo.num_threads = num_threads;
        xo.flags |= TFLITE_XNNPACK_DELEGATE_FLAG_ENABLE_LATEST_OPERATORS;
        if (id == BackendId::TFLITE_XNNPACK_FP16) {
            xo.flags |= TFLITE_XNNPACK_DELEGATE_FLAG_FORCE_FP16;
        }
        return TfLiteXNNPackDelegateCreate(&xo);
#else
        /* Desktop: load XNNPACK delegate from tensorflowlite_c.dll.
         * libLiteRt.dll also exports TfLiteXNNPackDelegateCreate but that
         * copy crashes with ACCESS_VIOLATION during init on x86-64.
         * The tensorflowlite_c.dll copy is verified working (4x speedup). */
        {
            static HMODULE tflite_dll = nullptr;
            static decltype(&TfLiteXNNPackDelegateCreate) pfn_create = nullptr;
            static decltype(&TfLiteXNNPackDelegateOptionsDefault) pfn_opts = nullptr;
            if (!tflite_dll) {
                tflite_dll = LoadLibraryA("tensorflowlite_c.dll");
                if (tflite_dll) {
                    pfn_create = (decltype(pfn_create))GetProcAddress(
                        tflite_dll, "TfLiteXNNPackDelegateCreate");
                    pfn_opts = (decltype(pfn_opts))GetProcAddress(
                        tflite_dll, "TfLiteXNNPackDelegateOptionsDefault");
                }
                if (!pfn_create || !pfn_opts) {
                    LOGE("TFLite: XNNPACK functions not found in tensorflowlite_c.dll");
                    return fail("XNNPACK functions not found in tensorflowlite_c.dll");
                }
            }
            TfLiteXNNPackDelegateOptions xo = pfn_opts();
            xo.num_threads = num_threads;
            xo.flags |= TFLITE_XNNPACK_DELEGATE_FLAG_ENABLE_LATEST_OPERATORS;
            if (id == BackendId::TFLITE_XNNPACK_FP16) {
                xo.flags |= TFLITE_XNNPACK_DELEGATE_FLAG_FORCE_FP16;
            }
            return pfn_create(&xo);
        }
#endif
    }
    case BackendId::TFLITE_NNAPI: {
#if defined(__ANDROID__) || defined(__android__)
        // StatefulNnApiDelegate extends TfLiteDelegate; heap-allocate so
        // the delegate outlives the interpreter. The caller (Cleanup) must
        // delete it via TfLiteDelegateFree or direct delete.
        auto *delegate = new tflite::StatefulNnApiDelegate();
        return delegate;
#else
        return fail("NNAPI only available on Android");
#endif
    }
    case BackendId::TFLITE_GPU:
    case BackendId::TFLITE_GPU_FP16: {
#if defined(__ANDROID__) || defined(__android__)
        TfLiteGpuDelegateOptionsV2 go = TfLiteGpuDelegateOptionsV2Default();
        /* Max speed: allow FP16, prefer min latency over memory savings */
        go.is_precision_loss_allowed = 1;
        go.inference_preference = TFLITE_GPU_INFERENCE_PREFERENCE_FAST_SINGLE_ANSWER;
        go.inference_priority1 = TFLITE_GPU_INFERENCE_PRIORITY_MIN_LATENCY;
        go.inference_priority2 = TFLITE_GPU_INFERENCE_PRIORITY_MIN_MEMORY_USAGE;
        go.inference_priority3 = TFLITE_GPU_INFERENCE_PRIORITY_MAX_PRECISION;
        return TfLiteGpuDelegateV2Create(&go);
#else
        return fail("GPU delegate only available on Android");
#endif
    }
    case BackendId::TFLITE_NPU: {
        /* Handled inline in Initialize() with dynamic loading */
        return nullptr;
    }
    default: {
        return fail("Unknown backend");
    }
    }
}

bool TFLiteBackend::Initialize(const char *model_path, int num_threads)
{
    auto t0 = std::chrono::high_resolution_clock::now();
    last_error_.clear();

    model_ = TFLiteModelCreateWithError(model_path);
    if (!model_) {
        LOGE("TFLite: failed to load model: %s", model_path);
        last_error_ = "TFLite: ";
        last_error_ += tflite_error_buf[0] ? tflite_error_buf : "failed to load model";
        return false;
    }

    opts_ = TfLiteInterpreterOptionsCreate();
    if (!opts_) {
        LOGE("TFLite: failed to create interpreter options");
        last_error_ = "TFLite: failed to create interpreter options";
        return false;
    }

    TfLiteInterpreterOptionsSetNumThreads(opts_, num_threads);

    std::string delegate_error;
    delegate_ = CreateDelegate(id_, num_threads, &delegate_error);
    if (delegate_) {
        TfLiteInterpreterOptionsAddDelegate(opts_, delegate_);
    }

    if (id_ == BackendId::TFLITE_NPU) {
#ifdef HAVE_QNN_DELEGATE
        TfLiteQnnDelegateOptions opts = TfLiteQnnDelegateOptionsDefault();
        opts.backend_type = kHtpBackend;
        delegate_ = TfLiteQnnDelegateCreate(&opts);
        if (!delegate_) {
            LOGW("TFLite: TfLiteQnnDelegateCreate failed, running as CPU");
        } else {
            TfLiteInterpreterOptionsAddDelegate(opts_, delegate_);
            LOGI("TFLite: TfLiteQnnDelegate created (HTP backend)");
        }
#else
        LOGW("TFLite: NPU delegate not available on this platform (QNN SDK not found)");
#endif
    }

    /* Non-CPU backends require a delegate; fail if unavailable
     * rather than silently falling back to CPU. */
    if (!delegate_ && id_ != BackendId::TFLITE_CPU) {
        std::string reason = "TFLite: delegate creation failed for backend ";
        reason += std::to_string(bid(id_));
        if (!delegate_error.empty()) {
            reason += " (" + delegate_error + ")";
        }
        LOGE("TFLite: delegate not available for backend %d, aborting", bid(id_));
        last_error_ = reason;
        return false;
    }

    /* Flex delegate temporarily disabled */
    (void)flex_delegate_;

    interp_ = TfLiteInterpreterCreate(model_, opts_);
    if (!interp_) {
        const char *tflite_err = tflite_error_buf[0] ? tflite_error_buf : "no TFLite error detail";
        LOGE("TFLite: interpreter create failed: %s", tflite_err);
        last_error_ = std::string("TFLite: interpreter create failed - ") + tflite_err;
        return false;
    }

    if (TfLiteInterpreterAllocateTensors(interp_) != kTfLiteOk) {
        LOGE("TFLite: tensor allocation failed");
        last_error_ = "TFLite: tensor allocation failed";
        return false;
    }

    num_inputs_ = TfLiteInterpreterGetInputTensorCount(interp_);
    num_outputs_ = TfLiteInterpreterGetOutputTensorCount(interp_);
    if (num_inputs_ > MAX_IO) {
        num_inputs_ = MAX_IO;
    }
    if (num_outputs_ > MAX_IO) {
        num_outputs_ = MAX_IO;
    }

    for (size_t i = 0; i < num_inputs_; ++i) {
        const TfLiteTensor *t = TfLiteInterpreterGetInputTensor(interp_, (int32_t)i);
        if (!t) {
            LOGE("TFLite: null input tensor at %zu", i);
            last_error_ = "TFLite: null input tensor";
            return false;
        }
        size_t elems = (size_t)TfLiteTensorByteSize(t) / sizeof(float);
        input_elems_.push_back(elems);
        std::vector<int> dims;
        int32_t nd = TfLiteTensorNumDims(t);
        for (int32_t d = 0; d < nd; ++d) {
            dims.push_back(TfLiteTensorDim(t, d));
        }
        input_shapes_.push_back(std::move(dims));
    }

    for (size_t i = 0; i < num_outputs_; ++i) {
        const TfLiteTensor *t = TfLiteInterpreterGetOutputTensor(interp_, (int32_t)i);
        if (!t) {
            LOGE("TFLite: null output tensor at %zu", i);
            last_error_ = "TFLite: null output tensor";
            return false;
        }
        size_t elems = (size_t)TfLiteTensorByteSize(t) / sizeof(float);
        output_elems_.push_back(elems);
        std::vector<int> dims;
        int32_t nd = TfLiteTensorNumDims(t);
        for (int32_t d = 0; d < nd; ++d) {
            dims.push_back(TfLiteTensorDim(t, d));
        }
        output_shapes_.push_back(std::move(dims));
    }

    init_ms_ = std::chrono::duration<double, std::milli>(
                   std::chrono::high_resolution_clock::now() - t0)
                   .count();

    LOGI("TFLite: init complete (%.1f ms), %zu in, %zu out",
         init_ms_, num_inputs_, num_outputs_);
    return true;
}

bool TFLiteBackend::QueryIOInfo(std::string &is, size_t &ie,
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
            is += ";";
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
            os += ";";
        }
        os += buf;
        oe += output_elems_[i];
    }
    return true;
}

void TFLiteBackend::SetSharedInput(const float *const *data, const size_t *counts)
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

bool TFLiteBackend::CopyOutputToFloat(size_t idx, float *dst, size_t n)
{
    const TfLiteTensor *t = TfLiteInterpreterGetOutputTensor(interp_, (int32_t)idx);
    if (!t) {
        return false;
    }
    switch (TfLiteTensorType(t)) {
    case kTfLiteFloat32: {
        memcpy(dst, TfLiteTensorData(t), n * sizeof(float));
        break;
    }
    case kTfLiteFloat16: {
        const uint16_t *src = (const uint16_t *)TfLiteTensorData(t);
        /* Simple half-to-float: not fully IEEE, but OK for benchmarking */
        for (size_t j = 0; j < n; ++j) {
            dst[j] = (float)src[j]; /* placeholder */
            break;
        }
    }
    default: {
        memset(dst, 0, n * sizeof(float));
        return false;
    }
    }
    return true;
}

bool TFLiteBackend::RunBenchmark(int warmup, int repeat, double &total,
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

    /* Allocate snapshots */
    std::vector<std::vector<float>> snaps(num_outputs_);
    for (size_t i = 0; i < num_outputs_; ++i) {
        snaps[i].resize(output_elems_[i] > 0 ? output_elems_[i] : 1);
    }

    auto feed = [&]() {
        for (size_t i = 0; i < num_inputs_; ++i) {
            TfLiteTensor *t = TfLiteInterpreterGetInputTensor(interp_, (int32_t)i);
            if (!t || !input_bufs_[i]) {
                continue;
            }
            float *dst = (float *)TfLiteTensorData(t);
            /* Shared inputs are in NCHW; TFLite expects NHWC.
             * Transpose 4D tensors from NCHW to NHWC. */
            if (input_shapes_[i].size() == 4) {
                int N = input_shapes_[i][0], H = input_shapes_[i][1];
                int W = input_shapes_[i][2], C = input_shapes_[i][3];
                const float *src = input_bufs_[i];
                for (int n = 0; n < N; n++) {
                    for (int h = 0; h < H; h++) {
                        for (int w = 0; w < W; w++) {
                            for (int c = 0; c < C; c++) {
                                dst[n * H * W * C + h * W * C + w * C + c] =
                                    src[n * C * H * W + c * H * W + h * W + w];
                            }
                        }
                    }
                }
            } else {
                memcpy(dst, input_bufs_[i], input_elems_[i] * sizeof(float));
            }
        }
    };

    (void)warmup;
    for (int w = 0; w < warmup; ++w) {
        feed();
        if (TfLiteInterpreterInvoke(interp_) != kTfLiteOk) {
            LOGE("TFLite: invoke failed (warmup)");
            return false;
        }
    }

    for (int r = 0; r < repeat; ++r) {
        /* Time input upload: feed() (NCHW->NHWC transpose + memcpy). */
        auto t_in0 = std::chrono::high_resolution_clock::now();
        feed();
        auto t_in1 = std::chrono::high_resolution_clock::now();
        transfer_in_ms_ +=
            std::chrono::duration<double, std::milli>(t_in1 - t_in0).count();

        auto t0 = std::chrono::high_resolution_clock::now();
        if (TfLiteInterpreterInvoke(interp_) != kTfLiteOk) {
            LOGE("TFLite: invoke failed at run %d", r);
            return false;
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        LOGD("TFLite: run %d took %.3f ms", r, ms);

        /* Time output download: snapshot memcpy (D2H for GPU delegates). */
        auto t_out0 = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < num_outputs_; ++i) {
            CopyOutputToFloat(i, snaps[i].data(), output_elems_[i]);
        }
        auto t_out1 = std::chrono::high_resolution_clock::now();
        transfer_out_ms_ +=
            std::chrono::duration<double, std::milli>(t_out1 - t_out0).count();

        total += ms;
        if (ms > maxv) {
            maxv = ms;
            maxi = r;
        }
        if (ms < minv) {
            minv = ms;
        }
    }

    odata.resize(num_outputs_);
    oelems.resize(num_outputs_);
    oshapes.resize(num_outputs_);
    odims.resize(num_outputs_);
    for (size_t i = 0; i < num_outputs_; ++i) {
        size_t n = output_elems_[i];
        float *buf = (float *)malloc(n * sizeof(float));
        if (!buf) {
            LOGE("TFLite: malloc(%zu) failed at output %zu, due to %s, %d", n * sizeof(float), i, strerror(errno), errno);
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
    /* Normalize transfer time to avg ms per repeat. */
    if (repeat > 0) {
        transfer_in_ms_ /= (double)repeat;
        transfer_out_ms_ /= (double)repeat;
    }
    return true;
}

void TFLiteBackend::GetTiming(std::array<double, 10> &timing)
{
    timing.fill(0);
    timing[0] = init_ms_;
}

void TFLiteBackend::GetTransferTiming(double &transfer_in_ms,
                                      double &transfer_out_ms)
{
    transfer_in_ms = transfer_in_ms_;
    transfer_out_ms = transfer_out_ms_;
}

void TFLiteBackend::Cleanup()
{
    if (interp_) {
        TfLiteInterpreterDelete(interp_);
        interp_ = nullptr;
    }
    if (opts_) {
        TfLiteInterpreterOptionsDelete(opts_);
        opts_ = nullptr;
    }
    if (delegate_) {
#if defined(__ANDROID__) || defined(__android__)
        if (id_ == BackendId::TFLITE_GPU) {
            TfLiteGpuDelegateV2Delete(delegate_);
        } else if (id_ == BackendId::TFLITE_NNAPI) {
            delete static_cast<tflite::StatefulNnApiDelegate *>(delegate_);
        } else if (id_ == BackendId::TFLITE_NPU) {
            TfLiteQnnDelegateDelete(delegate_);
        }
#endif
        delegate_ = nullptr;
    }
    if (flex_delegate_) {
        typedef void (*FlexDeleteFunc)(TfLiteDelegate *);
#ifdef _WIN32
        HMODULE tflib = GetModuleHandleA("tensorflowlite_c.dll");
        auto flex_delete = (FlexDeleteFunc)(tflib ? GetProcAddress(tflib, "TfLiteFlexDelegateDelete") : nullptr);
#else
        auto flex_delete = (FlexDeleteFunc)dlsym(RTLD_DEFAULT, "TfLiteFlexDelegateDelete");
#endif
        if (flex_delete) {
            flex_delete(flex_delegate_);
        }
        flex_delegate_ = nullptr;
    }
    if (model_) {
        TfLiteModelDelete(model_);
        model_ = nullptr;
    }

    for (size_t i = 0; i < input_bufs_.size(); ++i) {
        if (input_bufs_[i] && !input_external_[i]) {
            free(input_bufs_[i]);
        }
    }
    input_bufs_.clear();
}

BackendPtr CreateTfliteBackend(BackendId id)
{
    return std::make_unique<TFLiteBackend>(id);
}

#endif /* HAVE_TFLITE_BACKEND */
