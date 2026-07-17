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
#endif
#include <tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h>

#include <chrono>
#include <cstring>
#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

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
    bool PrepareInputs(float *&fd, size_t &fe, const char *arg,
                       bool random, const float *const *ext, const size_t *extc) override;
    void SetSharedInput(const float *const *data, const size_t *counts) override;
    bool RunBenchmark(int warmup, int repeat, double &total, double &maxv,
                      double &minv, int &maxi, std::vector<float *> &odata,
                      std::vector<size_t> &oelems,
                      std::vector<std::array<size_t, MAX_DIMENSIONS>> &oshapes,
                      std::vector<size_t> &odims) override;
    void GetTiming(std::array<double, 10> &timing) override;
    bool SaveOutputs(const char *suffix) override;

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
};

static TfLiteDelegate *CreateDelegate(BackendId id, int num_threads)
{
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
        /* Desktop: XNNPACK delegate creation hangs on some Intel GPU drivers.
         * TFLITE_XNNPACK_FP16 runs as CPU (no delegate). */
        if (id == BackendId::TFLITE_XNNPACK_FP16) {
            LOGW("TFLite: XNNPACK_FP16 not available on this platform, running as CPU");
            return nullptr;
        }
        TfLiteXNNPackDelegateOptions xo = TfLiteXNNPackDelegateOptionsDefault();
        xo.num_threads = num_threads;
        return TfLiteXNNPackDelegateCreate(&xo);
#endif
    }
    case BackendId::TFLITE_NNAPI:
#if defined(__ANDROID__) || defined(__android__)
    {
        // StatefulNnApiDelegate extends TfLiteDelegate; heap-allocate so
        // the delegate outlives the interpreter. The caller (Cleanup) must
        // delete it via TfLiteDelegateFree or direct delete.
        auto *delegate = new tflite::StatefulNnApiDelegate();
        return delegate;
    }
#else
        return nullptr;
#endif
    case BackendId::TFLITE_GPU:
    case BackendId::TFLITE_GPU_FP16:
#if defined(__ANDROID__) || defined(__android__)
    {
        TfLiteGpuDelegateOptionsV2 go = TfLiteGpuDelegateOptionsV2Default();
        /* Max speed: allow FP16, prefer min latency over memory savings */
        go.is_precision_loss_allowed = 1;
        go.inference_preference = TFLITE_GPU_INFERENCE_PREFERENCE_FAST_SINGLE_ANSWER;
        go.inference_priority1 = TFLITE_GPU_INFERENCE_PRIORITY_MIN_LATENCY;
        go.inference_priority2 = TFLITE_GPU_INFERENCE_PRIORITY_MIN_MEMORY_USAGE;
        go.inference_priority3 = TFLITE_GPU_INFERENCE_PRIORITY_MAX_PRECISION;
        return TfLiteGpuDelegateV2Create(&go);
    }
#else
        return nullptr;
#endif
    case BackendId::TFLITE_NPU:
#if defined(__ANDROID__) || defined(__android__)
    {
        /* QNN TFLite delegate for NPU acceleration.
         * Requires Qualcomm QNN SDK and TFLite delegate library.
         * Include: #include <QnnTFLiteDelegate.h>
         * Link with qnntflitedelegate or libQnnTFLiteDelegate.so */
        // TODO: Replace with actual QNN TFLite delegate initialization
        // Example:
        //   QnnTFLiteDelegateOptions qo = QnnTFLiteDelegateOptionsDefault();
        //   qo.backend_type = QNN_BACKEND_HTP;  // or QNN_BACKEND_ADRENO
        //   return QnnTFLiteDelegateCreate(&qo);
        LOGW("TFLite: NPU delegate not implemented - provide QNN TFLite delegate header");
        return nullptr;
    }
#else
        return nullptr;
#endif
    default:
        return nullptr;
    }
}

bool TFLiteBackend::Initialize(const char *model_path, int num_threads)
{
    auto t0 = std::chrono::high_resolution_clock::now();

    LOGI("TFLite: step 1/7: TfLiteModelCreateFromFile(%s)", model_path);
    model_ = TfLiteModelCreateFromFile(model_path);
    if (!model_) {
        LOGE("TFLite: failed to load model: %s", model_path);
        return false;
    }
    LOGI("TFLite: step 1 OK");

    LOGI("TFLite: step 2/7: TfLiteInterpreterOptionsCreate");
    opts_ = TfLiteInterpreterOptionsCreate();
    if (!opts_) {
        LOGE("TFLite: failed to create interpreter options");
        return false;
    }
    LOGI("TFLite: step 2 OK");

    TfLiteInterpreterOptionsSetNumThreads(opts_, num_threads);

    LOGI("TFLite: step 3/7: CreateDelegate");
    delegate_ = CreateDelegate(id_, num_threads);
    if (delegate_) {
        LOGI("TFLite: step 3a: TfLiteInterpreterOptionsAddDelegate");
        TfLiteInterpreterOptionsAddDelegate(opts_, delegate_);
        LOGI("TFLite: delegate attached for backend %d", bid(id_));
    }
    LOGI("TFLite: step 3 OK");

    /* Flex delegate temporarily disabled */
    (void)flex_delegate_;

    LOGI("TFLite: step 4/7: TfLiteInterpreterCreate");
    interp_ = TfLiteInterpreterCreate(model_, opts_);
    if (!interp_) {
        LOGE("TFLite: interpreter create failed");
        return false;
    }
    LOGI("TFLite: step 4 OK");

    LOGI("TFLite: step 5/7: TfLiteInterpreterAllocateTensors");
    if (TfLiteInterpreterAllocateTensors(interp_) != kTfLiteOk) {
        LOGE("TFLite: tensor allocation failed");
        return false;
    }
    LOGI("TFLite: step 5 OK");

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

bool TFLiteBackend::PrepareInputs(float *&fd, size_t &fe, const char * /*arg*/,
                                  bool random, const float *const *ext,
                                  const size_t *extc)
{
    for (size_t i = 0; i < num_inputs_; ++i) {
        size_t n = input_elems_[i] > 0 ? input_elems_[i] : 1;
        if (i >= input_bufs_.size()) {
            input_bufs_.resize(i + 1, nullptr);
            input_buf_elems_.resize(i + 1, 0);
            input_external_.resize(i + 1, false);
        }
        if (input_bufs_[i] && !input_external_[i]) {
            free(input_bufs_[i]);
        }

        if (ext && ext[i] && extc && extc[i] == n) {
            input_bufs_[i] = const_cast<float *>(ext[i]);
            input_external_[i] = true;
        } else {
            float *buf = (float *)malloc(n * sizeof(float));
            if (!buf) {
                return false;
            }
            if (random) {
                for (size_t j = 0; j < n; ++j) {
                    buf[j] = (float)rand() / (float)RAND_MAX;
                }
            } else {
                memset(buf, 0, n * sizeof(float));
            }
            input_bufs_[i] = buf;
            input_external_[i] = false;
        }
        input_buf_elems_[i] = n;
    }
    fd = num_inputs_ > 0 ? input_bufs_[0] : nullptr;
    fe = num_inputs_ > 0 ? input_buf_elems_[0] : 0;
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
    case kTfLiteFloat32:
        memcpy(dst, TfLiteTensorData(t), n * sizeof(float));
        break;
    case kTfLiteFloat16: {
        const uint16_t *src = (const uint16_t *)TfLiteTensorData(t);
        /* Simple half-to-float: not fully IEEE, but OK for benchmarking */
        for (size_t j = 0; j < n; ++j) {
            dst[j] = (float)src[j]; /* placeholder */
            break;
        }
    }
    default:
        memset(dst, 0, n * sizeof(float));
        return false;
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
                        for (int w = 0; w < W; w++)
                            for (int c = 0; c < C; c++)
                                dst[n * H * W * C + h * W * C + w * C + c] =
                                    src[n * C * H * W + c * H * W + h * W + w];
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
        feed();
        auto t0 = std::chrono::high_resolution_clock::now();
        if (TfLiteInterpreterInvoke(interp_) != kTfLiteOk) {
            LOGE("TFLite: invoke failed at run %d", r);
            return false;
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        for (size_t i = 0; i < num_outputs_; ++i) {
            CopyOutputToFloat(i, snaps[i].data(), output_elems_[i]);
        }

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

void TFLiteBackend::GetTiming(std::array<double, 10> &timing)
{
    timing.fill(0);
    timing[0] = init_ms_;
}

bool TFLiteBackend::SaveOutputs(const char * /*suffix*/) { return true; }

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
        } else if (id_ == BackendId::TFLITE_NNAPI)
            delete static_cast<tflite::StatefulNnApiDelegate *>(delegate_);
        else if (id_ == BackendId::TFLITE_NPU)
            TfLiteGpuDelegateV2Delete(delegate_); /* or QnnTFLiteDelegateDelete if available */
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
