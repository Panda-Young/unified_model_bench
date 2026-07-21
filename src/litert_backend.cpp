/*============================================================================
 * litert_backend.cpp - Google LiteRT backend (RAII)
 *
 * LiteRT is the next-generation TFLite runtime. It loads .tflite models
 * and supports NPU/GPU/CPU accelerators via dispatch libraries.
 *
 * NPU dispatch:
 *   - Qualcomm HTP: libLiteRtDispatch_Qualcomm.so (SoC-specific v69~v81)
 *   - Google Tensor: libLiteRtDispatch_GoogleTensor.so
 *   The dispatch library is selected at runtime based on ro.soc.model.
 *============================================================================*/

#include "backend_interface.hpp"
#include "log.hpp"

#ifdef HAVE_LITERT_BACKEND

#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <malloc.h>
#endif

/* LiteRT C API headers (from litert_cc_sdk) */
#include "litert/c/litert_common.h"
#include "litert/c/litert_compiled_model.h"
#include "litert/c/litert_environment.h"
#include "litert/c/litert_environment_options.h"
#include "litert/c/litert_layout.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_options.h"
#include "litert/c/litert_tensor_buffer.h"
#include "litert/c/litert_tensor_buffer_requirements.h"
#include "litert/c/litert_tensor_buffer_types.h"
#include "litert/c/litert_opaque_options.h"
#include "litert/c/options/litert_qualcomm_options.h"

/* ---------------------------------------------------------------------------
 * LiteRtStatus -> readable string (LiteRT has no built-in status-to-string)
 * -------------------------------------------------------------------------*/
static const char *LiteRtStatusStr(LiteRtStatus st)
{
    switch (st) {
    case kLiteRtStatusOk:
        return "Ok";
    case kLiteRtStatusErrorInvalidArgument:
        return "InvalidArgument";
    case kLiteRtStatusErrorMemoryAllocationFailure:
        return "MemoryAllocationFailure";
    case kLiteRtStatusErrorRuntimeFailure:
        return "RuntimeFailure";
    case kLiteRtStatusErrorMissingInputTensor:
        return "MissingInputTensor";
    case kLiteRtStatusErrorUnsupported:
        return "Unsupported";
    case kLiteRtStatusErrorNotFound:
        return "NotFound";
    case kLiteRtStatusErrorTimeoutExpired:
        return "TimeoutExpired";
    case kLiteRtStatusErrorWrongVersion:
        return "WrongVersion";
    case kLiteRtStatusErrorUnknown:
        return "Unknown";
    case kLiteRtStatusErrorAlreadyExists:
        return "AlreadyExists";
    case kLiteRtStatusErrorFileIO:
        return "FileIO";
    case kLiteRtStatusErrorInvalidFlatbuffer:
        return "InvalidFlatbuffer";
    case kLiteRtStatusErrorDynamicLoading:
        return "DynamicLoading";
    case kLiteRtStatusErrorSerialization:
        return "Serialization";
    case kLiteRtStatusErrorCompilation:
        return "Compilation";
    case kLiteRtStatusErrorIndexOOB:
        return "IndexOutOfBounds";
    default:
        return "UnknownStatus";
    }
}

/* ---------------------------------------------------------------------------
 * LiteRTBackend
 * -------------------------------------------------------------------------*/
class LiteRTBackend : public IBackend
{
public:
    explicit LiteRTBackend(BackendId id) { id_ = id; }
    ~LiteRTBackend() override { Cleanup(); }

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

    bool QueryIOMetadata();

    LiteRtEnvironment env_ = nullptr;
    LiteRtModel model_ = nullptr;
    LiteRtOptions comp_opts_ = nullptr;
    LiteRtCompiledModel compiled_ = nullptr;

    size_t num_inputs_ = 0;
    size_t num_outputs_ = 0;
    std::vector<size_t> input_elems_;
    std::vector<size_t> output_elems_;
    std::vector<std::vector<int32_t>> input_shapes_;
    std::vector<std::vector<int32_t>> output_shapes_;

    std::vector<float *> input_bufs_;
    std::vector<size_t> input_buf_elems_;
    std::vector<bool> input_external_;

    double init_ms_ = 0;

    /* NPU dispatch directory string (must outlive env_ which holds pointer) */
    std::string npu_dispatch_dir_;
};

/* ---------------------------------------------------------------------------
 * Query IO metadata from model signatures
 * -------------------------------------------------------------------------*/
bool LiteRTBackend::QueryIOMetadata()
{
    LiteRtParamIndex num_sigs;
    LiteRtStatus st_sig = LiteRtGetNumModelSignatures(model_, &num_sigs);
    if (st_sig != kLiteRtStatusOk || num_sigs == 0) {
        LOGE("LiteRT: no signatures in model");
        last_error_ = std::string("LiteRT: GetNumModelSignatures: ") + LiteRtStatusStr(st_sig);
        return false;
    }
    /* Use first signature */
    LiteRtSignature sig;
    LiteRtStatus st_get = LiteRtGetModelSignature(model_, 0, &sig);
    if (st_get != kLiteRtStatusOk) {
        LOGE("LiteRT: failed to get signature");
        last_error_ = std::string("LiteRT: GetModelSignature: ") + LiteRtStatusStr(st_get);
        return false;
    }

    /* Input count & info */
    LiteRtParamIndex num_in, num_out;
    {
        LiteRtStatus st = LiteRtGetNumSignatureInputs(sig, &num_in);
        if (st != kLiteRtStatusOk) {
            last_error_ = std::string("LiteRT: GetNumSignatureInputs: ") + LiteRtStatusStr(st);
            return false;
        }
    }
    {
        LiteRtStatus st = LiteRtGetNumSignatureOutputs(sig, &num_out);
        if (st != kLiteRtStatusOk) {
            last_error_ = std::string("LiteRT: GetNumSignatureOutputs: ") + LiteRtStatusStr(st);
            return false;
        }
    }
    num_inputs_ = (size_t)num_in;
    num_outputs_ = (size_t)num_out;

    for (LiteRtParamIndex i = 0; i < num_in; ++i) {
        LiteRtTensor tensor;
        {
            LiteRtStatus st = LiteRtGetSignatureInputTensorByIndex(sig, i, &tensor);
            if (st != kLiteRtStatusOk) {
                last_error_ = std::string("LiteRT: GetSignatureInputTensorByIndex: ") + LiteRtStatusStr(st);
                return false;
            }
        }
        LiteRtRankedTensorType rtype;
        {
            LiteRtStatus st = LiteRtGetRankedTensorType(tensor, &rtype);
            if (st != kLiteRtStatusOk) {
                last_error_ = std::string("LiteRT: GetRankedTensorType: ") + LiteRtStatusStr(st);
                return false;
            }
        }
        int32_t rank = (int32_t)rtype.layout.rank;
        const int32_t *dims = rtype.layout.dimensions;
        std::vector<int32_t> shape(dims, dims + rank);
        input_shapes_.push_back(std::move(shape));
        size_t elems = 1;
        for (int d = 0; d < rank; ++d) {
            elems *= (size_t)dims[d];
        }
        input_elems_.push_back(elems);
    }

    for (LiteRtParamIndex i = 0; i < num_out; ++i) {
        LiteRtTensor tensor;
        {
            LiteRtStatus st = LiteRtGetSignatureOutputTensorByIndex(sig, i, &tensor);
            if (st != kLiteRtStatusOk) {
                last_error_ = std::string("LiteRT: GetSignatureOutputTensorByIndex: ") + LiteRtStatusStr(st);
                return false;
            }
        }
        LiteRtRankedTensorType rtype;
        {
            LiteRtStatus st = LiteRtGetRankedTensorType(tensor, &rtype);
            if (st != kLiteRtStatusOk) {
                last_error_ = std::string("LiteRT: GetRankedTensorType: ") + LiteRtStatusStr(st);
                return false;
            }
        }
        int32_t rank = (int32_t)rtype.layout.rank;
        const int32_t *dims = rtype.layout.dimensions;
        std::vector<int32_t> shape(dims, dims + rank);
        output_shapes_.push_back(std::move(shape));
        size_t elems = 1;
        for (int d = 0; d < rank; ++d) {
            elems *= (size_t)dims[d];
        }
        output_elems_.push_back(elems);
    }

    LOGI("LiteRT: IO metadata: %zu in, %zu out", num_inputs_, num_outputs_);
    return true;
}

/* ---------------------------------------------------------------------------
 * Initialize
 * -------------------------------------------------------------------------*/
bool LiteRTBackend::Initialize(const char *model_path, int num_threads)
{
    auto t0 = std::chrono::high_resolution_clock::now();
    last_error_.clear();
    (void)num_threads;

    /* 1. Build environment options for NPU dispatch */
    LiteRtEnvOption env_opts[2];
    int num_env_opts = 0;

    if (id_ == BackendId::LITERT_NPU) {
        /* Set dispatch library dir for Qualcomm NPU.
         * Store in member var so c_str() outlives env_ which holds the pointer. */
        npu_dispatch_dir_ = "/data/local/tmp/bench_test/qnn";
        LOGI("LiteRT: NPU dispatch dir: %s", npu_dispatch_dir_.c_str());

        LiteRtAny val;
        val.type = kLiteRtAnyTypeString;
        val.str_value = npu_dispatch_dir_.c_str();
        env_opts[num_env_opts].tag = kLiteRtEnvOptionTagDispatchLibraryDir;
        env_opts[num_env_opts].value = val;
        ++num_env_opts;
    }

    /* 2. Create environment */
    LiteRtStatus st_env;
    st_env = LiteRtCreateEnvironment(num_env_opts, env_opts, &env_);
    if (st_env != kLiteRtStatusOk) {
        LOGE("LiteRT: CreateEnvironment failed: %d", (int)st_env);
        last_error_ = std::string("LiteRT: CreateEnvironment: ") + LiteRtStatusStr(st_env);
        return false;
    }

    /* 3. Load model */
    LiteRtStatus st_mod;
    st_mod = LiteRtCreateModelFromFile(env_, model_path, &model_);
    if (st_mod != kLiteRtStatusOk) {
        LOGE("LiteRT: CreateModelFromFile(%s) failed: %d", model_path, (int)st_mod);
        last_error_ = std::string("LiteRT: CreateModelFromFile: ") + LiteRtStatusStr(st_mod);
        return false;
    }

    /* 4. Query IO metadata */
    if (!QueryIOMetadata()) {
        if (last_error_.empty())
            last_error_ = "LiteRT: QueryIOMetadata failed";
        return false;
    }

    /* 5. Create compilation options with accelerator selection */
    LiteRtStatus st_opt;
    st_opt = LiteRtCreateOptions(&comp_opts_);
    if (st_opt != kLiteRtStatusOk) {
        LOGE("LiteRT: CreateOptions failed: %d", (int)st_opt);
        last_error_ = std::string("LiteRT: CreateOptions: ") + LiteRtStatusStr(st_opt);
        return false;
    }

    {
        LiteRtHwAcceleratorSet accel;
        if (id_ == BackendId::LITERT_GPU || id_ == BackendId::LITERT_GPU_FP16) {
            accel = kLiteRtHwAcceleratorGpu;
        } else if (id_ == BackendId::LITERT_NPU || id_ == BackendId::LITERT_NPU_FP16) {
            accel = kLiteRtHwAcceleratorNpu;
        } else {
            accel = kLiteRtHwAcceleratorCpu;
        }
        LiteRtStatus st_hw;
        st_hw = LiteRtSetOptionsHardwareAccelerators(comp_opts_, accel);
        if (st_hw != kLiteRtStatusOk) {
            LOGE("LiteRT: SetOptionsHardwareAccelerators failed: %d", (int)st_hw);
            last_error_ = std::string("LiteRT: SetAccelerators: ") + LiteRtStatusStr(st_hw);
            return false;
        }
    }

    /* 5b. For NPU backends: configure QNN-specific options.
     * Without this, the dispatch receives "Null Qualcomm options"
     * and CreateCompiledModel returns error 504.
     * NOTE: QNN functions only available in Android libLiteRt.so. */
#if defined(__ANDROID__) || defined(__android__)
    if (id_ == BackendId::LITERT_NPU || id_ == BackendId::LITERT_NPU_FP16) {
        LrtQualcommOptions qnn_opts = nullptr;
        LiteRtStatus st_qnn = LrtCreateQualcommOptions(&qnn_opts);
        if (st_qnn == kLiteRtStatusOk && qnn_opts) {
            LrtQualcommOptionsSetBackend(qnn_opts, kLiteRtQualcommBackendHtp);
            LrtQualcommOptionsSetHtpPerformanceMode(
                qnn_opts, kLiteRtQualcommHtpPerformanceModeBurst);
            LrtQualcommOptionsSetOptimizationLevel(
                qnn_opts, kHtpOptimizeForInferenceO3);

            /* Extract opaque payload and attach to compilation options */
            const char *payload_id = nullptr;
            void *payload_data = nullptr;
            void (*payload_dtor)(void *) = nullptr;
            if (LrtGetOpaqueQualcommOptionsData(
                    qnn_opts, &payload_id, &payload_data,
                    &payload_dtor) == kLiteRtStatusOk) {
                LiteRtOpaqueOptions opaque = nullptr;
                if (LiteRtCreateOpaqueOptions(
                        payload_id, payload_data, payload_dtor,
                        &opaque) == kLiteRtStatusOk) {
                    LiteRtAddOpaqueOptions(comp_opts_, opaque);
                    LOGI("LiteRT: QNN HTP options configured (burst, O3)");
                }
            }
            LrtDestroyQualcommOptions(qnn_opts);
        } else {
            LOGW("LiteRT: LrtCreateQualcommOptions failed (%d), "
                 "proceeding without QNN options", (int)st_qnn);
        }
    }
#endif

    /* 6. Create compiled model */
    LiteRtStatus st_comp;
    st_comp = LiteRtCreateCompiledModel(env_, model_, comp_opts_, &compiled_);
    if (st_comp != kLiteRtStatusOk) {
        LOGE("LiteRT: CreateCompiledModel failed: %d", (int)st_comp);
        last_error_ = std::string("LiteRT: CreateCompiledModel: ") + LiteRtStatusStr(st_comp);
        return false;
    }

    init_ms_ = std::chrono::duration<double, std::milli>(
                   std::chrono::high_resolution_clock::now() - t0)
                   .count();
    LOGI("LiteRT: init complete (%.1f ms), %zu in, %zu out",
         init_ms_, num_inputs_, num_outputs_);
    return true;
}

/* ---------------------------------------------------------------------------
 * QueryIOInfo
 * -------------------------------------------------------------------------*/
bool LiteRTBackend::QueryIOInfo(std::string &is, size_t &ie,
                                std::string &os, size_t &oe)
{
    is.clear();
    ie = 0;
    for (size_t i = 0; i < num_inputs_; ++i) {
        char buf[128] = {};
        int off = snprintf(buf, sizeof(buf), "[");
        for (size_t d = 0; d < input_shapes_[i].size(); ++d) {
            off += snprintf(buf + off, sizeof(buf) - off, "%s%ld", d > 0 ? "," : "", (long)input_shapes_[i][d]);
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
            off += snprintf(buf + off, sizeof(buf) - off, "%s%ld", d > 0 ? "," : "", (long)output_shapes_[i][d]);
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

/* ---------------------------------------------------------------------------
 * PrepareInputs / SetSharedInput
 * -------------------------------------------------------------------------*/
bool LiteRTBackend::PrepareInputs(float *&fd, size_t &fe, const char *,
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

void LiteRTBackend::SetSharedInput(const float *const *data, const size_t *counts)
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

/* ---------------------------------------------------------------------------
 * RunBenchmark
 * -------------------------------------------------------------------------*/
bool LiteRTBackend::RunBenchmark(int warmup, int repeat, double &total,
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

    /* Helper: clean up N first buffers (on partial failure) */
    auto cleanup_bufs = [](std::vector<LiteRtTensorBuffer> &bufs, size_t n) {
        for (size_t j = 0; j < n; ++j) {
            if (bufs[j])
                LiteRtDestroyTensorBuffer(bufs[j]);
        }
    };

    /* Helper: create input tensor buffers from host memory with proper alignment */
    auto create_input_buffers = [&](std::vector<LiteRtTensorBuffer> &bufs) -> bool {
        bufs.resize(num_inputs_, nullptr);
        for (size_t i = 0; i < num_inputs_; ++i) {
            /* Query buffer requirements */
            LiteRtTensorBufferRequirements reqs = nullptr;
            if (LiteRtGetCompiledModelInputBufferRequirements(
                    compiled_, 0, i, &reqs) != kLiteRtStatusOk) {
                LOGE("LiteRT: failed to get input %zu buffer requirements", i);
                cleanup_bufs(bufs, i);
                return false;
            }

            size_t buffer_size = 0;
            LiteRtGetTensorBufferRequirementsBufferSize(reqs, &buffer_size);

            /* Allocate aligned host memory and copy input data */
            void *host_mem = nullptr;
#if defined(_WIN32)
            host_mem = _aligned_malloc(buffer_size > 0 ? buffer_size : 1,
                                       LITERT_HOST_MEMORY_BUFFER_ALIGNMENT);
#else
            if (posix_memalign(&host_mem, LITERT_HOST_MEMORY_BUFFER_ALIGNMENT,
                               buffer_size > 0 ? buffer_size : 1) != 0)
                host_mem = nullptr;
#endif
            if (!host_mem) {
                LOGE("LiteRT: failed to allocate %zu bytes for input %zu",
                     buffer_size, i);
                cleanup_bufs(bufs, i);
                return false;
            }
            /* Copy input data into the aligned buffer */
            size_t copy_bytes = input_elems_[i] * sizeof(float);
            if (copy_bytes > buffer_size) {
                copy_bytes = buffer_size;
            }
            memcpy(host_mem, input_bufs_[i], copy_bytes);

            /* Create ranked tensor type from cached shape info */
            LiteRtRankedTensorType ttype;
            ttype.element_type = kLiteRtElementTypeFloat32;
            ttype.layout.rank = (unsigned)input_shapes_[i].size();
            ttype.layout.has_strides = false;
            memset(ttype.layout.dimensions, 0, sizeof(ttype.layout.dimensions));
            for (size_t d = 0; d < input_shapes_[i].size(); ++d) {
                ttype.layout.dimensions[d] = input_shapes_[i][d];
            }

            if (LiteRtCreateTensorBufferFromHostMemory(
                    &ttype, host_mem, buffer_size,
#if defined(_WIN32)
                    [](void *p) { _aligned_free(p); },
#else
                    [](void *p) { free(p); },
#endif
                    &bufs[i]) != kLiteRtStatusOk) {
                LOGE("LiteRT: failed to create input buffer %zu", i);
                cleanup_bufs(bufs, i);
#if defined(_WIN32)
                _aligned_free(host_mem);
#else
                free(host_mem);
#endif
                return false;
            }
        }
        return true;
    };

    /* Helper: create output tensor buffers using buffer requirements */
    auto create_output_buffers = [&](std::vector<LiteRtTensorBuffer> &bufs) -> bool {
        bufs.resize(num_outputs_, nullptr);
        for (size_t i = 0; i < num_outputs_; ++i) {
            /* Query buffer requirements from compiled model */
            LiteRtTensorBufferRequirements reqs = nullptr;
            if (LiteRtGetCompiledModelOutputBufferRequirements(
                    compiled_, 0, i, &reqs) != kLiteRtStatusOk) {
                LOGE("LiteRT: failed to get output %zu buffer requirements", i);
                cleanup_bufs(bufs, i);
                return false;
            }

            size_t buffer_size = 0;
            LiteRtGetTensorBufferRequirementsBufferSize(reqs, &buffer_size);

            /* Allocate host memory with required alignment */
            void *host_mem = nullptr;
#if defined(_WIN32)
            host_mem = _aligned_malloc(buffer_size > 0 ? buffer_size : 1,
                                       LITERT_HOST_MEMORY_BUFFER_ALIGNMENT);
#else
            if (posix_memalign(&host_mem, LITERT_HOST_MEMORY_BUFFER_ALIGNMENT,
                               buffer_size > 0 ? buffer_size : 1) != 0)
                host_mem = nullptr;
#endif
            if (!host_mem) {
                LOGE("LiteRT: failed to allocate %zu bytes for output %zu",
                     buffer_size, i);
                cleanup_bufs(bufs, i);
                return false;
            }
            memset(host_mem, 0, buffer_size);

            /* Create ranked tensor type from cached shape info */
            LiteRtRankedTensorType ttype;
            ttype.element_type = kLiteRtElementTypeFloat32;
            ttype.layout.rank = (unsigned)output_shapes_[i].size();
            ttype.layout.has_strides = false;
            memset(ttype.layout.dimensions, 0, sizeof(ttype.layout.dimensions));
            for (size_t d = 0; d < output_shapes_[i].size(); ++d) {
                ttype.layout.dimensions[d] = output_shapes_[i][d];
            }

            /* Create tensor buffer from host memory */
            if (LiteRtCreateTensorBufferFromHostMemory(
                    &ttype, host_mem, buffer_size,
#if defined(_WIN32)
                    [](void *p) { _aligned_free(p); },
#else
                    [](void *p) { free(p); },
#endif
                    &bufs[i]) != kLiteRtStatusOk) {
                LOGE("LiteRT: failed to create output buffer %zu", i);
                cleanup_bufs(bufs, i);
#if defined(_WIN32)
                _aligned_free(host_mem);
#else
                free(host_mem);
#endif
                return false;
            }
        }
        return true;
    };

    auto destroy_buffers = [](const char *label, std::vector<LiteRtTensorBuffer> &bufs) {
        for (auto &b : bufs) {
            if (b) {
                LiteRtDestroyTensorBuffer(b);
            }
        }
        bufs.clear();
    };

    /* Warmup */
    for (int w = 0; w < warmup; ++w) {
        std::vector<LiteRtTensorBuffer> in_bufs, out_bufs;
        if (!create_input_buffers(in_bufs)) {
            return false;
        }
        if (!create_output_buffers(out_bufs)) {
            destroy_buffers("warmup_out_err", in_bufs);
            return false;
        }
        LiteRtStatus warmup_st = LiteRtRunCompiledModel(compiled_, 0, num_inputs_, in_bufs.data(),
                                                        num_outputs_, out_bufs.data());
        if (warmup_st != kLiteRtStatusOk) {
            LOGE("LiteRT: warmup %d run failed: %s", w, LiteRtStatusStr(warmup_st));
            destroy_buffers("warmup_err", in_bufs);
            destroy_buffers("warmup_err", out_bufs);
            return false;
        }
        destroy_buffers("warmup_in", in_bufs);
        destroy_buffers("warmup_out", out_bufs);
    }

    /* Benchmark repeats */
    for (int r = 0; r < repeat; ++r) {
        std::vector<LiteRtTensorBuffer> in_bufs, out_bufs;
        if (!create_input_buffers(in_bufs)) {
            return false;
        }
        if (!create_output_buffers(out_bufs)) {
            destroy_buffers("repeat_buf_err", in_bufs);
            return false;
        }

        auto t0 = std::chrono::high_resolution_clock::now();
        LiteRtStatus st = LiteRtRunCompiledModel(compiled_, 0, num_inputs_, in_bufs.data(),
                                                 num_outputs_, out_bufs.data());
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        if (st != kLiteRtStatusOk) {
            LOGE("LiteRT: run failed at repeat %d", r);
            destroy_buffers("repeat_in_err", in_bufs);
            destroy_buffers("repeat_out_err", out_bufs);
            return false;
        }

        /* Copy outputs to snapshots */
        for (size_t i = 0; i < num_outputs_; ++i) {
            if (out_bufs[i]) {
                void *host_ptr = nullptr;
                LiteRtLockTensorBuffer(out_bufs[i], &host_ptr,
                                       kLiteRtTensorBufferLockModeReadWrite);
                if (host_ptr) {
                    memcpy(snaps[i].data(), host_ptr,
                           std::min(snaps[i].size() * sizeof(float),
                                    output_elems_[i] * sizeof(float)));
                }
                LiteRtUnlockTensorBuffer(out_bufs[i]);
            }
        }

        destroy_buffers("repeat_in", in_bufs);
        destroy_buffers("repeat_out", out_bufs);
        total += ms;
        if (ms > maxv) {
            maxv = ms;
            maxi = r;
        }
        if (ms < minv) {
            minv = ms;
        }
    }

    /* Allocate output data */
    odata.resize(num_outputs_);
    oelems.resize(num_outputs_);
    oshapes.resize(num_outputs_);
    odims.resize(num_outputs_);
    for (size_t i = 0; i < num_outputs_; ++i) {
        size_t n = output_elems_[i];
        float *buf = (float *)malloc(n * sizeof(float));
        if (!buf) {
            LOGE("LiteRT: malloc(%zu) failed at output %zu", n * sizeof(float), i);
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

/* ---------------------------------------------------------------------------
 * GetTiming / SaveOutputs
 * -------------------------------------------------------------------------*/
void LiteRTBackend::GetTiming(std::array<double, 10> &timing)
{
    timing.fill(0);
    timing[0] = init_ms_;
}

bool LiteRTBackend::SaveOutputs(const char * /*suffix*/) { return true; }

/* ---------------------------------------------------------------------------
 * Cleanup
 * -------------------------------------------------------------------------*/
void LiteRTBackend::Cleanup()
{
    if (compiled_) {
        LiteRtDestroyCompiledModel(compiled_);
        compiled_ = nullptr;
    }
    if (comp_opts_) {
        LiteRtDestroyOptions(comp_opts_);
        comp_opts_ = nullptr;
    }
    if (model_) {
        LiteRtDestroyModel(model_);
        model_ = nullptr;
    }
    if (env_) {
        LiteRtDestroyEnvironment(env_);
        env_ = nullptr;
    }

    for (size_t i = 0; i < input_bufs_.size(); ++i) {
        if (input_bufs_[i] && !input_external_[i]) {
            free(input_bufs_[i]);
        }
    }
    input_bufs_.clear();
}

/* ---------------------------------------------------------------------------
 * Factory
 * -------------------------------------------------------------------------*/
BackendPtr CreateLitertBackend(BackendId id)
{
    return std::make_unique<LiteRTBackend>(id);
}

#endif /* HAVE_LITERT_BACKEND */
