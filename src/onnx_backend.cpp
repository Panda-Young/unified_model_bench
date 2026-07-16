/*============================================================================
 * onnx_backend.cpp - ONNX Runtime backend (RAII, fixed API loading)
 *============================================================================*/

#include "backend_interface.hpp"
#include "file_ops.hpp"
#include "log.hpp"

#ifdef HAVE_ONNX_BACKEND

#include <chrono>
#include <cstring>
#include <onnxruntime_c_api.h>
#include <onnxruntime_session_options_config_keys.h>
#include <string>
#include <vector>

#if defined(__ANDROID__) || defined(__android__)
#include <nnapi_provider_factory.h>
#endif

#ifdef _WIN32
/* DML V2 API types (OrtDmlApi, OrtDmlDeviceOptions, etc.) */
#define ENABLE_NPU_ADAPTER_ENUMERATION /* enable OrtDmlDeviceFilter::Npu */
#include <dml_provider_factory.h>
#endif

/* DML / oneDNN / OpenVINO function pointer types (avoid including heavy provider headers) */
typedef OrtStatus *(ORT_API_CALL *PFN_OrtSessionOptionsAppendExecutionProvider_DML)(OrtSessionOptions *options, int device_id);
typedef OrtStatus *(ORT_API_CALL *PFN_OrtSessionOptionsAppendExecutionProvider_Dnnl)(OrtSessionOptions *options, int use_arena);
typedef OrtStatus *(ORT_API_CALL *PFN_OrtSessionOptionsAppendExecutionProvider_OpenVINO)(OrtSessionOptions *options, const char *device_type);

class ONNXBackend : public IBackend
{
public:
    explicit ONNXBackend(BackendId id) { id_ = id; }
    ~ONNXBackend() override { Cleanup(); }

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
    bool ConfigureEP();
    bool QueryIOMetadata();

    void *lib_handle_ = nullptr;
    const OrtApi *ort_ = nullptr;
    OrtEnv *env_ = nullptr;
    OrtSessionOptions *opts_ = nullptr;
    OrtSession *session_ = nullptr;
    OrtAllocator *alloc_ = nullptr;
    OrtMemoryInfo *mem_info_ = nullptr;

    size_t num_inputs_ = 0;
    size_t num_outputs_ = 0;
    std::vector<char *> input_names_;
    std::vector<char *> output_names_;
    std::vector<size_t> input_elems_;
    std::vector<size_t> output_elems_;
    std::vector<std::vector<int64_t>> input_shapes_;
    std::vector<std::vector<int64_t>> output_shapes_;

    std::vector<float *> input_bufs_;
    std::vector<size_t> input_buf_elems_;
    std::vector<bool> input_external_;

    double init_ms_ = 0;
    double timing_[10] = {};
};

/* ---------------------------------------------------------------------------
 * Configure EP
 * -------------------------------------------------------------------------*/
bool ONNXBackend::ConfigureEP()
{
    switch (id_) {
    case BackendId::ONNX_CPU:
        LOGI("ONNX: using default CPU EP");
        return true;
    case BackendId::ONNX_ONEDNN: {
        auto pfn = (PFN_OrtSessionOptionsAppendExecutionProvider_Dnnl)
            load_function(lib_handle_, "OrtSessionOptionsAppendExecutionProvider_Dnnl");
        if (!pfn) {
            LOGE("ONNX: oneDNN EP function not found in DLL");
            return false;
        }
        OrtStatus *st = pfn(opts_, 1);
        if (st) {
            LOGE("ONNX: oneDNN append failed: %s", ort_->GetErrorMessage(st));
            ort_->ReleaseStatus(st);
            return false;
        }
        LOGI("ONNX: oneDNN EP configured");
        return true;
    }
    case BackendId::ONNX_DML_GPU: {
#if defined(_WIN32)
        /* Try V2 API with HighPerformance preference (auto-select best GPU) */
        const OrtDmlApi *dml_api = nullptr;
        OrtStatus *st = ort_->GetExecutionProviderApi("DML", ORT_API_VERSION, (const void **)&dml_api);
        if (!st && dml_api && dml_api->SessionOptionsAppendExecutionProvider_DML2) {
            /* V2 API available — use HighPerformance + Gpu filter */
            OrtDmlDeviceOptions device_opts;
            device_opts.Preference = OrtDmlPerformancePreference::HighPerformance;
            device_opts.Filter = OrtDmlDeviceFilter::Gpu;
            st = dml_api->SessionOptionsAppendExecutionProvider_DML2(opts_, &device_opts);
            if (st) {
                LOGE("ONNX: DML GPU V2 append failed: %s", ort_->GetErrorMessage(st));
                ort_->ReleaseStatus(st);
                return false;
            }
            LOGI("ONNX: DML GPU EP configured (V2, HighPerformance)");
        } else {
            ort_->ReleaseStatus(st);
            /* Fallback to V1 API via function pointer */
            auto pfnDML = (PFN_OrtSessionOptionsAppendExecutionProvider_DML)
                load_function(lib_handle_, "OrtSessionOptionsAppendExecutionProvider_DML");
            if (!pfnDML) {
                LOGE("ONNX: DML EP function not found");
                return false;
            }
            /* First try device_id=1 (dGPU), fallback to device_id=0 (iGPU) */
            st = pfnDML(opts_, 1);
            if (st) {
                ort_->ReleaseStatus(st);
                st = pfnDML(opts_, 0);
                if (st) {
                    LOGE("ONNX: DML V1 device=0 also failed: %s", ort_->GetErrorMessage(st));
                    ort_->ReleaseStatus(st);
                    return false;
                }
                LOGI("ONNX: DML GPU EP configured (V1, device=0 fallback)");
            } else {
                LOGI("ONNX: DML GPU EP configured (V1, device=1)");
            }
        }
        return true;
#else
        LOGE("ONNX: DML GPU EP only available on Windows");
        return false;
#endif
    }
    case BackendId::ONNX_DML_NPU: {
#if defined(_WIN32)
        /* Must use V2 API with NPU filter (V1 has no NPU support) */
        const OrtDmlApi *dml_api = nullptr;
        OrtStatus *st = ort_->GetExecutionProviderApi("DML", ORT_API_VERSION, (const void **)&dml_api);
        if (st || !dml_api || !dml_api->SessionOptionsAppendExecutionProvider_DML2) {
            ort_->ReleaseStatus(st);
            LOGE("ONNX: DML_NPU requires V2 API, not available");
            return false;
        }
        OrtDmlDeviceOptions device_opts;
        device_opts.Preference = OrtDmlPerformancePreference::HighPerformance;
        device_opts.Filter = OrtDmlDeviceFilter::Npu;
        st = dml_api->SessionOptionsAppendExecutionProvider_DML2(opts_, &device_opts);
        if (st) {
            LOGE("ONNX: DML_NPU append failed: %s", ort_->GetErrorMessage(st));
            ort_->ReleaseStatus(st);
            return false;
        }
        LOGI("ONNX: DML_NPU EP configured (V2)");
        return true;
#else
        LOGE("ONNX: DML NPU EP only available on Windows");
        return false;
#endif
    }
    case BackendId::ONNX_OPENVINO_CPU: {
        /* Use V2 API via OrtApi struct with cache_dir to suppress warning and speed up reload */
        if (ort_->SessionOptionsAppendExecutionProvider_OpenVINO_V2) {
            const char *keys[] = {"device_type", "cache_dir"};
            const char *values[] = {"CPU", "model_cache"};
            OrtStatus *st = ort_->SessionOptionsAppendExecutionProvider_OpenVINO_V2(opts_, keys, values, 2);
            if (st) {
                LOGE("ONNX: OpenVINO CPU append failed: %s", ort_->GetErrorMessage(st));
                ort_->ReleaseStatus(st);
                return false;
            }
            LOGI("ONNX: OpenVINO CPU EP configured (V2, cache)");
        } else {
            /* Fallback to V1 API */
            auto pfn = (PFN_OrtSessionOptionsAppendExecutionProvider_OpenVINO)
                load_function(lib_handle_, "OrtSessionOptionsAppendExecutionProvider_OpenVINO");
            if (!pfn) {
                LOGE("ONNX: OpenVINO EP function not found in DLL");
                return false;
            }
            OrtStatus *st = pfn(opts_, "CPU");
            if (st) {
                LOGE("ONNX: OpenVINO CPU append failed: %s", ort_->GetErrorMessage(st));
                ort_->ReleaseStatus(st);
                return false;
            }
            LOGI("ONNX: OpenVINO CPU EP configured (V1)");
        }
        return true;
    }
    case BackendId::ONNX_OPENVINO_GPU: {
        /* Use V2 API via OrtApi struct to separate device_type and precision (GPU_FP32 deprecated) */
        if (ort_->SessionOptionsAppendExecutionProvider_OpenVINO_V2) {
            const char *keys[] = {"device_type", "precision", "cache_dir"};
            const char *values[] = {"GPU", "FP32", "model_cache"};
            OrtStatus *st = ort_->SessionOptionsAppendExecutionProvider_OpenVINO_V2(opts_, keys, values, 3);
            if (st) {
                LOGE("ONNX: OpenVINO GPU append failed: %s", ort_->GetErrorMessage(st));
                ort_->ReleaseStatus(st);
                return false;
            }
            LOGI("ONNX: OpenVINO GPU EP configured (V2, FP32, cache)");
        } else {
            /* Fallback to V1 API (loaded via dlsym for older runtimes without V2 in OrtApi) */
            auto pfnV1 = (PFN_OrtSessionOptionsAppendExecutionProvider_OpenVINO)
                load_function(lib_handle_, "OrtSessionOptionsAppendExecutionProvider_OpenVINO");
            if (!pfnV1) {
                LOGE("ONNX: OpenVINO EP function not found in DLL");
                return false;
            }
            OrtStatus *st = pfnV1(opts_, "GPU_FP32");
            if (st) {
                LOGE("ONNX: OpenVINO GPU append failed: %s", ort_->GetErrorMessage(st));
                ort_->ReleaseStatus(st);
                return false;
            }
            LOGW("ONNX: OpenVINO GPU EP configured (V1 fallback, GPU_FP32 deprecated)");
        }
        return true;
    }
    case BackendId::ONNX_OPENVINO_NPU: {
        /* Use V2 API via OrtApi struct with FP16 precision (NPU hardware optimized for FP16) */
        if (ort_->SessionOptionsAppendExecutionProvider_OpenVINO_V2) {
            const char *keys[] = {"device_type", "precision", "cache_dir"};
            const char *values[] = {"NPU", "FP16", "model_cache"};
            OrtStatus *st = ort_->SessionOptionsAppendExecutionProvider_OpenVINO_V2(opts_, keys, values, 3);
            if (st) {
                LOGE("ONNX: OpenVINO NPU append failed: %s", ort_->GetErrorMessage(st));
                ort_->ReleaseStatus(st);
                return false;
            }
            LOGI("ONNX: OpenVINO NPU EP configured (V2, FP16, cache)");
        } else {
            /* Fallback to V1 API with basic NPU device type */
            auto pfnV1 = (PFN_OrtSessionOptionsAppendExecutionProvider_OpenVINO)
                load_function(lib_handle_, "OrtSessionOptionsAppendExecutionProvider_OpenVINO");
            if (!pfnV1) {
                LOGE("ONNX: OpenVINO EP function not found in DLL");
                return false;
            }
            OrtStatus *st = pfnV1(opts_, "NPU");
            if (st) {
                LOGE("ONNX: OpenVINO NPU append failed: %s", ort_->GetErrorMessage(st));
                ort_->ReleaseStatus(st);
                return false;
            }
            LOGI("ONNX: OpenVINO NPU EP configured (V1 fallback)");
        }
        return true;
    }
    case BackendId::ONNX_NNAPI:
#if defined(__ANDROID__) || defined(__android__)
    {
        OrtStatus *st = OrtSessionOptionsAppendExecutionProvider_Nnapi(opts_, 0);
        if (st) {
            LOGW("ONNX: NNAPI failed");
            ort_->ReleaseStatus(st);
            return false;
        }
        return true;
    }
#else
        LOGE("ONNX: NNAPI only on Android");
        return false;
#endif
    case BackendId::ONNX_XNNPACK: {
#if defined(__ANDROID__) || defined(__android__)
        /* XNNPACK uses the generic SessionOptionsAppendExecutionProvider API,
         * NOT a dedicated provider-specific function pointer.
         * Reference: onnx_test/src/session_manager.c EP_XNNPACK branch */
        (void)ort_->AddSessionConfigEntry(opts_, kOrtSessionOptionsConfigAllowIntraOpSpinning, "0");
        (void)ort_->SetIntraOpNumThreads(opts_, 1);
        const char *keys[] = {"intra_op_num_threads"};
        const char *values[] = {"4"};
        OrtStatus *st = ort_->SessionOptionsAppendExecutionProvider(opts_, "XNNPACK",
                                                                    keys, values, 1);
        if (st) {
            LOGE("ONNX: XNNPACK append failed: %s", ort_->GetErrorMessage(st));
            ort_->ReleaseStatus(st);
            return false;
        }
        LOGI("ONNX: XNNPACK EP configured");
        return true;
#else
        LOGE("ONNX: XNNPACK EP not available in this build");
        return false;
#endif
    }
    case BackendId::ONNX_QNN_CPU:
    case BackendId::ONNX_QNN_GPU:
    case BackendId::ONNX_QNN_HTP: {
#if defined(__ANDROID__) || defined(__android__)
        OrtStatus *st;
        const char *backend_type;

        if (id_ == BackendId::ONNX_QNN_HTP) {
            /* QNN EP with HTP backend — full tuning for best NPU acceleration */
            backend_type = "htp";
            const char *htp_keys[] = {
                "backend_type",
                "soc_model",
                "htp_arch",
                "profiling_level",
                "profiling_file_path",
                "htp_performance_mode",
                "htp_graph_finalization_optimization_mode",
                "enable_htp_fp16_precision",
                "enable_htp_shared_memory_allocator",
            };
            const char *htp_values[] = {
                "htp", // backend_type: HTP NPU backend
                "0",   // soc_model: 0=auto
                "0",   // htp_arch: 0=auto
                "off", // profiling_level: off basic detailed
                "/data/local/tmp/qnn_htp_profiling.csv",
                "burst", // htp_performance_mode: burst balanced default high_performance ...
                "3",     // htp_graph_finalization_optimization_mode: 0 1 2 3
                "1",     // enable_htp_fp16_precision: 0 1
                "1",     // enable_htp_shared_memory_allocator: 0 1
            };
            int num_opts = sizeof(htp_keys) / sizeof(htp_keys[0]);
            st = ort_->SessionOptionsAppendExecutionProvider(
                opts_, "QNN", htp_keys, htp_values, num_opts);
        } else if (id_ == BackendId::ONNX_QNN_GPU) {
            /* QNN EP with Adreno GPU backend — FP32/FP16, no HTP-specific options */
            backend_type = "gpu";
            const char *gpu_keys[] = {
                "backend_type",
                "profiling_level",
                "profiling_file_path",
            };
            const char *gpu_values[] = {
                "gpu",
                "off",
                "/data/local/tmp/qnn_gpu_profiling.csv",
            };
            int num_opts = sizeof(gpu_keys) / sizeof(gpu_keys[0]);
            st = ort_->SessionOptionsAppendExecutionProvider(
                opts_, "QNN", gpu_keys, gpu_values, num_opts);
        } else {
            /* QNN EP with CPU backend — reference backend for graph validation */
            backend_type = "cpu";
            const char *cpu_keys[] = {
                "backend_type",
                "profiling_level",
                "profiling_file_path",
            };
            const char *cpu_values[] = {
                "cpu",
                "off",
                "/data/local/tmp/qnn_cpu_profiling.csv",
            };
            int num_opts = sizeof(cpu_keys) / sizeof(cpu_keys[0]);
            st = ort_->SessionOptionsAppendExecutionProvider(
                opts_, "QNN", cpu_keys, cpu_values, num_opts);
        }

        if (st) {
            LOGE("ONNX: QNN(%s) append failed: %s", backend_type, ort_->GetErrorMessage(st));
            ort_->ReleaseStatus(st);
            return false;
        }
        LOGI("ONNX: QNN(%s) EP configured", backend_type);
        return true;
#else
        LOGE("ONNX: QNN EP not available in this build");
        return false;
#endif
    }
    default:
        return true; /* fallback to CPU */
    }
}

/* ---------------------------------------------------------------------------
 * Initialize
 * -------------------------------------------------------------------------*/
bool ONNXBackend::Initialize(const char *model_path, int num_threads)
{
    auto t0 = std::chrono::high_resolution_clock::now();
    std::string dll_name = "onnxruntime.dll";

    /* EP-specific DLL paths relative to deps/ */
    const char *ep_subdir = nullptr;
    (void)ep_subdir; /* only used on desktop */
    switch (id_) {
    case BackendId::ONNX_CPU: /* use default CPU DLL path below */
        break;
    case BackendId::ONNX_DML_GPU:
    case BackendId::ONNX_DML_NPU:
        ep_subdir = "dml";
        break;
    case BackendId::ONNX_ONEDNN:
        ep_subdir = "onednn";
        break;
    case BackendId::ONNX_OPENVINO_CPU:
    case BackendId::ONNX_OPENVINO_GPU:
        ep_subdir = "openvino";
        break;
    case BackendId::ONNX_OPENVINO_NPU:
        ep_subdir = "openvino";
        break;
    default:
        break;
    }

    lib_handle_ = nullptr;

#if defined(__ANDROID__) || defined(__android__)
    /* Android: libonnxruntime.so is in LD_LIBRARY_PATH. QNN EP uses separate .so. */
    if (id_ == BackendId::ONNX_QNN_CPU || id_ == BackendId::ONNX_QNN_GPU ||
        id_ == BackendId::ONNX_QNN_HTP) {
        lib_handle_ = load_library("qnn/libonnxruntime.so");
    }
    if (!lib_handle_) {
        lib_handle_ = load_library("libonnxruntime.so");
    }
#else
    /* Desktop: deps/onnxruntime/lib/<arch>/<ep>/onnxruntime.dll */
    const char *arch_dir =
#if defined(_WIN64)
        "win-x64";
#elif defined(_WIN32)
        "win-x86";
#else
        "linux-x64";
#endif

    if (ep_subdir) {
        char ep_path[MAX_PATH_LEN];
        snprintf(ep_path, sizeof(ep_path), "deps/onnxruntime/lib/%s/%s/onnxruntime.dll",
                 arch_dir, ep_subdir);
        lib_handle_ = load_library(ep_path);
        if (!lib_handle_) {
            LOGW("ONNX: %s not found, trying default CPU DLL", ep_path);
        }
    }
    if (!lib_handle_) {
        /* Default CPU DLL */
        char cpu_path[MAX_PATH_LEN];
        snprintf(cpu_path, sizeof(cpu_path), "deps/onnxruntime/lib/%s/cpu/onnxruntime.dll",
                 arch_dir);
        lib_handle_ = load_library(cpu_path);
    }
    if (!lib_handle_) {
        /* Last resort: try current directory */
        lib_handle_ = load_library(dll_name.c_str());
    }
#endif
    if (!lib_handle_) {
        LOGE("ONNX: failed to load onnxruntime library");
        return false;
    }
    timing_[1] = std::chrono::duration<double, std::milli>(
                     std::chrono::high_resolution_clock::now() - t0)
                     .count();

    auto t1 = std::chrono::high_resolution_clock::now();
    auto GetApiBase = (const OrtApiBase *(*)())load_function(lib_handle_, "OrtGetApiBase");
    if (!GetApiBase) {
        LOGE("ONNX: OrtGetApiBase not found");
        return false;
    }
    const OrtApiBase *base = GetApiBase();
    if (!base) {
        LOGE("ONNX: OrtGetApiBase returned NULL");
        return false;
    }
    ort_ = base->GetApi(ORT_API_VERSION);
    if (!ort_) {
        LOGE("ONNX: GetApi(v=%u) returned NULL", ORT_API_VERSION);
        return false;
    }
    timing_[2] = std::chrono::duration<double, std::milli>(
                     std::chrono::high_resolution_clock::now() - t1)
                     .count();

    auto t2 = std::chrono::high_resolution_clock::now();
    OrtStatus *st = ort_->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "unified_bench", &env_);
    if (st) {
        LOGE("ONNX: CreateEnv failed");
        ort_->ReleaseStatus(st);
        return false;
    }
    timing_[3] = std::chrono::duration<double, std::milli>(
                     std::chrono::high_resolution_clock::now() - t2)
                     .count();

    auto t3 = std::chrono::high_resolution_clock::now();
    st = ort_->CreateSessionOptions(&opts_);
    if (st) {
        LOGE("ONNX: CreateSessionOptions failed");
        ort_->ReleaseStatus(st);
        return false;
    }
    (void)ort_->SetIntraOpNumThreads(opts_, num_threads);
    (void)ort_->SetSessionGraphOptimizationLevel(opts_, ORT_ENABLE_ALL);
    timing_[4] = std::chrono::duration<double, std::milli>(
                     std::chrono::high_resolution_clock::now() - t3)
                     .count();

    auto t4 = std::chrono::high_resolution_clock::now();
    if (!ConfigureEP()) {
        LOGE("ONNX: ConfigureEP failed for backend id=%d", bid(id_));
        return false;
    }
    timing_[5] = std::chrono::duration<double, std::milli>(
                     std::chrono::high_resolution_clock::now() - t4)
                     .count();

    auto t5 = std::chrono::high_resolution_clock::now();
#ifdef _WIN32
    std::wstring wpath = utf8_to_wide(model_path);
    st = ort_->CreateSession(env_, wpath.c_str(), opts_, &session_);
#else
    st = ort_->CreateSession(env_, model_path, opts_, &session_);
#endif
    if (st) {
        LOGE("ONNX: CreateSession failed: %s", ort_->GetErrorMessage(st));
        ort_->ReleaseStatus(st);
        return false;
    }
    timing_[6] = std::chrono::duration<double, std::milli>(
                     std::chrono::high_resolution_clock::now() - t5)
                     .count();

    (void)ort_->GetAllocatorWithDefaultOptions(&alloc_);

    auto t6 = std::chrono::high_resolution_clock::now();
    if (!QueryIOMetadata()) {
        return false;
    }
    timing_[7] = std::chrono::duration<double, std::milli>(
                     std::chrono::high_resolution_clock::now() - t6)
                     .count();

    init_ms_ = std::chrono::duration<double, std::milli>(
                   std::chrono::high_resolution_clock::now() - t0)
                   .count();
    LOGI("ONNX: init complete (%.1f ms), %zu in, %zu out", init_ms_, num_inputs_, num_outputs_);
    return true;
}

/* ---------------------------------------------------------------------------
 * Query IO metadata
 * -------------------------------------------------------------------------*/
bool ONNXBackend::QueryIOMetadata()
{
    OrtStatus *st;
    size_t n_in, n_out;
    st = ort_->SessionGetInputCount(session_, &n_in);
    if (st) {
        ort_->ReleaseStatus(st);
        return false;
    }
    st = ort_->SessionGetOutputCount(session_, &n_out);
    if (st) {
        ort_->ReleaseStatus(st);
        return false;
    }
    num_inputs_ = std::min(n_in, MAX_IO);
    num_outputs_ = std::min(n_out, MAX_IO);

    for (size_t i = 0; i < num_inputs_; ++i) {
        char *name = nullptr;
        (void)ort_->SessionGetInputName(session_, i, alloc_, &name);
        input_names_.push_back(name);
        OrtTypeInfo *ti;
        (void)ort_->SessionGetInputTypeInfo(session_, i, &ti);
        const OrtTensorTypeAndShapeInfo *si;
        (void)ort_->CastTypeInfoToTensorInfo(ti, &si);
        size_t nd;
        (void)ort_->GetDimensionsCount(si, &nd);
        std::vector<int64_t> dims(nd);
        (void)ort_->GetDimensions(si, dims.data(), nd);
        size_t elems = 1;
        for (auto d : dims) {
            if (d > 0) {
                elems *= (size_t)d;
            }
        }
        input_elems_.push_back(elems);
        input_shapes_.push_back(std::move(dims));
        ort_->ReleaseTypeInfo(ti);
    }
    for (size_t i = 0; i < num_outputs_; ++i) {
        char *name = nullptr;
        (void)ort_->SessionGetOutputName(session_, i, alloc_, &name);
        output_names_.push_back(name);
        OrtTypeInfo *ti;
        (void)ort_->SessionGetOutputTypeInfo(session_, i, &ti);
        const OrtTensorTypeAndShapeInfo *si;
        (void)ort_->CastTypeInfoToTensorInfo(ti, &si);
        size_t nd;
        (void)ort_->GetDimensionsCount(si, &nd);
        std::vector<int64_t> dims(nd);
        (void)ort_->GetDimensions(si, dims.data(), nd);
        size_t elems = 1;
        for (auto d : dims) {
            if (d > 0) {
                elems *= (size_t)d;
            }
        }
        output_elems_.push_back(elems);
        output_shapes_.push_back(std::move(dims));
        ort_->ReleaseTypeInfo(ti);
    }
    return true;
}

/* ---------------------------------------------------------------------------
 * QueryIOInfo
 * -------------------------------------------------------------------------*/
bool ONNXBackend::QueryIOInfo(std::string &is, size_t &ie, std::string &os, size_t &oe)
{
    is.clear();
    ie = 0;
    for (size_t i = 0; i < num_inputs_; ++i) {
        char b[128] = {};
        int o = snprintf(b, sizeof(b), "[");
        for (size_t d = 0; d < input_shapes_[i].size(); ++d) {
            o += snprintf(b + o, sizeof(b) - o, "%s%lld", d > 0 ? "," : "", (long long)input_shapes_[i][d]);
        }
        snprintf(b + o, sizeof(b) - o, "]");
        if (i > 0) {
            is += ";";
        }
        is += b;
        ie += input_elems_[i];
    }
    os.clear();
    oe = 0;
    for (size_t i = 0; i < num_outputs_; ++i) {
        char b[128] = {};
        int o = snprintf(b, sizeof(b), "[");
        for (size_t d = 0; d < output_shapes_[i].size(); ++d) {
            o += snprintf(b + o, sizeof(b) - o, "%s%lld", d > 0 ? "," : "", (long long)output_shapes_[i][d]);
        }
        snprintf(b + o, sizeof(b) - o, "]");
        if (i > 0) {
            os += ";";
        }
        os += b;
        oe += output_elems_[i];
    }
    return true;
}

/* ---------------------------------------------------------------------------
 * PrepareInputs / SetSharedInput
 * -------------------------------------------------------------------------*/
bool ONNXBackend::PrepareInputs(float *&fd, size_t &fe, const char *, bool random, const float *const *ext, const size_t *extc)
{
    for (size_t i = 0; i < num_inputs_; ++i) {
        size_t n = input_elems_[i] ? input_elems_[i] : 1;
        if (i >= input_bufs_.size()) {
            input_bufs_.resize(i + 1, nullptr);
            input_buf_elems_.resize(i + 1, 0);
            input_external_.resize(i + 1, false);
        }
        if (input_bufs_[i] && !input_external_[i]) {
            free(input_bufs_[i]);
        }
        if (ext && ext[i] && extc && extc[i] == n) {
            input_bufs_[i] = (float *)ext[i];
            input_external_[i] = true;
        } else {
            float *b = (float *)malloc(n * sizeof(float));
            if (!b) {
                LOGE("ONNX: malloc(%zu) failed at input %zu, due to %s, %d", n * sizeof(float), i, strerror(errno), errno);
                return false;
            }
            if (random) {
                for (size_t j = 0; j < n; ++j) {
                    b[j] = (float)rand() / (float)RAND_MAX;
                }
            } else {
                memset(b, 0, n * sizeof(float));
            }
            input_bufs_[i] = b;
            input_external_[i] = false;
        }
        input_buf_elems_[i] = n;
    }
    fd = num_inputs_ > 0 ? input_bufs_[0] : nullptr;
    fe = num_inputs_ > 0 ? input_buf_elems_[0] : 0;
    return true;
}
void ONNXBackend::SetSharedInput(const float *const *data, const size_t *counts)
{
    for (size_t i = 0; i < num_inputs_; ++i) {
        if (i >= input_bufs_.size()) {
            input_bufs_.resize(i + 1, nullptr);
            input_buf_elems_.resize(i + 1, 0);
            input_external_.resize(i + 1, false);
        }
        input_bufs_[i] = (float *)data[i];
        input_buf_elems_[i] = counts[i];
        input_external_[i] = true;
    }
}

/* ---------------------------------------------------------------------------
 * RunBenchmark
 * -------------------------------------------------------------------------*/
bool ONNXBackend::RunBenchmark(int warmup, int repeat, double &total, double &maxv, double &minv, int &maxi,
                               std::vector<float *> &odata, std::vector<size_t> &oelems,
                               std::vector<std::array<size_t, MAX_DIMENSIONS>> &oshapes, std::vector<size_t> &odims)
{
    total = 0;
    maxv = 0;
    minv = 1e12;
    maxi = 0;
    if (num_inputs_ == 0 || num_outputs_ == 0) {
        return false;
    }

    OrtStatus *st = ort_->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mem_info_);
    if (st) {
        ort_->ReleaseStatus(st);
        return false;
    }

    std::vector<std::vector<float>> snaps(num_outputs_);
    for (size_t i = 0; i < num_outputs_; ++i) {
        snaps[i].resize(output_elems_[i] ? output_elems_[i] : 1);
    }

    auto create_inputs = [&]() -> std::vector<OrtValue *> {
        std::vector<OrtValue *> vals(num_inputs_, nullptr);
        for (size_t i = 0; i < num_inputs_; ++i) {
            size_t n = input_elems_[i] ? input_elems_[i] : 1;
            st = ort_->CreateTensorWithDataAsOrtValue(mem_info_, input_bufs_[i], n * sizeof(float), input_shapes_[i].data(), input_shapes_[i].size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &vals[i]);
            if (st) {
                ort_->ReleaseStatus(st);
                for (size_t j = 0; j < i; ++j) {
                    ort_->ReleaseValue(vals[j]);
                }
                return {};
            }
        }
        return vals;
    };

    (void)warmup; /* total_runs not logged at INFO level */
    for (int w = 0; w < warmup; ++w) {
        auto in = create_inputs();
        if (in.empty()) {
            return false;
        }
        std::vector<OrtValue *> out(num_outputs_, nullptr);
        (void)ort_->Run(session_, nullptr, (const char *const *)input_names_.data(), in.data(), num_inputs_, (const char *const *)output_names_.data(), num_outputs_, out.data());
        for (auto &v : in) {
            ort_->ReleaseValue(v);
        }
        for (auto &v : out) {
            if (v) {
                ort_->ReleaseValue(v);
            }
        }
    }

    for (int r = 0; r < repeat; ++r) {
        auto in = create_inputs();
        if (in.empty()) {
            return false;
        }
        auto t0 = std::chrono::high_resolution_clock::now();
        std::vector<OrtValue *> out(num_outputs_, nullptr);
        st = ort_->Run(session_, nullptr, (const char *const *)input_names_.data(), in.data(), num_inputs_, (const char *const *)output_names_.data(), num_outputs_, out.data());
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (st) {
            LOGE("ONNX:Run failed at %d", r);
            ort_->ReleaseStatus(st);
            for (auto &v : in) {
                ort_->ReleaseValue(v);
            }
            for (auto &v : out) {
                if (v) {
                    ort_->ReleaseValue(v);
                }
            }
            return false;
        }
        for (size_t i = 0; i < num_outputs_; ++i) {
            if (out[i]) {
                float *fp = nullptr;
                (void)ort_->GetTensorMutableData(out[i], (void **)&fp);
                if (fp) {
                    memcpy(snaps[i].data(), fp, output_elems_[i] * sizeof(float));
                }
            }
        }
        for (auto &v : in) {
            ort_->ReleaseValue(v);
        }
        for (auto &v : out) {
            if (v) {
                ort_->ReleaseValue(v);
            }
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
        float *b = (float *)malloc(n * sizeof(float));
        if (!b) {
            LOGE("ONNX: malloc(%zu) failed at output %zu, due to %s, %d", n * sizeof(float), i, strerror(errno), errno);
            return false;
        }
        memcpy(b, snaps[i].data(), n * sizeof(float));
        odata[i] = b;
        oelems[i] = n;
        auto &sh = oshapes[i];
        sh.fill(0);
        for (size_t d = 0; d < output_shapes_[i].size() && d < MAX_DIMENSIONS; ++d) {
            sh[d] = (size_t)output_shapes_[i][d];
        }
        odims[i] = output_shapes_[i].size();
    }

    ort_->ReleaseMemoryInfo(mem_info_);
    mem_info_ = nullptr;
    return true;
}

void ONNXBackend::GetTiming(std::array<double, 10> &t)
{
    for (int i = 0; i < 10; ++i) {
        t[i] = timing_[i];
    }
    t[0] = init_ms_;
}
bool ONNXBackend::SaveOutputs(const char *) { return true; }

void ONNXBackend::Cleanup()
{
    if (mem_info_) {
        ort_->ReleaseMemoryInfo(mem_info_);
        mem_info_ = nullptr;
    }
    if (session_) {
        ort_->ReleaseSession(session_);
        session_ = nullptr;
    }
    if (opts_) {
        ort_->ReleaseSessionOptions(opts_);
        opts_ = nullptr;
    }
    if (env_) {
        ort_->ReleaseEnv(env_);
        env_ = nullptr;
    }
    for (auto *n : input_names_) {
        if (alloc_) {
            alloc_->Free(alloc_, n);
        }
    }
    for (auto *n : output_names_) {
        if (alloc_) {
            alloc_->Free(alloc_, n);
        }
    }
    input_names_.clear();
    output_names_.clear();
    for (size_t i = 0; i < input_bufs_.size(); ++i) {
        if (input_bufs_[i] && !input_external_[i]) {
            free(input_bufs_[i]);
        }
    }
    input_bufs_.clear();
}

BackendPtr CreateOnnxBackend(BackendId id) { return std::make_unique<ONNXBackend>(id); }

#endif /* HAVE_ONNX_BACKEND */
