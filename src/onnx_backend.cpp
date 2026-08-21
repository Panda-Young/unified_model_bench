/*============================================================================
 * onnx_backend.cpp - ONNX Runtime backend (RAII, fixed API loading)
 *============================================================================*/

#include "backend_interface.hpp"
#include "file_ops.hpp"
#include "log.hpp"
#include "qnn_soc.hpp"

#ifdef HAVE_ONNX_BACKEND

#include <chrono>
#include <cstring>
#include <onnxruntime_c_api.h>
#include <onnxruntime_session_options_config_keys.h>
#include <string>
#include <vector>

#if defined(__ANDROID__) || defined(__android__)
/* NNAPI symbols loaded dynamically -- avoid link-time dependency */
#endif

#ifdef _WIN32
/* DML V2 API types (OrtDmlApi, OrtDmlDeviceOptions, etc.) */
#define ENABLE_NPU_ADAPTER_ENUMERATION /* enable OrtDmlDeviceFilter::Npu */
#include <dml_provider_factory.h>
#endif

/* DML / oneDNN / OpenVINO / NNAPI function pointer types (avoid including heavy provider headers) */
typedef OrtStatus *(ORT_API_CALL *PFN_OrtSessionOptionsAppendExecutionProvider_DML)(OrtSessionOptions *options, int device_id);
typedef OrtStatus *(ORT_API_CALL *PFN_OrtSessionOptionsAppendExecutionProvider_Dnnl)(OrtSessionOptions *options, int use_arena);
typedef OrtStatus *(ORT_API_CALL *PFN_OrtSessionOptionsAppendExecutionProvider_OpenVINO)(OrtSessionOptions *options, const char *device_type);
typedef OrtStatus *(ORT_API_CALL *PFN_OrtSessionOptionsAppendExecutionProvider_Nnapi)(OrtSessionOptions *options, int use_nnapi);

/* ---------------------------------------------------------------------------
 * ORT status helper: every ORT C API call returns an OrtStatus*; a non-null
 * status means failure. Never ignore it - log the error, record it in the
 * backend error string and return false so the caller can abort cleanly.
 * Returns true when the call succeeded (status == nullptr).
 * -------------------------------------------------------------------------*/
static bool OrOk(const OrtApi *api, OrtStatus *st, const char *what, std::string &err)
{
    if (!st) {
        return true;
    }
    const char *msg = api ? api->GetErrorMessage(st) : "unknown error";
    LOGE("ONNX: %s failed: %s", what, msg);
    err = std::string("ONNX: ") + what + ": " + msg;
    api->ReleaseStatus(st);
    return false;
}

class ONNXBackend : public IBackend
{
public:
    explicit ONNXBackend(BackendId id) { id_ = id; }
    ~ONNXBackend() override { Cleanup(); }

    bool Initialize(const char *model_path, int num_threads) override;
    bool QueryIOInfo(std::string &is, size_t &ie, std::string &os, size_t &oe) override;
    void SetSharedInput(const float *const *data, const size_t *counts) override;
    bool RunBenchmark(int warmup, int repeat, double &total, double &maxv,
                      double &minv, int &maxi, std::vector<float *> &odata,
                      std::vector<size_t> &oelems,
                      std::vector<std::array<size_t, MAX_DIMENSIONS>> &oshapes,
                      std::vector<size_t> &odims) override;
    void GetTiming(std::array<double, 10> &timing) override;
    const std::vector<std::string> &GetOutputNames() const override
    {
        return output_names_str_;
    }

private:
    void Cleanup();
    bool ConfigureEP();
    bool QueryIOMetadata();

    void *lib_handle_ = nullptr;
    const OrtApi *ort_ = nullptr;
    uint32_t ort_api_ver_ = ORT_API_VERSION;
    OrtEnv *env_ = nullptr;
    OrtSessionOptions *opts_ = nullptr;
    OrtSession *session_ = nullptr;
    OrtAllocator *alloc_ = nullptr;
    OrtMemoryInfo *mem_info_ = nullptr;

    size_t num_inputs_ = 0;
    size_t num_outputs_ = 0;
    std::vector<char *> input_names_;
    std::vector<char *> output_names_;
    std::vector<std::string> output_names_str_; /* stable copies for GetOutputNames */
    std::vector<size_t> input_elems_;
    std::vector<size_t> output_elems_;
    std::vector<std::vector<int64_t>> input_shapes_;
    std::vector<std::vector<int64_t>> output_shapes_;

    std::vector<float *> input_bufs_;
    std::vector<size_t> input_buf_elems_;
    std::vector<bool> input_external_;

    double init_ms_ = 0;
    double timing_[10] = {};

    /* EP Context cache for QNN backends */
    std::string ep_context_path_;
    const char *model_path_for_create_ = nullptr;
};

/* ---------------------------------------------------------------------------
 * Configure EP
 * -------------------------------------------------------------------------*/
bool ONNXBackend::ConfigureEP()
{
    switch (id_) {
    case BackendId::ONNX_CPU: {
        LOGI("ONNX: using default CPU EP");
        return true;
    }
    case BackendId::ONNX_ONEDNN: {
        auto pfn = (PFN_OrtSessionOptionsAppendExecutionProvider_Dnnl)
            load_function(lib_handle_, "OrtSessionOptionsAppendExecutionProvider_Dnnl");
        if (!pfn) {
            LOGE("ONNX: oneDNN EP function not found in DLL");
            return false;
        }
        if (!OrOk(ort_, pfn(opts_, 1), "oneDNN append", last_error_)) {
            return false;
        }
        LOGI("ONNX: oneDNN EP configured");
        return true;
    }
    case BackendId::ONNX_DML_GPU: {
#if defined(_WIN32)
        /* Try V2 API with HighPerformance preference (auto-select best GPU) */
        const OrtDmlApi *dml_api = nullptr;
        OrtStatus *st = ort_->GetExecutionProviderApi("DML", ort_api_ver_, (const void **)&dml_api);
        if (!st && dml_api && dml_api->SessionOptionsAppendExecutionProvider_DML2) {
            /* V2 API available -- use HighPerformance + Gpu filter */
            OrtDmlDeviceOptions device_opts;
            device_opts.Preference = OrtDmlPerformancePreference::HighPerformance;
            device_opts.Filter = OrtDmlDeviceFilter::Gpu;
            if (!OrOk(ort_, dml_api->SessionOptionsAppendExecutionProvider_DML2(opts_, &device_opts),
                      "DML GPU V2 append", last_error_)) {
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
            if (!OrOk(ort_, pfnDML(opts_, 1), "DML GPU V1 append(device=1)", last_error_)) {
                if (!OrOk(ort_, pfnDML(opts_, 0), "DML GPU V1 append(device=0)", last_error_)) {
                    return false;
                }
                last_error_.clear(); /* device=0 fallback succeeded; drop the device=1 probe error */
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
        OrtStatus *st = ort_->GetExecutionProviderApi("DML", ort_api_ver_, (const void **)&dml_api);
        if (st || !dml_api || !dml_api->SessionOptionsAppendExecutionProvider_DML2) {
            ort_->ReleaseStatus(st);
            LOGE("ONNX: DML_NPU requires V2 API, not available");
            return false;
        }
        OrtDmlDeviceOptions device_opts;
        device_opts.Preference = OrtDmlPerformancePreference::HighPerformance;
        device_opts.Filter = OrtDmlDeviceFilter::Npu;
        if (!OrOk(ort_, dml_api->SessionOptionsAppendExecutionProvider_DML2(opts_, &device_opts),
                  "DML_NPU append", last_error_)) {
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
            if (!OrOk(ort_, ort_->SessionOptionsAppendExecutionProvider_OpenVINO_V2(opts_, keys, values, 2),
                      "OpenVINO CPU append", last_error_)) {
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
            if (!OrOk(ort_, pfn(opts_, "CPU"), "OpenVINO CPU append", last_error_)) {
                return false;
            }
            LOGI("ONNX: OpenVINO CPU EP configured (V1)");
        }
        return true;
    }
    case BackendId::ONNX_OPENVINO_GPU_FP16: {
        /* OpenVINO GPU FP16 explicit precision */
        if (ort_->SessionOptionsAppendExecutionProvider_OpenVINO_V2) {
            const char *keys[] = {"device_type", "precision", "cache_dir"};
            const char *values[] = {"GPU", "FP16", "model_cache"};
            if (!OrOk(ort_, ort_->SessionOptionsAppendExecutionProvider_OpenVINO_V2(opts_, keys, values, 3),
                      "OpenVINO GPU_FP16 append", last_error_)) {
                return false;
            }
            LOGI("ONNX: OpenVINO GPU_FP16 EP configured (V2, FP16, cache)");
        } else {
            auto pfnV1 = (PFN_OrtSessionOptionsAppendExecutionProvider_OpenVINO)
                load_function(lib_handle_, "OrtSessionOptionsAppendExecutionProvider_OpenVINO");
            if (!pfnV1) {
                LOGE("ONNX: OpenVINO EP function not found in DLL");
                return false;
            }
            if (!OrOk(ort_, pfnV1(opts_, "GPU_FP16"), "OpenVINO GPU_FP16 append", last_error_)) {
                return false;
            }
            LOGW("ONNX: OpenVINO GPU_FP16 EP configured (V1 fallback)");
        }
        return true;
    }
    case BackendId::ONNX_OPENVINO_GPU: {
        /* Use V2 API via OrtApi struct to separate device_type and precision (GPU_FP32 deprecated) */
        if (ort_->SessionOptionsAppendExecutionProvider_OpenVINO_V2) {
            const char *keys[] = {"device_type", "precision", "cache_dir"};
            const char *values[] = {"GPU", "FP32", "model_cache"};
            if (!OrOk(ort_, ort_->SessionOptionsAppendExecutionProvider_OpenVINO_V2(opts_, keys, values, 3),
                      "OpenVINO GPU append", last_error_)) {
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
            if (!OrOk(ort_, pfnV1(opts_, "GPU_FP32"), "OpenVINO GPU append", last_error_)) {
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
            if (!OrOk(ort_, ort_->SessionOptionsAppendExecutionProvider_OpenVINO_V2(opts_, keys, values, 3),
                      "OpenVINO NPU append", last_error_)) {
                return false;
            }
            LOGI("ONNX: OpenVINO NPU EP configured (V2, FP16, cache)");
        } else {
            /* Fallback to V1 API with basic NPU device type */
            auto pfnV1 = (PFN_OrtSessionOptionsAppendExecutionProvider_OpenVINO)
                load_function(lib_handle_, "OrtSessionOptionsAppendExecutionProvider_OpenVINO");
            if (!pfnV1) {
                LOGE("ONNX: OpenVINO EP function not found in DLL");
                last_error_ = "ONNX: OpenVINO EP function not found";
                return false;
            }
            if (!OrOk(ort_, pfnV1(opts_, "NPU"), "OpenVINO NPU append", last_error_)) {
                return false;
            }
            LOGI("ONNX: OpenVINO NPU EP configured (V1 fallback)");
        }
        return true;
    }
    case BackendId::ONNX_NNAPI: {
#if defined(__ANDROID__) || defined(__android__)
        auto pfnNnapi = (PFN_OrtSessionOptionsAppendExecutionProvider_Nnapi)
            load_function(lib_handle_, "OrtSessionOptionsAppendExecutionProvider_Nnapi");
        if (!pfnNnapi) {
            LOGE("ONNX: NNAPI EP function not found in ONNX Runtime library");
            last_error_ = "ONNX: NNAPI EP function not found";
            return false;
        }
        if (!OrOk(ort_, pfnNnapi(opts_, 0), "NNAPI append", last_error_)) {
            return false;
        }
        LOGI("ONNX: NNAPI EP configured");
        return true;
#else
        LOGE("ONNX: NNAPI only on Android");
        return false;
#endif
    }
    case BackendId::ONNX_XNNPACK: {
#if defined(__ANDROID__) || defined(__android__)
        /* XNNPACK uses the generic SessionOptionsAppendExecutionProvider API,
         * NOT a dedicated provider-specific function pointer.
         * Reference: onnx_test/src/session_manager.c EP_XNNPACK branch */
        if (!OrOk(ort_, ort_->AddSessionConfigEntry(opts_, kOrtSessionOptionsConfigAllowIntraOpSpinning, "0"),
                  "AddSessionConfigEntry(allow_intra_op_spinning)", last_error_)) {
            return false;
        }
        if (!OrOk(ort_, ort_->SetIntraOpNumThreads(opts_, 1), "SetIntraOpNumThreads(XNNPACK)", last_error_)) {
            return false;
        }
        const char *keys[] = {"intra_op_num_threads"};
        const char *values[] = {"4"};
        if (!OrOk(ort_, ort_->SessionOptionsAppendExecutionProvider(opts_, "XNNPACK", keys, values, 1),
                  "XNNPACK append", last_error_)) {
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
            /* QNN EP with HTP backend -- full tuning for best NPU acceleration */
            backend_type = "htp";
            /* Probe the device SoC/HTP arch at runtime (falls back to auto on
             * unknown SoCs) instead of hard-coding values for one device. */
            std::string soc_model("0");
            std::string htp_arch("0");
            /* Runtime SoC/HTP-arch detection (shared module) -> per-SoC
             * defaults; falls back to "0"/auto on unknown SoCs. */
            qnn_soc_detect(soc_model, htp_arch);
            /* VTCM per arch: 8 MB on v81+ (validated on SM8850), else 0 = QNN
             * max, so a value larger than the chip's VTCM (e.g. SM8450 / v69)
             * never fails the graph compile. */
            char vtcm_buf[16] = {0};
            snprintf(vtcm_buf, sizeof(vtcm_buf), "%u", qnn_recommended_vtcm_mb(htp_arch));
            LOGI("ONNX: QNN(htp) vtcm_mb=%s (0 = use SoC max)", vtcm_buf);
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
                "vtcm_mb",
                "rpc_control_latency",
                "qnn_context_priority",
            };
            const char *htp_values[] = {
                "htp", // backend_type: HTP NPU backend
                soc_model.c_str(), // soc_model: auto-probed (0 = unknown/auto)
                htp_arch.c_str(),  // htp_arch: bare arch number ("81", not "v81")
                "off", // profiling_level: off basic detailed
                "/data/local/tmp/qnn_htp_profiling.csv",
                "burst", // htp_performance_mode: burst balanced default high_performance ...
                "3",     // htp_graph_finalization_optimization_mode: 0 1 2 3
                "1",     // enable_htp_fp16_precision: 0 1
                "0",     // enable_htp_shared_memory_allocator: 0 1 (1 requires rpcmem attr2,
                         //   only present in system libcdsprpc via HIDL without attr2 -> cache-hit
                         //   fails; 0 keeps epContext cache reuse working)
                vtcm_buf, // vtcm_mb: 0 = use SoC max (8 MB on v81+)
                "100",   // rpc_control_latency: RPC control latency in microseconds
                "HIGH",  // qnn_context_priority: LOW NORMAL NORMAL_HIGH HIGH
            };
            int num_opts = sizeof(htp_keys) / sizeof(htp_keys[0]);
            st = ort_->SessionOptionsAppendExecutionProvider(
                opts_, "QNN", htp_keys, htp_values, num_opts);
            /* Do not silently fall back unsupported nodes to CPU - surface a
             * load failure instead, so QNN EP results are never mixed with
             * CPU execution ("never silently degrade" rule). */
            if (!OrOk(ort_, ort_->AddSessionConfigEntry(opts_, "session.disable_cpu_ep_fallback", "1"),
                      "AddSessionConfigEntry(disable_cpu_ep_fallback)", last_error_)) {
                return false;
            }
        } else if (id_ == BackendId::ONNX_QNN_GPU) {
            /* QNN EP with Adreno GPU backend -- FP32/FP16, no HTP-specific options */
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
            /* QNN EP with CPU backend -- reference backend for graph validation */
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

        std::string qnn_what = std::string("QNN(") + backend_type + ") append";
        if (!OrOk(ort_, st, qnn_what.c_str(), last_error_)) {
            return false;
        }
        LOGI("ONNX: QNN(%s) EP configured", backend_type);
        return true;
#else
        LOGE("ONNX: QNN EP not available in this build");
        return false;
#endif
    }
    default: {
        return true; /* fallback to CPU */
    }
    }
}

/* ---------------------------------------------------------------------------
 * Initialize
 * -------------------------------------------------------------------------*/
bool ONNXBackend::Initialize(const char *model_path, int num_threads)
{
    auto t0 = std::chrono::high_resolution_clock::now();
    last_error_.clear();
    std::string dll_name = "onnxruntime.dll";

    /* EP-specific DLL paths relative to deps/ */
    const char *ep_subdir = nullptr;
    (void)ep_subdir; /* only used on desktop */
    switch (id_) {
    case BackendId::ONNX_CPU: { /* use default CPU DLL path below */
        break;
    }
    case BackendId::ONNX_DML_GPU:
    case BackendId::ONNX_DML_NPU: {
        ep_subdir = "dml";
        break;
    }
    case BackendId::ONNX_ONEDNN: {
        ep_subdir = "onednn";
        break;
    }
    case BackendId::ONNX_OPENVINO_CPU:
    case BackendId::ONNX_OPENVINO_GPU:
    case BackendId::ONNX_OPENVINO_GPU_FP16: {
        ep_subdir = "openvino";
        break;
    }
    case BackendId::ONNX_OPENVINO_NPU: {
        ep_subdir = "openvino";
        break;
    }
    default: {
        break;
    }
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
        last_error_ = "ONNX: failed to load onnxruntime library";
        return false;
    }
    timing_[1] = std::chrono::duration<double, std::milli>(
                     std::chrono::high_resolution_clock::now() - t0)
                     .count();

    auto t1 = std::chrono::high_resolution_clock::now();
    auto GetApiBase = (const OrtApiBase *(*)())load_function(lib_handle_, "OrtGetApiBase");
    if (!GetApiBase) {
        LOGE("ONNX: OrtGetApiBase not found");
        last_error_ = "ONNX: OrtGetApiBase not found";
        return false;
    }
    const OrtApiBase *base = GetApiBase();
    if (!base) {
        LOGE("ONNX: OrtGetApiBase returned NULL");
        last_error_ = "ONNX: OrtGetApiBase returned NULL";
        return false;
    }

    /* Resolve API version at runtime -- not hardcoded ORT_API_VERSION.
     * GetVersionString() returns e.g. "1.22.0", minor version = API version. */
    const char *ort_ver_str = base->GetVersionString();
    ort_api_ver_ = ORT_API_VERSION; /* fallback for unparseable version */
    if (ort_ver_str) {
        LOGI("ONNX Runtime version: %s", ort_ver_str);
        const char *dot = strchr(ort_ver_str, '.');
        if (dot) {
            ort_api_ver_ = (uint32_t)atoi(dot + 1);
        }
    } else {
        LOGW("ONNX: failed to get runtime version string, falling back to API v=1");
    }

    ort_ = base->GetApi(ort_api_ver_);
    if (!ort_) {
        LOGE("ONNX: GetApi(v=%u) returned NULL", ort_api_ver_);
        last_error_ = "ONNX: GetApi returned NULL";
        return false;
    } else {
        LOGI("get ONNX Runtime API version: %s", ort_ver_str);
    }
    timing_[2] = std::chrono::duration<double, std::milli>(
                     std::chrono::high_resolution_clock::now() - t1)
                     .count();

    auto t2 = std::chrono::high_resolution_clock::now();
    /* Map app log level to ORT logging level: DBG->VERBOSE, INFO->INFO, WARN->WARNING, ERR->ERROR, OFF->FATAL */
    OrtLoggingLevel ort_lvl;
    switch (Logger::level) {
    case LogLevel::DBG: {
        ort_lvl = ORT_LOGGING_LEVEL_VERBOSE;
        break;
    }
    case LogLevel::INFO: {
        ort_lvl = ORT_LOGGING_LEVEL_INFO;
        break;
    }
    case LogLevel::WARN: {
        ort_lvl = ORT_LOGGING_LEVEL_WARNING;
        break;
    }
    case LogLevel::ERR: {
        ort_lvl = ORT_LOGGING_LEVEL_ERROR;
        break;
    }
    default: {
        ort_lvl = ORT_LOGGING_LEVEL_FATAL;
        break;
    }
    }
    if (!OrOk(ort_, ort_->CreateEnv(ort_lvl, "unified_bench", &env_),
              "CreateEnv", last_error_)) {
        return false;
    }
    timing_[3] = std::chrono::duration<double, std::milli>(
                     std::chrono::high_resolution_clock::now() - t2)
                     .count();

    auto t3 = std::chrono::high_resolution_clock::now();
    if (!OrOk(ort_, ort_->CreateSessionOptions(&opts_),
              "CreateSessionOptions", last_error_)) {
        return false;
    }
    if (!OrOk(ort_, ort_->SetIntraOpNumThreads(opts_, num_threads), "SetIntraOpNumThreads", last_error_)) {
        return false;
    }
    if (!OrOk(ort_, ort_->SetSessionGraphOptimizationLevel(opts_, ORT_ENABLE_ALL),
              "SetSessionGraphOptimizationLevel", last_error_)) {
        return false;
    }
    /* Enable CPU memory arena to reduce allocation overhead */
    if (!OrOk(ort_, ort_->EnableCpuMemArena(opts_), "EnableCpuMemArena", last_error_)) {
        return false;
    }
    if (id_ == BackendId::ONNX_DML_GPU || id_ == BackendId::ONNX_DML_NPU) {
        if (!OrOk(ort_, ort_->AddSessionConfigEntry(opts_, "ep.dml.disable_graph_fusion", "1"),
                  "AddSessionConfigEntry(ep.dml.disable_graph_fusion)", last_error_)) {
            return false;
        }
        LOGI("ONNX: DML graph fusion disabled (per-operator kernels + CPU fallback)");
    }
    timing_[4] = std::chrono::duration<double, std::milli>(
                     std::chrono::high_resolution_clock::now() - t3)
                     .count();

    /* EP Context cache for QNN backends: compile once, reuse later.*/
    model_path_for_create_ = model_path;
    bool ep_cache_hit = false;
    if (id_ == BackendId::ONNX_QNN_CPU || id_ == BackendId::ONNX_QNN_GPU ||
        id_ == BackendId::ONNX_QNN_HTP) {
        /* Generate EP context file path: <model_stem>-<backend_name>-epContext.onnx */
        ep_context_path_ = std::string(model_path);
        const BackendConfig *bcfg = BackendRegistry::GetConfig(id_);
        const char *bname = bcfg ? bcfg->name.c_str() : "QNN";
        auto dot = ep_context_path_.rfind('.');
        if (dot != std::string::npos) {
            /* Build name like "test_model-ONNX_QNN_HTP-epContext.onnx" */
            std::string stem = ep_context_path_.substr(0, dot);
            ep_context_path_ = stem + "-" + bname + "-epContext.onnx";
        } else {
            ep_context_path_ += "-" + std::string(bname) + "-epContext.onnx";
        }

        if (file_readable_nonzero(ep_context_path_.c_str())) {
            /* Cached context exists -- use it directly, skip recompilation */
            if (!OrOk(ort_, ort_->AddSessionConfigEntry(opts_, kOrtSessionOptionsDisableModelCompile, "1"),
                      "AddSessionConfigEntry(disable_model_compile)", last_error_)) {
                return false;
            }
            model_path_for_create_ = ep_context_path_.c_str();
            ep_cache_hit = true;
            LOGI("ONNX: EP Context cache hit: %s", ep_context_path_.c_str());
        } else {
            /* First run -- enable EP context creation */
            if (!OrOk(ort_, ort_->AddSessionConfigEntry(opts_, kOrtSessionOptionEpContextEnable, "1"),
                      "AddSessionConfigEntry(ep_context_enable)", last_error_)) {
                return false;
            }
            if (!OrOk(ort_, ort_->AddSessionConfigEntry(opts_, kOrtSessionOptionEpContextFilePath, ep_context_path_.c_str()),
                      "AddSessionConfigEntry(ep_context_file_path)", last_error_)) {
                return false;
            }
            if (!OrOk(ort_, ort_->AddSessionConfigEntry(opts_, kOrtSessionOptionEpContextEmbedMode, "0"),
                      "AddSessionConfigEntry(ep_context_embed_mode)", last_error_)) {
                return false;
            }
            LOGI("ONNX: EP Context will be created: %s", ep_context_path_.c_str());
        }
    }

    auto t4 = std::chrono::high_resolution_clock::now();
    if (!ConfigureEP()) {
        LOGE("ONNX: ConfigureEP failed for backend id=%d", bid(id_));
        if (last_error_.empty()) {
            last_error_ = "ONNX: ConfigureEP failed";
        }
        return false;
    }
    timing_[5] = std::chrono::duration<double, std::milli>(
                     std::chrono::high_resolution_clock::now() - t4)
                     .count();

    auto t5 = std::chrono::high_resolution_clock::now();
    OrtStatus *st;
#ifdef _WIN32
    std::wstring wpath = utf8_to_wide(model_path_for_create_);
    st = ort_->CreateSession(env_, wpath.c_str(), opts_, &session_);
#else
    st = ort_->CreateSession(env_, model_path_for_create_, opts_, &session_);
#endif
    if (!OrOk(ort_, st, "CreateSession", last_error_)) {
        return false;
    }
    timing_[6] = std::chrono::duration<double, std::milli>(
                     std::chrono::high_resolution_clock::now() - t5)
                     .count();

    /* Design rule (see README): a backend whose initialization failed must NOT
     * silently fall back to CPU and report a fake CPU-speed number. ORT's QNN EP
     * falls back to CPU when the QNN backend cannot be set up (wrong
     * libonnxruntime.so build, mismatched QNN libs, missing rpcmem/attr2, no
     * libcdsprpc/HIDL DSP access, ...). Detect it via the EP context file: when
     * the QNN EP actually partitioned the graph, the epContext file is written;
     * when it fell back to CPU, no file is produced. Fail loudly instead.
     * NOTE: only GPU/HTP write an epContext; the QNN CPU backend does not
     * produce one (it re-compiles each run and runs at CPU speed anyway), so it
     * is excluded from this check. */
    if ((id_ == BackendId::ONNX_QNN_GPU || id_ == BackendId::ONNX_QNN_HTP) &&
        !ep_cache_hit && !file_readable_nonzero(ep_context_path_.c_str())) {
        LOGE("ONNX: QNN EP did NOT activate - no epContext was generated: %s. "
             "The QNN backend silently fell back to CPU; marking this backend FAILED "
             "instead of reporting a misleading CPU-speed result. "
             "Check: libonnxruntime.so must be the QNN-enabled build; QNN backend libs "
             "(libQnnHtp.so/... ) must be reachable via LD_LIBRARY_PATH; libcdsprpc/HIDL "
             "DSP access; QNN SDK version match.",
             ep_context_path_.c_str());
        last_error_ = "ONNX: QNN EP inactive (silent CPU fallback) - no epContext generated";
        ort_->ReleaseSession(session_);
        session_ = nullptr;
        return false;
    }

    if (!OrOk(ort_, ort_->GetAllocatorWithDefaultOptions(&alloc_),
              "GetAllocatorWithDefaultOptions", last_error_)) {
        return false;
    }

    auto t6 = std::chrono::high_resolution_clock::now();
    if (!QueryIOMetadata()) {
        /* QueryIOMetadata sets a specific last_error_ (unsupported input type
         * / dynamic shape / ORT API failure); keep it, only fall back to a
         * generic message when nothing was recorded. */
        if (last_error_.empty()) {
            last_error_ = "ONNX: QueryIOMetadata failed";
        }
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
    size_t n_in, n_out;
    if (!OrOk(ort_, ort_->SessionGetInputCount(session_, &n_in),
              "SessionGetInputCount", last_error_)) {
        return false;
    }
    if (!OrOk(ort_, ort_->SessionGetOutputCount(session_, &n_out),
              "SessionGetOutputCount", last_error_)) {
        return false;
    }
    num_inputs_ = std::min(n_in, MAX_IO);
    num_outputs_ = std::min(n_out, MAX_IO);

    for (size_t i = 0; i < num_inputs_; ++i) {
        char *name = nullptr;
        if (!OrOk(ort_, ort_->SessionGetInputName(session_, i, alloc_, &name),
                  "SessionGetInputName", last_error_)) {
            return false;
        }
        input_names_.push_back(name);
        OrtTypeInfo *ti;
        if (!OrOk(ort_, ort_->SessionGetInputTypeInfo(session_, i, &ti),
                  "SessionGetInputTypeInfo", last_error_)) {
            return false;
        }
        const OrtTensorTypeAndShapeInfo *si;
        if (!OrOk(ort_, ort_->CastTypeInfoToTensorInfo(ti, &si),
                  "CastTypeInfoToTensorInfo", last_error_)) {
            return false;
        }
        size_t nd;
        if (!OrOk(ort_, ort_->GetDimensionsCount(si, &nd),
                  "GetDimensionsCount", last_error_)) {
            return false;
        }
        std::vector<int64_t> dims(nd);
        if (!OrOk(ort_, ort_->GetDimensions(si, dims.data(), nd),
                  "GetDimensions", last_error_)) {
            return false;
        }
        /* Early, explicit rejection of unsupported inputs. The benchmark feeds
         * float32 buffers with static shapes, so a dynamic dim (-1) or a
         * non-FLOAT element type would otherwise fail later with a cryptic
         * "tried creating tensor with negative value in shape" deep inside
         * CreateTensorWithDataAsOrtValue on the first Run. Fail here instead
         * with a reason that names the actual model mismatch. */
        ONNXTensorElementDataType elem_type;
        if (!OrOk(ort_, ort_->GetTensorElementType(si, &elem_type),
                  "GetTensorElementType", last_error_)) {
            ort_->ReleaseTypeInfo(ti);
            return false;
        }
        if (elem_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            const char *in_name = name ? name : "?";
            LOGE("ONNX: input '%s' element type %d is not FLOAT - model type "
                 "mismatch, benchmark only supports float32 inputs",
                 in_name, (int)elem_type);
            last_error_ = "ONNX: input '" + std::string(in_name) +
                          "' has non-float element type (" +
                          std::to_string((int)elem_type) +
                          "), not supported (benchmark feeds float32 inputs)";
            ort_->ReleaseTypeInfo(ti);
            return false;
        }
        for (auto d : dims) {
            if (d < 0) {
                const char *in_name = name ? name : "?";
                LOGE("ONNX: input '%s' has dynamic input shape (dim=%lld) - "
                     "model type mismatch, benchmark requires static input "
                     "shapes",
                     in_name, (long long)d);
                last_error_ = "ONNX: input '" + std::string(in_name) +
                              "' has dynamic input shape (dim=" +
                              std::to_string(d) +
                              "), not supported (static shape required)";
                ort_->ReleaseTypeInfo(ti);
                return false;
            }
        }
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
        if (!OrOk(ort_, ort_->SessionGetOutputName(session_, i, alloc_, &name),
                  "SessionGetOutputName", last_error_)) {
            return false;
        }
        output_names_.push_back(name);
        output_names_str_.emplace_back(name ? name : "");
        OrtTypeInfo *ti;
        if (!OrOk(ort_, ort_->SessionGetOutputTypeInfo(session_, i, &ti),
                  "SessionGetOutputTypeInfo", last_error_)) {
            return false;
        }
        const OrtTensorTypeAndShapeInfo *si;
        if (!OrOk(ort_, ort_->CastTypeInfoToTensorInfo(ti, &si),
                  "CastTypeInfoToTensorInfo", last_error_)) {
            return false;
        }
        size_t nd;
        if (!OrOk(ort_, ort_->GetDimensionsCount(si, &nd),
                  "GetDimensionsCount", last_error_)) {
            return false;
        }
        std::vector<int64_t> dims(nd);
        if (!OrOk(ort_, ort_->GetDimensions(si, dims.data(), nd),
                  "GetDimensions", last_error_)) {
            return false;
        }
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
 * SetSharedInput
 * -------------------------------------------------------------------------*/
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

    if (!OrOk(ort_, ort_->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mem_info_),
              "CreateCpuMemoryInfo", last_error_)) {
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
            if (!OrOk(ort_, ort_->CreateTensorWithDataAsOrtValue(mem_info_, input_bufs_[i], n * sizeof(float), input_shapes_[i].data(), input_shapes_[i].size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &vals[i]),
                      "CreateTensorWithDataAsOrtValue", last_error_)) {
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
        if (!OrOk(ort_, ort_->Run(session_, nullptr, (const char *const *)input_names_.data(), in.data(), num_inputs_, (const char *const *)output_names_.data(), num_outputs_, out.data()),
                  "warmup Run", last_error_)) {
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
        if (!OrOk(ort_, ort_->Run(session_, nullptr, (const char *const *)input_names_.data(), in.data(), num_inputs_, (const char *const *)output_names_.data(), num_outputs_, out.data()),
                  "Run", last_error_)) {
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
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        LOGD("ONNX: run %d took %.3f ms", r, ms);
        for (size_t i = 0; i < num_outputs_; ++i) {
            if (out[i]) {
                float *fp = nullptr;
                if (OrOk(ort_, ort_->GetTensorMutableData(out[i], (void **)&fp),
                         "GetTensorMutableData", last_error_) &&
                    fp) {
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
            for (size_t j = 0; j < i; ++j) {
                free(odata[j]);
            }
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
