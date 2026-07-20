/*============================================================================
 * backend_registry.cpp - Backend registration & factory
 *============================================================================*/

#include "backend_interface.hpp"
#include "log.hpp"
#include <algorithm>
#include <cctype>
#include <map>
#include <mutex>
#include <vector>

/* ---------------------------------------------------------------------------
 * Internal storage
 * -------------------------------------------------------------------------*/
struct RegistryEntry {
    BackendConfig config;
    BackendFactory factory;
};

static std::map<int, RegistryEntry> g_registry;
static std::mutex g_registry_mutex;

/* ---------------------------------------------------------------------------
 * Backend factory forward declarations (defined in respective .cpp files)
 * -------------------------------------------------------------------------*/
#ifdef HAVE_ONNX_BACKEND
extern BackendPtr CreateOnnxBackend(BackendId id);
#endif
#ifdef HAVE_TFLITE_BACKEND
extern BackendPtr CreateTfliteBackend(BackendId id);
#endif
#ifdef HAVE_NCNN_BACKEND
extern BackendPtr CreateNcnnBackend(BackendId id);
#endif
#ifdef HAVE_MNN_BACKEND
extern BackendPtr CreateMnnBackend(BackendId id);
#endif
#ifdef HAVE_LITERT_BACKEND
extern BackendPtr CreateLitertBackend(BackendId id);
#endif

/* ---------------------------------------------------------------------------
 * Public API
 * -------------------------------------------------------------------------*/
namespace BackendRegistry
{

    void Register(BackendId id, BackendConfig cfg, BackendFactory factory)
    {
        std::lock_guard<std::mutex> lock(g_registry_mutex);
        LOGD("Registered backend: %s (id=%d)", cfg.name.c_str(), bid(id));
        g_registry[bid(id)] = {std::move(cfg), std::move(factory)};
    }

    BackendPtr Create(BackendId id)
    {
        std::lock_guard<std::mutex> lock(g_registry_mutex);
        auto it = g_registry.find(bid(id));
        if (it == g_registry.end()) {
            LOGE("Unknown backend ID: %d", bid(id));
            return nullptr;
        }
        return it->second.factory(id);
    }

    std::vector<BackendConfig> GetAvailable(ModelFormat format)
    {
        std::lock_guard<std::mutex> lock(g_registry_mutex);
        std::vector<BackendConfig> result;
        for (const auto &[id, entry] : g_registry) {
            bool match = false;
            switch (format) {
            case ModelFormat::ONNX:
                match = is_onnx_backend(static_cast<BackendId>(id));
                break;
            case ModelFormat::TFLITE:
                match = is_tflite_backend(static_cast<BackendId>(id)) || is_litert_backend(static_cast<BackendId>(id));
                break;
            case ModelFormat::NCNN:
                match = is_ncnn_backend(static_cast<BackendId>(id));
                break;
            case ModelFormat::MNN:
                match = is_mnn_backend(static_cast<BackendId>(id));
                break;
            default:
                break;
            }
            if (match) {
                result.push_back(entry.config);
            }
        }
        /* Sort by ID */
        std::sort(result.begin(), result.end(),
                  [](const BackendConfig &a, const BackendConfig &b) {
                      return bid(a.id) < bid(b.id);
                  });
        return result;
    }

    const BackendConfig *GetConfig(BackendId id)
    {
        std::lock_guard<std::mutex> lock(g_registry_mutex);
        auto it = g_registry.find(bid(id));
        if (it == g_registry.end()) {
            return nullptr;
        }
        return &it->second.config;
    }

    int FindByName(const char *name)
    {
        std::lock_guard<std::mutex> lock(g_registry_mutex);
        /* Build lowercase version of the query */
        std::string lower(name);
        for (auto &c : lower)
            c = (char)tolower((unsigned char)c);
        for (const auto &[id, entry] : g_registry) {
            std::string ename = entry.config.name;
            for (auto &c : ename)
                c = (char)tolower((unsigned char)c);
            if (ename == lower) {
                return id;
            }
        }
        return -1;
    }

    /* ---------------------------------------------------------------------------
     * Platform-specific default registrations
     * -------------------------------------------------------------------------*/
    void InitDefaults()
    {
#if defined(__ANDROID__) || defined(__android__)
        /* --- Android --- */
#ifdef HAVE_ONNX_BACKEND
        Register(BackendId::ONNX_CPU, {BackendId::ONNX_CPU, BackendType::ONNX_EP, "ONNX_CPU", "", true}, CreateOnnxBackend);
        Register(BackendId::ONNX_NNAPI, {BackendId::ONNX_NNAPI, BackendType::ONNX_EP, "ONNX_NNAPI", "", false}, CreateOnnxBackend);
        Register(BackendId::ONNX_XNNPACK, {BackendId::ONNX_XNNPACK, BackendType::ONNX_EP, "ONNX_XNNPACK", "", false}, CreateOnnxBackend);
        Register(BackendId::ONNX_QNN_CPU, {BackendId::ONNX_QNN_CPU, BackendType::ONNX_EP, "ONNX_QNN_CPU", "", false}, CreateOnnxBackend);
        Register(BackendId::ONNX_QNN_GPU, {BackendId::ONNX_QNN_GPU, BackendType::ONNX_EP, "ONNX_QNN_GPU", "", false}, CreateOnnxBackend);
        Register(BackendId::ONNX_QNN_HTP, {BackendId::ONNX_QNN_HTP, BackendType::ONNX_EP, "ONNX_QNN_HTP", "", false}, CreateOnnxBackend);
#endif
#ifdef HAVE_TFLITE_BACKEND
        Register(BackendId::TFLITE_CPU, {BackendId::TFLITE_CPU, BackendType::TFLITE_DEL, "TFLITE_CPU", "", true}, CreateTfliteBackend);
        Register(BackendId::TFLITE_XNNPACK, {BackendId::TFLITE_XNNPACK, BackendType::TFLITE_DEL, "TFLITE_XNNPACK", "", false}, CreateTfliteBackend);
        Register(BackendId::TFLITE_XNNPACK_FP16, {BackendId::TFLITE_XNNPACK_FP16, BackendType::TFLITE_DEL, "TFLITE_XNNPACK_FP16", "", false}, CreateTfliteBackend);
        Register(BackendId::TFLITE_NNAPI, {BackendId::TFLITE_NNAPI, BackendType::TFLITE_DEL, "TFLITE_NNAPI", "", false}, CreateTfliteBackend);
        Register(BackendId::TFLITE_GPU, {BackendId::TFLITE_GPU, BackendType::TFLITE_DEL, "TFLITE_GPU", "", false}, CreateTfliteBackend);
        Register(BackendId::TFLITE_GPU_FP16, {BackendId::TFLITE_GPU_FP16, BackendType::TFLITE_DEL, "TFLITE_GPU_FP16", "", false}, CreateTfliteBackend);
        Register(BackendId::TFLITE_NPU, {BackendId::TFLITE_NPU, BackendType::TFLITE_DEL, "TFLITE_NPU", "", false}, CreateTfliteBackend);
#endif
#ifdef HAVE_LITERT_BACKEND
        Register(BackendId::LITERT_CPU, {BackendId::LITERT_CPU, BackendType::LITERT, "LiteRT_CPU", "", true}, CreateLitertBackend);
        Register(BackendId::LITERT_GPU, {BackendId::LITERT_GPU, BackendType::LITERT, "LiteRT_GPU", "", false}, CreateLitertBackend);
        Register(BackendId::LITERT_GPU_FP16, {BackendId::LITERT_GPU_FP16, BackendType::LITERT, "LiteRT_GPU_FP16", "", false}, CreateLitertBackend);
        Register(BackendId::LITERT_NPU, {BackendId::LITERT_NPU, BackendType::LITERT, "LiteRT_NPU", "", false}, CreateLitertBackend);
        Register(BackendId::LITERT_NPU_FP16, {BackendId::LITERT_NPU_FP16, BackendType::LITERT, "LiteRT_NPU_FP16", "", false}, CreateLitertBackend);
#endif
#ifdef HAVE_NCNN_BACKEND
        Register(BackendId::NCNN_CPU, {BackendId::NCNN_CPU, BackendType::NCNN, "NCNN_CPU", "", true}, CreateNcnnBackend);
        Register(BackendId::NCNN_CPU_FP16, {BackendId::NCNN_CPU_FP16, BackendType::NCNN, "NCNN_CPU_FP16", "", false}, CreateNcnnBackend);
        Register(BackendId::NCNN_CPU_BF16, {BackendId::NCNN_CPU_BF16, BackendType::NCNN, "NCNN_CPU_BF16", "", false}, CreateNcnnBackend);
        Register(BackendId::NCNN_VULKAN, {BackendId::NCNN_VULKAN, BackendType::NCNN, "NCNN_Vulkan", "", false}, CreateNcnnBackend);
        Register(BackendId::NCNN_VULKAN_FP16, {BackendId::NCNN_VULKAN_FP16, BackendType::NCNN, "NCNN_Vulkan_FP16", "", false}, CreateNcnnBackend);
        Register(BackendId::NCNN_VULKAN_BF16, {BackendId::NCNN_VULKAN_BF16, BackendType::NCNN, "NCNN_Vulkan_BF16", "", false}, CreateNcnnBackend);
#endif
#ifdef HAVE_MNN_BACKEND
        Register(BackendId::MNN_CPU, {BackendId::MNN_CPU, BackendType::MNN, "MNN_CPU", "", true}, CreateMnnBackend);
        Register(BackendId::MNN_OPENCL, {BackendId::MNN_OPENCL, BackendType::MNN, "MNN_OpenCL", "", false}, CreateMnnBackend);
        Register(BackendId::MNN_OPENCL_FP16, {BackendId::MNN_OPENCL_FP16, BackendType::MNN, "MNN_OpenCL_FP16", "", false}, CreateMnnBackend);
        Register(BackendId::MNN_OPENCL_BF16, {BackendId::MNN_OPENCL_BF16, BackendType::MNN, "MNN_OpenCL_BF16", "", false}, CreateMnnBackend);
        Register(BackendId::MNN_VULKAN, {BackendId::MNN_VULKAN, BackendType::MNN, "MNN_VULKAN", "", false}, CreateMnnBackend);
        Register(BackendId::MNN_VULKAN_FP16, {BackendId::MNN_VULKAN_FP16, BackendType::MNN, "MNN_VULKAN_FP16", "", false}, CreateMnnBackend);
        Register(BackendId::MNN_VULKAN_BF16, {BackendId::MNN_VULKAN_BF16, BackendType::MNN, "MNN_VULKAN_BF16", "", false}, CreateMnnBackend);
        Register(BackendId::MNN_OPENGL, {BackendId::MNN_OPENGL, BackendType::MNN, "MNN_OPENGL", "", false}, CreateMnnBackend);
        Register(BackendId::MNN_NN, {BackendId::MNN_NN, BackendType::MNN, "MNN_NN", "", false}, CreateMnnBackend);
#endif

#else
        /* --- Desktop --- */
#ifdef HAVE_ONNX_BACKEND
#if defined(_WIN64) || defined(__linux__)
        Register(BackendId::ONNX_CPU, {BackendId::ONNX_CPU, BackendType::ONNX_EP, "ONNX_CPU", "", true}, CreateOnnxBackend);
        Register(BackendId::ONNX_ONEDNN, {BackendId::ONNX_ONEDNN, BackendType::ONNX_EP, "ONNX_oneDNN", "", false}, CreateOnnxBackend);
        Register(BackendId::ONNX_DML_GPU, {BackendId::ONNX_DML_GPU, BackendType::ONNX_EP, "ONNX_DML_GPU", "", false}, CreateOnnxBackend);
        Register(BackendId::ONNX_DML_NPU, {BackendId::ONNX_DML_NPU, BackendType::ONNX_EP, "ONNX_DML_NPU", "", false}, CreateOnnxBackend);
        Register(BackendId::ONNX_OPENVINO_CPU, {BackendId::ONNX_OPENVINO_CPU, BackendType::ONNX_EP, "ONNX_OpenVINO_CPU", "", false}, CreateOnnxBackend);
        Register(BackendId::ONNX_OPENVINO_GPU, {BackendId::ONNX_OPENVINO_GPU, BackendType::ONNX_EP, "ONNX_OpenVINO_GPU", "", false}, CreateOnnxBackend);
        Register(BackendId::ONNX_OPENVINO_NPU, {BackendId::ONNX_OPENVINO_NPU, BackendType::ONNX_EP, "ONNX_OpenVINO_NPU", "", false}, CreateOnnxBackend);
        Register(BackendId::ONNX_DML_GPU_FP16, {BackendId::ONNX_DML_GPU_FP16, BackendType::ONNX_EP, "ONNX_DML_GPU_FP16", "", false}, CreateOnnxBackend);
        Register(BackendId::ONNX_OPENVINO_GPU_FP16, {BackendId::ONNX_OPENVINO_GPU_FP16, BackendType::ONNX_EP, "ONNX_OpenVINO_GPU_FP16", "", false}, CreateOnnxBackend);
#else /* 32-bit Windows */
        Register(BackendId::ONNX_CPU, {BackendId::ONNX_CPU, BackendType::ONNX_EP, "ONNX_CPU", "", true}, CreateOnnxBackend);
        Register(BackendId::ONNX_DML_GPU, {BackendId::ONNX_DML_GPU, BackendType::ONNX_EP, "ONNX_DML_GPU", "", false}, CreateOnnxBackend);
        Register(BackendId::ONNX_DML_NPU, {BackendId::ONNX_DML_NPU, BackendType::ONNX_EP, "ONNX_DML_NPU", "", false}, CreateOnnxBackend);
        Register(BackendId::ONNX_DML_GPU_FP16, {BackendId::ONNX_DML_GPU_FP16, BackendType::ONNX_EP, "ONNX_DML_GPU_FP16", "", false}, CreateOnnxBackend);
#endif
#endif
#ifdef HAVE_TFLITE_BACKEND
        Register(BackendId::TFLITE_CPU, {BackendId::TFLITE_CPU, BackendType::TFLITE_DEL, "TFLITE_CPU", "", true}, CreateTfliteBackend);
        Register(BackendId::TFLITE_XNNPACK, {BackendId::TFLITE_XNNPACK, BackendType::TFLITE_DEL, "TFLITE_XNNPACK", "", false}, CreateTfliteBackend);
        Register(BackendId::TFLITE_XNNPACK_FP16, {BackendId::TFLITE_XNNPACK_FP16, BackendType::TFLITE_DEL, "TFLITE_XNNPACK_FP16", "", false}, CreateTfliteBackend);
#endif
#ifdef HAVE_LITERT_BACKEND
        Register(BackendId::LITERT_CPU, {BackendId::LITERT_CPU, BackendType::LITERT, "LiteRT_CPU", "", true}, CreateLitertBackend);
        Register(BackendId::LITERT_GPU, {BackendId::LITERT_GPU, BackendType::LITERT, "LiteRT_GPU", "", false}, CreateLitertBackend);
        Register(BackendId::LITERT_GPU_FP16, {BackendId::LITERT_GPU_FP16, BackendType::LITERT, "LiteRT_GPU_FP16", "", false}, CreateLitertBackend);
#endif
#ifdef HAVE_NCNN_BACKEND
        Register(BackendId::NCNN_CPU, {BackendId::NCNN_CPU, BackendType::NCNN, "NCNN_CPU", "", true}, CreateNcnnBackend);
        Register(BackendId::NCNN_CPU_FP16, {BackendId::NCNN_CPU_FP16, BackendType::NCNN, "NCNN_CPU_FP16", "", false}, CreateNcnnBackend);
        Register(BackendId::NCNN_CPU_BF16, {BackendId::NCNN_CPU_BF16, BackendType::NCNN, "NCNN_CPU_BF16", "", false}, CreateNcnnBackend);
        Register(BackendId::NCNN_VULKAN, {BackendId::NCNN_VULKAN, BackendType::NCNN, "NCNN_Vulkan", "", false}, CreateNcnnBackend);
        Register(BackendId::NCNN_VULKAN_BF16, {BackendId::NCNN_VULKAN_BF16, BackendType::NCNN, "NCNN_Vulkan_BF16", "", false}, CreateNcnnBackend);
        Register(BackendId::NCNN_VULKAN_FP16, {BackendId::NCNN_VULKAN_FP16, BackendType::NCNN, "NCNN_Vulkan_FP16", "", false}, CreateNcnnBackend);
#endif
#ifdef HAVE_MNN_BACKEND
        Register(BackendId::MNN_CPU, {BackendId::MNN_CPU, BackendType::MNN, "MNN_CPU", "", true}, CreateMnnBackend);
        Register(BackendId::MNN_OPENCL, {BackendId::MNN_OPENCL, BackendType::MNN, "MNN_OpenCL", "", false}, CreateMnnBackend);
        Register(BackendId::MNN_OPENCL_FP16, {BackendId::MNN_OPENCL_FP16, BackendType::MNN, "MNN_OpenCL_FP16", "", false}, CreateMnnBackend);
        Register(BackendId::MNN_OPENCL_BF16, {BackendId::MNN_OPENCL_BF16, BackendType::MNN, "MNN_OpenCL_BF16", "", false}, CreateMnnBackend);
        Register(BackendId::MNN_VULKAN, {BackendId::MNN_VULKAN, BackendType::MNN, "MNN_VULKAN", "", false}, CreateMnnBackend);
        Register(BackendId::MNN_VULKAN_FP16, {BackendId::MNN_VULKAN_FP16, BackendType::MNN, "MNN_VULKAN_FP16", "", false}, CreateMnnBackend);
        Register(BackendId::MNN_VULKAN_BF16, {BackendId::MNN_VULKAN_BF16, BackendType::MNN, "MNN_VULKAN_BF16", "", false}, CreateMnnBackend);
        Register(BackendId::MNN_OPENGL, {BackendId::MNN_OPENGL, BackendType::MNN, "MNN_OPENGL", "", false}, CreateMnnBackend);
#endif
#endif

        LOGI("Backend registry initialized with %zu backends", g_registry.size());
    }

} // namespace BackendRegistry
