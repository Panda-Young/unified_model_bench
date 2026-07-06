/*============================================================================
 * backend_registry.cpp - Backend registration & factory
 *============================================================================*/

#include "backend_interface.hpp"
#include "log.hpp"
#include <map>
#include <mutex>
#include <vector>
#include <algorithm>

/* ---------------------------------------------------------------------------
 * Internal storage
 * -------------------------------------------------------------------------*/
struct RegistryEntry {
    BackendConfig  config;
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

/* ---------------------------------------------------------------------------
 * Public API
 * -------------------------------------------------------------------------*/
namespace BackendRegistry {

void Register(BackendId id, BackendConfig cfg, BackendFactory factory) {
    std::lock_guard<std::mutex> lock(g_registry_mutex);
    LOGD("Registered backend: %s (id=%d)", cfg.name.c_str(), bid(id));
    g_registry[bid(id)] = {std::move(cfg), std::move(factory)};
}

BackendPtr Create(BackendId id) {
    std::lock_guard<std::mutex> lock(g_registry_mutex);
    auto it = g_registry.find(bid(id));
    if (it == g_registry.end()) {
        LOGE("Unknown backend ID: %d", bid(id));
        return nullptr;
    }
    return it->second.factory(id);
}

std::vector<BackendConfig> GetAvailable(ModelFormat format) {
    std::lock_guard<std::mutex> lock(g_registry_mutex);
    std::vector<BackendConfig> result;
    for (const auto& [id, entry] : g_registry) {
        bool match = false;
        switch (format) {
        case ModelFormat::ONNX:   match = is_onnx_backend(static_cast<BackendId>(id)); break;
        case ModelFormat::TFLITE: match = is_tflite_backend(static_cast<BackendId>(id)); break;
        case ModelFormat::NCNN:   match = is_ncnn_backend(static_cast<BackendId>(id)); break;
        case ModelFormat::MNN:    match = is_mnn_backend(static_cast<BackendId>(id)); break;
        default: break;
        }
        if (match) result.push_back(entry.config);
    }
    /* Sort by ID */
    std::sort(result.begin(), result.end(),
              [](const BackendConfig& a, const BackendConfig& b) {
                  return bid(a.id) < bid(b.id);
              });
    return result;
}

const BackendConfig* GetConfig(BackendId id) {
    std::lock_guard<std::mutex> lock(g_registry_mutex);
    auto it = g_registry.find(bid(id));
    if (it == g_registry.end()) return nullptr;
    return &it->second.config;
}

/* ---------------------------------------------------------------------------
 * Platform-specific default registrations
 * -------------------------------------------------------------------------*/
void InitDefaults() {
#if defined(__ANDROID__) || defined(__android__)
    /* --- Android --- */
#  ifdef HAVE_ONNX_BACKEND
    Register(BackendId::ONNX_CPU,     {BackendId::ONNX_CPU,     BackendType::ONNX_EP, "CPU",     "", true},  CreateOnnxBackend);
    Register(BackendId::ONNX_NNAPI,   {BackendId::ONNX_NNAPI,   BackendType::ONNX_EP, "NNAPI",   "", false}, CreateOnnxBackend);
    Register(BackendId::ONNX_XNNPACK, {BackendId::ONNX_XNNPACK, BackendType::ONNX_EP, "XNNPACK", "", false}, CreateOnnxBackend);
    Register(BackendId::ONNX_QNN_CPU, {BackendId::ONNX_QNN_CPU, BackendType::ONNX_EP, "QNN_CPU", "", false}, CreateOnnxBackend);
    Register(BackendId::ONNX_QNN_GPU, {BackendId::ONNX_QNN_GPU, BackendType::ONNX_EP, "QNN_GPU", "", false}, CreateOnnxBackend);
    Register(BackendId::ONNX_QNN_HTP, {BackendId::ONNX_QNN_HTP, BackendType::ONNX_EP, "QNN_HTP", "", false}, CreateOnnxBackend);
#  endif
#  ifdef HAVE_TFLITE_BACKEND
    Register(BackendId::TFLITE_CPU,     {BackendId::TFLITE_CPU,     BackendType::TFLITE_DEL, "CPU",     "", true},  CreateTfliteBackend);
    Register(BackendId::TFLITE_XNNPACK, {BackendId::TFLITE_XNNPACK, BackendType::TFLITE_DEL, "XNNPACK", "", false}, CreateTfliteBackend);
    Register(BackendId::TFLITE_NNAPI,   {BackendId::TFLITE_NNAPI,   BackendType::TFLITE_DEL, "NNAPI",   "", false}, CreateTfliteBackend);
    Register(BackendId::TFLITE_GPU,     {BackendId::TFLITE_GPU,     BackendType::TFLITE_DEL, "GPU",     "", false}, CreateTfliteBackend);
#  endif
#  ifdef HAVE_NCNN_BACKEND
    Register(BackendId::NCNN_CPU,         {BackendId::NCNN_CPU,         BackendType::NCNN, "NCNN_CPU",         "", true},  CreateNcnnBackend);
    Register(BackendId::NCNN_VULKAN,      {BackendId::NCNN_VULKAN,      BackendType::NCNN, "NCNN_Vulkan",      "", false}, CreateNcnnBackend);
    Register(BackendId::NCNN_VULKAN_FP16, {BackendId::NCNN_VULKAN_FP16, BackendType::NCNN, "NCNN_Vulkan_FP16", "", false}, CreateNcnnBackend);
#  endif
#  ifdef HAVE_MNN_BACKEND
    Register(BackendId::MNN_CPU, {BackendId::MNN_CPU, BackendType::MNN, "MNN_CPU", "", true}, CreateMnnBackend);
    /* Android: only CPU is stable */
#  endif

#else
    /* --- Desktop --- */
#  ifdef HAVE_ONNX_BACKEND
#    if defined(_WIN64) || defined(__linux__)
    Register(BackendId::ONNX_CPU,           {BackendId::ONNX_CPU,           BackendType::ONNX_EP, "CPU",           "", true},  CreateOnnxBackend);
    // Register(BackendId::ONNX_ONEDNN,        {BackendId::ONNX_ONEDNN,        BackendType::ONNX_EP, "oneDNN",        "", false}, CreateOnnxBackend);
    // Register(BackendId::ONNX_DML,           {BackendId::ONNX_DML,           BackendType::ONNX_EP, "DML",           "", false}, CreateOnnxBackend);
    // Register(BackendId::ONNX_OPENVINO_CPU,  {BackendId::ONNX_OPENVINO_CPU,  BackendType::ONNX_EP, "OpenVINO_CPU",  "", false}, CreateOnnxBackend);
    // Register(BackendId::ONNX_OPENVINO_GPU,  {BackendId::ONNX_OPENVINO_GPU,  BackendType::ONNX_EP, "OpenVINO_GPU",  "", false}, CreateOnnxBackend);
#    else /* 32-bit Windows */
    Register(BackendId::ONNX_CPU, {BackendId::ONNX_CPU, BackendType::ONNX_EP, "CPU", "", true},  CreateOnnxBackend);
    Register(BackendId::ONNX_DML, {BackendId::ONNX_DML, BackendType::ONNX_EP, "DML", "", false}, CreateOnnxBackend);
#    endif
#  endif
#  ifdef HAVE_TFLITE_BACKEND
    Register(BackendId::TFLITE_CPU, {BackendId::TFLITE_CPU, BackendType::TFLITE_DEL, "CPU", "", true}, CreateTfliteBackend);
#  endif
#  ifdef HAVE_NCNN_BACKEND
    Register(BackendId::NCNN_CPU,         {BackendId::NCNN_CPU,         BackendType::NCNN, "NCNN_CPU",         "", true},  CreateNcnnBackend);
    Register(BackendId::NCNN_VULKAN,      {BackendId::NCNN_VULKAN,      BackendType::NCNN, "NCNN_Vulkan",      "", false}, CreateNcnnBackend);
    Register(BackendId::NCNN_VULKAN_FP16, {BackendId::NCNN_VULKAN_FP16, BackendType::NCNN, "NCNN_Vulkan_FP16", "", false}, CreateNcnnBackend);
#  endif
#  ifdef HAVE_MNN_BACKEND
    Register(BackendId::MNN_CPU,         {BackendId::MNN_CPU,         BackendType::MNN, "MNN_CPU",         "", true},  CreateMnnBackend);
    Register(BackendId::MNN_OPENCL,      {BackendId::MNN_OPENCL,      BackendType::MNN, "MNN_OpenCL",      "", false}, CreateMnnBackend);
    Register(BackendId::MNN_OPENCL_FP16, {BackendId::MNN_OPENCL_FP16, BackendType::MNN, "MNN_OpenCL_FP16", "", false}, CreateMnnBackend);
#  endif
#endif

    LOGI("Backend registry initialized with %zu backends", g_registry.size());
}

} // namespace BackendRegistry
