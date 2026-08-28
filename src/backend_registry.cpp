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
#ifdef HAVE_QNN_SDK_BACKEND
extern BackendPtr CreateQnnSdkBackend(BackendId id);
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
            case ModelFormat::ONNX: {
                match = is_onnx_backend(static_cast<BackendId>(id));
                break;
            }
            case ModelFormat::TFLITE: {
                match = is_tflite_backend(static_cast<BackendId>(id)) || is_litert_backend(static_cast<BackendId>(id));
                break;
            }
            case ModelFormat::NCNN: {
                match = is_ncnn_backend(static_cast<BackendId>(id));
                break;
            }
            case ModelFormat::MNN: {
                match = is_mnn_backend(static_cast<BackendId>(id));
                break;
            }
            case ModelFormat::QNN: {
                match = is_qnn_sdk_backend(static_cast<BackendId>(id));
                break;
            }
            default: {
                break;
            }
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
        for (auto &c : lower) {
            c = (char)tolower((unsigned char)c);
        }
        for (const auto &[id, entry] : g_registry) {
            std::string ename = entry.config.name;
            for (auto &c : ename) {
                c = (char)tolower((unsigned char)c);
            }
            if (ename == lower) {
                return id;
            }
        }
        return -1;
    }

    /* ---------------------------------------------------------------------------
     * Default registrations - DATA DRIVEN
     *
     * Platform availability used to be expressed by hand-written #if branches
     * (one big Android block + one big Desktop block + a nested 32/64-bit
     * split), which had to be kept in sync with the README matrix by hand.
     * It is now a single declaration list filtered by PlatformMask, so:
     *   - adding a backend means adding ONE row
     *   - "which backends exist on this platform?" is answerable at runtime
     *   - a new platform only needs a new PlatformMask bit
     *
     * Compile-time gating still applies on top (HAVE_*_BACKEND, and the
     * arch-specific #if below for oneDNN / OpenVINO, whose 32-bit libraries
     * simply do not exist).
     * -------------------------------------------------------------------------*/
    void InitDefaults()
    {
        /* One declaration per backend. `factory` is null for placeholders that
         * are not implemented on this build; those rows are skipped. */
        struct BackendDecl {
            BackendId id;
            BackendType type;
            const char *name;
            bool is_cpu_baseline;
            unsigned platforms;
            BackendFactory factory;
        };

        /* Backends whose upstream libraries ship for 64-bit desktop only
         * (no 32-bit builds exist), so they are unavailable on win-x86. */
#if defined(_WIN64) || defined(__linux__)
        constexpr unsigned kPlatDesktop64 = kPlatWin | kPlatLinux;
#else
        constexpr unsigned kPlatDesktop64 = kPlatLinux;
#endif

        const std::vector<BackendDecl> decls = {
#ifdef HAVE_ONNX_BACKEND
            {BackendId::ONNX_CPU, BackendType::ONNX_EP, "ONNX_CPU", true,
             kPlatWin | kPlatLinux | kPlatAndroid, CreateOnnxBackend},
            {BackendId::ONNX_ONEDNN, BackendType::ONNX_EP, "ONNX_oneDNN", false,
             kPlatDesktop64, CreateOnnxBackend},
            {BackendId::ONNX_DML_GPU, BackendType::ONNX_EP, "ONNX_DML_GPU", false,
             kPlatWin, CreateOnnxBackend},
            {BackendId::ONNX_DML_NPU, BackendType::ONNX_EP, "ONNX_DML_NPU", false,
             kPlatWin, CreateOnnxBackend},
            {BackendId::ONNX_OPENVINO_CPU, BackendType::ONNX_EP, "ONNX_OpenVINO_CPU", false,
             kPlatDesktop64, CreateOnnxBackend},
            {BackendId::ONNX_OPENVINO_GPU, BackendType::ONNX_EP, "ONNX_OpenVINO_GPU", false,
             kPlatDesktop64, CreateOnnxBackend},
            {BackendId::ONNX_OPENVINO_GPU_FP16, BackendType::ONNX_EP, "ONNX_OpenVINO_GPU_FP16", false,
             kPlatDesktop64, CreateOnnxBackend},
            {BackendId::ONNX_OPENVINO_NPU, BackendType::ONNX_EP, "ONNX_OpenVINO_NPU", false,
             kPlatDesktop64, CreateOnnxBackend},
            {BackendId::ONNX_NNAPI, BackendType::ONNX_EP, "ONNX_NNAPI", false,
             kPlatAndroid, CreateOnnxBackend},
            {BackendId::ONNX_XNNPACK, BackendType::ONNX_EP, "ONNX_XNNPACK", false,
             kPlatAndroid, CreateOnnxBackend},
            {BackendId::ONNX_QNN_CPU, BackendType::ONNX_EP, "ONNX_QNN_CPU", false,
             kPlatAndroid, CreateOnnxBackend},
            {BackendId::ONNX_QNN_GPU, BackendType::ONNX_EP, "ONNX_QNN_GPU", false,
             kPlatAndroid, CreateOnnxBackend},
            {BackendId::ONNX_QNN_HTP, BackendType::ONNX_EP, "ONNX_QNN_HTP", false,
             kPlatAndroid, CreateOnnxBackend},
#endif
#ifdef HAVE_TFLITE_BACKEND
            {BackendId::TFLITE_CPU, BackendType::TFLITE_DEL, "TFLITE_CPU", true,
             kPlatWin | kPlatLinux | kPlatAndroid, CreateTfliteBackend},
            {BackendId::TFLITE_XNNPACK, BackendType::TFLITE_DEL, "TFLITE_XNNPACK", false,
             kPlatWin | kPlatLinux | kPlatAndroid, CreateTfliteBackend},
            {BackendId::TFLITE_XNNPACK_FP16, BackendType::TFLITE_DEL, "TFLITE_XNNPACK_FP16", false,
             kPlatWin | kPlatLinux | kPlatAndroid, CreateTfliteBackend},
            {BackendId::TFLITE_NNAPI, BackendType::TFLITE_DEL, "TFLITE_NNAPI", false,
             kPlatAndroid, CreateTfliteBackend},
            {BackendId::TFLITE_GPU, BackendType::TFLITE_DEL, "TFLITE_GPU", false,
             kPlatAndroid, CreateTfliteBackend},
            {BackendId::TFLITE_GPU_FP16, BackendType::TFLITE_DEL, "TFLITE_GPU_FP16", false,
             kPlatAndroid, CreateTfliteBackend},
            {BackendId::TFLITE_NPU, BackendType::TFLITE_DEL, "TFLITE_NPU", false,
             kPlatAndroid, CreateTfliteBackend},
#endif
#ifdef HAVE_LITERT_BACKEND
            {BackendId::LITERT_CPU, BackendType::LITERT, "LiteRT_CPU", true,
             kPlatWin | kPlatLinux | kPlatAndroid, CreateLitertBackend},
            {BackendId::LITERT_GPU, BackendType::LITERT, "LiteRT_GPU", false,
             kPlatWin | kPlatLinux | kPlatAndroid, CreateLitertBackend},
            {BackendId::LITERT_GPU_FP16, BackendType::LITERT, "LiteRT_GPU_FP16", false,
             kPlatWin | kPlatLinux | kPlatAndroid, CreateLitertBackend},
            {BackendId::LITERT_NPU, BackendType::LITERT, "LiteRT_NPU", false,
             kPlatAndroid, CreateLitertBackend},
            {BackendId::LITERT_NPU_FP16, BackendType::LITERT, "LiteRT_NPU_FP16", false,
             kPlatAndroid, CreateLitertBackend},
#endif
#ifdef HAVE_QNN_SDK_BACKEND
            {BackendId::QNN_SDK_HTP, BackendType::QNN_SDK, "QNN_SDK_HTP", false,
             kPlatAndroid, CreateQnnSdkBackend},
            {BackendId::QNN_SDK_GPU, BackendType::QNN_SDK, "QNN_SDK_GPU", false,
             kPlatAndroid, CreateQnnSdkBackend},
            {BackendId::QNN_SDK_CPU, BackendType::QNN_SDK, "QNN_SDK_CPU", false,
             kPlatAndroid, CreateQnnSdkBackend},
#endif
#ifdef HAVE_NCNN_BACKEND
            {BackendId::NCNN_CPU, BackendType::NCNN, "NCNN_CPU", true,
             kPlatWin | kPlatLinux | kPlatAndroid, CreateNcnnBackend},
            {BackendId::NCNN_CPU_FP16, BackendType::NCNN, "NCNN_CPU_FP16", false,
             kPlatWin | kPlatLinux | kPlatAndroid, CreateNcnnBackend},
            {BackendId::NCNN_CPU_BF16, BackendType::NCNN, "NCNN_CPU_BF16", false,
             kPlatWin | kPlatLinux | kPlatAndroid, CreateNcnnBackend},
            {BackendId::NCNN_VK, BackendType::NCNN, "NCNN_VK", false,
             kPlatWin | kPlatLinux | kPlatAndroid, CreateNcnnBackend},
            {BackendId::NCNN_VK_FP16, BackendType::NCNN, "NCNN_VK_FP16", false,
             kPlatWin | kPlatLinux | kPlatAndroid, CreateNcnnBackend},
            {BackendId::NCNN_VK_BF16, BackendType::NCNN, "NCNN_VK_BF16", false,
             kPlatWin | kPlatLinux | kPlatAndroid, CreateNcnnBackend},
#endif
#ifdef HAVE_MNN_BACKEND
            {BackendId::MNN_CPU, BackendType::MNN, "MNN_CPU", true,
             kPlatWin | kPlatLinux | kPlatAndroid, CreateMnnBackend},
            {BackendId::MNN_OPENCL, BackendType::MNN, "MNN_OpenCL", false,
             kPlatWin | kPlatLinux | kPlatAndroid, CreateMnnBackend},
            {BackendId::MNN_OPENCL_FP16, BackendType::MNN, "MNN_OpenCL_FP16", false,
             kPlatWin | kPlatLinux | kPlatAndroid, CreateMnnBackend},
            {BackendId::MNN_OPENCL_BF16, BackendType::MNN, "MNN_OpenCL_BF16", false,
             kPlatWin | kPlatLinux | kPlatAndroid, CreateMnnBackend},
            {BackendId::MNN_VULKAN, BackendType::MNN, "MNN_VULKAN", false,
             kPlatWin | kPlatLinux | kPlatAndroid, CreateMnnBackend},
            {BackendId::MNN_VULKAN_FP16, BackendType::MNN, "MNN_VULKAN_FP16", false,
             kPlatWin | kPlatLinux | kPlatAndroid, CreateMnnBackend},
            {BackendId::MNN_VULKAN_BF16, BackendType::MNN, "MNN_VULKAN_BF16", false,
             kPlatWin | kPlatLinux | kPlatAndroid, CreateMnnBackend},
            {BackendId::MNN_OPENGL, BackendType::MNN, "MNN_OPENGL", false,
             kPlatWin | kPlatLinux | kPlatAndroid, CreateMnnBackend},
            {BackendId::MNN_NN, BackendType::MNN, "MNN_NN", false,
             kPlatWin | kPlatLinux | kPlatAndroid, CreateMnnBackend},
#endif
        };

        size_t skipped = 0;
        for (const auto &d : decls) {
            if (!platform_supports(d.platforms)) {
                ++skipped;
                continue;
            }
            Register(d.id,
                     {d.id, d.type, d.name, "", d.is_cpu_baseline, d.platforms},
                     d.factory);
        }

        LOGI("Backend registry initialized with %zu backends (%zu not available on %s)",
             g_registry.size(), skipped, ARCH_STR);
        if (skipped > 0) {
            LOGD("Platforms of registered backends:");
            for (const auto &[id, entry] : g_registry) {
                LOGD("  %-20s %s", entry.config.name.c_str(),
                     platform_mask_str(entry.config.platforms).c_str());
            }
        }
    }

} // namespace BackendRegistry
