#pragma once
/*============================================================================
 * backend_interface.hpp - Abstract backend interface & registry
 *============================================================================*/

#include "model_format.hpp"
#include "platform.hpp"
#include <array>
#include <functional>
#include <memory>
#include <string>
#include <vector>

/* ---------------------------------------------------------------------------
 * BackendType & BackendId enums
 * -------------------------------------------------------------------------*/
enum class BackendType { ONNX_EP = 0,
                         TFLITE_DEL = 1,
                         NCNN = 2,
                         MNN = 3,
                         LITERT = 4,
                         QNN_SDK = 5 };

/* ONNX Execution Providers:      0-17
 * TFLite Delegates:            100-107
 * NCNN backends:               200-206
 * MNN backends:                300-309
 * LiteRT backends:             400-405
 */
enum class BackendId {
    /* ONNX */
    ONNX_CPU = 0,
    ONNX_ONEDNN = 1,
    ONNX_DML_GPU = 2,
    // ONNX_DML_GPU_FP16 = 3, removed, same as ONNX_DML_GPU (DML auto-selects FP16 if supported)
    ONNX_DML_NPU = 4,
    ONNX_OPENVINO_CPU = 5,
    ONNX_OPENVINO_GPU = 6,
    ONNX_OPENVINO_GPU_FP16 = 7,
    /* 8 was ONNX_OPENVINO_GPU_BF16 - removed, unsupported by OpenVINO */
    ONNX_OPENVINO_NPU = 9,
    ONNX_CUDA = 10,
    ONNX_TENSORRT = 11,
    ONNX_NNAPI = 12,
    ONNX_XNNPACK = 13,
    ONNX_QNN_CPU = 14,
    ONNX_QNN_GPU = 15,
    ONNX_QNN_HTP = 16,
    ONNX_LAST = 17,

    /* TFLite */
    TFLITE_CPU = 100,
    TFLITE_XNNPACK = 101,
    TFLITE_XNNPACK_FP16 = 102,
    TFLITE_NNAPI = 103,
    TFLITE_GPU = 104,
    TFLITE_GPU_FP16 = 105,
    TFLITE_NPU = 106,
    TFLITE_LAST = 107,

    /* NCNN */
    NCNN_CPU = 200,
    NCNN_CPU_FP16 = 201,
    NCNN_CPU_BF16 = 202,
    NCNN_VK = 203,
    NCNN_VK_BF16 = 204,
    NCNN_VK_FP16 = 205,
    NCNN_LAST = 206,

    /* MNN */
    MNN_CPU = 300,
    MNN_OPENCL = 301,
    MNN_OPENCL_FP16 = 302,
    MNN_OPENCL_BF16 = 303,
    MNN_VULKAN = 304,
    MNN_VULKAN_FP16 = 305,
    MNN_VULKAN_BF16 = 306,
    MNN_OPENGL = 307,
    MNN_NN = 308,
    MNN_LAST = 309,

    /* LiteRT */
    LITERT_CPU = 400,
    LITERT_GPU = 401,
    LITERT_GPU_FP16 = 402,
    LITERT_NPU = 403,
    LITERT_NPU_FP16 = 404,
    LITERT_LAST = 405,

    /* QNN SDK (native QNN C API, context binary) */
    QNN_SDK_CPU = 500,
    QNN_SDK_GPU = 501,
    QNN_SDK_HTP = 502,
    QNN_SDK_LAST = 503,
};

inline int bid(BackendId id) { return static_cast<int>(id); }

/* ---------------------------------------------------------------------------
 * Platform availability mask
 *
 * Which platforms a backend is available on used to be encoded only in the
 * #if/#ifdef branches of BackendRegistry::InitDefaults(), which meant:
 *   - the README backend matrix had to be maintained by hand and drifted
 *   - "is backend X available here?" could not be answered at runtime
 *
 * BackendConfig now carries the mask as DATA, and InitDefaults() registers a
 * single list of declarations filtered by the mask. Compile-time gating
 * (HAVE_*_BACKEND, __ANDROID__) still applies on top - the mask only replaces
 * the hand-written per-platform duplication.
 * -------------------------------------------------------------------------*/
enum PlatformMask : unsigned {
    kPlatNone = 0,
    kPlatWin = 1u << 0,    /* Windows desktop (x86 / x64) */
    kPlatLinux = 1u << 1,  /* Linux desktop (x64) */
    kPlatAndroid = 1u << 2 /* Android (arm64) */
};

/* Mask of the platform this binary was built for. */
inline constexpr unsigned current_platform()
{
#if defined(__ANDROID__) || defined(__android__)
    return kPlatAndroid;
#elif defined(_WIN32)
    return kPlatWin;
#elif defined(__linux__)
    return kPlatLinux;
#else
    return kPlatNone;
#endif
}

inline bool platform_supports(unsigned mask) { return (mask & current_platform()) != 0; }

/* Short names used when printing an availability summary. */
inline std::string platform_mask_str(unsigned mask)
{
    std::string s;
    if (mask & kPlatWin) { s += "Win,"; }
    if (mask & kPlatLinux) { s += "Linux,"; }
    if (mask & kPlatAndroid) { s += "Android,"; }
    if (s.empty()) { return "-"; }
    s.pop_back(); /* trailing comma */
    return s;
}

/* ---------------------------------------------------------------------------
 * BackendConfig - describes a backend variant
 * -------------------------------------------------------------------------*/
struct BackendConfig {
    BackendId id;
    BackendType type;
    std::string name;
    std::string output_suffix;
    bool is_cpu_baseline = false;
    /* Platforms this backend is available on (PlatformMask bits).
     * kPlatNone = never registered on any platform (e.g. placeholder enum
     * values that were never implemented, like ONNX_CUDA). */
    unsigned platforms = kPlatNone;
};

/* ---------------------------------------------------------------------------
 * IBackend - abstract backend interface (pure virtual)
 * -------------------------------------------------------------------------*/
class IBackend
{
public:
    virtual ~IBackend() = default;

    virtual bool Initialize(const char *model_path, int num_threads) = 0;

    virtual bool QueryIOInfo(std::string &input_shape, size_t &input_elems,
                             std::string &output_shape, size_t &output_elems) = 0;

    virtual void SetSharedInput(const float *const *data,
                                const size_t *counts) = 0;

    virtual bool RunBenchmark(int warmup, int repeat,
                              double &total_ms, double &max_ms,
                              double &min_ms, int &max_idx,
                              std::vector<float *> &outputs,
                              std::vector<size_t> &output_elems,
                              std::vector<std::array<size_t, MAX_DIMENSIONS>> &output_shapes,
                              std::vector<size_t> &output_num_dims) = 0;

    virtual void GetTiming(std::array<double, 10> &timing) = 0;

    /**
     * Tensor transfer (CPU<->device) timing, average milliseconds per repeat.
     *
     * transfer_in_ms:  average time to move input tensors from the shared CPU
     *                  buffers to the device (H2D upload / host->device).
     * transfer_out_ms: average time to move output tensors back to CPU host
     *                  memory (D2H download / device->host, including the
     *                  memcpy into the snapshot buffers).
     *
     * Semantics differ per backend:
     *  - Backends with explicit upload/download steps (MNN GPU, TFLite,
     *    LiteRT, QNN SDK) measure the actual calls.
     *  - NCNN: extract() is the synchronous inference call itself and already
     *    includes the D2H copy internally; transfer_out only covers the
     *    explicit snapshot memcpy, and the D2H part stays inside avg_run_ms.
     *  - ONNX: inputs are wrapped zero-copy (CreateTensorWithDataAsOrtValue),
     *    so there is no separate H2D step; the synchronous OrtRun() already
     *    includes the device transfer for GPU EPs. transfer_in_ms stays 0
     *    and transfer_out_ms covers the post-run memcpy into snapshots.
     *  - CPU backends: both are ~0 (memcpy only, negligible).
     *
     * Both are averages over the repeat loop. Default no-op keeps existing
     * backends source-compatible.
     */
    virtual void GetTransferTiming(double &transfer_in_ms,
                                   double &transfer_out_ms)
    {
        transfer_in_ms = 0.0;
        transfer_out_ms = 0.0;
    }

    BackendId GetId() const { return id_; }
    void SetId(BackendId id) { id_ = id; }

    /** Return the last error message (set by backend on failure) */
    const char *GetLastError() const { return last_error_.c_str(); }

    /**
     * Output tensor names in the same order as RunBenchmark's outputs.
     * Enables name-based accuracy comparison: multi-output models can have a
     * different output ORDER across backends (e.g. QNN SDK model.so), so the
     * collector matches outputs by name instead of by position.
     * Returns an empty list by default -> position-based comparison.
     */
    virtual const std::vector<std::string> &GetOutputNames() const
    {
        static const std::vector<std::string> kEmpty;
        return kEmpty;
    }

protected:
    BackendId id_ = BackendId::ONNX_CPU;
    std::string last_error_;
};

using BackendPtr = std::unique_ptr<IBackend>;
using BackendFactory = std::function<BackendPtr(BackendId)>;

/* ---------------------------------------------------------------------------
 * Backend family checks
 * -------------------------------------------------------------------------*/
inline bool is_onnx_backend(BackendId id)
{
    int v = bid(id);
    return v >= bid(BackendId::ONNX_CPU) && v <= bid(BackendId::ONNX_LAST);
}
inline bool is_tflite_backend(BackendId id)
{
    int v = bid(id);
    return v >= bid(BackendId::TFLITE_CPU) && v <= bid(BackendId::TFLITE_LAST);
}
inline bool is_ncnn_backend(BackendId id)
{
    int v = bid(id);
    return v >= bid(BackendId::NCNN_CPU) && v <= bid(BackendId::NCNN_LAST);
}
inline bool is_mnn_backend(BackendId id)
{
    int v = bid(id);
    return v >= bid(BackendId::MNN_CPU) && v <= bid(BackendId::MNN_LAST);
}
inline bool is_litert_backend(BackendId id)
{
    int v = bid(id);
    return v >= bid(BackendId::LITERT_CPU) && v <= bid(BackendId::LITERT_LAST);
}
inline bool is_qnn_sdk_backend(BackendId id)
{
    int v = bid(id);
    return v >= bid(BackendId::QNN_SDK_CPU) && v <= bid(BackendId::QNN_SDK_LAST);
}

/* ---------------------------------------------------------------------------
 * Registry - maps BackendId -> BackendConfig + factory
 * -------------------------------------------------------------------------*/
namespace BackendRegistry
{

    /* Register a backend at static-init time */
    void Register(BackendId id, BackendConfig cfg, BackendFactory factory);

    /* Create a backend by ID (returns nullptr if not found) */
    BackendPtr Create(BackendId id);

    /* Get available backends for a model format */
    std::vector<BackendConfig> GetAvailable(ModelFormat format);

    /* Look up config by ID */
    const BackendConfig *GetConfig(BackendId id);

    /* Find backend ID by name (case-insensitive), returns -1 if not found */
    int FindByName(const char *name);

    /* Initialize all default registrations */
    void InitDefaults();

} // namespace BackendRegistry
