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
                         LITERT = 4 };

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
    ONNX_DML_GPU_FP16 = 3,
    ONNX_DML_NPU = 4,
    ONNX_OPENVINO_CPU = 5,
    ONNX_OPENVINO_GPU = 6,
    ONNX_OPENVINO_GPU_FP16 = 7,
    /* 8 was ONNX_OPENVINO_GPU_BF16 — removed, unsupported by OpenVINO */
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
};

inline int bid(BackendId id) { return static_cast<int>(id); }

/* ---------------------------------------------------------------------------
 * BackendConfig - describes a backend variant
 * -------------------------------------------------------------------------*/
struct BackendConfig {
    BackendId id;
    BackendType type;
    std::string name;
    std::string output_suffix;
    bool is_cpu_baseline = false;
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

    virtual bool PrepareInputs(float *&first_data, size_t &first_elems,
                               const char *input_arg, bool use_random,
                               const float *const *ext_data,
                               const size_t *ext_counts) = 0;

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

    virtual bool SaveOutputs(const char *suffix) = 0;

    BackendId GetId() const { return id_; }
    void SetId(BackendId id) { id_ = id; }

    /** Return the last error message (set by backend on failure) */
    const char *GetLastError() const { return last_error_.c_str(); }

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

/* ---------------------------------------------------------------------------
 * Registry - maps BackendId → BackendConfig + factory
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
