#pragma once
/*============================================================================
 * backend_interface.hpp - Abstract backend interface & registry
 *============================================================================*/

#include "platform.hpp"
#include "model_format.hpp"
#include <string>
#include <vector>
#include <array>
#include <memory>
#include <functional>

/* ---------------------------------------------------------------------------
 * BackendType & BackendId enums
 * -------------------------------------------------------------------------*/
enum class BackendType { ONNX_EP = 0, TFLITE_DEL = 1, NCNN = 2, MNN = 3, LITERT = 4 };   


/* ONNX Execution Providers:      0-15
 * TFLite Delegates:            100-103
 * NCNN backends:               200-202
 * MNN backends:                300-307
 */
enum class BackendId {
    /* ONNX */
    ONNX_CPU          = 0,
    ONNX_ONEDNN       = 1,
    ONNX_DML          = 2,
    ONNX_OPENVINO_CPU = 3,
    ONNX_OPENVINO_GPU = 4,
    ONNX_OPENVINO_NPU = 5,
    ONNX_CUDA         = 6,
    ONNX_TENSORRT     = 7,
    ONNX_NNAPI        = 8,
    ONNX_XNNPACK      = 9,
    ONNX_QNN_CPU      = 10,
    ONNX_QNN_GPU      = 11,
    ONNX_QNN_HTP      = 12,
    ONNX_LAST         = 13,

    /* TFLite */
    TFLITE_CPU         = 100,
    TFLITE_XNNPACK     = 101,
    TFLITE_NNAPI       = 102,
    TFLITE_GPU         = 103,
    TFLITE_NPU         = 104,

    /* NCNN */
    NCNN_CPU           = 200,
    NCNN_VULKAN        = 201,
    NCNN_VULKAN_FP16   = 202,

    /* MNN */
    MNN_CPU            = 300,
    MNN_OPENCL         = 301,
    MNN_OPENCL_FP16    = 302,
    MNN_OPENCL_BF16    = 303,
    MNN_VULKAN         = 304,
    MNN_VULKAN_FP16    = 305,
    MNN_VULKAN_BF16    = 306,
    MNN_OPENGL         = 307,

    /* LiteRT */
    LITERT_CPU         = 400,
    LITERT_GPU         = 401,
    LITERT_NPU         = 402,
};

inline int bid(BackendId id) { return static_cast<int>(id); }

/* ---------------------------------------------------------------------------
 * BackendConfig - describes a backend variant
 * -------------------------------------------------------------------------*/
struct BackendConfig {
    BackendId   id;
    BackendType type;
    std::string name;
    std::string output_suffix;
    bool        is_cpu_baseline = false;
};

/* ---------------------------------------------------------------------------
 * IBackend - abstract backend interface (pure virtual)
 * -------------------------------------------------------------------------*/
class IBackend {
public:
    virtual ~IBackend() = default;

    virtual bool Initialize(const char* model_path, int num_threads) = 0;

    virtual bool QueryIOInfo(std::string& input_shape, size_t& input_elems,
                             std::string& output_shape, size_t& output_elems) = 0;

    virtual bool PrepareInputs(float*& first_data, size_t& first_elems,
                               const char* input_arg, bool use_random,
                               const float* const* ext_data,
                               const size_t* ext_counts) = 0;

    virtual void SetSharedInput(const float* const* data,
                                const size_t* counts) = 0;

    virtual bool RunBenchmark(int warmup, int repeat,
                              double& total_ms, double& max_ms,
                              double& min_ms, int& max_idx,
                              std::vector<float*>& outputs,
                              std::vector<size_t>& output_elems,
                              std::vector<std::array<size_t, MAX_DIMENSIONS>>& output_shapes,
                              std::vector<size_t>& output_num_dims) = 0;

    virtual void GetTiming(std::array<double, 10>& timing) = 0;

    virtual bool SaveOutputs(const char* suffix) = 0;

    BackendId GetId() const { return id_; }
    void SetId(BackendId id) { id_ = id; }

protected:
    BackendId id_ = BackendId::ONNX_CPU;
};

using BackendPtr = std::unique_ptr<IBackend>;
using BackendFactory = std::function<BackendPtr(BackendId)>;

/* ---------------------------------------------------------------------------
 * Backend family checks
 * -------------------------------------------------------------------------*/
inline bool is_onnx_backend(BackendId id)   { int v=bid(id); return v>=0  && v<=15;  }
inline bool is_tflite_backend(BackendId id) { int v=bid(id); return v>=100&& v<=104; }
inline bool is_ncnn_backend(BackendId id)    { int v=bid(id); return v>=200&& v<=202; }
inline bool is_mnn_backend(BackendId id)     { int v=bid(id); return v>=300&& v<=307; }
inline bool is_litert_backend(BackendId id)  { int v=bid(id); return v>=400&& v<=402; }

/* ---------------------------------------------------------------------------
 * Registry - maps BackendId → BackendConfig + factory
 * -------------------------------------------------------------------------*/
namespace BackendRegistry {

/* Register a backend at static-init time */
void Register(BackendId id, BackendConfig cfg, BackendFactory factory);

/* Create a backend by ID (returns nullptr if not found) */
BackendPtr Create(BackendId id);

/* Get available backends for a model format */
std::vector<BackendConfig> GetAvailable(ModelFormat format);

/* Look up config by ID */
const BackendConfig* GetConfig(BackendId id);

/* Initialize all default registrations */
void InitDefaults();

} // namespace BackendRegistry
