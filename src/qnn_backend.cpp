/*============================================================================
 * qnn_backend.cpp - Native QNN SDK backend (context binary + shared buffer)
 *
 * Loads a QNN context binary and executes it directly through the
 * QNN C API, mirroring the flow of the SDK's SampleAppSharedBuffer example:
 *
 *   dlopen(libQnn<Backend>.so) -> QnnInterface_getProviders -> QnnInterface_t
 *     -> backendCreate -> deviceCreate -> contextCreateFromBinary
 *     -> graphRetrieve -> graphExecute
 *
 * Model format: QNN context binary (generated offline by
 *   qnn-context-binary-generator from a DLC).
 *
 * Buffers: on Android, a DMA-BUF is allocated directly via /dev/dma_heap/system
 * (no libcdsprpc / fastrpc involved) and registered via QnnMem_register for
 * zero-copy I/O; QNN hands the fd to the DSP through its own channel. Otherwise
 * falls back to plain client buffers (cross-platform).
 *
 * NOTE: do NOT try to call rpcmem_* from libcdsprpc.so directly - on some
 * devices (SM8550) the system libcdsprpc's rpcmem_alloc crashes in the shell
 * domain (its internal remote_register_buf path is SELinux-blocked). qnn-net-run
 * --shared_buffer works because QNN allocates/registers the buffer internally.
 *============================================================================*/

#include "backend_interface.hpp"
#include "log.hpp"

#ifdef HAVE_QNN_SDK_BACKEND

#include <QNN/HTP/QnnHtpDevice.h>
#include <QNN/HTP/QnnHtpGraph.h>
#include <QNN/HTP/QnnHtpMem.h>
#include <QNN/HTP/QnnHtpPerfInfrastructure.h>
#include <QNN/QnnBackend.h>
#include <QNN/QnnCommon.h>
#include <QNN/QnnContext.h>
#include <QNN/QnnDevice.h>
#include <QNN/QnnGraph.h>
#include <QNN/QnnInterface.h>
#include <QNN/QnnLog.h>
#include <QNN/QnnMem.h>
#include <QNN/QnnTensor.h>
#include <QNN/QnnTypes.h>
#include <QNN/System/QnnSystemContext.h>
#include <QNN/System/QnnSystemInterface.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include <dlfcn.h>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

#if defined(__ANDROID__) || defined(__android__)
#include <cerrno>
#include <fcntl.h>
#include <linux/dma-buf.h>
#include <linux/dma-heap.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

/* ---------------------------------------------------------------------------
 * DMA-BUF cache sync (dma_heap/system is a cached heap; without this the CPU
 * and DSP can see stale data when exchanging buffers).  No-op on uncached heaps.
 * -------------------------------------------------------------------------*/
#if defined(__ANDROID__) || defined(__android__)
static void dma_buf_sync(int fd, uint32_t flags)
{
    struct dma_buf_sync sync = {};
    sync.flags = flags;
    if (ioctl(fd, DMA_BUF_IOCTL_SYNC, &sync) != 0) {
        static bool warned = false;
        if (!warned) {
            LOGE("QNN: dma_buf_sync(0x%X) failed on fd %d, due to %s, %d",
                 flags, fd, strerror(errno), errno);
            warned = true;
        }
    }
}
#endif

/* ---------------------------------------------------------------------------
 * model.so (QnnModel_composeGraphs) wrapper types.
 * QnnModel_freeGraphsInfo and is compiled per-target at runtime, so the SAME
 * .so can drive CPU / GPU / HTP backends. Mirrors SampleApp's QnnWrapperUtils.
 * -------------------------------------------------------------------------*/
namespace qnn_wrapper_api
{
    typedef enum ModelError {
        MODEL_NO_ERROR = 0,
        MODEL_TENSOR_ERROR = 1,
        MODEL_PARAMS_ERROR = 2,
        MODEL_NODES_ERROR = 3,
        MODEL_GRAPH_ERROR = 4,
        MODEL_CONTEXT_ERROR = 5,
        MODEL_GENERATION_ERROR = 6,
        MODEL_SETUP_ERROR = 7,
        MODEL_INVALID_ARGUMENT_ERROR = 8,
        MODEL_FILE_ERROR = 9,
        MODEL_MEMORY_ALLOCATE_ERROR = 10,
        MODEL_UNKNOWN_ERROR = 0x7FFFFFFF
    } ModelError_t;
    typedef struct GraphInfo {
        Qnn_GraphHandle_t graph;
        char *graphName;
        Qnn_Tensor_t *inputTensors;
        uint32_t numInputTensors;
        Qnn_Tensor_t *outputTensors;
        uint32_t numOutputTensors;
    } GraphInfo_t;
    typedef GraphInfo_t *GraphInfoPtr_t;
    typedef struct GraphConfigInfo {
        char *graphName;
        const QnnGraph_Config_t **graphConfigs;
    } GraphConfigInfo_t;
} /* namespace qnn_wrapper_api */

typedef qnn_wrapper_api::ModelError_t (*QnnComposeGraphsFn_t)(
    Qnn_BackendHandle_t, QNN_INTERFACE_VER_TYPE, Qnn_ContextHandle_t,
    const qnn_wrapper_api::GraphConfigInfo_t **, const uint32_t,
    qnn_wrapper_api::GraphInfo_t ***, uint32_t *, bool,
    QnnLog_Callback_t, QnnLog_Level_t);
typedef qnn_wrapper_api::ModelError_t (*QnnFreeGraphInfoFn_t)(
    qnn_wrapper_api::GraphInfo_t ***, uint32_t);

/* QNN backend log sink - surfaces backend-side errors (e.g. HTP graphCreate). */
static void qnn_log_callback(const char *fmt, QnnLog_Level_t level,
                             uint64_t /*timestamp*/, va_list args)
{
    fprintf(stderr, "[QNN_LOG:%d] ", (int)level);
    vfprintf(stderr, fmt ? fmt : "", args);
    fprintf(stderr, "\n");
}

/* ---------------------------------------------------------------------------
 * IEEE 754 binary16 <-> binary32 conversion (QNN FP16 tensors).
 * QNN 2.48 SDK ships no QnnConvert.h helper, so implement it locally with
 * round-to-nearest-even. Pure integer/bit ops - portable across platforms.
 * -------------------------------------------------------------------------*/
static uint16_t float_to_half(float f)
{
    union {
        float f;
        uint32_t u;
    } x;
    x.f = f;
    uint32_t u = x.u;
    uint32_t sign = (u >> 16) & 0x8000u;
    int32_t exp = (int32_t)((u >> 23) & 0xFFu) - 127 + 15;
    uint32_t mant = u & 0x7FFFFFu;

    if (exp >= 31) { /* overflow (or inf/nan) -> inf */
        return (uint16_t)(sign | 0x7C00u);
    }
    if (exp <= 0) {
        if (exp < -10) { /* underflow to zero */
            return (uint16_t)sign;
        }
        /* subnormal: keep leading 1, shift right, round to nearest even */
        mant |= 0x800000u;
        uint32_t shift = (uint32_t)(14 - exp);
        uint32_t half = mant >> shift;
        uint32_t rem = mant & ((1u << shift) - 1u);
        if (rem > (1u << (shift - 1)) || (rem == (1u << (shift - 1)) && (half & 1u))) {
            ++half;
        }
        return (uint16_t)(sign | half);
    }
    /* normal: drop 13 low mantissa bits, round to nearest even */
    uint32_t half = ((uint32_t)exp << 10) | (mant >> 13);
    uint32_t rem = mant & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (half & 1u))) {
        ++half;
        if (half == 0x7C00u) { /* rounded up to inf */
            half = 0x7C00u;
        }
    }
    return (uint16_t)(sign | half);
}

static float half_to_float(uint16_t h)
{
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x3FFu;
    uint32_t u;
    if (exp == 0) {
        if (mant == 0) {
            u = sign; /* +/- 0 */
        } else {
            /* subnormal */
            int32_t e = -1;
            uint32_t m = mant;
            do {
                ++e;
                m <<= 1;
            } while ((m & 0x400u) == 0);
            u = sign | ((uint32_t)(127 - 15 - e) << 23) | ((m & 0x3FFu) << 13);
        }
    } else if (exp == 31) {
        u = sign | 0x7F800000u | (mant << 13); /* inf / nan */
    } else {
        u = sign | ((uint32_t)(exp - 15 + 127) << 23) | (mant << 13);
    }
    union {
        float f;
        uint32_t u;
    } out;
    out.u = u;
    return out.f;
}

/* ---------------------------------------------------------------------------
 * QnnSdkBackend
 * -------------------------------------------------------------------------*/
class QnnSdkBackend : public IBackend
{
public:
    explicit QnnSdkBackend(BackendId id) { id_ = id; }
    ~QnnSdkBackend() override { Cleanup(); }

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
    const std::vector<std::string> &GetOutputNames() const override
    {
        return out_names_;
    }

private:
    void Cleanup();

    bool LoadContextBinary(const char *path);
    bool LoadModelLibrary(const char *path);
    bool CreateBackendDeviceContext();
    bool SetupTensorsFromBinaryInfo();
    bool SetupTensorsFromModelSo();
    bool AllocateBuffers();
    bool ConfigureHtpPerformance();
    void PopulateInput(int idx, const float *data, size_t n);
    bool ReadOutput(int idx, float *dst, size_t n);

    /* Library handles */
    void *lib_backend_ = nullptr; /* libQnnHtp.so / libQnnGpu.so / libQnnCpu.so */
    void *lib_system_ = nullptr;  /* libQnnSystem.so */

    /* model.so support (QnnModel_composeGraphs) */
    bool is_model_so_ = false;
    std::string model_path_; /* original model path (graph name derivation) */
    void *lib_model_ = nullptr;
    QnnComposeGraphsFn_t compose_graphs_ = nullptr;
    QnnFreeGraphInfoFn_t free_graphs_info_ = nullptr;
    qnn_wrapper_api::GraphInfo_t **graphs_info_ = nullptr;
    uint32_t graphs_count_ = 0;

    /* QNN interfaces & handles (QNN 2.48: versioned structs) */
    const QNN_INTERFACE_VER_TYPE *qnn_ = nullptr;
    const QNN_SYSTEM_INTERFACE_VER_TYPE *sys_ = nullptr;
    Qnn_BackendHandle_t backend_ = nullptr;
    Qnn_DeviceHandle_t device_ = nullptr;
    Qnn_ContextHandle_t context_ = nullptr;
    Qnn_GraphHandle_t graph_ = nullptr;
    Qnn_LogHandle_t log_ = nullptr;

    /* Model binary */
    std::vector<uint8_t> binary_;

    /* Tensor metadata from context binary */
    std::vector<Qnn_Tensor_t> in_tensors_;
    std::vector<Qnn_Tensor_t> out_tensors_;
    std::vector<std::vector<uint32_t>> in_dims_;
    std::vector<std::vector<uint32_t>> out_dims_;
    std::vector<std::string> in_names_; /* owned copies of tensor names */
    std::vector<std::string> out_names_;
    std::vector<size_t> in_elems_;
    std::vector<size_t> out_elems_;
    std::vector<Qnn_DataType_t> in_dtypes_;
    std::vector<Qnn_DataType_t> out_dtypes_;
    std::vector<Qnn_ScaleOffset_t> in_quant_;
    std::vector<Qnn_ScaleOffset_t> out_quant_;
    std::vector<bool> in_is_quant_;
    std::vector<bool> out_is_quant_;

    /* Buffers (shared dma-buf or client) */
    std::vector<void *> in_bufs_;
    std::vector<void *> out_bufs_;
    std::vector<Qnn_MemHandle_t> in_mem_handles_;
    std::vector<Qnn_MemHandle_t> out_mem_handles_;
    std::vector<bool> in_is_shared_;
    std::vector<bool> out_is_shared_;

    /* DMA-BUF descriptors for shared buffers: ONE combined buffer per
     * direction (in/out), each tensor registered at its own 4K-aligned offset
     * into the same fd. Cleaned up with munmap + close. */
    struct DmaBuf {
        int fd = -1;
        size_t size = 0;
        void *addr = nullptr;
    };
    std::vector<DmaBuf> in_dma_;  /* 0 or 1 element (combined input buffer) */
    std::vector<DmaBuf> out_dma_; /* 0 or 1 element (combined output buffer) */
    std::vector<size_t> in_offsets_; /* byte offset of each input inside in_dma_[0] */
    std::vector<size_t> out_offsets_;

    /* HTP performance infrastructure (DCVS v3 power vote, ~ORT burst mode) */
    uint32_t power_config_id_ = 0;
    QnnHtpPerfInfrastructure_DestroyPowerConfigIdFn_t destroy_power_config_ = nullptr;

    /* Inputs from runner (float, shared across backends) */
    std::vector<float *> input_bufs_;
    std::vector<size_t> input_buf_elems_;
    std::vector<bool> input_external_;

    size_t num_inputs_ = 0;
    size_t num_outputs_ = 0;

    int num_threads_ = 4; /* CPU threads for input conversion */

    double init_ms_ = 0;
};

/* ---------------------------------------------------------------------------
 * Helpers
 * -------------------------------------------------------------------------*/
static const char *qnn_backend_lib(BackendId id)
{
    switch (id) {
    case BackendId::QNN_SDK_GPU: {
        return "libQnnGpu.so";
    }
    case BackendId::QNN_SDK_CPU: {
        return "libQnnCpu.so";
    }
    case BackendId::QNN_SDK_HTP:
    default: {
        return "libQnnHtp.so";
    }
    }
}

static const char *qnn_backend_type(BackendId id)
{
    switch (id) {
    case BackendId::QNN_SDK_GPU: {
        return "gpu";
    }
    case BackendId::QNN_SDK_CPU: {
        return "cpu";
    }
    case BackendId::QNN_SDK_HTP:
    default: {
        return "htp";
    }
    }
}

/* QNN 2.48 tensor accessors: fields id/name/type/dataFormat/dataType/
 * quantizeParams/rank/dimensions/memType/clientBuf are identical in v1 & v2. */
static inline Qnn_TensorV1_t *qnn_tensor_v1(Qnn_Tensor_t *t)
{
    return &t->v1;
}
static inline const Qnn_TensorV1_t *qnn_tensor_v1(const Qnn_Tensor_t *t)
{
    return &t->v1;
}

/* Unified view over QnnSystemContext_GraphInfo_t (V1/V2/V3 share the same
 * field names for name / inputs / outputs). */
struct QnnGraphView {
    const char *name = nullptr;
    uint32_t numInputs = 0;
    const Qnn_Tensor_t *inputs = nullptr;
    uint32_t numOutputs = 0;
    const Qnn_Tensor_t *outputs = nullptr;
};
static QnnGraphView qnn_graph_view(const QnnSystemContext_GraphInfo_t *g)
{
    QnnGraphView v;
    if (!g) {
        return v;
    }
    switch (g->version) {
    case QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_2: {
        v.name = g->graphInfoV2.graphName;
        v.numInputs = g->graphInfoV2.numGraphInputs;
        v.inputs = g->graphInfoV2.graphInputs;
        v.numOutputs = g->graphInfoV2.numGraphOutputs;
        v.outputs = g->graphInfoV2.graphOutputs;
        break;
    }
    case QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_3: {
        v.name = g->graphInfoV3.graphName;
        v.numInputs = g->graphInfoV3.numGraphInputs;
        v.inputs = g->graphInfoV3.graphInputs;
        v.numOutputs = g->graphInfoV3.numGraphOutputs;
        v.outputs = g->graphInfoV3.graphOutputs;
        break;
    }
    default: { /* V1 */
        v.name = g->graphInfoV1.graphName;
        v.numInputs = g->graphInfoV1.numGraphInputs;
        v.inputs = g->graphInfoV1.graphInputs;
        v.numOutputs = g->graphInfoV1.numGraphOutputs;
        v.outputs = g->graphInfoV1.graphOutputs;
        break;
    }
    }
    return v;
}

/* Fetch graphs + count from a BinaryInfo of any version (V1/V2/V3). */
static void qnn_binary_graphs(const QnnSystemContext_BinaryInfo_t *b,
                              const QnnSystemContext_GraphInfo_t **graphs,
                              uint32_t *num_graphs)
{
    if (!b || !graphs || !num_graphs) {
        return;
    }
    switch (b->version) {
    case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1: {
        *graphs = b->contextBinaryInfoV1.graphs;
        *num_graphs = b->contextBinaryInfoV1.numGraphs;
        break;
    }
    case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2: {
        *graphs = b->contextBinaryInfoV2.graphs;
        *num_graphs = b->contextBinaryInfoV2.numGraphs;
        break;
    }
    case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_3: {
        *graphs = b->contextBinaryInfoV3.graphs;
        *num_graphs = b->contextBinaryInfoV3.numGraphs;
        break;
    }
    default: {
        *graphs = nullptr;
        *num_graphs = 0;
        break;
    }
    }
}

/* Copy tensor metadata into owned storage (dims/name) and force V1 layout.
 * Used by both the context-binary path and the model.so path. */
static void qnn_setup_tensor(Qnn_Tensor_t *dst, const Qnn_Tensor_t *src,
                             std::vector<std::vector<uint32_t>> &dims,
                             std::vector<std::string> &names,
                             std::vector<size_t> &elems,
                             std::vector<Qnn_DataType_t> &dtypes,
                             std::vector<Qnn_ScaleOffset_t> &quant,
                             std::vector<bool> &is_quant)
{
    const Qnn_TensorV1_t *s = qnn_tensor_v1(src);
    Qnn_TensorV1_t *d = qnn_tensor_v1(dst);

    dst->version = QNN_TENSOR_VERSION_1;
    d->id = s->id;
    d->name = nullptr;
    d->type = s->type;
    d->dataFormat = s->dataFormat;
    d->dataType = s->dataType;
    d->quantizeParams = s->quantizeParams;
    d->rank = s->rank;
    d->dimensions = nullptr;
    d->memType = QNN_TENSORMEMTYPE_UNDEFINED;
    d->clientBuf.data = nullptr;
    d->clientBuf.dataSize = 0;

    /* Owned copies (source tensors are freed by backend/model later) */
    dims.push_back(std::vector<uint32_t>(s->dimensions, s->dimensions + s->rank));
    d->dimensions = dims.back().data();
    names.emplace_back(s->name ? s->name : "");
    d->name = names.back().c_str();

    size_t n = 1;
    for (auto dd : dims.back()) {
        n *= (size_t)dd;
    }
    elems.push_back(n);
    dtypes.push_back(s->dataType);
    bool q = (s->quantizeParams.encodingDefinition ==
              (Qnn_Definition_t)QNN_QUANTIZATION_ENCODING_SCALE_OFFSET);
    is_quant.push_back(q);
    quant.push_back(q ? s->quantizeParams.scaleOffsetEncoding
                      : Qnn_ScaleOffset_t{1.0f, 0});
}

/* ELF64 shared object -> model.so (composeGraphs) instead of raw context binary */
static bool is_elf_shared_object(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) {
        return false;
    }
    unsigned char h[5] = {0};
    size_t n = fread(h, 1, 5, f);
    fclose(f);
    return n == 5 && h[0] == 0x7f && h[1] == 'E' && h[2] == 'L' &&
           h[3] == 'F' && h[4] == 2;
}

/* ---------------------------------------------------------------------------
 * Load context binary into memory
 * -------------------------------------------------------------------------*/
bool QnnSdkBackend::LoadContextBinary(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) {
        LOGE("QNN: cannot open context binary: %s", path);
        last_error_ = "QNN: context binary not found";
        return false;
    }
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (size <= 0) {
        fclose(f);
        LOGE("QNN: empty context binary: %s", path);
        last_error_ = "QNN: empty context binary";
        return false;
    }
    binary_.resize((size_t)size);
    if (fread(binary_.data(), 1, (size_t)size, f) != (size_t)size) {
        fclose(f);
        LOGE("QNN: short read on context binary");
        return false;
    }
    fclose(f);
    LOGI("QNN: loaded context binary %s (%ld bytes)", path, size);
    return true;
}

/* ---------------------------------------------------------------------------
 * Load a QNN model library (.so) exporting QnnModel_composeGraphs.
 * The same model.so can be compiled for any backend (CPU/GPU/HTP).
 * -------------------------------------------------------------------------*/
bool QnnSdkBackend::LoadModelLibrary(const char *path)
{
    lib_model_ = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!lib_model_) {
        LOGE("QNN: dlopen(model.so) failed: %s", dlerror());
        last_error_ = std::string("QNN: cannot load model library ") + path;
        return false;
    }
    void *sym_compose = dlsym(lib_model_, "QnnModel_composeGraphs");
    void *sym_free = dlsym(lib_model_, "QnnModel_freeGraphsInfo");
    memcpy(&compose_graphs_, &sym_compose, sizeof(sym_compose));
    memcpy(&free_graphs_info_, &sym_free, sizeof(sym_free));
    if (!compose_graphs_ || !free_graphs_info_) {
        LOGE("QNN: model.so missing QnnModel_composeGraphs/freeGraphsInfo");
        last_error_ = "QNN: model.so missing composeGraphs symbols";
        return false;
    }
    LOGI("QNN: loaded model library %s", path);
    return true;
}

/* ---------------------------------------------------------------------------
 * Load QNN backend + system libraries, create backend/device/context/graph
 * -------------------------------------------------------------------------*/
bool QnnSdkBackend::CreateBackendDeviceContext()
{
    /* 1. Load backend library and get QnnInterface */
    lib_backend_ = dlopen(qnn_backend_lib(id_), RTLD_NOW | RTLD_GLOBAL);
    if (!lib_backend_) {
        LOGE("QNN: dlopen(%s) failed: %s", qnn_backend_lib(id_), dlerror());
        last_error_ = std::string("QNN: cannot load ") + qnn_backend_lib(id_);
        return false;
    }
    auto get_providers = (Qnn_ErrorHandle_t (*)(const QnnInterface_t ***, uint32_t *))
        dlsym(lib_backend_, "QnnInterface_getProviders");
    if (!get_providers) {
        LOGE("QNN: QnnInterface_getProviders not found in %s", qnn_backend_lib(id_));
        return false;
    }
    const QnnInterface_t **providers = nullptr;
    uint32_t num_providers = 0;
    if (QNN_SUCCESS != get_providers(&providers, &num_providers) || !providers || !num_providers) {
        LOGE("QNN: failed to get interface providers");
        return false;
    }
    bool found = false;
    for (uint32_t i = 0; i < num_providers; ++i) {
        if (providers[i] &&
            providers[i]->apiVersion.coreApiVersion.major == QNN_API_VERSION_MAJOR &&
            providers[i]->apiVersion.coreApiVersion.minor >= QNN_API_VERSION_MINOR) {
            qnn_ = &providers[i]->QNN_INTERFACE_VER_NAME;
            found = true;
            break;
        }
    }
    if (!found) {
        LOGE("QNN: no compatible interface provider in %s", qnn_backend_lib(id_));
        return false;
    }
    LOGI("QNN: loaded backend %s", qnn_backend_lib(id_));

    /* 2. Create log handle (best effort) - errors only, keeps output clean */
    if (qnn_->logCreate) {
        (void)qnn_->logCreate(qnn_log_callback, QNN_LOG_LEVEL_ERROR, &log_);
    }

    /* 3. Create backend */
    if (!qnn_->backendCreate ||
        QNN_BACKEND_NO_ERROR != qnn_->backendCreate(log_, nullptr, &backend_)) {
        LOGE("QNN: backendCreate failed");
        return false;
    }
    LOGI("QNN: backend created (%s)", qnn_backend_lib(id_));

    /* 3b. Report detailed backend version info: core + backend-specific API
     * version and the build id string (e.g. "qaisw-v2.48.0.260626120635"). */
    Qnn_ApiVersion_t api_ver = QNN_API_VERSION_INIT;
    if (qnn_->backendGetApiVersion &&
        QNN_SUCCESS == qnn_->backendGetApiVersion(&api_ver)) {
        LOGI("QNN: backend API version: core %u.%u.%u, backend %u.%u.%u",
             api_ver.coreApiVersion.major, api_ver.coreApiVersion.minor,
             api_ver.coreApiVersion.patch, api_ver.backendApiVersion.major,
             api_ver.backendApiVersion.minor, api_ver.backendApiVersion.patch);
    }
    const char *build_id = nullptr;
    if (qnn_->backendGetBuildId &&
        QNN_SUCCESS == qnn_->backendGetBuildId(&build_id) && build_id) {
        LOGI("QNN: backend build id: %s", build_id);
    }

    /* 4. Create device. HTP requires a device; CPU has no device concept and
     * GPU device creation is best-effort, so only surface a failure for HTP
     * (where a missing device means the graph cannot run). */
    if (qnn_->deviceCreate) {
        const QnnDevice_Config_t *dev_configs[] = {nullptr};
        if (QNN_DEVICE_NO_ERROR != qnn_->deviceCreate(log_, dev_configs, &device_)) {
            if (id_ == BackendId::QNN_SDK_HTP) {
                LOGW("QNN: deviceCreate failed, continuing without device");
            }
            device_ = nullptr;
        } else {
            LOGI("QNN: device created");
        }
    }

    /* 5. model.so: create an empty context and compose the graph via
     * QnnModel_composeGraphs (works for any backend). */
    if (is_model_so_) {
        if (!qnn_->contextCreate ||
            QNN_CONTEXT_NO_ERROR != qnn_->contextCreate(backend_, device_, nullptr, &context_)) {
            LOGE("QNN: contextCreate failed (model.so)");
            return false;
        }
        LOGI("QNN: context created (model.so)");

        /* HTP graph-level configs for runtime compose (mirrors the offline
         * htp_config JSON: O-level, VTCM, HVX threads, activation fusion).
         * num_cores is deliberately NOT set - the unsigned-PD runtime exposes
         * a single NSP, so multi-core has no effect (docs 5.24/5.25). Env
         * tunable for A/B: QNN_HTP_O (3), QNN_HTP_VTCM_MB (8),
         * QNN_HTP_HVX_THREADS (8), QNN_HTP_AAF (1). GPU/CPU backends keep the
         * default (no HTP-specific graph config). */
        const qnn_wrapper_api::GraphConfigInfo_t *gci_arr[2] = {nullptr, nullptr};
        const qnn_wrapper_api::GraphConfigInfo_t **gcis = nullptr;
        uint32_t num_gcis = 0;
        QnnHtpGraph_CustomConfig_t htp_cfg[4] = {};
        QnnGraph_Config_t gcfg[4] = {};
        const QnnGraph_Config_t *gcfg_ptrs[5] = {};
        std::string graph_name;
        qnn_wrapper_api::GraphConfigInfo_t gci = {};
        if (id_ == BackendId::QNN_SDK_HTP) {
            uint32_t o_level = 3;
            uint32_t vtcm_mb = 8;
            uint64_t hvx_threads = 8;
            bool aaf = true;
            const char *env = getenv("QNN_HTP_O");
            if (env && env[0] != '\0') {
                o_level = (uint32_t)atoi(env);
            }
            env = getenv("QNN_HTP_VTCM_MB");
            if (env && env[0] != '\0') {
                vtcm_mb = (uint32_t)atoi(env);
            }
            env = getenv("QNN_HTP_HVX_THREADS");
            if (env && env[0] != '\0') {
                hvx_threads = (uint64_t)atoi(env);
            }
            env = getenv("QNN_HTP_AAF");
            if (env && env[0] != '\0') {
                aaf = (atoi(env) != 0);
            }

            htp_cfg[0].option = QNN_HTP_GRAPH_CONFIG_OPTION_OPTIMIZATION;
            htp_cfg[0].optimizationOption.type =
                QNN_HTP_GRAPH_OPTIMIZATION_TYPE_FINALIZE_OPTIMIZATION_FLAG;
            htp_cfg[0].optimizationOption.floatValue = (float)o_level;
            htp_cfg[1].option = QNN_HTP_GRAPH_CONFIG_OPTION_VTCM_SIZE;
            htp_cfg[1].vtcmSizeInMB = vtcm_mb;
            htp_cfg[2].option = QNN_HTP_GRAPH_CONFIG_OPTION_NUM_HVX_THREADS;
            htp_cfg[2].numHvxThreads = hvx_threads;
            htp_cfg[3].option = QNN_HTP_GRAPH_CONFIG_OPTION_ADVANCED_ACTIVATION_FUSION;
            htp_cfg[3].advancedActivationFusion = aaf;

            for (uint32_t i = 0; i < 4; ++i) {
                gcfg[i].option = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
                gcfg[i].customConfig = &htp_cfg[i];
                gcfg_ptrs[i] = &gcfg[i];
            }
            gcfg_ptrs[4] = nullptr;

            /* Graph name = model file stem minus "lib" prefix (matches the
             * QnnModel_composeGraphs graph name, e.g. libfoo.so -> foo). */
            graph_name = model_path_;
            auto slash = graph_name.rfind('/');
            if (slash != std::string::npos) {
                graph_name = graph_name.substr(slash + 1);
            }
            if (graph_name.size() > 3 &&
                graph_name.compare(graph_name.size() - 3, 3, ".so") == 0) {
                graph_name = graph_name.substr(0, graph_name.size() - 3);
            }
            if (graph_name.size() > 3 && graph_name.compare(0, 3, "lib") == 0) {
                graph_name = graph_name.substr(3);
            }

            gci.graphName = const_cast<char *>(graph_name.c_str());
            gci.graphConfigs = gcfg_ptrs;
            gci_arr[0] = &gci;
            gcis = gci_arr;
            num_gcis = 1;
            LOGI("QNN: HTP graph config: O%u vtcm=%uMB hvx=%llu aaf=%d graph=%s",
                 o_level, vtcm_mb, (unsigned long long)hvx_threads, aaf ? 1 : 0,
                 graph_name.c_str());
        }

        qnn_wrapper_api::GraphInfo_t **gi = nullptr;
        uint32_t ng = 0;
        qnn_wrapper_api::ModelError_t mrc = compose_graphs_(
            backend_, *qnn_, context_, gcis, num_gcis, &gi, &ng,
            false, nullptr, QNN_LOG_LEVEL_INFO);
        LOGI("QNN: composeGraphs rc=%d graphs=%u", (int)mrc, ng);
        if (mrc != qnn_wrapper_api::MODEL_NO_ERROR || !gi || ng == 0) {
            LOGE("QNN: composeGraphs failed");
            last_error_ = "QNN: composeGraphs failed";
            return false;
        }
        graphs_info_ = gi;
        graphs_count_ = ng;
        graph_ = gi[0]->graph;
        LOGI("QNN: graph[0] name=%s",
             gi[0]->graphName ? gi[0]->graphName : "(null)");

        /* composeGraphs returns a non-finalized graph; finalize before execute */
        if (!qnn_->graphFinalize ||
            QNN_GRAPH_NO_ERROR != qnn_->graphFinalize(graph_, nullptr, nullptr)) {
            LOGE("QNN: graphFinalize failed");
            last_error_ = "QNN: graphFinalize failed";
            return false;
        }
        LOGI("QNN: graph finalized (model.so)");
        return true;
    }

    /* 5b. Load QNN system library for binary info */
    lib_system_ = dlopen("libQnnSystem.so", RTLD_NOW | RTLD_LOCAL);
    if (!lib_system_) {
        LOGE("QNN: dlopen(libQnnSystem.so) failed: %s", dlerror());
        return false;
    }
    auto get_sys_providers = (Qnn_ErrorHandle_t (*)(const QnnSystemInterface_t ***, uint32_t *))
        dlsym(lib_system_, "QnnSystemInterface_getProviders");
    if (!get_sys_providers) {
        LOGE("QNN: QnnSystemInterface_getProviders not found");
        return false;
    }
    const QnnSystemInterface_t **sys_providers = nullptr;
    uint32_t num_sys_providers = 0;
    if (QNN_SUCCESS != get_sys_providers(&sys_providers, &num_sys_providers) ||
        !sys_providers || !num_sys_providers) {
        LOGE("QNN: failed to get system interface providers");
        return false;
    }
    for (uint32_t i = 0; i < num_sys_providers; ++i) {
        if (sys_providers[i] &&
            sys_providers[i]->systemApiVersion.major == QNN_SYSTEM_API_VERSION_MAJOR &&
            sys_providers[i]->systemApiVersion.minor >= QNN_SYSTEM_API_VERSION_MINOR) {
            sys_ = &sys_providers[i]->QNN_SYSTEM_INTERFACE_VER_NAME;
            break;
        }
    }
    if (!sys_) {
        LOGE("QNN: no compatible system interface");
        return false;
    }

    /* 6. Restore context from binary */
    if (!qnn_->contextCreateFromBinary ||
        QNN_CONTEXT_NO_ERROR != qnn_->contextCreateFromBinary(
                                    backend_, device_, nullptr,
                                    binary_.data(), (Qnn_ContextBinarySize_t)binary_.size(),
                                    &context_, nullptr)) {
        LOGE("QNN: contextCreateFromBinary failed");
        last_error_ = "QNN: contextCreateFromBinary failed";
        return false;
    }
    LOGI("QNN: context restored from binary");

    /* 7. Retrieve graph by name from binary info */
    QnnSystemContext_Handle_t sys_ctx = nullptr;
    const QnnSystemContext_BinaryInfo_t *bin_info = nullptr;
    Qnn_ContextBinarySize_t info_size = 0;
    const char *graph_name = nullptr;
    Qnn_ErrorHandle_t bin_rc = QNN_SUCCESS;
    if (sys_ && sys_->systemContextCreate) {
        bin_rc = sys_->systemContextCreate(&sys_ctx);
        LOGI("QNN: systemContextCreate rc=%d", (int)bin_rc);
    }
    if (QNN_SUCCESS == bin_rc && sys_ && sys_->systemContextGetBinaryInfo) {
        bin_rc = sys_->systemContextGetBinaryInfo(
            sys_ctx, binary_.data(), (Qnn_ContextBinarySize_t)binary_.size(),
            &bin_info, &info_size);
        LOGI("QNN: systemContextGetBinaryInfo rc=%d size=%llu", (int)bin_rc,
             (unsigned long long)info_size);
    }
    if (QNN_SUCCESS == bin_rc && bin_info) {
        uint32_t num_graphs = 0;
        const QnnSystemContext_GraphInfo_t *graphs = nullptr;
        qnn_binary_graphs(bin_info, &graphs, &num_graphs);
        LOGI("QNN: binary info version=%u num_graphs=%u",
             (unsigned)bin_info->version, num_graphs);
        if (graphs && num_graphs > 0) {
            graph_name = qnn_graph_view(&graphs[0]).name;
            LOGI("QNN: binary graph[0] name: %s", graph_name ? graph_name : "(null)");
        }
    }
    if (sys_ && sys_ctx) {
        sys_->systemContextFree(sys_ctx);
    }

    if (!qnn_->graphRetrieve) {
        LOGE("QNN: graphRetrieve not available");
        return false;
    }
    Qnn_ErrorHandle_t grc = qnn_->graphRetrieve(context_, graph_name, &graph_);
    LOGI("QNN: graphRetrieve(name=%s) rc=%d", graph_name ? graph_name : "(null)", (int)grc);
    if (QNN_SUCCESS != grc) {
        LOGE("QNN: graphRetrieve failed");
        return false;
    }
    LOGI("QNN: graph retrieved");
    return true;
}

/* ---------------------------------------------------------------------------
 * Configure HTP performance mode (DCVS v3 power vote, ORT burst equivalent).
 *
 * Mirrors ONNX Runtime's QNN EP "burst" profile
 * (qnn_htp_power_config_manager.cc, HtpPerformanceMode::kHtpBurst):
 *   - dcvsEnable = 0: DISABLE DCVS dynamic voltage scaling so the voltage
 *     corners below are hard-pinned by the hardware. Keeping DCVS enabled
 *     (our earlier NOM->TURBO target vote) lets the system power manager
 *     (perfd / DCVS arbiter) re-scale the HTP down to a low corner after
 *     ~100-300ms, which is exactly the stair-step 6ms<->16ms seen in runs.
 *   - bus/core voltage corners Min/Target/Max = MAX_VOLTAGE_CORNER: lock the
 *     highest corner, so the system cannot lower it.
 *   - sleepDisable = 1 (CONTROL experiment 2026-08-06): force the HTP awake
 *     instead of ORT's sleepLatency=40us, to test whether the rare ~100ms
 *     single-frame spikes (run 796/806) are HTP sleep/wakeup. If the spikes
 *     disappear, HTP sleep was the cause; revert to sleepLatency=40us if
 *     keeping the ORT-identical config matters more than the spikes.
 * This is a one-shot vote at init - like ORT (which only re-applies when the
 * mode actually changes), no per-frame refresh is needed because DCVS is off.
 * Non-fatal: on failure the backend still runs at the default power profile.
 * -------------------------------------------------------------------------*/
bool QnnSdkBackend::ConfigureHtpPerformance()
{
    if (id_ != BackendId::QNN_SDK_HTP || !device_ || !qnn_->deviceGetInfrastructure) {
        return true; /* only HTP has a DCVS power vote */
    }
    /* deviceGetInfrastructure writes back a pointer to the HTP-specific
     * infrastructure (QnnHtpDevice_Infrastructure_t) through the generic
     * QnnDevice_Infrastructure_t* alias. */
    QnnHtpDevice_Infrastructure_t *infra = nullptr;
    if (QNN_SUCCESS != qnn_->deviceGetInfrastructure(&infra) || !infra) {
        LOGW("QNN: deviceGetInfrastructure failed - perf mode disabled");
        return false;
    }
    if (infra->infraType != QNN_HTP_DEVICE_INFRASTRUCTURE_TYPE_PERF ||
        !infra->perfInfra.createPowerConfigId || !infra->perfInfra.setPowerConfig ||
        !infra->perfInfra.destroyPowerConfigId) {
        LOGW("QNN: HTP perf infrastructure unavailable - perf mode disabled");
        return false;
    }

    uint32_t device_id = 0; /* QNN_DEVICE_DEFAULT_DEVICE_ID */
    uint32_t core_id = 0;   /* QNN_DEVICE_DEFAULT_CORE_ID */
    if (QNN_SUCCESS != infra->perfInfra.createPowerConfigId(device_id, core_id, &power_config_id_)) {
        LOGW("QNN: createPowerConfigId failed - perf mode disabled");
        return false;
    }
    destroy_power_config_ = infra->perfInfra.destroyPowerConfigId;

    QnnHtpPerfInfrastructure_PowerConfig_t cfg = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIG_INIT;
    cfg.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3;
    cfg.dcvsV3Config.contextId = power_config_id_;
    /* ORT burst: disable DCVS dynamic scaling and pin the highest corner */
    cfg.dcvsV3Config.setDcvsEnable = 1;
    cfg.dcvsV3Config.dcvsEnable = 0; /* kDcvsDisable: hardware-lock the corners */
    cfg.dcvsV3Config.powerMode = QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE;
    /* Control experiment: force awake (no HTP sleep/wakeup jitter) */
    cfg.dcvsV3Config.setSleepDisable = 1;
    cfg.dcvsV3Config.sleepDisable = 1;
    cfg.dcvsV3Config.setBusParams = 1;
    cfg.dcvsV3Config.busVoltageCornerMin = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
    cfg.dcvsV3Config.busVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
    cfg.dcvsV3Config.busVoltageCornerMax = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
    cfg.dcvsV3Config.setCoreParams = 1;
    cfg.dcvsV3Config.coreVoltageCornerMin = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
    cfg.dcvsV3Config.coreVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
    cfg.dcvsV3Config.coreVoltageCornerMax = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;

    const QnnHtpPerfInfrastructure_PowerConfig_t *configs[] = {&cfg, nullptr};
    Qnn_ErrorHandle_t rc = infra->perfInfra.setPowerConfig(power_config_id_, configs);
    if (QNN_SUCCESS != rc) {
        LOGE("QNN: setPowerConfig failed rc=0x%X - perf mode disabled", (unsigned)rc);
        if (destroy_power_config_) {
            destroy_power_config_(power_config_id_);
        }
        power_config_id_ = 0;
        destroy_power_config_ = nullptr;
        return false;
    }
    LOGI("QNN: HTP perf configured (DCVS_V3 burst: DCVS off, MAX corner, sleep disabled)");
    return true;
}

/* ---------------------------------------------------------------------------
 * Extract tensor metadata from binary info and build Qnn_Tensor_t list
 * -------------------------------------------------------------------------*/
bool QnnSdkBackend::SetupTensorsFromBinaryInfo()
{
    if (!sys_ || !sys_->systemContextCreate || !sys_->systemContextGetBinaryInfo) {
        LOGE("QNN: system context API not available");
        return false;
    }
    QnnSystemContext_Handle_t sys_ctx = nullptr;
    if (QNN_SUCCESS != sys_->systemContextCreate(&sys_ctx)) {
        LOGE("QNN: systemContextCreate failed");
        return false;
    }
    const QnnSystemContext_BinaryInfo_t *bin_info = nullptr;
    Qnn_ContextBinarySize_t info_size = 0;
    if (QNN_SUCCESS != sys_->systemContextGetBinaryInfo(
                           sys_ctx, binary_.data(), (Qnn_ContextBinarySize_t)binary_.size(),
                           &bin_info, &info_size)) {
        sys_->systemContextFree(sys_ctx);
        LOGE("QNN: systemContextGetBinaryInfo failed");
        return false;
    }

    const QnnSystemContext_GraphInfo_t *graph_info = nullptr;
    uint32_t num_graphs = 0;
    qnn_binary_graphs(bin_info, &graph_info, &num_graphs);
    if (!graph_info || num_graphs == 0) {
        sys_->systemContextFree(sys_ctx);
        LOGE("QNN: no graphs in binary info (version %d)", (int)bin_info->version);
        return false;
    }
    QnnGraphView g = qnn_graph_view(&graph_info[0]);

    for (uint32_t i = 0; i < g.numInputs; ++i) {
        Qnn_Tensor_t t = QNN_TENSOR_INIT;
        qnn_setup_tensor(&t, &g.inputs[i], in_dims_, in_names_, in_elems_, in_dtypes_,
                         in_quant_, in_is_quant_);
        in_tensors_.push_back(t);
    }
    for (uint32_t i = 0; i < g.numOutputs; ++i) {
        Qnn_Tensor_t t = QNN_TENSOR_INIT;
        qnn_setup_tensor(&t, &g.outputs[i], out_dims_, out_names_, out_elems_, out_dtypes_,
                         out_quant_, out_is_quant_);
        out_tensors_.push_back(t);
    }
    sys_->systemContextFree(sys_ctx);

    num_inputs_ = in_tensors_.size();
    num_outputs_ = out_tensors_.size();
    LOGI("QNN: binary info: %zu in, %zu out", num_inputs_, num_outputs_);
    return num_inputs_ > 0 && num_outputs_ > 0;
}

/* ---------------------------------------------------------------------------
 * Extract tensor metadata from a model.so composeGraphs result
 * -------------------------------------------------------------------------*/
bool QnnSdkBackend::SetupTensorsFromModelSo()
{
    if (!graphs_info_ || graphs_count_ == 0 || !graphs_info_[0]) {
        LOGE("QNN: no graphs info from model.so");
        return false;
    }
    qnn_wrapper_api::GraphInfo_t *gi = graphs_info_[0];

    for (uint32_t i = 0; i < gi->numInputTensors; ++i) {
        Qnn_Tensor_t t = QNN_TENSOR_INIT;
        qnn_setup_tensor(&t, &gi->inputTensors[i], in_dims_, in_names_, in_elems_,
                         in_dtypes_, in_quant_, in_is_quant_);
        in_tensors_.push_back(t);
    }
    for (uint32_t i = 0; i < gi->numOutputTensors; ++i) {
        Qnn_Tensor_t t = QNN_TENSOR_INIT;
        qnn_setup_tensor(&t, &gi->outputTensors[i], out_dims_, out_names_, out_elems_,
                         out_dtypes_, out_quant_, out_is_quant_);
        out_tensors_.push_back(t);
    }

    num_inputs_ = in_tensors_.size();
    num_outputs_ = out_tensors_.size();
    LOGI("QNN: model.so: %zu in, %zu out", num_inputs_, num_outputs_);
    return num_inputs_ > 0 && num_outputs_ > 0;
}

/* ---------------------------------------------------------------------------
 * Allocate buffers for all tensors.
 * Android: allocate DMA-BUF from /dev/dma_heap/system and register with
 * QnnMem_register for zero-copy I/O; fall back to client buffer.
 * -------------------------------------------------------------------------*/
bool QnnSdkBackend::AllocateBuffers()
{
    LOGI("QNN: AllocateBuffers: %zu in, %zu out", in_tensors_.size(), out_tensors_.size());
    /* dma-heap zero-copy is HTP-specific: the registered descriptor is a
     * QnnMemHtp_Descriptor_t, so CPU/GPU backends would always fail
     * memRegister. Skip the attempt for them to keep the log clean; the
     * "buffers: x/y shared" summary below still reports the result.
     * QNN_NO_DMA_HEAP=1 forces client buffers (debug switch for isolating
     * DMA/quantized-graph issues). */
    bool try_dma_heap = (id_ == BackendId::QNN_SDK_HTP);
    const char *no_dma = getenv("QNN_NO_DMA_HEAP");
    if (no_dma && no_dma[0] == '1') {
        try_dma_heap = false;
        LOGW("QNN: dma-heap disabled by QNN_NO_DMA_HEAP=1 - using client buffers");
    }
    bool dma_heap_warned = false;
    auto warn_dma_heap_once = [&](const char *msg) {
        if (!dma_heap_warned) {
            LOGW("QNN: %s - fallback client buffer", msg);
            dma_heap_warned = true;
        }
    };

    auto aligned_bytes = [](size_t bytes) -> size_t {
        return (bytes + 4095u) & ~(size_t)4095u;
    };
    auto tensor_bytes = [](size_t elems, Qnn_DataType_t dt) -> size_t {
        size_t item_size = 1; /* quantized 8-bit default */
        switch (dt) {
        case QNN_DATATYPE_FLOAT_32: {
            item_size = 4;
            break;
        }
        case QNN_DATATYPE_FLOAT_16:
        case QNN_DATATYPE_BFLOAT_16: {
            item_size = 2;
            break;
        }
        case QNN_DATATYPE_UFIXED_POINT_8:
        case QNN_DATATYPE_SFIXED_POINT_8: {
            item_size = 1;
            break;
        }
        case QNN_DATATYPE_UFIXED_POINT_16:
        case QNN_DATATYPE_SFIXED_POINT_16: {
            item_size = 2;
            break;
        }
        default: {
            break;
        }
        }
        return elems * item_size;
    };

    /* Lay all tensors of one direction into ONE combined dma-buf: each tensor
     * gets a 4K-aligned offset and its own QnnMem_register handle (fd+offset).
     * A single dma_buf_sync then covers the whole direction per inference
     * (1 ioctl instead of one per tensor), cutting per-run DMA overhead. */
    auto build_layout = [&](const std::vector<size_t> &elems,
                            const std::vector<Qnn_DataType_t> &dtypes,
                            std::vector<size_t> &offsets,
                            std::vector<size_t> &sizes) -> size_t {
        offsets.assign(elems.size(), 0);
        sizes.assign(elems.size(), 0);
        size_t total = 0;
        for (size_t i = 0; i < elems.size(); ++i) {
            size_t a = aligned_bytes(tensor_bytes(elems[i], dtypes[i]));
            sizes[i] = a;
            offsets[i] = total;
            total += a;
        }
        return total;
    };
    std::vector<size_t> in_sizes, out_sizes;
    size_t in_total = build_layout(in_elems_, in_dtypes_, in_offsets_, in_sizes);
    size_t out_total = build_layout(out_elems_, out_dtypes_, out_offsets_, out_sizes);

    auto dt_name = [](Qnn_DataType_t dt) -> const char * {
        switch (dt) {
        case QNN_DATATYPE_FLOAT_32: {
            return "F32";
        }
        case QNN_DATATYPE_FLOAT_16: {
            return "F16";
        }
        case QNN_DATATYPE_UFIXED_POINT_8: {
            return "QU8";
        }
        case QNN_DATATYPE_SFIXED_POINT_8: {
            return "QS8";
        }
        case QNN_DATATYPE_UFIXED_POINT_16: {
            return "QU16";
        }
        case QNN_DATATYPE_SFIXED_POINT_16: {
            return "QS16";
        }
        default: {
            return "OTHER";
        }
        }
    };
    /* Per-tensor IO details are DEBUG-only: 44 lines (22 in + 22 out) every
     * init is too noisy for INFO. Enable with --log-level DBG when diagnosing
     * quantized/FP16 graph data types. */
    for (size_t i = 0; i < in_tensors_.size(); ++i) {
        LOGD("QNN:  in[%zu] %s elems=%zu bytes=%zu quant=%d scale=%g off=%d",
             i, dt_name(in_dtypes_[i]), in_elems_[i], in_sizes[i],
             in_is_quant_[i] ? 1 : 0, in_quant_[i].scale, in_quant_[i].offset);
    }
    for (size_t i = 0; i < out_tensors_.size(); ++i) {
        LOGD("QNN: out[%zu] %s elems=%zu bytes=%zu quant=%d scale=%g off=%d",
             i, dt_name(out_dtypes_[i]), out_elems_[i], out_sizes[i],
             out_is_quant_[i] ? 1 : 0, out_quant_[i].scale, out_quant_[i].offset);
    }

    auto alloc_direction = [&](std::vector<Qnn_Tensor_t> &tensors,
                               const std::vector<size_t> &elems,
                               const std::vector<Qnn_DataType_t> &dtypes,
                               const std::vector<size_t> &offsets,
                               const std::vector<size_t> &sizes,
                               size_t total_bytes,
                               std::vector<void *> &bufs,
                               std::vector<Qnn_MemHandle_t> &mem,
                               std::vector<bool> &is_shared,
                               std::vector<DmaBuf> &dma) -> bool {
        void *base = nullptr;
        int buf_fd = -1;
        bool shared = false;
#if defined(__ANDROID__) || defined(__android__)
        if (try_dma_heap && !tensors.empty()) {
            /* One DMA-BUF from the system dma-heap. This needs no
             * libcdsprpc/fastrpc on the app side - QNN registers the fd with the
             * DSP through its own channel. (Direct rpcmem_* calls into the system
             * libcdsprpc crash in the shell domain on SM8550.) */
            size_t alloc_size = aligned_bytes(total_bytes > 0 ? total_bytes : 1);
            int dmafd = open("/dev/dma_heap/system", O_RDONLY | O_CLOEXEC);
            if (dmafd >= 0) {
                struct dma_heap_allocation_data dmabuf = {};
                dmabuf.len = alloc_size;
                dmabuf.fd_flags = O_RDWR | O_CLOEXEC;
                if (ioctl(dmafd, DMA_HEAP_IOCTL_ALLOC, &dmabuf) == 0) {
                    buf_fd = (int)dmabuf.fd;
                    void *addr = mmap(nullptr, alloc_size, PROT_READ | PROT_WRITE,
                                      MAP_SHARED, buf_fd, 0);
                    if (addr != MAP_FAILED) {
                        bool all_ok = true;
                        for (size_t i = 0; i < tensors.size(); ++i) {
                            /* HTP backend requires a CUSTOM memory descriptor with a
                             * QnnMemHtp_Descriptor_t of type QNN_HTP_MEM_SHARED_BUFFER.
                             * Using QNN_MEM_TYPE_DMA_BUF here crashes libQnnHtp (the
                             * backend reads the union as a custom descriptor). */
                            QnnMemHtp_Descriptor_t htp_desc = {};
                            htp_desc.type = QNN_HTP_MEM_SHARED_BUFFER;
                            /* QnnHtpMem.h: for QNN_HTP_MEM_SHARED_BUFFER the size
                             * field must be the TOTAL size of the whole shared
                             * buffer (all tensors in this direction), NOT the
                             * per-tensor size. Passing the per-tensor size makes
                             * QnnDsp reject every register ("fd ... already mapped
                             * with mismatched size" / "calculated buffer size ... is
                             * more than the actual buffer size") and the whole
                             * direction silently falls back to client buffers. */
                            htp_desc.size = (uint32_t)alloc_size;
                            htp_desc.sharedBufferConfig.fd = buf_fd;
                            htp_desc.sharedBufferConfig.offset = offsets[i];

                            Qnn_MemDescriptor_t desc = QNN_MEM_DESCRIPTOR_INIT;
                            desc.dataType = dtypes[i];
                            desc.memType = QNN_MEM_TYPE_CUSTOM;
                            desc.customInfo = &htp_desc;
                            Qnn_MemHandle_t h = nullptr;
                            if (qnn_->memRegister &&
                                QNN_SUCCESS == qnn_->memRegister(context_, &desc, 1, &h)) {
                                mem.push_back(h);
                            } else {
                                mem.push_back(nullptr);
                                all_ok = false;
                            }
                        }
                        if (all_ok) {
                            shared = true;
                            base = addr;
                            dma.push_back({buf_fd, alloc_size, addr});
                            buf_fd = -1; /* ownership moved into dma */
                        } else {
                            /* roll back partial registrations */
                            std::vector<Qnn_MemHandle_t> ok;
                            for (auto &h : mem) {
                                if (h) {
                                    ok.push_back(h);
                                }
                            }
                            if (qnn_->memDeRegister && !ok.empty()) {
                                qnn_->memDeRegister(ok.data(), (uint32_t)ok.size());
                            }
                            mem.clear();
                            munmap(addr, alloc_size);
                            close(buf_fd);
                            buf_fd = -1;
                            warn_dma_heap_once("dma-heap memRegister failed");
                        }
                    } else {
                        close(buf_fd);
                        buf_fd = -1;
                        warn_dma_heap_once("dma-heap mmap failed");
                    }
                } else {
                    warn_dma_heap_once("dma-heap DMA_HEAP_IOCTL_ALLOC failed");
                }
                close(dmafd);
            } else {
                warn_dma_heap_once("open /dev/dma_heap/system failed");
            }
        } /* if (try_dma_heap) */
#endif
        if (!shared) {
            for (size_t i = 0; i < tensors.size(); ++i) {
                mem.push_back(nullptr);
            }
            dma.push_back({}); /* empty marker: direction is client-backed */
        }
        for (size_t i = 0; i < tensors.size(); ++i) {
            size_t bytes = tensor_bytes(elems[i], dtypes[i]);
            void *p = shared ? (uint8_t *)base + offsets[i]
                             : malloc(bytes > 0 ? bytes : 1);
            if (!p) {
                LOGE("QNN: buffer alloc failed at index %zu", i);
                return false;
            }
            bufs.push_back(p);
            is_shared.push_back(shared);

            /* Attach buffer to tensor (QNN 2.48 direct struct access) */
            Qnn_TensorV1_t *tv = qnn_tensor_v1(&tensors[i]);
            if (shared) {
                tv->memType = QNN_TENSORMEMTYPE_MEMHANDLE;
                tv->memHandle = mem[i];
            } else {
                tv->memType = QNN_TENSORMEMTYPE_RAW;
                tv->clientBuf.data = p;
                tv->clientBuf.dataSize = bytes;
            }
        }
        return true;
    };

    if (!alloc_direction(in_tensors_, in_elems_, in_dtypes_, in_offsets_, in_sizes,
                         in_total, in_bufs_, in_mem_handles_, in_is_shared_, in_dma_)) {
        LOGE("QNN: input buffer allocation failed");
        return false;
    }
    if (!alloc_direction(out_tensors_, out_elems_, out_dtypes_, out_offsets_, out_sizes,
                         out_total, out_bufs_, out_mem_handles_, out_is_shared_, out_dma_)) {
        LOGE("QNN: output buffer allocation failed");
        return false;
    }

    /* Report whether zero-copy shared buffers actually engaged */
    size_t n_shared = 0, n_total = in_is_shared_.size() + out_is_shared_.size();
    for (bool s : in_is_shared_) {
        if (s) {
            ++n_shared;
        }
    }
    for (bool s : out_is_shared_) {
        if (s) {
            ++n_shared;
        }
    }
    LOGI("QNN: buffers: %zu/%zu shared (dma-heap zero-copy), rest client buffer",
         n_shared, n_total);
    return true;
}

/* ---------------------------------------------------------------------------
 * Initialize
 * -------------------------------------------------------------------------*/
bool QnnSdkBackend::Initialize(const char *model_path, int num_threads)
{
    auto t0 = std::chrono::high_resolution_clock::now();
    last_error_.clear();
    num_threads_ = num_threads > 0 ? num_threads : 4;
    model_path_ = model_path;

    /* ELF shared object -> model.so (composeGraphs), otherwise context binary */
    if (is_elf_shared_object(model_path)) {
        is_model_so_ = true;
        if (!LoadModelLibrary(model_path)) {
            return false;
        }
    } else {
        if (!LoadContextBinary(model_path)) {
            return false;
        }
    }
    if (!CreateBackendDeviceContext()) {
        return false;
    }
    if (is_model_so_) {
        if (!SetupTensorsFromModelSo()) {
            return false;
        }
    } else {
        if (!SetupTensorsFromBinaryInfo()) {
            return false;
        }
    }
    if (!AllocateBuffers()) {
        return false;
    }
    if (!ConfigureHtpPerformance()) {
        LOGW("QNN: HTP performance mode not configured (non-fatal)");
    }

    init_ms_ = std::chrono::duration<double, std::milli>(
                   std::chrono::high_resolution_clock::now() - t0)
                   .count();
    LOGI("QNN: init complete (%.1f ms), %zu in, %zu out, backend=%s",
         init_ms_, num_inputs_, num_outputs_, qnn_backend_type(id_));
    return true;
}

/* ---------------------------------------------------------------------------
 * QueryIOInfo
 * -------------------------------------------------------------------------*/
bool QnnSdkBackend::QueryIOInfo(std::string &is, size_t &ie,
                                std::string &os, size_t &oe)
{
    is.clear();
    ie = 0;
    for (size_t i = 0; i < num_inputs_; ++i) {
        char buf[128] = {};
        int off = snprintf(buf, sizeof(buf), "[");
        for (size_t d = 0; d < in_dims_[i].size(); ++d) {
            off += snprintf(buf + off, sizeof(buf) - off, "%s%u", d > 0 ? "," : "",
                            in_dims_[i][d]);
        }
        snprintf(buf + off, sizeof(buf) - off, "]");
        if (i > 0) {
            is += ";";
        }
        is += buf;
        ie += in_elems_[i];
    }
    os.clear();
    oe = 0;
    for (size_t i = 0; i < num_outputs_; ++i) {
        char buf[128] = {};
        int off = snprintf(buf, sizeof(buf), "[");
        for (size_t d = 0; d < out_dims_[i].size(); ++d) {
            off += snprintf(buf + off, sizeof(buf) - off, "%s%u", d > 0 ? "," : "",
                            out_dims_[i][d]);
        }
        snprintf(buf + off, sizeof(buf) - off, "]");
        if (i > 0) {
            os += ";";
        }
        os += buf;
        oe += out_elems_[i];
    }
    return true;
}

/* ---------------------------------------------------------------------------
 * PrepareInputs / SetSharedInput
 * -------------------------------------------------------------------------*/
bool QnnSdkBackend::PrepareInputs(float *&fd, size_t &fe, const char * /*arg*/,
                                  bool random, const float *const *ext,
                                  const size_t *extc)
{
    for (size_t i = 0; i < num_inputs_; ++i) {
        size_t n = in_elems_[i] > 0 ? in_elems_[i] : 1;
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
            float *b = (float *)malloc(n * sizeof(float));
            if (!b) {
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

void QnnSdkBackend::SetSharedInput(const float *const *data, const size_t *counts)
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
 * Populate one input: quantize float [0,1] into tensor buffer if needed
 * -------------------------------------------------------------------------*/
void QnnSdkBackend::PopulateInput(int idx, const float *data, size_t n)
{
    void *dst = in_bufs_[idx];
    if (!dst || !data) {
        return;
    }
    /* Dispatch on the tensor DATA TYPE first (fixed-point dtypes must never
     * fall through to the float memcpy path: the buffers are sized for the
     * narrow dtype and a 4-byte copy would overrun into the guard page).
     * Quant scale/offset come from the tensor metadata (binary info when
     * present, otherwise the model.so GraphInfo). */
    if (in_dtypes_[idx] == QNN_DATATYPE_UFIXED_POINT_16 ||
        in_dtypes_[idx] == QNN_DATATYPE_SFIXED_POINT_16) {
        /* 16-bit fixed point (A16): quantize float into int16 buffer */
        float scale = in_quant_[idx].scale;
        int32_t offset = in_quant_[idx].offset;
        int16_t *q = (int16_t *)dst;
        for (size_t j = 0; j < n; ++j) {
            float v = data[j] / scale + (float)offset;
            if (v < -32768.0f) {
                q[j] = (int16_t)-32768;
            } else if (v > 32767.0f) {
                q[j] = (int16_t)32767;
            } else {
                q[j] = (int16_t)(v >= 0.0f ? (int32_t)(v + 0.5f) : (int32_t)(v - 0.5f));
            }
        }
    } else if (in_dtypes_[idx] == QNN_DATATYPE_UFIXED_POINT_8 ||
               in_dtypes_[idx] == QNN_DATATYPE_SFIXED_POINT_8) {
        /* 8-bit fixed point (A8): quantize float into uint8 buffer */
        float scale = in_quant_[idx].scale;
        int32_t offset = in_quant_[idx].offset;
        uint8_t *q = (uint8_t *)dst;
        for (size_t j = 0; j < n; ++j) {
            float v = data[j] / scale + (float)offset;
            q[j] = (uint8_t)(v < 0.0f ? 0 : (v > 255.0f ? 255 : (uint8_t)(v + 0.5f)));
        }
    } else if (in_dtypes_[idx] == QNN_DATATYPE_FLOAT_16) {
        /* Convert float -> FP16 into the 2-byte tensor buffer. On arm64 this
         * is a single NEON FCVT (4 lanes per instruction) instead of the
         * scalar bit-twiddling fallback; these models move ~7-14M elements
         * per inference, so the scalar loop was ~47ms of every frame. */
        uint16_t *h = (uint16_t *)dst;
#if defined(__aarch64__)
        size_t j = 0;
        for (; j + 4 <= n; j += 4) {
            float32x4_t f4 = vld1q_f32(data + j);
            vst1_u16(h + j, vreinterpret_u16_f16(vcvt_f16_f32(f4)));
        }
        for (; j < n; ++j) {
            h[j] = float_to_half(data[j]);
        }
#else
        for (size_t j = 0; j < n; ++j) {
            h[j] = float_to_half(data[j]);
        }
#endif
    } else {
        memcpy(dst, data, n * sizeof(float));
    }
}

/* ---------------------------------------------------------------------------
 * Read one output: dequantize tensor buffer into float
 * -------------------------------------------------------------------------*/
bool QnnSdkBackend::ReadOutput(int idx, float *dst, size_t n)
{
    const void *src = out_bufs_[idx];
    if (!src || !dst) {
        return false;
    }
    /* Same dtype-first dispatch as PopulateInput: fixed-point dtypes must
     * dequantize, never fall through to the 4-byte float memcpy. */
    if (out_dtypes_[idx] == QNN_DATATYPE_UFIXED_POINT_16 ||
        out_dtypes_[idx] == QNN_DATATYPE_SFIXED_POINT_16) {
        /* 16-bit fixed point (A16): dequantize int16 buffer to float */
        float scale = out_quant_[idx].scale;
        int32_t offset = out_quant_[idx].offset;
        const int16_t *q = (const int16_t *)src;
        for (size_t j = 0; j < n; ++j) {
            dst[j] = ((float)q[j] - (float)offset) * scale;
        }
    } else if (out_dtypes_[idx] == QNN_DATATYPE_UFIXED_POINT_8 ||
               out_dtypes_[idx] == QNN_DATATYPE_SFIXED_POINT_8) {
        /* 8-bit fixed point (A8): dequantize uint8 buffer to float */
        float scale = out_quant_[idx].scale;
        int32_t offset = out_quant_[idx].offset;
        const uint8_t *q = (const uint8_t *)src;
        for (size_t j = 0; j < n; ++j) {
            dst[j] = ((float)q[j] - (float)offset) * scale;
        }
    } else if (out_dtypes_[idx] == QNN_DATATYPE_FLOAT_16) {
        /* Convert FP16 -> float from the 2-byte tensor buffer (NEON FCVT on
         * arm64, scalar fallback elsewhere - see PopulateInput). */
        const uint16_t *h = (const uint16_t *)src;
#if defined(__aarch64__)
        size_t j = 0;
        for (; j + 4 <= n; j += 4) {
            float16x4_t h4 = vreinterpret_f16_u16(vld1_u16(h + j));
            vst1q_f32(dst + j, vcvt_f32_f16(h4));
        }
        for (; j < n; ++j) {
            dst[j] = half_to_float(h[j]);
        }
#else
        for (size_t j = 0; j < n; ++j) {
            dst[j] = half_to_float(h[j]);
        }
#endif
    } else {
        memcpy(dst, src, n * sizeof(float));
    }
    return true;
}

/* ---------------------------------------------------------------------------
 * RunBenchmark
 * -------------------------------------------------------------------------*/
bool QnnSdkBackend::RunBenchmark(int warmup, int repeat, double &total,
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
    if (num_inputs_ == 0 || num_outputs_ == 0 || !qnn_->graphExecute) {
        return false;
    }

    std::vector<std::vector<float>> snaps(num_outputs_);
    for (size_t i = 0; i < num_outputs_; ++i) {
        snaps[i].resize(out_elems_[i] > 0 ? out_elems_[i] : 1);
    }

    auto execute_once = [&](double &t_pop, double &t_sync, double &t_exec) -> bool {
        auto c0 = std::chrono::high_resolution_clock::now();
#if defined(__ANDROID__) || defined(__android__)
        /* Combined buffer: a single sync covers every input tensor */
        if (!in_dma_.empty() && in_dma_[0].fd >= 0) {
            dma_buf_sync(in_dma_[0].fd, DMA_BUF_SYNC_START | DMA_BUF_SYNC_WRITE);
        }
#endif
        auto c1 = std::chrono::high_resolution_clock::now();
        /* PopulateInput: convert all inputs from shared float buffers into the
         * tensor buffers (FP16/quantized). These 22 tensors are independent,
         * so convert them across CPU threads - for these sports models the
         * ~27-55MB of input conversion is the single biggest per-frame cost
         * after the NEON fix, and threads hide most of the memory bandwidth. */
        {
            size_t n_in = num_inputs_;
            size_t nthreads = (size_t)num_threads_;
            if (nthreads < 1) {
                nthreads = 1;
            }
            if (nthreads > n_in) {
                nthreads = n_in;
            }
            if (nthreads <= 1 || n_in <= 1) {
                for (size_t i = 0; i < n_in; ++i) {
                    PopulateInput((int)i, input_bufs_[i],
                                  input_buf_elems_[i] > 0 ? input_buf_elems_[i] : in_elems_[i]);
                }
            } else {
                size_t chunk = (n_in + nthreads - 1) / nthreads;
                std::vector<std::thread> ths;
                ths.reserve(nthreads);
                for (size_t t = 0; t < nthreads; ++t) {
                    ths.emplace_back([&, t]() {
                        for (size_t i = t * chunk; i < n_in && i < (t + 1) * chunk; ++i) {
                            PopulateInput((int)i, input_bufs_[i],
                                          input_buf_elems_[i] > 0 ? input_buf_elems_[i] : in_elems_[i]);
                        }
                    });
                }
                for (auto &th : ths) {
                    th.join();
                }
            }
        }
        auto c2 = std::chrono::high_resolution_clock::now();
#if defined(__ANDROID__) || defined(__android__)
        if (!in_dma_.empty() && in_dma_[0].fd >= 0) {
            dma_buf_sync(in_dma_[0].fd, DMA_BUF_SYNC_END | DMA_BUF_SYNC_WRITE);
        }
#endif
        Qnn_ErrorHandle_t st = qnn_->graphExecute(
            graph_,
            in_tensors_.data(), (uint32_t)in_tensors_.size(),
            out_tensors_.data(), (uint32_t)out_tensors_.size(),
            nullptr, nullptr);
        auto c3 = std::chrono::high_resolution_clock::now();
        /* Breakdown (DEBUG only): populate = CPU float->FP16 conversion of all
         * inputs, sync = DMA-BUF cache sync + fixed call overhead, exec = the
         * actual HTP inference. This isolates where the per-frame time goes. */
        t_pop = std::chrono::duration<double, std::milli>(c2 - c1).count();
        t_sync = std::chrono::duration<double, std::milli>(c1 - c0).count() +
                 std::chrono::duration<double, std::milli>(c3 - c2).count();
        t_exec = std::chrono::duration<double, std::milli>(c3 - c2).count();
        if (st != QNN_GRAPH_NO_ERROR) {
            LOGE("QNN: graphExecute rc=%d (0x%X)", (int)st, (unsigned)st);
        }
        return st == QNN_GRAPH_NO_ERROR;
    };

    for (int w = 0; w < warmup; ++w) {
        double t_pop = 0, t_sync = 0, t_exec = 0;
        if (!execute_once(t_pop, t_sync, t_exec)) {
            LOGE("QNN: warmup execute failed");
            return false;
        }
    }
    double acc_pop = 0, acc_sync = 0, acc_exec = 0;
    for (int r = 0; r < repeat; ++r) {
        auto t0 = std::chrono::high_resolution_clock::now();
        double t_pop = 0, t_sync = 0, t_exec = 0;
        if (!execute_once(t_pop, t_sync, t_exec)) {
            LOGE("QNN: execute failed at run %d", r);
            return false;
        }
        acc_pop += t_pop;
        acc_sync += t_sync;
        acc_exec += t_exec;
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        LOGD("QNN: run %d took %.3f ms (pop=%.2f sync=%.2f exec=%.2f)",
             r, ms, t_pop, t_sync, t_exec);
        /* Copy outputs only on the last iteration: intermediate snapshots are
         * never consumed, so reading ~10 MB of outputs every run would only add
         * CPU-side copy + DMA sync latency to each measurement. */
        if (r == repeat - 1) {
#if defined(__ANDROID__) || defined(__android__)
            if (!out_dma_.empty() && out_dma_[0].fd >= 0) {
                dma_buf_sync(out_dma_[0].fd, DMA_BUF_SYNC_START | DMA_BUF_SYNC_READ);
            }
#endif
            for (size_t i = 0; i < num_outputs_; ++i) {
                ReadOutput((int)i, snaps[i].data(), out_elems_[i]);
            }
#if defined(__ANDROID__) || defined(__android__)
            if (!out_dma_.empty() && out_dma_[0].fd >= 0) {
                dma_buf_sync(out_dma_[0].fd, DMA_BUF_SYNC_END | DMA_BUF_SYNC_READ);
            }
#endif
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
    if (repeat > 0) {
        LOGI("QNN: avg breakdown pop=%.2f sync=%.2f exec=%.2f ms (of avg %.2f ms)",
             acc_pop / repeat, acc_sync / repeat, acc_exec / repeat,
             total / repeat);
    }

    odata.resize(num_outputs_);
    oelems.resize(num_outputs_);
    oshapes.resize(num_outputs_);
    odims.resize(num_outputs_);
    for (size_t i = 0; i < num_outputs_; ++i) {
        size_t n = out_elems_[i];
        float *b = (float *)malloc(n * sizeof(float));
        if (!b) {
            LOGE("QNN: malloc(%zu) failed at output %zu", n * sizeof(float), i);
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
        for (size_t d = 0; d < out_dims_[i].size() && d < MAX_DIMENSIONS; ++d) {
            sh[d] = (size_t)out_dims_[i][d];
        }
        odims[i] = out_dims_[i].size();
    }
    return true;
}

void QnnSdkBackend::GetTiming(std::array<double, 10> &timing)
{
    timing.fill(0);
    timing[0] = init_ms_;
}

bool QnnSdkBackend::SaveOutputs(const char * /*suffix*/) { return true; }

/* ---------------------------------------------------------------------------
 * Cleanup
 * -------------------------------------------------------------------------*/
void QnnSdkBackend::Cleanup()
{
    /* Free model.so graph-info wrapper memory before releasing the context */
    if (free_graphs_info_ && graphs_info_) {
        free_graphs_info_(&graphs_info_, graphs_count_);
        graphs_info_ = nullptr;
        graphs_count_ = 0;
    }
    if (qnn_) {
        /* Release the HTP power vote before tearing down the context */
        if (power_config_id_ && destroy_power_config_) {
            destroy_power_config_(power_config_id_);
            power_config_id_ = 0;
            destroy_power_config_ = nullptr;
        }
        /* Deregister shared memory before freeing buffers */
        if (qnn_->memDeRegister) {
            std::vector<Qnn_MemHandle_t> all;
            for (auto &h : in_mem_handles_) {
                if (h) {
                    all.push_back(h);
                }
            }
            for (auto &h : out_mem_handles_) {
                if (h) {
                    all.push_back(h);
                }
            }
            if (!all.empty()) {
                qnn_->memDeRegister(all.data(), (uint32_t)all.size());
            }
        }
        if (context_ && qnn_->contextFree) {
            qnn_->contextFree(context_, nullptr);
        }
        if (device_ && qnn_->deviceFree) {
            qnn_->deviceFree(device_);
        }
        if (backend_ && qnn_->backendFree) {
            qnn_->backendFree(backend_);
        }
        if (log_ && qnn_->logFree) {
            qnn_->logFree(log_);
        }
    }
    for (size_t i = 0; i < in_bufs_.size(); ++i) {
        if (in_bufs_[i]) {
#if defined(__ANDROID__) || defined(__android__)
            /* Shared-direction pointers point into in_dma_[0]; unmapped below */
            if (in_dma_.empty() || in_dma_[0].fd < 0) {
                free(in_bufs_[i]);
            }
#else
            free(in_bufs_[i]);
#endif
        }
    }
    for (size_t i = 0; i < out_bufs_.size(); ++i) {
        if (out_bufs_[i]) {
#if defined(__ANDROID__) || defined(__android__)
            if (out_dma_.empty() || out_dma_[0].fd < 0) {
                free(out_bufs_[i]);
            }
#else
            free(out_bufs_[i]);
#endif
        }
    }
    in_bufs_.clear();
    out_bufs_.clear();
    in_mem_handles_.clear();
    out_mem_handles_.clear();
#if defined(__ANDROID__) || defined(__android__)
    for (auto &db : in_dma_) {
        if (db.fd >= 0) {
            munmap(db.addr, db.size);
            close(db.fd);
        }
    }
    for (auto &db : out_dma_) {
        if (db.fd >= 0) {
            munmap(db.addr, db.size);
            close(db.fd);
        }
    }
#endif
    in_dma_.clear();
    out_dma_.clear();

    if (lib_system_) {
        dlclose(lib_system_);
        lib_system_ = nullptr;
    }
    if (lib_model_) {
        dlclose(lib_model_);
        lib_model_ = nullptr;
        compose_graphs_ = nullptr;
        free_graphs_info_ = nullptr;
    }
    if (lib_backend_) {
        dlclose(lib_backend_);
        lib_backend_ = nullptr;
    }
    qnn_ = nullptr;
    sys_ = nullptr;
    backend_ = nullptr;
    device_ = nullptr;
    context_ = nullptr;
    graph_ = nullptr;
    log_ = nullptr;

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
BackendPtr CreateQnnSdkBackend(BackendId id)
{
    return std::make_unique<QnnSdkBackend>(id);
}

#endif /* HAVE_QNN_SDK_BACKEND */
