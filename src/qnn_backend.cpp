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
 * NOTE: do NOT try to call rpcmem_* from libcdsprpc.so directly — on some
 * devices (SM8550) the system libcdsprpc's rpcmem_alloc crashes in the shell
 * domain (its internal remote_register_buf path is SELinux-blocked). qnn-net-run
 * --shared_buffer works because QNN allocates/registers the buffer internally.
 *============================================================================*/

#include "backend_interface.hpp"
#include "log.hpp"

#ifdef HAVE_QNN_SDK_BACKEND

#include <QNN/QnnBackend.h>
#include <QNN/QnnCommon.h>
#include <QNN/QnnContext.h>
#include <QNN/QnnDevice.h>
#include <QNN/QnnGraph.h>
#include <QNN/QnnInterface.h>
#include <QNN/QnnLog.h>
#include <QNN/QnnMem.h>
#include <QNN/System/QnnSystemContext.h>
#include <QNN/System/QnnSystemInterface.h>
#include <QNN/QnnTensor.h>
#include <QNN/QnnTypes.h>
#include <QNN/HTP/QnnHtpMem.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <dlfcn.h>

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
    (void)ioctl(fd, DMA_BUF_IOCTL_SYNC, &sync);
}
#endif

/* ---------------------------------------------------------------------------
 * model.so (QnnModel_composeGraphs) wrapper types.
 * QnnModel_freeGraphsInfo and is compiled per-target at runtime, so the SAME
 * .so can drive CPU / GPU / HTP backends. Mirrors SampleApp's QnnWrapperUtils.
 * -------------------------------------------------------------------------*/
namespace qnn_wrapper_api {
typedef enum ModelError {
    MODEL_NO_ERROR               = 0,
    MODEL_TENSOR_ERROR           = 1,
    MODEL_PARAMS_ERROR           = 2,
    MODEL_NODES_ERROR            = 3,
    MODEL_GRAPH_ERROR            = 4,
    MODEL_CONTEXT_ERROR          = 5,
    MODEL_GENERATION_ERROR       = 6,
    MODEL_SETUP_ERROR            = 7,
    MODEL_INVALID_ARGUMENT_ERROR = 8,
    MODEL_FILE_ERROR             = 9,
    MODEL_MEMORY_ALLOCATE_ERROR  = 10,
    MODEL_UNKNOWN_ERROR          = 0x7FFFFFFF
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

/* QNN backend log sink — surfaces backend-side errors (e.g. HTP graphCreate). */
static void qnn_log_callback(const char *fmt, QnnLog_Level_t level,
                             uint64_t /*timestamp*/, va_list args)
{
    fprintf(stderr, "[QNN_LOG:%d] ", (int)level);
    vfprintf(stderr, fmt ? fmt : "", args);
    fprintf(stderr, "\n");
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
    void PopulateInput(int idx, const float *data, size_t n);
    bool ReadOutput(int idx, float *dst, size_t n);

    /* Library handles */
    void *lib_backend_ = nullptr;  /* libQnnHtp.so / libQnnGpu.so / libQnnCpu.so */
    void *lib_system_ = nullptr;   /* libQnnSystem.so */

    /* model.so support (QnnModel_composeGraphs) */
    bool is_model_so_ = false;
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
    std::vector<std::string> in_names_;   /* owned copies of tensor names */
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

    /* DMA-BUF descriptors for shared buffers (to munmap + close on cleanup) */
    struct DmaBuf {
        int fd = -1;
        size_t size = 0;
        void *addr = nullptr;
    };
    std::vector<DmaBuf> in_dma_;
    std::vector<DmaBuf> out_dma_;

    /* Inputs from runner (float, shared across backends) */
    std::vector<float *> input_bufs_;
    std::vector<size_t> input_buf_elems_;
    std::vector<bool> input_external_;

    size_t num_inputs_ = 0;
    size_t num_outputs_ = 0;

    double init_ms_ = 0;
};

/* ---------------------------------------------------------------------------
 * Helpers
 * -------------------------------------------------------------------------*/
static const char *qnn_backend_lib(BackendId id)
{
    switch (id) {
    case BackendId::QNN_SDK_GPU:
        return "libQnnGpu.so";
    case BackendId::QNN_SDK_CPU:
        return "libQnnCpu.so";
    case BackendId::QNN_SDK_HTP:
    default:
        return "libQnnHtp.so";
    }
}

static const char *qnn_backend_type(BackendId id)
{
    switch (id) {
    case BackendId::QNN_SDK_GPU:
        return "gpu";
    case BackendId::QNN_SDK_CPU:
        return "cpu";
    case BackendId::QNN_SDK_HTP:
    default:
        return "htp";
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
};static QnnGraphView qnn_graph_view(const QnnSystemContext_GraphInfo_t *g)
{
    QnnGraphView v;
    if (!g) {
        return v;
    }
    switch (g->version) {
    case QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_2:
        v.name = g->graphInfoV2.graphName;
        v.numInputs = g->graphInfoV2.numGraphInputs;
        v.inputs = g->graphInfoV2.graphInputs;
        v.numOutputs = g->graphInfoV2.numGraphOutputs;
        v.outputs = g->graphInfoV2.graphOutputs;
        break;
    case QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_3:
        v.name = g->graphInfoV3.graphName;
        v.numInputs = g->graphInfoV3.numGraphInputs;
        v.inputs = g->graphInfoV3.graphInputs;
        v.numOutputs = g->graphInfoV3.numGraphOutputs;
        v.outputs = g->graphInfoV3.graphOutputs;
        break;
    default: /* V1 */
        v.name = g->graphInfoV1.graphName;
        v.numInputs = g->graphInfoV1.numGraphInputs;
        v.inputs = g->graphInfoV1.graphInputs;
        v.numOutputs = g->graphInfoV1.numGraphOutputs;
        v.outputs = g->graphInfoV1.graphOutputs;
        break;
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
    case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1:
        *graphs = b->contextBinaryInfoV1.graphs;
        *num_graphs = b->contextBinaryInfoV1.numGraphs;
        break;
    case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2:
        *graphs = b->contextBinaryInfoV2.graphs;
        *num_graphs = b->contextBinaryInfoV2.numGraphs;
        break;
    case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_3:
        *graphs = b->contextBinaryInfoV3.graphs;
        *num_graphs = b->contextBinaryInfoV3.numGraphs;
        break;
    default:
        *graphs = nullptr;
        *num_graphs = 0;
        break;
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

/* ELF64 shared object → model.so (composeGraphs) instead of raw context binary */
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
    auto get_providers = (Qnn_ErrorHandle_t(*)(const QnnInterface_t ***, uint32_t *))
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

    /* 2. Create log handle (best effort) — errors only, keeps output clean */
    if (qnn_->logCreate) {
        (void)qnn_->logCreate(qnn_log_callback, QNN_LOG_LEVEL_ERROR, &log_);
    }

    /* 3. Create backend */
    if (!qnn_->backendCreate ||
        QNN_BACKEND_NO_ERROR != qnn_->backendCreate(log_, nullptr, &backend_)) {
        LOGE("QNN: backendCreate failed");
        return false;
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

        qnn_wrapper_api::GraphInfo_t **gi = nullptr;
        uint32_t ng = 0;
        qnn_wrapper_api::ModelError_t mrc = compose_graphs_(
            backend_, *qnn_, context_, nullptr, 0, &gi, &ng,
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
    auto get_sys_providers = (Qnn_ErrorHandle_t(*)(const QnnSystemInterface_t ***, uint32_t *))
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
     * "buffers: x/y shared" summary below still reports the result. */
    const bool try_dma_heap = (id_ == BackendId::QNN_SDK_HTP);
    bool dma_heap_warned = false;
    auto warn_dma_heap_once = [&](const char *msg) {
        if (!dma_heap_warned) {
            LOGW("QNN: %s - fallback client buffer", msg);
            dma_heap_warned = true;
        }
    };
    auto alloc_one = [&](Qnn_Tensor_t &t, size_t elems, Qnn_DataType_t dt,
                         std::vector<void *> &bufs, std::vector<Qnn_MemHandle_t> &mem,
                         std::vector<bool> &is_shared, std::vector<DmaBuf> &dma) -> bool {
        size_t bytes = elems * (dt == QNN_DATATYPE_FLOAT_32 ? 4u : 1u);
        void *p = nullptr;
        bool shared = false;
#if defined(__ANDROID__) || defined(__android__)
        if (try_dma_heap) {
            /* Allocate a DMA-BUF from the system dma-heap. This needs no
             * libcdsprpc/fastrpc on the app side — QNN registers the fd with the
             * DSP through its own channel. (Direct rpcmem_* calls into the system
             * libcdsprpc crash in the shell domain on SM8550.) */
            size_t alloc_size = (bytes + 4095u) & ~(size_t)4095u;
            int dmafd = open("/dev/dma_heap/system", O_RDONLY | O_CLOEXEC);
            if (dmafd >= 0) {
                struct dma_heap_allocation_data dmabuf = {};
                dmabuf.len = alloc_size;
                dmabuf.fd_flags = O_RDWR | O_CLOEXEC;
                if (ioctl(dmafd, DMA_HEAP_IOCTL_ALLOC, &dmabuf) == 0) {
                    int fd = (int)dmabuf.fd;
                    void *addr = mmap(nullptr, alloc_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
                    if (addr != MAP_FAILED) {
                        Qnn_MemHandle_t h = nullptr;
                        /* HTP backend requires a CUSTOM memory descriptor with a
                         * QnnMemHtp_Descriptor_t of type QNN_HTP_MEM_SHARED_BUFFER.
                         * Using QNN_MEM_TYPE_DMA_BUF here crashes libQnnHtp (the
                         * backend reads the union as a custom descriptor). */
                        QnnMemHtp_Descriptor_t htp_desc = {};
                        htp_desc.type = QNN_HTP_MEM_SHARED_BUFFER;
                        htp_desc.size = alloc_size;
                        htp_desc.sharedBufferConfig.fd = fd;
                        htp_desc.sharedBufferConfig.offset = 0;

                        Qnn_MemDescriptor_t desc = QNN_MEM_DESCRIPTOR_INIT;
                        desc.dataType = dt;
                        desc.memType = QNN_MEM_TYPE_CUSTOM;
                        desc.customInfo = &htp_desc;
                        if (qnn_->memRegister && QNN_SUCCESS == qnn_->memRegister(context_, &desc, 1, &h)) {
                            shared = true;
                            mem.push_back(h);
                            dma.push_back({fd, alloc_size, addr});
                            p = addr;
                            /* Per-tensor detail at DBG level (floods the log on
                             * multi-tensor models); the "buffers: x/y shared"
                             * INFO summary below reports the outcome. */
                            LOGD("QNN: dma-heap %zu B (aligned %zu) fd=%d memHandle=%p (shared)",
                                 bytes, alloc_size, fd, (void *)h);
                        } else {
                            munmap(addr, alloc_size);
                            close(fd);
                            warn_dma_heap_once("dma-heap memRegister failed");
                        }
                    } else {
                        close(fd);
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
        if (!p) {
            p = malloc(bytes > 0 ? bytes : 1);
            mem.push_back(nullptr);
            dma.push_back({});
        }
        bufs.push_back(p);
        is_shared.push_back(shared);

        /* Attach buffer to tensor (QNN 2.48 direct struct access) */
        Qnn_TensorV1_t *tv = qnn_tensor_v1(&t);
        if (shared) {
            tv->memType = QNN_TENSORMEMTYPE_MEMHANDLE;
            tv->memHandle = mem.back();
        } else {
            tv->memType = QNN_TENSORMEMTYPE_RAW;
            tv->clientBuf.data = p;
            tv->clientBuf.dataSize = bytes;
        }
        return p != nullptr;
    };

    for (size_t i = 0; i < in_tensors_.size(); ++i) {
        if (!alloc_one(in_tensors_[i], in_elems_[i], in_dtypes_[i],
                       in_bufs_, in_mem_handles_, in_is_shared_, in_dma_)) {
            LOGE("QNN: input %zu alloc failed", i);
            return false;
        }
    }
    for (size_t i = 0; i < out_tensors_.size(); ++i) {
        if (!alloc_one(out_tensors_[i], out_elems_[i], out_dtypes_[i],
                       out_bufs_, out_mem_handles_, out_is_shared_, out_dma_)) {
            LOGE("QNN: output %zu alloc failed", i);
            return false;
        }
    }

    /* Report whether zero-copy shared buffers actually engaged */
    size_t n_shared = 0, n_total = in_is_shared_.size() + out_is_shared_.size();
    for (bool s : in_is_shared_)
        if (s) ++n_shared;
    for (bool s : out_is_shared_)
        if (s) ++n_shared;
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
    (void)num_threads;

    /* ELF shared object → model.so (composeGraphs), otherwise context binary */
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
    if (in_is_quant_[idx]) {
        float scale = in_quant_[idx].scale;
        int32_t offset = in_quant_[idx].offset;
        uint8_t *q = (uint8_t *)dst;
        for (size_t j = 0; j < n; ++j) {
            float v = data[j] / scale + (float)offset;
            q[j] = (uint8_t)(v < 0.0f ? 0 : (v > 255.0f ? 255 : (uint8_t)(v + 0.5f)));
        }
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
    if (out_is_quant_[idx]) {
        float scale = out_quant_[idx].scale;
        int32_t offset = out_quant_[idx].offset;
        const uint8_t *q = (const uint8_t *)src;
        for (size_t j = 0; j < n; ++j) {
            dst[j] = ((float)q[j] - (float)offset) * scale;
        }
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

    auto execute_once = [&]() -> bool {
#if defined(__ANDROID__) || defined(__android__)
        for (size_t i = 0; i < num_inputs_; ++i) {
            if (in_is_shared_[i] && i < in_dma_.size() && in_dma_[i].fd >= 0)
                dma_buf_sync(in_dma_[i].fd, DMA_BUF_SYNC_START | DMA_BUF_SYNC_WRITE);
        }
#endif
        for (size_t i = 0; i < num_inputs_; ++i) {
            PopulateInput((int)i, input_bufs_[i],
                          input_buf_elems_[i] > 0 ? input_buf_elems_[i] : in_elems_[i]);
        }
#if defined(__ANDROID__) || defined(__android__)
        for (size_t i = 0; i < num_inputs_; ++i) {
            if (in_is_shared_[i] && i < in_dma_.size() && in_dma_[i].fd >= 0)
                dma_buf_sync(in_dma_[i].fd, DMA_BUF_SYNC_END | DMA_BUF_SYNC_WRITE);
        }
#endif
        Qnn_ErrorHandle_t st = qnn_->graphExecute(
            graph_,
            in_tensors_.data(), (uint32_t)in_tensors_.size(),
            out_tensors_.data(), (uint32_t)out_tensors_.size(),
            nullptr, nullptr);
        if (st != QNN_GRAPH_NO_ERROR) {
            LOGE("QNN: graphExecute rc=%d (0x%X)", (int)st, (unsigned)st);
        }
        return st == QNN_GRAPH_NO_ERROR;
    };

    for (int w = 0; w < warmup; ++w) {
        if (!execute_once()) {
            LOGE("QNN: warmup execute failed");
            return false;
        }
    }
    for (int r = 0; r < repeat; ++r) {
        auto t0 = std::chrono::high_resolution_clock::now();
        if (!execute_once()) {
            LOGE("QNN: execute failed at run %d", r);
            return false;
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        for (size_t i = 0; i < num_outputs_; ++i) {
#if defined(__ANDROID__) || defined(__android__)
            if (out_is_shared_[i] && i < out_dma_.size() && out_dma_[i].fd >= 0)
                dma_buf_sync(out_dma_[i].fd, DMA_BUF_SYNC_START | DMA_BUF_SYNC_READ);
#endif
            ReadOutput((int)i, snaps[i].data(), out_elems_[i]);
#if defined(__ANDROID__) || defined(__android__)
            if (out_is_shared_[i] && i < out_dma_.size() && out_dma_[i].fd >= 0)
                dma_buf_sync(out_dma_[i].fd, DMA_BUF_SYNC_END | DMA_BUF_SYNC_READ);
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
            if (in_is_shared_[i] && i < in_dma_.size() && in_dma_[i].fd >= 0) {
                munmap(in_dma_[i].addr, in_dma_[i].size);
                close(in_dma_[i].fd);
            } else
#endif
                free(in_bufs_[i]);
        }
    }
    for (size_t i = 0; i < out_bufs_.size(); ++i) {
        if (out_bufs_[i]) {
#if defined(__ANDROID__) || defined(__android__)
            if (out_is_shared_[i] && i < out_dma_.size() && out_dma_[i].fd >= 0) {
                munmap(out_dma_[i].addr, out_dma_[i].size);
                close(out_dma_[i].fd);
            } else
#endif
                free(out_bufs_[i]);
        }
    }
    in_bufs_.clear();
    out_bufs_.clear();
    in_mem_handles_.clear();
    out_mem_handles_.clear();
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
