/*============================================================================
 * mnn_backend.cpp - MNN backend (RAII)
 *============================================================================*/

#include "backend_interface.hpp"
#include "log.hpp"

#ifdef HAVE_MNN_BACKEND

#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

class MNNBackend : public IBackend
{
public:
    explicit MNNBackend(BackendId id) { id_ = id; }
    ~MNNBackend() override { Cleanup(); }

    bool Initialize(const char *model_path, int num_threads) override;
    bool QueryIOInfo(std::string &is, size_t &ie, std::string &os, size_t &oe) override;
    void SetSharedInput(const float *const *data, const size_t *counts) override;
    bool RunBenchmark(int warmup, int repeat, double &total, double &maxv,
                      double &minv, int &maxi, std::vector<float *> &odata,
                      std::vector<size_t> &oelems,
                      std::vector<std::array<size_t, MAX_DIMENSIONS>> &oshapes,
                      std::vector<size_t> &odims) override;
    void GetTiming(std::array<double, 10> &timing) override;
    void GetTransferTiming(double &transfer_in_ms,
                           double &transfer_out_ms) override;

private:
    void Cleanup();

    /* Verify the session really executes on the requested forward type.
     * MNN's ScheduleConfig defaults backupType to MNN_FORWARD_CPU, so a backend
     * that cannot be created (no Vulkan/OpenGL driver, no NNAPI, unsupported
     * precision) silently degrades to CPU: createSession() still succeeds and
     * returns plausible CPU timings under a GPU backend name. That violates the
     * tool's "unsupported = report failure, never silently degrade" rule.
     * Returns false (and fills last_error_) when the outputs do not live on the
     * requested backend. */
    bool VerifyActualForwardType();

    std::shared_ptr<MNN::Interpreter> interp_;
    MNN::Session *session_ = nullptr;
    MNN::ScheduleConfig sched_;
    MNN::BackendConfig bcfg_;
    MNNForwardType requested_type_ = MNN_FORWARD_CPU;

    size_t num_inputs_ = 0;
    size_t num_outputs_ = 0;
    std::vector<MNN::Tensor *> input_tensors_;
    std::vector<MNN::Tensor *> output_tensors_;
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    std::vector<size_t> input_elems_;
    std::vector<size_t> output_elems_;
    std::vector<std::vector<int>> input_shapes_;
    std::vector<std::vector<int>> output_shapes_;

    std::vector<float *> input_bufs_;
    std::vector<size_t> input_buf_elems_;
    std::vector<bool> input_external_;

    double init_ms_ = 0;

    /* Tensor transfer timing (avg ms per repeat):
     * transfer_in_ms_  = host -> device input upload (feed_inputs)
     * transfer_out_ms_ = device -> host output download (copyToHostTensor +
     *                    snapshot memcpy) */
    double transfer_in_ms_ = 0.0;
    double transfer_out_ms_ = 0.0;
};

bool MNNBackend::Initialize(const char *model_path, int num_threads)
{
    auto t0 = std::chrono::high_resolution_clock::now();
    last_error_.clear();

    LOGI("MNN: version=%s", MNN::getVersion());
    sched_.numThread = num_threads;

    /* Max speed: pre-allocate all memory, max power */
    bcfg_.memory = MNN::BackendConfig::Memory_High;
    bcfg_.power = MNN::BackendConfig::Power_High;

    switch (id_) {
    case BackendId::MNN_CPU: {
        sched_.type = MNN_FORWARD_CPU;
        break;
    }
    case BackendId::MNN_OPENCL: {
        sched_.type = MNN_FORWARD_OPENCL;
        bcfg_.precision = MNN::BackendConfig::Precision_High;
        break;
    }
    case BackendId::MNN_OPENCL_FP16: {
        sched_.type = MNN_FORWARD_OPENCL;
        bcfg_.precision = MNN::BackendConfig::Precision_Low;
        break;
    }
    case BackendId::MNN_OPENCL_BF16: {
        sched_.type = MNN_FORWARD_OPENCL;
        bcfg_.precision = MNN::BackendConfig::Precision_Low_BF16;
        break;
    }
    case BackendId::MNN_VULKAN: {
        sched_.type = MNN_FORWARD_VULKAN;
        bcfg_.precision = MNN::BackendConfig::Precision_High;
        break;
    }
    case BackendId::MNN_VULKAN_FP16: {
        sched_.type = MNN_FORWARD_VULKAN;
        bcfg_.precision = MNN::BackendConfig::Precision_Low;
        break;
    }
    case BackendId::MNN_VULKAN_BF16: {
        sched_.type = MNN_FORWARD_VULKAN;
        bcfg_.precision = MNN::BackendConfig::Precision_Low_BF16;
        break;
    }
    case BackendId::MNN_OPENGL: {
        sched_.type = MNN_FORWARD_OPENGL;
        bcfg_.precision = MNN::BackendConfig::Precision_High;
        break;
    }
    case BackendId::MNN_NN: {
        sched_.type = MNN_FORWARD_NN;
        bcfg_.precision = MNN::BackendConfig::Precision_High;
        break;
    }
    default: {
        sched_.type = MNN_FORWARD_CPU;
        break;
    }
    }

    /* Always set backendConfig for all modes to apply Memory_High/Power_High */
    sched_.backendConfig = &bcfg_;
    requested_type_ = sched_.type;

    /* NO SILENT FALLBACK (tool-wide rule: unsupported => report failure).
     *
     * ScheduleConfig::backupType defaults to MNN_FORWARD_CPU and MNN uses it
     * whenever the requested backend cannot be created. That turns "Vulkan is
     * unavailable" into "run on CPU and label it MNN_VULKAN" - the CSV row
     * looks normal, so a CPU number gets compared as if it were a GPU number.
     *
     * Setting backupType to the requested type makes MNN fail instead of
     * degrading. Verified by VerifyActualForwardType() below, which also
     * catches any other path that could land the session on CPU. */
    sched_.backupType = sched_.type;

    LOGI("MNN: forward=%d threads=%d precision=%d backup=%d",
         sched_.type, num_threads, (int)bcfg_.precision, (int)sched_.backupType);

    interp_ = std::shared_ptr<MNN::Interpreter>(
        MNN::Interpreter::createFromFile(model_path));
    if (!interp_) {
        LOGE("MNN: createFromFile failed");
        last_error_ = "MNN: createFromFile failed";
        return false;
    }

    session_ = interp_->createSession(sched_);
    if (!session_) {
        LOGE("MNN: createSession failed (backend unusable, no fallback)");
        interp_ = nullptr;
        last_error_ = "MNN: createSession failed - requested backend unavailable "
                      "(no silent fallback to CPU)";
        return false;
    }

    /* NOTE: VerifyActualForwardType() runs further down, after the I/O tensors
     * are discovered - it needs output_tensors_[0]. */
    if (!session_) {
        LOGE("MNN: createSession failed (backend unusable, no fallback)");
        interp_ = nullptr;
        last_error_ = "MNN: createSession failed - requested backend unavailable "
                      "(no silent fallback to CPU)";
        return false;
    }

    /* Discover I/O tensors */
    auto in_map = interp_->getSessionInputAll(session_);
    auto out_map = interp_->getSessionOutputAll(session_);

    /* Sort by numeric suffix */
    using Pair = std::pair<std::string, MNN::Tensor *>;
    std::vector<Pair> sorted_in(in_map.begin(), in_map.end());
    std::vector<Pair> sorted_out(out_map.begin(), out_map.end());

    auto num_suffix = [](const std::string &s) -> int {
        size_t p = s.length();
        while (p > 0 && isdigit((unsigned char)s[p - 1])) {
            --p;
        }
        return (p < s.length()) ? atoi(s.c_str() + p) : 0;
    };
    std::sort(sorted_in.begin(), sorted_in.end(),
              [&](const Pair &a, const Pair &b) { return num_suffix(a.first) < num_suffix(b.first); });
    std::sort(sorted_out.begin(), sorted_out.end(),
              [&](const Pair &a, const Pair &b) { return num_suffix(a.first) < num_suffix(b.first); });

    for (auto &[name, t] : sorted_in) {
        input_tensors_.push_back(t);
        input_names_.push_back(name);
        auto shape = t->shape();
        std::vector<int> dims;
        size_t elems = 1;
        for (auto d : shape) {
            dims.push_back(d);
            elems *= (size_t)d;
        }
        input_shapes_.push_back(dims);
        input_elems_.push_back(elems);
    }
    for (auto &[name, t] : sorted_out) {
        output_tensors_.push_back(t);
        output_names_.push_back(name);
        auto shape = t->shape();
        std::vector<int> dims;
        size_t elems = 1;
        for (auto d : shape) {
            dims.push_back(d);
            elems *= (size_t)d;
        }
        output_shapes_.push_back(dims);
        output_elems_.push_back(elems);
    }

    num_inputs_ = input_tensors_.size();
    num_outputs_ = output_tensors_.size();

    /* Confirm the session really executes on the requested backend BEFORE
     * accepting it. Runs here (not right after createSession) because it
     * inspects output_tensors_[0], which is only discovered above. */
    if (!VerifyActualForwardType()) {
        LOGE("MNN: session does not run on the requested backend - "
             "reporting failure instead of degrading to CPU");
        interp_->releaseSession(session_);
        session_ = nullptr;
        interp_ = nullptr;
        return false;
    }

    init_ms_ = std::chrono::duration<double, std::milli>(
                   std::chrono::high_resolution_clock::now() - t0)
                   .count();

    LOGI("MNN: init complete (%.1f ms), %zu in, %zu out", init_ms_, num_inputs_, num_outputs_);
    return true;
}

/* ---------------------------------------------------------------------------
 * Backend verification
 *
 * MNN's ScheduleConfig::backupType defaults to MNN_FORWARD_CPU, so a backend
 * that cannot be created (no Vulkan/OpenGL driver, no NNAPI, unsupported
 * precision) silently degrades: createSession() succeeds and returns CPU
 * timings under a GPU backend name. That breaks the tool's core rule
 * ("unsupported => report failure, never silently degrade").
 *
 * Detection: the output tensor's halide_buffer_t tells us where the data
 * lives. For a HOST (CPU) tensor both `device` and `device_interface` are
 * zero/null; for a DEVICE tensor MNN sets them. So:
 *   requested CPU  -> outputs must have no device handle
 *   requested !CPU -> outputs must have a device handle
 *
 * This inspects only Tensor.hpp (vendored) - Backend.hpp is NOT shipped in
 * deps/mnn/include, so Backend::type() cannot be used, and comparing Backend
 * pointers is useless because MNN creates a separate Backend per session
 * (a CPU reference session's Backend differs from another CPU session's).
 * -------------------------------------------------------------------------*/
bool MNNBackend::VerifyActualForwardType()
{
    if (num_outputs_ == 0 || !output_tensors_[0]) {
        LOGE("MNN: no output tensor to verify the backend against");
        last_error_ = "MNN: no output tensor to verify the backend against";
        return false;
    }

    const halide_buffer_t &buf = output_tensors_[0]->buffer();
    /* Measured on MNN 3.6.0 (Windows, this model):
     *   CPU session  -> device == 1 (sentinel), host != nullptr
     *   OpenCL       -> device == a real device address (>1), host == nullptr
     * So `device > 1` distinguishes device-resident from host-resident.
     * `device != 0` alone does NOT work: CPU tensors carry the sentinel 1. */
    const bool on_device = (buf.device > 1);
    const bool want_device = (requested_type_ != MNN_FORWARD_CPU);

    /* Cross-check every output: a partially degraded session would otherwise
     * slip through on the strength of its first tensor. */
    size_t on_device_cnt = 0;
    for (size_t i = 0; i < num_outputs_; ++i) {
        if (!output_tensors_[i]) {
            continue;
        }
        const halide_buffer_t &b = output_tensors_[i]->buffer();
        if (b.device > 1) {
            ++on_device_cnt;
        }
    }

    LOGI("MNN: backend check requested=%d want_device=%d outputs_on_device=%zu/%zu",
         (int)requested_type_, (int)want_device, on_device_cnt, num_outputs_);

    if (want_device && on_device_cnt == 0) {
        /* Every output lives in host memory: the session runs on CPU even
         * though a device backend was requested - a silent fallback. */
        last_error_ = "MNN: requested a non-CPU backend but every output tensor "
                      "is host-resident - the session degraded to CPU "
                      "(no silent fallback: reporting failure)";
        return false;
    }
    if (!want_device && on_device_cnt > 0) {
        /* Requested CPU yet the data is on a device: the timing semantics this
         * backend assumes (host-resident I/O) would not hold. */
        last_error_ = "MNN: requested CPU but output tensors are device-resident "
                      "(unexpected session configuration)";
        return false;
    }

    if (want_device && on_device_cnt < num_outputs_) {
        LOGW("MNN: only %zu/%zu outputs are device-resident - partial device "
             "execution", on_device_cnt, num_outputs_);
    }
    return true;
}

bool MNNBackend::QueryIOInfo(std::string &is, size_t &ie,
                             std::string &os, size_t &oe)
{
    is.clear();
    ie = 0;
    for (size_t i = 0; i < num_inputs_; ++i) {
        char buf[128] = {};
        int off = snprintf(buf, sizeof(buf), "[");
        for (size_t d = 0; d < input_shapes_[i].size(); ++d) {
            off += snprintf(buf + off, sizeof(buf) - off, "%s%d", d > 0 ? "," : "", input_shapes_[i][d]);
        }
        snprintf(buf + off, sizeof(buf) - off, "]");
        if (i > 0) {
            is += " ";
        }
        is += buf;
        ie += input_elems_[i];
    }
    os.clear();
    oe = 0;
    for (size_t i = 0; i < num_outputs_; ++i) {
        char buf[128] = {};
        int off = snprintf(buf, sizeof(buf), "[");
        for (size_t d = 0; d < output_shapes_[i].size(); ++d) {
            off += snprintf(buf + off, sizeof(buf) - off, "%s%d", d > 0 ? "," : "", output_shapes_[i][d]);
        }
        snprintf(buf + off, sizeof(buf) - off, "]");
        if (i > 0) {
            os += " ";
        }
        os += buf;
        oe += output_elems_[i];
    }
    return true;
}

void MNNBackend::SetSharedInput(const float *const *data, const size_t *counts)
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

bool MNNBackend::RunBenchmark(int warmup, int repeat, double &total,
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
    if (num_inputs_ == 0 || num_outputs_ == 0) {
        return false;
    }

    /* Allocate output snapshots */
    std::vector<std::vector<float>> snaps(num_outputs_);
    for (size_t i = 0; i < num_outputs_; ++i) {
        snaps[i].resize(output_elems_[i] > 0 ? output_elems_[i] : 1);
    }

    /* Pre-allocate float host tensors for GPU output (reused across repeats) */
    const bool is_device = (requested_type_ != MNN_FORWARD_CPU);
    std::vector<MNN::Tensor *> gpu_out_tensors;
    if (is_device) {
        gpu_out_tensors.resize(num_outputs_, nullptr);
        for (size_t i = 0; i < num_outputs_; ++i) {
            if (output_tensors_[i]) {
                gpu_out_tensors[i] = MNN::Tensor::create<float>(
                    output_tensors_[i]->shape(), nullptr,
                    output_tensors_[i]->getDimensionType());
            }
        }
    }

    /* Pre-allocate float host tensors for GPU input (reused across feeds) */
    std::vector<MNN::Tensor *> gpu_in_tensors;
    if (is_device) {
        gpu_in_tensors.resize(num_inputs_, nullptr);
        for (size_t i = 0; i < num_inputs_; ++i) {
            MNN::Tensor *t = interp_->getSessionInput(session_, input_names_[i].c_str());
            if (t && input_bufs_[i]) {
                gpu_in_tensors[i] = MNN::Tensor::create<float>(
                    t->shape(), input_bufs_[i], t->getDimensionType());
            }
        }
    }

    /* Helper: copy host data to MNN input tensors */
    auto feed_inputs = [&]() {
        for (size_t i = 0; i < num_inputs_; ++i) {
            MNN::Tensor *t = interp_->getSessionInput(session_, input_names_[i].c_str());
            if (!t || !input_bufs_[i]) {
                continue;
            }
            size_t n = input_elems_[i];
            float *host = t->host<float>();
            if (host && !is_device) {
                memcpy(host, input_bufs_[i], n * sizeof(float));
                continue;
            }
            /* GPU input: use pre-allocated host tensor with copyFromHostTensor
             * for automatic FP32->FP16 conversion */
            if (i < gpu_in_tensors.size() && gpu_in_tensors[i]) {
                t->copyFromHostTensor(gpu_in_tensors[i]);
                continue;
            }
        }
    };

    interp_->resizeSession(session_);
    feed_inputs();

    (void)warmup;
    for (int w = 0; w < warmup; ++w) {
        if (interp_->runSession(session_) != MNN::NO_ERROR) {
            LOGE("MNN: warmup run failed");
            return false;
        }
    }

    /* Reset after warmup */
    interp_->resizeSession(session_);
    feed_inputs();

    for (int r = 0; r < repeat; ++r) {
        /* Time the host->device input upload (feed_inputs). */
        auto t_in0 = std::chrono::high_resolution_clock::now();
        if (r > 0) {
            interp_->resizeSession(session_);
            feed_inputs();
        }
        auto t_in1 = std::chrono::high_resolution_clock::now();
        transfer_in_ms_ +=
            std::chrono::duration<double, std::milli>(t_in1 - t_in0).count();

        auto t0 = std::chrono::high_resolution_clock::now();
        if (interp_->runSession(session_) != MNN::NO_ERROR) {
            LOGE("MNN: run %d failed", r);
            return false;
        }
        /* MNN device backends execute ASYNCHRONOUSLY: runSession() only
         * enqueues the work and returns immediately. Timed alone it measures
         * command submission, not inference - on this model that reported
         * ~5 ms "avg_run_ms" while the real cost (~220 ms) landed in
         * transfer_out_ms, because copyToHostTensor() is what actually blocks.
         *
         * That made avg_run_ms meaningless and inflated transfer_out_ms by the
         * entire GPU compute time. Force the sync INSIDE the timed window so
         * avg_run_ms is the true inference time and transfer_out_ms is the
         * download alone. */
        if (is_device) {
            for (size_t i = 0; i < num_outputs_; ++i) {
                if (i < gpu_out_tensors.size() && gpu_out_tensors[i]) {
                    output_tensors_[i]->copyToHostTensor(gpu_out_tensors[i]);
                }
            }
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        LOGD("MNN: run %d took %.3f ms", r, ms);

        /* Now time the remaining host-side work: the D2H copy the sync above
         * already produced (cached, so this is the memcpy into our snapshots)
         * plus the snapshot memcpy itself. */
        auto t_out0 = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < num_outputs_; ++i) {
            MNN::Tensor *t = output_tensors_[i];
            if (!t) {
                continue;
            }
            if (i < gpu_out_tensors.size() && gpu_out_tensors[i]) {
                /* GPU: use pre-allocated float host tensor for automatic
                 * FP16->FP32 conversion */
                t->copyToHostTensor(gpu_out_tensors[i]);
                float *hp = gpu_out_tensors[i]->host<float>();
                if (hp) {
                    memcpy(snaps[i].data(), hp,
                           output_elems_[i] * sizeof(float));
                }
            } else {
                /* CPU: direct copy from host memory */
                float *host = t->host<float>();
                if (host) {
                    memcpy(snaps[i].data(), host,
                           output_elems_[i] * sizeof(float));
                }
            }
        }
        auto t_out1 = std::chrono::high_resolution_clock::now();
        transfer_out_ms_ +=
            std::chrono::duration<double, std::milli>(t_out1 - t_out0).count();

        total += ms;
        if (ms > maxv) {
            maxv = ms;
            maxi = r;
        }
        if (ms < minv) {
            minv = ms;
        }
    }

    /* Normalize transfer time to avg ms per repeat. */
    if (repeat > 0) {
        transfer_in_ms_ /= (double)repeat;
        transfer_out_ms_ /= (double)repeat;
    }

    /* Cleanup pre-allocated GPU host tensors */
    for (auto *ht : gpu_out_tensors) {
        MNN::Tensor::destroy(ht);
    }
    for (auto *ht : gpu_in_tensors) {
        MNN::Tensor::destroy(ht);
    }

    odata.resize(num_outputs_);
    oelems.resize(num_outputs_);
    oshapes.resize(num_outputs_);
    odims.resize(num_outputs_);
    for (size_t i = 0; i < num_outputs_; ++i) {
        size_t n = output_elems_[i];
        float *buf = (float *)malloc(n * sizeof(float));
        if (!buf) {
            LOGE("MNN: malloc(%zu) failed at output %zu, due to %s, %d", n * sizeof(float), i, strerror(errno), errno);
            for (size_t j = 0; j < i; ++j) {
                free(odata[j]);
            }
            return false;
        }
        memcpy(buf, snaps[i].data(), n * sizeof(float));
        odata[i] = buf;
        oelems[i] = n;
        auto &sh = oshapes[i];
        sh.fill(0);
        for (size_t d = 0; d < output_shapes_[i].size() && d < MAX_DIMENSIONS; ++d) {
            sh[d] = (size_t)output_shapes_[i][d];
        }
        odims[i] = output_shapes_[i].size();
    }
    return true;
}

void MNNBackend::GetTiming(std::array<double, 10> &timing)
{
    timing.fill(0);
    timing[0] = init_ms_;
}

void MNNBackend::GetTransferTiming(double &transfer_in_ms,
                                   double &transfer_out_ms)
{
    transfer_in_ms = transfer_in_ms_;
    transfer_out_ms = transfer_out_ms_;
}

void MNNBackend::Cleanup()
{
    if (interp_ && session_) {
        interp_->releaseSession(session_);
        session_ = nullptr;
    }
    interp_ = nullptr;

    for (size_t i = 0; i < input_bufs_.size(); ++i) {
        if (input_bufs_[i] && !input_external_[i]) {
            free(input_bufs_[i]);
        }
    }
    input_bufs_.clear();
}

BackendPtr CreateMnnBackend(BackendId id)
{
    return std::make_unique<MNNBackend>(id);
}

#endif /* HAVE_MNN_BACKEND */
