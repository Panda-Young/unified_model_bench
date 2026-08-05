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

private:
    void Cleanup();

    std::shared_ptr<MNN::Interpreter> interp_;
    MNN::Session *session_ = nullptr;
    MNN::ScheduleConfig sched_;
    MNN::BackendConfig bcfg_;

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

    LOGI("MNN: forward=%d threads=%d precision=%d",
         sched_.type, num_threads, (int)bcfg_.precision);

    interp_ = std::shared_ptr<MNN::Interpreter>(
        MNN::Interpreter::createFromFile(model_path));
    if (!interp_) {
        LOGE("MNN: createFromFile failed");
        last_error_ = "MNN: createFromFile failed";
        return false;
    }

    session_ = interp_->createSession(sched_);
    if (!session_) {
        LOGE("MNN: createSession failed");
        interp_ = nullptr;
        last_error_ = "MNN: createSession failed";
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

    init_ms_ = std::chrono::duration<double, std::milli>(
                   std::chrono::high_resolution_clock::now() - t0)
                   .count();

    LOGI("MNN: init complete (%.1f ms), %zu in, %zu out", init_ms_, num_inputs_, num_outputs_);
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

bool MNNBackend::PrepareInputs(float *&fd, size_t &fe, const char * /*arg*/,
                               bool random, const float *const *ext,
                               const size_t *extc)
{
    for (size_t i = 0; i < num_inputs_; ++i) {
        size_t n = input_elems_[i] > 0 ? input_elems_[i] : 1;
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
            float *buf = (float *)malloc(n * sizeof(float));
            if (!buf) {
                return false;
            }
            if (random) {
                for (size_t j = 0; j < n; ++j) {
                    buf[j] = (float)rand() / (float)RAND_MAX;
                }
            } else {
                memset(buf, 0, n * sizeof(float));
            }
            input_bufs_[i] = buf;
            input_external_[i] = false;
        }
        input_buf_elems_[i] = n;
    }
    fd = num_inputs_ > 0 ? input_bufs_[0] : nullptr;
    fe = num_inputs_ > 0 ? input_buf_elems_[0] : 0;
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
    bool is_gpu = (sched_.type != MNN_FORWARD_CPU);
    std::vector<MNN::Tensor *> gpu_out_tensors;
    if (is_gpu) {
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
    if (is_gpu) {
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
            if (host && !is_gpu) {
                memcpy(host, input_bufs_[i], n * sizeof(float));
                continue;
            }
            /* GPU input: use pre-allocated host tensor with copyFromHostTensor
             * for automatic FP32→FP16 conversion */
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
        if (r > 0) {
            interp_->resizeSession(session_);
            feed_inputs();
        }

        auto t0 = std::chrono::high_resolution_clock::now();
        if (interp_->runSession(session_) != MNN::NO_ERROR) {
            LOGE("MNN: run %d failed", r);
            return false;
        }
        /* Snapshot outputs */
        for (size_t i = 0; i < num_outputs_; ++i) {
            MNN::Tensor *t = output_tensors_[i];
            if (!t) {
                continue;
            }
            if (i < gpu_out_tensors.size() && gpu_out_tensors[i]) {
                /* GPU: use pre-allocated float host tensor for automatic
                 * FP16→FP32 conversion */
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
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        total += ms;
        if (ms > maxv) {
            maxv = ms;
            maxi = r;
        }
        if (ms < minv) {
            minv = ms;
        }
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

bool MNNBackend::SaveOutputs(const char * /*suffix*/) { return true; }

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
