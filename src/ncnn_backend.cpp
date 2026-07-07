/*============================================================================
 * ncnn_backend.cpp - NCNN backend (RAII)
 *============================================================================*/

#include "backend_interface.hpp"
#include "log.hpp"

#ifdef HAVE_NCNN_BACKEND

#include <ncnn/net.h>

/* ncnn/platform.h may define min/max macros and backend preprocessor names.
 * Undefine all potential conflicts before our C++ code. */
#ifdef min
#  undef min
#endif
#ifdef max
#  undef max
#endif
#ifdef NCNN_VULKAN
#  undef NCNN_VULKAN
#endif

#include <vector>
#include <string>
#include <cstring>
#include <cstdio>
#include <chrono>
#include <cerrno>

class NCNNBackend : public IBackend {
public:
    explicit NCNNBackend(BackendId id) { id_ = id; }
    ~NCNNBackend() override { Cleanup(); }

    bool Initialize(const char* model_path, int num_threads) override;
    bool QueryIOInfo(std::string& is, size_t& ie, std::string& os, size_t& oe) override;
    bool PrepareInputs(float*& fd, size_t& fe, const char* arg,
                       bool random, const float* const* ext, const size_t* extc) override;
    void SetSharedInput(const float* const* data, const size_t* counts) override;
    bool RunBenchmark(int warmup, int repeat, double& total, double& maxv,
                      double& minv, int& maxi, std::vector<float*>& odata,
                      std::vector<size_t>& oelems,
                      std::vector<std::array<size_t, MAX_DIMENSIONS>>& oshapes,
                      std::vector<size_t>& odims) override;
    void GetTiming(std::array<double, 10>& timing) override;
    bool SaveOutputs(const char* suffix) override;

private:
    void Cleanup();
    bool ReadShapesFile(const char* shapes_path);

    ncnn::Net* net_ = nullptr;
    int gpu_device_ = -1; /* -1 = CPU only */

    size_t num_inputs_ = 0;
    size_t num_outputs_ = 0;
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    std::vector<size_t> input_elems_;
    std::vector<size_t> output_elems_;
    std::vector<std::vector<int>> input_shapes_;
    std::vector<std::vector<int>> output_shapes_;

    std::vector<float*> input_bufs_;
    std::vector<size_t> input_buf_elems_;
    std::vector<bool> input_external_;

    double init_ms_ = 0;
};

/* ---------------------------------------------------------------------------
 * Read .ncnn.shapes file
 * -------------------------------------------------------------------------*/
bool NCNNBackend::ReadShapesFile(const char* shapes_path) {
    FILE* f = fopen(shapes_path, "r");
    if (!f) { LOGE("NCNN: cannot open shapes file: %s, due to %s, %d", shapes_path, strerror(errno), errno); return false; }

    int num_in = 0, num_out = 0;
    (void)num_in; (void)num_out; /* read from file but used only for validation */
    char line[512];

    while (fgets(line, sizeof(line), f)) {
        size_t len = strlen(line);
        while (len > 0 && (line[len-1] == '\n' || line[len-1] == '\r'))
            line[--len] = '\0';

        if (strncmp(line, "inputs=", 7) == 0) {
            num_in = atoi(line + 7);
        } else if (strncmp(line, "outputs=", 8) == 0) {
            num_out = atoi(line + 8);
        } else if (strncmp(line, "in", 2) == 0 && strncmp(line, "input", 5) != 0) {
            /* in0=1,4,2048,8 */
            char* eq = strchr(line, '=');
            if (!eq) continue;
            /* Name before = */
            std::string name(line, eq - line);
            input_names_.push_back(name);

            const char* dims = eq + 1;
            std::vector<int> shape;
            size_t elems = 1;
            const char* p = dims;
            while (*p) {
                int d = atoi(p);
                shape.push_back(d);
                elems *= (size_t)d;
                p = strchr(p, ',');
                if (!p) break;
                ++p;
            }
            input_shapes_.push_back(shape);
            input_elems_.push_back(elems);
        } else if (strncmp(line, "out", 3) == 0) {
            char* eq = strchr(line, '=');
            if (!eq) continue;
            std::string name(line, eq - line);
            output_names_.push_back(name);

            const char* dims = eq + 1;
            std::vector<int> shape;
            size_t elems = 1;
            const char* p = dims;
            while (*p) {
                int d = atoi(p);
                shape.push_back(d);
                elems *= (size_t)d;
                p = strchr(p, ',');
                if (!p) break;
                ++p;
            }
            output_shapes_.push_back(shape);
            output_elems_.push_back(elems);
        }
    }
    if (fclose(f) != 0)
        LOGW("NCNN: fclose(%s) failed: %s, %d", shapes_path, strerror(errno), errno);

    num_inputs_ = input_names_.size();
    num_outputs_ = output_names_.size();

    LOGI("NCNN: shapes loaded: %zu in, %zu out", num_inputs_, num_outputs_);
    return num_inputs_ > 0 && num_outputs_ > 0;
}

/* ---------------------------------------------------------------------------
 * Initialize
 * -------------------------------------------------------------------------*/
bool NCNNBackend::Initialize(const char* model_path, int num_threads) {
    auto t0 = std::chrono::high_resolution_clock::now();

    /* Derive paths: model.ncnn.param -> model.ncnn.bin, model.ncnn.shapes */
    std::string path(model_path);
    std::string base = path;
    /* Strip .param */
    if (base.length() > 6) {
        auto tail6 = base.substr(base.length() - 6);
        if (stricmp_(tail6.c_str(), ".param") == 0)
            base = base.substr(0, base.length() - 6);
    }
    /* Strip .ncnn if present (the file is model.ncnn.param, we want model) */
    if (base.length() > 5) {
        auto tail5 = base.substr(base.length() - 5);
        if (stricmp_(tail5.c_str(), ".ncnn") == 0)
            base = base.substr(0, base.length() - 5);
    }

    std::string bin_path = base + ".ncnn.bin";
    std::string shapes_path = base + ".shapes";

    /* Read shapes first */
    if (!ReadShapesFile(shapes_path.c_str())) return false;

    net_ = new ncnn::Net();
    net_->opt.num_threads = num_threads;
    net_->opt.lightmode = true;
    net_->opt.use_packing_layout = true;

    /* For Vulkan_FP16: prefer FP16-converted model.
     * The FP16 model has weights already quantized, giving smaller and
     * more accurate results than runtime FP32→FP16 conversion.
     * .param is shared with FP32 (identical structure), only .bin differs. */
    if (id_ == BackendId::NCNN_VULKAN_FP16) {
        std::string fp16_bin = base + "_fp16.ncnn.bin";
        FILE* test_fp16 = fopen(fp16_bin.c_str(), "r");
        if (test_fp16) {
            if (fclose(test_fp16) != 0)
                LOGW("NCNN: fclose(%s) failed: %s, %d", fp16_bin.c_str(), strerror(errno), errno);
            bin_path = fp16_bin;
            LOGI("NCNN: using FP16 weights: %s", fp16_bin.c_str());
        } else {
            LOGW("NCNN: FP16 weights not found (%s), falling back to FP32 "
                 "(run: python onnx_to_ncnn.py model.onnx --dual)",
                 fp16_bin.c_str());
        }
    }
    if (id_ == BackendId::NCNN_VULKAN || id_ == BackendId::NCNN_VULKAN_FP16) {
        gpu_device_ = ncnn::get_default_gpu_index();
        if (gpu_device_ < 0) {
            LOGW("NCNN: no Vulkan device found, falling back to CPU");
        } else {
            net_->opt.use_vulkan_compute = true;
            if (id_ == BackendId::NCNN_VULKAN) {
                /* Force FP32 path: disable all FP16 optimizations */
                net_->opt.use_fp16_packed     = false;
                net_->opt.use_fp16_storage    = false;
                net_->opt.use_fp16_arithmetic = false;
            } else {
                /* FP16 path */
                net_->opt.use_fp16_packed     = true;
                net_->opt.use_fp16_storage    = true;
                net_->opt.use_fp16_arithmetic = true;
            }
            LOGI("NCNN: Vulkan enabled (gpu=%d, fp16=%d)", gpu_device_,
                 net_->opt.use_fp16_packed);
        }
    }

    /* Load model */
    LOGI("NCNN: loading param: %s", path.c_str());
    if (net_->load_param(path.c_str()) != 0) {
        LOGE("NCNN: failed to load param: %s, due to %s, %d",
             path.c_str(), strerror(errno), errno);
        LOGE("NCNN: possible cause - missing/unsupported layer type, or file not found");
        return false;
    }
    LOGI("NCNN: param loaded, loading bin: %s", bin_path.c_str());
    if (net_->load_model(bin_path.c_str()) != 0) {
        LOGE("NCNN: failed to load bin: %s, due to %s, %d",
             bin_path.c_str(), strerror(errno), errno);
        return false;
    }
    LOGI("NCNN: model loaded successfully");

    init_ms_ = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - t0).count();

    LOGI("NCNN: init complete (%.1f ms)", init_ms_);
    return true;
}

bool NCNNBackend::QueryIOInfo(std::string& is, size_t& ie,
                               std::string& os, size_t& oe) {
    is.clear(); ie = 0;
    for (size_t i = 0; i < num_inputs_; ++i) {
        char buf[128] = {}; int off = snprintf(buf, sizeof(buf), "[");
        for (size_t d = 0; d < input_shapes_[i].size(); ++d)
            off += snprintf(buf + off, sizeof(buf) - off, "%s%d", d > 0 ? "," : "", input_shapes_[i][d]);
        snprintf(buf + off, sizeof(buf) - off, "]");
        if (i > 0) is += " "; is += buf; ie += input_elems_[i];
    }
    os.clear(); oe = 0;
    for (size_t i = 0; i < num_outputs_; ++i) {
        char buf[128] = {}; int off = snprintf(buf, sizeof(buf), "[");
        for (size_t d = 0; d < output_shapes_[i].size(); ++d)
            off += snprintf(buf + off, sizeof(buf) - off, "%s%d", d > 0 ? "," : "", output_shapes_[i][d]);
        snprintf(buf + off, sizeof(buf) - off, "]");
        if (i > 0) os += " "; os += buf; oe += output_elems_[i];
    }
    return true;
}

bool NCNNBackend::PrepareInputs(float*& fd, size_t& fe, const char* /*arg*/,
                                 bool random, const float* const* ext,
                                 const size_t* extc) {
    for (size_t i = 0; i < num_inputs_; ++i) {
        size_t n = input_elems_[i] > 0 ? input_elems_[i] : 1;
        if (i >= input_bufs_.size()) {
            input_bufs_.resize(i + 1, nullptr);
            input_buf_elems_.resize(i + 1, 0);
            input_external_.resize(i + 1, false);
        }
        if (input_bufs_[i] && !input_external_[i]) free(input_bufs_[i]);

        if (ext && ext[i] && extc && extc[i] == n) {
            input_bufs_[i] = const_cast<float*>(ext[i]);
            input_external_[i] = true;
        } else {
            float* buf = (float*)malloc(n * sizeof(float));
            if (!buf) return false;
            if (random) { for (size_t j = 0; j < n; ++j) buf[j] = (float)rand() / (float)RAND_MAX; }
            else { memset(buf, 0, n * sizeof(float)); }
            input_bufs_[i] = buf;
            input_external_[i] = false;
        }
        input_buf_elems_[i] = n;
    }
    fd = num_inputs_ > 0 ? input_bufs_[0] : nullptr;
    fe = num_inputs_ > 0 ? input_buf_elems_[0] : 0;
    return true;
}

void NCNNBackend::SetSharedInput(const float* const* data, const size_t* counts) {
    for (size_t i = 0; i < num_inputs_; ++i) {
        if (i >= input_bufs_.size()) {
            input_bufs_.resize(i + 1, nullptr);
            input_buf_elems_.resize(i + 1, 0);
            input_external_.resize(i + 1, false);
        }
        input_bufs_[i] = const_cast<float*>(data[i]);
        input_buf_elems_[i] = counts[i];
        input_external_[i] = true;
    }
}

bool NCNNBackend::RunBenchmark(int warmup, int repeat, double& total,
                                double& maxv, double& minv, int& maxi,
                                std::vector<float*>& odata,
                                std::vector<size_t>& oelems,
                                std::vector<std::array<size_t, MAX_DIMENSIONS>>& oshapes,
                                std::vector<size_t>& odims) {
    total = 0; maxv = 0; minv = 1e12; maxi = 0;
    if (num_inputs_ == 0 || num_outputs_ == 0) return false;

    std::vector<std::vector<float>> snaps(num_outputs_);
    for (size_t i = 0; i < num_outputs_; ++i)
        snaps[i].resize(output_elems_[i] > 0 ? output_elems_[i] : 1);

    /* Build ncnn::Mat inputs (NCNN uses [w,h,c] order internally) */
    auto build_inputs = [&]() -> std::vector<ncnn::Mat> {
        std::vector<ncnn::Mat> mats;
        for (size_t i = 0; i < num_inputs_; ++i) {
            auto& sh = input_shapes_[i];
            /* Convert ONNX [N,C,H,W] to NCNN [w,h,c] */
            int w = (int)sh[3], h = (int)sh[2], c = (int)sh[1];
            ncnn::Mat m(w, h, c, input_bufs_[i]);
            mats.push_back(m);
        }
        return mats;
    };

    (void)warmup;
    for (int w = 0; w < warmup; ++w) {
        ncnn::Extractor ex = net_->create_extractor();
        auto mats = build_inputs();
        for (size_t i = 0; i < num_inputs_; ++i)
            ex.input(input_names_[i].c_str(), mats[i]);
        for (size_t i = 0; i < num_outputs_; ++i) {
            ncnn::Mat out;
            ex.extract(output_names_[i].c_str(), out);
        }
    }

    for (int r = 0; r < repeat; ++r) {
        ncnn::Extractor ex = net_->create_extractor();
        auto mats = build_inputs();
        for (size_t i = 0; i < num_inputs_; ++i)
            ex.input(input_names_[i].c_str(), mats[i]);

        auto t0 = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < num_outputs_; ++i) {
            ncnn::Mat out;
            ex.extract(output_names_[i].c_str(), out);
            /* Convert NCNN [w,h,c] back to flat float */
            if (!out.empty()) {
                /* NCNN Mat is channels-first for continuous memory */
                memcpy(snaps[i].data(), out.channel(0),
                       std::min(snaps[i].size() * sizeof(float),
                                out.total() * sizeof(float)));
            }
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        total += ms;
        if (ms > maxv) { maxv = ms; maxi = r; }
        if (ms < minv) minv = ms;
    }

    odata.resize(num_outputs_); oelems.resize(num_outputs_);
    oshapes.resize(num_outputs_); odims.resize(num_outputs_);
    for (size_t i = 0; i < num_outputs_; ++i) {
        size_t n = output_elems_[i];
        float* buf = (float*)malloc(n * sizeof(float));
        if (!buf) { LOGE("NCNN: malloc(%zu) failed at output %zu, due to %s, %d", n * sizeof(float), i, strerror(errno), errno); return false; }
        memcpy(buf, snaps[i].data(), n * sizeof(float));
        odata[i] = buf; oelems[i] = n;
        auto& sh = oshapes[i]; sh.fill(0);
        for (size_t d = 0; d < output_shapes_[i].size() && d < MAX_DIMENSIONS; ++d)
            sh[d] = (size_t)output_shapes_[i][d];
        odims[i] = output_shapes_[i].size();
    }
    return true;
}

void NCNNBackend::GetTiming(std::array<double, 10>& timing) {
    timing.fill(0);
    timing[0] = init_ms_;
}

bool NCNNBackend::SaveOutputs(const char* /*suffix*/) { return true; }

void NCNNBackend::Cleanup() {
    if (net_) {
        if (gpu_device_ >= 0) {
            ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device(gpu_device_);
            (void)vkdev; /* Don't destroy - GPU might be shared */
        }
        delete net_;
        net_ = nullptr;
    }
    for (size_t i = 0; i < input_bufs_.size(); ++i)
        if (input_bufs_[i] && !input_external_[i]) free(input_bufs_[i]);
    input_bufs_.clear();
}

BackendPtr CreateNcnnBackend(BackendId id) {
    return std::make_unique<NCNNBackend>(id);
}

#endif /* HAVE_NCNN_BACKEND */
