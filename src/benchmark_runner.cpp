/*============================================================================
 * benchmark_runner.cpp - Main benchmark orchestrator (clean)
 *============================================================================*/

#include "benchmark_runner.hpp"
#include "backend_interface.hpp"
#include "input_provider.hpp"
#include "log.hpp"
#include "model_loader.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iomanip>
#include <sstream>

/* ---------------------------------------------------------------------------
 * Time helpers
 * -------------------------------------------------------------------------*/
static std::string now_date()
{
    time_t t = time(nullptr);
    struct tm b;
#ifdef _WIN32
    localtime_s(&b, &t);
#else
    localtime_r(&t, &b);
#endif
    char buf[16];
    strftime(buf, sizeof(buf), "%Y-%m-%d", &b);
    return buf;
}
static std::string now_time()
{
    time_t t = time(nullptr);
    struct tm b;
#ifdef _WIN32
    localtime_s(&b, &t);
#else
    localtime_r(&t, &b);
#endif
    char buf[16];
    strftime(buf, sizeof(buf), "%H:%M:%S", &b);
    return buf;
}

/* ---------------------------------------------------------------------------
 * Parse shape string like "[1,4,2048,8];[1,128,2,512]" into element counts
 * -------------------------------------------------------------------------*/
static std::vector<size_t> parse_input_element_counts(const std::string &shape_str)
{
    std::vector<size_t> counts;
    std::string s = shape_str;
    size_t pos = 0;
    while ((pos = s.find('[')) != std::string::npos) {
        auto end = s.find(']', pos);
        if (end == std::string::npos)
            break;
        std::string dims = s.substr(pos + 1, end - pos - 1);
        size_t elems = 1;
        size_t cp = 0;
        while (cp < dims.length()) {
            auto nc = dims.find(',', cp);
            if (nc == std::string::npos)
                nc = dims.length();
            int d = atoi(dims.substr(cp, nc - cp).c_str());
            if (d > 0)
                elems *= (size_t)d;
            cp = nc + 1;
        }
        counts.push_back(elems);
        s = s.substr(end + 1);
    }
    return counts;
}

static const char *app_name_str()
{
#if defined(_WIN64)
    return "unified_bench_win_x64.exe";
#elif defined(_WIN32)
    return "unified_bench_win_x86.exe";
#elif defined(__ANDROID__) || defined(__android__)
    return "unified_bench_arm64-v8a";
#else
    return "unified_bench_linux_x64";
#endif
}

/* ---------------------------------------------------------------------------
 * Constructor
 * -------------------------------------------------------------------------*/
BenchmarkRunner::BenchmarkRunner(const BenchConfig &cfg, ResultCollector &collector)
    : cfg_(cfg), collector_(collector)
{
    device_info_ = get_device_info_csv();
    arch_ = ARCH_STR;
}

/* ---------------------------------------------------------------------------
 * Run
 * -------------------------------------------------------------------------*/
bool BenchmarkRunner::Run()
{
    batch_date_ = now_date();
    batch_time_ = now_time();

    LOGI("========================================");
    LOGI("Unified Benchmark v2.0 (C++)");
    LOGI("========================================");

    auto bundle = search_model_variants(cfg_.model_path);
    auto variants = bundle.all_found();
    if (variants.empty()) {
        LOGE("No model variants found");
        return false;
    }

    /* Reference = first found variant */
    const ModelSearchResult *ref_var = variants[0];
    LOGI("Reference: %s (%s)", model_format_name(ref_var->format),
         ref_var->path.c_str());

    for (const auto *var : variants)
        TestVariant(*var, *ref_var);

    /* Records are already written incrementally via AppendCsv.
     * No final ExportCsv needed (it would truncate previous runs). */

    LOGI("========================================");
    LOGI("Done: %zu record(s)", collector_.Count());
    for (size_t i = 0; i < collector_.Count(); ++i) {
        auto &r = collector_.Get(i);
        LOGI("  %-18s avg=%.1f ms  diff=%.6f  accel=%.2fx",
             r.backend_name.c_str(), r.avg_run_ms,
             r.max_output_diff, r.acceleration_vs_cpu);
    }
    LOGI("========================================");
    return collector_.Count() > 0;
}

/* ---------------------------------------------------------------------------
 * TestVariant
 * -------------------------------------------------------------------------*/
bool BenchmarkRunner::TestVariant(const ModelSearchResult &variant,
                                  const ModelSearchResult &ref_variant)
{
    ModelFormat fmt = variant.format;
    ModelFormat ref_fmt = ref_variant.format;

    auto backends = BackendRegistry::GetAvailable(fmt);
    if (backends.empty()) {
        LOGW("No backends for %s", model_format_name(fmt));
        return false;
    }

    LOGI("=== %s: %zu backend(s) ===", variant.path.c_str(), backends.size());

    /* Find CPU baseline backend ID */
    BackendId cpu_id = BackendId::ONNX_CPU;
    for (auto &bc : backends) {
        if (bc.is_cpu_baseline) {
            cpu_id = bc.id;
            break;
        }
    }

    /* Generate shared deterministic inputs:
     * 1. Create a temp CPU backend to query shapes
     * 2. Parse shapes to get element counts per input
     * 3. Generate InputProvider with seed=42 */
    InputProvider shared;

    /* NCNN: read .ncnn.shapes */
    std::string ncnn_shapes;
    if (fmt == ModelFormat::NCNN) {
        std::string base = variant.path;
        /* Strip trailing .param or .ncnn.param to get base name */
        if (base.length() > 6) {
            auto tail6 = base.substr(base.length() - 6);
            if (stricmp_(tail6.c_str(), ".param") == 0)
                base = base.substr(0, base.length() - 6);
        }
        if (base.length() > 5) {
            auto tail5 = base.substr(base.length() - 5);
            if (stricmp_(tail5.c_str(), ".ncnn") == 0)
                base = base.substr(0, base.length() - 5);
        }
        ncnn_shapes = base + ".shapes";

        /* Read shapes file to get element counts */
        FILE *f = fopen(ncnn_shapes.c_str(), "r");
        if (!f) {
            LOGW("NCNN shapes file not found: %s, due to %s, %d", ncnn_shapes.c_str(), strerror(errno), errno);
        } else {
            char line[512];
            std::vector<size_t> sizes;
            while (fgets(line, sizeof(line), f)) {
                /* Skip the "inputs=N" / "outputs=N" header lines */
                if (strncmp(line, "inputs=", 7) == 0 || strncmp(line, "outputs=", 8) == 0)
                    continue;
                if (strncmp(line, "in", 2) == 0 && strchr(line, '=')) {
                    const char *dims = strchr(line, '=') + 1;
                    size_t elems = 1;
                    const char *p = dims;
                    while (*p) {
                        elems *= (size_t)atoi(p);
                        p = strchr(p, ',');
                        if (!p)
                            break;
                        ++p;
                    }
                    sizes.push_back(elems);
                }
            }
            fclose(f);
            shared.GenerateFromSizes(sizes);
        }
    }

    /* For non-NCNN: use temp backend to query shapes */
    if (shared.Empty() && fmt != ModelFormat::NCNN) {
        std::vector<BackendId> candidate_ids = {cpu_id};
        /* TFLite CPU backend may fail for models with Select TF ops (FlexErf).
         * Fall back to GPU delegate which handles Flex ops via GL shaders. */
        if (fmt == ModelFormat::TFLITE) {
            for (auto &bc : backends) {
                if (bc.id != cpu_id)
                    candidate_ids.push_back(bc.id);
            }
        }
        for (auto cid : candidate_ids) {
            LOGI("Creating temp backend (id=%d) to query shapes...", bid(cid));
            fflush(stderr);
            auto tmp = BackendRegistry::Create(cid);
            if (tmp && tmp->Initialize(variant.path.c_str(), cfg_.num_threads)) {
                std::string is, os;
                size_t ie, oe;
                tmp->QueryIOInfo(is, ie, os, oe);
                auto sizes = parse_input_element_counts(is);
                shared.GenerateFromSizes(sizes);
                break;
            }
        }
    }

    if (shared.Empty()) {
        LOGW("Cannot generate shared inputs for %s", model_format_name(fmt));
        return false;
    }

    /* Test each backend */
    bool any_ok = false;
    for (auto &bcfg : backends) {
        if (TestBackend(bcfg, variant.path, fmt, ref_fmt,
                        shared, ncnn_shapes))
            any_ok = true;
    }
    return any_ok;
}

/* ---------------------------------------------------------------------------
 * TestBackend
 * -------------------------------------------------------------------------*/
bool BenchmarkRunner::TestBackend(const BackendConfig &bcfg,
                                  const std::string &model_path,
                                  ModelFormat fmt, ModelFormat ref_fmt,
                                  InputProvider &shared,
                                  const std::string & /*ncnn_shapes*/)
{
    LOGI("--- %s ---", bcfg.name.c_str());

    auto backend = BackendRegistry::Create(bcfg.id);
    if (!backend) {
        LOGW("Create failed: %s", bcfg.name.c_str());
        return false;
    }

    if (!backend->Initialize(model_path.c_str(), cfg_.num_threads)) {
        LOGW("Init failed: %s", bcfg.name.c_str());
        return false;
    }

    /* Query IO info */
    std::string is, os;
    size_t ie, oe;
    backend->QueryIOInfo(is, ie, os, oe);

    /* Set shared inputs (ALL backends use shared inputs for fairness) */
    backend->SetSharedInput(shared.DataPtrs().data(),
                            shared.ElementCounts().data());

    /* Run benchmark */
    double total_ms, max_ms, min_ms;
    int max_idx;
    std::vector<float *> odata;
    std::vector<size_t> oelems;
    std::vector<std::array<size_t, MAX_DIMENSIONS>> oshapes;
    std::vector<size_t> odims;

    if (!backend->RunBenchmark(cfg_.warmup_runs, cfg_.repeat,
                               total_ms, max_ms, min_ms, max_idx,
                               odata, oelems, oshapes, odims)) {
        LOGW("Benchmark failed: %s", bcfg.name.c_str());
        return false;
    }

    double avg_ms = (cfg_.repeat > 0) ? total_ms / cfg_.repeat : 0;

    /* Baseline or comparison */
    double max_diff = 0, avg_diff = 0, accel = 1.0;
    int64_t elem_count = 0;

    float *cmp_data = odata.empty() ? nullptr : odata[0];
    size_t cmp_elems = oelems.empty() ? 0 : oelems[0];

    if (bcfg.is_cpu_baseline && fmt == ref_fmt) {
        /* Store as baseline */
        if (cmp_data) {
            collector_.SetBaseline(fmt, cmp_data, cmp_elems, avg_ms, bcfg.id);
            LOGI("Baseline set: %s avg=%.3f ms", bcfg.name.c_str(), avg_ms);
        }
    } else {
        /* Compare with baseline */
        if (collector_.HasBaseline(ref_fmt) && cmp_data) {
            collector_.CompareWithBaseline(ref_fmt, cmp_data, cmp_elems,
                                           max_diff, avg_diff, elem_count);
            double cpu_ms = collector_.GetCpuBaselineMs(ref_fmt);
            if (cpu_ms > 0)
                accel = cpu_ms / avg_ms;
        }
    }

    /* Timing breakdown */
    std::array<double, 10> timing;
    backend->GetTiming(timing);

    /* Notes */
    std::ostringstream notes;
    notes << std::fixed << std::setprecision(3);
    if (bcfg.type == BackendType::ONNX_EP) {
        notes << "load_lib=" << timing[1] << ",get_api=" << timing[2]
              << ",create_env=" << timing[3] << ",session_opts=" << timing[4]
              << ",config_opts=" << timing[5] << ",append_prov=" << timing[5]
              << ",create_session=" << timing[6] << ",get_io=" << timing[7];
    } else {
        notes << "load_lib=" << timing[0];
    }

    /* Record */
    BenchmarkRecord rec;
    rec.model_name = model_path;
    /* For NCNN: use .bin instead of .param (weights file varies, param is same) */
    if (fmt == ModelFormat::NCNN) {
        auto pos = rec.model_name.rfind(".ncnn.param");
        if (pos != std::string::npos) {
            if (bcfg.id == BackendId::NCNN_VULKAN_FP16)
                rec.model_name.replace(pos, 12, "_fp16.ncnn.bin");
            else
                rec.model_name.replace(pos, 12, ".ncnn.bin");
        }
    }
    rec.input_shape_str = is;
    rec.input_elements = ie;
    rec.output_shape_str = os;
    rec.output_elements = oe;
    rec.warmup_runs = cfg_.warmup_runs;
    rec.repeat_runs = cfg_.repeat;
    rec.num_threads = cfg_.num_threads;
    rec.total_run_ms = total_ms;
    rec.avg_run_ms = avg_ms;
    rec.max_run_ms = max_ms;
    rec.max_run_idx = max_idx;
    rec.init_ms = timing[0];
    rec.max_output_diff = max_diff;
    rec.avg_output_diff = avg_diff;
    rec.acceleration_vs_cpu = accel;
    rec.backend_name = bcfg.name;
    rec.device_info = device_info_;
    rec.arch = arch_;
    rec.app_name = app_name_str();
    rec.notes = notes.str();

    collector_.Add(rec);

    /* Append CSV (crash-safe), using frozen batch timestamp */
    if (cfg_.enable_csv)
        collector_.AppendCsv(rec, cfg_.csv_path.c_str(),
                             batch_date_.c_str(), batch_time_.c_str(),
                             app_name_str());

    /* Free output buffers */
    for (auto *p : odata)
        free(p);

    return true;
}
