#pragma once
/*============================================================================
 * result_collector.hpp - Result collection, accuracy comparison & CSV export
 *============================================================================*/

#include "platform.hpp"
#include "model_format.hpp"
#include "backend_interface.hpp"
#include <string>
#include <vector>
#include <array>
#include <cstdint>

/* ---------------------------------------------------------------------------
 * OutputData - owns a float buffer
 * -------------------------------------------------------------------------*/
struct OutputData {
    std::vector<float> data;

    OutputData() = default;
    explicit OutputData(size_t n) : data(n) {}
    OutputData(const float* src, size_t n) : data(src, src + n) {}

    size_t size() const { return data.size(); }
    bool   empty() const { return data.empty(); }
    float* ptr()        { return data.data(); }
    const float* ptr() const { return data.data(); }
};

/* ---------------------------------------------------------------------------
 * Format baseline (per-format CPU reference output)
 * -------------------------------------------------------------------------*/
struct FormatBaseline {
    bool       has_baseline = false;
    OutputData output;
    double     cpu_avg_ms = 0.0;
    BackendId  cpu_backend_id = BackendId::ONNX_CPU;
};

/* ---------------------------------------------------------------------------
 * BenchmarkRecord - one row of CSV output
 * -------------------------------------------------------------------------*/
struct BenchmarkRecord {
    /* Model */
    std::string model_name;
    std::string input_shape_str;
    size_t      input_elements = 0;
    std::string output_shape_str;
    size_t      output_elements = 0;

    /* Benchmark config */
    int warmup_runs = 1;
    int repeat_runs = 100;
    int num_threads = 4;

    /* Performance */
    double total_run_ms = 0.0;
    double avg_run_ms   = 0.0;
    double max_run_ms   = 0.0;
    int    max_run_idx  = 0;
    double init_ms      = 0.0;

    /* Accuracy vs baseline */
    double max_output_diff = 0.0;
    double avg_output_diff = 0.0;
    double acceleration_vs_cpu = 0.0;

    /* Metadata */
    std::string backend_name;
    std::string device_info;
    std::string arch;
    std::string app_name;
    std::string notes;
};

/* ---------------------------------------------------------------------------
 * ResultCollector
 * -------------------------------------------------------------------------*/
class ResultCollector {
public:
    ResultCollector() { baselines_.resize(5); }  /* UNKNOWN..MNN */

    /* Record management */
    void Add(const BenchmarkRecord& rec);
    size_t Count() const { return records_.size(); }
    const BenchmarkRecord& Get(size_t i) const { return records_[i]; }

    /* Baseline management */
    bool SetBaseline(ModelFormat fmt, const float* data, size_t n,
                     double cpu_avg_ms, BackendId cpu_id);
    bool HasBaseline(ModelFormat fmt) const;
    double GetCpuBaselineMs(ModelFormat fmt) const;

    /* Accuracy comparison */
    bool CompareWithBaseline(ModelFormat fmt, const float* data, size_t n,
                             double& max_diff, double& avg_diff,
                             int64_t& element_count);

    /* CSV export */
    bool ExportCsv(const char* path, const char* date, const char* time,
                   const char* app_name) const;
    bool AppendCsv(const BenchmarkRecord& rec, const char* path,
                   const char* date, const char* time, const char* app_name) const;

private:
    std::vector<BenchmarkRecord> records_;
    std::vector<FormatBaseline>  baselines_;

    FormatBaseline* get_baseline(ModelFormat fmt);
    const FormatBaseline* get_baseline(ModelFormat fmt) const;
};
