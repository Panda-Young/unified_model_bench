#pragma once
/*============================================================================
 * result_collector.hpp - Result collection, accuracy comparison & CSV export
 *============================================================================*/

#include "backend_interface.hpp"
#include "model_format.hpp"
#include "platform.hpp"
#include <array>
#include <cstdint>
#include <string>
#include <vector>

/* ---------------------------------------------------------------------------
 * OutputData - owns a float buffer
 * -------------------------------------------------------------------------*/
struct OutputData {
    std::vector<float> data;

    OutputData() = default;
    explicit OutputData(size_t n) : data(n) {}
    OutputData(const float *src, size_t n) : data(src, src + n) {}

    size_t size() const { return data.size(); }
    bool empty() const { return data.empty(); }
    float *ptr() { return data.data(); }
    const float *ptr() const { return data.data(); }
};

/* ---------------------------------------------------------------------------
 * Format baseline (per-format CPU reference output)
 * -------------------------------------------------------------------------*/
struct FormatBaseline {
    bool has_baseline = false;
    OutputData output;
    std::string output_name; /* name of the stored output, for name-based matching */
    double cpu_avg_ms = 0.0;
    BackendId cpu_backend_id = BackendId::ONNX_CPU;
};

/* Sentinel for BenchmarkRecord::acceleration_vs_cpu when the run succeeded but
 * there was no baseline to compare against (e.g. --no-baseline single-backend
 * runs). CSV writes "-" for the comparison columns while keeping the timing
 * columns intact. -1.0 remains the "failed" sentinel. */
constexpr double kNoBaselineAccel = -2.0;

/* ---------------------------------------------------------------------------
 * BenchmarkRecord - one row of CSV output
 * -------------------------------------------------------------------------*/
struct BenchmarkRecord {
    /* Model */
    std::string model_name;
    std::string input_shape_str;
    size_t input_elements = 0;
    std::string output_shape_str;
    size_t output_elements = 0;

    /* Benchmark config */
    int warmup_runs = 1;
    int repeat_runs = 100;
    int num_threads = 4;

    /* Performance */
    double total_run_ms = 0.0;
    double avg_run_ms = 0.0;
    double max_run_ms = 0.0;
    int max_run_idx = 0;
    double init_ms = 0.0;

    /* Accuracy vs baseline */
    double max_output_diff = 0.0;
    double avg_output_diff = 0.0;
    double acceleration_vs_cpu = 0.0;

    /* Memory (MB) - deployment info (directly measured, no estimation)
     * weight_mem_mb:      model weights (parsed for ONNX / NCNN, else file
     *                     size approximation), backend-independent
     * peak_mem_mb:        process peak working set / RSS after this run
     * resident_mem_mb:    process resident working set / RSS after this run */
    double weight_mem_mb = 0.0;
    double peak_mem_mb = 0.0;
    double resident_mem_mb = 0.0;

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
class ResultCollector
{
public:
    ResultCollector() { baselines_.resize(5); } /* UNKNOWN..MNN */

    /* Record management */
    void Add(const BenchmarkRecord &rec);
    size_t Count() const { return records_.size(); }
    const BenchmarkRecord &Get(size_t i) const { return records_[i]; }

    /* Baseline management */
    bool SetBaseline(ModelFormat fmt, const std::vector<float *> &data,
                     const std::vector<size_t> &elems, double cpu_avg_ms,
                     BackendId cpu_id, const std::vector<std::string> &names);
    bool HasBaseline(ModelFormat fmt) const;
    double GetCpuBaselineMs(ModelFormat fmt) const;

    /* Accuracy comparison */
    bool CompareWithBaseline(ModelFormat fmt, const std::vector<float *> &data,
                             const std::vector<size_t> &elems, double &max_diff,
                             double &avg_diff, int64_t &element_count,
                             const std::vector<std::string> &names);

    /* CSV export */
    bool ExportCsv(const char *path, const char *date, const char *time,
                   const char *app_name) const;
    bool AppendCsv(const BenchmarkRecord &rec, const char *path,
                   const char *date, const char *time, const char *app_name) const;

private:
    std::vector<BenchmarkRecord> records_;
    std::vector<FormatBaseline> baselines_;

    FormatBaseline *get_baseline(ModelFormat fmt);
    const FormatBaseline *get_baseline(ModelFormat fmt) const;
};
