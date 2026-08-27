#pragma once
/*============================================================================
 * result_collector.hpp - Result collection & CSV export
 *============================================================================*/

#include <cstdint>
#include <string>
#include <vector>

/* Sentinel for BenchmarkRecord::acceleration_vs_cpu when the run succeeded but
 * there was no baseline to compare against (e.g. --no-baseline single-backend
 * runs). CSV writes "-" for the comparison columns while keeping the timing
 * columns intact. -1.0 remains the "failed" sentinel. */
constexpr double kNoBaselineAccel = -2.0;

/* Number of columns in the CSV schema (must match kCsvHeader in
 * result_collector.cpp). Kept in the header so any consumer can validate
 * parsed rows against it; result_collector.cpp static_asserts on it. */
constexpr size_t kCsvExpectedColumns = 29;

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

    /* Tensor transfer (CPU<->device) time, average ms per repeat.
     * See IBackend::GetTransferTiming() for per-backend semantics.
     * transfer_in_ms:  H2D input upload
     * transfer_out_ms: D2H output download (incl. snapshot memcpy)
     * transfer_total_ms: transfer_in_ms + transfer_out_ms (convenience) */
    double transfer_in_ms = 0.0;
    double transfer_out_ms = 0.0;
    double transfer_total_ms = 0.0;

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
 *
 * Accuracy comparison is NOT done here anymore: since the per-backend-process
 * scheduler architecture, the reference output is dumped to a file by the
 * baseline worker and compared inside each worker (benchmark_runner.cpp).
 * This class only aggregates records and appends them to the CSV.
 * -------------------------------------------------------------------------*/
class ResultCollector
{
public:
    /* Record management */
    void Add(const BenchmarkRecord &rec);
    size_t Count() const { return records_.size(); }
    const BenchmarkRecord &Get(size_t i) const { return records_[i]; }

    /* CSV export (append-only; the header is written when the file is new) */
    bool AppendCsv(const BenchmarkRecord &rec, const char *path,
                   const char *date, const char *time,
                   const char *app_name) const;

private:
    std::vector<BenchmarkRecord> records_;
};
