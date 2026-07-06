#pragma once
/*============================================================================
 * benchmark_runner.hpp - Main benchmark orchestrator
 *============================================================================*/

#include "platform.hpp"
#include "cmd_args.hpp"
#include "result_collector.hpp"
#include "input_provider.hpp"
#include "model_loader.hpp"
#include "device_info.hpp"
#include <string>

class BenchmarkRunner {
public:
    BenchmarkRunner(const BenchConfig& cfg, ResultCollector& collector);

    /* Run the full benchmark suite */
    bool Run();

    /* Access results */
    const ResultCollector& Results() const { return collector_; }

private:
    const BenchConfig& cfg_;
    ResultCollector&   collector_;
    std::string        device_info_;
    std::string        arch_;
    std::string        batch_date_;  /* frozen once per Run() */
    std::string        batch_time_;

    /* Test one model variant across all its backends */
    bool TestVariant(const ModelSearchResult& variant,
                     const ModelSearchResult& ref_variant);

    /* Test one specific backend */
    bool TestBackend(const BackendConfig& bcfg,
                     const std::string& model_path,
                     ModelFormat fmt,
                     ModelFormat ref_fmt,
                     InputProvider& shared_inputs,
                     const std::string& ncnn_shapes_path);

    /* Get current date/time strings */
    static std::string current_date();
    static std::string current_time();
};
