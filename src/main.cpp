/*============================================================================
 * main.cpp - Entry point
 *============================================================================*/
#include "benchmark_runner.hpp"
#include "cmd_args.hpp"
#include "backend_interface.hpp"
#include "log.hpp"
#include <cstdio>
#include <exception>
#include <cstdlib>

int main(int argc, char* argv[]) {
    try {
        setvbuf(stderr, nullptr, _IONBF, 0);
        Logger::init("unified_bench");
        Logger::level = LogLevel::INFO;

        LOGI("Unified Benchmark Tool v2.0 (C++)  Arch: %s", ARCH_STR);

        BenchConfig cfg;
        if (!parse_cmd_args(argc, argv, cfg)) return 1;

        LOGI("Model: %s  Repeat: %d  Warmup: %d  Threads: %d",
             cfg.model_path.c_str(), cfg.repeat, cfg.warmup_runs, cfg.num_threads);

        BackendRegistry::InitDefaults();
        ResultCollector collector;
        BenchmarkRunner runner(cfg, collector);
        bool ok = runner.Run();
        if (!ok) LOGW("Completed with warnings");
        return ok ? 0 : 1;

    } catch (const std::exception& e) {
        fprintf(stderr, "FATAL: %s\n", e.what());
        return 2;
    } catch (...) {
        fprintf(stderr, "FATAL: unknown exception\n");
        return 3;
    }
}
