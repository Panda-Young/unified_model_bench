/*============================================================================
 * main.cpp - Entry point
 *============================================================================*/
#include "backend_interface.hpp"
#include "benchmark_runner.hpp"
#include "cmd_args.hpp"
#include "log.hpp"
#include <cstdio>
#include <cstdlib>
#include <exception>

int main(int argc, char *argv[])
{
    try {
        setvbuf(stderr, nullptr, _IONBF, 0);
        Logger::init("unified_bench");

        BenchConfig cfg;

        BackendRegistry::InitDefaults();

        if (!parse_cmd_args(argc, argv, cfg)) {
            return 1;
        }

        Logger::level = static_cast<LogLevel>(cfg.log_level);

        LOGI("Unified Benchmark Tool v2.0 (C++)  Arch: %s", ARCH_STR);

        LOGI("Model: %s  Repeat: %d  Warmup: %d  Threads: %d",
             cfg.model_path.c_str(), cfg.repeat, cfg.warmup_runs, cfg.num_threads);

        ResultCollector collector;
        BenchmarkRunner runner(cfg, collector, argc, argv);
        bool ok = runner.Run();
        if (!ok) {
            LOGW("Completed with warnings");
        }
        return ok ? 0 : 1;

    } catch (const std::exception &e) {
        fprintf(stderr, "FATAL: %s\n", e.what());
        return 2;
    } catch (...) {
        fprintf(stderr, "FATAL: unknown exception\n");
        return 3;
    }
}
