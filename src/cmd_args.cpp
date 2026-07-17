/*============================================================================
 * cmd_args.cpp - CLI argument parser
 *============================================================================*/

#include "cmd_args.hpp"
#include "backend_interface.hpp"
#include "log.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cctype>

void print_usage(const char *prog)
{
    printf("Unified Benchmark Tool - Multi-framework inference benchmark\n\n");
    printf("Usage: %s <model_path> [options]\n\n", prog);
    printf("Options:\n");
    printf("  --model <path>      Model file path\n");
    printf("  --input <path>      Input data file (binary float32)\n");
    printf("  --repeat <N>        Benchmark repeat count (default: 100)\n");
    printf("  --warmup <N>        Warmup runs (default: 1)\n");
    printf("  --threads <N>       Number of threads (default: 4)\n");
    printf("  --output-dir <dir>  Output directory for saved outputs\n");
    printf("  --csv <path>        CSV output path (default: summary.csv)\n");
    printf("  --save-input        Save generated input to file\n");
    printf("  --no-save-output    Don't save output tensors\n");
    printf("  --no-csv            Don't write CSV\n");
    printf("  --no-output-print   Don't print output summary\n");
    printf("  --backend <list>    Comma-separated backend names (default: all)\n");
    printf("                      Examples: --backend ONNX_CPU,MNN_OpenCL_FP16\n");
    printf("  --help              Show this help\n");
    printf("  --version           Show version\n");
}

bool parse_cmd_args(int argc, char *argv[], BenchConfig &cfg)
{
    bool has_backend_filter = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return false;
        }
        if (strcmp(argv[i], "--version") == 0) {
            printf("Unified Benchmark Tool v2.0 (C++)\n");
            return false;
        }
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            cfg.model_path = argv[++i];
        } else if (strcmp(argv[i], "--input") == 0 && i + 1 < argc) {
            cfg.input_path = argv[++i];
        } else if (strcmp(argv[i], "--repeat") == 0 && i + 1 < argc) {
            cfg.repeat = atoi(argv[++i]);
            if (cfg.repeat < 1) {
                cfg.repeat = 1;
            }
        } else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
            cfg.warmup_runs = atoi(argv[++i]);
            if (cfg.warmup_runs < 0) {
                cfg.warmup_runs = 0;
            }
        } else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            cfg.num_threads = atoi(argv[++i]);
            if (cfg.num_threads < 1) {
                cfg.num_threads = 1;
            }
        } else if (strcmp(argv[i], "--output-dir") == 0 && i + 1 < argc) {
            cfg.output_dir = argv[++i];
        } else if (strcmp(argv[i], "--csv") == 0 && i + 1 < argc) {
            cfg.csv_path = argv[++i];
        } else if (strcmp(argv[i], "--save-input") == 0) {
            cfg.save_input = true;
        } else if (strcmp(argv[i], "--no-save-output") == 0) {
            cfg.save_output = false;
        } else if (strcmp(argv[i], "--no-csv") == 0) {
            cfg.enable_csv = false;
        } else if (strcmp(argv[i], "--no-output-print") == 0) {
            cfg.enable_output = false;
        } else if (strcmp(argv[i], "--backend") == 0 && i + 1 < argc) {
            has_backend_filter = true;
            const char *list = argv[++i];
            const char *p = list;
            while (*p) {
                /* Skip leading spaces */
                while (*p == ' ') ++p;
                if (!*p) break;
                /* Find end of this token (comma or end) */
                const char *end = p;
                while (*end && *end != ',') ++end;
                /* Extract token */
                std::string token(p, end - p);
                if (!token.empty()) {
                    int found = BackendRegistry::FindByName(token.c_str());
                    if (found >= 0) {
                        cfg.backend_ids.push_back(found);
                    } else {
                        LOGW("Unknown backend name: %s", token.c_str());
                    }
                }
                p = (*end == ',') ? end + 1 : end;
            }
        } else if (argv[i][0] != '-') {
            /* Positional argument = model path */
            if (cfg.model_path.empty()) {
                cfg.model_path = argv[i];
            } else {
                LOGW("Ignoring extra positional argument: %s", argv[i]);
            }
        } else {
            LOGW("Unknown option: %s", argv[i]);
        }
    }

    if (cfg.model_path.empty()) {
        LOGE("No model path specified. Use --model <path> or positional arg.");
        print_usage(argv[0]);
        return false;
    }

    if (has_backend_filter && cfg.backend_ids.empty()) {
        LOGE("--backend specified but no valid backend names found");
        return false;
    }

    return true;
}
