#pragma once
/*============================================================================
 * cmd_args.hpp - CLI argument parsing
 *============================================================================*/

#include <string>

struct BenchConfig {
    std::string model_path;
    std::string input_path;
    std::string output_dir;
    std::string csv_path = "summary.csv";

    int repeat = 100;
    int warmup_runs = 1;
    int num_threads = 4;

    bool enable_csv = true;
    bool enable_output = true;
    bool save_input = false;
    bool save_output = true;

    bool valid() const { return !model_path.empty(); }
};

/* Returns true on success, false on error or --help */
bool parse_cmd_args(int argc, char *argv[], BenchConfig &cfg);
void print_usage(const char *prog);
