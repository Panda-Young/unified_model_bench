#pragma once
/*============================================================================
 * cmd_args.hpp - CLI argument parsing
 *============================================================================*/

#include <string>
#include <vector>

/* Input data format for --input-list files */
enum class InputDataFormat { Auto = 0,
                             Float32,
                             UInt8 };

struct BenchConfig {
    std::string model_path;
    std::string input_path;
    std::string input_list_path;                          /* --input-list: file of .bin input paths */
    InputDataFormat input_format = InputDataFormat::Auto; /* --input-format */
    std::string output_dir;
    std::string csv_path = "summary.csv";

    int repeat = 100;
    int warmup_runs = 1;
    int num_threads = 4;
    int log_level = 2; // 0=OFF, 1=DBG, 2=INFO, 3=WARN, 4=ERR

    bool enable_csv = true;
    bool enable_output = true;
    bool save_input = false;
    bool save_output = true;

    /* Empty = run all available backends. Populated by --backend <name1,name2,...> */
    std::vector<int> backend_ids;

    /* Backends to EXCLUDE (blacklist). Populated by --no-backend <name1,...>.
     * Applied after the --backend whitelist filter. */
    std::vector<int> no_backend_ids;

    bool valid() const { return !model_path.empty(); }
};

/* Returns true on success, false on error or --help */
bool parse_cmd_args(int argc, char *argv[], BenchConfig &cfg);
void print_usage(const char *prog);
