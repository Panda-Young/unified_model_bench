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
    std::string input_list_path;                          /* --input-list: file of .bin input paths */
    InputDataFormat input_format = InputDataFormat::Auto; /* --input-format */
    std::string output_dir;
    std::string csv_path = "summary.csv";

    int repeat = 100;
    int warmup_runs = 1;
    int num_threads = 4;
    int log_level = 2; // 0=OFF, 1=DBG, 2=INFO, 3=WARN, 4=ERR

    bool enable_csv = true;

    /* Empty = run all available backends. Populated by --backend <name1,name2,...> */
    std::vector<int> backend_ids;

    /* Backends to EXCLUDE (blacklist). Populated by --no-backend <name1,...>.
     * Applied after the --backend whitelist filter. */
    std::vector<int> no_backend_ids;

    /* Internal (worker-process) flags, set only by the scheduler's child
     * command lines - not exposed in --help:
     *  - worker_mode:  this process is a child worker (run a single backend,
     *    no scheduling); without it the process is the scheduler that spawns
     *    one child process per (variant, backend)
     *  - batch_time:   frozen batch timestamp shared by all workers of one
     *    scheduler run, so one batch produces one CSV "time" value
     *  - dump_output:  after the run, write the first output tensor to this
     *    file (uint64 count + float data) plus a "<path>.avg" sidecar holding
     *    avg_run_ms; used by the baseline worker so the scheduler can hand
     *    the reference to the other workers
     *  - baseline_file: load a dumped baseline output and compute real
     *    max_diff / avg_diff instead of the no-baseline "-"
     *  - baseline_ms:   baseline avg_run_ms from the sidecar, for accel */
    bool worker_mode = false;
    std::string batch_time;
    std::string dump_output;
    std::string baseline_file;
    double baseline_ms = 0.0;

    bool valid() const { return !model_path.empty(); }
};

/* Returns true on success, false on error or --help */
bool parse_cmd_args(int argc, char *argv[], BenchConfig &cfg);
void print_usage(const char *prog);
