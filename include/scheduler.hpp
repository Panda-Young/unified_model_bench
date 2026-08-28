#pragma once
/*============================================================================
 * scheduler.hpp - Per-backend process scheduler helpers
 *
 * Owns the "one worker process per (variant, backend)" machinery: child
 * process spawning, backend filtering, cross-process baseline output files
 * and failure-row recording. BenchmarkRunner::RunPerProcess() (declared in
 * benchmark_runner.hpp) is implemented in scheduler.cpp.
 *============================================================================*/

#include "backend_interface.hpp"
#include "cmd_args.hpp"
#include "model_format.hpp"
#include "result_collector.hpp"

#include <string>
#include <vector>

/* Apply the user's --no-backend blacklist only. Used for the baseline
 * selection, which is the accuracy reference rather than a user-requested
 * run: it must ignore the --backend whitelist but still honor exclusions. */
void apply_backend_blacklist(std::vector<BackendConfig> &backends,
                             const BenchConfig &cfg);

/* Apply the user's --backend (whitelist) / --no-backend (blacklist) filters.
 * STRICT: when the whitelist matches nothing, the result is empty (the caller
 * skips the variant) - a name that is valid in the registry but unavailable
 * for this model format must not silently re-enable everything. */
void filter_backends_by_user(std::vector<BackendConfig> &backends,
                             const BenchConfig &cfg);

/* Whether a command-line option takes a separate value argument
 * ("--repeat 2" as opposed to "--no-csv" / "--repeat=2").
 *
 * The scheduler must carry such values over to the worker command line AND
 * skip them in the positional-argument logic, otherwise the value is mistaken
 * for the model path: "--repeat 2" used to produce a worker running with model
 * path "2", failing as "no model variants found". Only --backend/--no-backend/
 * --model were skipped before, so every other option with a value was broken.
 * Exposed for unit testing. */
bool takes_value_arg(const std::string &opt);

/* QNN context binaries are offline-compiled and backend-specific: keep
 * QNN_SDK_HTP only. A model.so (lib{base}.so) is runtime-composed and runs
 * on all QNN backends (CPU/GPU/HTP), so it keeps the full list. */
void filter_qnn_context_backends(std::vector<BackendConfig> &backends,
                                 const std::string &path);

/* Dump the first output tensor to "<path>" as
 *   [uint64 element_count][float data...]
 * Cross-process baseline handoff file (see RunPerProcess). */
bool write_output_file(const std::string &path, const float *data,
                       size_t count, double avg_ms);

/* Load a dumped baseline output file (uint64 count + float data). */
bool load_output_file(const std::string &path, std::vector<float> &out);

/* Default location of the dumped baseline output of one (variant, backend):
 * "<dir>/<base>_<backend>.out" (dir = --output-dir or the current directory). */
std::string baseline_output_path(const std::string &model_path,
                                 const std::string &backend_name,
                                 const std::string &out_dir);

/* Executable name for the CSV "app_name" column. */
const char *app_name_str();

/* Fill the fields shared by failure rows and success rows: model name
 * (directory stripped; NCNN reports the weights file, .bin, not the .param),
 * benchmark config, and the QNN HTP thread-count special case. */
void init_record_common(BenchmarkRecord &rec, const BackendConfig &bcfg,
                        const std::string &model_path, ModelFormat fmt,
                        const BenchConfig &cfg);

/* Record a failed backend into the collector and append its CSV row
 * (failure sentinel: accel = -1.0, memory columns printed as "-"). */
void RecordFailure(ResultCollector &collector, const BackendConfig &bcfg,
                   const std::string &model_path, ModelFormat fmt,
                   const BenchConfig &cfg, const std::string &device_info,
                   const std::string &arch, const char *app,
                   const char *date, const char *time, const char *reason);
