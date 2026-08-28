/*============================================================================
 * scheduler.cpp - Per-backend process scheduler (worker spawn & bookkeeping)
 *============================================================================*/

#include "scheduler.hpp"

#include "benchmark_runner.hpp"
#include "csv_utils.hpp"
#include "log.hpp"
#include "model_loader.hpp"
#include "platform.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#ifndef _WIN32
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

/* ---------------------------------------------------------------------------
 * Process spawning (one worker process per backend)
 * -------------------------------------------------------------------------*/

/* Absolute path of the currently running executable, so the scheduler can
 * re-invoke itself for each backend. */
static std::string get_self_exe_path()
{
#ifdef _WIN32
    char buf[MAX_PATH_LEN];
    DWORD n = GetModuleFileNameA(nullptr, buf, (DWORD)sizeof(buf));
    return (n > 0 && n < sizeof(buf)) ? std::string(buf, n) : std::string();
#else
    char buf[MAX_PATH_LEN];
    ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (n > 0) {
        buf[n] = '\0';
        return std::string(buf);
    }
    return std::string();
#endif
}

/* Run one child process synchronously, return its exit code
 * (-1 when the process could not be spawned). */
static int spawn_process(const std::string &cmdline, std::string &err_out)
{
#ifdef _WIN32
    STARTUPINFOA si;
    memset(&si, 0, sizeof(si));
    si.cb = sizeof(si);
    PROCESS_INFORMATION pi;
    memset(&pi, 0, sizeof(pi));
    std::string mut = cmdline; /* CreateProcessA may modify the buffer */
    if (!CreateProcessA(nullptr, &mut[0], nullptr, nullptr, TRUE, 0,
                        nullptr, nullptr, &si, &pi)) {
        err_out = "CreateProcess failed, error=" + std::to_string((long long)GetLastError());
        return -1;
    }
    CloseHandle(pi.hThread);
    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD code = 0;
    GetExitCodeProcess(pi.hProcess, &code);
    CloseHandle(pi.hProcess);
    return (int)code;
#else
    pid_t pid = fork();
    if (pid < 0) {
        err_out = "fork failed";
        return -1;
    }
    if (pid == 0) {
        execl("/bin/sh", "sh", "-c", cmdline.c_str(), (char *)nullptr);
        _exit(127);
    }
    int status = 0;
    if (waitpid(pid, &status, 0) < 0) {
        return -1;
    }
    if (WIFEXITED(status)) {
        return WEXITSTATUS(status);
    }
    return -1;
#endif
}

/* Rebuild the child (worker) command line: this executable + the user's
 * original args (minus the backend-filter/model-path tokens) + the target
 * model variant + --worker --backend <name>. The scheduler appends
 * --batch-time and the baseline flags (--dump-output / --baseline-file /
 * --baseline-ms) after this. */
/* Options that take a separate value argument, i.e. "--opt value" (as opposed
 * to "--opt" flags or "--opt=value"). Their value must be carried over to the
 * child command line verbatim AND skipped by the positional-argument logic,
 * otherwise the value is mistaken for the model path (the first bare token
 * replaces the model path) - e.g. "--repeat 2" used to make the worker run
 * with model path "2", which failed as "no model variants found". */
bool takes_value_arg(const std::string &opt)
{
    static const char *const kValueOpts[] = {
        "--model", "--backend", "--no-backend", "--input-list",
        "--input-format", "--repeat", "--warmup", "--threads",
        "--csv", "--log-level", "--output-dir"};
    for (const char *o : kValueOpts) {
        if (opt == o) {
            return true;
        }
    }
    return false;
}

static std::string build_child_cmdline(const std::string &exe,
                                       const std::string &model_path,
                                       const std::string &backend_name,
                                       int argc, char **argv)
{
    std::string cmd = "\"" + exe + "\"";
    bool saw_positional = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--backend" || a == "--no-backend" || a == "--model") {
            ++i; /* skip the option's value - replaced below */
            continue;
        }
        /* Carry over "value" of any other value-taking option untouched, and
         * do not let it look like a positional argument. */
        if (takes_value_arg(a)) {
            cmd += " " + a;
            if (i + 1 < argc) {
                cmd += " \"" + std::string(argv[++i]) + "\"";
            }
            continue;
        }
        if (!a.empty() && a[0] != '-') {
            if (!saw_positional) { /* first positional = model path, replaced */
                saw_positional = true;
                continue;
            }
        }
        cmd += " " + a;
    }
    cmd += " \"" + model_path + "\"";
    cmd += " --worker";
    cmd += " --backend " + backend_name;
    return cmd;
}

/* ---------------------------------------------------------------------------
 * Backend list filtering
 * -------------------------------------------------------------------------*/
void apply_backend_blacklist(std::vector<BackendConfig> &backends,
                             const BenchConfig &cfg)
{
    if (cfg.no_backend_ids.empty()) {
        return;
    }
    std::vector<BackendConfig> kept;
    for (auto &b : backends) {
        bool excluded = false;
        for (int want : cfg.no_backend_ids) {
            if (bid(b.id) == want) {
                excluded = true;
                break;
            }
        }
        if (!excluded) {
            kept.push_back(b);
        }
    }
    if (kept.empty()) {
        LOGW("All backends excluded by --no-backend");
    }
    backends = std::move(kept);
}

void filter_backends_by_user(std::vector<BackendConfig> &backends,
                             const BenchConfig &cfg)
{
    if (!cfg.backend_ids.empty()) {
        std::vector<BackendConfig> filtered;
        for (auto &b : backends) {
            for (int want : cfg.backend_ids) {
                if (bid(b.id) == want) {
                    filtered.push_back(b);
                    break;
                }
            }
        }
        backends = std::move(filtered);
    }
    apply_backend_blacklist(backends, cfg);
}

void filter_qnn_context_backends(std::vector<BackendConfig> &backends,
                                 const std::string &path)
{
    bool is_model_so = (path.size() > 3 &&
                        stricmp_(path.c_str() + path.size() - 3, ".so") == 0);
    if (is_model_so) {
        return;
    }
    std::vector<BackendConfig> keep;
    for (auto &b : backends) {
        if (b.id == BackendId::QNN_SDK_HTP) {
            keep.push_back(b);
        }
    }
    backends = std::move(keep);
}

/* ---------------------------------------------------------------------------
 * Cross-process baseline output files (scheduler workers)
 *
 * The baseline child dumps its first output tensor to "<path>" as
 *   [uint64 element_count][float data...]
 * The baseline worker's avg_run_ms travels back to the scheduler via its own
 * CSV row (read by csv_read_avg_run_ms), NOT via a sidecar file - the
 * scheduler then passes it to the other workers with --baseline-ms for the
 * acceleration ratio.
 * -------------------------------------------------------------------------*/
bool write_output_file(const std::string &path, const float *data,
                       size_t count, double /*avg_ms*/)
{
    FILE *f = fopen(path.c_str(), "wb");
    if (!f) {
        LOGE("write_output_file: cannot open %s", path.c_str());
        return false;
    }
    uint64_t n = (uint64_t)count;
    fwrite(&n, sizeof(n), 1, f);
    if (count > 0) {
        fwrite(data, sizeof(float), count, f);
    }
    fclose(f);
    return true;
}

bool load_output_file(const std::string &path, std::vector<float> &out)
{
    out.clear();
    FILE *f = fopen(path.c_str(), "rb");
    if (!f) {
        return false;
    }
    uint64_t n = 0;
    if (fread(&n, sizeof(n), 1, f) != 1 || n > (1ull << 30)) {
        fclose(f);
        return false;
    }
    out.resize((size_t)n);
    if (n > 0 && fread(out.data(), sizeof(float), (size_t)n, f) != n) {
        fclose(f);
        return false;
    }
    fclose(f);
    return true;
}

std::string baseline_output_path(const std::string &model_path,
                                 const std::string &backend_name,
                                 const std::string &out_dir)
{
    std::string dir = out_dir.empty() ? std::string(".") : out_dir;
    std::string base = extract_base_name(model_path);
    return dir + PATH_SEP + base + "_" + backend_name + ".out";
}

/* ---------------------------------------------------------------------------
 * Record bookkeeping shared by the scheduler and the worker flow
 * -------------------------------------------------------------------------*/
const char *app_name_str()
{
#if defined(_WIN64)
    return "unified_bench_win_x64.exe";
#elif defined(_WIN32)
    return "unified_bench_win_x86.exe";
#elif defined(__ANDROID__) || defined(__android__)
    return "unified_bench_arm64-v8a";
#else
    return "unified_bench_linux_x64";
#endif
}

void init_record_common(BenchmarkRecord &rec, const BackendConfig &bcfg,
                        const std::string &model_path, ModelFormat fmt,
                        const BenchConfig &cfg)
{
    auto last_slash = model_path.find_last_of("/\\");
    rec.model_name = (last_slash != std::string::npos) ? model_path.substr(last_slash + 1) : model_path;
    if (fmt == ModelFormat::NCNN) {
        /* NCNN reports the weights file (.bin), not the .param; the VK FP16
         * variant uses its own converted weights file. */
        auto pos = rec.model_name.rfind(".ncnn.param");
        if (pos != std::string::npos) {
            if (bcfg.id == BackendId::NCNN_VK_FP16) {
                rec.model_name.replace(pos, 12, "_fp16.ncnn.bin");
            } else {
                rec.model_name.replace(pos, 12, ".ncnn.bin");
            }
        }
    }
    rec.warmup_runs = cfg.warmup_runs;
    rec.repeat_runs = cfg.repeat;
    /* QNN SDK HTP (offline context binary): the CSV "threads" column cannot
     * report the HTP compute threads - hvx_threads is a compile-time parameter
     * baked into the .serialized.bin and not readable at runtime. Use -1 so
     * the CSV prints "-" instead of a misleading CPU thread count. */
    rec.num_threads = (bcfg.id == BackendId::QNN_SDK_HTP) ? -1 : cfg.num_threads;
    rec.backend_name = bcfg.name;
}

void RecordFailure(ResultCollector &collector, const BackendConfig &bcfg,
                   const std::string &model_path, ModelFormat fmt,
                   const BenchConfig &cfg, const std::string &device_info,
                   const std::string &arch, const char *app,
                   const char *date, const char *time, const char *reason)
{
    if (!cfg.enable_csv) {
        return;
    }
    BenchmarkRecord rec;
    init_record_common(rec, bcfg, model_path, fmt, cfg);
    rec.device_info = device_info;
    rec.arch = arch;
    rec.app_name = app;
    rec.notes = csv_safe(reason);
    rec.acceleration_vs_cpu = -1.0; /* sentinel: no valid measurement */
    /* Memory fields stay 0.0; the CSV writer prints "-" for all memory
     * columns on failed records (no valid measurement). */
    collector.Add(rec);
    collector.AppendCsv(rec, cfg.csv_path.c_str(), date, time, app);
}

/* ---------------------------------------------------------------------------
 * RunPerProcess - the scheduler (the only mode of the tool)
 *
 * The CPU backend of the ENTRY model format is the SINGLE global accuracy
 * baseline: its output is dumped once and handed to EVERY other backend
 * worker (all model variants included) via --baseline-file, so every
 * backend's diff is measured against that reference - never against another
 * converted model format. E.g. entry test_model.onnx -> ONNX_CPU baseline,
 * test_model.mnn -> MNN_CPU baseline, test_model.ncnn.bin -> NCNN_CPU.
 *
 * For every (variant, backend) the runner spawns a FRESH worker process of
 * this executable with "--worker --backend <name> --batch-time <t>", so:
 *   - each backend measures peak/resident memory in a clean process (zero
 *     cross-backend residue from previously loaded frameworks/DLLs)
 *   - a backend that crashes (segfault / access violation / ncnn heap
 *     corruption) only kills its own worker; the others still finish
 *   - all workers of one scheduler run share a single batch timestamp, so the
 *     CSV "time" column is the same for the whole batch
 *   - an abnormal worker exit (crash, code != 0/1) is merged into the notes
 *     column of that backend's own row; only if the worker died BEFORE writing
 *     any record is a separate failure row appended.
 * -------------------------------------------------------------------------*/
bool BenchmarkRunner::RunPerProcess()
{
    auto bundle = search_model_variants(cfg_.model_path);
    auto variants = bundle.all_found();
    if (variants.empty()) {
        LOGE("No model variants found");
        return false;
    }

    std::string exe = get_self_exe_path();
    if (exe.empty()) {
        LOGE("Scheduler: cannot locate this executable");
        return false;
    }

    /* ---- 1) Global baseline: the CPU backend of the ENTRY model format ----
     * e.g. entry test_model.onnx -> ONNX_CPU, test_model.mnn -> MNN_CPU,
     * test_model.ncnn.bin -> NCNN_CPU, and so on. Its output is dumped once
     * and handed to EVERY other backend worker (all model variants included)
     * via --baseline-file, so every backend's diff is measured against the
     * entry-format CPU reference - never against another converted format. */
    ModelFormat entry_fmt = detect_model_format(cfg_.model_path);
    const ModelSearchResult *base_var = nullptr;
    for (const auto *var : variants) {
        if (var->format == entry_fmt) {
            base_var = var;
            break;
        }
    }
    std::string base_out;
    double baseline_ms = 0.0;
    bool base_ok = false;
    bool have_base = false;
    BackendConfig base_cfg;
    if (base_var) {
        auto bs = BackendRegistry::GetAvailable(base_var->format);
        /* The baseline is the accuracy REFERENCE, not a user-requested run:
         * ignore the --backend whitelist (so e.g. "--backend MNN_CPU" on an
         * ONNX entry model still gets its ONNX_CPU baseline), but respect
         * an explicit --no-backend exclusion. */
        apply_backend_blacklist(bs, cfg_);
        for (auto &b : bs) {
            if (b.is_cpu_baseline) {
                base_cfg = b;
                have_base = true;
                break;
            }
        }
        if (have_base) {
            base_out = baseline_output_path(base_var->path, base_cfg.name,
                                            cfg_.output_dir);
            std::string cmd = build_child_cmdline(exe, base_var->path,
                                                  base_cfg.name, argc_, argv_);
            cmd += " --batch-time \"" + batch_time_ + "\"";
            cmd += " --dump-output \"" + base_out + "\"";
            LOGI("=== global baseline: spawning %s ===", base_cfg.name.c_str());
            fflush(stderr);
            std::string err;
            int code = spawn_process(cmd, err);
            if (code == 0) {
                base_ok = true;
                /* avg_run_ms comes back via the baseline worker's own CSV
                 * row (it appended the record before exiting) - no sidecar. */
                baseline_ms = csv_read_avg_run_ms(cfg_.csv_path, batch_time_,
                                                  base_cfg.name);
                LOGI("--- %s: OK (baseline avg=%.3f ms) ---",
                     base_cfg.name.c_str(), baseline_ms);
            } else if (code == 1) {
                /* exit 1: the worker either recorded its own failure row, or
                 * died silently (e.g. shape probe failed) - make sure the CSV
                 * is not missing the backend entirely. */
                if (!csv_has_backend_record(cfg_.csv_path, batch_time_,
                                            base_cfg.name)) {
                    std::string reason =
                        "worker process exited with code 1 (no CSV record written)";
                    RecordFailure(collector_, base_cfg, base_var->path,
                                  base_var->format, cfg_, device_info_, arch_,
                                  app_name_str(), batch_date_.c_str(),
                                  batch_time_.c_str(), reason.c_str());
                    LOGW("--- %s: exit=1 but no CSV record - failure row appended ---",
                         base_cfg.name.c_str());
                } else {
                    LOGI("--- %s: expected failure (worker recorded it) ---",
                         base_cfg.name.c_str());
                }
                base_out.clear();
            } else {
                LOGW("--- %s: FAILED (exit=%d)%s%s ---", base_cfg.name.c_str(),
                     code, err.empty() ? "" : "; ", err.empty() ? "" : err.c_str());
                std::string reason = "worker process exited abnormally with code " +
                                     std::to_string(code) +
                                     (err.empty() ? "" : ("; " + err));
                if (!csv_append_note(cfg_.csv_path, batch_time_,
                                     base_cfg.name, reason)) {
                    RecordFailure(collector_, base_cfg, base_var->path,
                                  base_var->format, cfg_, device_info_, arch_,
                                  app_name_str(), batch_date_.c_str(),
                                  batch_time_.c_str(), reason.c_str());
                }
                base_out.clear();
            }
        }
    }
    if (!have_base) {
        LOGW("No %s_CPU baseline available - accuracy diff will be '-' for all backends",
             model_format_name(entry_fmt));
    }

    /* ---- 2) All variants x backends, one worker process each ---- */
    bool any_ok = false;
    auto spawn_worker = [&](const ModelSearchResult &var, const BackendConfig &bcfg) {
        std::string cmd = build_child_cmdline(exe, var.path, bcfg.name, argc_, argv_);
        cmd += " --batch-time \"" + batch_time_ + "\"";
        if (base_ok && !base_out.empty()) {
            cmd += " --baseline-file \"" + base_out + "\"";
            if (baseline_ms > 0.0) {
                std::ostringstream ms;
                ms << std::fixed << std::setprecision(3) << baseline_ms;
                cmd += " --baseline-ms " + ms.str();
            }
        }
        LOGI("--- spawning %s ---", bcfg.name.c_str());
        fflush(stderr);
        std::string err;
        int code = spawn_process(cmd, err);
        if (code == 0) {
            any_ok = true;
            LOGI("--- %s: OK ---", bcfg.name.c_str());
        } else if (code == 1) {
            /* exit 1 = expected failure: the worker usually wrote its own
             * failure row (Initialize/RunBenchmark error). But a worker can
             * also exit 1 WITHOUT recording anything (e.g. TestVariant shape
             * probe failed) - detect that and append a failure row so no
             * backend silently disappears from the CSV. */
            if (!csv_has_backend_record(cfg_.csv_path, batch_time_,
                                        bcfg.name)) {
                std::string reason =
                    "worker process exited with code 1 (no CSV record written)";
                RecordFailure(collector_, bcfg, var.path, var.format, cfg_,
                              device_info_, arch_, app_name_str(),
                              batch_date_.c_str(), batch_time_.c_str(),
                              reason.c_str());
                LOGW("--- %s: exit=1 but no CSV record - failure row appended ---",
                     bcfg.name.c_str());
            } else {
                LOGI("--- %s: expected failure (worker recorded it) ---",
                     bcfg.name.c_str());
            }
        } else {
            LOGW("--- %s: FAILED (exit=%d)%s%s ---", bcfg.name.c_str(), code,
                 err.empty() ? "" : "; ", err.empty() ? "" : err.c_str());
            /* Abnormal exit (crash): merge the exit code into the notes column
             * of the worker's own row; append a failure row only when the
             * worker died before writing any record. */
            std::string reason = "worker process exited abnormally with code " +
                                 std::to_string(code) +
                                 (err.empty() ? "" : ("; " + err));
            if (!csv_append_note(cfg_.csv_path, batch_time_, bcfg.name, reason)) {
                RecordFailure(collector_, bcfg, var.path, var.format, cfg_,
                              device_info_, arch_, app_name_str(),
                              batch_date_.c_str(), batch_time_.c_str(),
                              reason.c_str());
            }
        }
    };

    for (const auto *var : variants) {
        auto backends = BackendRegistry::GetAvailable(var->format);
        if (backends.empty()) {
            continue;
        }
        /* QNN context binaries are backend-specific: keep HTP only (matches
         * the in-process TestVariant behavior). */
        if (var->format == ModelFormat::QNN) {
            filter_qnn_context_backends(backends, var->path);
        }
        filter_backends_by_user(backends, cfg_);
        if (backends.empty()) {
            continue;
        }

        LOGI("=== %s: %zu backend(s), one process each ===",
             var->path.c_str(), backends.size());
        for (auto &bcfg : backends) {
            if (have_base && bcfg.id == base_cfg.id) {
                continue; /* global baseline already ran */
            }
            spawn_worker(*var, bcfg);
        }
    }

    /* ---- 3) Cleanup the temporary baseline handoff file ---- */
    if (base_ok && !base_out.empty()) {
        remove(base_out.c_str());
        LOGI("Baseline temp files cleaned: %s", base_out.c_str());
    }
    return any_ok;
}
