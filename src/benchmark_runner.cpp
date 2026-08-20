/*============================================================================
 * benchmark_runner.cpp - Main benchmark orchestrator (clean)
 *============================================================================*/

#include "benchmark_runner.hpp"
#include "backend_interface.hpp"
#include "input_provider.hpp"
#include "log.hpp"
#include "model_loader.hpp"

#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iomanip>
#include <sstream>

#ifndef _WIN32
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

/* ---------------------------------------------------------------------------
 * scheduler helpers (one worker process per backend)
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
            ++i; /* skip the option's value */
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

/* Apply the user's --backend / --no-backend filters (no auto-added baseline). */
static void filter_backends_by_user(std::vector<BackendConfig> &backends,
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
        if (!filtered.empty()) {
            backends = std::move(filtered);
        }
    }
    if (!cfg.no_backend_ids.empty()) {
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
        if (!kept.empty()) {
            backends = std::move(kept);
        }
    }
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
static bool write_output_file(const std::string &path, const float *data,
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

static bool load_output_file(const std::string &path, std::vector<float> &out)
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

/* Check whether the CSV already contains any row for this (batch_time,
 * backend). Used by the scheduler to decide whether a worker that exited
 * with code 1 actually wrote its own failure row (expected failure) or died
 * without recording anything (silent loss - a failure row must be added). */
static bool csv_has_backend_record(const std::string &path,
                                   const std::string &batch_time,
                                   const std::string &backend_name)
{
    FILE *f = fopen(path.c_str(), "r");
    if (!f) {
        return false;
    }
    char line[8192];
    bool found = false;
    while (fgets(line, sizeof(line), f)) {
        if (strstr(line, batch_time.c_str()) &&
            strstr(line, backend_name.c_str())) {
            found = true;
            break;
        }
    }
    fclose(f);
    return found;
}

/* Default location for the dumped baseline output of one (variant, backend):
 * "<dir>/<base>_<backend>.out" (dir = --output-dir or the current directory). */
static std::string baseline_output_path(const std::string &model_path,
                                        const std::string &backend_name,
                                        const std::string &out_dir)
{
    std::string dir = out_dir.empty() ? std::string(".") : out_dir;
    std::string base = extract_base_name(model_path);
    return dir + PATH_SEP + base + "_" + backend_name + ".out";
}

/* Forward declaration: used by the scheduler (RunPerProcess) to record failed
 * worker processes into the CSV. */
static void RecordFailure(ResultCollector &collector, const BackendConfig &bcfg,
                          const std::string &model_path, ModelFormat fmt,
                          const BenchConfig &cfg, const std::string &device_info,
                          const std::string &arch, const char *app,
                          const char *date, const char *time,
                          const char *reason);

/* ---------------------------------------------------------------------------
 * CSV row helpers - parse/serialize one CSV line (quote-aware) and append a
 * note to the notes column of an existing record. Used to merge an abnormal
 * worker exit code into the worker's own row instead of adding a second row.
 * -------------------------------------------------------------------------*/
static std::vector<std::string> csv_parse_line(const std::string &line)
{
    std::vector<std::string> out;
    std::string cur;
    bool in_q = false;
    size_t i = 0;
    const size_t n = line.size();
    while (i < n) {
        char c = line[i];
        if (in_q) {
            if (c == '"') {
                if (i + 1 < n && line[i + 1] == '"') { /* escaped "" */
                    cur += '"';
                    i += 2;
                    continue;
                }
                in_q = false;
            } else {
                cur += c;
            }
            ++i;
        } else if (c == '"') {
            in_q = true;
            ++i;
        } else if (c == ',') {
            out.push_back(cur);
            cur.clear();
            ++i;
        } else if (c == '\r' || c == '\n') {
            ++i;
        } else {
            cur += c;
            ++i;
        }
    }
    out.push_back(cur);
    return out;
}

/* Read avg_run_ms of the (batch_time, backend) row from the CSV (column 11
 * of the 26-column schema; backend_name is column 21). The baseline worker
 * appends its own record - including avg_run_ms - before exiting, so the
 * scheduler can read it straight back into a double variable and hand it to
 * the other workers via --baseline-ms. No .avg sidecar file is needed.
 * Returns 0.0 when the row is missing or holds "-" (failed baseline). */
static double csv_read_avg_run_ms(const std::string &path,
                                  const std::string &batch_time,
                                  const std::string &backend_name)
{
    FILE *f = fopen(path.c_str(), "r");
    if (!f) {
        return 0.0;
    }
    char line[8192];
    double avg = 0.0;
    while (fgets(line, sizeof(line), f)) {
        auto flds = csv_parse_line(line);
        if (flds.size() < 26 || flds[1] != batch_time ||
            flds[21] != backend_name) {
            continue;
        }
        if (flds[11].size() > 0 && flds[11] != "-") {
            avg = atof(flds[11].c_str());
        }
        break;
    }
    fclose(f);
    return avg;
}

static std::string csv_quote_field(const std::string &s)
{
    if (s.find_first_of(",\"\r\n") == std::string::npos) {
        return s;
    }
    std::string q = "\"";
    for (char c : s) {
        if (c == '"') {
            q += "\"\"";
        } else {
            q += c;
        }
    }
    q += '"';
    return q;
}

/* Append "note" to the notes column of the record whose (time, backend_name)
 * match. Returns true when a row was modified. */
static bool csv_append_note(const std::string &path, const std::string &batch_time,
                            const std::string &backend_name, const std::string &note)
{
    std::vector<std::string> lines;
    FILE *f = fopen(path.c_str(), "r");
    if (!f) {
        return false;
    }
    char buf[16384];
    while (fgets(buf, sizeof(buf), f)) {
        lines.emplace_back(buf);
    }
    fclose(f);

    bool changed = false;
    for (size_t li = 1; li < lines.size() && !changed; ++li) { /* skip header */
        auto flds = csv_parse_line(lines[li]);
        /* 26 columns: [1]=time, [21]=backend_name, [25]=notes */
        if (flds.size() < 26 || flds[1] != batch_time || flds[21] != backend_name) {
            continue;
        }
        std::string &notes = flds[25];
        if (notes.find(note) == std::string::npos) {
            notes += notes.empty() ? note : ("; " + note);
            /* rebuild the line */
            std::string nl;
            for (size_t k = 0; k < flds.size(); ++k) {
                if (k > 0) {
                    nl += ',';
                }
                nl += csv_quote_field(flds[k]);
            }
            lines[li] = nl + "\n";
            changed = true;
        }
    }
    if (changed) {
        FILE *w = fopen(path.c_str(), "w");
        if (!w) {
            return false;
        }
        for (auto &l : lines) {
            fputs(l.c_str(), w);
        }
        fclose(w);
    }
    return changed;
}

/* ---------------------------------------------------------------------------
 * Time helpers
 * -------------------------------------------------------------------------*/
static std::string now_date()
{
    time_t t = time(nullptr);
    struct tm b;
#ifdef _WIN32
    localtime_s(&b, &t);
#else
    localtime_r(&t, &b);
#endif
    char buf[16];
    strftime(buf, sizeof(buf), "%Y-%m-%d", &b);
    return buf;
}
static std::string now_time()
{
    time_t t = time(nullptr);
    struct tm b;
#ifdef _WIN32
    localtime_s(&b, &t);
#else
    localtime_r(&t, &b);
#endif
    char buf[16];
    strftime(buf, sizeof(buf), "%H:%M:%S", &b);
    return buf;
}

/* ---------------------------------------------------------------------------
 * Parse shape string like "[1,4,2048,8];[1,128,2,512]" into element counts
 * -------------------------------------------------------------------------*/
static std::vector<size_t> parse_input_element_counts(const std::string &shape_str)
{
    std::vector<size_t> counts;
    std::string s = shape_str;
    size_t pos = 0;
    while ((pos = s.find('[')) != std::string::npos) {
        auto end = s.find(']', pos);
        if (end == std::string::npos) {
            break;
        }
        std::string dims = s.substr(pos + 1, end - pos - 1);
        size_t elems = 1;
        size_t cp = 0;
        while (cp < dims.length()) {
            auto nc = dims.find(',', cp);
            if (nc == std::string::npos) {
                nc = dims.length();
            }
            int d = atoi(dims.substr(cp, nc - cp).c_str());
            if (d > 0) {
                elems *= (size_t)d;
            }
            cp = nc + 1;
        }
        counts.push_back(elems);
        s = s.substr(end + 1);
    }
    return counts;
}

/* ---------------------------------------------------------------------------
 * Parse an input list file (generated by tools/generate_test_data_for_onnx.py).
 * Entries may be separated by whitespace/newlines/commas; lines starting with
 * '#' are comments and skipped.  Returns the raw tokens (paths as written).
 * -------------------------------------------------------------------------*/
static std::vector<std::string> parse_input_list_file(const std::string &list_path)
{
    std::vector<std::string> tokens;
    FILE *f = fopen(list_path.c_str(), "r");
    if (!f) {
        LOGE("Cannot open input list: %s, due to %s, %d",
             list_path.c_str(), strerror(errno), errno);
        return tokens;
    }
    char line[4096];
    while (fgets(line, sizeof(line), f)) {
        char *p = line;
        /* Skip leading whitespace */
        while (*p && isspace((unsigned char)*p)) {
            ++p;
        }
        if (!*p || *p == '#') {
            continue; /* blank or comment */
        }
        /* Tokenize the line by whitespace or commas */
        char *tok = strtok(p, " \t\r\n,");
        while (tok) {
            if (*tok != '#') {
                tokens.push_back(tok);
            }
            tok = strtok(nullptr, " \t\r\n,");
        }
    }
    fclose(f);
    return tokens;
}

/* ---------------------------------------------------------------------------
 * Resolve input-list entries to absolute paths.
 * Relative paths are resolved against the directory containing the list file,
 * so the list can be used from any working directory.
 * -------------------------------------------------------------------------*/
static std::vector<std::string> resolve_input_list_paths(
    const std::string &list_path, const std::vector<std::string> &tokens)
{
    std::vector<std::string> paths;
    /* Extract directory of the list file */
    std::string base_dir;
    auto slash = list_path.find_last_of("/\\");
    if (slash != std::string::npos) {
        base_dir = list_path.substr(0, slash + 1);
    }
    for (auto &t : tokens) {
        /* Absolute path (unix or windows) or already resolvable */
        bool is_abs = !t.empty() && (t[0] == '/' || t[0] == '\\' || (t.size() > 1 && t[1] == ':'));
        std::string full = is_abs ? t : base_dir + t;
        /* If relative to CWD failed, fall back to raw token */
        if (!is_abs) {
            FILE *test = fopen(full.c_str(), "rb");
            if (!test) {
                FILE *test2 = fopen(t.c_str(), "rb");
                if (test2) {
                    fclose(test2);
                    full = t;
                } else {
                    LOGW("Input path not found (list-relative nor CWD): %s", t.c_str());
                }
            } else {
                fclose(test);
            }
        }
        paths.push_back(full);
    }
    return paths;
}

static const char *app_name_str()
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

/* ---------------------------------------------------------------------------
 * Sanitize a string for CSV export -- replace characters that break CSV.
 * -------------------------------------------------------------------------*/
static std::string csv_safe(const std::string &s)
{
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        switch (c) {
        case '\n': {
            out += ' ';
            break;
        }
        case '\r': {
            out += ' ';
            break;
        }
        case '"': {
            out += "'";
            break;
        }
        default: {
            out += c;
            break;
        }
        }
    }
    return out;
}

/* ---------------------------------------------------------------------------
 * Constructor
 * -------------------------------------------------------------------------*/
BenchmarkRunner::BenchmarkRunner(const BenchConfig &cfg, ResultCollector &collector,
                                 int argc, char **argv)
    : cfg_(cfg), collector_(collector), argc_(argc), argv_(argv)
{
    device_info_ = get_device_info_csv();
    arch_ = ARCH_STR;
}

/* ---------------------------------------------------------------------------
 * Run
 * -------------------------------------------------------------------------*/
bool BenchmarkRunner::Run()
{
    batch_date_ = now_date();
    batch_time_ = cfg_.batch_time.empty() ? now_time() : cfg_.batch_time;

    LOGI("========================================");
    LOGI("Unified Benchmark v2.0 (C++)");
    LOGI("========================================");

    /* The tool always runs in per-backend-process scheduler mode. A child
     * (worker) process - flagged with --worker - runs the classic single
     * variant/backend flow below; anything else is the scheduler that spawns
     * one clean process per (variant, backend). */
    if (!cfg_.worker_mode) {
        return RunPerProcess();
    }

    auto bundle = search_model_variants(cfg_.model_path);
    auto variants = bundle.all_found();
    if (variants.empty()) {
        LOGE("No model variants found");
        return false;
    }

    /* Reference = first found variant */
    const ModelSearchResult *ref_var = variants[0];
    LOGI("Reference: %s (%s)", model_format_name(ref_var->format),
         ref_var->path.c_str());

    for (const auto *var : variants) {
        TestVariant(*var, *ref_var);
    }

    /* Records are already written incrementally via AppendCsv.
     * No final ExportCsv needed (it would truncate previous runs). */

    LOGI("========================================");
    LOGI("Done: %zu record(s)", collector_.Count());
    for (size_t i = 0; i < collector_.Count(); ++i) {
        auto &r = collector_.Get(i);
        if (r.acceleration_vs_cpu == kNoBaselineAccel) {
            LOGI("  %-18s avg=%.1f ms  diff=-      accel=-      peak=%7.2f MB",
                 r.backend_name.c_str(), r.avg_run_ms, r.peak_mem_mb);
        } else if (r.acceleration_vs_cpu == -1.0) {
            LOGI("  %-18s failed (no measurement)", r.backend_name.c_str());
        } else {
            LOGI("  %-18s avg=%.1f ms  diff=%.6f  accel=%.2fx  peak=%7.2f MB",
                 r.backend_name.c_str(), r.avg_run_ms,
                 r.max_output_diff, r.acceleration_vs_cpu, r.peak_mem_mb);
        }
    }
    LOGI("========================================");
    return collector_.Count() > 0;
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
        filter_backends_by_user(bs, cfg_);
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
            bool is_model_so = (var->path.size() > 3 &&
                                stricmp_(var->path.c_str() + var->path.size() - 3,
                                         ".so") == 0);
            if (!is_model_so) {
                std::vector<BackendConfig> keep;
                for (auto &b : backends) {
                    if (b.id == BackendId::QNN_SDK_HTP) {
                        keep.push_back(b);
                    }
                }
                backends = std::move(keep);
            }
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

bool BenchmarkRunner::TestVariant(const ModelSearchResult &variant,
                                  const ModelSearchResult &ref_variant)
{
    ModelFormat fmt = variant.format;
    ModelFormat ref_fmt = ref_variant.format;

    auto backends = BackendRegistry::GetAvailable(fmt);
    if (backends.empty()) {
        return false;
    }

    /* QNN: a model.so (lib{base}.so) is runtime-composed and runs on ALL QNN
     * SDK backends (CPU/GPU/HTP); a context binary (.serialized.bin/.bin/.dlc)
     * is offline-compiled and backend-specific -- default to HTP only. */
    if (fmt == ModelFormat::QNN) {
        bool is_model_so = (variant.path.size() > 3 &&
                            stricmp_(variant.path.c_str() + variant.path.size() - 3,
                                     ".so") == 0);
        if (!is_model_so) {
            std::vector<BackendConfig> keep;
            for (auto &b : backends) {
                if (b.id == BackendId::QNN_SDK_HTP) {
                    keep.push_back(b);
                }
            }
            backends = std::move(keep);
        }
    }

    /* Filter by --backend (whitelist) if specified */
    if (!cfg_.backend_ids.empty()) {
        std::vector<BackendConfig> filtered;
        for (auto &b : backends) {
            for (int want : cfg_.backend_ids) {
                if (bid(b.id) == want) {
                    filtered.push_back(b);
                    break;
                }
            }
        }
        if (filtered.empty()) {
            return false;
        }
        /* NOTE: no auto-added CPU baseline here. The tool always runs in the
         * per-backend-process scheduler mode; accuracy comparison happens
         * cross-process via the dumped baseline output file, so the in-process
         * "force include the CPU baseline" behavior is intentionally gone. */
        backends = std::move(filtered);
    }

    /* Filter by --no-backend (blacklist) -- applied after the whitelist */
    if (!cfg_.no_backend_ids.empty()) {
        std::vector<BackendConfig> kept;
        for (auto &b : backends) {
            bool excluded = false;
            for (int want : cfg_.no_backend_ids) {
                if (bid(b.id) == want) {
                    LOGI("Excluded backend: %s", b.name.c_str());
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
            return false;
        }
        backends = std::move(kept);
    }

    LOGI("=== %s: %zu backend(s) ===", variant.path.c_str(), backends.size());

    /* Find a CPU backend of THIS format for shape probing. A worker may only
     * run one non-CPU backend (e.g. MNN_OPENCL), so the filtered list has no
     * is_cpu_baseline entry - fall back to the full registry list. ONNX_CPU
     * is only a valid shape probe for ONNX models, never for other formats
     * (loading a .mnn/.tflite with the ONNX backend fails Protobuf parsing). */
    BackendId cpu_id = BackendId::ONNX_CPU;
    for (auto &bc : backends) {
        if (bc.is_cpu_baseline) {
            cpu_id = bc.id;
            break;
        }
    }
    if (cpu_id == BackendId::ONNX_CPU) {
        auto all = BackendRegistry::GetAvailable(fmt);
        for (auto &bc : all) {
            if (bc.is_cpu_baseline) {
                cpu_id = bc.id;
                break;
            }
        }
    }

    /* Generate shared deterministic inputs:
     * 1. Create a temp CPU backend to query shapes
     * 2. Parse shapes to get element counts per input
     * 3. Either load inputs from --input-list files, or generate with seed=42 */
    InputProvider shared;
    std::vector<size_t> input_sizes;

    /* NCNN: read .ncnn.shapes */
    std::string ncnn_shapes;
    if (fmt == ModelFormat::NCNN) {
        std::string base = variant.path;
        /* Strip trailing .param or .ncnn.param to get base name */
        if (base.length() > 6) {
            auto tail6 = base.substr(base.length() - 6);
            if (stricmp_(tail6.c_str(), ".param") == 0) {
                base = base.substr(0, base.length() - 6);
            }
        }
        if (base.length() > 5) {
            auto tail5 = base.substr(base.length() - 5);
            if (stricmp_(tail5.c_str(), ".ncnn") == 0) {
                base = base.substr(0, base.length() - 5);
            }
        }
        ncnn_shapes = base + ".shapes";

        /* Read shapes file to get element counts */
        FILE *f = fopen(ncnn_shapes.c_str(), "r");
        if (!f) {
            LOGW("NCNN shapes file not found: %s, due to %s, %d", ncnn_shapes.c_str(), strerror(errno), errno);
        } else {
            char line[512];
            while (fgets(line, sizeof(line), f)) {
                /* Skip the "inputs=N" / "outputs=N" header lines */
                if (strncmp(line, "inputs=", 7) == 0 || strncmp(line, "outputs=", 8) == 0) {
                    continue;
                }
                if (strncmp(line, "in", 2) == 0 && strchr(line, '=')) {
                    const char *dims = strchr(line, '=') + 1;
                    size_t elems = 1;
                    const char *p = dims;
                    while (*p) {
                        elems *= (size_t)atoi(p);
                        p = strchr(p, ',');
                        if (!p) {
                            break;
                        }
                        ++p;
                    }
                    input_sizes.push_back(elems);
                }
            }
            fclose(f);
        }
    }

    /* For non-NCNN: use temp backend to query shapes */
    if (input_sizes.empty() && fmt != ModelFormat::NCNN) {
        std::vector<BackendId> candidate_ids;
        if (fmt == ModelFormat::QNN) {
            /* QNN context binaries are backend-specific (HTP/GPU/CPU), so the
             * ONNX CPU temp backend cannot parse them. Query shapes via a QNN
             * SDK backend that can restore the binary. */
            for (auto &bc : backends) {
                if (is_qnn_sdk_backend(bc.id)) {
                    candidate_ids.push_back(bc.id);
                }
            }
        } else {
            candidate_ids.push_back(cpu_id);
        }
        /* TFLite CPU backend may fail for models with Select TF ops (FlexErf).
         * Fall back to GPU delegate which handles Flex ops via GL shaders. */
        if (fmt == ModelFormat::TFLITE) {
            for (auto &bc : backends) {
                if (bc.id != cpu_id) {
                    candidate_ids.push_back(bc.id);
                }
            }
        }
        for (auto cid : candidate_ids) {
            LOGI("Creating temp backend (id=%d) to query shapes...", bid(cid));
            fflush(stderr);
            auto tmp = BackendRegistry::Create(cid);
            if (tmp && tmp->Initialize(variant.path.c_str(), cfg_.num_threads)) {
                std::string is, os;
                size_t ie, oe;
                tmp->QueryIOInfo(is, ie, os, oe);
                input_sizes = parse_input_element_counts(is);
                break;
            }
        }
    }

    if (input_sizes.empty()) {
        LOGW("Cannot determine input shapes for %s", model_format_name(fmt));
        return false;
    }

    /* Populate shared inputs: from --input-list files, or deterministic random */
    if (!cfg_.input_list_path.empty()) {
        auto tokens = parse_input_list_file(cfg_.input_list_path);
        if (tokens.empty()) {
            LOGW("Input list is empty or unreadable: %s", cfg_.input_list_path.c_str());
            return false;
        }
        auto paths = resolve_input_list_paths(cfg_.input_list_path, tokens);
        if (!shared.LoadFromFiles(paths, input_sizes, cfg_.input_format)) {
            LOGW("Failed to load inputs from list: %s", cfg_.input_list_path.c_str());
            return false;
        }
    } else {
        shared.GenerateFromSizes(input_sizes);
    }

    /* Test each backend */
    bool any_ok = false;
    double weight_mb = estimate_weight_mb(variant);
    for (auto &bcfg : backends) {
        if (TestBackend(bcfg, variant.path, fmt, ref_fmt,
                        shared, ncnn_shapes, weight_mb)) {
            any_ok = true;
        }
    }
    return any_ok;
}

/* ---------------------------------------------------------------------------
 * Record a failed backend
 * -------------------------------------------------------------------------*/
static void RecordFailure(ResultCollector &collector, const BackendConfig &bcfg,
                          const std::string &model_path, ModelFormat fmt,
                          const BenchConfig &cfg, const std::string &device_info,
                          const std::string &arch, const char *app,
                          const char *date, const char *time,
                          const char *reason)
{
    if (!cfg.enable_csv) {
        return;
    }
    BenchmarkRecord rec;
    auto last_slash = model_path.find_last_of("/\\");
    rec.model_name = (last_slash != std::string::npos) ? model_path.substr(last_slash + 1) : model_path;
    if (fmt == ModelFormat::NCNN) {
        auto pos = rec.model_name.rfind(".ncnn.param");
        if (pos != std::string::npos) {
            rec.model_name.replace(pos, 12, ".ncnn.bin");
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
 * TestBackend
 * -------------------------------------------------------------------------*/
bool BenchmarkRunner::TestBackend(const BackendConfig &bcfg,
                                  const std::string &model_path,
                                  ModelFormat fmt, ModelFormat ref_fmt,
                                  InputProvider &shared,
                                  const std::string & /*ncnn_shapes*/,
                                  double weight_mem_mb)
{
    LOGI("--- %s ---", bcfg.name.c_str());

    auto backend = BackendRegistry::Create(bcfg.id);
    if (!backend) {
        LOGW("Create failed: %s", bcfg.name.c_str());
        RecordFailure(collector_, bcfg, model_path, fmt, cfg_,
                      device_info_, arch_, app_name_str(),
                      batch_date_.c_str(), batch_time_.c_str(),
                      "Backend not registered");
        return false;
    }

    if (!backend->Initialize(model_path.c_str(), cfg_.num_threads)) {
        LOGW("Init failed: %s", bcfg.name.c_str());
        const char *reason = backend->GetLastError();
        if (!reason || !*reason) {
            reason = "Initialization failed";
        }
        RecordFailure(collector_, bcfg, model_path, fmt, cfg_,
                      device_info_, arch_, app_name_str(),
                      batch_date_.c_str(), batch_time_.c_str(),
                      reason);
        return false;
    }

    /* Query IO info */
    std::string is, os;
    size_t ie, oe;
    backend->QueryIOInfo(is, ie, os, oe);

    /* Set shared inputs (ALL backends use shared inputs for fairness) */
    backend->SetSharedInput(shared.DataPtrs().data(),
                            shared.ElementCounts().data());

    /* Run benchmark */
    double total_ms, max_ms, min_ms;
    int max_idx;
    std::vector<float *> odata;
    std::vector<size_t> oelems;
    std::vector<std::array<size_t, MAX_DIMENSIONS>> oshapes;
    std::vector<size_t> odims;

    if (!backend->RunBenchmark(cfg_.warmup_runs, cfg_.repeat,
                               total_ms, max_ms, min_ms, max_idx,
                               odata, oelems, oshapes, odims)) {
        LOGW("Benchmark failed: %s", bcfg.name.c_str());
        const char *reason = backend->GetLastError();
        if (!reason || !*reason) {
            reason = "Benchmark failed";
        }
        RecordFailure(collector_, bcfg, model_path, fmt, cfg_,
                      device_info_, arch_, app_name_str(),
                      batch_date_.c_str(), batch_time_.c_str(),
                      reason);
        return false;
    }

    /* Process memory after the run (peak is monotonic since process start) */
    double mem_peak = 0.0, mem_resident = 0.0;
    get_process_mem_mb(mem_peak, mem_resident);

    double avg_ms = (cfg_.repeat > 0) ? total_ms / cfg_.repeat : 0;

    /* Baseline or comparison */
    double max_diff = 0, avg_diff = 0, accel = 1.0;
    int64_t elem_count = 0;

    /* Output tensor names enable name-based accuracy comparison: multi-output
     * models can have a different output ORDER across backends (e.g. QNN SDK
     * model.so vs ONNX), so pass all outputs + names and let the collector
     * match by name (falls back to position when names are unavailable). */
    const std::vector<std::string> &out_names = backend->GetOutputNames();

    /* Compare with baseline. The tool only ever runs in scheduler/worker mode
     * (one backend per process), so there is no in-process baseline: the
     * comparison source is the cross-process dumped file handed to this
     * worker via --baseline-file. A worker WITHOUT a baseline file (the
     * baseline worker itself, flagged with --dump-output) shows "-" for the
     * comparison columns - it IS the reference and does not compare to
     * itself. */
    bool compared = false;
    if (!odata.empty() && odata[0] && !oelems.empty() && oelems[0] > 0) {
        if (!cfg_.baseline_file.empty()) {
            std::vector<float> base;
            if (load_output_file(cfg_.baseline_file, base) && !base.empty()) {
                size_t n = std::min(base.size(), oelems[0]);
                double sum = 0.0, mx = 0.0;
                const float *cur = odata[0];
                for (size_t k = 0; k < n; ++k) {
                    double d = fabs((double)cur[k] - (double)base[k]);
                    sum += d;
                    if (d > mx) {
                        mx = d;
                    }
                }
                max_diff = mx;
                avg_diff = (n > 0) ? sum / (double)n : 0.0;
                elem_count = (int64_t)n;
                if (cfg_.baseline_ms > 0.0 && avg_ms > 0.0) {
                    accel = cfg_.baseline_ms / avg_ms;
                } else {
                    accel = kNoBaselineAccel; /* diff present, no timing ref */
                }
                compared = true;
                LOGI("Baseline compare (file): max_diff=%.8f avg_diff=%.8f (%zu elems)",
                     max_diff, avg_diff, n);
            } else {
                LOGW("Baseline file unreadable: %s", cfg_.baseline_file.c_str());
            }
        } else if (collector_.HasBaseline(ref_fmt)) {
            collector_.CompareWithBaseline(ref_fmt, odata, oelems,
                                           max_diff, avg_diff, elem_count, out_names);
            double cpu_ms = collector_.GetCpuBaselineMs(ref_fmt);
            if (cpu_ms > 0) {
                accel = cpu_ms / avg_ms;
            }
            compared = true;
        }
    }
    if (!compared) {
        if (!cfg_.dump_output.empty()) {
            /* The baseline worker: it IS the reference, so its own row must
             * not look like a failure. Report the identity comparison -
             * diff=0, accel=1.000x - instead of the no-baseline "-". */
            max_diff = 0.0;
            avg_diff = 0.0;
            elem_count = (int64_t)(oelems.empty() ? 0 : oelems[0]);
            accel = 1.0;
            LOGI("Baseline row: diff=0 accel=1.000x (reference, no self-comparison)");
        } else {
            /* No baseline to compare against (e.g. a worker that received no
             * baseline file because the entry-format CPU baseline was
             * unavailable): mark accel with the no-baseline sentinel so the
             * CSV shows "-" for the comparison columns instead of a
             * misleading 1.000x. */
            accel = kNoBaselineAccel;
        }
    }

    /* Timing breakdown */
    std::array<double, 10> timing;
    backend->GetTiming(timing);

    /* Notes */
    std::ostringstream notes;
    notes << std::fixed << std::setprecision(3);
    if (bcfg.type == BackendType::ONNX_EP) {
        notes << "load_lib=" << timing[1] << ",get_api=" << timing[2]
              << ",create_env=" << timing[3] << ",session_opts=" << timing[4]
              << ",config_opts=" << timing[5] << ",append_prov=" << timing[5]
              << ",create_session=" << timing[6] << ",get_io=" << timing[7];
    } else {
        notes << "load_lib=" << timing[0];
    }

    /* Record */
    BenchmarkRecord rec;
    /* Strip directory path, keep only the filename */
    {
        auto last_slash = model_path.find_last_of("/\\");
        rec.model_name = (last_slash != std::string::npos) ? model_path.substr(last_slash + 1) : model_path;
    }
    /* For NCNN: use .bin instead of .param (weights file varies, param is same) */
    if (fmt == ModelFormat::NCNN) {
        auto pos = rec.model_name.rfind(".ncnn.param");
        if (pos != std::string::npos) {
            if (bcfg.id == BackendId::NCNN_VK_FP16) {
                rec.model_name.replace(pos, 12, "_fp16.ncnn.bin");
            } else {
                rec.model_name.replace(pos, 12, ".ncnn.bin");
            }
        }
    }
    rec.input_shape_str = is;
    rec.input_elements = ie;
    rec.output_shape_str = os;
    rec.output_elements = oe;
    rec.warmup_runs = cfg_.warmup_runs;
    rec.repeat_runs = cfg_.repeat;
    /* QNN SDK HTP (offline context binary): the CSV "threads" column cannot
     * report the HTP compute threads - hvx_threads is a compile-time parameter
     * baked into the .serialized.bin and not readable at runtime. Use -1 so
     * the CSV prints "-" instead of a misleading CPU thread count. */
    rec.num_threads = (bcfg.id == BackendId::QNN_SDK_HTP) ? -1 : cfg_.num_threads;
    rec.total_run_ms = total_ms;
    rec.avg_run_ms = avg_ms;
    rec.max_run_ms = max_ms;
    rec.max_run_idx = max_idx;
    rec.init_ms = timing[0];
    rec.max_output_diff = max_diff;
    rec.avg_output_diff = avg_diff;
    rec.acceleration_vs_cpu = accel;
    rec.backend_name = bcfg.name;
    rec.device_info = device_info_;
    rec.arch = arch_;
    rec.app_name = app_name_str();
    rec.notes = csv_safe(notes.str());

    /* Deployment memory info (MB) */
    rec.weight_mem_mb = weight_mem_mb;
    rec.peak_mem_mb = mem_peak;
    rec.resident_mem_mb = mem_resident;

    collector_.Add(rec);

    /* Append CSV (crash-safe), using frozen batch timestamp */
    if (cfg_.enable_csv) {
        collector_.AppendCsv(rec, cfg_.csv_path.c_str(),
                             batch_date_.c_str(), batch_time_.c_str(),
                             app_name_str());
    }

    /* Deployment summary - one glance for the customer */
    LOGI("  --- deployment memory (%s) ---", bcfg.name.c_str());
    LOGI("  weight_mem    : %8.2f MB", rec.weight_mem_mb);
    LOGI("  peak_mem      : %8.2f MB", rec.peak_mem_mb);
    LOGI("  resident_mem  : %8.2f MB", rec.resident_mem_mb);

    /* Dump first output for cross-process baseline (--dump-output, used by
     * the scheduler's baseline worker). */
    if (!cfg_.dump_output.empty() && !odata.empty() && odata[0] &&
        !oelems.empty() && oelems[0] > 0) {
        if (write_output_file(cfg_.dump_output, odata[0], oelems[0], avg_ms)) {
            LOGI("Output dumped for baseline: %s (%zu elems, avg=%.3f ms)",
                 cfg_.dump_output.c_str(), oelems[0], avg_ms);
        } else {
            LOGW("Failed to dump output: %s", cfg_.dump_output.c_str());
        }
    }

    /* Free output buffers */
    for (auto *p : odata) {
        free(p);
    }

    return true;
}
