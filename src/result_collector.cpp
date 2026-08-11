/*============================================================================
 * result_collector.cpp - Result collection & CSV export
 *============================================================================*/

#include "result_collector.hpp"
#include "log.hpp"
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstring>

/* ---------------------------------------------------------------------------
 * Private helpers
 * -------------------------------------------------------------------------*/
FormatBaseline *ResultCollector::get_baseline(ModelFormat fmt)
{
    int idx = static_cast<int>(fmt);
    if (idx < 0 || idx >= (int)baselines_.size()) {
        return nullptr;
    }
    return &baselines_[idx];
}

const FormatBaseline *ResultCollector::get_baseline(ModelFormat fmt) const
{
    int idx = static_cast<int>(fmt);
    if (idx < 0 || idx >= (int)baselines_.size()) {
        return nullptr;
    }
    return &baselines_[idx];
}

/* ---------------------------------------------------------------------------
 * Record management
 * -------------------------------------------------------------------------*/
void ResultCollector::Add(const BenchmarkRecord &rec)
{
    records_.push_back(rec);
    LOGI("Record added: %s [%s] avg=%.3f ms, max_diff=%.8f",
         rec.backend_name.c_str(), rec.model_name.c_str(),
         rec.avg_run_ms, rec.max_output_diff);
}

/* ---------------------------------------------------------------------------
 * Baseline management
 * -------------------------------------------------------------------------*/
bool ResultCollector::SetBaseline(ModelFormat fmt, const std::vector<float *> &data,
                                  const std::vector<size_t> &elems, double cpu_avg_ms,
                                  BackendId cpu_id, const std::vector<std::string> &names)
{
    FormatBaseline *fb = get_baseline(fmt);
    if (!fb) {
        LOGE("No baseline slot for format %d", (int)fmt);
        return false;
    }

    /* Store the first output as the reference and remember its name so the
     * comparison can later match by name (output order may differ across
     * backends for multi-output models). */
    if (!data.empty() && data[0] && !elems.empty() && elems[0] > 0) {
        fb->output = OutputData(data[0], elems[0]);
        fb->has_baseline = true;
    }
    fb->output_name = (!names.empty()) ? names[0] : "";
    fb->cpu_avg_ms = cpu_avg_ms;
    fb->cpu_backend_id = cpu_id;

    LOGI("[%s] Baseline stored: %zu elements, first_output='%s', cpu_avg=%.3f ms",
         model_format_name(fmt), fb->output.size(), fb->output_name.c_str(),
         cpu_avg_ms);
    return true;
}

bool ResultCollector::HasBaseline(ModelFormat fmt) const
{
    const FormatBaseline *fb = get_baseline(fmt);
    return fb && fb->has_baseline;
}

double ResultCollector::GetCpuBaselineMs(ModelFormat fmt) const
{
    const FormatBaseline *fb = get_baseline(fmt);
    return (fb && fb->has_baseline) ? fb->cpu_avg_ms : 0.0;
}

/* ---------------------------------------------------------------------------
 * Accuracy comparison
 * -------------------------------------------------------------------------*/
bool ResultCollector::CompareWithBaseline(ModelFormat fmt,
                                          const std::vector<float *> &data,
                                          const std::vector<size_t> &elems,
                                          double &max_diff, double &avg_diff,
                                          int64_t &element_count,
                                          const std::vector<std::string> &names)
{
    max_diff = 0.0;
    avg_diff = 0.0;
    element_count = 0;

    const FormatBaseline *fb = get_baseline(fmt);
    if (!fb || !fb->has_baseline || fb->output.empty()) {
        LOGW("[%s] Baseline not available for comparison", model_format_name(fmt));
        return false;
    }

    /* Choose the output to compare:
     * - Name-based: when the baseline recorded its first output's name and the
     *   current backend exposes output names, compare the output with the SAME
     *   name (backends may reorder multi-outputs, e.g. QNN SDK model.so).
     * - Fallback: position 0 when either side provides no names. */
    size_t match_idx = 0;
    if (!fb->output_name.empty() && !names.empty()) {
        bool found = false;
        for (size_t i = 0; i < names.size(); ++i) {
            if (names[i] == fb->output_name) {
                match_idx = i;
                found = true;
                break;
            }
        }
        if (!found) {
            LOGW("[%s] Output name '%s' not found in current outputs; "
                 "fallback to position 0",
                 model_format_name(fmt), fb->output_name.c_str());
        }
    }

    if (match_idx >= data.size() || match_idx >= elems.size() ||
        !data[match_idx]) {
        LOGW("[%s] No output available for comparison (idx=%zu)",
             model_format_name(fmt), match_idx);
        return false;
    }

    const float *data_ptr = data[match_idx];
    size_t n = elems[match_idx];
    element_count = (int64_t)n;

    if (n != fb->output.size()) {
        LOGW("[%s] Output size mismatch: current=%zu, baseline=%zu",
             model_format_name(fmt), n, fb->output.size());
        return false;
    }

    if (n == 0) {
        return false;
    }

    double sum_abs = 0.0;
    double max_abs = 0.0;
    const float *base = fb->output.ptr();
    int64_t nan_count = 0;

    for (size_t i = 0; i < n; ++i) {
        double da = static_cast<double>(data_ptr[i]);
        double db = static_cast<double>(base[i]);
        if (std::isnan(da) || std::isnan(db)) {
            if (nan_count < 10) {
                LOGW("[%s] NaN at element %zu: output=%.8f baseline=%.8f - skipping",
                     model_format_name(fmt), i, da, db);
            }
            ++nan_count;
            continue;
        }
        double diff = fabs(da - db);
        sum_abs += diff;
        if (diff > max_abs) {
            max_abs = diff;
        }
    }

    if (nan_count > 0) {
        LOGW("[%s] %lld NaN elements skipped out of %zu",
             model_format_name(fmt), (long long)nan_count, n);
    }

    avg_diff = sum_abs / static_cast<double>(n);
    max_diff = max_abs;
    return true;
}

/* ---------------------------------------------------------------------------
 * CSV export
 * -------------------------------------------------------------------------*/
bool ResultCollector::ExportCsv(const char *path, const char *date,
                                const char *time, const char *app_name) const
{
    FILE *f = fopen(path, "w");
    if (!f) {
        LOGE("Failed to open CSV: %s, due to %s, %d", path, strerror(errno), errno);
        return false;
    }

    fprintf(f, "date,time,model_name,model_input_shape,input_elements,"
               "model_output_shape,output_elements,warmup_runs,repeat_runs,threads,"
               "total_run_ms,avg_run_ms,max_run_ms,max_run_idx,init_ms,"
               "max_output_diff,avg_output_diff,acceleration_vs_cpu,"
               "backend_name,device_info,arch,app_name,notes\n");

    for (const auto &r : records_) {
        fprintf(f, "\"%s\",\"%s\",\"%s\",", date, time, r.model_name.c_str());
        fprintf(f, "\"%s\",%zu,\"%s\",%zu,",
                r.input_shape_str.c_str(), r.input_elements,
                r.output_shape_str.c_str(), r.output_elements);
        fprintf(f, "%d,%d,", r.warmup_runs, r.repeat_runs);
        if (r.num_threads < 0) {
            fputs("-", f);
        } else {
            fprintf(f, "%d", r.num_threads);
        }
        fputc(',', f);
        if (r.acceleration_vs_cpu < 0) {
            fprintf(f, "-,-,-,%d,", r.max_run_idx);
        } else {
            fprintf(f, "%.3f,%.3f,%.3f,%d,",
                    r.total_run_ms, r.avg_run_ms, r.max_run_ms, r.max_run_idx);
        }
        if (r.acceleration_vs_cpu < 0) {
            fprintf(f, "-,-,-,");
        } else {
            fprintf(f, "%.3f,%.8f,%.8f,",
                    r.init_ms, r.max_output_diff, r.avg_output_diff);
        }
        if (r.acceleration_vs_cpu < 0) {
            fprintf(f, "-");
        } else {
            fprintf(f, "%.3fx", r.acceleration_vs_cpu);
        }
        fprintf(f, ",\"%s\",\"%s\",\"%s\",\"%s\",\"%s\"\n",
                r.backend_name.c_str(), r.device_info.c_str(),
                r.arch.c_str(), app_name, r.notes.c_str());
    }

    if (fclose(f) != 0) {
        LOGE("fclose(%s) failed: %s, %d", path, strerror(errno), errno);
        return false;
    }
    LOGI("CSV exported: %s (%zu records)", path, records_.size());
    return true;
}

bool ResultCollector::AppendCsv(const BenchmarkRecord &rec, const char *path,
                                const char *date, const char *time,
                                const char *app_name) const
{
    /* Check if file exists and has content */
    bool need_header = true;
    FILE *check = fopen(path, "r");
    if (check) {
        if (fseek(check, 0, SEEK_END) != 0) {
            LOGW("fseek(%s) failed: %s, %d", path, strerror(errno), errno);
        }
        long sz = ftell(check);
        if (sz < 0) {
            LOGW("ftell(%s) failed: %s, %d", path, strerror(errno), errno);
        }
        if (fclose(check) != 0) {
            LOGW("fclose(%s) failed: %s, %d", path, strerror(errno), errno);
        }
        if (sz > 0) {
            need_header = false;
        }
    }

    FILE *f = fopen(path, "a");
    if (!f) {
        LOGE("Failed to open CSV for append: %s, due to %s, %d", path, strerror(errno), errno);
        return false;
    }

    if (need_header) {
        fprintf(f, "date,time,model_name,model_input_shape,input_elements,"
                   "model_output_shape,output_elements,warmup_runs,repeat_runs,threads,"
                   "total_run_ms,avg_run_ms,max_run_ms,max_run_idx,init_ms,"
                   "max_output_diff,avg_output_diff,acceleration_vs_cpu,"
                   "backend_name,device_info,arch,app_name,notes\n");
    }

    fprintf(f, "\"%s\",\"%s\",\"%s\",", date, time, rec.model_name.c_str());
    fprintf(f, "\"%s\",%zu,\"%s\",%zu,",
            rec.input_shape_str.c_str(), rec.input_elements,
            rec.output_shape_str.c_str(), rec.output_elements);
    fprintf(f, "%d,%d,", rec.warmup_runs, rec.repeat_runs);
    if (rec.num_threads < 0) {
        fputs("-", f);
    } else {
        fprintf(f, "%d", rec.num_threads);
    }
    fputc(',', f);
    if (rec.acceleration_vs_cpu < 0) {
        fprintf(f, "-,-,-,%d,", rec.max_run_idx);
    } else {
        fprintf(f, "%.3f,%.3f,%.3f,%d,",
                rec.total_run_ms, rec.avg_run_ms, rec.max_run_ms, rec.max_run_idx);
    }
    if (rec.acceleration_vs_cpu < 0) {
        fprintf(f, "-,-,-,");
    } else {
        fprintf(f, "%.3f,%.8f,%.8f,",
                rec.init_ms, rec.max_output_diff, rec.avg_output_diff);
    }
    if (rec.acceleration_vs_cpu < 0) {
        fprintf(f, "-");
    } else {
        fprintf(f, "%.3fx", rec.acceleration_vs_cpu);
    }
    fprintf(f, ",\"%s\",\"%s\",\"%s\",\"%s\",\"%s\"\n",
            rec.backend_name.c_str(), rec.device_info.c_str(),
            rec.arch.c_str(), app_name, rec.notes.c_str());

    fclose(f);
    return true;
}
