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
 * CSV schema (single source of truth for header + record rows)
 *
 * 26 columns, in order:
 *   0 date            1 time          2 model_name    3 model_input_shape
 *   4 input_elements  5 model_output_shape  6 output_elements
 *   7 warmup_runs     8 repeat_runs   9 threads
 *  10 total_run_ms   11 avg_run_ms   12 max_run_ms   13 max_run_idx
 *  14 init_ms        15 max_output_diff  16 avg_output_diff
 *  17 acceleration_vs_cpu
 *  18 weight_mem_mb  19 peak_mem_mb  20 resident_mem_mb
 *  21 backend_name   22 device_info  23 arch         24 app_name
 *  25 notes
 * -------------------------------------------------------------------------*/
static const char *kCsvHeader =
    "date,time,model_name,model_input_shape,input_elements,"
    "model_output_shape,output_elements,warmup_runs,repeat_runs,threads,"
    "total_run_ms,avg_run_ms,max_run_ms,max_run_idx,init_ms,"
    "max_output_diff,avg_output_diff,acceleration_vs_cpu,"
    "weight_mem_mb,peak_mem_mb,resident_mem_mb,"
    "backend_name,device_info,arch,app_name,notes\n";

/* Write one record row (quote-aware for string fields). Shared by every
 * append so the row format can never drift between callers. */
static void write_csv_record(FILE *f, const BenchmarkRecord &r,
                             const char *date, const char *time,
                             const char *app_name)
{
    fprintf(f, "\"%s\",\"%s\",\"%s\",", date, time, r.model_name.c_str());
    fprintf(f, "\"%s\",%zu,\"%s\",%zu,",
            r.input_shape_str.c_str(), r.input_elements,
            r.output_shape_str.c_str(), r.output_elements);
    fprintf(f, "%d,%d,", r.warmup_runs, r.repeat_runs);
    if (r.num_threads < 0) {
        fputs("-", f); /* e.g. QNN HTP: thread count not meaningful */
    } else {
        fprintf(f, "%d", r.num_threads);
    }
    fputc(',', f);
    bool failed = r.acceleration_vs_cpu == -1.0;               /* failed sentinel */
    bool no_base = r.acceleration_vs_cpu == kNoBaselineAccel;  /* -2 sentinel */
    if (failed) {
        fprintf(f, "-,-,-,%d,", r.max_run_idx);
    } else {
        fprintf(f, "%.3f,%.3f,%.3f,%d,",
                r.total_run_ms, r.avg_run_ms, r.max_run_ms, r.max_run_idx);
    }
    if (failed) {
        fprintf(f, "-,-,-,");
    } else if (no_base) {
        fprintf(f, "%.3f,-,-,", r.init_ms); /* timing kept, no comparison */
    } else {
        fprintf(f, "%.3f,%.8f,%.8f,",
                r.init_ms, r.max_output_diff, r.avg_output_diff);
    }
    if (failed || no_base) {
        fprintf(f, "-");
    } else {
        fprintf(f, "%.3fx", r.acceleration_vs_cpu);
    }
    /* Failed backends: memory columns are "-" too (no valid measurement);
     * no-baseline runs keep the memory values (they are the point). */
    if (failed) {
        fprintf(f, ",-,-,-");
    } else {
        fprintf(f, ",%.3f,%.3f,%.3f",
                r.weight_mem_mb, r.peak_mem_mb, r.resident_mem_mb);
    }
    fprintf(f, ",\"%s\",\"%s\",\"%s\",\"%s\",\"%s\"\n",
            r.backend_name.c_str(), r.device_info.c_str(),
            r.arch.c_str(), app_name, r.notes.c_str());
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
 * CSV export (append-only, crash-safe)
 * -------------------------------------------------------------------------*/
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
        fputs(kCsvHeader, f);
    }
    write_csv_record(f, rec, date, time, app_name);

    if (fclose(f) != 0) {
        LOGE("fclose(%s) failed: %s, %d", path, strerror(errno), errno);
        return false;
    }
    return true;
}
