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
 * 29 columns, in order:
 *   0 date            1 time          2 model_name    3 model_input_shape
 *   4 input_elements  5 model_output_shape  6 output_elements
 *   7 weight_mem_mb
 *   8 warmup_runs     9 repeat_runs  10 threads
 *  11 total_run_ms   12 avg_run_ms   13 max_run_ms   14 max_run_idx
 *  15 init_ms
 *  16 transfer_in_ms 17 transfer_out_ms 18 transfer_total_ms
 *  19 max_output_diff 20 avg_output_diff
 *  21 acceleration_vs_cpu
 *  22 peak_mem_mb    23 resident_mem_mb
 *  24 backend_name   25 device_info  26 arch         27 app_name
 *  28 notes
 * -------------------------------------------------------------------------*/
static const char *kCsvHeader =
    "date,time,model_name,model_input_shape,input_elements,"
    "model_output_shape,output_elements,weight_mem_mb,"
    "warmup_runs,repeat_runs,threads,"
    "total_run_ms,avg_run_ms,max_run_ms,max_run_idx,init_ms,"
    "transfer_in_ms,transfer_out_ms,transfer_total_ms,"
    "max_output_diff,avg_output_diff,acceleration_vs_cpu,"
    "peak_mem_mb,resident_mem_mb,"
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
    bool failed = r.acceleration_vs_cpu == -1.0;               /* failed sentinel */
    bool no_base = r.acceleration_vs_cpu == kNoBaselineAccel;  /* -2 sentinel */
    /* weight memory (deployment info): right after the output shapes.
     * Failed backends have no valid measurement -> "-". */
    if (failed) {
        fputs("-", f);
    } else {
        fprintf(f, "%.3f", r.weight_mem_mb);
    }
    fputc(',', f);
    fprintf(f, "%d,%d,", r.warmup_runs, r.repeat_runs);
    if (r.num_threads < 0) {
        fputs("-", f); /* e.g. QNN HTP: thread count not meaningful */
    } else {
        fprintf(f, "%d", r.num_threads);
    }
    fputc(',', f);
    if (failed) {
        fprintf(f, "-,-,-,%d,", r.max_run_idx);
    } else {
        fprintf(f, "%.3f,%.3f,%.3f,%d,",
                r.total_run_ms, r.avg_run_ms, r.max_run_ms, r.max_run_idx);
    }
    /* init_ms first, then the transfer block, then the diff block (matching
     * the header order: init_ms, transfer_in/out/total, max/avg_diff). */
    if (failed) {
        fprintf(f, "-,");
    } else {
        fprintf(f, "%.3f,", r.init_ms);
    }
    if (failed) {
        fprintf(f, "-,-,-,");
    } else {
        fprintf(f, "%.6f,%.6f,%.6f,",
                r.transfer_in_ms, r.transfer_out_ms, r.transfer_total_ms);
    }
    if (failed) {
        fprintf(f, "-,-,");
    } else if (no_base) {
        fprintf(f, "-,-,"); /* no comparison: diff columns are "-" */
    } else {
        fprintf(f, "%.8f,%.8f,",
                r.max_output_diff, r.avg_output_diff);
    }
    if (failed || no_base) {
        fprintf(f, "-");
    } else {
        fprintf(f, "%.3fx", r.acceleration_vs_cpu);
    }
    /* Failed backends: peak/resident are "-" too (no valid measurement);
     * no-baseline runs keep the memory values (they are the point).
     * weight_mem_mb was already emitted after the output shapes above. */
    if (failed) {
        fprintf(f, ",-,-");
    } else {
        fprintf(f, ",%.3f,%.3f", r.peak_mem_mb, r.resident_mem_mb);
    }
    fprintf(f, ",\"%s\",\"%s\",\"%s\",\"%s\",\"%s\"\n",
            r.backend_name.c_str(), r.device_info.c_str(),
            r.arch.c_str(), app_name, r.notes.c_str());
}

/* The header/row schema is fixed at 29 columns; every write path must produce
 * exactly that many fields or the CSV silently misaligns. Verify at build time
 * via the shared constant. */
static_assert(kCsvExpectedColumns == 29,
              "CSV schema changed - update kCsvExpectedColumns");

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

    /* Guard against appending to a CSV that has a different (older or
     * hand-edited) column layout: the existing header column count must match
     * the current schema, otherwise every new row would be shifted and silently
     * corrupt the file (seen in the wild: warmup_runs populated with the
     * weight_mem_mb value). Refuse to append and tell the user to fix the
     * file, instead of writing misaligned rows. */
    if (!need_header) {
        FILE *hdr = fopen(path, "r");
        if (hdr) {
            char first_line[4096] = {0};
            if (fgets(first_line, sizeof(first_line), hdr) != nullptr) {
                size_t commas = 0;
                bool in_quotes = false;
                for (const char *p = first_line; *p != '\0' && *p != '\n'; ++p) {
                    if (*p == '"') {
                        in_quotes = !in_quotes;
                    } else if (*p == ',' && !in_quotes) {
                        ++commas;
                    }
                }
                const size_t cols = commas + 1; /* N fields -> N-1 commas */
                if (cols != kCsvExpectedColumns) {
                    LOGW("CSV %s has %zu columns but schema requires %zu - "
                         "refusing to append (fix or remove the file first)",
                         path, cols, (size_t)kCsvExpectedColumns);
                    if (fclose(hdr) != 0) {
                        LOGW("fclose(%s) failed: %s, %d", path, strerror(errno), errno);
                    }
                    return false;
                }
            }
            if (fclose(hdr) != 0) {
                LOGW("fclose(%s) failed: %s, %d", path, strerror(errno), errno);
            }
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
