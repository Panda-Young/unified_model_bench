/*============================================================================
 * test_result_collector.cpp - CSV schema regression tests
 *
 * Guards the exact 29-column layout written by AppendCsv - in particular the
 * weight_mem_mb position (column 7, right after the output shapes) and the
 * tensor-transfer block (columns 16-18). A schema drift here silently breaks
 * csv_utils row lookups and csv_to_excel.py.
 *============================================================================*/
#include "csv_utils.hpp"
#include "result_collector.hpp"
#include "doctest.h"

#include <cstdio>
#include <string>
#include <vector>

namespace {

const char *kTmpCsv = "ut_result_collector_tmp.csv";

/* A fully-populated success record. */
BenchmarkRecord make_record()
{
    BenchmarkRecord r;
    r.model_name = "test_model.onnx";
    r.input_shape_str = "[1,4]";
    r.input_elements = 4;
    r.output_shape_str = "[1,2]";
    r.output_elements = 2;
    r.weight_mem_mb = 0.178;
    r.warmup_runs = 1;
    r.repeat_runs = 10;
    r.num_threads = 4;
    r.total_run_ms = 50.0;
    r.avg_run_ms = 5.0;
    r.max_run_ms = 6.0;
    r.max_run_idx = 3;
    r.init_ms = 1.0;
    r.transfer_in_ms = 0.5;
    r.transfer_out_ms = 0.7;
    r.transfer_total_ms = 1.2;
    r.max_output_diff = 0.0001;
    r.avg_output_diff = 0.00001;
    r.acceleration_vs_cpu = 2.5;
    r.peak_mem_mb = 100.0;
    r.resident_mem_mb = 90.0;
    r.backend_name = "ONNX_CPU";
    r.device_info = "dev";
    r.arch = "win_x64";
    r.app_name = "unified_bench_win_x64.exe";
    r.notes = "ok";
    return r;
}

} // namespace

TEST_CASE("AppendCsv: 29 columns with weight_mem_mb at column 7")
{
    std::remove(kTmpCsv);
    ResultCollector c;
    BenchmarkRecord r = make_record();
    CHECK(c.AppendCsv(r, kTmpCsv, "2026-08-21", "10:00:00", r.app_name.c_str()));

    FILE *f = fopen(kTmpCsv, "r");
    REQUIRE(f != nullptr);
    char line[4096];
    REQUIRE(fgets(line, sizeof(line), f) != nullptr); /* header */
    REQUIRE(fgets(line, sizeof(line), f) != nullptr); /* data row */
    fclose(f);

    auto flds = csv_parse_line(line);
    REQUIRE(flds.size() == (size_t)kCsvColumnCount);
    /* layout contract: shapes... output_elements(6), weight(7), config(8-10),
     * timing(11-14), init(15), transfer(16-18), diff(19-20), accel(21),
     * peak/resident(22-23), meta(24-28) */
    CHECK(flds[6] == "2");
    CHECK(flds[7] == "0.178");          /* weight_mem_mb right after shapes */
    CHECK(flds[8] == "1");              /* warmup_runs */
    CHECK(flds[kCsvColAvgRunMs] == "5.000");
    CHECK(flds[16] == "0.500000");      /* transfer_in_ms */
    CHECK(flds[17] == "0.700000");      /* transfer_out_ms */
    CHECK(flds[18] == "1.200000");      /* transfer_total_ms */
    CHECK(flds[22] == "100.000");       /* peak_mem_mb */
    CHECK(flds[23] == "90.000");        /* resident_mem_mb */
    CHECK(flds[kCsvColBackendName] == "ONNX_CPU");
    CHECK(flds[28] == "ok");            /* notes */
    std::remove(kTmpCsv);
}

TEST_CASE("AppendCsv: failed row keeps 29 columns with '-' placeholders")
{
    std::remove(kTmpCsv);
    ResultCollector c;
    BenchmarkRecord r = make_record();
    r.acceleration_vs_cpu = -1.0; /* failed sentinel */
    CHECK(c.AppendCsv(r, kTmpCsv, "2026-08-21", "10:00:00", r.app_name.c_str()));

    FILE *f = fopen(kTmpCsv, "r");
    REQUIRE(f != nullptr);
    char line[4096];
    REQUIRE(fgets(line, sizeof(line), f) != nullptr); /* header */
    REQUIRE(fgets(line, sizeof(line), f) != nullptr); /* data row */
    fclose(f);

    auto flds = csv_parse_line(line);
    REQUIRE(flds.size() == (size_t)kCsvColumnCount);
    CHECK(flds[7] == "-");              /* weight '-': no measurement */
    CHECK(flds[11] == "-");             /* total_run_ms */
    CHECK(flds[kCsvColAvgRunMs] == "-");
    CHECK(flds[16] == "-");             /* transfer_in_ms */
    CHECK(flds[17] == "-");             /* transfer_out_ms */
    CHECK(flds[18] == "-");             /* transfer_total_ms */
    CHECK(flds[22] == "-");             /* peak_mem_mb */
    CHECK(flds[23] == "-");             /* resident_mem_mb */
    std::remove(kTmpCsv);
}

TEST_CASE("AppendCsv: refuses to append to a misaligned existing CSV")
{
    /* A pre-existing file with the WRONG column count (e.g. produced by an
     * older build or hand-edited). Appending a correct row would shift every
     * column (the warmup_runs-gets-weight_mem_mb corruption seen in the wild),
     * so AppendCsv must refuse instead of writing misaligned data. */
    std::remove(kTmpCsv);
    {
        FILE *f = fopen(kTmpCsv, "w");
        REQUIRE(f != nullptr);
        fputs("date,time,model_name,only4\n", f); /* 4 columns != 29 */
        fclose(f);
    }
    ResultCollector c;
    BenchmarkRecord r = make_record();
    CHECK_FALSE(c.AppendCsv(r, kTmpCsv, "2026-08-21", "10:00:00", r.app_name.c_str()));
    std::remove(kTmpCsv);
}
