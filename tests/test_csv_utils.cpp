/*============================================================================
 * test_csv_utils.cpp - Unit tests for the CSV row-level utilities
 *============================================================================*/
#include "csv_utils.hpp"
#include "doctest.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

/* Build a CSV row with the full schema width (quotes applied). */
static std::string make_csv_row(const std::string &time,
                                const std::string &avg_ms,
                                const std::string &backend,
                                const std::string &notes)
{
    std::vector<std::string> f(kCsvColumnCount);
    f[kCsvColTime] = time;
    f[kCsvColAvgRunMs] = avg_ms;
    f[kCsvColBackendName] = backend;
    f[kCsvColNotes] = notes;
    std::string out;
    for (size_t i = 0; i < f.size(); ++i) {
        if (i > 0) {
            out += ',';
        }
        out += csv_quote_field(f[i]);
    }
    return out;
}

TEST_CASE("csv_parse_line: plain and quoted fields")
{
    auto f = csv_parse_line("a,b,c");
    REQUIRE(f.size() == 3);
    CHECK(f[0] == "a");
    CHECK(f[2] == "c");

    /* quoted field containing a comma */
    auto q = csv_parse_line("a,\"x,y\",c");
    REQUIRE(q.size() == 3);
    CHECK(q[1] == "x,y");

    /* escaped quotes "" inside a quoted field */
    auto e = csv_parse_line("\"he said \"\"hi\"\"\",2");
    REQUIRE(e.size() == 2);
    CHECK(e[0] == "he said \"hi\"");
}

TEST_CASE("csv_parse_line: empty fields and CRLF")
{
    auto f = csv_parse_line(",b,\r\n");
    REQUIRE(f.size() == 3);
    CHECK(f[0].empty());
    CHECK(f[1] == "b");
    CHECK(f[2].empty());
}

TEST_CASE("csv_quote_field: only quotes when needed")
{
    CHECK(csv_quote_field("plain") == "plain");
    CHECK(csv_quote_field("a,b") == "\"a,b\"");
    CHECK(csv_quote_field("say \"hi\"") == "\"say \"\"hi\"\"\"");
}

TEST_CASE("csv_safe: sanitizes newlines and quotes")
{
    CHECK(csv_safe("line1\nline2") == "line1 line2");
    CHECK(csv_safe("cr\rhere") == "cr here");
    CHECK(csv_safe("a\"b") == "a'b");
    CHECK(csv_safe("clean") == "clean");
}

/* ---------------------------------------------------------------------------
 * Row lookups against a real CSV file (temporary)
 * -------------------------------------------------------------------------*/
namespace {

const char *kTmpCsv = "ut_csv_utils_tmp.csv";

/* header + 2 data rows sharing one batch_time */
void write_fixture_csv(const char *path)
{
    FILE *f = fopen(path, "w");
    REQUIRE(f != nullptr);
    fputs("date,time,model_name,model_input_shape,input_elements,"
          "model_output_shape,output_elements,warmup_runs,repeat_runs,threads,"
          "total_run_ms,avg_run_ms,max_run_ms,max_run_idx,init_ms,"
          "max_output_diff,avg_output_diff,acceleration_vs_cpu,"
          "weight_mem_mb,peak_mem_mb,resident_mem_mb,"
          "backend_name,device_info,arch,app_name,notes\n", f);
    fputs((make_csv_row("10:00:00", "12.500", "ONNX_CPU", "ok") + "\n").c_str(), f);
    fputs((make_csv_row("10:00:00", "-", "DML_GPU", "failed") + "\n").c_str(), f);
    fclose(f);
}

} // namespace

TEST_CASE("csv_has_backend_record: exact column match")
{
    write_fixture_csv(kTmpCsv);
    /* backend_name column matches */
    CHECK(csv_has_backend_record(kTmpCsv, "10:00:00", "ONNX_CPU"));
    CHECK(csv_has_backend_record(kTmpCsv, "10:00:00", "DML_GPU"));
    /* substring trap: "ONNX_CPU" must NOT match a row whose backend is
     * "ONNX_CPU_FP16" - and vice versa */
    CHECK_FALSE(csv_has_backend_record(kTmpCsv, "10:00:00", "ONNX_CPU_FP16"));
    CHECK_FALSE(csv_has_backend_record(kTmpCsv, "10:00:00", "CPU"));
    /* wrong batch_time */
    CHECK_FALSE(csv_has_backend_record(kTmpCsv, "11:00:00", "ONNX_CPU"));
    /* missing file */
    CHECK_FALSE(csv_has_backend_record("ut_no_such_file.csv", "10:00:00", "ONNX_CPU"));
    std::remove(kTmpCsv);
}

TEST_CASE("csv_read_avg_run_ms: reads value / dash / missing")
{
    write_fixture_csv(kTmpCsv);
    CHECK(csv_read_avg_run_ms(kTmpCsv, "10:00:00", "ONNX_CPU") == doctest::Approx(12.500));
    /* "-" (failed baseline) -> 0.0 */
    CHECK(csv_read_avg_run_ms(kTmpCsv, "10:00:00", "DML_GPU") == 0.0);
    /* no matching row -> 0.0 */
    CHECK(csv_read_avg_run_ms(kTmpCsv, "10:00:00", "NCNN_CPU") == 0.0);
    std::remove(kTmpCsv);
}

TEST_CASE("csv_append_note: merges into notes, deduplicates")
{
    write_fixture_csv(kTmpCsv);
    CHECK(csv_append_note(kTmpCsv, "10:00:00", "ONNX_CPU", "crashed code 5"));
    /* second append with the same note is a no-op (returns true, no dup) */
    CHECK(csv_append_note(kTmpCsv, "10:00:00", "ONNX_CPU", "crashed code 5"));

    FILE *f = fopen(kTmpCsv, "r");
    REQUIRE(f != nullptr);
    char line[4096];
    int count = 0;
    while (fgets(line, sizeof(line), f)) {
        if (strstr(line, "ONNX_CPU")) {
            ++count;
            CHECK(strstr(line, "crashed code 5") != nullptr);
        }
    }
    fclose(f);
    CHECK(count == 1); /* still exactly one ONNX_CPU row */
    std::remove(kTmpCsv);
}
