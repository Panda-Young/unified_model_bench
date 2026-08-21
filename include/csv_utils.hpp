#pragma once
/*============================================================================
 * csv_utils.hpp - CSV row-level helpers for worker result bookkeeping
 *
 * Pure line/field utilities used by the scheduler (worker exit-code merge,
 * baseline avg_run_ms round-trip, missing-row detection). No dependency on
 * the benchmark runner or any backend.
 *============================================================================*/

#include <string>
#include <vector>

/* CSV column indices of the 26-column schema written by result_collector.cpp.
 * Keep in sync with kCsvHeader there. weight_mem_mb sits at column 7 (right
 * after the output shapes); the timing block was pushed down accordingly. */
enum CsvCol {
    kCsvColTime = 1,        /* batch timestamp */
    kCsvColWeightMemMb = 7, /* weight memory (deployment info) */
    kCsvColAvgRunMs = 12,   /* avg_run_ms of a worker's own row */
    kCsvColBackendName = 21,
    kCsvColNotes = 25,
    kCsvColumnCount = 26
};

/* Parse one CSV line (quote-aware, handles "" escapes) into fields. */
std::vector<std::string> csv_parse_line(const std::string &line);

/* Quote a field when it contains ',', '"' or a newline; otherwise verbatim. */
std::string csv_quote_field(const std::string &s);

/* Sanitize a string for CSV storage: replace \n / \r with space, '"' with '. */
std::string csv_safe(const std::string &s);

/* Whether the CSV already contains a row for this (batch_time, backend_name).
 * Matches the backend_name COLUMN exactly - a substring search would let
 * "ONNX_CPU" match an "ONNX_CPU_FP16" row. Used by the scheduler to decide
 * whether a worker that exited with code 1 wrote its own failure row
 * (expected failure) or died without recording anything (silent loss). */
bool csv_has_backend_record(const std::string &path,
                            const std::string &batch_time,
                            const std::string &backend_name);

/* Read avg_run_ms (kCsvColAvgRunMs) of the (batch_time, backend_name) row.
 * Returns 0.0 when the row is missing or holds "-" (failed baseline). */
double csv_read_avg_run_ms(const std::string &path,
                           const std::string &batch_time,
                           const std::string &backend_name);

/* Append "note" to the notes column of the matching row (merging instead of
 * adding a second row). Returns true when a matching row exists and is
 * guaranteed to contain the note (idempotent - an already-present note still
 * counts as success); false only when no row matches. */
bool csv_append_note(const std::string &path, const std::string &batch_time,
                     const std::string &backend_name, const std::string &note);
