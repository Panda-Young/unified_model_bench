/*============================================================================
 * csv_utils.cpp - CSV row-level helpers for worker result bookkeeping
 *============================================================================*/

#include "csv_utils.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

/* ---------------------------------------------------------------------------
 * Field-level helpers
 * -------------------------------------------------------------------------*/
std::vector<std::string> csv_parse_line(const std::string &line)
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

std::string csv_quote_field(const std::string &s)
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

std::string csv_safe(const std::string &s)
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
 * Row lookups (scheduler bookkeeping)
 * -------------------------------------------------------------------------*/
bool csv_has_backend_record(const std::string &path,
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
        auto flds = csv_parse_line(line);
        if (flds.size() >= (size_t)kCsvColumnCount &&
            flds[kCsvColTime] == batch_time &&
            flds[kCsvColBackendName] == backend_name) {
            found = true;
            break;
        }
    }
    fclose(f);
    return found;
}

double csv_read_avg_run_ms(const std::string &path,
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
        if (flds.size() < (size_t)kCsvColumnCount ||
            flds[kCsvColTime] != batch_time ||
            flds[kCsvColBackendName] != backend_name) {
            continue;
        }
        if (flds[kCsvColAvgRunMs].size() > 0 && flds[kCsvColAvgRunMs] != "-") {
            avg = atof(flds[kCsvColAvgRunMs].c_str());
        }
        break;
    }
    fclose(f);
    return avg;
}

bool csv_append_note(const std::string &path, const std::string &batch_time,
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
        if (flds.size() < (size_t)kCsvColumnCount ||
            flds[kCsvColTime] != batch_time ||
            flds[kCsvColBackendName] != backend_name) {
            continue;
        }
        std::string &notes = flds[kCsvColNotes];
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
        }
        changed = true;
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
