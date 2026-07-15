/*============================================================================
 * device_info.cpp - Device information collection
 *============================================================================*/

#include "device_info.hpp"
#include "log.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>

#ifdef _WIN32
#include <comdef.h>
#include <windows.h>
#else
#include <sys/utsname.h>
#include <unistd.h>
#endif

/* ---------------------------------------------------------------------------
 * Helpers
 * -------------------------------------------------------------------------*/
static std::string trim(const std::string &s)
{
    size_t b = 0, e = s.length();
    while (b < e && (s[b] == ' ' || s[b] == '\t' || s[b] == '\r' || s[b] == '\n')) {
        ++b;
    }
    while (e > b && (s[e - 1] == ' ' || s[e - 1] == '\t' || s[e - 1] == '\r' || s[e - 1] == '\n')) {
        --e;
    }
    return s.substr(b, e - b);
}

static std::string run_cmd(const char *cmd)
{
    std::string result;
#ifdef _WIN32
    FILE *p = _popen(cmd, "r");
#else
    FILE *p = popen(cmd, "r");
#endif
    if (!p) {
        return {};
    }
    char buf[1024];
    while (fgets(buf, sizeof(buf), p)) {
        result += buf;
    }
#ifdef _WIN32
    _pclose(p);
#else
    pclose(p);
#endif
    return result;
}

#ifdef _WIN32
static std::string human_readable_bytes(uint64_t bytes)
{
    char buf[64];
    if (bytes >= 1073741824ULL) {
        snprintf(buf, sizeof(buf), "%.2f GB", bytes / 1073741824.0);
    } else if (bytes >= 1048576ULL) {
        snprintf(buf, sizeof(buf), "%.2f MB", bytes / 1048576.0);
    } else if (bytes >= 1024ULL) {
        snprintf(buf, sizeof(buf), "%.2f KB", bytes / 1024.0);
    } else {
        snprintf(buf, sizeof(buf), "%llu B", (unsigned long long)bytes);
    }
    return buf;
}
#endif

#if defined(__ANDROID__) || defined(__android__)
/* ---------------------------------------------------------------------------
 * Android device info
 * -------------------------------------------------------------------------*/
static std::string android_getprop(const char *key)
{
    char cmd[256];
    snprintf(cmd, sizeof(cmd), "getprop %s 2>/dev/null", key);
    return trim(run_cmd(cmd));
}

static std::string simplify_soc_name(const std::string &raw)
{
    /* Extract the key model token */
    std::string s = raw;
    /* Remove leading/trailing non-alphanumeric */
    auto alpha = [](char c) { return isalnum((unsigned char)c) || c == '-'; };
    size_t b = 0;
    while (b < s.length() && !alpha(s[b])) {
        ++b;
    }
    size_t e = s.length();
    while (e > b && !alpha(s[e - 1])) {
        --e;
    }
    return s.substr(b, e - b);
}

std::string get_device_info_csv()
{
    std::ostringstream oss;

    /* SOC */
    std::string soc = android_getprop("ro.soc.model");
    if (soc.empty()) {
        soc = android_getprop("ro.board.platform");
    }
    if (soc.empty()) {
        soc = android_getprop("ro.product.board");
    }
    if (soc.empty()) {
        soc = "Unknown";
    }

    /* CPU cores */
    std::string cpuinfo = run_cmd("grep -c ^processor /proc/cpuinfo 2>/dev/null");
    int cores = atoi(trim(cpuinfo).c_str());
    if (cores <= 0) {
        cores = (int)sysconf(_SC_NPROCESSORS_CONF);
    }

    /* GPU */
    std::string gpu;
    FILE *gf = fopen("/sys/class/kgsl/kgsl-3d0/gpu_model", "r");
    if (gf) {
        char buf[256] = {};
        fgets(buf, sizeof(buf), gf);
        if (fclose(gf) != 0) {
            LOGW("fclose(gpu_model) failed: %s, %d", strerror(errno), errno);
        }
        gpu = trim(buf);
    }
    if (gpu.empty()) {
        gpu = android_getprop("ro.vendor.gpu");
    }

    oss << "CPU: " << simplify_soc_name(soc)
        << "; Logical: " << cores;
    if (!gpu.empty()) {
        oss << "; GPU: " << gpu;
    }

    return oss.str();
}

#else /* Desktop Windows / Linux */
/* ---------------------------------------------------------------------------
 * Windows / Linux device info
 * -------------------------------------------------------------------------*/
std::string get_device_info_csv()
{
    std::ostringstream oss;
    std::string cpu_name;
    std::string cpu_cores;
    std::string gpu_name;
    std::string gpu_ram;

#ifdef _WIN32
    /* CPU: try registry first */
    int cpu_mhz = 0;
    HKEY hKey;
    if (RegOpenKeyExA(HKEY_LOCAL_MACHINE,
                      "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0",
                      0, KEY_READ, &hKey) == ERROR_SUCCESS) {
        char buf[256] = {};
        DWORD sz = sizeof(buf);
        if (RegQueryValueExA(hKey, "ProcessorNameString", nullptr, nullptr,
                             (LPBYTE)buf, &sz) == ERROR_SUCCESS) {
            cpu_name = trim(buf);
        }
        RegCloseKey(hKey);
    }

    /* MaxClockSpeed: WMIC first, PowerShell fallback (WMIC removed in Win11 24H2) */
    {
        std::string raw = run_cmd("wmic cpu get MaxClockSpeed /format:list 2>nul");
        if (raw.find("MaxClockSpeed=") == std::string::npos) {
            raw = run_cmd("powershell -NoProfile -Command \"Get-CimInstance -ClassName Win32_Processor | Select-Object -First 1 MaxClockSpeed | Format-List\"");
        }
        /* Parse with flexible "Key...: Value" or "Key=Value" */
        auto parse_key = [&](const std::string &key) -> std::string {
            /* Try Key=Value first */
            auto p = raw.find(key + "=");
            if (p != std::string::npos) {
                p += key.length() + 1;
                auto e = raw.find('\n', p);
                return trim(raw.substr(p, e == std::string::npos ? e : e - p));
            }
            /* Try Key...: Value */
            p = raw.find(key);
            if (p != std::string::npos) {
                p += key.length();
                while (p < raw.length() && (raw[p] == ' ' || raw[p] == '\t'))
                    ++p;
                if (p < raw.length() && raw[p] == ':') {
                    ++p;
                    while (p < raw.length() && (raw[p] == ' ' || raw[p] == '\t'))
                        ++p;
                    auto e = raw.find('\n', p);
                    return trim(raw.substr(p, e == std::string::npos ? e : e - p));
                }
            }
            return {};
        };
        std::string val = parse_key("MaxClockSpeed");
        if (!val.empty())
            cpu_mhz = atoi(val.c_str());
    }

    /* CPU cores from WMIC, fallback to GetSystemInfo */
    {
        std::string wmic = run_cmd("wmic cpu get NumberOfCores,NumberOfLogicalProcessors /format:list 2>nul");
        /* Parse key=value pairs */
        auto parse_val = [&](const std::string &key) -> std::string {
            auto p = wmic.find(key + "=");
            if (p == std::string::npos)
                return {};
            p += key.length() + 1;
            auto eol = wmic.find('\n', p);
            std::string v = wmic.substr(p, eol == std::string::npos ? eol : eol - p);
            return trim(v);
        };
        std::string nc = parse_val("NumberOfCores");
        std::string nl = parse_val("NumberOfLogicalProcessors");
        if (!nc.empty())
            cpu_cores = nc;
        if (cpu_cores.empty() && !nl.empty())
            cpu_cores = nl;
    }
    if (cpu_cores.empty()) {
        SYSTEM_INFO si;
        GetSystemInfo(&si);
        cpu_cores = std::to_string(si.dwNumberOfProcessors);
    }

    /* GPU: WMIC with /format:list to enumerate all GPUs (skip remote display adapter) */
    {
        std::string raw = run_cmd("wmic path win32_videocontroller get Name,AdapterRAM /format:list 2>nul");
        if (raw.empty()) {
            raw = run_cmd("powershell -NoProfile -Command \"Get-CimInstance Win32_VideoController | Select-Object Name,AdapterRAM | Format-List\"");
        }
        /* Parse into blocks separated by blank lines */
        std::string gpu_list;
        std::string cur_block;
        size_t i = 0;
        while (i < raw.length()) {
            size_t eol = raw.find('\n', i);
            std::string line = raw.substr(i, eol == std::string::npos ? eol : eol - i);
            /* Remove trailing \r */
            if (!line.empty() && line.back() == '\r')
                line.pop_back();
            if (line.empty()) {
                /* Blank line = end of a block */
                if (!cur_block.empty()) {
                    /* Extract Name and AdapterRAM from this block */
                    auto get_val = [&](const std::string &key) -> std::string {
                        /* Try "Key=Value" (WMIC /format:list) first */
                        auto p = cur_block.find(key + "=");
                        if (p != std::string::npos) {
                            p += key.length() + 1;
                            auto e = cur_block.find('\n', p);
                            return trim(cur_block.substr(p, e == std::string::npos ? e : e - p));
                        }
                        /* Try "Key...: Value" (PowerShell Format-List, variable whitespace before colon) */
                        p = cur_block.find(key);
                        if (p != std::string::npos) {
                            p += key.length();
                            /* Skip whitespace to colon */
                            while (p < cur_block.length() && (cur_block[p] == ' ' || cur_block[p] == '\t'))
                                ++p;
                            if (p < cur_block.length() && cur_block[p] == ':') {
                                ++p; /* skip colon */
                                while (p < cur_block.length() && (cur_block[p] == ' ' || cur_block[p] == '\t'))
                                    ++p;
                                auto e = cur_block.find('\n', p);
                                return trim(cur_block.substr(p, e == std::string::npos ? e : e - p));
                            }
                        }
                        return {};
                    };
                    std::string name = get_val("Name");
                    std::string ram = get_val("AdapterRAM");
                    if (!name.empty()) {
                        /* Lowercase check for "Microsoft Remote Display Adapter" */
                        std::string lower = name;
                        for (auto &ch : lower)
                            ch = (char)tolower((unsigned char)ch);
                        if (!(lower.find("remote") != std::string::npos &&
                              lower.find("display") != std::string::npos)) {
                            if (!gpu_list.empty())
                                gpu_list += "; ";
                            gpu_list += name;
                            if (!ram.empty()) {
                                char *end = nullptr;
                                uint64_t bytes = strtoull(ram.c_str(), &end, 10);
                                if (bytes > 0) {
                                    gpu_list += " (RAM:" + human_readable_bytes(bytes) + ")";
                                }
                            }
                        }
                    }
                    cur_block.clear();
                }
            } else {
                if (!cur_block.empty())
                    cur_block += '\n';
                cur_block += line;
            }
            if (eol == std::string::npos)
                break;
            i = eol + 1;
        }
        /* Last block */
        if (!cur_block.empty()) {
            auto get_val = [&](const std::string &key) -> std::string {
                auto p = cur_block.find(key + "=");
                if (p != std::string::npos) {
                    p += key.length() + 1;
                    auto e = cur_block.find('\n', p);
                    return trim(cur_block.substr(p, e == std::string::npos ? e : e - p));
                }
                p = cur_block.find(key);
                if (p != std::string::npos) {
                    p += key.length();
                    while (p < cur_block.length() && (cur_block[p] == ' ' || cur_block[p] == '\t'))
                        ++p;
                    if (p < cur_block.length() && cur_block[p] == ':') {
                        ++p;
                        while (p < cur_block.length() && (cur_block[p] == ' ' || cur_block[p] == '\t'))
                            ++p;
                        auto e = cur_block.find('\n', p);
                        return trim(cur_block.substr(p, e == std::string::npos ? e : e - p));
                    }
                }
                return {};
            };
            std::string name = get_val("Name");
            std::string ram = get_val("AdapterRAM");
            if (!name.empty()) {
                std::string lower = name;
                for (auto &ch : lower)
                    ch = (char)tolower((unsigned char)ch);
                if (!(lower.find("remote") != std::string::npos &&
                      lower.find("display") != std::string::npos)) {
                    if (!gpu_list.empty())
                        gpu_list += "; ";
                    gpu_list += name;
                    if (!ram.empty()) {
                        char *end = nullptr;
                        uint64_t bytes = strtoull(ram.c_str(), &end, 10);
                        if (bytes > 0) {
                            gpu_list += " (RAM:" + human_readable_bytes(bytes) + ")";
                        }
                    }
                }
            }
        }
        /* Split into name and RAM for output */
        if (!gpu_list.empty()) {
            gpu_name = gpu_list;
        }
    }

    /* Simplify CPU name: extract model token + frequency */
    if (!cpu_name.empty()) {
        std::string s = cpu_name;
        /* Find frequency */
        std::string freq;
        auto at_pos = s.find('@');
        if (at_pos != std::string::npos) {
            freq = trim(s.substr(at_pos + 1));
            s = trim(s.substr(0, at_pos));
        } else {
            /* Case-insensitive search for GHz frequency marker */
            std::string lower_s = s;
            for (auto &ch : lower_s)
                ch = (char)tolower((unsigned char)ch);
            auto ghz_pos = lower_s.find("ghz");
            if (ghz_pos != std::string::npos) {
                /* Find the start of the frequency token */
                auto start = ghz_pos;
                while (start > 0 && s[start] != ' ' && s[start] != '\t')
                    --start;
                if (start > 0 && (s[start - 1] == ' ' || s[start - 1] == '\t'))
                    --start; /* include the space before frequency */
                freq = trim(s.substr(start));
                s = trim(s.substr(0, start));
            }
        }
        /* Find the last token with at least one digit and at least one letter/dash */
        std::string best;
        size_t pos = 0;
        while (pos < s.length()) {
            while (pos < s.length() && (s[pos] == ' ' || s[pos] == '\t'))
                ++pos;
            if (pos >= s.length())
                break;
            size_t end = pos;
            while (end < s.length() && s[end] != ' ' && s[end] != '\t')
                ++end;
            std::string tok = s.substr(pos, end - pos);
            bool has_digit = false, has_alpha_or_dash = false;
            for (auto ch : tok) {
                if (isdigit((unsigned char)ch))
                    has_digit = true;
                if (isalpha((unsigned char)ch) || ch == '-')
                    has_alpha_or_dash = true;
            }
            if (has_digit && has_alpha_or_dash)
                best = tok;
            pos = end;
        }
        if (!best.empty()) {
            cpu_name = best;
        } else {
            /* Fallback: take last token */
            auto last_space = s.rfind(' ');
            if (last_space != std::string::npos) {
                cpu_name = trim(s.substr(last_space + 1));
            } else if (!s.empty()) {
                cpu_name = s;
            }
        }
        if (!freq.empty()) {
            cpu_name += " " + freq;
        } else if (cpu_mhz > 0) {
            /* Append frequency from registery ~MHz (e.g. 2700 -> "2.70 GHz") */
            char ghzbuf[32];
            snprintf(ghzbuf, sizeof(ghzbuf), "%.2f GHz", cpu_mhz / 1000.0);
            cpu_name += " " + std::string(ghzbuf);
        }
    }

#else /* Linux */
    /* CPU */
    std::string cpuinfo = run_cmd("grep 'model name' /proc/cpuinfo 2>/dev/null | head -1");
    auto colon = cpuinfo.find(':');
    if (colon != std::string::npos) {
        cpu_name = trim(cpuinfo.substr(colon + 1));
    }

    std::string nproc = run_cmd("nproc 2>/dev/null");
    cpu_cores = trim(nproc);

    /* GPU */
    std::string lspci = run_cmd("lspci 2>/dev/null | grep -i vga | head -1");
    if (!lspci.empty()) {
        auto pos = lspci.find(':');
        if (pos != std::string::npos) {
            gpu_name = trim(lspci.substr(pos + 1));
        }
    }
#endif

    oss << "CPU: " << cpu_name;
    if (!cpu_cores.empty()) {
        oss << "; Cores: " << cpu_cores;
    }
    if (!gpu_name.empty()) {
        oss << "; GPU: " << gpu_name;
        if (!gpu_ram.empty()) {
            oss << " (RAM:" << gpu_ram << ")";
        }
    }

    return oss.str();
}
#endif
