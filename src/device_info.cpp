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
    while (b < e && (s[b] == ' ' || s[b] == '\t' || s[b] == '\r' || s[b] == '\n'))
        ++b;
    while (e > b && (s[e - 1] == ' ' || s[e - 1] == '\t' || s[e - 1] == '\r' || s[e - 1] == '\n'))
        --e;
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
    if (!p)
        return {};
    char buf[1024];
    while (fgets(buf, sizeof(buf), p))
        result += buf;
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
    if (bytes >= 1073741824ULL)
        snprintf(buf, sizeof(buf), "%.2f GB", bytes / 1073741824.0);
    else
        snprintf(buf, sizeof(buf), "%.2f GB", bytes / 1048576.0);
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
    while (b < s.length() && !alpha(s[b]))
        ++b;
    size_t e = s.length();
    while (e > b && !alpha(s[e - 1]))
        --e;
    return s.substr(b, e - b);
}

std::string get_device_info_csv()
{
    std::ostringstream oss;

    /* SOC */
    std::string soc = android_getprop("ro.soc.model");
    if (soc.empty())
        soc = android_getprop("ro.board.platform");
    if (soc.empty())
        soc = android_getprop("ro.product.board");
    if (soc.empty())
        soc = "Unknown";

    /* CPU cores */
    std::string cpuinfo = run_cmd("grep -c ^processor /proc/cpuinfo 2>/dev/null");
    int cores = atoi(trim(cpuinfo).c_str());
    if (cores <= 0)
        cores = (int)sysconf(_SC_NPROCESSORS_CONF);

    /* GPU */
    std::string gpu;
    FILE *gf = fopen("/sys/class/kgsl/kgsl-3d0/gpu_model", "r");
    if (gf) {
        char buf[256] = {};
        fgets(buf, sizeof(buf), gf);
        if (fclose(gf) != 0)
            LOGW("fclose(gpu_model) failed: %s, %d", strerror(errno), errno);
        gpu = trim(buf);
    }
    if (gpu.empty())
        gpu = android_getprop("ro.vendor.gpu");

    oss << "CPU: " << simplify_soc_name(soc)
        << "; Logical: " << cores;
    if (!gpu.empty())
        oss << "; GPU: " << gpu;

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

    /* WMIC fallback */
    if (cpu_name.empty()) {
        std::string wmic = run_cmd("wmic cpu get Name,NumberOfCores,NumberOfLogicalProcessors,MaxClockSpeed 2>nul");
        /* Parse: skip header, take first data line */
    }

    /* GetSystemInfo fallback */
    if (cpu_name.empty()) {
        SYSTEM_INFO si;
        GetSystemInfo(&si);
        char buf[64];
        snprintf(buf, sizeof(buf), "%u-core CPU @ ? GHz", si.dwNumberOfProcessors);
        cpu_name = buf;
    }

    /* Parse cores from cpu_name or WMIC */
    {
        std::string wmic = run_cmd("wmic cpu get NumberOfLogicalProcessors 2>nul");
        auto pos = wmic.find('\n');
        if (pos != std::string::npos) {
            std::string line = trim(wmic.substr(pos + 1));
            int c = atoi(line.c_str());
            if (c > 0)
                cpu_cores = std::to_string(c);
        }
    }
    if (cpu_cores.empty()) {
        SYSTEM_INFO si;
        GetSystemInfo(&si);
        cpu_cores = std::to_string(si.dwNumberOfProcessors);
    }

    /* GPU: WMIC */
    std::string gpu_info = run_cmd("wmic path win32_videocontroller get Name,AdapterRAM 2>nul");
    auto pos1 = gpu_info.find('\n');
    if (pos1 != std::string::npos) {
        auto pos2 = gpu_info.find('\n', pos1 + 1);
        if (pos2 != std::string::npos) {
            std::string line = trim(gpu_info.substr(pos1 + 1, pos2 - pos1 - 1));
            /* Name comes first, then AdapterRAM */
            auto comma = line.rfind(',');
            if (comma != std::string::npos) {
                gpu_ram = trim(line.substr(comma + 1));
                gpu_name = trim(line.substr(0, comma));
            } else {
                gpu_name = line;
            }
            /* Convert RAM bytes to human-readable */
            if (!gpu_ram.empty()) {
                char *end;
                uint64_t bytes = strtoull(gpu_ram.c_str(), &end, 10);
                if (bytes > 0)
                    gpu_ram = human_readable_bytes(bytes);
            }
        }
    }

    /* Simplify CPU name */
    {
        std::string s = cpu_name;
        /* Extract model like "i5-1135G7" */
        auto pos = s.find("CPU");
        if (pos == std::string::npos)
            pos = s.find("Core");
        if (pos != std::string::npos) {
            /* Try to find freq */
            auto ghz = s.find("GHz");
            std::string freq;
            if (ghz != std::string::npos && ghz > 0) {
                auto start = s.rfind(' ', ghz);
                if (start != std::string::npos)
                    freq = s.substr(start + 1, ghz - start + 2);
            }
            cpu_name = trim(s.substr(0, pos + 12)); /* approximate */
            if (!freq.empty())
                cpu_name += " " + freq;
        }
    }

#else /* Linux */
    /* CPU */
    std::string cpuinfo = run_cmd("grep 'model name' /proc/cpuinfo 2>/dev/null | head -1");
    auto colon = cpuinfo.find(':');
    if (colon != std::string::npos)
        cpu_name = trim(cpuinfo.substr(colon + 1));

    std::string nproc = run_cmd("nproc 2>/dev/null");
    cpu_cores = trim(nproc);

    /* GPU */
    std::string lspci = run_cmd("lspci 2>/dev/null | grep -i vga | head -1");
    if (!lspci.empty()) {
        auto pos = lspci.find(':');
        if (pos != std::string::npos)
            gpu_name = trim(lspci.substr(pos + 1));
    }
#endif

    oss << "CPU: " << cpu_name;
    if (!cpu_cores.empty())
        oss << "; Cores: " << cpu_cores;
    if (!gpu_name.empty()) {
        oss << "; GPU: " << gpu_name;
        if (!gpu_ram.empty())
            oss << " (RAM:" << gpu_ram << ")";
    }

    return oss.str();
}
#endif
