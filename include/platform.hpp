#pragma once
/*============================================================================
 * platform.hpp - Platform abstraction & shared constants
 *============================================================================*/

/* glibc only exposes clock_gettime()/CLOCK_MONOTONIC when a POSIX feature
 * macro is defined before <time.h> is first included.  Define _DEFAULT_SOURCE
 * up front so the Linux timer code below compiles regardless of which system
 * headers a translation unit pulls in first. */
#if defined(__linux__) && !defined(_DEFAULT_SOURCE)
#define _DEFAULT_SOURCE
#endif

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#endif

/* ---------------------------------------------------------------------------
 * Architecture string
 * -------------------------------------------------------------------------*/
#if defined(__ANDROID__) || defined(__android__)
#define ARCH_STR "android_aarch64"
#elif defined(_WIN64)
#define ARCH_STR "win_x64"
#elif defined(_WIN32)
#define ARCH_STR "win_x86"
#elif defined(__linux__)
#define ARCH_STR "linux_x64"
#else
#define ARCH_STR "unknown"
#endif

/* ---------------------------------------------------------------------------
 * Constants (shared across all modules)
 * -------------------------------------------------------------------------*/
constexpr size_t MAX_DIMENSIONS = 8;
constexpr size_t MAX_PATH_LEN = 1024;
constexpr size_t MAX_IO = 128; /* max inputs/outputs per model */

/* ---------------------------------------------------------------------------
 * Platform-specific path separator
 * -------------------------------------------------------------------------*/
#ifdef _WIN32
constexpr char PATH_SEP = '\\';
#else
constexpr char PATH_SEP = '/';
#endif

/* ---------------------------------------------------------------------------
 * High-resolution monotonic timer (ms)
 * -------------------------------------------------------------------------*/
inline double get_time_ms()
{
#ifdef _WIN32
    LARGE_INTEGER freq, cnt;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&cnt);
    return (double)cnt.QuadPart * 1000.0 / (double)freq.QuadPart;
#elif defined(__ANDROID__) || defined(__linux__)
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
#else
#error "Unsupported platform for get_time_ms()"
#endif
}

/* ---------------------------------------------------------------------------
 * Process memory usage (MB). Returns false if the OS counters are unavailable.
 *  - peak_mb:     process peak working set / RSS (Windows PeakWorkingSetSize,
 *                 Linux VmHWM) - monotonic since process start
 *  - resident_mb: current working set / RSS (Windows WorkingSetSize,
 *                 Linux VmRSS)
 * Both are PROCESS-level aggregates (framework arena, thread stacks, DLLs,
 * weights and activations all included); they are NOT per-tensor values.
 * -------------------------------------------------------------------------*/
inline bool get_process_mem_mb(double &peak_mb, double &resident_mb)
{
    peak_mb = 0.0;
    resident_mb = 0.0;
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS pmc;
    if (!GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        return false;
    }
    peak_mb = (double)pmc.PeakWorkingSetSize / (1024.0 * 1024.0);
    resident_mb = (double)pmc.WorkingSetSize / (1024.0 * 1024.0);
    return true;
#elif defined(__ANDROID__) || defined(__linux__)
    FILE *f = fopen("/proc/self/status", "r");
    if (!f) {
        return false;
    }
    char line[256];
    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "VmHWM:", 6) == 0) {
            peak_mb = atof(line + 6) / 1024.0; /* kB -> MB */
        } else if (strncmp(line, "VmRSS:", 6) == 0) {
            resident_mb = atof(line + 6) / 1024.0; /* kB -> MB */
        }
    }
    fclose(f);
    return (peak_mb > 0.0 || resident_mb > 0.0);
#else
    (void)peak_mb;
    (void)resident_mb;
    return false;
#endif
}

/* ---------------------------------------------------------------------------
 * Helper: extract filename from __FILE__ (handles both / and \)
 * -------------------------------------------------------------------------*/
constexpr const char *filename_from_path(const char *path)
{
    const char *p = path;
    while (*path) {
        if (*path == '/' || *path == '\\') {
            p = path + 1;
        }
        ++path;
    }
    return p;
}
#define FILENAME__ filename_from_path(__FILE__)

/* ---------------------------------------------------------------------------
 * Helper: cross-platform case-insensitive string compare
 * -------------------------------------------------------------------------*/
inline int stricmp_(const char *a, const char *b)
{
#ifdef _WIN32
    return _stricmp(a, b);
#else
    return strcasecmp(a, b);
#endif
}
inline int numeric_suffix(const std::string &s)
{
    size_t pos = s.length();
    while (pos > 0 && isdigit(static_cast<unsigned char>(s[pos - 1]))) {
        --pos;
    }
    return (pos < s.length()) ? std::atoi(s.c_str() + pos) : 0;
}
