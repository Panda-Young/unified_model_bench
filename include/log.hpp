#pragma once
/*============================================================================
 * log.hpp - Lightweight logging system (C++ header-only)
 *============================================================================*/

#include "platform.hpp"
#include <cstdarg>
#include <cstdio>
#include <ctime>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
/* Windows SDK pollutes global namespace with macros that conflict with our enum */
#ifdef ERROR
#undef ERROR
#endif
#ifdef WARN
#undef WARN
#endif
#ifdef INFO
#undef INFO
#endif
#ifdef DEBUG
#undef DEBUG
#endif
#else
#include <sys/syscall.h>
#include <unistd.h>
#endif

/* ---------------------------------------------------------------------------
 * Log levels
 * -------------------------------------------------------------------------*/
enum class LogLevel { OFF = 0,
                      DBG = 1,
                      INFO = 2,
                      WARN = 3,
                      ERR = 4 };

/* Re-export as convenience macros using the clean names */
constexpr LogLevel LOG_ERR = LogLevel::ERR;
constexpr LogLevel LOG_WARN = LogLevel::WARN;
constexpr LogLevel LOG_INFO = LogLevel::INFO;
constexpr LogLevel LOG_DBG = LogLevel::DBG;

/* ---------------------------------------------------------------------------
 * Logger class - all static, minimal overhead
 * -------------------------------------------------------------------------*/
class Logger
{
public:
    static LogLevel level;
    static const char *prog_name;

    static void init(const char *name) { prog_name = name; }

    static void log(LogLevel lv, const char *file, int line,
                    const char *func, const char *fmt, ...)
    {
        if (lv > level) {
            return;
        }

        /* Timestamp */
        time_t now = time(nullptr);
        struct tm tm_buf;
#ifdef _WIN32
        localtime_s(&tm_buf, &now);
#else
        localtime_r(&now, &tm_buf);
#endif
        char ts[32];
        strftime(ts, sizeof(ts), "%Y-%m-%d %H:%M:%S", &tm_buf);

        /* PID / TID */
#ifdef _WIN32
        DWORD pid = GetCurrentProcessId();
        DWORD tid = GetCurrentThreadId();
#else
        pid_t pid = getpid();
#ifdef SYS_gettid
        pid_t tid = syscall(SYS_gettid);
#else
        pid_t tid = pid;
#endif
#endif

        const char *lv_str = "?";
        const char *color = "";
        switch (lv) {
        case LogLevel::ERR:
            lv_str = "ERROR";
            color = "\033[1;31m";
            break;
        case LogLevel::WARN:
            lv_str = "WARN";
            color = "\033[1;33m";
            break;
        case LogLevel::INFO:
            lv_str = "INFO";
            color = "\033[1;32m";
            break;
        case LogLevel::DBG:
            lv_str = "DEBUG";
            color = "\033[1;36m";
            break;
        default:
            break;
        }
        (void)lv_str; /* suppress unused warning on non-color terminals */

        fprintf(stderr, "%s%s.%03ld %s [%d.%d] %s:%d @%s  ",
                color, ts, 0L, prog_name ? prog_name : "bench",
                (int)pid, (int)tid, file, line, func);

        va_list ap;
        va_start(ap, fmt);
        vfprintf(stderr, fmt, ap);
        va_end(ap);
        fprintf(stderr, "\033[0m\n");
        fflush(stderr);
    }
};

/* Convenience macros */
#define LOGE(fmt, ...) Logger::log(LogLevel::ERR, FILENAME__, __LINE__, __FUNCTION__, fmt, ##__VA_ARGS__)
#define LOGW(fmt, ...) Logger::log(LogLevel::WARN, FILENAME__, __LINE__, __FUNCTION__, fmt, ##__VA_ARGS__)
#define LOGI(fmt, ...) Logger::log(LogLevel::INFO, FILENAME__, __LINE__, __FUNCTION__, fmt, ##__VA_ARGS__)
#define LOGD(fmt, ...) Logger::log(LogLevel::DBG, FILENAME__, __LINE__, __FUNCTION__, fmt, ##__VA_ARGS__)
