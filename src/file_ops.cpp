/*============================================================================
 * file_ops.cpp - File & dynamic library helpers
 *============================================================================*/

#include "file_ops.hpp"
#include "log.hpp"
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/stat.h>

#ifdef _WIN32
#include <io.h>
#include <windows.h>
#define access _access
#define F_OK 0
#else
#include <dlfcn.h>
#include <unistd.h>
#endif

/* ---------------------------------------------------------------------------
 * Dynamic library loading
 * -------------------------------------------------------------------------*/
void *load_library(const char *path)
{
#ifdef _WIN32
    std::wstring wpath = utf8_to_wide(path);
    /* LOAD_WITH_ALTERED_SEARCH_PATH (0x8): search dependencies in the DLL's
     * own directory first. Critical for EP-specific DLLs like dml\onnxruntime.dll
     * which depend on DirectML.dll in the same subdirectory. */
    HMODULE h = LoadLibraryExW(wpath.c_str(), nullptr,
                               LOAD_WITH_ALTERED_SEARCH_PATH);
    if (!h) {
        DWORD err = GetLastError();
        LOGE("LoadLibraryExW(%s) failed: %lu", path, (unsigned long)err);
    }
    return (void *)h;
#else
    void *h = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!h)
        LOGE("dlopen(%s) failed: %s", path, dlerror());
    return h;
#endif
}

void *load_function(void *lib, const char *name)
{
    if (!lib)
        return nullptr;
#ifdef _WIN32
    return (void *)GetProcAddress((HMODULE)lib, name);
#else
    return dlsym(lib, name);
#endif
}

void release_library(void *lib)
{
    if (!lib)
        return;
#ifdef _WIN32
    FreeLibrary((HMODULE)lib);
#else
    dlclose(lib);
#endif
}

/* ---------------------------------------------------------------------------
 * Path conversion
 * -------------------------------------------------------------------------*/
std::string wide_to_utf8(const wchar_t *wstr)
{
    if (!wstr || !*wstr)
        return {};
#ifdef _WIN32
    int len = WideCharToMultiByte(CP_UTF8, 0, wstr, -1, nullptr, 0, nullptr, nullptr);
    if (len <= 0)
        return {};
    std::string result(len - 1, '\0');
    WideCharToMultiByte(CP_UTF8, 0, wstr, -1, &result[0], len, nullptr, nullptr);
    return result;
#else
    (void)wstr;
    return {};
#endif
}

std::wstring utf8_to_wide(const char *str)
{
    if (!str || !*str)
        return {};
#ifdef _WIN32
    int len = MultiByteToWideChar(CP_UTF8, 0, str, -1, nullptr, 0);
    if (len <= 0)
        return {};
    std::wstring result(len - 1, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, str, -1, &result[0], len);
    return result;
#else
    (void)str;
    return {};
#endif
}

/* ---------------------------------------------------------------------------
 * File utilities
 * -------------------------------------------------------------------------*/
bool file_exists(const char *path)
{
    return access(path, F_OK) == 0;
}

bool file_readable_nonzero(const char *path)
{
    struct stat st;
    if (stat(path, &st) != 0)
        return false;
    if (st.st_size <= 0)
        return false;
    return access(path, /*R_OK*/ 4) == 0;
}

std::string build_model_cache_dir()
{
    /* Beside the executable */
#ifdef _WIN32
    wchar_t exe_path[MAX_PATH];
    if (!GetModuleFileNameW(nullptr, exe_path, MAX_PATH)) {
        DWORD err = GetLastError();
        LOGE("GetModuleFileNameW failed: %lu", (unsigned long)err);
        wcsncpy(exe_path, L".", MAX_PATH);
    }
    std::string exe_dir = wide_to_utf8(exe_path);
    auto pos = exe_dir.find_last_of("/\\");
    if (pos != std::string::npos)
        exe_dir = exe_dir.substr(0, pos);
    std::string dir = exe_dir + "\\model_cache\\";
    if (!CreateDirectoryW(utf8_to_wide(dir.c_str()).c_str(), nullptr) && GetLastError() != ERROR_ALREADY_EXISTS)
        LOGW("CreateDirectoryW(%s) failed: %lu", dir.c_str(), (unsigned long)GetLastError());
    return dir;
#else
    char exe_path[1024];
    ssize_t n = readlink("/proc/self/exe", exe_path, sizeof(exe_path) - 1);
    if (n > 0)
        exe_path[n] = '\0';
    else
        strncpy(exe_path, "./bench", sizeof(exe_path));
    std::string exe_dir = exe_path;
    auto pos = exe_dir.find_last_of('/');
    if (pos != std::string::npos)
        exe_dir = exe_dir.substr(0, pos);
    std::string dir = exe_dir + "/model_cache/";
    if (mkdir(dir.c_str(), 0755) != 0 && errno != EEXIST)
        LOGW("mkdir(%s) failed: %s, %d", dir.c_str(), strerror(errno), errno);
    return dir;
#endif
}

std::string build_compiled_model_path(const char *ep_name)
{
    return build_model_cache_dir() + "model_" + ep_name + ".onnxr";
}
