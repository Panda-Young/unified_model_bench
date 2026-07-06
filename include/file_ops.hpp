#pragma once
/*============================================================================
 * file_ops.hpp - File & dynamic library helpers
 *============================================================================*/

#include <string>

/* ---------------------------------------------------------------------------
 * Dynamic library loading (RAII wrapper not needed; manual for flexibility)
 * -------------------------------------------------------------------------*/
void* load_library(const char* path);
void* load_function(void* lib, const char* name);
void  release_library(void* lib);

/* ---------------------------------------------------------------------------
 * Path conversion (Windows wide-char ↔ UTF-8)
 * -------------------------------------------------------------------------*/
std::string  wide_to_utf8(const wchar_t* wstr);
std::wstring utf8_to_wide(const char* str);

/* ---------------------------------------------------------------------------
 * File utilities
 * -------------------------------------------------------------------------*/
bool file_exists(const char* path);
bool file_readable_nonzero(const char* path);

/* Build model cache directory path and compiled model path */
std::string build_model_cache_dir();
std::string build_compiled_model_path(const char* ep_name);
