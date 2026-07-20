/*============================================================================
 * log.cpp - Logger static member definitions
 *============================================================================*/

#include "log.hpp"

LogLevel Logger::level = LogLevel::WARN;
const char *Logger::prog_name = "unified_bench";
