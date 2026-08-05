/*============================================================================
 * litert_qualcomm_stubs.cpp -- Minimal Qualcomm options stubs for Android
 *
 * On Android, libLiteRt.so does NOT export the LrtQualcommOptions* helper
 * functions (they live in the static litert_cc_api library which is meant
 * to be compiled into the application).  On Windows, libLiteRt.dll is a
 * monolithic build that exports everything, so no stubs are needed.
 *
 * This file provides minimal implementations of the 6 Qualcomm options
 * functions used by litert_backend.cpp, compiled only for Android.
 *============================================================================*/

#ifdef HAVE_LITERT_BACKEND

#include "litert/c/litert_common.h"
#include "litert/c/options/litert_qualcomm_options.h"

/* Only compile on Android -- Windows libLiteRt.dll exports these natively. */
#if defined(__ANDROID__) || defined(__android__)

#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>

/* ---------------------------------------------------------------------------
 * Internal options struct -- mirrors LrtQualcommOptionsT from the SDK.
 * We only define the fields actually used by our code path:
 *   qnn_backend, htp_performance_mode, optimization_level
 * -------------------------------------------------------------------------*/
struct LiteRTQualcommOptionsInternal {
    int backend = 0;              // LrtQualcommOptionsBackend
    int htp_performance_mode = 0; // LrtQualcommOptionsHtpPerformanceMode
    int optimization_level = 0;   // LrtQualcommOptionsOptimizationLevel
    bool has_backend = false;
    bool has_htp_performance_mode = false;
    bool has_optimization_level = false;
};

/* ---------------------------------------------------------------------------
 * Public API
 * -------------------------------------------------------------------------*/

LiteRtStatus LrtCreateQualcommOptions(LrtQualcommOptions *options)
{
    if (!options) {
        return kLiteRtStatusErrorInvalidArgument;
    }
    *options = reinterpret_cast<LrtQualcommOptions>(
        new LiteRTQualcommOptionsInternal());
    return kLiteRtStatusOk;
}

void LrtDestroyQualcommOptions(LrtQualcommOptions options)
{
    delete reinterpret_cast<LiteRTQualcommOptionsInternal *>(options);
}

LiteRtStatus LrtQualcommOptionsSetBackend(
    LrtQualcommOptions options, LrtQualcommOptionsBackend backend)
{
    if (!options) {
        return kLiteRtStatusErrorInvalidArgument;
    }
    auto *o = reinterpret_cast<LiteRTQualcommOptionsInternal *>(options);
    o->backend = static_cast<int>(backend);
    o->has_backend = true;
    return kLiteRtStatusOk;
}

LiteRtStatus LrtQualcommOptionsSetHtpPerformanceMode(
    LrtQualcommOptions options,
    LrtQualcommOptionsHtpPerformanceMode mode)
{
    if (!options) {
        return kLiteRtStatusErrorInvalidArgument;
    }
    auto *o = reinterpret_cast<LiteRTQualcommOptionsInternal *>(options);
    o->htp_performance_mode = static_cast<int>(mode);
    o->has_htp_performance_mode = true;
    return kLiteRtStatusOk;
}

LiteRtStatus LrtQualcommOptionsSetOptimizationLevel(
    LrtQualcommOptions options,
    LrtQualcommOptionsOptimizationLevel level)
{
    if (!options) {
        return kLiteRtStatusErrorInvalidArgument;
    }
    auto *o = reinterpret_cast<LiteRTQualcommOptionsInternal *>(options);
    o->optimization_level = static_cast<int>(level);
    o->has_optimization_level = true;
    return kLiteRtStatusOk;
}

LiteRtStatus LrtGetOpaqueQualcommOptionsData(
    LrtQualcommOptions options,
    const char **identifier,
    void **payload,
    void (**payload_deleter)(void *))
{
    if (!options || !identifier || !payload || !payload_deleter) {
        return kLiteRtStatusErrorInvalidArgument;
    }

    auto *o = reinterpret_cast<LiteRTQualcommOptionsInternal *>(options);

    /* Serialize to TOML -- same format as the SDK's implementation */
    std::ostringstream toml;
    if (o->has_backend) {
        toml << "qnn_backend = " << o->backend << "\n";
    }
    if (o->has_htp_performance_mode) {
        toml << "htp_performance_mode = " << o->htp_performance_mode << "\n";
    }
    if (o->has_optimization_level) {
        toml << "optimization_level = " << o->optimization_level << "\n";
    }

    *identifier = "qualcomm";

    std::string str = toml.str();
    char *buf = new char[str.size() + 1];
    std::memcpy(buf, str.data(), str.size());
    buf[str.size()] = '\0';
    *payload = buf;
    *payload_deleter = [](void *p) { delete[] static_cast<char *>(p); };

    return kLiteRtStatusOk;
}

#endif /* defined(__ANDROID__) || defined(__android__) */

#endif /* HAVE_LITERT_BACKEND */
