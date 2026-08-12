#pragma once
/*============================================================================
 * qnn_soc.hpp - QNN HTP SoC / HTP-architecture detection (shared)
 *
 * Used by both the ORT QNN EP path (src/onnx_backend.cpp) and the native QNN
 * SDK path (src/qnn_backend.cpp) so per-SoC defaults (VTCM size, tuning set)
 * come from the detected HTP architecture instead of being hard-coded for a
 * single device (e.g. SM8850). All helpers are platform-neutral; on non-Android
 * qnn_soc_detect() reports "unknown" and callers fall back to safe defaults.
 *============================================================================*/

#include <cstdint>
#include <string>

// One entry of the Snapdragon -> QNN socId / HTP arch map (source: QAIRT 2.48.40
// docs, overview.html "Supported Snapdragon devices"). arch is the HTP
// architecture number (81 = hexagon v81, 79 = v79, 75 = v75, 73 = v73, 69 = v69,
// 68 = v68, ...).
struct QnnSocEntry {
    const char *name;   // ro.soc.model string, e.g. "SM8850"
    uint32_t socId;     // QNN soc id (QNN_SOC_MODEL_*), e.g. 87
    int arch;           // HTP architecture number, e.g. 81
};

// Runtime detection. On Android reads ro.soc.model and maps it to the QNN
// "soc_model"/"htp_arch" option strings. htp_arch is the BARE number ("81"),
// because ORT's ParseHtpArchitecture() accepts only "68"/"69"/"73"/"75"/"81"
// (not "v81"). Returns true when recognized; on failure/unknown both outputs
// are left unchanged (caller should default them to "0" for auto).
bool qnn_soc_detect(std::string &soc_model, std::string &htp_arch);

// True when the (bare) arch string is a number >= min_arch. Unknown/empty or
// non-numeric arch returns false (conservative: caller picks safe defaults).
bool qnn_arch_at_least(const std::string &htp_arch, int min_arch);

// Recommended VTCM size in MB for the detected arch. Returns 0 meaning "use
// the target SoC's maximum VTCM" (QNN_HTP_GRAPH_CONFIG_OPTION_MAX) for every
// arch except v81+ (SM8750/SM8850) where 8 MB is empirically validated. 0 is
// the portable default: QNN always resolves it to a valid per-SoC size, so it
// never fails on older chips with a smaller VTCM (e.g. SM8450 / v69).
uint32_t qnn_recommended_vtcm_mb(const std::string &htp_arch);
