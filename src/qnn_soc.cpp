/*============================================================================
 * qnn_soc.cpp - QNN HTP SoC / HTP-architecture detection (shared)
 *
 * Implements include/qnn_soc.hpp. Keeping the SoC table and helpers in one
 * module avoids duplicating the mapping in the ORT and native QNN paths and
 * makes per-SoC tuning defaults consistent across the tool.
 *============================================================================*/

#include "qnn_soc.hpp"
#include "log.hpp"
#include "platform.hpp"

#include <cstdlib>
#include <cstring>

#if defined(__ANDROID__) || defined(__android__)
#include <sys/system_properties.h>
#endif

namespace {

// Snapdragon -> QNN socId / HTP arch. Source: QAIRT 2.48.40 docs
// (overview.html "Supported Snapdragon devices"). Values match the QNN
// QNN_SOC_MODEL_* enum (QnnTypes.h) for socId.
const QnnSocEntry kSocMap[] = {
    {"SC8480XP", 88, 81}, {"SM8850", 87, 81}, {"SM8750", 69, 79},
    {"SM8650", 57, 75},   {"SM7750", 86, 73}, {"SM8550", 43, 73},
    {"SM7635", 73, 73},   {"SC8380XP", 60, 73}, {"SM8475", 42, 69},
    {"SM8450", 36, 69},   {"SM7450", 41, 69}, {"SM8350", 30, 68},
    {"SM7325", 35, 68},   {"SC8280X", 37, 68},
};

} // namespace

bool qnn_soc_detect(std::string &soc_model, std::string &htp_arch)
{
#if defined(__ANDROID__) || defined(__android__)
    char buf[PROP_VALUE_MAX] = {0};
    if (__system_property_get("ro.soc.model", buf) <= 0) {
        LOGW("QNN: cannot read ro.soc.model - per-SoC tuning left auto");
        return false;
    }
    for (const auto &e : kSocMap) {
        if (stricmp_(buf, e.name) == 0) {
            char soc_buf[16] = {0};
            char arch_buf[16] = {0};
            snprintf(soc_buf, sizeof(soc_buf), "%u", e.socId);
            snprintf(arch_buf, sizeof(arch_buf), "%d", e.arch);
            soc_model = soc_buf;
            htp_arch = arch_buf;
            LOGI("QNN: detected SoC %s -> soc_id=%u htp_arch=%d", buf, e.socId, e.arch);
            return true;
        }
    }
    LOGW("QNN: unknown SoC '%s' - per-SoC tuning left auto", buf);
#else
    (void)soc_model;
    (void)htp_arch;
#endif
    return false;
}

bool qnn_arch_at_least(const std::string &htp_arch, int min_arch)
{
    if (htp_arch.empty()) {
        return false;
    }
    char *end = nullptr;
    long value = strtol(htp_arch.c_str(), &end, 10);
    if (!end || end == htp_arch.c_str() || *end != '\0') {
        return false; // not a bare decimal number
    }
    return value >= min_arch;
}

uint32_t qnn_recommended_vtcm_mb(const std::string &htp_arch)
{
    // 0 == QNN_HTP_GRAPH_CONFIG_OPTION_MAX ("use the SoC's maximum VTCM").
    // Only v81+ (SM8750/SM8850) gets an explicit 8 MB, validated on SM8850
    // (16 MB fails; default is 4). Every other arch falls back to MAX so QNN
    // picks a size that always fits the device (e.g. SM8450 / v69).
    if (qnn_arch_at_least(htp_arch, 81)) {
        return 8;
    }
    return 0;
}
