#pragma once
/*============================================================================
 * model_format.hpp - Model format detection & naming
 *============================================================================*/

#include "platform.hpp"
#include <cstring>
#include <string>

enum class ModelFormat {
    UNKNOWN = 0,
    ONNX = 1,
    TFLITE = 2,
    NCNN = 3,
    MNN = 4
};

inline const char *model_format_name(ModelFormat fmt)
{
    switch (fmt) {
    case ModelFormat::ONNX:
        return "ONNX";
    case ModelFormat::TFLITE:
        return "TFLite";
    case ModelFormat::NCNN:
        return "NCNN";
    case ModelFormat::MNN:
        return "MNN";
    default:
        return "UNKNOWN";
    }
}

struct ModelInfo {
    ModelFormat format = ModelFormat::UNKNOWN;
    std::string path;
    std::string filename;

    bool valid() const { return format != ModelFormat::UNKNOWN && !path.empty(); }
};

/* Detect format by extension, then magic number */
inline ModelFormat detect_model_format(const std::string &path)
{
    /* Extension check */
    auto ends_with = [](const std::string &s, const std::string &suffix) -> bool {
        if (s.length() < suffix.length()) {
            return false;
        }
        return stricmp_(s.c_str() + s.length() - suffix.length(), suffix.c_str()) == 0;
    };

    if (ends_with(path, ".onnx")) {
        return ModelFormat::ONNX;
    }
    if (ends_with(path, ".tflite")) {
        return ModelFormat::TFLITE;
    }
    if (ends_with(path, ".param")) {
        return ModelFormat::NCNN;
    }
    if (ends_with(path, ".mnn")) {
        return ModelFormat::MNN;
    }

    /* Magic-number fallback for ONNX */
    FILE *f = fopen(path.c_str(), "rb");
    if (f) {
        char buf[12] = {};
        size_t n = fread(buf, 1, 12, f);
        fclose(f);
        if (n >= 12 && memcmp(buf + 8, "ONNX", 4) == 0) {
            return ModelFormat::ONNX;
        }
    }
    return ModelFormat::UNKNOWN;
}

inline ModelInfo extract_model_info(const std::string &path)
{
    ModelInfo info;
    info.format = detect_model_format(path);
    info.path = path;

    /* Extract filename */
    auto pos = path.find_last_of("/\\");
    info.filename = (pos != std::string::npos) ? path.substr(pos + 1) : path;
    return info;
}
