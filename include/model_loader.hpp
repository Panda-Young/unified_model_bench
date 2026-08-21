#pragma once
/*============================================================================
 * model_loader.hpp - Multi-format model variant discovery
 *============================================================================*/

#include "model_format.hpp"
#include <string>
#include <vector>

struct ModelSearchResult {
    bool found = false;
    std::string path;
    ModelFormat format = ModelFormat::UNKNOWN;
    std::string variant_name; /* e.g. "Original", "NCNN converted" */
};

struct ModelBundle {
    std::string base_name;
    std::string directory;

    ModelSearchResult onnx_model;
    ModelSearchResult tflite_model;
    ModelSearchResult ncnn_model;
    ModelSearchResult mnn_model;
    /* QNN can have multiple model sources: a model.so (lib{base}.so) runs on
     * CPU/GPU/HTP, plus context binaries ({base}.serialized.bin/.bin/.dlc)
     * which are backend-specific. Priority: model.so first. */
    std::vector<ModelSearchResult> qnn_models;

    int total_variants = 0;

    /* Return all found variants in priority order */
    std::vector<const ModelSearchResult *> all_found() const;
};

/* Given one model path, search same directory for all format variants */
ModelBundle search_model_variants(const std::string &ref_path);

/* Extract base name (strip dir + extension) */
std::string extract_base_name(const std::string &path);

/* Build path: dir/base.ext */
std::string build_model_path(const std::string &dir,
                             const std::string &base,
                             const std::string &ext);

/* Estimate the model's WEIGHT memory footprint in MB (deployment info).
 *  - ONNX: parses the graph initializers (minimal protobuf wire reader,
 *    no external deps) for an exact byte count of all weight tensors.
 *  - NCNN: the .ncnn.bin file IS the weights payload (param is text), so its
 *    file size is exact.
 *  - TFLite: minimal FlatBuffers reader sums every Buffer.data payload
 *    (exact weight bytes; graph/metadata overhead excluded).
 *  - MNN / QNN: approximated by file size (no parser for the .mnn schema /
 *    the closed QNN context-binary format; weights dominate, but
 *    graph/metadata overhead is included).
 * Returns 0.0 when the model or its weights cannot be read. */
double estimate_weight_mb(const ModelSearchResult &model);
