/*============================================================================
 * model_loader.cpp - Multi-format model variant discovery
 *============================================================================*/

#include "model_loader.hpp"
#include "log.hpp"
#include <cstdio>
#include <cstring>
#include <sys/stat.h>

#ifdef _WIN32
#include <io.h>
#define access _access
#define F_OK 0
#else
#include <unistd.h>
#endif

/* ---------------------------------------------------------------------------
 * Helpers
 * -------------------------------------------------------------------------*/
static bool file_exists(const char *path)
{
    return access(path, F_OK) == 0;
}

static std::string get_directory(const std::string &path)
{
    auto pos = path.find_last_of("/\\");
    return (pos != std::string::npos) ? path.substr(0, pos + 1) : std::string("./");
}

static bool ends_with_icase(const std::string &s, const std::string &suffix)
{
    if (s.length() < suffix.length()) {
        return false;
    }
#ifdef _WIN32
    return stricmp_(s.c_str() + s.length() - suffix.length(), suffix.c_str()) == 0;
#else
    return strcasecmp(s.c_str() + s.length() - suffix.length(), suffix.c_str()) == 0;
#endif
}

/* ---------------------------------------------------------------------------
 * Extract base name
 * -------------------------------------------------------------------------*/
std::string extract_base_name(const std::string &path)
{
    /* Get filename */
    auto pos = path.find_last_of("/\\");
    std::string fname = (pos != std::string::npos) ? path.substr(pos + 1) : path;

    /* Handle compound extensions like .ncnn.param */
    if (ends_with_icase(fname, ".ncnn.param")) {
        fname = fname.substr(0, fname.length() - 12);
    } else if (ends_with_icase(fname, ".ncnn.bin")) {
        fname = fname.substr(0, fname.length() - 9);
    }

    /* Strip last extension */
    auto dot = fname.find_last_of('.');
    if (dot != std::string::npos) {
        fname = fname.substr(0, dot);
    }

    return fname;
}

/* ---------------------------------------------------------------------------
 * Build model path
 * -------------------------------------------------------------------------*/
std::string build_model_path(const std::string &dir, const std::string &base,
                             const std::string &ext)
{
    std::string p = dir;
    if (!p.empty() && p.back() != '/' && p.back() != '\\') {
        p += PATH_SEP;
    }
    p += base;
    if (!ext.empty() && ext[0] != '.') {
        p += '.';
    }
    p += ext;
    return p;
}

/* ---------------------------------------------------------------------------
 * Search one variant
 * -------------------------------------------------------------------------*/
static ModelSearchResult search_variant(const std::string &dir,
                                        const std::string &base,
                                        const std::string &ext,
                                        const std::string &bin_ext,
                                        const std::string &variant_name,
                                        ModelFormat fmt)
{
    ModelSearchResult r;
    r.variant_name = variant_name;

    std::string primary = build_model_path(dir, base, ext);
    if (!file_exists(primary.c_str())) {
        return r;
    }

    /* For NCNN: need both .param and .bin */
    if (fmt == ModelFormat::NCNN) {
        std::string bin = build_model_path(dir, base, bin_ext);
        if (!file_exists(bin.c_str())) {
            return r;
        }
    }

    r.found = true;
    r.path = primary;
    r.format = fmt;
    return r;
}

/* ---------------------------------------------------------------------------
 * Search all QNN model sources.
 * 1) model.so forms (lib{base}.so / {base}.so) -- runtime-composed, the SAME
 *    library can run on CPU/GPU/HTP.
 * 2) context binaries ({base}.serialized.bin / {base}.bin / {base}.dlc) --
 *    offline-compiled and backend-specific (typically HTP).
 * model.so is returned first (priority).
 * -------------------------------------------------------------------------*/
static std::vector<ModelSearchResult> search_qnn_variants(const std::string &dir,
                                                          const std::string &base)
{
    std::vector<ModelSearchResult> out;
    auto add = [&](const std::string &p, const char *name) {
        ModelSearchResult r;
        r.found = true;
        r.path = p;
        r.format = ModelFormat::QNN;
        r.variant_name = name;
        out.push_back(r);
    };

    /* 1. model.so (priority) */
    std::string lib_so = dir + "lib" + base + ".so";
    if (file_exists(lib_so.c_str())) {
        add(lib_so, "QNN model.so");
    }
    std::string so = build_model_path(dir, base, "so");
    if (file_exists(so.c_str())) {
        add(so, "QNN model.so");
    }

    /* 2. context binaries */
    const char *exts[] = {"serialized.bin", "bin", "dlc", nullptr};
    for (int i = 0; exts[i]; ++i) {
        std::string p = build_model_path(dir, base, exts[i]);
        if (file_exists(p.c_str())) {
            add(p, "QNN context binary");
        }
    }
    return out;
}

/* ---------------------------------------------------------------------------
 * Search all variants
 * -------------------------------------------------------------------------*/
ModelBundle search_model_variants(const std::string &ref_path)
{
    ModelBundle bundle;
    bundle.base_name = extract_base_name(ref_path);
    bundle.directory = get_directory(ref_path);

    auto &dir = bundle.directory;
    auto &base = bundle.base_name;

    bundle.onnx_model = search_variant(dir, base, "onnx", "", "Original", ModelFormat::ONNX);
    bundle.tflite_model = search_variant(dir, base, "tflite", "", "TFLite conv", ModelFormat::TFLITE);
    bundle.ncnn_model = search_variant(dir, base, "ncnn.param",
                                       "ncnn.bin", "NCNN converted", ModelFormat::NCNN);
    bundle.mnn_model = search_variant(dir, base, "mnn", "", "MNN converted", ModelFormat::MNN);

    /* If the reference path itself is a QNN model, use it directly -- it is
     * not auto-derived from the other formats. */
    if (detect_model_format(ref_path) == ModelFormat::QNN) {
        ModelSearchResult r;
        r.found = true;
        r.path = ref_path;
        r.format = ModelFormat::QNN;
        r.variant_name = (ref_path.size() > 3 &&
                          stricmp_(ref_path.c_str() + ref_path.size() - 3, ".so") == 0)
                             ? "QNN model.so"
                             : "QNN context binary";
        bundle.qnn_models.push_back(r);
    } else {
        bundle.qnn_models = search_qnn_variants(dir, base);
    }

    bundle.total_variants = 0;
    if (bundle.onnx_model.found) {
        ++bundle.total_variants;
    }
    if (bundle.tflite_model.found) {
        ++bundle.total_variants;
    }
    if (bundle.ncnn_model.found) {
        ++bundle.total_variants;
    }
    if (bundle.mnn_model.found) {
        ++bundle.total_variants;
    }
    bundle.total_variants += (int)bundle.qnn_models.size();

    LOGI("Model discovery: base='%s', found %d variant(s)", base.c_str(), bundle.total_variants);
    if (bundle.onnx_model.found) {
        LOGI("  ONNX:   %s", bundle.onnx_model.path.c_str());
    }
    if (bundle.tflite_model.found) {
        LOGI("  TFLite: %s", bundle.tflite_model.path.c_str());
    }
    if (bundle.ncnn_model.found) {
        LOGI("  NCNN:   %s", bundle.ncnn_model.path.c_str());
    }
    if (bundle.mnn_model.found) {
        LOGI("  MNN:    %s", bundle.mnn_model.path.c_str());
    }
    for (size_t i = 0; i < bundle.qnn_models.size(); ++i) {
        LOGI("  QNN[%zu]: %s (%s)", i, bundle.qnn_models[i].path.c_str(),
             bundle.qnn_models[i].variant_name.c_str());
    }

    return bundle;
}

std::vector<const ModelSearchResult *> ModelBundle::all_found() const
{
    std::vector<const ModelSearchResult *> v;
    if (onnx_model.found) {
        v.push_back(&onnx_model);
    }
    if (tflite_model.found) {
        v.push_back(&tflite_model);
    }
    if (ncnn_model.found) {
        v.push_back(&ncnn_model);
    }
    if (mnn_model.found) {
        v.push_back(&mnn_model);
    }
    for (auto &q : qnn_models) {
        v.push_back(&q);
    }
    return v;
}
