/*============================================================================
 * model_loader.cpp - Multi-format model variant discovery
 *============================================================================*/

#include "model_loader.hpp"
#include "log.hpp"
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <sys/stat.h>
#include <vector>

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

    /* Handle compound extensions like .ncnn.param (11 chars) / .ncnn.bin (9) */
    if (ends_with_icase(fname, ".ncnn.param")) {
        fname = fname.substr(0, fname.length() - 11);
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

/* ---------------------------------------------------------------------------
 * Weight memory estimation
 * -------------------------------------------------------------------------*/

/* Minimal protobuf wire-format reader (no external deps). Only supports the
 * field kinds used below: varint(0), 64-bit(1), length-delimited(2), 32-bit(5).
 * -------------------------------------------------------------------------*/
namespace {

struct PbReader {
    const uint8_t *p;
    size_t n;
    size_t pos = 0;
    bool ok = true;

    PbReader(const void *data, size_t size) : p((const uint8_t *)data), n(size) {}

    bool eof() const { return pos >= n; }

    uint64_t varint()
    {
        uint64_t v = 0;
        int shift = 0;
        while (pos < n) {
            uint8_t b = p[pos++];
            v |= (uint64_t)(b & 0x7F) << shift;
            if (!(b & 0x80)) {
                return v;
            }
            shift += 7;
            if (shift >= 64) {
                ok = false;
                return 0;
            }
        }
        ok = false;
        return 0;
    }

    /* Skip one field whose tag was already read. Returns false on bad wire. */
    bool skip(unsigned wire)
    {
        switch (wire) {
        case 0: {
            varint();
            break;
        }
        case 1: {
            return take(8) != nullptr;
        }
        case 2: {
            size_t l = (size_t)varint();
            return take(l) != nullptr;
        }
        case 5: {
            return take(4) != nullptr;
        }
        default: {
            ok = false;
            return false;
        }
        }
        return ok;
    }

    const uint8_t *take(size_t len)
    {
        if (pos + len > n || !ok) {
            ok = false;
            return nullptr;
        }
        const uint8_t *r = p + pos;
        pos += len;
        return r;
    }
};

/* ONNX ModelProto: graph = field 7 (length-delimited GraphProto).
 * GraphProto:      initializer = field 5 (repeated TensorProto).
 * TensorProto payload fields: float_data=4, int32_data=5, string_data=6,
 * int64_data=7, raw_data=9 (exact bytes), double_data=10, uint64_data=11.
 * raw_data, when present, is the exact serialized weight payload. */
size_t onnx_initializer_bytes(const uint8_t *data, size_t size)
{
    PbReader r(data, size);
    while (!r.eof() && r.ok) {
        uint64_t t = r.varint();
        unsigned fld = (unsigned)(t >> 3);
        unsigned wire = (unsigned)(t & 7);
        if (fld == 7 && wire == 2) {
            size_t glen = (size_t)r.varint();
            const uint8_t *graph = r.take(glen);
            if (!graph) {
                return 0;
            }
            PbReader g(graph, glen);
            size_t total = 0;
            while (!g.eof() && g.ok) {
                uint64_t gt = g.varint();
                unsigned gf = (unsigned)(gt >> 3);
                unsigned gw = (unsigned)(gt & 7);
                if (gf == 5 && gw == 2) {
                    size_t tlen = (size_t)g.varint();
                    const uint8_t *tp = g.take(tlen);
                    if (!tp) {
                        return 0;
                    }
                    /* Sum all payload fields of one TensorProto */
                    PbReader tr(tp, tlen);
                    size_t tensor_bytes = 0;
                    bool have_raw = false;
                    size_t raw_len = 0;
                    while (!tr.eof() && tr.ok) {
                        uint64_t tt = tr.varint();
                        unsigned tf = (unsigned)(tt >> 3);
                        unsigned tw = (unsigned)(tt & 7);
                        if (tf == 9 && tw == 2) { /* raw_data: exact bytes */
                            size_t l = (size_t)tr.varint();
                            if (!tr.take(l)) {
                                break;
                            }
                            have_raw = true;
                            raw_len = l;
                        } else if ((tf == 4 || tf == 5 || tf == 6 ||
                                    tf == 7 || tf == 10 || tf == 11) &&
                                   tw == 2) {
                            /* float_data/int32_data/string_data/int64_data/
                             * double_data/uint64_data: payload length is the
                             * weight byte count (strings: content bytes) */
                            size_t l = (size_t)tr.varint();
                            if (!tr.take(l)) {
                                break;
                            }
                            tensor_bytes += l;
                        } else if (!tr.skip(tw)) {
                            break;
                        }
                    }
                    total += have_raw ? raw_len : tensor_bytes;
                } else if (!g.skip(gw)) {
                    return 0;
                }
            }
            return total;
        }
        if (!r.skip(wire)) {
            return 0;
        }
    }
    return 0;
}

} // namespace

double estimate_weight_mb(const ModelSearchResult &model)
{
    if (!model.found || model.path.empty()) {
        return 0.0;
    }

    FILE *f = fopen(model.path.c_str(), "rb");
    if (!f) {
        LOGW("estimate_weight_mb: cannot open %s", model.path.c_str());
        return 0.0;
    }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz <= 0) {
        fclose(f);
        return 0.0;
    }

    double mb = 0.0;
    if (model.format == ModelFormat::ONNX) {
        /* Exact: sum of all graph initializer payloads */
        std::vector<uint8_t> buf((size_t)sz);
        bool parsed = fread(buf.data(), 1, (size_t)sz, f) == (size_t)sz;
        fclose(f);
        if (parsed) {
            size_t bytes = onnx_initializer_bytes(buf.data(), buf.size());
            if (bytes > 0) {
                mb = (double)bytes / (1024.0 * 1024.0);
                LOGI("Weight memory (ONNX, parsed): %.2f MB (%zu bytes)",
                     mb, bytes);
                return mb;
            }
            LOGW("estimate_weight_mb: ONNX initializer parse failed, "
                 "falling back to file size");
        }
        /* Parser failed / unreadable - approximate with file size */
        mb = (double)sz / (1024.0 * 1024.0);
        return mb;
    }

    /* NCNN: the weights payload IS the .ncnn.bin file */
    if (model.format == ModelFormat::NCNN) {
        std::string bin = build_model_path(get_directory(model.path),
                                           extract_base_name(model.path),
                                           "ncnn.bin");
        fclose(f);
        FILE *bf = fopen(bin.c_str(), "rb");
        if (!bf) {
            LOGW("estimate_weight_mb: NCNN weights file not found: %s",
                 bin.c_str());
            return 0.0;
        }
        fseek(bf, 0, SEEK_END);
        long bsz = ftell(bf);
        fclose(bf);
        mb = bsz > 0 ? (double)bsz / (1024.0 * 1024.0) : 0.0;
        LOGI("Weight memory (NCNN, .bin size): %.2f MB (%ld bytes)", mb, bsz);
        return mb;
    }

    /* TFLite / MNN / QNN: no parser yet - file size approximation
     * (weights dominate, but graph/metadata overhead is included) */
    fclose(f);
    mb = (double)sz / (1024.0 * 1024.0);
    LOGI("Weight memory (%s, approx. file size): %.2f MB",
         model_format_name(model.format), mb);
    return mb;
}
