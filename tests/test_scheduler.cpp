/*============================================================================
 * test_scheduler.cpp - Unit tests for scheduler helpers
 *  (backend filtering, cross-process baseline files)
 *============================================================================*/
#include "scheduler.hpp"
#include "doctest.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace {

/* Convenience factory for a BackendConfig (factory fn not needed for tests). */
BackendConfig make_cfg(BackendId id, const char *name, bool is_cpu = false)
{
    BackendConfig c;
    c.id = id;
    c.type = BackendType::ONNX_EP;
    c.name = name;
    c.is_cpu_baseline = is_cpu;
    return c;
}

} // namespace

/* ---------------------------------------------------------------------------
 * Backend list filtering
 * -------------------------------------------------------------------------*/
TEST_CASE("apply_backend_blacklist: removes excluded, keeps the rest")
{
    std::vector<BackendConfig> bs = {
        make_cfg(BackendId::ONNX_CPU, "ONNX_CPU"),
        make_cfg(BackendId::ONNX_DML_GPU, "ONNX_DML_GPU"),
        make_cfg(BackendId::ONNX_CUDA, "ONNX_CUDA"),
    };
    BenchConfig cfg;
    cfg.no_backend_ids = {bid(BackendId::ONNX_DML_GPU)};

    apply_backend_blacklist(bs, cfg);
    REQUIRE(bs.size() == 2);
    CHECK(bs[0].id == BackendId::ONNX_CPU);
    CHECK(bs[1].id == BackendId::ONNX_CUDA);
}

TEST_CASE("apply_backend_blacklist: empty blacklist is a no-op")
{
    std::vector<BackendConfig> bs = {make_cfg(BackendId::ONNX_CPU, "ONNX_CPU")};
    BenchConfig cfg; /* no_backend_ids empty */
    apply_backend_blacklist(bs, cfg);
    REQUIRE(bs.size() == 1);
}

TEST_CASE("filter_backends_by_user: whitelist hit")
{
    std::vector<BackendConfig> bs = {
        make_cfg(BackendId::ONNX_CPU, "ONNX_CPU"),
        make_cfg(BackendId::ONNX_DML_GPU, "ONNX_DML_GPU"),
    };
    BenchConfig cfg;
    cfg.backend_ids = {bid(BackendId::ONNX_DML_GPU)};

    filter_backends_by_user(bs, cfg);
    REQUIRE(bs.size() == 1);
    CHECK(bs[0].id == BackendId::ONNX_DML_GPU);
}

TEST_CASE("filter_backends_by_user: whitelist miss leaves list EMPTY (strict)")
{
    /* Regression: a whitelist name that is valid in the registry but not
     * available for this model format must not silently re-enable everything. */
    std::vector<BackendConfig> bs = {
        make_cfg(BackendId::ONNX_CPU, "ONNX_CPU"),
        make_cfg(BackendId::ONNX_DML_GPU, "ONNX_DML_GPU"),
    };
    BenchConfig cfg;
    cfg.backend_ids = {bid(BackendId::MNN_CPU)}; /* no match in this list */

    filter_backends_by_user(bs, cfg);
    CHECK(bs.empty());
}

TEST_CASE("filter_backends_by_user: whitelist then blacklist combined")
{
    std::vector<BackendConfig> bs = {
        make_cfg(BackendId::ONNX_CPU, "ONNX_CPU"),
        make_cfg(BackendId::ONNX_ONEDNN, "ONNX_oneDNN"),
        make_cfg(BackendId::ONNX_DML_GPU, "ONNX_DML_GPU"),
    };
    BenchConfig cfg;
    cfg.backend_ids = {bid(BackendId::ONNX_CPU), bid(BackendId::ONNX_DML_GPU)};
    cfg.no_backend_ids = {bid(BackendId::ONNX_DML_GPU)};

    filter_backends_by_user(bs, cfg);
    REQUIRE(bs.size() == 1);
    CHECK(bs[0].id == BackendId::ONNX_CPU);
}

TEST_CASE("filter_qnn_context_backends: model.so keeps all QNN backends")
{
    std::vector<BackendConfig> bs = {
        make_cfg(BackendId::QNN_SDK_CPU, "QNN_SDK_CPU"),
        make_cfg(BackendId::QNN_SDK_GPU, "QNN_SDK_GPU"),
        make_cfg(BackendId::QNN_SDK_HTP, "QNN_SDK_HTP"),
    };
    filter_qnn_context_backends(bs, "/data/local/tmp/libtest_model.so");
    REQUIRE(bs.size() == 3);
}

TEST_CASE("filter_qnn_context_backends: context binary keeps HTP only")
{
    std::vector<BackendConfig> bs = {
        make_cfg(BackendId::QNN_SDK_CPU, "QNN_SDK_CPU"),
        make_cfg(BackendId::QNN_SDK_GPU, "QNN_SDK_GPU"),
        make_cfg(BackendId::QNN_SDK_HTP, "QNN_SDK_HTP"),
    };
    filter_qnn_context_backends(bs, "test_model.serialized.bin");
    REQUIRE(bs.size() == 1);
    CHECK(bs[0].id == BackendId::QNN_SDK_HTP);
}

/* ---------------------------------------------------------------------------
 * Cross-process baseline output files
 * -------------------------------------------------------------------------*/
TEST_CASE("baseline_output_path: dir handling and base extraction")
{
    /* default dir (empty) -> "." prefix */
    CHECK(baseline_output_path("C:/models/test_model.onnx", "ONNX_CPU", "") ==
          std::string(".") + PATH_SEP + "test_model_ONNX_CPU.out");
    /* explicit dir */
    CHECK(baseline_output_path("C:/models/test_model.onnx", "ONNX_CPU", "out") ==
          std::string("out") + PATH_SEP + "test_model_ONNX_CPU.out");
    /* NCNN compound extension is stripped to the base name */
    CHECK(baseline_output_path("test_model.ncnn.param", "NCNN_CPU", "out") ==
          std::string("out") + PATH_SEP + "test_model_NCNN_CPU.out");
}

TEST_CASE("write/load_output_file: float roundtrip")
{
    const std::string path = "ut_baseline_tmp.out";
    const float src[] = {1.0f, -2.5f, 3.25f, 0.0f, 1e-8f};
    CHECK(write_output_file(path, src, 5, 0.0));

    std::vector<float> out;
    CHECK(load_output_file(path, out));
    REQUIRE(out.size() == 5);
    for (size_t i = 0; i < 5; ++i) {
        CHECK(out[i] == src[i]);
    }
    std::remove(path.c_str());
}

TEST_CASE("write/load_output_file: empty payload and missing file")
{
    CHECK(write_output_file("ut_baseline_tmp.out", nullptr, 0, 0.0));
    std::vector<float> out;
    CHECK(load_output_file("ut_baseline_tmp.out", out));
    CHECK(out.empty());
    std::remove("ut_baseline_tmp.out");

    CHECK_FALSE(load_output_file("ut_no_such_file.out", out));
}

/* ---------------------------------------------------------------------------
 * build_child_cmdline option classification
 * -------------------------------------------------------------------------*/
TEST_CASE("takes_value_arg: every option with a separate value is recognised")
{
    /* Regression: the scheduler previously only skipped the value of
     * --backend/--no-backend/--model. The value of any OTHER value-taking
     * option (--repeat 2, --threads 8, ...) was treated as a positional
     * argument and replaced the model path, so the worker ran with the model
     * path "2" and failed with "no model variants found". */
    const char *const value_opts[] = {
        "--model", "--backend", "--no-backend", "--input-list",
        "--input-format", "--repeat", "--warmup", "--threads",
        "--csv", "--log-level", "--output-dir"};
    for (const char *o : value_opts) {
        CHECK_MESSAGE(takes_value_arg(o), "should take a value: " << o);
    }
}

TEST_CASE("takes_value_arg: flags and --opt=value forms are not value options")
{
    /* Boolean flags take no separate value. */
    CHECK_FALSE(takes_value_arg("--no-csv"));
    CHECK_FALSE(takes_value_arg("--worker"));
    CHECK_FALSE(takes_value_arg("--help"));
    CHECK_FALSE(takes_value_arg("--version"));

    /* "--opt=value" is a single token: there is no separate value to skip, and
     * treating it as a value option would wrongly consume the NEXT argument. */
    CHECK_FALSE(takes_value_arg("--repeat=2"));
    CHECK_FALSE(takes_value_arg("--threads=8"));

    /* A positional argument is not an option at all. */
    CHECK_FALSE(takes_value_arg("test_model.onnx"));
    CHECK_FALSE(takes_value_arg("2"));
    CHECK_FALSE(takes_value_arg(""));
}
