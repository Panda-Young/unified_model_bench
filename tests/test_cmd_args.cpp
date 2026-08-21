/*============================================================================
 * test_cmd_args.cpp - Unit tests for CLI argument parsing
 *============================================================================*/
#include "backend_interface.hpp"
#include "cmd_args.hpp"
#include "doctest.h"

#include <string>
#include <vector>

namespace {

/* Register a minimal set of fake backends so --backend/--no-backend name
 * resolution has something to look up. The factory is never invoked by the
 * tests (it would only run when a backend is actually created). */
void register_fake_backends()
{
    auto make = [](BackendId id, const char *name) {
        BackendConfig c;
        c.id = id;
        c.type = BackendType::ONNX_EP;
        c.name = name;
        BackendRegistry::Register(id, c, [](BackendId) -> BackendPtr { return nullptr; });
    };
    make(BackendId::ONNX_CPU, "ONNX_CPU");
    make(BackendId::ONNX_DML_GPU, "ONNX_DML_GPU");
    make(BackendId::NCNN_CPU, "NCNN_CPU");
}

/* Parse an argv list (first element = program name). */
bool parse(std::vector<const char *> args, BenchConfig &cfg)
{
    std::vector<char *> argv;
    for (auto *a : args) {
        argv.push_back(const_cast<char *>(a));
    }
    return parse_cmd_args((int)argv.size(), argv.data(), cfg);
}

} // namespace

TEST_CASE("parse_cmd_args: positional model path")
{
    register_fake_backends();
    BenchConfig cfg;
    const char *args[] = {"unified_bench", "test_model.onnx"};
    CHECK(parse_cmd_args(2, const_cast<char **>(args), cfg));
    CHECK(cfg.model_path == "test_model.onnx");
}

TEST_CASE("parse_cmd_args: --backend comma list resolves to ids")
{
    register_fake_backends();
    BenchConfig cfg;
    const char *args[] = {"unified_bench", "--model", "m.onnx",
                          "--backend", "ONNX_CPU,ONNX_DML_GPU"};
    CHECK(parse_cmd_args(5, const_cast<char **>(args), cfg));
    REQUIRE(cfg.backend_ids.size() == 2);
    CHECK(cfg.backend_ids[0] == bid(BackendId::ONNX_CPU));
    CHECK(cfg.backend_ids[1] == bid(BackendId::ONNX_DML_GPU));
}

TEST_CASE("parse_cmd_args: --no-backend blacklist")
{
    register_fake_backends();
    BenchConfig cfg;
    const char *args[] = {"unified_bench", "m.onnx", "--no-backend", "NCNN_CPU"};
    CHECK(parse_cmd_args(4, const_cast<char **>(args), cfg));
    REQUIRE(cfg.no_backend_ids.size() == 1);
    CHECK(cfg.no_backend_ids[0] == bid(BackendId::NCNN_CPU));
}

TEST_CASE("parse_cmd_args: unknown backend name warns but does not abort")
{
    register_fake_backends();
    BenchConfig cfg;
    const char *args[] = {"unified_bench", "m.onnx", "--backend", "NO_SUCH_BACKEND,ONNX_CPU"};
    CHECK(parse_cmd_args(4, const_cast<char **>(args), cfg));
    /* only the valid name survives */
    REQUIRE(cfg.backend_ids.size() == 1);
    CHECK(cfg.backend_ids[0] == bid(BackendId::ONNX_CPU));
}

TEST_CASE("parse_cmd_args: options and worker flags")
{
    register_fake_backends();
    BenchConfig cfg;
    const char *args[] = {"unified_bench", "m.onnx", "--repeat", "50",
                          "--threads", "2", "--warmup", "3",
                          "--worker", "--batch-time", "10:00:00",
                          "--baseline-ms", "12.5"};
    CHECK(parse_cmd_args(13, const_cast<char **>(args), cfg));
    CHECK(cfg.repeat == 50);
    CHECK(cfg.num_threads == 2);
    CHECK(cfg.warmup_runs == 3);
    CHECK(cfg.worker_mode);
    CHECK(cfg.batch_time == "10:00:00");
    CHECK(cfg.baseline_ms == doctest::Approx(12.5));
}

TEST_CASE("parse_cmd_args: missing model path fails")
{
    register_fake_backends();
    BenchConfig cfg;
    const char *args[] = {"unified_bench", "--repeat", "10"};
    CHECK_FALSE(parse_cmd_args(3, const_cast<char **>(args), cfg));
}
