/*============================================================================
 * test_input_provider.cpp - Unit tests for deterministic shared inputs
 *============================================================================*/
#include "input_provider.hpp"
#include "doctest.h"

#include <cstdio>
#include <vector>

TEST_CASE("GenerateFromSizes: deterministic and size-correct")
{
    InputProvider a, b;
    std::vector<size_t> sizes = {150528, 48, 10};
    a.GenerateFromSizes(sizes);
    b.GenerateFromSizes(sizes);

    REQUIRE(a.Count() == 3);
    CHECK(a.ElementCounts() == sizes);

    /* same seed (42) -> identical data across instances */
    for (size_t i = 0; i < a.Count(); ++i) {
        REQUIRE(a[i].data.size() == b[i].data.size());
        for (size_t j = 0; j < a[i].data.size(); ++j) {
            CHECK(a[i].data[j] == b[i].data[j]);
        }
    }
}

TEST_CASE("LoadFromFiles: float32 roundtrip and mismatch detection")
{
    /* write two float32 bin files with known data */
    const float src1[] = {0.1f, 0.2f, 0.3f, 0.4f};
    const float src2[] = {1.0f, -1.0f};
    FILE *f = fopen("ut_in1.bin", "wb");
    REQUIRE(f != nullptr);
    fwrite(src1, sizeof(float), 4, f);
    fclose(f);
    f = fopen("ut_in2.bin", "wb");
    REQUIRE(f != nullptr);
    fwrite(src2, sizeof(float), 2, f);
    fclose(f);

    InputProvider p;
    std::vector<std::string> paths = {"ut_in1.bin", "ut_in2.bin"};
    std::vector<size_t> counts = {4, 2};
    CHECK(p.LoadFromFiles(paths, counts, InputDataFormat::Float32));
    REQUIRE(p.Count() == 2);
    for (size_t i = 0; i < 4; ++i) {
        CHECK(p[0].data[i] == src1[i]);
    }
    CHECK(p[1].data[1] == src2[1]);

    /* element-count mismatch -> load fails cleanly */
    std::vector<size_t> wrong = {4, 3};
    CHECK_FALSE(p.LoadFromFiles(paths, wrong, InputDataFormat::Float32));

    /* missing file -> fails cleanly */
    std::vector<std::string> missing = {"ut_no_such.bin"};
    std::vector<size_t> one = {4};
    CHECK_FALSE(p.LoadFromFiles(missing, one, InputDataFormat::Float32));

    std::remove("ut_in1.bin");
    std::remove("ut_in2.bin");
}
