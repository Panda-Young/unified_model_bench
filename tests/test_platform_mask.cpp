/*============================================================================
 * test_platform_mask.cpp - Unit tests for the platform availability mask
 *
 * BackendConfig::platforms is data, so it can be tested without a registry or
 * any backend. Guards:
 *  - current_platform() resolves to exactly one bit (never 0 / never multiple)
 *  - platform_supports() is a pure bit test
 *  - platform_mask_str() renders every combination deterministically
 *  - the mask of THIS build's platform is actually supported (a mis-declared
 *    backend would silently never register)
 *============================================================================*/
#include "backend_interface.hpp"
#include "doctest.h"

#include <string>

TEST_CASE("current_platform: exactly one platform bit on supported builds")
{
    const unsigned p = current_platform();
    CHECK(p != kPlatNone);
    /* Exactly one bit set: p & (p - 1) clears the lowest set bit, so it is 0
     * only when p is a power of two. */
    CHECK((p & (p - 1)) == 0u);
    const bool known = (p == kPlatWin) || (p == kPlatLinux) || (p == kPlatAndroid);
    CHECK(known);
}

TEST_CASE("platform_supports: pure bit test")
{
    CHECK(platform_supports(kPlatWin | kPlatLinux | kPlatAndroid));

    /* A mask for a foreign platform must NOT be supported here. Pick one
     * that is definitely not the current platform. */
    const unsigned cur = current_platform();
    unsigned foreign = kPlatWin | kPlatLinux | kPlatAndroid;
    foreign &= ~cur;
    CHECK(foreign != 0u);
    CHECK_FALSE(platform_supports(foreign));

    CHECK_FALSE(platform_supports(kPlatNone));

    /* Adding the current platform bit to a foreign mask makes it supported. */
    CHECK(platform_supports(foreign | cur));
}

TEST_CASE("platform_mask_str: deterministic, comma-separated, no trailing comma")
{
    CHECK(platform_mask_str(kPlatNone) == "-");
    CHECK(platform_mask_str(kPlatWin) == "Win");
    CHECK(platform_mask_str(kPlatLinux) == "Linux");
    CHECK(platform_mask_str(kPlatAndroid) == "Android");
    CHECK(platform_mask_str(kPlatWin | kPlatLinux) == "Win,Linux");
    CHECK(platform_mask_str(kPlatWin | kPlatLinux | kPlatAndroid) == "Win,Linux,Android");

    /* No trailing comma: the last character is never ','. */
    const std::string all = platform_mask_str(kPlatWin | kPlatLinux | kPlatAndroid);
    REQUIRE_FALSE(all.empty());
    CHECK(all.back() != ',');
}

TEST_CASE("BackendConfig: platforms defaults to none (never registered)")
{
    BackendConfig c;
    c.id = BackendId::ONNX_CPU;
    c.name = "ONNX_CPU";
    CHECK(c.platforms == kPlatNone);
    CHECK_FALSE(platform_supports(c.platforms));
}

TEST_CASE("this platform is supported by its own mask")
{
    /* Sanity: a backend declared for all platforms must register on any build.
     * Catches a mismatch between current_platform() and the mask constants. */
    const BackendConfig c{BackendId::ONNX_CPU, BackendType::ONNX_EP, "ONNX_CPU", "",
                          true, kPlatWin | kPlatLinux | kPlatAndroid};
    CHECK(platform_supports(c.platforms));
}
