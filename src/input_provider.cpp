/*============================================================================
 * input_provider.cpp - Deterministic shared input generation
 *============================================================================*/

#include "input_provider.hpp"
#include "log.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>

void InputProvider::GenerateFromSizes(const std::vector<size_t> &element_counts)
{
    Clear();
    srand(42); /* deterministic seed */
    for (size_t n : element_counts) {
        InputSlot slot;
        slot.element_count = n;
        slot.data.resize(n);
        for (size_t j = 0; j < n; ++j) {
            slot.data[j] = (float)rand() / (float)RAND_MAX;
        }
        slots_.push_back(std::move(slot));
    }
    LOGI("Generated %zu shared input(s), total %zu elements (seed=42)",
         slots_.size(),
         [this]() { size_t t = 0; for (auto& s : slots_) t += s.element_count; return t; }());
}

bool InputProvider::LoadFromFiles(const std::vector<std::string> &paths,
                                  const std::vector<size_t> &element_counts,
                                  InputDataFormat fmt)
{
    if (paths.size() != element_counts.size()) {
        LOGE("Input list has %zu entries but model has %zu inputs",
             paths.size(), element_counts.size());
        return false;
    }
    Clear();
    for (size_t i = 0; i < paths.size(); ++i) {
        /* Read whole file */
        FILE *f = fopen(paths[i].c_str(), "rb");
        if (!f) {
            LOGE("Input file not found: %s", paths[i].c_str());
            Clear();
            return false;
        }
        fseek(f, 0, SEEK_END);
        long file_size = ftell(f);
        fseek(f, 0, SEEK_SET);

        size_t n = element_counts[i];
        InputSlot slot;
        slot.element_count = n;

        /* Detect format from file size when Auto */
        InputDataFormat use_fmt = fmt;
        if (use_fmt == InputDataFormat::Auto) {
            if (file_size == (long)(n * sizeof(float))) {
                use_fmt = InputDataFormat::Float32;
            } else if (file_size == (long)n) {
                use_fmt = InputDataFormat::UInt8;
            } else {
                LOGE("Input[%zu] %s: file size %ld does not match %zu "
                     "elements (float32=%zu bytes, uint8=%zu bytes)",
                     i, paths[i].c_str(), file_size, n,
                     n * sizeof(float), n);
                fclose(f);
                Clear();
                return false;
            }
        }

        slot.data.resize(n);
        if (use_fmt == InputDataFormat::Float32) {
            if (file_size != (long)(n * sizeof(float))) {
                LOGE("Input[%zu] %s: expected %zu bytes (float32), got %ld",
                     i, paths[i].c_str(), n * sizeof(float), file_size);
                fclose(f);
                Clear();
                return false;
            }
            if (fread(slot.data.data(), sizeof(float), n, f) != n) {
                LOGE("Input[%zu] %s: short read", i, paths[i].c_str());
                fclose(f);
                Clear();
                return false;
            }
        } else { /* UInt8 */
            if (file_size != (long)n) {
                LOGE("Input[%zu] %s: expected %zu bytes (uint8), got %ld",
                     i, paths[i].c_str(), n, file_size);
                fclose(f);
                Clear();
                return false;
            }
            std::vector<unsigned char> raw(n);
            if (fread(raw.data(), 1, n, f) != n) {
                LOGE("Input[%zu] %s: short read", i, paths[i].c_str());
                fclose(f);
                Clear();
                return false;
            }
            for (size_t j = 0; j < n; ++j) {
                slot.data[j] = (float)raw[j] / 255.0f;
            }
        }
        fclose(f);
        slots_.push_back(std::move(slot));
        LOGI("Loaded input[%zu]: %s (%zu elements, %s)", i,
             paths[i].c_str(), n,
             use_fmt == InputDataFormat::Float32 ? "float32" : "uint8");
    }
    return true;
}

std::vector<const float *> InputProvider::DataPtrs() const
{
    std::vector<const float *> v;
    v.reserve(slots_.size());
    for (auto &s : slots_) {
        v.push_back(s.data.data());
    }
    return v;
}

std::vector<size_t> InputProvider::ElementCounts() const
{
    std::vector<size_t> v;
    v.reserve(slots_.size());
    for (auto &s : slots_) {
        v.push_back(s.element_count);
    }
    return v;
}
