/*============================================================================
 * input_provider.cpp - Deterministic shared input generation
 *============================================================================*/

#include "input_provider.hpp"
#include "log.hpp"
#include <cstdlib>

void InputProvider::GenerateFromSizes(const std::vector<size_t> &element_counts)
{
    Clear();
    srand(42); /* deterministic seed */
    for (size_t n : element_counts) {
        InputSlot slot;
        slot.element_count = n;
        slot.data.resize(n);
        for (size_t j = 0; j < n; ++j)
            slot.data[j] = (float)rand() / (float)RAND_MAX;
        slots_.push_back(std::move(slot));
    }
    LOGI("Generated %zu shared input(s), total %zu elements (seed=42)",
         slots_.size(),
         [this]() { size_t t = 0; for (auto& s : slots_) t += s.element_count; return t; }());
}

std::vector<const float *> InputProvider::DataPtrs() const
{
    std::vector<const float *> v;
    v.reserve(slots_.size());
    for (auto &s : slots_)
        v.push_back(s.data.data());
    return v;
}

std::vector<size_t> InputProvider::ElementCounts() const
{
    std::vector<size_t> v;
    v.reserve(slots_.size());
    for (auto &s : slots_)
        v.push_back(s.element_count);
    return v;
}
