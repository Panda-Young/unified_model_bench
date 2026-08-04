#pragma once
/*============================================================================
 * input_provider.hpp - Deterministic shared input management (RAII)
 *============================================================================*/

#include "cmd_args.hpp" /* InputDataFormat */
#include "platform.hpp"
#include <string>
#include <vector>

class InputProvider
{
public:
    struct InputSlot {
        std::vector<float> data;
        size_t element_count = 0;
        const float *ptr() const { return data.data(); }
    };

    /* Generate deterministic inputs (seed=42) from element counts */
    void GenerateFromSizes(const std::vector<size_t> &element_counts);

    /* Load inputs from raw binary files.
     *   - paths[i] must exist and contain exactly element_counts[i] floats (float32)
     *     or element_counts[i] bytes (uint8). With Auto, the format is detected
     *     from the file size (n*4 bytes -> float32, n bytes -> uint8).
     *   - Returns false on any error (missing file, size mismatch). */
    bool LoadFromFiles(const std::vector<std::string> &paths,
                       const std::vector<size_t> &element_counts,
                       InputDataFormat fmt = InputDataFormat::Auto);

    size_t Count() const { return slots_.size(); }
    bool Empty() const { return slots_.empty(); }
    void Clear() { slots_.clear(); }

    std::vector<const float *> DataPtrs() const;
    std::vector<size_t> ElementCounts() const;

    const InputSlot &operator[](size_t i) const { return slots_[i]; }
    InputSlot &operator[](size_t i) { return slots_[i]; }

    std::vector<InputSlot> &Slots() { return slots_; }
    const std::vector<InputSlot> &Slots() const { return slots_; }

private:
    std::vector<InputSlot> slots_;
};
