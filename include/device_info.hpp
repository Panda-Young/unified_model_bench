#pragma once
/*============================================================================
 * device_info.hpp - Device information collection
 *============================================================================*/

#include <string>

/* Returns a CSV-safe device info string:
 * "CPU: i5-1135G7 2.40GHz; GPU: Intel(R) Iris(R) Xe Graphics (RAM:1.00 GB)" */
std::string get_device_info_csv();
