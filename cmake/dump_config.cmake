# Dump the fully resolved target configuration of the main executable.
#
# Included from CMakeLists.txt when -DUB_DUMP_BUILD_CONFIG=ON. Used to prove
# that build-system refactors are behaviour-preserving: dump the config before
# and after the change and diff the two outputs.
#
# Usage:
#   cmake -S . -B build -DUB_DUMP_BUILD_CONFIG=ON
#
# Directory-scoped properties (include_directories() / add_definitions()) are
# part of the output too, since the project relies on both mechanisms.

if(NOT TARGET unified_bench_${TARGET_SUFFIX})
    message(FATAL_ERROR "UB_DUMP_BUILD_CONFIG: target unified_bench_${TARGET_SUFFIX} not defined")
endif()

set(_ub_target unified_bench_${TARGET_SUFFIX})

message(STATUS "==== UB_BUILD_CONFIG_BEGIN ====")
message(STATUS "TARGET_SUFFIX=${TARGET_SUFFIX}")
message(STATUS "ARCH_DIR=${ARCH_DIR}")

foreach(_prop SOURCES INCLUDE_DIRECTORIES COMPILE_DEFINITIONS COMPILE_OPTIONS LINK_LIBRARIES)
    get_target_property(_val ${_ub_target} ${_prop})
    if(NOT _val)
        set(_val "")
    endif()
    # Sort for a stable diff; sources may be reordered by refactors without
    # changing the build result.
    list(REMOVE_DUPLICATES _val)
    list(SORT _val)
    message(STATUS "PROP ${_prop}:")
    foreach(_item IN LISTS _val)
        message(STATUS "  ${_prop}.${_item}")
    endforeach()
endforeach()

# Directory-scoped equivalents, which the target also inherits.
get_directory_property(_dir_inc INCLUDE_DIRECTORIES)
get_directory_property(_dir_def COMPILE_DEFINITIONS)
list(SORT _dir_inc)
list(SORT _dir_def)
message(STATUS "PROP DIR_INCLUDE_DIRECTORIES:")
foreach(_item IN LISTS _dir_inc)
    message(STATUS "  DIR_INCLUDE_DIRECTORIES.${_item}")
endforeach()
message(STATUS "PROP DIR_COMPILE_DEFINITIONS:")
foreach(_item IN LISTS _dir_def)
    message(STATUS "  DIR_COMPILE_DEFINITIONS.${_item}")
endforeach()

# POST_BUILD commands are not introspectable, so re-declare the DLL set the
# same way CMakeLists.txt does and print it for comparison.
message(STATUS "PROP POST_BUILD_COPY_DLLS:")
foreach(_dll IN LISTS UB_POST_BUILD_DLLS)
    message(STATUS "  POST_BUILD_COPY_DLLS.${_dll}")
endforeach()

message(STATUS "==== UB_BUILD_CONFIG_END ====")
