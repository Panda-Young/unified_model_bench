# =============================================================================
# backends.cmake - Backend declaration & wiring (included by CMakeLists.txt)
#
# Purpose: adding a backend used to touch six separate places in
# CMakeLists.txt (option, add_definitions, include_directories, SRC_FILES,
# target_link_libraries, POST_BUILD copies). This module centralises them
# behind one ub_row() entry per backend.
#
# IMPORTANT: this is a REFACTOR ONLY - the resolved build configuration must
# stay identical. Verified with cmake/dump_config.cmake (see README §3.8).
#
# Two-phase by necessity: add_definitions()/include_directories() are
# directory-scoped and must run before add_executable() to keep the exact same
# semantics, while target_link_libraries()/add_custom_command(POST_BUILD)
# require the target to already exist.
#   Phase 1 (before add_executable): ub_backends_phase1(<out_sources_var>)
#   Phase 2 (after  add_executable): ub_backends_phase2()
#
# NOTE: deps/onnxruntime/include is deliberately NOT part of this table -
# CMakeLists.txt adds it unconditionally (independent of HAVE_ONNX_BACKEND),
# and this refactor preserves that behaviour.
# =============================================================================

# ---------------------------------------------------------------------------
# Row storage: ub_row() is called at include time (directory scope), so the
# UB_ROW_* variables it sets remain visible to the phase functions below.
# ---------------------------------------------------------------------------
set(UB_ROWS "")

set(UB_ROW_FIELDS
    DEFINITION CONDITION INCLUDE_DIR SOURCES
    WIN32_LIBS ANDROID_LIBS LINUX_LIBS NONWIN_LIBS
    UNIX_DEFS WIN32_DLLS)

macro(ub_row NAME)
    cmake_parse_arguments(R "" "DEFINITION;CONDITION;INCLUDE_DIR"
        "SOURCES;WIN32_LIBS;ANDROID_LIBS;LINUX_LIBS;NONWIN_LIBS;UNIX_DEFS;WIN32_DLLS"
        ${ARGN})
    list(APPEND UB_ROWS "${NAME}")
    foreach(_f IN LISTS UB_ROW_FIELDS)
        set(UB_ROW_${NAME}_${_f} "${R_${_f}}")
    endforeach()
endmacro()

# ---------------------------------------------------------------------------
# The table - one entry per backend. Keep in sync with BackendRegistry.
# ---------------------------------------------------------------------------

# ONNX: headers are always included (see note above). On Windows the runtime
# DLL is loaded dynamically, so no import library is linked; elsewhere the .so
# is linked directly.
ub_row(ONNX
    DEFINITION HAVE_ONNX_BACKEND
    SOURCES onnx_backend.cpp
    ANDROID_LIBS "${DEPS_DIR}/onnxruntime/lib/${ARCH_DIR}/libonnxruntime.so"
    LINUX_LIBS   "${DEPS_DIR}/onnxruntime/lib/${ARCH_DIR}/libonnxruntime.so")

# TFLite: on Windows we intentionally link NOTHING - tensorflowlite_c.dll is
# loaded at runtime for the XNNPACK/GPU delegates (linking its .if.lib
# conflicts with libLiteRt.lib). Only tensorflowlite_c.dll is copied.
ub_row(TFLITE
    DEFINITION HAVE_TFLITE_BACKEND
    INCLUDE_DIR "${TFLITE_INC}"
    SOURCES tflite_backend.cpp
    NONWIN_LIBS "${DEPS_DIR}/tflite/lib/${ARCH_DIR}/libtensorflowlite_c.so"
                "${DEPS_DIR}/tflite/lib/${ARCH_DIR}/libtensorflowlite_gpu_delegate.so"
    WIN32_DLLS  "${DEPS_DIR}/tflite/lib/${ARCH_DIR}/tensorflowlite_c.dll")

# NCNN: static on Android/Linux (no __declspec), NCNN_STATIC_DEFINE required.
# Linux desktop additionally needs the bundled glslang static libs + OpenMP.
ub_row(NCNN
    DEFINITION HAVE_NCNN_BACKEND
    INCLUDE_DIR "${NCNN_INC}"
    SOURCES ncnn_backend.cpp
    UNIX_DEFS  NCNN_STATIC_DEFINE
    WIN32_LIBS "${DEPS_DIR}/ncnn/lib/${ARCH_DIR}/ncnn.lib"
    ANDROID_LIBS "${DEPS_DIR}/ncnn/lib/${ARCH_DIR}/libncnn.so"
    LINUX_LIBS   "${DEPS_DIR}/ncnn/lib/${ARCH_DIR}/libncnn.a"
                 "${DEPS_DIR}/ncnn/lib/${ARCH_DIR}/libglslang.a"
                 "${DEPS_DIR}/ncnn/lib/${ARCH_DIR}/libglslang-default-resource-limits.a"
                 "${DEPS_DIR}/ncnn/lib/${ARCH_DIR}/libMachineIndependent.a"
                 "${DEPS_DIR}/ncnn/lib/${ARCH_DIR}/libGenericCodeGen.a"
                 "${DEPS_DIR}/ncnn/lib/${ARCH_DIR}/libOSDependent.a"
                 "${DEPS_DIR}/ncnn/lib/${ARCH_DIR}/libSPIRV.a"
                 OpenMP::OpenMP_CXX
    WIN32_DLLS "${DEPS_DIR}/ncnn/lib/${ARCH_DIR}/ncnn.dll")

ub_row(MNN
    DEFINITION HAVE_MNN_BACKEND
    INCLUDE_DIR "${MNN_INC}"
    SOURCES mnn_backend.cpp
    WIN32_LIBS "${DEPS_DIR}/mnn/lib/${ARCH_DIR}/MNN.lib"
    # NOTE: upstream uses arm64-v8a unconditionally in the non-Windows branch,
    # so Android and Linux share the same path here.
    ANDROID_LIBS "${DEPS_DIR}/mnn/lib/arm64-v8a/libMNN.so"
    LINUX_LIBS   "${DEPS_DIR}/mnn/lib/arm64-v8a/libMNN.so"
    WIN32_DLLS "${DEPS_DIR}/mnn/lib/${ARCH_DIR}/MNN.dll")

# LiteRT: the Qualcomm options stubs ship with this backend. The DXC (DirectX
# Shader Compiler) probe is NOT part of this table - it is a Windows SDK
# version search rather than simple backend wiring, so it stays inline in
# CMakeLists.txt (it appends to UB_POST_BUILD_DLLS for the config dump).
ub_row(LITERT
    DEFINITION HAVE_LITERT_BACKEND
    INCLUDE_DIR "${LITERT_INC}"
    SOURCES litert_backend.cpp litert_qualcomm_stubs.cpp
    WIN32_LIBS "${DEPS_DIR}/litert/liteRT_runtime/windows_x86_64/libLiteRt.lib"
    ANDROID_LIBS "${DEPS_DIR}/litert/liteRT_runtime/android_arm64/libLiteRt.so"
    WIN32_DLLS "${DEPS_DIR}/litert/liteRT_runtime/windows_x86_64/libLiteRt.dll"
               "${DEPS_DIR}/litert/liteRT_runtime/windows_x86_64/libLiteRtWebGpuAccelerator.dll")

# QNN SDK: only meaningful on Android with the SDK present, hence CONDITION.
# The QNN libraries are dlopen'd at runtime, so there is no link entry.
ub_row(QNN_SDK
    DEFINITION HAVE_QNN_SDK_BACKEND
    CONDITION "HAVE_QNN_SDK_BACKEND AND ANDROID AND HAVE_QNN_DELEGATE"
    SOURCES qnn_backend.cpp)

# ---------------------------------------------------------------------------
# Phase 1: definitions, include dirs, source collection.
# ---------------------------------------------------------------------------
function(ub_backends_phase1 out_sources)
    set(_srcs "")
    foreach(_name IN LISTS UB_ROWS)
        set(_def "${UB_ROW_${_name}_DEFINITION}")
        set(_cond "${UB_ROW_${_name}_CONDITION}")
        if(NOT ${_def})
            continue()
        endif()
        if(_cond AND NOT (${_cond}))
            message(STATUS "Backend ${_name}: option is ON but platform "
                           "preconditions are not met - disabled")
            continue()
        endif()

        add_definitions(-D${_def})
        if(UB_ROW_${_name}_UNIX_DEFS AND NOT WIN32)
            foreach(_d IN LISTS UB_ROW_${_name}_UNIX_DEFS)
                add_definitions(-D${_d})
            endforeach()
        endif()
        if(UB_ROW_${_name}_INCLUDE_DIR)
            include_directories("${UB_ROW_${_name}_INCLUDE_DIR}")
        endif()
        foreach(_s IN LISTS UB_ROW_${_name}_SOURCES)
            list(APPEND _srcs "${CMAKE_SOURCE_DIR}/src/${_s}")
        endforeach()
    endforeach()
    set(${out_sources} ${_srcs} PARENT_SCOPE)
endfunction()

# ---------------------------------------------------------------------------
# Phase 2: link libraries + Windows POST_BUILD DLL copies.
# ---------------------------------------------------------------------------
function(ub_backends_phase2)
    set(_dlls_out "${UB_POST_BUILD_DLLS}")
    foreach(_name IN LISTS UB_ROWS)
        set(_def "${UB_ROW_${_name}_DEFINITION}")
        set(_cond "${UB_ROW_${_name}_CONDITION}")
        if(NOT ${_def})
            continue()
        endif()
        if(_cond AND NOT (${_cond}))
            continue()
        endif()

        if(WIN32)
            set(_libs "${UB_ROW_${_name}_WIN32_LIBS}")
        elseif(ANDROID)
            set(_libs "${UB_ROW_${_name}_ANDROID_LIBS}")
        else()
            set(_libs "${UB_ROW_${_name}_LINUX_LIBS}")
        endif()
        # NONWIN_LIBS (e.g. TFLite) applies to both Android and Linux.
        if(NOT WIN32 AND UB_ROW_${_name}_NONWIN_LIBS)
            list(APPEND _libs ${UB_ROW_${_name}_NONWIN_LIBS})
        endif()
        if(_libs)
            target_link_libraries(unified_bench_${TARGET_SUFFIX} PRIVATE ${_libs})
        endif()

        if(WIN32 AND UB_ROW_${_name}_WIN32_DLLS)
            foreach(_dll IN LISTS UB_ROW_${_name}_WIN32_DLLS)
                list(APPEND _dlls_out "${_dll}")
                add_custom_command(TARGET unified_bench_${TARGET_SUFFIX} POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${_dll}" "${OUT_DIR}/"
                    COMMENT "Copying ${_dll}")
            endforeach()
        endif()
    endforeach()
    set(UB_POST_BUILD_DLLS ${_dlls_out} PARENT_SCOPE)
endfunction()

# Flags that undefine every backend macro, for the unit-test target (it must
# compile the shared modules with an empty backend registry).
set(UB_TEST_UNDEF_FLAGS "")
foreach(_opt IN LISTS UB_ALL_BACKEND_OPTIONS)
    if(MSVC)
        list(APPEND UB_TEST_UNDEF_FLAGS "/U${_opt}")
    else()
        list(APPEND UB_TEST_UNDEF_FLAGS "-U${_opt}")
    endif()
endforeach()
