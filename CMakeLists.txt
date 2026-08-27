cmake_minimum_required(VERSION 3.16)
project(unified_bench VERSION 2.0 LANGUAGES C CXX)

# NCNN static builds (Linux desktop prebuilt) require OpenMP at link time.
find_package(OpenMP QUIET)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---- Options ----
option(HAVE_ONNX_BACKEND  "Build ONNX Runtime backend"  ON)
option(HAVE_TFLITE_BACKEND "Build TFLite backend"       OFF)
option(HAVE_NCNN_BACKEND  "Build NCNN backend"          ON)
option(HAVE_MNN_BACKEND   "Build MNN backend"           OFF)
option(HAVE_LITERT_BACKEND "Build LiteRT backend"       OFF)
option(HAVE_QNN_SDK_BACKEND "Build native QNN SDK backend" OFF)

# ---- Architecture detection ----
if(ANDROID)
    set(ARCH_DIR "android/arm64-v8a")
    set(TARGET_SUFFIX "android")
elseif(WIN32)
    if(CMAKE_SIZEOF_VOID_P EQUAL 8)
        set(ARCH_DIR "win-x64")
        set(TARGET_SUFFIX "win-x64")
    else()
        set(ARCH_DIR "win-x86")
        set(TARGET_SUFFIX "win-x86")
    endif()
elseif(UNIX AND NOT ANDROID)
    # Linux desktop: vendored prebuilt libs live under linux-x64
    # (deps/onnxruntime/lib/linux-x64, deps/ncnn/lib/linux-x64). The suffix is
    # kept as "linux-x64" so the produced binary name is self-describing.
    set(ARCH_DIR "linux-x64")
    set(TARGET_SUFFIX "linux-x64")
endif()

# ---- Include paths ----
set(INC_DIR  "${CMAKE_SOURCE_DIR}/include")
set(DEPS_DIR "${CMAKE_SOURCE_DIR}/deps")

set(ONNX_INC "${DEPS_DIR}/onnxruntime/include")
set(TFLITE_INC "${DEPS_DIR}/tflite/include")
if(ANDROID)
    set(NCNN_INC "${DEPS_DIR}/ncnn/include-android")
elseif(UNIX AND NOT APPLE)
    # Linux desktop uses the Linux prebuilt headers (platform macros differ
    # from the Windows ones in deps/ncnn/include).
    set(NCNN_INC "${DEPS_DIR}/ncnn/include-linux")
else()
    set(NCNN_INC "${DEPS_DIR}/ncnn/include")
endif()
set(MNN_INC   "${DEPS_DIR}/mnn/include")
set(LITERT_INC "${DEPS_DIR}/litert/litert_cc_sdk")

# QNN SDK (Qualcomm AI Engine Direct) — for TFLITE_NPU on Android
if(ANDROID AND (HAVE_TFLITE_BACKEND OR HAVE_QNN_SDK_BACKEND))
    if(NOT DEFINED QNN_SDK_ROOT)
        set(QNN_SDK_ROOT "C:/Qualcomm/AIStack/QAIRT/2.48.40.260702"
            CACHE PATH "Qualcomm AI Engine Direct SDK root")
    endif()
    if(EXISTS "${QNN_SDK_ROOT}/include/QNN/TFLiteDelegate/QnnTFLiteDelegate.h" OR
       EXISTS "${QNN_SDK_ROOT}/include/QNN/QnnInterface.h")
        include_directories("${QNN_SDK_ROOT}/include"
                            "${QNN_SDK_ROOT}/include/QNN")
        set(HAVE_QNN_DELEGATE TRUE)
        message(STATUS "QNN SDK found: ${QNN_SDK_ROOT}")
        if(EXISTS "${QNN_SDK_ROOT}/include/QNN/TFLiteDelegate/QnnTFLiteDelegate.h")
            add_definitions(-DHAVE_QNN_DELEGATE)
        endif()
    else()
        message(WARNING "QNN SDK not found at ${QNN_SDK_ROOT}; QNN backends disabled")
    endif()
endif()

include_directories(${INC_DIR} ${ONNX_INC})
if(HAVE_TFLITE_BACKEND)
    include_directories(${TFLITE_INC})
endif()
if(HAVE_NCNN_BACKEND)
    include_directories(${NCNN_INC})
endif()
if(HAVE_MNN_BACKEND)
    include_directories(${MNN_INC})
endif()
if(HAVE_LITERT_BACKEND)
    include_directories(${LITERT_INC})
endif()

# ---- Compile definitions ----
add_definitions(-D_CRT_SECURE_NO_WARNINGS -DNOMINMAX)
if(HAVE_ONNX_BACKEND)
    add_definitions(-DHAVE_ONNX_BACKEND)
endif()
if(HAVE_TFLITE_BACKEND)
    add_definitions(-DHAVE_TFLITE_BACKEND)
endif()
if(HAVE_NCNN_BACKEND)
    add_definitions(-DHAVE_NCNN_BACKEND)
    if(NOT WIN32)
        # Android / Linux: NCNN statically linked, no __declspec
        add_definitions(-DNCNN_STATIC_DEFINE)
    endif()
endif()
if(HAVE_MNN_BACKEND)
    add_definitions(-DHAVE_MNN_BACKEND)
endif()
if(HAVE_LITERT_BACKEND)
    add_definitions(-DHAVE_LITERT_BACKEND)
endif()
if(HAVE_QNN_SDK_BACKEND AND ANDROID AND HAVE_QNN_DELEGATE)
    add_definitions(-DHAVE_QNN_SDK_BACKEND)
endif()

# ---- NEON FP16 conversion (arm64) ----
# qnn_backend.cpp uses vcvt_f16_f32 / vcvt_f32_f16 for fast FP16 I/O
# conversion. These intrinsics need the ARMv8.2-A FP16 extension, so enable it
# for that translation unit only (the rest of the project stays on armv8-a).
if(ANDROID AND HAVE_QNN_SDK_BACKEND)
    set_source_files_properties(${CMAKE_SOURCE_DIR}/src/qnn_backend.cpp PROPERTIES
        COMPILE_OPTIONS "-march=armv8.2-a+fp16")
endif()

# ---- Source files ----
# Core (always compiled): orchestration, registry, CLI, IO, collection.
# NOTE: keep in sync when adding a new core module; backend files are appended
# conditionally below so disabled backends cost zero build time (each backend
# .cpp is additionally wrapped in #ifdef HAVE_*_BACKEND as a safety net).
set(SRC_FILES
    "${CMAKE_SOURCE_DIR}/src/main.cpp"
    "${CMAKE_SOURCE_DIR}/src/benchmark_runner.cpp"
    "${CMAKE_SOURCE_DIR}/src/scheduler.cpp"
    "${CMAKE_SOURCE_DIR}/src/csv_utils.cpp"
    "${CMAKE_SOURCE_DIR}/src/backend_registry.cpp"
    "${CMAKE_SOURCE_DIR}/src/cmd_args.cpp"
    "${CMAKE_SOURCE_DIR}/src/device_info.cpp"
    "${CMAKE_SOURCE_DIR}/src/file_ops.cpp"
    "${CMAKE_SOURCE_DIR}/src/input_provider.cpp"
    "${CMAKE_SOURCE_DIR}/src/log.cpp"
    "${CMAKE_SOURCE_DIR}/src/model_loader.cpp"
    "${CMAKE_SOURCE_DIR}/src/result_collector.cpp"
    # Shared QNN SoC detection, used by both the ORT QNN EP path and the
    # native QNN SDK backend; standalone (no QNN SDK headers needed).
    "${CMAKE_SOURCE_DIR}/src/qnn_soc.cpp"
)
if(HAVE_ONNX_BACKEND)
    list(APPEND SRC_FILES "${CMAKE_SOURCE_DIR}/src/onnx_backend.cpp")
endif()
if(HAVE_TFLITE_BACKEND)
    list(APPEND SRC_FILES "${CMAKE_SOURCE_DIR}/src/tflite_backend.cpp")
endif()
if(HAVE_NCNN_BACKEND)
    list(APPEND SRC_FILES "${CMAKE_SOURCE_DIR}/src/ncnn_backend.cpp")
endif()
if(HAVE_MNN_BACKEND)
    list(APPEND SRC_FILES "${CMAKE_SOURCE_DIR}/src/mnn_backend.cpp")
endif()
if(HAVE_LITERT_BACKEND)
    # Qualcomm options stubs are only needed by the LiteRT backend (Android).
    list(APPEND SRC_FILES
        "${CMAKE_SOURCE_DIR}/src/litert_backend.cpp"
        "${CMAKE_SOURCE_DIR}/src/litert_qualcomm_stubs.cpp")
endif()
if(HAVE_QNN_SDK_BACKEND)
    list(APPEND SRC_FILES "${CMAKE_SOURCE_DIR}/src/qnn_backend.cpp")
endif()

# ---- Executable ----
add_executable(unified_bench_${TARGET_SUFFIX} ${SRC_FILES})
set_target_properties(unified_bench_${TARGET_SUFFIX} PROPERTIES OUTPUT_NAME "unified_bench")

# ---- Link libraries ----
if(WIN32)
    target_link_libraries(unified_bench_${TARGET_SUFFIX} PRIVATE Advapi32.lib psapi.lib)
endif()

if(HAVE_ONNX_BACKEND AND NOT WIN32)
    # ONNX Runtime .so -- loaded dynamically, linked for NNAPI symbol
    target_link_libraries(unified_bench_${TARGET_SUFFIX} PRIVATE
        "${DEPS_DIR}/onnxruntime/lib/${ARCH_DIR}/libonnxruntime.so")
endif()

if(HAVE_NCNN_BACKEND)
    if(WIN32)
        target_link_libraries(unified_bench_${TARGET_SUFFIX} PRIVATE
            "${DEPS_DIR}/ncnn/lib/${ARCH_DIR}/ncnn.lib")
    elseif(ANDROID)
        target_link_libraries(unified_bench_${TARGET_SUFFIX} PRIVATE
            "${DEPS_DIR}/ncnn/lib/${ARCH_DIR}/libncnn.so")
    else()
        # Linux desktop: official Ubuntu prebuilt ships a static libncnn.a
        # built with OpenMP + Vulkan/Glslang, so we must pull in libgomp and
        # the bundled glslang static libraries.
        target_link_libraries(unified_bench_${TARGET_SUFFIX} PRIVATE
            "${DEPS_DIR}/ncnn/lib/${ARCH_DIR}/libncnn.a"
            "${DEPS_DIR}/ncnn/lib/${ARCH_DIR}/libglslang.a"
            "${DEPS_DIR}/ncnn/lib/${ARCH_DIR}/libglslang-default-resource-limits.a"
            "${DEPS_DIR}/ncnn/lib/${ARCH_DIR}/libMachineIndependent.a"
            "${DEPS_DIR}/ncnn/lib/${ARCH_DIR}/libGenericCodeGen.a"
            "${DEPS_DIR}/ncnn/lib/${ARCH_DIR}/libOSDependent.a"
            "${DEPS_DIR}/ncnn/lib/${ARCH_DIR}/libSPIRV.a"
            OpenMP::OpenMP_CXX)
    endif()
endif()

if(HAVE_MNN_BACKEND)
    if(WIN32)
        target_link_libraries(unified_bench_${TARGET_SUFFIX} PRIVATE
            "${DEPS_DIR}/mnn/lib/${ARCH_DIR}/MNN.lib")
    else()
        target_link_libraries(unified_bench_${TARGET_SUFFIX} PRIVATE
            "${DEPS_DIR}/mnn/lib/arm64-v8a/libMNN.so")
    endif()
endif()

if(HAVE_TFLITE_BACKEND)
    if(WIN32)
        # On Windows, libLiteRt.dll also exports the full TFLite C API
        # (TfLiteModelCreate, TfLiteInterpreterCreate, etc.) which works
        # correctly.  We link libLiteRt.lib to get basic TFLite symbols and
        # dynamically load XNNPACK/GPU delegates from tensorflowlite_c.dll
        # at runtime to avoid the buggy XNNPACK in libLiteRt.
        # Do NOT link tensorflowlite_c.dll.if.lib — it conflicts with
        # libLiteRt.lib for TfLite* symbols.
        if(HAVE_LITERT_BACKEND)
            message(STATUS "TFLite + LiteRT coexistence: TFLite symbols from libLiteRt.lib, "
                           "XNNPACK from tensorflowlite_c.dll (dynamic)")
        endif()
    else()
        target_link_libraries(unified_bench_${TARGET_SUFFIX} PRIVATE
            "${DEPS_DIR}/tflite/lib/${ARCH_DIR}/libtensorflowlite_c.so"
            "${DEPS_DIR}/tflite/lib/${ARCH_DIR}/libtensorflowlite_gpu_delegate.so")
        if(HAVE_QNN_DELEGATE)
            target_link_libraries(unified_bench_${TARGET_SUFFIX} PRIVATE
                "${QNN_SDK_ROOT}/lib/aarch64-android/libQnnTFLiteDelegate.so")
        endif()
    endif()
endif()

if(HAVE_LITERT_BACKEND)
    if(WIN32)
        target_link_libraries(unified_bench_${TARGET_SUFFIX} PRIVATE
            "${DEPS_DIR}/litert/liteRT_runtime/windows_x86_64/libLiteRt.lib")
    else()
        target_link_libraries(unified_bench_${TARGET_SUFFIX} PRIVATE
            "${DEPS_DIR}/litert/liteRT_runtime/android_arm64/libLiteRt.so")
    endif()
endif()

# ---- Post-build: copy dependency DLLs to output dir (Windows only) ----
if(WIN32)
set(OUT_DIR "${CMAKE_BINARY_DIR}/$<CONFIG>")
if(HAVE_NCNN_BACKEND)
    add_custom_command(TARGET unified_bench_${TARGET_SUFFIX} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${DEPS_DIR}/ncnn/lib/${ARCH_DIR}/ncnn.dll"
            "${OUT_DIR}/"
        COMMENT "Copying ncnn.dll")
endif()
if(HAVE_MNN_BACKEND)
    add_custom_command(TARGET unified_bench_${TARGET_SUFFIX} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${DEPS_DIR}/mnn/lib/${ARCH_DIR}/MNN.dll"
            "${OUT_DIR}/"
        COMMENT "Copying MNN.dll")
endif()
if(HAVE_TFLITE_BACKEND)
    # Always copy tensorflowlite_c.dll — it's loaded dynamically for
    # XNNPACK/GPU delegates on Windows (to avoid libLiteRt's buggy copy).
    add_custom_command(TARGET unified_bench_${TARGET_SUFFIX} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${DEPS_DIR}/tflite/lib/${ARCH_DIR}/tensorflowlite_c.dll"
            "${OUT_DIR}/"
        COMMENT "Copying tensorflowlite_c.dll (dynamic load)")
endif()
if(HAVE_LITERT_BACKEND)
    add_custom_command(TARGET unified_bench_${TARGET_SUFFIX} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${DEPS_DIR}/litert/liteRT_runtime/windows_x86_64/libLiteRt.dll"
            "${DEPS_DIR}/litert/liteRT_runtime/windows_x86_64/libLiteRtWebGpuAccelerator.dll"
            "${OUT_DIR}/"
        COMMENT "Copying LiteRT DLLs")

    # LiteRT GPU accelerator uses Dawn/WebGPU which requires DXC (DirectX Shader Compiler)
    # at runtime for D3D12 shader compilation.  Search Windows SDK for matching DLLs.
    if(CMAKE_SIZEOF_VOID_P EQUAL 8)
        set(DXC_ARCH "x64")
    else()
        set(DXC_ARCH "x86")
    endif()
    # Try latest Windows SDK version first
    foreach(sdk_ver "10.0.26100.0" "10.0.22621.0" "10.0.22000.0" "10.0.19041.0")
        set(DXC_DIR "C:/Program Files (x86)/Windows Kits/10/bin/${sdk_ver}/${DXC_ARCH}")
        if(EXISTS "${DXC_DIR}/dxcompiler.dll")
            message(STATUS "Found DXC DLLs in: ${DXC_DIR}")
            add_custom_command(TARGET unified_bench_${TARGET_SUFFIX} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${DXC_DIR}/dxcompiler.dll"
                    "${DXC_DIR}/dxil.dll"
                    "${OUT_DIR}/"
                COMMENT "Copying DXC DLLs (${sdk_ver})")
            break()
        endif()
    endforeach()
endif()
endif()

# ---- Unit tests (doctest, single-header) ----
# The test runner compiles the pure-logic modules WITHOUT the HAVE_*_BACKEND
# macros, so no backend (and its 3rd-party libs) is pulled into the link.
option(BUILD_TESTING "Build the doctest unit test runner" ON)
if(BUILD_TESTING)
    enable_testing()
    add_executable(unified_bench_tests
        tests/test_main.cpp
        tests/test_csv_utils.cpp
        tests/test_scheduler.cpp
        tests/test_model_loader.cpp
        tests/test_cmd_args.cpp
        tests/test_input_provider.cpp
        tests/test_result_collector.cpp
        # Units under test (no backend macros -> registry stays empty)
        src/csv_utils.cpp
        src/scheduler.cpp
        src/backend_registry.cpp
        src/model_loader.cpp
        src/result_collector.cpp
        src/cmd_args.cpp
        src/input_provider.cpp
        src/log.cpp
    )
    target_include_directories(unified_bench_tests PRIVATE
        "${CMAKE_SOURCE_DIR}/include" "${DEPS_DIR}/doctest")
    target_compile_definitions(unified_bench_tests PRIVATE
        _CRT_SECURE_NO_WARNINGS NOMINMAX)
    # The three estimate_weight_mb cases in test_model_loader.cpp need the
    # real repo test models, which are .gitignored ("user-provided") and thus
    # absent on CI runners. Only define these macros when the models are
    # actually present, so the tests adapt to the environment instead of
    # failing on a missing file.
    if(EXISTS "${CMAKE_SOURCE_DIR}/test_model.onnx")
        target_compile_definitions(unified_bench_tests PRIVATE
            "UB_TEST_MODEL_PATH=\"${CMAKE_SOURCE_DIR}/test_model.onnx\""
            "UB_TEST_MODEL_DIR=\"${CMAKE_SOURCE_DIR}\"")
    endif()
    # The top-level add_definitions(-DHAVE_*_BACKEND) is directory-scoped, so
    # this target inherits every backend macro. Undefine them here: the units
    # under test must compile with an EMPTY backend registry, otherwise
    # backend_registry.cpp's InitDefaults() references Create*Backend() whose
    # implementations live in the backend .cpp files that are deliberately NOT
    # part of this target (and would pull in 3rd-party libs).
    if(MSVC)
        target_compile_options(unified_bench_tests PRIVATE
            /UHAVE_ONNX_BACKEND /UHAVE_TFLITE_BACKEND /UHAVE_NCNN_BACKEND
            /UHAVE_MNN_BACKEND /UHAVE_LITERT_BACKEND /UHAVE_QNN_SDK_BACKEND
            # doctest.h specializes std::tuple (C5285/N5014) - 3rd-party header, ignore
            /wd5285)
    else()
        target_compile_options(unified_bench_tests PRIVATE
            -UHAVE_ONNX_BACKEND -UHAVE_TFLITE_BACKEND -UHAVE_NCNN_BACKEND
            -UHAVE_MNN_BACKEND -UHAVE_LITERT_BACKEND -UHAVE_QNN_SDK_BACKEND)
    endif()
    add_test(NAME unit_tests COMMAND unified_bench_tests)
endif()
