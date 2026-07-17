cmake_minimum_required(VERSION 3.16)
project(unified_bench VERSION 2.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---- Options ----
option(HAVE_ONNX_BACKEND  "Build ONNX Runtime backend"  ON)
option(HAVE_TFLITE_BACKEND "Build TFLite backend"       OFF)
option(HAVE_NCNN_BACKEND  "Build NCNN backend"          ON)
option(HAVE_MNN_BACKEND   "Build MNN backend"           OFF)
option(HAVE_LITERT_BACKEND "Build LiteRT backend"       OFF)

# ---- Architecture detection ----
if(ANDROID)
    set(ARCH_DIR "android/arm64-v8a")
    set(TARGET_SUFFIX "android")
elseif(CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(ARCH_DIR "win-x64")
    set(TARGET_SUFFIX "win-x64")
else()
    set(ARCH_DIR "win-x86")
    set(TARGET_SUFFIX "win-x86")
endif()

# ---- Include paths ----
set(INC_DIR  "${CMAKE_SOURCE_DIR}/include")
set(DEPS_DIR "${CMAKE_SOURCE_DIR}/deps")

set(ONNX_INC "${DEPS_DIR}/onnxruntime/include")
set(TFLITE_INC "${DEPS_DIR}/tflite/include")
if(ANDROID)
    set(NCNN_INC "${DEPS_DIR}/ncnn/include-android")
else()
    set(NCNN_INC "${DEPS_DIR}/ncnn/include")
endif()
set(MNN_INC   "${DEPS_DIR}/mnn/include")
set(LITERT_INC "${DEPS_DIR}/litert/litert_cc_sdk")

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

# ---- Source files (glob all .cpp in src/) ----
file(GLOB SRC_FILES "${CMAKE_SOURCE_DIR}/src/*.cpp")

# ---- Executable ----
add_executable(unified_bench_${TARGET_SUFFIX} ${SRC_FILES})
set_target_properties(unified_bench_${TARGET_SUFFIX} PROPERTIES OUTPUT_NAME "unified_bench")

# ---- Link libraries ----
if(WIN32)
    target_link_libraries(unified_bench_${TARGET_SUFFIX} PRIVATE Advapi32.lib)
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
    else()
        target_link_libraries(unified_bench_${TARGET_SUFFIX} PRIVATE
            "${DEPS_DIR}/ncnn/lib/${ARCH_DIR}/libncnn.so")
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

if(HAVE_LITERT_BACKEND)
    if(WIN32)
        target_link_libraries(unified_bench_${TARGET_SUFFIX} PRIVATE
            "${DEPS_DIR}/litert/liteRT_runtime/windows_x86_64/libLiteRt.lib")
    else()
        target_link_libraries(unified_bench_${TARGET_SUFFIX} PRIVATE
            "${DEPS_DIR}/litert/liteRT_runtime/android_arm64/libLiteRt.so")
    endif()
endif()

if(HAVE_TFLITE_BACKEND)
    if(WIN32)
        target_link_libraries(unified_bench_${TARGET_SUFFIX} PRIVATE
            "${DEPS_DIR}/tflite/lib/${ARCH_DIR}/tensorflowlite_c.dll.if.lib")
    else()
        target_link_libraries(unified_bench_${TARGET_SUFFIX} PRIVATE
            "${DEPS_DIR}/tflite/lib/${ARCH_DIR}/libtensorflowlite_c.so"
            "${DEPS_DIR}/tflite/lib/${ARCH_DIR}/libtensorflowlite_gpu_delegate.so")
    endif()
endif()

# ---- Post-build: copy dependency DLLs to output dir (Windows only) ----
if(WIN32)
set(OUT_DIR "${CMAKE_BINARY_DIR}/Release")
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
    add_custom_command(TARGET unified_bench_${TARGET_SUFFIX} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${DEPS_DIR}/tflite/lib/${ARCH_DIR}/tensorflowlite_c.dll"
            "${OUT_DIR}/"
        COMMENT "Copying tensorflowlite_c.dll")
endif()
if(HAVE_LITERT_BACKEND)
    add_custom_command(TARGET unified_bench_${TARGET_SUFFIX} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${DEPS_DIR}/litert/liteRT_runtime/windows_x86_64/libLiteRt.dll"
            "${DEPS_DIR}/litert/liteRT_runtime/windows_x86_64/libLiteRtWebGpuAccelerator.dll"
            "${OUT_DIR}/"
        COMMENT "Copying LiteRT DLLs")
endif()
endif()
