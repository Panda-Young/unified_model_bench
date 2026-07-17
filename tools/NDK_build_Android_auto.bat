@echo off
setlocal enabledelayedexpansion

REM ============================================================
REM  Unified Benchmark - Android NDK build (arm64-v8a)
REM  Requires: ANDROID_NDK_ROOT
REM ============================================================

IF NOT DEFINED ANDROID_NDK_ROOT (
    echo ERROR: ANDROID_NDK_ROOT is not set
    exit /b 1
)

REM Trim trailing spaces from ANDROID_NDK_ROOT (common copy-paste issue)
for /f "tokens=*" %%a in ("%ANDROID_NDK_ROOT%") do set "ANDROID_NDK_ROOT=%%a"

set "NDK=%ANDROID_NDK_ROOT%\toolchains\llvm\prebuilt\windows-x86_64\bin"
set "CC=%NDK%\aarch64-linux-android21-clang"
set "CXX=%NDK%\aarch64-linux-android21-clang++"

REM Verify compiler exists before building
if not exist "%CC%.exe" if not exist "%CC%.cmd" if not exist "%CC%" (
    echo ERROR: Compiler not found: "%CC%"
    echo   ANDROID_NDK_ROOT=%ANDROID_NDK_ROOT%
    echo   Check: dir "%NDK%" 
    dir "%NDK%" 2>nul
    exit /b 1
)
echo NDK: %ANDROID_NDK_ROOT%
echo CC:  %CC%

set OUT=unified_bench
set ROOT=%~dp0..
set INC=%ROOT%\include
set SRC=%ROOT%\src
set ONNX_INC=%ROOT%\deps\onnxruntime\include
set TFLITE_INC=%ROOT%\deps\tflite\include
set TFLITE_LIB=%ROOT%\deps\tflite\lib\android\arm64-v8a
set ONNX_LIB=%ROOT%\deps\onnxruntime\lib\android\arm64-v8a
set QNN_LIB=%ROOT%\deps\onnxruntime\lib\android\qnn\arm64-v8a
set NCNN_INC=%ROOT%\deps\ncnn\include-android
set NCNN_LIB=%ROOT%\deps\ncnn\lib\android\arm64-v8a
set MNN_INC=%ROOT%\deps\mnn\include
set MNN_LIB=%ROOT%\deps\mnn\lib\arm64-v8a

REM LiteRT (Google next-gen TFLite runtime)
set LITERT_INC=%ROOT%\deps\litert\litert_cc_sdk
set LITERT_LIB_DIR=%ROOT%\deps\litert\liteRT_runtime\android_arm64
set LITERT_LIB_PATH=%LITERT_LIB_DIR%\libLiteRt.so

set "BUILD_DIR=%ROOT%\build\android-arm64"

REM ---- Clear stale CMake cache (generator mismatch) ----
if exist "%BUILD_DIR%\CMakeCache.txt" (
    echo Removing stale CMake cache from previous generator...
    del /Q "%BUILD_DIR%\CMakeCache.txt" 2>nul
    rmdir /S /Q "%BUILD_DIR%\CMakeFiles" 2>nul
)

REM ---- Add NDK prebuilt + VS bundled CMake/Ninja to PATH ----
set "PATH=%ANDROID_NDK_ROOT%\prebuilt\windows-x86_64\bin;%PATH%"
for /f "usebackq tokens=*" %%i in (
    `"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -prerelease -property installationPath`
) do set "PATH=%%i\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja;%PATH%"
set "PATH=%LOCALAPPDATA%\Android\Sdk\cmake\3.22.1\bin;%PATH%"

REM ---- Detect build tool (prefer Ninja for speed) ----
where ninja >nul 2>nul
if errorlevel 1 (
    set "CMAKE_GEN=Unix Makefiles"
    echo Generator: Unix Makefiles
) else (
    set "CMAKE_GEN=Ninja"
    echo Generator: Ninja
)

echo ============================================================
echo  Configuring (CMake + NDK)...
echo ============================================================
cmake -S "%ROOT%" -B "%BUILD_DIR%" -G "%CMAKE_GEN%" ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DCMAKE_TOOLCHAIN_FILE="%ANDROID_NDK_ROOT%\build\cmake\android.toolchain.cmake" ^
    -DANDROID_ABI=arm64-v8a ^
    -DANDROID_PLATFORM=21 ^
    -DANDROID_STL=c++_static ^
    -DHAVE_ONNX_BACKEND=ON ^
    -DHAVE_TFLITE_BACKEND=ON ^
    -DHAVE_NCNN_BACKEND=ON ^
    -DHAVE_MNN_BACKEND=ON ^
    -DHAVE_LITERT_BACKEND=ON
if errorlevel 1 (
    echo CMake configuration failed.
    exit /b 1
)

echo.
echo ============================================================
echo  Building %OUT% (CMake + Ninja, incremental build)...
echo ============================================================
cmake --build "%BUILD_DIR%" --verbose
if errorlevel 1 (
    echo.
    echo ============================================================
    echo  BUILD FAILED
    echo ============================================================
    exit /b 1
)

echo.
echo ============================================================
echo  BUILD SUCCESS: %BUILD_DIR%\unified_bench
echo ============================================================

REM --- Push and run ---
adb devices | findstr "device$" >nul
if errorlevel 1 (
    echo No device. Build only.
    goto :eof
)

echo Pushing to device...
adb wait-for-device
adb shell "mkdir -p /data/local/tmp/bench_test/qnn 2>/dev/null"
adb push "%BUILD_DIR%\unified_bench" /data/local/tmp/bench_test/
adb push "%ROOT%\test_model.onnx" /data/local/tmp/bench_test/ 2>nul
adb push "%ROOT%\test_model.tflite" /data/local/tmp/bench_test/ 2>nul
adb push "%ROOT%\test_model.ncnn.param" /data/local/tmp/bench_test/ 2>nul
adb push "%ROOT%\test_model.ncnn.bin"   /data/local/tmp/bench_test/ 2>nul
adb push "%ROOT%\test_model_fp16.ncnn.bin" /data/local/tmp/bench_test/ 2>nul
adb push "%ROOT%\test_model.shapes" /data/local/tmp/bench_test/ 2>nul
adb push "%ROOT%\test_model.mnn" /data/local/tmp/bench_test/ 2>nul

REM --- Push .so files (skip if already on device) ---
adb shell "test -f /data/local/tmp/bench_test/libonnxruntime.so" >nul 2>&1
if errorlevel 1 (
    echo Pushing ONNX Runtime .so...
    adb push "%ONNX_LIB%\libonnxruntime.so" /data/local/tmp/bench_test/ || (
        echo ERROR: Failed to push ONNX Runtime .so
        exit /b 1
    )
) else ( echo ONNX Runtime .so already on device, skip. )

if exist "%QNN_LIB%\libonnxruntime.so" (
    adb shell "test -f /data/local/tmp/bench_test/qnn/libQnnHtp.so" >nul 2>&1
    if errorlevel 1 (
        echo Pushing QNN ONNX Runtime + backend libs...
        adb push "%QNN_LIB%\libonnxruntime.so"     /data/local/tmp/bench_test/qnn/ || (echo WARNING: QNN .so push failed)
        adb push "%QNN_LIB%\libQnnCpu.so"          /data/local/tmp/bench_test/qnn/ 2>nul
        adb push "%QNN_LIB%\libQnnGpu.so"          /data/local/tmp/bench_test/qnn/ 2>nul
        adb push "%QNN_LIB%\libQnnHtp.so"          /data/local/tmp/bench_test/qnn/ 2>nul
        adb push "%QNN_LIB%\libQnnHtpPrepare.so"   /data/local/tmp/bench_test/qnn/ 2>nul
        adb push "%QNN_LIB%\libQnnSaver.so"        /data/local/tmp/bench_test/qnn/ 2>nul
        adb push "%QNN_LIB%\libQnnSystem.so"       /data/local/tmp/bench_test/qnn/ 2>nul
    ) else ( echo QNN libs already on device, skip. )

    REM Auto-detect SOC → hexagon version → push matching stub + DSP skeleton libs
    for /f "usebackq tokens=*" %%i in (`adb shell getprop ro.soc.model 2^>nul`) do set "SOC=%%i"
    if defined SOC (
        set "SOC=!SOC:[=!"
        set "SOC=!SOC:]=!"
        for /f "tokens=*" %%a in ("!SOC!") do set "SOC=%%a"
        echo Device SOC: !SOC!

        set HEXVER=
        if /i "!SOC!"=="SM8850" set HEXVER=hexagon-v81
        if /i "!SOC!"=="SM8750" set HEXVER=hexagon-v79
        if /i "!SOC!"=="SM8650" set HEXVER=hexagon-v75
        if /i "!SOC!"=="SM7750" set HEXVER=hexagon-v73
        if /i "!SOC!"=="SM8550" set HEXVER=hexagon-v73
        if /i "!SOC!"=="SC8380XP" set HEXVER=hexagon-v73
        if /i "!SOC!"=="SM7635" set HEXVER=hexagon-v73
        if /i "!SOC!"=="SM8475" set HEXVER=hexagon-v69
        if /i "!SOC!"=="SM8450" set HEXVER=hexagon-v69
        if /i "!SOC!"=="SM7450" set HEXVER=hexagon-v69
        if /i "!SOC!"=="SC8280X" set HEXVER=hexagon-v68
        if /i "!SOC!"=="SC7280X" set HEXVER=hexagon-v68
        if /i "!SOC!"=="SM8350P" set HEXVER=hexagon-v68
        if /i "!SOC!"=="SM8350" set HEXVER=hexagon-v68
        if /i "!SOC!"=="SM7325" set HEXVER=hexagon-v68
        if /i "!SOC!"=="QCM6490" set HEXVER=hexagon-v68

        if defined HEXVER (
            REM Derive directory name (e.g. v73) and lib version (e.g. V73) from hexagon-v73
            set "HEX_SUBDIR=!HEXVER:hexagon-=!"
            set "HEXLIBVER=!HEXVER:hexagon-v=V!"
            set QNN_HEX_SDK=%ROOT%\deps\onnxruntime\lib\android\qnn\hexagon\!HEX_SUBDIR!
            if exist "!QNN_HEX_SDK!\libQnnHtp!HEXLIBVER!.so" (
                echo Pushing hexagon DSP libs for !HEXVER!...
                adb shell "mkdir -p /data/local/tmp/bench_test/qnn/hexagon 2>/dev/null"
                REM Push the matching stub (only one, detected by SOC)
                adb push "%QNN_LIB%\libQnnHtp!HEXLIBVER!Stub.so"    /data/local/tmp/bench_test/qnn/ 2>nul
                REM Push hexagon DSP skeleton libs
                adb push "!QNN_HEX_SDK!\libQnnHtp!HEXLIBVER!.so"      /data/local/tmp/bench_test/qnn/hexagon/ 2>nul
                adb push "!QNN_HEX_SDK!\libQnnHtp!HEXLIBVER!Skel.so"  /data/local/tmp/bench_test/qnn/hexagon/ 2>nul
                adb push "!QNN_HEX_SDK!\libQnnSaver.so"               /data/local/tmp/bench_test/qnn/hexagon/ 2>nul
                adb push "!QNN_HEX_SDK!\libQnnSystem.so"              /data/local/tmp/bench_test/qnn/hexagon/ 2>nul
            ) else ( echo WARNING: Hexagon libs not found at !QNN_HEX_SDK! )
        ) else ( echo WARNING: Unknown SOC '!SOC!', skip hexagon DSP libs. )
    ) else ( echo WARNING: Could not detect SOC, skip hexagon DSP libs. )
) else (
    echo WARNING: QNN SDK not found, QNN backends will be unavailable
)

adb shell "test -f /data/local/tmp/bench_test/libncnn.so" >nul 2>&1
if errorlevel 1 (
    echo Pushing NCNN .so...
    adb push "%NCNN_LIB%\libncnn.so" /data/local/tmp/bench_test/ || exit /b 1
) else ( echo NCNN .so already on device, skip. )

adb shell "test -f /data/local/tmp/bench_test/libtensorflowlite_c.so" >nul 2>&1
if errorlevel 1 (
    echo Pushing TFLite .so...
    adb push "%TFLITE_LIB%\libtensorflowlite_c.so" /data/local/tmp/bench_test/ || exit /b 1
    adb push "%TFLITE_LIB%\libtensorflowlite_gpu_delegate.so" /data/local/tmp/bench_test/ || exit /b 1
    if exist "%TFLITE_LIB%\libtensorflowlite_flex.so" (
        echo Pushing TFLite Flex delegate...
        adb push "%TFLITE_LIB%\libtensorflowlite_flex.so" /data/local/tmp/bench_test/ || exit /b 1
    )
) else ( echo TFLite .so already on device, skip. )

REM --- Push MNN libs ---
adb shell "test -f /data/local/tmp/bench_test/libMNN.so" >nul 2>&1
if errorlevel 1 (
    echo Pushing MNN .so files...
    adb push "%MNN_LIB%\libMNN.so" /data/local/tmp/bench_test/ || exit /b 1
    adb push "%MNN_LIB%\libMNN_CL.so" /data/local/tmp/bench_test/ 2>nul
    adb push "%MNN_LIB%\libMNN_Vulkan.so" /data/local/tmp/bench_test/ 2>nul
    adb push "%MNN_LIB%\libc++_shared.so" /data/local/tmp/bench_test/ 2>nul
) else ( echo MNN .so already on device, skip. )

REM --- Push LiteRT libs ---
adb shell "test -f /data/local/tmp/bench_test/libLiteRt.so" >nul 2>&1
if errorlevel 1 (
    echo Pushing LiteRT .so files...
    if exist "%LITERT_LIB_DIR%\libLiteRt.so" (
        adb push "%LITERT_LIB_DIR%\libLiteRt.so" /data/local/tmp/bench_test/ || exit /b 1
    )
    if exist "%LITERT_LIB_DIR%\libLiteRtClGlAccelerator.so" (
        adb push "%LITERT_LIB_DIR%\libLiteRtClGlAccelerator.so" /data/local/tmp/bench_test/ 2>nul
    )
) else ( echo LiteRT .so already on device, skip. )

REM --- Push LiteRT NPU dispatch library to qnn/ dir (reuses SOC/HEXVER from QNN section above) ---
if defined HEXVER (
    set "DISPATCH_VER=!HEXVER:hexagon-=!"
    set "LITERT_NPU_ROOT=%ROOT%\deps\litert\litert_npu_runtime_libraries"
    set "DISPATCH_SRC=!LITERT_NPU_ROOT!\qualcomm_runtime_!DISPATCH_VER!\src\main\jni\arm64-v8a\libLiteRtDispatch_Qualcomm.so"
    if exist "!DISPATCH_SRC!" (
        echo Pushing LiteRT Qualcomm dispatch for !DISPATCH_VER!...
        adb push "!DISPATCH_SRC!" /data/local/tmp/bench_test/qnn/ || echo WARNING: Dispatch push failed
    ) else (
        echo WARNING: Dispatch library not found: !DISPATCH_SRC!
    )
) else (
    echo WARNING: HEXVER not set, skip LiteRT NPU dispatch push.
)

echo.

echo.
echo ============================================================
echo  Running FULL benchmark (warmup=5, repeat=100)...
echo ============================================================
adb shell "rm -f /data/local/tmp/bench_test/summary.csv"
adb shell "cd /data/local/tmp/bench_test && chmod +x ./%OUT% && LD_LIBRARY_PATH=.:./qnn ADSP_LIBRARY_PATH=./qnn/hexagon ./%OUT% test_model.onnx"
set BENCH_EXIT=%ERRORLEVEL%

echo.
echo ============================================================
echo  Benchmark exit code: %BENCH_EXIT%
echo ============================================================

echo Pulling results (appending data rows only)...
adb shell "tail -n +2 /data/local/tmp/bench_test/summary.csv" >> "%ROOT%\summary.csv"

echo.
echo ============================================================
echo  DONE.
echo ============================================================

endlocal
