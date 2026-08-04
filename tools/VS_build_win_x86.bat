@echo off
setlocal enabledelayedexpansion
REM ============================================================
REM  Unified Benchmark - Windows x86 build (CMake + MSVC)
REM  Supported: ONNX, NCNN  (TFLite/MNN/LiteRT not available on x86)
REM  Requires: Visual Studio 2019+, CMake >= 3.16
REM ============================================================

REM ---- Locate Visual Studio ----
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" (echo ERROR: vswhere.exe not found & exit /b 1)
for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -prerelease -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSROOT=%%i"
if not defined VSROOT (echo ERROR: VS not found & exit /b 1)
call "%VSROOT%\VC\Auxiliary\Build\vcvars32.bat"

REM ---- Project root ----
set "PROJ_ROOT=%~dp0.."
set "BUILD_DIR=%PROJ_ROOT%\build\win-x86"

REM ---- Detect Visual Studio version for CMake generator ----
for /f "usebackq tokens=*" %%v in (
    `"%VSWHERE%" -latest -products * -prerelease -property catalog_productLineVersion`
) do set "VS_VER=%%v"
if not defined VS_VER (
    echo ERROR: Could not detect VS version.
    exit /b 1
)
echo Detected Visual Studio %VS_VER%

REM ---- Optional clean rebuild: "VS_build_win_x86.bat clean|rebuild" ----
set "CLEAN_BUILD=0"
if /i "%~1"=="clean" set "CLEAN_BUILD=1"
if /i "%~1"=="rebuild" set "CLEAN_BUILD=1"
if "%CLEAN_BUILD%"=="1" (
    if exist "%BUILD_DIR%" (
        echo [clean] Removing build dir for full rebuild...
        rmdir /S /Q "%BUILD_DIR%"
    )
)

echo ============================================================
echo  Configuring (CMake) - x86 with ONNX + NCNN backends
 echo  (incremental; use "%~nx0 clean" for full rebuild)
echo ============================================================
cmake -S "%PROJ_ROOT%" -B "%BUILD_DIR%" ^
    -G "Visual Studio %VS_VER%" -A Win32 ^
    -DHAVE_ONNX_BACKEND=ON ^
    -DHAVE_TFLITE_BACKEND=OFF ^
    -DHAVE_NCNN_BACKEND=ON ^
    -DHAVE_MNN_BACKEND=OFF ^
    -DHAVE_LITERT_BACKEND=OFF
if errorlevel 1 (
    REM Likely a generator mismatch with an existing cache -> clean and retry once
    echo CMake configure failed; cleaning build dir and retrying...
    rmdir /S /Q "%BUILD_DIR%"
    cmake -S "%PROJ_ROOT%" -B "%BUILD_DIR%" ^
        -G "Visual Studio %VS_VER%" -A Win32 ^
        -DHAVE_ONNX_BACKEND=ON ^
        -DHAVE_TFLITE_BACKEND=OFF ^
        -DHAVE_NCNN_BACKEND=ON ^
        -DHAVE_MNN_BACKEND=OFF ^
        -DHAVE_LITERT_BACKEND=OFF
    if errorlevel 1 (
        echo CMake configuration failed.
        exit /b 1
    )
)

echo.
echo ============================================================
echo  Building (CMake)...
echo ============================================================
cmake --build "%BUILD_DIR%" --config Release --verbose
if errorlevel 1 (
    echo.
    echo ============================================================
    echo  BUILD FAILED
    echo ============================================================
    exit /b 1
)

set "OUT_PATH=%BUILD_DIR%\Release\unified_bench.exe"

echo.
echo ============================================================
echo  BUILD SUCCESS: %OUT_PATH%
echo ============================================================

REM Note: ONNX Runtime DLLs are loaded dynamically from deps/onnxruntime/lib/win-x86/<ep>/
REM Dependency DLLs (ncnn) are copied by CMake post-build.

"%OUT_PATH%" --help

echo.
echo ============================================================
echo  Quick rebuild (only changed files):
echo    cmake --build "%BUILD_DIR%" --config Release
echo ============================================================

endlocal
