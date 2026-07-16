@echo off
cls
setlocal enabledelayedexpansion

REM ============================================================
REM  Unified Benchmark Tool - Windows x64 build (CMake + MSVC)
REM  Requires: Visual Studio 2019+ with C++ tools, CMake >= 3.16
REM ============================================================

REM ---- Locate Visual Studio ----
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" (
    echo ERROR: vswhere.exe not found. Install Visual Studio.
    exit /b 1
)
for /f "usebackq tokens=*" %%i in (
    `"%VSWHERE%" -latest -products * -prerelease ^
     -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 ^
     -property installationPath`
) do call "%%i\VC\Auxiliary\Build\vcvars64.bat"

REM ---- Project root ----
set "PROJ_ROOT=%~dp0.."
set "BUILD_DIR=%PROJ_ROOT%\build\win-x64"

REM ---- Generate LiteRT import library if .dll exists but .lib missing ----
set "LITERT_LIB_DIR=%PROJ_ROOT%\deps\litert\LiteRT_runtime\windows_x86_64"
set "LITERT_LIB=%LITERT_LIB_DIR%\libLiteRt.lib"
set "LITERT_DLL=%LITERT_LIB_DIR%\libLiteRt.dll"
if exist "%LITERT_DLL%" (
    if not exist "%LITERT_LIB%" (
        echo Generating LiteRT import library...
        powershell -ExecutionPolicy Bypass -File "%~dp0gen_litert_lib.ps1"
        if errorlevel 1 (
            echo WARNING: LiteRT import library generation failed.
        )
    )
)

REM ---- Detect Visual Studio version for CMake generator ----
for /f "usebackq tokens=*" %%v in (
    `"%VSWHERE%" -latest -products * -prerelease -property catalog_productLineVersion`
) do set "VS_VER=%%v"
if not defined VS_VER (
    echo ERROR: Could not detect VS version.
    exit /b 1
)
echo Detected Visual Studio %VS_VER%

REM ---- Clear stale CMake cache (generator mismatch on version change) ----
if exist "%BUILD_DIR%\CMakeCache.txt" (
    echo Removing stale CMake cache from previous generator...
    del /Q "%BUILD_DIR%\CMakeCache.txt" 2>nul
    rmdir /S /Q "%BUILD_DIR%\CMakeFiles" 2>nul
)

echo ============================================================
echo  Configuring (CMake)...
echo ============================================================
cmake -S "%PROJ_ROOT%" -B "%BUILD_DIR%" ^
    -G "Visual Studio %VS_VER%" -A x64 ^
    -DCMAKE_BUILD_TYPE=Release ^
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

set "OUT_PATH=%BUILD_DIR%\Release\unified_bench_win_x64.exe"

echo.
echo ============================================================
echo  BUILD SUCCESS
echo  %OUT_PATH%
echo ============================================================

REM Note: ONNX Runtime DLLs are loaded dynamically from deps/onnxruntime/lib/win-x64/<ep>/
REM Dependency DLLs (ncnn, TFLite, MNN, LiteRT) are copied by CMake post-build.

REM ---- Copy dxil.dll + dxcompiler.dll for WebGPU (needed by LiteRT GPU backend) ----
call :copy_dxil "%BUILD_DIR%\Release"

"%OUT_PATH%" --help

echo.
echo ============================================================
echo  Quick rebuild (only changed files):
echo    cmake --build "%BUILD_DIR%" --config Release
echo ============================================================
goto :eof

REM ---- helper: copy WebGPU D3D shader DLLs ----
:copy_dxil
if exist "%~1\dxil.dll" if exist "%~1\dxcompiler.dll" echo WebGPU DLLs already present. & goto :eof
set "D3D_REDIST=%ProgramFiles(x86)%\Windows Kits\10\Redist\D3D\x64"
if exist "!D3D_REDIST!\dxil.dll" (
    copy /Y "!D3D_REDIST!\dxil.dll" "%~1\" >nul
    copy /Y "!D3D_REDIST!\dxcompiler.dll" "%~1\" >nul
    echo Copied WebGPU DLLs to %~1 & goto :eof
)
set "D3D_SDK=%ProgramFiles(x86)%\Windows Kits\10\bin\10.0.26100.0\x64"
if exist "!D3D_SDK!\dxil.dll" (
    copy /Y "!D3D_SDK!\dxil.dll" "%~1\" >nul
    copy /Y "!D3D_SDK!\dxcompiler.dll" "%~1\" >nul
    echo Copied WebGPU DLLs to %~1 & goto :eof
)
echo WARNING: dxil.dll/dxcompiler.dll not found - LiteRT GPU (WebGPU) backend will not work
goto :eof

endlocal
