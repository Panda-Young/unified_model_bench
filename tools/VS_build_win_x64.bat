@echo off
cls
setlocal enabledelayedexpansion

REM ============================================================
REM  Unified Benchmark Tool - Windows x64 build (MSVC)
REM  Requires: Visual Studio 2019+ with C++ tools
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

REM ---- Paths ----
set "PROJ_ROOT=%~dp0.."
set "INC=%PROJ_ROOT%\include"
set "SRC=%PROJ_ROOT%\src"

REM ONNX Runtime (header-only, loaded dynamically at runtime)
set "ONNX_INC=%~dp0..\deps\onnxruntime\include"
set "ONNX_ROOT=%~dp0..\deps\onnxruntime"

REM TensorFlow Lite
set "TFLITE_ROOT=%~dp0..\deps\tflite"
set "TFLITE_INC=%TFLITE_ROOT%\include"
set "TFLITE_LIB=%TFLITE_ROOT%\lib\win-x64\tensorflowlite_c.dll.if.lib"

REM ncnn
set "NCNN_ROOT=%~dp0..\deps\ncnn"
set "NCNN_INC=%NCNN_ROOT%\include"
set "NCNN_LIB=%NCNN_ROOT%\lib\win-x64\ncnn.lib"

REM MNN
set "MNN_ROOT=%~dp0..\deps\mnn"
set "MNN_INC=%MNN_ROOT%\include"
set "MNN_LIB=%MNN_ROOT%\lib\win-x64\MNN.lib"
set "MNN_DLL=%MNN_ROOT%\lib\win-x64\MNN.dll"

REM LiteRT (Google next-gen TFLite runtime)
set "LITERT_ROOT=%~dp0..\deps\litert"
set "LITERT_INC=%LITERT_ROOT%\litert_cc_sdk"
set "LITERT_LIB_DIR=%LITERT_ROOT%\LiteRT_runtime\windows_x86_64"
set "LITERT_LIB=%LITERT_LIB_DIR%\libLiteRt.lib"
set "LITERT_DLL=%LITERT_LIB_DIR%\libLiteRt.dll"

set "OUT_NAME=unified_bench_win_x64.exe"
set "OUT_PATH=%PROJ_ROOT%\%OUT_NAME%"

del "%OUT_PATH%" 2>nul

echo ============================================================
echo  Building %OUT_NAME%
echo ============================================================

REM ---- Generate LiteRT import library if needed ----
if exist "%LITERT_DLL%" (
    if not exist "%LITERT_LIB%" (
        echo Generating LiteRT import library...
        powershell -ExecutionPolicy Bypass -File "%~dp0gen_litert_lib.ps1"
        if errorlevel 1 (
            echo WARNING: LiteRT import library generation failed, building without LiteRT.
            set "LITERT_LIB="
        )
    )
)

REM ---- Compile ----
cl /EHa /std:c++17 /utf-8 /O2 /DNDEBUG /W3 /arch:AVX2 /wd4251 /wd4273 /D_CRT_SECURE_NO_WARNINGS /D_CRT_NONSTDC_NO_DEPRECATE /DNOMINMAX ^
    /DHAVE_ONNX_BACKEND ^
    /DHAVE_TFLITE_BACKEND ^
    /DHAVE_NCNN_BACKEND ^
    /DHAVE_MNN_BACKEND ^
    /DHAVE_LITERT_BACKEND ^
    /I"%INC%" ^
    /I"%ONNX_INC%" ^
    /I"%TFLITE_INC%" ^
    /I"%NCNN_INC%" ^
    /I"%MNN_INC%" ^
    /I"%LITERT_INC%" ^
    "%SRC%\log.cpp" ^
    "%SRC%\cmd_args.cpp" ^
    "%SRC%\device_info.cpp" ^
    "%SRC%\file_ops.cpp" ^
    "%SRC%\model_loader.cpp" ^
    "%SRC%\backend_registry.cpp" ^
    "%SRC%\onnx_backend.cpp" ^
    "%SRC%\tflite_backend.cpp" ^
    "%SRC%\ncnn_backend.cpp" ^
    "%SRC%\mnn_backend.cpp" ^
    "%SRC%\litert_backend.cpp" ^
    "%SRC%\input_provider.cpp" ^
    "%SRC%\result_collector.cpp" ^
    "%SRC%\benchmark_runner.cpp" ^
    "%SRC%\main.cpp" ^
    /link "%TFLITE_LIB%" "%NCNN_LIB%" "%MNN_LIB%" %LITERT_LIB% Advapi32.lib /OUT:"%OUT_PATH%"
set BUILD_ERR=%ERRORLEVEL%
del *.obj 2>nul

if %BUILD_ERR% neq 0 (
    echo.
    echo ============================================================
    echo  BUILD FAILED
    echo ============================================================
    exit /b 1
)

echo.
echo ============================================================
echo  BUILD SUCCESS
echo  %OUT_PATH%
echo ============================================================

REM ---- Copy runtime DLLs to exe directory ----
echo Copying dependency DLLs...
copy /Y "%NCNN_ROOT%\lib\win-x64\ncnn.dll"              "%PROJ_ROOT%\" 2>nul
copy /Y "%TFLITE_ROOT%\lib\win-x64\tensorflowlite_c.dll" "%PROJ_ROOT%\" 2>nul
copy /Y "%MNN_DLL%"                                       "%PROJ_ROOT%\" 2>nul
if exist "%LITERT_DLL%" (
    copy /Y "%LITERT_DLL%" "%PROJ_ROOT%\" 2>nul
    if exist "%LITERT_LIB_DIR%\libLiteRtWebGpuAccelerator.dll" (
        copy /Y "%LITERT_LIB_DIR%\libLiteRtWebGpuAccelerator.dll" "%PROJ_ROOT%\" 2>nul
    )
)

echo Done.
REM Note: ONNX Runtime DLLs are loaded directly from deps/onnxruntime/lib/win-x64/<ep>/

REM ---- Copy dxil.dll + dxcompiler.dll for WebGPU (needed by LiteRT GPU backend) ----
call :copy_dxil "%PROJ_ROOT%"

"%OUT_PATH%" --help
goto :eof

REM ---- helper: copy WebGPU D3D shader DLLs ----
:copy_dxil
if exist "%~1\dxil.dll" if exist "%~1\dxcompiler.dll" echo WebGPU DLLs already present. & goto :eof
set "D3D_REDIST=%ProgramFiles(x86)%\Windows Kits\10\Redist\D3D\x64"
if exist "!D3D_REDIST!\dxil.dll" (
    copy /Y "!D3D_REDIST!\dxil.dll" "%~1\" >nul
    copy /Y "!D3D_REDIST!\dxcompiler.dll" "%~1\" >nul
    echo Copied WebGPU DLLs from Windows Kits Redist & goto :eof
)
set "D3D_SDK=%ProgramFiles(x86)%\Windows Kits\10\bin\10.0.26100.0\x64"
if exist "!D3D_SDK!\dxil.dll" (
    copy /Y "!D3D_SDK!\dxil.dll" "%~1\" >nul
    copy /Y "!D3D_SDK!\dxcompiler.dll" "%~1\" >nul
    echo Copied WebGPU DLLs from Windows Kits SDK & goto :eof
)
echo WARNING: dxil.dll/dxcompiler.dll not found - LiteRT GPU (WebGPU) backend will not work
goto :eof

endlocal
