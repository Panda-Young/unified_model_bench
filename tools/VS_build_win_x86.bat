@echo off
setlocal enabledelayedexpansion
REM ============================================================
REM  Unified Benchmark - Visual Studio x86 build
REM  Supported: ONNX, NCNN  (TFLite/MNN not available on x86)
REM ============================================================

REM ---- Locate Visual Studio ----
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" (echo ERROR: vswhere.exe not found & exit /b 1)
for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -prerelease -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSROOT=%%i"
if not defined VSROOT (echo ERROR: VS not found & exit /b 1)
call "%VSROOT%\VC\Auxiliary\Build\vcvars32.bat"

REM ---- Paths ----
set "PROJ_ROOT=%~dp0.."
set "INC=%PROJ_ROOT%\include"
set "SRC=%PROJ_ROOT%\src"

REM ONNX Runtime
set "ONNX_INC=%~dp0..\deps\onnxruntime\include"
set "ONNX_ROOT=%~dp0..\deps\onnxruntime"

REM TensorFlow Lite (x86 .lib not available - TFLite skipped)
set "TFLITE_ROOT=%~dp0..\deps\tflite"
set "TFLITE_INC=%TFLITE_ROOT%\include"

REM ncnn
set "NCNN_ROOT=%~dp0..\deps\ncnn"
set "NCNN_INC=%NCNN_ROOT%\include"
set "NCNN_LIB=%NCNN_ROOT%\lib\win-x86\ncnn.lib"

set "OUT_NAME=unified_bench_win_x86.exe"
set "OUT_PATH=%PROJ_ROOT%\%OUT_NAME%"

del "%OUT_PATH%" 2>nul

echo ============================================================
echo  Building %OUT_NAME% (x86 - ONNX + NCNN, no TFLite)
echo ============================================================

REM ---- Compile (via response file to avoid command-line length limit) ----
set "RSP=%PROJ_ROOT%\build_x86.rsp"
(
echo /EHa /std:c++17 /utf-8 /O2 /DNDEBUG /W3 /wd4251 /wd4273 /D_CRT_SECURE_NO_WARNINGS /DNOMINMAX
echo /DHAVE_ONNX_BACKEND
echo /DHAVE_NCNN_BACKEND
echo /I"%INC%"
echo /I"%ONNX_INC%"
echo /I"%TFLITE_INC%"
echo /I"%NCNN_INC%"
echo "%SRC%\log.cpp"
echo "%SRC%\cmd_args.cpp"
echo "%SRC%\device_info.cpp"
echo "%SRC%\file_ops.cpp"
echo "%SRC%\model_loader.cpp"
echo "%SRC%\backend_registry.cpp"
echo "%SRC%\onnx_backend.cpp"
echo "%SRC%\ncnn_backend.cpp"
echo "%SRC%\input_provider.cpp"
echo "%SRC%\result_collector.cpp"
echo "%SRC%\benchmark_runner.cpp"
echo "%SRC%\main.cpp"
echo /link "%NCNN_LIB%" Advapi32.lib /OUT:"%OUT_PATH%"
) > "%RSP%"

cl @"%RSP%"
set BUILD_ERR=%ERRORLEVEL%
del "%RSP%" 2>nul
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
echo  BUILD SUCCESS: %OUT_PATH%
echo ============================================================

REM ---- Copy runtime DLLs to exe directory (non-ORT backends only) ----
echo Copying dependency DLLs...
copy /Y "%NCNN_ROOT%\lib\win-x86\ncnn.dll" "%PROJ_ROOT%\" 2>nul
echo Done.
REM Note: ONNX Runtime DLLs are loaded directly from deps/onnxruntime/lib/win-x86/<ep>/

"%OUT_PATH%" --help

endlocal
