@echo off
setlocal enabledelayedexpansion

REM ============================================================
REM  Unified Benchmark - Capture ONNX QNN HTP partition blockers
REM
REM  PURPOSE
REM    Diagnose why `ONNX_QNN_HTP` fails with:
REM      "This session contains graph nodes that are assigned to the
REM       default CPU EP, but fallback to CPU EP has been explicitly
REM       disabled by the user."
REM
REM    ORT does a whole-graph partition before it decides whether to
REM    fall back, so at VERBOSE log level it prints *all* nodes it could
REM    not place on the QNN EP (plus the reason), even though CreateSession
REM    later aborts. This script captures that and extracts a blocklist.
REM
REM  Usage: capture_qnn_blockers.bat [<model_name>] [--trust-device]
REM
REM    <model_name>    Name as deployed under /data/local/tmp/bench_test/
REM                    (default: test_model.onnx)
REM    --trust-device  Skip re-pushing model + QNN .so (assume what's already
REM                    on device is the build you want to capture)
REM
REM  Output (in the project build dir):
REM    build\qnn_capture_debug.log   raw VERBOSE ORT/benchmark output
REM    build\qnn_blockers.txt        extracted QNN fallback/unsupported lines
REM ============================================================

set "ROOT=%~dp0.."
set "DEST=/data/local/tmp/bench_test"
set "MODEL_NAME=test_model.onnx"
set "TRUST_DEVICE=0"

:parse
if "%~1"=="" goto :parsed
if /i "%~1"=="--trust-device" ( set "TRUST_DEVICE=1" ) & shift & goto :parse
set "MODEL_NAME=%~1"
shift
goto :parse
:parsed

REM ---- Require adb + device ----
adb get-state 2>nul >nul
if errorlevel 1 (
    echo ERROR: No connected Android device. Plug in the device and retry.
    exit /b 1
)

echo ============================================================
echo  Capturing ONNX QNN HTP partition blockers ...
echo  model      : %MODEL_NAME%
echo  device dir : %DEST%
echo ============================================================

if "%TRUST_DEVICE%"=="1" goto :ensure_libs

REM ---- Ensure the model is on device (re-push if missing) ----
adb shell "test -f %DEST%/%MODEL_NAME%" >nul 2>&1
if errorlevel 1 (
    echo Model %MODEL_NAME% not on device - looking for a local copy...
    if exist "%ROOT%\%MODEL_NAME%" (
        adb push "%ROOT%\%MODEL_NAME%" %DEST%/ >nul || ( echo ERROR: model push failed & exit /b 1 )
    ) else (
        echo ERROR: %MODEL_NAME% not found in %ROOT% and not on device.
        echo   Push it yourself with:
        echo     adb push D:\your\path\%MODEL_NAME% %DEST%/
        exit /b 1
    )
)

:ensure_libs
REM ---- Ensure the QNN ORT runtime (.so) is on device ----
adb shell "test -f %DEST%/qnn/libonnxruntime.so" >nul 2>&1
if errorlevel 1 (
    if exist "%ROOT%\deps\onnxruntime\lib\android\qnn\arm64-v8a\libonnxruntime.so" (
        echo Pushing QNN ORT runtime...
        adb push "%ROOT%\deps\onnxruntime\lib\android\qnn\arm64-v8a\libonnxruntime.so" %DEST%/qnn/ >nul || echo WARNING: QNN ORT .so push failed
    ) else (
        echo WARNING: local QNN libonnxruntime.so not found; using whatever is on device.
    )
)

set "RAW=%ROOT%\build\qnn_capture_debug.log"
set "SUM=%ROOT%\build\qnn_blockers.txt"
if not exist "%ROOT%\build" mkdir "%ROOT%\build"
if exist "%RAW%" del /Q "%RAW%"
if exist "%SUM%" del /Q "%SUM%"

echo.
echo Running on-device ONNX-QNN-HTP at --log-level debug ...
echo   (prints ORT's full QNN partition, incl. the nodes it cannot place)
adb shell "cd %DEST% && LD_LIBRARY_PATH=.:./qnn ADSP_LIBRARY_PATH=./qnn ./unified_bench %MODEL_NAME% --log-level debug 2>&1" > "%RAW%"

echo.
echo Capture written: %RAW%
echo.
echo --- Extracting ORT QNN fallback / unsupported-node lines ---
findstr /i /c:"QNN" /c:"not supported" /c:"fallback" /c:"assigned to ep" /c:"partition" "%RAW%" > "%SUM%"
if errorlevel 1 (
    echo [warn] No QNN/fallback markers matched in the raw log.
    echo        Check %RAW% directly for ORT's verbose partition messages.
) else (
    echo.
    type "%SUM%"
)

echo.
echo ============================================================
echo  DONE.  Raw log : %RAW%
echo  Blockers      : %SUM%
echo ============================================================
echo.
echo Next: paste %SUM% (or search %RAW% for "not supported" /
echo "assigned to ep of type" / "QNN") and we will fix the exact op(s).

endlocal