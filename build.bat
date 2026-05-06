@echo off
setlocal

set "ROOT=%~dp0"
pushd "%ROOT%"

echo === Configuring (preset: default) ===
cmake --preset default
if errorlevel 1 goto :fail

echo.
echo === Building (Release) ===
cmake --build --preset default
if errorlevel 1 goto :fail

echo.
echo === Done ===
echo   build\Release\collision_benchmark.exe
echo   build\visualization\raylib-cuda-interop\Release\raylib_cuda_visualizer.exe
popd
endlocal
exit /b 0

:fail
echo.
echo Build failed.
popd
endlocal
exit /b 1
