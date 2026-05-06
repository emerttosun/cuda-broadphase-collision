@echo off
setlocal

set "ROOT=%~dp0"
pushd "%ROOT%"

echo === Configuring (preset: benchmark-only) ===
cmake --preset benchmark-only
if errorlevel 1 goto :fail

echo.
echo === Building (Release) ===
cmake --build --preset benchmark-only
if errorlevel 1 goto :fail

echo.
echo === Done ===
echo   build\Release\collision_benchmark.exe
popd
endlocal
exit /b 0

:fail
echo.
echo Build failed.
popd
endlocal
exit /b 1
