@echo off
setlocal

set "ROOT=%~dp0"
pushd "%ROOT%"

if exist "build" (
    echo Removing build/ ...
    rmdir /s /q "build"
) else (
    echo No build/ directory to remove.
)

if exist "CMakeUserPresets.json" (
    echo Removing CMakeUserPresets.json ...
    del /q "CMakeUserPresets.json"
)

popd
echo Done.
endlocal
