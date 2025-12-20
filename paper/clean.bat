@echo off
REM Clean build artifacts for VSAX paper

echo Cleaning build files...

REM Remove build directory
if exist "build" (
    rmdir /s /q build
    echo Removed build directory
)

REM Remove PDF from main directory
if exist "vsax_paper.pdf" (
    del vsax_paper.pdf
    echo Removed vsax_paper.pdf
)

echo.
echo Cleanup complete!
