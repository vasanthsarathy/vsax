@echo off
REM Build script for VSAX paper on Windows
REM Requires latexmk to be installed

echo Building VSAX paper...
echo.

REM Create build directory if it doesn't exist
if not exist "build" mkdir build

REM Compile with latexmk, putting aux files in build/
latexmk -pdf -output-directory=build -interaction=nonstopmode vsax_paper.tex

REM Copy the PDF to the main directory
if exist "build\vsax_paper.pdf" (
    copy "build\vsax_paper.pdf" "vsax_paper.pdf" >nul
    echo.
    echo ========================================
    echo Success! PDF generated: vsax_paper.pdf
    echo ========================================
    echo.
    echo Build files are in: build\
) else (
    echo.
    echo ========================================
    echo ERROR: Compilation failed!
    echo Check build\vsax_paper.log for details
    echo ========================================
    exit /b 1
)
