@echo off
REM Build script for VSAX paper on Windows
REM Requires latexmk to be installed

echo Building VSAX paper...
echo.

REM Create build directory if it doesn't exist
if not exist "build" mkdir build

REM Clean previous auxiliary files
if exist "build\*.aux" del /q "build\*.aux"
if exist "build\*.toc" del /q "build\*.toc"
if exist "build\*.out" del /q "build\*.out"

echo Running first pass to generate TOC data...
pdflatex -output-directory=build -interaction=nonstopmode vsax_paper.tex >nul

echo Running second pass to include TOC...
pdflatex -output-directory=build -interaction=nonstopmode vsax_paper.tex >nul

echo Running third pass to finalize...
pdflatex -output-directory=build -interaction=nonstopmode vsax_paper.tex

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
