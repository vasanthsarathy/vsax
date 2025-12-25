@echo off
REM Build script for VSAX papers on Windows
REM Requires pdflatex and bibtex to be installed

setlocal enabledelayedexpansion

REM Parse command line argument
set PAPER=%1
if "%PAPER%"=="" set PAPER=all

echo ========================================
echo Building VSAX papers...
echo ========================================
echo.

REM Create build directory if it doesn't exist
if not exist "build" mkdir build

if "%PAPER%"=="all" goto build_all
if "%PAPER%"=="main" goto build_main
if "%PAPER%"=="mloss" goto build_mloss
echo Unknown paper: %PAPER%
echo Usage: build.bat [all^|main^|mloss]
exit /b 1

:build_all
call :build_paper vsax_paper
if errorlevel 1 exit /b 1
call :build_paper vsax_mloss
if errorlevel 1 exit /b 1
echo.
echo ========================================
echo All papers built successfully!
echo ========================================
echo   - vsax_paper.pdf (main paper)
echo   - vsax_mloss.pdf (MLOSS submission)
echo ========================================
exit /b 0

:build_main
call :build_paper vsax_paper
exit /b %errorlevel%

:build_mloss
call :build_paper vsax_mloss
exit /b %errorlevel%

:build_paper
set BASENAME=%~1
echo.
echo Building %BASENAME%.pdf...
echo ----------------------------------------

REM Clean previous auxiliary files for this paper
if exist "build\%BASENAME%.aux" del /q "build\%BASENAME%.aux"
if exist "build\%BASENAME%.toc" del /q "build\%BASENAME%.toc"
if exist "build\%BASENAME%.out" del /q "build\%BASENAME%.out"
if exist "build\%BASENAME%.bbl" del /q "build\%BASENAME%.bbl"

echo [1/5] First pdflatex pass...
pdflatex -output-directory=build -interaction=nonstopmode %BASENAME%.tex >nul 2>&1

echo [2/5] Running bibtex...
pushd build
bibtex %BASENAME% >nul 2>&1
popd

echo [3/5] Second pdflatex pass...
pdflatex -output-directory=build -interaction=nonstopmode %BASENAME%.tex >nul 2>&1

echo [4/5] Third pdflatex pass...
pdflatex -output-directory=build -interaction=nonstopmode %BASENAME%.tex >nul 2>&1

echo [5/5] Final pdflatex pass...
pdflatex -output-directory=build -interaction=nonstopmode %BASENAME%.tex

REM Copy the PDF to the main directory
if exist "build\%BASENAME%.pdf" (
    copy "build\%BASENAME%.pdf" "%BASENAME%.pdf" >nul
    echo.
    echo SUCCESS: %BASENAME%.pdf generated!
    echo.
    exit /b 0
) else (
    echo.
    echo ERROR: Compilation of %BASENAME%.pdf failed!
    echo Check build\%BASENAME%.log for details
    echo.
    exit /b 1
)
