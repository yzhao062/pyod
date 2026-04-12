@echo off
REM Build PyOD docs locally (Windows).
REM
REM Usage:
REM   docs\build.bat          - Build once
REM   docs\build.bat clean    - Clean build
REM   docs\build.bat watch    - Auto-rebuild on change
REM
REM Dependencies (install once):
REM   pip install sphinxcontrib-bibtex furo sphinx-rtd-theme sphinx-autobuild

cd /d "%~dp0"

set MODE=%1
if "%MODE%"=="" set MODE=build

if "%MODE%"=="clean" (
    if exist _build rmdir /s /q _build
    echo Cleaned _build\
    exit /b 0
)

if "%MODE%"=="watch" (
    sphinx-autobuild . _build\html --open-browser --port 8000
    exit /b %errorlevel%
)

python -m sphinx -b html . _build\html
echo.
echo Built docs. Open: docs\_build\html\index.html
