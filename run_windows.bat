@echo off
setlocal
if not exist .venv (
    python -m venv .venv
)
call .venv\Scripts\activate
pip install -r requirements.txt
python app.py
if errorlevel 1 (
    echo.
    echo ----- ERROR -----
    echo If you saw "Qt bindings not found", try:
    echo   pip install PySide6 matplotlib numpy
    echo.
    pause
)
