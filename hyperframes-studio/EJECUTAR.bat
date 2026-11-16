@echo off
title Regresion Lineal - Minimos Cuadrados
cd /d "%~dp0"

echo ==========================================
echo   PIPELINE COMPLETO: datos nuevos + results/
echo ==========================================
echo.

python src\main.py --sin-gui

if errorlevel 1 (
    echo.
    echo [ERROR] Hubo un problema. Revisa el mensaje de arriba.
) else (
    echo.
    echo [OK] data/input, data/generated y results/ actualizados
    echo   - results\video\explicacion_optimizacion.mp4
    echo   - results\html\dashboard.html  (abrelo en el navegador)
)
echo.
pause
