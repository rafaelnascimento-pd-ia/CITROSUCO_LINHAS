@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"
title Interface Web - IA Linhas Citrosuco

echo ==============================================================================
echo       INICIANDO INTERFACE WEB - PROJETO IA LINHAS CITROSUCO
echo ==============================================================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo [ERRO] Ambiente virtual .venv nao encontrado!
    echo Execute primeiro o arquivo "1_INSTALAR_DEPENDENCIAS.bat".
    echo.
    pause
    exit /b 1
)

echo Ativando ambiente virtual...
call ".venv\Scripts\activate.bat"

echo Iniciando servidor Flask...
echo Acesse no navegador: http://localhost:5000
echo Para encerrar o servidor, feche esta janela ou pressione Ctrl + C.
echo.

start http://localhost:5000
python app.py
pause
