@echo off
chcp 65001 >nul
title Interface Web - Citrosuco IA Linhas
echo ==============================================================================
echo       INICIANDO INTERFACE WEB - PROJETO IA LINHAS CITROSUCO
echo ==============================================================================
echo.
if not exist ".venv\Scripts\python.exe" (
    echo [ERRO] Ambiente virtual .venv nao encontrado!
    echo Execute primeiro o instalador "instalar_ambiente.bat".
    echo.
    pause
    exit /b 1
)

echo Iniciando servidor Flask...
echo Acesse no navegador: http://localhost:5000
echo Pressione Ctrl + C nesta janela para encerrar o servidor.
echo.
start http://localhost:5000
.venv\Scripts\python.exe app.py
pause
