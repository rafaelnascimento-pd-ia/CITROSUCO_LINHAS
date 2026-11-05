@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"
title Instalador de Dependencias - IA Linhas Citrosuco

echo ==============================================================================
echo       INSTALADOR DE AMBIENTE E DEPENDENCIAS - IA LINHAS CITROSUCO
echo ==============================================================================
echo.

REM 1. Localizar Python no sistema
set "BASE_PYTHON="

REM Opcao 1: Python Launcher (py)
py -0 >nul 2>&1
if !errorlevel! == 0 set "BASE_PYTHON=py -3"

REM Opcao 2: python direto no PATH
if "!BASE_PYTHON!"=="" (
    python --version >nul 2>&1
    if !errorlevel! == 0 set "BASE_PYTHON=python"
)

REM Opcao 3: Anaconda / Miniconda em diretorios comuns
if "!BASE_PYTHON!"=="" (
    if exist "%USERPROFILE%\anaconda3\python.exe" set "BASE_PYTHON=%USERPROFILE%\anaconda3\python.exe"
    if exist "%USERPROFILE%\miniconda3\python.exe" set "BASE_PYTHON=%USERPROFILE%\miniconda3\python.exe"
    if exist "C:\anaconda3\python.exe" set "BASE_PYTHON=C:\anaconda3\python.exe"
    if exist "C:\ProgramData\anaconda3\python.exe" set "BASE_PYTHON=C:\ProgramData\anaconda3\python.exe"
)

if "!BASE_PYTHON!"=="" (
    echo [ERRO] Python nao foi encontrado neste computador!
    echo.
    echo Por favor, instale o Python 3.10, 3.11 ou 3.12:
    echo 1. Acesse: https://www.python.org/downloads/
    echo 2. Baixe o instalador do Windows.
    echo 3. IMPORTANTE: Na primeira tela do instalador, MARQUE a caixa:
    echo    "Add python.exe to PATH" (Adicionar Python ao PATH).
    echo 4. Conclua a instalacao e execute este arquivo novamente.
    echo.
    pause
    exit /b 1
)

echo [OK] Python localizado: !BASE_PYTHON!
!BASE_PYTHON! --version
echo.

REM 2. Criar ambiente virtual .venv
if not exist ".venv\Scripts\python.exe" (
    echo [1/3] Criando ambiente virtual isolado (.venv)...
    !BASE_PYTHON! -m venv .venv
    if !errorlevel! neq 0 (
        echo [ERRO] Falha ao criar a pasta .venv.
        pause
        exit /b 1
    )
    echo [OK] Ambiente .venv criado com sucesso.
) else (
    echo [1/3] Ambiente .venv ja existente.
)
echo.

REM 3. Atualizar pip
echo [2/3] Atualizando o pip...
call ".venv\Scripts\activate.bat"
python -m pip install --upgrade pip --quiet
echo [OK] Pip atualizado.
echo.

REM 4. Instalar PyTorch CUDA e todas as bibliotecas
echo [3/3] Instalando PyTorch (CUDA 12.1) e bibliotecas geoespaciais/IA...
echo       (rasterio, geopandas, shapely, opencv, segmentation-models-pytorch, albumentations, flask)
echo       Isso pode levar alguns minutos dependendo da sua conexao...
echo.

if exist "requirements.txt" (
    python -m pip install -r requirements.txt
    if !errorlevel! neq 0 (
        echo.
        echo [AVISO] Tentando instalar PyTorch e dependencias em modo seguro...
        python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
        python -m pip install -r requirements.txt
    )
) else (
    echo [ERRO] Arquivo requirements.txt nao encontrado!
    pause
    exit /b 1
)

echo.
echo ==============================================================================
echo       VERIFICACAO DO AMBIENTE E SUPORTE A GPU / CUDA
echo ==============================================================================
python -c "import torch; print(f'PyTorch: {torch.__version__} | CUDA Disponivel: {torch.cuda.is_available()} | Dispositivo: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU (Sem GPU)\"}')"
python -c "import rasterio, geopandas, shapely, cv2, albumentations; print('[SUCESSO] Todas as bibliotecas geoespaciais e de visao computacional foram carregadas!')"

echo.
echo ==============================================================================
echo       INSTALACAO CONCLUIDA COM SUCESSO!
echo ==============================================================================
echo.
echo Agora o projeto esta 100%% pronto para uso.
echo - Para abrir a Interface Grafica Web: execute "2_INICIAR_INTERFACE_WEB.bat"
echo - Para testar o ambiente a qualquer momento: execute "3_TESTAR_AMBIENTE.bat"
echo.
pause
