@echo off
chcp 65001 >nul
echo ==============================================================================
echo       INSTALADOR AUTOMATICO - PROJETO IA LINHAS CITROSUCO
echo ==============================================================================
echo.

:: 1. Verificar Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERRO] Python nao encontrado no PATH do Windows!
    echo Por favor, instale o Python 3.10, 3.11 ou 3.12 e marque "Add python.exe to PATH".
    pause
    exit /b 1
)

echo [1/4] Verificando versao do Python...
python --version

:: 2. Criar Ambiente Virtual (.venv)
if not exist ".venv" (
    echo [2/4] Criando ambiente virtual (.venv)...
    python -m venv .venv
    if %errorlevel% neq 0 (
        echo [ERRO] Falha ao criar ambiente virtual.
        pause
        exit /b 1
    )
) else (
    echo [2/4] Ambiente virtual .venv ja existe.
)

:: 3. Atualizar pip
echo [3/4] Atualizando pip...
.venv\Scripts\python.exe -m pip install --upgrade pip

:: 4. Instalar PyTorch com CUDA 12.1 e dependencias
echo [4/4] Instalando PyTorch (CUDA 12.1) e todas as bibliotecas...
.venv\Scripts\python.exe -m pip install -r requirements.txt

if %errorlevel% neq 0 (
    echo.
    echo [AVISO] Ocorreu um erro durante a instalacao. Tentando instalar dependencias em modo seguro...
    .venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    .venv\Scripts\python.exe -m pip install -r requirements.txt
)

echo.
echo ==============================================================================
echo       VERIFICACAO DO AMBIENTE E SUPORTE A GPU / CUDA
echo ==============================================================================
.venv\Scripts\python.exe -c "import torch; print(f'PyTorch: {torch.__version__} | CUDA Disponivel: {torch.cuda.is_available()} | Dispositivo: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

echo.
echo [SUCESSO] Ambiente configurado e pronto para uso!
echo - Para iniciar a interface web: iniciar_interface_web.bat
echo - Para testar o ambiente: testar_ambiente.bat
echo - Para ativar no terminal: .venv\Scripts\activate
echo.
pause
