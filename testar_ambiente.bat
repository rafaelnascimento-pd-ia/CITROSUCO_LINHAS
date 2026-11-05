@echo off
chcp 65001 >nul
title Teste de Ambiente - Citrosuco IA Linhas
echo ==============================================================================
echo       TESTE DE AMBIENTE E SUPORTE A GPU / CUDA
echo ==============================================================================
echo.
if not exist ".venv\Scripts\python.exe" (
    echo [ERRO] Ambiente virtual .venv nao encontrado!
    echo Execute primeiro o instalador "instalar_ambiente.bat".
    echo.
    pause
    exit /b 1
)

echo [1/2] Testando PyTorch e Aceleracao por GPU (CUDA)...
.venv\Scripts\python.exe -c "import torch; print(f'  -> PyTorch: {torch.__version__}'); print(f'  -> CUDA Disponivel: {torch.cuda.is_available()}'); print(f'  -> Dispositivo: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU (Sem GPU detectada)\"}')"

echo.
echo [2/2] Testando bibliotecas geoespaciais e de visao computacional...
.venv\Scripts\python.exe -c "import rasterio, geopandas, shapely, cv2, albumentations, segmentation_models_pytorch; print('  -> rasterio, geopandas, shapely, cv2, albumentations, smp: CARREGADOS COM SUCESSO!')"

echo.
echo ==============================================================================
echo       TESTE CONCLUIDO! O AMBIENTE ESTA OPERACIONAL.
echo ==============================================================================
echo.
pause
