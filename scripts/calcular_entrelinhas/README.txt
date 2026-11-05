===============================================================================
   SCRIPT: calcular_entrelinhas.py
===============================================================================

1. O QUE ESTE SCRIPT FAZ:
-------------------------------------------------------------------------------
Script de pós-processamento geográfico para gerar as linhas centrais das entrelinhas de plantio a partir das fileiras (linhas de plantio) detectadas.
Suporta linhas retas e curvas por interpolação ponto a ponto adaptável.

2. COMANDO PARA EXECUTAR NO TERMINAL:
-------------------------------------------------------------------------------
.\.venv\Scripts\python.exe scripts/calcular_entrelinhas/calcular_entrelinhas.py -i "D:\raphael\84650\entregas\VETORES\LINHAS_MASSA.geojson" -o "D:\raphael\84650\entregas\VETORES\retas_curvas.geojson"

===============================================================================
