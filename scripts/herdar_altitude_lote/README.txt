===============================================================================
   SCRIPT: herdar_altitude_lote.py
===============================================================================

1. O QUE ESTE SCRIPT FAZ:
-------------------------------------------------------------------------------
Script de pós-processamento geográfico em lote para ler subpastas de talhões, projetar altitude (Z) de um MDT raster nas linhas e no polígono de buffer do talhão (bordadura de 6.5m), salvando em WGS84 Single Part.

2. COMANDO PARA EXECUTAR NO TERMINAL:
-------------------------------------------------------------------------------
python scripts/herdar_altitude_lote/herdar_altitude_lote.py -i "caminho/pasta_input" -m "caminho/mdt.tif" -o "caminho/pasta_output"
===============================================================================
