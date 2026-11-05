===============================================================================
   SCRIPT: adicionar_coordenadas_mdt.py
===============================================================================

1. O QUE ESTE SCRIPT FAZ:
-------------------------------------------------------------------------------
Script para adicionar coordenadas (Z / altitude do MDT, Lat/Lon WGS84, X/Y UTM) em geometrias de LINHAS e POLÍGONOS.
Não altera nenhum arquivo original; os resultados são salvos espelhando a estrutura de subpastas.

2. COMANDO PARA EXECUTAR NO TERMINAL:
-------------------------------------------------------------------------------
python scripts/adicionar_coordenadas_mdt/adicionar_coordenadas_mdt.py -i "caminho/pasta_entrada" -m "caminho/mdt.tif" -o "caminho/pasta_saida"
===============================================================================
