===============================================================================
   SCRIPT: adicionar_coordenadas_lote.py
===============================================================================

1. O QUE ESTE SCRIPT FAZ:
-------------------------------------------------------------------------------
Script de utilidade para adicionar coordenadas geográficas (WGS84) e métricas (UTM) a arquivos vetoriais (linhas e polígonos) em lote de forma automática.
Permite opcionalmente informar um MDT raster para converter as geometrias em 3D e preencher os atributos de altitude (Z).
Preserva as geometrias e atributos originais intactos.

2. COMANDO PARA EXECUTAR NO TERMINAL:
-------------------------------------------------------------------------------
python scripts/adicionar_coordenadas_lote/adicionar_coordenadas_lote.py -i "caminho/pasta_entrada" -m "caminho/mdt.tif" -o "caminho/pasta_saida"
===============================================================================
