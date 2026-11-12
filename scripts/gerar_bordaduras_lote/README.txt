===============================================================================
   SCRIPT: gerar_bordaduras_lote.py
===============================================================================

1. O QUE ESTE SCRIPT FAZ:
-------------------------------------------------------------------------------
Script para processamento em lote de polígonos de talhões (bordadura).
Aplica buffer de 7m e projeta a altitude (Z) de um MDT raster nos vértices.
Salva o polígono de bordadura e a rota pelo contorno externo como shapefiles
Single Part WGS84. A rota é MultiLineStringZ e usa codificação UTF-8.

2. COMANDO PARA EXECUTAR NO TERMINAL:
-------------------------------------------------------------------------------
python scripts/gerar_bordaduras_lote/gerar_bordaduras_lote.py -i "caminho/pasta_entrada" -m "caminho/mdt.tif" -o "caminho/pasta_saida"
===============================================================================
