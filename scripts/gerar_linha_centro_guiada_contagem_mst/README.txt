===============================================================================
   SCRIPT: gerar_linha_centro_guiada_contagem_mst.py
===============================================================================

1. O QUE ESTE SCRIPT FAZ:
-------------------------------------------------------------------------------
Traça linhas centrais de plantio curvas e contínuas a partir de polígonos locais e pontos guia de contagem.
Utiliza o algoritmo de Árvore Geradora Mínima (MST) otimizado com R-Tree para ordenar sequencialmente pontos em curvas acentuadas.

2. COMANDO PARA EXECUTAR NO TERMINAL:
-------------------------------------------------------------------------------
python scripts/gerar_linha_centro_guiada_contagem_mst/gerar_linha_centro_guiada_contagem_mst.py -p "caminho/linhas_plantio.geojson" -pts "caminho/contagem.geojson" -o "caminho/linhas_centro.shp"
===============================================================================
