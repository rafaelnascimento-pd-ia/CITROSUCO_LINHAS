===============================================================================
   SCRIPT: densificar_vertices_linhas.py
===============================================================================

1. O QUE ESTE SCRIPT FAZ:
-------------------------------------------------------------------------------
Script de pós-processamento geográfico para reamostrar e densificar vértices em 
linhas de plantio a intervalos regulares (padrão 10 metros).

Navega por toda a estrutura de subpastas aninhadas da pasta mãe (incluindo 
sub-subpastas como parte1, parte2, etc.), insere os vértices espaçados a cada N 
metros nas geometrias de linha e copia todos os demais arquivos (polígonos de 
talhão, bordaduras e arquivos auxiliares .dbf, .prj, .cpg) mantendo a estrutura 
espelhada na nova pasta de saída.

2. COMANDO PARA EXECUTAR NO TERMINAL:
-------------------------------------------------------------------------------
python scripts/densificar_vertices_linhas/densificar_vertices_linhas.py -i "C:\BRUNO\PROJETOS\linhas_citrosuco\completo\Nova pasta" -dist 10.0 -o "C:\BRUNO\PROJETOS\linhas_citrosuco\completo\Nova_pasta_10m"
===============================================================================
