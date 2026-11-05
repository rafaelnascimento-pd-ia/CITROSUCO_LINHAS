===============================================================================
   SCRIPT: calcular_entrelinhas_curvas_v2.py
===============================================================================

1. O QUE ESTE SCRIPT FAZ:
-------------------------------------------------------------------------------
Gera entrelinhas de alta precisão e perfeitamente suaves para talhões com 
fileiras de plantio curvas (ex: plantios em curvas de nível e matações).

Diferenciais da Versão 2:
- Elimina 100% dos ângulos agudos / ziguezagues artificiais.
- Trata matações (pontas de rua e fileiras em cunha) de forma inteligente.
- Utiliza Projeção Paramétrica Ortogonal Monotônica ao longo do arco das fileiras.
- Filtro angular que evita distorções nas extremidades das curvas.
- Exporta tanto GeoJSON quanto Shapefile com backups automáticos.

2. COMANDO PARA EXECUTAR NO TERMINAL:
-------------------------------------------------------------------------------
.\.venv\Scripts\python.exe scripts/calcular_entrelinhas_curvas_v2/calcular_entrelinhas_curvas_v2.py -i "D:\raphael\84650\entregas\VETORES\LINHAS_MASSA.geojson" -o "D:\raphael\84650\entregas\VETORES\entrelinhas_curvas_v2.geojson"

===============================================================================
