# Separação automática de talhões

O fluxo com `export_as_lines: true` identifica regiões de fileiras usando
orientação **local**, proximidade e suporte de plantio. A opção está disponível
em `infer_post`, `train_infer`, `post_only`, `resultado` e no pós-processamento
da interface web. Não altera o modelo de IA nem requer novas dependências.

## Funcionamento

1. Reprojeta para coordenadas métricas, conserva cada polígono de origem e suas
   exclusões, e trata componentes desconectados separadamente.
2. Estima o espaçamento por sondas normais locais, dentro da faixa configurada
   em `min_row_spacing_m` / `max_row_spacing_m`. Sem vizinhos suficientes, usa o
   ponto médio dessa faixa e registra o espaçamento utilizado.
3. Amostra tangentes ao longo das fileiras. Usa estatística de ângulo duplo:
   inverter os vértices ou usar 0°/180° não altera a orientação agrícola.
4. Acumula orientações em células locais proporcionais ao espaçamento. Células
   com orientações conflitantes não conectam grupos. Um grafo esparso conecta
   células vizinhas de orientação compatível, sem cruzar exclusões vetoriais.
5. Exige suporte de várias linhas, comprimento e extensão transversal. Um
   fragmento isolado não é suficiente para criar um talhão. Curvas e mudanças
   graduais podem permanecer conectadas, mesmo com orientações globais diversas.
6. Havendo dois ou mais grupos sustentados, propaga seus rótulos conjuntamente
   por watershed no domínio permitido. Recorta os polígonos no limite vetorial
   original, restitui pequenas sobras e valida cobertura e ausência de sobreposição.
7. Recorta as fileiras por filho e calcula entrelinhas exclusivamente dentro
   do grupo associado. Identificadores permanecem na normalização final 3D.

A extensão da fronteira em trechos sem fileiras é uma inferência espacial,
não uma identificação de estrada ou obstáculo. Obstáculos precisam estar
representados nas exclusões do vetor de entrada; ausência de predição não
é automaticamente convertida em exclusão. Não existe um detector específico
de enforcamentos: descontinuidades de suporte podem separar regiões, mas um
estreitamento com suporte contínuo não força um corte.

## Configuração

```yaml
export_as_lines: true
min_row_spacing_m: 4.0
max_row_spacing_m: 8.5
field_partition:
  enabled: true
  angle_tolerance_deg: 20.0
  cell_spacing_factor: 2.0
  min_rows: 3
  min_support_spacing_factor: 2.0
  sample_step_m: null
  grid_resolution_m: 1.0
  max_grid_cells: 4000000
```

- `angle_tolerance_deg`: compatibilidade entre células vizinhas, não faixas
  globais de azimute. O padrão 20° reutiliza a escala do teste de paralelismo
  existente. A concentração mínima local é `cos(2 × tolerância)`. Calibrar em
  áreas revisadas para distinguir ruído, curvas e mudanças de bloco.
- `cell_spacing_factor`: lado da célula em múltiplos do espaçamento; duas
  entrelinhas permitem observar orientação da vizinhança. Células maiores
  agregam mais evidência, mas podem misturar fronteiras próximas.
- `min_rows`: número mínimo de geometrias de fileira que sustentam um grupo.
  O padrão três evita aceitar um par isolado. A extensão transversal também
  precisa representar pelo menos `min_rows - 1` espaçamentos, para reduzir
  falsos grupos formados por fragmentos da mesma fileira.
- `min_support_spacing_factor`: comprimento total mínimo por fileira de
  suporte, expresso em espaçamentos. O total mínimo é
  `min_rows × espaçamento × fator`. O padrão dois exige evidência longitudinal
  além de um encontro pontual. Regiões menores que `min_rows × espaçamento²`
  não viram filhos independentes na partição.
- `sample_step_m`: quando nulo, metade do espaçamento medido. A janela da
  tangente também deriva da metade do espaçamento.
- `grid_resolution_m`: precisão da grade de partição, em metros. Quando nulo,
  usa o menor entre 1 m e um quarto do espaçamento. O comando `resultado`
  também repassa a resolução configurada ao gerador de bordaduras.
- `max_grid_cells`: proteção de memória por domínio processado. O algoritmo
  interrompe com mensagem explícita se o limite for excedido; não reduz a
  precisão silenciosamente. Bordaduras desconectadas são processadas em separado.

Os padrões são pontos de partida configuráveis, baseados nos parâmetros
geométricos existentes e no suporte mínimo de vizinhança; não são uma
calibração agronômica universal. Os espaçamentos de referência inspecionados
em 86508/68094 ficaram próximos de 6,4–6,5 m.

`enabled: false` desativa a descoberta de subdivisões por orientação. Ainda
preserva os talhões fornecidos e seus identificadores nas etapas seguintes.
A API web aceita o mesmo objeto em `field_partition` no JSON de processamento.

## Arquivos e associação

- `talhoes_separados.geojson`: polígonos filhos e atributos de origem.
- `shp/TALHOES_SEPARADOS.*`: limites em WGS84, com campos curtos compatíveis
  com Shapefile (`parent_id`, `field_id`, `area_m2`, `angle_deg`, `n_rows`,
  `support_m`). Estes limites permanecem 2D.
- `predicao_linhas.*`: fileiras recortadas, com `parent_id`, `field_id` e `row_id`.
- `predicao_entrelinhas.*`: entrelinhas por filho; conserva também as referências
  `source_a`, `source_b`, `row_a` e `row_b`. Os índices numéricos são globais
  ao arquivo de entrada, não reiniciam por bloco.
- `separacao_metadata.json`: parâmetros e diagnóstico por componente de origem.

No comando `resultado`, são entregues o SHP de entrelinhas com o nome
`lines_output_name` e os SHPs de bordadura/rota. O SHP `TALHOES_SEPARADOS`
é intermediário e não é incluído na entrega; componentes desse produto que
tenham sobrado de execuções anteriores são removidos da pasta final.
As tabelas dos SHPs finais contêm apenas `X`, `Y` e `Z`. Os identificadores
permanecem nos intermediários até terminar o processamento geométrico.
Há uma feição de bordadura/rota por talhão, podendo conter várias partes. As geometrias de
linhas, bordaduras e rotas continuam 3D, com Z amostrado no MDT.

O buffer externo das bordaduras é dividido conjuntamente. A borda interna
entre filhos não avança sobre o plantio do vizinho. Exclusões interiores são
preservadas, inclusive quando o buffer positivo as fecharia. Bordaduras de
filhos diferentes nunca são dissolvidas em uma só feição.

**A rota gerada é um contorno geométrico, não uma garantia de corredor livre
para dirigir.** Uma divisão por orientação não cria espaço físico de manobra.
As rotas seguem contornos externos das partes; não são criadas rotas de
contorno para os buracos de obstáculos.

Limites compartilhados e rotas associadas não são simplificados ou arredondados
independentemente, pois isso poderia criar sobreposição ou atravessar outra
região. Por esse motivo, os limites de vértices das opções anteriores não são
aplicados a esses produtos identificados; eles podem conter mais vértices.

## Execução

```bash
python main.py resultado --config configs/resultado.yaml
```

Para avaliar sem nova inferência, `src.field_partition.export_partition` aceita
GeoDataFrames de fileiras existentes e talhões originais, além de uma pasta de
saída separada. Essa avaliação não modifica as entradas. Os limites de saída
devem ser conferidos sobre a ortofoto antes de uso operacional.
