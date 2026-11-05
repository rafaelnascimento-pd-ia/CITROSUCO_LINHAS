# Plano técnico final para melhoria da vetorização de linhas de plantio em citros

## 1. Objetivo

Melhorar a **fidelidade das linhas curvas, pontas e continuidade** no pipeline de detecção e vetorização de linhas de plantio, preservando a compatibilidade com o fluxo existente:

```text
inference → postprocess → exportação geoespacial
```

O plano prioriza correções incrementais, mensuráveis e reversíveis. A implementação deve manter fallback para o comportamento atual até que cada melhoria seja validada.

## 2. Regras obrigatórias de implementação

1. Não remover nem alterar de forma incompatível as assinaturas públicas existentes.
2. Novos parâmetros opcionais devem ser adicionados somente ao final das assinaturas.
3. Não interromper o fluxo atual `inference → postprocess`.
4. Implementar uma fase por vez.
5. Preservar o algoritmo atual como fallback configurável.
6. Registrar logs das etapas novas e das falhas recuperáveis.
7. Executar todas as operações métricas em CRS projetado ou em geometria previamente reprojetada para unidades métricas.
8. Não aceitar melhoria visual como único critério de sucesso.
9. Medir o baseline atual antes de alterar os algoritmos.
10. Rejeitar resultados cuja geometria seja inválida ou esteja fora do talhão sem registrar a causa.

## 3. Diagnóstico resumido do estado atual

O pipeline atual é funcional para fileiras retas e máscaras relativamente limpas, mas apresenta risco para curvas, pontas, fragmentos próximos e talhões complexos.

A função `extract_centerline_from_polygon` utiliza rasterização, esqueletização e escolha do maior caminho do esqueleto. Esse método pode produzir uma linha plausível, mas não garante o centro geométrico exato do polígono. A posição da linha depende da resolução e do alinhamento da grade raster.

A função `merge_collinear_lines` usa uma estratégia gulosa. Ela aceita o primeiro par elegível, usa somente segmentos terminais para calcular a direção e não avalia suficientemente a compatibilidade da curvatura ou a presença de fileiras intermediárias.

A função `deduplicate_close_lines` considera apenas a menor distância geométrica entre linhas. Ela não verifica sobreposição longitudinal, paralelismo ou a proporção da linha que permanece próxima da outra.

Na inferência, a normalização por mínimo e máximo do próprio recorte pode variar entre talhões. Também faltam proteções explícitas para imagens constantes, `nodata`, CRS geográfico e configuração do batch.

## 4. Ordem recomendada das fases

A ordem deve ser:

1. **Fase 0 — Instrumentação e baseline**
2. **Fase 0.5 — Correções de segurança**
3. **Fase 1 — Extração e validação da centerline**
4. **Fase 2 — Deduplicação preliminar e final**
5. **Fase 3 — União de fragmentos**
6. **Fase 4 — Parametrização por GSD**
7. **Fase 5 — Melhorias probabilísticas na inferência**

A deduplicação deve ocorrer antes do merge para reduzir conexões falsas entre duplicatas. Recomenda-se também uma deduplicação final após o merge:

```text
extração → deduplicação preliminar → merge → deduplicação final
```

## 5. Fase 0 — Instrumentação e baseline

### 5.1 Novo módulo de avaliação

Criar:

```text
src/evaluation.py
```

Adicionar funções compatíveis com geometrias em CRS métrico:

```python
def compute_center_error(line_pred, line_ref, step_m=1.0):
    """Calcula erro médio e percentil 95 entre linhas amostradas por comprimento."""


def compute_length_coverage(line_pred, line_ref, tolerance_m=1.0):
    """Calcula a fração do comprimento da referência coberta pela predição."""


def compute_endpoint_error(line_pred, line_ref):
    """Calcula o erro entre endpoints correspondentes após orientação consistente."""


def compute_parallelism(line1, line2, step_m=1.0):
    """Calcula a diferença angular média entre tangentes amostradas."""


def compute_overlap_ratio(line1, line2, tolerance_m=1.0):
    """Calcula a fração de uma linha contida no buffer da outra."""
```

As métricas devem validar o CRS ou receber somente geometrias previamente transformadas para unidades métricas.

### 5.2 Definições formais das métricas

| Métrica | Definição recomendada |
|---|---|
| Erro de centro | Distância média e percentil 95 entre pontos amostrados por comprimento |
| Cobertura longitudinal | Fração do comprimento da referência dentro de um buffer da predição |
| Erro de ponta | Distância entre endpoints correspondentes depois de orientar as linhas na mesma direção |
| Paralelismo | Média do menor ângulo entre tangentes locais amostradas |
| Sobreposição | Fração de uma linha contida em um buffer da outra, com denominador explicitamente definido |

Para `compute_overlap_ratio`, o denominador padrão deve ser o comprimento da linha avaliada. A função deve permitir indicar se a avaliação é feita da predição para a referência ou da referência para a predição.

### 5.3 Logs obrigatórios

Registrar, por talhão e no total:

- número de polígonos recebidos;
- número de polígonos descartados por área;
- número de polígonos descartados por elongação;
- número de linhas extraídas;
- número de falhas na extração da centerline;
- número de linhas deduplicadas;
- número de candidatos de merge;
- número de merges realizados;
- número de merges rejeitados e motivo da rejeição;
- número de linhas estendidas até a borda;
- número de geometrias inválidas;
- tempo e memória por etapa.

### 5.4 Baseline obrigatório

Antes de qualquer alteração algorítmica, executar o pipeline atual e salvar:

- vetores resultantes;
- imagens de diagnóstico;
- métricas de qualidade;
- tempo de execução;
- pico de RAM e memória GPU;
- quantidade de polígonos, linhas, merges e deduplicações.

## 6. Fase 0.5 — Correções de segurança

### 6.1 Normalização radiométrica

Substituir a normalização implícita por mínimo e máximo local pela normalização usada no treinamento do modelo. Caso a normalização local precise ser mantida temporariamente, proteger o caso constante:

```python
value_range = img_hwc.max() - img_hwc.min()
if value_range <= 1e-6:
    img_hwc = np.zeros_like(img_hwc, dtype=np.uint8)
else:
    img_hwc = ((img_hwc - img_hwc.min()) / value_range * 255).astype(np.uint8)
```

Valores `nodata` devem ser excluídos das estatísticas. O número de canais deve ser validado contra o checkpoint, em vez de descartar silenciosamente canais além dos três primeiros.

### 6.2 Validação de CRS

Antes de calcular área, comprimento, distância ou espaçamento:

1. verificar se o CRS existe;
2. verificar se é projetado;
3. verificar se as unidades são métricas;
4. reprojetar temporariamente quando necessário;
5. restaurar o CRS de saída apenas na exportação.

### 6.3 Geometrias inválidas

Registrar e tratar explicitamente polígonos inválidos. Não utilizar `except Exception` como substituto de validação geométrica. Cada falha deve registrar o componente, bounds, área, operação e causa.

## 7. Fase 1 — Extração da linha central

Arquivo principal:

```text
src/postprocess.py
```

Função principal:

```python
extract_centerline_from_polygon
```

O algoritmo atual deve permanecer disponível como fallback.

### 7.1 Poda do esqueleto

Criar:

```python
def prune_skeleton_graph(G, min_length_px, protected_path=None):
    """Remove ramos laterais curtos preservando o caminho principal."""
```

O parâmetro deve ser expresso em pixels ou convertido explicitamente:

```python
min_length_px = min_length_m / resolution
```

Não remover indiscriminadamente todos os caminhos curtos ligados a nós de grau 1. Extremidades reais da fileira também possuem grau 1. O procedimento deve:

1. identificar o maior caminho principal;
2. proteger as duas extremidades desse caminho;
3. remover somente ramos laterais curtos;
4. preservar componentes principais com comprimento suficiente;
5. registrar o número de ramos removidos.

### 7.2 Escolha dos endpoints

A escolha primária deve ser baseada na maior distância geodésica do grafo. PCA não deve substituir esse critério em curvas.

Uso recomendado de PCA:

- estimar orientação inicial;
- desempatar componentes semelhantes;
- auxiliar na ordenação de candidatos;
- fornecer uma referência para fileiras predominantemente retas.

Para curvas, a sequência prioritária é:

```text
maior caminho geodésico → endpoints desse caminho → PCA como apoio
```

### 7.3 Reamostragem por comprimento

Substituir:

```python
longest_path[::step]
```

por:

```python
def resample_line_by_length(coords, step_m, preserve_curvature=True):
    """Reamostra uma linha por comprimento acumulado em unidades métricas."""
```

Requisitos:

- calcular distância acumulada real;
- produzir espaçamento aproximadamente uniforme;
- preservar primeiro e último ponto;
- operar em CRS métrico;
- preservar pontos de alta curvatura quando necessário;
- evitar dependência do número de pixels do esqueleto.

A reamostragem uniforme não deve eliminar automaticamente vértices associados a mudanças abruptas de direção.

### 7.4 Suavização ao longo do arco

Remover a suavização independente em X e Y. Criar uma função baseada no comprimento acumulado:

```python
def smooth_line_arc_length(coords, smoothing=1.0, preserve_endpoints=True):
    """Suaviza uma linha parametrizada pelo comprimento acumulado."""
```

Preferir spline regularizada ou filtro parametrizado por comprimento acumulado. Uma média móvel simples só deve ser usada como fallback.

Requisitos:

- preservar endpoints;
- limitar o deslocamento máximo;
- controlar a redução de comprimento;
- evitar que a linha saia do polígono;
- registrar o método e o parâmetro utilizados.

### 7.5 Garantia de permanência no polígono

Criar:

```python
def enforce_line_inside_polygon(line, polygon, tolerance_m=0.0):
    """Valida e corrige a linha mantendo a componente contínua mais próxima."""
```

Não basta projetar pontos individualmente. Em polígonos côncavos, segmentos entre pontos internos podem atravessar regiões externas.

Procedimento recomendado:

1. verificar validade e `line.difference(polygon)`;
2. construir um corredor estreito em torno da linha;
3. intersectar o corredor com o polígono;
4. escolher a componente contínua mais próxima da linha original;
5. preservar a orientação e os endpoints quando possível;
6. validar novamente a geometria resultante;
7. rejeitar e registrar a linha quando a correção não for confiável.

### 7.6 Validação da centerline

Para cada linha extraída, calcular:

- comprimento;
- distância máxima ao polígono;
- fração dentro do polígono;
- número de vértices;
- curvatura máxima;
- presença de auto-interseção;
- distância entre endpoints e borda do polígono.

## 8. Fase 2 — Deduplicação robusta

Função alvo:

```python
deduplicate_close_lines
```

### 8.1 Reamostragem

Reamostrar todas as linhas em passo comum, por exemplo, 1 metro, em CRS métrico.

### 8.2 Métricas por par

Para cada par candidato, calcular:

- distância média entre amostras;
- percentil 95 da distância;
- diferença angular média;
- fração de sobreposição longitudinal;
- diferença de comprimento;
- distância entre envelopes;
- diferença de espaçamento para linhas vizinhas.

### 8.3 Critério de duplicata

Remover uma linha somente quando todas as condições principais forem satisfeitas:

- sobreposição longitudinal alta, inicialmente acima de 70%;
- distância média baixa;
- percentil 95 dentro da tolerância;
- paralelismo alto;
- ausência de evidência de que as linhas representam fileiras distintas.

Os limites devem ser configuráveis. O valor de 70% é um ponto inicial de calibração, não uma verdade universal.

A linha mantida pode ser escolhida por:

1. maior comprimento válido;
2. maior confiança média da rede;
3. maior cobertura da máscara;
4. menor erro de centro, se houver referência.

## 9. Fase 3 — União de fragmentos

Função alvo:

```python
merge_collinear_lines
```

### 9.1 Índice espacial

Usar `shapely.strtree.STRtree` ou o índice espacial do GeoPandas para reduzir comparações. O índice deve recuperar somente linhas próximas das pontas candidatas.

O índice espacial acelera a busca, mas não substitui os critérios geométricos.

### 9.2 Geração de candidatos

Criar candidatos considerando:

- distância entre endpoints;
- tangente média em uma janela de 3 a 5 metros;
- offset lateral;
- diferença de curvatura;
- continuidade da orientação;
- distância entre os segmentos inteiros;
- presença de outra linha entre os fragmentos;
- compatibilidade com máscara ou mapa de probabilidade.

A direção não deve ser estimada somente pelos dois últimos pontos do primeiro fragmento e pelos dois primeiros do segundo.

### 9.3 Função de custo normalizada

A função de custo deve normalizar cada termo:

```python
def merge_cost(gap, angle_rad, offset_m, curvature_diff,
               max_gap_m, max_angle_rad, max_offset_m,
               max_curvature_diff,
               weights):
    return (
        weights["gap"] * gap / max_gap_m +
        weights["angle"] * angle_rad / max_angle_rad +
        weights["offset"] * offset_m / max_offset_m +
        weights["curve"] * curvature_diff / max_curvature_diff
    )
```

Termos com denominador zero devem ser rejeitados ou protegidos explicitamente.

### 9.4 Resolução dos merges

1. Ordenar candidatos por menor custo.
2. Aplicar merges sem reutilizar a mesma ponta em múltiplas conexões.
3. Recalcular candidatos quando a geometria resultante mudar significativamente.
4. Rejeitar conexões que atravessem outra fileira.
5. Rejeitar conexões que saiam do corredor plausível da máscara.
6. Registrar cada merge com custo e critérios utilizados.

### 9.5 Conexão suave

Não concatenar os fragmentos diretamente quando houver gap relevante. Inserir uma transição suave que respeite:

- endpoint inicial;
- endpoint final;
- tangente de entrada;
- tangente de saída;
- limite de curvatura;
- corredor da máscara.

A conexão deve ser validada antes de ser aceita.

## 10. Fase 4 — Melhorias na inferência

Arquivo:

```text
src/inference.py
```

### 10.1 Seleção de tiles

Não substituir diretamente:

```python
if np.sum(talhao_tile == 255) > 0:
```

por:

```python
if np.mean(mask) > threshold:
```

Essa substituição é incorreta quando a máscara usa valores `0` e `255`, pois `threshold` normalmente está em `[0, 1]`. Além disso, exigir 50% de cobertura pode eliminar tiles legítimos de borda.

Usar uma fração configurável:

```python
coverage = np.mean(talhao_tile > 0)
if coverage >= min_tile_coverage:
    coords.append((y, x))
```

O padrão deve preservar a compatibilidade atual, processando qualquer tile que intercepte o talhão. Um valor baixo, como `0.01` ou `0.05`, deve ser calibrado empiricamente.

### 10.2 Batch configurável

Adicionar ao final da assinatura, preservando chamadas existentes:

```python
def run_inference(..., logger=print, batch_size=4):
```

Aplicar a mesma regra a `run_segmented_inference_pipeline`, se necessário.

O batch deve ser validado contra a memória disponível. Também avaliar:

- `torch.inference_mode()`;
- transferência assíncrona;
- `pin_memory` quando aplicável;
- autocast somente quando suportado pelo dispositivo.

### 10.3 Pós-processamento probabilístico

Considerar suavização espacial ou otimização de caminho diretamente no `prob_map`, mas somente depois de estabelecer o baseline das máscaras binárias.

CRF ou métodos equivalentes devem ser introduzidos apenas se houver benefício mensurável. O custo computacional e o comportamento nas bordas devem ser avaliados.

## 11. Fase 5 — Parametrização por GSD

Os parâmetros atualmente definidos em pixels devem ser convertidos a partir do GSD do raster quando representarem efeitos físicos:

- `close_size`;
- `open_size`;
- resolução de rasterização;
- comprimento mínimo de ramo;
- passo de reamostragem;
- tolerâncias de endpoint;
- distância de deduplicação;
- gap máximo de merge.

Exemplo:

```python
kernel_px = max(1, round(kernel_m / gsd_m))
```

O tamanho final do kernel deve ser ajustado para uma dimensão ímpar quando exigido pelo operador morfológico.

Parâmetros de merge e deduplicação que já são métricos devem permanecer em metros. Não converter metros para pixels fora das operações raster.

## 12. Critérios quantitativos de aceite

Os valores finais devem ser definidos após medir o baseline, mas a aprovação deve seguir critérios mensuráveis:

| Critério | Condição mínima recomendada |
|---|---|
| Erro médio de centro | Redução em relação ao baseline, inicialmente alvo de pelo menos 10% |
| Percentil 95 do erro de centro | Não pode piorar em linhas retas e deve reduzir em curvas |
| Erro de endpoint | Abaixo de tolerância definida pelo GSD e pela aplicação agronômica |
| Cobertura longitudinal | Aumento sem crescimento significativo de falsos positivos |
| Falso merge | Redução em relação ao baseline |
| Duplicatas | Redução em relação ao baseline |
| Linhas retas | Nenhuma degradação superior a 5% nas métricas principais |
| Geometrias válidas | 100% das geometrias exportadas válidas |
| Permanência no talhão | 100% ou desvio máximo explicitamente tolerado |
| Desempenho | Limite definido por hectare, número de fragmentos e memória disponível |

Os percentuais são metas iniciais. A decisão final deve considerar a distribuição por talhão, e não apenas a média global.

## 13. Casos de teste obrigatórios

O conjunto de validação deve conter:

1. fileiras retas paralelas;
2. curvas suaves;
3. curvas com raio pequeno;
4. máscaras com falhas internas;
5. pontas próximas ao limite do talhão;
6. polígonos com largura variável;
7. componentes próximos pertencentes a fileiras distintas;
8. bifurcações artificiais produzidas por ruído;
9. talhões côncavos;
10. talhões adjacentes ou sobrepostos;
11. CRS projetado;
12. CRS geográfico;
13. diferentes GSDs;
14. imagens constantes ou com `nodata` relevante;
15. tiles parcialmente cobertos pelo talhão.

Cada caso deve salvar máscara, polígonos, centerlines intermediárias, resultado final e métricas.

## 14. Estratégia de rollout e fallback

Cada nova etapa deve ser controlada por configuração, por exemplo:

```yaml
centerline_method: skeleton_v2
prune_skeleton: true
resample_step_m: 1.0
smooth_centerline: true
validate_inside_polygon: true
merge_method: spatial_cost
merge_enabled: true
dedup_method: overlap_parallelism
dedup_enabled: true
min_tile_coverage: 0.01
batch_size: 4
```

Valores recomendados:

- `skeleton_v2`: novo método;
- `skeleton_legacy`: comportamento atual;
- `spatial_cost`: novo merge;
- `greedy_legacy`: merge atual;
- `overlap_parallelism`: nova deduplicação;
- `distance_legacy`: deduplicação atual.

O fallback deve ser acionável sem alterar o código-fonte.

## 15. Conclusão

O plano é tecnicamente viável, mas a implementação deve respeitar cinco cuidados centrais:

1. **Usar unidades métricas de forma explícita.** Poda, reamostragem, distâncias e parâmetros físicos não devem depender implicitamente de pixels.
2. **Não usar PCA como substituto do caminho geodésico em curvas.** PCA deve ser apenas auxiliar.
3. **Não corrigir linhas fora do polígono apenas projetando pontos isolados.** A linha inteira precisa ser validada.
4. **Normalizar os termos do custo de merge.** Metros, ângulos e curvatura não podem ser somados diretamente sem escala comum.
5. **Não usar `np.mean(mask) > threshold` sem normalizar a máscara e definir a cobertura mínima.** Essa alteração pode eliminar tiles de borda e quebrar o comportamento atual.

A implementação deve começar pela instrumentação e pelo baseline. Depois devem ser feitas as correções de segurança, a melhoria da centerline, a deduplicação e, somente então, o merge otimizado. O sucesso deve ser determinado por métricas de erro, continuidade, falso merge, duplicação, validade geométrica e desempenho.

## Referências do projeto

[1]: ./src/postprocess.py "Código de pós-processamento, extração de linhas, união e deduplicação"

[2]: ./src/inference.py "Código de inferência por janelas e processamento segmentado"

[3]: ./configs/post.yaml "Configuração de pós-processamento"

[4]: ./requirements.txt "Dependências declaradas do projeto"
