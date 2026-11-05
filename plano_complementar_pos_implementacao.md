# Plano complementar pós-implementação

## Ajustes finais para produção: curvas, robustez e rastreabilidade

## 1. Contexto

O plano técnico anterior já foi implementado pela IDE. Este documento **não substitui nem repete o plano anterior**. Ele define somente os complementos que devem ser adicionados depois que a implementação atual estiver concluída e validada.

O objetivo é elevar o pipeline de uma solução geométrica melhorada para uma solução mais robusta em produção, principalmente nos casos de:

- curvas reais;
- máscaras ruidosas ou assimétricas;
- falhas entre árvores;
- pontas próximas às bordas;
- fileiras vizinhas próximas;
- múltiplos talhões;
- regressões entre versões do algoritmo.

A implementação deve ser incremental. Cada item deve ser habilitado por configuração e comparado com a versão já implementada.

## 2. Regra principal

Não reimplementar o que já foi entregue no plano anterior.

Antes de iniciar qualquer alteração, verificar se já existem no código:

- reamostragem por comprimento;
- poda do esqueleto;
- suavização baseada no arco;
- validação dentro do polígono;
- deduplicação preliminar e final;
- merge espacial com custo;
- parametrização por GSD;
- tratamento de `nodata` e divisão por zero;
- métricas básicas de erro, cobertura e endpoints.

Caso um item já exista, somente complementar testes, logs, validação ou integração. Não criar uma segunda implementação concorrente.

## 3. Complementos obrigatórios

### 3.1 Refinamento da centerline usando o `prob_map`

#### Lacuna

A centerline pode permanecer deslocada em relação à região de maior probabilidade da rede, mesmo quando está dentro do polígono segmentado. O polígono contém informação binária; o `prob_map` contém informação contínua que pode melhorar a posição do eixo.

#### Implementação

Adicionar uma etapa opcional depois da extração, reamostragem e suavização da centerline:

```python
def refine_line_with_prob_map(
    line,
    prob_map,
    transform,
    polygon=None,
    weight=0.5,
    search_radius_m=1.5,
    step_m=1.0,
    max_shift_m=0.75,
):
    """Refina a linha usando a probabilidade local sem perder continuidade."""
```

#### Estratégia recomendada

Para cada ponto amostrado por comprimento:

1. calcular a tangente local;
2. obter a direção normal;
3. amostrar um perfil transversal limitado por `search_radius_m`;
4. converter as coordenadas para índices do `prob_map`;
5. calcular a probabilidade ao longo do perfil;
6. escolher um deslocamento candidato limitado por `max_shift_m`;
7. penalizar deslocamentos grandes em relação à linha original;
8. penalizar variação abrupta entre deslocamentos consecutivos;
9. rejeitar candidatos que saiam do polígono ou corredor válido;
10. reamostrar e suavizar levemente a linha refinada;
11. comparar a qualidade da linha refinada com a original;
12. usar fallback para a linha original quando a melhoria não for consistente.

O refinamento não deve simplesmente mover cada ponto para o máximo local independente. Essa estratégia pode saltar para uma fileira vizinha ou produzir zig-zag.

#### Função objetivo sugerida

```text
custo total =
    - probabilidade média sobre a linha
    + penalidade de deslocamento
    + penalidade de curvatura
    + penalidade de descontinuidade
    + penalidade por sair do corredor
```

#### Métricas obrigatórias

Registrar antes e depois do refinamento:

- probabilidade média na linha;
- mediana da probabilidade;
- percentil 10 da probabilidade;
- fração de amostras acima do limiar;
- deslocamento médio e máximo;
- erro de centro, quando houver referência;
- mudança no comprimento;
- mudança na curvatura.

O refinamento só deve ser aceito quando melhorar a qualidade probabilística sem ultrapassar os limites geométricos.

### 3.2 Métrica de alinhamento com o `prob_map`

Adicionar ao módulo de avaliação:

```python
def compute_prob_alignment(
    line,
    prob_map,
    transform,
    buffer_m=0.5,
    step_m=1.0,
):
    """Mede o alinhamento da linha com regiões de alta probabilidade."""
```

A métrica deve amostrar a linha por comprimento e retornar pelo menos:

```python
{
    "mean": ...,
    "median": ...,
    "p10": ...,
    "p25": ...,
    "fraction_above_threshold": ...,
}
```

A avaliação deve ser feita em um corredor estreito em torno da linha, e não somente nos vértices originais.

Também é recomendável comparar a linha com perfis paralelos deslocados. Isso ajuda a verificar se o alinhamento obtido é realmente melhor do que deslocamentos laterais próximos.

### 3.3 Controle explícito de curvatura e anti-zig-zag

#### Lacuna

A suavização pode reduzir ruído, mas não garante limite de curvatura. Uma linha pode continuar apresentando microcurvas artificiais, oscilações ou inversões de direção.

#### Funções

Adicionar, caso ainda não existam:

```python
def compute_local_curvature(line, step_m=1.0):
    """Calcula curvatura local em unidades métricas."""


def enforce_curvature_limit(
    line,
    max_curvature,
    step_m=1.0,
    preserve_endpoints=True,
):
    """Reduz picos de curvatura sem eliminar curvas reais."""
```

#### Regras

- calcular curvatura em uma linha reamostrada por comprimento;
- detectar picos isolados e sequências de microcurvas;
- preservar curvas sustentadas ao longo de vários pontos;
- preservar endpoints;
- limitar o deslocamento máximo durante a correção;
- evitar alteração excessiva do comprimento;
- validar novamente contra polígono, corredor e `prob_map`.

#### Métricas de suavidade

Registrar:

- curvatura média;
- curvatura mediana;
- percentil 95 da curvatura;
- curvatura máxima;
- número de inversões de direção;
- número de microcurvas;
- variação angular por metro.

Não usar somente um limiar fixo sem considerar o GSD e o tipo de plantio. O limite deve ser configurável por perfil.

### 3.4 Validação contra corredor plausível da fileira

#### Lacuna

Uma linha pode estar dentro do polígono, mas posicionada em uma borda lateral ou em uma região que não representa o eixo da fileira.

#### Função

Adicionar ou complementar:

```python
def validate_line_corridor(
    line,
    polygon,
    prob_map=None,
    transform=None,
    max_width_m=None,
    step_m=1.0,
):
    """Valida se a linha permanece em um corredor plausível da fileira."""
```

O corredor deve considerar, quando disponíveis:

- largura local da máscara;
- direção local da linha;
- distância aos limites laterais do polígono;
- probabilidade local;
- distância às fileiras vizinhas;
- continuidade do corredor ao longo do percurso.

A validação não deve presumir que o centro geométrico do polígono é sempre o centro agronômico. Copas assimétricas podem deslocar o centro geométrico.

A função deve retornar diagnóstico, não somente `True` ou `False`:

```python
{
    "valid": True,
    "inside_fraction": ...,
    "max_outside_distance_m": ...,
    "mean_lateral_margin_m": ...,
    "prob_alignment": ...,
    "reason": None,
}
```

### 3.5 Controle topológico do conjunto final

#### Lacuna

As validações individuais não garantem que o conjunto final de linhas seja coerente. Após merge e deduplicação, podem permanecer cruzamentos, sobreposições ou inversões de ordem.

#### Função

Adicionar:

```python
def validate_topology(
    lines,
    talhao_geom=None,
    expected_spacing_m=None,
    tolerance_m=1.0,
):
    """Valida a coerência espacial e agrícola do conjunto de linhas."""
```

#### Verificações

- linhas que se cruzam de forma anômala;
- sobreposição de linhas por grande extensão;
- convergência excessiva entre fileiras;
- inversão da ordem local das fileiras;
- espaçamento local incompatível;
- duplicatas residuais;
- linhas fora do talhão;
- linhas excessivamente curtas;
- gaps incompatíveis com a continuidade esperada.

A função deve classificar as linhas em:

- aceitas;
- suspeitas;
- rejeitadas;

sem apagar automaticamente uma geometria sem registrar o motivo.

O relatório topológico deve conter:

```python
{
    "num_lines": ...,
    "num_crossings": ...,
    "num_overlaps": ...,
    "num_spacing_anomalies": ...,
    "num_outside_talhao": ...,
    "num_suspect": ...,
    "warnings": [...],
}
```

## 4. Estabilidade numérica transversal

Este item deve ser aplicado a todas as funções novas e existentes.

### Proteções obrigatórias

- usar epsilon em divisões;
- rejeitar linhas com menos de dois pontos;
- exigir número mínimo de pontos para calcular curvatura;
- proteger normalização de vetores de comprimento zero;
- limitar argumentos de `arccos` para `[-1, 1]`;
- proteger `np.percentile` em arrays vazios;
- tratar geometrias degeneradas;
- tratar componentes sem endpoints válidos;
- limitar dimensões de rasterização;
- validar valores `NaN` e infinitos;
- manter fallback para a geometria anterior.

Toda proteção deve registrar o motivo quando alterar ou descartar um resultado.

## 5. Performance: complementos condicionais

### 5.1 Redução do NetworkX

Se o perfil de execução confirmar que NetworkX é gargalo, avaliar a substituição do grafo por:

- arrays NumPy;
- componentes conectados de `scipy.ndimage` ou `skimage`;
- representação esparsa;
- algoritmo especializado para caminhos no skeleton.

A troca não deve ser feita apenas por expectativa. O comportamento deve ser comparado com o algoritmo atual em máscaras sintéticas e reais.

### 5.2 Rasterização por polígono

Não substituir automaticamente o processamento por:

```text
máscara global → skeleton → separar depois
```

Essa abordagem pode criar conexões artificiais entre fileiras vizinhas antes da separação.

Ela só deve ser considerada se houver:

- separação confiável entre fileiras;
- controle dos fechamentos morfológicos;
- poda topológica robusta;
- método de particionamento posterior;
- testes específicos contra falsos merges.

Como primeira alternativa, otimizar o processamento por componente, limitar a janela ao bounding box com margem e reutilizar estruturas raster quando possível.

### 5.3 Paralelização

A paralelização por talhão deve ser introduzida somente após medir o custo e a memória.

Regras:

- limitar número de processos;
- evitar múltiplos processos disputando a mesma GPU;
- controlar escrita concorrente;
- preservar logs por talhão;
- tornar o resultado determinístico quando possível;
- comparar tempo total, RAM e memória GPU.

## 6. Controle de experimentos e rastreabilidade

Criar um arquivo por execução:

```text
outputs/run_metadata.json
```

Formato mínimo:

```json
{
  "algorithm_version": "centerline-v2.1",
  "timestamp_utc": "...",
  "centerline_method": "skeleton_pruned_prob_refined",
  "merge_method": "spatial_cost",
  "dedup_method": "overlap_parallelism",
  "profile": "curvas",
  "parameters": {
    "resample_step_m": 1.0,
    "max_curvature": 0.08,
    "max_gap_m": 12.0,
    "max_offset_m": 1.5,
    "max_angle_deg": 25.0,
    "prob_refinement_enabled": true,
    "prob_search_radius_m": 1.5,
    "min_tile_coverage": 0.01,
    "batch_size": 4
  },
  "input": {
    "raster": "...",
    "talhao": "...",
    "checkpoint": "...",
    "crs": "...",
    "gsd_m": null
  },
  "counts": {
    "num_polygons": 0,
    "num_centerlines": 0,
    "num_merges": 0,
    "num_deduplicated_pre": 0,
    "num_deduplicated_final": 0,
    "num_centerline_failures": 0,
    "num_topology_warnings": 0
  },
  "metrics": {
    "center_error_mean_m": null,
    "center_error_p95_m": null,
    "length_coverage": null,
    "endpoint_error_mean_m": null,
    "prob_alignment_mean": null,
    "curvature_p95": null,
    "spacing_error_mean_m": null
  },
  "performance": {
    "elapsed_seconds": null,
    "peak_ram_mb": null,
    "peak_gpu_mb": null
  }
}
```

Os parâmetros efetivamente usados devem ser registrados após a resolução da configuração, e não apenas os valores padrão.

## 7. Critérios de aceite adicionais

Além dos critérios já implementados no plano anterior, validar:

| Critério | Condição de aceite |
|---|---|
| Alinhamento probabilístico | A centerline refinada não pode reduzir a probabilidade mediana sem justificativa geométrica |
| Curvatura | Redução de microcurvas sem eliminar curvas reais |
| Zig-zag | Redução de inversões angulares anômalas |
| Corredor | Fração dentro do corredor acima do limite definido |
| Topologia | Nenhum cruzamento anômalo não classificado |
| Espaçamento | Redução de anomalias locais sem apagar fileiras válidas |
| Reprodutibilidade | Mesma configuração produz métricas equivalentes entre execuções |
| Rastreamento | Cada saída possui metadados de versão e parâmetros |
| Fallback | Falha do refinamento retorna a centerline anterior sem interromper o pipeline |

Os valores absolutos devem ser definidos a partir do baseline já produzido pelo plano anterior. Não usar a afirmação de que o plano anterior resolveu “80%” sem benchmark.

## 8. Ordem de implementação depois do plano anterior

Implementar nesta ordem:

1. **Verificar o que já foi implementado e evitar duplicação.**
2. **Adicionar métricas de alinhamento com o `prob_map`.**
3. **Adicionar refinamento opcional da centerline pelo `prob_map`.**
4. **Adicionar cálculo e controle explícito de curvatura.**
5. **Adicionar validação contra corredor plausível.**
6. **Adicionar validação topológica do conjunto final.**
7. **Adicionar proteções numéricas transversais.**
8. **Adicionar metadados e controle de experimentos.**
9. **Medir performance e somente então avaliar otimizações estruturais.**
10. **Validar regressão em linhas retas, curvas e casos degenerados.**

## 9. Estratégia de ativação

Adicionar configurações independentes para cada complemento:

```yaml
prob_refinement_enabled: false
prob_refinement_weight: 0.5
prob_search_radius_m: 1.5
prob_max_shift_m: 0.75
curvature_control_enabled: false
max_curvature: null
corridor_validation_enabled: false
topology_validation_enabled: true
write_run_metadata: true
```

Durante a validação:

1. executar com todos os complementos desativados;
2. ativar somente o alinhamento probabilístico;
3. ativar refinamento da centerline;
4. ativar controle de curvatura;
5. ativar corredor;
6. ativar validação topológica;
7. comparar métricas e produtos em cada etapa.

Nenhuma etapa deve ser ativada globalmente antes de comparar seu efeito nos casos de teste.

## 10. Conclusão

O plano anterior deve ser considerado a base já implementada. Este documento cobre somente as lacunas adicionais mais relevantes para produção:

- uso efetivo do `prob_map` na centerline;
- métrica de alinhamento probabilístico;
- controle explícito de curvatura e anti-zig-zag;
- validação contra corredor plausível;
- controle topológico do conjunto final;
- estabilidade numérica transversal;
- rastreabilidade de versões, parâmetros e métricas.

A proposta de skeleton global seguido de separação posterior não deve ser adotada automaticamente. A paralelização também deve ser condicionada a medições de memória e desempenho.

O resultado final deve ser aceito somente quando demonstrar melhoria mensurável em curvas e pontas sem regressão significativa em linhas retas, mantendo geometrias válidas, rastreabilidade completa e fallback seguro.

## Referências do projeto

[1]: ./src/postprocess.py "Código de pós-processamento, extração de linhas, união e deduplicação"

[2]: ./src/inference.py "Código de inferência por janelas e processamento segmentado"

[3]: ./configs/post.yaml "Configuração de pós-processamento"

[4]: ./requirements.txt "Dependências declaradas do projeto"
