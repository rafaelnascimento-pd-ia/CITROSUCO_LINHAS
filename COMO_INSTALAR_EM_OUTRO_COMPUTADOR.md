# 🚀 Guia de Instalação e Execução em Outro Computador
## Projeto IA Linhas Citrosuco (Treinamento, Inferência e Vetorização)

Este pacote contém a solução completa para detecção, vetorização e interpolação de entrelinhas de citros com Inteligência Artificial, incluindo:
- **Código-fonte completo**: Pré-processamento, Treinamento, Inferência, Vetorização Geoespacial e Interface Web.
- **Modelos Treinados (`models/`)**: Checkpoints das redes neurais (`best_model.pth`, modelos de 9 ortofotos, etc.).
- **Dataset Bruto (`data_set/`)**: Ortofotos de alta resolução e shapefiles de linhas de plantio das fazendas.
- **Patches Processados (`data/`)**: Conjuntos de treino, validação e teste fatiados em patches.
- **Casos de Inferência (`inferencia/`)**: Execuções de teste com ortomosaicos e saídas vetoriais (Shapefile/GeoJSON).

---

## 📋 1. Pré-Requisitos no Novo Computador

1. **Sistema Operacional**: Windows 10 ou 11 (64 bits).
2. **Python**: Versão **3.10**, **3.11** ou **3.12** instalada.
   - [Download do Python oficial](https://www.python.org/downloads/)
   - ⚠️ **MUITO IMPORTANTE**: Na primeira tela do instalador do Python, marque a caixa:
     ☑️ **"Add python.exe to PATH"** (Adicionar python ao PATH) antes de clicar em "Install Now".
3. **Hardware Recomendado**:
   - **Placa de Vídeo (GPU)**: Placa NVIDIA (GTX 1660, RTX 2060/3060/4060 ou superior) com drivers atualizados para aceleração CUDA.
   - *Nota*: Se o computador não tiver GPU NVIDIA, o código funcionará normalmente utilizando o processador (CPU), embora o tempo de treinamento seja mais demorado.
   - **Memória RAM**: Recomendado 16 GB ou mais.
   - **Espaço em Disco**: Pelo menos 40 a 50 GB livres para descompactação completa do projeto com datasets e modelos.

---

## ⚡ 2. Instalação Rápida (1 Clique)

1. Extraia todo o conteúdo do arquivo ZIP em uma pasta do computador (exemplo: `C:\Projetos\projeto_ia_linhas_citrosuco` ou `D:\projeto_ia_linhas`).
   > 💡 **Dica**: Evite caminhos com caracteres especiais, acentos ou espaços excessivos.
2. Na pasta do projeto, localize e dê um **duplo-clique** no arquivo:
   ```text
   1_INSTALAR_DEPENDENCIAS.bat
   ```
3. O instalador automático irá:
   - Identificar o Python do seu computador (Python oficial, py launcher ou Anaconda).
   - Criar o ambiente virtual isolado `.venv`.
   - Atualizar o gerenciador de pacotes `pip`.
   - Instalar o PyTorch com aceleração CUDA 12.1 e todas as dependências (`requirements.txt`).
   - Testar o ambiente e confirmar se a GPU foi reconhecida.
4. Ao final, pressione qualquer tecla para fechar. O ambiente estará 100% pronto!

---

## 🔍 3. Teste do Ambiente

Para confirmar que tudo está instalado corretamente, dê um duplo-clique em:
```text
3_TESTAR_AMBIENTE.bat
```
Ele exibirá a versão do PyTorch, se a GPU CUDA está ativa e se as bibliotecas geoespaciais (`rasterio`, `geopandas`, `shapely`, `cv2`) foram importadas com sucesso.

---

## 🖥️ 4. Como Executar o Projeto

### A) Interface Web com Painel Interativo (Recomendado)
Para usar a interface gráfica no navegador:
1. Dê um duplo-clique em:
   ```text
   2_INICIAR_INTERFACE_WEB.bat
   ```
   *(Ou abra o terminal e digite: `.venv\Scripts\python.exe app.py`)*
2. O navegador abrirá automaticamente em:
   ```text
   http://localhost:5000
   ```
3. Pela interface web você pode:
   - Visualizar métricas e gráficos de treinamento.
   - Disparar treinamentos e acompanhar a barra de progresso em tempo real.
   - Rodar inferência em lote ou em ortomosaicos individuais.
   - Visualizar as predições e os shapes gerados.

---

### B) Execução via Linha de Comando (Terminal / CLI)

Abra o prompt de comando (CMD ou PowerShell) na pasta do projeto e use os comandos abaixo:

#### 1. Treinamento com Todos os Datasets das Fazendas:
```bash
.venv\Scripts\python.exe main.py train_multi --config configs/train_all_datasets.yaml
```

#### 2. Treinamento com Dataset Único:
```bash
.venv\Scripts\python.exe main.py train --config configs/train.yaml
```

#### 3. Inferência em Lote de Ortomosaicos:
```bash
.venv\Scripts\python.exe main.py infer_batch --config configs/infer_batch.yaml
```

#### 4. Fatiamento / Pré-processamento de Novo Ortomosaico:
```bash
.venv\Scripts\python.exe main.py preprocess --config configs/preprocess.yaml
```

#### 5. Algoritmo de Entrelinhas Curvas e Interpolação Robusta:
```bash
.venv\Scripts\python.exe scripts\calcular_entrelinhas_curvas\calcular_entrelinhas_curvas_robusto.py
```

---

## 🛠️ 5. Configuração no VS Code ou Cursor

Se for programar ou editar o código:
1. Abra a pasta do projeto no VS Code: `File` -> `Open Folder...`.
2. Pressione `Ctrl + Shift + P` e digite:
   `Python: Select Interpreter`
3. Selecione o interpretador do ambiente virtual recém-criado:
   `.\.venv\Scripts\python.exe`
4. O terminal integrado do VS Code ativará o ambiente automaticamente!

---

## 📁 6. Estrutura do Pacote

```text
projeto_ia_linhas_citrosuco/
├── 1_INSTALAR_DEPENDENCIAS.bat           -> Instalador automático de 1 clique
├── 2_INICIAR_INTERFACE_WEB.bat           -> Atalho para abrir o painel web no navegador
├── 3_TESTAR_AMBIENTE.bat                 -> Atalho para teste de GPU e bibliotecas
├── COMO_INSTALAR_EM_OUTRO_COMPUTADOR.md  -> Este guia completo
├── LEIAME_INSTALACAO.txt                 -> Guia rápido em texto puro
├── requirements.txt                      -> Dependências Python
├── app.py                                -> Servidor Flask e API da Interface Web
├── main.py                               -> CLI principal do projeto
│
├── configs/                              -> Arquivos de configuração (.yaml)
│   ├── train.yaml                        -> Configuração de treino único
│   ├── train_all_datasets.yaml           -> Configuração de treino multi-fazenda
│   ├── infer_batch.yaml                  -> Configuração de inferência em lote
│   └── preprocess.yaml                   -> Configuração de recorte de patches
│
├── models/                               -> Checkpoints e pesos dos modelos treinados (.pth)
│   ├── best_model.pth                    -> Melhor modelo geral para inferência
│   └── best_model_*_9orthos.pth          -> Modelos treinados com 9 ortofotos
│
├── data_set/                             -> Datasets brutos (ortofotos TIF + shapefiles das fazendas)
│   ├── 63995/
│   ├── 63996/
│   ├── 64000/
│   ├── 64001/
│   ├── ...
│   └── 86508/
│
├── data/                                 -> Patches fatiados para Treino/Val/Teste
│   ├── train/
│   ├── val/
│   └── test/
│
├── inferencia/                           -> Casos de inferência executados (TIFs, predições e SHPs)
│
├── src/                                  -> Módulos centrais em Python
│   ├── dataset.py                        -> Carregador de dados e augmentations
│   ├── model.py                          -> Definição da arquitetura Unet / DeepLabV3+
│   ├── train.py                          -> Loop de treino, validação e salvamento de pesos
│   ├── inference.py                      -> Predição sobre ortomosaicos grandes em tiles
│   ├── preprocess.py                     -> Extração de patches de ortofoto + máscara de linhas
│   └── postprocess.py                    -> Vetorização de raster para Shapefile / GeoJSON
│
├── scripts/                              -> Scripts auxiliares e pós-processamentos especializados
│   └── calcular_entrelinhas_curvas/      -> Pós-processador geométrico de entrelinhas
│
└── static/                               -> Interface gráfica web (HTML5, CSS3, JS)
```

---

## ❓ 7. Solução de Problemas Comuns

1. **"python não é reconhecido como um comando interno..."**:
   - O Python não foi adicionado ao PATH do Windows durante a instalação. Reinstale o Python e lembre-se de marcar ☑️ **"Add python.exe to PATH"**.

2. **"CUDA Disponível: False" (Executando em CPU)**:
   - Se você tem uma GPU NVIDIA, verifique se instalou os drivers oficiais da NVIDIA mais recentes pelo site da NVIDIA ou GeForce Experience.

3. **Erro de permissão ou caminho longo no Windows**:
   - Recomendamos colocar a pasta em caminhos curtos como `C:\Projetos\` ou `D:\Projetos\`.
   - Se o Windows alertar sobre nomes de arquivo longos, habilite o suporte a *Long Paths* no Windows (`reg add "HKLM\SYSTEM\CurrentControlSet\Control\FileSystem" /v LongPathsEnabled /t REG_DWORD /d 1 /f` como Administrador).
