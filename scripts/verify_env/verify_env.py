"""
verify_env.py — Verifica o ambiente antes de rodar o app.
Testa: CUDA, dados sintéticos, pré-processamento e geração de tiles.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

def check_torch():
    print("=" * 50)
    print("1. TORCH & CUDA")
    import torch
    print(f"   PyTorch  : {torch.__version__}")
    print(f"   CUDA OK  : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"   GPU Name : {props.name}")
        print(f"   VRAM     : {props.total_memory / 1024**3:.2f} GB")
    else:
        print("   [AVISO] CUDA não disponível — treino usará CPU!")

def check_geo():
    print("=" * 50)
    print("2. BIBLIOTECAS GEOESPACIAIS")
    import rasterio, geopandas, shapely, pyproj
    print(f"   rasterio   : {rasterio.__version__}")
    print(f"   geopandas  : {geopandas.__version__}")
    print(f"   shapely    : {shapely.__version__}")
    print(f"   pyproj     : {pyproj.__version__}")

def check_gpkg():
    print("=" * 50)
    print("3. LEITURA DO GEOPACKAGE 63996.gpkg")
    import rasterio
    if not os.path.exists("63996.gpkg"):
        print("   [ERRO] 63996.gpkg não encontrado!")
        return
    with rasterio.open("63996.gpkg") as src:
        print(f"   CRS    : {src.crs}")
        print(f"   Bounds : {src.bounds}")
        print(f"   Bands  : {src.count}")
        print(f"   Width  : {src.width}")
        print(f"   Height : {src.height}")

def generate_and_check_synthetic():
    print("=" * 50)
    print("4. GERAÇÃO DE DADOS SINTÉTICOS")
    from src.generate_synthetic_data import generate_data
    generate_data(output_dir="data/synthetic")
    print("   Dados sintéticos gerados com sucesso!")
    import os
    for f in ["data/synthetic/talhao.geojson", "data/synthetic/linhas_centrais.geojson"]:
        size = os.path.getsize(f)
        print(f"   {f}  ({size} bytes)")

def check_preprocess():
    print("=" * 50)
    print("5. TESTE DE PRÉ-PROCESSAMENTO (dados sintéticos)")
    from src.preprocess import run_preprocessing
    result = run_preprocessing(
        raster_path="63996.gpkg",
        lines_path="data/synthetic/linhas_centrais.geojson",
        talhao_path="data/synthetic/talhao.geojson",
        buffer_size=1.5,
        tile_size=256,
        stride=128
    )
    print(f"   Tiles gerados   : {result['num_tiles']}")
    print(f"   Treino (train)  : {result['train_count']}")
    print(f"   Validação (val) : {result['val_count']}")
    print(f"   Tamanho da área : {result['width']}x{result['height']} pixels")

if __name__ == "__main__":
    check_torch()
    check_geo()
    check_gpkg()
    generate_and_check_synthetic()
    check_preprocess()
    print("=" * 50)
    print("[OK] AMBIENTE VERIFICADO COM SUCESSO -- execute run.bat para iniciar a interface web.")
    print("=" * 50)
