import sys
import os

# Ensure the src module can be found
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.preprocess import run_preprocessing
from src.train import train_model

def main():
    print("Iniciando processamento em lote...")
    
    # 1. Processar 63996 (Append = False)
    print("\n--- Processando 63996 ---")
    run_preprocessing(
        raster_path="63996.gpkg",
        lines_path=os.path.join("data_set", "63996", "linhas.shp"),
        talhao_path=os.path.join("data_set", "63996", "talhao.shp"),
        output_dir="data",
        buffer_size=1.5,
        tile_size=256,
        stride=128,
        append=False
    )
    
    # 2. Processar 64000 (Append = True)
    print("\n--- Processando 64000 ---")
    run_preprocessing(
        raster_path="64000.gpkg",
        lines_path=os.path.join("data_set", "64000", "linhas.shp"),
        talhao_path=os.path.join("data_set", "64000", "talhoes.shp"),
        output_dir="data",
        buffer_size=1.5,
        tile_size=256,
        stride=128,
        append=True
    )
    
    # 3. Treinar Modelo
    print("\n--- Iniciando Treinamento ---")
    train_model(
        data_dir="data",
        encoder_name="resnet18",
        batch_size=2,      # Optimizado para RTX 2050 4GB
        epochs=20,         # Recomendado
        lr=0.0001
    )
    
    print("\nTREINAMENTO CONCLUIDO COM SUCESSO!")

if __name__ == "__main__":
    main()
