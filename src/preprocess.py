import os
import shutil
import glob
import json
import datetime
import numpy as np
import geopandas as gpd
import rasterio
import rasterio.mask
import rasterio.features
from rasterio.windows import Window
from shapely.geometry import mapping, box
from sklearn.model_selection import train_test_split
import cv2

def run_preprocessing(raster_path, lines_path, talhao_path, output_dir="data", 
                      buffer_size=1.5, tile_size=256, stride=128, val_size=0.15, test_size=0.10,
                      append=False, logger=print):
    """
    Runs the entire preprocessing pipeline:
    1. Read and reproject vectors.
    2. Crop orthomosaic by talhao.
    3. Generate buffer for centerlines.
    4. Rasterize buffer into a binary mask.
    5. Extract tiles (256x256 or 512x512).
    6. Split into train (75%), validation (15%) and test (10%) folders.
    If append is True, it adds tiles to existing data instead of clearing the folder.
    """
    logger(f"Iniciando pré-processamento...")
    logger(f"  Raster: {raster_path}")
    logger(f"  Linhas: {lines_path}")
    logger(f"  Talhão: {talhao_path}")
    logger(f"  Buffer das linhas: {buffer_size}m")
    logger(f"  Tamanho do tile: {tile_size}x{tile_size}")
    logger(f"  Passo (stride): {stride}")
    
    # Paths for output tiles
    tiles_img_dir = os.path.join(output_dir, "tiles", "images")
    tiles_mask_dir = os.path.join(output_dir, "tiles", "masks")
    info_path = os.path.join(output_dir, "dataset_info.json")
    
    # Clear previous tiles if not appending
    if not append:
        if os.path.exists(os.path.join(output_dir, "tiles")):
            shutil.rmtree(os.path.join(output_dir, "tiles"))
        for split in ["train", "val", "test"]:
            split_dir = os.path.join(output_dir, split)
            if os.path.exists(split_dir):
                shutil.rmtree(split_dir)
        if os.path.exists(info_path):
            os.remove(info_path)
    else:
        logger("Modo APPEND: adicionando tiles ao dataset existente.")
                
    os.makedirs(tiles_img_dir, exist_ok=True)
    os.makedirs(tiles_mask_dir, exist_ok=True)
    
    # 1. Open orthomosaic and get its CRS
    with rasterio.open(raster_path) as src:
        raster_crs = src.crs
        logger(f"CRS do Orthomosaico: {raster_crs}")
        
        # 2. Read vectors and reproject to raster CRS
        logger("Carregando vetores...")
        talhao_gdf = gpd.read_file(talhao_path)
        lines_gdf = gpd.read_file(lines_path)
        logger(f"  CRS original talhão: {talhao_gdf.crs}")
        logger(f"  CRS original linhas: {lines_gdf.crs}")

        if talhao_gdf.crs is None or lines_gdf.crs is None:
            raise ValueError(
                "CRS indefinido em talhão ou linhas. Verifique se os arquivos GeoJSON" \
                " possuem um CRS válido antes de tentar o pré-processamento."
            )

        if raster_crs is None or not raster_crs.is_projected:
            raise ValueError(
                "O raster deve estar em um CRS projetado (por exemplo UTM) para usar buffer em metros."
            )

        if talhao_gdf.crs.is_geographic or lines_gdf.crs.is_geographic:
            logger("  Vetores estão em CRS geográfico. Reprojetando para o CRS do raster antes do buffer...")

        talhao_gdf = talhao_gdf.to_crs(raster_crs)
        lines_gdf = lines_gdf.to_crs(raster_crs)
        logger(f"  Reprojetado para CRS do raster: {raster_crs}")
        
        # Merge all talhao geometries into a single geometry if multiple
        talhao_geom = talhao_gdf.unary_union
        
        # Clip lines to talhao boundary to ensure we only buffer lines inside
        logger("Recortando linhas centrais pelo talhão...")
        lines_clipped = gpd.clip(lines_gdf, talhao_gdf)
        
        if lines_clipped.empty:
            raise ValueError("Nenhuma linha central intercepta o talhão fornecido!")
            
        # 3. Compute the tile grid from the talhao bounding box without loading everything at once
        logger("Calculando área de recorte do talhão...")
        raster_bounds = src.bounds
        t_bounds = talhao_geom.bounds  # (minx, miny, maxx, maxy)
        clipped_bounds = (
            max(t_bounds[0], raster_bounds.left),
            max(t_bounds[1], raster_bounds.bottom),
            min(t_bounds[2], raster_bounds.right),
            min(t_bounds[3], raster_bounds.top)
        )
        if clipped_bounds[0] >= clipped_bounds[2] or clipped_bounds[1] >= clipped_bounds[3]:
            raise ValueError(
                f"O talhão {t_bounds} não intercepta o raster {tuple(raster_bounds)}. "
                "Verifique se o CRS do vetor está correto (deve ser WGS 84 ou UTM)."
            )
        talhao_window = rasterio.windows.from_bounds(*clipped_bounds, transform=src.transform)
        talhao_window = talhao_window.round_offsets().round_shape(op='ceil', pixel_precision=1)
        talhao_window = talhao_window.intersection(Window(0, 0, src.width, src.height))

        window_width = int(talhao_window.width)
        window_height = int(talhao_window.height)
        if window_width == 0 or window_height == 0:
            raise ValueError("A janela calculada para o talhão é vazia. Verifique os vetores e o raster.")

        logger(f"Janela de recorte em tiles: {window_width}x{window_height}")

        # 4. Generate buffer on centerlines
        logger(f"Gerando buffer de {buffer_size}m nas linhas centrais...")
        buffered_lines = lines_clipped.geometry.buffer(buffer_size)
        
        # 5. Extract tiles directly from raster using a windowed read
        logger("Fatiando área em tiles sem carregar toda a imagem na memória...")
        tile_count = 0
        import uuid
        run_prefix = str(uuid.uuid4())[:8]

        # Tile loop uses the window origin
        start_col = int(talhao_window.col_off)
        start_row = int(talhao_window.row_off)
        end_col = start_col + window_width
        end_row = start_row + window_height

        for row in range(start_row, end_row - tile_size + 1, stride):
            for col in range(start_col, end_col - tile_size + 1, stride):
                tile_window = Window(col_off=col, row_off=row, width=tile_size, height=tile_size)
                
                # Check spatial overlap first in vector space (100x faster than reading raster)
                tile_bounds_coords = rasterio.windows.bounds(tile_window, src.transform)
                tile_box = box(*tile_bounds_coords)
                
                if not talhao_geom.intersects(tile_box):
                    continue
                    
                # Calculate vector-based overlap ratio to avoid rasterizing non-overlapping tiles
                overlap_ratio = talhao_geom.intersection(tile_box).area / tile_box.area
                if overlap_ratio < 0.10:
                    continue

                # Read the tile from raster only if it actually overlaps enough
                img_tile = src.read([1, 2, 3], window=tile_window)
                img_tile = np.transpose(img_tile, (1, 2, 0))
                if img_tile.dtype != np.uint8:
                    if img_tile.max() > 255:
                        img_tile = ((img_tile - img_tile.min()) / 
                                    (img_tile.max() - img_tile.min()) * 255).astype(np.uint8)
                    else:
                        img_tile = img_tile.astype(np.uint8)

                tile_transform = src.window_transform(tile_window)

                # Rasterize talhao and buffer mask for this tile only
                talhao_tile_mask = np.zeros((tile_size, tile_size), dtype=np.uint8)
                rasterio.features.rasterize(
                    shapes=[(talhao_geom, 255)],
                    out=talhao_tile_mask,
                    transform=tile_transform,
                    fill=0,
                    default_value=255
                )

                mask_tile = np.zeros((tile_size, tile_size), dtype=np.uint8)
                shapes = [
                    (geom, 255) for geom in buffered_lines 
                    if not geom.is_empty and geom.intersects(tile_box)
                ]
                if shapes:
                    rasterio.features.rasterize(
                        shapes=shapes,
                        out=mask_tile,
                        transform=tile_transform,
                        fill=0,
                        default_value=255
                    )

                mask_tile = cv2.bitwise_and(mask_tile, talhao_tile_mask)

                tile_id = f"tile_{run_prefix}_{row}_{col}"
                cv2.imwrite(os.path.join(tiles_img_dir, f"{tile_id}.png"), cv2.cvtColor(img_tile, cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join(tiles_mask_dir, f"{tile_id}.png"), mask_tile)
                tile_count += 1

        logger(f"Gerados {tile_count} novos tiles de treino.")
        if tile_count == 0:
            raise ValueError("Nenhum tile gerado! Verifique se as dimensões do talhão são compatíveis com o tamanho de tile.")

        # 6. Split into Train/Val/Test sets (75% / 15% / 10%)
        logger("Dividindo em conjuntos de Treinamento (75%), Validação (15%) e Teste (10%)...")
        all_imgs = glob.glob(os.path.join(tiles_img_dir, f"*{run_prefix}*.png"))
        basenames = [os.path.basename(f) for f in all_imgs]
        
        test_basenames = []
        if test_size and test_size > 0.0 and len(basenames) >= 10:
            train_val_basenames, test_basenames = train_test_split(
                basenames, test_size=test_size, random_state=42
            )
            val_ratio = val_size / (1.0 - test_size)
            train_basenames, val_basenames = train_test_split(
                train_val_basenames, test_size=val_ratio, random_state=42
            )
        else:
            train_basenames, val_basenames = train_test_split(
                basenames, test_size=val_size, random_state=42
            )
        
        # Create directories for train/val/test split
        splits_to_create = ["train", "val"]
        if test_basenames:
            splits_to_create.append("test")
            
        for split in splits_to_create:
            os.makedirs(os.path.join(output_dir, split, "images"), exist_ok=True)
            os.makedirs(os.path.join(output_dir, split, "masks"), exist_ok=True)
            
        # Move files (move instead of copy to avoid 3x disk usage)
        for name in train_basenames:
            shutil.move(os.path.join(tiles_img_dir, name), os.path.join(output_dir, "train", "images", name))
            shutil.move(os.path.join(tiles_mask_dir, name), os.path.join(output_dir, "train", "masks", name))
            
        for name in val_basenames:
            shutil.move(os.path.join(tiles_img_dir, name), os.path.join(output_dir, "val", "images", name))
            shutil.move(os.path.join(tiles_mask_dir, name), os.path.join(output_dir, "val", "masks", name))

        for name in test_basenames:
            shutil.move(os.path.join(tiles_img_dir, name), os.path.join(output_dir, "test", "images", name))
            shutil.move(os.path.join(tiles_mask_dir, name), os.path.join(output_dir, "test", "masks", name))

        # Remove now-empty tiles dir to free disk space
        if os.path.exists(os.path.join(output_dir, "tiles")):
            shutil.rmtree(os.path.join(output_dir, "tiles"))
            
        logger(f"Divisão concluída:")
        logger(f"  Treino (train): {len(train_basenames)} tiles")
        logger(f"  Validação (val): {len(val_basenames)} tiles")
        if test_basenames:
            logger(f"  Teste cego (test): {len(test_basenames)} tiles")
        
        # Save dataset info
        info_path = os.path.join(output_dir, "dataset_info.json")
        existing_orthos = []
        if append and os.path.exists(info_path):
            try:
                with open(info_path, "r", encoding="utf-8") as f:
                    d = json.load(f)
                    existing_orthos = d.get("orthomosaics", [])
            except Exception:
                pass
                
        ortho_name = os.path.basename(raster_path) if raster_path else "orthomosaic"
        if ortho_name not in existing_orthos:
            existing_orthos.append(ortho_name)
            
        with open(info_path, "w", encoding="utf-8") as f:
            json.dump({
                "orthomosaics": existing_orthos,
                "num_orthomosaics": len(existing_orthos),
                "last_raster": ortho_name,
                "updated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }, f, indent=2)

        width = window_width
        height = window_height
        cropped_transform = rasterio.windows.transform(talhao_window, src.transform)

        return {
            "num_tiles": tile_count,
            "train_count": len(train_basenames),
            "val_count": len(val_basenames),
            "num_orthomosaics": len(existing_orthos),
            "width": width,
            "height": height,
            "transform": [float(x) for x in cropped_transform]
        }

if __name__ == "__main__":
    # Test script locally
    import sys
    # Example paths
    run_preprocessing(
        raster_path="63996.gpkg",
        lines_path="data/synthetic/linhas_centrais.geojson",
        talhao_path="data/synthetic/talhao.geojson",
        buffer_size=1.5,
        tile_size=256,
        stride=128
    )
