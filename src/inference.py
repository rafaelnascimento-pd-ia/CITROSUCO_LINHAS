# -*- coding: utf-8 -*-
import os
import datetime
import torch
import numpy as np
import cv2
import rasterio
import rasterio.mask
import rasterio.features
from shapely.geometry import shape, mapping
import geopandas as gpd
from src.model import get_segmentation_model
from src.dataset import get_transforms

def generate_taper_window(tile_size):
    """
    Generates a 2D window weight with edge tapering to blend overlapping tiles smoothly.
    Uses a 1D linear taper at the edges (1/8 of tile size).
    """
    taper = tile_size // 8
    if taper < 1:
        taper = 1
    w1d = np.ones(tile_size, dtype=np.float32)
    w1d[:taper] = np.linspace(0.01, 1.0, taper)
    w1d[-taper:] = np.linspace(1.0, 0.01, taper)
    w2d = np.outer(w1d, w1d)
    return w2d

def run_inference(raster_path, talhao_path, checkpoint_path, output_dir="data",
                  tile_size=256, stride=128, device=None, logger=print,
                  min_tile_coverage=0.0, batch_size=4):
    """
    Runs sliding window inference on the orthomosaic cropped by the talhao polygon.
    """
    logger(f"Iniciando inferência...")
    logger(f"  Raster: {raster_path}")
    logger(f"  Talhão: {talhao_path}")
    logger(f"  Checkpoint: {checkpoint_path}")
    logger(f"  Tamanho do tile: {tile_size}")
    logger(f"  Stride: {stride}")
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger(f"Dispositivo para inferência: {device}")
    
    # 1. Load checkpoint and instantiate model
    logger("Carregando modelo...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    encoder_name = checkpoint.get("encoder_name", "resnet18")
    logger(f"  Encoder detectado no checkpoint: {encoder_name}")
    
    model = get_segmentation_model(encoder_name=encoder_name)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    
    # 2. Open raster and talhao, crop raster
    with rasterio.open(raster_path) as src:
        raster_crs = src.crs
        
        logger("Carregando e reprojetando talhão...")
        talhao_gdf = gpd.read_file(talhao_path).to_crs(raster_crs)
        talhao_geom = talhao_gdf.unary_union
        
        logger("Recortando o raster da área de interesse (talhão)...")
        try:
            cropped_img, cropped_transform = rasterio.mask.mask(
                src, [mapping(talhao_geom)], crop=True, nodata=0
            )
        except Exception as e:
            logger(f"Fallback: lendo pelo bounding box do talhão (erro: {e}).")
            raster_bounds = src.bounds
            t_bounds = talhao_geom.bounds
            clipped_bounds = (
                max(t_bounds[0], raster_bounds.left),
                max(t_bounds[1], raster_bounds.bottom),
                min(t_bounds[2], raster_bounds.right),
                min(t_bounds[3], raster_bounds.top)
            )
            if clipped_bounds[0] >= clipped_bounds[2] or clipped_bounds[1] >= clipped_bounds[3]:
                raise ValueError(
                    f"O talhão {t_bounds} não intercepta o raster {tuple(raster_bounds)}. "
                    "Verifique se o CRS do vetor está correto."
                )
            window = src.window(*clipped_bounds)
            window = window.round_offsets().round_shape(op='ceil', pixel_precision=1)
            cropped_img = src.read(window=window)
            cropped_transform = src.window_transform(window)

        channels, height, width = cropped_img.shape
        logger(f"Dimensões do recorte para inferência: {width}x{height} pixels (canais: {channels})")
        
        # Prepare image: (H, W, C), keep first 3 channels
        if channels < 3:
            cropped_img = np.repeat(cropped_img[:1], 3, axis=0)
        img_hwc = np.transpose(cropped_img[:3], (1, 2, 0)).astype(np.float32)
        
        # Safe radiometric normalization & nodata handling
        if src.nodata is not None:
            valid_mask = ~(cropped_img[:3] == src.nodata).any(axis=0)
        else:
            valid_mask = np.ones((height, width), dtype=bool)

        if cropped_img.dtype != np.uint8:
            if valid_mask.any():
                v_min = float(img_hwc[valid_mask].min())
                v_max = float(img_hwc[valid_mask].max())
            else:
                v_min, v_max = float(img_hwc.min()), float(img_hwc.max())
            value_range = v_max - v_min
            if value_range <= 1e-6:
                img_hwc = np.zeros((height, width, 3), dtype=np.uint8)
            else:
                img_hwc = np.clip((img_hwc - v_min) / value_range * 255.0, 0, 255).astype(np.uint8)
        else:
            img_hwc = img_hwc.astype(np.uint8)
                
        # Generate talhao raster mask to restrict prediction area later
        talhao_mask = np.zeros((height, width), dtype=np.uint8)
        rasterio.features.rasterize(
            shapes=[(talhao_geom, 255)],
            out=talhao_mask,
            transform=cropped_transform,
            fill=0,
            default_value=255
        )
        
        # 3. Setup sliding window accumulation arrays
        pred_map = np.zeros((height, width), dtype=np.float32)
        weight_map = np.zeros((height, width), dtype=np.float32)
        window_weight = generate_taper_window(tile_size)
        
        transform = get_transforms(img_size=tile_size, is_train=False)
        
        # Pre-pad image if it is smaller than tile_size
        pad_y = max(0, tile_size - height)
        pad_x = max(0, tile_size - width)
        if pad_y > 0 or pad_x > 0:
            img_hwc = cv2.copyMakeBorder(img_hwc, 0, pad_y, 0, pad_x, cv2.BORDER_CONSTANT, value=0)
            pred_map = cv2.copyMakeBorder(pred_map, 0, pad_y, 0, pad_x, cv2.BORDER_CONSTANT, value=0)
            weight_map = cv2.copyMakeBorder(weight_map, 0, pad_y, 0, pad_x, cv2.BORDER_CONSTANT, value=0)
            talhao_mask = cv2.copyMakeBorder(talhao_mask, 0, pad_y, 0, pad_x, cv2.BORDER_CONSTANT, value=0)
            logger(f"Imagem menor que o tile_size. Adicionado padding: Y={pad_y}, X={pad_x}")
            
        new_h, new_w = img_hwc.shape[0], img_hwc.shape[1]
        
        # 4. Sliding window loop
        logger("Executando predição por janela deslizante...")
        
        def is_tile_active(tile):
            if min_tile_coverage <= 0.0:
                return np.sum(tile == 255) > 0
            return float(np.mean(tile > 0)) >= min_tile_coverage

        coords = []
        for y in range(0, new_h - tile_size + 1, stride):
            for x in range(0, new_w - tile_size + 1, stride):
                talhao_tile = talhao_mask[y:y+tile_size, x:x+tile_size]
                if is_tile_active(talhao_tile):
                    coords.append((y, x))
                    
        # Add boundary cases
        for y in range(0, new_h - tile_size + 1, stride):
            x = new_w - tile_size
            tile = talhao_mask[y:y+tile_size, x:x+tile_size]
            if is_tile_active(tile) and (y, x) not in coords:
                coords.append((y, x))
        for x in range(0, new_w - tile_size + 1, stride):
            y = new_h - tile_size
            tile = talhao_mask[y:y+tile_size, x:x+tile_size]
            if is_tile_active(tile) and (y, x) not in coords:
                coords.append((y, x))
        y, x = new_h - tile_size, new_w - tile_size
        tile = talhao_mask[y:y+tile_size, x:x+tile_size]
        if is_tile_active(tile) and (y, x) not in coords:
            coords.append((y, x))
                
        logger(f"  Número de janelas ativas a processar: {len(coords)}")
        
        effective_batch_size = max(1, int(batch_size or 4))
        
        for i in range(0, len(coords), effective_batch_size):
            batch_coords = coords[i:i+effective_batch_size]
            tiles_batch = []
            
            for y, x in batch_coords:
                tile = img_hwc[y:y+tile_size, x:x+tile_size]
                transformed = transform(image=tile)
                tiles_batch.append(transformed['image'])
                
            inputs = torch.stack(tiles_batch).to(device)
            
            with torch.inference_mode():
                with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                    outputs = model(inputs)
                    probs = torch.sigmoid(outputs)
                    
            probs = probs.cpu().squeeze(1).numpy()
            if probs.ndim == 2:
                probs = np.expand_dims(probs, axis=0)
                
            for idx, (y, x) in enumerate(batch_coords):
                pred_map[y:y+tile_size, x:x+tile_size] += probs[idx] * window_weight
                weight_map[y:y+tile_size, x:x+tile_size] += window_weight
                
        weight_map = np.maximum(weight_map, 1e-5)
        prob_map = pred_map / weight_map
        
        if pad_y > 0 or pad_x > 0:
            prob_map = prob_map[:height, :width]
            
        logger("Predição do raster concluída.")
        return prob_map, cropped_transform, raster_crs, talhao_geom


def _save_intermediate_tifs(prob_map, transform, crs, output_dir, threshold=0.5, logger=print, suffix=""):
    """
    Salva os GeoTIFFs intermediários: mapa de probabilidade contínua (0 a 1) e máscara binária bruta.
    """
    tif_dir = os.path.join(output_dir, "outputs", "tifs")
    os.makedirs(tif_dir, exist_ok=True)
    height, width = prob_map.shape

    prob_path = os.path.join(tif_dir, f"probabilidade_linhas{suffix}.tif")
    logger(f"  [TIF Intermediário] Salvando mapa de probabilidade em: {prob_path}")
    with rasterio.open(
        prob_path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype=rasterio.float32,
        crs=crs,
        transform=transform,
        nodata=0
    ) as dst:
        dst.write(prob_map, 1)

    bin_mask = (prob_map >= threshold).astype(np.uint8) * 255
    bin_path = os.path.join(tif_dir, f"predicao_binaria{suffix}.tif")
    logger(f"  [TIF Intermediário] Salvando máscara binária bruta em: {bin_path}")
    with rasterio.open(
        bin_path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype=rasterio.uint8,
        crs=crs,
        transform=transform,
        nodata=0
    ) as dst:
        dst.write(bin_mask, 1)


def _save_raw_geojson(prob_map, transform, crs, talhao_geom, output_dir, threshold=0.5, min_area_m2=5.0, logger=print, suffix=""):
    """
    Polygoniza a máscara bruta sem filtros morfológicos ou aproximações de linhas,
    gerando o GeoJSON dos polígonos detectados diretamente pela rede neural.
    """
    height, width = prob_map.shape
    binary_mask = (prob_map >= threshold).astype(np.uint8) * 255

    # Restringe ao talhão se existir
    if talhao_geom is not None and not talhao_geom.is_empty:
        talhao_mask = np.zeros((height, width), dtype=np.uint8)
        rasterio.features.rasterize(
            shapes=[(talhao_geom, 255)],
            out=talhao_mask,
            transform=transform,
            fill=0,
            default_value=255
        )
        binary_mask = cv2.bitwise_and(binary_mask, talhao_mask)

    shapes_gen = rasterio.features.shapes(binary_mask, mask=(binary_mask > 0), transform=transform)
    polygons = []
    for geom_dict, val in shapes_gen:
        if val == 255:
            poly_shape = shape(geom_dict)
            if poly_shape.area >= min_area_m2:
                polygons.append(poly_shape)

    if not polygons:
        logger(f"  [Raw Vector] Nenhum polígono bruto acima de {min_area_m2}m² no recorte{suffix}.")
        return None

    gdf_utm = gpd.GeoDataFrame(geometry=polygons, crs=crs)
    gdf_utm['area_m2'] = gdf_utm.geometry.area
    gdf_wgs84 = gdf_utm.to_crs("EPSG:4326")

    raw_dir = os.path.join(output_dir, "outputs")
    os.makedirs(raw_dir, exist_ok=True)
    if not suffix:
        raw_path = os.path.join(raw_dir, "predicao_linhas_raw.geojson")
        logger(f"  [Raw Vector] Salvando GeoJSON bruto sem pós-processamento ({len(polygons)} polígonos) em: {raw_path}")
        gdf_wgs84.to_file(raw_path, driver="GeoJSON")

    return gdf_wgs84


def run_segmented_inference_pipeline(raster_path, talhao_path, checkpoint_path, output_dir="data",
                                     tile_size=256, stride=128, threshold=0.5, min_area_m2=10.0,
                                     close_size=5, open_size=3, skip_postprocess=False,
                                     export_as_lines=False, path_linhas=None,
                                     profile="tiro_reto", calc_entrelinhas=False,
                                     extend_to_boundary=None, min_row_spacing_m=4.0,
                                     max_row_spacing_m=8.5, device=None, logger=print,
                                     centerline_method="skeleton_v2",
                                     merge_method="spatial_cost",
                                     dedup_method="overlap_parallelism",
                                     min_tile_coverage=0.0, batch_size=4,
                                     resample_step_m=1.0, gsd_m=None,
                                     prob_refinement_enabled=False,
                                     prob_refinement_weight=0.5,
                                     prob_search_radius_m=1.5,
                                     prob_max_shift_m=0.75,
                                     curvature_control_enabled=False,
                                     max_curvature=None,
                                     corridor_validation_enabled=False,
                                     topology_validation_enabled=True,
                                     write_run_metadata_file=True):
    """
    Executa a inferência de forma segmentada (talhão por talhão) para evitar estouro de memória (RAM)
    em imagens gigantescas. Suporta os perfis 'tiro_reto' e 'curvas'.
    """
    import geopandas as gpd
    import shutil
    import pandas as pd
    from src.postprocess import run_postprocessing, calculate_lines_dominant_angle, calculate_dominant_angle, write_run_metadata

    logger("=== INICIANDO INFERÊNCIA SEGMENTADA POR TALHÃO (OTIMIZADA PARA MEMÓRIA) ===")
    logger(f"  Perfil selecionado: {profile}")
    logger(f"  Método de Centerline: {centerline_method} | Merge: {merge_method} | Dedup: {dedup_method}")
    
    # 1. Read talhao features
    talhao_gdf = gpd.read_file(talhao_path)
    num_features = len(talhao_gdf)
    logger(f"Total de talhões detectados: {num_features}")

    if num_features <= 1:
        logger("Apenas 1 talhão detectado. Executando fluxo padrão...")
        prob_map, transform, crs, talhao_geom = run_inference(
            raster_path=raster_path,
            talhao_path=talhao_path,
            checkpoint_path=checkpoint_path,
            output_dir=output_dir,
            tile_size=tile_size,
            stride=stride,
            device=device,
            logger=logger,
            min_tile_coverage=min_tile_coverage,
            batch_size=batch_size
        )
        # Salva TIFs intermediários no fluxo de talhão único
        _save_intermediate_tifs(prob_map, transform, crs, output_dir, threshold, logger)
        # Salva geojson bruto (sem pós-processamento)
        _save_raw_geojson(prob_map, transform, crs, talhao_geom, output_dir, threshold, min_area_m2, logger)
        return run_postprocessing(
            prob_map=prob_map,
            transform=transform,
            crs=crs,
            talhao_geom=talhao_geom,
            output_dir=output_dir,
            threshold=threshold,
            min_area_m2=min_area_m2,
            close_size=close_size,
            open_size=open_size,
            skip_postprocess=skip_postprocess,
            export_as_lines=export_as_lines,
            path_linhas=path_linhas,
            profile=profile,
            calc_entrelinhas=calc_entrelinhas,
            extend_to_boundary=extend_to_boundary,
            min_row_spacing_m=min_row_spacing_m,
            max_row_spacing_m=max_row_spacing_m,
            logger=logger,
            centerline_method=centerline_method,
            merge_method=merge_method,
            dedup_method=dedup_method,
            resample_step_m=resample_step_m,
            gsd_m=gsd_m,
            prob_refinement_enabled=prob_refinement_enabled,
            prob_refinement_weight=prob_refinement_weight,
            prob_search_radius_m=prob_search_radius_m,
            prob_max_shift_m=prob_max_shift_m,
            curvature_control_enabled=curvature_control_enabled,
            max_curvature=max_curvature,
            corridor_validation_enabled=corridor_validation_enabled,
            topology_validation_enabled=topology_validation_enabled,
            write_run_metadata_file=write_run_metadata_file
        )

    # Multi-feature segmented flow
    all_gdfs = []
    all_raw_gdfs = []
    all_entrelinhas_gdfs = []
    temp_dir = os.path.join(output_dir, "temp_segments")
    os.makedirs(temp_dir, exist_ok=True)
    final_output_dir = os.path.join(output_dir, "outputs")
    os.makedirs(final_output_dir, exist_ok=True)

    for idx, row in talhao_gdf.iterrows():
        segment_num = idx + 1
        logger(f"\n--- Processando Talhão {segment_num}/{num_features} ---")
        
        single_talhao_gdf = gpd.GeoDataFrame([row], crs=talhao_gdf.crs)
        temp_talhao_path = os.path.join(temp_dir, f"temp_talhao_{idx}.geojson")
        single_talhao_gdf.to_file(temp_talhao_path, driver="GeoJSON")
        
        segment_output_dir = os.path.join(temp_dir, f"output_{idx}")
        os.makedirs(segment_output_dir, exist_ok=True)
        
        try:
            prob_map, transform, crs, talhao_geom = run_inference(
                raster_path=raster_path,
                talhao_path=temp_talhao_path,
                checkpoint_path=checkpoint_path,
                output_dir=segment_output_dir,
                tile_size=tile_size,
                stride=stride,
                device=device,
                logger=logger,
                min_tile_coverage=min_tile_coverage,
                batch_size=batch_size
            )

            # Salva TIFs intermediários do segmento no diretório final
            _save_intermediate_tifs(
                prob_map, transform, crs, output_dir, threshold, logger,
                suffix=f"_talhao{segment_num}"
            )
            # Salva geojson bruto do segmento antes do pós-processamento
            raw_gdf = _save_raw_geojson(
                prob_map, transform, crs, talhao_geom, output_dir, threshold, min_area_m2, logger,
                suffix=f"_talhao{segment_num}"
            )
            if raw_gdf is not None and not raw_gdf.empty:
                all_raw_gdfs.append(raw_gdf)

            seg_results = run_postprocessing(
                prob_map=prob_map,
                transform=transform,
                crs=crs,
                talhao_geom=talhao_geom,
                output_dir=segment_output_dir,
                threshold=threshold,
                min_area_m2=min_area_m2,
                close_size=close_size,
                open_size=open_size,
                skip_postprocess=skip_postprocess,
                export_as_lines=export_as_lines,
                path_linhas=path_linhas,
                profile=profile,
                calc_entrelinhas=calc_entrelinhas,
                extend_to_boundary=extend_to_boundary,
                min_row_spacing_m=min_row_spacing_m,
                max_row_spacing_m=max_row_spacing_m,
                logger=logger,
                centerline_method=centerline_method,
                merge_method=merge_method,
                dedup_method=dedup_method,
                resample_step_m=resample_step_m,
                gsd_m=gsd_m,
                prob_refinement_enabled=prob_refinement_enabled,
                prob_refinement_weight=prob_refinement_weight,
                prob_search_radius_m=prob_search_radius_m,
                prob_max_shift_m=prob_max_shift_m,
                curvature_control_enabled=curvature_control_enabled,
                max_curvature=max_curvature,
                corridor_validation_enabled=corridor_validation_enabled,
                topology_validation_enabled=topology_validation_enabled,
                write_run_metadata_file=write_run_metadata_file
            )

            seg_geojson_path = os.path.join(segment_output_dir, "outputs", "predicao_linhas.geojson")
            if os.path.exists(seg_geojson_path):
                seg_gdf = gpd.read_file(seg_geojson_path)
                if not seg_gdf.empty:
                    all_gdfs.append(seg_gdf)
                    logger(f"[OK] Talhão {segment_num} processado: {len(seg_gdf)} linhas detectadas.")

            seg_entrelinhas_path = os.path.join(segment_output_dir, "outputs", "predicao_entrelinhas.geojson")
            if os.path.exists(seg_entrelinhas_path):
                seg_entrelinhas_gdf = gpd.read_file(seg_entrelinhas_path)
                if not seg_entrelinhas_gdf.empty:
                    all_entrelinhas_gdfs.append(seg_entrelinhas_gdf)

        except Exception as e:
            logger(f"[ERRO] Falha ao processar o Talhão {segment_num}: {e}")
            import traceback
            logger(traceback.format_exc())

        finally:
            if os.path.exists(segment_output_dir):
                try:
                    shutil.rmtree(segment_output_dir)
                except Exception:
                    pass
            if os.path.exists(temp_talhao_path):
                try:
                    os.remove(temp_talhao_path)
                except Exception:
                    pass

    if os.path.exists(temp_dir):
        try:
            shutil.rmtree(temp_dir)
        except Exception:
            pass

    # 4. Merge results and save final outputs
    logger("\n=== Agrupando resultados de todos os talhões ===")
    run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    final_output_dir = os.path.join(output_dir, run_timestamp)
    os.makedirs(final_output_dir, exist_ok=True)
    logger(f"  Diretório desta execução: {final_output_dir}")
    
    geojson_path = os.path.join(final_output_dir, "predicao_linhas.geojson")
    shp_dir = os.path.join(final_output_dir, "shp")
    shp_path = os.path.join(shp_dir, "predicao_linhas.shp")
    
    if all_gdfs:
        merged_gdf = gpd.GeoDataFrame(pd.concat(all_gdfs, ignore_index=True), crs="EPSG:4326")
        logger(f"Total combinado: {len(merged_gdf)} linhas detectadas em toda a área.")
        merged_gdf.to_file(geojson_path, driver="GeoJSON")
        os.makedirs(shp_dir, exist_ok=True)
        merged_gdf.to_file(shp_path)
        logger(f"GeoJSON final salvo em: {geojson_path}")
    else:
        logger("[WARN] Nenhuma linha detectada em nenhum talhão. Gerando arquivos vazios.")
        merged_gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        merged_gdf.to_file(geojson_path, driver="GeoJSON")

    # Salva geojson bruto combinado (sem pós-processamento)
    raw_geojson_path = os.path.join(final_output_dir, "predicao_linhas_raw.geojson")
    if all_raw_gdfs:
        merged_raw_gdf = gpd.GeoDataFrame(pd.concat(all_raw_gdfs, ignore_index=True), crs="EPSG:4326")
        merged_raw_gdf.to_file(raw_geojson_path, driver="GeoJSON")
        logger(f"GeoJSON bruto (sem pós-processamento) salvo em: {raw_geojson_path}")
    else:
        gpd.GeoDataFrame(geometry=[], crs="EPSG:4326").to_file(raw_geojson_path, driver="GeoJSON")

    if write_run_metadata_file:
        write_run_metadata(final_output_dir, {
            "profile": profile,
            "num_talhoes": num_features,
            "num_lines": len(merged_gdf) if all_gdfs else 0,
            "centerline_method": centerline_method,
            "merge_method": merge_method,
            "dedup_method": dedup_method,
            "prob_refinement_enabled": prob_refinement_enabled,
            "curvature_control_enabled": curvature_control_enabled,
            "topology_validation_enabled": topology_validation_enabled
        })

    # Merge entrelinhas if present
    entrelinhas_path = None
    entrelinhas_count = 0
    if all_entrelinhas_gdfs:
        merged_entrelinhas = gpd.GeoDataFrame(pd.concat(all_entrelinhas_gdfs, ignore_index=True), crs="EPSG:4326")
        entrelinhas_count = len(merged_entrelinhas)
        entrelinhas_path = os.path.join(final_output_dir, "predicao_entrelinhas.geojson")
        merged_entrelinhas.to_file(entrelinhas_path, driver="GeoJSON")
        logger(f"Total de entrelinhas curvas combinadas: {entrelinhas_count} salvas em: {entrelinhas_path}")

    reported_angle = None
    if profile != "curvas":
        if path_linhas and os.path.exists(path_linhas):
            with rasterio.open(raster_path) as r_src:
                r_crs = r_src.crs
            reported_angle = calculate_lines_dominant_angle(path_linhas, target_crs=r_crs)
        elif all_gdfs and not merged_gdf.empty:
            with rasterio.open(raster_path) as r_src:
                r_crs = r_src.crs
            gdf_proj = merged_gdf.to_crs(r_crs)
            geom_type = gdf_proj.iloc[0].geometry.geom_type
            if 'LineString' in geom_type:
                angles = []
                for geom in gdf_proj.geometry:
                    if geom is None or geom.is_empty:
                        continue
                    coords = np.array(geom.coords)
                    if len(coords) >= 2:
                        v = coords[-1] - coords[0]
                        if np.linalg.norm(v) > 0:
                            angle = np.arctan2(v[1], v[0])
                            while angle < -np.pi/2:
                                angle += np.pi
                            while angle > np.pi/2:
                                angle -= np.pi
                            angles.append(angle)
                if angles:
                    reported_angle = np.median(angles)
            else:
                reported_angle = calculate_dominant_angle(gdf_proj.geometry)

    logger("=== INFERÊNCIA SEGMENTADA CONCLUÍDA COM SUCESSO ===")
    
    results = {
        "profile": profile,
        "num_polygons": len(merged_gdf),
        "geotiff": None,
        "geojson": geojson_path,
        "shapefile": shp_path if all_gdfs else None,
        "entrelinhas_geojson": entrelinhas_path,
        "num_entrelinhas": entrelinhas_count
    }
    
    if reported_angle is not None:
        results["dominant_angle_deg"] = float(np.degrees(reported_angle))
        results["dominant_azimuth_deg"] = float((90.0 - np.degrees(reported_angle)) % 180)
        
    return results


if __name__ == "__main__":
    pass
