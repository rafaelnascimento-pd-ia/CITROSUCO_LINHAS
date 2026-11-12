# -*- coding: utf-8 -*-
import argparse
import os
import sys
import subprocess
import tempfile
import shutil

# Desativa PyGEOS para compatibilidade com Shapely 2.0+ e suprime warning
os.environ['USE_PYGEOS'] = '0'
# Permite OpenMP duplicado entre PyTorch e OpenCV no Windows
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import yaml
import torch
import rasterio
import geopandas as gpd
import numpy as np
from pyproj import Transformer
from shapely.geometry import LineString, MultiLineString, Polygon, MultiPolygon
from shapely.ops import unary_union

from src.inference import run_inference, run_segmented_inference_pipeline
from src.postprocess import run_postprocessing


def load_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuração não encontrada: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def normalize_path(path, base_dir=None):
    if not path:
        return None
    if base_dir and not os.path.isabs(path):
        path = os.path.join(base_dir, path)
    return os.path.abspath(path)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def read_prob_map_from_raster(raster_path):
    with rasterio.open(raster_path) as src:
        prob_map = src.read(1).astype('float32')
        if prob_map.max() > 1.0:
            prob_map = prob_map / 255.0
        return prob_map, src.transform, src.crs


def wrap_with_defaults(cfg):
    defaults = {
        'data_dir': 'data',
        'models_dir': 'models',
        'output_dir': 'data',
        'encoder_name': 'resnet18',
        'buffer_size': 1.37,
        'tile_size': 256,
        'stride': 128,
        'append': False,
        'val_size': 0.15,
        'test_size': 0.10,
        'epochs': 10,
        'batch_size': 2,
        'lr': 0.0001,
        'early_stopping_patience': 20,
        'min_delta': 0.001,
        'skip_preprocess': False,
        'threshold': 0.5,
        'min_area_m2': 10.0,
        'close_size': 5,
        'open_size': 3,
        'skip_postprocess': False,
        'export_as_lines': False,
        'profile': 'tiro_reto',
        'calc_entrelinhas': False,
        'extend_to_boundary': None,
        'min_row_spacing_m': 4.0,
        'max_row_spacing_m': 8.5,
        'centerline_method': 'skeleton_v2',
        'merge_method': 'spatial_cost',
        'dedup_method': 'overlap_parallelism',
        'min_tile_coverage': 0.0,
        'resample_step_m': 1.0,
        'gsd_m': None,
        'prob_refinement_enabled': False,
        'prob_refinement_weight': 0.5,
        'prob_search_radius_m': 1.5,
        'prob_max_shift_m': 0.75,
        'curvature_control_enabled': False,
        'max_curvature': None,
        'corridor_validation_enabled': False,
        'topology_validation_enabled': True,
        'write_run_metadata_file': True,
        'field_partition': {},
    }
    for key, value in defaults.items():
        cfg.setdefault(key, value)
    return cfg


def normalize_dataset_entry(dataset, base_dir=None):
    dataset['gpkg_path'] = normalize_path(dataset.get('gpkg_path'), base_dir)
    dataset['lines_path'] = normalize_path(dataset.get('lines_path'), base_dir)
    dataset['talhao_path'] = normalize_path(dataset.get('talhao_path'), base_dir)
    return dataset


def normalize_config_paths(cfg, base_dir=None):
    cfg['data_dir'] = normalize_path(cfg.get('data_dir'), base_dir)
    cfg['models_dir'] = normalize_path(cfg.get('models_dir'), base_dir)
    cfg['output_dir'] = normalize_path(cfg.get('output_dir'), base_dir)
    return cfg


def generate_interrows(
    lines_path,
    talhao_path,
    output_dir,
    min_spacing,
    max_spacing,
    base_dir,
    max_bridge_gap=0.75,
):
    """Gera entrelinhas curvas usando as linhas já exportadas pelo pipeline."""
    script_path = os.path.join(
        base_dir,
        'scripts',
        'calcular_entrelinhas_curvas',
        'calcular_entrelinhas_curvas_robusto.py'
    )
    if not os.path.isfile(lines_path):
        raise FileNotFoundError(f'Arquivo de linhas para calcular entrelinhas não encontrado: {lines_path}')
    if not os.path.isfile(script_path):
        raise FileNotFoundError(f'Script de cálculo de entrelinhas não encontrado: {script_path}')

    command = [
        sys.executable,
        script_path,
        '--input_lines', lines_path,
        '--talhao_path', talhao_path,
        '--output_dir', output_dir,
        '--min_row_spacing', str(min_spacing),
        '--max_row_spacing', str(max_spacing),
        '--max_bridge_gap', str(max_bridge_gap),
    ]
    print('=== Calculando entrelinhas curvas a partir das linhas detectadas ===')
    subprocess.run(command, check=True)

    interrows_path = os.path.join(output_dir, 'predicao_entrelinhas.geojson')
    if not os.path.isfile(interrows_path):
        raise FileNotFoundError(f'O cálculo terminou sem gerar o arquivo esperado: {interrows_path}')
    interrows = gpd.read_file(interrows_path)
    print(f'[OK] {len(interrows)} trecho(s) de entrelinha salvo(s) em: {interrows_path}')
    return interrows_path, len(interrows)


def normalize_result_shapefiles(
    search_roots,
    mdt_path,
    max_line_vertices=80,
    simplify_tolerance_m=0.1,
    route_smoothing_radius_m=0.5,
    route_smoothing_min_turn_deg=15.0,
    route_simplify_tolerance_m=0.035,
    max_boundary_vertices=100,
    boundary_simplify_tolerance_m=0.1,
):
    """Define geometrias 3D finais e grava os shapefiles em WGS84/UTF-8."""
    with rasterio.open(mdt_path) as mdt:
        to_mdt_cache = {}
        to_wgs84 = Transformer.from_crs(mdt.crs, 'EPSG:4326', always_xy=True)

        def elevated_wgs84(coord, source_transformer):
            x, y = source_transformer.transform(coord[0], coord[1])
            value = float(next(mdt.sample([(x, y)]))[0])
            if not value or not (value == value) or (mdt.nodata is not None and value == mdt.nodata):
                value = 0.0
            lon, lat = to_wgs84.transform(x, y)
            return (lon, lat, value)

        def line_parts(geom):
            if geom.geom_type == 'LineString':
                return [geom]
            if geom.geom_type == 'MultiLineString':
                return list(geom.geoms)
            if geom.geom_type == 'GeometryCollection':
                return [part for child in geom.geoms for part in line_parts(child)]
            return []

        def polygon_parts(geom):
            if geom.geom_type == 'Polygon':
                return [geom]
            if geom.geom_type == 'MultiPolygon':
                return list(geom.geoms)
            return []

        for search_root in search_roots:
            if not os.path.isdir(search_root):
                continue
            for root, _, filenames in os.walk(search_root):
                for filename in filenames:
                    if not filename.lower().endswith('.shp'):
                        continue
                    upper_name = filename.upper()
                    is_boundary = upper_name.startswith('BORDADURA_')
                    is_route = upper_name.startswith('ROTA_BORDADURA_')
                    is_line = (
                        upper_name.startswith('PREDICAO_LINHAS')
                        or upper_name.startswith('PREDICAO_ENTRELINHAS')
                        or is_route
                    )
                    if not (is_boundary or is_line):
                        continue

                    path = os.path.join(root, filename)
                    gdf = gpd.read_file(path)
                    if gdf.empty or gdf.crs is None:
                        continue
                    crs_key = str(gdf.crs)
                    if crs_key not in to_mdt_cache:
                        to_mdt_cache[crs_key] = Transformer.from_crs(gdf.crs, mdt.crs, always_xy=True)
                    source_transformer = to_mdt_cache[crs_key]
                    output_geometries = []
                    anchors = []

                    for geom in gdf.geometry:
                        if geom is None or geom.is_empty:
                            output_geometries.append(None)
                            anchors.append(None)
                        elif is_boundary:
                            parts = []
                            anchor = None
                            for poly in polygon_parts(geom):
                                exterior = [elevated_wgs84(c, source_transformer) for c in poly.exterior.coords]
                                if anchor is None and exterior:
                                    anchor = exterior[0]
                                interiors = [
                                    [elevated_wgs84(c, source_transformer) for c in ring.coords]
                                    for ring in poly.interiors
                                ]
                                parts.append(Polygon(exterior, interiors))
                            output_geometries.append(MultiPolygon(parts) if parts else None)
                            anchors.append(anchor)
                        else:
                            parts = []
                            for line in line_parts(geom):
                                coords = [elevated_wgs84(c, source_transformer) for c in line.coords]
                                if len(coords) >= 2:
                                    parts.append(coords)
                            output_geometries.append(MultiLineString(parts) if parts else None)
                            anchors.append(parts[0][0] if parts else None)

                    if is_line:
                        valid_coords = [
                            coord[:2]
                            for geom in output_geometries if geom is not None
                            for line in geom.geoms for coord in line.coords
                        ]
                        if len(valid_coords) >= 2:
                            centered = np.asarray(valid_coords, dtype=float)
                            _, _, vectors = np.linalg.svd(centered - centered.mean(axis=0), full_matrices=False)
                            direction = vectors[0]
                            if direction[0] < 0 or (abs(direction[0]) < 1e-12 and direction[1] < 0):
                                direction = -direction
                        else:
                            direction = np.array([1.0, 0.0])
                        across = np.array([-direction[1], direction[0]])

                        records = []
                        for idx, geom in enumerate(output_geometries):
                            if geom is None:
                                records.append((float('inf'), idx, None, None))
                                continue
                            parts = []
                            for line in geom.geoms:
                                coords = list(line.coords)
                                start_proj = np.dot(np.asarray(coords[0][:2]), direction)
                                end_proj = np.dot(np.asarray(coords[-1][:2]), direction)
                                if end_proj < start_proj:
                                    coords.reverse()
                                parts.append(coords)
                            parts.sort(key=lambda coords: float(np.mean([np.dot(np.asarray(c[:2]), across) for c in coords])))
                            normalized = MultiLineString(parts)
                            all_coords = [coord for part in parts for coord in part]
                            order_key = float(np.mean([np.dot(np.asarray(c[:2]), across) for c in all_coords]))
                            records.append((order_key, idx, normalized, parts[0][0] if parts else None))

                        records.sort(key=lambda item: item[0])
                        output_geometries = [item[2] for item in records]
                        anchors = [item[3] for item in records]
                        gdf = gdf.iloc[[item[1] for item in records]].reset_index(drop=True)

                    gdf = gdf.drop(columns=[gdf.geometry.name]).copy()
                    gdf = gpd.GeoDataFrame(gdf, geometry=output_geometries, crs='EPSG:4326')
                    # Replace old coordinate attributes, retaining field/source IDs.
                    gdf = gdf.drop(columns=[c for c in gdf.columns if c.lower() in ('x', 'y', 'z')])
                    gdf['X'] = [anchor[0] if anchor else None for anchor in anchors]
                    gdf['Y'] = [anchor[1] if anchor else None for anchor in anchors]
                    gdf['Z'] = [anchor[2] if anchor else None for anchor in anchors]
                    if is_line and not gdf.empty:
                        metric_crs = gdf.estimate_utm_crs()
                        metric_gdf = gdf.to_crs(metric_crs)

                        def vertex_count(geom):
                            if geom is None or geom.is_empty:
                                return 0
                            if geom.geom_type == 'LineString':
                                return len(geom.coords)
                            if geom.geom_type == 'MultiLineString':
                                return sum(len(part.coords) for part in geom.geoms)
                            return 0

                        def force_multipart(geom):
                            if geom is None or geom.is_empty:
                                return geom
                            if geom.geom_type == 'MultiLineString' and len(geom.geoms) > 1:
                                return geom
                            if geom.geom_type == 'LineString':
                                coords = list(geom.coords)
                            elif geom.geom_type == 'MultiLineString' and geom.geoms:
                                coords = list(geom.geoms[0].coords)
                            else:
                                return geom
                            if len(coords) < 2:
                                return geom
                            if len(coords) == 2:
                                midpoint_coord = tuple(
                                    (start + end) / 2.0
                                    for start, end in zip(coords[0], coords[1])
                                )
                                return MultiLineString([
                                    [coords[0], midpoint_coord],
                                    [midpoint_coord, coords[1]],
                                ])
                            midpoint = len(coords) // 2
                            return MultiLineString([coords[:midpoint + 1], coords[midpoint:]])

                        def smooth_route_line(line):
                            coords = np.asarray(line.coords, dtype=float)
                            if len(coords) < 3:
                                return line
                            closed = np.linalg.norm(coords[0, :2] - coords[-1, :2]) < 1e-6
                            core = coords[:-1].copy() if closed else coords.copy()
                            if len(core) < 3:
                                return line

                            if closed:
                                turns = []
                                for idx in range(len(core)):
                                    incoming = core[idx, :2] - core[(idx - 1) % len(core), :2]
                                    outgoing = core[(idx + 1) % len(core), :2] - core[idx, :2]
                                    incoming_len = np.linalg.norm(incoming)
                                    outgoing_len = np.linalg.norm(outgoing)
                                    if incoming_len <= 1e-9 or outgoing_len <= 1e-9:
                                        turns.append(float('inf'))
                                        continue
                                    cosine = np.dot(incoming / incoming_len, outgoing / outgoing_len)
                                    turns.append(float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))))
                                turns = np.asarray(turns)

                            smoothed = []
                            for idx, point in enumerate(core):
                                if (closed and idx == 0) or (not closed and idx in (0, len(core) - 1)):
                                    smoothed.append(tuple(point))
                                    continue
                                prev_idx = (idx - 1) % len(core)
                                next_idx = (idx + 1) % len(core)
                                incoming = point[:2] - core[prev_idx, :2]
                                outgoing = core[next_idx, :2] - point[:2]
                                incoming_len = np.linalg.norm(incoming)
                                outgoing_len = np.linalg.norm(outgoing)
                                if incoming_len <= 1e-9 or outgoing_len <= 1e-9:
                                    smoothed.append(tuple(point))
                                    continue
                                if closed:
                                    turn = turns[idx]
                                else:
                                    in_unit = incoming / incoming_len
                                    out_unit = outgoing / outgoing_len
                                    turn = float(np.degrees(np.arccos(np.clip(np.dot(in_unit, out_unit), -1.0, 1.0))))
                                if turn < route_smoothing_min_turn_deg:
                                    smoothed.append(tuple(point))
                                    continue
                                cut = min(
                                    route_smoothing_radius_m,
                                    incoming_len * 0.25,
                                    outgoing_len * 0.25,
                                )
                                before = point.copy()
                                after = point.copy()
                                before[:2] = point[:2] - cut * incoming / incoming_len
                                after[:2] = point[:2] + cut * outgoing / outgoing_len
                                if len(point) > 2:
                                    before[2] = point[2] - cut / incoming_len * (point[2] - core[prev_idx, 2])
                                    after[2] = point[2] + cut / outgoing_len * (core[next_idx, 2] - point[2])
                                smoothed.extend([tuple(before), tuple(after)])

                            if closed:
                                smoothed.append(smoothed[0])
                            return LineString(smoothed)

                        tolerances = (
                            [route_simplify_tolerance_m]
                            if is_route
                            else [simplify_tolerance_m * step for step in range(1, 6)]
                        )
                        simplified = []
                        for geom in metric_gdf.geometry:
                            base_geom = geom
                            if is_route and 'field_id' in gdf.columns:
                                # Shared cuts must remain on the exact boundary;
                                # smoothing would enter a neighbouring planting block.
                                simplified.append(force_multipart(base_geom))
                                continue
                            if geom.geom_type == 'MultiLineString' and not is_route and 'field_id' in gdf.columns:
                                simplified.append(force_multipart(base_geom))
                                continue
                            if is_route and geom.geom_type == 'MultiLineString':
                                route_parts = [smooth_route_line(part) for part in geom.geoms]
                                base_geom = MultiLineString(route_parts)
                            elif is_route and geom.geom_type == 'LineString':
                                base_geom = smooth_route_line(geom)
                            candidate = base_geom
                            for tolerance in tolerances:
                                candidate = base_geom.simplify(tolerance, preserve_topology=True)
                                if is_route or vertex_count(candidate) <= max_line_vertices:
                                    break
                            simplified.append(force_multipart(candidate))
                        metric_gdf.geometry = gpd.GeoSeries(
                            simplified, index=metric_gdf.index, crs=metric_crs
                        )
                        gdf = metric_gdf.to_crs('EPSG:4326')
                    elif is_boundary and not gdf.empty and 'field_id' not in gdf.columns:
                        metric_crs = gdf.estimate_utm_crs()
                        metric_gdf = gdf.to_crs(metric_crs)

                        def polygon_vertex_count(geom):
                            if geom is None or geom.is_empty:
                                return 0
                            polygons = geom.geoms if geom.geom_type == 'MultiPolygon' else [geom]
                            return sum(
                                len(ring.coords)
                                for polygon in polygons
                                for ring in [polygon.exterior, *polygon.interiors]
                            )

                        # Keep one record per child. Never dissolve across field IDs.
                        boundaries = []
                        for geom in metric_gdf.geometry:
                            candidate = geom
                            if 'field_id' not in metric_gdf.columns:
                                for step in range(1, 6):
                                    candidate = geom.simplify(boundary_simplify_tolerance_m * step, preserve_topology=True)
                                    if polygon_vertex_count(candidate) <= max_boundary_vertices:
                                        break
                            boundaries.append(MultiPolygon([candidate]) if candidate.geom_type == 'Polygon' else candidate)
                        metric_gdf.geometry = gpd.GeoSeries(boundaries, index=metric_gdf.index, crs=metric_crs)
                        gdf = metric_gdf.to_crs('EPSG:4326')
                    if is_boundary:
                        from src.field_partition import project_polygons
                        gdf = project_polygons(gdf, 'EPSG:4326')
                    # IDs are needed above to preserve child boundaries, but
                    # the delivery attribute table contains coordinates only.
                    gdf = gdf[['X', 'Y', 'Z', gdf.geometry.name]]
                    gdf.to_file(path, driver='ESRI Shapefile', encoding='UTF-8')


def train_multi_command(cfg, base_dir=None):
    """
    Faz pré-processamento de MÚLTIPLOS datasets e treina um ÚNICO modelo com todos os dados.
    """
    from src.preprocess import run_preprocessing
    from src.train import train_model

    cfg = wrap_with_defaults(cfg)
    cfg = normalize_config_paths(cfg, base_dir)
    ensure_dir(cfg['models_dir'])
    ensure_dir(cfg['data_dir'])

    datasets = cfg.get('datasets', [])
    if not datasets:
        raise ValueError('Config deve conter "datasets" com lista de datasets')

    skip_prep = cfg.get('skip_preprocess', False)
    train_images_dir = os.path.join(cfg['data_dir'], 'train', 'images')
    if skip_prep and (not os.path.exists(train_images_dir) or len(os.listdir(train_images_dir)) == 0):
        print(f'=== [AVISO] skip_preprocess=True mas pasta {train_images_dir} nao existe ou esta vazia. Executando pre-processamento...')
        skip_prep = False

    if not skip_prep:
        print(f'=== Processando {len(datasets)} dataset(s) para treinamento único...\n')
        for idx, dataset in enumerate(datasets, 1):
            dataset = normalize_dataset_entry(dataset, base_dir)
            print(f'- Dataset {idx}/{len(datasets)}: {dataset.get("gpkg_path", "?")}')
            
            should_append = (idx > 1)
            
            dataset_buffer = float(dataset.get('buffer_size', cfg['buffer_size']))
            print(f'  Buffer aplicado: {dataset_buffer}m')
            run_preprocessing(
                raster_path=dataset['gpkg_path'],
                lines_path=dataset['lines_path'],
                talhao_path=dataset['talhao_path'],
                output_dir=cfg['data_dir'],
                buffer_size=dataset_buffer,
                tile_size=int(cfg['tile_size']),
                stride=int(cfg['stride']),
                val_size=float(cfg['val_size']),
                test_size=float(cfg['test_size']),
                append=should_append,
                logger=print
            )
            print()
    else:
        print('=== Pulando pré-processamento (utilizando tiles já gerados em data/)...\n')

    print('=== Iniciando treinamento do modelo (com todos os datasets)...\n')
    saved_model = train_model(
        data_dir=cfg['data_dir'],
        models_dir=cfg['models_dir'],
        encoder_name=cfg['encoder_name'],
        epochs=int(cfg['epochs']),
        batch_size=int(cfg['batch_size']),
        lr=float(cfg['lr']),
        patience=int(cfg.get('early_stopping_patience', cfg.get('patience', 20))),
        min_delta=float(cfg.get('min_delta', 0.001)),
        num_orthos=len(datasets)
    )
    print('\n[OK] Treinamento concluído com sucesso!')
    print(f'[OK] Checkpoint salvo em: {saved_model}')


def train_command(cfg, base_dir=None):
    from src.preprocess import run_preprocessing
    from src.train import train_model

    cfg = wrap_with_defaults(cfg)
    cfg = normalize_config_paths(cfg, base_dir)
    ensure_dir(cfg['models_dir'])
    ensure_dir(cfg['data_dir'])

    raster_path = normalize_path(cfg.get('gpkg_path'), base_dir) if cfg.get('gpkg_path') else None
    lines_path = normalize_path(cfg.get('lines_path') or cfg.get('linhas_path'), base_dir) if (cfg.get('lines_path') or cfg.get('linhas_path')) else None
    talhao_path = normalize_path(cfg.get('talhao_path'), base_dir) if cfg.get('talhao_path') else None

    if raster_path and lines_path and talhao_path:
        print('Executando pré-processamento para o dataset único...')
        run_preprocessing(
            raster_path=raster_path,
            lines_path=lines_path,
            talhao_path=talhao_path,
            output_dir=cfg['data_dir'],
            buffer_size=float(cfg['buffer_size']),
            tile_size=int(cfg['tile_size']),
            stride=int(cfg['stride']),
            val_size=float(cfg['val_size']),
            test_size=float(cfg['test_size']),
            append=bool(cfg.get('append', False)),
            logger=print
        )
        print()

    print('Iniciando treinamento do modelo (1 ortomosaico)...')
    saved_model = train_model(
        data_dir=cfg['data_dir'],
        models_dir=cfg['models_dir'],
        encoder_name=cfg['encoder_name'],
        epochs=int(cfg['epochs']),
        batch_size=int(cfg['batch_size']),
        lr=float(cfg['lr']),
        patience=int(cfg.get('early_stopping_patience', cfg.get('patience', 20))),
        min_delta=float(cfg.get('min_delta', 0.001)),
        num_orthos=1
    )
    print(f'Treinamento concluído com sucesso! Modelo salvo em: {saved_model}')


def train_infer_command(cfg, base_dir=None):
    from src.preprocess import run_preprocessing
    from src.train import train_model

    cfg = wrap_with_defaults(cfg)
    cfg = normalize_config_paths(cfg, base_dir)
    ensure_dir(cfg['models_dir'])
    ensure_dir(cfg['output_dir'])

    raster_path = normalize_path(cfg.get('gpkg_path'), base_dir)
    lines_path = normalize_path(cfg.get('linhas_path') or cfg.get('lines_path'), base_dir)
    talhao_path = normalize_path(cfg.get('talhao_path'), base_dir)

    print('Executando pré-processamento...')
    run_preprocessing(
        raster_path=raster_path,
        lines_path=lines_path,
        talhao_path=talhao_path,
        output_dir=cfg['data_dir'],
        buffer_size=float(cfg['buffer_size']),
        tile_size=int(cfg['tile_size']),
        stride=int(cfg['stride']),
        val_size=float(cfg['val_size']),
        test_size=float(cfg['test_size']),
        append=bool(cfg['append']),
        logger=print
    )

    print('Executando treinamento...')
    saved_model = train_model(
        data_dir=cfg['data_dir'],
        models_dir=cfg['models_dir'],
        encoder_name=cfg['encoder_name'],
        epochs=int(cfg['epochs']),
        batch_size=int(cfg['batch_size']),
        lr=float(cfg['lr']),
        num_orthos=1
    )

    checkpoint_path = cfg.get('checkpoint_path') or saved_model or os.path.join(cfg['models_dir'], 'best_model.pth')
    path_linhas = normalize_path(cfg.get('path_linhas'), base_dir)
    print(f'Executando inferência e pós-processamento com checkpoint: {checkpoint_path}')
    run_segmented_inference_pipeline(
        raster_path=raster_path,
        talhao_path=talhao_path,
        checkpoint_path=checkpoint_path,
        output_dir=cfg['output_dir'],
        tile_size=int(cfg['tile_size']),
        stride=int(cfg['stride']),
        threshold=float(cfg['threshold']),
        min_area_m2=float(cfg['min_area_m2']),
        close_size=int(cfg['close_size']),
        open_size=int(cfg['open_size']),
        skip_postprocess=bool(cfg['skip_postprocess']),
        export_as_lines=bool(cfg.get('export_as_lines', False)),
        path_linhas=path_linhas,
        profile=cfg.get('profile', 'tiro_reto'),
        calc_entrelinhas=bool(cfg.get('calc_entrelinhas', False)),
        extend_to_boundary=cfg.get('extend_to_boundary'),
        min_row_spacing_m=float(cfg.get('min_row_spacing_m', 4.0)),
        max_row_spacing_m=float(cfg.get('max_row_spacing_m', 8.5)),
        logger=print,
        centerline_method=cfg.get('centerline_method', 'skeleton_v2'),
        merge_method=cfg.get('merge_method', 'spatial_cost'),
        dedup_method=cfg.get('dedup_method', 'overlap_parallelism'),
        min_tile_coverage=float(cfg.get('min_tile_coverage', 0.0)),
        batch_size=int(cfg.get('batch_size', 4)),
        resample_step_m=float(cfg.get('resample_step_m', 1.0)),
        gsd_m=cfg.get('gsd_m'),
        prob_refinement_enabled=bool(cfg.get('prob_refinement_enabled', False)),
        prob_refinement_weight=float(cfg.get('prob_refinement_weight', 0.5)),
        prob_search_radius_m=float(cfg.get('prob_search_radius_m', 1.5)),
        prob_max_shift_m=float(cfg.get('prob_max_shift_m', 0.75)),
        curvature_control_enabled=bool(cfg.get('curvature_control_enabled', False)),
        max_curvature=float(cfg['max_curvature']) if cfg.get('max_curvature') is not None else None,
        corridor_validation_enabled=bool(cfg.get('corridor_validation_enabled', False)),
        topology_validation_enabled=bool(cfg.get('topology_validation_enabled', True)),
        write_run_metadata_file=bool(cfg.get('write_run_metadata_file', True)),
        field_partition_options=cfg.get('field_partition'),
    )
    print('Fluxo train_infer concluído.')


def infer_post_command(cfg, base_dir=None):
    cfg = wrap_with_defaults(cfg)
    cfg = normalize_config_paths(cfg, base_dir)
    ensure_dir(cfg['output_dir'])

    raster_path = normalize_path(cfg.get('gpkg_path'), base_dir)
    talhao_path = normalize_path(cfg.get('talhao_path'), base_dir)
    checkpoint_path = cfg.get('checkpoint_path') or os.path.join(cfg['models_dir'], 'best_model.pth')
    path_linhas = normalize_path(cfg.get('path_linhas'), base_dir)
    print(f'Executando inferência e pós-processamento [Perfil: {cfg.get("profile", "tiro_reto")}] com checkpoint: {checkpoint_path}')
    results = run_segmented_inference_pipeline(
        raster_path=raster_path,
        talhao_path=talhao_path,
        checkpoint_path=checkpoint_path,
        output_dir=cfg['output_dir'],
        tile_size=int(cfg['tile_size']),
        stride=int(cfg['stride']),
        threshold=float(cfg['threshold']),
        min_area_m2=float(cfg['min_area_m2']),
        close_size=int(cfg['close_size']),
        open_size=int(cfg['open_size']),
        skip_postprocess=bool(cfg['skip_postprocess']),
        export_as_lines=bool(cfg.get('export_as_lines', False)),
        path_linhas=path_linhas,
        profile=cfg.get('profile', 'tiro_reto'),
        calc_entrelinhas=bool(cfg.get('calc_entrelinhas', False)),
        extend_to_boundary=cfg.get('extend_to_boundary'),
        min_row_spacing_m=float(cfg.get('min_row_spacing_m', 4.0)),
        max_row_spacing_m=float(cfg.get('max_row_spacing_m', 8.5)),
        logger=print,
        centerline_method=cfg.get('centerline_method', 'skeleton_v2'),
        merge_method=cfg.get('merge_method', 'spatial_cost'),
        dedup_method=cfg.get('dedup_method', 'overlap_parallelism'),
        min_tile_coverage=float(cfg.get('min_tile_coverage', 0.0)),
        batch_size=int(cfg.get('batch_size', 4)),
        resample_step_m=float(cfg.get('resample_step_m', 1.0)),
        gsd_m=cfg.get('gsd_m'),
        prob_refinement_enabled=bool(cfg.get('prob_refinement_enabled', False)),
        prob_refinement_weight=float(cfg.get('prob_refinement_weight', 0.5)),
        prob_search_radius_m=float(cfg.get('prob_search_radius_m', 1.5)),
        prob_max_shift_m=float(cfg.get('prob_max_shift_m', 0.75)),
        curvature_control_enabled=bool(cfg.get('curvature_control_enabled', False)),
        max_curvature=float(cfg['max_curvature']) if cfg.get('max_curvature') is not None else None,
        corridor_validation_enabled=bool(cfg.get('corridor_validation_enabled', False)),
        topology_validation_enabled=bool(cfg.get('topology_validation_enabled', True)),
        write_run_metadata_file=bool(cfg.get('write_run_metadata_file', True)),
        field_partition_options=cfg.get('field_partition'),
    )

    if bool(cfg.get('calc_entrelinhas', False)):
        if not bool(cfg.get('export_as_lines', False)):
            raise ValueError('Para calcular entrelinhas, configure também export_as_lines: true.')
        lines_path = results.get('geojson')
        interrows_path, interrows_count = generate_interrows(
            lines_path=lines_path,
            talhao_path=results.get('talhoes_geojson') or talhao_path,
            output_dir=os.path.dirname(lines_path),
            min_spacing=float(cfg.get('min_row_spacing_m', 4.0)),
            max_spacing=float(cfg.get('max_row_spacing_m', 8.5)),
            base_dir=base_dir or os.path.dirname(os.path.abspath(__file__)),
            max_bridge_gap=float(cfg.get('interrow_max_bridge_gap_m', 0.75)),
        )
        results['entrelinhas_geojson'] = interrows_path
        results['num_entrelinhas'] = interrows_count
    print('Fluxo infer_post concluído.')
    return results


def resultado_command(cfg, base_dir=None):
    """Gera os shapefiles finais juntos em uma única pasta de saída."""
    cfg = wrap_with_defaults(cfg)
    cfg = normalize_config_paths(cfg, base_dir)

    talhao_path = normalize_path(cfg.get('talhao_path'), base_dir)
    mdt_path = normalize_path(cfg.get('mdt_path'), base_dir)
    if not talhao_path or not os.path.isfile(talhao_path):
        raise FileNotFoundError(f'Talhão não encontrado: {talhao_path}')
    if not mdt_path or not os.path.isfile(mdt_path):
        raise FileNotFoundError(f'MDT não encontrado: {mdt_path}')

    # O gerador existente recebe uma pasta mãe com subpastas por talhão.
    script_path = os.path.join(base_dir, 'scripts', 'gerar_bordaduras_lote', 'gerar_bordaduras_lote.py')
    if not os.path.isfile(script_path):
        raise FileNotFoundError(f'Script de bordadura não encontrado: {script_path}')

    final_output_dir = ensure_dir(cfg['output_dir'])
    # Todo o processamento intermediário fica temporário; só os shapefiles
    # finais e seus arquivos auxiliares são copiados para a pasta de entrega.
    with tempfile.TemporaryDirectory(prefix='citros_resultado_') as staging_dir:
        inference_dir = os.path.join(staging_dir, 'inferencia')
        bordadura_dir = os.path.join(staging_dir, 'bordaduras')
        staged_cfg = dict(cfg)
        staged_cfg['output_dir'] = inference_dir

        results = infer_post_command(staged_cfg, base_dir=base_dir)

        talhao_dir = os.path.join(staging_dir, 'entrada', 'talhao')
        os.makedirs(talhao_dir)
        # Use child polygons, with stable IDs, for both borders and routes.
        border_input = results.get('talhoes_geojson') or talhao_path
        staged_talhao = os.path.join(talhao_dir, 'talhoes.geojson')
        shutil.copy2(border_input, staged_talhao)
        command = [
            sys.executable, script_path,
            '--input_dir', os.path.dirname(talhao_dir),
            '--mdt_path', mdt_path,
            '--output_dir', bordadura_dir,
            '--buffer_distance', str(float(cfg.get('bordadura_buffer_distance', 7.0))),
            '--partition_resolution', str(float((cfg.get('field_partition') or {}).get('grid_resolution_m') or 1.0)),
            '--partition_max_cells', str(int((cfg.get('field_partition') or {}).get('max_grid_cells', 4_000_000))),
        ]
        print('=== Gerando bordadura do talhão com altitude do MDT ===')
        subprocess.run(command, check=True)

        normalize_result_shapefiles(
            (inference_dir, bordadura_dir),
            mdt_path,
            max_line_vertices=int(cfg.get('max_line_vertices', 80)),
            simplify_tolerance_m=float(cfg.get('line_simplify_tolerance_m', 0.1)),
            route_smoothing_radius_m=float(cfg.get('route_smoothing_radius_m', 0.5)),
            route_smoothing_min_turn_deg=float(cfg.get('route_smoothing_min_turn_deg', 15.0)),
            route_simplify_tolerance_m=float(cfg.get('route_simplify_tolerance_m', 0.035)),
            max_boundary_vertices=int(cfg.get('max_boundary_vertices', 100)),
            boundary_simplify_tolerance_m=float(cfg.get('boundary_simplify_tolerance_m', 0.1)),
        )

        shp_paths = []
        for search_root in (inference_dir, bordadura_dir):
            for root, _, filenames in os.walk(search_root):
                shp_paths.extend(
                    os.path.join(root, name)
                    for name in filenames
                    if name.lower().endswith('.shp')
                    and os.path.splitext(name)[0].lower() not in ('predicao_linhas', 'talhoes_separados')
                )
        if not shp_paths:
            raise FileNotFoundError('O fluxo terminou sem gerar arquivos Shapefile.')

        copied = set()
        for shp_path in shp_paths:
            stem = os.path.splitext(shp_path)[0]
            output_stem = os.path.basename(stem)
            if output_stem.lower() == 'predicao_entrelinhas':
                output_stem = cfg.get('lines_output_name', 'Linhas')
            elif output_stem.lower().startswith('rota_bordadura_'):
                output_stem = cfg.get('route_output_name', output_stem)
            for companion in os.listdir(os.path.dirname(shp_path)):
                if os.path.splitext(companion)[0] != os.path.basename(stem):
                    continue
                source = os.path.join(os.path.dirname(shp_path), companion)
                target_name = output_stem + os.path.splitext(companion)[1]
                target = os.path.join(final_output_dir, target_name)
                shutil.copy2(source, target)
                copied.add(target)

        metadata_path = os.path.join(os.path.dirname(results['geojson']), 'separacao_metadata.json')
        if os.path.isfile(metadata_path):
            target = os.path.join(final_output_dir, 'separacao_metadata.json')
            shutil.copy2(metadata_path, target)
            copied.add(target)

        # Remove this retired product from earlier runs of the same delivery.
        for filename in os.listdir(final_output_dir):
            stem, extension = os.path.splitext(filename.lower())
            if ((stem == 'talhoes_separados' and extension in
                 ('.shp', '.shx', '.dbf', '.prj', '.cpg', '.qix', '.sbn', '.sbx'))
                    or filename.lower() == 'talhoes_separados.shp.xml'):
                os.remove(os.path.join(final_output_dir, filename))

    print(f'Fluxo resultado concluído. {len(copied)} arquivo(s) Shapefile e auxiliares em: {final_output_dir}')

def post_only_command(cfg, base_dir=None):
    cfg = wrap_with_defaults(cfg)
    cfg = normalize_config_paths(cfg, base_dir)
    ensure_dir(cfg['output_dir'])

    input_raster = cfg.get('input_raster') or os.path.join(cfg['data_dir'], 'outputs', 'predicao_linhas.tif')
    input_raster = normalize_path(input_raster, base_dir)
    if not os.path.exists(input_raster):
        raise FileNotFoundError(f'Arquivo de entrada não encontrado: {input_raster}')

    talhao_path = cfg.get('talhao_path')
    talhao_path = normalize_path(talhao_path, base_dir)
    if not talhao_path or not os.path.exists(talhao_path):
        print('[WARN] Talhão não encontrado. Usando talhão sintético.')
        talhao_path = normalize_path('data/synthetic/talhao.geojson', base_dir)

    prob_map, transform, crs = read_prob_map_from_raster(input_raster)
    talhao_gdf = gpd.read_file(talhao_path).to_crs(crs)
    talhao_geom = talhao_gdf.unary_union

    path_linhas = normalize_path(cfg.get('path_linhas'), base_dir)

    print(f'Executando pós-processamento sobre máscara existente [Perfil: {cfg.get("profile", "tiro_reto")}]...')
    run_postprocessing(
        prob_map=prob_map,
        transform=transform,
        crs=crs,
        talhao_geom=talhao_geom,
        output_dir=cfg['output_dir'],
        threshold=float(cfg['threshold']),
        min_area_m2=float(cfg['min_area_m2']),
        close_size=int(cfg['close_size']),
        open_size=int(cfg['open_size']),
        skip_postprocess=bool(cfg['skip_postprocess']),
        export_as_lines=bool(cfg.get('export_as_lines', False)),
        path_linhas=path_linhas,
        profile=cfg.get('profile', 'tiro_reto'),
        calc_entrelinhas=bool(cfg.get('calc_entrelinhas', False)),
        extend_to_boundary=cfg.get('extend_to_boundary'),
        min_row_spacing_m=float(cfg.get('min_row_spacing_m', 4.0)),
        max_row_spacing_m=float(cfg.get('max_row_spacing_m', 8.5)),
        logger=print,
        centerline_method=cfg.get('centerline_method', 'skeleton_v2'),
        merge_method=cfg.get('merge_method', 'spatial_cost'),
        dedup_method=cfg.get('dedup_method', 'overlap_parallelism'),
        resample_step_m=float(cfg.get('resample_step_m', 1.0)),
        gsd_m=cfg.get('gsd_m'),
        prob_refinement_enabled=bool(cfg.get('prob_refinement_enabled', False)),
        prob_refinement_weight=float(cfg.get('prob_refinement_weight', 0.5)),
        prob_search_radius_m=float(cfg.get('prob_search_radius_m', 1.5)),
        prob_max_shift_m=float(cfg.get('prob_max_shift_m', 0.75)),
        curvature_control_enabled=bool(cfg.get('curvature_control_enabled', False)),
        max_curvature=float(cfg['max_curvature']) if cfg.get('max_curvature') is not None else None,
        corridor_validation_enabled=bool(cfg.get('corridor_validation_enabled', False)),
        topology_validation_enabled=bool(cfg.get('topology_validation_enabled', True)),
        write_run_metadata_file=bool(cfg.get('write_run_metadata_file', True)),
        parent_fields=talhao_gdf,
        field_partition_options=cfg.get('field_partition'),
    )
    print('Fluxo post_only concluído.')


def main():
    parser = argparse.ArgumentParser(description='CLI para executar o fluxo de treinamento e inferência local')
    subparsers = parser.add_subparsers(dest='command', required=True)

    parser_train = subparsers.add_parser('train', help='Executa apenas o treinamento do modelo')
    parser_train.add_argument('--config', '-c', required=True, help='Arquivo de configuração YAML')

    parser_train_multi = subparsers.add_parser('train_multi', help='Treina 1 modelo com MÚLTIPLOS datasets de uma vez')
    parser_train_multi.add_argument('--config', '-c', required=True, help='Arquivo de configuração YAML com lista de datasets')

    parser_train_infer = subparsers.add_parser('train_infer', help='Executa treinamento e inferência em sequência')
    parser_train_infer.add_argument('--config', '-c', required=True, help='Arquivo de configuração YAML')

    parser_infer_post = subparsers.add_parser('infer_post', help='Executa inferência e pós-processamento a partir de checkpoint')
    parser_infer_post.add_argument('--config', '-c', required=True, help='Arquivo de configuração YAML')

    parser_resultado = subparsers.add_parser('resultado', help='Executa inferência e gera a bordadura do talhão')
    parser_resultado.add_argument('--config', '-c', required=True, help='Arquivo de configuração YAML do fluxo completo')

    parser_post_only = subparsers.add_parser('post_only', help='Executa apenas pós-processamento sobre máscara existente')
    parser_post_only.add_argument('--config', '-c', required=True, help='Arquivo de configuração YAML')

    args = parser.parse_args()
    cfg = load_config(args.config)
    base_dir = os.path.dirname(os.path.abspath(__file__))

    if args.command == 'train':
        train_command(cfg, base_dir=base_dir)
    elif args.command == 'train_multi':
        train_multi_command(cfg, base_dir=base_dir)
    elif args.command == 'train_infer':
        train_infer_command(cfg, base_dir=base_dir)
    elif args.command == 'infer_post':
        infer_post_command(cfg, base_dir=base_dir)
    elif args.command == 'resultado':
        resultado_command(cfg, base_dir=base_dir)
    elif args.command == 'post_only':
        post_only_command(cfg, base_dir=base_dir)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == '__main__':
    main()
