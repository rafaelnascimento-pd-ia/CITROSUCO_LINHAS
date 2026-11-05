# -*- coding: utf-8 -*-
import argparse
import os
import sys

# Desativa PyGEOS para compatibilidade com Shapely 2.0+ e suprime warning
os.environ['USE_PYGEOS'] = '0'
# Permite OpenMP duplicado entre PyTorch e OpenCV no Windows
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import yaml
import torch
import rasterio
import geopandas as gpd

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
        write_run_metadata_file=bool(cfg.get('write_run_metadata_file', True))
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
        write_run_metadata_file=bool(cfg.get('write_run_metadata_file', True))
    )
    print('Fluxo infer_post concluído.')


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
        write_run_metadata_file=bool(cfg.get('write_run_metadata_file', True))
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
    elif args.command == 'post_only':
        post_only_command(cfg, base_dir=base_dir)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == '__main__':
    main()
