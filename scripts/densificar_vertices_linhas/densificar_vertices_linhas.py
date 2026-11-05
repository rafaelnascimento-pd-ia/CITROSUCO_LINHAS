#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
densificar_vertices_linhas.py
--------------------------------------------------------------------------------
Script de pós-processamento geográfico para reamostrar e densificar vértices em 
linhas de plantio a cada N metros (padrão 10m).

Navega por toda a estrutura de subpastas aninhadas da pasta mãe, aplica a 
densificação de vértices nas linhas e copia os demais arquivos (talhões, 
bordaduras e arquivos auxiliares) mantendo a estrutura idêntica na pasta de saída.
--------------------------------------------------------------------------------
"""

import os
import sys
import shutil
import argparse
import numpy as np
import geopandas as gpd
from shapely.geometry import LineString, MultiLineString, Point, Polygon, MultiPolygon
from pyproj import CRS

def get_driver(filepath):
    """Retorna o driver do fiona adequado para a extensão do arquivo."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext == '.shp':
        return 'ESRI Shapefile'
    elif ext == '.geojson':
        return 'GeoJSON'
    elif ext == '.gpkg':
        return 'GPKG'
    return 'ESRI Shapefile'

def densify_linestring(geom, step_meters):
    """
    Reamostra uma LineString inserindo vértices a cada `step_meters`.
    Preserva a elevação (Z) se a geometria for 3D.
    """
    if geom is None or geom.is_empty:
        return geom
        
    length = geom.length
    if length <= 0:
        return geom

    # Gerar distâncias ao longo da linha
    distances = np.arange(0, length, step_meters)
    if len(distances) == 0 or not np.isclose(distances[-1], length):
        distances = np.append(distances, length)

    has_z = geom.has_z
    new_coords = []
    
    for dist in distances:
        pt = geom.interpolate(dist)
        if has_z:
            new_coords.append((pt.x, pt.y, pt.z))
        else:
            new_coords.append((pt.x, pt.y))

    if len(new_coords) < 2:
        return geom

    return LineString(new_coords)

def densify_geometry(geom, step_meters):
    """Aplica a densificação para LineString ou MultiLineString."""
    if geom is None or geom.is_empty:
        return geom
        
    if geom.geom_type == 'LineString':
        return densify_linestring(geom, step_meters)
    elif geom.geom_type == 'MultiLineString':
        parts = [densify_linestring(part, step_meters) for part in geom.geoms]
        return MultiLineString(parts)
    return geom

def is_line_file(filepath, gdf):
    """Determina se o arquivo vetorial contém geometrias de linhas."""
    fname_lower = os.path.basename(filepath).lower()
    if "linha" in fname_lower or "lines" in fname_lower:
        return True
    
    geom_types = gdf.geometry.geom_type.dropna().unique()
    for gtype in geom_types:
        if "LineString" in gtype:
            return True
    return False

def process_file(in_file, out_file, distance_meters):
    """Processa um arquivo espacial de linha densificando vértices."""
    try:
        gdf = gpd.read_file(in_file)
        if gdf.empty:
            os.makedirs(os.path.dirname(out_file), exist_ok=True)
            shutil.copy2(in_file, out_file)
            return True

        orig_crs = gdf.crs

        # 1. Determinar CRS métrico para interpolação correta em metros
        if orig_crs and orig_crs.is_geographic:
            metric_crs = gdf.estimate_utm_crs()
        else:
            metric_crs = orig_crs or "EPSG:3857"

        # Garantir CRS 2D para projeção
        if isinstance(metric_crs, str):
            metric_crs = CRS.from_user_input(metric_crs)
        if hasattr(metric_crs, "to_2d"):
            metric_crs = metric_crs.to_2d()

        gdf_metric = gdf.to_crs(metric_crs)

        # 2. Densificar vértices a cada N metros
        densified_geoms = []
        for geom in gdf_metric.geometry:
            densified_geoms.append(densify_geometry(geom, distance_meters))

        gdf_metric['geometry'] = densified_geoms

        # 3. Reprojetar de volta para o CRS original (ou WGS84 se indefinido)
        target_crs = orig_crs if orig_crs else "EPSG:4326"
        gdf_out = gdf_metric.to_crs(target_crs)

        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        driver = get_driver(out_file)

        gdf_out.to_file(out_file, driver=driver)
        print(f"  [OK - LINHAS DENSIFICADAS] Salvo: {os.path.basename(out_file)} (Espaçamento: {distance_meters}m)")
        return True
    except Exception as e:
        print(f"  [AVISO] Falha ao processar {os.path.basename(in_file)}: {e}. Copiando original...")
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        shutil.copy2(in_file, out_file)
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Insere vértices a cada N metros em arquivos de linhas de plantio em toda a árvore de subpastas.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-i", "--input", required=True,
        help="Caminho do diretório raiz (pasta mãe) de entrada."
    )
    parser.add_argument(
        "-o", "--output", required=True,
        help="Caminho do diretório de saída."
    )
    parser.add_argument(
        "-dist", "--distance", type=float, default=10.0,
        help="Distância de espaçamento entre vértices em metros."
    )

    args = parser.parse_args()

    input_dir = os.path.abspath(args.input)
    output_dir = os.path.abspath(args.output)
    distance_meters = args.distance

    if not os.path.exists(input_dir):
        print(f"[ERRO] Diretório de entrada não encontrado: {input_dir}")
        sys.exit(1)

    print("=" * 75)
    print("      DENSIFICAÇÃO DE VÉRTICES DE LINHAS (ESPAÇAMENTO REGULAR)")
    print("=" * 75)
    print(f"Entrada              : {input_dir}")
    print(f"Saída                : {output_dir}")
    print(f"Espaçamento Vértices : {distance_meters} metros")
    print("-" * 75)

    spatial_exts = ('.shp', '.geojson', '.gpkg')
    sidecar_exts = ('.dbf', '.prj', '.cpg', '.shx', '.sbn', '.sbx')

    files_processed = 0
    files_copied = 0

    for root, dirs, files in os.walk(input_dir):
        # Ignorar pasta de saída se estiver dentro da entrada
        try:
            if os.path.commonpath([output_dir, root]) == output_dir:
                continue
        except ValueError:
            pass

        rel_path = os.path.relpath(root, input_dir)
        current_out_dir = os.path.join(output_dir, rel_path) if rel_path != "." else output_dir
        os.makedirs(current_out_dir, exist_ok=True)

        for f in files:
            in_file = os.path.join(root, f)
            out_file = os.path.join(current_out_dir, f)
            f_lower = f.lower()

            if f_lower.endswith(spatial_exts):
                try:
                    gdf_check = gpd.read_file(in_file)
                    if is_line_file(in_file, gdf_check):
                        process_file(in_file, out_file, distance_meters)
                        files_processed += 1
                    else:
                        # É um polígono/ponto (não-linha), copiar arquivo e seus auxiliares
                        print(f"  [COPIANDO NÃO-LINHA] {f}")
                        shutil.copy2(in_file, out_file)
                        files_copied += 1
                except Exception:
                    shutil.copy2(in_file, out_file)
                    files_copied += 1
            else:
                # Arquivo não espacial ou sidecar (.dbf, .prj, etc) - copiar se não existir
                if not os.path.exists(out_file):
                    shutil.copy2(in_file, out_file)
                    files_copied += 1

    print("\n" + "=" * 75)
    print(f" DENSIFICAÇÃO CONCLUÍDA!")
    print(f" Arquivos de linhas reamostrados : {files_processed}")
    print(f" Demais arquivos mantidos/copiados: {files_copied}")
    print(f" Diretório de saída final         : {output_dir}")
    print("=" * 75)

if __name__ == "__main__":
    main()
