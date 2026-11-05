#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
herdar_altitude_mdt.py
--------------------------------------------------------------------------------
Script para projetar a altitude (coordenada Z) de um Modelo Digital de Terreno (MDT)
nas linhas de entrelinhas, gerando arquivos 3D (LineStringZ) e adicionando atributos
estatísticos de elevação e declividade.
--------------------------------------------------------------------------------
"""

import os
import sys
import argparse
import datetime
import numpy as np
import geopandas as gpd
import rasterio
from shapely.geometry import LineString


def main():
    parser = argparse.ArgumentParser(
        description="Projeta a altitude (Z) de um raster DTM/MDT nas geometrias das entrelinhas.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--input_lines", "-i", required=True,
        help="Caminho do arquivo de linhas de plantio ou entrelinhas (GeoJSON ou Shapefile)."
    )
    parser.add_argument(
        "--mdt_path", "-m", required=True,
        help="Caminho do raster DTM/MDT (GeoTIFF)."
    )
    parser.add_argument(
        "--output_dir", "-o", default="data/outputs/entrelinhas_3d",
        help="Diretório onde as saídas serão salvas."
    )
    parser.add_argument(
        "--timestamp", default=None,
        help="Sufixo de timestamp opcional para os nomes de arquivo (formato YYYYMMDD_HHMMSS)."
    )

    args = parser.parse_args()

    # 1. Obter timestamp
    timestamp = args.timestamp
    if not timestamp:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 70)
    print("   CITROSROWS AI - INJEÇÃO DE ALTITUDE (Z) NAS ENTRELINHAS (3D)")
    print("=" * 70)
    print(f"Linhas de Entrada  : {args.input_lines}")
    print(f"Raster DTM/MDT     : {args.mdt_path}")
    print(f"Diretório de Saída : {args.output_dir}")
    print("-" * 70)

    # 2. Carregar arquivos e validar
    if not os.path.exists(args.input_lines):
        print(f"[ERRO] Arquivo de linhas não encontrado: {args.input_lines}")
        sys.exit(1)
    if not os.path.exists(args.mdt_path):
        print(f"[ERRO] Raster MDT não encontrado: {args.mdt_path}")
        sys.exit(1)

    try:
        gdf_lines = gpd.read_file(args.input_lines)
        src_mdt = rasterio.open(args.mdt_path)
    except Exception as e:
        print(f"[ERRO] Falha ao ler arquivos: {e}")
        sys.exit(1)

    if gdf_lines.empty:
        print("[WARN] Arquivo de linhas de entrada está vazio.")
        sys.exit(0)

    original_crs = gdf_lines.crs
    print(f"[OK] {len(gdf_lines)} linhas carregadas.")
    print(f"[OK] MDT carregado. Resolução: {src_mdt.res} | CRS: {src_mdt.crs.to_string()}")

    # 3. Remover todos os atributos originais para começar do zero (apenas geometria é mantida)
    geom_col = gdf_lines.geometry.name
    gdf_lines = gdf_lines[[geom_col]].copy()

    # Explodir MultiLineStrings
    gdf_lines = gdf_lines.explode(index_parts=False)

    # Reprojetar para o mesmo CRS do MDT
    print("[INFO] Reprojetando linhas para o CRS do MDT para amostragem...")
    gdf_proj = gdf_lines.to_crs(src_mdt.crs)

    # 4. Acumular coordenadas em lote para amostragem super rápida (vértices e ponto médio)
    all_coords = []
    line_lengths = []
    midpoint_coords = []
    for geom in gdf_proj.geometry:
        if geom is None or geom.is_empty:
            line_lengths.append(0)
            midpoint_coords.append(None)
            continue
            
        # Se for um Polígono ou MultiPolígono, extrai o contorno (boundary/exterior)
        if geom.geom_type in ['Polygon', 'MultiPolygon']:
            geom_linear = geom.boundary
            # Se o boundary for MultiLineString, pega a maior parte para simplificar
            if geom_linear.geom_type == 'MultiLineString':
                geom_linear = max(geom_linear.geoms, key=lambda g: g.length) if hasattr(geom_linear, 'geoms') else geom_linear
        else:
            geom_linear = geom
            
        coords = [(c[0], c[1]) for c in geom_linear.coords]
        all_coords.extend(coords)
        line_lengths.append(len(coords))
        
        # Ponto médio (50% do comprimento)
        mid_pt = geom_linear.interpolate(0.5, normalized=True)
        midpoint_coords.append((mid_pt.x, mid_pt.y))

    if not all_coords:
        print("[ERRO] Nenhuma coordenada de linha encontrada para amostragem.")
        sys.exit(1)

    # Filtrar midpoints válidos para amostragem
    valid_midpoints = [pt for pt in midpoint_coords if pt is not None]

    # 5. Amostragem em lote no MDT
    print(f"[INFO] Amostrando {len(all_coords) + len(valid_midpoints)} pontos no raster MDT...")
    try:
        # Junta todas as coordenadas para amostrar em uma única chamada de E/S
        sampled_vals = list(src_mdt.sample(all_coords + valid_midpoints))
    except Exception as e:
        print(f"[ERRO] Falha na amostragem no MDT: {e}")
        sys.exit(1)

    nodata = src_mdt.nodata
    z_values = []
    for val in sampled_vals[:len(all_coords)]:
        z = float(val[0])
        # Tratar valor nodata do raster
        if np.isnan(z) or (nodata is not None and np.isclose(z, nodata)):
            z = 0.0
        z_values.append(z)

    # Valores de Z para os pontos médios
    mid_z_vals = []
    mid_idx = len(all_coords)
    for pt in midpoint_coords:
        if pt is None:
            mid_z_vals.append(0.0)
            continue
        val = sampled_vals[mid_idx]
        mid_idx += 1
        z = float(val[0])
        if np.isnan(z) or (nodata is not None and np.isclose(z, nodata)):
            z = 0.0
        mid_z_vals.append(z)

    # 6. Reconstruir geometrias 3D (LineStringZ) e calcular estatísticas de declividade
    new_geometries = []
    start_zs = []
    end_zs = []
    min_zs = []
    max_zs = []
    mean_zs = []
    slope_pcts = []

    idx = 0
    for geom, length in zip(gdf_proj.geometry, line_lengths):
        if length == 0 or geom is None:
            new_geometries.append(None)
            start_zs.append(0.0)
            end_zs.append(0.0)
            min_zs.append(0.0)
            max_zs.append(0.0)
            mean_zs.append(0.0)
            slope_pcts.append(0.0)
            continue

        line_coords = all_coords[idx : idx + length]
        line_zs = z_values[idx : idx + length]
        idx += length

        # Montar coordenadas 3D (X, Y, Z)
        coords_3d = [(x, y, z) for (x, y), z in zip(line_coords, line_zs)]
        new_line = LineString(coords_3d)
        new_geometries.append(new_line)

        # Estatísticas
        start_z = line_zs[0]
        end_z = line_zs[-1]
        min_z = float(np.min(line_zs))
        max_z = float(np.max(line_zs))
        mean_z = float(np.mean(line_zs))

        # Comprimento horizontal da linha no CRS métrico (projetado)
        horiz_len = geom.length
        if horiz_len > 0:
            slope = ((end_z - start_z) / horiz_len) * 100
        else:
            slope = 0.0

        start_zs.append(start_z)
        end_zs.append(end_z)
        min_zs.append(min_z)
        max_zs.append(max_z)
        mean_zs.append(mean_z)
        slope_pcts.append(slope)

    # Adicionar as novas colunas e geometria reconstruída no DataFrame
    gdf_proj['geometry'] = new_geometries
    gdf_proj['start_z'] = start_zs
    gdf_proj['end_z'] = end_zs
    gdf_proj['min_z'] = min_zs
    gdf_proj['max_z'] = max_zs
    gdf_proj['mean_z'] = mean_zs
    gdf_proj['slope_pct'] = slope_pcts
    gdf_proj['z'] = mid_z_vals

    # 7. Adicionar/recalcular atributos de coordenadas X, Y, Z (lat, lon, x, y, z) utilizando o MDT como referência de CRS
    print("[INFO] Calculando atributos de coordenadas geográficas (WGS84) e métricas (UTM/MDT)...")
    
    # Determinar o CRS métrico para cálculo de X, Y e comprimento (Length)
    # Se o CRS do MDT for projetado, usamos ele diretamente. Caso contrário, estimamos o UTM.
    if src_mdt.crs.is_projected:
        metric_crs = src_mdt.crs
    else:
        try:
            metric_crs = gdf_proj.estimate_utm_crs()
        except Exception:
            metric_crs = "EPSG:3857" # Fallback caso falhe
            
    print(f"[INFO] CRS métrico de referência: {metric_crs.to_string() if hasattr(metric_crs, 'to_string') else metric_crs}")
    
    # Reprojetar as geometrias 3D para o CRS métrico
    gdf_metric = gdf_proj.to_crs(metric_crs)
    
    # Criar transformador de coordenadas do CRS métrico para WGS84 (EPSG:4326)
    # always_xy=True garante que o resultado seja (longitude, latitude)
    from pyproj import Transformer
    transformer = Transformer.from_crs(metric_crs, "EPSG:4326", always_xy=True)
    
    lat_list, lon_list, x_list, y_list = [], [], [], []
    start_lat_list, start_lon_list, start_x_list, start_y_list = [], [], [], []
    end_lat_list, end_lon_list, end_x_list, end_y_list = [], [], [], []
    lengths_list = []
    
    for k in range(len(gdf_metric)):
        geom_metric = gdf_metric.iloc[k].geometry
        
        if geom_metric is None or geom_metric.is_empty:
            lat_list.append(0.0)
            lon_list.append(0.0)
            x_list.append(0.0)
            y_list.append(0.0)
            start_lat_list.append(0.0)
            start_lon_list.append(0.0)
            start_x_list.append(0.0)
            start_y_list.append(0.0)
            end_lat_list.append(0.0)
            end_lon_list.append(0.0)
            end_x_list.append(0.0)
            end_y_list.append(0.0)
            lengths_list.append(0.0)
            continue
            
        # 1. Comprimento horizontal real no CRS métrico (MDT/UTM)
        h_len = geom_metric.length
        lengths_list.append(h_len)
        
        # 2. Ponto de início (start) no CRS métrico
        s_coord = geom_metric.coords[0]
        sx, sy = s_coord[0], s_coord[1]
        slon, slat = transformer.transform(sx, sy)
        
        start_x_list.append(sx)
        start_y_list.append(sy)
        start_lon_list.append(slon)
        start_lat_list.append(slat)
        
        # 3. Ponto de fim (end) no CRS métrico
        e_coord = geom_metric.coords[-1]
        ex, ey = e_coord[0], e_coord[1]
        elon, elat = transformer.transform(ex, ey)
        
        end_x_list.append(ex)
        end_y_list.append(ey)
        end_lon_list.append(elon)
        end_lat_list.append(elat)
        
        # 4. Ponto médio (midpoint) calculado ao longo do comprimento métrico real
        mid_pt = geom_metric.interpolate(0.5, normalized=True)
        mx, my = mid_pt.x, mid_pt.y
        mlon, mlat = transformer.transform(mx, my)
        
        x_list.append(mx)
        y_list.append(my)
        lon_list.append(mlon)
        lat_list.append(mlat)
        
    gdf_proj['lat'] = lat_list
    gdf_proj['lon'] = lon_list
    gdf_proj['x'] = x_list
    gdf_proj['y'] = y_list
    gdf_proj['start_lat'] = start_lat_list
    gdf_proj['start_lon'] = start_lon_list
    gdf_proj['start_x'] = start_x_list
    gdf_proj['start_y'] = start_y_list
    gdf_proj['end_lat'] = end_lat_list
    gdf_proj['end_lon'] = end_lon_list
    gdf_proj['end_x'] = end_x_list
    gdf_proj['end_y'] = end_y_list
    gdf_proj['Length'] = lengths_list

    # 8. Reprojetar para o CRS WGS84 (EPSG:4326) e forçar Single Part
    print("[INFO] Reprojetando dados 3D finais para WGS84 (EPSG:4326) e garantindo Single Part...")
    gdf_out = gdf_proj.to_crs("EPSG:4326")
    gdf_out = gdf_out.explode(index_parts=False)

    # 9. Salvar saídas (GeoJSON e Shapefile 3D)
    os.makedirs(args.output_dir, exist_ok=True)

    geojson_out = os.path.join(args.output_dir, "entrelinhas_3d.geojson")
    geojson_ts_out = os.path.join(args.output_dir, f"entrelinhas_3d_{timestamp}.geojson")

    shp_dir = os.path.join(args.output_dir, "shp")
    shp_out = os.path.join(shp_dir, "entrelinhas_3d.shp")
    shp_ts_out = os.path.join(shp_dir, f"entrelinhas_3d_{timestamp}.shp")

    print("-" * 70)
    print(f"Salvando resultados em: {args.output_dir}")

    # Exportar GeoJSON 3D
    try:
        gdf_out.to_file(geojson_out, driver="GeoJSON")
        gdf_out.to_file(geojson_ts_out, driver="GeoJSON")
        print(f"[OK] GeoJSON 3D salvo com sucesso: {geojson_out}")
        print(f"[OK] GeoJSON 3D (timestamp) salvo: {geojson_ts_out}")
    except Exception as e:
        print(f"[WARN] Falha ao exportar GeoJSON: {e}. Usando fallback...")
        try:
            with open(geojson_out, 'w', encoding='utf-8') as f:
                f.write(gdf_out.to_json())
            with open(geojson_ts_out, 'w', encoding='utf-8') as f:
                f.write(gdf_out.to_json())
            print(f"[OK] GeoJSON 3D salvo via fallback.")
        except Exception as ex:
            print(f"[ERRO] Falha total ao exportar GeoJSON: {ex}")

    # Exportar Shapefile 3D (LineStringZ)
    try:
        os.makedirs(shp_dir, exist_ok=True)
        gdf_out.to_file(shp_out)
        gdf_out.to_file(shp_ts_out)
        print(f"[OK] Shapefile 3D (LineStringZ) salvo com sucesso: {shp_out}")
        print(f"[OK] Shapefile 3D (timestamp) salvo: {shp_ts_out}")
    except Exception as e:
        print(f"[WARN] Falha ao exportar Shapefile: {e}")

    # Fechar raster
    src_mdt.close()

    print("=" * 70)
    print("           PROCESSAMENTO CONCLUÍDO COM SUCESSO!")
    print("=" * 70)


if __name__ == "__main__":
    main()
