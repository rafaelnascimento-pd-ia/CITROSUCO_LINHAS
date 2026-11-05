#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
herdar_altitude_mdt_poligonos.py
--------------------------------------------------------------------------------
Script para projetar a altitude (coordenada Z) de um Modelo Digital de Terreno (MDT)
nos vértices de polígonos, gerando arquivos 3D (PolygonZ/MultiPolygonZ) e preservando
todos os atributos originais dos polígonos.
--------------------------------------------------------------------------------
"""

import os
import sys
import argparse
import datetime
import numpy as np
import geopandas as gpd
import rasterio
from shapely.geometry import Polygon, MultiPolygon


def make_polygon_3d(geom, mdt_src):
    """
    Converte um Polygon ou MultiPolygon 2D para 3D (PolygonZ / MultiPolygonZ)
    amostrando Z no MDT para cada vértice.
    """
    if geom is None or geom.is_empty:
        return geom

    def process_ring(ring_coords):
        xy_coords = [(coords[0], coords[1]) for coords in ring_coords]
        sampled = list(mdt_src.sample(xy_coords))
        new_coords = []
        for (x, y), val in zip(xy_coords, sampled):
            z = float(val[0])
            if np.isnan(z) or (mdt_src.nodata is not None and np.isclose(z, mdt_src.nodata)):
                z = 0.0
            new_coords.append((x, y, z))
        return new_coords

    if geom.geom_type == 'Polygon':
        ext_coords = list(geom.exterior.coords)
        new_ext = process_ring(ext_coords)
        new_ints = []
        for interior in geom.interiors:
            int_coords = list(interior.coords)
            new_ints.append(process_ring(int_coords))
        return Polygon(new_ext, new_ints)

    elif geom.geom_type == 'MultiPolygon':
        parts = []
        for poly in geom.geoms:
            parts.append(make_polygon_3d(poly, mdt_src))
        return MultiPolygon(parts)

    return geom


def main():
    parser = argparse.ArgumentParser(
        description="Projeta a altitude (Z) de um raster DTM/MDT nas geometrias de polígonos, mantendo-os como polígonos 3D.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--input_polygons", "-i", required=True,
        help="Caminho do arquivo de polígonos de entrada (GeoJSON ou Shapefile)."
    )
    parser.add_argument(
        "--mdt_path", "-m", required=True,
        help="Caminho do raster DTM/MDT (GeoTIFF)."
    )
    parser.add_argument(
        "--output_dir", "-o", default="data/outputs/poligonos_3d",
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
    print("   CITROSROWS AI - INJEÇÃO DE ALTITUDE (Z) EM POLÍGONOS (3D)")
    print("=" * 70)
    print(f"Polígonos de Entrada : {args.input_polygons}")
    print(f"Raster DTM/MDT       : {args.mdt_path}")
    print(f"Diretório de Saída   : {args.output_dir}")
    print("-" * 70)

    # 2. Carregar arquivos e validar
    if not os.path.exists(args.input_polygons):
        print(f"[ERRO] Arquivo de polígonos não encontrado: {args.input_polygons}")
        sys.exit(1)
    if not os.path.exists(args.mdt_path):
        print(f"[ERRO] Raster MDT não encontrado: {args.mdt_path}")
        sys.exit(1)

    try:
        gdf_polys = gpd.read_file(args.input_polygons)
        src_mdt = rasterio.open(args.mdt_path)
    except Exception as e:
        print(f"[ERRO] Falha ao ler arquivos: {e}")
        sys.exit(1)

    if gdf_polys.empty:
        print("[WARN] Arquivo de polígonos de entrada está vazio.")
        sys.exit(0)

    print(f"[OK] {len(gdf_polys)} feições carregadas.")
    print(f"[OK] MDT carregado. Resolução: {src_mdt.res} | CRS: {src_mdt.crs.to_string()}")

    # Reprojetar para o mesmo CRS do MDT
    print("[INFO] Reprojetando polígonos para o CRS do MDT para amostragem...")
    gdf_proj = gdf_polys.to_crs(src_mdt.crs)

    # 3. Converter geometrias para 3D (PolygonZ / MultiPolygonZ)
    print("[INFO] Amostrando altitudes no raster MDT para cada vértice dos polígonos...")
    geoms_3d = []
    for geom in gdf_proj.geometry:
        geom_3d = make_polygon_3d(geom, src_mdt)
        geoms_3d.append(geom_3d)

    gdf_proj['geometry'] = geoms_3d

    # 4. Adicionar/recalcular atributos de coordenadas X, Y, Z (lat, lon, x, y, z) utilizando o MDT como referência de CRS
    print("[INFO] Calculando atributos de coordenadas geográficas (Centróides em WGS84) e estatísticas de elevação...")

    # Determinar o CRS métrico para cálculo de X, Y
    if src_mdt.crs.is_projected:
        metric_crs = src_mdt.crs
    else:
        try:
            metric_crs = gdf_proj.estimate_utm_crs()
        except Exception:
            metric_crs = "EPSG:3857" # Fallback

    # Garantir que o CRS métrico seja 2D
    from pyproj import CRS
    if isinstance(metric_crs, str):
        metric_crs = CRS.from_user_input(metric_crs)
    if hasattr(metric_crs, "to_2d"):
        metric_crs = metric_crs.to_2d()

    print(f"[INFO] CRS métrico de referência: {metric_crs.to_string() if hasattr(metric_crs, 'to_string') else metric_crs}")

    # Reprojetar para o CRS métrico para centróide correto
    gdf_metric = gdf_proj.to_crs(metric_crs)

    # Criar transformador de coordenadas do CRS métrico para WGS84 (EPSG:4326)
    from pyproj import Transformer
    transformer = Transformer.from_crs(metric_crs, "EPSG:4326", always_xy=True)

    x_list, y_list, z_list = [], [], []
    lat_list, lon_list = [], []
    min_zs, max_zs, mean_zs = [], [], []

    for k in range(len(gdf_metric)):
        geom_m = gdf_metric.iloc[k].geometry
        geom_proj = gdf_proj.iloc[k].geometry

        if geom_m is None or geom_m.is_empty:
            x_list.append(0.0)
            y_list.append(0.0)
            z_list.append(0.0)
            lat_list.append(0.0)
            lon_list.append(0.0)
            min_zs.append(0.0)
            max_zs.append(0.0)
            mean_zs.append(0.0)
            continue

        # A. Centróide da geometria no CRS métrico
        centroid = geom_m.centroid
        cx, cy = centroid.x, centroid.y
        x_list.append(cx)
        y_list.append(cy)

        # Converter centróide para lat, lon (WGS84)
        clon, clat = transformer.transform(cx, cy)
        lat_list.append(clat)
        lon_list.append(clon)

        # B. Amostrar Z do centróide no MDT
        sampled = list(src_mdt.sample([(cx, cy)]))
        cz = float(sampled[0][0])
        if np.isnan(cz) or (src_mdt.nodata is not None and np.isclose(cz, src_mdt.nodata)):
            cz = 0.0
        z_list.append(cz)

        # C. Estatísticas de Z das coordenadas 3D do polígono
        poly_z_vals = []
        if geom_proj:
            if geom_proj.geom_type == 'Polygon':
                for coord in geom_proj.exterior.coords:
                    if len(coord) >= 3:
                        poly_z_vals.append(coord[2])
                for interior in geom_proj.interiors:
                    for coord in interior.coords:
                        if len(coord) >= 3:
                            poly_z_vals.append(coord[2])
            elif geom_proj.geom_type == 'MultiPolygon':
                for poly in geom_proj.geoms:
                    for coord in poly.exterior.coords:
                        if len(coord) >= 3:
                            poly_z_vals.append(coord[2])
                    for interior in poly.interiors:
                        for coord in interior.coords:
                            if len(coord) >= 3:
                                poly_z_vals.append(coord[2])

        if poly_z_vals:
            min_zs.append(float(np.min(poly_z_vals)))
            max_zs.append(float(np.max(poly_z_vals)))
            mean_zs.append(float(np.mean(poly_z_vals)))
        else:
            min_zs.append(0.0)
            max_zs.append(0.0)
            mean_zs.append(0.0)

    # Injetar os novos campos estatísticos no GeoDataFrame projetado
    gdf_proj['lat'] = lat_list
    gdf_proj['lon'] = lon_list
    gdf_proj['x'] = x_list
    gdf_proj['y'] = y_list
    gdf_proj['z'] = z_list
    gdf_proj['min_z'] = min_zs
    gdf_proj['max_z'] = max_zs
    gdf_proj['mean_z'] = mean_zs

    # 5. Reprojetar para o CRS WGS84 (EPSG:4326) e forçar Single Part
    print("[INFO] Reprojetando dados 3D finais para WGS84 (EPSG:4326) e garantindo Single Part...")
    gdf_out = gdf_proj.to_crs("EPSG:4326")
    gdf_out = gdf_out.explode(index_parts=False)

    # 6. Salvar saídas (GeoJSON e Shapefile 3D)
    os.makedirs(args.output_dir, exist_ok=True)

    # Pegamos o nome base do arquivo de entrada para dar um nome mais descritivo
    base_name = os.path.splitext(os.path.basename(args.input_polygons))[0]

    geojson_out = os.path.join(args.output_dir, f"{base_name}_3d.geojson")
    geojson_ts_out = os.path.join(args.output_dir, f"{base_name}_3d_{timestamp}.geojson")

    shp_dir = os.path.join(args.output_dir, "shp")
    shp_out = os.path.join(shp_dir, f"{base_name}_3d.shp")
    shp_ts_out = os.path.join(shp_dir, f"{base_name}_3d_{timestamp}.shp")

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

    # Exportar Shapefile 3D
    try:
        os.makedirs(shp_dir, exist_ok=True)
        gdf_out.to_file(shp_out)
        gdf_out.to_file(shp_ts_out)
        print(f"[OK] Shapefile 3D (PolygonZ) salvo com sucesso: {shp_out}")
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
