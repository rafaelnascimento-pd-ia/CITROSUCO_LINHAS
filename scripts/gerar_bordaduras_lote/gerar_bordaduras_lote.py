#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
gerar_bordaduras_lote.py
--------------------------------------------------------------------------------
Script simplificado para processamento em lote de polígonos de talhões (bordadura).
Aplica buffer de 7m, projeta a altitude (Z) de um MDT raster nos vértices do talhão,
mantendo partes simples (Single Part), coordenadas em WGS84 (EPSG:4326),
e salvando os resultados em subpastas organizadas.
--------------------------------------------------------------------------------
"""

import os
import sys
import argparse
import numpy as np
import geopandas as gpd
import rasterio
from shapely.geometry import Polygon
from pyproj import CRS, Transformer

def sanitize_filename(name):
    """
    Remove caracteres inválidos para nomes de arquivos e pastas no Windows.
    Garante que não haja hífens (-) no nome resultante.
    """
    if not name:
        return ""
    name = str(name).replace("-", "")
    name = name.replace(" ", "_")
    name = "".join(c for c in name if c.isalnum() or c == "_")
    while "__" in name:
        name = name.replace("__", "_")
    return name.strip("_")

def make_polygon_3d(poly, mdt_src):
    """
    Converte um polígono 2D para 3D (PolygonZ) amostrando Z no MDT para cada vértice.
    """
    if poly is None or poly.is_empty:
        return poly
        
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

    ext_coords = list(poly.exterior.coords)
    new_ext = process_ring(ext_coords)
    
    new_ints = []
    for interior in poly.interiors:
        int_coords = list(interior.coords)
        new_ints.append(process_ring(int_coords))
        
    return Polygon(new_ext, new_ints)

def process_talhao(talhao_path, src_mdt, output_path, transformer, metric_crs, buffer_distance):
    """
    Processa o talhão: gera buffer, converte em 3D, e calcula atributos.
    """
    gdf_talhao = gpd.read_file(talhao_path)
    if gdf_talhao.empty:
        return
        
    # 1. Remover atributos originais (manter apenas geometria)
    geom_col = gdf_talhao.geometry.name
    gdf_talhao = gdf_talhao[[geom_col]].copy()
    
    # Reprojetar para o CRS métrico
    gdf_talhao_m = gdf_talhao.to_crs(metric_crs)
    
    # 2. Aplicar buffer (padrão 7.0 metros)
    gdf_talhao_m['geometry'] = gdf_talhao_m.geometry.buffer(buffer_distance)
    
    # Explodir para Single Part
    gdf_talhao_m = gdf_talhao_m.explode(index_parts=False)
    
    # 3. Reprojetar para o CRS do MDT
    gdf_talhao_proj = gdf_talhao_m.to_crs(src_mdt.crs)
    
    # 4. Converter geometrias para 3D (PolygonZ) amostrando no MDT
    geoms_3d = []
    for geom in gdf_talhao_proj.geometry:
        if geom is None or geom.is_empty:
            geoms_3d.append(None)
            continue
        if geom.geom_type == 'Polygon':
            geom_3d = make_polygon_3d(geom, src_mdt)
            geoms_3d.append(geom_3d)
        else:
            geoms_3d.append(geom)
            
    gdf_talhao_proj['geometry'] = geoms_3d
    
    # Reprojetar de volta ao CRS métrico para centróide
    gdf_metric = gdf_talhao_proj.to_crs(metric_crs)
    
    x_list, y_list, z_list = [], [], []
    lat_list, lon_list = [], []
    min_zs, max_zs, mean_zs = [], [], []
    
    for k in range(len(gdf_metric)):
        geom_m = gdf_metric.iloc[k].geometry
        geom_proj = gdf_talhao_proj.iloc[k].geometry
        
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
        if geom_proj and geom_proj.geom_type == 'Polygon':
            for coord in geom_proj.exterior.coords:
                if len(coord) >= 3:
                    poly_z_vals.append(coord[2])
            for interior in geom_proj.interiors:
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
            
    # Adicionar atributos
    gdf_talhao_proj['x'] = x_list
    gdf_talhao_proj['y'] = y_list
    gdf_talhao_proj['z'] = z_list
    gdf_talhao_proj['lat'] = lat_list
    gdf_talhao_proj['lon'] = lon_list
    gdf_talhao_proj['min_z'] = min_zs
    gdf_talhao_proj['max_z'] = max_zs
    gdf_talhao_proj['mean_z'] = mean_zs
    
    # 5. Reprojetar para WGS84 e salvar
    gdf_out = gdf_talhao_proj.to_crs("EPSG:4326")
    gdf_out = gdf_out.explode(index_parts=False)
    
    gdf_out.to_file(output_path)

def main():
    parser = argparse.ArgumentParser(
        description="Processa em lote subpastas de talhões, aplicando buffer de 7m (bordadura) e altitude MDT.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--input_dir", "-i", required=True,
        help="Diretório contendo as subpastas dos talhões (pasta mãe)."
    )
    parser.add_argument(
        "--mdt_path", "-m", required=True,
        help="Caminho do arquivo raster do MDT (GeoTIFF)."
    )
    parser.add_argument(
        "--output_dir", "-o", default=None,
        help="Diretório base de saída. Padrão: subpasta 'bordaduras' dentro da pasta mãe."
    )
    parser.add_argument(
        "--buffer_distance", "-b", type=float, default=7.0,
        help="Distância do buffer/bordadura do talhão em metros."
    )

    args = parser.parse_args()

    # 1. Validar caminhos
    if not os.path.exists(args.input_dir):
        print(f"[ERRO] Diretório de entrada não encontrado: {args.input_dir}")
        sys.exit(1)
    if not os.path.exists(args.mdt_path):
        print(f"[ERRO] MDT não encontrado: {args.mdt_path}")
        sys.exit(1)

    # Definir diretório de saída padrão
    if not args.output_dir:
        args.output_dir = os.path.join(args.input_dir, "bordaduras")

    print("=" * 75)
    print("   CITROSROWS AI - CRIAÇÃO DE BORDADURAS EM LOTE (7M + ALTITUDE MDT)")
    print("=" * 75)
    print(f"Pasta Mãe              : {args.input_dir}")
    print(f"Raster MDT             : {args.mdt_path}")
    print(f"Diretório de Saída     : {args.output_dir}")
    print(f"Distância da Bordadura : {args.buffer_distance}m")
    print("-" * 75)

    # 2. Carregar raster MDT
    try:
        src_mdt = rasterio.open(args.mdt_path)
    except Exception as e:
        print(f"[ERRO] Falha ao abrir relevo MDT: {e}")
        sys.exit(1)

    # Encontrar todas as subpastas
    subdirs = [d for d in os.listdir(args.input_dir) if os.path.isdir(os.path.join(args.input_dir, d))]
    
    # Se o output_dir estiver dentro da input_dir, vamos ignorar a subpasta do output_dir para evitar loop infinito/processamento recursivo
    output_dir_name = os.path.basename(os.path.normpath(args.output_dir))
    subdirs = [d for d in subdirs if os.path.normpath(d) != os.path.normpath(output_dir_name)]

    if not subdirs:
        print(f"[WARN] Nenhuma subpasta encontrada no diretório de entrada: {args.input_dir}")
        src_mdt.close()
        sys.exit(0)

    print(f"[INFO] Encontradas {len(subdirs)} subpastas para processamento.")

    # 3. Determinar CRS métrico para os cálculos
    metric_crs = None
    for subdir in subdirs:
        sub_path = os.path.join(args.input_dir, subdir)
        found = False
        for f in os.listdir(sub_path):
            if f.lower().endswith(('.shp', '.geojson', '.gpkg')):
                try:
                    gdf_temp = gpd.read_file(os.path.join(sub_path, f))
                    if not gdf_temp.empty and gdf_temp.crs:
                        if gdf_temp.crs.is_geographic:
                            metric_crs = gdf_temp.estimate_utm_crs()
                        else:
                            metric_crs = gdf_temp.crs
                        found = True
                        break
                except Exception:
                    pass
        if found:
            break

    if not metric_crs:
        if src_mdt.crs.is_projected:
            metric_crs = src_mdt.crs
        else:
            metric_crs = "EPSG:3857"

    # Garantir que o CRS métrico seja 2D
    if isinstance(metric_crs, str):
        metric_crs = CRS.from_user_input(metric_crs)
    if hasattr(metric_crs, "to_2d"):
        metric_crs = metric_crs.to_2d()

    print(f"[OK] CRS métrico de referência determinado: {metric_crs}")

    # Criar transformador de coordenadas métricas -> WGS84
    transformer = Transformer.from_crs(metric_crs, "EPSG:4326", always_xy=True)

    # 4. Processamento
    processed_count = 0
    os.makedirs(args.output_dir, exist_ok=True)

    for idx, subdir in enumerate(subdirs):
        sub_path_in = os.path.join(args.input_dir, subdir)
        subdir_clean = sanitize_filename(subdir)
        if not subdir_clean:
            print(f"[WARN] Nome de pasta inválido: '{subdir}'. Pulando.")
            continue

        # Criar a subpasta correspondente no diretório de saída
        sub_path_out = os.path.join(args.output_dir, subdir_clean)
        os.makedirs(sub_path_out, exist_ok=True)

        # Encontrar arquivos de talões espaciais (.shp, .geojson, .gpkg)
        # Ignora arquivos que contêm "linhas" no nome
        spatial_files = []
        valid_extensions = ('.shp', '.geojson', '.gpkg')
        for f in os.listdir(sub_path_in):
            f_lower = f.lower()
            if f_lower.endswith(valid_extensions) and "linhas" not in f_lower:
                spatial_files.append(f)
                
        if not spatial_files:
            print(f"\n>> [{idx+1}/{len(subdirs)}] Subpasta: '{subdir}' -> [INFO] Nenhum polígono de talhão encontrado.")
            continue

        # Usar o primeiro arquivo de talhão encontrado
        talhao_file = os.path.join(sub_path_in, spatial_files[0])
        print(f"\n>> [{idx+1}/{len(subdirs)}] Processando Subpasta: '{subdir}' -> '{subdir_clean}'")
        print(f"  - Talhão de origem: {os.path.basename(talhao_file)}")

        # Nome do arquivo final: mantém o formato anterior 'BORDADURA_{subdir_clean}.shp'
        out_buf_name = f"BORDADURA_{subdir_clean}.shp"
        out_buf_path = os.path.join(sub_path_out, out_buf_name)

        try:
            process_talhao(talhao_file, src_mdt, out_buf_path, transformer, metric_crs, args.buffer_distance)
            print(f"    [OK] Salvo: {out_buf_name}")
            processed_count += 1
        except Exception as e:
            print(f"    [ERRO] Falha ao processar talhão/buffer: {e}")

    src_mdt.close()
    print("\n" + "=" * 75)
    print(f"   LOTE CONCLUÍDO! {processed_count} TALHÕES PROCESSADOS COM SUCESSO.")
    print("=" * 75)

if __name__ == "__main__":
    main()
