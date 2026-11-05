#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
herdar_altitude_lote_manter_campos.py
--------------------------------------------------------------------------------
Script de pós-processamento geográfico em lote para ler múltiplas subpastas de talhões,
projetar a altitude (Z) de um MDT raster nas linhas e nos polígonos.

Diferencial:
- Para o arquivo de linhas: PRESERVA todos os atributos originais (incluindo 'line_order')
  e apenas atualiza/injeta as colunas de coordenadas/estatísticas.
- Para o arquivo de polígonos: PRESERVA todos os atributos originais
  e apenas atualiza/injeta as colunas de coordenadas/estatísticas.
- Permite configurar distância de buffer para os polígonos (padrão 0.0 para manter intacto).
- Salva os arquivos de saída com os mesmos nomes dos arquivos de entrada originais.
--------------------------------------------------------------------------------
"""

import os
import sys
import argparse
import datetime
import numpy as np
import geopandas as gpd
import rasterio
from shapely.geometry import Polygon, LineString, MultiPolygon
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

def process_lines(lines_path, src_mdt, output_path, transformer, metric_crs):
    """
    Processa o arquivo de linhas: projeta em 3D e calcula os atributos espaciais/altitude.
    PRESERVA todos os atributos originais, especialmente 'line_order'.
    """
    gdf_lines = gpd.read_file(lines_path)
    if gdf_lines.empty:
        return
        
    # NOTA: Não removemos os atributos originais! Mantemos gdf_lines intacto.
    
    # Explodir MultiLineStrings mantendo atributos
    gdf_lines = gdf_lines.explode(index_parts=False)
    
    # Reprojetar para o CRS do MDT para fazer amostragem
    gdf_proj = gdf_lines.to_crs(src_mdt.crs)
    
    # Acumular coordenadas para amostragem em lote rápida
    all_coords = []
    line_lengths = []
    midpoint_coords = []
    for geom in gdf_proj.geometry:
        if geom is None or geom.is_empty:
            line_lengths.append(0)
            midpoint_coords.append(None)
            continue
        coords = [(c[0], c[1]) for c in geom.coords]
        all_coords.extend(coords)
        line_lengths.append(len(coords))
        
        # Ponto médio (50% do comprimento)
        mid_pt = geom.interpolate(0.5, normalized=True)
        midpoint_coords.append((mid_pt.x, mid_pt.y))
        
    if not all_coords:
        return
        
    # Amostragem em lote
    valid_midpoints = [pt for pt in midpoint_coords if pt is not None]
    sampled_vals = list(src_mdt.sample(all_coords + valid_midpoints))
    
    nodata = src_mdt.nodata
    z_values = []
    for val in sampled_vals[:len(all_coords)]:
        z = float(val[0])
        if np.isnan(z) or (nodata is not None and np.isclose(z, nodata)):
            z = 0.0
        z_values.append(z)
        
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
        
    # Reconstruir geometrias 3D
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
        
        coords_3d = [(x, y, z) for (x, y), z in zip(line_coords, line_zs)]
        new_line = LineString(coords_3d)
        new_geometries.append(new_line)
        
        start_z = line_zs[0]
        end_z = line_zs[-1]
        min_z = float(np.min(line_zs))
        max_z = float(np.max(line_zs))
        mean_z = float(np.mean(line_zs))
        
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
        
    gdf_proj['geometry'] = new_geometries
    gdf_proj['start_z'] = start_zs
    gdf_proj['end_z'] = end_zs
    gdf_proj['min_z'] = min_zs
    gdf_proj['max_z'] = max_zs
    gdf_proj['mean_z'] = mean_zs
    gdf_proj['slope_pct'] = slope_pcts
    gdf_proj['z'] = mid_z_vals
    
    # Calcular coordenadas geográficas e métricas
    gdf_metric = gdf_proj.to_crs(metric_crs)
    
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
            
        h_len = geom_metric.length
        lengths_list.append(h_len)
        
        s_coord = geom_metric.coords[0]
        sx, sy = s_coord[0], s_coord[1]
        slon, slat = transformer.transform(sx, sy)
        start_x_list.append(sx)
        start_y_list.append(sy)
        start_lon_list.append(slon)
        start_lat_list.append(slat)
        
        e_coord = geom_metric.coords[-1]
        ex, ey = e_coord[0], e_coord[1]
        elon, elat = transformer.transform(ex, ey)
        end_x_list.append(ex)
        end_y_list.append(ey)
        end_lon_list.append(elon)
        end_lat_list.append(elat)
        
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
    
    # Exportar reprojetado para WGS84 e Single Part
    gdf_out = gdf_proj.to_crs("EPSG:4326")
    gdf_out = gdf_out.explode(index_parts=False)
    
    # Salvar
    gdf_out.to_file(output_path)

def process_talhao(talhao_path, src_mdt, output_path, transformer, metric_crs, buffer_distance):
    """
    Processa o talhão: limpa atributos originais, opcionalmente gera buffer,
    converte em 3D e calcula os atributos.
    """
    gdf_talhao = gpd.read_file(talhao_path)
    if gdf_talhao.empty:
        return
        
    # 1. Preservar atributos originais do polígono
    
    # Reprojetar para o CRS métrico
    gdf_talhao_m = gdf_talhao.to_crs(metric_crs)
    
    # 2. Aplicar buffer se for maior que zero
    if buffer_distance > 0.0:
        print(f"  - Aplicando buffer de {buffer_distance}m no talhão...")
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
        description="Processa subpastas de talhões, injetando altitude e coordenadas. Preserva atributos originais do arquivo de linhas.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--input_dir", "-i", required=True,
        help="Diretório contendo as subpastas dos talhões."
    )
    parser.add_argument(
        "--mdt_path", "-m", required=True,
        help="Caminho do arquivo raster do MDT (GeoTIFF)."
    )
    parser.add_argument(
        "--output_dir", "-o", required=True,
        help="Diretório base de saída."
    )
    parser.add_argument(
        "--buffer_distance", "-b", type=float, default=0.0,
        help="Distância do buffer/bordadura para o polígono em metros (0.0 para manter intacto)."
    )

    args = parser.parse_args()

    # Sanitizar argumentos de caminho contra aspas extras, espaços e quebras de linha/CRLF
    if args.input_dir:
        args.input_dir = args.input_dir.strip(' \t\n\r"')
    if args.mdt_path:
        args.mdt_path = args.mdt_path.strip(' \t\n\r"')
    if args.output_dir:
        args.output_dir = args.output_dir.strip(' \t\n\r"')

    # 1. Validar caminhos
    if not os.path.exists(args.input_dir):
        print(f"[ERRO] Diretório de entrada não encontrado: {args.input_dir}")
        sys.exit(1)
    if not os.path.exists(args.mdt_path):
        print(f"[ERRO] MDT não encontrado: {args.mdt_path}")
        sys.exit(1)

    print("=" * 75)
    print("   CITROSROWS AI - INJEÇÃO DE ALTITUDE COM PRESERVAÇÃO DE CAMPOS DE LINHA")
    print("=" * 75)
    print(f"Diretório de Entrada   : {args.input_dir}")
    print(f"Raster MDT             : {args.mdt_path}")
    print(f"Diretório de Saída     : {args.output_dir}")
    print(f"Distância do Buffer    : {args.buffer_distance}m")
    print("-" * 75)

    # 2. Carregar raster MDT
    try:
        src_mdt = rasterio.open(args.mdt_path)
    except Exception as e:
        print(f"[ERRO] Falha ao abrir relevo MDT: {e}")
        sys.exit(1)

    # Encontrar subpastas
    subdirs = [d for d in os.listdir(args.input_dir) if os.path.isdir(os.path.join(args.input_dir, d))]
    
    # Ignorar pasta de saída se estiver dentro da de entrada
    output_dir_name = os.path.basename(os.path.normpath(args.output_dir))
    subdirs = [d for d in subdirs if os.path.normpath(d) != os.path.normpath(output_dir_name)]

    if not subdirs:
        print(f"[WARN] Nenhuma subpasta encontrada no diretório: {args.input_dir}")
        src_mdt.close()
        sys.exit(0)

    print(f"[INFO] Encontradas {len(subdirs)} subpastas para processamento.")

    # 3. Determinar CRS métrico para cálculos
    metric_crs = None
    for subdir in subdirs:
        sub_path = os.path.join(args.input_dir, subdir)
        found = False
        for f in os.listdir(sub_path):
            if f.lower().endswith(".shp"):
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

    # Garantir CRS métrico 2D
    if isinstance(metric_crs, str):
        metric_crs = CRS.from_user_input(metric_crs)
    if hasattr(metric_crs, "to_2d"):
        metric_crs = metric_crs.to_2d()

    print(f"[OK] CRS métrico de referência: {metric_crs}")

    # Criar transformador de coordenadas métricas -> WGS84
    transformer = Transformer.from_crs(metric_crs, "EPSG:4326", always_xy=True)

    # 4. Processamento das pastas
    processed_count = 0
    os.makedirs(args.output_dir, exist_ok=True)

    for idx, subdir in enumerate(subdirs):
        sub_path_in = os.path.join(args.input_dir, subdir)
        subdir_clean = sanitize_filename(subdir)
        if not subdir_clean:
            continue

        sub_path_out = os.path.join(args.output_dir, subdir_clean)
        os.makedirs(sub_path_out, exist_ok=True)

        spatial_files = []
        valid_extensions = ('.shp', '.geojson', '.gpkg')
        for f in os.listdir(sub_path_in):
            if f.lower().endswith(valid_extensions):
                spatial_files.append(f)

        lines_file = None
        talhao_file = None

        # Identificar arquivos
        for f in spatial_files:
            if "linhas" in f.lower() or "line" in f.lower():
                lines_file = os.path.join(sub_path_in, f)
                break

        for f in spatial_files:
            f_full = os.path.join(sub_path_in, f)
            if lines_file and f_full == lines_file:
                continue
            talhao_file = f_full
            break

        print(f"\n>> [{idx+1}/{len(subdirs)}] Processando Subpasta: '{subdir}' -> '{subdir_clean}'")

        # Processar Linhas
        if lines_file:
            orig_name = os.path.basename(lines_file)
            out_lines_path = os.path.join(sub_path_out, orig_name)
            print(f"  - Processando Linhas (Preservando campos): {orig_name}")
            try:
                process_lines(lines_file, src_mdt, out_lines_path, transformer, metric_crs)
                print(f"    [OK] Salvo: {orig_name}")
            except Exception as e:
                print(f"    [ERRO] Falha nas linhas: {e}")
        else:
            print("  - [INFO] Sem arquivo de linhas.")

        # Processar Talhão/Polígono
        if talhao_file:
            orig_name = os.path.basename(talhao_file)
            out_talhao_path = os.path.join(sub_path_out, orig_name)
            print(f"  - Processando Polígono: {orig_name}")
            try:
                process_talhao(talhao_file, src_mdt, out_talhao_path, transformer, metric_crs, args.buffer_distance)
                print(f"    [OK] Salvo: {orig_name}")
            except Exception as e:
                print(f"    [ERRO] Falha no polígono: {e}")
        else:
            print("  - [INFO] Sem arquivo de polígono.")

        processed_count += 1

    src_mdt.close()
    print("\n" + "=" * 75)
    print(f"   LOTE CONCLUÍDO! {processed_count} SUBPASTAS PROCESSADAS COM SUCESSO.")
    print("=" * 75)

if __name__ == "__main__":
    main()
