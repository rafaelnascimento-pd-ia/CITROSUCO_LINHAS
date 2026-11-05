#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
herdar_altitude_lote_sem_buffer.py
--------------------------------------------------------------------------------
Script de pós-processamento geográfico em lote para ler múltiplas subpastas de talhões,
dissolver e explodir linhas contíguas, simplificá-las com Douglas-Peucker (tolerância de 0.02m),
reamostrá-las (espaçamento de 8m) e projetar a altitude (Z) de um MDT raster nelas,
além de limpar atributos e recalcular coordenadas 3D para o talhão original (sem aplicar buffer).
--------------------------------------------------------------------------------
"""

import os
import sys
import argparse
import numpy as np
import geopandas as gpd
import rasterio
from shapely.geometry import Polygon, LineString
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

def resample_line(line, spacing=8.0):
    """
    Reamostra a linha para garantir espaçamento uniforme dos vértices.
    """
    if line is None or line.is_empty:
        return line
    length = line.length
    if length <= 0:
        return line
        
    if length < 2.0:
        distances = [0.0, length]
    elif length <= spacing:
        distances = [0.0, length / 3.0, 2.0 * (length / 3.0), length]
    else:
        distances = list(np.arange(0, length, spacing))
        if distances[-1] < length:
            distances.append(length)
            
    points = [line.interpolate(d) for d in distances]
    return LineString(points)

def process_lines(lines_path, src_mdt, output_path, transformer, metric_crs, tolerance, spacing):
    """
    Processa o arquivo de linhas: dissolve e explode para unificar segmentos contíguos,
    simplifica usando Douglas-Peucker no CRS métrico, reamostra em intervalos regulares,
    e projeta em 3D amostrando Z no MDT.
    """
    gdf_lines = gpd.read_file(lines_path)
    if gdf_lines.empty:
        return
        
    # 1. Remover atributos originais
    geom_col = gdf_lines.geometry.name
    gdf_lines = gdf_lines[[geom_col]].copy()
    
    # 2. Reprojetar para o CRS métrico para realizar processamentos em metros
    gdf_metric = gdf_lines.to_crs(metric_crs)
    
    # 2.5 Dissolver e Explodir
    print("  - Dissolvendo e explodindo linhas para unificar segmentos contíguos...")
    union_geom = gdf_metric.geometry.unary_union
    gdf_unified = gpd.GeoDataFrame(geometry=[union_geom], crs=gdf_metric.crs)
    gdf_exploded = gdf_unified.explode(index_parts=False)
    gdf_metric = gdf_exploded.reset_index(drop=True)
    
    # 3. Simplificar e Reamostrar no CRS métrico
    print(f"  - Simplificando linhas (Douglas-Peucker, tol={tolerance}m) e reamostrando (spacing={spacing}m)...")
    new_geoms = []
    for geom in gdf_metric.geometry:
        if geom is None or geom.is_empty:
            new_geoms.append(geom)
            continue
            
        simp_geom = geom.simplify(tolerance, preserve_topology=True)
        
        if simp_geom.geom_type == 'LineString':
            resamp_geom = resample_line(simp_geom, spacing)
            new_geoms.append(resamp_geom)
        elif simp_geom.geom_type == 'MultiLineString':
            parts = []
            for part in simp_geom.geoms:
                parts.append(resample_line(part, spacing))
            from shapely.geometry import MultiLineString
            new_geoms.append(MultiLineString(parts))
        else:
            new_geoms.append(simp_geom)
            
    gdf_metric['geometry'] = new_geoms
    
    # Explodir
    gdf_metric = gdf_metric.explode(index_parts=False)
    
    # 4. Reprojetar para CRS do MDT
    gdf_proj = gdf_metric.to_crs(src_mdt.crs)
    
    # 5. Extrair coordenadas para amostragem
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
        
    # 6. Reconstruir geometrias 3D
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
    
    # 7. Calcular coordenadas
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
    
    # 8. Exportar reprojetado para WGS84 e Single Part
    gdf_out = gdf_proj.to_crs("EPSG:4326")
    gdf_out = gdf_out.explode(index_parts=False)
    
    gdf_out.to_file(output_path)

def process_talhao(talhao_path, src_mdt, output_path, transformer, metric_crs):
    """
    Processa o talhão: limpa atributos antigos, converte em 3D (Z do MDT) sem aplicar buffer,
    e recalcula as coordenadas/altitudes do centróide.
    """
    gdf_talhao = gpd.read_file(talhao_path)
    if gdf_talhao.empty:
        return
        
    # 1. Remover atributos originais
    geom_col = gdf_talhao.geometry.name
    gdf_talhao = gdf_talhao[[geom_col]].copy()
    
    # Reprojetar para o CRS métrico
    gdf_talhao_m = gdf_talhao.to_crs(metric_crs)
    
    # Explodir para Single Part (mantendo geometria original intacta)
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
        description="Processa em lote subpastas de talhões, aplicando simplificação nas linhas e apenas atualizando coordenadas/Z no talhão original (sem buffer).",
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
        "--output_dir", "-o", default=None,
        help="Diretório base de saída."
    )
    parser.add_argument(
        "--tolerance", "-t", type=float, default=0.02,
        help="Tolerância de simplificação prévia de linhas em metros (Douglas-Peucker)."
    )
    parser.add_argument(
        "--spacing", "-s", type=float, default=8.0,
        help="Espaçamento de reamostragem das linhas em metros."
    )

    args = parser.parse_args()

    # 1. Validar caminhos
    if not os.path.exists(args.input_dir):
        print(f"[ERRO] Diretório de entrada não encontrado: {args.input_dir}")
        sys.exit(1)
    if not os.path.exists(args.mdt_path):
        print(f"[ERRO] MDT não encontrado: {args.mdt_path}")
        sys.exit(1)

    if not args.output_dir:
        args.output_dir = os.path.join(args.input_dir, "convertidos_sem_buffer")

    print("=" * 75)
    print("   CITROSROWS AI - INJEÇÃO DE ALTITUDE SEM APLICAR BUFFER NO TALHÃO")
    print("=" * 75)
    print(f"Diretório de Entrada   : {args.input_dir}")
    print(f"Raster MDT             : {args.mdt_path}")
    print(f"Diretório de Saída     : {args.output_dir}")
    print(f"Tolerância Simplif.    : {args.tolerance}m")
    print(f"Espaçamento Reamost.   : {args.spacing}m")
    print("-" * 75)

    # 2. Carregar raster MDT
    try:
        src_mdt = rasterio.open(args.mdt_path)
    except Exception as e:
        print(f"[ERRO] Falha ao abrir relevo MDT: {e}")
        sys.exit(1)

    # Encontrar todas as subpastas
    subdirs = [d for d in os.listdir(args.input_dir) if os.path.isdir(os.path.join(args.input_dir, d))]
    
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

        sub_path_out = os.path.join(args.output_dir, subdir_clean)
        os.makedirs(sub_path_out, exist_ok=True)

        spatial_files = []
        valid_extensions = ('.shp', '.geojson', '.gpkg')
        for f in os.listdir(sub_path_in):
            f_lower = f.lower()
            if f_lower.endswith(valid_extensions):
                spatial_files.append(f)
                
        lines_file = None
        talhao_file = None
        
        # 1. Procurar o arquivo de linhas (contém "linhas" no nome)
        for f in spatial_files:
            if "linhas" in f.lower():
                lines_file = os.path.join(sub_path_in, f)
                break
                
        # 2. O outro arquivo espacial será considerado o talhão
        for f in spatial_files:
            f_full = os.path.join(sub_path_in, f)
            if lines_file and f_full == lines_file:
                continue
            talhao_file = f_full
            break

        print(f"\n>> [{idx+1}/{len(subdirs)}] Processando Subpasta: '{subdir}' -> '{subdir_clean}'")

        # Processar as Linhas
        if lines_file:
            print(f"  - Processando Linhas: {os.path.basename(lines_file)}")
            out_lines_name = f"LINHAS_{subdir_clean}.shp"
            out_lines_path = os.path.join(sub_path_out, out_lines_name)
            try:
                process_lines(lines_file, src_mdt, out_lines_path, transformer, metric_crs, args.tolerance, args.spacing)
                print(f"    [OK] Salvo: {out_lines_name}")
            except Exception as e:
                print(f"    [ERRO] Falha ao processar linhas: {e}")
        else:
            print("  - [INFO] Nenhum arquivo de linhas encontrado nesta subpasta.")

        # Processar os Talhões (Mantendo a geometria original intacta - Sem Buffer!)
        if talhao_file:
            print(f"  - Processando Talhão: {os.path.basename(talhao_file)}")
            out_talhao_name = f"BORDADURA_{subdir_clean}.shp"
            out_talhao_path = os.path.join(sub_path_out, out_talhao_name)
            try:
                process_talhao(talhao_file, src_mdt, out_talhao_path, transformer, metric_crs)
                print(f"    [OK] Salvo: {out_talhao_name}")
            except Exception as e:
                print(f"    [ERRO] Falha ao processar talhão: {e}")
        else:
            print("  - [INFO] Nenhum arquivo de talhão encontrado nesta subpasta.")

        processed_count += 1

    src_mdt.close()
    print("\n" + "=" * 75)
    print(f"   LOTE CONCLUÍDO! {processed_count} SUBPASTAS CONVERTIDAS COM SUCESSO.")
    print("=" * 75)

if __name__ == "__main__":
    main()
