#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
calcular_entrelinhas_por_talhao.py
--------------------------------------------------------------------------------
Script de pós-processamento geográfico para gerar as linhas centrais das
entrelinhas de plantio individualmente por talhão, aplicando um buffer de 4 metros
e salvando os resultados segmentados em subpastas.
--------------------------------------------------------------------------------
"""

import os
import sys
import argparse
import datetime
import numpy as np
import geopandas as gpd
from shapely.geometry import LineString, MultiLineString
from shapely.ops import unary_union


def calculate_lines_dominant_angle(lines):
    """
    Calcula o ângulo dominante (mediana em radianos) das fileiras de plantio.
    """
    angles = []
    for geom in lines:
        if geom is None or geom.is_empty:
            continue
        coords = np.array(geom.coords)
        if len(coords) >= 2:
            v = coords[-1] - coords[0]
            length = np.linalg.norm(v)
            if length > 0:
                angle = np.arctan2(v[1], v[0])
                # Normaliza para o intervalo [-pi/2, pi/2]
                while angle < -np.pi/2:
                    angle += np.pi
                while angle > np.pi/2:
                    angle -= np.pi
                angles.append(angle)
    
    if not angles:
        return 0.0
    return np.median(angles)


def calculate_distance_stats(line1, line2, num_sample_points=30):
    """
    Calcula com precisão métrica a distância mínima, máxima e média entre duas linhas.
    """
    if line1.length < line2.length:
        shorter_line, longer_line = line1, line2
    else:
        shorter_line, longer_line = line2, line1
        
    distances = []
    for f in np.linspace(0, 1, num_sample_points):
        pt = shorter_line.interpolate(f, normalized=True)
        dist = pt.distance(longer_line)
        distances.append(dist)
        
    distances = np.array(distances)
    return {
        'mean': float(distances.mean()),
        'min': float(distances.min()),
        'max': float(distances.max())
    }


def generate_midline(line1, line2, step_meters=1.5, min_points=10):
    """
    Gera a linha média (entrelinha) entre duas linhas de plantio.
    """
    coords1 = list(line1.coords)
    coords2 = list(line2.coords)
    
    p1_start, p1_end = np.array(coords1[0]), np.array(coords1[-1])
    p2_start, p2_end = np.array(coords2[0]), np.array(coords2[-1])
    
    v1 = p1_end - p1_start
    v2 = p2_end - p2_start
    
    if np.dot(v1, v2) < 0:
        coords2 = coords2[::-1]
        line2_aligned = LineString(coords2)
    else:
        line2_aligned = line2
        
    shorter_length = min(line1.length, line2.length)
    num_points = int(np.ceil(shorter_length / step_meters))
    num_points = max(min_points, num_points)
    
    mid_coords = []
    for f in np.linspace(0, 1, num_points):
        pt1 = line1.interpolate(f, normalized=True)
        pt2 = line2_aligned.interpolate(f, normalized=True)
        
        mid_x = (pt1.x + pt2.x) / 2.0
        mid_y = (pt1.y + pt2.y) / 2.0
        mid_coords.append((mid_x, mid_y))
        
    return LineString(mid_coords)


def extend_line_to_boundary(line, block_geom):
    """
    Estende as extremidades de uma LineString ao longo de suas tangentes
    terminais até tocar a borda do polígono do talhão (bloco).
    """
    if block_geom is None or block_geom.is_empty:
        return line
        
    coords = list(line.coords)
    if len(coords) < 2:
        return line
        
    bounds = block_geom.bounds
    diag = np.sqrt((bounds[2] - bounds[0])**2 + (bounds[3] - bounds[1])**2)
    ext_length = diag * 1.5
    
    p0 = np.array(coords[0])
    p1 = np.array(coords[1])
    v_start = p0 - p1
    len_start = np.linalg.norm(v_start)
    if len_start > 0:
        u_start = v_start / len_start
        p_start_extended = p0 + ext_length * u_start
    else:
        p_start_extended = p0
        
    pn = np.array(coords[-1])
    pn1 = np.array(coords[-2])
    v_end = pn - pn1
    len_end = np.linalg.norm(v_end)
    if len_end > 0:
        u_end = v_end / len_end
        p_end_extended = pn + ext_length * u_end
    else:
        p_end_extended = pn
        
    extended_coords = [tuple(p_start_extended)] + coords + [tuple(p_end_extended)]
    long_line = LineString(extended_coords)
    
    intersection = long_line.intersection(block_geom)
    
    if intersection.is_empty:
        return line
        
    if intersection.geom_type == 'LineString':
        return intersection
    elif intersection.geom_type == 'MultiLineString':
        valid_lines = [geom for geom in intersection.geoms if not geom.is_empty]
        if valid_lines:
            return max(valid_lines, key=lambda l: l.length)
            
    return line


def main():
    parser = argparse.ArgumentParser(
        description="Gera entrelinhas de plantio individualmente por talhão, salvando os resultados em subpastas.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--input_lines", "-i", required=True,
        help="Caminho do arquivo de linhas de plantio (GeoJSON ou Shapefile)."
    )
    parser.add_argument(
        "--talhao_path", "-t", required=True,
        help="Caminho do arquivo de talhões (GeoJSON ou Shapefile)."
    )
    parser.add_argument(
        "--output_dir", "-o", default="data/outputs/entrelinhas_por_talhao",
        help="Diretório de saída para as subpastas dos talhões."
    )
    parser.add_argument(
        "--min_row_spacing", type=float, default=4.0,
        help="Espaçamento lateral mínimo em metros para considerar fileiras distintas."
    )
    parser.add_argument(
        "--max_row_spacing", type=float, default=8.5,
        help="Espaçamento lateral máximo em metros para considerar fileiras vizinhas."
    )
    parser.add_argument(
        "--step_meters", type=float, default=1.5,
        help="Resolução espacial em metros para a amostragem dinâmica."
    )
    parser.add_argument(
        "--min_points", type=int, default=10,
        help="Número mínimo de pontos por linha de amostragem."
    )
    parser.add_argument(
        "--reference_lines", "-r", default=None,
        help="Caminho opcional para arquivo de linhas guia de direção."
    )
    parser.add_argument(
        "--buffer_distance", "-b", type=float, default=6.5,
        help="Distância do buffer em metros para estender e recortar as entrelinhas."
    )
    parser.add_argument(
        "--timestamp", default=None,
        help="Sufixo de timestamp opcional para os nomes de arquivo (formato YYYYMMDD_HHMMSS)."
    )
    
    args = parser.parse_args()
    
    timestamp = args.timestamp
    if not timestamp:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("=" * 70)
    print("   CITROSROWS AI - CÁLCULO DE ENTRELINHAS SEGMENTADAS POR TALHÃO")
    print("=" * 70)
    print(f"Linhas de Entrada  : {args.input_lines}")
    print(f"Arquivo de Talhões : {args.talhao_path}")
    print(f"Diretório de Saída : {args.output_dir}")
    print(f"Limiares de Busca  : Mínimo = {args.min_row_spacing}m | Máximo = {args.max_row_spacing}m")
    print(f"Buffer do Talhão   : {args.buffer_distance}m")
    print("-" * 70)
    
    # 1. Carregar arquivos
    if not os.path.exists(args.input_lines):
        print(f"[ERRO] Arquivo de linhas não encontrado: {args.input_lines}")
        sys.exit(1)
    if not os.path.exists(args.talhao_path):
        print(f"[ERRO] Arquivo de talhões não encontrado: {args.talhao_path}")
        sys.exit(1)
        
    try:
        gdf_lines = gpd.read_file(args.input_lines)
        gdf_talhao = gpd.read_file(args.talhao_path)
    except Exception as e:
        print(f"[ERRO] Falha ao ler arquivos: {e}")
        sys.exit(1)
        
    if gdf_lines.empty:
        print("[ERRO] O arquivo de linhas de entrada está vazio.")
        sys.exit(0)
    if gdf_talhao.empty:
        print("[ERRO] O arquivo de talhões está vazio.")
        sys.exit(0)
        
    original_crs = gdf_lines.crs
    print(f"[OK] {len(gdf_lines)} linhas e {len(gdf_talhao)} talhões carregados.")
    
    # 2. Reprojeção dinâmica para sistema métrico (UTM) para cálculos espaciais precisos
    if original_crs is not None:
        if original_crs.is_geographic:
            try:
                utm_crs = gdf_lines.estimate_utm_crs()
                gdf_lines = gdf_lines.to_crs(utm_crs)
                gdf_talhao = gdf_talhao.to_crs(utm_crs)
                print(f"[INFO] Reprojetando para UTM local: {utm_crs.to_string()}")
            except Exception as e:
                gdf_lines = gdf_lines.to_crs("EPSG:3857")
                gdf_talhao = gdf_talhao.to_crs("EPSG:3857")
                print(f"[WARN] Falha ao estimar UTM ({e}). Reprojetando para EPSG:3857 (Web Mercator).")
        else:
            gdf_talhao = gdf_talhao.to_crs(gdf_lines.crs)
            print(f"[INFO] Dados já em formato métrico: {original_crs.to_string()}")
            
    # Explodir MultiLineStrings
    gdf_lines = gdf_lines.explode(index_parts=False)
    lines = []
    for geom in gdf_lines.geometry:
        if geom is not None and not geom.is_empty:
            if geom.geom_type == 'LineString':
                lines.append(geom)
            elif geom.geom_type == 'MultiLineString':
                for g in geom.geoms:
                    if not g.is_empty:
                        lines.append(g)
                        
    if not lines:
        print("[ERRO] Nenhuma geometria de linha válida restante após explosão.")
        sys.exit(1)
        
    # Carregar linhas de referência (opcional)
    gdf_ref = None
    if args.reference_lines:
        if os.path.exists(args.reference_lines):
            try:
                gdf_ref = gpd.read_file(args.reference_lines).to_crs(gdf_lines.crs)
                gdf_ref = gdf_ref.explode(index_parts=False)
                print(f"[OK] {len(gdf_ref)} linhas de referência carregadas.")
            except Exception as e:
                print(f"[WARN] Falha ao ler linhas de referência: {e}")
                
    # 3. Laço de processamento por talhão
    os.makedirs(args.output_dir, exist_ok=True)
    
    for idx, row in gdf_talhao.iterrows():
        geom_t = row.geometry
        if geom_t is None or geom_t.is_empty:
            continue
            
        # Determinar nome da subpasta do talhão
        talhao_id = None
        for col in ['TALHAO', 'talhao', 'ID', 'id', 'NOME', 'nome', 'CD_TALHAO']:
            if col in row and row[col] is not None:
                talhao_id = str(row[col])
                break
        if not talhao_id:
            talhao_id = f"talhao_{idx+1}"
        else:
            talhao_id = "".join(c for c in talhao_id if c.isalnum() or c in ('_', '-')).strip()
            if not talhao_id:
                talhao_id = f"talhao_{idx+1}"
                
        print(f"\n>> Processando Talhão: {talhao_id} ({idx+1}/{len(gdf_talhao)})")
        
        # ETAPA A: Filtrar linhas que pertencem a este talhão (limite original / sem buffer)
        block_lines = [line for line in lines if line.intersects(geom_t)]
        print(f"  - Fileiras pertencentes (sem buffer): {len(block_lines)}")
        
        if len(block_lines) < 2:
            print("  - [INFO] Fileiras insuficientes (< 2) para processar entrelinhas. Pulando.")
            continue
            
        # ETAPA B: Aplicar buffer no talhão para o cálculo e corte final
        block_geom_buffered = geom_t.buffer(args.buffer_distance)
        
        # Determinar direção dominante local
        local_angle = None
        if gdf_ref is not None:
            ref_lines_block = [g for g in gdf_ref.geometry if g is not None and not g.is_empty and g.intersects(geom_t)]
            if ref_lines_block:
                local_angle = calculate_lines_dominant_angle(ref_lines_block)
                
        if local_angle is None:
            local_angle = calculate_lines_dominant_angle(block_lines)
            
        local_u_perp = np.array([-np.sin(local_angle), np.cos(local_angle)])
        
        # Ordenação transversal
        local_lines_pos = []
        for line in block_lines:
            coords = np.array(line.coords)
            midpoint = coords.mean(axis=0)
            pos = np.dot(midpoint, local_u_perp)
            local_lines_pos.append((pos, line))
        local_lines_pos.sort(key=lambda x: x[0])
        sorted_local_lines = [item[1] for item in local_lines_pos]
        
        # Pareamento e geração de entrelinhas
        block_inter_rows = []
        all_attributes = {
            'dist_mean': [],
            'dist_min': [],
            'dist_max': [],
            'row_len_1': [],
            'row_len_2': []
        }
        
        for i in range(len(sorted_local_lines) - 1):
            l1 = sorted_local_lines[i]
            l2 = sorted_local_lines[i + 1]
            
            stats = calculate_distance_stats(l1, l2, num_sample_points=30)
            mean_d = stats['mean']
            
            if mean_d < args.min_row_spacing or mean_d > args.max_row_spacing:
                continue
                
            midline = generate_midline(l1, l2, step_meters=args.step_meters, min_points=args.min_points)
            if midline is not None and not midline.is_empty:
                # Estende e recorta no talhão com buffer de 4m
                extended_midline = extend_line_to_boundary(midline, block_geom_buffered)
                if extended_midline is not None and not extended_midline.is_empty:
                    block_inter_rows.append(extended_midline)
                    all_attributes['dist_mean'].append(stats['mean'])
                    all_attributes['dist_min'].append(stats['min'])
                    all_attributes['dist_max'].append(stats['max'])
                    all_attributes['row_len_1'].append(l1.length)
                    all_attributes['row_len_2'].append(l2.length)
                    
        print(f"  - Entrelinhas geradas: {len(block_inter_rows)}")
        
        # ETAPA C: Salvamento em subpasta
        sub_dir = os.path.join(args.output_dir, talhao_id)
        os.makedirs(sub_dir, exist_ok=True)
        
        # 1. Salvar talhão original (padrão e com timestamp)
        gdf_t_orig = gpd.GeoDataFrame(geometry=[geom_t], crs=gdf_lines.crs).to_crs(original_crs)
        gdf_t_orig.to_file(os.path.join(sub_dir, "talhao.geojson"), driver="GeoJSON")
        gdf_t_orig.to_file(os.path.join(sub_dir, f"talhao_{timestamp}.geojson"), driver="GeoJSON")
        
        # 2. Salvar talhão com buffer (padrão e com timestamp)
        buf_suffix = str(int(args.buffer_distance)) if args.buffer_distance.is_integer() else str(args.buffer_distance).replace('.', '_')
        gdf_t_buff = gpd.GeoDataFrame(geometry=[block_geom_buffered], crs=gdf_lines.crs).to_crs(original_crs)
        gdf_t_buff.to_file(os.path.join(sub_dir, f"talhao_buffer_{buf_suffix}m.geojson"), driver="GeoJSON")
        gdf_t_buff.to_file(os.path.join(sub_dir, f"talhao_buffer_{buf_suffix}m_{timestamp}.geojson"), driver="GeoJSON")
        
        # 3. Salvar linhas de plantio recortadas (padrão e com timestamp)
        gdf_block_lines = gpd.GeoDataFrame(geometry=block_lines, crs=gdf_lines.crs).to_crs(original_crs)
        gdf_block_lines.to_file(os.path.join(sub_dir, "linhas_plantio.geojson"), driver="GeoJSON")
        gdf_block_lines.to_file(os.path.join(sub_dir, f"linhas_plantio_{timestamp}.geojson"), driver="GeoJSON")
        
        # 4. Salvar entrelinhas com atributos de coordenadas (lat, lon, x, y para início, meio e fim)
        if block_inter_rows:
            gdf_inter = gpd.GeoDataFrame(all_attributes, geometry=block_inter_rows, crs=gdf_lines.crs)
            gdf_inter['length_m'] = gdf_inter.geometry.length
            
            # Reprojeta temporariamente para obter coordenadas corretas
            gdf_inter_wgs = gdf_inter.to_crs("EPSG:4326")
            gdf_inter_proj = gdf_inter.to_crs(gdf_lines.crs) # UTM métrico
            
            lat_list, lon_list, x_list, y_list = [], [], [], []
            start_lat_list, start_lon_list, start_x_list, start_y_list = [], [], [], []
            end_lat_list, end_lon_list, end_x_list, end_y_list = [], [], [], []
            
            for k in range(len(gdf_inter)):
                g_wgs = gdf_inter_wgs.iloc[k].geometry
                g_proj = gdf_inter_proj.iloc[k].geometry
                
                # Coordenadas do meio (centroide/interpolação 50%)
                mid_wgs = g_wgs.interpolate(0.5, normalized=True)
                mid_proj = g_proj.interpolate(0.5, normalized=True)
                lat_list.append(mid_wgs.y)
                lon_list.append(mid_wgs.x)
                x_list.append(mid_proj.x)
                y_list.append(mid_proj.y)
                
                # Coordenadas de início
                s_wgs = g_wgs.coords[0]
                s_proj = g_proj.coords[0]
                start_lon_list.append(s_wgs[0])
                start_lat_list.append(s_wgs[1])
                start_x_list.append(s_proj[0])
                start_y_list.append(s_proj[1])
                
                # Coordenadas de fim
                e_wgs = g_wgs.coords[-1]
                e_proj = g_proj.coords[-1]
                end_lon_list.append(e_wgs[0])
                end_lat_list.append(e_wgs[1])
                end_x_list.append(e_proj[0])
                end_y_list.append(e_proj[1])
                
            # Adiciona ao dataframe
            gdf_inter['lat'] = lat_list
            gdf_inter['lon'] = lon_list
            gdf_inter['x'] = x_list
            gdf_inter['y'] = y_list
            
            gdf_inter['start_lat'] = start_lat_list
            gdf_inter['start_lon'] = start_lon_list
            gdf_inter['start_x'] = start_x_list
            gdf_inter['start_y'] = start_y_list
            
            gdf_inter['end_lat'] = end_lat_list
            gdf_inter['end_lon'] = end_lon_list
            gdf_inter['end_x'] = end_x_list
            gdf_inter['end_y'] = end_y_list
            
            # Reprojeta de volta para o CRS original do GeoJSON para salvar
            gdf_inter_out = gdf_inter.to_crs(original_crs)
            gdf_inter_out.to_file(os.path.join(sub_dir, "entrelinhas.geojson"), driver="GeoJSON")
            gdf_inter_out.to_file(os.path.join(sub_dir, f"entrelinhas_{timestamp}.geojson"), driver="GeoJSON")
            print(f"  - Arquivos salvos na subpasta: {sub_dir}")
        else:
            print("  - [INFO] Nenhuma entrelinha válida gerada para este talhão.")

    print("\n" + "=" * 70)
    print("               PROCESSAMENTO DE LOTE CONCLUÍDO COM SUCESSO!")
    print("=" * 70)


if __name__ == "__main__":
    main()
