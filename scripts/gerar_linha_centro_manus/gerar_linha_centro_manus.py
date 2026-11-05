#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
gerar_linha_centro_pca.py (v5)
--------------------------------------------------------------------------------
Script para traccar linhas de centro de fileiras de plantio curvas.

Melhorias da v5:
    - FUSÃO DE LINHAS (Merging): Une linhas que pertencem à mesma fileira mas foram
      fragmentadas por falhas de plantio ou divisão de polígonos.
    - CRITÉRIOS DE UNIÃO: Baseado em distância entre pontas e alinhamento angular.
    - PREENCHIMENTO DE BURACOS: Mantém a lógica da v4 de ignorar buracos internos.
--------------------------------------------------------------------------------
"""

import os
import sys
import argparse
import numpy as np
import geopandas as gpd
from scipy.spatial import distance_matrix
from shapely.geometry import LineString, Polygon, MultiPolygon, Point
from shapely.ops import linemerge, unary_union


def order_points_nearest_neighbor(pts_coords):
    pts = np.array(pts_coords)
    n = len(pts)
    if n <= 2: return pts
    D = distance_matrix(pts, pts)
    i, j = np.unravel_index(np.argmax(D), D.shape)
    start = i
    visited = [start]
    remaining = set(range(n))
    remaining.remove(start)
    current = start
    while remaining:
        next_idx = min(remaining, key=lambda idx: D[current, idx])
        visited.append(next_idx)
        remaining.remove(next_idx)
        current = next_idx
    return pts[visited]


def smooth_path(path, window=3):
    n = len(path)
    if n < 3: return path
    window = min(window, n)
    if window % 2 == 0: window -= 1
    if window < 3: return path
    smoothed = []
    half = window // 2
    for k in range(n):
        start = max(0, k - half)
        end = min(n, k + half + 1)
        if k == 0 or k == n - 1:
            smoothed.append(path[k])
        else:
            smoothed.append(path[start:end].mean(axis=0))
    return np.array(smoothed)


def get_filled_polygon(geom):
    if geom.geom_type == 'Polygon':
        return Polygon(geom.exterior)
    elif geom.geom_type == 'MultiPolygon':
        return MultiPolygon([Polygon(p.exterior) for p in geom.geoms])
    return geom


def centerline_from_points(poly, pts_coords, min_points=5, smooth_window=3):
    n = len(pts_coords)
    if n < min_points: return None
    try:
        ordered_pts = order_points_nearest_neighbor(pts_coords)
        smoothed = smooth_path(ordered_pts, window=smooth_window)
        if len(smoothed) < 2: return None

        # Criar linha baseada nos pontos suavizados
        line = LineString(smoothed)
        
        # Extensão para garantir que toque as bordas
        bounds = poly.bounds
        diag = np.sqrt((bounds[2] - bounds[0])**2 + (bounds[3] - bounds[1])**2)
        ext = diag * 0.3 # Aumentado para facilitar a união de fileiras fragmentadas

        # Extensão Preditiva (usa a tendência dos últimos pontos para projetar a curva)
        def predict_extension(pts, direction='forward', length=5.0):
            if len(pts) < 3:
                # Fallback para linear se poucos pontos
                v = pts[-1] - pts[0] if direction == 'forward' else pts[0] - pts[-1]
                norm = np.linalg.norm(v)
                return pts[-1] + (v/norm * length) if norm > 0 else pts[-1]
            
            # Usar os últimos 3 pontos para estimar direção
            subset = pts[-3:] if direction == 'forward' else pts[:3][::-1]
            v1 = subset[1] - subset[0]
            v2 = subset[2] - subset[1]
            # Vetor médio de direção
            v_dir = (v1 + v2) / 2
            norm = np.linalg.norm(v_dir)
            return subset[2] + (v_dir/norm * length) if norm > 0 else subset[2]

        p_start_ext = predict_extension(smoothed, direction='backward', length=ext)
        p_end_ext = predict_extension(smoothed, direction='forward', length=ext)

        long_line = LineString([p_start_ext] + list(smoothed) + [p_end_ext])
        poly_filled = get_filled_polygon(poly)
        intersection = long_line.intersection(poly_filled)
        
        if intersection.is_empty: return None
        if intersection.geom_type == 'LineString': return intersection
        elif hasattr(intersection, 'geoms'):
            parts = [g for g in intersection.geoms if g.geom_type == 'LineString']
            if parts: return max(parts, key=lambda l: l.length)
        return None
    except Exception: return None


def get_vector(line, end='end', n_points=3):
    """Calcula o vetor de direção na extremidade da linha."""
    coords = np.array(line.coords)
    if len(coords) < 2: return None
    if end == 'end':
        v = coords[-1] - coords[-max(2, min(n_points, len(coords)))]
    else:
        v = coords[0] - coords[min(n_points, len(coords))-1]
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else None

def merge_lines(lines, max_dist=15.0, max_angle_diff=45):
    """
    Une linhas próximas que apontam uma para a outra (mesma fileira) usando KDTree.
    """
    if not lines:
        return []
        
    # Extrair extremidades das linhas
    endpoints = []
    for idx, line in enumerate(lines):
        coords = list(line.coords)
        if len(coords) < 2:
            continue
        endpoints.append((coords[0], idx, False))  # False = início
        endpoints.append((coords[-1], idx, True))   # True = fim
        
    if not endpoints:
        return lines
        
    coords_arr = np.array([ep[0] for ep in endpoints])
    
    from scipy.spatial import KDTree
    tree = KDTree(coords_arr)
    
    # Buscar pares dentro de max_dist
    pairs = tree.query_pairs(max_dist)
    
    candidates = []
    for idx1, idx2 in pairs:
        pt1, line_idx1, is_end1 = endpoints[idx1]
        pt2, line_idx2, is_end2 = endpoints[idx2]
        
        if line_idx1 == line_idx2:
            continue
            
        line1 = lines[line_idx1]
        line2 = lines[line_idx2]
        
        c1 = list(line1.coords)
        c2 = list(line2.coords)
        
        # Orientar temporariamente as linhas no sentido da conexão
        if is_end1:
            c1_temp = c1
        else:
            c1_temp = c1[::-1]
            
        if not is_end2:
            c2_temp = c2
        else:
            c2_temp = c2[::-1]
            
        v1 = np.array(c1_temp[-1]) - np.array(c1_temp[-max(2, len(c1_temp)//4)])
        v2 = np.array(c2_temp[min(max(2, len(c2_temp)//4), len(c2_temp)-1)]) - np.array(c2_temp[0])
        v_gap = np.array(c2_temp[0]) - np.array(c1_temp[-1])
        
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        ng = np.linalg.norm(v_gap)
        
        if n1 > 0 and n2 > 0:
            v1_n = v1 / n1
            v2_n = v2 / n2
            cos_sim = np.clip(np.dot(v1_n, v2_n), -1, 1)
            angle = np.degrees(np.arccos(cos_sim))
            
            # Validar se as linhas apontam uma em direção à outra através do gap
            if ng > 1.0:
                v_gap_n = v_gap / ng
                cos_g1 = np.dot(v1_n, v_gap_n)
                cos_g2 = np.dot(v_gap_n, v2_n)
                if cos_g1 < 0.7 or cos_g2 < 0.7:
                    continue
            
            if angle < max_angle_diff:
                candidates.append((ng, line_idx1, is_end1, line_idx2, is_end2))
                
    # Ordenar conexões candidatas por distância (menor primeiro)
    candidates.sort(key=lambda x: x[0])
    
    connections = {}
    matched = set()
    
    for dist, l1, e1, l2, e2 in candidates:
        ep1 = (l1, e1)
        ep2 = (l2, e2)
        if ep1 in matched or ep2 in matched:
            continue
        matched.add(ep1)
        matched.add(ep2)
        connections[ep1] = ep2
        connections[ep2] = ep1

    # Construir caminhos contínuos a partir das conexões greedy
    visited_lines = set()
    merged_lines = []
    
    # Começar com linhas que tenham extremidades iniciais desemparelhadas (inícios de caminhos)
    unmatched_starts = []
    for idx in range(len(lines)):
        if (idx, False) not in connections:
            unmatched_starts.append(idx)
            
    for idx in unmatched_starts:
        if idx in visited_lines:
            continue
            
        current_line_idx = idx
        visited_lines.add(current_line_idx)
        current_coords = list(lines[current_line_idx].coords)
        
        curr_end = (current_line_idx, True)
        while curr_end in connections:
            next_ep = connections[curr_end]
            next_line_idx, next_is_end = next_ep
            
            if next_line_idx in visited_lines:
                break
                
            visited_lines.add(next_line_idx)
            next_coords = list(lines[next_line_idx].coords)
            
            # Adiciona os pontos respeitando a orientação da junção
            if not next_is_end:
                current_coords.extend(next_coords)
                curr_end = (next_line_idx, True)
            else:
                current_coords.extend(next_coords[::-1])
                curr_end = (next_line_idx, False)
                
        merged_lines.append(LineString(current_coords))

    # Tratar quaisquer ciclos (loops fechados) restantes que não possuam inícios livres
    for idx in range(len(lines)):
        if idx not in visited_lines:
            current_line_idx = idx
            visited_lines.add(current_line_idx)
            current_coords = list(lines[current_line_idx].coords)
            
            curr_end = (current_line_idx, True)
            while curr_end in connections:
                next_ep = connections[curr_end]
                next_line_idx, next_is_end = next_ep
                if next_line_idx in visited_lines:
                    break
                visited_lines.add(next_line_idx)
                next_coords = list(lines[next_line_idx].coords)
                if not next_is_end:
                    current_coords.extend(next_coords)
                    curr_end = (next_line_idx, True)
                else:
                    current_coords.extend(next_coords[::-1])
                    curr_end = (next_line_idx, False)
                    
            merged_lines.append(LineString(current_coords))
            
    # Suavização final leve nos caminhos unidos
    final_lines = []
    for line in merged_lines:
        coords = np.array(line.coords)
        if len(coords) > 5:
            smoothed = smooth_path(coords, window=5)
            final_lines.append(LineString(smoothed))
        else:
            final_lines.append(line)
            
    return final_lines


def main():
    parser = argparse.ArgumentParser(description="Gerador de Linhas de Centro v5 (Merging)")
    parser.add_argument("--input_polygon", "-p", required=True)
    parser.add_argument("--input_points", "-pts", required=True)
    parser.add_argument("--output_shp", "-o", required=True)
    parser.add_argument("--min_points", type=int, default=5)
    parser.add_argument("--max_merge_dist", type=float, default=10.0, help="Distancia maxima para unir fileiras fragmentadas")

    args = parser.parse_args()

    print("[INFO] Carregando dados...")
    gdf_poly = gpd.read_file(args.input_polygon)
    gdf_pts = gpd.read_file(args.input_points)

    original_crs = gdf_poly.crs
    if original_crs and original_crs.is_geographic:
        metric_crs = gdf_poly.estimate_utm_crs()
        gdf_poly = gdf_poly.to_crs(metric_crs)
    else:
        metric_crs = original_crs

    if gdf_pts.crs != metric_crs:
        gdf_pts = gdf_pts.to_crs(metric_crs)

    gdf_poly = gdf_poly.explode(index_parts=False).reset_index(drop=True)
    gdf_poly['_poly_id'] = range(len(gdf_poly))

    gdf_poly_filled = gdf_poly.copy()
    gdf_poly_filled['geometry'] = gdf_poly['geometry'].apply(get_filled_polygon)
    
    joined = gpd.sjoin(gdf_pts, gdf_poly_filled[['geometry', '_poly_id']], how='inner', predicate='within')
    points_by_poly = {}
    if not joined.empty:
        for poly_id, group in joined.groupby('_poly_id'):
            points_by_poly[poly_id] = np.array([(g.x, g.y) for g in group.geometry])

    print("[INFO] Gerando linhas base...")
    base_lines = []
    for idx, row in gdf_poly.iterrows():
        pts = points_by_poly.get(row['_poly_id'], [])
        line = centerline_from_points(row.geometry, pts, args.min_points)
        if line: base_lines.append(line)

    print(f"[INFO] {len(base_lines)} linhas base geradas. Iniciando fusão (merging)...")
    merged_lines = merge_lines(base_lines, max_dist=args.max_merge_dist)
    print(f"[INFO] Fusão concluída: {len(base_lines)} -> {len(merged_lines)} linhas.")

    if merged_lines:
        gdf_out = gpd.GeoDataFrame(geometry=merged_lines, crs=metric_crs)
        gdf_out.to_crs(original_crs).to_file(args.output_shp)
        print(f"[OK] Sucesso! Resultado salvo em {args.output_shp}")
    else:
        print("[ERRO] Nenhuma linha gerada.")

if __name__ == "__main__":
    main()