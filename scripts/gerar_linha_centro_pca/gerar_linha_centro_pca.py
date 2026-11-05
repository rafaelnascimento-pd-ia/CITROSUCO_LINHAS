#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
gerar_linha_centro_pca.py (v4)
--------------------------------------------------------------------------------
Script para traccar linhas de centro de fileiras de plantio curvas.

Melhorias da v4:
    - IGNORA BURACOS: O recorte da linha agora usa apenas o contorno externo (exterior)
      do polígono, evitando que a linha seja fragmentada pelos buracos das árvores.
    - ORDENAÇÃO ROBUSTA: Algoritmo de vizinho mais próximo otimizado para manter a 
      sequência lógica da fileira.
    - EXTENSÃO E RECORTE: Garante que a linha cubra toda a extensão do polígono.
--------------------------------------------------------------------------------
"""

import os
import sys
import argparse
import numpy as np
import geopandas as gpd
from scipy.spatial import distance_matrix
from shapely.geometry import LineString, Polygon, MultiPolygon


def order_points_nearest_neighbor(pts_coords):
    """
    Ordena os pontos seguindo o caminho mais provável da fileira.
    """
    pts = np.array(pts_coords)
    n = len(pts)
    if n <= 2:
        return pts

    D = distance_matrix(pts, pts)
    
    # Encontrar as duas extremidades (os dois pontos mais distantes entre si)
    i, j = np.unravel_index(np.argmax(D), D.shape)
    start = i
    
    visited = [start]
    remaining = set(range(n))
    remaining.remove(start)
    current = start
    
    while remaining:
        # Encontrar o próximo ponto mais próximo
        next_idx = min(remaining, key=lambda idx: D[current, idx])
        visited.append(next_idx)
        remaining.remove(next_idx)
        current = next_idx
        
    return pts[visited]


def get_centerline_via_pca(poly, pts_coords):
    """
    Fallback: Gera uma linha reta via PCA se houver poucos pontos ou falha na ordenação.
    """
    from sklearn.decomposition import PCA
    pca = PCA(n_components=1)
    pca.fit(pts_coords)
    
    # Projeta os pontos no primeiro componente principal
    projected = pca.transform(pts_coords)
    p_min, p_max = projected.min(), projected.max()
    
    # Reconstrói os pontos extremos
    pts_extrema = pca.inverse_transform(np.array([[p_min], [p_max]]))
    
    # Estende e recorta
    line = LineString(pts_extrema)
    return line


def smooth_path(path, window=3):
    """
    Suavização leve por média móvel. 
    Janela reduzida para 3 para preservar melhor a curvatura original.
    """
    n = len(path)
    if n < 3:
        return path
    
    window = min(window, n)
    if window % 2 == 0: window -= 1
    if window < 3: return path

    smoothed = []
    half = window // 2
    for k in range(n):
        start = max(0, k - half)
        end = min(n, k + half + 1)
        # Manter os pontos originais nas extremidades para evitar encurtamento
        if k == 0 or k == n - 1:
            smoothed.append(path[k])
        else:
            smoothed.append(path[start:end].mean(axis=0))
    return np.array(smoothed)


def get_filled_polygon(geom):
    """
    Retorna o polígono sem buracos internos (apenas o anel externo).
    """
    if geom.geom_type == 'Polygon':
        return Polygon(geom.exterior)
    elif geom.geom_type == 'MultiPolygon':
        return MultiPolygon([Polygon(p.exterior) for p in geom.geoms])
    return geom


def centerline_from_points(poly, pts_coords, min_points=5, smooth_window=5):
    """
    Gera a linha de centro ignorando buracos internos do polígono.
    """
    n = len(pts_coords)
    if n < min_points:
        return None

    try:
        # 1. Ordenação
        ordered_pts = order_points_nearest_neighbor(pts_coords)

        # 2. Suavização
        win = smooth_window if n >= smooth_window else (3 if n >= 3 else 1)
        smoothed = smooth_path(ordered_pts, window=win)

        if len(smoothed) < 2:
            # Fallback para PCA se a ordenação falhar em gerar pontos suficientes
            long_line = get_centerline_via_pca(poly, pts_coords)
        else:
            # 3. Criar linha e estender
            bounds = poly.bounds
            diag = np.sqrt((bounds[2] - bounds[0])**2 + (bounds[3] - bounds[1])**2)
            ext = diag * 0.5

            p0, p1 = smoothed[0], smoothed[1]
            v_start = p0 - p1
            dist_start = np.linalg.norm(v_start)
            p_start_ext = p0 + (v_start / dist_start * ext) if dist_start > 0 else p0

            pn_1, pn = smoothed[-2], smoothed[-1]
            v_end = pn - pn_1
            dist_end = np.linalg.norm(v_end)
            p_end_ext = pn + (v_end / dist_end * ext) if dist_end > 0 else pn

            long_line = LineString([p_start_ext] + list(smoothed) + [p_end_ext])

        # 4. RECORTE CRÍTICO: Usar polígono preenchido (sem buracos)
        poly_filled = get_filled_polygon(poly)
        intersection = long_line.intersection(poly_filled)
        
        if intersection.is_empty:
            # Se a linha estendida não cruza, tentamos sem a extensão
            intersection = long_line.intersection(poly_filled)
            if intersection.is_empty: return None
            
        # Retornar a maior parte contínua
        if intersection.geom_type == 'LineString':
            return intersection
        elif intersection.geom_type in ['MultiLineString', 'GeometryCollection']:
            parts = [g for g in (intersection.geoms if hasattr(intersection, 'geoms') else []) if g.geom_type == 'LineString']
            if parts:
                return max(parts, key=lambda l: l.length)
            
        return None

    except Exception as e:
        # print(f"Erro no polígono: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Gerador de Linhas de Centro v4 (Ignore Holes)")
    parser.add_argument("--input_polygon", "-p", required=True)
    parser.add_argument("--input_points", "-pts", required=True)
    parser.add_argument("--output_shp", "-o", required=True)
    parser.add_argument("--min_points", type=int, default=5)
    parser.add_argument("--smooth_window", type=int, default=5)

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

    # Sjoin usando polígono preenchido para capturar pontos que podem estar nos buracos
    gdf_poly_filled = gdf_poly.copy()
    gdf_poly_filled['geometry'] = gdf_poly['geometry'].apply(get_filled_polygon)
    
    print("[INFO] Associando pontos...")
    joined = gpd.sjoin(gdf_pts, gdf_poly_filled[['geometry', '_poly_id']], how='inner', predicate='within')
    
    points_by_poly = {}
    if not joined.empty:
        for poly_id, group in joined.groupby('_poly_id'):
            points_by_poly[poly_id] = np.array([(g.x, g.y) for g in group.geometry])

    print("[INFO] Gerando linhas...")
    result_lines = []
    result_attrs = []
    
    for idx, row in gdf_poly.iterrows():
        poly_id = row['_poly_id']
        pts = points_by_poly.get(poly_id, [])
        
        line = centerline_from_points(row.geometry, pts, args.min_points, args.smooth_window)
        
        if line:
            result_lines.append(line)
            attr = {k: v for k, v in row.items() if k not in ('geometry', '_poly_id')}
            result_attrs.append(attr)

    if result_lines:
        gdf_out = gpd.GeoDataFrame(result_attrs, geometry=result_lines, crs=metric_crs)
        gdf_out.to_crs(original_crs).to_file(args.output_shp)
        print(f"[OK] Sucesso! {len(result_lines)} linhas salvas em {args.output_shp}")
    else:
        print("[ERRO] Nenhuma linha gerada.")

if __name__ == "__main__":
    main()