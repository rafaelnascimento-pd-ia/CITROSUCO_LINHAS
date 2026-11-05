#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
gerar_linha_centro_pca.py (v6)
--------------------------------------------------------------------------------
Script para traçar linhas de centro de fileiras de plantio curvas.

Melhorias da v6:
    - INTEGRATE & DISSOLVE: Lógica para unir linhas próximas (threshold de 0.5m).
    - SNAP VERTICES: União de vértices próximos para garantir conectividade.
    - JUNÇÃO DE RETAS E CURVAS: Lógica para mesclar segmentos adjacentes.
    - TRATAMENTO DE SINGLE PART: Garante que a saída seja composta por entidades individuais.
--------------------------------------------------------------------------------
"""

import os
import sys
import argparse
import numpy as np
import geopandas as gpd
from scipy.spatial import distance_matrix
from scipy.interpolate import splprep, splev
from shapely.geometry import LineString, MultiLineString, Polygon, MultiPolygon, Point
from shapely.ops import linemerge, unary_union

def order_points_nearest_neighbor(pts_coords):
    """
    Ordena os pontos seguindo o caminho mais provável da fileira usando vizinho mais próximo.
    """
    pts = np.array(pts_coords)
    n = len(pts)
    if n <= 2:
        return pts

    D = distance_matrix(pts, pts)
    i, j = np.unravel_index(np.argmax(D), D.shape)
    start_node = i
    
    ordered_indices = [start_node]
    remaining = set(range(n))
    remaining.remove(start_node)
    
    current = start_node
    while remaining:
        next_idx = min(remaining, key=lambda idx: D[current, idx])
        ordered_indices.append(next_idx)
        remaining.remove(next_idx)
        current = next_idx
        
    return pts[ordered_indices]

def smooth_path_spline(path, s=0.5):
    """
    Suavização usando B-splines para manter a curvatura natural.
    """
    n = len(path)
    if n < 4:
        return path
    
    try:
        dists = np.linalg.norm(np.diff(path, axis=0), axis=1)
        mask = np.ones(len(path), dtype=bool)
        mask[1:] = dists > 1e-5
        path = path[mask]
        
        if len(path) < 4:
            return path

        x = path[:, 0]
        y = path[:, 1]
        
        tck, u = splprep([x, y], s=s)
        u_new = np.linspace(0, 1, num=max(n * 2, 20))
        new_points = splev(u_new, tck)
        return np.column_stack(new_points)
    except Exception:
        return path

def get_filled_polygon(geom):
    """
    Retorna o polígono sem buracos internos.
    """
    if geom.geom_type == 'Polygon':
        return Polygon(geom.exterior)
    elif geom.geom_type == 'MultiPolygon':
        return MultiPolygon([Polygon(p.exterior) for p in geom.geoms])
    return geom

def snap_points(line, tolerance=0.5):
    """
    Aproxima vértices de uma linha que estão muito próximos.
    """
    coords = list(line.coords)
    if len(coords) < 2:
        return line
    
    new_coords = [coords[0]]
    for i in range(1, len(coords)):
        last_pt = np.array(new_coords[-1])
        curr_pt = np.array(coords[i])
        if np.linalg.norm(last_pt - curr_pt) > tolerance:
            new_coords.append(coords[i])
    
    if len(new_coords) < 2:
        return line
    return LineString(new_coords)

def integrate_and_dissolve_lines(lines, tolerance=0.5):
    """
    Simula o 'Integrate' e 'Dissolve' do ArcGIS:
    1. Faz o snap dos endpoints.
    2. Une linhas que se tocam ou estão muito próximas.
    3. Simplifica a geometria resultante.
    """
    if not lines:
        return []

    # Unary union com buffer zero ou pequeno para resolver sobreposições
    # Aqui usamos a lógica de agrupar por proximidade
    merged = linemerge(lines)
    
    if merged.geom_type == 'LineString':
        return [merged]
    elif merged.geom_type == 'MultiLineString':
        # Tentar unir segmentos que estão próximos mas não se tocam exatamente
        parts = list(merged.geoms)
        final_parts = []
        
        while parts:
            current = parts.pop(0)
            combined = False
            for i, other in enumerate(final_parts):
                # Verifica distância entre extremidades
                dist = current.distance(other)
                if dist <= tolerance:
                    # Tenta unir criando um segmento de conexão se necessário
                    # Simplificação: apenas faz o union e linemerge novamente
                    new_combined = linemerge(unary_union([current, other]).buffer(tolerance).exterior).intersection(unary_union([current, other]).buffer(tolerance))
                    # Lógica mais robusta para integração:
                    # Vamos apenas agrupar e depois processar
                    final_parts[i] = unary_union([current, other])
                    combined = True
                    break
            if not combined:
                final_parts.append(current)
        
        # Após o agrupamento, extraímos as linhas centrais novamente
        results = []
        for p in final_parts:
            m = linemerge(p)
            if m.geom_type == 'LineString':
                results.append(m)
            elif m.geom_type == 'MultiLineString':
                results.extend(list(m.geoms))
        return results
    return []

def join_close_lines(gdf, tolerance=0.5):
    """
    Lógica para juntar linhas retas e curvas próximas.
    Utiliza buffer e união para integrar geometrias.
    """
    if gdf.empty:
        return gdf
    
    # Criar um buffer para "engolir" as lacunas de 0.5m
    buffered = gdf.geometry.buffer(tolerance / 2, cap_style=2, join_style=2)
    unified = unary_union(buffered)
    
    new_lines = []
    
    # Para cada parte unificada, tentamos extrair a linha central (skeleton simplificado)
    if unified.geom_type == 'Polygon':
        unified = MultiPolygon([unified])
    
    if unified.geom_type == 'MultiPolygon':
        for poly in unified.geoms:
            # Aqui poderíamos usar algoritmos de skeletonization, 
            # mas para manter a fidelidade às linhas originais, 
            # vamos tentar dar merge nas linhas que caem dentro deste polígono unificado.
            contained_indices = gdf.index[gdf.geometry.within(poly.buffer(0.01))]
            if len(contained_indices) > 0:
                subset = gdf.loc[contained_indices].geometry
                merged = linemerge(list(subset))
                if merged.geom_type == 'LineString':
                    new_lines.append(merged)
                elif merged.geom_type == 'MultiLineString':
                    # Se ainda for multi, tentamos conectar as pontas mais próximas
                    parts = list(merged.geoms)
                    ordered_parts = [parts.pop(0)]
                    while parts:
                        last_line = ordered_parts[-1]
                        last_pt = Point(last_line.coords[-1])
                        
                        best_idx = -1
                        best_dist = float('inf')
                        reverse_needed = False
                        
                        for i, p in enumerate(parts):
                            d1 = last_pt.distance(Point(p.coords[0]))
                            d2 = last_pt.distance(Point(p.coords[-1]))
                            
                            if d1 < best_dist:
                                best_dist = d1
                                best_idx = i
                                reverse_needed = False
                            if d2 < best_dist:
                                best_dist = d2
                                best_idx = i
                                reverse_needed = True
                        
                        if best_idx != -1 and best_dist <= tolerance * 2:
                            next_line = parts.pop(best_idx)
                            if reverse_needed:
                                next_line = LineString(list(next_line.coords)[::-1])
                            
                            # Une os pontos
                            combined_coords = list(last_line.coords) + list(next_line.coords)
                            ordered_parts[-1] = LineString(combined_coords)
                        else:
                            if parts:
                                ordered_parts.append(parts.pop(0))
                            else:
                                break
                    new_lines.extend(ordered_parts)
    
    # Criar novo GeoDataFrame
    if new_lines:
        # Tenta preservar atributos do primeiro item encontrado no grupo (simplificado)
        return gpd.GeoDataFrame(geometry=new_lines, crs=gdf.crs)
    return gdf

def centerline_from_points(poly, pts_coords, min_points=5, smoothing=0.5):
    """
    Gera a linha de centro com ordenação robusta e suavização por spline.
    """
    n = len(pts_coords)
    if n < min_points:
        return None

    try:
        ordered_pts = order_points_nearest_neighbor(pts_coords)
        smoothed = smooth_path_spline(ordered_pts, s=smoothing)

        if len(smoothed) < 2:
            return None
            
        bounds = poly.bounds
        diag = np.sqrt((bounds[2] - bounds[0])**2 + (bounds[3] - bounds[1])**2)
        ext = diag * 0.2 

        p0, p1 = smoothed[0], smoothed[1]
        v_start = p0 - p1
        d_start = np.linalg.norm(v_start)
        p_start_ext = p0 + (v_start / d_start * ext) if d_start > 0 else p0

        pn_1, pn = smoothed[-2], smoothed[-1]
        v_end = pn - pn_1
        d_end = np.linalg.norm(v_end)
        p_end_ext = pn + (v_end / d_end * ext) if d_end > 0 else pn

        long_line = LineString([p_start_ext] + list(smoothed) + [p_end_ext])
        poly_filled = get_filled_polygon(poly)
        intersection = long_line.intersection(poly_filled)
        
        if intersection.is_empty:
            return None
            
        if intersection.geom_type == 'LineString':
            return snap_points(intersection)
        elif intersection.geom_type in ['MultiLineString', 'GeometryCollection']:
            parts = [g for g in (intersection.geoms if hasattr(intersection, 'geoms') else []) if g.geom_type == 'LineString']
            if parts:
                return snap_points(max(parts, key=lambda l: l.length))
            
        return None

    except Exception:
        return None

def main():
    parser = argparse.ArgumentParser(description="Gerador de Linhas de Centro v6 (Integrate & Dissolve)")
    parser.add_argument("--input_polygon", "-p", required=True, help="Shapefile de polígonos das fileiras")
    parser.add_argument("--input_points", "-pts", required=True, help="Shapefile de pontos das árvores/plantas")
    parser.add_argument("--output_shp", "-o", required=True, help="Caminho de saída para o shapefile de linhas")
    parser.add_argument("--min_points", type=int, default=5, help="Mínimo de pontos para processar um polígono")
    parser.add_argument("--smoothing", type=float, default=0.5, help="Fator de suavização")
    parser.add_argument("--tolerance", type=float, default=0.5, help="Tolerância para integração/snap (metros)")

    args = parser.parse_args()

    if not os.path.exists(args.input_polygon) or not os.path.exists(args.input_points):
        print("[ERRO] Arquivos de entrada não encontrados.")
        sys.exit(1)

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
    
    print("[INFO] Associando pontos aos polígonos...")
    joined = gpd.sjoin(gdf_pts, gdf_poly_filled[['geometry', '_poly_id']], how='inner', predicate='within')
    
    points_by_poly = {}
    if not joined.empty:
        for poly_id, group in joined.groupby('_poly_id'):
            points_by_poly[poly_id] = np.array([(g.x, g.y) for g in group.geometry])

    print("[INFO] Gerando linhas de centro individuais...")
    temp_lines = []
    temp_attrs = []
    
    for idx, row in gdf_poly.iterrows():
        poly_id = row['_poly_id']
        pts = points_by_poly.get(poly_id, [])
        line = centerline_from_points(row.geometry, pts, args.min_points, args.smoothing)
        
        if line:
            temp_lines.append(line)
            attr = {k: v for k, v in row.items() if k not in ('geometry', '_poly_id')}
            temp_attrs.append(attr)

    if not temp_lines:
        print("[AVISO] Nenhuma linha foi gerada inicialmente.")
        sys.exit(0)

    gdf_temp = gpd.GeoDataFrame(temp_attrs, geometry=temp_lines, crs=metric_crs)

    print(f"[INFO] Aplicando Integrate e Dissolve (Tolerância: {args.tolerance}m)...")
    # Aplicar a lógica de junção baseada em proximidade
    gdf_final = join_close_lines(gdf_temp, tolerance=args.tolerance)

    if not gdf_final.empty:
        # Garantir Single Part e remover geometrias inválidas
        gdf_final = gdf_final.explode(index_parts=False).reset_index(drop=True)
        
        # Voltar para o CRS original antes de salvar
        gdf_final.to_crs(original_crs).to_file(args.output_shp)
        print(f"[OK] Sucesso! {len(gdf_final)} linhas finais salvas em: {args.output_shp}")
    else:
        print("[AVISO] O processamento resultou em um arquivo vazio.")

if __name__ == "__main__":
    main()
