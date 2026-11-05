#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
gerar_linha_centro_guiada_contagem_mst.py
--------------------------------------------------------------------------------
Script utilitário para traçar linhas centrais de plantio curvas e contínuas a partir 
de polígonos locais e pontos guia (contagem de plantas).

Utiliza o algoritmo de Árvore Geradora Mínima (MST) para ordenar sequencialmente
os pontos de cada fileira curva, garantindo que as linhas centrais acompanhem 
perfeitamente as curvas das árvores (mesmo em curvas "U" ou "S").

Otimizado com R-Tree (STRtree) e Union-Find para processamento de alta performance.
--------------------------------------------------------------------------------
"""

import os
import sys
import argparse
import numpy as np
import geopandas as gpd
from shapely.geometry import LineString, Polygon, MultiPolygon
from shapely.strtree import STRtree

def bfs_furthest_node(adj, start):
    """
    Executa busca em largura (BFS) para encontrar o nó mais distante em um grafo/árvore.
    Retorna o nó mais distante, o dicionário de distâncias e o dicionário de pais.
    """
    n = len(adj)
    dist = {start: 0}
    parent = {start: None}
    queue = [start]
    
    while queue:
        curr = queue.pop(0)
        for neighbor in adj[curr]:
            if neighbor not in dist:
                dist[neighbor] = dist[curr] + 1
                parent[neighbor] = curr
                queue.append(neighbor)
                
    furthest_node = max(dist, key=dist.get)
    return furthest_node, dist, parent

def extract_centerline_guided(poly, points_inside, deg=2, min_points=2):
    """
    Extrai a linha central de um polígono ordenando os pontos guia internos
    via Árvore Geradora Mínima (MST), estendendo as extremidades tangencialmente
    e recortando no polígono original.
    """
    if poly is None or poly.is_empty:
        return None
        
    # Se houver pontos guia suficientes
    if points_inside is not None and len(points_inside) >= min_points:
        try:
            # Extrai as coordenadas x, y dos pontos de forma robusta
            pts_coords = np.array([(geom.centroid.x, geom.centroid.y) for geom in points_inside.geometry if geom is not None])
            n = len(pts_coords)
            
            if n >= 2:
                # 1. Calcula a matriz de distâncias euclidianas entre todos os pontos
                dists = np.linalg.norm(pts_coords[:, None, :] - pts_coords[None, :, :], axis=-1)
                
                # 2. Constrói a Árvore Geradora Mínima (MST) usando o algoritmo de Prim
                visited = [False] * n
                min_dist = [float('inf')] * n
                mst_parent = [-1] * n
                min_dist[0] = 0.0
                
                for _ in range(n):
                    u = -1
                    for i in range(n):
                        if not visited[i] and (u == -1 or min_dist[i] < min_dist[u]):
                            u = i
                    visited[u] = True
                    
                    for v in range(n):
                        if not visited[v] and dists[u, v] < min_dist[v]:
                            min_dist[v] = dists[u, v]
                            mst_parent[v] = u
                            
                # Lista de adjacência da MST
                adj = [[] for _ in range(n)]
                for v in range(1, n):
                    u = mst_parent[v]
                    adj[u].append(v)
                    adj[v].append(u)
                    
                # 3. Encontra o diâmetro da MST (o caminho mais longo na árvore)
                # BFS 1: Encontra o nó A mais distante a partir do nó 0
                node_A, _, _ = bfs_furthest_node(adj, 0)
                # BFS 2: Encontra o nó B mais distante a partir de A
                node_B, _, parent_bfs = bfs_furthest_node(adj, node_A)
                
                # Reconstroi o caminho ordenado de A a B
                curr = node_B
                path_indices = []
                while curr is not None:
                    path_indices.append(curr)
                    curr = parent_bfs[curr]
                
                # Coordenadas do caminho ordenado (sequencial ao longo da curva)
                path = pts_coords[path_indices]
                n_path = len(path)
                
                # 4. Suavização leve do caminho para remover pequenos desvios locais
                if n_path >= 5:
                    window = 5
                elif n_path >= 3:
                    window = 3
                else:
                    window = 1
                    
                smoothed_path = []
                for k in range(n_path):
                    start_w = max(0, k - window // 2)
                    end_w = min(n_path, k + window // 2 + 1)
                    mean_pt = path[start_w:end_w].mean(axis=0)
                    smoothed_path.append(mean_pt)
                smoothed_path = np.array(smoothed_path)
                
                # 5. Prolongamento tangencial nas duas extremidades
                bounds = poly.bounds
                diag = np.sqrt((bounds[2] - bounds[0])**2 + (bounds[3] - bounds[1])**2)
                ext_len = diag * 2.0
                
                p_start = smoothed_path[0]
                p_second = smoothed_path[1] if n_path > 1 else p_start
                p_end = smoothed_path[-1]
                p_penultimate = smoothed_path[-2] if n_path > 1 else p_end
                
                # Extensão no início (direção de p_second para p_start)
                v_start = p_start - p_second
                len_start = np.linalg.norm(v_start)
                if len_start > 0:
                    u_start = v_start / len_start
                    p_start_ext = p_start + ext_len * u_start
                else:
                    p_start_ext = p_start
                    
                # Extensão no fim (direção de p_penultimate para p_end)
                v_end = p_end - p_penultimate
                len_end = np.linalg.norm(v_end)
                if len_end > 0:
                    u_end = v_end / len_end
                    p_end_ext = p_end + ext_len * u_end
                else:
                    p_end_ext = p_end
                    
                extended_coords = [tuple(p_start_ext)] + [tuple(pt) for pt in smoothed_path] + [tuple(p_end_ext)]
                line = LineString(extended_coords)
                
                # 6. Recorta a linha na borda do polígono original
                intersection = line.intersection(poly)
                if intersection.is_empty:
                    return line
                if intersection.geom_type == 'LineString':
                    return intersection
                elif intersection.geom_type == 'MultiLineString':
                    valid_lines = [geom for geom in intersection.geoms if not geom.is_empty]
                    if valid_lines:
                        return max(valid_lines, key=lambda l: l.length)
                return line
                
        except Exception as e:
            print(f"[WARN] Falha ao rastrear linha curva via MST: {e}. Usando fallback do polígono.")
            
    # Lógica de fallback baseada nas coordenadas externas do polígono
    coords = np.array(poly.exterior.coords)
    if len(coords) < 3:
        return None
        
    mean = coords.mean(axis=0)
    coords_centered = coords - mean
    
    try:
        cov = np.cov(coords_centered.T)
        if cov.ndim < 2:
            return None
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        idx_max = np.argmax(eigenvalues)
        ex = eigenvectors[:, idx_max]
        ey = np.array([-ex[1], ex[0]])
        R = np.vstack([ex, ey])
        
        coords_rot = coords_centered @ R.T
        X = coords_rot[:, 0]
        Y = coords_rot[:, 1]
        x_min, x_max = X.min(), X.max()
        row_length = x_max - x_min
        
        fit_deg = deg if row_length > 10.0 else 1
        coefs = np.polyfit(X, Y, fit_deg)
        poly_func = np.poly1d(coefs)
        
        xs = np.linspace(x_min, x_max, 50 if fit_deg > 1 else 2)
        ys = poly_func(xs)
        pts_rot = np.vstack([xs, ys]).T
        pts_orig = (pts_rot @ R) + mean
        line = LineString(pts_orig)
        
        intersection = line.intersection(poly)
        if intersection.is_empty:
            return line
        if intersection.geom_type == 'LineString':
            return intersection
        elif intersection.geom_type == 'MultiLineString':
            valid_lines = [geom for geom in intersection.geoms if not geom.is_empty]
            if valid_lines:
                return max(valid_lines, key=lambda l: l.length)
        return line
    except Exception as e:
        print(f"[WARN] Falha no cálculo de fallback para polígono: {e}")
        try:
            cov = np.cov(coords_centered.T)
            eigenvalues, eigenvectors = np.linalg.eig(cov)
            idx_max = np.argmax(eigenvalues)
            ex = eigenvectors[:, idx_max]
            ey = np.array([-ex[1], ex[0]])
            R = np.vstack([ex, ey])
            
            coords_rot = coords_centered @ R.T
            X = coords_rot[:, 0]
            Y = coords_rot[:, 1]
            x_min, x_max = X.min(), X.max()
            
            xs = np.array([x_min, x_max])
            ys = np.array([Y[np.argmin(X)], Y[np.argmax(X)]])
            pts_rot = np.vstack([xs, ys]).T
            pts_orig = (pts_rot @ R) + mean
            line = LineString(pts_orig)
            
            intersection = line.intersection(poly)
            if not intersection.is_empty:
                if intersection.geom_type == 'LineString':
                    return intersection
                elif intersection.geom_type == 'MultiLineString':
                    valid_lines = [geom for geom in intersection.geoms if not geom.is_empty]
                    if valid_lines:
                        return max(valid_lines, key=lambda l: l.length)
            return line
        except Exception:
            return None

def find_parent(parent, i):
    if parent[i] == i:
        return i
    parent[i] = find_parent(parent, parent[i])
    return parent[i]

def union_sets(parent, rank, i, j):
    root_i = find_parent(parent, i)
    root_j = find_parent(parent, j)
    if root_i != root_j:
        if rank[root_i] < rank[root_j]:
            parent[root_i] = root_j
        elif rank[root_i] > rank[root_j]:
            parent[root_j] = root_i
        else:
            parent[root_j] = root_i
            rank[root_i] += 1
        return True
    return False

def merge_collinear_lines(lines, max_gap_m=18.0, max_offset_m=1.8, max_angle_deg=15.0):
    """
    Mescla linhas centrais (LineStrings) curvas que estão alinhadas e próximas,
    fechando lacunas de árvores mortas/erradicadas sem misturar linhas paralelas.
    Preserva a curvatura concatenando as coordenadas das linhas ordenadamente.
    """
    if not lines:
        return []
        
    n = len(lines)
    parent = list(range(n))
    rank = [0] * n
    
    # Criar mapeamento de id para índice
    geom_to_idx = {id(g): i for i, g in enumerate(lines)}
    tree = STRtree(lines)
    
    for i in range(n):
        l1 = lines[i]
        coords1 = np.array(l1.coords)
        if len(coords1) < 2:
            continue
        p1_start, p1_end = coords1[0], coords1[-1]
        v1 = p1_end - p1_start
        len1 = np.linalg.norm(v1)
        if len1 == 0:
            continue
        u1 = v1 / len1
        perp1 = np.array([-u1[1], u1[0]])
        
        # Buscar candidatos próximos
        buffer_geom = l1.buffer(max_gap_m + max_offset_m)
        candidate_geoms = tree.query(buffer_geom)
        
        if len(candidate_geoms) > 0:
            if isinstance(candidate_geoms[0], (int, np.integer)):
                candidate_indices = list(candidate_geoms)
            else:
                candidate_indices = [geom_to_idx[id(g)] for g in candidate_geoms if id(g) in geom_to_idx]
        else:
            candidate_indices = []
            
        for j in candidate_indices:
            if j <= i:
                continue
                
            l2 = lines[j]
            coords2 = np.array(l2.coords)
            if len(coords2) < 2:
                continue
            p2_start, p2_end = coords2[0], coords2[-1]
            v2 = p2_end - p2_start
            len2 = np.linalg.norm(v2)
            if len2 == 0:
                continue
            u2 = v2 / len2
            
            # 1. Checar ângulo
            cos_theta = np.abs(np.dot(u1, u2))
            angle_deg = np.arccos(np.clip(cos_theta, -1.0, 1.0)) * 180.0 / np.pi
            if angle_deg > max_angle_deg:
                continue
                
            # 2. Checar gap longitudinal
            combos = [
                (p1_end, p2_start),
                (p1_end, p2_end),
                (p1_start, p2_start),
                (p1_start, p2_end)
            ]
            valid_gap = False
            for pt_a, pt_b in combos:
                d = np.linalg.norm(pt_a - pt_b)
                if d <= max_gap_m:
                    valid_gap = True
                    break
            if not valid_gap:
                continue
                
            # 3. Checar desvio lateral (offset)
            offset_start2 = np.abs(np.dot(p2_start - p1_start, perp1))
            offset_end2 = np.abs(np.dot(p2_end - p1_start, perp1))
            if offset_start2 > max_offset_m or offset_end2 > max_offset_m:
                continue
                
            union_sets(parent, rank, i, j)
            
    # Agrupar linhas pelos conjuntos Union-Find
    groups = {}
    for i in range(n):
        root = find_parent(parent, i)
        if root not in groups:
            groups[root] = []
        groups[root].append(i)
        
    merged_lines = []
    for root, indices in groups.items():
        if len(indices) == 1:
            merged_lines.append(lines[indices[0]])
            continue
            
        # Coletar pontos
        all_pts = []
        for idx in indices:
            coords = np.array(lines[idx].coords)
            all_pts.append(coords[0])
            all_pts.append(coords[-1])
        all_pts = np.array(all_pts)
        
        # Projetar na direção da linha mais longa do grupo
        lengths = [lines[idx].length for idx in indices]
        longest_idx = indices[np.argmax(lengths)]
        l_ref = lines[longest_idx]
        ref_start = np.array(l_ref.coords[0])
        ref_end = np.array(l_ref.coords[-1])
        ref_v = ref_end - ref_start
        ref_len = np.linalg.norm(ref_v)
        
        if ref_len > 0:
            ref_u = ref_v / ref_len
            
            # Ordenar índices com base no centroide de cada linha projetado em ref_u
            sorted_indices = sorted(indices, key=lambda idx: np.dot(np.array(lines[idx].coords).mean(axis=0) - ref_start, ref_u))
            
            merged_coords = []
            for idx in sorted_indices:
                line_coords = list(lines[idx].coords)
                l_v = np.array(line_coords[-1]) - np.array(line_coords[0])
                if np.dot(l_v, ref_u) < 0:
                    line_coords = line_coords[::-1]
                    
                if not merged_coords:
                    merged_coords.extend(line_coords)
                else:
                    if np.linalg.norm(np.array(merged_coords[-1]) - np.array(line_coords[0])) < 0.001:
                        merged_coords.extend(line_coords[1:])
                    else:
                        merged_coords.extend(line_coords)
            
            if len(merged_coords) >= 2:
                merged_lines.append(LineString(merged_coords))
            else:
                merged_lines.append(lines[indices[0]])
        else:
            merged_lines.append(lines[indices[0]])
            
    return merged_lines

def extend_and_deduplicate_lines(lines, talhao_geom, min_row_spacing_m=4.5, dominant_angle=None):
    """
    Estende as pontas de cada linha curva tangencialmente até a borda do talhão,
    preservando as curvas internas originais, e remove fileiras duplicadas.
    """
    if not lines or talhao_geom.is_empty:
        return []
        
    bounds = talhao_geom.bounds
    diag = np.sqrt((bounds[2] - bounds[0])**2 + (bounds[3] - bounds[1])**2)
    ext_length = diag * 1.5
    
    extended_lines = []
    
    for line in lines:
        coords = list(line.coords)
        if len(coords) < 2:
            continue
            
        p_start = np.array(coords[0])
        p_second = np.array(coords[1])
        p_end = np.array(coords[-1])
        p_penultimate = np.array(coords[-2])
        
        # Extensão no início
        v_start = p_start - p_second
        len_start = np.linalg.norm(v_start)
        if len_start > 0:
            u_start = v_start / len_start
            p_start_ext = p_start + ext_length * u_start
        else:
            p_start_ext = p_start
            
        # Extensão no fim
        v_end = p_end - p_penultimate
        len_end = np.linalg.norm(v_end)
        if len_end > 0:
            u_end = v_end / len_end
            p_end_ext = p_end + ext_length * u_end
        else:
            p_end_ext = p_end
            
        extended_coords = [tuple(p_start_ext)] + coords + [tuple(p_end_ext)]
        long_line = LineString(extended_coords)
        
        intersection = long_line.intersection(talhao_geom)
        if intersection.is_empty:
            continue
            
        if intersection.geom_type == 'LineString':
            extended_lines.append(intersection)
        elif intersection.geom_type == 'MultiLineString':
            for geom in intersection.geoms:
                if not geom.is_empty:
                    extended_lines.append(geom)
                    
    # Deduplicação por distância curva real
    deduplicated = []
    for line in extended_lines:
        is_duplicate = False
        for ref_line in deduplicated:
            dist = line.distance(ref_line)
            if dist < (min_row_spacing_m * 0.6):
                is_duplicate = True
                break
        if not is_duplicate:
            deduplicated.append(line)
            
    return deduplicated

def main():
    parser = argparse.ArgumentParser(
        description="Gera linhas centrais baseadas em MST a partir de pontos de árvores, preservando curvas.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--input_polygon", "-p", required=True,
        help="Arquivo de polígonos de linhas locais (GeoJSON, Shapefile ou GPKG)."
    )
    parser.add_argument(
        "--input_points", "-pts", required=True,
        help="Arquivo de pontos guia de contagem de plantas (GeoJSON ou Shapefile)."
    )
    parser.add_argument(
        "--output_shp", "-o", required=True,
        help="Arquivo Shapefile (.shp) de saída."
    )
    parser.add_argument(
        "--boundary", "-b", default=None,
        help="Polígono limite do talhão para recorte e extensão."
    )
    parser.add_argument(
        "--deg", "-d", type=int, default=2,
        help="Grau do ajuste polinomial de fallback."
    )
    parser.add_argument(
        "--min_points", type=int, default=2,
        help="Mínimo de pontos guia para ativar ajuste estruturado."
    )
    parser.add_argument(
        "--max_gap", type=float, default=18.0,
        help="Distância longitudinal máxima de gap (metros) para união."
    )
    parser.add_argument(
        "--max_offset", type=float, default=1.8,
        help="Deslocamento lateral máximo (metros) para união."
    )
    parser.add_argument(
        "--min_row_spacing", type=float, default=3.5,
        help="Espaçamento lateral mínimo (metros) entre fileiras paralelas."
    )
    
    args = parser.parse_args()
    
    print("=" * 75)
    print("   GERADOR DE LINHAS DE CENTRO CURVAS BASEADO EM MST")
    print("=" * 75)
    print(f"Polígonos : {args.input_polygon}")
    print(f"Pontos    : {args.input_points}")
    print(f"Limite    : {args.boundary}")
    print(f"Saída     : {args.output_shp}")
    print("-" * 75)
    
    if not os.path.exists(args.input_polygon):
        print(f"[ERRO] Polígonos de entrada não encontrados: {args.input_polygon}")
        sys.exit(1)
    if not os.path.exists(args.input_points):
        print(f"[ERRO] Pontos guia não encontrados: {args.input_points}")
        sys.exit(1)
        
    try:
        print("[INFO] Carregando polígonos...")
        gdf_polygons = gpd.read_file(args.input_polygon)
        print("[INFO] Carregando pontos...")
        gdf_points = gpd.read_file(args.input_points)
    except Exception as e:
        print(f"[ERRO] Falha ao ler arquivos: {e}")
        sys.exit(1)
        
    gdf_boundary = None
    if args.boundary and os.path.exists(args.boundary):
        try:
            gdf_boundary = gpd.read_file(args.boundary)
            print("[OK] Limite do talhão carregado com sucesso.")
        except Exception as e:
            print(f"[WARN] Falha ao carregar limite do talhão: {e}")
            
    original_crs = gdf_polygons.crs
    if original_crs is None:
        original_crs = "EPSG:4326"
        gdf_polygons.set_crs(original_crs, inplace=True)
        
    if gdf_points.crs is None:
        gdf_points.set_crs(original_crs, inplace=True)
        
    is_geographic = original_crs.is_geographic if hasattr(original_crs, 'is_geographic') else True
    if is_geographic:
        try:
            metric_crs = gdf_polygons.estimate_utm_crs()
            print(f"[INFO] Projetando dados para UTM local ({metric_crs.to_string()})...")
            gdf_poly_proj = gdf_polygons.to_crs(metric_crs)
            gdf_points_proj = gdf_points.to_crs(metric_crs)
            if gdf_boundary is not None:
                gdf_boundary_proj = gdf_boundary.to_crs(metric_crs)
        except Exception as e:
            metric_crs = "EPSG:3857"
            print(f"[WARN] Falha ao estimar UTM ({e}). Projetando para EPSG:3857...")
            gdf_poly_proj = gdf_polygons.to_crs(metric_crs)
            gdf_points_proj = gdf_points.to_crs(metric_crs)
            if gdf_boundary is not None:
                gdf_boundary_proj = gdf_boundary.to_crs(metric_crs)
    else:
        metric_crs = original_crs
        gdf_poly_proj = gdf_polygons
        gdf_points_proj = gdf_points.to_crs(metric_crs)
        if gdf_boundary is not None:
            gdf_boundary_proj = gdf_boundary.to_crs(metric_crs)
            
    # Filtrar espacialmente se limite fornecido
    if gdf_boundary is not None and not gdf_boundary_proj.empty:
        print("[INFO] Filtrando polígonos dentro do limite do talhão...")
        boundary_geom = gdf_boundary_proj.geometry.union_all()
        gdf_poly_proj = gdf_poly_proj[gdf_poly_proj.geometry.intersects(boundary_geom)].reset_index(drop=True)
        print(f"  --> Restaram {len(gdf_poly_proj)} polígonos locais dentro do talhão.")
        
    # Explodir Multipolygons
    gdf_poly_proj = gdf_poly_proj.explode(index_parts=False).reset_index(drop=True)
    gdf_poly_proj['temp_poly_id'] = range(len(gdf_poly_proj))
    
    # sjoin espacial
    print("[INFO] 1. Associando pontos aos polígonos locais via sjoin...")
    points_by_poly = {}
    try:
        joined = gpd.sjoin(gdf_points_proj, gdf_poly_proj[['geometry', 'temp_poly_id']], how='inner', predicate='intersects')
        if not joined.empty:
            for poly_id, group in joined.groupby('temp_poly_id'):
                points_by_poly[poly_id] = group
        print(f"  [OK] Junção concluída. {len(points_by_poly)} polígonos contêm pontos guia.")
    except Exception as e:
        print(f"[WARN] Falha ao executar sjoin em lote: {e}. Usando busca sindex...")
        points_by_poly = None
        
    local_segments = []
    print("[INFO] 2. Extraindo segmentos de linhas curvas locais via MST...")
    for idx, row in gdf_poly_proj.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
            
        poly_id = row['temp_poly_id']
        pts_inside = None
        if points_by_poly is not None:
            pts_inside = points_by_poly.get(poly_id)
        else:
            if gdf_points_proj.sindex is not None:
                possible_idx = list(gdf_points_proj.sindex.intersection(geom.bounds))
                possible = gdf_points_proj.iloc[possible_idx]
                pts_inside = possible[possible.geometry.intersects(geom)]
                
        if pts_inside is None:
            pts_inside = gpd.GeoDataFrame(geometry=[], crs=gdf_points_proj.crs)
            
        line = extract_centerline_guided(geom, pts_inside, deg=args.deg, min_points=args.min_points)
        if line is not None and not line.is_empty:
            local_segments.append(line)
            
    if not local_segments:
        print("[ERRO] Nenhum segmento pôde ser gerado.")
        sys.exit(1)
        
    print(f"  --> {len(local_segments)} segmentos locais gerados.")
    
    # Mesclagem colinear curvatura preservada
    print(f"[INFO] 3. Mesclando segmentos colineares (gap max={args.max_gap}m)...")
    merged_lines = merge_collinear_lines(
        local_segments, max_gap_m=args.max_gap, max_offset_m=args.max_offset, max_angle_deg=15.0
    )
    print(f"  --> {len(merged_lines)} linhas após mesclagem.")
    
    # Extensão tangencial e recorte (se talhão disponível)
    final_lines = merged_lines
    if gdf_boundary is not None and not gdf_boundary_proj.empty:
        print("[INFO] 4. Estendendo e deduplicando as fileiras curvas até as bordas do talhão...")
        talhao_geom = gdf_boundary_proj.geometry.union_all()
        final_lines = extend_and_deduplicate_lines(
            merged_lines, talhao_geom, min_row_spacing_m=args.min_row_spacing
        )
        print(f"  --> {len(final_lines)} fileiras curvas finais geradas.")
        
    # Exportação
    print("[INFO] 5. Exportando resultados para WGS84 (EPSG:4326)...")
    gdf_out_proj = gpd.GeoDataFrame(geometry=final_lines, crs=metric_crs)
    gdf_out_proj['length_m'] = gdf_out_proj.geometry.length
    
    gdf_out_wgs84 = gdf_out_proj.to_crs(original_crs)
    
    os.makedirs(os.path.dirname(os.path.abspath(args.output_shp)), exist_ok=True)
    if not args.output_shp.lower().endswith(".shp"):
        args.output_shp = os.path.splitext(args.output_shp)[0] + ".shp"
        
    try:
        gdf_out_wgs84.to_file(args.output_shp)
        print(f"\n[OK] Execução concluída! Arquivo salvo em:\n     {args.output_shp}")
    except Exception as e:
        print(f"[ERRO] Falha ao exportar shapefile de saída: {e}")
        sys.exit(1)
        
    print("=" * 75)

if __name__ == "__main__":
    main()
