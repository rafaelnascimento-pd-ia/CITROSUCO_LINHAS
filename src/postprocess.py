import os
import sys
import time
import json
import datetime
import numpy as np
import cv2
import rasterio
import rasterio.features
from scipy.ndimage import binary_fill_holes, gaussian_filter1d
from scipy.interpolate import splprep, splev
import geopandas as gpd
from shapely.geometry import shape, MultiPolygon, LineString, MultiLineString, Point, Polygon, GeometryCollection
from shapely.ops import unary_union, substring
from shapely.strtree import STRtree
from skimage.morphology import skeletonize
import networkx as nx

from src.evaluation import (
    compute_center_error,
    compute_parallelism,
    compute_overlap_ratio,
    compute_prob_alignment,
    compute_local_curvature
)


def calculate_lines_dominant_angle(lines_path, target_crs=None):
    """
    Calcula a direcao dominante (angulo em radianos) a partir de um arquivo de linhas (.geojson ou .shp).
    """
    try:
        gdf = gpd.read_file(lines_path)
        if gdf.empty:
            return None
        if target_crs is not None and gdf.crs is not None:
            gdf = gdf.to_crs(target_crs)
            
        angles = []
        for geom in gdf.geometry:
            if geom is None or geom.is_empty:
                continue
            if geom.geom_type == 'LineString':
                coords = np.array(geom.coords)
                if len(coords) >= 2:
                    v = coords[-1] - coords[0]
                    length = np.linalg.norm(v)
                    if length > 0:
                        angle = np.arctan2(v[1], v[0])
                        while angle < -np.pi / 2:
                            angle += np.pi
                        while angle > np.pi / 2:
                            angle -= np.pi
                        angles.append(angle)
            elif geom.geom_type == 'MultiLineString':
                for line in geom.geoms:
                    coords = np.array(line.coords)
                    if len(coords) >= 2:
                        v = coords[-1] - coords[0]
                        length = np.linalg.norm(v)
                        if length > 0:
                            angle = np.arctan2(v[1], v[0])
                            while angle < -np.pi / 2:
                                angle += np.pi
                            while angle > np.pi / 2:
                                angle -= np.pi
                            angles.append(angle)
        if not angles:
            return None
            
        return float(np.median(angles))
    except Exception as e:
        print(f"[WARN] Erro ao calcular angulo das linhas de referencia: {e}")
        return None


def calculate_dominant_angle(polygons):
    """
    Calcula a direcao dominante (angulo em radianos) das fileiras de plantio no talhao.
    """
    angles = []
    for poly in polygons:
        if poly.geom_type == 'MultiPolygon':
            if poly.is_empty:
                continue
            poly = max(poly.geoms, key=lambda p: p.area)
            
        coords = np.array(poly.exterior.coords)
        if len(coords) < 3:
            continue
            
        try:
            mean = coords.mean(axis=0)
            coords_centered = coords - mean
            cov = np.cov(coords_centered.T)
            if cov.ndim < 2:
                continue
            eigenvalues, eigenvectors = np.linalg.eigh(cov.real)
            val_max = eigenvalues[-1]
            val_min = eigenvalues[0]
            
            if val_min > 0 and (val_max / val_min) >= 3.0:
                ex = eigenvectors[:, -1]
                angle = np.arctan2(ex[1], ex[0])
                while angle < -np.pi / 2:
                    angle += np.pi
                while angle > np.pi / 2:
                    angle -= np.pi
                angles.append((angle, poly.area))
        except Exception:
            continue

    if not angles:
        return 0.0
        
    angles.sort(key=lambda x: x[1], reverse=True)
    top_n = max(3, int(len(angles) * 0.3))
    top_angles = [a[0] for a in angles[:top_n]]
    
    return float(np.median(top_angles))


def detect_profile_from_polygons(polygons, logger=print):
    """
    Analisa a variancia angular dos poligonos detectados para decidir automaticamente
    se o talhao e de 'tiro_reto' ou 'curva'.
    """
    angles = []
    for poly in polygons:
        if poly.geom_type == 'MultiPolygon':
            if poly.is_empty:
                continue
            poly = max(poly.geoms, key=lambda p: p.area)
        coords = np.array(poly.exterior.coords)
        if len(coords) < 3:
            continue
        try:
            mean = coords.mean(axis=0)
            coords_centered = coords - mean
            cov = np.cov(coords_centered.T)
            if cov.ndim < 2:
                continue
            eigenvalues, eigenvectors = np.linalg.eigh(cov.real)
            val_max, val_min = eigenvalues[-1], eigenvalues[0]
            if val_min > 0 and (val_max / val_min) >= 2.5:
                ex = eigenvectors[:, -1]
                angle = np.arctan2(ex[1], ex[0])
                while angle < -np.pi / 2:
                    angle += np.pi
                while angle > np.pi / 2:
                    angle -= np.pi
                angles.append(np.degrees(angle))
        except Exception:
            continue

    if len(angles) < 3:
        logger("  [Auto Detect] Poucos poligonos com elongacao confiavel. Assumindo 'tiro_reto'.")
        return "tiro_reto"

    std_dev = float(np.std(angles))
    logger(f"  [Auto Detect] Dispersao angular: {std_dev:.2f} graus (limiar: 14.0 graus).")
    if std_dev < 14.0:
        return "tiro_reto"
    else:
        return "curva"


def smooth_linestring_gaussian(line, sigma=1.0):
    """
    Aplica suavizacao Gaussiana 1D ao longo dos eixos X e Y de uma LineString,
    mantendo os pontos extremos (start/end) perfeitamente intactos.
    """
    if line is None or line.is_empty:
        return line
    coords = np.array(line.coords)
    if len(coords) < 4:
        return line

    try:
        x_smooth = gaussian_filter1d(coords[:, 0], sigma=sigma, mode='nearest')
        y_smooth = gaussian_filter1d(coords[:, 1], sigma=sigma, mode='nearest')

        # Garante que as pontas extremas originais nao se movam
        x_smooth[0], x_smooth[-1] = coords[0, 0], coords[-1, 0]
        y_smooth[0], y_smooth[-1] = coords[0, 1], coords[-1, 1]

        smoothed = LineString(np.column_stack([x_smooth, y_smooth]))
        return smoothed if (smoothed.is_valid and not smoothed.is_empty) else line
    except Exception:
        return line


def filter_polygons_by_elongation(polygons, min_elongation=3.5, logger=print):
    """
    Filtra e descarta poligonos que nao tem formato alongado de fileira (ruidos, entrelinhas, manchas).
    """
    filtered = []
    discarded = 0
    for poly in polygons:
        if poly.is_empty:
            continue
        cur = poly
        if cur.geom_type == 'MultiPolygon':
            cur = max(cur.geoms, key=lambda p: p.area)
        coords = np.array(cur.exterior.coords)
        if len(coords) < 4:
            filtered.append(poly)
            continue
        try:
            mean = coords.mean(axis=0)
            coords_centered = coords - mean
            cov = np.cov(coords_centered.T)
            if cov.ndim < 2:
                filtered.append(poly)
                continue
            eigenvalues, _ = np.linalg.eigh(cov.real)
            val_min = eigenvalues[0]
            val_max = eigenvalues[-1]
            if val_min <= 1e-6:
                filtered.append(poly)
                continue
            elongation = val_max / val_min
            if elongation >= min_elongation:
                filtered.append(poly)
            else:
                discarded += 1
        except Exception:
            filtered.append(poly)

    logger(f"  [Filtro Elongacao] {len(filtered)} poligonos mantidos, {discarded} descartados como entrelinhas/ruidos.")
    return filtered


def prune_skeleton_graph(G, min_length_px, protected_path=None):
    """
    Remove ramos laterais curtos (< min_length_px) preservando o caminho principal e seus endpoints reais.
    """
    if G.number_of_nodes() < 3 or min_length_px <= 0:
        return G

    H = G.copy()
    protected_nodes = set(protected_path) if protected_path else set()

    changed = True
    iteration = 0
    max_iter = int(min_length_px) + 2

    while changed and iteration < max_iter:
        changed = False
        iteration += 1
        degree_1 = [n for n, d in H.degree() if d == 1 and n not in protected_nodes]
        if not degree_1:
            break

        for n in degree_1:
            if not H.has_node(n):
                continue
            curr = n
            branch = [curr]
            visited = {curr}
            while True:
                unvisited = [nbr for nbr in H.neighbors(curr) if nbr not in visited]
                if len(unvisited) != 1:
                    break
                nxt = unvisited[0]
                if nxt in protected_nodes or H.degree(nxt) > 2:
                    break
                branch.append(nxt)
                visited.add(nxt)
                curr = nxt

            if len(branch) <= min_length_px:
                for b_node in branch:
                    if b_node not in protected_nodes and H.has_node(b_node):
                        H.remove_node(b_node)
                        changed = True

    return H


def resample_line_by_length(coords, step_m=1.0, preserve_curvature=True):
    """
    Reamostra uma sequencia de coordenadas por comprimento acumulado em unidades metricas,
    garantindo que curvas acentuadas e endpoints sejam preservados.
    """
    coords = np.asarray(coords)
    if len(coords) < 2:
        return coords

    diffs = coords[1:] - coords[:-1]
    seg_lens = np.linalg.norm(diffs, axis=1)
    cum_dist = np.insert(np.cumsum(seg_lens), 0, 0.0)
    total_len = cum_dist[-1]

    if total_len <= 1e-4:
        return coords[:1]

    num_samples = max(2, int(np.ceil(total_len / max(0.1, step_m))) + 1)
    target_dists = list(np.linspace(0, total_len, num_samples))

    if preserve_curvature and len(coords) >= 3:
        v1 = diffs[:-1]
        v2 = diffs[1:]
        norms = seg_lens[:-1] * seg_lens[1:]
        valid_idx = np.where(norms > 1e-6)[0]
        for idx in valid_idx:
            cos_a = np.clip(np.dot(v1[idx], v2[idx]) / norms[idx], -1.0, 1.0)
            ang_deg = np.degrees(np.arccos(cos_a))
            if ang_deg > 15.0:
                curv_dist = cum_dist[idx + 1]
                target_dists.append(curv_dist)

    target_dists = sorted(set(target_dists))

    resampled_x = np.interp(target_dists, cum_dist, coords[:, 0])
    resampled_y = np.interp(target_dists, cum_dist, coords[:, 1])

    resampled_x[0], resampled_x[-1] = coords[0, 0], coords[-1, 0]
    resampled_y[0], resampled_y[-1] = coords[0, 1], coords[-1, 1]

    return np.column_stack([resampled_x, resampled_y])


def smooth_line_arc_length(coords, smoothing=1.0, preserve_endpoints=True):
    """
    Suaviza uma linha parametrizada pelo comprimento acumulado do arco.
    Usa spline parametrica regularizada com fallback para filtro gaussiano com endpoints intactos.
    """
    coords = np.asarray(coords)
    if len(coords) < 4:
        return coords

    try:
        diffs = coords[1:] - coords[:-1]
        dist = np.linalg.norm(diffs, axis=1)
        keep = np.insert(dist > 1e-4, 0, True)
        clean_coords = coords[keep]
        if len(clean_coords) < 4:
            return coords

        x, y = clean_coords[:, 0], clean_coords[:, 1]
        m = len(x)
        k = 3 if m > 3 else 2
        s = float(m) * max(0.1, float(smoothing))

        tck, u = splprep([x, y], s=s, k=k)
        u_new = np.linspace(0, 1, m)
        new_x, new_y = splev(u_new, tck)

        new_coords = np.column_stack([new_x, new_y])
        if preserve_endpoints:
            new_coords[0] = coords[0]
            new_coords[-1] = coords[-1]
        return new_coords
    except Exception:
        x_sm = gaussian_filter1d(coords[:, 0], sigma=smoothing, mode='nearest')
        y_sm = gaussian_filter1d(coords[:, 1], sigma=smoothing, mode='nearest')
        if preserve_endpoints:
            x_sm[0], x_sm[-1] = coords[0, 0], coords[-1, 0]
            y_sm[0], y_sm[-1] = coords[0, 1], coords[-1, 1]
        return np.column_stack([x_sm, y_sm])


def enforce_line_inside_polygon(line, polygon, tolerance_m=0.0):
    """
    Garante que a linha permaneca estritamente dentro do poligono da copa/fileira.
    Se a linha escapar em trechos concavos, extrai a componente continua dominante.
    """
    if line is None or line.is_empty or polygon is None or polygon.is_empty:
        return line

    try:
        if polygon.contains(line):
            return line

        poly_buf = polygon.buffer(tolerance_m) if tolerance_m > 0 else polygon
        if poly_buf.contains(line):
            return line

        inter = line.intersection(poly_buf)
        if inter.is_empty:
            return line

        if inter.geom_type == 'LineString':
            return inter if inter.length > 0.5 else line
        elif inter.geom_type == 'MultiLineString':
            valid_geoms = [g for g in inter.geoms if g.length > 0.5]
            if valid_geoms:
                return max(valid_geoms, key=lambda g: g.length)
        elif inter.geom_type == 'GeometryCollection':
            lines = [g for g in inter.geoms if g.geom_type == 'LineString' and g.length > 0.5]
            if lines:
                return max(lines, key=lambda g: g.length)

        return line
    except Exception:
        return line


def extract_centerline_from_polygon(poly, resolution=0.25, centerline_method="skeleton_v2", resample_step_m=1.0):
    """
    Extrai a linha central (LineString) de um poligono de fileira usando o esqueleto (eixo medial).
    Suporta centerline_method="skeleton_v2" (com poda de ramos, reamostragem metrica e retencao interna)
    e centerline_method="skeleton_legacy" (comportamento legado).
    """
    if poly is None or poly.is_empty:
        return None
    if poly.geom_type == 'MultiPolygon':
        poly = max(poly.geoms, key=lambda p: p.area)
    if poly.area < 1.0:
        return None

    try:
        minx, miny, maxx, maxy = poly.bounds
        dx, dy = maxx - minx, maxy - miny
        if dx <= 0 or dy <= 0:
            return None

        # Resolucao adaptativa para precisao e velocidade
        res = max(0.2, min(resolution, min(dx, dy) / 6.0))
        w = int(np.ceil(dx / res)) + 4
        h = int(np.ceil(dy / res)) + 4

        # Protecao contra dimensoes exageradas
        if w * h > 3000000:
            res = max(dx, dy) / 1200.0
            w = int(np.ceil(dx / res)) + 4
            h = int(np.ceil(dy / res)) + 4

        transform = rasterio.transform.from_origin(minx - 2 * res, maxy + 2 * res, res, res)
        mask = rasterio.features.rasterize([(poly, 1)], out_shape=(h, w), transform=transform, fill=0, dtype=np.uint8)

        skel = skeletonize(mask > 0)
        pts_y, pts_x = np.where(skel)
        if len(pts_x) < 3:
            return None

        # Grafo 8-conectado dos pixels do esqueleto
        G = nx.Graph()
        for y, x in zip(pts_y, pts_x):
            G.add_node((y, x))
            for d_y in [-1, 0, 1]:
                for d_x in [-1, 0, 1]:
                    if d_y == 0 and d_x == 0:
                        continue
                    ny, nx_ = y + d_y, x + d_x
                    if 0 <= ny < h and 0 <= nx_ < w and skel[ny, nx_]:
                        G.add_edge((y, x), (ny, nx_), weight=np.hypot(d_y, d_x))

        # Encontra o caminho mais longo (diametro do esqueleto)
        longest_path = []
        max_path_len = 0.0

        for comp in nx.connected_components(G):
            subG = G.subgraph(comp)
            nodes = list(subG.nodes())
            if len(nodes) < 2:
                continue
            deg1 = [n for n, d in subG.degree() if d == 1]
            start_node = deg1[0] if len(deg1) > 0 else nodes[0]
            lengths = nx.single_source_dijkstra_path_length(subG, start_node, weight='weight')
            farthest_1 = max(lengths, key=lengths.get)
            lengths2, paths2 = nx.single_source_dijkstra(subG, farthest_1, weight='weight')
            farthest_2 = max(lengths2, key=lengths2.get)
            if lengths2[farthest_2] > max_path_len:
                max_path_len = lengths2[farthest_2]
                longest_path = paths2[farthest_2]

        if len(longest_path) < 2:
            return None

        if centerline_method == "skeleton_v2":
            # Poda ramos laterais espurios protegendo o caminho principal
            min_prune_px = max(3, int(1.2 / res))
            G_pruned = prune_skeleton_graph(G, min_length_px=min_prune_px, protected_path=longest_path)
            
            # Recalcula se o grafo podado manteve componentes validas
            if G_pruned.number_of_nodes() >= 2:
                for comp in nx.connected_components(G_pruned):
                    if longest_path[0] in comp and longest_path[-1] in comp:
                        subG_p = G_pruned.subgraph(comp)
                        lengths_p, paths_p = nx.single_source_dijkstra(subG_p, longest_path[0], weight='weight')
                        if longest_path[-1] in paths_p:
                            longest_path = paths_p[longest_path[-1]]
                        break

            # Converte todos os pixels do caminho para coordenadas metricas do mapa
            raw_coords = np.array([transform * (x + 0.5, y + 0.5) for y, x in longest_path])
            if len(raw_coords) < 2:
                return None

            # Reamostragem por comprimento metrico acumulado com preservacao de curvas
            resampled = resample_line_by_length(raw_coords, step_m=resample_step_m, preserve_curvature=True)
            
            # Suavizacao ao longo do arco (spline parametrica)
            smoothed_coords = smooth_line_arc_length(resampled, smoothing=1.0, preserve_endpoints=True)
            line = LineString(smoothed_coords)

            # Restringe a linha para ficar estritamente no poligono
            line = enforce_line_inside_polygon(line, poly)
            return line if (line is not None and line.is_valid and not line.is_empty) else None

        elif centerline_method == "skeleton_hybrid":
            # HIBRIDO: Poda leve de ramos (v2) + suavizacao gaussiana robusta (legacy)
            # Sem enforce_line_inside_polygon (evita cortar linhas validas em concavidades)
            min_prune_px = max(3, int(1.2 / res))
            G_pruned = prune_skeleton_graph(G, min_length_px=min_prune_px, protected_path=longest_path)

            # Recalcula caminho no grafo podado
            if G_pruned.number_of_nodes() >= 2:
                for comp in nx.connected_components(G_pruned):
                    if longest_path[0] in comp and longest_path[-1] in comp:
                        subG_p = G_pruned.subgraph(comp)
                        lengths_p, paths_p = nx.single_source_dijkstra(subG_p, longest_path[0], weight='weight')
                        if longest_path[-1] in paths_p:
                            longest_path = paths_p[longest_path[-1]]
                        break

            # Amostragem do caminho (metodo legacy — mais estavel)
            step = max(1, int(1.0 / res))
            sampled = longest_path[::step]
            if sampled[-1] != longest_path[-1]:
                sampled.append(longest_path[-1])

            coords = np.array([transform * (x + 0.5, y + 0.5) for y, x in sampled])
            if len(coords) < 2:
                return None

            # Suavizacao gaussiana (legacy — mais robusta para centralizacao)
            if len(coords) >= 4:
                coords_sm_x = gaussian_filter1d(coords[:, 0], sigma=1.0)
                coords_sm_y = gaussian_filter1d(coords[:, 1], sigma=1.0)
                coords_sm_x[0], coords_sm_x[-1] = coords[0, 0], coords[-1, 0]
                coords_sm_y[0], coords_sm_y[-1] = coords[0, 1], coords[-1, 1]
                line = LineString(np.vstack([coords_sm_x, coords_sm_y]).T)
            else:
                line = LineString(coords)

            return line if line.is_valid and not line.is_empty else None

        else:
            # Fallback para o metodo legado puro (sem poda)
            step = max(1, int(1.0 / res))
            sampled = longest_path[::step]
            if sampled[-1] != longest_path[-1]:
                sampled.append(longest_path[-1])

            coords = np.array([transform * (x + 0.5, y + 0.5) for y, x in sampled])
            if len(coords) < 2:
                return None

            if len(coords) >= 4:
                coords_sm_x = gaussian_filter1d(coords[:, 0], sigma=1.0)
                coords_sm_y = gaussian_filter1d(coords[:, 1], sigma=1.0)
                coords_sm_x[0], coords_sm_x[-1] = coords[0, 0], coords[-1, 0]
                coords_sm_y[0], coords_sm_y[-1] = coords[0, 1], coords[-1, 1]
                line = LineString(np.vstack([coords_sm_x, coords_sm_y]).T)
            else:
                line = LineString(coords)

            return line if line.is_valid and not line.is_empty else None

    except Exception:
        try:
            coords = np.array(poly.exterior.coords)
            c = coords.mean(axis=0)
            cov = np.cov((coords - c).T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov.real)
            ex = eigenvectors[:, -1]
            projs = np.dot(coords - c, ex)
            p1 = c + ex * projs.min()
            p2 = c + ex * projs.max()
            return LineString([p1, p2])
        except Exception:
            return None


def _get_endpoint_tangent(coords, is_start, window_m=4.0):
    coords = np.asarray(coords)
    if len(coords) < 2:
        return np.array([1.0, 0.0])
    diffs = coords[1:] - coords[:-1]
    dists = np.linalg.norm(diffs, axis=1)
    if is_start:
        acc = 0.0
        target_idx = 1
        for i, d in enumerate(dists):
            acc += d
            target_idx = i + 1
            if acc >= window_m:
                break
        v = coords[0] - coords[target_idx]
    else:
        acc = 0.0
        target_idx = len(coords) - 2
        for i in range(len(dists) - 1, -1, -1):
            acc += dists[i]
            target_idx = i
            if acc >= window_m:
                break
        v = coords[-1] - coords[target_idx]
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-6 else np.array([1.0, 0.0])


def merge_collinear_lines(lines, max_gap_m=12.0, max_offset_m=1.5, max_angle_deg=25.0,
                          merge_method="spatial_cost", weights=None, active_boundary=None):
    """
    Mescla linhas centrais que estao na mesma fileira (fechando gaps de arvores falhadas/matacoes),
    preservando toda a sinuosidade e todos os vertices da curva sem achatar para reta.
    Suporta merge_method="spatial_cost" (STRtree acelerado e custo multiobjetivo normalizado)
    e merge_method="greedy_legacy" (comportamento legado).
    """
    if not lines:
        return []

    if merge_method == "greedy_legacy":
        active_lines = list(lines)
        merged_any = True

        while merged_any:
            merged_any = False
            n = len(active_lines)
            to_remove = set()

            for i in range(n):
                if i in to_remove:
                    continue
                l1 = active_lines[i]
                c1 = list(l1.coords)
                if len(c1) < 2:
                    continue

                p1_start = np.array(c1[0])
                p1_end = np.array(c1[-1])

                merged_index = -1
                best_order = None

                for j in range(i + 1, n):
                    if j in to_remove:
                        continue
                    l2 = active_lines[j]
                    c2 = list(l2.coords)
                    if len(c2) < 2:
                        continue

                    p2_start = np.array(c2[0])
                    p2_end = np.array(c2[-1])

                    combos = [
                        ('end_start', np.linalg.norm(p1_end - p2_start), c1, c2),
                        ('end_end', np.linalg.norm(p1_end - p2_end), c1, c2[::-1]),
                        ('start_start', np.linalg.norm(p1_start - p2_start), c1[::-1], c2),
                        ('start_end', np.linalg.norm(p1_start - p2_end), c2, c1)
                    ]

                    best_combo = min(combos, key=lambda x: x[1])
                    mode, gap, part1, part2 = best_combo

                    if gap > max_gap_m:
                        continue

                    v1 = np.array(part1[-1]) - np.array(part1[-2])
                    v2 = np.array(part2[1]) - np.array(part2[0])
                    len1 = np.linalg.norm(v1)
                    len2 = np.linalg.norm(v2)
                    if len1 < 1e-6 or len2 < 1e-6:
                        continue
                    u1 = v1 / len1
                    u2 = v2 / len2

                    cos_theta = np.clip(np.dot(u1, u2), -1.0, 1.0)
                    angle_deg = np.arccos(cos_theta) * 180.0 / np.pi
                    if angle_deg > max_angle_deg:
                        continue

                    v_gap = np.array(part2[0]) - np.array(part1[-1])
                    perp1 = np.array([-u1[1], u1[0]])
                    offset = abs(np.dot(v_gap, perp1))
                    if offset > max_offset_m:
                        continue

                    merged_index = j
                    best_order = part1 + part2
                    break

                if merged_index != -1 and best_order is not None:
                    try:
                        merged_line = LineString(best_order)
                        if merged_line.is_valid and not merged_line.is_empty:
                            active_lines[i] = merged_line
                            to_remove.add(merged_index)
                            merged_any = True
                            break
                    except Exception:
                        pass

            active_lines = [active_lines[k] for k in range(len(active_lines)) if k not in to_remove]

        return active_lines

    # Metodo moderno: spatial_cost com STRtree e custo multiobjetivo
    weights = weights or {"gap": 0.35, "angle": 0.35, "offset": 0.30}
    active_lines = [ln for ln in lines if ln is not None and not ln.is_empty and ln.length > 0.1]
    merged_any = True
    pass_count = 0

    while merged_any and pass_count < 10:
        merged_any = False
        pass_count += 1
        n = len(active_lines)
        if n < 2:
            break

        tree = STRtree(active_lines)
        candidates = []

        for i in range(n):
            l1 = active_lines[i]
            c1 = list(l1.coords)
            if len(c1) < 2:
                continue

            p1_start = np.array(c1[0])
            p1_end = np.array(c1[-1])
            t1_start = _get_endpoint_tangent(c1, is_start=True, window_m=4.0)
            t1_end = _get_endpoint_tangent(c1, is_start=False, window_m=4.0)

            # Consulta vizinhos espaciais usando o envelope expandido de l1
            neighbor_indices = tree.query(l1.buffer(max_gap_m))

            for j in neighbor_indices:
                if j <= i:
                    continue
                l2 = active_lines[j]
                c2 = list(l2.coords)
                if len(c2) < 2:
                    continue

                p2_start = np.array(c2[0])
                p2_end = np.array(c2[-1])
                t2_start = _get_endpoint_tangent(c2, is_start=True, window_m=4.0)
                t2_end = _get_endpoint_tangent(c2, is_start=False, window_m=4.0)

                combos = [
                    ('end_start', np.linalg.norm(p1_end - p2_start), c1, c2, t1_end, -t2_start),
                    ('end_end', np.linalg.norm(p1_end - p2_end), c1, c2[::-1], t1_end, t2_end),
                    ('start_start', np.linalg.norm(p1_start - p2_start), c1[::-1], c2, -t1_start, -t2_start),
                    ('start_end', np.linalg.norm(p1_start - p2_end), c2, c1, -t2_end, t1_start)
                ]

                best_combo = min(combos, key=lambda x: x[1])
                mode, gap, part1, part2, u1, u2 = best_combo

                if gap > max_gap_m:
                    continue

                cos_theta = np.clip(np.dot(u1, u2), -1.0, 1.0)
                angle_deg = np.arccos(cos_theta) * 180.0 / np.pi
                if angle_deg > max_angle_deg:
                    continue

                v_gap = np.array(part2[0]) - np.array(part1[-1])
                perp1 = np.array([-u1[1], u1[0]])
                offset = abs(float(np.dot(v_gap, perp1)))
                if offset > max_offset_m:
                    continue

                # Valida ponte: nao pode cruzar outras fileiras
                bridge = LineString([part1[-1], part2[0]])
                bridge_valid = True
                if bridge.length > 0.5:
                    bridge_crossings = tree.query(bridge)
                    for b_idx in bridge_crossings:
                        if b_idx != i and b_idx != j:
                            other_line = active_lines[b_idx]
                            if bridge.crosses(other_line) or bridge.distance(other_line) < 0.2:
                                bridge_valid = False
                                break

                if not bridge_valid:
                    continue

                # Custo normalizado
                c_gap = gap / max(1.0, max_gap_m)
                c_ang = angle_deg / max(1.0, max_angle_deg)
                c_off = offset / max(0.1, max_offset_m)
                cost = (weights["gap"] * c_gap +
                        weights["angle"] * c_ang +
                        weights["offset"] * c_off)

                candidates.append((cost, i, j, part1, part2))

        if not candidates:
            break

        candidates.sort(key=lambda x: x[0])
        used_indices = set()
        new_active = []
        merged_map = {}

        for cost, i, j, part1, part2 in candidates:
            if i in used_indices or j in used_indices:
                continue

            try:
                # Conexao com preservacao dos vertices
                merged_coords = list(part1) + list(part2)
                merged_line = LineString(merged_coords)
                if merged_line.is_valid and not merged_line.is_empty:
                    merged_map[i] = merged_line
                    used_indices.add(i)
                    used_indices.add(j)
                    merged_any = True
            except Exception:
                continue

        for idx, ln in enumerate(active_lines):
            if idx in merged_map:
                new_active.append(merged_map[idx])
            elif idx not in used_indices:
                new_active.append(ln)

        active_lines = new_active

    return active_lines


def deduplicate_close_lines(lines, min_spacing_m=3.0, dedup_method="overlap_parallelism",
                            min_overlap_ratio=0.70, max_angle_deg=20.0, step_m=1.0,
                            dedup_factor=0.45):
    """
    Remove linhas duplicadas que estao muito proximas na mesma fileira.
    - overlap_parallelism: Exige alta sobreposicao longitudinal (>70%),
      paralelismo angular (<=20°) e proximidade metrica (< dedup_factor * min_spacing_m).
    - distance_legacy: Modo legado baseado apenas na menor distancia geometrica.
    """
    if not lines:
        return []

    sorted_lines = sorted(lines, key=lambda l: l.length, reverse=True)

    if dedup_method == "distance_legacy":
        kept = []
        for line in sorted_lines:
            duplicate = False
            for k in kept:
                if line.distance(k) < min_spacing_m * dedup_factor:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(line)
        return kept

    # Metodo moderno: overlap_parallelism com STRtree
    kept = []
    dist_thresh = min_spacing_m * dedup_factor
    overlap_tol = min(1.2, dist_thresh)

    for line in sorted_lines:
        if line is None or line.is_empty or line.length < 0.2:
            continue

        is_dup = False
        if kept:
            tree = STRtree(kept)
            cand_indices = tree.query(line.buffer(dist_thresh))
            for idx in cand_indices:
                k = kept[idx]
                d = line.distance(k)
                if d > dist_thresh:
                    continue

                overlap = compute_overlap_ratio(line, k, tolerance_m=overlap_tol)
                if overlap < min_overlap_ratio:
                    continue

                ang_diff = compute_parallelism(line, k, step_m=step_m)
                if not np.isnan(ang_diff) and ang_diff <= max_angle_deg:
                    is_dup = True
                    break

        if not is_dup:
            kept.append(line)

    return kept


def refine_line_with_prob_map(line, prob_map, transform, polygon=None, weight=0.5,
                              search_radius_m=1.5, step_m=1.0, max_shift_m=0.75):
    """
    Refina a posicao da linha usando a distribuicao continua de probabilidade da rede,
    deslocando amostras na direcao perpendicular a tangente local sem perder continuidade.
    """
    if line is None or line.is_empty or prob_map is None or transform is None:
        return line

    coords = np.asarray(line.coords)
    if len(coords) < 3:
        return line

    diffs = coords[1:] - coords[:-1]
    seg_lens = np.linalg.norm(diffs, axis=1)
    cum_dist = np.insert(np.cumsum(seg_lens), 0, 0.0)
    total_len = cum_dist[-1]
    if total_len < 2.0:
        return line

    num_samples = max(3, int(np.ceil(total_len / max(0.2, step_m))) + 1)
    target_dists = np.linspace(0, total_len, num_samples)
    sampled_pts = [line.interpolate(d) for d in target_dists]
    n_pts = len(sampled_pts)

    inv_trans = ~transform
    h_map, w_map = prob_map.shape
    new_coords = []
    prev_shift = 0.0

    # Mantem endpoints fixos para estabilidade
    new_coords.append([sampled_pts[0].x, sampled_pts[0].y])

    # Amostra offsets transversais
    offsets_m = np.linspace(-min(search_radius_m, max_shift_m), min(search_radius_m, max_shift_m), 15)

    for i in range(1, n_pts - 1):
        pt = sampled_pts[i]
        pt_prev = sampled_pts[i - 1]
        pt_next = sampled_pts[i + 1]

        # Vetor tangente
        tang = np.array([pt_next.x - pt_prev.x, pt_next.y - pt_prev.y])
        t_norm = np.linalg.norm(tang)
        if t_norm < 1e-6:
            new_coords.append([pt.x, pt.y])
            continue
        tang = tang / t_norm
        # Normal perpendicular (rotacao de 90 graus)
        normal = np.array([-tang[1], tang[0]])

        best_score = -1e9
        best_pt = [pt.x, pt.y]
        best_shift = 0.0

        for off in offsets_m:
            cand_x = pt.x + normal[0] * off
            cand_y = pt.y + normal[1] * off

            col, row = inv_trans * (cand_x, cand_y)
            col_i, row_i = int(round(col)), int(round(row))

            if 0 <= row_i < h_map and 0 <= col_i < w_map:
                prob = float(prob_map[row_i, col_i])
            else:
                prob = 0.0

            # Penalidade de deslocamento e continuidade
            dist_pen = weight * (abs(off) / max(1e-4, max_shift_m)) ** 2
            smooth_pen = 0.25 * (abs(off - prev_shift) / max(1e-4, max_shift_m)) ** 2
            score = prob - dist_pen - smooth_pen

            cand_geom = Point(cand_x, cand_y)
            if polygon is not None and not polygon.contains(cand_geom):
                score -= 2.0

            if score > best_score:
                best_score = score
                best_pt = [cand_x, cand_y]
                best_shift = off

        new_coords.append(best_pt)
        prev_shift = best_shift

    new_coords.append([sampled_pts[-1].x, sampled_pts[-1].y])

    try:
        candidate_line = LineString(new_coords)
        if not candidate_line.is_valid or candidate_line.is_empty:
            return line

        # Verifica se o alinhamento com a probabilidade realmente melhorou
        align_orig = compute_prob_alignment(line, prob_map, transform)
        align_cand = compute_prob_alignment(candidate_line, prob_map, transform)

        if align_cand["mean"] >= align_orig["mean"] - 0.02:
            return candidate_line
        else:
            return line
    except Exception:
        return line


def enforce_curvature_limit(line, max_curvature=None, step_m=1.0, preserve_endpoints=True):
    """
    Detecta e relaxa picos de curvatura excessiva sem eliminar curvaturas reais do plantio.
    """
    if line is None or line.is_empty or max_curvature is None or max_curvature <= 0:
        return line

    curv_info = compute_local_curvature(line, step_m=step_m)
    if curv_info["max"] <= max_curvature:
        return line

    coords = np.asarray(line.coords)
    if len(coords) < 3:
        return line

    # Reamostra densamente para suavizacao adequada
    dense = resample_line_by_length(coords, step_m=step_m, preserve_curvature=True)
    if len(dense) < 4:
        dense = coords

    current_line = line
    best_max_curv = curv_info["max"]

    for s_factor in [2.0, 5.0, 10.0, 20.0]:
        smooth_coords = smooth_line_arc_length(dense, smoothing=s_factor, preserve_endpoints=preserve_endpoints)
        candidate = LineString(smooth_coords)
        if candidate.is_valid and not candidate.is_empty:
            cand_curv = compute_local_curvature(candidate, step_m=step_m)
            if cand_curv["max"] < best_max_curv:
                best_max_curv = cand_curv["max"]
                current_line = candidate
                if cand_curv["max"] <= max_curvature:
                    break

    return current_line


def validate_line_corridor(line, polygon, prob_map=None, transform=None, max_width_m=None, step_m=1.0):
    """
    Valida se a linha permanece em um corredor plausivel da fileira.
    Retorna diagnostico estruturado.
    """
    if line is None or line.is_empty:
        return {"valid": False, "inside_fraction": 0.0, "max_outside_distance_m": 999.0, "reason": "empty"}

    if polygon is None or polygon.is_empty:
        return {"valid": True, "inside_fraction": 1.0, "max_outside_distance_m": 0.0, "reason": None}

    pts = [line.interpolate(d) for d in np.linspace(0, line.length, max(2, int(np.ceil(line.length / step_m))))]
    inside_count = sum(1 for p in pts if polygon.contains(p))
    inside_frac = inside_count / len(pts) if pts else 0.0

    outside_dists = [p.distance(polygon) for p in pts if not polygon.contains(p)]
    max_out_dist = max(outside_dists) if outside_dists else 0.0

    valid = (inside_frac >= 0.85) and (max_out_dist <= 2.0)
    reason = None if valid else f"inside_frac={inside_frac:.2f}, max_out={max_out_dist:.2f}m"

    prob_align = None
    if prob_map is not None and transform is not None:
        prob_align = compute_prob_alignment(line, prob_map, transform)["mean"]

    return {
        "valid": valid,
        "inside_fraction": float(inside_frac),
        "max_outside_distance_m": float(max_out_dist),
        "prob_alignment": prob_align,
        "reason": reason
    }


def validate_topology(lines, talhao_geom=None, expected_spacing_m=None, tolerance_m=1.0):
    """
    Valida a coerencia espacial e agronomica do conjunto de linhas finais.
    Identifica cruzamentos, sobreposicoes e espacamentos anomalos.
    """
    if not lines:
        return {
            "num_lines": 0,
            "num_crossings": 0,
            "num_overlaps": 0,
            "num_spacing_anomalies": 0,
            "num_outside_talhao": 0,
            "num_suspect": 0,
            "warnings": []
        }

    tree = STRtree(lines)
    num_crossings = 0
    num_overlaps = 0
    num_spacing_anomalies = 0
    num_outside = 0
    warnings = []
    suspect_indices = set()

    for i, l1 in enumerate(lines):
        if talhao_geom is not None and not talhao_geom.is_empty:
            if not talhao_geom.contains(l1):
                inter = l1.intersection(talhao_geom)
                if inter.is_empty or inter.length / max(1e-6, l1.length) < 0.8:
                    num_outside += 1
                    suspect_indices.add(i)

        neighbors = tree.query(l1)
        for j in neighbors:
            if j <= i:
                continue
            l2 = lines[j]
            if l1.crosses(l2):
                num_crossings += 1
                suspect_indices.add(i)
                suspect_indices.add(j)
                warnings.append(f"Cruzamento detectado entre linhas {i} e {j}")

            d = l1.distance(l2)
            if d < 0.8:
                ov = compute_overlap_ratio(l1, l2, tolerance_m=1.0)
                if ov > 0.6:
                    num_overlaps += 1
                    suspect_indices.add(i)
                    warnings.append(f"Sobreposicao residual entre linhas {i} e {j} (overlap={ov:.2f})")

            if expected_spacing_m is not None and expected_spacing_m > 0:
                if 0.5 < d < expected_spacing_m * 0.4:
                    num_spacing_anomalies += 1

    return {
        "num_lines": len(lines),
        "num_crossings": int(num_crossings),
        "num_overlaps": int(num_overlaps),
        "num_spacing_anomalies": int(num_spacing_anomalies),
        "num_outside_talhao": int(num_outside),
        "num_suspect": len(suspect_indices),
        "warnings": warnings[:20]
    }


def write_run_metadata(output_dir, metadata_dict):
    """
    Grava o arquivo run_metadata.json na pasta outputs para rastreabilidade de versao e parametros.
    """
    try:
        meta_path = os.path.join(output_dir, "outputs", "run_metadata.json")
        os.makedirs(os.path.dirname(meta_path), exist_ok=True)
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(metadata_dict, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[WARN] Nao foi possivel gravar run_metadata.json: {e}")


def extend_endpoints_to_talhao_boundary(lines, talhao_geom, max_dist_m=3.0, logger=print):
    """
    Estende APENAS as pontas das linhas que estao a menos de max_dist_m (2-3 metros)
    da borda do talhao, fazendo-as tocar a borda.
    Linhas no interior do talhao (matacoes, quebras, meio do talhao) NUNCA sao estendidas.
    """
    if not lines or talhao_geom is None or talhao_geom.is_empty:
        return lines

    boundary = talhao_geom.boundary
    extended_count = 0
    result = []

    for line in lines:
        coords = list(line.coords)
        if len(coords) < 2:
            result.append(line)
            continue

        new_coords = [list(c) for c in coords]
        modified = False

        # Ponta inicial
        p_start = np.array(coords[0])
        p_next = np.array(coords[1])
        dist_start = Point(p_start).distance(boundary)
        if dist_start <= max_dist_m:
            v = p_start - p_next
            v_len = np.linalg.norm(v)
            if v_len > 1e-6:
                u = v / v_len
                ray = LineString([p_start, p_start + u * (dist_start + 2.0)])
                inter = ray.intersection(boundary)
                if not inter.is_empty:
                    if inter.geom_type == 'Point':
                        new_coords[0] = [inter.x, inter.y]
                        modified = True
                    elif inter.geom_type == 'MultiPoint':
                        closest = min(inter.geoms, key=lambda pt: Point(p_start).distance(pt))
                        new_coords[0] = [closest.x, closest.y]
                        modified = True

        # Ponta final
        p_end = np.array(coords[-1])
        p_prev = np.array(coords[-2])
        dist_end = Point(p_end).distance(boundary)
        if dist_end <= max_dist_m:
            v = p_end - p_prev
            v_len = np.linalg.norm(v)
            if v_len > 1e-6:
                u = v / v_len
                ray = LineString([p_end, p_end + u * (dist_end + 2.0)])
                inter = ray.intersection(boundary)
                if not inter.is_empty:
                    if inter.geom_type == 'Point':
                        new_coords[-1] = [inter.x, inter.y]
                        modified = True
                    elif inter.geom_type == 'MultiPoint':
                        closest = min(inter.geoms, key=lambda pt: Point(p_end).distance(pt))
                        new_coords[-1] = [closest.x, closest.y]
                        modified = True

        if modified:
            try:
                new_line = LineString([tuple(c) for c in new_coords])
                if new_line.is_valid and not new_line.is_empty:
                    result.append(new_line)
                    extended_count += 1
                    continue
            except Exception:
                pass

        result.append(line)

    if logger and extended_count > 0:
        logger(f"  [Extend Bordas] {extended_count} pontas de linhas estendidas ate o limite do talhao (distancia <= {max_dist_m}m).")
    return result


def run_postprocessing(prob_map, transform, crs, talhao_geom=None, output_dir="data",
                       threshold=0.5, min_area_m2=10.0, close_size=5, open_size=3,
                       skip_postprocess=False, export_as_lines=False,
                       path_linhas=None, profile="tiro_reto",
                       calc_entrelinhas=False,
                       extend_to_boundary=None, min_row_spacing_m=4.0,
                       max_row_spacing_m=8.5, logger=print,
                       centerline_method="skeleton_v2",
                       merge_method="spatial_cost",
                       dedup_method="overlap_parallelism",
                       resample_step_m=1.0,
                       gsd_m=None,
                       prob_refinement_enabled=False,
                       prob_refinement_weight=0.5,
                       prob_search_radius_m=1.5,
                       prob_max_shift_m=0.75,
                       curvature_control_enabled=False,
                       max_curvature=None,
                       corridor_validation_enabled=False,
                       topology_validation_enabled=True,
                       write_run_metadata_file=True):
    """
    Executa o fluxo limpo, fiel e robusto de pos-processamento, thresholding, vetorizacao e extracao das linhas.
    """
    t_start = time.time()
    profile = str(profile or "tiro_reto").strip().lower()

    logger(f"Iniciando pos-processamento [Perfil: {profile.upper()}]...")
    logger(f"  Config: Centerline={centerline_method} | Merge={merge_method} | Dedup={dedup_method}")
    if prob_refinement_enabled:
        logger(f"  [Prob Refinement ATIVADO] search_radius={prob_search_radius_m}m, max_shift={prob_max_shift_m}m")
    if curvature_control_enabled:
        logger(f"  [Curvature Control ATIVADO] max_curvature={max_curvature}")
    if corridor_validation_enabled:
        logger("  [Corridor Validation ATIVADO]")
    if topology_validation_enabled:
        logger("  [Topology Validation ATIVADO]")

    os.makedirs(os.path.join(output_dir, "outputs"), exist_ok=True)

    if gsd_m is None and transform is not None:
        try:
            gsd_m = abs(float(transform[0]))
        except Exception:
            gsd_m = 0.25

    height, width = prob_map.shape

    # 1. Thresholding com os parametros da configuracao do usuario
    logger(f"  Aplicando threshold de probabilidade: {threshold:.2f}")
    binary_mask = (prob_map >= threshold).astype(np.uint8) * 255


    if not skip_postprocess:
        # 2. Fechamento Morfologico (conecta falhas pequenas na mesma linha)
        if close_size > 0:
            logger(f"  Aplicando fechamento morfologico (kernel: {close_size}x{close_size})...")
            kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel_close)

        # 3. Preenchimento de buracos internos
        logger("  Preenchendo buracos internos nas copas...")
        binary_mask = binary_fill_holes(binary_mask > 0).astype(np.uint8) * 255

        # 4. Abertura Morfologica (remove pequenos ruidos isolados)
        if open_size > 0:
            logger(f"  Aplicando abertura morfologica (kernel: {open_size}x{open_size})...")
            kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size))
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel_open)
    else:
        logger("  [MODO BRUTO] Filtros morfologicos pulados.")

    # 5. Restringe a mascara ao talhao
    if talhao_geom is not None and not talhao_geom.is_empty:
        logger("  Restringindo resultado final ao limite do talhao...")
        talhao_mask = np.zeros((height, width), dtype=np.uint8)
        rasterio.features.rasterize(
            shapes=[(talhao_geom, 255)],
            out=talhao_mask,
            transform=transform,
            fill=0,
            default_value=255
        )
        binary_mask = cv2.bitwise_and(binary_mask, talhao_mask)

    # 6. Grava GeoTIFFs de probabilidade e mascara binaria
    prob_tiff_path = os.path.join(output_dir, "outputs", "probabilidade_linhas.tif")
    with rasterio.open(
        prob_tiff_path, 'w', driver='GTiff',
        height=height, width=width, count=1,
        dtype=rasterio.float32, crs=crs, transform=transform, nodata=0
    ) as dst:
        dst.write(prob_map, 1)

    geotiff_path = os.path.join(output_dir, "outputs", "predicao_linhas.tif")
    logger(f"  Gravando GeoTIFF georreferenciado em: {geotiff_path}")
    with rasterio.open(
        geotiff_path, 'w', driver='GTiff',
        height=height, width=width, count=1,
        dtype=rasterio.uint8, crs=crs, transform=transform, nodata=0
    ) as dst:
        dst.write(binary_mask, 1)

    # 7. Polygoniza a mascara binaria
    logger("  Polygonizando a mascara binaria resultante...")
    shapes_gen = rasterio.features.shapes(binary_mask, mask=(binary_mask > 0), transform=transform)

    polygons = []
    for geom_dict, val in shapes_gen:
        if val == 255:
            poly_shape = shape(geom_dict)
            if skip_postprocess or poly_shape.area >= min_area_m2:
                polygons.append(poly_shape)

    filter_msg = "(modo bruto)" if skip_postprocess else f"(area minima: {min_area_m2}m2)"
    logger(f"  Gerados {len(polygons)} poligonos de fileiras {filter_msg}.")

    # 7a. Filtro de elongacao — descarta blocos compactos que nao sao fileiras
    if not skip_postprocess and len(polygons) > 0:
        polygons = filter_polygons_by_elongation(polygons, min_elongation=3.5, logger=logger)

    dominant_angle = None
    if len(polygons) > 0:
        dominant_angle = calculate_dominant_angle(polygons)

    # 7.1. Vetorizacao para Linhas Centrais (LineStrings)
    final_lines = []
    fail_count = 0
    corridor_fail_count = 0
    topology_report = {}

    if export_as_lines and len(polygons) > 0:
        logger(f"  Extraindo linhas centrais [{centerline_method.upper()}] de cada poligono...")

        lines = []
        for poly in polygons:
            line_geom = extract_centerline_from_polygon(
                poly,
                resolution=0.25,
                centerline_method=centerline_method,
                resample_step_m=resample_step_m
            )
            if line_geom is not None and not line_geom.is_empty:
                # Refinamento continuo pelo prob_map
                if prob_refinement_enabled:
                    line_geom = refine_line_with_prob_map(
                        line_geom,
                        prob_map=prob_map,
                        transform=transform,
                        polygon=poly,
                        weight=prob_refinement_weight,
                        search_radius_m=prob_search_radius_m,
                        step_m=resample_step_m,
                        max_shift_m=prob_max_shift_m
                    )

                # Validacao contra corredor plausivel
                if corridor_validation_enabled:
                    diag = validate_line_corridor(line_geom, poly, prob_map, transform, step_m=resample_step_m)
                    if not diag["valid"]:
                        corridor_fail_count += 1
                        continue

                lines.append(line_geom)
            else:
                fail_count += 1

        logger(f"  {len(lines)} linhas centrais extraidas dos poligonos ({fail_count} sem centerline viavel, {corridor_fail_count} rejeitadas por corredor).")

        # 7.2. Deduplicacao preliminar (antes do merge para evitar conexoes espurias)
        if len(lines) > 1:
            logger(f"  Deduplicacao preliminar [{dedup_method}]...")
            lines = deduplicate_close_lines(
                lines,
                min_spacing_m=min_row_spacing_m,
                dedup_method=dedup_method,
                step_m=resample_step_m
            )
            logger(f"  {len(lines)} linhas apos deduplicacao preliminar.")

        # 7.3. Mescla fileiras colineares (arvores mortas/gaps) preservando a curvatura
        logger(f"  Mesclando fragmentos da mesma fileira [{merge_method}]...")
        merged_lines = merge_collinear_lines(
            lines,
            max_gap_m=12.0,
            max_offset_m=1.5,
            max_angle_deg=25.0,
            merge_method=merge_method,
            active_boundary=talhao_geom
        )
        logger(f"  {len(merged_lines)} linhas apos uniao colinear.")

        # 7.4. Deduplicacao final (pos-merge)
        if len(merged_lines) > 1:
            logger(f"  Deduplicacao final [{dedup_method}]...")
            dedup_lines = deduplicate_close_lines(
                merged_lines,
                min_spacing_m=min_row_spacing_m,
                dedup_method=dedup_method,
                step_m=resample_step_m
            )
            logger(f"  {len(dedup_lines)} linhas apos deduplicacao final.")
        else:
            dedup_lines = merged_lines

        # 7.5. Extensao APENAS das pontas proximas a borda (2-3 metros)
        # NUNCA estende linhas no meio do talhao (evita retas infinitas)
        should_extend = True if extend_to_boundary is None else bool(extend_to_boundary)
        if should_extend and talhao_geom is not None:
            logger("  Estendendo pontas proximas a borda (ultimos 2-3m) ate o talhao...")
            final_lines = extend_endpoints_to_talhao_boundary(dedup_lines, talhao_geom, max_dist_m=3.0, logger=logger)
        else:
            final_lines = dedup_lines

        # 7.6. Suavizacao Gaussiana final suave
        smoothed_lines = []
        for ln in final_lines:
            smoothed = smooth_linestring_gaussian(ln, sigma=1.0)
            smoothed_lines.append(smoothed if (smoothed is not None and not smoothed.is_empty) else ln)
        final_lines = smoothed_lines

        # 7.7. Controle de Curvatura (Anti-zig-zag)
        if curvature_control_enabled and max_curvature is not None:
            logger(f"  Aplicando controle de curvatura maxima ({max_curvature:.4f})...")
            final_lines = [enforce_curvature_limit(ln, max_curvature=max_curvature, step_m=resample_step_m) for ln in final_lines]

        # 7.8. Validacao Topologica do conjunto final
        if topology_validation_enabled and len(final_lines) > 0:
            topology_report = validate_topology(
                final_lines,
                talhao_geom=talhao_geom,
                expected_spacing_m=min_row_spacing_m
            )
            logger(f"  [Topologia] Cruzamentos: {topology_report['num_crossings']} | Sobreposicoes: {topology_report['num_overlaps']} | Suspeitas: {topology_report['num_suspect']}")

        features_to_export = final_lines
        logger(f"  Total de {len(final_lines)} linhas centrais geradas com sucesso.")
    else:
        features_to_export = polygons

    # Monta GeoDataFrame de saida
    if len(features_to_export) == 0:
        logger("  [Aviso] Nenhuma feicao resultante apos o processamento!")
        gdf_wgs84 = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    else:
        gdf_utm = gpd.GeoDataFrame(geometry=features_to_export, crs=crs)
        gdf_utm['area_m2'] = gdf_utm.geometry.area
        gdf_utm['length_m'] = gdf_utm.geometry.length
        logger("  Reprojetando resultados para WGS 84 (EPSG:4326)...")
        gdf_wgs84 = gdf_utm.to_crs("EPSG:4326")

    # 8. Exporta vetores
    geojson_path = os.path.join(output_dir, "outputs", "predicao_linhas.geojson")
    shp_path = os.path.join(output_dir, "outputs", "shp", "predicao_linhas.shp")

    logger(f"  Salvando GeoJSON em: {geojson_path}")
    try:
        gdf_wgs84.to_file(geojson_path, driver="GeoJSON")
    except Exception as e:
        logger(f"  [WARN] Falha ao salvar via geopandas: {e}. Usando fallback...")
        with open(geojson_path, 'w', encoding='utf-8') as f:
            f.write(gdf_wgs84.to_json())

    logger(f"  Salvando Shapefile em: {shp_path}")
    os.makedirs(os.path.dirname(shp_path), exist_ok=True)
    if not gdf_wgs84.empty:
        try:
            gdf_wgs84.to_file(shp_path)
        except Exception as e:
            logger(f"  [WARN] Falha ao salvar Shapefile: {e}")

    # Preview
    preview_img_path = os.path.join(output_dir, "debug", "predicao_preview.png")
    os.makedirs(os.path.dirname(preview_img_path), exist_ok=True)
    cv2.imwrite(preview_img_path, binary_mask)

    elapsed_sec = time.time() - t_start
    if write_run_metadata_file:
        prob_align_mean = None
        curv_p95 = None
        if len(final_lines) > 0 and export_as_lines:
            try:
                sample_ln = max(final_lines, key=lambda l: l.length)
                prob_align_mean = compute_prob_alignment(sample_ln, prob_map, transform)["mean"]
                curv_p95 = compute_local_curvature(sample_ln, step_m=resample_step_m)["p95"]
            except Exception:
                pass

        metadata = {
            "algorithm_version": "centerline-v2.1-prod",
            "timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z",
            "centerline_method": centerline_method,
            "merge_method": merge_method,
            "dedup_method": dedup_method,
            "profile": profile,
            "parameters": {
                "threshold": threshold,
                "min_area_m2": min_area_m2,
                "close_size": close_size,
                "open_size": open_size,
                "min_row_spacing_m": min_row_spacing_m,
                "max_row_spacing_m": max_row_spacing_m,
                "resample_step_m": resample_step_m,
                "gsd_m": gsd_m,
                "prob_refinement_enabled": prob_refinement_enabled,
                "prob_refinement_weight": prob_refinement_weight,
                "prob_search_radius_m": prob_search_radius_m,
                "prob_max_shift_m": prob_max_shift_m,
                "curvature_control_enabled": curvature_control_enabled,
                "max_curvature": max_curvature,
                "corridor_validation_enabled": corridor_validation_enabled,
                "topology_validation_enabled": topology_validation_enabled
            },
            "counts": {
                "num_polygons": len(polygons),
                "num_centerlines": len(final_lines) if export_as_lines else 0,
                "num_centerline_failures": fail_count if export_as_lines else 0,
                "topology_warnings": len(topology_report.get("warnings", [])) if topology_validation_enabled else 0,
                "topology_crossings": topology_report.get("num_crossings", 0) if topology_validation_enabled else 0
            },
            "metrics": {
                "prob_alignment_mean": prob_align_mean,
                "curvature_p95": curv_p95
            },
            "performance": {
                "elapsed_seconds": round(elapsed_sec, 2)
            }
        }
        write_run_metadata(output_dir, metadata)

    logger(f"Pos-processamento concluido com sucesso [Perfil: {profile}] ({elapsed_sec:.2f}s).")

    results = {
        "profile": profile,
        "num_polygons": len(features_to_export),
        "geotiff": geotiff_path,
        "geojson": geojson_path,
        "shapefile": shp_path if not gdf_wgs84.empty else None,
        "topology_report": topology_report
    }

    if dominant_angle is not None:
        results["dominant_angle_deg"] = float(np.degrees(dominant_angle))
        results["dominant_azimuth_deg"] = float((90.0 - np.degrees(dominant_angle)) % 180)

    return results


if __name__ == '__main__':
    pass
