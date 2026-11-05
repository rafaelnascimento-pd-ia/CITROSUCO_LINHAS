# -*- coding: utf-8 -*-
"""
Módulo de avaliação quantitativa e métricas geométricas para linhas de plantio em citros.
Projetado para operar em coordenadas métricas (CRS projetado, ex.: UTM).
"""

import numpy as np
from shapely.geometry import LineString, Point


def _sample_line_by_length(line, step_m=1.0):
    """
    Amostra pontos ao longo de uma LineString em intervalos regulares de comprimento acumulado.
    """
    if line is None or line.is_empty:
        return []
    total_len = line.length
    if total_len <= 0:
        coords = list(line.coords)
        return [Point(coords[0])] if coords else []
    
    num_steps = max(1, int(np.ceil(total_len / step_m)))
    distances = np.linspace(0, total_len, num_steps + 1)
    points = [line.interpolate(dist) for dist in distances]
    return points


def _sample_tangents_by_length(line, step_m=1.0, window_m=1.0):
    """
    Amostra vetores tangentes normalizados ao longo de uma LineString em intervalos step_m.
    """
    if line is None or line.is_empty:
        return [], []
    total_len = line.length
    if total_len <= 0:
        return [], []
    
    num_steps = max(1, int(np.ceil(total_len / step_m)))
    distances = np.linspace(0, total_len, num_steps + 1)
    tangents = []
    points = []
    
    delta = min(window_m, total_len / 2.0) if total_len > 0 else 0.5
    for dist in distances:
        p_prev = line.interpolate(max(0.0, dist - delta))
        p_next = line.interpolate(min(total_len, dist + delta))
        v = np.array([p_next.x - p_prev.x, p_next.y - p_prev.y])
        norm = np.linalg.norm(v)
        if norm > 1e-7:
            tangents.append(v / norm)
            points.append(line.interpolate(dist))
            
    return points, tangents


def compute_center_error(line_pred, line_ref, step_m=1.0):
    """
    Calcula o erro médio e o percentil 95 da distância entre amostras de line_pred
    até a linha de referência line_ref (em metros).
    """
    if line_pred is None or line_ref is None or line_pred.is_empty or line_ref.is_empty:
        return {"mean_error_m": float("nan"), "p95_error_m": float("nan"), "max_error_m": float("nan")}
    
    pts = _sample_line_by_length(line_pred, step_m=step_m)
    if not pts:
        return {"mean_error_m": 0.0, "p95_error_m": 0.0, "max_error_m": 0.0}
    
    distances = [pt.distance(line_ref) for pt in pts]
    return {
        "mean_error_m": float(np.mean(distances)),
        "p95_error_m": float(np.percentile(distances, 95)),
        "max_error_m": float(np.max(distances))
    }


def compute_length_coverage(line_pred, line_ref, tolerance_m=1.0):
    """
    Calcula a fração do comprimento da referência line_ref coberta pela predição line_pred
    dentro de uma tolerância métrica (buffer).
    """
    if line_pred is None or line_ref is None or line_pred.is_empty or line_ref.is_empty:
        return 0.0
    ref_len = line_ref.length
    if ref_len <= 1e-6:
        return 0.0
    
    buf = line_pred.buffer(tolerance_m)
    intersected = line_ref.intersection(buf)
    if intersected.is_empty:
        return 0.0
    
    covered_len = intersected.length
    return float(min(1.0, covered_len / ref_len))


def compute_endpoint_error(line_pred, line_ref):
    """
    Calcula a distância entre endpoints correspondentes após alinhar as direções das duas linhas.
    Retorna a média dos erros nos dois endpoints (start e end).
    """
    if line_pred is None or line_ref is None or line_pred.is_empty or line_ref.is_empty:
        return {"mean_endpoint_error_m": float("nan"), "start_error_m": float("nan"), "end_error_m": float("nan")}
    
    p_coords = list(line_pred.coords)
    r_coords = list(line_ref.coords)
    if len(p_coords) < 2 or len(r_coords) < 2:
        return {"mean_endpoint_error_m": 0.0, "start_error_m": 0.0, "end_error_m": 0.0}
    
    p_start, p_end = np.array(p_coords[0][:2]), np.array(p_coords[-1][:2])
    r_start, r_end = np.array(r_coords[0][:2]), np.array(r_coords[-1][:2])
    
    # Orientação direta
    d_direct = np.linalg.norm(p_start - r_start) + np.linalg.norm(p_end - r_end)
    # Orientação invertida
    d_reverse = np.linalg.norm(p_start - r_end) + np.linalg.norm(p_end - r_start)
    
    if d_direct <= d_reverse:
        e_start = float(np.linalg.norm(p_start - r_start))
        e_end = float(np.linalg.norm(p_end - r_end))
    else:
        e_start = float(np.linalg.norm(p_start - r_end))
        e_end = float(np.linalg.norm(p_end - r_start))
        
    return {
        "mean_endpoint_error_m": float((e_start + e_end) / 2.0),
        "start_error_m": e_start,
        "end_error_m": e_end
    }


def compute_parallelism(line1, line2, step_m=1.0):
    """
    Calcula a diferença angular média (em graus) entre tangentes locais amostradas.
    Retorna ângulo médio entre 0° (perfeitamente paralelo) e 90° (ortogonal).
    """
    if line1 is None or line2 is None or line1.is_empty or line2.is_empty:
        return float("nan")
    
    pts1, tang1 = _sample_tangents_by_length(line1, step_m=step_m)
    pts2, tang2 = _sample_tangents_by_length(line2, step_m=step_m)
    
    if not tang1 or not tang2:
        return 0.0
    
    angles_deg = []
    # Para cada ponto em line1, encontra a tangente mais próxima em line2
    for p1, t1 in zip(pts1, tang1):
        # Encontra o ponto e tangente mais próximos em line2
        dists = [p1.distance(p2) for p2 in pts2]
        closest_idx = int(np.argmin(dists))
        t2 = tang2[closest_idx]
        
        # Produto escalar absoluto (paralelismo independente do sentido da linha)
        cos_sim = abs(float(np.dot(t1, t2)))
        cos_sim = min(1.0, max(0.0, cos_sim))
        ang = np.degrees(np.arccos(cos_sim))
        angles_deg.append(ang)
        
    return float(np.mean(angles_deg)) if angles_deg else 0.0


def compute_overlap_ratio(line1, line2, tolerance_m=1.0):
    """
    Calcula a fração do comprimento de line1 contida no buffer de line2.
    O denominador é sempre o comprimento da própria line1.
    """
    if line1 is None or line2 is None or line1.is_empty or line2.is_empty:
        return 0.0
    l1_len = line1.length
    if l1_len <= 1e-6:
        return 0.0
    
    buf = line2.buffer(tolerance_m)
    inter = line1.intersection(buf)
    if inter.is_empty:
        return 0.0
    return float(min(1.0, inter.length / l1_len))


def compute_prob_alignment(line, prob_map, transform, buffer_m=0.5, step_m=1.0, threshold=0.35):
    """
    Mede o alinhamento da linha com as regioes de alta probabilidade do prob_map.
    Amostra pontos ao longo da linha por comprimento acumulado e converte para coordenadas raster.
    """
    if line is None or line.is_empty or prob_map is None or transform is None:
        return {
            "mean": 0.0,
            "median": 0.0,
            "p10": 0.0,
            "p25": 0.0,
            "fraction_above_threshold": 0.0
        }

    pts = _sample_line_by_length(line, step_m=step_m)
    if not pts:
        return {
            "mean": 0.0,
            "median": 0.0,
            "p10": 0.0,
            "p25": 0.0,
            "fraction_above_threshold": 0.0
        }

    inv_trans = ~transform
    height, width = prob_map.shape
    values = []

    for pt in pts:
        col, row = inv_trans * (pt.x, pt.y)
        col_i, row_i = int(round(col)), int(round(row))
        if 0 <= row_i < height and 0 <= col_i < width:
            values.append(float(prob_map[row_i, col_i]))
        else:
            values.append(0.0)

    if not values:
        return {
            "mean": 0.0,
            "mean_prob": 0.0,
            "median": 0.0,
            "p10": 0.0,
            "p25": 0.0,
            "fraction_above_threshold": 0.0
        }

    v_arr = np.array(values)
    mean_val = float(np.mean(v_arr))
    return {
        "mean": mean_val,
        "mean_prob": mean_val,
        "median": float(np.median(v_arr)),
        "p10": float(np.percentile(v_arr, 10)),
        "p25": float(np.percentile(v_arr, 25)),
        "fraction_above_threshold": float(np.mean(v_arr >= threshold))
    }


def compute_local_curvature(line, step_m=1.0):
    """
    Calcula a curvatura local (em 1/metro) ao longo de uma linha reamostrada por comprimento acumulado.
    Calcula a taxa de variacao angular por comprimento de arco d_theta/ds e conta inversoes de sinal (zig-zag).
    """
    if line is None or line.is_empty or line.length < 2.0 * step_m:
        return {
            "mean": 0.0,
            "mean_curvature": 0.0,
            "median": 0.0,
            "p95": 0.0,
            "max": 0.0,
            "max_curvature": 0.0,
            "num_inversions": 0,
            "curvatures": []
        }

    pts = _sample_line_by_length(line, step_m=step_m)
    if len(pts) < 3:
        return {
            "mean": 0.0,
            "mean_curvature": 0.0,
            "median": 0.0,
            "p95": 0.0,
            "max": 0.0,
            "max_curvature": 0.0,
            "num_inversions": 0,
            "curvatures": []
        }

    coords = np.array([[p.x, p.y] for p in pts])
    diffs = coords[1:] - coords[:-1]
    ds = np.linalg.norm(diffs, axis=1)

    curvatures = []
    cross_signs = []

    for i in range(len(diffs) - 1):
        v1 = diffs[i]
        v2 = diffs[i + 1]
        norm1 = ds[i]
        norm2 = ds[i + 1]
        if norm1 < 1e-6 or norm2 < 1e-6:
            continue

        cos_a = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)
        d_theta = np.arccos(cos_a)
        avg_ds = (norm1 + norm2) / 2.0
        kappa = d_theta / max(0.1, avg_ds)
        curvatures.append(kappa)

        cross_val = v1[0] * v2[1] - v1[1] * v2[0]
        if abs(cross_val) > 1e-4:
            cross_signs.append(1 if cross_val > 0 else -1)

    if not curvatures:
        return {
            "mean": 0.0,
            "mean_curvature": 0.0,
            "median": 0.0,
            "p95": 0.0,
            "max": 0.0,
            "max_curvature": 0.0,
            "num_inversions": 0,
            "curvatures": []
        }

    num_inversions = 0
    for i in range(len(cross_signs) - 1):
        if cross_signs[i] != cross_signs[i + 1]:
            num_inversions += 1

    curv_arr = np.array(curvatures)
    mean_c = float(np.mean(curv_arr))
    max_c = float(np.max(curv_arr))

    return {
        "mean": mean_c,
        "mean_curvature": mean_c,
        "median": float(np.median(curv_arr)),
        "p95": float(np.percentile(curv_arr, 95)),
        "max": max_c,
        "max_curvature": max_c,
        "num_inversions": int(num_inversions),
        "curvatures": [float(c) for c in curvatures]
    }
