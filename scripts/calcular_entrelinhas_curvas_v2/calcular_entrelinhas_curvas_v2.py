#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
calcular_entrelinhas_curvas_v2.py
--------------------------------------------------------------------------------
Geração de entrelinhas de alta precisão para fileiras de plantio curvas e 
em curvas de nível, utilizando Projeção Paramétrica Ortogonal Monotônica.

Correções implementadas:
  1. Curvas fechadas (U-turn): amostragem densa (0.5m) que acompanha 100% o
     ápice redondo da curva, sem cortar cantos retos ou "quadrados".
  2. Eliminação de linhas diagonais/cruzadas: filtro de paralelismo vetorial
     (rejeita fileiras não-paralelas em carreadores e estradas).
  3. Remoção de fragmentos desnecessários: filtro de comprimento mínimo
     (--min_length padrão 15m) e quebra de continuidade em lacunas.
  4. Suporte nativo a matações (pontas de rua e fileiras em cunha).
--------------------------------------------------------------------------------
"""

import os
import sys
import argparse
import datetime
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, LineString, MultiLineString
from shapely.strtree import STRtree


def get_tangent(line, dist, window=0.8):
    """Calcula o vetor tangente unitário na distância `dist` ao longo da linha."""
    line_len = line.length
    d0 = max(0.0, dist - window)
    d1 = min(line_len, dist + window)
    if d1 - d0 < 1e-4:
        return None
    p0 = line.interpolate(d0)
    p1 = line.interpolate(d1)
    dx = p1.x - p0.x
    dy = p1.y - p0.y
    norm = np.sqrt(dx**2 + dy**2)
    if norm < 1e-6:
        return None
    return np.array([dx / norm, dy / norm])


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

    p0 = np.array(coords[0][:2])
    p1 = np.array(coords[1][:2])
    v_start = p0 - p1
    len_start = np.linalg.norm(v_start)
    if len_start > 0:
        u_start = v_start / len_start
        p_start_extended = p0 + ext_length * u_start
    else:
        p_start_extended = p0

    pn = np.array(coords[-1][:2])
    pn1 = np.array(coords[-2][:2])
    v_end = pn - pn1
    len_end = np.linalg.norm(v_end)
    if len_end > 0:
        u_end = v_end / len_end
        p_end_extended = pn + ext_length * u_end
    else:
        p_end_extended = pn

    extended_coords = [tuple(p_start_extended)] + [(c[0], c[1]) for c in coords] + [tuple(p_end_extended)]
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


def compute_smooth_inter_row(line_a, line_b, min_spacing=3.5, max_spacing=8.5, sample_step=0.5, min_parallel_cos=0.72, min_length=15.0):
    """
    Gera a linha média (entrelinha) perfeitamente suave entre duas fileiras adjacentes.
    Usa amostragem densa para curvas fechadas e filtro de paralelismo para evitar diagonais.
    """
    # Avalia a partir da linha de referência mais longa
    candidates = []

    for base_line, target_line in [(line_a, line_b), (line_b, line_a)]:
        line_len = base_line.length
        if line_len < min_length:
            continue

        num_samples = int(np.ceil(line_len / sample_step)) + 1
        num_samples = max(10, num_samples)

        sampled_segments = []
        current_pts = []
        current_dists = []

        for dist in np.linspace(0.0, line_len, num_samples):
            P = base_line.interpolate(dist)

            # Projeta P na linha vizinha
            target_dist = target_line.project(P)
            Q = target_line.interpolate(target_dist)

            v = np.array([Q.x - P.x, Q.y - P.y])
            d = np.linalg.norm(v)

            # Valida distância lateral
            if d < min_spacing or d > max_spacing:
                if current_pts:
                    if len(current_pts) >= 4:
                        sampled_segments.append((current_pts, current_dists))
                    current_pts = []
                    current_dists = []
                continue

            # Filtro de paralelismo: descarta linhas cruzadas/diagonais em carreadores
            u_base = get_tangent(base_line, dist, window=0.8)
            u_target = get_tangent(target_line, target_dist, window=0.8)

            if u_base is not None and u_target is not None:
                parallel_cos = abs(np.dot(u_base, u_target))
                if parallel_cos < min_parallel_cos:
                    if current_pts:
                        if len(current_pts) >= 4:
                            sampled_segments.append((current_pts, current_dists))
                        current_pts = []
                        current_dists = []
                    continue

            M = ((P.x + Q.x) / 2.0, (P.y + Q.y) / 2.0)

            # Evita pontes sobre lacunas (ex: travessia de carreadores)
            if current_pts:
                gap = np.hypot(M[0] - current_pts[-1][0], M[1] - current_pts[-1][1])
                if gap > sample_step * 3.5:
                    if len(current_pts) >= 4:
                        sampled_segments.append((current_pts, current_dists))
                    current_pts = []
                    current_dists = []

            current_pts.append(M)
            current_dists.append(d)

        if current_pts and len(current_pts) >= 4:
            sampled_segments.append((current_pts, current_dists))

        # Constrói segmentos válidos que respeitam o comprimento mínimo
        valid_for_this_dir = []
        for seg_pts, seg_dists in sampled_segments:
            if len(seg_pts) >= 4:
                ls = LineString(seg_pts)
                if ls.length >= min_length:
                    valid_for_this_dir.append((ls, seg_dists, base_line.length, target_line.length))

        if valid_for_this_dir:
            total_len = sum(item[0].length for item in valid_for_this_dir)
            candidates.append((total_len, valid_for_this_dir))

    if not candidates:
        return []

    # Retorna a direção que cobriu a maior extensão longitudinal
    candidates.sort(key=lambda c: c[0], reverse=True)
    return candidates[0][1]


def main():
    parser = argparse.ArgumentParser(
        description="Gera entrelinhas suaves de alta fidelidade para talhões em curvas de nível (Citrosrows AI v2).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--input_lines", "-i", required=True,
        help="Caminho do arquivo de linhas de plantio (GeoJSON ou Shapefile)."
    )
    parser.add_argument(
        "--talhao_path", "-t", default=None,
        help="Caminho do limite do talhão (GeoJSON ou Shapefile) para recorte nos carreadores."
    )
    parser.add_argument(
        "--output_dir", "-o", default="data/outputs/entrelinhas_curvas_v2",
        help="Diretório de saída onde os arquivos GeoJSON e Shapefile serão salvos."
    )
    parser.add_argument(
        "--min_row_spacing", type=float, default=3.5,
        help="Espaçamento lateral mínimo em metros aceito entre fileiras vizinhas."
    )
    parser.add_argument(
        "--max_row_spacing", type=float, default=8.5,
        help="Espaçamento lateral máximo em metros aceito entre fileiras vizinhas."
    )
    parser.add_argument(
        "--sample_step", type=float, default=0.5,
        help="Passo de amostragem ao longo das curvas (em metros). Valores baixos (0.5m) garantem curvas 100% redondas no ápice de U-turns."
    )
    parser.add_argument(
        "--min_length", type=float, default=15.0,
        help="Comprimento mínimo em metros para manter uma entrelinha (descarta pequenos fragmentos desnecessários)."
    )
    parser.add_argument(
        "--min_probe_hits", type=int, default=3,
        help="Número mínimo de sondagens mútuas para confirmar pareamento de fileiras vizinhas."
    )

    args = parser.parse_args()

    print("=" * 68)
    print("      CITROSROWS AI - CÁLCULO DE ENTRELINHAS CURVAS V2 (SUAVES)")
    print("=" * 68)
    print(f"Linhas de Entrada: {args.input_lines}")
    print(f"Limite do Talhão : {args.talhao_path or 'Não fornecido'}")
    print(f"Limiares de Busca: Mínimo = {args.min_row_spacing}m | Máximo = {args.max_row_spacing}m")
    print(f"Passo Amostragem : {args.sample_step}m | Comprimento Mín: {args.min_length}m")
    print("-" * 68)

    if not os.path.exists(args.input_lines):
        print(f"[ERRO] Arquivo de entrada não encontrado: {args.input_lines}")
        sys.exit(1)

    try:
        gdf_lines = gpd.read_file(args.input_lines)
    except Exception as e:
        print(f"[ERRO] Falha ao abrir arquivo de linhas: {e}")
        sys.exit(1)

    if gdf_lines.empty:
        print("[ERRO] Arquivo de linhas está vazio.")
        sys.exit(0)

    original_crs = gdf_lines.crs
    print(f"[OK] {len(gdf_lines)} feições de linhas carregadas.")

    # 1. Reprojeção automática para sistema métrico (UTM)
    if original_crs is not None and original_crs.is_geographic:
        try:
            utm_crs = gdf_lines.estimate_utm_crs()
            gdf_lines = gdf_lines.to_crs(utm_crs)
            print(f"[INFO] Dados geográficos detectados. Reprojetando para UTM local: {utm_crs.to_string()}")
        except Exception as e:
            gdf_lines = gdf_lines.to_crs("EPSG:3857")
            print(f"[WARN] Reprojetando para EPSG:3857 (Web Mercator): {e}")
    elif original_crs is not None:
        print(f"[INFO] Dados já em formato projetado/métrico: {original_crs.to_string()}")

    # 2. Explodir MultiLineStrings em LineStrings individuais
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
        print("[ERRO] Nenhuma LineString válida para processamento.")
        sys.exit(1)

    print(f"[OK] Total de {len(lines)} fileiras (LineStrings) individuais para processamento.")

    # 3. Particionamento por talhão (se fornecido)
    blocks_to_process = []
    if args.talhao_path and os.path.exists(args.talhao_path):
        try:
            gdf_talhao = gpd.read_file(args.talhao_path)
            if not gdf_talhao.empty:
                gdf_talhao = gdf_talhao.to_crs(gdf_lines.crs)
                for idx, row in gdf_talhao.iterrows():
                    geom_t = row.geometry
                    if geom_t is not None and not geom_t.is_empty:
                        blocks_to_process.append((f"Talhão {idx+1}", geom_t))
            else:
                print("[WARN] Arquivo de talhão vazio. Processando globalmente.")
        except Exception as e:
            print(f"[WARN] Falha ao ler limites do talhão ({e}). Processando globalmente.")

    if not blocks_to_process:
        print("[INFO] Processando todas as fileiras como um único bloco global.")
        blocks_to_process.append(("Global", None))

    all_inter_rows = []
    all_attributes = {
        'dist_mean': [],
        'dist_min': [],
        'dist_max': [],
        'row_len_1': [],
        'row_len_2': []
    }

    # 4. Processamento de cada bloco
    for block_name, block_geom in blocks_to_process:
        if block_geom is not None:
            block_lines = [l for l in lines if l.intersects(block_geom)]
            print(f"\n--- Processando {block_name} ({len(block_lines)} fileiras) ---")
        else:
            block_lines = lines
            print(f"\n--- Processando Bloco Global ({len(block_lines)} fileiras) ---")

        if len(block_lines) < 2:
            print(f"  [INFO] Poucas fileiras ({len(block_lines)}) neste bloco. Entrelinhas ignoradas.")
            continue

        tree = STRtree(block_lines)

        # 4.1 Identificação robusta de pares vizinhos laterais com filtro de paralelismo
        pair_scores = {}
        for i, line_i in enumerate(block_lines):
            line_len = line_i.length
            if line_len < args.min_length:
                continue

            num_samples = max(5, int(np.ceil(line_len / 1.5)))
            for dist in np.linspace(0.1, line_len - 0.1, num_samples):
                P = line_i.interpolate(dist)
                u_tangent = get_tangent(line_i, dist, window=0.8)
                if u_tangent is None:
                    continue

                u_normal = np.array([-u_tangent[1], u_tangent[0]])

                p_start = Point(P.x - args.max_row_spacing * u_normal[0], P.y - args.max_row_spacing * u_normal[1])
                p_end   = Point(P.x + args.max_row_spacing * u_normal[0], P.y + args.max_row_spacing * u_normal[1])
                probe = LineString([p_start, p_end])

                hits = tree.query(probe, predicate="intersects")
                closest_left = None
                min_d_left = float('inf')
                closest_right = None
                min_d_right = float('inf')

                for j in hits:
                    if j == i:
                        continue
                    line_j = block_lines[j]
                    sect = probe.intersection(line_j)
                    if sect.is_empty:
                        continue

                    pts = []
                    if sect.geom_type == 'Point':
                        pts = [sect]
                    elif sect.geom_type == 'MultiPoint':
                        pts = list(sect.geoms)
                    elif sect.geom_type in ('LineString', 'MultiLineString'):
                        coords_s = sect.coords if sect.geom_type == 'LineString' else [c for sg in sect.geoms for c in sg.coords]
                        pts = [Point(c[0], c[1]) for c in coords_s]

                    for pt in pts:
                        d = P.distance(pt)
                        if d < args.min_row_spacing or d > args.max_row_spacing:
                            continue

                        # Verifica paralelismo na interseção para descartar diagonais em carreadores
                        j_dist = line_j.project(pt)
                        u_j = get_tangent(line_j, j_dist, window=0.8)
                        if u_j is not None and abs(np.dot(u_tangent, u_j)) < 0.70:
                            continue

                        w = np.array([pt.x - P.x, pt.y - P.y])
                        proj = np.dot(w, u_normal)

                        if proj > 0 and d < min_d_right:
                            min_d_right = d
                            closest_right = j
                        elif proj < 0 and d < min_d_left:
                            min_d_left = d
                            closest_left = j

                for neighbor in [closest_left, closest_right]:
                    if neighbor is not None:
                        pair = (min(i, neighbor), max(i, neighbor))
                        pair_scores[pair] = pair_scores.get(pair, 0) + 1

        print(f"  Pares vizinhos candidatos identificados: {len(pair_scores)}")

        # 4.2 Geração contínua e suave das entrelinhas
        block_inter_rows = []
        for (i, j), score in pair_scores.items():
            if score < args.min_probe_hits:
                continue

            segs = compute_smooth_inter_row(
                block_lines[i],
                block_lines[j],
                min_spacing=args.min_row_spacing,
                max_spacing=args.max_row_spacing,
                sample_step=args.sample_step,
                min_parallel_cos=0.72,
                min_length=args.min_length
            )

            for midline, dists, l1, l2 in segs:
                if block_geom is not None:
                    ext = extend_line_to_boundary(midline, block_geom)
                    if ext is not None and not ext.is_empty:
                        midline = ext

                block_inter_rows.append((midline, dists, l1, l2))

        print(f"  Entrelinhas curvas geradas com sucesso: {len(block_inter_rows)}")

        for geom_mid, dist_arr, l1_len, l2_len in block_inter_rows:
            all_inter_rows.append(geom_mid)
            all_attributes['dist_mean'].append(float(np.mean(dist_arr)))
            all_attributes['dist_min'].append(float(np.min(dist_arr)))
            all_attributes['dist_max'].append(float(np.max(dist_arr)))
            all_attributes['row_len_1'].append(l1_len)
            all_attributes['row_len_2'].append(l2_len)

    print("\n" + "=" * 68)
    print(f"[OK] Processamento concluído com sucesso!")
    print(f"  - Entrelinhas curvas totais geradas: {len(all_inter_rows)}")

    if not all_inter_rows:
        print("[WARN] Nenhuma entrelinha gerada com os parâmetros informados.")
        sys.exit(0)

    # 5. Criar GeoDataFrame e retornar ao CRS original
    gdf_out = gpd.GeoDataFrame(all_attributes, geometry=all_inter_rows, crs=gdf_lines.crs)
    gdf_out['length_m'] = gdf_out.geometry.length

    if original_crs is not None:
        print(f"[INFO] Reprojetando resultados para CRS original: {original_crs.to_string()}")
        gdf_out = gdf_out.to_crs(original_crs)

    # 6. Salvar arquivos com backup de timestamp
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    geojson_out = os.path.join(args.output_dir, "predicao_entrelinhas.geojson")
    geojson_ts  = os.path.join(args.output_dir, f"predicao_entrelinhas_{timestamp}.geojson")
    shp_dir     = os.path.join(args.output_dir, "shp")
    shp_out     = os.path.join(shp_dir, "predicao_entrelinhas.shp")
    shp_ts      = os.path.join(shp_dir, f"predicao_entrelinhas_{timestamp}.shp")

    print("-" * 68)
    print(f"Salvando resultados em: {args.output_dir}")

    # Salva GeoJSON principal
    try:
        gdf_out.to_file(geojson_out, driver="GeoJSON")
        print(f"[OK] GeoJSON salvo: {geojson_out}")
    except Exception as e:
        print(f"[WARN] Salvando GeoJSON via fallback ({e})...")
        try:
            with open(geojson_out, 'w', encoding='utf-8') as f:
                f.write(gdf_out.to_json())
            print(f"[OK] GeoJSON principal salvo com sucesso.")
        except Exception as ex:
            print(f"[ERRO] Falha ao salvar GeoJSON principal: {ex}")

    # Salva GeoJSON com timestamp
    try:
        with open(geojson_ts, 'w', encoding='utf-8') as f:
            f.write(gdf_out.to_json())
        print(f"[OK] GeoJSON (timestamp) salvo: {geojson_ts}")
    except Exception as e:
        print(f"[WARN] Falha ao salvar GeoJSON timestamp: {e}")

    # Salva Shapefiles
    try:
        os.makedirs(shp_dir, exist_ok=True)
        gdf_out.to_file(shp_out)
        gdf_out.to_file(shp_ts)
        print(f"[OK] Shapefile salvo: {shp_out}")
        print(f"[OK] Shapefile (timestamp) salvo: {shp_ts}")
    except Exception as e:
        print(f"[WARN] Falha ao salvar Shapefiles: {e}")

    print("=" * 68)
    print("      CÁLCULO DE ENTRELINHAS CURVAS V2 CONCLUÍDO COM SUCESSO!")
    print("=" * 68)


if __name__ == "__main__":
    main()
