# -*- coding: utf-8 -*-

"""
calcular_entrelinhas_curvas_robusto.py
-------------------------------------------------------------------------------
Gera entrelinhas em fileiras curvas com controles de continuidade e segurança.

A rotina foi desenhada para dados em que:
- as fileiras podem ter lacunas ou estar divididas em vários trechos;
- linhas próximas podem cruzar, mudar de direção ou ficar momentaneamente
  mais próximas umas das outras;
- a entrelinha não deve ser desenhada atravessando uma lacuna nem saltando de
  uma fileira para outra.

Regras principais:
1. A fileira mais longa do par é sempre a referência.
2. O vizinho precisa estar na normal local e ter direção compatível.
3. O vizinho escolhido é mantido enquanto sua distância variar de forma
   plausível; uma troca de vizinho inicia outro par, nunca uma continuação.
4. Lacunas maiores que --max_bridge_gap dividem a saída em feições separadas.
5. Nenhuma simplificação é aplicada à curva reconstruída.
-------------------------------------------------------------------------------
"""

import argparse
import os
import sys

import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, Point
from shapely.strtree import STRtree


EPS = 1e-12


def as_index(value, geometries):
    """Converte o retorno do STRtree em índice, compatível com Shapely 1 e 2."""
    if isinstance(value, (int, np.integer)):
        return int(value)

    # Shapely 1.x pode retornar a própria geometria no STRtree.
    for idx, geom in enumerate(geometries):
        if value is geom or value.equals(geom):
            return idx
    return None


def geometry_points(geom):
    """Extrai os pontos relevantes de uma interseção geométrica."""
    if geom is None or geom.is_empty:
        return []

    if geom.geom_type == "Point":
        return [geom]
    if geom.geom_type == "MultiPoint":
        return [part for part in geom.geoms if not part.is_empty]
    if geom.geom_type == "LineString":
        coords = list(geom.coords)
        if len(coords) == 1:
            return [Point(coords[0])]
        if len(coords) >= 2:
            return [Point(coords[0]), Point(coords[-1])]
        return []
    if hasattr(geom, "geoms"):
        points = []
        for part in geom.geoms:
            points.extend(geometry_points(part))
        return points
    return []


def local_tangent(line, distance, window):
    """Calcula uma tangente por diferença centrada ao longo da linha."""
    if line is None or line.is_empty or line.length <= EPS:
        return None

    half_window = max(float(window), 0.05)
    half_window = min(half_window, max(line.length / 2.0, 0.05))
    d0 = max(0.0, float(distance) - half_window)
    d1 = min(float(line.length), float(distance) + half_window)

    p0 = line.interpolate(d0)
    p1 = line.interpolate(d1)
    vector = np.array([p1.x - p0.x, p1.y - p0.y], dtype=float)
    norm = np.linalg.norm(vector)
    if norm <= EPS:
        return None
    return vector / norm


def sample_distances(length, step):
    """Gera amostras uniformes incluindo exatamente as duas extremidades."""
    if length <= EPS:
        return np.array([], dtype=float)
    count = max(2, int(np.ceil(length / max(step, 0.05))) + 1)
    return np.linspace(0.0, float(length), count)


def angle_between_parallel_directions(a, b):
    """Retorna o menor ângulo entre duas direções, ignorando o sentido."""
    if a is None or b is None:
        return 180.0
    cosine = float(np.clip(abs(np.dot(a, b)), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def endpoint_and_inside_direction(line, at_start, tangent_window):
    """Retorna ponto terminal e direção apontando para dentro da fileira."""
    distance = 0.0 if at_start else line.length
    point = line.interpolate(distance)
    tangent = local_tangent(line, distance, tangent_window)
    if tangent is None:
        return point, None
    inside = tangent if at_start else -tangent
    return point, inside


def extend_line_to_boundary(line, block_geom):
    """Estende as extremidades pelas tangentes terminais e recorta no talhão."""
    if block_geom is None or block_geom.is_empty or line is None or line.is_empty:
        return line

    coords = list(line.coords)
    if len(coords) < 2:
        return line

    minx, miny, maxx, maxy = block_geom.bounds
    extension = max(np.hypot(maxx - minx, maxy - miny) * 2.0, line.length * 2.0, 1.0)

    p0 = np.array(coords[0], dtype=float)
    p1 = np.array(coords[1], dtype=float)
    start_vector = p0 - p1
    start_norm = np.linalg.norm(start_vector)
    if start_norm > EPS:
        p0 = p0 + extension * start_vector / start_norm

    pn = np.array(coords[-1], dtype=float)
    pn1 = np.array(coords[-2], dtype=float)
    end_vector = pn - pn1
    end_norm = np.linalg.norm(end_vector)
    if end_norm > EPS:
        pn = pn + extension * end_vector / end_norm

    extended = LineString([tuple(p0)] + coords + [tuple(pn)])
    clipped = extended.intersection(block_geom)
    if clipped.is_empty:
        return line
    if clipped.geom_type == "LineString":
        return clipped
    if clipped.geom_type == "MultiLineString":
        parts = [part for part in clipped.geoms if not part.is_empty]
        if parts:
            return max(parts, key=lambda part: part.length)
    return line


def find_neighbor_candidates(
    anchor_idx,
    anchor_point,
    anchor_tangent,
    normal,
    block_lines,
    tree,
    min_spacing,
    max_spacing,
    tangent_window,
    max_tangent_angle,
    eligible_indices,
    preferred_indices=None,
    recovery_tangent_angle=85.0,
):
    """Encontra candidatos válidos nos dois lados da normal local.

    ``preferred_indices`` contém os vizinhos que já estavam sendo usados na
    amostra anterior. Eles recebem uma recuperação controlada para curvas
    críticas: se a interseção exata com a normal falhar, o ponto mais próximo
    da mesma linha pode ser usado, desde que ainda esteja na faixa lateral e
    dentro do ângulo máximo de recuperação. Uma linha nova nunca recebe essa
    tolerância.
    """
    preferred_indices = set(preferred_indices or set())
    p_start = Point(
        anchor_point.x - max_spacing * normal[0],
        anchor_point.y - max_spacing * normal[1],
    )
    p_end = Point(
        anchor_point.x + max_spacing * normal[0],
        anchor_point.y + max_spacing * normal[1],
    )
    probe = LineString([p_start, p_end])

    try:
        query_result = tree.query(probe, predicate="intersects")
    except TypeError:
        query_result = tree.query(probe)

    left_candidates = []
    right_candidates = []
    cosine_limit = np.cos(np.radians(max_tangent_angle))
    recovery_cosine_limit = np.cos(np.radians(recovery_tangent_angle))

    # O índice espacial normalmente devolve apenas linhas que cruzam a normal.
    # Incluímos também os vizinhos anteriores para permitir a recuperação
    # controlada quando a normal falhar exatamente em um ângulo crítico.
    candidate_indices = []
    for candidate in query_result:
        idx_j = as_index(candidate, block_lines)
        if idx_j is not None and idx_j not in candidate_indices:
            candidate_indices.append(idx_j)
    for idx_j in preferred_indices:
        if idx_j not in candidate_indices:
            candidate_indices.append(idx_j)

    for idx_j in candidate_indices:
        if idx_j == anchor_idx or idx_j not in eligible_indices:
            continue

        line_j = block_lines[idx_j]
        hits = [(hit, False) for hit in geometry_points(probe.intersection(line_j))]

        # Em uma curva muito fechada a normal pode deixar de cruzar a linha
        # vizinha por uma amostra. Só a mesma linha anterior pode ser
        # recuperada pelo ponto mais próximo; isso evita saltos para outra
        # fileira.
        if not hits and idx_j in preferred_indices:
            target_distance = line_j.project(anchor_point)
            nearest_hit = line_j.interpolate(target_distance)
            nearest_distance = anchor_point.distance(nearest_hit)
            if min_spacing <= nearest_distance <= max_spacing:
                hits = [(nearest_hit, True)]

        for hit, is_recovery in hits:
            distance = anchor_point.distance(hit)
            if distance < min_spacing or distance > max_spacing:
                continue

            vector = np.array([hit.x - anchor_point.x, hit.y - anchor_point.y], dtype=float)
            signed_side = float(np.dot(vector, normal))
            if abs(signed_side) <= EPS:
                continue

            # Uma recuperação por ponto mais próximo não pode ser uma ligação
            # predominantemente longitudinal. Ela ainda precisa estar de lado
            # da referência, e não na frente ou atrás dela.
            if is_recovery and abs(signed_side) < max(min_spacing * 0.5, distance * 0.25):
                continue

            target_distance = line_j.project(hit)
            target_tangent = local_tangent(line_j, target_distance, tangent_window)
            if target_tangent is None:
                continue

            parallel_cosine = abs(float(np.dot(anchor_tangent, target_tangent)))
            if parallel_cosine < cosine_limit:
                # A tolerância maior só é permitida para o vizinho que já
                # estava selecionado; candidatos novos continuam estritos.
                if idx_j not in preferred_indices or parallel_cosine < recovery_cosine_limit:
                    continue

            candidate_item = (idx_j, hit, float(distance), target_tangent)
            if signed_side > 0:
                right_candidates.append(candidate_item)
            else:
                left_candidates.append(candidate_item)

    left_candidates.sort(key=lambda item: item[2])
    right_candidates.sort(key=lambda item: item[2])
    return left_candidates, right_candidates


def choose_stable_candidate(candidates, previous_idx, previous_distance, max_spacing_jump):
    """Mantém o vizinho e nunca troca de linha no meio de um trecho."""
    if previous_idx is not None:
        same_line = [item for item in candidates if item[0] == previous_idx]
        if same_line:
            candidate = same_line[0]
            if previous_distance is None or abs(candidate[2] - previous_distance) <= max_spacing_jump:
                return candidate
        # Se o vizinho anterior não passou nos testes, encerra o trecho atual.
        # A aquisição de outro vizinho só pode começar como um novo trecho,
        # evitando que a geometria salte silenciosamente para outra fileira.
        return None

    return candidates[0] if candidates else None


def split_contiguous_samples(samples, max_bridge_gap, max_midpoint_jump, min_points):
    """Divide amostras em feições sem desenhar através de lacunas."""
    if not samples:
        return []

    ordered = sorted(samples, key=lambda item: item[0])
    runs = []
    current = [ordered[0]]

    for item in ordered[1:]:
        previous = current[-1]
        reference_gap = item[0] - previous[0]
        midpoint_gap = previous[1].distance(item[1])

        if reference_gap > max_bridge_gap or midpoint_gap > max_midpoint_jump:
            if len(current) >= min_points:
                runs.append(current)
            current = [item]
        else:
            current.append(item)

    if len(current) >= min_points:
        runs.append(current)
    return runs


def load_lines(path):
    """Carrega e explode todas as geometrias em LineStrings individuais."""
    gdf = gpd.read_file(path)
    if gdf.empty:
        return gdf, []

    exploded = gdf.explode(index_parts=False)
    lines = []
    for geom in exploded.geometry:
        if geom is None or geom.is_empty:
            continue
        if geom.geom_type == "LineString":
            lines.append(geom)
        elif geom.geom_type == "MultiLineString":
            lines.extend(part for part in geom.geoms if not part.is_empty)
    return gdf, lines


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Gera entrelinhas curvas sem ligar trechos por cima de lacunas "
            "ou trocar silenciosamente de fileira."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input_lines", "-i", required=True, help="Arquivo de linhas.")
    parser.add_argument("--talhao_path", "-t", default=None, help="Limite do talhão.")
    parser.add_argument(
        "--output_dir", "-o", default="data/outputs/entrelinhas_curvas",
        help="Diretório dos resultados.",
    )
    parser.add_argument("--min_row_spacing", type=float, default=4.0, help="Espaçamento mínimo.")
    parser.add_argument("--max_row_spacing", type=float, default=8.5, help="Espaçamento máximo.")
    parser.add_argument(
        "--sample_step", type=float, default=0.50,
        help="Passo de amostragem ao longo da referência; reduza para 0.25 em curvas fechadas.",
    )
    parser.add_argument(
        "--tangent_window", type=float, default=1.0,
        help="Janela usada para calcular a tangente local.",
    )
    parser.add_argument(
        "--max_tangent_angle", type=float, default=65.0,
        help="Ângulo máximo entre as direções das fileiras vizinhas novas.",
    )
    parser.add_argument(
        "--recovery_tangent_angle", type=float, default=85.0,
        help=(
            "Ângulo máximo de recuperação do mesmo vizinho em curvas críticas. "
            "Não é usado para aceitar uma linha nova."
        ),
    )
    parser.add_argument(
        "--max_bridge_gap", type=float, default=3.0,
        help=(
            "Maior lacuna, em metros ao longo da referência, que pode ficar "
            "dentro da mesma entrelinha. Lacunas maiores geram feições separadas."
        ),
    )
    parser.add_argument(
        "--max_spacing_jump", type=float, default=None,
        help="Variação máxima de espaçamento entre amostras consecutivas.",
    )
    parser.add_argument(
        "--min_points_per_segment", type=int, default=6,
        help="Número mínimo de amostras para aceitar um trecho de entrelinha.",
    )
    parser.add_argument(
        "--min_segment_length", type=float, default=10.0,
        help="Comprimento mínimo, em metros, para manter um trecho de saída.",
    )
    parser.add_argument(
        "--extend_to_boundary", action="store_true",
        help="Estende cada trecho até o limite do talhão; desativado por segurança.",
    )

    args = parser.parse_args()
    if args.min_row_spacing <= 0 or args.max_row_spacing <= args.min_row_spacing:
        print("[ERRO] É necessário max_row_spacing > min_row_spacing > 0.")
        sys.exit(1)
    if args.sample_step <= 0 or args.tangent_window <= 0:
        print("[ERRO] sample_step e tangent_window devem ser maiores que zero.")
        sys.exit(1)
    if not 0 < args.max_tangent_angle < 90:
        print("[ERRO] max_tangent_angle deve estar entre 0 e 90 graus.")
        sys.exit(1)
    if not args.max_tangent_angle <= args.recovery_tangent_angle < 90:
        print("[ERRO] recovery_tangent_angle deve ser >= max_tangent_angle e < 90 graus.")
        sys.exit(1)
    if args.max_bridge_gap <= 0 or args.min_points_per_segment < 2 or args.min_segment_length <= 0:
        print("[ERRO] max_bridge_gap deve ser positivo e min_points_per_segment >= 2.")
        sys.exit(1)

    spacing_jump = args.max_spacing_jump
    if spacing_jump is None:
        spacing_jump = max(0.75, args.max_row_spacing * 0.35)

    print("=" * 76)
    print("      CITROSROWS AI - ENTRELINHAS CURVAS COM CONTINUIDADE SEGURA")
    print("=" * 76)
    print(f"Linhas de entrada : {args.input_lines}")
    print(f"Limite do talhão  : {args.talhao_path or 'Não fornecido'}")
    print(f"Faixa de busca    : {args.min_row_spacing} m a {args.max_row_spacing} m")
    print(f"Amostragem        : {args.sample_step} m")
    print(f"Janela da tangente: {args.tangent_window} m")
    print(f"Ângulo máximo     : {args.max_tangent_angle} graus")
    print(f"Ângulo recuperação: {args.recovery_tangent_angle} graus")
    print(f"Lacuna máxima     : {args.max_bridge_gap} m")
    print(f"Trecho mínimo     : {args.min_segment_length} m")
    print(f"Salto de distância: {spacing_jump:.2f} m")
    print(f"Extensão à borda  : {'sim' if args.extend_to_boundary else 'não'}")
    print("-" * 76)

    if not os.path.exists(args.input_lines):
        print(f"[ERRO] Arquivo não encontrado: {args.input_lines}")
        sys.exit(1)

    try:
        original_gdf, lines = load_lines(args.input_lines)
    except Exception as exc:
        print(f"[ERRO] Falha ao abrir as linhas: {exc}")
        sys.exit(1)

    if not lines:
        print("[ERRO] Nenhuma LineString válida foi encontrada.")
        sys.exit(1)

    original_crs = original_gdf.crs
    gdf_crs = original_crs
    # Distâncias e ângulos precisam ser calculados em coordenadas métricas.
    lines_gdf = gpd.GeoDataFrame({"geometry": lines}, crs=original_crs)
    if original_crs is not None and original_crs.is_geographic:
        try:
            metric_crs = lines_gdf.estimate_utm_crs()
            lines_gdf = lines_gdf.to_crs(metric_crs)
            lines = list(lines_gdf.geometry)
            gdf_crs = metric_crs
            print(f"[INFO] CRS geográfico detectado; cálculos em UTM: {metric_crs}.")
        except Exception as exc:
            lines_gdf = lines_gdf.to_crs("EPSG:3857")
            lines = list(lines_gdf.geometry)
            gdf_crs = lines_gdf.crs
            print(f"[WARN] Falha ao estimar UTM ({exc}); cálculos em EPSG:3857.")
    else:
        print(f"[INFO] CRS das linhas: {original_crs or 'não informado'}.")

    # Talhões são reprojetados para o mesmo CRS métrico das linhas.
    blocks_to_process = []
    if args.talhao_path:
        if not os.path.exists(args.talhao_path):
            print(f"[WARN] Limite não encontrado; processamento global: {args.talhao_path}")
        else:
            try:
                blocks = gpd.read_file(args.talhao_path)
                if gdf_crs is not None:
                    blocks = blocks.to_crs(gdf_crs)
                for idx, row in blocks.iterrows():
                    geom = row.geometry
                    if geom is not None and not geom.is_empty:
                        blocks_to_process.append((f"Bloco {idx + 1}", geom))
            except Exception as exc:
                print(f"[WARN] Falha ao abrir o limite ({exc}); processamento global: {exc}")

    if not blocks_to_process:
        blocks_to_process = [("Global", None)]

    all_geometries = []
    all_attributes = {
        "dist_mean": [],
        "dist_min": [],
        "dist_max": [],
        "row_len_1": [],
        "row_len_2": [],
        "reference_len": [],
        "source_a": [],
        "source_b": [],
        "segment_no": [],
    }

    for block_name, block_geom in blocks_to_process:
        if block_geom is None:
            block_lines = lines
        else:
            block_lines = [line for line in lines if line.intersects(block_geom)]

        print(f"\n--- Processando {block_name}: {len(block_lines)} fileiras/fragmentos ---")
        if len(block_lines) < 2:
            print("  [INFO] Menos de duas geometrias; bloco ignorado.")
            continue

        tree = STRtree(block_lines)
        pair_data = {}

        # A cada ponto da linha âncora, busca somente vizinhos cujo comprimento
        # não seja maior que o da âncora. Assim cada par é processado a partir
        # da fileira maior, nunca a partir da menor.
        for anchor_idx, anchor_line in enumerate(block_lines):
            if anchor_line.length < 1.0:
                continue

            eligible_indices = set()
            for candidate_idx, candidate_line in enumerate(block_lines):
                if candidate_idx == anchor_idx:
                    continue
                if candidate_line.length < anchor_line.length:
                    eligible_indices.add(candidate_idx)
                elif np.isclose(candidate_line.length, anchor_line.length) and candidate_idx > anchor_idx:
                    eligible_indices.add(candidate_idx)

            if not eligible_indices:
                continue

            previous_left_idx = None
            previous_left_distance = None
            previous_right_idx = None
            previous_right_distance = None

            for anchor_distance in sample_distances(anchor_line.length, args.sample_step):
                anchor_point = anchor_line.interpolate(float(anchor_distance))
                anchor_tangent = local_tangent(
                    anchor_line,
                    float(anchor_distance),
                    args.tangent_window,
                )
                if anchor_tangent is None:
                    continue

                normal = np.array([-anchor_tangent[1], anchor_tangent[0]], dtype=float)
                left_candidates, right_candidates = find_neighbor_candidates(
                    anchor_idx,
                    anchor_point,
                    anchor_tangent,
                    normal,
                    block_lines,
                    tree,
                    args.min_row_spacing,
                    args.max_row_spacing,
                    args.tangent_window,
                    args.max_tangent_angle,
                    eligible_indices,
                    preferred_indices={idx for idx in (previous_left_idx, previous_right_idx) if idx is not None},
                    recovery_tangent_angle=args.recovery_tangent_angle,
                )

                left = choose_stable_candidate(
                    left_candidates,
                    previous_left_idx,
                    previous_left_distance,
                    spacing_jump,
                )
                right = choose_stable_candidate(
                    right_candidates,
                    previous_right_idx,
                    previous_right_distance,
                    spacing_jump,
                )

                if left is None:
                    previous_left_idx = None
                    previous_left_distance = None
                else:
                    previous_left_idx = left[0]
                    previous_left_distance = left[2]
                    pair_key = (min(anchor_idx, left[0]), max(anchor_idx, left[0]))
                    midpoint = Point(
                        (anchor_point.x + left[1].x) / 2.0,
                        (anchor_point.y + left[1].y) / 2.0,
                    )
                    pair_data.setdefault(pair_key, []).append(
                        (float(anchor_distance), midpoint, float(left[2]))
                    )

                if right is None:
                    previous_right_idx = None
                    previous_right_distance = None
                else:
                    previous_right_idx = right[0]
                    previous_right_distance = right[2]
                    pair_key = (min(anchor_idx, right[0]), max(anchor_idx, right[0]))
                    midpoint = Point(
                        (anchor_point.x + right[1].x) / 2.0,
                        (anchor_point.y + right[1].y) / 2.0,
                    )
                    pair_data.setdefault(pair_key, []).append(
                        (float(anchor_distance), midpoint, float(right[2]))
                    )

        block_results = []
        for pair_key, samples in pair_data.items():
            idx1, idx2 = pair_key
            line1 = block_lines[idx1]
            line2 = block_lines[idx2]
            reference_idx = idx1 if line1.length >= line2.length else idx2
            reference_line = block_lines[reference_idx]

            # Importante: uma única LineString não deve atravessar um trecho
            # em que não houve correspondência ou em que a correspondência
            # pulou de posição. A saída é dividida nesses pontos.
            runs = split_contiguous_samples(
                samples,
                max_bridge_gap=args.max_bridge_gap,
                max_midpoint_jump=max(args.max_row_spacing * 2.0, args.max_bridge_gap * 2.0),
                min_points=args.min_points_per_segment,
            )

            for segment_no, run in enumerate(runs, start=1):
                # Remove pontos duplicados consecutivos sem simplificar a forma.
                cleaned = []
                for item in run:
                    if cleaned and item[1].distance(cleaned[-1][1]) < 1e-4:
                        continue
                    cleaned.append(item)
                if len(cleaned) < args.min_points_per_segment:
                    continue

                midline = LineString([item[1].coords[0] for item in cleaned])
                if midline.is_empty or midline.length <= EPS:
                    continue
                if midline.length < args.min_segment_length:
                    continue

                output_line = midline
                if args.extend_to_boundary and block_geom is not None:
                    output_line = extend_line_to_boundary(output_line, block_geom)

                distances = np.array([item[2] for item in cleaned], dtype=float)
                block_results.append(
                    (
                        output_line,
                        distances,
                        line1.length,
                        line2.length,
                        reference_line.length,
                        idx1,
                        idx2,
                        segment_no,
                    )
                )

        print(f"  Entrelinhas/trechos gerados: {len(block_results)}")
        for (
            geom,
            distances,
            length1,
            length2,
            reference_length,
            source_a,
            source_b,
            segment_no,
        ) in block_results:
            all_geometries.append(geom)
            all_attributes["dist_mean"].append(float(distances.mean()))
            all_attributes["dist_min"].append(float(distances.min()))
            all_attributes["dist_max"].append(float(distances.max()))
            all_attributes["row_len_1"].append(float(length1))
            all_attributes["row_len_2"].append(float(length2))
            all_attributes["reference_len"].append(float(reference_length))
            all_attributes["source_a"].append(int(source_a))
            all_attributes["source_b"].append(int(source_b))
            all_attributes["segment_no"].append(int(segment_no))

    print("\n" + "=" * 76)
    print(f"[OK] Trechos de entrelinha gerados: {len(all_geometries)}")
    if not all_geometries:
        print("[WARN] Nenhum trecho foi gerado com os critérios informados.")
        sys.exit(0)

    output_gdf = gpd.GeoDataFrame(all_attributes, geometry=all_geometries, crs=gdf_crs)
    output_gdf["length_m"] = output_gdf.geometry.length

    if original_crs is not None and gdf_crs is not None and gdf_crs != original_crs:
        print(f"[INFO] Reprojetando resultados para o CRS original: {original_crs}.")
        output_gdf = output_gdf.to_crs(original_crs)

    os.makedirs(args.output_dir, exist_ok=True)
    geojson_out = os.path.join(args.output_dir, "predicao_entrelinhas.geojson")
    shp_dir = os.path.join(args.output_dir, "shp")
    shp_out = os.path.join(shp_dir, "predicao_entrelinhas.shp")

    try:
        output_gdf.to_file(geojson_out, driver="GeoJSON")
        print(f"[OK] GeoJSON salvo: {geojson_out}")
    except Exception as exc:
        print(f"[WARN] Falha ao salvar GeoJSON com GeoPandas: {exc}")
        try:
            with open(geojson_out, "w", encoding="utf-8") as output_file:
                output_file.write(output_gdf.to_json())
            print(f"[OK] GeoJSON salvo via fallback: {geojson_out}")
        except Exception as fallback_exc:
            print(f"[ERRO] Falha total no GeoJSON: {fallback_exc}")

    try:
        os.makedirs(shp_dir, exist_ok=True)
        output_gdf.to_file(shp_out)
        print(f"[OK] Shapefile salvo: {shp_out}")
    except Exception as exc:
        print(f"[WARN] Falha ao salvar Shapefile: {exc}")

    print("=" * 76)
    print("      CÁLCULO DE ENTRELINHAS CURVAS CONCLUÍDO")
    print("=" * 76)


if __name__ == "__main__":
    main()
