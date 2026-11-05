#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
calcular_entrelinhas_curvas.py
--------------------------------------------------------------------------------
Script de pós-processamento geográfico para gerar linhas médias (entrelinhas)
em talhões com fileiras de plantio curvas (ex: plantios em curva de nível).

Diferente da ordenação global, este script implementa uma busca espacial por 
normal local de vizinhança, suportando perfeitamente curvas acentuadas e 
geometrias sinuosas sem a necessidade de um ângulo dominante global.
--------------------------------------------------------------------------------
"""

import os
import sys
import argparse
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, LineString, MultiPoint
from shapely.strtree import STRtree


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
        
    # Tamanho da diagonal do talhão para estender com folga
    bounds = block_geom.bounds
    diag = np.sqrt((bounds[2] - bounds[0])**2 + (bounds[3] - bounds[1])**2)
    ext_length = diag * 1.5
    
    # Vetor tangente no início da linha (aponta para fora/trás)
    p0 = np.array(coords[0])
    p1 = np.array(coords[1])
    v_start = p0 - p1
    len_start = np.linalg.norm(v_start)
    if len_start > 0:
        u_start = v_start / len_start
        p_start_extended = p0 + ext_length * u_start
    else:
        p_start_extended = p0
        
    # Vetor tangente no fim da linha (aponta para fora/frente)
    pn = np.array(coords[-1])
    pn1 = np.array(coords[-2])
    v_end = pn - pn1
    len_end = np.linalg.norm(v_end)
    if len_end > 0:
        u_end = v_end / len_end
        p_end_extended = pn + ext_length * u_end
    else:
        p_end_extended = pn
        
    # Cria a linha estendida
    extended_coords = [tuple(p_start_extended)] + coords + [tuple(p_end_extended)]
    long_line = LineString(extended_coords)
    
    # Intersecciona com a borda do talhão (recorta exatamente na borda!)
    intersection = long_line.intersection(block_geom)
    
    if intersection.is_empty:
        return line
        
    if intersection.geom_type == 'LineString':
        return intersection
    elif intersection.geom_type == 'MultiLineString':
        # Se for MultiLineString, retorna a maior componente
        valid_lines = [geom for geom in intersection.geoms if not geom.is_empty]
        if valid_lines:
            return max(valid_lines, key=lambda l: l.length)
            
    return line


def main():
    parser = argparse.ArgumentParser(
        description="Gera entrelinhas para talhões curvos (curva de nível) usando busca espacial por normal local.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--input_lines", "-i", required=True,
        help="Caminho do arquivo de linhas de plantio (GeoJSON ou Shapefile)."
    )
    parser.add_argument(
        "--talhao_path", "-t", default=None,
        help="Caminho do limite do talhão (GeoJSON ou Shapefile). Essencial para delimitar as curvas."
    )
    parser.add_argument(
        "--output_dir", "-o", default="data/outputs/entrelinhas_curvas",
        help="Diretório onde os arquivos de saída serão salvos."
    )
    parser.add_argument(
        "--min_row_spacing", type=float, default=4.0,
        help="Espaçamento lateral mínimo em metros tolerado entre fileiras vizinhas distintas."
    )
    parser.add_argument(
        "--max_row_spacing", type=float, default=8.5,
        help="Espaçamento lateral máximo em metros tolerado entre fileiras vizinhas distintas."
    )
    parser.add_argument(
        "--sample_step", type=float, default=1.5,
        help="Espaçamento em metros para amostragem dinâmica ao longo das curvas."
    )
    
    args = parser.parse_args()
    
    print("=" * 65)
    print("      CITROSROWS AI - CÁLCULO DE ENTRELINHAS CURVAS (LOCAL)")
    print("=" * 65)
    print(f"Linhas de Entrada: {args.input_lines}")
    print(f"Limite do Talhão : {args.talhao_path or 'Não fornecido'}")
    print(f"Limiares de Busca: Mínimo = {args.min_row_spacing}m | Máximo = {args.max_row_spacing}m")
    print(f"Amostragem      : Cada {args.sample_step}m ao longo de cada curva")
    print("-" * 65)
    
    if not os.path.exists(args.input_lines):
        print(f"[ERRO] Arquivo de linhas não encontrado: {args.input_lines}")
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
    
    # 1. Reprojeção dinâmica para sistema métrico (UTM)
    is_projected = False
    if original_crs is not None:
        if original_crs.is_geographic:
            try:
                utm_crs = gdf_lines.estimate_utm_crs()
                gdf_lines = gdf_lines.to_crs(utm_crs)
                is_projected = True
                print(f"[INFO] Dados geográficos detectados. Reprojetando para UTM local: {utm_crs.to_string()}")
            except Exception as e:
                gdf_lines = gdf_lines.to_crs("EPSG:3857")
                is_projected = True
                print(f"[WARN] Falha ao estimar UTM ({e}). Reprojetando para EPSG:3857 (Web Mercator) para cálculos métricos.")
        else:
            print(f"[INFO] Dados já em formato métrico/projetado: {original_crs.to_string()}")
            
    # 2. Explodir MultiLineStrings
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
        print("[ERRO] Nenhuma LineString válida restante após a explosão.")
        sys.exit(1)
        
    print(f"[OK] Total de {len(lines)} fileiras (LineStrings) individuais para processamento.")
    
    # 3. Particionar por blocos (talhão) ou processamento global
    blocks_to_process = []
    if args.talhao_path and os.path.exists(args.talhao_path):
        try:
            gdf_talhao = gpd.read_file(args.talhao_path)
            if not gdf_talhao.empty:
                gdf_talhao = gdf_talhao.to_crs(gdf_lines.crs)
                for idx, row in gdf_talhao.iterrows():
                    geom_t = row.geometry
                    if geom_t is not None and not geom_t.is_empty:
                        blocks_to_process.append((f"Bloco {idx+1}", geom_t))
            else:
                print("[WARN] Arquivo de talhão vazio. Processando de forma global.")
        except Exception as e:
            print(f"[WARN] Falha ao abrir limite do talhão: {e}. Usando processamento global.")
            
    if not blocks_to_process:
        print("[INFO] Processando todas as fileiras como um único bloco global.")
        blocks_to_process.append(("Global", None))
        
    # Acumulador global de entrelinhas e atributos
    all_inter_rows = []
    all_attributes = {
        'dist_mean': [],
        'dist_min': [],
        'dist_max': [],
        'row_len_1': [],
        'row_len_2': []
    }
    
    # 4. Processamento bloco a bloco (independência geométrica absoluta)
    for block_name, block_geom in blocks_to_process:
        if block_geom is not None:
            block_lines = [line for line in lines if line.intersects(block_geom)]
            print(f"\n--- Processando {block_name} ({len(block_lines)} fileiras encontradas) ---")
        else:
            block_lines = lines
            print(f"\n--- Processando Bloco Global ({len(block_lines)} fileiras) ---")
            
        if len(block_lines) < 2:
            print(f"  [INFO] Poucas fileiras ({len(block_lines)}) neste bloco. Entrelinhas não podem ser geradas.")
            continue
            
        # Criar índice espacial (STRtree) das linhas deste bloco para buscas rápidas
        tree = STRtree(block_lines)
        
        # Calcula espaçamento adaptativo lateral para as curvas deste bloco
        local_min_row_spacing = args.min_row_spacing
        local_max_row_spacing = args.max_row_spacing
        
        distancias_bloco = []
        for idx_a, line_a in enumerate(block_lines):
            min_d = float('inf')
            for idx_b, line_b in enumerate(block_lines):
                if idx_a == idx_b:
                    continue
                d = line_a.distance(line_b)
                if d < min_d:
                    min_d = d
            if min_d < float('inf') and min_d > 0.5: # ignora duplicatas perfeitas
                distancias_bloco.append(min_d)
                
        if distancias_bloco:
            spacing_median = np.median(distancias_bloco)
            local_min_row_spacing = spacing_median * 0.6
            local_max_row_spacing = spacing_median * 1.4
            print(f"  [Autoadaptação] Espaçamento mediano: {spacing_median:.2f}m | Limites: [{local_min_row_spacing:.2f}m, {local_max_row_spacing:.2f}m]")
            
        # Dicionário para acumular nuvens de pontos médios por par de vizinhos
        # Chave: (min_idx, max_idx), Valor: {'midpoints': [], 'distances': []}
        pair_data = {}
        
        # 5. Para cada fileira, amostrar pontos e projetar normais à esquerda e direita
        for i, line_i in enumerate(block_lines):
            line_len = line_i.length
            if line_len < 3.0:
                continue
                
            # Amostra pontos a intervalos regulares ao longo da curva
            num_samples = int(np.ceil(line_len / args.sample_step))
            num_samples = max(5, num_samples)
            
            for dist in np.linspace(0.1, line_len - 0.1, num_samples):
                P_k = line_i.interpolate(dist)
                
                # Calcular vetor tangente local usando uma diferença finita minúscula (10cm)
                pt_next = line_i.interpolate(dist + 0.1 if dist + 0.1 <= line_len else dist - 0.1)
                dx = pt_next.x - P_k.x
                dy = pt_next.y - P_k.y
                
                # Inverte direção caso tenhamos retrocedido no final da linha
                if dist + 0.1 > line_len:
                    dx, dy = -dx, -dy
                    
                len_t = np.sqrt(dx**2 + dy**2)
                if len_t == 0:
                    continue
                    
                u_tangent = np.array([dx, dy]) / len_t
                
                # Vetor normal local rotacionando 90 graus
                u_normal = np.array([-u_tangent[1], u_tangent[0]])
                
                # Lança o segmento de busca (probe segment) lateral
                p_start = Point(P_k.x - args.max_row_spacing * u_normal[0], P_k.y - args.max_row_spacing * u_normal[1])
                p_end = Point(P_k.x + args.max_row_spacing * u_normal[0], P_k.y + args.max_row_spacing * u_normal[1])
                probe_seg = LineString([p_start, p_end])
                
                # Consultar de forma extremamente rápida o índice espacial
                intersect_indices = tree.query(probe_seg, predicate="intersects")
                
                closest_left = None
                min_d_left = float('inf')
                
                closest_right = None
                min_d_right = float('inf')
                
                for idx_j in intersect_indices:
                    if idx_j == i:
                        continue # Evita auto-interseção
                        
                    line_j = block_lines[idx_j]
                    intersection = probe_seg.intersection(line_j)
                    
                    if intersection.is_empty:
                        continue
                        
                    # Extrair todos os pontos de interseção (tratando Point, MultiPoint, etc.)
                    points = []
                    if intersection.geom_type == 'Point':
                        points.append(intersection)
                    elif intersection.geom_type == 'MultiPoint':
                        points.extend(intersection.geoms)
                    elif intersection.geom_type == 'LineString':
                        points.append(Point(intersection.coords[0]))
                        points.append(Point(intersection.coords[-1]))
                    elif intersection.geom_type == 'MultiLineString':
                        for sub_g in intersection.geoms:
                            if not sub_g.is_empty:
                                points.append(Point(sub_g.coords[0]))
                                points.append(Point(sub_g.coords[-1]))
                                
                    # Encontrar o ponto mais próximo ao longo do probe
                    for pt in points:
                        d = P_k.distance(pt)
                        if d < local_min_row_spacing or d > local_max_row_spacing:
                            continue
                            
                        # Determinar orientação (esquerda ou direita) por projeção escalar
                        w = np.array([pt.x - P_k.x, pt.y - P_k.y])
                        proj = np.dot(w, u_normal)
                        
                        if proj > 0: # Lado Direito
                            if d < min_d_right:
                                min_d_right = d
                                closest_right = (idx_j, pt, d)
                        elif proj < 0: # Lado Esquerdo
                            if d < min_d_left:
                                min_d_left = d
                                closest_left = (idx_j, pt, d)
                                
                # Se encontrou vizinho esquerdo consistente, acumula ponto médio
                if closest_left is not None:
                    j_idx, pt_left, dist_val = closest_left
                    mid_x = (P_k.x + pt_left.x) / 2.0
                    mid_y = (P_k.y + pt_left.y) / 2.0
                    pair_key = (min(i, j_idx), max(i, j_idx))
                    if pair_key not in pair_data:
                        pair_data[pair_key] = {'midpoints': [], 'distances': []}
                    pair_data[pair_key]['midpoints'].append(Point(mid_x, mid_y))
                    pair_data[pair_key]['distances'].append(dist_val)
                    
                # Se encontrou vizinho direito consistente, acumula ponto médio
                if closest_right is not None:
                    j_idx, pt_right, dist_val = closest_right
                    mid_x = (P_k.x + pt_right.x) / 2.0
                    mid_y = (P_k.y + pt_right.y) / 2.0
                    pair_key = (min(i, j_idx), max(i, j_idx))
                    if pair_key not in pair_data:
                        pair_data[pair_key] = {'midpoints': [], 'distances': []}
                    pair_data[pair_key]['midpoints'].append(Point(mid_x, mid_y))
                    pair_data[pair_key]['distances'].append(dist_val)
                    
        # 6. Para cada par acumulado, ordenar curvilinearmente, reconstruir e estender
        block_inter_rows = []
        for pair_key, data in pair_data.items():
            pts = data['midpoints']
            dists = data['distances']
            
            # Filtro de robustez para ruídos (exige pelo menos 6 pontos amostrados)
            if len(pts) < 6:
                continue
                
            idx1, idx2 = pair_key
            parent_line = block_lines[idx1] # Usa a primeira fileira do par para projetar a curva
            
            # Ordena a nuvem de pontos médios projetando as coordenadas ao longo da curva da fileira mãe
            sorted_items = sorted(zip(pts, dists), key=lambda x: parent_line.project(x[0]))
            
            sorted_pts = [item[0] for item in sorted_items]
            sorted_dists = np.array([item[1] for item in sorted_items])
            
            # Gera a LineString unificada da entrelinha
            midline = LineString(sorted_pts)
            
            # Aplica simplificação suave para remover micro-oscilações de amostragem
            midline = midline.simplify(0.1, preserve_topology=True)
            
            if midline is not None and not midline.is_empty:
                # Estende e recorta perfeitamente nos carreadores (limite do bloco)
                if block_geom is not None:
                    extended_midline = extend_line_to_boundary(midline, block_geom)
                    if extended_midline is not None and not extended_midline.is_empty:
                        block_inter_rows.append((extended_midline, sorted_dists, parent_line.length, block_lines[idx2].length))
                else:
                    block_inter_rows.append((midline, sorted_dists, parent_line.length, block_lines[idx2].length))
                    
        print(f"  Entrelinhas curvas geradas neste bloco: {len(block_inter_rows)}")
        
        # Consolida atributos
        for geom_mid, dist_arr, l1_len, l2_len in block_inter_rows:
            all_inter_rows.append(geom_mid)
            all_attributes['dist_mean'].append(float(dist_arr.mean()))
            all_attributes['dist_min'].append(float(dist_arr.min()))
            all_attributes['dist_max'].append(float(dist_arr.max()))
            all_attributes['row_len_1'].append(l1_len)
            all_attributes['row_len_2'].append(l2_len)
            
    print("\n" + "=" * 65)
    print(f"[OK] Processamento de todos os blocos curvos concluído:")
    print(f"  - Entrelinhas curvas totais geradas : {len(all_inter_rows)}")
    
    if not all_inter_rows:
        print("[WARN] Nenhuma entrelinha curva pôde ser gerada com os espaçamentos especificados.")
        sys.exit(0)
        
    # 7. Criar o GeoDataFrame final com atributos e reverter projeção
    gdf_out = gpd.GeoDataFrame(all_attributes, geometry=all_inter_rows, crs=gdf_lines.crs)
    gdf_out['length_m'] = gdf_out.geometry.length
    
    if original_crs is not None:
        print(f"[INFO] Reprojetando resultados de volta para o CRS original: {original_crs.to_string()}")
        gdf_out = gdf_out.to_crs(original_crs)
        
    # 8. Exportar para GeoJSON e Shapefile
    os.makedirs(args.output_dir, exist_ok=True)
    
    geojson_out = os.path.join(args.output_dir, "predicao_entrelinhas.geojson")
    shp_dir = os.path.join(args.output_dir, "shp")
    shp_out = os.path.join(shp_dir, "predicao_entrelinhas.shp")
    
    print("-" * 65)
    print(f"Salvando resultados em: {args.output_dir}")
    
    try:
        gdf_out.to_file(geojson_out, driver="GeoJSON")
        print(f"[OK] GeoJSON salvo com sucesso: {geojson_out}")
    except Exception as e:
        print(f"[WARN] Falha ao salvar GeoJSON com GeoPandas: {e}. Usando fallback...")
        try:
            with open(geojson_out, 'w', encoding='utf-8') as f:
                f.write(gdf_out.to_json())
            print(f"[OK] GeoJSON salvo via fallback.")
        except Exception as ex:
            print(f"[ERRO] Falha total no GeoJSON: {ex}")
            
    try:
        os.makedirs(shp_dir, exist_ok=True)
        gdf_out.to_file(shp_out)
        print(f"[OK] Shapefile salvo com sucesso: {shp_out}")
    except Exception as e:
        print(f"[WARN] Falha ao salvar Shapefile: {e}")
        
    print("=" * 65)
    print("      CÁLCULO DE ENTRELINHAS CURVAS CONCLUÍDO COM SUCESSO!")
    print("=" * 65)


if __name__ == "__main__":
    main()
