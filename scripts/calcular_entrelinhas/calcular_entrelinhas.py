#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
calcular_entrelinhas.py
--------------------------------------------------------------------------------
Script de pós-processamento geográfico para gerar as linhas centrais das
entrelinhas de plantio a partir das fileiras (linhas de plantio) detectadas.

Suporta linhas retas e curvas por interpolação ponto a ponto adaptável,
preservando a tendência geométrica das fileiras e gerando métricas de qualidade.
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
    Garante o alinhamento perpendicular preciso para ordenação espacial.
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
    Amostra pontos ao longo da linha mais curta e projeta sobre a mais longa
    para mitigar efeitos de borda (linhas de comprimentos muito discrepantes).
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
    Amostra pontos dinamicamente a cada `step_meters` e alinha a direção do fluxo.
    """
    coords1 = list(line1.coords)
    coords2 = list(line2.coords)
    
    p1_start, p1_end = np.array(coords1[0]), np.array(coords1[-1])
    p2_start, p2_end = np.array(coords2[0]), np.array(coords2[-1])
    
    v1 = p1_end - p1_start
    v2 = p2_end - p2_start
    
    # Se as linhas correm em sentidos opostos, inverte a segunda para evitar torções
    if np.dot(v1, v2) < 0:
        coords2 = coords2[::-1]
        line2_aligned = LineString(coords2)
    else:
        line2_aligned = line2
        
    # Amostragem dinâmica baseada no comprimento
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
        description="Gera entrelinhas de plantio a partir de linhas de fileiras de laranjeiras (Citrosrows AI).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--input_lines", "-i", required=True,
        help="Caminho do arquivo de linhas de plantio (GeoJSON ou Shapefile)."
    )
    parser.add_argument(
        "--talhao_path", "-t", default=None,
        help="Caminho do limite do talhão (opcional). Se fornecido, as entrelinhas serão recortadas na borda."
    )
    parser.add_argument(
        "--output_dir", "-o", default="data/outputs/entrelinhas",
        help="Diretório onde os arquivos de saída serão salvos."
    )
    parser.add_argument(
        "--min_row_spacing", type=float, default=4.0,
        help="Espaçamento lateral mínimo em metros para considerar fileiras distintas (evita duplicatas)."
    )
    parser.add_argument(
        "--max_row_spacing", type=float, default=8.5,
        help="Espaçamento lateral máximo em metros para considerar fileiras vizinhas (evita ligar fileiras distantes devido a falhas)."
    )
    parser.add_argument(
        "--step_meters", type=float, default=1.5,
        help="Resolução espacial em metros para a amostragem dinâmica ao longo das linhas."
    )
    parser.add_argument(
        "--min_points", type=int, default=10,
        help="Número mínimo de pontos por linha de amostragem (garante fidelidade em segmentos curtos)."
    )
    parser.add_argument(
        "--reference_lines", "-r", default=None,
        help="Caminho opcional para arquivo de linhas guia/referência de direção (GeoJSON ou Shapefile)."
    )
    parser.add_argument(
        "--timestamp", default=None,
        help="Sufixo de timestamp opcional para os nomes de arquivo (formato YYYYMMDD_HHMMSS)."
    )
    
    args = parser.parse_args()
    
    timestamp = args.timestamp
    if not timestamp:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("=" * 65)
    print("      CITROSROWS AI - CÁLCULO DE ENTRELINHAS DE PLANTIO")
    print("=" * 65)
    print(f"Linhas de Entrada: {args.input_lines}")
    print(f"Limite do Talhão : {args.talhao_path or 'Não fornecido (sem recorte na borda)'}")
    print(f"Limiares de Busca: Mínimo = {args.min_row_spacing}m | Máximo = {args.max_row_spacing}m")
    print(f"Amostragem      : Cada {args.step_meters}m (Mínimo {args.min_points} pontos por linha)")
    print("-" * 65)
    
    # 1. Carregar arquivo de linhas de plantio
    if not os.path.exists(args.input_lines):
        print(f"[ERRO] Arquivo de linhas não encontrado: {args.input_lines}")
        sys.exit(1)
        
    try:
        gdf_lines = gpd.read_file(args.input_lines)
    except Exception as e:
        print(f"[ERRO] Falha ao abrir o arquivo de linhas: {e}")
        sys.exit(1)
        
    if gdf_lines.empty:
        print("[ERRO] O arquivo de linhas de entrada está vazio.")
        sys.exit(0)
        
    # Guardar CRS original para a exportação final
    original_crs = gdf_lines.crs
    print(f"[OK] {len(gdf_lines)} feições de linhas carregadas.")
    
    # 2. Reprojeção dinâmica para sistema métrico (UTM) para cálculos espaciais precisos
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
            
    # 3. Explodir MultiLineStrings e limpar geometrias
    gdf_lines = gdf_lines.explode(index_parts=False)
    lines = []
    for geom in gdf_lines.geometry:
        if geom is not None and not geom.is_empty:
            if geom.geom_type == 'LineString':
                lines.append(geom)
            elif geom.geom_type == 'MultiLineString':
                # Só por segurança caso o explode não cubra algum caso
                for g in geom.geoms:
                    if not g.is_empty:
                        lines.append(g)
                        
    if not lines:
        print("[ERRO] Nenhuma geometria de linha válida restante após a explosão.")
        sys.exit(1)
        
    print(f"[OK] Total de {len(lines)} fileiras (LineStrings) individuais para processamento.")
    
    # Carregar arquivo opcional de linhas de referência para direcionamento
    gdf_ref = None
    if args.reference_lines:
        if not os.path.exists(args.reference_lines):
            print(f"[WARN] Arquivo de linhas de referência não encontrado: {args.reference_lines}. Ignorando.")
        else:
            try:
                gdf_ref = gpd.read_file(args.reference_lines)
                if not gdf_ref.empty:
                    # Garantir que está no mesmo CRS métrico do processamento
                    gdf_ref = gdf_ref.to_crs(gdf_lines.crs)
                    gdf_ref = gdf_ref.explode(index_parts=False)
                    print(f"[OK] {len(gdf_ref)} feições de linhas de referência carregadas para direcionamento.")
                else:
                    print("[WARN] Arquivo de linhas de referência está vazio. Ignorando.")
                    gdf_ref = None
            except Exception as e:
                print(f"[WARN] Falha ao abrir linhas de referência: {e}. Ignorando.")
                gdf_ref = None
                
    # 4. Processar por bloco (se talhão for fornecido e tiver geometrias) ou processamento global
    all_inter_rows = []
    all_attributes = {
        'dist_mean': [],
        'dist_min': [],
        'dist_max': [],
        'row_len_1': [],
        'row_len_2': []
    }
    
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
                print("[WARN] Arquivo de talhão vazio. Processando tudo de forma global.")
        except Exception as e:
            print(f"[WARN] Falha ao ler arquivo de talhão para particionamento: {e}. Usando processamento global.")
            
    if not blocks_to_process:
        print("[INFO] Processando todas as fileiras como um único bloco global.")
        blocks_to_process.append(("Global", None))
        
    total_ignored_close = 0
    total_ignored_far = 0
    
    for block_name, block_geom in blocks_to_process:
        # Filtrar as linhas pertencentes a este bloco
        if block_geom is not None:
            # Uma linha pertence ao bloco se ela intersecciona ou está contida no bloco
            block_lines = [line for line in lines if line.intersects(block_geom)]
            print(f"\n--- Processando {block_name} ({len(block_lines)} fileiras encontradas) ---")
        else:
            block_lines = lines
            print(f"\n--- Processando Bloco Global ({len(block_lines)} fileiras) ---")
            
        if len(block_lines) < 2:
            print(f"  [INFO] Poucas fileiras ({len(block_lines)}) para este bloco. Entrelinhas não podem ser calculadas.")
            continue
            
        # Determinar ângulo dominante local a partir de linhas de referência ou das fileiras
        local_angle = None
        if gdf_ref is not None and block_geom is not None:
            # Filtra apenas as linhas guia que intersectam este bloco
            ref_lines_block = [geom for geom in gdf_ref.geometry if geom is not None and not geom.is_empty and geom.intersects(block_geom)]
            if ref_lines_block:
                local_angle = calculate_lines_dominant_angle(ref_lines_block)
                print(f"  Direção dominante local (Linhas guia/referência): {np.degrees(local_angle):.2f}°")
                
        if local_angle is None:
            local_angle = calculate_lines_dominant_angle(block_lines)
            print(f"  Direção dominante local (Fileiras detectadas): {np.degrees(local_angle):.2f}°")
        
        local_u_perp = np.array([-np.sin(local_angle), np.cos(local_angle)])
        
        # Ordenação transversal local no bloco
        local_lines_pos = []
        for line in block_lines:
            coords = np.array(line.coords)
            midpoint = coords.mean(axis=0)
            pos = np.dot(midpoint, local_u_perp)
            local_lines_pos.append((pos, line))
            
        local_lines_pos.sort(key=lambda x: x[0])
        sorted_local_lines = [item[1] for item in local_lines_pos]
        
        # Varredura e pareamento local
        block_inter_rows = []
        for i in range(len(sorted_local_lines) - 1):
            l1 = sorted_local_lines[i]
            l2 = sorted_local_lines[i + 1]
            
            stats = calculate_distance_stats(l1, l2, num_sample_points=30)
            mean_d = stats['mean']
            
            if mean_d < args.min_row_spacing:
                total_ignored_close += 1
                continue
                
            if mean_d > args.max_row_spacing:
                total_ignored_far += 1
                # Imprime aviso detalhado sobre o espaçamento
                print(f"  [AVISO] Espaçamento de {mean_d:.2f}m entre a fileira {i+1} e {i+2} excede o limite. Nenhuma entrelinha gerada.")
                continue
                
            # Par válido! Calcular entrelinha por amostragem dinâmica
            midline = generate_midline(l1, l2, step_meters=args.step_meters, min_points=args.min_points)
            
            if midline is not None and not midline.is_empty:
                # Estende a entrelinha até a borda física do bloco (carreadores) e recorta
                if block_geom is not None:
                    extended_midline = extend_line_to_boundary(midline, block_geom)
                    if extended_midline is not None and not extended_midline.is_empty:
                        block_inter_rows.append((extended_midline, stats, l1.length, l2.length))
                else:
                    block_inter_rows.append((midline, stats, l1.length, l2.length))
                    
        print(f"  Entrelinhas locais geradas neste bloco: {len(block_inter_rows)}")
        
        # Adiciona os resultados locais ao acumulador global
        for item, stats, len1, len2 in block_inter_rows:
            all_inter_rows.append(item)
            all_attributes['dist_mean'].append(stats['mean'])
            all_attributes['dist_min'].append(stats['min'])
            all_attributes['dist_max'].append(stats['max'])
            all_attributes['row_len_1'].append(len1)
            all_attributes['row_len_2'].append(len2)
            
    print("\n" + "=" * 65)
    print(f"[OK] Processamento de todos os blocos concluído:")
    print(f"  - Entrelinhas totais geradas : {len(all_inter_rows)}")
    print(f"  - Pares muito próximos ignorados  : {total_ignored_close}")
    print(f"  - Pares muito afastados ignorados : {total_ignored_far}")
    
    if not all_inter_rows:
        print("[WARN] Nenhuma entrelinha pôde ser gerada em nenhum dos blocos.")
        sys.exit(0)
        
    # 7. Criar o GeoDataFrame final acumulado
    # O CRS é herdado do processamento projetado (UTM/EPSG:3857)
    gdf_out = gpd.GeoDataFrame(all_attributes, geometry=all_inter_rows, crs=gdf_lines.crs)
    gdf_out['length_m'] = gdf_out.geometry.length
    
    # 8. Reprojetar de volta para o CRS original
    if original_crs is not None:
        print(f"[INFO] Reprojetando resultados de volta para o CRS original: {original_crs.to_string()}")
        gdf_out = gdf_out.to_crs(original_crs)
        
    # 10. Salvar saídas (GeoJSON e Shapefile)
    os.makedirs(args.output_dir, exist_ok=True)
    
    geojson_out = os.path.join(args.output_dir, "predicao_entrelinhas.geojson")
    geojson_ts_out = os.path.join(args.output_dir, f"predicao_entrelinhas_{timestamp}.geojson")
    shp_dir = os.path.join(args.output_dir, "shp")
    shp_out = os.path.join(shp_dir, "predicao_entrelinhas.shp")
    shp_ts_out = os.path.join(shp_dir, f"predicao_entrelinhas_{timestamp}.shp")
    
    print("-" * 65)
    print(f"Salvando resultados em: {args.output_dir}")
    
    try:
        # GeoJSON (padrão e com timestamp)
        gdf_out.to_file(geojson_out, driver="GeoJSON")
        gdf_out.to_file(geojson_ts_out, driver="GeoJSON")
        print(f"[OK] GeoJSON salvo com sucesso: {geojson_out}")
        print(f"[OK] GeoJSON (timestamp) salvo com sucesso: {geojson_ts_out}")
    except Exception as e:
        print(f"[WARN] Falha ao salvar GeoJSON com GeoPandas: {e}. Tentando fallback direto...")
        try:
            with open(geojson_out, 'w', encoding='utf-8') as f:
                f.write(gdf_out.to_json())
            with open(geojson_ts_out, 'w', encoding='utf-8') as f:
                f.write(gdf_out.to_json())
            print(f"[OK] GeoJSON salvos via fallback do sistema.")
        except Exception as ex:
            print(f"[ERRO] Falha total ao exportar GeoJSON: {ex}")
            
    try:
        # Shapefile (padrão e com timestamp)
        os.makedirs(shp_dir, exist_ok=True)
        gdf_out.to_file(shp_out)
        gdf_out.to_file(shp_ts_out)
        print(f"[OK] Shapefile salvo com sucesso: {shp_out}")
        print(f"[OK] Shapefile (timestamp) salvo com sucesso: {shp_ts_out}")
    except Exception as e:
        print(f"[WARN] Falha ao salvar Shapefile: {e}")
        
    print("=" * 65)
    print("           PROCESSAMENTO CONCLUÍDO COM SUCESSO!")
    print("=" * 65)


if __name__ == "__main__":
    main()
