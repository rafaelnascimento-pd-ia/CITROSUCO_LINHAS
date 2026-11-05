#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
separar_entrelinhas_por_talhao.py
--------------------------------------------------------------------------------
Script de pós-processamento geográfico para filtrar as entrelinhas finais de
plantio para cada talhão, mantendo todos os atributos originais intactos,
aplicando um buffer de 6.5 metros e salvando os resultados em subpastas
organizadas no formato Shapefile (.shp), em WGS84 (EPSG:4326) e em Single Part.
--------------------------------------------------------------------------------
"""

import os
import sys
import argparse
import datetime
import geopandas as gpd
from shapely.geometry import MultiPolygon, MultiLineString

def sanitize_filename(name):
    """
    Remove caracteres inválidos para nomes de arquivos e pastas no Windows.
    Garante que não haja hífens (-) no nome resultante.
    """
    if not name:
        return ""
    # Remover hífens (-)
    name = str(name).replace("-", "")
    # Substituir espaços por underscore
    name = name.replace(" ", "_")
    # Manter apenas alfanuméricos e underscores
    name = "".join(c for c in name if c.isalnum() or c == "_")
    # Reduzir múltiplos underscores consecutivos
    while "__" in name:
        name = name.replace("__", "_")
    # Remover underscores das pontas
    return name.strip("_")

def main():
    parser = argparse.ArgumentParser(
        description="Filtra e separa entrelinhas finais por talhão usando o campo BALAYER.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--input_lines", "-i", required=True,
        help="Caminho do arquivo de entrelinhas (GeoJSON ou Shapefile)."
    )
    parser.add_argument(
        "--talhao_path", "-t", required=True,
        help="Caminho do arquivo de talões (GeoJSON ou Shapefile) contendo o campo BALAYER."
    )
    parser.add_argument(
        "--output_dir", "-o", default="inferencia/entrelinhas_separadas",
        help="Diretório base de saída onde as subpastas dos talhões serão salvas."
    )
    parser.add_argument(
        "--buffer_distance", "-b", type=float, default=6.5,
        help="Distância do buffer em metros para o talhão."
    )
    parser.add_argument(
        "--timestamp", default=None,
        help="Sufixo opcional de data/hora nos nomes dos arquivos (formato YYYYMMDD_HHMMSS)."
    )

    args = parser.parse_args()

    # 1. Validar caminhos
    if not os.path.exists(args.input_lines):
        print(f"[ERRO] Arquivo de entrelinhas não encontrado: {args.input_lines}")
        sys.exit(1)
    if not os.path.exists(args.talhao_path):
        print(f"[ERRO] Arquivo de talhões não encontrado: {args.talhao_path}")
        sys.exit(1)

    print("=" * 75)
    print("   CITROSROWS AI - SEPARAÇÃO DE ENTRELINHAS POR TALHÃO (BALAYER)")
    print("=" * 75)
    print(f"Entrelinhas de Entrada  : {args.input_lines}")
    print(f"Arquivo de Talhões      : {args.talhao_path}")
    print(f"Diretório de Saída      : {args.output_dir}")
    print(f"Buffer do Talhão        : {args.buffer_distance}m")
    print("-" * 75)

    # 2. Carregar dados
    try:
        print("[INFO] Carregando arquivos...")
        gdf_lines = gpd.read_file(args.input_lines)
        gdf_talhao = gpd.read_file(args.talhao_path)
    except Exception as e:
        print(f"[ERRO] Falha ao carregar os dados: {e}")
        sys.exit(1)

    if gdf_lines.empty:
        print("[WARN] Arquivo de entrelinhas está vazio.")
        sys.exit(0)
    if gdf_talhao.empty:
        print("[WARN] Arquivo de talhões está vazio.")
        sys.exit(0)

    print(f"[OK] {len(gdf_lines)} entrelinhas carregadas.")
    print(f"[OK] {len(gdf_talhao)} talhões carregados.")

    # 3. Validar se o campo BALAYER existe nos talhões (case-insensitive)
    balayer_col = None
    for col in gdf_talhao.columns:
        if col.upper() == 'BALAYER':
            balayer_col = col
            break

    if not balayer_col:
        print("[WARN] Campo 'BALAYER' não encontrado nos atributos do talhão.")
        # Procurar alternativas comuns caso não exista
        alternatives = ['TALHAO', 'talhao', 'ID', 'id', 'NOME', 'nome', 'CD_TALHAO']
        for col in gdf_talhao.columns:
            if col in alternatives:
                balayer_col = col
                print(f"[INFO] Utilizando coluna '{balayer_col}' como alternativa para identificação.")
                break
        if not balayer_col:
            # Fallback final: criar coluna com índice
            gdf_talhao['id_temp'] = [f"talhao_{i+1}" for i in range(len(gdf_talhao))]
            balayer_col = 'id_temp'
            print("[INFO] Nenhuma coluna identificadora encontrada. Criando ids temporários.")
    else:
        print(f"[OK] Campo de identificação encontrado: '{balayer_col}'")

    # 4. Determinar CRS métrico para cálculo de intersecção e buffer precisos
    original_lines_crs = gdf_lines.crs
    original_talhao_crs = gdf_talhao.crs

    # Usar UTM local estimado a partir das linhas ou do talhão
    try:
        if original_lines_crs and original_lines_crs.is_geographic:
            metric_crs = gdf_lines.estimate_utm_crs()
        elif original_talhao_crs and original_talhao_crs.is_geographic:
            metric_crs = gdf_talhao.estimate_utm_crs()
        elif original_lines_crs:
            metric_crs = original_lines_crs
        else:
            metric_crs = "EPSG:3857"
    except Exception:
        metric_crs = "EPSG:3857"

    print(f"[INFO] CRS métrico para cálculos espaciais: {metric_crs}")

    # Reprojetar para o CRS métrico
    print("[INFO] Reprojetando dados para o CRS métrico...")
    gdf_lines_m = gdf_lines.to_crs(metric_crs)
    gdf_talhao_m = gdf_talhao.to_crs(metric_crs)

    # 5. Processamento por talhão
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = args.timestamp
    if not timestamp:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    processed_count = 0

    for idx, row in gdf_talhao_m.iterrows():
        geom_talhao = row.geometry
        if geom_talhao is None or geom_talhao.is_empty:
            continue

        # Obter nome do talhão
        raw_name = row[balayer_col]
        if raw_name is None or str(raw_name).strip() == "":
            talhao_name = f"talhao_{idx+1}"
        else:
            talhao_name = sanitize_filename(raw_name)
            if not talhao_name:
                talhao_name = f"talhao_{idx+1}"

        print(f"\n>> Processando Talhão: '{talhao_name}' ({idx+1}/{len(gdf_talhao_m)})")

        # Criar a subpasta do talhão
        talhao_dir = os.path.join(args.output_dir, talhao_name)
        os.makedirs(talhao_dir, exist_ok=True)

        # A. Filtrar entrelinhas que tocam ou estão dentro do talhão original (sem buffer)
        print("  - Filtrando entrelinhas dentro do talhão...")
        # Usamos intersects para selecionar as feições
        intersects_mask = gdf_lines_m.geometry.intersects(geom_talhao)
        gdf_lines_filtered = gdf_lines_m[intersects_mask].copy()

        print(f"  - Encontradas {len(gdf_lines_filtered)} entrelinhas.")

        # Gerar polígono com buffer de 6.5m
        print(f"  - Gerando buffer de {args.buffer_distance}m no talhão...")
        geom_buffered = geom_talhao.buffer(args.buffer_distance)

        # B. Explodir feições para garantir Single Part (explode de MultiLineStrings e MultiPolygons)
        # Para as linhas:
        if not gdf_lines_filtered.empty:
            gdf_lines_single = gdf_lines_filtered.explode(index_parts=False)
        else:
            gdf_lines_single = gpd.GeoDataFrame(columns=gdf_lines_filtered.columns, crs=metric_crs)

        # Para o talhão original (preservando atributos):
        gdf_talhao_single = gdf_talhao_m.loc[[idx]].explode(index_parts=False)
        
        # Para o talhão com buffer (preservando atributos):
        gdf_talhao_buf = gdf_talhao_m.loc[[idx]].copy()
        gdf_talhao_buf.geometry = [geom_buffered]
        gdf_talhao_buf_single = gdf_talhao_buf.explode(index_parts=False)

        # C. Reprojetar tudo para WGS84 (EPSG:4326) em graus decimais
        print("  - Reprojetando geometrias finais para WGS84 (EPSG:4326)...")
        gdf_lines_wgs84 = gdf_lines_single.to_crs("EPSG:4326")
        gdf_talhao_wgs84 = gdf_talhao_single.to_crs("EPSG:4326")
        gdf_talhao_buf_wgs84 = gdf_talhao_buf_single.to_crs("EPSG:4326")

        # D. Definir nomes de arquivos de saída
        lines_shp_name = f"Linhas_{talhao_name}.shp"
        talhao_buf_shp_name = f"TALHAO_{talhao_name}.shp"
        talhao_orig_shp_name = f"TALHAO_{talhao_name}_original.shp"

        lines_out_path = os.path.join(talhao_dir, lines_shp_name)
        talhao_buf_out_path = os.path.join(talhao_dir, talhao_buf_shp_name)
        talhao_orig_out_path = os.path.join(talhao_dir, talhao_orig_shp_name)

        # E. Salvar como Shapefile (.shp)
        print("  - Salvando arquivos em formato Shapefile...")

        # 1. Salvar entrelinhas (se houver alguma)
        if not gdf_lines_wgs84.empty:
            try:
                gdf_lines_wgs84.to_file(lines_out_path)
                print(f"    [OK] Salvo: {lines_shp_name}")
            except Exception as e:
                print(f"    [ERRO] Falha ao salvar entrelinhas: {e}")
        else:
            print("    [INFO] Nenhuma entrelinha para salvar neste talhão.")

        # 2. Salvar talhão original
        try:
            gdf_talhao_wgs84.to_file(talhao_orig_out_path)
            print(f"    [OK] Salvo: {talhao_orig_shp_name}")
        except Exception as e:
            print(f"    [ERRO] Falha ao salvar talhão original: {e}")

        # 3. Salvar talhão com buffer
        try:
            gdf_talhao_buf_wgs84.to_file(talhao_buf_out_path)
            print(f"    [OK] Salvo: {talhao_buf_shp_name}")
        except Exception as e:
            print(f"    [ERRO] Falha ao salvar talhão com buffer: {e}")

        processed_count += 1

    print("\n" + "=" * 75)
    print(f"   PROCESSAMENTO CONCLUÍDO! {processed_count} TALHÕES PROCESSADOS.")
    print("=" * 75)

if __name__ == "__main__":
    main()
