#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ordenar_linhas_norte_sul.py
--------------------------------------------------------------------------------
Script de pós-processamento geográfico para ler uma pasta mãe contendo subpastas,
identificar arquivos espaciais contendo geometrias de linha (LineString/MultiLineString)
e de polígono (Polygon/MultiPolygon). 
As linhas são ordenadas geometricamente (ex: Norte para Sul), recebendo o campo
'orden_geome' e salvas com o mesmo nome original na nova estrutura.
Os polígonos são copiados intactos com todos os seus arquivos auxiliares.
--------------------------------------------------------------------------------
"""

import os
import sys
import shutil
import argparse
import geopandas as gpd

def sanitize_filename(name):
    """
    Remove caracteres inválidos para nomes de arquivos e pastas no Windows.
    Garante que não haja hífens (-) no nome resultante.
    """
    if not name:
        return ""
    name = str(name).replace("-", "")
    name = name.replace(" ", "_")
    name = "".join(c for c in name if c.isalnum() or c == "_")
    while "__" in name:
        name = name.replace("__", "_")
    return name.strip("_")

def main():
    parser = argparse.ArgumentParser(
        description="Ordena geometricamente linhas em subpastas de uma pasta mãe, copiando polígonos intactos.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--input_dir", "-i", required=True,
        help="Caminho do diretório mãe contendo as subpastas."
    )
    parser.add_argument(
        "--output_dir", "-o", required=True,
        help="Diretório de saída (nova pasta mãe) onde a estrutura será salva."
    )
    parser.add_argument(
        "--direction", "-d", default="N-S", choices=["N-S", "S-N", "W-E", "E-W"],
        help="Direção do ordenamento geométrico: N-S (Norte-Sul), S-N (Sul-Norte), W-E (Oeste-Leste), E-W (Leste-Oeste)."
    )

    args = parser.parse_args()

    # 1. Validar caminhos
    if not os.path.exists(args.input_dir):
        print(f"[ERRO] Diretório de entrada não encontrado: {args.input_dir}")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 75)
    print("   CITROSROWS AI - REPLICAÇÃO E ORDENAMENTO GEOMÉTRICO DE LINHAS")
    print("=" * 75)
    print(f"Diretório Mãe (Entrada) : {args.input_dir}")
    print(f"Diretório Mãe (Saída)   : {args.output_dir}")
    print(f"Direção do Ordenamento  : {args.direction}")
    print("-" * 75)

    # 2. Encontrar todas as subpastas
    subdirs = [d for d in os.listdir(args.input_dir) if os.path.isdir(os.path.join(args.input_dir, d))]
    
    # Se a pasta de saída estiver dentro da entrada, ignore-a
    output_dir_name = os.path.basename(os.path.normpath(args.output_dir))
    subdirs = [d for d in subdirs if os.path.normpath(d) != os.path.normpath(output_dir_name)]

    if not subdirs:
        # Se não houver subpastas, trata a própria input_dir como a única pasta de trabalho
        print("[INFO] Nenhuma subpasta encontrada. Processando arquivos diretamente no diretório de entrada.")
        subdirs = ["."]

    processed_files_count = 0
    copied_files_count = 0

    for idx, subdir in enumerate(subdirs):
        sub_path_in = os.path.normpath(os.path.join(args.input_dir, subdir))
        
        subdir_clean = sanitize_filename(subdir) if subdir != "." else ""
        sub_path_out = os.path.normpath(os.path.join(args.output_dir, subdir_clean))
        os.makedirs(sub_path_out, exist_ok=True)

        # Listar arquivos espaciais na subpasta
        spatial_files = []
        valid_extensions = ('.shp', '.geojson', '.gpkg')
        
        if os.path.exists(sub_path_in):
            for f in os.listdir(sub_path_in):
                if f.lower().endswith(valid_extensions):
                    spatial_files.append(f)

        if not spatial_files:
            continue

        print(f"\n>> [{idx+1}/{len(subdirs)}] Processando Pasta: '{subdir}'")

        for f in spatial_files:
            file_path_in = os.path.join(sub_path_in, f)
            file_path_out = os.path.join(sub_path_out, f)

            try:
                gdf = gpd.read_file(file_path_in)
                if gdf.empty:
                    # Copiar arquivo vazio diretamente
                    shutil.copy(file_path_in, file_path_out)
                    continue

                # Identificar tipo de geometria (LineString/MultiLineString vs Polygon/MultiPolygon)
                first_geom = gdf.geometry.iloc[0]
                geom_type = first_geom.geom_type if first_geom is not None else ""

                if "Line" in geom_type:
                    # CASO LINHAS: Ordenar e injetar campo orden_geome
                    print(f"  - [LINHAS] Processando: '{f}' ({len(gdf)} feições, tipo: {geom_type})")
                    
                    # Calcular coordenadas do centróide para ordenamento
                    centroids = gdf.geometry.centroid
                    gdf['_centroid_x'] = centroids.x
                    gdf['_centroid_y'] = centroids.y

                    # Aplicar ordenamento geométrico
                    if args.direction == "N-S":
                        # Norte para Sul: Maior Latitude (Y) primeiro
                        gdf_sorted = gdf.sort_values(by='_centroid_y', ascending=False).copy()
                    elif args.direction == "S-N":
                        # Sul para Norte: Menor Latitude (Y) primeiro
                        gdf_sorted = gdf.sort_values(by='_centroid_y', ascending=True).copy()
                    elif args.direction == "W-E":
                        # Oeste para Leste: Menor Longitude (X) primeiro
                        gdf_sorted = gdf.sort_values(by='_centroid_x', ascending=True).copy()
                    elif args.direction == "E-W":
                        # Leste para Oeste: Maior Longitude (X) primeiro
                        gdf_sorted = gdf.sort_values(by='_centroid_x', ascending=False).copy()
                    else:
                        gdf_sorted = gdf.copy()

                    # Criar campo orden_geome sequencial (iniciando em 1)
                    gdf_sorted['orden_geome'] = range(1, len(gdf_sorted) + 1)

                    # Remover colunas auxiliares
                    gdf_sorted = gdf_sorted.drop(columns=['_centroid_x', '_centroid_y'])

                    # Salvar com o mesmo nome original
                    gdf_sorted.to_file(file_path_out)
                    print(f"    [OK] Salvo e ordenado em: '{file_path_out}'")
                    processed_files_count += 1

                elif "Polygon" in geom_type:
                    # CASO POLÍGONOS: Copiar intacto com todos os arquivos auxiliares (SHP, DBF, PRJ, SHX, CPG, etc.)
                    print(f"  - [POLÍGONO] Copiando intacto: '{f}' (tipo: {geom_type})")
                    base_name = os.path.splitext(f)[0]
                    
                    copied_aux_count = 0
                    for file_in_sub in os.listdir(sub_path_in):
                        # Copia todos os arquivos que compartilham o mesmo prefixo
                        if file_in_sub.startswith(base_name + "."):
                            src_file = os.path.join(sub_path_in, file_in_sub)
                            dst_file = os.path.join(sub_path_out, file_in_sub)
                            shutil.copy(src_file, dst_file)
                            copied_aux_count += 1
                            
                    print(f"    [OK] Copiados {copied_aux_count} arquivos auxiliares em: '{sub_path_out}'")
                    copied_files_count += 1
                
                else:
                    # Outro tipo de geometria (ex: Pontos) -> Apenas copiar
                    print(f"  - [OUTRO] Copiando arquivo de geometria '{geom_type}': '{f}'")
                    shutil.copy(file_path_in, file_path_out)
                    copied_files_count += 1

            except Exception as e:
                print(f"    [ERRO] Falha ao processar arquivo '{f}': {e}")

    print("\n" + "=" * 75)
    print(f"   REPLICAÇÃO E ORDENAMENTO CONCLUÍDOS COM SUCESSO!")
    print(f"   - Arquivos de linhas ordenados e salvos: {processed_files_count}")
    print(f"   - Arquivos de polígonos copiados intactos: {copied_files_count}")
    print("=" * 75)

if __name__ == "__main__":
    main()
