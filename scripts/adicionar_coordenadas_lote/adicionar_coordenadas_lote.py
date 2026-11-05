#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
adicionar_coordenadas_lote.py
--------------------------------------------------------------------------------
Script de utilidade para adicionar coordenadas geográficas (WGS84) e métricas (UTM)
a arquivos vetoriais (linhas e polígonos) de forma automática.
Permite opcionalmente informar um MDT raster para converter as geometrias em 3D
e preencher os atributos de altitude (Z).
Preserva as geometrias e atributos originais intactos (sem simplificar ou aplicar buffers),
apenas injetando/atualizando as colunas de coordenadas no arquivo.
--------------------------------------------------------------------------------
"""

import os
import sys
import argparse
import numpy as np
import geopandas as gpd
import rasterio
from shapely.geometry import Point, LineString, MultiLineString, Polygon, MultiPolygon
from pyproj import CRS, Transformer

def get_driver(filepath):
    """Retorna o driver do fiona adequado para a extensão do arquivo."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext == '.shp':
        return 'ESRI Shapefile'
    elif ext == '.geojson':
        return 'GeoJSON'
    elif ext == '.gpkg':
        return 'GPKG'
    return None

def make_line_3d(geom, mdt_src):
    """Converte LineString ou MultiLineString 2D em 3D usando o MDT raster."""
    if geom is None or geom.is_empty:
        return geom
    if geom.geom_type == 'LineString':
        coords = [(c[0], c[1]) for c in geom.coords]
        sampled = list(mdt_src.sample(coords))
        new_coords = []
        for (x, y), val in zip(coords, sampled):
            z = float(val[0])
            if np.isnan(z) or (mdt_src.nodata is not None and np.isclose(z, mdt_src.nodata)):
                z = 0.0
            new_coords.append((x, y, z))
        return LineString(new_coords)
    elif geom.geom_type == 'MultiLineString':
        new_lines = []
        for part in geom.geoms:
            new_lines.append(make_line_3d(part, mdt_src))
        return MultiLineString(new_lines)
    return geom

def make_polygon_3d(poly, mdt_src):
    """Converte Polygon ou MultiPolygon 2D em 3D usando o MDT raster."""
    if poly is None or poly.is_empty:
        return poly
    if poly.geom_type == 'Polygon':
        def process_ring(ring_coords):
            xy_coords = [(coords[0], coords[1]) for coords in ring_coords]
            sampled = list(mdt_src.sample(xy_coords))
            new_coords = []
            for (x, y), val in zip(xy_coords, sampled):
                z = float(val[0])
                if np.isnan(z) or (mdt_src.nodata is not None and np.isclose(z, mdt_src.nodata)):
                    z = 0.0
                new_coords.append((x, y, z))
            return new_coords

        ext_coords = list(poly.exterior.coords)
        new_ext = process_ring(ext_coords)
        
        new_ints = []
        for interior in poly.interiors:
            int_coords = list(interior.coords)
            new_ints.append(process_ring(int_coords))
            
        return Polygon(new_ext, new_ints)
    elif poly.geom_type == 'MultiPolygon':
        new_polys = []
        for part in poly.geoms:
            new_polys.append(make_polygon_3d(part, mdt_src))
        return MultiPolygon(new_polys)
    return poly

def process_spatial_file(filepath, mdt_path=None, output_path=None):
    """
    Processa um arquivo espacial (linhas ou polígonos) injetando as coordenadas e altitudes.
    Sem fazer nenhuma outra alteração nas geometrias originais ou campos de dados.
    """
    print(f"\n[LENDO] {filepath}")
    
    # 1. Ler o arquivo original
    try:
        gdf = gpd.read_file(filepath)
    except Exception as e:
        print(f"  [ERRO] Falha ao ler arquivo: {e}")
        return False
        
    if gdf.empty:
        print("  [AVISO] Arquivo vazio. Pulando...")
        return False

    # Guardar CRS original
    original_crs = gdf.crs
    if not original_crs:
        print("  [AVISO] CRS não definido. Assumindo EPSG:4326 (WGS84)...")
        gdf.crs = "EPSG:4326"
        original_crs = gdf.crs

    # 2. Definir CRS métrico e geográfico para os cálculos
    # Se o original já for projetado (métrico), usamos ele. Senão, estimamos UTM.
    if original_crs.is_projected:
        metric_crs = original_crs
    else:
        try:
            metric_crs = gdf.estimate_utm_crs()
        except Exception:
            metric_crs = CRS.from_user_input("EPSG:3857")

    if hasattr(metric_crs, "to_2d"):
        metric_crs = metric_crs.to_2d()

    # Abrir MDT se fornecido
    src_mdt = None
    if mdt_path:
        try:
            src_mdt = rasterio.open(mdt_path)
            print(f"  - MDT carregado: {os.path.basename(mdt_path)} (CRS: {src_mdt.crs.to_string()})")
        except Exception as e:
            print(f"  - [ERRO] Falha ao carregar o MDT: {e}")
            return False

    # Obter os DataFrames projetados para os cálculos de coordenadas e amostragem MDT
    gdf_wgs84 = gdf.to_crs("EPSG:4326")
    gdf_metric = gdf.to_crs(metric_crs)
    gdf_mdt = gdf.to_crs(src_mdt.crs) if src_mdt else None

    # Atualizar geometrias originais para 3D se o MDT estiver ativado
    if src_mdt:
        print("  - Atualizando geometrias para 3D (inserindo altitude Z do MDT)...")
        geoms_3d = []
        for idx in range(len(gdf)):
            geom_mdt = gdf_mdt.iloc[idx].geometry
            geom_orig = gdf.iloc[idx].geometry
            
            if geom_orig is None or geom_orig.is_empty:
                geoms_3d.append(geom_orig)
                continue
                
            if "linestring" in geom_orig.geom_type.lower():
                # Converte para 3D no CRS do MDT
                geom_mdt_3d = make_line_3d(geom_mdt, src_mdt)
                # Reprojeta de volta para o CRS original
                gdf_temp = gpd.GeoDataFrame(geometry=[geom_mdt_3d], crs=src_mdt.crs)
                geom_orig_3d = gdf_temp.to_crs(original_crs).iloc[0].geometry
                geoms_3d.append(geom_orig_3d)
            elif "polygon" in geom_orig.geom_type.lower():
                # Converte para 3D no CRS do MDT
                geom_mdt_3d = make_polygon_3d(geom_mdt, src_mdt)
                # Reprojeta de volta para o CRS original
                gdf_temp = gpd.GeoDataFrame(geometry=[geom_mdt_3d], crs=src_mdt.crs)
                geom_orig_3d = gdf_temp.to_crs(original_crs).iloc[0].geometry
                geoms_3d.append(geom_orig_3d)
            else:
                geoms_3d.append(geom_orig)
        gdf['geometry'] = geoms_3d

    # Identificar o tipo predominante de geometria
    geom_types = gdf.geometry.type.unique()
    is_line = any("linestring" in t.lower() for t in geom_types)
    is_polygon = any("polygon" in t.lower() for t in geom_types)

    print(f"  - CRS Original: {original_crs.to_string()}")
    print(f"  - CRS Métrico para cálculo: {metric_crs.to_string()}")
    print(f"  - Tipos de geometria: {list(geom_types)}")

    if is_line:
        print("  - Processando atributos de LINHA...")
        
        lat_list, lon_list = [], []
        x_list, y_list = [], []
        start_lat_list, start_lon_list = [], []
        start_x_list, start_y_list = [], []
        end_lat_list, end_lon_list = [], []
        end_x_list, end_y_list = [], []
        lengths_list = []
        
        # Z e Altitude
        z_list = []
        start_z_list, end_z_list = [], []
        min_z_list, max_z_list, mean_z_list = [], [], []
        slope_pct_list = []

        for idx in range(len(gdf)):
            geom_orig = gdf.iloc[idx].geometry
            geom_wgs = gdf_wgs84.iloc[idx].geometry
            geom_met = gdf_metric.iloc[idx].geometry
            geom_mdt_val = gdf_mdt.iloc[idx].geometry if gdf_mdt is not None else None

            if geom_orig is None or geom_orig.is_empty:
                for lst in [lat_list, lon_list, x_list, y_list, start_lat_list, start_lon_list,
                            start_x_list, start_y_list, end_lat_list, end_lon_list, end_x_list,
                            end_y_list, lengths_list, z_list, start_z_list, end_z_list,
                            min_z_list, max_z_list, mean_z_list, slope_pct_list]:
                    lst.append(0.0)
                continue

            # Comprimento métrico
            lengths_list.append(float(geom_met.length))

            # Extração de pontos WGS84
            if geom_wgs.geom_type == 'LineString':
                w_start = geom_wgs.coords[0]
                w_end = geom_wgs.coords[-1]
                w_mid_pt = geom_wgs.interpolate(0.5, normalized=True)
                w_mid = (w_mid_pt.x, w_mid_pt.y)
            elif geom_wgs.geom_type == 'MultiLineString':
                parts = list(geom_wgs.geoms)
                w_start = parts[0].coords[0] if parts else (0.0, 0.0)
                w_end = parts[-1].coords[-1] if parts else (0.0, 0.0)
                w_mid_pt = geom_wgs.interpolate(0.5, normalized=True)
                w_mid = (w_mid_pt.x, w_mid_pt.y)
            else:
                w_start = w_end = w_mid = (0.0, 0.0)

            start_lon_list.append(float(w_start[0]))
            start_lat_list.append(float(w_start[1]))
            end_lon_list.append(float(w_end[0]))
            end_lat_list.append(float(w_end[1]))
            lon_list.append(float(w_mid[0]))
            lat_list.append(float(w_mid[1]))

            # Extração de pontos métricos (UTM)
            if geom_met.geom_type == 'LineString':
                m_start = geom_met.coords[0]
                m_end = geom_met.coords[-1]
                m_mid_pt = geom_met.interpolate(0.5, normalized=True)
                m_mid = (m_mid_pt.x, m_mid_pt.y)
            elif geom_met.geom_type == 'MultiLineString':
                parts = list(geom_met.geoms)
                m_start = parts[0].coords[0] if parts else (0.0, 0.0)
                m_end = parts[-1].coords[-1] if parts else (0.0, 0.0)
                m_mid_pt = geom_met.interpolate(0.5, normalized=True)
                m_mid = (m_mid_pt.x, m_mid_pt.y)
            else:
                m_start = m_end = m_mid = (0.0, 0.0)

            start_x_list.append(float(m_start[0]))
            start_y_list.append(float(m_start[1]))
            end_x_list.append(float(m_end[0]))
            end_y_list.append(float(m_end[1]))
            x_list.append(float(m_mid[0]))
            y_list.append(float(m_mid[1]))

            # Extração de altitude Z (MDT)
            if src_mdt and geom_mdt_val:
                # Amostrar altitudes para toda a linha
                if geom_mdt_val.geom_type == 'LineString':
                    pts = list(geom_mdt_val.coords)
                    mid_pt_mdt = geom_mdt_val.interpolate(0.5, normalized=True)
                    pts_to_sample = pts + [(mid_pt_mdt.x, mid_pt_mdt.y)]
                elif geom_mdt_val.geom_type == 'MultiLineString':
                    parts = list(geom_mdt_val.geoms)
                    pts = []
                    for part in parts:
                        pts.extend(list(part.coords))
                    mid_pt_mdt = geom_mdt_val.interpolate(0.5, normalized=True)
                    pts_to_sample = pts + [(mid_pt_mdt.x, mid_pt_mdt.y)]
                else:
                    pts_to_sample = []

                if pts_to_sample:
                    sampled = list(src_mdt.sample(pts_to_sample))
                    z_vals = []
                    for val in sampled[:-1]:
                        val_z = float(val[0])
                        if np.isnan(val_z) or (src_mdt.nodata is not None and np.isclose(val_z, src_mdt.nodata)):
                            val_z = 0.0
                        z_vals.append(val_z)
                    
                    mid_z = float(sampled[-1][0])
                    if np.isnan(mid_z) or (src_mdt.nodata is not None and np.isclose(mid_z, src_mdt.nodata)):
                        mid_z = 0.0
                        
                    s_z = z_vals[0] if z_vals else 0.0
                    e_z = z_vals[-1] if z_vals else 0.0
                    min_z = float(np.min(z_vals)) if z_vals else 0.0
                    max_z = float(np.max(z_vals)) if z_vals else 0.0
                    mean_z = float(np.mean(z_vals)) if z_vals else 0.0
                    
                    horiz_len = geom_met.length
                    slope = ((e_z - s_z) / horiz_len) * 100 if horiz_len > 0 else 0.0
                else:
                    s_z = e_z = mid_z = min_z = max_z = mean_z = slope = 0.0
            else:
                # Sem MDT, mantemos os valores de Z se já estiverem na geometria
                if geom_orig.has_z:
                    if geom_orig.geom_type == 'LineString':
                        z_vals = [c[2] for c in geom_orig.coords if len(c) >= 3]
                        mid_pt_z = geom_orig.interpolate(0.5, normalized=True)
                        mid_z = mid_pt_z.z if hasattr(mid_pt_z, 'z') else (z_vals[len(z_vals)//2] if z_vals else 0.0)
                    else:
                        z_vals = []
                        for part in geom_orig.geoms:
                            z_vals.extend([c[2] for c in part.coords if len(c) >= 3])
                        mid_pt_z = geom_orig.interpolate(0.5, normalized=True)
                        mid_z = mid_pt_z.z if hasattr(mid_pt_z, 'z') else (z_vals[len(z_vals)//2] if z_vals else 0.0)
                        
                    s_z = z_vals[0] if z_vals else 0.0
                    e_z = z_vals[-1] if z_vals else 0.0
                    min_z = float(np.min(z_vals)) if z_vals else 0.0
                    max_z = float(np.max(z_vals)) if z_vals else 0.0
                    mean_z = float(np.mean(z_vals)) if z_vals else 0.0
                    horiz_len = geom_met.length
                    slope = ((e_z - s_z) / horiz_len) * 100 if horiz_len > 0 else 0.0
                else:
                    s_z = e_z = mid_z = min_z = max_z = mean_z = slope = 0.0

            start_z_list.append(s_z)
            end_z_list.append(e_z)
            z_list.append(mid_z)
            min_z_list.append(min_z)
            max_z_list.append(max_z)
            mean_z_list.append(mean_z)
            slope_pct_list.append(slope)

        # Atualizar/adicionar colunas no original
        gdf['lat'] = lat_list
        gdf['lon'] = lon_list
        gdf['x'] = x_list
        gdf['y'] = y_list
        gdf['start_lat'] = start_lat_list
        gdf['start_lon'] = start_lon_list
        gdf['start_x'] = start_x_list
        gdf['start_y'] = start_y_list
        gdf['end_lat'] = end_lat_list
        gdf['end_lon'] = end_lon_list
        gdf['end_x'] = end_x_list
        gdf['end_y'] = end_y_list
        gdf['Length'] = lengths_list

        # Colunas de altitude (Z)
        gdf['z'] = z_list
        gdf['start_z'] = start_z_list
        gdf['end_z'] = end_z_list
        gdf['min_z'] = min_z_list
        gdf['max_z'] = max_z_list
        gdf['mean_z'] = mean_z_list
        gdf['slope_pct'] = slope_pct_list

    elif is_polygon:
        print("  - Processando atributos de POLÍGONO...")
        
        lat_list, lon_list = [], []
        x_list, y_list = [], []
        z_list = []
        min_z_list, max_z_list, mean_z_list = [], [], []

        for idx in range(len(gdf)):
            geom_orig = gdf.iloc[idx].geometry
            geom_wgs = gdf_wgs84.iloc[idx].geometry
            geom_met = gdf_metric.iloc[idx].geometry
            geom_mdt_val = gdf_mdt.iloc[idx].geometry if gdf_mdt is not None else None

            if geom_orig is None or geom_orig.is_empty:
                for lst in [lat_list, lon_list, x_list, y_list, z_list, min_z_list, max_z_list, mean_z_list]:
                    lst.append(0.0)
                continue

            # Centróide WGS84
            w_centroid = geom_wgs.centroid
            lon_list.append(float(w_centroid.x))
            lat_list.append(float(w_centroid.y))

            # Centróide métrico (UTM)
            m_centroid = geom_met.centroid
            x_list.append(float(m_centroid.x))
            y_list.append(float(m_centroid.y))

            # Centróide MDT Z e Estatísticas de Altitude
            if src_mdt and geom_mdt_val:
                # Amostrar Z do centróide
                sampled_mid = list(src_mdt.sample([(geom_mdt_val.centroid.x, geom_mdt_val.centroid.y)]))
                cz = float(sampled_mid[0][0])
                if np.isnan(cz) or (src_mdt.nodata is not None and np.isclose(cz, src_mdt.nodata)):
                    cz = 0.0
                
                # Amostrar Z de todos os vértices do polígono
                poly_z_vals = []
                
                def extract_poly_z(poly_g):
                    if poly_g.geom_type == 'Polygon':
                        coords = list(poly_g.exterior.coords)
                        for interior in poly_g.interiors:
                            coords.extend(list(interior.coords))
                        return coords
                    elif poly_g.geom_type == 'MultiPolygon':
                        coords = []
                        for part in poly_g.geoms:
                            coords.extend(extract_poly_z(part))
                        return coords
                    return []

                xy_coords = [(c[0], c[1]) for c in extract_poly_z(geom_mdt_val)]
                if xy_coords:
                    sampled_poly = list(src_mdt.sample(xy_coords))
                    for val in sampled_poly:
                        val_z = float(val[0])
                        if np.isnan(val_z) or (src_mdt.nodata is not None and np.isclose(val_z, src_mdt.nodata)):
                            val_z = 0.0
                        poly_z_vals.append(val_z)

                min_z = float(np.min(poly_z_vals)) if poly_z_vals else 0.0
                max_z = float(np.max(poly_z_vals)) if poly_z_vals else 0.0
                mean_z = float(np.mean(poly_z_vals)) if poly_z_vals else 0.0
            else:
                # Sem MDT, usar valores existentes da geometria se houver Z
                if geom_orig.has_z:
                    poly_z_vals = []
                    
                    def extract_z_from_geom(g):
                        if g.geom_type == 'Polygon':
                            coords = [c[2] for c in g.exterior.coords if len(c) >= 3]
                            for interior in g.interiors:
                                coords.extend([c[2] for c in interior.coords if len(c) >= 3])
                            return coords
                        elif g.geom_type == 'MultiPolygon':
                            coords = []
                            for part in g.geoms:
                                coords.extend(extract_z_from_geom(part))
                            return coords
                        return []

                    poly_z_vals = extract_z_from_geom(geom_orig)
                    cz = geom_orig.centroid.z if hasattr(geom_orig.centroid, 'z') else (poly_z_vals[0] if poly_z_vals else 0.0)
                    min_z = float(np.min(poly_z_vals)) if poly_z_vals else 0.0
                    max_z = float(np.max(poly_z_vals)) if poly_z_vals else 0.0
                    mean_z = float(np.mean(poly_z_vals)) if poly_z_vals else 0.0
                else:
                    cz = min_z = max_z = mean_z = 0.0

            z_list.append(cz)
            min_z_list.append(min_z)
            max_z_list.append(max_z)
            mean_z_list.append(mean_z)

        # Atualizar/adicionar colunas no original
        gdf['lat'] = lat_list
        gdf['lon'] = lon_list
        gdf['x'] = x_list
        gdf['y'] = y_list
        gdf['z'] = z_list
        gdf['min_z'] = min_z_list
        gdf['max_z'] = max_z_list
        gdf['mean_z'] = mean_z_list
        
    else:
        print("  - [AVISO] Nenhuma geometria de linha ou polígono encontrada para processamento. Pulando...")
        if src_mdt:
            src_mdt.close()
        return False

    if src_mdt:
        src_mdt.close()

    # 3. Salvar os resultados preservando o formato original e CRS original
    target_path = output_path if output_path else filepath
    driver = get_driver(target_path)
    
    print(f"  - Gravando de volta para: {target_path} (Driver: {driver})")
    try:
        # Garantir diretório de destino
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        # Salva o gdf no CRS original e com as novas colunas
        gdf.to_file(target_path, driver=driver)
        print("  - [SUCESSO] Arquivo atualizado com coordenadas.")
        return True
    except Exception as e:
        print(f"  - [ERRO] Falha ao salvar arquivo: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Adiciona colunas de coordenadas (lat, lon, x, y, start_lat, start_lon, end_lat, end_lon, Length, z, min_z, max_z, mean_z) a feições espaciais sem modificar geometria de contorno ou outros atributos de dados."
    )
    parser.add_argument("-i", "--input_dir", required=True, help="Diretório de entrada contendo arquivos espaciais (.shp, .geojson, .gpkg)")
    parser.add_argument("-m", "--mdt", help="Caminho do arquivo MDT raster (.tif) opcional para extrair altitudes Z.")
    parser.add_argument("-o", "--output_dir", help="Diretório de saída. Se não informado, modifica os arquivos originais in-place.")
    parser.add_argument("--overwrite", action="store_true", help="Sobrescrever arquivos de entrada sem confirmação manual.")
    args = parser.parse_args()

    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir) if args.output_dir else None
    mdt_path = os.path.abspath(args.mdt) if args.mdt else None

    if not os.path.exists(input_dir):
        print(f"[ERRO] Diretório de entrada não existe: {input_dir}")
        sys.exit(1)

    if mdt_path and not os.path.exists(mdt_path):
        print(f"[ERRO] MDT não encontrado no caminho: {mdt_path}")
        sys.exit(1)

    # Coletar arquivos espaciais válidos recursivamente
    spatial_files = []
    valid_extensions = ('.shp', '.geojson', '.gpkg')
    
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(valid_extensions):
                spatial_files.append(os.path.join(root, file))

    if not spatial_files:
        print(f"[AVISO] Nenhum arquivo espacial (.shp, .geojson, .gpkg) encontrado em: {input_dir}")
        sys.exit(0)

    print("=" * 80)
    print(f"   CITROSROWS AI - INJEÇÃO DE COORDENADAS E ALTITUDES VETORIAIS")
    print("=" * 80)
    print(f"Diretório de Entrada: {input_dir}")
    print(f"Arquivos localizados: {len(spatial_files)}")
    if mdt_path:
        print(f"MDT Raster:           {mdt_path}")
    if output_dir:
        print(f"Diretório de Saída:   {output_dir}")
    else:
        print("Modo de execução:     SOBRESCREVER EM PASTA ORIGINAL (in-place)")
        if not args.overwrite:
            confirm = input("\nATENÇÃO: Os arquivos originais serão modificados. Deseja continuar? (s/n): ")
            if confirm.lower() != 's':
                print("[CANCELADO] Execução cancelada pelo usuário.")
                sys.exit(0)

    success_count = 0
    fail_count = 0

    for filepath in spatial_files:
        # Calcular caminho de saída
        if output_dir:
            # Manter a estrutura de pastas relativas
            rel_path = os.path.relpath(filepath, input_dir)
            out_filepath = os.path.join(output_dir, rel_path)
        else:
            out_filepath = filepath

        if process_spatial_file(filepath, mdt_path, out_filepath):
            success_count += 1
        else:
            fail_count += 1

    print("\n" + "=" * 80)
    print(f"   PROCESSAMENTO CONCLUÍDO!")
    print(f"   Sucessos: {success_count} | Falhas/Ignorados: {fail_count}")
    print("=" * 80)

if __name__ == "__main__":
    main()
