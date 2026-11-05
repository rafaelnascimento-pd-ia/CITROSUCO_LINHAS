#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
adicionar_coordenadas_mdt.py
--------------------------------------------------------------------------------
Script para adicionar coordenadas (Z / altitude do MDT, Lat/Lon WGS84, X/Y UTM)
em geometrias de LINHAS e POLÍGONOS.

ESTE SCRIPT NÃO ALTERA NENHUM ARQUIVO ORIGINAL.
Os resultados são salvos no diretório de saída espelhando a estrutura de subpastas
e mantendo os mesmos nomes originais dos arquivos.
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
    return 'GeoJSON'


def process_line_3d(geom, src_mdt):
    """Converte LineString ou MultiLineString 2D para 3D (Z do MDT)."""
    if geom is None or geom.is_empty:
        return geom

    if geom.geom_type == 'LineString':
        coords = [(c[0], c[1]) for c in geom.coords]
        sampled = list(src_mdt.sample(coords))
        new_coords = []
        nodata = src_mdt.nodata
        for (x, y), val in zip(coords, sampled):
            z = float(val[0])
            if np.isnan(z) or (nodata is not None and np.isclose(z, nodata)):
                z = 0.0
            new_coords.append((x, y, z))
        return LineString(new_coords)

    elif geom.geom_type == 'MultiLineString':
        new_parts = []
        for part in geom.geoms:
            new_parts.append(process_line_3d(part, src_mdt))
        return MultiLineString(new_parts)

    return geom


def process_polygon_3d(geom, src_mdt):
    """Converte Polygon ou MultiPolygon 2D para 3D (Z do MDT)."""
    if geom is None or geom.is_empty:
        return geom

    nodata = src_mdt.nodata

    def sample_ring(ring_coords):
        xy_coords = [(c[0], c[1]) for c in ring_coords]
        sampled = list(src_mdt.sample(xy_coords))
        new_coords = []
        for (x, y), val in zip(xy_coords, sampled):
            z = float(val[0])
            if np.isnan(z) or (nodata is not None and np.isclose(z, nodata)):
                z = 0.0
            new_coords.append((x, y, z))
        return new_coords

    if geom.geom_type == 'Polygon':
        ext_coords = list(geom.exterior.coords)
        new_ext = sample_ring(ext_coords)
        new_ints = []
        for interior in geom.interiors:
            new_ints.append(sample_ring(list(interior.coords)))
        return Polygon(new_ext, new_ints)

    elif geom.geom_type == 'MultiPolygon':
        new_parts = []
        for part in geom.geoms:
            new_parts.append(process_polygon_3d(part, src_mdt))
        return MultiPolygon(new_parts)

    return geom


def process_single_file(input_file, mdt_path, output_file):
    """
    Processa um arquivo vetorial (linhas ou polígonos),
    atribui coordenadas 3D via MDT e salva no caminho de saída mantendo o nome original.
    """
    print(f"\n[PROCESSANDO] {input_file}")
    if not os.path.exists(input_file):
        print(f"  [ERRO] Arquivo de entrada não existe: {input_file}")
        return False

    if not os.path.exists(mdt_path):
        print(f"  [ERRO] Raster MDT não encontrado: {mdt_path}")
        return False

    try:
        gdf = gpd.read_file(input_file)
        src_mdt = rasterio.open(mdt_path)
    except Exception as e:
        print(f"  [ERRO] Falha ao abrir arquivos: {e}")
        return False

    if gdf.empty:
        print("  [AVISO] Arquivo vetorial está vazio. Pulando.")
        src_mdt.close()
        return False

    original_crs = gdf.crs
    if not original_crs:
        print("  [AVISO] CRS não definido no arquivo original. Assumindo EPSG:4326.")
        original_crs = CRS.from_user_input("EPSG:4326")
        gdf.crs = original_crs

    print(f"  - Total de feições: {len(gdf)}")
    print(f"  - CRS original: {original_crs.to_string()}")
    print(f"  - MDT CRS: {src_mdt.crs.to_string()} | Resolução: {src_mdt.res}")

    # Determinar CRS métrico (para X, Y em metros e comprimento/área real)
    if src_mdt.crs.is_projected:
        metric_crs = src_mdt.crs
    elif original_crs.is_projected:
        metric_crs = original_crs
    else:
        try:
            metric_crs = gdf.estimate_utm_crs()
        except Exception:
            metric_crs = CRS.from_user_input("EPSG:3857")

    if hasattr(metric_crs, "to_2d"):
        metric_crs = metric_crs.to_2d()

    # Reprojetar cópias de trabalho para amostragem no MDT e cálculo métrico
    gdf_mdt = gdf.to_crs(src_mdt.crs)
    gdf_metric = gdf.to_crs(metric_crs)

    # Transformador de coordenadas para WGS84 (lat/lon)
    transformer_met_to_wgs = Transformer.from_crs(metric_crs, "EPSG:4326", always_xy=True)

    # Identificar tipo predominante de geometria
    geom_types = set(g.geom_type.lower() for g in gdf.geometry if g is not None)
    is_polygon_layer = any('polygon' in t for t in geom_types)

    geoms_3d_mdt = []

    print("  - Amostrando altitudes no MDT e gerando geometrias 3D...")

    if is_polygon_layer:
        # ---- PROCESSAMENTO DE POLÍGONOS ----
        lat_list, lon_list = [], []
        x_list, y_list = [], []
        z_list = []
        min_z_list, max_z_list, mean_z_list = [], [], []
        area_ha_list = []
        area_m2_list = []
        perim_m_list = []

        for idx in range(len(gdf)):
            geom_mdt = gdf_mdt.iloc[idx].geometry
            geom_met = gdf_metric.iloc[idx].geometry

            if geom_mdt is None or geom_mdt.is_empty:
                geoms_3d_mdt.append(None)
                for lst in [lat_list, lon_list, x_list, y_list, z_list,
                            min_z_list, max_z_list, mean_z_list,
                            area_ha_list, area_m2_list, perim_m_list]:
                    lst.append(0.0)
                continue

            # 1. Geometria 3D
            geom_3d = process_polygon_3d(geom_mdt, src_mdt)
            geoms_3d_mdt.append(geom_3d)

            # 2. Centróide
            c_met = geom_met.centroid
            c_lon, c_lat = transformer_met_to_wgs.transform(c_met.x, c_met.y)
            c_mdt = geom_mdt.centroid
            sampled_mid = list(src_mdt.sample([(c_mdt.x, c_mdt.y)]))
            cz = float(sampled_mid[0][0])
            if np.isnan(cz) or (src_mdt.nodata is not None and np.isclose(cz, src_mdt.nodata)):
                cz = 0.0

            x_list.append(float(c_met.x))
            y_list.append(float(c_met.y))
            lon_list.append(float(c_lon))
            lat_list.append(float(c_lat))
            z_list.append(float(cz))

            # 3. Estatísticas Z das coordenadas dos vértices do polígono
            def get_poly_zs(p):
                zs = []
                if p.geom_type == 'Polygon':
                    zs.extend([c[2] for c in p.exterior.coords if len(c) >= 3])
                    for interior in p.interiors:
                        zs.extend([c[2] for c in interior.coords if len(c) >= 3])
                elif p.geom_type == 'MultiPolygon':
                    for part in p.geoms:
                        zs.extend(get_poly_zs(part))
                return zs

            poly_z_vals = get_poly_zs(geom_3d)
            min_z = float(np.min(poly_z_vals)) if poly_z_vals else 0.0
            max_z = float(np.max(poly_z_vals)) if poly_z_vals else 0.0
            mean_z = float(np.mean(poly_z_vals)) if poly_z_vals else 0.0

            # 4. Área e Perímetro
            area_m2 = float(geom_met.area)
            area_ha = float(area_m2 / 10000.0)
            perim_m = float(geom_met.length)

            min_z_list.append(min_z)
            max_z_list.append(max_z)
            mean_z_list.append(mean_z)
            area_ha_list.append(area_ha)
            area_m2_list.append(area_m2)
            perim_m_list.append(perim_m)

        src_mdt.close()

        # Montar GeoDataFrame de saída (APENAS campos de polígonos)
        gdf_out = gdf.copy()
        gdf_out_3d = gpd.GeoDataFrame(gdf_out, geometry=geoms_3d_mdt, crs=src_mdt.crs)
        gdf_final = gdf_out_3d.to_crs("EPSG:4326")

        gdf_final['lat'] = lat_list
        gdf_final['lon'] = lon_list
        gdf_final['x'] = x_list
        gdf_final['y'] = y_list
        gdf_final['z'] = z_list
        gdf_final['min_z'] = min_z_list
        gdf_final['max_z'] = max_z_list
        gdf_final['mean_z'] = mean_z_list
        gdf_final['Area_ha'] = area_ha_list
        gdf_final['Area_m2'] = area_m2_list
        gdf_final['Perim_m'] = perim_m_list

    else:
        # ---- PROCESSAMENTO DE LINHAS ----
        lat_list, lon_list = [], []
        x_list, y_list = [], []
        z_list = []
        start_lat_list, start_lon_list, start_x_list, start_y_list, start_z_list = [], [], [], [], []
        end_lat_list, end_lon_list, end_x_list, end_y_list, end_z_list = [], [], [], [], []
        min_z_list, max_z_list, mean_z_list = [], [], []
        slope_pct_list = []
        length_list = []

        for idx in range(len(gdf)):
            geom_mdt = gdf_mdt.iloc[idx].geometry
            geom_met = gdf_metric.iloc[idx].geometry

            if geom_mdt is None or geom_mdt.is_empty:
                geoms_3d_mdt.append(None)
                for lst in [lat_list, lon_list, x_list, y_list, z_list,
                            start_lat_list, start_lon_list, start_x_list, start_y_list, start_z_list,
                            end_lat_list, end_lon_list, end_x_list, end_y_list, end_z_list,
                            min_z_list, max_z_list, mean_z_list, slope_pct_list, length_list]:
                    lst.append(0.0)
                continue

            # 1. Geometria 3D
            geom_3d = process_line_3d(geom_mdt, src_mdt)
            geoms_3d_mdt.append(geom_3d)

            # Coordenadas Z dos vértices da linha
            if geom_3d.geom_type == 'LineString':
                z_vals = [c[2] for c in geom_3d.coords]
                s_coord = geom_3d.coords[0]
                e_coord = geom_3d.coords[-1]
            else: # MultiLineString
                z_vals = []
                for part in geom_3d.geoms:
                    z_vals.extend([c[2] for c in part.coords])
                s_coord = geom_3d.geoms[0].coords[0] if geom_3d.geoms else (0,0,0)
                e_coord = geom_3d.geoms[-1].coords[-1] if geom_3d.geoms else (0,0,0)

            # Início / Fim métrico e geográfico
            s_met = geom_met.coords[0] if geom_met.geom_type == 'LineString' else (0,0)
            e_met = geom_met.coords[-1] if geom_met.geom_type == 'LineString' else (0,0)
            if geom_met.geom_type == 'MultiLineString' and geom_met.geoms:
                s_met = geom_met.geoms[0].coords[0]
                e_met = geom_met.geoms[-1].coords[-1]

            s_lon, s_lat = transformer_met_to_wgs.transform(s_met[0], s_met[1])
            e_lon, e_lat = transformer_met_to_wgs.transform(e_met[0], e_met[1])

            start_x_list.append(float(s_met[0]))
            start_y_list.append(float(s_met[1]))
            start_lon_list.append(float(s_lon))
            start_lat_list.append(float(s_lat))
            start_z_list.append(float(s_coord[2] if len(s_coord) >= 3 else 0.0))

            end_x_list.append(float(e_met[0]))
            end_y_list.append(float(e_met[1]))
            end_lon_list.append(float(e_lon))
            end_lat_list.append(float(e_lat))
            end_z_list.append(float(e_coord[2] if len(e_coord) >= 3 else 0.0))

            # Ponto Médio
            mid_pt_met = geom_met.interpolate(0.5, normalized=True)
            m_lon, m_lat = transformer_met_to_wgs.transform(mid_pt_met.x, mid_pt_met.y)
            mid_pt_mdt = geom_mdt.interpolate(0.5, normalized=True)
            sampled_mid = list(src_mdt.sample([(mid_pt_mdt.x, mid_pt_mdt.y)]))
            mid_z = float(sampled_mid[0][0])
            if np.isnan(mid_z) or (src_mdt.nodata is not None and np.isclose(mid_z, src_mdt.nodata)):
                mid_z = float(np.mean(z_vals)) if z_vals else 0.0

            x_list.append(float(mid_pt_met.x))
            y_list.append(float(mid_pt_met.y))
            lon_list.append(float(m_lon))
            lat_list.append(float(m_lat))
            z_list.append(float(mid_z))

            # Estatísticas Z
            min_z = float(np.min(z_vals)) if z_vals else 0.0
            max_z = float(np.max(z_vals)) if z_vals else 0.0
            mean_z = float(np.mean(z_vals)) if z_vals else 0.0
            horiz_len = float(geom_met.length)

            s_z_val = start_z_list[-1]
            e_z_val = end_z_list[-1]
            slope = ((e_z_val - s_z_val) / horiz_len) * 100.0 if horiz_len > 0 else 0.0

            min_z_list.append(min_z)
            max_z_list.append(max_z)
            mean_z_list.append(mean_z)
            slope_pct_list.append(slope)
            length_list.append(horiz_len)

        src_mdt.close()

        # Montar GeoDataFrame de saída (APENAS campos de linhas)
        gdf_out = gdf.copy()
        gdf_out_3d = gpd.GeoDataFrame(gdf_out, geometry=geoms_3d_mdt, crs=src_mdt.crs)
        gdf_final = gdf_out_3d.to_crs("EPSG:4326")

        gdf_final['lat'] = lat_list
        gdf_final['lon'] = lon_list
        gdf_final['x'] = x_list
        gdf_final['y'] = y_list
        gdf_final['z'] = z_list
        gdf_final['start_lat'] = start_lat_list
        gdf_final['start_lon'] = start_lon_list
        gdf_final['start_x'] = start_x_list
        gdf_final['start_y'] = start_y_list
        gdf_final['start_z'] = start_z_list
        gdf_final['end_lat'] = end_lat_list
        gdf_final['end_lon'] = end_lon_list
        gdf_final['end_x'] = end_x_list
        gdf_final['end_y'] = end_y_list
        gdf_final['end_z'] = end_z_list
        gdf_final['min_z'] = min_z_list
        gdf_final['max_z'] = max_z_list
        gdf_final['mean_z'] = mean_z_list
        gdf_final['slope_pct'] = slope_pct_list
        gdf_final['Length'] = length_list

    # Garantir criação do diretório de destino (incluindo subpastas)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    driver = get_driver(output_file)

    print(f"  - Salvando NOVO arquivo em: {output_file} (Driver: {driver})")

    try:
        gdf_final.to_file(output_file, driver=driver)
        print(f"  [SUCESSO] Arquivo gerado com sucesso!")
        return True
    except Exception as e:
        print(f"  [AVISO] Falha na gravação padrão ({e}). Tentando fallback GeoJSON...")
        try:
            name, _ = os.path.splitext(output_file)
            fallback_path = f"{name}.geojson"
            gdf_final.to_file(fallback_path, driver="GeoJSON")
            print(f"  [SUCESSO] Arquivo salvo via fallback em: {fallback_path}")
            return True
        except Exception as ex:
            print(f"  [ERRO] Falha total ao salvar: {ex}")
            return False


def main():
    parser = argparse.ArgumentParser(
        description="Adiciona altitude Z do MDT e colunas de coordenadas a arquivos de linhas e polígonos SEM ALTERAR OS ARQUIVOS ORIGINAIS.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-i", "--input", required=True,
        help="Caminho do arquivo vetorial (.geojson, .shp, .gpkg) ou diretório contendo arquivos."
    )
    parser.add_argument(
        "-m", "--mdt", required=True,
        help="Caminho do arquivo raster DTM/MDT (.tif / .tiff)."
    )
    parser.add_argument(
        "-o", "--output_dir", default="outputs_coordenadas",
        help="Diretório de saída para salvar os novos arquivos 3D com coordenadas."
    )

    args = parser.parse_args()

    print("=" * 75)
    print("      INJEÇÃO DE COORDENADAS E ALTITUDE MDT (LINHAS & POLÍGONOS 3D)")
    print("=" * 75)
    print(f"Entrada    : {args.input}")
    print(f"Raster MDT : {args.mdt}")
    print(f"Saída      : {args.output_dir}")
    print("-" * 75)

    input_path = os.path.abspath(args.input)
    mdt_path = os.path.abspath(args.mdt)
    output_dir = os.path.abspath(args.output_dir)

    if not os.path.exists(input_path):
        print(f"[ERRO] Entrada não encontrada: {input_path}")
        sys.exit(1)

    if not os.path.exists(mdt_path):
        print(f"[ERRO] MDT não encontrado: {mdt_path}")
        sys.exit(1)

    valid_exts = ('.geojson', '.shp', '.gpkg')
    files_to_process = []

    if os.path.isfile(input_path):
        out_file = os.path.join(output_dir, os.path.basename(input_path))
        files_to_process.append((input_path, out_file))
    else:
        for root, _, files in os.walk(input_path):
            # Ignorar a própria pasta de saída se ela estiver dentro do diretório de entrada
            try:
                if os.path.commonpath([output_dir, root]) == output_dir:
                    continue
            except ValueError:
                pass
            for f in files:
                if f.lower().endswith(valid_exts):
                    in_file = os.path.join(root, f)
                    # Manter o mesmo caminho relativo de subpasta
                    rel_path = os.path.relpath(in_file, input_path)
                    out_file = os.path.join(output_dir, rel_path)
                    files_to_process.append((in_file, out_file))

    if not files_to_process:
        print("[ERRO] Nenhum arquivo vetorial válido (.geojson, .shp, .gpkg) encontrado.")
        sys.exit(1)

    success_count = 0
    for in_file, out_file in files_to_process:
        if process_single_file(in_file, mdt_path, out_file):
            success_count += 1

    print("\n" + "=" * 75)
    print(f" CONCLUÍDO: {success_count}/{len(files_to_process)} arquivo(s) processados com sucesso.")
    print(f" Os arquivos originais PERMANECERAM INTACTOS.")
    print(f" Novos arquivos salvos em: {output_dir}")
    print("=" * 75)


if __name__ == "__main__":
    main()
