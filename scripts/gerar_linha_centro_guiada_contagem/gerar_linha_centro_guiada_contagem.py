#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
gerar_linha_centro_guiada.py
--------------------------------------------------------------------------------
Script para traçar linhas centrais de polígonos utilizando pontos guia internos.
Ajusta uma regressão polinomial em coordenadas rotacionadas pelo eixo principal (PCA),
gerando um arquivo final em formato Shapefile (.shp) projetado em WGS84 (EPSG:4326).
--------------------------------------------------------------------------------
"""

import os
import sys
import argparse
import numpy as np
import geopandas as gpd
from shapely.geometry import LineString, Polygon, MultiPolygon

def extract_centerline_guided(poly, points_gdf, deg=2, min_points=2):
    """
    Extrai a linha central de um polígono ordenando e traçando um caminho contínuo
    pelos pontos guia (árvores) internos (Nearest Neighbor path), estendendo as extremidades
    até a borda e recortando no polígono.
    """
    if poly is None or poly.is_empty:
        return None
        
    # Encontra pontos dentro do polígono usando índice espacial se disponível
    if points_gdf.sindex is not None:
        possible_matches_idx = list(points_gdf.sindex.intersection(poly.bounds))
        possible_matches = points_gdf.iloc[possible_matches_idx]
        points_inside = possible_matches[possible_matches.geometry.intersects(poly)]
    else:
        points_inside = points_gdf[points_gdf.geometry.intersects(poly)]
        
    # Se houver pontos guia suficientes
    if len(points_inside) >= min_points:
        try:
            # Extrai as coordenadas x, y dos pontos de forma robusta usando o centroide
            pts_coords = np.array([(geom.centroid.x, geom.centroid.y) for geom in points_inside.geometry if geom is not None])
            n = len(pts_coords)
            
            if n >= 2:
                # 1. Calcula a matriz de distâncias euclidianas entre todos os pontos
                dists = np.linalg.norm(pts_coords[:, None, :] - pts_coords[None, :, :], axis=-1)
                
                # 2. Encontra os dois pontos mais distantes entre si (as extremidades da fileira)
                i, j = np.unravel_index(np.argmax(dists), dists.shape)
                
                # 3. Reconstrói o caminho de pontos de forma sequencial (Nearest Neighbor walk)
                current_idx = i
                visited = {current_idx}
                path = [pts_coords[current_idx]]
                
                while len(visited) < n:
                    # Encontra o vizinho não visitado mais próximo
                    unvisited_indices = [idx for idx in range(n) if idx not in visited]
                    distances_to_unvisited = dists[current_idx, unvisited_indices]
                    nearest_unvisited_idx = unvisited_indices[np.argmin(distances_to_unvisited)]
                    
                    current_idx = nearest_unvisited_idx
                    visited.add(current_idx)
                    path.append(pts_coords[current_idx])
                    
                path = np.array(path)
                
                # 4. Suavização leve para remover pequenos ruídos/zig-zags
                if n >= 5:
                    window = 5
                elif n >= 3:
                    window = 3
                else:
                    window = 1
                    
                smoothed_path = []
                for k in range(n):
                    start = max(0, k - window // 2)
                    end = min(n, k + window // 2 + 1)
                    mean_pt = path[start:end].mean(axis=0)
                    smoothed_path.append(mean_pt)
                smoothed_path = np.array(smoothed_path)
                
                # 5. Estende as extremidades nas direções tangentes para cobrir o polígono "de fora a fora"
                bounds = poly.bounds
                diag = np.sqrt((bounds[2] - bounds[0])**2 + (bounds[3] - bounds[1])**2)
                ext_len = diag * 2.0
                
                # Extensão no início (P_0 estendido na direção oposta de P_1)
                v_start = smoothed_path[0] - smoothed_path[1]
                len_start = np.linalg.norm(v_start)
                if len_start > 0:
                    u_start = v_start / len_start
                    p_start_ext = smoothed_path[0] + ext_len * u_start
                else:
                    p_start_ext = smoothed_path[0]
                    
                # Extensão no fim (P_N estendido na direção de P_{N-1} -> P_N)
                v_end = smoothed_path[-1] - smoothed_path[-2]
                len_end = np.linalg.norm(v_end)
                if len_end > 0:
                    u_end = v_end / len_end
                    p_end_ext = smoothed_path[-1] + ext_len * u_end
                else:
                    p_end_ext = smoothed_path[-1]
                    
                extended_coords = [tuple(p_start_ext)] + [tuple(pt) for pt in smoothed_path] + [tuple(p_end_ext)]
                line = LineString(extended_coords)
                
                # 6. Recorta a linha estendida na borda do polígono original
                intersection = line.intersection(poly)
                if intersection.is_empty:
                    return line
                if intersection.geom_type == 'LineString':
                    return intersection
                elif intersection.geom_type == 'MultiLineString':
                    valid_lines = [geom for geom in intersection.geoms if not geom.is_empty]
                    if valid_lines:
                        return max(valid_lines, key=lambda l: l.length)
                return line
                
        except Exception as e:
            # Log de aviso e fallback para lógica de polígono
            print(f"[WARN] Falha ao rastrear linha pelos pontos guia: {e}. Usando geometria do polígono.")
            
    # Lógica de fallback baseada apenas nas coordenadas da borda do polígono
    coords = np.array(poly.exterior.coords)
    if len(coords) < 3:
        return None
        
    mean = coords.mean(axis=0)
    coords_centered = coords - mean
    
    try:
        cov = np.cov(coords_centered.T)
        if cov.ndim < 2:
            return None
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        idx_max = np.argmax(eigenvalues)
        ex = eigenvectors[:, idx_max]
        ey = np.array([-ex[1], ex[0]])
        R = np.vstack([ex, ey])
        
        coords_rot = coords_centered @ R.T
        X = coords_rot[:, 0]
        Y = coords_rot[:, 1]
        x_min, x_max = X.min(), X.max()
        row_length = x_max - x_min
        
        fit_deg = deg if row_length > 10.0 else 1
        coefs = np.polyfit(X, Y, fit_deg)
        poly_func = np.poly1d(coefs)
        
        xs = np.linspace(x_min, x_max, 50 if fit_deg > 1 else 2)
        ys = poly_func(xs)
        pts_rot = np.vstack([xs, ys]).T
        pts_orig = (pts_rot @ R) + mean
        line = LineString(pts_orig)
        
        intersection = line.intersection(poly)
        if intersection.is_empty:
            return line
        if intersection.geom_type == 'LineString':
            return intersection
        elif intersection.geom_type == 'MultiLineString':
            valid_lines = [geom for geom in intersection.geoms if not geom.is_empty]
            if valid_lines:
                return max(valid_lines, key=lambda l: l.length)
        return line
    except Exception as e:
        print(f"[WARN] Falha no cálculo de fallback para polígono: {e}")
        # Último caso: Linha conectando extremos do retângulo envolvente rotacionado
        try:
            cov = np.cov(coords_centered.T)
            eigenvalues, eigenvectors = np.linalg.eig(cov)
            idx_max = np.argmax(eigenvalues)
            ex = eigenvectors[:, idx_max]
            ey = np.array([-ex[1], ex[0]])
            R = np.vstack([ex, ey])
            
            coords_rot = coords_centered @ R.T
            X = coords_rot[:, 0]
            Y = coords_rot[:, 1]
            x_min, x_max = X.min(), X.max()
            
            xs = np.array([x_min, x_max])
            ys = np.array([Y[np.argmin(X)], Y[np.argmax(X)]])
            pts_rot = np.vstack([xs, ys]).T
            pts_orig = (pts_rot @ R) + mean
            line = LineString(pts_orig)
            
            intersection = line.intersection(poly)
            if not intersection.is_empty:
                if intersection.geom_type == 'LineString':
                    return intersection
                elif intersection.geom_type == 'MultiLineString':
                    valid_lines = [geom for geom in intersection.geoms if not geom.is_empty]
                    if valid_lines:
                        return max(valid_lines, key=lambda l: l.length)
            return line
        except Exception:
            return None

def main():
    parser = argparse.ArgumentParser(
        description="Gera linhas centrais para polígonos usando pontos internos como guias.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--input_polygon", "-p", required=True,
        help="Caminho do arquivo de polígonos (GeoJSON, Shapefile ou GPKG)."
    )
    parser.add_argument(
        "--input_points", "-pts", required=True,
        help="Caminho do arquivo de pontos guia (GeoJSON, Shapefile ou GPKG)."
    )
    parser.add_argument(
        "--output_shp", "-o", required=True,
        help="Caminho do Shapefile (.shp) de saída (será salvo em WGS84)."
    )
    parser.add_argument(
        "--deg", "-d", type=int, default=2,
        help="Grau do ajuste polinomial para as linhas centrais."
    )
    parser.add_argument(
        "--min_points", type=int, default=2,
        help="Número mínimo de pontos internos para ativar o ajuste baseado em guias."
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("   GERADOR DE LINHA CENTRAL COMPATÍVEL COM PONTOS GUIA")
    print("=" * 70)
    print(f"Polígonos de entrada : {args.input_polygon}")
    print(f"Pontos guia de entrada: {args.input_points}")
    print(f"Shapefile de saída   : {args.output_shp}")
    print(f"Grau do polinômio    : {args.deg}")
    print(f"Mínimo de pontos guia: {args.min_points}")
    print("-" * 70)
    
    # Valida caminhos de entrada
    if not os.path.exists(args.input_polygon):
        print(f"[ERRO] Arquivo de polígonos não encontrado: {args.input_polygon}")
        sys.exit(1)
    if not os.path.exists(args.input_points):
        print(f"[ERRO] Arquivo de pontos guia não encontrado: {args.input_points}")
        sys.exit(1)
        
    # Garante que a saída termina com .shp
    if not args.output_shp.lower().endswith(".shp"):
        print("[AVISO] O arquivo de saída deve terminar com .shp. O nome será ajustado.")
        args.output_shp = os.path.splitext(args.output_shp)[0] + ".shp"
        
    try:
        print("[INFO] Carregando arquivos...")
        gdf_polygons = gpd.read_file(args.input_polygon)
        gdf_points = gpd.read_file(args.input_points)
    except Exception as e:
        print(f"[ERRO] Falha ao carregar camadas vetoriais: {e}")
        sys.exit(1)
        
    if gdf_polygons.empty:
        print("[ERRO] A camada de polígonos está vazia.")
        sys.exit(0)
    if gdf_points.empty:
        print("[AVISO] A camada de pontos guia está vazia. O script funcionará apenas com fallback.")
        
    # Guarda o CRS original
    original_crs = gdf_polygons.crs
    if original_crs is None:
        print("[AVISO] A camada de entrada não possui CRS. Assumindo EPSG:4326.")
        original_crs = "EPSG:4326"
        gdf_polygons.set_crs(original_crs, inplace=True)
        
    if gdf_points.crs is None:
        gdf_points.set_crs(original_crs, inplace=True)
        
    # Reprojeção dinâmica para sistema métrico local (UTM) para cálculos métricos precisos e sem distorção
    is_geographic = original_crs.is_geographic if hasattr(original_crs, 'is_geographic') else True
    if is_geographic:
        try:
            metric_crs = gdf_polygons.estimate_utm_crs()
            print(f"[INFO] Projetando dados para UTM local ({metric_crs.to_string()}) para cálculos espaciais...")
            gdf_poly_proj = gdf_polygons.to_crs(metric_crs)
            gdf_points_proj = gdf_points.to_crs(metric_crs)
        except Exception as e:
            metric_crs = "EPSG:3857"
            print(f"[WARN] Falha ao obter UTM ({e}). Reprojetando para EPSG:3857...")
            gdf_poly_proj = gdf_polygons.to_crs(metric_crs)
            gdf_points_proj = gdf_points.to_crs(metric_crs)
    else:
        metric_crs = original_crs
        gdf_poly_proj = gdf_polygons
        gdf_points_proj = gdf_points.to_crs(metric_crs)
        
    # Explode MultiPolygons em Polygons normais para garantir consistência
    gdf_poly_proj = gdf_poly_proj.explode(index_parts=False)
    
    lines_list = []
    attributes_list = []
    
    print("[INFO] Processando geometrias...")
    for idx, row in gdf_poly_proj.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
            
        line = extract_centerline_guided(geom, gdf_points_proj, deg=args.deg, min_points=args.min_points)
        if line is not None and not line.is_empty:
            lines_list.append(line)
            # Copiar atributos do polígono original
            attr_dict = row.drop('geometry').to_dict()
            attributes_list.append(attr_dict)
            
    if not lines_list:
        print("[ERRO] Nenhuma linha central pôde ser gerada.")
        sys.exit(1)
        
    # Cria o GeoDataFrame resultante no CRS métrico
    gdf_lines_proj = gpd.GeoDataFrame(attributes_list, geometry=lines_list, crs=metric_crs)
    
    # Reprojeta para WGS84 para exportação final
    print("[INFO] Reprojetando resultados para WGS84 (EPSG:4326)...")
    gdf_lines_wgs84 = gdf_lines_proj.to_crs("EPSG:4326")
    
    # Salva o arquivo de saída
    try:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_shp)), exist_ok=True)
        gdf_lines_wgs84.to_file(args.output_shp)
        print(f"\n[OK] Linhas centrais geradas com sucesso e salvas em:\n     {args.output_shp}")
    except Exception as e:
        print(f"[ERRO] Falha ao salvar arquivo Shapefile: {e}")
        sys.exit(1)
        
    print("=" * 70)

if __name__ == "__main__":
    main()
