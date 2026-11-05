#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import geopandas as gpd
from scipy.spatial import distance_matrix
from scipy.interpolate import splprep, splev
from shapely.geometry import LineString, MultiLineString, Polygon, MultiPolygon, Point
from shapely.ops import linemerge, unary_union

# -----------------------------------------------------------------------------
# CORE ALGORITHMS AND HELPERS
# -----------------------------------------------------------------------------

def order_points_nearest_neighbor(pts_coords):
    pts = np.array(pts_coords)
    n = len(pts)
    if n <= 2:
        return pts

    D = distance_matrix(pts, pts)
    i, j = np.unravel_index(np.argmax(D), D.shape)
    start_node = i
    
    ordered_indices = [start_node]
    remaining = set(range(n))
    remaining.remove(start_node)
    
    current = start_node
    while remaining:
        next_idx = min(remaining, key=lambda idx: D[current, idx])
        ordered_indices.append(next_idx)
        remaining.remove(next_idx)
        current = next_idx
        
    return pts[ordered_indices]

def smooth_path_v4(path, window=3):
    n = len(path)
    if n < 3:
        return path
    window = min(window, n)
    if window % 2 == 0: 
        window -= 1
    if window < 3: 
        return path

    smoothed = []
    half = window // 2
    for k in range(n):
        start = max(0, k - half)
        end = min(n, k + half + 1)
        if k == 0 or k == n - 1:
            smoothed.append(path[k])
        else:
            smoothed.append(path[start:end].mean(axis=0))
    return np.array(smoothed)

def smooth_path_spline(path, s=0.5):
    n = len(path)
    if n < 4:
        return path
    
    try:
        dists = np.linalg.norm(np.diff(path, axis=0), axis=1)
        mask = np.ones(len(path), dtype=bool)
        mask[1:] = dists > 1e-5
        path = path[mask]
        
        if len(path) < 4:
            return path

        x = path[:, 0]
        y = path[:, 1]
        
        tck, u = splprep([x, y], s=s)
        u_new = np.linspace(0, 1, num=max(n * 2, 20))
        new_points = splev(u_new, tck)
        return np.column_stack(new_points)
    except Exception:
        return path

def get_centerline_via_pca(poly, pts_coords):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=1)
    pca.fit(pts_coords)
    projected = pca.transform(pts_coords)
    p_min, p_max = projected.min(), projected.max()
    pts_extrema = pca.inverse_transform(np.array([[p_min], [p_max]]))
    line = LineString(pts_extrema)
    return line

def get_filled_polygon(geom):
    if geom.geom_type == 'Polygon':
        return Polygon(geom.exterior)
    elif geom.geom_type == 'MultiPolygon':
        return MultiPolygon([Polygon(p.exterior) for p in geom.geoms])
    return geom

def snap_points(line, tolerance=0.5):
    coords = list(line.coords)
    if len(coords) < 2:
        return line
    
    new_coords = [coords[0]]
    for i in range(1, len(coords)):
        last_pt = np.array(new_coords[-1])
        curr_pt = np.array(coords[i])
        if np.linalg.norm(last_pt - curr_pt) > tolerance:
            new_coords.append(coords[i])
    
    if len(new_coords) < 2:
        return line
    return LineString(new_coords)

def join_close_lines(gdf, tolerance=0.5):
    if gdf.empty:
        return gdf
    
    buffered = gdf.geometry.buffer(tolerance / 2, cap_style=2, join_style=2)
    unified = unary_union(buffered)
    
    new_lines = []
    if unified.geom_type == 'Polygon':
        unified = MultiPolygon([unified])
    
    if unified.geom_type == 'MultiPolygon':
        for poly in unified.geoms:
            contained_indices = gdf.index[gdf.geometry.within(poly.buffer(0.01))]
            if len(contained_indices) > 0:
                subset = gdf.loc[contained_indices].geometry
                merged = linemerge(list(subset))
                if merged.geom_type == 'LineString':
                    new_lines.append(merged)
                elif merged.geom_type == 'MultiLineString':
                    parts = list(merged.geoms)
                    ordered_parts = [parts.pop(0)]
                    while parts:
                        last_line = ordered_parts[-1]
                        last_pt = Point(last_line.coords[-1])
                        
                        best_idx = -1
                        best_dist = float('inf')
                        reverse_needed = False
                        
                        for i, p in enumerate(parts):
                            d1 = last_pt.distance(Point(p.coords[0]))
                            d2 = last_pt.distance(Point(p.coords[-1]))
                            
                            if d1 < best_dist:
                                best_dist = d1
                                best_idx = i
                                reverse_needed = False
                            if d2 < best_dist:
                                best_dist = d2
                                best_idx = i
                                reverse_needed = True
                        
                        if best_idx != -1 and best_dist <= tolerance * 2:
                            next_line = parts.pop(best_idx)
                            if reverse_needed:
                                next_line = LineString(list(next_line.coords)[::-1])
                            
                            combined_coords = list(last_line.coords) + list(next_line.coords)
                            ordered_parts[-1] = LineString(combined_coords)
                        else:
                            if parts:
                                ordered_parts.append(parts.pop(0))
                            else:
                                break
                    new_lines.extend(ordered_parts)
    
    if new_lines:
        return gpd.GeoDataFrame(geometry=new_lines, crs=gdf.crs)
    return gdf

def centerline_from_points_v6(poly, pts_coords, min_points=5, smoothing=0.5):
    n = len(pts_coords)
    if n < min_points:
        return None

    try:
        ordered_pts = order_points_nearest_neighbor(pts_coords)
        smoothed = smooth_path_spline(ordered_pts, s=smoothing)

        if len(smoothed) < 2:
            return None
            
        bounds = poly.bounds
        diag = np.sqrt((bounds[2] - bounds[0])**2 + (bounds[3] - bounds[1])**2)
        ext = diag * 0.2 

        p0, p1 = smoothed[0], smoothed[1]
        v_start = p0 - p1
        d_start = np.linalg.norm(v_start)
        p_start_ext = p0 + (v_start / d_start * ext) if d_start > 0 else p0

        pn_1, pn = smoothed[-2], smoothed[-1]
        v_end = pn - pn_1
        d_end = np.linalg.norm(v_end)
        p_end_ext = pn + (v_end / d_end * ext) if d_end > 0 else pn

        long_line = LineString([p_start_ext] + list(smoothed) + [p_end_ext])
        poly_filled = get_filled_polygon(poly)
        intersection = long_line.intersection(poly_filled)
        
        if intersection.is_empty:
            return None
            
        if intersection.geom_type == 'LineString':
            return snap_points(intersection)
        elif intersection.geom_type in ['MultiLineString', 'GeometryCollection']:
            parts = [g for g in (intersection.geoms if hasattr(intersection, 'geoms') else []) if g.geom_type == 'LineString']
            if parts:
                return snap_points(max(parts, key=lambda l: l.length))
            
        return None
    except Exception:
        return None

def centerline_from_points_v4(poly, pts_coords, min_points=5, smooth_window=5):
    n = len(pts_coords)
    if n < min_points:
        return None

    try:
        ordered_pts = order_points_nearest_neighbor(pts_coords)
        win = smooth_window if n >= smooth_window else (3 if n >= 3 else 1)
        smoothed = smooth_path_v4(ordered_pts, window=win)

        if len(smoothed) < 2:
            long_line = get_centerline_via_pca(poly, pts_coords)
        else:
            bounds = poly.bounds
            diag = np.sqrt((bounds[2] - bounds[0])**2 + (bounds[3] - bounds[1])**2)
            ext = diag * 0.5

            p0, p1 = smoothed[0], smoothed[1]
            v_start = p0 - p1
            dist_start = np.linalg.norm(v_start)
            p_start_ext = p0 + (v_start / dist_start * ext) if dist_start > 0 else p0

            pn_1, pn = smoothed[-2], smoothed[-1]
            v_end = pn - pn_1
            dist_end = np.linalg.norm(v_end)
            p_end_ext = pn + (v_end / dist_end * ext) if dist_end > 0 else pn

            long_line = LineString([p_start_ext] + list(smoothed) + [p_end_ext])

        poly_filled = get_filled_polygon(poly)
        intersection = long_line.intersection(poly_filled)
        
        if intersection.is_empty:
            intersection = long_line.intersection(poly_filled)
            if intersection.is_empty: 
                return None
            
        if intersection.geom_type == 'LineString':
            return intersection
        elif intersection.geom_type in ['MultiLineString', 'GeometryCollection']:
            parts = [g for g in (intersection.geoms if hasattr(intersection, 'geoms') else []) if g.geom_type == 'LineString']
            if parts:
                return max(parts, key=lambda l: l.length)
            
        return None
    except Exception:
        return None

# -----------------------------------------------------------------------------
# RUN LOGIC FROM GUI
# -----------------------------------------------------------------------------

def run_centerline_generation(method, input_poly, input_pts, boundary_file, output_path, min_points, smoothing, tolerance, update_console, on_complete):
    try:
        update_console("[INFO] Iniciando processamento...\n")
        
        if not os.path.exists(input_poly):
            update_console(f"[ERRO] Arquivo de polígonos não encontrado: {input_poly}\n")
            on_complete(False)
            return

        if not os.path.exists(input_pts):
            update_console(f"[ERRO] Arquivo de pontos guia não encontrado: {input_pts}\n")
            on_complete(False)
            return

        update_console("[INFO] Carregando arquivos...\n")
        gdf_poly = gpd.read_file(input_poly)
        gdf_pts = gpd.read_file(input_pts)

        original_crs = gdf_poly.crs
        
        if original_crs and original_crs.is_geographic:
            update_console("[INFO] Projetando dados para UTM local...\n")
            metric_crs = gdf_poly.estimate_utm_crs()
            gdf_poly = gdf_poly.to_crs(metric_crs)
        else:
            metric_crs = original_crs

        if gdf_pts.crs != metric_crs:
            gdf_pts = gdf_pts.to_crs(metric_crs)

        # Se houver limite do talhão opcional, recortamos os dados de entrada
        if boundary_file and os.path.exists(boundary_file):
            update_console("[INFO] Carregando e aplicando limite do talhão...\n")
            gdf_boundary = gpd.read_file(boundary_file)
            if not gdf_boundary.empty:
                if gdf_boundary.crs != metric_crs:
                    gdf_boundary = gdf_boundary.to_crs(metric_crs)
                
                talhao_geom = gdf_boundary.geometry.unary_union
                
                # Manter apenas elementos que interceptam o talhão
                gdf_poly = gdf_poly[gdf_poly.geometry.intersects(talhao_geom)].reset_index(drop=True)
                gdf_pts = gdf_pts[gdf_pts.geometry.intersects(talhao_geom)].reset_index(drop=True)
                update_console(f"[INFO] Elementos dentro do talhão: {len(gdf_poly)} polígonos, {len(gdf_pts)} pontos.\n")

        gdf_poly = gdf_poly.explode(index_parts=False).reset_index(drop=True)
        gdf_poly['_poly_id'] = range(len(gdf_poly))

        gdf_poly_filled = gdf_poly.copy()
        gdf_poly_filled['geometry'] = gdf_poly['geometry'].apply(get_filled_polygon)
        
        update_console("[INFO] Realizando associação espacial de pontos aos polígonos...\n")
        joined = gpd.sjoin(gdf_pts, gdf_poly_filled[['geometry', '_poly_id']], how='inner', predicate='within')
        
        points_by_poly = {}
        if not joined.empty:
            for poly_id, group in joined.groupby('_poly_id'):
                points_by_poly[poly_id] = np.array([(g.x, g.y) for g in group.geometry])

        update_console(f"[INFO] Gerando linhas de centro individuais (Método: {method})...\n")
        temp_lines = []
        temp_attrs = []
        
        for idx, row in gdf_poly.iterrows():
            poly_id = row['_poly_id']
            pts = points_by_poly.get(poly_id, [])
            
            if method == "PCA v6 (Integrate & Dissolve)":
                line = centerline_from_points_v6(row.geometry, pts, min_points, smoothing)
            else:  # PCA v4
                # Usar a janela de suavização correspondente ao smoothing convertido em int
                win = int(smoothing)
                line = centerline_from_points_v4(row.geometry, pts, min_points, win)
            
            if line:
                temp_lines.append(line)
                attr = {k: v for k, v in row.items() if k not in ('geometry', '_poly_id')}
                temp_attrs.append(attr)

        if not temp_lines:
            update_console("[AVISO] Nenhuma linha foi gerada inicialmente com os parâmetros fornecidos.\n")
            on_complete(False)
            return

        gdf_temp = gpd.GeoDataFrame(temp_attrs, geometry=temp_lines, crs=metric_crs)

        if method == "PCA v6 (Integrate & Dissolve)":
            update_console(f"[INFO] Aplicando Integrate e Dissolve (Tolerância: {tolerance}m)...\n")
            gdf_final = join_close_lines(gdf_temp, tolerance=tolerance)
        else:
            gdf_final = gdf_temp

        if not gdf_final.empty:
            gdf_final = gdf_final.explode(index_parts=False).reset_index(drop=True)
            
            # Garantir diretório da saída
            out_dir = os.path.dirname(os.path.abspath(output_path))
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
                
            gdf_final.to_crs(original_crs).to_file(output_path)
            update_console(f"\n[OK] Sucesso! {len(gdf_final)} linhas salvas em:\n{output_path}\n")
            on_complete(True)
        else:
            update_console("[AVISO] O processamento resultou em um conjunto de dados vazio.\n")
            on_complete(False)

    except Exception as e:
        import traceback
        err_msg = traceback.format_exc()
        update_console(f"\n[ERRO CRÍTICO] Falha durante a execução:\n{err_msg}\n")
        on_complete(False)

# -----------------------------------------------------------------------------
# TEXT REDIRECTOR FOR CONSOLE
# -----------------------------------------------------------------------------

class ConsoleRedirector:
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, message):
        self.text_widget.configure(state='normal')
        self.text_widget.insert(tk.END, message)
        self.text_widget.see(tk.END)
        self.text_widget.configure(state='disabled')
        self.text_widget.update_idletasks()

    def flush(self):
        pass

# -----------------------------------------------------------------------------
# TKINTER WINDOW CLASS
# -----------------------------------------------------------------------------

class GeradorLinhasApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Gerador de Linha de Centro - CitrosRows / Bemagro")
        self.geometry("760x680")
        self.minsize(700, 600)
        
        # Style
        self.style = ttk.Style()
        self.style.theme_use("clam")
        
        # Color palette - Sleek slate blue
        self.style.configure("TFrame", background="#F3F4F6")
        self.style.configure("TLabel", background="#F3F4F6", foreground="#374151", font=("Segoe UI", 10))
        self.style.configure("Header.TLabel", background="#F3F4F6", foreground="#1F2937", font=("Segoe UI", 12, "bold"))
        self.style.configure("TButton", font=("Segoe UI", 9, "bold"), borderwidth=1)
        self.style.configure("Action.TButton", background="#2563EB", foreground="white", font=("Segoe UI", 10, "bold"))
        self.style.map("Action.TButton", background=[("active", "#1D4ED8")])
        self.style.configure("Browse.TButton", background="#E5E7EB", foreground="#374151")
        self.style.map("Browse.TButton", background=[("active", "#D1D5DB")])
        self.configure(background="#F3F4F6")
        
        self.create_widgets()
        
    def create_widgets(self):
        # Container principal com margens
        main_frame = ttk.Frame(self, padding="20 20 20 20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Título
        title_label = ttk.Label(main_frame, text="Gerador de Linhas Centrais de Plantio", font=("Segoe UI", 16, "bold"), foreground="#1E3A8A")
        title_label.pack(anchor=tk.W, pady=(0, 15))
        
        # --- SEÇÃO DE ARQUIVOS ---
        files_frame = ttk.LabelFrame(main_frame, text=" Arquivos de Entrada e Saída ", padding="15 15 15 15")
        files_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Linhas de Plantio
        ttk.Label(files_frame, text="Polígonos de Linha de Plantio (Required):").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.poly_entry = ttk.Entry(files_frame, width=60)
        self.poly_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)
        ttk.Button(files_frame, text="Procurar...", style="Browse.TButton", command=self.browse_poly).grid(row=0, column=2, padx=5, pady=5)
        
        # Contagem de Plantas (Pontos)
        ttk.Label(files_frame, text="Pontos de Contagem (Required):").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.pts_entry = ttk.Entry(files_frame, width=60)
        self.pts_entry.grid(row=1, column=1, padx=5, pady=5, sticky=tk.EW)
        ttk.Button(files_frame, text="Procurar...", style="Browse.TButton", command=self.browse_pts).grid(row=1, column=2, padx=5, pady=5)
        
        # Talhão/Área (Opcional)
        ttk.Label(files_frame, text="Área/Talhão Limite (Optional):").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.boundary_entry = ttk.Entry(files_frame, width=60)
        self.boundary_entry.grid(row=2, column=1, padx=5, pady=5, sticky=tk.EW)
        ttk.Button(files_frame, text="Procurar...", style="Browse.TButton", command=self.browse_boundary).grid(row=2, column=2, padx=5, pady=5)
        
        # Arquivo de Saída
        ttk.Label(files_frame, text="Caminho do Arquivo de Saída:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.out_entry = ttk.Entry(files_frame, width=60)
        self.out_entry.grid(row=3, column=1, padx=5, pady=5, sticky=tk.EW)
        ttk.Button(files_frame, text="Salvar como...", style="Browse.TButton", command=self.browse_output).grid(row=3, column=2, padx=5, pady=5)
        
        files_frame.columnconfigure(1, weight=1)
        
        # --- SEÇÃO DE CONFIGURAÇÃO ---
        config_frame = ttk.LabelFrame(main_frame, text=" Parâmetros de Configuração ", padding="15 15 15 15")
        config_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Escolha do Método
        ttk.Label(config_frame, text="Método de Extração:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.method_combo = ttk.Combobox(config_frame, values=["PCA v6 (Integrate & Dissolve)", "PCA v4 (Ignore Holes)"], state="readonly", width=30)
        self.method_combo.set("PCA v6 (Integrate & Dissolve)")
        self.method_combo.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        self.method_combo.bind("<<ComboboxSelected>>", self.on_method_change)
        
        # Min Pontos
        ttk.Label(config_frame, text="Mínimo de Pontos por Polígono:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.min_pts_spin = ttk.Spinbox(config_frame, from_=2, to=100, width=10)
        self.min_pts_spin.set(5)
        self.min_pts_spin.grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)
        
        # Suavização
        self.smooth_label = ttk.Label(config_frame, text="Suavização (Spline Factor):")
        self.smooth_label.grid(row=2, column=0, sticky=tk.W, pady=5)
        self.smooth_spin = ttk.Spinbox(config_frame, from_=0.0, to=10.0, increment=0.1, width=10)
        self.smooth_spin.set(0.5)
        self.smooth_spin.grid(row=2, column=1, padx=5, pady=5, sticky=tk.W)
        
        # Tolerância Snap
        self.tolerance_label = ttk.Label(config_frame, text="Tolerância de Snap/Integração (metros):")
        self.tolerance_label.grid(row=3, column=0, sticky=tk.W, pady=5)
        self.tolerance_spin = ttk.Spinbox(config_frame, from_=0.0, to=10.0, increment=0.1, width=10)
        self.tolerance_spin.set(0.5)
        self.tolerance_spin.grid(row=3, column=1, padx=5, pady=5, sticky=tk.W)
        
        # --- SEÇÃO DE EXECUÇÃO E CONSOLE ---
        ttk.Label(main_frame, text="Logs de Processamento:", style="Header.TLabel").pack(anchor=tk.W, pady=(0, 5))
        
        console_frame = ttk.Frame(main_frame)
        console_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
        
        self.console_text = tk.Text(console_frame, height=10, state="disabled", font=("Consolas", 9), background="#1F2937", foreground="#F9FAFB")
        self.console_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(console_frame, orient="vertical", command=self.console_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.console_text.configure(yscrollcommand=scrollbar.set)
        
        # Botão de Ação Principal
        self.run_button = ttk.Button(main_frame, text="GERAR LINHAS DE CENTRO", style="Action.TButton", command=self.start_processing)
        self.run_button.pack(fill=tk.X, ipady=8)
        
    def on_method_change(self, event=None):
        method = self.method_combo.get()
        if method == "PCA v6 (Integrate & Dissolve)":
            self.smooth_label.config(text="Suavização (Spline Factor):")
            self.smooth_spin.set(0.5)
            self.tolerance_label.grid()
            self.tolerance_spin.grid()
        else:
            self.smooth_label.config(text="Janela de Suavização (Média Móvel):")
            self.smooth_spin.set(5)
            self.tolerance_label.grid_remove()
            self.tolerance_spin.grid_remove()
            
    def browse_poly(self):
        filename = filedialog.askopenfilename(filetypes=[("GeoJSON / Shapefiles", "*.geojson;*.shp"), ("Todos os Arquivos", "*.*")])
        if filename:
            self.poly_entry.delete(0, tk.END)
            self.poly_entry.insert(0, filename)
            
    def browse_pts(self):
        filename = filedialog.askopenfilename(filetypes=[("GeoJSON / Shapefiles", "*.geojson;*.shp"), ("Todos os Arquivos", "*.*")])
        if filename:
            self.pts_entry.delete(0, tk.END)
            self.pts_entry.insert(0, filename)
            
    def browse_boundary(self):
        filename = filedialog.askopenfilename(filetypes=[("GeoJSON / Shapefiles", "*.geojson;*.shp"), ("Todos os Arquivos", "*.*")])
        if filename:
            self.boundary_entry.delete(0, tk.END)
            self.boundary_entry.insert(0, filename)
            
    def browse_output(self):
        filename = filedialog.asksaveasfilename(defaultextension=".geojson", filetypes=[("GeoJSON", "*.geojson"), ("Shapefile", "*.shp"), ("Todos os Arquivos", "*.*")])
        if filename:
            self.out_entry.delete(0, tk.END)
            self.out_entry.insert(0, filename)
            
    def update_console(self, text):
        self.console_text.configure(state="normal")
        self.console_text.insert(tk.END, text)
        self.console_text.see(tk.END)
        self.console_text.configure(state="disabled")
        self.console_text.update_idletasks()
        
    def start_processing(self):
        # Validação
        poly = self.poly_entry.get().strip()
        pts = self.pts_entry.get().strip()
        boundary = self.boundary_entry.get().strip()
        output = self.out_entry.get().strip()
        
        if not poly or not pts or not output:
            messagebox.showerror("Erro de Validação", "Por favor, preencha os campos obrigatórios:\n- Polígonos de Linha de Plantio\n- Pontos de Contagem\n- Arquivo de Saída")
            return
            
        method = self.method_combo.get()
        try:
            min_points = int(self.min_pts_spin.get())
            smoothing = float(self.smooth_spin.get())
            tolerance = float(self.tolerance_spin.get())
        except ValueError:
            messagebox.showerror("Erro de Entrada", "Os parâmetros numéricos (Min Pontos, Suavização, Tolerância) devem ser válidos.")
            return
            
        self.run_button.config(state="disabled")
        self.console_text.configure(state="normal")
        self.console_text.delete("1.0", tk.END)
        self.console_text.configure(state="disabled")
        
        # Executar em uma thread separada para não travar a interface
        thread = threading.Thread(
            target=run_centerline_generation,
            args=(method, poly, pts, boundary, output, min_points, smoothing, tolerance, self.update_console, self.on_processing_complete)
        )
        thread.daemon = True
        thread.start()
        
    def on_processing_complete(self, success):
        self.run_button.config(state="normal")
        if success:
            messagebox.showinfo("Sucesso", "Processamento concluído com sucesso!")
        else:
            messagebox.showerror("Erro", "Ocorreu um erro durante o processamento. Veja o log do console para mais detalhes.")

if __name__ == "__main__":
    app = GeradorLinhasApp()
    app.mainloop()
