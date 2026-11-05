#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
mesclar_mdts.py
--------------------------------------------------------------------------------
Script para unir/mosaicar múltiplos rasters de MDT (GeoTIFF) de voos diferentes
em um único arquivo unificado para injeção de altitude nos talhões e linhas.
--------------------------------------------------------------------------------
"""

import os
import sys
import argparse
import rasterio
from rasterio.merge import merge

def mesclar_mdts(mdt_paths, output_path):
    print(f"[INFO] Abrindo {len(mdt_paths)} arquivo(s) de MDT...")
    src_files_to_mosaic = []
    
    for path in mdt_paths:
        if not os.path.exists(path):
            print(f"[ERRO] Arquivo não encontrado: {path}")
            sys.exit(1)
        src = rasterio.open(path)
        print(f"  - Carregado: {path} (CRS: {src.crs}, Dimensões: {src.width}x{src.height})")
        src_files_to_mosaic.append(src)

    print("[INFO] Executando o mosaico (merge)...")
    mosaic, out_trans = merge(src_files_to_mosaic)

    out_meta = src_files_to_mosaic[0].meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_trans,
        "crs": src_files_to_mosaic[0].crs,
        "nodata": src_files_to_mosaic[0].nodata
    })

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with rasterio.open(output_path, "w", **out_meta) as dest:
        dest.write(mosaic)

    for src in src_files_to_mosaic:
        src.close()

    print(f"\n[SUCESSO] MDT unificado gerado em: {output_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Junta 2 ou mais rasters MDT (GeoTIFF) em um único arquivo raster unificado."
    )
    parser.add_argument(
        "--inputs", "-i", nargs="+", required=True,
        help="Caminhos dos rasters MDT de entrada (separados por espaço)."
    )
    parser.add_argument(
        "--output", "-o", required=True,
        help="Caminho do arquivo GeoTIFF de saída unificado (ex: mdt_unificado.tif)."
    )

    args = parser.parse_args()
    mesclar_mdts(args.inputs, args.output)

if __name__ == "__main__":
    main()
