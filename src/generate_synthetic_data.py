import os
import json
import shapely.geometry as geom
import geopandas as gpd
from pyproj import Transformer

def generate_data(output_dir="data/synthetic"):
    os.makedirs(output_dir, exist_ok=True)
    
    # GPKG real bounds (EPSG:32722):
    # left=699013.60, bottom=7478362.37, right=705283.21, top=7481742.50
    # Use a 400m block near the center of the raster
    center_x = 702148   # (699013 + 705283) / 2
    center_y = 7480052  # (7478362 + 7481742) / 2
    size = 400  # 400m x 400m block
    
    # 1. Create Talhão (Block) Polygon in UTM
    min_x, max_x = center_x - size/2, center_x + size/2
    min_y, max_y = center_y - size/2, center_y + size/2
    
    talhao_poly = geom.Polygon([
        (min_x, min_y),
        (min_x, max_y),
        (max_x, max_y),
        (max_x, min_y),
        (min_x, min_y)
    ])
    
    # 2. Create parallel Centerlines spaced 6.5 meters apart inside the block
    # We will orient them at an angle of 15 degrees to represent realistic rows
    import numpy as np
    spacing = 6.5  # 6.5m row spacing as requested
    lines = []
    
    # Generate line starting points along an axis orthogonal to the rows
    angle_rad = np.radians(15)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    
    # Rotate standard grid lines
    # Row spacing is along the rotated X axis, line extends along rotated Y axis
    # We cover from -size to size to ensure full coverage of the block
    for offset in np.arange(-size, size, spacing):
        # Line from bottom to top, shifted by offset
        # Rotated coordinates
        x1 = center_x + offset * cos_a - (size) * sin_a
        y1 = center_y + offset * sin_a + (size) * cos_a
        
        x2 = center_x + offset * cos_a + (size) * sin_a
        y2 = center_y + offset * sin_a - (size) * cos_a
        
        line = geom.LineString([(x1, y1), (x2, y2)])
        # Intersect line with the talhao polygon to keep it inside
        clipped_line = line.intersection(talhao_poly)
        if not clipped_line.is_empty:
            if clipped_line.geom_type == 'LineString':
                lines.append(clipped_line)
            elif clipped_line.geom_type == 'MultiLineString':
                for part in clipped_line.geoms:
                    lines.append(part)
                    
    # Create GeoDataFrames in UTM (EPSG:32722)
    talhao_gdf = gpd.GeoDataFrame(geometry=[talhao_poly], crs="EPSG:32722")
    lines_gdf = gpd.GeoDataFrame(geometry=lines, crs="EPSG:32722")
    
    # 3. Reproject to WGS 84 (EPSG:4326) as requested
    talhao_wgs84 = talhao_gdf.to_crs("EPSG:4326")
    lines_wgs84 = lines_gdf.to_crs("EPSG:4326")
    
    # 4. Save to GeoJSON and Shapefile
    # Save GeoJSON
    talhao_wgs84.to_file(os.path.join(output_dir, "talhao.geojson"), driver="GeoJSON")
    lines_wgs84.to_file(os.path.join(output_dir, "linhas_centrais.geojson"), driver="GeoJSON")
    
    # Save Shapefiles
    shp_dir = os.path.join(output_dir, "shp")
    os.makedirs(shp_dir, exist_ok=True)
    talhao_wgs84.to_file(os.path.join(shp_dir, "talhao.shp"))
    lines_wgs84.to_file(os.path.join(shp_dir, "linhas_centrais.shp"))
    
    print(f"Synthetic vectors generated successfully in {output_dir}")
    print(f"Generated {len(lines)} parallel rows inside the talhão.")

if __name__ == "__main__":
    generate_data()
