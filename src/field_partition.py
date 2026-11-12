"""Spatially continuous planting blocks, measured in a local metric CRS.

Orientation is axial (modulo 180 degrees). A sparse graph of local orientation
cells keeps curved rows together without using their endpoint bearing. Raster
labels define a joint partition; the original vector boundary and holes remain
authoritative. No crop-free manoeuvring corridor is inferred at an internal cut.
"""

import json
import math
import os

import geopandas as gpd
import numpy as np
import rasterio.features
from rasterio.transform import from_origin
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from shapely import line_interpolate_point, make_valid
from shapely.geometry import LineString, shape
from shapely.ops import unary_union
from shapely.strtree import STRtree
from skimage.segmentation import watershed


DEFAULTS = {
    "enabled": True,
    # Same angular scale as the existing duplicate/parallelism check. This is
    # a local compatibility threshold, never a global bearing bin.
    "angle_tolerance_deg": 20.0,
    # Two row spacings make a cell observe neighbouring rows, not just one
    # vertex; three independent rows are the minimum block evidence.
    "cell_spacing_factor": 2.0,
    "min_rows": 3,
    "min_support_spacing_factor": 2.0,
    "sample_step_m": None,
    "grid_resolution_m": None,
    "max_grid_cells": 4_000_000,
}


def parts(geom, kind):
    if geom is None or geom.is_empty:
        return []
    if geom.geom_type == kind:
        return [geom]
    if hasattr(geom, "geoms"):
        return [part for child in geom.geoms for part in parts(child, kind)]
    return []


def _polygonal(geom):
    """Discard zero-area intersection artefacts before polygon export."""
    return unary_union(parts(make_valid(geom), "Polygon"))


def project_polygons(gdf, crs):
    """Repair sub-ULP ring contacts introduced by CRS transformation."""
    projected = gdf.to_crs(crs)
    projected.geometry = projected.geometry.map(_polygonal)
    if not projected.is_valid.all():
        raise ValueError("Reprojeção produziu um limite de talhão inválido.")
    return projected


def metric_frame(gdf):
    if gdf.crs is None:
        raise ValueError("Separação de talhões exige CRS definido.")
    axes = gdf.crs.axis_info
    if gdf.crs.is_projected and axes and all(
        abs(axis.unit_conversion_factor - 1.0) < 1e-9 for axis in axes[:2]
    ):
        target = gdf.crs
    else:
        target = gdf.estimate_utm_crs()
    if gdf.geom_type.isin(["Polygon", "MultiPolygon"]).all():
        return project_polygons(gdf, target)
    return gdf.to_crs(target)


def _options(options):
    cfg = dict(DEFAULTS)
    if options is not None and not isinstance(options, dict):
        raise ValueError("field_partition deve ser um mapa de parâmetros YAML/JSON.")
    if options:
        unknown = set(options) - set(cfg)
        if unknown:
            raise ValueError(f"Parâmetros de separação desconhecidos: {sorted(unknown)}")
        cfg.update(options)
    if not isinstance(cfg["enabled"], bool):
        raise ValueError("field_partition.enabled deve ser true ou false.")
    if not 0 < float(cfg["angle_tolerance_deg"]) < 45:
        raise ValueError("angle_tolerance_deg deve estar entre 0 e 45 graus.")
    if int(cfg["min_rows"]) < 2:
        raise ValueError("min_rows deve ser pelo menos 2.")
    for key in ("cell_spacing_factor", "min_support_spacing_factor", "max_grid_cells"):
        if float(cfg[key]) <= 0:
            raise ValueError(f"{key} deve ser positivo.")
    for key in ("sample_step_m", "grid_resolution_m"):
        if cfg[key] is not None and float(cfg[key]) <= 0:
            raise ValueError(f"{key} deve ser positivo.")
    return cfg


def _samples(lines, step, tangent_window):
    xy, angles, weights, owners = [], [], [], []
    for owner, line in enumerate(lines):
        count = max(1, int(math.ceil(line.length / step)))
        distances = (np.arange(count) + 0.5) * line.length / count
        points = line_interpolate_point(line, distances)
        before = line_interpolate_point(line, np.maximum(0, distances - tangent_window))
        after = line_interpolate_point(line, np.minimum(line.length, distances + tangent_window))
        for p, a, b in zip(points, before, after):
            dx, dy = b.x - a.x, b.y - a.y
            if math.hypot(dx, dy) <= 1e-9:
                continue
            xy.append((p.x, p.y))
            angles.append(math.atan2(dy, dx))
            weights.append(line.length / count)
            owners.append(owner)
    return np.asarray(xy).reshape(-1, 2), np.asarray(angles), np.asarray(weights), np.asarray(owners)


def _estimate_spacing(lines, minimum, maximum):
    """Estimate spacing with bounded local normal probes, not all-pairs distances."""
    if not lines:
        return (minimum + maximum) / 2
    tree = STRtree(lines)
    distances = []
    # Bounded diagnostic sample independent of farm size. No threshold is
    # inferred from the sampling count; it only bounds estimator work.
    for i in np.linspace(0, len(lines) - 1, min(len(lines), 256), dtype=int):
        line = lines[i]
        d = line.length / 2
        p = line.interpolate(d)
        a, b = line.interpolate(max(0, d - minimum / 2)), line.interpolate(min(line.length, d + minimum / 2))
        tangent = np.array([b.x - a.x, b.y - a.y])
        norm = np.linalg.norm(tangent)
        if norm == 0:
            continue
        normal = np.array([-tangent[1], tangent[0]]) / norm
        center = np.array([p.x, p.y])
        probe = LineString([center - maximum * normal, center + maximum * normal])
        hits = []
        for j in tree.query(probe, predicate="intersects"):
            if i == j:
                continue
            for hit in parts(probe.intersection(lines[j]), "Point"):
                distance = p.distance(hit)
                if minimum <= distance <= maximum:
                    hits.append(distance)
        if hits:
            distances.append(min(hits))
    return float(np.median(distances)) if distances else (minimum + maximum) / 2


def _orientation_groups(lines, domain, spacing, maximum_distance, cfg):
    step = float(cfg["sample_step_m"] or spacing / 2)
    xy, angle, weight, owner = _samples(lines, step, spacing / 2)
    if not len(xy):
        return [], {"spacing_m": spacing, "samples": 0}
    cell_size = spacing * float(cfg["cell_spacing_factor"])
    origin = np.asarray(domain.bounds[:2])
    cells = np.floor((xy - origin) / cell_size).astype(np.int64)
    unique, inverse = np.unique(cells, axis=0, return_inverse=True)
    total = np.bincount(inverse, weights=weight)
    cosine = np.bincount(inverse, weights=weight * np.cos(2 * angle))
    sine = np.bincount(inverse, weights=weight * np.sin(2 * angle))
    concentration = np.hypot(cosine, sine) / total
    orientations = np.arctan2(sine, cosine) / 2
    tolerance = np.radians(float(cfg["angle_tolerance_deg"]))
    # For a symmetric angular spread +/- tolerance, R=cos(2*tolerance).
    # Mixed crossing cells below that evidence level cannot join blocks.
    reliable = concentration >= np.cos(2 * tolerance)
    ordered_samples = np.argsort(inverse, kind="stable")
    cell_samples = np.split(ordered_samples, np.flatnonzero(np.diff(inverse[ordered_samples])) + 1)
    cell_trees = {}
    lookup = {tuple(cell): i for i, cell in enumerate(unique) if reliable[i]}
    edges_a, edges_b = [], []
    for (x, y), i in lookup.items():
        for dx, dy in ((1, 0), (0, 1), (1, 1), (-1, 1)):
            j = lookup.get((x + dx, y + dy))
            if j is None:
                continue
            delta = abs(orientations[i] - orientations[j]) % np.pi
            if min(delta, np.pi - delta) > tolerance:
                continue
            # Cell adjacency alone can bridge a road or an empty gap. Require
            # actual sample proximity within the existing row-neighbour range.
            if i not in cell_trees:
                cell_trees[i] = cKDTree(xy[cell_samples[i]])
            distances, neighbors = cell_trees[i].query(xy[cell_samples[j]], distance_upper_bound=maximum_distance)
            nearest = int(np.argmin(distances))
            if not np.isfinite(distances[nearest]):
                continue
            a = xy[cell_samples[i][neighbors[nearest]]]
            b = xy[cell_samples[j][nearest]]
            # Do not let a graph edge jump across a vector exclusion.
            bridge = LineString([a, b])
            if domain.covers(bridge):
                edges_a.append(i)
                edges_b.append(j)
    graph = sparse.coo_matrix((np.ones(len(edges_a)), (edges_a, edges_b)), shape=(len(unique), len(unique)))
    _, labels = connected_components(graph, directed=False)
    candidates = []
    sample_labels = labels[inverse]
    reliable_samples = np.flatnonzero(reliable[inverse])
    ordered = reliable_samples[np.argsort(sample_labels[reliable_samples], kind="stable")]
    sample_groups = np.split(ordered, np.flatnonzero(np.diff(sample_labels[ordered])) + 1)
    for selected in sample_groups:
        if not len(selected):
            continue
        row_count = len(np.unique(owner[selected]))
        support = float(weight[selected].sum())
        if row_count < int(cfg["min_rows"]) or support < cfg["min_rows"] * spacing * cfg["min_support_spacing_factor"]:
            continue
        # Many fragments along one physical row are not independent rows.
        z = np.sum(weight[selected] * np.exp(2j * angle[selected]))
        theta = float(np.angle(z) / 2)
        across = np.array([-np.sin(theta), np.cos(theta)])
        span = float(np.ptp(xy[selected] @ across))
        if span < (int(cfg["min_rows"]) - 1) * spacing:
            continue
        candidates.append({"xy": xy[selected], "rows": row_count, "support_m": support,
                           "angle_deg": float(np.degrees(theta) % 180),
                           "coherence": float(abs(z) / support)})
    candidates.sort(key=lambda group: tuple(np.min(group["xy"], axis=0)))
    return candidates, {"spacing_m": spacing, "samples": len(xy), "orientation_cells": len(unique),
                        "ambiguous_cells": int((~reliable).sum()), "candidate_groups": len(candidates)}


def _grid(domain, resolution, max_cells):
    minx, miny, maxx, maxy = domain.bounds
    width = max(1, int(math.ceil((maxx - minx) / resolution)))
    height = max(1, int(math.ceil((maxy - miny) / resolution)))
    if width * height > max_cells:
        # Fail explicitly rather than silently changing boundary precision.
        raise ValueError(f"Partição exige {width * height:,} células; limite {max_cells:,}. "
                         "Aumente grid_resolution_m ou max_grid_cells.")
    transform = from_origin(minx, maxy, resolution, resolution)
    mask = rasterio.features.rasterize([(domain, 1)], out_shape=(height, width), transform=transform,
                                       all_touched=True, dtype="uint8").astype(bool)
    return transform, mask


def _polygon_partition(domain, markers, transform, mask):
    """Flood all labels together, then restore the exact vector domain."""
    flooded = watershed(np.zeros(mask.shape, dtype=np.uint8), markers=markers, mask=mask,
                        connectivity=1).astype(np.int32)
    grouped = {}
    for geometry, label in rasterio.features.shapes(flooded, mask=flooded > 0, transform=transform):
        grouped.setdefault(int(label), []).append(shape(geometry))
    polygons = {label: _polygonal(unary_union(geoms).intersection(domain)) for label, geoms in grouped.items()}
    polygons = {label: geom for label, geom in polygons.items() if not geom.is_empty}
    # Very narrow vector tails/holes may not be represented by pixel centres.
    # Restore uncovered vector pieces without changing shared boundaries.
    covered = unary_union(list(polygons.values()))
    for leftover in parts(make_valid(domain.difference(covered)), "Polygon"):
        if not polygons:
            raise ValueError("Partição sem sementes no domínio.")
        label = max(polygons, key=lambda key: (leftover.boundary.intersection(polygons[key].boundary).length,
                                               -leftover.distance(polygons[key]), -key))
        polygons[label] = _polygonal(polygons[label].union(leftover))
    return polygons


def _split_component(domain, lines, minimum, maximum, cfg):
    spacing = _estimate_spacing(lines, minimum, maximum)
    groups, diagnostics = _orientation_groups(lines, domain, spacing, maximum, cfg)
    if len(groups) < 2:
        return [(domain, groups[0] if groups else {})], diagnostics
    resolution = float(cfg["grid_resolution_m"] or min(1.0, spacing / 4))
    transform, mask = _grid(domain, resolution, int(cfg["max_grid_cells"]))
    markers = np.zeros(mask.shape, dtype=np.int32)
    for label, group in enumerate(groups, 1):
        columns = np.floor((group["xy"][:, 0] - transform.c) / resolution).astype(int)
        rows = np.floor((transform.f - group["xy"][:, 1]) / resolution).astype(int)
        rows = np.clip(rows, 0, mask.shape[0] - 1)
        columns = np.clip(columns, 0, mask.shape[1] - 1)
        markers[rows, columns] = label
    polygons = _polygon_partition(domain, markers, transform, mask)
    result = []
    for label, polygon in sorted(polygons.items()):
        components = parts(make_valid(polygon), "Polygon")
        for component in components:
            # Tiny disconnected raster remnants do not become new fields.
            if component.area < spacing * spacing * cfg["min_rows"]:
                continue
            result.append((component, groups[label - 1]))
    if len(result) < 2:
        diagnostics["decision"] = "insufficient_area_support"
        return [(domain, {})], diagnostics
    # Reassign discarded slivers to a touching retained region, jointly.
    missing = domain.difference(unary_union([p for p, _ in result]))
    for sliver in parts(make_valid(missing), "Polygon"):
        best = max(range(len(result)), key=lambda i: (
            sliver.boundary.intersection(result[i][0].boundary).length,
            -sliver.distance(result[i][0]), -i))
        geom, group = result[best]
        result[best] = (_polygonal(geom.union(sliver)), group)
    diagnostics["grid_resolution_m"] = resolution
    return result, diagnostics


def partition_lines(lines_gdf, parents_gdf, options=None, min_spacing=4.0, max_spacing=8.5, logger=print):
    """Return clipped rows, child fields and diagnostics, preserving parent IDs."""
    cfg = _options(options)
    if min_spacing <= 0 or max_spacing <= min_spacing:
        raise ValueError("Faixa de espaçamento inválida para separar talhões.")
    parents = metric_frame(parents_gdf).reset_index(drop=True)
    rows = lines_gdf.to_crs(parents.crs).reset_index(drop=True)
    source_lines = [(i, part) for i, geom in enumerate(rows.geometry) for part in parts(geom, "LineString")]
    line_geometries = [line for _, line in source_lines]
    tree = STRtree(line_geometries)
    fields, clipped, diagnostics = [], [], []
    child_counts = {}
    for parent_no, parent in parents.iterrows():
        parent_id = str(parent.get("parent_id", f"P{parent_no + 1:05d}"))
        child_no = child_counts.get(parent_id, 0)
        first_child = child_no
        for component in parts(make_valid(parent.geometry), "Polygon"):
            local = []
            for candidate in tree.query(component, predicate="intersects"):
                local.extend(parts(line_geometries[candidate].intersection(component), "LineString"))
            if cfg["enabled"] and local:
                children, diagnostic = _split_component(component, local, min_spacing, max_spacing, cfg)
            else:
                children, diagnostic = [(component, {})], {"decision": "disabled" if not cfg["enabled"] else "no_rows"}
            diagnostics.append({"parent_id": parent_id, **diagnostic, "children": len(children)})
            for child, group in children:
                child_no += 1
                field_id = f"{parent_id}_{child_no:03d}"
                record = parent.drop(labels=[parents.geometry.name]).to_dict()
                record.update(parent_id=parent_id, field_id=field_id, area_m2=child.area,
                              angle_deg=group.get("angle_deg"), n_rows=group.get("rows", len(local)),
                              support_m=group.get("support_m", 0.0), geometry=child)
                fields.append(record)
                for candidate in tree.query(child, predicate="intersects"):
                    source_index, source_line = source_lines[candidate]
                    for piece in parts(source_line.intersection(child), "LineString"):
                        if piece.length <= 1e-7:
                            continue
                        attributes = rows.iloc[source_index].drop(labels=[rows.geometry.name]).to_dict()
                        attributes.update(parent_id=parent_id, field_id=field_id, row_id=f"{parent_id}_R{source_index + 1:07d}",
                                          length_m=piece.length, area_m2=0.0, geometry=piece)
                        clipped.append(attributes)
        child_counts[parent_id] = child_no
        logger(f"  [Talhões] {parent_id}: {child_no - first_child} região(ões).")
    fields_gdf = gpd.GeoDataFrame(fields, geometry="geometry", crs=parents.crs)
    if clipped:
        clipped_gdf = gpd.GeoDataFrame(clipped, geometry="geometry", crs=parents.crs)
    else:
        clipped_gdf = gpd.GeoDataFrame(columns=["parent_id", "field_id", "row_id", "length_m", "area_m2", "geometry"],
                                       geometry="geometry", crs=parents.crs)
    _validate_partition(fields_gdf, parents)
    return clipped_gdf.to_crs(lines_gdf.crs), project_polygons(fields_gdf, parents_gdf.crs), diagnostics


def _validate_partition(fields, parents):
    if "field_id" in fields.columns and fields["field_id"].duplicated().any():
        raise ValueError("Identificadores de talhão duplicados na partição.")
    if not fields.geom_type.isin(["Polygon", "MultiPolygon"]).all() or fields.is_empty.any():
        raise ValueError("Talhões devem ser polígonos não vazios.")
    domain = unary_union([make_valid(g) for g in parents.geometry])
    coverage = unary_union(list(fields.geometry))
    # Projection/serialization can move touching rings by a few coordinate
    # ULPs. Bound the resulting area error by perimeter times roundoff, not
    # by an agricultural distance threshold that could hide real overlaps.
    roundoff = np.spacing(max(abs(value) for value in domain.bounds)) * domain.length * 32
    tolerance = max(1e-6, domain.area * 1e-10, roundoff)
    if not fields.is_valid.all() or coverage.symmetric_difference(domain).area > tolerance:
        raise ValueError("Partição inválida: limites/exclusões ou cobertura não preservados.")
    if fields.area.sum() - coverage.area > tolerance:
        raise ValueError("Talhões sobrepostos na entrada ou na partição; corrija os limites antes de exportar.")


def export_partition(lines_gdf, parents_gdf, output_dir, options=None, min_spacing=4.0, max_spacing=8.5, logger=print):
    rows, fields, diagnostics = partition_lines(lines_gdf, parents_gdf, options, min_spacing, max_spacing, logger)
    os.makedirs(output_dir, exist_ok=True)
    fields_path = os.path.join(output_dir, "talhoes_separados.geojson")
    fields = project_polygons(fields, 4326)
    fields.to_file(fields_path, driver="GeoJSON")
    shp_dir = os.path.join(output_dir, "shp")
    os.makedirs(shp_dir, exist_ok=True)
    # Stable short field names avoid Shapefile's ten-character truncation.
    fields[["parent_id", "field_id", "area_m2", "angle_deg", "n_rows", "support_m", "geometry"]].to_file(
        os.path.join(shp_dir, "TALHOES_SEPARADOS.shp"), encoding="UTF-8")
    with open(os.path.join(output_dir, "separacao_metadata.json"), "w", encoding="utf-8") as stream:
        json.dump({"options": _options(options), "parents": diagnostics}, stream, ensure_ascii=False, indent=2)
    return rows, fields_path


def build_borders(fields_gdf, distance, resolution=1.0, max_cells=4_000_000):
    """Allocate external buffers jointly; an internal cut never invades a sibling.

    These are geometric boundary routes, not proof of an available driving lane.
    Existing holes are subtracted even when a positive buffer would close them.
    """
    if distance < 0:
        raise ValueError("Distância da bordadura deve ser não negativa.")
    if resolution <= 0 or max_cells < 1:
        raise ValueError("Resolução e limite de células da bordadura devem ser positivos.")
    fields = metric_frame(fields_gdf).reset_index(drop=True)
    original = unary_union(list(fields.geometry))
    holes = [shape({"type": "Polygon", "coordinates": [list(ring.coords)]})
             for poly in parts(original, "Polygon") for ring in poly.interiors]
    expanded = unary_union(list(fields.geometry.buffer(distance))).difference(unary_union(holes))
    if distance == 0 or len(fields) == 1:
        fields.geometry = [_polygonal(geom.buffer(distance).difference(unary_union(holes))) for geom in fields.geometry]
        return project_polygons(fields, fields_gdf.crs)
    partition = {}
    tree = STRtree(list(fields.geometry))
    for component in parts(expanded, "Polygon"):
        indices = tree.query(component, predicate="intersects")
        transform, mask = _grid(component, resolution, max_cells)
        markers = rasterio.features.rasterize([(fields.geometry.iloc[i], int(i) + 1) for i in indices],
                                             out_shape=mask.shape, transform=transform, dtype="int32")
        # A narrow field must not silently disappear at the chosen resolution.
        if len(np.unique(markers[markers > 0])) != len(indices):
            raise ValueError("Resolução da bordadura omitiu um talhão; reduza grid_resolution_m.")
        for label, geom in _polygon_partition(component, markers, transform, mask).items():
            partition[label] = partition[label].union(geom) if label in partition else geom
    external = expanded.difference(original)
    fields.geometry = [_polygonal(geom.union(partition[i + 1].intersection(external))) for i, geom in enumerate(fields.geometry)]
    _validate_partition(fields, gpd.GeoDataFrame(geometry=[expanded], crs=fields.crs))
    return project_polygons(fields, fields_gdf.crs)
