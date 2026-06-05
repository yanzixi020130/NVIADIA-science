"""PhysicsNeMo Mesh adapter for PyVista meshes and point-cloud reconstruction."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np


SUPPORTED_MESH_FILE_EXTENSIONS = {".vtp", ".vtk", ".stl", ".ply", ".obj"}
SUPPORTED_RECONSTRUCTION_MODES = {"delaunay_2d", "surface_reconstruction"}


def is_physicsnemo_mesh_available() -> bool:
    """Return whether physicsnemo.mesh can be imported in the current runtime."""
    try:
        from physicsnemo.mesh import Mesh  # noqa: F401
    except Exception:
        return False
    return True


def _import_torch():
    try:
        import torch
    except Exception as exc:
        raise RuntimeError(f"PyTorch is required for PhysicsNeMo Mesh: {exc}") from exc
    return torch


def _import_pyvista():
    try:
        import pyvista as pv
    except Exception as exc:
        raise RuntimeError(f"PyVista is required for mesh analysis: {exc}") from exc
    return pv


def _torch_device(device: Optional[str]):
    torch = _import_torch()
    requested = device or "cuda"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA device but torch.cuda.is_available() is False")
    return torch.device(requested)


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def to_jsonable(value: Any) -> Any:
    """Convert tensors, arrays, and scalar values to JSON-serializable objects."""
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, np.generic):
        return to_jsonable(value.item())
    if hasattr(value, "detach") or hasattr(value, "numpy"):
        return to_jsonable(_as_numpy(value))
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return to_jsonable(value.item())
        array = value.astype(float, copy=False) if np.issubdtype(value.dtype, np.number) else value
        if np.issubdtype(array.dtype, np.number):
            array = np.where(np.isfinite(array), array, np.nan)
            return [[None if np.isnan(item) else float(item) for item in row]
                    for row in array.reshape(array.shape[0], -1)] if array.ndim > 1 else [
                        None if np.isnan(item) else float(item) for item in array.tolist()
                    ]
        return array.tolist()
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]
    return str(value)


def _triangulated_pyvista_mesh(pv_mesh: Any) -> Any:
    mesh = pv_mesh.extract_surface() if hasattr(pv_mesh, "extract_surface") else pv_mesh
    if hasattr(mesh, "triangulate"):
        mesh = mesh.triangulate()
    if getattr(mesh, "n_cells", 0) < 1:
        raise ValueError("Mesh contains no cells")
    return mesh


def _extract_triangles(pv_mesh: Any) -> np.ndarray:
    mesh = _triangulated_pyvista_mesh(pv_mesh)
    faces = np.asarray(getattr(mesh, "faces", []), dtype=np.int64)
    if faces.size == 0:
        raise ValueError("Mesh has no polygon faces to convert")

    triangles = []
    cursor = 0
    while cursor < len(faces):
        size = int(faces[cursor])
        cursor += 1
        indices = faces[cursor:cursor + size]
        cursor += size
        if size == 3:
            triangles.append(indices)
        elif size > 3:
            first = indices[0]
            for i in range(1, size - 1):
                triangles.append([first, indices[i], indices[i + 1]])
    if not triangles:
        raise ValueError("Mesh has no triangular faces after triangulation")
    return np.asarray(triangles, dtype=np.int64)


def _attach_data_from_pyvista(mesh: Any, pv_mesh: Any, device: Any) -> None:
    torch = _import_torch()
    point_data = getattr(pv_mesh, "point_data", {})
    cell_data = getattr(pv_mesh, "cell_data", {})

    for key in point_data.keys():
        values = np.asarray(point_data[key])
        if values.size:
            mesh.point_data[str(key)] = torch.as_tensor(values, dtype=torch.float32, device=device)
    for key in cell_data.keys():
        values = np.asarray(cell_data[key])
        if values.size:
            mesh.cell_data[str(key)] = torch.as_tensor(values, dtype=torch.float32, device=device)


def pyvista_to_physicsnemo_mesh(pv_mesh: Any, device: Optional[str] = "cuda") -> Any:
    """Convert a PyVista mesh to physicsnemo.mesh.Mesh."""
    target_device = _torch_device(device)
    pv_mesh = _triangulated_pyvista_mesh(pv_mesh)

    try:
        from physicsnemo.mesh.io import from_pyvista

        mesh = from_pyvista(pv_mesh)
        if hasattr(mesh, "to"):
            mesh = mesh.to(target_device)
        return mesh
    except Exception:
        pass

    try:
        from physicsnemo.mesh import Mesh
    except Exception as exc:
        raise RuntimeError(f"physicsnemo.mesh is required for mesh analysis: {exc}") from exc

    torch = _import_torch()
    points = torch.as_tensor(np.asarray(pv_mesh.points), dtype=torch.float32, device=target_device)
    cells = torch.as_tensor(_extract_triangles(pv_mesh), dtype=torch.long, device=target_device)
    mesh = Mesh(points=points, cells=cells)
    _attach_data_from_pyvista(mesh, pv_mesh, target_device)
    return mesh


def read_mesh_file(path: Path, device: Optional[str] = "cuda") -> Tuple[Any, Any]:
    """Read a mesh file with PyVista and convert it to PhysicsNeMo Mesh."""
    pv = _import_pyvista()
    if path.suffix.lower() not in SUPPORTED_MESH_FILE_EXTENSIONS:
        raise ValueError(f"Unsupported mesh file extension '{path.suffix}'")
    pv_mesh = pv.read(str(path))
    return pyvista_to_physicsnemo_mesh(pv_mesh, device=device), pv_mesh


def point_cloud_to_pyvista_mesh(
    points: Sequence[Sequence[Optional[float]]],
    scalars: Sequence[Optional[float]],
    mode: str,
    projection_plane: str = "xy",
    delaunay_alpha: Optional[float] = None,
    surface_method: str = "reconstruct_surface",
    surface_alpha: Optional[float] = None,
    nbr_sz: int = 20,
    sample_spacing: Optional[float] = None,
) -> Any:
    """Build a PyVista triangular mesh from point-cloud samples."""
    if mode not in SUPPORTED_RECONSTRUCTION_MODES:
        raise ValueError(
            "CSV point clouds require visualization_mode='delaunay_2d' or "
            "visualization_mode='surface_reconstruction' before PhysicsNeMo Mesh analysis"
        )
    from .point_cloud_visualizer import _build_mesh

    points_np = np.asarray(points, dtype=float)
    scalars_np = np.asarray(scalars, dtype=float) if len(scalars) == len(points) else None
    _, mesh, _ = _build_mesh(
        points_np,
        scalars_np,
        mode,
        projection_plane,
        delaunay_alpha,
        surface_method,
        surface_alpha,
        nbr_sz,
        sample_spacing,
        False,
        25,
        None,
    )
    if mesh is None:
        raise ValueError("Point-cloud reconstruction did not produce a mesh")
    return _triangulated_pyvista_mesh(mesh)


def _call_optional(mesh: Any, name: str) -> Any:
    method = getattr(mesh, name, None)
    if callable(method):
        try:
            return to_jsonable(method())
        except Exception as exc:
            return {"error": str(exc)}
    return None


def _data_keys(container: Any) -> list:
    if container is None:
        return []
    if hasattr(container, "keys"):
        return sorted(str(key) for key in container.keys())
    return []


def _mesh_array(mesh: Any, name: str) -> Optional[np.ndarray]:
    value = getattr(mesh, name, None)
    if callable(value):
        value = value()
    if value is None:
        return None
    return _as_numpy(value)


def _mesh_bounds(points: np.ndarray) -> Dict[str, list]:
    if points.size == 0:
        return {"x": [None, None], "y": [None, None], "z": [None, None]}
    if points.shape[1] < 3:
        padded = np.zeros((points.shape[0], 3), dtype=points.dtype)
        padded[:, :points.shape[1]] = points
        points = padded
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    return {
        "x": [float(mins[0]), float(maxs[0])],
        "y": [float(mins[1]), float(maxs[1])],
        "z": [float(mins[2]), float(maxs[2])],
    }


def _validation_report(mesh: Any) -> Dict[str, Any]:
    validate = getattr(mesh, "validate", None)
    if not callable(validate):
        return {"available": False}
    try:
        report = validate()
    except Exception as exc:
        return {"available": True, "error": str(exc)}
    if isinstance(report, dict):
        converted = to_jsonable(report)
    elif hasattr(report, "__dict__"):
        converted = to_jsonable(vars(report))
    else:
        converted = to_jsonable(report)
    return {"available": True, "report": converted}


def _validation_valid(validation: Dict[str, Any]) -> Optional[bool]:
    report = validation.get("report")
    if isinstance(report, dict):
        for key in ("valid", "is_valid", "success", "ok"):
            if isinstance(report.get(key), bool):
                return bool(report[key])
    if validation.get("error"):
        return False
    return None


def analyze_mesh(mesh: Any, include_centroids: int = 5) -> Dict[str, Any]:
    """Return stable JSON-compatible mesh quality and geometry statistics."""
    points = _mesh_array(mesh, "points")
    cells = _mesh_array(mesh, "cells")
    if points is None:
        raise ValueError("PhysicsNeMo Mesh does not expose points")
    if cells is None:
        raise ValueError("PhysicsNeMo Mesh does not expose cells")

    areas = _mesh_array(mesh, "cell_areas")
    centroids = _mesh_array(mesh, "cell_centroids")
    validation = _validation_report(mesh)
    valid = _validation_valid(validation)

    result: Dict[str, Any] = {
        "engine": "physicsnemo.mesh",
        "device": str(getattr(getattr(mesh, "points", None), "device", "unknown")),
        "valid": valid,
        "num_points": int(len(points)),
        "num_cells": int(len(cells)),
        "bounds": _mesh_bounds(np.asarray(points)),
        "area_sum": None,
        "area_min": None,
        "area_max": None,
        "area_mean": None,
        "centroid_sample": [],
        "validation": validation,
        "is_manifold": _call_optional(mesh, "is_manifold"),
        "is_watertight": _call_optional(mesh, "is_watertight"),
        "has_point_data": _data_keys(getattr(mesh, "point_data", None)),
        "has_cell_data": _data_keys(getattr(mesh, "cell_data", None)),
    }

    if areas is not None and areas.size:
        flat = np.asarray(areas, dtype=float).reshape(-1)
        finite = flat[np.isfinite(flat)]
        if finite.size:
            result.update({
                "area_sum": float(finite.sum()),
                "area_min": float(finite.min()),
                "area_max": float(finite.max()),
                "area_mean": float(finite.mean()),
            })
    if centroids is not None and include_centroids > 0:
        result["centroid_sample"] = to_jsonable(np.asarray(centroids)[:include_centroids])

    return result


def compute_scalar_gradient(mesh: Any, key: str, method: str = "lsq") -> Any:
    """Compute a scalar point-data gradient using PhysicsNeMo when available."""
    compute = getattr(mesh, "compute_point_derivatives", None)
    if not callable(compute):
        raise RuntimeError("This PhysicsNeMo Mesh version does not expose compute_point_derivatives")
    result = compute(keys=key, method=method)
    return result if result is not None else mesh


def _container_get(container: Any, key: str) -> Any:
    if container is None:
        return None
    try:
        return container[key]
    except Exception:
        return None


def _field_array(mesh: Any, key: str) -> Tuple[np.ndarray, str]:
    value = _container_get(getattr(mesh, "point_data", None), key)
    if value is not None:
        return _as_numpy(value), "point"
    value = _container_get(getattr(mesh, "cell_data", None), key)
    if value is not None:
        return _as_numpy(value), "cell"
    raise ValueError(f"Field '{key}' was not found in mesh point_data or cell_data")


def _normalized_cells(mesh: Any) -> np.ndarray:
    cells = _mesh_array(mesh, "cells")
    if cells is None:
        raise ValueError("Mesh does not expose cells")
    cells = np.asarray(cells, dtype=np.int64)
    if cells.ndim != 2 or cells.shape[1] < 3:
        raise ValueError("Mesh cells must be a 2D array with at least three vertices per cell")
    return cells[:, :3]


def _normalized_points(mesh: Any) -> np.ndarray:
    points = _mesh_array(mesh, "points")
    if points is None:
        raise ValueError("Mesh does not expose points")
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] < 3:
        padded = np.zeros((points.shape[0], 3), dtype=float)
        padded[:, :points.shape[1]] = points
        points = padded
    return points[:, :3]


def _cell_areas(mesh: Any, points: Optional[np.ndarray] = None, cells: Optional[np.ndarray] = None) -> np.ndarray:
    areas = _mesh_array(mesh, "cell_areas")
    if areas is not None:
        return np.asarray(areas, dtype=float).reshape(-1)
    points = _normalized_points(mesh) if points is None else points
    cells = _normalized_cells(mesh) if cells is None else cells
    vectors_a = points[cells[:, 1]] - points[cells[:, 0]]
    vectors_b = points[cells[:, 2]] - points[cells[:, 0]]
    return 0.5 * np.linalg.norm(np.cross(vectors_a, vectors_b), axis=1)


def _cell_normals(points: np.ndarray, cells: np.ndarray) -> np.ndarray:
    vectors_a = points[cells[:, 1]] - points[cells[:, 0]]
    vectors_b = points[cells[:, 2]] - points[cells[:, 0]]
    normals = np.cross(vectors_a, vectors_b)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    return np.divide(normals, lengths, out=np.zeros_like(normals), where=lengths > 0)


def _field_to_cell_values(values: np.ndarray, location: str, cells: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    if location == "cell":
        return values
    return values[cells].mean(axis=1)


def _integral_value(values: np.ndarray, areas: np.ndarray) -> Any:
    weighted = values * areas.reshape(-1, 1)
    integrated = weighted.sum(axis=0)
    if integrated.size == 1:
        return float(integrated[0])
    return integrated.tolist()


def surface_integral(mesh: Any, key: str) -> Dict[str, Any]:
    """Integrate a scalar or vector field over mesh surface cells."""
    cells = _normalized_cells(mesh)
    points = _normalized_points(mesh)
    areas = _cell_areas(mesh, points, cells)
    values, location = _field_array(mesh, key)
    cell_values = _field_to_cell_values(values, location, cells)
    finite = np.isfinite(cell_values).all(axis=1) & np.isfinite(areas)
    if not finite.any():
        raise ValueError(f"Field '{key}' has no finite values available for integration")
    return {
        "operation": "surface_integral",
        "field_key": key,
        "field_location": location,
        "num_cells_used": int(finite.sum()),
        "area_sum": float(areas[finite].sum()),
        "integral": _integral_value(cell_values[finite], areas[finite]),
    }


def _vector_from_component_keys(mesh: Any, vector_keys: Sequence[str], cells: np.ndarray) -> Tuple[np.ndarray, str]:
    if len(vector_keys) != 3:
        raise ValueError("vector_keys must contain exactly three component field names")
    arrays = []
    locations = []
    for key in vector_keys:
        values, location = _field_array(mesh, key)
        arrays.append(_field_to_cell_values(values, location, cells).reshape(len(cells), -1)[:, 0])
        locations.append(location)
    location = locations[0] if all(item == locations[0] for item in locations) else "mixed"
    return np.column_stack(arrays), location


def _vector_field_to_cells(mesh: Any, key: str, cells: np.ndarray) -> Tuple[np.ndarray, str]:
    values, location = _field_array(mesh, key)
    cell_values = _field_to_cell_values(values, location, cells)
    if cell_values.ndim != 2 or cell_values.shape[1] < 3:
        raise ValueError(f"Field '{key}' is not a 3-component vector field")
    return cell_values[:, :3], location


def flux_integral(
    mesh: Any,
    field_key: Optional[str] = None,
    vector_keys: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Integrate vector-field flux through mesh surface cells."""
    cells = _normalized_cells(mesh)
    points = _normalized_points(mesh)
    areas = _cell_areas(mesh, points, cells)
    normals = _cell_normals(points, cells)
    if vector_keys:
        vectors, location = _vector_from_component_keys(mesh, vector_keys, cells)
        source = list(vector_keys)
    elif field_key:
        vectors, location = _vector_field_to_cells(mesh, field_key, cells)
        source = field_key
    else:
        raise ValueError("flux_integral requires either field_key or vector_keys")

    density = np.einsum("ij,ij->i", vectors, normals)
    finite = np.isfinite(density) & np.isfinite(areas)
    if not finite.any():
        raise ValueError("Vector field has no finite values available for flux integration")
    return {
        "operation": "flux_integral",
        "field_key": source,
        "field_location": location,
        "num_cells_used": int(finite.sum()),
        "area_sum": float(areas[finite].sum()),
        "flux_integral": float((density[finite] * areas[finite]).sum()),
        "flux_density_mean": float(density[finite].mean()),
    }


def scalar_gradient_report(mesh: Any, key: str, method: str = "lsq", sample_limit: int = 5) -> Dict[str, Any]:
    """Compute and summarize a scalar gradient field."""
    _field_array(mesh, key)
    updated = compute_scalar_gradient(mesh, key, method=method)
    gradient_key = f"{key}_gradient"
    gradient = _container_get(getattr(updated, "point_data", None), gradient_key)
    if gradient is None:
        gradient = _container_get(getattr(mesh, "point_data", None), gradient_key)
    if gradient is None:
        component_keys = [f"{key}__x", f"{key}__y", f"{key}__z"]
        components = []
        for component_key in component_keys:
            component = _container_get(getattr(updated, "point_data", None), component_key)
            if component is None:
                component = _container_get(getattr(mesh, "point_data", None), component_key)
            components.append(component)
        if all(component is not None for component in components):
            gradient = np.column_stack([_as_numpy(component).reshape(-1) for component in components])
            gradient_key = component_keys
    if gradient is None:
        raise RuntimeError(
            f"compute_point_derivatives completed but no '{gradient_key}' field was found in point_data"
        )

    gradient_np = np.asarray(_as_numpy(gradient), dtype=float)
    if gradient_np.ndim == 1:
        gradient_np = gradient_np.reshape(-1, 1)
    norms = np.linalg.norm(gradient_np, axis=1)
    finite = np.isfinite(norms)
    return {
        "operation": "scalar_gradient",
        "field_key": key,
        "gradient_key": gradient_key,
        "method": method,
        "num_points": int(len(gradient_np)),
        "gradient_sample": to_jsonable(gradient_np[:sample_limit]),
        "gradient_norm_min": float(norms[finite].min()) if finite.any() else None,
        "gradient_norm_max": float(norms[finite].max()) if finite.any() else None,
        "gradient_norm_mean": float(norms[finite].mean()) if finite.any() else None,
    }


def analyze_field(
    mesh: Any,
    operation: str,
    field_key: Optional[str] = None,
    vector_keys: Optional[Sequence[str]] = None,
    method: str = "lsq",
    sample_limit: int = 5,
) -> Dict[str, Any]:
    """Run a PhysicsNeMo Mesh field-analysis operation and return JSON-safe data."""
    op = operation.lower().strip()
    if op in {"temperature_gradient", "stress_gradient", "scalar_gradient"}:
        key = field_key or op.removesuffix("_gradient")
        return scalar_gradient_report(mesh, key, method=method, sample_limit=sample_limit)
    if op == "surface_integral":
        if not field_key:
            raise ValueError("surface_integral requires field_key")
        return surface_integral(mesh, field_key)
    if op == "flux_integral":
        return flux_integral(mesh, field_key=field_key, vector_keys=vector_keys)
    raise ValueError(
        "operation must be one of temperature_gradient, stress_gradient, "
        "scalar_gradient, surface_integral, or flux_integral"
    )
