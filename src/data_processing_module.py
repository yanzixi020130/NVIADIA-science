import json
import os
import struct
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from fastapi import APIRouter, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

try:
    from dotenv import load_dotenv
    load_dotenv(override=False)
except Exception:
    pass

from .data_field_mapper import FieldDetection, detect_fields, parse_column_list
from .physicsnemo_mesh_adapter import (
    SUPPORTED_MESH_FILE_EXTENSIONS,
    SUPPORTED_RECONSTRUCTION_MODES,
    analyze_field,
    analyze_mesh,
    is_physicsnemo_mesh_available,
    point_cloud_to_pyvista_mesh,
    pyvista_to_physicsnemo_mesh,
    read_mesh_file,
)
from .point_cloud_visualizer import render_pyvista_dataset, render_static_visualization, render_time_animation
from .text_to_3d_generator import (
    SUPPORTED_GENERATED_TYPES,
    TextTo3DGenerationError,
    call_text_to_3d_llm,
    normalize_text_to_3d_payload,
)

try:
    import pandas as pd
except Exception:
    pd = None


BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_ROOT = BASE_DIR / "upload"
IMAGE_ROOT = BASE_DIR / "images"
MAX_PREVIEW_ROWS = 100
MAX_POINT_ROWS = 5000
MAX_MESH_TRIANGLES = 8000
MAX_SAMPLE_POINTS = 200000
MAX_RETURN_POINTS = 20000

router = APIRouter(prefix="/data-processing", tags=["data-processing"])


class GeometrySampleRequest(BaseModel):
    taskid: str = "default"
    dataset_id: Optional[str] = None
    filename: Optional[str] = None
    path: Optional[str] = None
    sample_points: int = Field(default=10000, ge=1, le=MAX_SAMPLE_POINTS)
    return_points: int = Field(default=5000, ge=1, le=MAX_RETURN_POINTS)
    include_boundary: bool = True
    include_interior: bool = True
    compute_sdf_derivatives: bool = True
    device: Optional[str] = None
    seed: Optional[int] = None


class MeshAnalyzeRequest(BaseModel):
    taskid: str = "default"
    dataset_id: Optional[str] = None
    filename: Optional[str] = None
    path: Optional[str] = None
    device: str = "cuda"
    include_centroids: int = Field(default=5, ge=0, le=100)
    max_points: int = Field(default=MAX_POINT_ROWS, ge=1, le=MAX_SAMPLE_POINTS)
    coord_cols: Optional[str] = None
    color_col: Optional[str] = None
    time_col: Optional[str] = None
    vector_cols: Optional[str] = None
    visualization_mode: str = "point_cloud"
    projection_plane: str = "xy"
    delaunay_alpha: Optional[float] = None
    surface_method: str = "reconstruct_surface"
    surface_alpha: Optional[float] = None
    nbr_sz: int = Field(default=20, ge=1, le=200)
    sample_spacing: Optional[float] = None
    auto_color: bool = True
    auto_time: bool = True
    fill_box_boundary: bool = False
    box_boundary_resolution: int = Field(default=25, ge=2, le=200)


class MeshFieldAnalysisRequest(MeshAnalyzeRequest):
    operation: str = "temperature_gradient"
    field_key: Optional[str] = None
    vector_keys: Optional[List[str]] = None
    method: str = "lsq"
    sample_limit: int = Field(default=5, ge=0, le=100)


class TextTo3DRequest(BaseModel):
    taskid: str = "default"
    prompt: str = Field(..., min_length=1)
    output_type: str = "auto"
    max_points: int = Field(default=3000, ge=1, le=MAX_SAMPLE_POINTS)
    max_faces: int = Field(default=6000, ge=1, le=MAX_SAMPLE_POINTS)
    include_mesh_analysis: bool = True
    device: str = "cuda"
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    timeout: float = Field(default=60.0, ge=1.0, le=300.0)
    show_edges: bool = True
    show_grid: bool = True
    mesh_opacity: float = Field(default=1.0, ge=0.0, le=1.0)
    mesh_color: str = "#9bbfc2"
    mesh_edge_color: str = "#111827"
    mesh_edge_width: float = Field(default=1.0, ge=0.1, le=10.0)
    colormap: str = "viridis"
    camera_zoom: float = Field(default=1.0, gt=0.0, le=10.0)
    canvas_size: int = Field(default=1000, ge=200, le=4000)
    plot_title: Optional[str] = None
    x_axis_title: Optional[str] = None
    y_axis_title: Optional[str] = None
    z_axis_title: Optional[str] = None
    include_raw_response: bool = False


def _safe_filename(filename: str) -> str:
    name = Path(filename or "uploaded-file").name
    return name.replace("\\", "_").replace("/", "_")


def _dataset_dir(taskid: str) -> Path:
    safe_taskid = "".join(ch for ch in taskid if ch.isalnum() or ch in "-_")[:80] or "default"
    path = UPLOAD_ROOT / safe_taskid / "data_processing"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _image_dir(taskid: str) -> Path:
    safe_taskid = "".join(ch for ch in taskid if ch.isalnum() or ch in "-_")[:80] or "default"
    path = IMAGE_ROOT / "data_processing" / safe_taskid
    path.mkdir(parents=True, exist_ok=True)
    return path


def _detect_type(filename: str, columns: Optional[List[str]] = None) -> str:
    ext = Path(filename).suffix.lower()
    lower_columns = {c.strip().lower() for c in columns or []}
    if ext in {".csv", ".txt", ".dat"}:
        if {"x", "y", "z"}.issubset(lower_columns):
            return "point_cloud"
        return "table"
    if ext in {".json", ".jsonl"}:
        return "json"
    if ext == ".stl":
        return "geometry_stl"
    if ext in {".obj", ".ply", ".vtp", ".vtk"}:
        return "geometry"
    return "unknown"


def _read_json_preview(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in text.splitlines()[:MAX_PREVIEW_ROWS] if line.strip()]
        return {"rows": rows, "row_preview_count": len(rows)}
    data = json.loads(text)
    if isinstance(data, list):
        return {"rows": data[:MAX_PREVIEW_ROWS], "row_preview_count": min(len(data), MAX_PREVIEW_ROWS)}
    if isinstance(data, dict):
        return {"object": data}
    return {"value": data}


def _read_stl_preview(path: Path) -> Dict[str, Any]:
    raw = path.read_bytes()
    if len(raw) < 84:
        text = raw.decode("utf-8", errors="ignore")
    else:
        text = ""
    triangles: List[List[List[float]]] = []
    triangle_count = 0
    if len(raw) >= 84:
        binary_count = struct.unpack("<I", raw[80:84])[0]
        expected_size = 84 + binary_count * 50
        if expected_size == len(raw):
            triangle_count = int(binary_count)
            offset = 84
            for _ in range(min(triangle_count, MAX_MESH_TRIANGLES)):
                values = struct.unpack("<12fH", raw[offset:offset + 50])
                triangles.append([
                    [values[3], values[4], values[5]],
                    [values[6], values[7], values[8]],
                    [values[9], values[10], values[11]],
                ])
                offset += 50
    if not triangles:
        text = text or raw.decode("utf-8", errors="ignore")
        current: List[List[float]] = []
        for line in text.splitlines():
            parts = line.strip().split()
            if len(parts) == 4 and parts[0].lower() == "vertex":
                try:
                    current.append([float(parts[1]), float(parts[2]), float(parts[3])])
                except ValueError:
                    current = []
            if len(current) == 3:
                triangle_count += 1
                if len(triangles) < MAX_MESH_TRIANGLES:
                    triangles.append(current)
                current = []
    if not triangles:
        raise HTTPException(status_code=400, detail="No STL triangles found")
    points = np.array([p for tri in triangles for p in tri], dtype=float)
    return {
        "triangle_count": triangle_count or len(triangles),
        "triangles_preview_count": len(triangles),
        "triangles": triangles,
        "bounds": {"min": points.min(axis=0).tolist(), "max": points.max(axis=0).tolist()},
    }


def _bad_table_request(message: str, columns: Sequence[str]) -> None:
    raise HTTPException(status_code=400, detail={"message": message, "columns": list(columns)})


def _read_table_file(path: Path) -> "pd.DataFrame":
    """Read CSV/TXT/DAT files as headered tables according to the upload contract."""
    if pd is None:
        raise HTTPException(status_code=500, detail="pandas is required for table inspection")

    ext = path.suffix.lower()
    try:
        if ext == ".csv":
            df = pd.read_csv(path)
        elif ext in {".txt", ".dat"}:
            try:
                df = pd.read_csv(path)
                if len(df.columns) == 1 and len(str(df.columns[0]).split()) > 1:
                    df = pd.read_csv(path, sep=r"\s+", engine="python")
            except Exception:
                df = pd.read_csv(path, sep=r"\s+", engine="python")
        else:
            raise HTTPException(status_code=400, detail="Only .csv, .txt, and .dat files are supported for table inspection")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read table file: {exc}") from exc

    df.columns = [str(column).strip() for column in df.columns]
    return df


def _numeric_columns(df: "pd.DataFrame") -> List[str]:
    numeric_columns: List[str] = []
    for column in df.columns:
        values = pd.to_numeric(df[column], errors="coerce")
        if values.notna().any():
            numeric_columns.append(str(column))
    return numeric_columns


def _parse_mapping_inputs(
    columns: Sequence[str],
    coord_cols: Optional[str],
    color_col: Optional[str],
    time_col: Optional[str],
    vector_cols: Optional[str],
) -> Tuple[Optional[List[str]], Optional[str], Optional[str], Optional[List[str]]]:
    try:
        parsed_coord_cols = parse_column_list(coord_cols, "coord_cols")
        parsed_vector_cols = parse_column_list(vector_cols, "vector_cols")
    except ValueError as exc:
        _bad_table_request(str(exc), columns)

    parsed_color_col = color_col.strip() if color_col and color_col.strip() else None
    parsed_time_col = time_col.strip() if time_col and time_col.strip() else None
    return parsed_coord_cols, parsed_color_col, parsed_time_col, parsed_vector_cols


def _inspect_table_mapping(
    df: "pd.DataFrame",
    coord_cols: Optional[str],
    color_col: Optional[str],
    time_col: Optional[str],
    vector_cols: Optional[str],
    auto_color: bool = True,
    auto_time: bool = True,
) -> Tuple[List[str], List[str], FieldDetection]:
    columns = [str(column) for column in df.columns]
    numeric_columns = _numeric_columns(df)
    parsed_coord_cols, parsed_color_col, parsed_time_col, parsed_vector_cols = _parse_mapping_inputs(
        columns, coord_cols, color_col, time_col, vector_cols
    )
    try:
        detection = detect_fields(
            columns=columns,
            numeric_columns=numeric_columns,
            explicit_coord_cols=parsed_coord_cols,
            explicit_color_col=parsed_color_col,
            explicit_time_col=parsed_time_col,
            explicit_vector_cols=parsed_vector_cols,
            auto_color=auto_color,
            auto_time=auto_time,
        )
    except ValueError as exc:
        _bad_table_request(str(exc), columns)
    return columns, numeric_columns, detection


def _coordinate_bounds(df: "pd.DataFrame", coord_cols: Sequence[str]) -> Optional[List[List[float]]]:
    if not coord_cols:
        return None
    numeric_df = df[list(coord_cols)].apply(pd.to_numeric, errors="coerce")
    numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan).dropna(subset=list(coord_cols))
    if numeric_df.empty:
        return None
    mins = numeric_df.min(axis=0).astype(float).tolist()
    maxs = numeric_df.max(axis=0).astype(float).tolist()
    return [mins, maxs]


def _nullable_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except Exception:
        return None
    if np.isnan(number) or np.isinf(number):
        return None
    return number


def _records_from_columns(df: "pd.DataFrame", columns: Sequence[str]) -> List[List[Optional[float]]]:
    rows: List[List[Optional[float]]] = []
    for values in df[list(columns)].to_numpy().tolist():
        rows.append([_nullable_float(value) for value in values])
    return rows


def _build_table_inspection_response(
    taskid: str,
    filename: str,
    df: "pd.DataFrame",
    detection: FieldDetection,
    columns: Sequence[str],
    numeric_columns: Sequence[str],
) -> Dict[str, Any]:
    return {
        "success": True,
        "taskid": taskid,
        "filename": filename,
        "columns": list(columns),
        "num_rows": int(len(df)),
        "num_columns": int(len(columns)),
        "numeric_columns": list(numeric_columns),
        "detected": detection.detected_payload(),
        "available_scalar_fields": detection.available_scalar_fields,
        "available_time_fields": detection.available_time_fields,
        "available_vector_fields": detection.available_vector_fields,
        "need_user_confirm": detection.need_user_confirm,
    }


def _build_point_cloud_preview(
    df: "pd.DataFrame",
    detection: FieldDetection,
    columns: Sequence[str],
    numeric_columns: Sequence[str],
    max_points: int,
) -> Dict[str, Any]:
    if max_points < 1:
        _bad_table_request("max_points must be greater than 0", columns)
    if not detection.coord_cols:
        _bad_table_request("Unable to detect coordinate columns; pass coord_cols explicitly.", columns)

    required_columns: List[str] = list(detection.coord_cols)
    optional_columns = [detection.color_col] if detection.color_col else []
    optional_columns += [detection.time_col] if detection.time_col else []
    optional_columns += list(detection.vector_cols)
    preview_columns = list(dict.fromkeys(required_columns + [column for column in optional_columns if column]))

    numeric_df = df[preview_columns].apply(pd.to_numeric, errors="coerce")
    numeric_df = numeric_df.dropna(subset=detection.coord_cols)
    if numeric_df.empty:
        _bad_table_request("No rows contain valid numeric coordinate values.", columns)
    if len(numeric_df) > max_points:
        numeric_df = numeric_df.sample(n=max_points, random_state=42)

    points = _records_from_columns(numeric_df, detection.coord_cols)
    color_values = (
        [_nullable_float(value) for value in numeric_df[detection.color_col].tolist()]
        if detection.color_col
        else []
    )
    vector_values = _records_from_columns(numeric_df, detection.vector_cols) if detection.vector_cols else []
    time_values = (
        [_nullable_float(value) for value in numeric_df[detection.time_col].tolist()]
        if detection.time_col
        else []
    )

    return {
        "type": "point_cloud",
        "mapping": {
            "coord_cols": detection.coord_cols,
            "color_col": detection.color_col,
            "time_col": detection.time_col,
            "vector_cols": detection.vector_cols,
        },
        "field_info": {
            "columns": list(columns),
            "num_rows": int(len(df)),
            "num_columns": int(len(columns)),
            "numeric_columns": list(numeric_columns),
            "detected": detection.detected_payload(),
            "available_scalar_fields": detection.available_scalar_fields,
            "available_time_fields": detection.available_time_fields,
            "available_vector_fields": detection.available_vector_fields,
            "need_user_confirm": detection.need_user_confirm,
        },
        "preview": {
            "num_preview_points": int(len(points)),
            "points": points,
            "color_values": color_values,
            "time_values": time_values,
            "vector_values": vector_values,
        },
    }


def _build_preview(path: Path, filename: str) -> Dict[str, Any]:
    ext = path.suffix.lower()
    columns: Optional[List[str]] = None
    if ext in {".csv", ".txt", ".dat"}:
        df = _read_table_file(path)
        columns = [str(column) for column in df.columns]
        rows = df.head(MAX_PREVIEW_ROWS).replace({np.nan: None}).to_dict(orient="records")
        preview = {"columns": columns, "rows": rows, "row_preview_count": len(rows)}
    elif ext in {".json", ".jsonl"}:
        preview = _read_json_preview(path)
    elif ext == ".stl":
        preview = _read_stl_preview(path)
    else:
        preview = {"message": "Preview is not available for this file type yet."}
    return {"data_type": _detect_type(filename, columns), "preview": preview}


def _array_to_points(data: Dict[str, Any], limit: int) -> List[Dict[str, float]]:
    if not {"x", "y", "z"}.issubset(data):
        return []
    keys = [k for k in [
        "x", "y", "z", "normal_x", "normal_y", "normal_z",
        "sdf", "sdf__x", "sdf__y", "sdf__z", "area",
    ] if k in data]
    arrays = {k: np.asarray(data[k]).reshape(-1) for k in keys}
    count = min(limit, *(len(v) for v in arrays.values()))
    points = []
    for i in range(count):
        item = {}
        for key in keys:
            try:
                item[key] = float(arrays[key][i])
            except Exception:
                pass
        points.append(item)
    return points


def _find_dataset_file(taskid: str, dataset_id: Optional[str], filename: Optional[str]) -> Path:
    candidates = [p for p in _dataset_dir(taskid).iterdir() if p.is_file()]
    if dataset_id:
        candidates = [p for p in candidates if p.name.startswith(dataset_id)]
    if filename:
        safe = _safe_filename(filename)
        candidates = [p for p in candidates if p.name.endswith(safe) or p.name == safe]
    stl_candidates = [p for p in candidates if p.suffix.lower() == ".stl"]
    if stl_candidates:
        return sorted(stl_candidates)[-1]
    if candidates:
        return sorted(candidates)[-1]
    raise HTTPException(status_code=404, detail="Dataset file not found")


def _path_under_workspace(path: str) -> Path:
    resolved = Path(path).resolve()
    base_dir = BASE_DIR.resolve()
    try:
        resolved.relative_to(base_dir)
    except ValueError:
        raise HTTPException(status_code=400, detail="Path must be under the service workspace")
    if not resolved.exists():
        raise HTTPException(status_code=404, detail="File path not found")
    return resolved


def _find_mesh_analysis_file(taskid: str, dataset_id: Optional[str], filename: Optional[str]) -> Path:
    candidates = [p for p in _dataset_dir(taskid).iterdir() if p.is_file()]
    if dataset_id:
        candidates = [p for p in candidates if p.name.startswith(dataset_id)]
    if filename:
        safe = _safe_filename(filename)
        candidates = [p for p in candidates if p.name.endswith(safe) or p.name == safe]
    if not candidates:
        raise HTTPException(status_code=404, detail="Dataset file not found")
    return sorted(candidates)[-1]


def _resolve_geometry_path(req: GeometrySampleRequest) -> Path:
    if req.path:
        return _path_under_workspace(req.path)
    return _find_dataset_file(req.taskid, req.dataset_id, req.filename)


def _resolve_mesh_analysis_path(req: MeshAnalyzeRequest) -> Path:
    if req.path:
        return _path_under_workspace(req.path)
    return _find_mesh_analysis_file(req.taskid, req.dataset_id, req.filename)


def _geometry_engine_status() -> Dict[str, Any]:
    status: Dict[str, Any] = {}
    try:
        import physicsnemo  # type: ignore
        import physicsnemo.sym  # type: ignore
        status["physicsnemo_available"] = True
        status["physicsnemo_path"] = getattr(physicsnemo, "__file__", None)
    except Exception as exc:
        status["physicsnemo_available"] = False
        status["physicsnemo_error"] = str(exc)
    try:
        from src.geometry import Tessellation  # noqa: F401
        status["local_tessellation_available"] = True
    except Exception as exc:
        status["local_tessellation_available"] = False
        status["local_tessellation_error"] = str(exc)
    status["physicsnemo_mesh_available"] = is_physicsnemo_mesh_available()
    return status


def _load_tessellation(path: Path, req: GeometrySampleRequest):
    try:
        from physicsnemo.sym.geometry.tessellation_warp import Tessellation
        try:
            return "physicsnemo.sym.geometry.tessellation_warp.Tessellation", Tessellation.from_stl(
                str(path), airtight=True, device=req.device, seed=req.seed
            )
        except TypeError:
            return "physicsnemo.sym.geometry.tessellation_warp.Tessellation", Tessellation.from_stl(str(path), airtight=True)
    except Exception:
        from src.geometry import Tessellation
        return "src.geometry.tessellation.Tessellation", Tessellation.from_stl(str(path))


def _sample_geometry(req: GeometrySampleRequest) -> Dict[str, Any]:
    stl_path = _resolve_geometry_path(req)
    if stl_path.suffix.lower() != ".stl":
        raise HTTPException(status_code=400, detail="Geometry sampling currently requires an STL file")
    engine, geometry = _load_tessellation(stl_path, req)
    result: Dict[str, Any] = {
        "engine": engine,
        "source": str(stl_path),
        "sample_points": req.sample_points,
        "return_points": req.return_points,
    }
    if req.include_boundary:
        boundary = geometry.sample_boundary(nr_points=req.sample_points)
        result["boundary"] = {
            "points": _array_to_points(boundary, req.return_points),
            "keys": sorted(boundary.keys()),
            "total_points": int(req.sample_points),
        }
        if "area" in boundary:
            result["boundary"]["area_sum"] = float(np.asarray(boundary["area"]).sum())
    if req.include_interior:
        interior = geometry.sample_interior(
            nr_points=req.sample_points,
            compute_sdf_derivatives=req.compute_sdf_derivatives,
        )
        result["interior"] = {
            "points": _array_to_points(interior, req.return_points),
            "keys": sorted(interior.keys()),
            "total_points": int(req.sample_points),
        }
        if "area" in interior:
            result["interior"]["area_sum"] = float(np.asarray(interior["area"]).sum())
    return result


def _analyze_csv_point_cloud(path: Path, req: MeshAnalyzeRequest) -> Dict[str, Any]:
    df = _read_table_file(path)
    columns, numeric_columns, detection = _inspect_table_mapping(
        df,
        req.coord_cols,
        req.color_col,
        req.time_col,
        req.vector_cols,
        auto_color=req.auto_color,
        auto_time=req.auto_time,
    )
    point_preview = _build_point_cloud_preview(df, detection, columns, numeric_columns, req.max_points)
    pv_mesh = point_cloud_to_pyvista_mesh(
        points=point_preview["preview"]["points"],
        scalars=point_preview["preview"]["color_values"],
        mode=req.visualization_mode.lower(),
        projection_plane=req.projection_plane,
        delaunay_alpha=req.delaunay_alpha,
        surface_method=req.surface_method,
        surface_alpha=req.surface_alpha,
        nbr_sz=req.nbr_sz,
        sample_spacing=req.sample_spacing,
    )
    mesh = pyvista_to_physicsnemo_mesh(pv_mesh, device=req.device)
    analysis = analyze_mesh(mesh, include_centroids=req.include_centroids)
    return {
        "success": True,
        "source": str(path),
        "source_type": "point_cloud",
        "reconstruction": {
            "visualization_mode": req.visualization_mode.lower(),
            "projection_plane": req.projection_plane,
            "surface_method": req.surface_method,
            "num_input_points": int(point_preview["preview"]["num_preview_points"]),
        },
        "mapping": point_preview["mapping"],
        "field_info": point_preview["field_info"],
        **analysis,
    }


def _set_mesh_point_data_alias(mesh: Any, source_key: str, alias_key: Optional[str]) -> None:
    if not alias_key or alias_key == source_key:
        return
    point_data = getattr(mesh, "point_data", None)
    if point_data is None:
        return
    try:
        if source_key in point_data and alias_key not in point_data:
            point_data[alias_key] = point_data[source_key]
    except Exception:
        pass


def _set_mesh_point_data_values(mesh: Any, key: str, values: Any) -> None:
    point_data = getattr(mesh, "point_data", None)
    points = getattr(mesh, "points", None)
    if point_data is None or points is None:
        return
    try:
        import torch

        point_data[key] = torch.as_tensor(values, dtype=torch.float32, device=points.device)
    except Exception:
        pass


def _attach_csv_field_data(mesh: Any, point_preview: Dict[str, Any]) -> None:
    mapping = point_preview.get("mapping", {})
    preview = point_preview.get("preview", {})
    _set_mesh_point_data_alias(mesh, "color_values", mapping.get("color_col"))

    vector_cols = mapping.get("vector_cols") or []
    vector_values = preview.get("vector_values") or []
    if not vector_cols or not vector_values:
        return
    values_np = np.asarray(vector_values, dtype=float)
    if values_np.ndim != 2 or values_np.shape[1] != len(vector_cols):
        return
    for index, key in enumerate(vector_cols):
        _set_mesh_point_data_values(mesh, str(key), values_np[:, index])
    if len(vector_cols) == 3:
        _set_mesh_point_data_values(mesh, "_".join(str(key) for key in vector_cols), values_np[:, :3])


def _analyze_mesh_file(path: Path, req: MeshAnalyzeRequest) -> Dict[str, Any]:
    mesh, _ = read_mesh_file(path, device=req.device)
    analysis = analyze_mesh(mesh, include_centroids=req.include_centroids)
    return {
        "success": True,
        "source": str(path),
        "source_type": "mesh_file",
        **analysis,
    }


def _mesh_from_request(req: MeshAnalyzeRequest) -> Tuple[Any, Dict[str, Any]]:
    path = _resolve_mesh_analysis_path(req)
    ext = path.suffix.lower()
    if ext in SUPPORTED_MESH_FILE_EXTENSIONS:
        mesh, _ = read_mesh_file(path, device=req.device)
        return mesh, {"source": str(path), "source_type": "mesh_file"}
    if ext in {".csv", ".txt", ".dat"}:
        df = _read_table_file(path)
        columns, numeric_columns, detection = _inspect_table_mapping(
            df,
            req.coord_cols,
            req.color_col,
            req.time_col,
            req.vector_cols,
            auto_color=req.auto_color,
            auto_time=req.auto_time,
        )
        point_preview = _build_point_cloud_preview(df, detection, columns, numeric_columns, req.max_points)
        pv_mesh = point_cloud_to_pyvista_mesh(
            points=point_preview["preview"]["points"],
            scalars=point_preview["preview"]["color_values"],
            mode=req.visualization_mode.lower(),
            projection_plane=req.projection_plane,
            delaunay_alpha=req.delaunay_alpha,
            surface_method=req.surface_method,
            surface_alpha=req.surface_alpha,
            nbr_sz=req.nbr_sz,
            sample_spacing=req.sample_spacing,
        )
        mesh = pyvista_to_physicsnemo_mesh(pv_mesh, device=req.device)
        _attach_csv_field_data(mesh, point_preview)
        return mesh, {
            "source": str(path),
            "source_type": "point_cloud",
            "reconstruction": {
                "visualization_mode": req.visualization_mode.lower(),
                "projection_plane": req.projection_plane,
                "surface_method": req.surface_method,
                "num_input_points": int(point_preview["preview"]["num_preview_points"]),
            },
            "mapping": point_preview["mapping"],
            "field_info": point_preview["field_info"],
        }
    raise HTTPException(
        status_code=400,
        detail=(
            "mesh analysis supports mesh files "
            f"{sorted(SUPPORTED_MESH_FILE_EXTENSIONS)} or CSV/TXT/DAT point-cloud tables"
        ),
    )


def _analyze_mesh_request(req: MeshAnalyzeRequest) -> Dict[str, Any]:
    mesh, metadata = _mesh_from_request(req)
    return {"success": True, **metadata, **analyze_mesh(mesh, include_centroids=req.include_centroids)}


def _field_analysis_request(req: MeshFieldAnalysisRequest) -> Dict[str, Any]:
    mesh, metadata = _mesh_from_request(req)
    field_result = analyze_field(
        mesh,
        operation=req.operation,
        field_key=req.field_key,
        vector_keys=req.vector_keys,
        method=req.method,
        sample_limit=req.sample_limit,
    )
    return {
        "success": True,
        **metadata,
        "engine": "physicsnemo.mesh",
        "device": str(getattr(getattr(mesh, "points", None), "device", "unknown")),
        "field_analysis": field_result,
    }


def _upload_mesh_analysis(
    point_preview: Dict[str, Any],
    visualization_info: Dict[str, Any],
    visualization_mode: str,
    projection_plane: str,
    delaunay_alpha: Optional[float],
    surface_method: str,
    surface_alpha: Optional[float],
    nbr_sz: int,
    sample_spacing: Optional[float],
    device: str,
    include_centroids: int = 5,
) -> Dict[str, Any]:
    mode = visualization_mode.lower()
    if mode not in SUPPORTED_RECONSTRUCTION_MODES:
        return {}
    try:
        mesh_path = visualization_info.get("mesh_path")
        if mesh_path and Path(mesh_path).exists():
            mesh, _ = read_mesh_file(Path(mesh_path), device=device)
        else:
            pv_mesh = point_cloud_to_pyvista_mesh(
                points=point_preview["preview"]["points"],
                scalars=point_preview["preview"]["color_values"],
                mode=mode,
                projection_plane=projection_plane,
                delaunay_alpha=delaunay_alpha,
                surface_method=surface_method,
                surface_alpha=surface_alpha,
                nbr_sz=nbr_sz,
                sample_spacing=sample_spacing,
            )
            mesh = pyvista_to_physicsnemo_mesh(pv_mesh, device=device)
            _attach_csv_field_data(mesh, point_preview)
        return {"mesh_analysis": analyze_mesh(mesh, include_centroids=include_centroids)}
    except Exception as exc:
        return {"mesh_analysis_error": str(exc)}


def _import_pyvista_for_generated_geometry() -> Any:
    try:
        import pyvista as pv
    except Exception as exc:
        raise RuntimeError(f"PyVista is required for generated 3D geometry: {exc}") from exc
    pv.OFF_SCREEN = True
    return pv


def _generated_geometry_to_pyvista(geometry: Dict[str, Any]) -> Tuple[Any, Optional[str]]:
    pv = _import_pyvista_for_generated_geometry()
    points = np.asarray(geometry["points"], dtype=float)
    scalar_fields = geometry.get("scalars") if isinstance(geometry.get("scalars"), dict) else {}
    color_col = next(iter(scalar_fields.keys()), None)

    if geometry["type"] == "triangle_mesh":
        faces = np.asarray(geometry["faces"], dtype=np.int64)
        pv_faces = np.column_stack([np.full(len(faces), 3, dtype=np.int64), faces]).reshape(-1)
        dataset = pv.PolyData(points, pv_faces).triangulate()
    else:
        dataset = pv.PolyData(points)

    for key, values in scalar_fields.items():
        try:
            array = np.asarray(values, dtype=float)
        except Exception:
            continue
        if array.ndim == 1 and len(array) == len(points) and np.isfinite(array).all():
            dataset.point_data[str(key)] = array
    if color_col and color_col in dataset.point_data:
        dataset.point_data["color_values"] = dataset.point_data[color_col]
    return dataset, color_col


def _save_generated_geometry(
    taskid: str,
    dataset_id: str,
    geometry: Dict[str, Any],
    dataset: Any,
) -> Dict[str, str]:
    data_dir = _dataset_dir(taskid)
    geometry_path = data_dir / f"{dataset_id}_{geometry['type']}.json"
    dataset_path = data_dir / f"{dataset_id}_{geometry['type']}.vtp"
    geometry_path.write_text(json.dumps(geometry, ensure_ascii=False, indent=2), encoding="utf-8")
    dataset.save(dataset_path)
    return {
        "generated_json_path": str(geometry_path),
        "path": str(dataset_path),
    }


def _analyze_generated_mesh(dataset: Any, req: TextTo3DRequest) -> Dict[str, Any]:
    if not req.include_mesh_analysis:
        return {}
    if getattr(dataset, "n_cells", 0) < 1:
        return {}
    try:
        mesh = pyvista_to_physicsnemo_mesh(dataset, device=req.device)
        return {"mesh_analysis": analyze_mesh(mesh, include_centroids=5)}
    except Exception as exc:
        return {"mesh_analysis_error": str(exc)}


@router.get("/health")
def data_processing_health() -> Dict[str, Any]:
    return {"status": "ok", "module": "data-processing"}


@router.get("/physicsnemo/status")
def physicsnemo_status() -> Dict[str, Any]:
    return _geometry_engine_status()


@router.post("/inspect-table")
async def inspect_table(
    taskid: str = Form(...),
    file: UploadFile = File(...),
    coord_cols: Optional[str] = Form(None),
    color_col: Optional[str] = Form(None),
    time_col: Optional[str] = Form(None),
    vector_cols: Optional[str] = Form(None),
    auto_color: bool = Form(True),
    auto_time: bool = Form(True),
) -> JSONResponse:
    filename = _safe_filename(file.filename or "uploaded-file")
    if Path(filename).suffix.lower() not in {".csv", ".txt", ".dat"}:
        raise HTTPException(status_code=400, detail="inspect-table only supports .csv, .txt, and .dat files")

    dataset_id = f"inspect_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    save_path = _dataset_dir(taskid) / f"{dataset_id}_{filename}"
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")
    save_path.write_bytes(content)

    df = _read_table_file(save_path)
    columns, numeric_columns, detection = _inspect_table_mapping(
        df,
        coord_cols,
        color_col,
        time_col,
        vector_cols,
        auto_color=auto_color,
        auto_time=auto_time,
    )
    return JSONResponse(_build_table_inspection_response(taskid, filename, df, detection, columns, numeric_columns))


@router.post("/upload-preview")
async def upload_preview(
    file: UploadFile = File(...),
    taskid: str = Form("default"),
    coord_cols: Optional[str] = Form(None),
    color_col: Optional[str] = Form(None),
    time_col: Optional[str] = Form(None),
    vector_cols: Optional[str] = Form(None),
    max_points: int = Form(MAX_POINT_ROWS),
    max_frames: int = Form(60),
    visualization_mode: str = Form("point_cloud"),
    projection_plane: str = Form("xy"),
    delaunay_alpha: Optional[float] = Form(None),
    surface_method: str = Form("reconstruct_surface"),
    surface_alpha: Optional[float] = Form(None),
    nbr_sz: int = Form(20),
    sample_spacing: Optional[float] = Form(None),
    show_edges: bool = Form(False),
    show_grid: bool = Form(True),
    mesh_opacity: float = Form(1.0),
    mesh_color: str = Form("#9bbfc2"),
    mesh_edge_color: str = Form("#111827"),
    mesh_edge_width: float = Form(1.0),
    colormap: str = Form("viridis"),
    camera_zoom: float = Form(1.0),
    canvas_size: int = Form(1000),
    include_mesh_analysis: bool = Form(True),
    mesh_analysis_device: str = Form("cuda"),
    mesh_analysis_centroids: int = Form(5),
    auto_color: bool = Form(True),
    auto_time: bool = Form(True),
    fill_box_boundary: bool = Form(False),
    box_boundary_resolution: int = Form(25),
    plot_title: Optional[str] = Form(None),
    x_axis_title: Optional[str] = Form(None),
    y_axis_title: Optional[str] = Form(None),
    z_axis_title: Optional[str] = Form(None),
) -> JSONResponse:
    filename = _safe_filename(file.filename or "uploaded-file")
    dataset_id = f"ds_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    save_path = _dataset_dir(taskid) / f"{dataset_id}_{filename}"
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")
    save_path.write_bytes(content)

    if save_path.suffix.lower() in {".csv", ".txt", ".dat"}:
        mesh_opacity = max(0.0, min(float(mesh_opacity), 1.0))
        mesh_edge_width = max(0.1, min(float(mesh_edge_width), 10.0))
        camera_zoom = max(0.1, min(float(camera_zoom), 10.0))
        canvas_size = max(200, min(int(canvas_size), 4000))
        box_boundary_resolution = max(2, min(int(box_boundary_resolution), 200))
        axis_titles = (x_axis_title, y_axis_title, z_axis_title)
        df = _read_table_file(save_path)
        columns, numeric_columns, detection = _inspect_table_mapping(
            df,
            coord_cols,
            color_col,
            time_col,
            vector_cols,
            auto_color=auto_color,
            auto_time=auto_time,
        )
        point_preview = _build_point_cloud_preview(df, detection, columns, numeric_columns, max_points)
        coord_bounds = _coordinate_bounds(df, detection.coord_cols)
        try:
            visualization_info = render_static_visualization(
                taskid=taskid,
                dataset_id=dataset_id,
                image_root=IMAGE_ROOT,
                points=point_preview["preview"]["points"],
                color_values=point_preview["preview"]["color_values"],
                mapping=point_preview["mapping"],
                mode=visualization_mode,
                projection_plane=projection_plane,
                delaunay_alpha=delaunay_alpha,
                surface_method=surface_method,
                surface_alpha=surface_alpha,
                nbr_sz=nbr_sz,
                sample_spacing=sample_spacing,
                show_edges=show_edges,
                show_grid=show_grid,
                mesh_opacity=mesh_opacity,
                mesh_color=mesh_color,
                mesh_edge_color=mesh_edge_color,
                mesh_edge_width=mesh_edge_width,
                colormap=colormap,
                camera_zoom=camera_zoom,
                canvas_size=canvas_size,
                fill_box_boundary=fill_box_boundary,
                box_boundary_resolution=box_boundary_resolution,
                box_bounds=coord_bounds,
                plot_title=plot_title,
                axis_titles=axis_titles,
            )
            animation_info = render_time_animation(
                taskid=taskid,
                dataset_id=dataset_id,
                image_root=IMAGE_ROOT,
                df=df,
                coord_cols=detection.coord_cols,
                color_col=detection.color_col,
                time_col=detection.time_col,
                mode=visualization_mode,
                max_points=max_points,
                max_frames=max_frames,
                projection_plane=projection_plane,
                delaunay_alpha=delaunay_alpha,
                surface_method=surface_method,
                surface_alpha=surface_alpha,
                nbr_sz=nbr_sz,
                sample_spacing=sample_spacing,
                show_edges=show_edges,
                show_grid=show_grid,
                mesh_opacity=mesh_opacity,
                mesh_color=mesh_color,
                mesh_edge_color=mesh_edge_color,
                mesh_edge_width=mesh_edge_width,
                colormap=colormap,
                camera_zoom=camera_zoom,
                canvas_size=canvas_size,
                fill_box_boundary=fill_box_boundary,
                box_boundary_resolution=box_boundary_resolution,
                box_bounds=coord_bounds,
                plot_title=plot_title,
                axis_titles=axis_titles,
            )
        except ValueError as exc:
            _bad_table_request(str(exc), columns)
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        mesh_analysis_info: Dict[str, Any] = {}
        if include_mesh_analysis:
            mesh_analysis_info = _upload_mesh_analysis(
                point_preview=point_preview,
                visualization_info=visualization_info,
                visualization_mode=visualization_mode,
                projection_plane=projection_plane,
                delaunay_alpha=delaunay_alpha,
                surface_method=surface_method,
                surface_alpha=surface_alpha,
                nbr_sz=nbr_sz,
                sample_spacing=sample_spacing,
                device=mesh_analysis_device,
                include_centroids=max(0, min(mesh_analysis_centroids, 100)),
            )
        return JSONResponse({
            "success": True,
            "taskid": taskid,
            "dataset_id": dataset_id,
            "filename": filename,
            "size": len(content),
            "path": str(save_path),
            "data_type": "point_cloud",
            **point_preview,
            "visualization": {
                **visualization_info,
                **animation_info,
                **mesh_analysis_info,
            },
            **mesh_analysis_info,
            **visualization_info,
            **animation_info,
        })

    result = _build_preview(save_path, filename)
    return JSONResponse({
        "success": True,
        "taskid": taskid,
        "dataset_id": dataset_id,
        "filename": filename,
        "size": len(content),
        "path": str(save_path),
        **result,
    })


@router.post("/text-to-3d")
async def text_to_3d(req: TextTo3DRequest) -> JSONResponse:
    output_type = req.output_type.strip().lower()
    if output_type not in SUPPORTED_GENERATED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"output_type must be one of {sorted(SUPPORTED_GENERATED_TYPES)}",
        )

    try:
        raw_payload, raw_response = await call_text_to_3d_llm(
            prompt=req.prompt,
            output_type=output_type,
            max_points=req.max_points,
            max_faces=req.max_faces,
            temperature=req.temperature,
            timeout=req.timeout,
        )
        geometry = normalize_text_to_3d_payload(
            raw_payload,
            requested_output_type=output_type,
            max_points=req.max_points,
            max_faces=req.max_faces,
        )
        dataset, color_col = _generated_geometry_to_pyvista(geometry)
    except TextTo3DGenerationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    dataset_id = f"gen_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    saved_info = _save_generated_geometry(req.taskid, dataset_id, geometry, dataset)
    plot_title = (
        req.plot_title.strip()
        if req.plot_title and req.plot_title.strip()
        else geometry.get("description") or geometry["type"]
    )
    try:
        visualization_info = render_pyvista_dataset(
            taskid=req.taskid,
            dataset_id=dataset_id,
            image_root=IMAGE_ROOT,
            dataset=dataset,
            mode=geometry["type"],
            title=plot_title,
            color_col=color_col,
            show_edges=req.show_edges,
            show_grid=req.show_grid,
            mesh_opacity=req.mesh_opacity,
            mesh_color=req.mesh_color,
            mesh_edge_color=req.mesh_edge_color,
            mesh_edge_width=req.mesh_edge_width,
            colormap=req.colormap,
            camera_zoom=req.camera_zoom,
            canvas_size=req.canvas_size,
            axis_titles=(req.x_axis_title, req.y_axis_title, req.z_axis_title),
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    mesh_analysis_info = _analyze_generated_mesh(dataset, req)
    response: Dict[str, Any] = {
        "success": True,
        "taskid": req.taskid,
        "dataset_id": dataset_id,
        "generated_type": geometry["type"],
        "description": geometry.get("description"),
        "unit": geometry.get("unit"),
        "num_points": len(geometry["points"]),
        "num_faces": len(geometry.get("faces", [])),
        "scalar_fields": sorted((geometry.get("scalars") or {}).keys()),
        **saved_info,
        "visualization": {
            **visualization_info,
            **mesh_analysis_info,
        },
        **visualization_info,
        **mesh_analysis_info,
    }
    if req.include_raw_response:
        response["llm_payload"] = raw_payload
        response["llm_raw_response"] = raw_response
    return JSONResponse(response)


@router.post("/geometry/sample")
def geometry_sample(req: GeometrySampleRequest) -> Dict[str, Any]:
    return _sample_geometry(req)


@router.post("/mesh/analyze")
def mesh_analyze(req: MeshAnalyzeRequest) -> Dict[str, Any]:
    try:
        return _analyze_mesh_request(req)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/mesh/field-analysis")
def mesh_field_analysis(req: MeshFieldAnalysisRequest) -> Dict[str, Any]:
    try:
        return _field_analysis_request(req)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/files")
def data_processing_files(taskid: str = "default") -> Dict[str, Any]:
    files = [p.name for p in sorted(_dataset_dir(taskid).iterdir()) if p.is_file()]
    return {"taskid": taskid, "files": files}


@router.get("/demo", response_class=HTMLResponse)
def data_processing_demo() -> str:
    return HTML_DEMO


def create_app() -> FastAPI:
    app = FastAPI(title="Science42 Data Processing")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    IMAGE_ROOT.mkdir(parents=True, exist_ok=True)
    app.mount("/images", StaticFiles(directory=str(IMAGE_ROOT)), name="images")
    app.include_router(router)
    return app


app = create_app()


HTML_DEMO = """
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Science42 数据处理预览</title>
  <style>
    body { margin: 0; font-family: Arial, "Microsoft YaHei", sans-serif; background: #eef4fb; color: #172033; }
    .shell { max-width: 1180px; margin: 32px auto; padding: 0 20px; }
    .panel { background: rgba(255,255,255,.92); border: 1px solid #d8e4f2; border-radius: 8px; box-shadow: 0 16px 48px rgba(34,74,124,.12); overflow: hidden; }
    .head { padding: 18px 22px; border-bottom: 1px solid #d8e4f2; display:flex; justify-content:space-between; gap:16px; align-items:center; }
    h1 { font-size: 20px; margin: 0; }
    .upload { padding: 18px 22px; display:flex; gap:12px; align-items:center; flex-wrap:wrap; }
    input[type=file] { border: 1px dashed #7fa5cf; padding: 12px; border-radius: 6px; background: #f8fbff; }
    button { border: 0; background: #1f63b7; color: white; padding: 11px 16px; border-radius: 6px; cursor: pointer; }
    button:disabled { opacity:.45; cursor:not-allowed; }
    .meta { padding: 0 22px 18px; color: #526070; font-size: 13px; }
    .grid { display:grid; grid-template-columns: 1.1fr .9fr; border-top:1px solid #d8e4f2; }
    .view, .json { padding: 18px 22px; min-height: 420px; }
    .view { border-right: 1px solid #d8e4f2; background:#fff; }
    pre { white-space: pre-wrap; word-break: break-word; background:#0e1726; color:#dbeafe; padding:14px; border-radius:6px; overflow:auto; max-height:520px; }
    table { border-collapse: collapse; width: 100%; font-size: 13px; }
    th, td { border-bottom: 1px solid #e4edf7; padding: 8px; text-align: left; max-width: 220px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    th { background: #f5f9fe; position: sticky; top: 0; }
    canvas { width: 100%; height: 420px; background: linear-gradient(180deg, #f8fbff, #eaf2fb); border: 1px solid #d8e4f2; border-radius: 6px; }
    .hint { color:#607086; margin-top:12px; font-size:13px; }
    @media (max-width: 900px) { .grid { grid-template-columns: 1fr; } .view { border-right:0; border-bottom:1px solid #d8e4f2; } }
  </style>
</head>
<body>
  <div class="shell">
    <div class="panel">
      <div class="head"><h1>Science42 数据处理预览</h1><span id="status">等待上传</span></div>
      <div class="upload">
        <input id="file" type="file" accept=".csv,.txt,.dat,.json,.jsonl,.stl" />
        <button id="submit">上传并预览</button>
      </div>
      <div class="meta" id="meta"></div>
      <div class="grid"><div class="view" id="view"><div class="hint">CSV/TXT 会展示表格；包含 x,y,z 的数据会展示点云；STL 会展示几何投影。</div></div><div class="json"><pre id="json">{}</pre></div></div>
    </div>
  </div>
  <script>
    const fileInput = document.getElementById('file'), submit = document.getElementById('submit'), statusEl = document.getElementById('status'), metaEl = document.getElementById('meta'), viewEl = document.getElementById('view'), jsonEl = document.getElementById('json');
    submit.onclick = async () => {
      const file = fileInput.files[0]; if (!file) return; submit.disabled = true; statusEl.textContent = '处理中...';
      const fd = new FormData(); fd.append('file', file); fd.append('taskid', 'demo');
      try {
        const res = await fetch('/data-processing/upload-preview', { method: 'POST', body: fd });
        const data = await res.json(); if (!res.ok) throw new Error(data.detail || data.error || '请求失败');
        jsonEl.textContent = JSON.stringify(data, null, 2);
        metaEl.textContent = `${data.filename} | ${data.data_type} | ${data.size} bytes | ${data.dataset_id}`;
        render(data); statusEl.textContent = '预览完成';
      } catch (err) { statusEl.textContent = '处理失败'; viewEl.innerHTML = `<div class="hint">${escapeHtml(err.message)}</div>`; }
      finally { submit.disabled = false; }
    };
    function escapeHtml(s) { return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
    function render(data) {
      const preview = data.preview || {};
      if (preview.points && preview.points.length) return renderCanvas(preview.points, []);
      if (preview.triangles && preview.triangles.length) return renderCanvas([], preview.triangles);
      if (preview.rows && preview.rows.length) return renderTable(preview.rows);
      viewEl.innerHTML = `<div class="hint">${escapeHtml(preview.message || '暂无可视化预览')}</div>`;
    }
    function renderTable(rows) {
      const columns = Object.keys(rows[0] || {});
      viewEl.innerHTML = `<div style="overflow:auto; max-height:520px"><table><thead><tr>${columns.map(c=>`<th>${escapeHtml(c)}</th>`).join('')}</tr></thead><tbody>${rows.map(r=>`<tr>${columns.map(c=>`<td>${escapeHtml(r[c] ?? '')}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;
    }
    function renderCanvas(points, triangles) {
      viewEl.innerHTML = '<canvas id="canvas" width="900" height="520"></canvas>';
      const canvas = document.getElementById('canvas'), ctx = canvas.getContext('2d');
      const all = points.length ? points : triangles.flat();
      const xs = all.map(p => p.x ?? p[0]), ys = all.map(p => p.y ?? p[1]), zs = all.map(p => p.z ?? p[2]);
      const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys), minZ = Math.min(...zs), maxZ = Math.max(...zs);
      const scale = Math.min(canvas.width / Math.max(maxX - minX, 1), canvas.height / Math.max(maxY - minY, 1)) * 0.75;
      const project = p => {
        const x = p.x ?? p[0], y = p.y ?? p[1], z = p.z ?? p[2];
        return [canvas.width / 2 + (x - (minX + maxX)/2) * scale + (z - (minZ + maxZ)/2) * scale * 0.22, canvas.height / 2 - (y - (minY + maxY)/2) * scale + (z - (minZ + maxZ)/2) * scale * 0.12];
      };
      ctx.clearRect(0,0,canvas.width,canvas.height);
      if (triangles.length) {
        ctx.strokeStyle = '#1f63b7'; ctx.globalAlpha = 0.22;
        triangles.forEach(t => { const a = project(t[0]), b = project(t[1]), c = project(t[2]); ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.lineTo(c[0], c[1]); ctx.closePath(); ctx.stroke(); });
      } else {
        ctx.fillStyle = '#1f63b7'; points.forEach(p => { const q = project(p); ctx.fillRect(q[0], q[1], 2, 2); });
      }
    }
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.data_processing_module:app", host="0.0.0.0", port=1212, reload=False)
