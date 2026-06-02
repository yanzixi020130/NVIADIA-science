from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


VISUALIZATION_MODES = {"point_cloud", "delaunay_2d", "surface_reconstruction"}
DEFAULT_WINDOW_SIZE = (1000, 1000)


def _safe_taskid(taskid: str) -> str:
    return "".join(ch for ch in taskid if ch.isalnum() or ch in "-_")[:80] or "default"


def _output_dir(image_root: Path, taskid: str) -> Path:
    path = image_root / "data_processing" / _safe_taskid(taskid)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _url_for(path: Path, image_root: Path) -> str:
    return f"/images/{path.relative_to(image_root).as_posix()}"


def _clean_points(points: Sequence[Sequence[Optional[float]]]) -> np.ndarray:
    values = np.asarray(points, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("Point cloud data must be an N x 3 array")
    values = values[np.isfinite(values).all(axis=1)]
    if len(values) < 1:
        raise ValueError("Point cloud contains no valid finite points")
    return values


def _clean_scalars(color_values: Sequence[Optional[float]], point_count: int) -> Optional[np.ndarray]:
    if len(color_values) != point_count:
        return None
    scalars = np.asarray(color_values, dtype=float)
    if not np.isfinite(scalars).any():
        return None
    return scalars


def _import_pyvista() -> Any:
    try:
        import pyvista as pv
    except Exception as exc:
        raise RuntimeError(f"PyVista is required for point cloud visualization: {exc}") from exc
    pv.OFF_SCREEN = True
    return pv


def _project_points(points: np.ndarray, projection_plane: str) -> np.ndarray:
    plane = projection_plane.lower()
    projected = np.zeros_like(points)
    if plane == "xy":
        projected[:, 0], projected[:, 1] = points[:, 0], points[:, 1]
    elif plane == "xz":
        projected[:, 0], projected[:, 1] = points[:, 0], points[:, 2]
    elif plane == "yz":
        projected[:, 0], projected[:, 1] = points[:, 1], points[:, 2]
    elif plane == "pca":
        centered = points - points.mean(axis=0)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        projected_2d = centered @ vt[:2].T
        projected[:, :2] = projected_2d
    else:
        raise ValueError("projection_plane must be one of xy, xz, yz, or pca")
    return projected


def _polydata(points: np.ndarray, scalars: Optional[np.ndarray]) -> Any:
    pv = _import_pyvista()
    cloud = pv.PolyData(points)
    if scalars is not None and len(scalars) == len(points):
        cloud.point_data["color_values"] = scalars
    return cloud


def _box_bounds(points: np.ndarray, explicit_bounds: Optional[Sequence[Sequence[float]]] = None) -> Tuple[np.ndarray, np.ndarray]:
    if explicit_bounds is not None:
        bounds = np.asarray(explicit_bounds, dtype=float)
        if bounds.shape == (2, 3) and np.isfinite(bounds).all():
            return bounds[0], bounds[1]
    return points.min(axis=0), points.max(axis=0)


def _box_boundary_points(
    points: np.ndarray,
    resolution: int,
    explicit_bounds: Optional[Sequence[Sequence[float]]] = None,
) -> np.ndarray:
    resolution = int(max(2, min(resolution, 200)))
    mins, maxs = _box_bounds(points, explicit_bounds)
    if not np.isfinite(mins).all() or not np.isfinite(maxs).all():
        return np.empty((0, 3), dtype=float)
    if np.any(maxs <= mins):
        return np.empty((0, 3), dtype=float)

    axes = [np.linspace(mins[i], maxs[i], resolution) for i in range(3)]
    faces: List[np.ndarray] = []
    for fixed_axis in range(3):
        free_axes = [axis for axis in range(3) if axis != fixed_axis]
        grid_a, grid_b = np.meshgrid(axes[free_axes[0]], axes[free_axes[1]], indexing="ij")
        for fixed_value in (mins[fixed_axis], maxs[fixed_axis]):
            face = np.empty((grid_a.size, 3), dtype=float)
            face[:, fixed_axis] = fixed_value
            face[:, free_axes[0]] = grid_a.ravel()
            face[:, free_axes[1]] = grid_b.ravel()
            faces.append(face)
    return np.unique(np.vstack(faces), axis=0)


def _nearest_scalar_values(points: np.ndarray, scalars: np.ndarray, targets: np.ndarray) -> np.ndarray:
    finite = np.isfinite(scalars)
    if not finite.any() or len(targets) == 0:
        return np.full(len(targets), np.nan, dtype=float)

    source_points = points[finite]
    source_scalars = scalars[finite]
    values = np.empty(len(targets), dtype=float)
    chunk_size = 1024
    for start in range(0, len(targets), chunk_size):
        chunk = targets[start:start + chunk_size]
        distances = ((chunk[:, None, :] - source_points[None, :, :]) ** 2).sum(axis=2)
        values[start:start + len(chunk)] = source_scalars[np.argmin(distances, axis=1)]
    return values


def _augment_box_boundary(
    points: np.ndarray,
    scalars: Optional[np.ndarray],
    resolution: int,
    explicit_bounds: Optional[Sequence[Sequence[float]]] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    boundary = _box_boundary_points(points, resolution, explicit_bounds)
    if len(boundary) == 0:
        return points, scalars
    augmented_points = np.vstack([points, boundary])
    if scalars is None:
        return augmented_points, None
    augmented_scalars = np.concatenate([scalars, _nearest_scalar_values(points, scalars, boundary)])
    return augmented_points, augmented_scalars


def _build_mesh(
    points: np.ndarray,
    scalars: Optional[np.ndarray],
    mode: str,
    projection_plane: str,
    delaunay_alpha: Optional[float],
    surface_method: str,
    surface_alpha: Optional[float],
    nbr_sz: int,
    sample_spacing: Optional[float],
    fill_box_boundary: bool,
    box_boundary_resolution: int,
    box_bounds: Optional[Sequence[Sequence[float]]],
) -> Tuple[Any, Optional[Any], str]:
    pv = _import_pyvista()
    if mode == "point_cloud":
        return _polydata(points, scalars), None, "Point Cloud"

    if mode == "delaunay_2d":
        if len(points) < 3:
            raise ValueError("delaunay_2d requires at least 3 points")
        projected = _project_points(points, projection_plane)
        cloud = _polydata(projected, scalars)
        mesh = cloud.delaunay_2d(alpha=delaunay_alpha or 0.0)
        return mesh, mesh, f"Delaunay 2D ({projection_plane})"

    if mode == "surface_reconstruction":
        if len(points) < 4:
            raise ValueError("surface_reconstruction requires at least 4 points")
        if fill_box_boundary:
            points, scalars = _augment_box_boundary(points, scalars, box_boundary_resolution, box_bounds)
        cloud = _polydata(points, scalars)
        if surface_method == "delaunay_3d":
            volume = cloud.delaunay_3d(alpha=surface_alpha or 0.0)
            mesh = volume.extract_surface(algorithm="dataset_surface")
            if "vtkOriginalCellIds" in mesh.cell_data:
                del mesh.cell_data["vtkOriginalCellIds"]
            mesh = mesh.triangulate()
        elif surface_method == "reconstruct_surface":
            mesh = cloud.reconstruct_surface(nbr_sz=nbr_sz, sample_spacing=sample_spacing)
        else:
            raise ValueError("surface_method must be reconstruct_surface or delaunay_3d")
        if scalars is not None and "color_values" not in mesh.point_data:
            try:
                mesh = mesh.interpolate(cloud, radius=float(np.ptp(points, axis=0).max()) * 0.08)
            except Exception:
                pass
        return mesh, mesh, f"Surface Reconstruction ({surface_method})"

    raise ValueError(f"Unsupported visualization_mode '{mode}'")


def _add_to_plotter(
    plotter: Any,
    dataset: Any,
    mode: str,
    title: str,
    color_col: Optional[str],
    show_edges: bool,
    show_grid: bool,
    mesh_opacity: float,
    mesh_color: str,
    mesh_edge_color: str,
    mesh_edge_width: float,
) -> None:
    has_scalars = "color_values" in dataset.point_data or "color_values" in dataset.cell_data
    if mode == "point_cloud":
        plotter.add_mesh(
            dataset,
            scalars="color_values" if has_scalars else None,
            cmap="viridis",
            render_points_as_spheres=True,
            point_size=6,
            scalar_bar_args={"title": color_col or "value"} if has_scalars else None,
        )
    else:
        plotter.add_mesh(
            dataset,
            scalars="color_values" if has_scalars else None,
            color=None if has_scalars else mesh_color,
            cmap="viridis",
            show_edges=show_edges,
            edge_color=mesh_edge_color,
            line_width=mesh_edge_width,
            opacity=mesh_opacity,
            scalar_bar_args={"title": color_col or "value"} if has_scalars else None,
        )
    plotter.add_title(title, font_size=12)
    plotter.show_axes()
    if show_grid:
        try:
            plotter.show_grid(color="#6b7280", grid="back", location="outer", ticks="outside")
        except TypeError:
            plotter.show_grid()
    plotter.camera_position = "iso"
    plotter.reset_camera()


def render_static_visualization(
    taskid: str,
    dataset_id: str,
    image_root: Path,
    points: Sequence[Sequence[Optional[float]]],
    color_values: Sequence[Optional[float]],
    mapping: Dict[str, Any],
    mode: str = "point_cloud",
    projection_plane: str = "xy",
    delaunay_alpha: Optional[float] = None,
    surface_method: str = "reconstruct_surface",
    surface_alpha: Optional[float] = None,
    nbr_sz: int = 20,
    sample_spacing: Optional[float] = None,
    show_edges: bool = False,
    show_grid: bool = True,
    mesh_opacity: float = 1.0,
    mesh_color: str = "#9bbfc2",
    mesh_edge_color: str = "#111827",
    mesh_edge_width: float = 1.0,
    fill_box_boundary: bool = False,
    box_boundary_resolution: int = 25,
    box_bounds: Optional[Sequence[Sequence[float]]] = None,
) -> Dict[str, Any]:
    mode = mode.lower()
    if mode not in VISUALIZATION_MODES:
        raise ValueError(f"visualization_mode must be one of {sorted(VISUALIZATION_MODES)}")

    clean_points = _clean_points(points)
    scalars = _clean_scalars(color_values, len(clean_points))
    dataset, mesh, title = _build_mesh(
        clean_points,
        scalars,
        mode,
        projection_plane,
        delaunay_alpha,
        surface_method,
        surface_alpha,
        nbr_sz,
        sample_spacing,
        fill_box_boundary,
        box_boundary_resolution,
        box_bounds,
    )

    pv = _import_pyvista()
    output_dir = _output_dir(image_root, taskid)
    image_path = output_dir / f"{dataset_id}_{mode}.png"

    plotter = pv.Plotter(off_screen=True, window_size=DEFAULT_WINDOW_SIZE)
    plotter.set_background("white")
    _add_to_plotter(
        plotter,
        dataset,
        mode,
        title,
        mapping.get("color_col"),
        show_edges,
        show_grid,
        mesh_opacity,
        mesh_color,
        mesh_edge_color,
        mesh_edge_width,
    )
    plotter.screenshot(str(image_path))
    plotter.close()

    result: Dict[str, Any] = {
        "visualization_mode": mode,
        "image_path": str(image_path),
        "image_url": _url_for(image_path, image_root),
    }
    if mesh is not None:
        mesh_path = output_dir / f"{dataset_id}_{mode}.vtp"
        mesh.save(mesh_path)
        result["mesh_path"] = str(mesh_path)
        result["mesh_url"] = _url_for(mesh_path, image_root)
    return result


def render_pyvista_dataset(
    taskid: str,
    dataset_id: str,
    image_root: Path,
    dataset: Any,
    mode: str,
    title: str,
    color_col: Optional[str] = None,
    show_edges: bool = True,
    show_grid: bool = True,
    mesh_opacity: float = 1.0,
    mesh_color: str = "#9bbfc2",
    mesh_edge_color: str = "#111827",
    mesh_edge_width: float = 1.0,
) -> Dict[str, Any]:
    """Render an already-built PyVista point cloud or mesh to PNG and VTP."""
    pv = _import_pyvista()
    output_dir = _output_dir(image_root, taskid)
    image_path = output_dir / f"{dataset_id}_{mode}.png"
    dataset_path = output_dir / f"{dataset_id}_{mode}.vtp"

    plotter = pv.Plotter(off_screen=True, window_size=DEFAULT_WINDOW_SIZE)
    plotter.set_background("white")
    _add_to_plotter(
        plotter,
        dataset,
        "point_cloud" if mode == "point_cloud" else "triangle_mesh",
        title,
        color_col,
        show_edges,
        show_grid,
        mesh_opacity,
        mesh_color,
        mesh_edge_color,
        mesh_edge_width,
    )
    plotter.screenshot(str(image_path))
    plotter.close()
    dataset.save(dataset_path)

    return {
        "visualization_mode": mode,
        "image_path": str(image_path),
        "image_url": _url_for(image_path, image_root),
        "mesh_path": str(dataset_path),
        "mesh_url": _url_for(dataset_path, image_root),
    }


def _frame_values(
    df: Any,
    coord_cols: Sequence[str],
    color_col: Optional[str],
    time_col: str,
    max_points: int,
    max_frames: int,
) -> List[Tuple[float, np.ndarray, List[Optional[float]]]]:
    required_columns = list(coord_cols) + [time_col]
    if color_col:
        required_columns.append(color_col)
    pd = __import__("pandas")
    numeric_df = df[list(dict.fromkeys(required_columns))].apply(pd.to_numeric, errors="coerce")
    numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan).dropna(subset=list(coord_cols) + [time_col])
    if numeric_df.empty:
        return []

    time_values = sorted(numeric_df[time_col].dropna().unique().tolist())
    if len(time_values) > max_frames:
        indices = np.linspace(0, len(time_values) - 1, max_frames).round().astype(int)
        time_values = [time_values[int(index)] for index in sorted(set(indices.tolist()))]

    frames: List[Tuple[float, np.ndarray, List[Optional[float]]]] = []
    for time_value in time_values:
        frame_df = numeric_df[numeric_df[time_col] == time_value]
        if len(frame_df) > max_points:
            frame_df = frame_df.sample(n=max_points, random_state=42)
        points = frame_df[list(coord_cols)].to_numpy(dtype=float)
        colors = frame_df[color_col].tolist() if color_col else []
        if len(points):
            frames.append((float(time_value), points, colors))
    return frames


def render_time_animation(
    taskid: str,
    dataset_id: str,
    image_root: Path,
    df: Any,
    coord_cols: Sequence[str],
    color_col: Optional[str],
    time_col: Optional[str],
    mode: str = "point_cloud",
    max_points: int = 5000,
    max_frames: int = 60,
    projection_plane: str = "xy",
    delaunay_alpha: Optional[float] = None,
    surface_method: str = "reconstruct_surface",
    surface_alpha: Optional[float] = None,
    nbr_sz: int = 20,
    sample_spacing: Optional[float] = None,
    show_edges: bool = False,
    show_grid: bool = True,
    mesh_opacity: float = 1.0,
    mesh_color: str = "#9bbfc2",
    mesh_edge_color: str = "#111827",
    mesh_edge_width: float = 1.0,
    fill_box_boundary: bool = False,
    box_boundary_resolution: int = 25,
    box_bounds: Optional[Sequence[Sequence[float]]] = None,
) -> Dict[str, Any]:
    if not time_col:
        return {}
    if max_frames < 1:
        raise ValueError("max_frames must be greater than 0")

    mode = mode.lower()
    frames = _frame_values(df, coord_cols, color_col, time_col, max_points, max_frames)
    if not frames:
        return {}

    pv = _import_pyvista()
    output_dir = _output_dir(image_root, taskid)
    animation_path = output_dir / f"{dataset_id}_{mode}.gif"
    plotter = pv.Plotter(off_screen=True, window_size=DEFAULT_WINDOW_SIZE)
    plotter.set_background("white")
    plotter.open_gif(str(animation_path), fps=3)

    for time_value, points, colors in frames:
        plotter.clear()
        scalars = _clean_scalars(colors, len(points))
        dataset, _, title = _build_mesh(
            points,
            scalars,
            mode,
            projection_plane,
            delaunay_alpha,
            surface_method,
            surface_alpha,
            nbr_sz,
            sample_spacing,
            fill_box_boundary,
            box_boundary_resolution,
            box_bounds,
        )
        _add_to_plotter(
            plotter,
            dataset,
            mode,
            f"{title} | {time_col}={time_value:g}",
            color_col,
            show_edges,
            show_grid,
            mesh_opacity,
            mesh_color,
            mesh_edge_color,
            mesh_edge_width,
        )
        plotter.write_frame()

    plotter.close()
    return {
        "animation_path": str(animation_path),
        "animation_url": _url_for(animation_path, image_root),
        "animation_type": "gif",
        "time_col": time_col,
        "num_animation_frames": len(frames),
    }
