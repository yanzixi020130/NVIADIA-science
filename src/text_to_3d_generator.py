"""Natural-language to 3D point-cloud or triangle-mesh generation helpers."""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from dotenv import load_dotenv
    load_dotenv(override=False)
except Exception:
    pass


DEFAULT_TEXT_TO_3D_BASE_URL = "http://www.science42.vip:40200/v1"
DEFAULT_TEXT_TO_3D_MODEL = "SE_V0.0"
SUPPORTED_GENERATED_TYPES = {"auto", "point_cloud", "triangle_mesh", "parametric_surface"}


class TextTo3DGenerationError(ValueError):
    """Raised when the LLM response cannot be converted into valid 3D data."""


@dataclass(frozen=True)
class TextTo3DConfig:
    base_url: str
    api_key: str
    model: str


def load_text_to_3d_config() -> TextTo3DConfig:
    api_key = (
        os.getenv("TEXT_TO_3D_LLM_API_KEY")
        or os.getenv("XIMU_LLM_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )
    if not api_key:
        raise TextTo3DGenerationError(
            "Missing TEXT_TO_3D_LLM_API_KEY, XIMU_LLM_API_KEY, or OPENAI_API_KEY"
        )
    return TextTo3DConfig(
        base_url=os.getenv("TEXT_TO_3D_LLM_BASE_URL", DEFAULT_TEXT_TO_3D_BASE_URL),
        api_key=api_key,
        model=os.getenv("TEXT_TO_3D_LLM_MODEL", DEFAULT_TEXT_TO_3D_MODEL),
    )


def _system_prompt(output_type: str, max_points: int, max_faces: int) -> str:
    return f"""
You generate compact, valid 3D geometry for scientific visualization.
Return exactly one JSON object and no markdown.

Allowed output types:
1. point_cloud:
   {{
     "type": "point_cloud",
     "points": [[x, y, z], ...],
     "scalars": {{"temperature": [number, ...]}},
     "description": "short description"
   }}

2. triangle_mesh:
   {{
     "type": "triangle_mesh",
     "points": [[x, y, z], ...],
     "faces": [[i, j, k], ...],
     "scalars": {{"temperature": [number, ...]}},
     "description": "short description"
   }}

3. parametric_surface, preferred for complex objects:
   {{
     "type": "parametric_surface",
     "shape": "wing|sphere|cylinder|wavy_surface|plane",
     "resolution": [u_count, v_count],
     "parameters": {{}},
     "fields": {{"temperature": "linear_x|linear_z|radial|sinusoidal"}}
   }}

Constraints:
- requested output_type: {output_type}
- points <= {max_points}
- triangle faces <= {max_faces}
- all coordinates and scalar values must be finite numbers
- faces must be zero-based integer indices into points
- scalars must have exactly one value per point
- use meters unless the user asks otherwise
""".strip()


async def call_text_to_3d_llm(
    prompt: str,
    output_type: str,
    max_points: int,
    max_faces: int,
    temperature: float = 0.2,
    timeout: float = 60.0,
    config: Optional[TextTo3DConfig] = None,
) -> Tuple[Dict[str, Any], str]:
    """Call an OpenAI-compatible chat model and parse its JSON response."""
    try:
        from openai import AsyncOpenAI
    except Exception as exc:
        raise TextTo3DGenerationError(f"openai package is required: {exc}") from exc

    cfg = config or load_text_to_3d_config()
    client = AsyncOpenAI(api_key=cfg.api_key, base_url=cfg.base_url)
    response = await client.chat.completions.create(
        model=cfg.model,
        messages=[
            {"role": "system", "content": _system_prompt(output_type, max_points, max_faces)},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        timeout=timeout,
    )
    text = response.choices[0].message.content if response.choices else ""
    if not text:
        raise TextTo3DGenerationError("LLM returned an empty response")
    return _parse_json_object(text), text


def _parse_json_object(text: str) -> Dict[str, Any]:
    stripped = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, flags=re.S | re.I)
    if fence:
        stripped = fence.group(1).strip()
    if not stripped.startswith("{"):
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start >= 0 and end > start:
            stripped = stripped[start:end + 1]
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise TextTo3DGenerationError(f"LLM response is not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise TextTo3DGenerationError("LLM response must be a JSON object")
    return parsed


def normalize_text_to_3d_payload(
    payload: Dict[str, Any],
    requested_output_type: str = "auto",
    max_points: int = 5000,
    max_faces: int = 10000,
) -> Dict[str, Any]:
    """Validate or expand a generated geometry payload into points/faces arrays."""
    if requested_output_type not in SUPPORTED_GENERATED_TYPES:
        raise TextTo3DGenerationError(
            f"output_type must be one of {sorted(SUPPORTED_GENERATED_TYPES)}"
        )

    payload_type = str(payload.get("type") or requested_output_type or "auto").strip().lower()
    if payload_type == "auto":
        payload_type = "parametric_surface" if payload.get("shape") else "triangle_mesh"
    if payload_type == "parametric_surface":
        payload = _generate_parametric_surface(payload, requested_output_type, max_points, max_faces)
        payload_type = payload["type"]

    if payload_type not in {"point_cloud", "triangle_mesh"}:
        raise TextTo3DGenerationError(
            "Generated type must be point_cloud, triangle_mesh, or parametric_surface"
        )
    if requested_output_type == "point_cloud" and payload_type != "point_cloud":
        payload_type = "point_cloud"
    if requested_output_type == "triangle_mesh" and payload_type != "triangle_mesh":
        raise TextTo3DGenerationError("Requested triangle_mesh but LLM did not provide faces")

    points = _validate_points(payload.get("points"), max_points)
    faces = None
    if payload_type == "triangle_mesh":
        faces = _validate_faces(payload.get("faces"), len(points), max_faces)

    scalars = _validate_scalar_fields(payload.get("scalars") or payload.get("fields"), len(points))
    if not scalars:
        scalars = {"height": points[:, 2].astype(float).tolist()}

    result: Dict[str, Any] = {
        "type": payload_type,
        "points": points.astype(float).tolist(),
        "scalars": scalars,
        "description": str(payload.get("description") or payload.get("shape") or "").strip(),
        "unit": str(payload.get("unit") or "m"),
    }
    if faces is not None:
        result["faces"] = faces.astype(int).tolist()
    return result


def _validate_points(points: Any, max_points: int) -> np.ndarray:
    array = np.asarray(points, dtype=float)
    if array.ndim != 2 or array.shape[1] != 3:
        raise TextTo3DGenerationError("points must be an N x 3 numeric array")
    if len(array) < 1:
        raise TextTo3DGenerationError("points must contain at least one point")
    if len(array) > max_points:
        raise TextTo3DGenerationError(f"points exceed max_points={max_points}")
    if not np.isfinite(array).all():
        raise TextTo3DGenerationError("points contain NaN or infinite values")
    return array


def _validate_faces(faces: Any, point_count: int, max_faces: int) -> np.ndarray:
    array = np.asarray(faces, dtype=int)
    if array.ndim != 2 or array.shape[1] != 3:
        raise TextTo3DGenerationError("faces must be an M x 3 integer array")
    if len(array) < 1:
        raise TextTo3DGenerationError("triangle_mesh requires at least one face")
    if len(array) > max_faces:
        raise TextTo3DGenerationError(f"faces exceed max_faces={max_faces}")
    if array.min() < 0 or array.max() >= point_count:
        raise TextTo3DGenerationError("faces contain point indices outside the points array")
    return array


def _validate_scalar_fields(fields: Any, point_count: int) -> Dict[str, List[float]]:
    if not isinstance(fields, dict):
        return {}
    result: Dict[str, List[float]] = {}
    for key, values in fields.items():
        if isinstance(values, str):
            continue
        array = np.asarray(values, dtype=float)
        if array.ndim != 1 or len(array) != point_count or not np.isfinite(array).all():
            continue
        result[str(key)] = array.astype(float).tolist()
    return result


def _generate_parametric_surface(
    payload: Dict[str, Any],
    requested_output_type: str,
    max_points: int,
    max_faces: int,
) -> Dict[str, Any]:
    shape = str(payload.get("shape") or "wavy_surface").strip().lower()
    params = payload.get("parameters") if isinstance(payload.get("parameters"), dict) else {}
    resolution = _resolution(payload.get("resolution"), max_points, max_faces, shape)

    if shape in {"sphere", "ball"}:
        points, faces = _sphere(params, resolution)
    elif shape in {"cylinder", "tube", "pipe"}:
        points, faces = _cylinder(params, resolution)
    elif shape in {"wing", "airfoil", "airplane_wing"}:
        points, faces = _wing(params, resolution)
    elif shape in {"plane", "surface", "wavy_surface", "wave"}:
        points, faces = _wavy_surface(params, resolution)
    else:
        points, faces = _wavy_surface(params, resolution)
        shape = "wavy_surface"

    if len(points) > max_points:
        raise TextTo3DGenerationError(f"generated surface exceeds max_points={max_points}")
    if len(faces) > max_faces:
        raise TextTo3DGenerationError(f"generated surface exceeds max_faces={max_faces}")

    return {
        "type": "point_cloud" if requested_output_type == "point_cloud" else "triangle_mesh",
        "points": points.tolist(),
        "faces": faces.tolist(),
        "scalars": _parametric_fields(points, payload.get("fields"), params),
        "shape": shape,
        "description": str(payload.get("description") or f"parametric {shape}"),
        "unit": str(payload.get("unit") or "m"),
    }


def _resolution(value: Any, max_points: int, max_faces: int, shape: str) -> Tuple[int, int]:
    defaults = {
        "sphere": (48, 24),
        "cylinder": (48, 24),
        "wing": (72, 28),
        "airfoil": (72, 28),
        "plane": (64, 40),
        "wavy_surface": (64, 40),
    }
    u, v = defaults.get(shape, (64, 40))
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        try:
            u, v = int(value[0]), int(value[1])
        except Exception:
            pass
    u = max(3, min(u, 200))
    v = max(3, min(v, 200))
    while u * v > max_points or 2 * (u - 1) * (v - 1) > max_faces:
        if u >= v and u > 3:
            u -= 1
        elif v > 3:
            v -= 1
        else:
            break
    return u, v


def _grid_faces(u_count: int, v_count: int, wrap_u: bool = False) -> np.ndarray:
    faces: List[List[int]] = []
    u_limit = u_count if wrap_u else u_count - 1
    for i in range(u_limit):
        ni = (i + 1) % u_count
        for j in range(v_count - 1):
            a = i * v_count + j
            b = ni * v_count + j
            c = ni * v_count + j + 1
            d = i * v_count + j + 1
            faces.append([a, b, c])
            faces.append([a, c, d])
    return np.asarray(faces, dtype=int)


def _float_param(params: Dict[str, Any], key: str, default: float) -> float:
    try:
        return float(params.get(key, default))
    except Exception:
        return default


def _wavy_surface(params: Dict[str, Any], resolution: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    u_count, v_count = resolution
    width = _float_param(params, "width", 4.0)
    depth = _float_param(params, "depth", 3.0)
    amplitude = _float_param(params, "amplitude", 0.25)
    frequency = _float_param(params, "frequency", 2.0)
    xs = np.linspace(-width / 2, width / 2, u_count)
    ys = np.linspace(-depth / 2, depth / 2, v_count)
    points = []
    for x in xs:
        for y in ys:
            z = amplitude * math.sin(frequency * x) * math.cos(frequency * y)
            points.append([x, y, z])
    return np.asarray(points, dtype=float), _grid_faces(u_count, v_count)


def _wing(params: Dict[str, Any], resolution: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    u_count, v_count = resolution
    span = _float_param(params, "span", 5.0)
    root_chord = _float_param(params, "root_chord", _float_param(params, "chord", 1.4))
    tip_chord = _float_param(params, "tip_chord", root_chord * 0.45)
    camber = _float_param(params, "camber", 0.08)
    twist_deg = _float_param(params, "twist", 5.0)
    ys = np.linspace(-span / 2, span / 2, u_count)
    chord_pos = np.linspace(0.0, 1.0, v_count)
    points = []
    for y in ys:
        frac = abs(y) / (span / 2)
        chord = root_chord + (tip_chord - root_chord) * frac
        twist = math.radians(twist_deg * frac)
        for xp in chord_pos:
            x_local = (xp - 0.35) * chord
            z_local = camber * chord * 4.0 * xp * (1.0 - xp)
            x = x_local * math.cos(twist) + z_local * math.sin(twist)
            z = -x_local * math.sin(twist) + z_local * math.cos(twist)
            points.append([x, y, z])
    return np.asarray(points, dtype=float), _grid_faces(u_count, v_count)


def _sphere(params: Dict[str, Any], resolution: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    u_count, v_count = resolution
    radius = _float_param(params, "radius", 1.0)
    phis = np.linspace(0.0, 2.0 * math.pi, u_count, endpoint=False)
    thetas = np.linspace(0.001, math.pi - 0.001, v_count)
    points = []
    for phi in phis:
        for theta in thetas:
            points.append([
                radius * math.sin(theta) * math.cos(phi),
                radius * math.sin(theta) * math.sin(phi),
                radius * math.cos(theta),
            ])
    return np.asarray(points, dtype=float), _grid_faces(u_count, v_count, wrap_u=True)


def _cylinder(params: Dict[str, Any], resolution: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    u_count, v_count = resolution
    radius = _float_param(params, "radius", 1.0)
    height = _float_param(params, "height", 3.0)
    angles = np.linspace(0.0, 2.0 * math.pi, u_count, endpoint=False)
    zs = np.linspace(-height / 2, height / 2, v_count)
    points = []
    for angle in angles:
        for z in zs:
            points.append([radius * math.cos(angle), radius * math.sin(angle), z])
    return np.asarray(points, dtype=float), _grid_faces(u_count, v_count, wrap_u=True)


def _parametric_fields(points: np.ndarray, fields: Any, params: Dict[str, Any]) -> Dict[str, List[float]]:
    if not isinstance(fields, dict) or not fields:
        fields = {"height": "linear_z"}
    result: Dict[str, List[float]] = {}
    for key, spec in fields.items():
        values = _field_values(points, str(spec), params)
        result[str(key)] = values.astype(float).tolist()
    return result


def _field_values(points: np.ndarray, spec: str, params: Dict[str, Any]) -> np.ndarray:
    lower = spec.lower().strip()
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    if lower in {"linear_x", "temperature_gradient_x"}:
        return _normalize(x, 280.0, 360.0)
    if lower in {"linear_y"}:
        return _normalize(y, 280.0, 360.0)
    if lower in {"linear_z", "height"}:
        return _normalize(z, 0.0, 1.0)
    if lower in {"radial", "radius"}:
        return np.linalg.norm(points, axis=1)
    if lower in {"sinusoidal", "wave"}:
        return 0.5 + 0.5 * np.sin(2.0 * x) * np.cos(2.0 * y)
    try:
        return np.full(len(points), float(spec), dtype=float)
    except Exception:
        return _normalize(z, 0.0, 1.0)


def _normalize(values: np.ndarray, low: float, high: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    span = float(values.max() - values.min()) if len(values) else 0.0
    if span <= 1e-12:
        return np.full(len(values), (low + high) / 2.0)
    return low + (values - values.min()) / span * (high - low)
