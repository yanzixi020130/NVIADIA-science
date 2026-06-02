import json
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence


COORDINATE_ALIASES: List[List[str]] = [
    ["x", "y", "z"],
    ["coord_x", "coord_y", "coord_z"],
    ["pos_x", "pos_y", "pos_z"],
    ["point_x", "point_y", "point_z"],
    ["px", "py", "pz"],
]

SCALAR_ALIASES: List[str] = [
    "value",
    "val",
    "scalar",
    "field",
    "temperature",
    "temp",
    "t",
    "pressure",
    "press",
    "p",
    "density",
    "rho",
    "speed",
    "velocity_magnitude",
    "vel_mag",
]

TIME_ALIASES: List[str] = [
    "time",
    "times",
    "time_s",
    "time[s]",
    "t[s]",
    "时间",
    "时间[s]",
    "时间t[s]",
    "step",
    "frame",
]

VECTOR_ALIASES: List[List[str]] = [
    ["u", "v", "w"],
    ["vx", "vy", "vz"],
    ["velocity_x", "velocity_y", "velocity_z"],
    ["vel_x", "vel_y", "vel_z"],
]


@dataclass(frozen=True)
class FieldDetection:
    coord_cols: List[str]
    coord_reason: str
    coord_confidence: float
    color_col: Optional[str]
    color_reason: str
    color_confidence: float
    time_col: Optional[str]
    time_reason: str
    time_confidence: float
    vector_cols: List[str]
    vector_reason: str
    vector_confidence: float
    available_scalar_fields: List[str]
    available_time_fields: List[str]
    available_vector_fields: List[List[str]]
    need_user_confirm: bool

    def detected_payload(self) -> Dict[str, object]:
        return {
            "coord_cols": self.coord_cols,
            "coord_reason": self.coord_reason,
            "coord_confidence": self.coord_confidence,
            "color_col": self.color_col,
            "color_reason": self.color_reason,
            "color_confidence": self.color_confidence,
            "time_col": self.time_col,
            "time_reason": self.time_reason,
            "time_confidence": self.time_confidence,
            "vector_cols": self.vector_cols,
            "vector_reason": self.vector_reason,
            "vector_confidence": self.vector_confidence,
        }


def _canonical_name(name: str) -> str:
    """Normalize separators and case while preserving enough structure for aliases."""
    return re.sub(r"[\s\-]+", "_", str(name).strip().lower())


def _column_lookup(columns: Sequence[str]) -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    for column in columns:
        key = _canonical_name(column)
        if key not in lookup:
            lookup[key] = column
    return lookup


def parse_column_list(raw_value: Optional[str], field_name: str, expected_count: int = 3) -> Optional[List[str]]:
    if raw_value is None or raw_value.strip() == "":
        return None

    text = raw_value.strip()
    if text.startswith("["):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{field_name} must be a JSON array or comma-separated string") from exc
        if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
            raise ValueError(f"{field_name} must contain only column name strings")
        values = [item.strip() for item in parsed if item.strip()]
    else:
        values = [item.strip().strip("\"'") for item in text.split(",") if item.strip()]

    if len(values) != expected_count:
        raise ValueError(f"{field_name} must contain exactly {expected_count} columns")
    return values


def resolve_column(columns: Sequence[str], requested: str) -> Optional[str]:
    if requested in columns:
        return requested
    return _column_lookup(columns).get(_canonical_name(requested))


def resolve_column_list(columns: Sequence[str], requested: Sequence[str]) -> List[str]:
    resolved: List[str] = []
    for column in requested:
        actual = resolve_column(columns, column)
        if actual is None:
            raise ValueError(f"Column '{column}' does not exist")
        resolved.append(actual)
    return resolved


def _match_group(columns: Sequence[str], groups: Sequence[Sequence[str]]) -> Optional[List[str]]:
    lookup = _column_lookup(columns)
    for group in groups:
        matched: List[str] = []
        for alias in group:
            actual = lookup.get(_canonical_name(alias))
            if actual is None:
                matched = []
                break
            matched.append(actual)
        if matched:
            return matched
    return None


def _match_one(columns: Sequence[str], aliases: Sequence[str]) -> Optional[str]:
    lookup = _column_lookup(columns)
    for alias in aliases:
        actual = lookup.get(_canonical_name(alias))
        if actual is not None:
            return actual
    return None


def _available_vector_fields(columns: Sequence[str], numeric_columns: Sequence[str]) -> List[List[str]]:
    numeric = set(numeric_columns)
    fields: List[List[str]] = []
    for group in VECTOR_ALIASES:
        match = _match_group(columns, [group])
        if match and all(column in numeric for column in match):
            fields.append(match)
    return fields


def detect_fields(
    columns: Sequence[str],
    numeric_columns: Sequence[str],
    explicit_coord_cols: Optional[Sequence[str]] = None,
    explicit_color_col: Optional[str] = None,
    explicit_time_col: Optional[str] = None,
    explicit_vector_cols: Optional[Sequence[str]] = None,
    auto_color: bool = True,
    auto_time: bool = True,
) -> FieldDetection:
    numeric_set = set(numeric_columns)
    available_scalar_fields = [column for column in columns if column in numeric_set]
    available_time_fields = [column for column in columns if column in numeric_set]
    available_vectors = _available_vector_fields(columns, numeric_columns)

    if explicit_coord_cols:
        coord_cols = resolve_column_list(columns, explicit_coord_cols)
        coord_reason = "User explicitly provided coord_cols."
        coord_confidence = 1.0
    else:
        coord_cols = _match_group(columns, COORDINATE_ALIASES) or []
        coord_reason = "Detected coordinate columns from known coordinate aliases." if coord_cols else (
            "No coordinate alias group was found; pass coord_cols explicitly."
        )
        coord_confidence = 0.95 if coord_cols else 0.0

    if explicit_time_col:
        time_col = resolve_column(columns, explicit_time_col)
        if time_col is None:
            raise ValueError(f"Column '{explicit_time_col}' does not exist")
        time_reason = "User explicitly provided time_col."
        time_confidence = 1.0
    elif auto_time:
        time_col = _match_one(columns, TIME_ALIASES)
        if time_col and time_col not in numeric_set:
            time_col = None
        time_reason = "Detected time column from known time aliases." if time_col else "No time column alias was found."
        time_confidence = 0.9 if time_col else 0.0
    else:
        time_col = None
        time_reason = "Automatic time detection disabled."
        time_confidence = 0.0

    if explicit_color_col:
        color_col = resolve_column(columns, explicit_color_col)
        if color_col is None:
            raise ValueError(f"Column '{explicit_color_col}' does not exist")
        color_reason = "User explicitly provided color_col."
        color_confidence = 1.0
    elif auto_color:
        color_col = None
        lookup = _column_lookup(columns)
        for alias in SCALAR_ALIASES:
            candidate = lookup.get(_canonical_name(alias))
            if candidate and candidate in numeric_set and candidate not in coord_cols and candidate != time_col:
                color_col = candidate
                break
        color_reason = "Detected scalar field from known scalar aliases." if color_col else "No scalar alias field was found."
        color_confidence = 0.9 if color_col else 0.0
    else:
        color_col = None
        color_reason = "Automatic scalar field detection disabled."
        color_confidence = 0.0

    if explicit_vector_cols:
        vector_cols = resolve_column_list(columns, explicit_vector_cols)
        vector_reason = "User explicitly provided vector_cols."
        vector_confidence = 1.0
    else:
        vector_cols = []
        for group in available_vectors:
            if not set(group).intersection(coord_cols):
                vector_cols = group
                break
        vector_reason = "Detected vector columns from known vector aliases." if vector_cols else "No vector alias group was found."
        vector_confidence = 0.9 if vector_cols else 0.0

    return FieldDetection(
        coord_cols=coord_cols,
        coord_reason=coord_reason,
        coord_confidence=coord_confidence,
        color_col=color_col,
        color_reason=color_reason,
        color_confidence=color_confidence,
        time_col=time_col,
        time_reason=time_reason,
        time_confidence=time_confidence,
        vector_cols=vector_cols,
        vector_reason=vector_reason,
        vector_confidence=vector_confidence,
        available_scalar_fields=available_scalar_fields,
        available_time_fields=available_time_fields,
        available_vector_fields=available_vectors,
        need_user_confirm=not bool(coord_cols),
    )
