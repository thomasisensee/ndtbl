from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray: TypeAlias = NDArray[np.float32] | NDArray[np.float64]


class NdtblFormatError(RuntimeError):
    """Raised when an .ndtbl file cannot be parsed safely."""


@dataclass(frozen=True, slots=True)
class UniformAxis:
    min: float
    max: float
    size: int

    def __post_init__(self) -> None:
        min_value = float(self.min)
        max_value = float(self.max)
        size = int(self.size)

        if size < 1:
            raise ValueError("uniform axis must contain at least one point")
        if size > 1 and not max_value > min_value:
            raise ValueError("uniform axis requires max > min when size > 1")
        if size == 1:
            max_value = min_value

        object.__setattr__(self, "min", min_value)
        object.__setattr__(self, "max", max_value)
        object.__setattr__(self, "size", size)

    @property
    def kind(self) -> str:
        return "uniform"

    def coordinates(self) -> NDArray[np.float64]:
        if self.size == 1:
            return np.asarray([self.min], dtype=np.float64)
        return np.linspace(self.min, self.max, self.size, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class ExplicitAxis:
    coordinates_values: tuple[float, ...]

    def __init__(self, coordinates: ArrayLike) -> None:
        raw = np.asarray(coordinates, dtype=np.float64)
        if raw.ndim != 1:
            raise ValueError(
                "explicit axis coordinates must be one-dimensional"
            )
        if raw.size < 1:
            raise ValueError("explicit axis must contain at least one point")
        if raw.size > 1 and np.any(np.diff(raw) <= 0.0):
            raise ValueError(
                "explicit axis coordinates must be strictly increasing"
            )
        object.__setattr__(
            self, "coordinates_values", tuple(float(x) for x in raw)
        )

    @property
    def kind(self) -> str:
        return "explicit"

    @property
    def size(self) -> int:
        return len(self.coordinates_values)

    @property
    def min(self) -> float:
        return self.coordinates_values[0]

    @property
    def max(self) -> float:
        return self.coordinates_values[-1]

    def coordinates(self) -> NDArray[np.float64]:
        return np.asarray(self.coordinates_values, dtype=np.float64)


Axis: TypeAlias = UniformAxis | ExplicitAxis


def normalize_dtype(
    dtype: np.dtype[np.generic] | str,
) -> np.dtype[np.float32] | np.dtype[np.float64]:
    normalized = np.dtype(dtype)
    if normalized == np.dtype(np.float32):
        return np.dtype(np.float32)
    if normalized == np.dtype(np.float64):
        return np.dtype(np.float64)
    raise ValueError("ndtbl only supports float32 and float64 payloads")


def _normalize_axes(axes: tuple[Axis, ...] | list[Axis]) -> tuple[Axis, ...]:
    normalized = tuple(axes)
    if not normalized:
        raise ValueError("field groups must contain at least one axis")
    for axis in normalized:
        if not isinstance(axis, UniformAxis | ExplicitAxis):
            raise TypeError(
                "axes must be UniformAxis or ExplicitAxis instances"
            )
    return normalized


def _normalize_field_names(
    field_names: tuple[str, ...] | list[str],
) -> tuple[str, ...]:
    normalized = tuple(str(name) for name in field_names)
    if not normalized:
        raise ValueError("field groups must contain at least one field")
    if any(not name for name in normalized):
        raise ValueError("field names must be non-empty strings")
    return normalized


def _normalize_values(values: ArrayLike) -> FloatArray:
    array = np.asarray(values)
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError("field values must be numeric")
    if array.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        array = array.astype(np.float64)
    return np.ascontiguousarray(array)


@dataclass(frozen=True, slots=True)
class GroupMetadata:
    axes: tuple[Axis, ...]
    field_names: tuple[str, ...]
    dtype: np.dtype[np.float32] | np.dtype[np.float64]

    def __post_init__(self) -> None:
        object.__setattr__(self, "axes", _normalize_axes(self.axes))
        object.__setattr__(
            self, "field_names", _normalize_field_names(self.field_names)
        )
        object.__setattr__(self, "dtype", normalize_dtype(self.dtype))

    @property
    def dimension(self) -> int:
        return len(self.axes)

    @property
    def field_count(self) -> int:
        return len(self.field_names)

    @property
    def axis_sizes(self) -> tuple[int, ...]:
        return tuple(axis.size for axis in self.axes)

    @property
    def point_count(self) -> int:
        point_count = 1
        for size in self.axis_sizes:
            point_count *= size
        return point_count

    @property
    def dtype_name(self) -> str:
        return str(self.dtype)


@dataclass(slots=True)
class FieldGroup:
    axes: tuple[Axis, ...]
    field_names: tuple[str, ...]
    values: FloatArray

    def __post_init__(self) -> None:
        self.axes = _normalize_axes(self.axes)
        self.field_names = _normalize_field_names(self.field_names)
        self.values = _normalize_values(self.values)

        expected_shape = (*self.axis_sizes, self.field_count)
        if self.values.ndim != self.dimension + 1:
            raise ValueError(
                "field values must have shape (*axis_sizes, field_count)"
            )
        if self.values.shape != expected_shape:
            raise ValueError(
                f"field values shape {self.values.shape} does not match "
                f"expected {expected_shape}"
            )

    @property
    def dimension(self) -> int:
        return len(self.axes)

    @property
    def field_count(self) -> int:
        return len(self.field_names)

    @property
    def axis_sizes(self) -> tuple[int, ...]:
        return tuple(axis.size for axis in self.axes)

    @property
    def point_count(self) -> int:
        point_count = 1
        for size in self.axis_sizes:
            point_count *= size
        return point_count

    @property
    def dtype(self) -> np.dtype[np.float32] | np.dtype[np.float64]:
        return normalize_dtype(self.values.dtype)

    @property
    def dtype_name(self) -> str:
        return str(self.dtype)

    def metadata(self) -> GroupMetadata:
        return GroupMetadata(
            axes=self.axes,
            field_names=self.field_names,
            dtype=self.dtype,
        )
