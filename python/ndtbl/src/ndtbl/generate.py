from dataclasses import dataclass

import numpy as np

from .model import FieldGroup, UniformAxis, normalize_dtype


@dataclass(frozen=True, slots=True)
class LinearFieldSpec:
    name: str
    input_axis: int
    a: float
    b: float


def _linear_field(
    axes: tuple[UniformAxis, ...],
    spec: LinearFieldSpec,
    dtype: np.dtype[np.float32] | np.dtype[np.float64],
) -> np.ndarray:
    if not 0 <= spec.input_axis < len(axes):
        raise ValueError(
            f"input axis {spec.input_axis} is out of range for{len(axes)} axes"
        )

    coordinates = axes[spec.input_axis].coordinates().astype(dtype, copy=False)
    broadcast_shape = [1] * len(axes)
    broadcast_shape[spec.input_axis] = axes[spec.input_axis].size
    coordinates = coordinates.reshape(broadcast_shape)
    return spec.a * coordinates + spec.b


GENERATOR_REGISTRY = {
    "linear": _linear_field,
}


def generate_group(
    axes: tuple[UniformAxis, ...],
    field_specs: tuple[LinearFieldSpec, ...],
    dtype: np.dtype[np.float32] | np.dtype[np.float64] | str = np.float64,
) -> FieldGroup:
    if not axes:
        raise ValueError("at least one axis is required for generation")
    if not field_specs:
        raise ValueError("at least one field is required for generation")

    normalized_dtype = normalize_dtype(dtype)
    axis_sizes = tuple(axis.size for axis in axes)
    values = np.empty((*axis_sizes, len(field_specs)), dtype=normalized_dtype)

    for field_index, spec in enumerate(field_specs):
        generator = GENERATOR_REGISTRY["linear"]
        values[..., field_index] = generator(axes, spec, normalized_dtype)

    field_names = tuple(spec.name for spec in field_specs)
    return FieldGroup(axes=axes, field_names=field_names, values=values)
