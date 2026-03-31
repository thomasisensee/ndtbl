from pathlib import Path

import click
import numpy as np

from .generate import LinearFieldSpec, generate_group
from .io import read_group, write_group
from .model import ExplicitAxis, FieldGroup, GroupMetadata, UniformAxis


def _format_axis(axis: UniformAxis | ExplicitAxis, axis_index: int) -> str:
    if isinstance(axis, UniformAxis):
        return (
            f"axis[{axis_index}]: uniform "
            f"size={axis.size} min={axis.min:g} max={axis.max:g}"
        )

    coordinates = ", ".join(f"{value:g}" for value in axis.coordinates_values)
    return (
        f"axis[{axis_index}]: explicit size={axis.size} "
        f"coordinates=[{coordinates}]"
    )


def _echo_metadata(path: Path, metadata: GroupMetadata) -> None:
    click.echo(f"file: {path}")
    click.echo(f"dimension: {metadata.dimension}")
    click.echo(f"fields: {metadata.field_count}")
    click.echo(f"points: {metadata.point_count}")
    click.echo(f"dtype: {metadata.dtype_name}")

    for axis_index, axis in enumerate(metadata.axes):
        click.echo(_format_axis(axis, axis_index))

    for field_index, field_name in enumerate(metadata.field_names):
        click.echo(f"field[{field_index}]: {field_name}")


def _echo_samples(group: FieldGroup, samples: int) -> None:
    flat_values = group.values.reshape(
        group.point_count, group.field_count, order="C"
    )
    sample_count = min(samples, group.point_count)
    click.echo(f"samples: {sample_count}")
    for sample_index in range(sample_count):
        values = ", ".join(f"{value:g}" for value in flat_values[sample_index])
        click.echo(f"sample[{sample_index}]: ({values})")


def _parse_axes(
    axis_specs: tuple[tuple[float, float, int], ...],
) -> tuple[UniformAxis, ...]:
    if not axis_specs:
        raise click.UsageError(
            "at least one --axis MIN MAX SIZE option is required"
        )

    axes: list[UniformAxis] = []
    for min_value, max_value, size in axis_specs:
        try:
            axes.append(UniformAxis(min=min_value, max=max_value, size=size))
        except ValueError as error:
            raise click.UsageError(str(error)) from error
    return tuple(axes)


def _parse_linear_fields(
    field_specs: tuple[tuple[str, int, float, float], ...],
) -> tuple[LinearFieldSpec, ...]:
    if not field_specs:
        raise click.UsageError(
            "at least one --field-linear NAME INPUT_AXIS A B optionis required"
        )

    return tuple(
        LinearFieldSpec(name=name, input_axis=input_axis, a=a, b=b)
        for name, input_axis, a, b in field_specs
    )


@click.group()
def main() -> None:
    """Read, inspect, and generate .ndtbl files."""


@main.command("inspect")
@click.argument(
    "file", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.option(
    "--samples",
    "-s",
    default=5,
    show_default=True,
    type=click.IntRange(min=0),
)
def inspect_command(file: Path, samples: int) -> None:
    """Print metadata and sample payload values from an .ndtbl file."""

    try:
        group = read_group(file)
    except (OSError, ValueError) as error:
        raise click.ClickException(str(error)) from error
    _echo_metadata(file, group.metadata())
    _echo_samples(group, samples)


@main.command("generate")
@click.argument("output", type=click.Path(dir_okay=False, path_type=Path))
@click.option(
    "--axis",
    "-a",
    "axis_specs",
    multiple=True,
    nargs=3,
    type=(float, float, int),
    metavar="MIN MAX SIZE",
    help="Define one uniform axis. Repeat for each axis in storage order.",
)
@click.option(
    "--field-linear",
    "-f",
    "field_specs",
    multiple=True,
    nargs=4,
    type=(str, int, float, float),
    metavar="NAME INPUT_AXIS A B",
    help="Define one linear field A*x+B based on the chosen zero-based axis.",
)
@click.option(
    "--dtype",
    "-t",
    type=click.Choice(("float32", "float64"), case_sensitive=True),
    default="float64",
    show_default=True,
)
def generate_command(
    output: Path,
    axis_specs: tuple[tuple[float, float, int], ...],
    field_specs: tuple[tuple[str, int, float, float], ...],
    dtype: str,
) -> None:
    """Generate a simple .ndtbl file with predefined linear fields."""

    axes = _parse_axes(axis_specs)
    linear_fields = _parse_linear_fields(field_specs)
    try:
        group = generate_group(axes, linear_fields, np.dtype(dtype))
        write_group(output, group)
    except (OSError, ValueError) as error:
        raise click.ClickException(str(error)) from error
    click.echo(f"wrote {output}")
