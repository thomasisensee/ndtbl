from pathlib import Path

import click
import numpy as np

from .generate import (
    GenerationSizeEstimate,
    LinearFieldSpec,
    estimate_generated_group_size,
    generate_group,
)
from .io import read_group, write_group
from .model import ExplicitAxis, FieldGroup, GroupMetadata, UniformAxis

DEFAULT_MAX_SIZE_MIB = 128.0


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
    tokens: tuple[str, ...],
    axis_count: int,
) -> tuple[Path, tuple[LinearFieldSpec, ...]]:
    parsed_specs: list[LinearFieldSpec] = []
    output: Path | None = None
    index = 0
    while index < len(tokens):
        option = tokens[index]
        if option not in ("--field-linear", "-f"):
            if output is not None:
                raise click.UsageError(f"unexpected extra argument: {option}")
            output = Path(option)
            index += 1
            continue
        next_index = index + 1
        expected_parts = axis_count + 2
        next_index = index + 1 + expected_parts

        parts = tokens[index + 1 : next_index]
        if len(parts) != expected_parts:
            raise click.UsageError(
                "each --field-linear option requires NAME OFFSET and"
                "one coefficient per axis"
            )

        try:
            name = parts[0]
            offset = float(parts[1])
            coefficients = tuple(float(value) for value in parts[2:])
        except ValueError as error:
            raise click.UsageError(
                "field offset and coefficients must be floats"
            ) from error

        if len(coefficients) != axis_count:
            raise click.UsageError(
                "field coefficient count must match the number of axes"
            )

        parsed_specs.append(
            LinearFieldSpec(
                name=name,
                offset=offset,
                coefficients=coefficients,
            )
        )
        index = next_index

    if output is None:
        raise click.UsageError("missing output path")
    if not parsed_specs:
        raise click.UsageError(
            "at least one --field-linear NAME OFFSET C0 [C1 ...] option"
            "is required"
        )

    return output, tuple(parsed_specs)


def _format_mib(size_bytes: int) -> str:
    return f"{size_bytes / (1024 * 1024):.2f}"


def _enforce_generation_size_limit(
    estimate: GenerationSizeEstimate,
    dtype_name: str,
    max_size_mib: float,
) -> None:
    if estimate.estimated_file_mib <= max_size_mib:
        return

    raise click.ClickException(
        "generated table exceeds the configured size limit: "
        f"{estimate.point_count} points, "
        f"{estimate.field_count} fields, "
        f"dtype={dtype_name}, "
        f"estimated size={_format_mib(estimate.estimated_file_bytes)} MiB, "
        f"limit={max_size_mib:.2f} MiB. "
        "Use --max-size-mib to raise the limit explicitly."
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


@main.command(
    "generate",
    context_settings={
        "allow_extra_args": True,
        "ignore_unknown_options": True,
    },
)
@click.argument(
    "output", required=False, type=click.Path(dir_okay=False, path_type=Path)
)
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
    "--dtype",
    "-t",
    type=click.Choice(("float32", "float64"), case_sensitive=True),
    default="float64",
    show_default=True,
)
@click.option(
    "--max-size-mib",
    "-m",
    type=click.FloatRange(min=0.0, min_open=True),
    default=DEFAULT_MAX_SIZE_MIB,
    show_default=True,
    help="Abort generation when the estimated file size exceeds this limit.",
)
@click.pass_context
def generate_command(
    ctx: click.Context,
    output: Path | None,
    axis_specs: tuple[tuple[float, float, int], ...],
    dtype: str,
    max_size_mib: float,
) -> None:
    """Generate a simple .ndtbl file with predefined linear fields.

    Field syntax:
      --field-linear NAME OFFSET C0 [C1 ...]

    Provide one coefficient per axis in axis order.
    """

    axes = _parse_axes(axis_specs)
    extra_tokens = tuple(ctx.args)
    if output is not None:
        extra_tokens = (str(output), *extra_tokens)
    output_path, linear_fields = _parse_linear_fields(extra_tokens, len(axes))
    estimate = estimate_generated_group_size(
        axes, linear_fields, np.dtype(dtype)
    )
    _enforce_generation_size_limit(estimate, dtype, max_size_mib)
    try:
        group = generate_group(axes, linear_fields, np.dtype(dtype))
        write_group(output_path, group)
    except (OSError, ValueError) as error:
        raise click.ClickException(str(error)) from error
    click.echo(f"wrote {output_path}")
