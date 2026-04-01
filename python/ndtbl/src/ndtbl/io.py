from pathlib import Path

from ._binary import (
    open_for_read,
    open_for_write,
    read_group_from_stream,
    read_metadata_from_stream,
    write_group_to_stream,
)
from .model import FieldGroup, GroupMetadata


def read_metadata(path: str | Path) -> GroupMetadata:
    """Read only metadata from an ndtbl file.

    Args:
        path: Path to the input ``.ndtbl`` file.

    Returns:
        The parsed group metadata.
    """

    with open_for_read(path) as stream:
        return read_metadata_from_stream(stream)


def read_group(path: str | Path) -> FieldGroup:
    """Read a complete ndtbl file into memory.

    Args:
        path: Path to the input ``.ndtbl`` file.

    Returns:
        The parsed field group including payload values.
    """

    with open_for_read(path) as stream:
        return read_group_from_stream(stream)


def write_group(path: str | Path, group: FieldGroup) -> None:
    """Write a field group to an ndtbl file.

    Args:
        path: Destination path for the output ``.ndtbl`` file.
        group: In-memory field group to serialize.
    """

    with open_for_write(path) as stream:
        write_group_to_stream(stream, group)
