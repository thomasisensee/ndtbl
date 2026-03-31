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
    with open_for_read(path) as stream:
        return read_metadata_from_stream(stream)


def read_group(path: str | Path) -> FieldGroup:
    with open_for_read(path) as stream:
        return read_group_from_stream(stream)


def write_group(path: str | Path, group: FieldGroup) -> None:
    with open_for_write(path) as stream:
        write_group_to_stream(stream, group)
