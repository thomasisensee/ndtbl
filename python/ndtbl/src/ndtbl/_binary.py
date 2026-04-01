import struct
from pathlib import Path
from typing import BinaryIO

import numpy as np

from .model import (
    ExplicitAxis,
    FieldGroup,
    GroupMetadata,
    NdtblFormatError,
    UniformAxis,
)

MAGIC = b"NDTBL1\0\0"
ENDIAN_MARKER = 0x01020304
VERSION = 1

AXIS_KIND_UNIFORM = 1
AXIS_KIND_EXPLICIT = 2

SCALAR_FLOAT32 = 1
SCALAR_FLOAT64 = 2

NATIVE_PREFIX = "="
UINT8 = struct.Struct(f"{NATIVE_PREFIX}B")
UINT16 = struct.Struct(f"{NATIVE_PREFIX}H")
UINT32 = struct.Struct(f"{NATIVE_PREFIX}I")
UINT64 = struct.Struct(f"{NATIVE_PREFIX}Q")
DOUBLE = struct.Struct(f"{NATIVE_PREFIX}d")

DTYPE_TO_TAG = {
    np.dtype(np.float32): SCALAR_FLOAT32,
    np.dtype(np.float64): SCALAR_FLOAT64,
}
TAG_TO_DTYPE = {
    SCALAR_FLOAT32: np.dtype(np.float32),
    SCALAR_FLOAT64: np.dtype(np.float64),
}


def _read_exact(stream: BinaryIO, size: int) -> bytes:
    """Read exactly ``size`` bytes from a binary stream.

    Args:
        stream: Open binary stream positioned at the next payload bytes.
        size: Number of bytes to read.

    Returns:
        The requested byte sequence.
    """

    data = stream.read(size)
    if len(data) != size:
        raise NdtblFormatError("unexpected end of ndtbl file")
    return data


def _read_uint8(stream: BinaryIO) -> int:
    """Read one unsigned 8-bit integer from a stream."""
    return UINT8.unpack(_read_exact(stream, UINT8.size))[0]


def _read_uint16(stream: BinaryIO) -> int:
    """Read one unsigned 16-bit integer from a stream."""
    return UINT16.unpack(_read_exact(stream, UINT16.size))[0]


def _read_uint32(stream: BinaryIO) -> int:
    """Read one unsigned 32-bit integer from a stream."""
    return UINT32.unpack(_read_exact(stream, UINT32.size))[0]


def _read_uint64(stream: BinaryIO) -> int:
    """Read one unsigned 64-bit integer from a stream."""
    return UINT64.unpack(_read_exact(stream, UINT64.size))[0]


def _read_double(stream: BinaryIO) -> float:
    """Read one IEEE-754 double-precision value from a stream."""
    return DOUBLE.unpack(_read_exact(stream, DOUBLE.size))[0]


def _write_uint8(stream: BinaryIO, value: int) -> None:
    """Write one unsigned 8-bit integer to a stream.

    Args:
        stream: Open binary stream positioned at the write offset.
        value: Integer value to encode.
    """

    stream.write(UINT8.pack(value))


def _write_uint16(stream: BinaryIO, value: int) -> None:
    """Write one unsigned 16-bit integer to a stream.

    Args:
        stream: Open binary stream positioned at the write offset.
        value: Integer value to encode.
    """

    stream.write(UINT16.pack(value))


def _write_uint32(stream: BinaryIO, value: int) -> None:
    """Write one unsigned 32-bit integer to a stream.

    Args:
        stream: Open binary stream positioned at the write offset.
        value: Integer value to encode.
    """

    stream.write(UINT32.pack(value))


def _write_uint64(stream: BinaryIO, value: int) -> None:
    """Write one unsigned 64-bit integer to a stream.

    Args:
        stream: Open binary stream positioned at the write offset.
        value: Integer value to encode.
    """

    stream.write(UINT64.pack(value))


def _write_double(stream: BinaryIO, value: float) -> None:
    """Write one double-precision float to a stream.

    Args:
        stream: Open binary stream positioned at the write offset.
        value: Floating-point value to encode.
    """

    stream.write(DOUBLE.pack(value))


def _read_string(stream: BinaryIO) -> str:
    """Read a length-prefixed UTF-8 string from a stream.

    Args:
        stream: Open binary stream positioned at the string header.

    Returns:
        The decoded string value.
    """

    size = _read_uint64(stream)
    data = _read_exact(stream, size)
    return data.decode("utf-8")


def _write_string(stream: BinaryIO, value: str) -> None:
    """Write a length-prefixed UTF-8 string to a stream.

    Args:
        stream: Open binary stream positioned at the string header.
        value: String value to encode.
    """

    encoded = value.encode("utf-8")
    _write_uint64(stream, len(encoded))
    stream.write(encoded)


def _read_axis(stream: BinaryIO) -> UniformAxis | ExplicitAxis:
    """Read one serialized axis definition from a stream.

    Args:
        stream: Open binary stream positioned at an axis header.

    Returns:
        The decoded axis object.
    """

    axis_tag = _read_uint8(stream)
    _read_uint8(stream)
    _read_uint16(stream)
    size = _read_uint64(stream)

    if axis_tag == AXIS_KIND_UNIFORM:
        return UniformAxis(
            min=_read_double(stream),
            max=_read_double(stream),
            size=size,
        )

    if axis_tag == AXIS_KIND_EXPLICIT:
        coordinates = [_read_double(stream) for _ in range(size)]
        return ExplicitAxis(coordinates)

    raise NdtblFormatError(f"unsupported ndtbl axis kind: {axis_tag}")


def _write_axis(stream: BinaryIO, axis: UniformAxis | ExplicitAxis) -> None:
    """Write one axis definition to a stream.

    Args:
        stream: Open binary stream positioned at an axis header.
        axis: Axis object to encode.
    """

    if isinstance(axis, UniformAxis):
        _write_uint8(stream, AXIS_KIND_UNIFORM)
        _write_uint8(stream, 0)
        _write_uint16(stream, 0)
        _write_uint64(stream, axis.size)
        _write_double(stream, axis.min)
        _write_double(stream, axis.max)
        return

    _write_uint8(stream, AXIS_KIND_EXPLICIT)
    _write_uint8(stream, 0)
    _write_uint16(stream, 0)
    _write_uint64(stream, axis.size)
    for coordinate in axis.coordinates_values:
        _write_double(stream, coordinate)


def read_metadata_from_stream(stream: BinaryIO) -> GroupMetadata:
    """Read ndtbl metadata from an already opened stream.

    Args:
        stream: Open binary stream positioned at the file start.

    Returns:
        Parsed metadata without loading the numeric payload.
    """

    magic = _read_exact(stream, len(MAGIC))
    if magic != MAGIC:
        raise NdtblFormatError("invalid ndtbl magic header")

    marker = _read_uint32(stream)
    if marker != ENDIAN_MARKER:
        raise NdtblFormatError("unsupported ndtbl endianness")

    version = _read_uint8(stream)
    if version != VERSION:
        raise NdtblFormatError(f"unsupported ndtbl version: {version}")

    scalar_tag = _read_uint8(stream)
    _read_uint16(stream)

    try:
        dtype = TAG_TO_DTYPE[scalar_tag]
    except KeyError as error:
        raise NdtblFormatError(
            f"unsupported ndtbl scalar type: {scalar_tag}"
        ) from error

    dimension = _read_uint64(stream)
    field_count = _read_uint64(stream)
    point_count = _read_uint64(stream)

    axes = tuple(_read_axis(stream) for _ in range(dimension))
    field_names = tuple(_read_string(stream) for _ in range(field_count))

    metadata = GroupMetadata(axes=axes, field_names=field_names, dtype=dtype)
    if metadata.point_count != point_count:
        raise NdtblFormatError("ndtbl point count does not match axis extents")
    return metadata


def read_group_from_stream(stream: BinaryIO) -> FieldGroup:
    """Read an entire ndtbl file from an already opened stream.

    Args:
        stream: Open binary stream positioned at the file start.

    Returns:
        Parsed field group including payload values.
    """

    metadata = read_metadata_from_stream(stream)
    value_count = metadata.point_count * metadata.field_count
    payload_size = value_count * metadata.dtype.itemsize
    payload = _read_exact(stream, payload_size)
    values = np.frombuffer(payload, dtype=metadata.dtype).copy()
    shaped = values.reshape(
        (*metadata.axis_sizes, metadata.field_count), order="C"
    )
    return FieldGroup(
        axes=metadata.axes,
        field_names=metadata.field_names,
        values=shaped,
    )


def write_group_to_stream(stream: BinaryIO, group: FieldGroup) -> None:
    """Write a field group to an already opened stream.

    Args:
        stream: Open binary stream positioned at the file start.
        group: Field group to serialize.
    """

    metadata = group.metadata()

    try:
        scalar_tag = DTYPE_TO_TAG[metadata.dtype]
    except KeyError as error:
        raise ValueError(
            "ndtbl only supports float32 and float64 payloads"
        ) from error

    stream.write(MAGIC)
    _write_uint32(stream, ENDIAN_MARKER)
    _write_uint8(stream, VERSION)
    _write_uint8(stream, scalar_tag)
    _write_uint16(stream, 0)
    _write_uint64(stream, metadata.dimension)
    _write_uint64(stream, metadata.field_count)
    _write_uint64(stream, metadata.point_count)

    for axis in metadata.axes:
        _write_axis(stream, axis)

    for field_name in metadata.field_names:
        _write_string(stream, field_name)

    stream.write(group.values.reshape(-1, order="C").tobytes(order="C"))


def open_for_read(path: str | Path) -> BinaryIO:
    """Open an ndtbl file for binary reading.

    Args:
        path: File path to open.

    Returns:
        Binary file object opened in read mode.
    """

    return Path(path).open("rb")


def open_for_write(path: str | Path) -> BinaryIO:
    """Open an ndtbl file for binary writing.

    Args:
        path: File path to open.

    Returns:
        Binary file object opened in write mode.
    """

    return Path(path).open("wb")
