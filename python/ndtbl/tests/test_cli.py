import numpy as np

from ndtbl import read_group, write_group
from ndtbl.cli import main


def test_inspect_prints_key_metadata_and_default_samples(
    runner,
    tmp_path,
    sample_uniform_group,
) -> None:
    path = tmp_path / "inspect.ndtbl"
    write_group(path, sample_uniform_group)

    result = runner.invoke(main, ["inspect", str(path)])

    assert result.exit_code == 0
    assert f"file: {path}" in result.output
    assert "dimension: 2" in result.output
    assert "fields: 2" in result.output
    assert "points: 6" in result.output
    assert "dtype: float64" in result.output
    assert "axis[0]: uniform size=2 min=0 max=1" in result.output
    assert "field[0]: A" in result.output
    assert "samples: 5" in result.output
    assert "sample[0]: (0, 1)" in result.output
    assert "sample[4]: (8, 9)" in result.output
    assert "sample[5]" not in result.output


def test_inspect_short_samples_option_limits_output(
    runner,
    tmp_path,
    sample_uniform_group,
) -> None:
    path = tmp_path / "inspect-short.ndtbl"
    write_group(path, sample_uniform_group)

    result = runner.invoke(main, ["inspect", str(path), "-s", "1"])

    assert result.exit_code == 0
    assert "samples: 1" in result.output
    assert "sample[0]: (0, 1)" in result.output
    assert "sample[1]" not in result.output


def test_generate_writes_expected_file_with_long_options(
    runner, tmp_path
) -> None:
    path = tmp_path / "generated-long.ndtbl"

    result = runner.invoke(
        main,
        [
            "generate",
            str(path),
            "--axis",
            "0",
            "1",
            "3",
            "--axis",
            "10",
            "20",
            "2",
            "--field-linear",
            "A",
            "0",
            "2.0",
            "1.0",
            "--field-linear",
            "B",
            "1",
            "-1.0",
            "5.0",
            "--dtype",
            "float32",
        ],
    )

    assert result.exit_code == 0
    assert f"wrote {path}" in result.output

    group = read_group(path)
    assert group.field_names == ("A", "B")
    assert group.dtype == np.dtype(np.float32)
    np.testing.assert_allclose(
        group.values[..., 0],
        np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], dtype=np.float32),
    )


def test_generate_writes_expected_file_with_short_options(
    runner, tmp_path
) -> None:
    path = tmp_path / "generated-short.ndtbl"

    result = runner.invoke(
        main,
        [
            "generate",
            str(path),
            "-a",
            "0",
            "1",
            "2",
            "-f",
            "A",
            "0",
            "3.0",
            "2.0",
            "-t",
            "float64",
        ],
    )

    assert result.exit_code == 0
    assert f"wrote {path}" in result.output

    group = read_group(path)
    np.testing.assert_allclose(group.values[..., 0], np.array([2.0, 5.0]))


def test_generate_requires_axis_option(runner, tmp_path) -> None:
    path = tmp_path / "missing-axis.ndtbl"

    result = runner.invoke(
        main,
        [
            "generate",
            str(path),
            "--field-linear",
            "A",
            "0",
            "1.0",
            "0.0",
        ],
    )

    assert result.exit_code != 0
    assert (
        "at least one --axis MIN MAX SIZE option is required" in result.output
    )


def test_generate_reports_invalid_axis_reference(runner, tmp_path) -> None:
    path = tmp_path / "bad-axis-ref.ndtbl"

    result = runner.invoke(
        main,
        [
            "generate",
            str(path),
            "--axis",
            "0",
            "1",
            "2",
            "--field-linear",
            "A",
            "9",
            "1.0",
            "0.0",
        ],
    )

    assert result.exit_code != 0
    assert "input axis 9 is out of range" in result.output
