from click.testing import CliRunner

from ndtbl.__main__ import main


def test_ndtbl_cli():
    runner = CliRunner()
    result = runner.invoke(main, ())
    assert result.exit_code == 0
