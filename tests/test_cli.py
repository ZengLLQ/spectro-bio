"""Tests for SpectroBio CLI commands."""

import pytest
from spectro_bio.cli import main


def test_cli_help(capsys):
    """Test CLI help output."""
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert "spectro-bio" in captured.out


def test_cli_ctf(capsys):
    """Test CLI ctf command."""
    code = main(["ctf", "--defocus", "12000.0", "--voltage", "300.0"])
    assert code == 0
    captured = capsys.readouterr()
    assert "Contrast Transfer Function" in captured.out
    assert "First Thon Ring Zero" in captured.out


def test_cli_fsc(capsys):
    """Test CLI fsc command."""
    code = main(["fsc", "--size", "16", "--pixel-size", "1.2"])
    assert code == 0
    captured = capsys.readouterr()
    assert "Fourier Shell Correlation" in captured.out
    assert "Resolution" in captured.out


def test_cli_pipeline(capsys):
    """Test CLI pipeline execution with modest size for rapid test suite."""
    code = main(["pipeline", "--size", "16", "--n-projections", "8", "--snr", "5.0"])
    assert code == 0
    captured = capsys.readouterr()
    assert "Pipeline Execution Completed Successfully" in captured.out
