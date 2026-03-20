"""Unit tests for CLI argument parsing and help text."""

from typer.testing import CliRunner

from synth2surge.cli.main import app

runner = CliRunner()


class TestCLIHelp:
    def test_main_help(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "Translate arbitrary VST synth patches" in result.stdout

    def test_capture_help(self):
        result = runner.invoke(app, ["capture", "--help"])
        assert result.exit_code == 0
        assert "--plugin" in result.stdout

    def test_optimize_help(self):
        result = runner.invoke(app, ["optimize", "--help"])
        assert result.exit_code == 0
        assert "--target" in result.stdout
        assert "--trials-t1" in result.stdout

    def test_build_prior_help(self):
        result = runner.invoke(app, ["build-prior", "--help"])
        assert result.exit_code == 0
        assert "--factory-dir" in result.stdout

    def test_inspect_help(self):
        result = runner.invoke(app, ["inspect", "--help"])
        assert result.exit_code == 0
        assert "--patch" in result.stdout

    def test_serve_help(self):
        result = runner.invoke(app, ["serve", "--help"])
        assert result.exit_code == 0
        assert "--port" in result.stdout


class TestCLIInspect:
    def test_inspect_factory_patch(self):
        from pathlib import Path

        fixture = Path(__file__).parent.parent / "fixtures" / "patches" / "init_fm2.fxp"
        result = runner.invoke(app, ["inspect", "--patch", str(fixture)])
        assert result.exit_code == 0
        assert "Init FM2" in result.stdout
        assert "Parameters" in result.stdout
