"""Tests for batch CLI commands — help text and argument validation."""

import pytest
from typer.testing import CliRunner

from synth2surge.cli.main import app

runner = CliRunner()


@pytest.mark.unit
class TestQueueCLI:
    def test_queue_help(self):
        result = runner.invoke(app, ["queue", "--help"])
        assert result.exit_code == 0
        assert "plugin" in result.output
        assert "queue-dir" in result.output
        assert "probe-mode" in result.output

    def test_queue_requires_plugin(self):
        result = runner.invoke(app, ["queue"])
        assert result.exit_code != 0


@pytest.mark.unit
class TestBatchOptimizeCLI:
    def test_batch_optimize_help(self):
        result = runner.invoke(app, ["batch-optimize", "--help"])
        assert result.exit_code == 0
        assert "queue-dir" in result.output
        assert "input" in result.output
        assert "output-dir" in result.output
        assert "warm-start" in result.output

    def test_requires_input_source(self):
        """Neither --queue-dir nor --input should fail."""
        result = runner.invoke(app, ["batch-optimize"])
        assert result.exit_code != 0

    def test_rejects_both_inputs(self):
        """Both --queue-dir and --input should fail."""
        result = runner.invoke(app, [
            "batch-optimize",
            "--queue-dir", "/tmp/q",
            "--input", "/tmp/w",
        ])
        assert result.exit_code != 0
