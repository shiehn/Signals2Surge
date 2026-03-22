"""Tests for batch CLI commands — help text and argument validation."""

import re

import pytest
from typer.testing import CliRunner

from synth2surge.cli.main import app

runner = CliRunner()


def _strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences from Rich output."""
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


@pytest.mark.unit
class TestQueueCLI:
    def test_queue_help(self):
        result = runner.invoke(app, ["queue", "--help"])
        assert result.exit_code == 0
        text = _strip_ansi(result.output)
        assert "--plugin" in text
        assert "--queue-dir" in text
        assert "--probe-mode" in text

    def test_queue_requires_plugin(self):
        result = runner.invoke(app, ["queue"])
        assert result.exit_code != 0


@pytest.mark.unit
class TestBatchOptimizeCLI:
    def test_batch_optimize_help(self):
        result = runner.invoke(app, ["batch-optimize", "--help"])
        assert result.exit_code == 0
        text = _strip_ansi(result.output)
        assert "--queue-dir" in text
        assert "--input" in text
        assert "--output-dir" in text
        assert "--warm-start" in text

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
