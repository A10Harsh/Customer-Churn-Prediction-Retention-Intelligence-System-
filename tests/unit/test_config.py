"""Unit tests for configuration utilities."""

from pathlib import Path

import pytest

from retention.utils.config import load_config


def test_load_valid_config(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"

    config_file.write_text(
        "random_state: 42\n",
        encoding="utf-8",
    )

    config = load_config(config_file)

    assert config["random_state"] == 42


def test_missing_config_file(tmp_path: Path) -> None:
    config_file = tmp_path / "missing.yaml"

    with pytest.raises(FileNotFoundError):
        load_config(config_file)


def test_invalid_config_structure(tmp_path: Path) -> None:
    config_file = tmp_path / "invalid.yaml"

    config_file.write_text(
        "- item1\n- item2\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_config(config_file)
