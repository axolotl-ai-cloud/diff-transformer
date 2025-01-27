"""End-to-end tests for differential transformer conversion."""
# pylint: disable=redefined-outer-name
# pylint: disable=duplicate-code

from pathlib import Path
from typing import Optional

import pytest
import yaml
from axolotl.cli.config import load_cfg
from axolotl.cli.main import cli

from axolotl_diff_transformer.convert_diff_transformer import convert_diff_transformer
from axolotl_diff_transformer.plugin.cli import ConvertDiffTransformerCliArgs


def test_cli_validation(cli_runner):
    # Test missing config file
    result = cli_runner.invoke(cli, ["convert-diff-transformer"])
    assert result.exit_code != 0
    assert "Error: Missing argument 'CONFIG'." in result.output

    # Test non-existent config file
    result = cli_runner.invoke(cli, ["convert-diff-transformer", "nonexistent.yml"])
    assert result.exit_code != 0
    assert "Error: Invalid value for 'CONFIG'" in result.output


def test_conversion_cli_basic(tmp_path: Path, base_config):
    output_dir = tmp_path / "converted"
    base_config["output_dir"] = str(output_dir)

    config_path = tmp_path / "config.yml"
    with open(config_path, "w", encoding="utf-8") as file:
        yaml.dump(base_config, file)

    cfg = load_cfg(str(config_path))
    cli_args = ConvertDiffTransformerCliArgs()
    _, debug_info = convert_diff_transformer(cfg, cli_args, str(config_path))

    assert not debug_info
    assert (output_dir / "model.safetensors").exists()
    assert (output_dir / "config.json").exists()
    assert (output_dir / "axolotl_config.yml").exists()


def test_conversion_cli_debug(tmp_path: Path, base_config):
    output_dir = tmp_path / "converted"
    base_config["output_dir"] = str(output_dir)

    config_path = tmp_path / "config.yml"
    with open(config_path, "w", encoding="utf-8") as file:
        yaml.dump(base_config, file)

    cfg = load_cfg(str(config_path))
    cli_args = ConvertDiffTransformerCliArgs(debug=True)
    _, debug_info = convert_diff_transformer(cfg, cli_args, str(config_path))

    assert not debug_info["generations_match"]
    assert not debug_info["match_expected"]
    assert (output_dir / "model.safetensors").exists()
    assert (output_dir / "config.json").exists()
    assert (output_dir / "axolotl_config.yml").exists()


def test_conversion_cli_reproduce(tmp_path: Path, base_config):
    output_dir = tmp_path / "converted"
    base_config["output_dir"] = str(output_dir)

    config_path = tmp_path / "config.yml"
    with open(config_path, "w", encoding="utf-8") as file:
        yaml.dump(base_config, file)

    cfg = load_cfg(str(config_path))
    cli_args = ConvertDiffTransformerCliArgs(
        debug=True, zero_init=True, sublayer_norm=False
    )
    _, debug_info = convert_diff_transformer(cfg, cli_args, str(config_path))

    assert debug_info["generations_match"] is True
    assert (output_dir / "model.safetensors").exists()
    assert (output_dir / "config.json").exists()
    assert (output_dir / "axolotl_config.yml").exists()


@pytest.mark.parametrize(
    "attention", ["eager_attention", "sdp_attention", "flash_attention"]
)
def test_conversion_cli_repoduce_attentions(
    tmp_path: Path, base_config, attention: Optional[str]
):
    output_dir = tmp_path / "converted"
    base_config["output_dir"] = str(output_dir)
    base_config[attention] = True

    config_path = tmp_path / "config.yml"
    with open(config_path, "w", encoding="utf-8") as file:
        yaml.dump(base_config, file)

    cfg = load_cfg(str(config_path))
    cli_args = ConvertDiffTransformerCliArgs(
        debug=True, zero_init=True, sublayer_norm=False
    )
    _, debug_info = convert_diff_transformer(cfg, cli_args, str(config_path))

    assert debug_info["generations_match"] is True
    assert (output_dir / "model.safetensors").exists()
    assert (output_dir / "config.json").exists()
    assert (output_dir / "axolotl_config.yml").exists()


@pytest.mark.parametrize(
    "attention", ["eager_attention", "sdp_attention", "flash_attention"]
)
def test_conversion_cli_split_heads(tmp_path: Path, base_config, attention: str):
    output_dir = tmp_path / "converted"

    # Smallest model with an even number of attention heads
    base_config["base_model"] = "HuggingFaceTB/SmolLM2-1.7B"
    base_config["output_dir"] = str(output_dir)
    base_config[attention] = True

    config_path = tmp_path / "config.yml"
    with open(config_path, "w", encoding="utf-8") as file:
        yaml.dump(base_config, file)

    cfg = load_cfg(str(config_path))
    cli_args = ConvertDiffTransformerCliArgs(debug=True, split_heads=True)
    _, debug_info = convert_diff_transformer(cfg, cli_args, str(config_path))

    assert debug_info["generations_match"] is False
    assert (output_dir / "model.safetensors").exists()
    assert (output_dir / "config.json").exists()
    assert (output_dir / "axolotl_config.yml").exists()
