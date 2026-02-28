"""Tests for fbrl/config.py — ExperimentConfig, load_config, roundtrip."""
import os
import tempfile
import yaml
import pytest

from fbrl.config import ExperimentConfig, load_config, config_to_dict, config_from_dict


def test_defaults():
    cfg = ExperimentConfig()
    assert cfg.model_type == 'letter'
    assert cfg.latent_dim == 256
    assert cfg.epochs == 200
    assert cfg.guide_weight == 8.0
    assert cfg.scan_guide_weight is None
    assert cfg.scaffold_epochs is None
    assert cfg.motor_enabled is False


def test_load_config_from_yaml():
    data = {'model_type': 'bigram', 'epochs': 50, 'batch_size': 16}
    with tempfile.NamedTemporaryFile('w', suffix='.yaml', delete=False) as f:
        yaml.dump(data, f)
        path = f.name
    try:
        cfg = load_config(path)
        assert cfg.model_type == 'bigram'
        assert cfg.epochs == 50
        assert cfg.batch_size == 16
    finally:
        os.unlink(path)


def test_cli_overrides():
    data = {'epochs': 50, 'batch_size': 16}
    with tempfile.NamedTemporaryFile('w', suffix='.yaml', delete=False) as f:
        yaml.dump(data, f)
        path = f.name
    try:
        cfg = load_config(path, cli_overrides={'epochs': 100, 'batch_size': None})
        assert cfg.epochs == 100
        assert cfg.batch_size == 16  # None override is skipped
    finally:
        os.unlink(path)


def test_scan_patch_size_list_to_tuple():
    data = {'scan_patch_size': [12, 18]}
    with tempfile.NamedTemporaryFile('w', suffix='.yaml', delete=False) as f:
        yaml.dump(data, f)
        path = f.name
    try:
        cfg = load_config(path)
        assert cfg.scan_patch_size == (12, 18)
        assert isinstance(cfg.scan_patch_size, tuple)
    finally:
        os.unlink(path)


def test_scan_guide_weight_defaults_to_guide_weight():
    data = {'guide_weight': 5.0}
    with tempfile.NamedTemporaryFile('w', suffix='.yaml', delete=False) as f:
        yaml.dump(data, f)
        path = f.name
    try:
        cfg = load_config(path)
        assert cfg.scan_guide_weight == 5.0
    finally:
        os.unlink(path)


def test_scaffold_epochs_computed():
    data = {'epochs': 100, 'scaffold_ratio': 0.5}
    with tempfile.NamedTemporaryFile('w', suffix='.yaml', delete=False) as f:
        yaml.dump(data, f)
        path = f.name
    try:
        cfg = load_config(path)
        assert cfg.scaffold_epochs == 50
    finally:
        os.unlink(path)


def test_extra_keys_filtered():
    data = {'epochs': 10, 'nonexistent_field': 'should_be_ignored'}
    with tempfile.NamedTemporaryFile('w', suffix='.yaml', delete=False) as f:
        yaml.dump(data, f)
        path = f.name
    try:
        cfg = load_config(path)
        assert cfg.epochs == 10
        assert not hasattr(cfg, 'nonexistent_field')
    finally:
        os.unlink(path)


def test_roundtrip_config_to_dict_from_dict():
    cfg = ExperimentConfig(
        model_type='word', epochs=300, scan_patch_size=(12, 18),
        guide_weight=5.0, motor_enabled=True,
    )
    d = config_to_dict(cfg)
    assert isinstance(d['scan_patch_size'], list)
    cfg2 = config_from_dict(d)
    assert cfg2.model_type == 'word'
    assert cfg2.epochs == 300
    assert cfg2.scan_patch_size == (12, 18)
    assert cfg2.guide_weight == 5.0
    assert cfg2.motor_enabled is True
