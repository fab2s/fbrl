"""Experiment configuration: flat dataclass + YAML load/save."""
import yaml
from dataclasses import dataclass, field, fields, asdict
from typing import Optional


@dataclass
class ExperimentConfig:
    # Model type
    model_type: str = 'letter'  # 'letter' | 'bigram' | 'word'

    # Model geometry
    n_positions: int = 1
    n_scan_glimpses: int = 0
    n_read_glimpses: int = 10
    scan_patch_size: tuple = (12, 18)
    read_patch_size: int = 12
    latent_dim: int = 256
    n_scales: int = 1

    # Training
    epochs: int = 200
    batch_size: int = 32
    checkpoint_interval: int = 10
    device: str = 'auto'
    multi_head: bool = False
    amp: bool = False

    # Attention & diversity
    guide_weight: float = 8.0
    scan_guide_weight: Optional[float] = None  # None = same as guide_weight
    blur_sigma_ratio: float = 0.16
    diversity_weight: float = 1.0
    diversity_sigma: float = 0.1
    scan_vy: float = 0.3
    read_vy: float = 1.5
    content_weight: float = 0.5
    edge_weight: float = 0.0

    # Scaffold & transfer
    scaffold_ratio: float = 0.67
    scaffold_floor: float = 0.0
    scaffold_epochs: Optional[int] = None  # None = compute from ratio
    transfer: Optional[str] = None

    # Letter-specific
    recode_weight: float = 1.0
    diversity_vy: float = 1.0  # single-letter diversity vy

    # Bigram-specific
    mask_weight: float = 0.5

    # Isolation (word only)
    isolation_weight: float = 0.5
    isolation_data_dir: Optional[str] = None
    isolation_random_prob: float = 0.0

    # Motor trace
    motor_enabled: bool = False
    n_trajectory_points: int = 32
    render_sigma: float = 1.5
    traj_weight: float = 1.0
    traj_scaffold_ratio: float = 0.67
    traj_scaffold_floor: float = 0.1
    rr_cls_weight: float = 1.0
    trajectory_data_dir: Optional[str] = None
    canonical_font: str = 'dejavu-sans'

    # Paths
    data_dir: str = ''
    save_dir: str = ''
    resume: Optional[str] = None


def load_config(yaml_path, cli_overrides=None):
    """Load config from YAML, apply CLI overrides (non-None values only).

    Args:
        yaml_path: Path to YAML config file.
        cli_overrides: dict of field_name -> value. None values are skipped.
    Returns:
        ExperimentConfig instance.
    """
    with open(yaml_path) as f:
        data = yaml.safe_load(f) or {}

    # Convert scan_patch_size list -> tuple
    if 'scan_patch_size' in data and isinstance(data['scan_patch_size'], list):
        data['scan_patch_size'] = tuple(data['scan_patch_size'])

    # Apply CLI overrides
    if cli_overrides:
        for k, v in cli_overrides.items():
            if v is not None:
                data[k] = v

    # Convert scan_patch_size from override too
    if 'scan_patch_size' in data and isinstance(data['scan_patch_size'], list):
        data['scan_patch_size'] = tuple(data['scan_patch_size'])

    # Filter to valid fields only
    valid = {f.name for f in fields(ExperimentConfig)}
    filtered = {k: v for k, v in data.items() if k in valid}

    cfg = ExperimentConfig(**filtered)

    # Resolve scan_guide_weight default
    if cfg.scan_guide_weight is None:
        cfg.scan_guide_weight = cfg.guide_weight

    # Resolve scaffold_epochs from ratio if not set
    if cfg.scaffold_epochs is None:
        cfg.scaffold_epochs = int(cfg.scaffold_ratio * cfg.epochs)

    return cfg


def config_to_dict(cfg):
    """Serialize config for checkpoint storage."""
    d = asdict(cfg)
    # Ensure scan_patch_size is a list for YAML/JSON compat
    if isinstance(d.get('scan_patch_size'), tuple):
        d['scan_patch_size'] = list(d['scan_patch_size'])
    return d


def config_from_dict(d):
    """Restore config from checkpoint dict."""
    if 'scan_patch_size' in d and isinstance(d['scan_patch_size'], list):
        d['scan_patch_size'] = tuple(d['scan_patch_size'])
    valid = {f.name for f in fields(ExperimentConfig)}
    filtered = {k: v for k, v in d.items() if k in valid}
    return ExperimentConfig(**filtered)
