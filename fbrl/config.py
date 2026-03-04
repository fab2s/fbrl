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
    void_weight: float = 0.0             # void repulsion (reads)
    scan_void_weight: float = 0.0       # void repulsion (scans)

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

    # Grouped read (word only)
    read_anchor_scan_indices: Optional[tuple] = None  # e.g. (1,2,3,4)
    n_read_per_group: Optional[int] = None            # glimpses per group
    learnable_scan_x: bool = False                    # learn optimal scan x positions
    interleaved: bool = False                        # interleaved scan-read (1 scan + N reads per position)

    # Motor trace
    motor_enabled: bool = False
    n_trajectory_points: int = 48
    render_sigma: float = 1.5
    traj_weight: float = 1.0
    traj_scaffold_ratio: float = 0.67
    traj_scaffold_floor: float = 0.1
    rr_cls_weight: float = 1.0
    trajectory_data_dir: Optional[str] = None
    canonical_font: str = 'dejavu-sans'

    # Enhanced motor losses (v2)
    case_filter: Optional[str] = None          # 'upper', 'lower', or None (both)
    motor_target_case: Optional[str] = None    # 'lower'/'upper' — force motor trace case (None = match input)
    latent_match_weight: float = 0.0           # 0 = disabled (backward compat)
    frozen_rr_weight: float = 0.0              # 0 = disabled
    render_match_weight: float = 0.0           # 0 = disabled
    rr_cls_warmup_ratio: float = 0.0           # ratio of epochs for inverse-cosine ramp 0→1
    motor_coupling_lr: float = 0.0001          # LR for motor→encoder coupling (0 = no coupling)

    # Constrained motor (v5)
    motor_n_strokes: int = 4
    motor_points_per_stroke: int = 20
    motor_hidden_dim: int = 256
    ink_weight: float = 1.0
    ink_blur_sigma: float = 3.0               # final blur sigma (or fixed if no anneal)
    ink_blur_sigma_start: float = 0.0          # starting blur sigma (0 = no anneal, use ink_blur_sigma)
    ink_void_weight: float = 0.5
    vision_checkpoint: Optional[str] = None   # frozen backbone checkpoint path

    # Three-phase reading
    n_meta_glimpses: int = 4
    n_sub_per_meta: int = 3
    n_read_per_sub: int = 3
    meta_patch_pixels: tuple = (32, 96)
    meta_blur_sigma: float = 6.0
    sub_patch_pixels: tuple = (20, 28)
    sub_blur_sigma: float = 2.0
    meta_guide_weight: float = 8.0
    sub_guide_weight: float = 8.0
    read_void_weight: float = 0.5
    meta_content_weight: float = 0.5
    sub_content_weight: float = 0.5

    # Differential cosine decay (per-head T_max ratios, multiply epochs)
    tmax_attn_ratio: float = 1.0
    tmax_cls_ratio: float = 1.0
    tmax_recon_ratio: float = 1.0
    tmax_motor_ratio: float = 1.0
    lr_floor: float = 0.0               # eta_min for all schedulers

    # Data filtering
    train_fonts: Optional[str] = None    # None = all fonts; comma-separated names to filter

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

    # Convert list -> tuple for tuple fields
    _tuple_fields = ['scan_patch_size', 'read_anchor_scan_indices',
                     'meta_patch_pixels', 'sub_patch_pixels']
    for tf in _tuple_fields:
        if tf in data and isinstance(data[tf], list):
            data[tf] = tuple(data[tf])

    # Apply CLI overrides
    if cli_overrides:
        for k, v in cli_overrides.items():
            if v is not None:
                data[k] = v

    # Convert from override too
    for tf in _tuple_fields:
        if tf in data and isinstance(data[tf], list):
            data[tf] = tuple(data[tf])

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
    # Ensure tuples are lists for YAML/JSON compat
    for tf in ('scan_patch_size', 'read_anchor_scan_indices',
               'meta_patch_pixels', 'sub_patch_pixels'):
        if isinstance(d.get(tf), tuple):
            d[tf] = list(d[tf])
    return d


def config_from_dict(d):
    """Restore config from checkpoint dict."""
    for tf in ('scan_patch_size', 'read_anchor_scan_indices',
               'meta_patch_pixels', 'sub_patch_pixels'):
        if tf in d and isinstance(d[tf], list):
            d[tf] = tuple(d[tf])
    valid = {f.name for f in fields(ExperimentConfig)}
    filtered = {k: v for k, v in d.items() if k in valid}
    return ExperimentConfig(**filtered)
