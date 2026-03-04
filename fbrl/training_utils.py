"""Shared training infrastructure: loss tracking, logging, checkpoints, transfer, plotting."""
import torch
import gzip
import io
import os
import subprocess
import sys
import time
import yaml
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from fbrl.config import config_to_dict


def _build_make_command(cfg):
    """Reconstruct the make command equivalent from config and sys.argv."""
    # subcommand -> (make target, config var)
    _TARGETS = {
        'train': ('train', 'CONFIG'),
        'train_bigrams': ('train-bigrams', 'BIGRAM_CONFIG'),
        'train_words': ('train-words', 'WORD_CONFIG'),
        'train_motor': ('train-motor', 'MOTOR_CONFIG'),
        'train_counting': ('train-counting', 'COUNTING_CONFIG'),
    }
    # Find subcommand in argv
    subcmd = None
    for arg in sys.argv[1:]:
        if arg in _TARGETS:
            subcmd = arg
            break
    if subcmd is None:
        return None

    target, config_var = _TARGETS[subcmd]

    # Extract CLI overrides (non-default values passed via --flag)
    parts = [f'make {target}']
    argv = sys.argv[1:]
    # Map CLI flags to Makefile vars
    _FLAG_MAP = {
        '--device': 'DEVICE', '--epochs': 'EPOCHS', '--batch_size': 'BATCH',
        '--transfer': 'TRANSFER', '--checkpoint_interval': 'CKPT',
    }
    # Check if a non-default config was used
    config_path = None
    i = 0
    while i < len(argv):
        if argv[i] == '--config' and i + 1 < len(argv):
            config_path = argv[i + 1]
            i += 2
        elif argv[i] in _FLAG_MAP and i + 1 < len(argv):
            parts.append(f'{_FLAG_MAP[argv[i]]}={argv[i + 1]}')
            i += 2
        else:
            i += 1

    # Only include config var if non-default path was used
    defaults = {
        'CONFIG': 'configs/letter.yaml', 'BIGRAM_CONFIG': 'configs/bigram.yaml',
        'WORD_CONFIG': 'configs/word.yaml', 'MOTOR_CONFIG': 'configs/letter_motor.yaml',
        'COUNTING_CONFIG': 'configs/counting.yaml',
    }
    if config_path and config_path != defaults.get(config_var):
        parts.insert(1, f'{config_var}={config_path}')

    return ' '.join(parts)


def save_run_info(save_dir, cfg):
    """Write config.yaml and info.txt into save_dir at training start.

    config.yaml: resolved config (includes CLI overrides).
    info.txt: git hash, date, make command, CLI command, transfer source.
    """
    # config.yaml — resolved (not the source file)
    cfg_dict = config_to_dict(cfg)
    with open(os.path.join(save_dir, 'config.yaml'), 'w') as f:
        yaml.dump(cfg_dict, f, default_flow_style=False, sort_keys=False)

    # info.txt
    try:
        git_hash = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            stderr=subprocess.DEVNULL).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        git_hash = 'n/a'

    lines = [
        f'git: {git_hash}',
        f'date: {datetime.now().astimezone().isoformat()}',
    ]
    make_cmd = _build_make_command(cfg)
    if make_cmd:
        lines.append(f'make: {make_cmd}')
    lines.append(f'cli: {" ".join(sys.argv)}')
    if cfg.transfer:
        lines.append(f'transfer: {cfg.transfer}')
    if cfg.resume:
        lines.append(f'resume: {cfg.resume}')

    with open(os.path.join(save_dir, 'info.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')


class LossTracker:
    """Tracks per-epoch loss averages across named loss terms."""

    def __init__(self, loss_names):
        self.names = list(loss_names)
        self.history = {name: [] for name in self.names}
        self._epoch_sums = {}
        self._epoch_count = 0

    def reset_epoch(self):
        self._epoch_sums = {name: 0.0 for name in self.names}
        self._epoch_count = 0

    def update(self, **losses):
        """Add a batch of losses. Keys must be subset of loss_names."""
        for name, val in losses.items():
            if name in self._epoch_sums:
                self._epoch_sums[name] += (val.item() if torch.is_tensor(val) else val)
        self._epoch_count += 1

    def end_epoch(self):
        """Compute averages, append to history, return dict of averages."""
        n = max(self._epoch_count, 1)
        avgs = {}
        for name in self.names:
            avg = self._epoch_sums.get(name, 0.0) / n
            self.history[name].append(avg)
            avgs[name] = avg
        return avgs

    def get_history_dict(self):
        """Return history dict for checkpoint storage."""
        return dict(self.history)

    def restore_history(self, hist_dict):
        """Restore history from checkpoint."""
        for name in self.names:
            if name in hist_dict:
                self.history[name] = hist_dict[name]


class TrainingLogger:
    """Manages the per-epoch training log file."""

    def __init__(self, save_dir, header, start_epoch=0):
        self.log_path = os.path.join(save_dir, 'training.log')
        if start_epoch == 0 and os.path.exists(self.log_path):
            ts = datetime.fromtimestamp(
                os.path.getmtime(self.log_path)
            ).strftime('%Y%m%d_%H%M%S')
            os.rename(self.log_path, os.path.join(save_dir, f'training_{ts}.log'))
        self._file = open(self.log_path, 'a')
        if start_epoch == 0:
            self._file.write(header + '\n')
            self._file.write('-' * len(header) + '\n')
            self._file.flush()

    def write_line(self, line):
        self._file.write(line + '\n')
        self._file.flush()

    def close(self):
        self._file.close()
        print(f"Training log saved to {self.log_path}")


def save_checkpoint(model, epoch, path, cfg=None, losses_dict=None, extra=None):
    """Save a model checkpoint.

    Args:
        model: nn.Module
        epoch: current epoch (0-indexed)
        path: save path
        cfg: ExperimentConfig (optional, stored as dict)
        losses_dict: dict of loss histories
        extra: dict of additional fields to include
    """
    ckpt = {
        'epoch': epoch,
        'model': {k: v.cpu() for k, v in model.state_dict().items()},
    }
    if cfg is not None:
        ckpt['config'] = config_to_dict(cfg)
        ckpt['model_type'] = cfg.model_type
    if losses_dict is not None:
        ckpt['losses'] = losses_dict
    if extra is not None:
        ckpt.update(extra)
    torch.save(ckpt, path)


def load_checkpoint(path, device):
    """Load checkpoint, handling gzip transparently."""
    if path.endswith('.gz'):
        with gzip.open(path, 'rb') as f:
            return torch.load(io.BytesIO(f.read()), map_location=device,
                              weights_only=False)
    return torch.load(path, map_location=device, weights_only=False)


def apply_transfer(model, source_path, key_mappings, device,
                   broadcast_keys=None, freeze_keys=None):
    """Apply transfer learning weights with key remapping.

    Args:
        model: target model
        source_path: path to source checkpoint (.pth or .pth.gz)
        key_mappings: list of (src_prefix, dst_prefix) tuples for remapping
        device: torch device
        broadcast_keys: list of (src_key_suffix, [dst_key_patterns]) for broadcasting
                        e.g. ('letter_classifier.weight', ['classifiers.{i}.weight'])
        freeze_keys: list of key prefixes to freeze after transfer
    Returns:
        number of transferred tensors
    """
    ckpt = load_checkpoint(source_path, device)
    src = ckpt['model'] if 'model' in ckpt else ckpt
    dst = model.state_dict()
    n_transferred = 0

    n_skipped = 0
    for key in src:
        for src_prefix, dst_prefix in key_mappings:
            if key.startswith(src_prefix):
                new_key = key.replace(src_prefix, dst_prefix, 1)
                if new_key in dst:
                    if dst[new_key].shape == src[key].shape:
                        dst[new_key] = src[key].float()
                        n_transferred += 1
                    else:
                        n_skipped += 1
                break

    if broadcast_keys:
        for src_suffix, dst_patterns in broadcast_keys:
            if src_suffix in src:
                for pattern in dst_patterns:
                    dk = pattern
                    if dk in dst:
                        if dst[dk].shape == src[src_suffix].shape:
                            dst[dk] = src[src_suffix].float()
                            n_transferred += 1
                        else:
                            n_skipped += 1

    model.load_state_dict(dst)
    skipped_str = f" ({n_skipped} skipped: shape mismatch)" if n_skipped else ""
    print(f"Transfer: {n_transferred} tensors from {source_path}{skipped_str}")

    if freeze_keys:
        for name, p in model.named_parameters():
            for fk in freeze_keys:
                if name.startswith(fk):
                    p.requires_grad = False
                    break

    return n_transferred


def plot_training_metrics(tracker, save_path, subplot_specs):
    """Plot training metrics from tracker history.

    Args:
        tracker: LossTracker with history
        save_path: path to save PNG
        subplot_specs: list of dicts, each with:
            - 'keys': list of history key names to plot
            - 'labels': list of labels (same order as keys)
            - 'colors': list of colors
            - 'styles': list of linestyles (optional, default '-')
            - 'title': subplot title
            - 'ylabel': y-axis label
            - 'hlines': optional list of (y, label, color) horizontal reference lines
            - 'ylim': optional (min, max) for y-axis
    """
    n_plots = len(subplot_specs)
    if n_plots == 0:
        return

    # Determine epoch range from first non-empty key
    n_epochs = 0
    for spec in subplot_specs:
        for key in spec['keys']:
            if key in tracker.history and tracker.history[key]:
                n_epochs = max(n_epochs, len(tracker.history[key]))
    if n_epochs == 0:
        return

    epochs_x = range(1, n_epochs + 1)
    fig, axes = plt.subplots(n_plots, 1, figsize=(8, 2.3 * n_plots), sharex=True)
    if n_plots == 1:
        axes = [axes]

    for ax, spec in zip(axes, subplot_specs):
        styles = spec.get('styles', ['-'] * len(spec['keys']))
        for key, label, color, style in zip(spec['keys'], spec['labels'],
                                            spec['colors'], styles):
            data = tracker.history.get(key, [])
            if data:
                # Align x-axis if history is shorter than total
                x = range(n_epochs - len(data) + 1, n_epochs + 1)
                ax.plot(x, data, label=label, color=color, linestyle=style)

        for y, label, color in spec.get('hlines', []):
            ax.axhline(y=y, color=color, linestyle='--', label=label)

        ax.set_ylabel(spec['ylabel'])
        ax.legend(loc='upper right')
        ax.set_title(spec['title'])
        if 'ylim' in spec:
            ax.set_ylim(*spec['ylim'])

    axes[-1].set_xlabel('Epoch')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def format_eta(elapsed, done, remaining_ep):
    """Format ETA string from timing info."""
    if done == 0:
        return "?"
    eta_sec = remaining_ep * (elapsed / done)
    eta_min, eta_s = divmod(int(eta_sec), 60)
    return f"{eta_min}m{eta_s:02d}s"
