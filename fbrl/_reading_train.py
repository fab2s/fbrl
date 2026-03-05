"""Three-phase reading training functions — imported by train.py."""
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import time

from fbrl import _resolve_device
from fbrl.data import CountingDataset
from fbrl.model import ReadingModel
from fbrl.losses import (attention_content_loss, fixation_diversity_loss,
                          fixation_hit_rate, void_repulsion)
from fbrl.training_utils import (LossTracker, TrainingLogger, save_checkpoint,
                                  plot_training_metrics, format_eta, save_run_info)


def assign_groups_spatial(barycenter_anchors, char_positions, char_labels, n_groups):
    """Assign ground-truth letters to read groups by spatial proximity.

    For each letter (non-NaN position), find the closest barycenter anchor x
    and assign that letter's label to that group. Unassigned groups get void (26).

    Uses detached anchor positions for assignment (no gradient through matching).

    Args:
        barycenter_anchors: list of n_groups × (B, 2) tensors
        char_positions: (B, max_count) float tensor, NaN-padded
        char_labels: (B, max_count) long tensor, 26-padded (void)
        n_groups: int

    Returns:
        group_targets: (B, n_groups) long tensor of letter indices (0-25 or 26=void)
    """
    B = char_positions.shape[0]
    device = char_positions.device

    # Stack anchor x-coords: (n_groups, B)
    anchor_xs = torch.stack([a[:, 0].detach() for a in barycenter_anchors], dim=0)  # (n_groups, B)

    # Initialize all groups as void
    group_targets = torch.full((B, n_groups), 26, dtype=torch.long, device=device)

    # For each letter in each batch sample, assign to closest anchor
    max_count = char_positions.shape[1]
    for li in range(max_count):
        pos = char_positions[:, li]   # (B,)
        label = char_labels[:, li]    # (B,)
        valid = ~torch.isnan(pos)     # (B,)

        if not valid.any():
            continue

        # Distance from this letter to each anchor: (n_groups, B)
        dists = (anchor_xs - pos.unsqueeze(0)).abs()  # (n_groups, B)
        # Find closest anchor per sample
        closest = dists.argmin(dim=0)  # (B,)

        # Assign label to closest group (only for valid samples)
        for b_idx in range(B):
            if valid[b_idx]:
                g = closest[b_idx].item()
                # Only assign if this group is still void (first-come for ties)
                if group_targets[b_idx, g] == 26:
                    group_targets[b_idx, g] = label[b_idx]

    return group_targets


def train_reading_model(cfg):
    """Train a ReadingModel from an ExperimentConfig.

    Three-phase hierarchical attention: meta-scan (peripheral) ->
    sub-scan (parafoveal) -> isolated read groups (foveal).
    Letter classification (27-class: a-z + void) on each read group.
    Count derived from non-void predictions.
    """
    n_meta = cfg.n_meta_glimpses
    n_sub_per_meta = cfg.n_sub_per_meta
    n_read_per_sub = cfg.n_read_per_sub
    n_read_per_group = cfg.n_read_glimpses_per_group
    n_letter_classes = cfg.n_letter_classes
    meta_patch_pixels = cfg.meta_patch_pixels
    meta_blur_sigma = cfg.meta_blur_sigma
    sub_patch_pixels = cfg.sub_patch_pixels
    sub_blur_sigma = cfg.sub_blur_sigma
    read_patch_size = cfg.read_patch_size
    latent_dim = cfg.latent_dim
    n_scales = cfg.n_scales
    meta_guide_weight = cfg.meta_guide_weight
    sub_guide_weight = cfg.sub_guide_weight
    read_void_weight = cfg.read_void_weight
    meta_content_weight = cfg.meta_content_weight
    sub_content_weight = cfg.sub_content_weight
    meta_x_drift = cfg.meta_x_drift
    sub_x_drift = cfg.sub_x_drift
    blur_sigma_ratio = cfg.blur_sigma_ratio
    diversity_weight = cfg.diversity_weight
    diversity_sigma = cfg.diversity_sigma
    scan_vy = cfg.scan_vy
    batch_size = cfg.batch_size
    epochs = cfg.epochs
    resume = cfg.resume
    save_dir = cfg.save_dir
    data_dir = cfg.data_dir
    checkpoint_interval = cfg.checkpoint_interval

    device = _resolve_device(cfg.device)
    n_discovery = 1 + n_meta + n_meta * n_sub_per_meta
    n_reads = n_meta * n_read_per_group
    n_total = n_discovery + n_reads
    print(f"Reading training on: {device}")
    print(f"Three-phase: meta={n_meta} ({meta_patch_pixels}, blur={meta_blur_sigma}) + "
          f"sub={n_sub_per_meta}/meta ({sub_patch_pixels}, blur={sub_blur_sigma}) + "
          f"read={n_read_per_group}/group × {n_meta} groups = {n_total} total steps")
    print(f"Letter classes: {n_letter_classes} (26 letters + void)")
    print(f"Attention: meta_guide={meta_guide_weight}  sub_guide={sub_guide_weight}  "
          f"read_void={read_void_weight}  blur_sigma_ratio={blur_sigma_ratio}  "
          f"diversity={diversity_weight}  diversity_sigma={diversity_sigma}  "
          f"scan_vy={scan_vy}  batch_size={batch_size}")

    os.makedirs(save_dir, exist_ok=True)
    save_run_info(save_dir, cfg)

    dataset = CountingDataset(data_dir, train_fonts=cfg.train_fonts)
    use_cuda = device.type == 'cuda'
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                             pin_memory=use_cuda)

    model = ReadingModel(
        n_meta=n_meta, n_sub_per_meta=n_sub_per_meta,
        n_read_per_sub=n_read_per_sub,
        meta_patch_pixels=meta_patch_pixels, meta_blur_sigma=meta_blur_sigma,
        sub_patch_pixels=sub_patch_pixels, sub_blur_sigma=sub_blur_sigma,
        read_patch_size=read_patch_size, latent_dim=latent_dim,
        n_scales=n_scales, max_count=3,
        n_letter_classes=n_letter_classes,
        n_read_glimpses_per_group=n_read_per_group,
        meta_x_drift=meta_x_drift, sub_x_drift=sub_x_drift,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"ReadingModel: {n_params:,} parameters")

    loss_names = ['letter_ce', 'meta_attn', 'sub_attn', 'read_void',
                  'meta_content', 'sub_content', 'div',
                  'letter_acc', 'void_acc', 'count_acc',
                  'hit_rate', 'hit_intensity']
    tracker = LossTracker(loss_names)
    start_epoch = 0

    if resume:
        checkpoint = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model'], strict=False)
        start_epoch = checkpoint['epoch'] + 1
        if 'losses' in checkpoint:
            tracker.restore_history(checkpoint['losses'])
        print(f"Resumed from epoch {start_epoch}")

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    bce = torch.nn.BCEWithLogitsLoss()

    end_epoch = start_epoch + epochs
    train_start = time.time()

    header = (f"{'epoch':>5s}  {'ltr_ce':>7s}  {'ltr_a':>6s}  {'void_a':>6s}  "
              f"{'cnt_a':>6s}  {'m_att':>6s}  {'s_att':>6s}  "
              f"{'r_void':>6s}  {'m_cnt':>6s}  {'s_cnt':>6s}  {'div':>6s}  "
              f"{'hr':>6s}  {'hi':>6s}  {'lr':>8s}  time")
    logger = TrainingLogger(save_dir, header, start_epoch)

    for epoch in range(start_epoch, end_epoch):
        epoch_start = time.time()
        tracker.reset_epoch()
        total_letter_correct = 0
        total_letter_total = 0
        total_void_correct = 0
        total_void_total = 0
        total_count_correct = 0
        total_samples = 0

        for batch in dataloader:
            img, clean, counts_batch, _letters, _fonts, char_pos, char_lab = batch
            img = img.to(device)
            clean = clean.to(device)
            char_pos = char_pos.to(device)
            char_lab = char_lab.to(device)
            B = img.shape[0]

            group_logits, enc = model(img)  # (B, n_meta, 27)

            # Assign ground-truth labels to groups by spatial proximity
            group_targets = assign_groups_spatial(
                enc.barycenter_anchors, char_pos, char_lab, n_meta)  # (B, n_meta)

            # Dynamic void class weight: inverse frequency per batch
            flat_targets = group_targets.view(-1)
            n_void = (flat_targets == 26).sum().float()
            n_nonvoid = (flat_targets != 26).sum().float()
            total_fg = n_void + n_nonvoid
            if n_void > 0 and n_nonvoid > 0:
                class_weights = torch.ones(n_letter_classes, device=device)
                class_weights[:26] = total_fg / (2 * n_nonvoid)
                class_weights[26] = total_fg / (2 * n_void)
            else:
                class_weights = None

            # Primary: letter CE
            letter_ce = F.cross_entropy(
                group_logits.view(-1, n_letter_classes),
                flat_targets,
                weight=class_weights,
            )

            # Phase-specific location lists for auxiliary losses
            init_loc = enc.locations[0]
            meta_locs = [init_loc] + enc.meta_positions
            sub_locs = [init_loc] + enc.sub_positions

            # Read locations (from enc, added by ReadingModel)
            read_locs = [enc.locations[i] for i, tag in enumerate(enc.phase_tags) if tag == 'read']

            # Attention guides (meta + sub toward ink)
            meta_attn = meta_guide_weight * attention_content_loss(
                clean, meta_locs, blur_sigma_ratio=blur_sigma_ratio)
            sub_attn = sub_guide_weight * attention_content_loss(
                clean, sub_locs, blur_sigma_ratio=blur_sigma_ratio)

            # Read void repulsion
            read_void_loss = torch.tensor(0.0, device=device)
            if read_void_weight > 0 and len(read_locs) > 0:
                read_void_loss = read_void_weight * void_repulsion(
                    clean, read_locs, read_patch_size, read_patch_size)

            # Diversity (meta + sub)
            meta_div = fixation_diversity_loss(meta_locs, sigma=diversity_sigma, vy=scan_vy)
            sub_div = fixation_diversity_loss(sub_locs, sigma=diversity_sigma, vy=scan_vy)
            div_loss = meta_div + sub_div

            # Content BCE (meta)
            meta_content_loss = torch.tensor(0.0, device=device)
            if meta_content_weight > 0 and len(enc.meta_content_logits) > 0:
                for loc, logit in zip(enc.meta_positions, enc.meta_content_logits):
                    grid = loc.view(B, 1, 1, 2)
                    sampled = F.grid_sample(clean, grid, align_corners=True,
                                            padding_mode='zeros')
                    label = (sampled.view(-1, 1) > 0.1).float()
                    meta_content_loss = meta_content_loss + bce(logit, label)
                meta_content_loss = meta_content_loss / len(enc.meta_content_logits)

            # Content BCE (sub)
            sub_content_loss = torch.tensor(0.0, device=device)
            if sub_content_weight > 0 and len(enc.sub_content_logits) > 0:
                for loc, logit in zip(enc.sub_positions, enc.sub_content_logits):
                    grid = loc.view(B, 1, 1, 2)
                    sampled = F.grid_sample(clean, grid, align_corners=True,
                                            padding_mode='zeros')
                    label = (sampled.view(-1, 1) > 0.1).float()
                    sub_content_loss = sub_content_loss + bce(logit, label)
                sub_content_loss = sub_content_loss / len(enc.sub_content_logits)

            total_loss = (letter_ce
                          + meta_attn + sub_attn + read_void_loss
                          + diversity_weight * div_loss
                          + meta_content_weight * meta_content_loss
                          + sub_content_weight * sub_content_loss)

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            # Metrics
            with torch.no_grad():
                preds = group_logits.argmax(2)  # (B, n_meta)
                targets = group_targets

                # Letter accuracy (non-void groups only)
                nonvoid_mask = targets != 26
                if nonvoid_mask.any():
                    total_letter_correct += (preds[nonvoid_mask] == targets[nonvoid_mask]).sum().item()
                    total_letter_total += nonvoid_mask.sum().item()

                # Void accuracy
                void_mask = targets == 26
                if void_mask.any():
                    total_void_correct += (preds[void_mask] == 26).sum().item()
                    total_void_total += void_mask.sum().item()

                # Derived count accuracy
                pred_counts = (preds != 26).sum(dim=1)  # (B,)
                true_counts = counts_batch.to(device) if torch.is_tensor(counts_batch) else torch.tensor(counts_batch, device=device)
                total_count_correct += (pred_counts == true_counts).sum().item()
                total_samples += B

                hr, hi = fixation_hit_rate(clean, enc.locations)

            tracker.update(letter_ce=letter_ce, meta_attn=meta_attn,
                           sub_attn=sub_attn, read_void=read_void_loss,
                           meta_content=meta_content_loss, sub_content=sub_content_loss,
                           div=div_loss,
                           letter_acc=0, void_acc=0, count_acc=0,
                           hit_rate=hr, hit_intensity=hi)

        avgs = tracker.end_epoch()
        letter_acc = total_letter_correct / max(total_letter_total, 1)
        void_acc = total_void_correct / max(total_void_total, 1)
        count_acc = total_count_correct / max(total_samples, 1)
        avgs['letter_acc'] = letter_acc
        avgs['void_acc'] = void_acc
        avgs['count_acc'] = count_acc
        tracker.history['letter_acc'][-1] = letter_acc
        tracker.history['void_acc'][-1] = void_acc
        tracker.history['count_acc'][-1] = count_acc

        epoch_time = time.time() - epoch_start
        elapsed = time.time() - train_start
        done = epoch - start_epoch + 1
        eta = format_eta(elapsed, done, epochs - done)
        current_lr = scheduler.get_last_lr()[0]

        print(f"Epoch {epoch+1}/{end_epoch}: "
              f"LtrCE {avgs['letter_ce']:.4f}  LtrA {letter_acc:.0%}  "
              f"VoidA {void_acc:.0%}  CntA {count_acc:.0%}  "
              f"MetaA {avgs['meta_attn']:.4f}  SubA {avgs['sub_attn']:.4f}  "
              f"Void {avgs['read_void']:.4f}  Div {avgs['div']:.4f}  "
              f"Hit {avgs['hit_rate']:.0%}  "
              f"lr {current_lr:.6f}  [{epoch_time:.1f}s  ETA {eta}]")

        logger.write_line(
            f"{epoch+1:>5d}  {avgs['letter_ce']:>7.4f}  {letter_acc:>6.4f}  {void_acc:>6.4f}  "
            f"{count_acc:>6.4f}  {avgs['meta_attn']:>6.4f}  {avgs['sub_attn']:>6.4f}  "
            f"{avgs['read_void']:>6.4f}  {avgs['meta_content']:>6.4f}  {avgs['sub_content']:>6.4f}  "
            f"{avgs['div']:>6.4f}  "
            f"{avgs['hit_rate']:>6.4f}  {avgs['hit_intensity']:>6.4f}  "
            f"{current_lr:>8.6f}  {epoch_time:.1f}s")
        scheduler.step()

        if (epoch + 1 - start_epoch) % checkpoint_interval == 0:
            _save_reading_checkpoint(model, epoch, save_dir, cfg, tracker,
                                     n_meta, n_sub_per_meta, n_read_per_sub,
                                     meta_patch_pixels, meta_blur_sigma,
                                     sub_patch_pixels, sub_blur_sigma,
                                     read_patch_size, latent_dim, n_scales,
                                     n_letter_classes, n_read_per_group,
                                     meta_x_drift, sub_x_drift,
                                     name=f'checkpoint_epoch_{epoch+1}.pth')

    logger.close()

    _save_reading_checkpoint(model, end_epoch - 1, save_dir, cfg, tracker,
                             n_meta, n_sub_per_meta, n_read_per_sub,
                             meta_patch_pixels, meta_blur_sigma,
                             sub_patch_pixels, sub_blur_sigma,
                             read_patch_size, latent_dim, n_scales,
                             n_letter_classes, n_read_per_group,
                             meta_x_drift, sub_x_drift,
                             name='model_final.pth')

    specs = [
        {'keys': ['letter_ce'], 'labels': ['Letter CE'], 'colors': ['tab:red'],
         'title': 'Letter Classification (27-class)', 'ylabel': 'Cross-Entropy',
         'hlines': [(np.log(27), f'Random ({np.log(27):.2f})', 'gray')]},
        {'keys': ['letter_acc', 'void_acc', 'count_acc'],
         'labels': ['Letter', 'Void', 'Count'],
         'colors': ['tab:blue', 'tab:green', 'tab:orange'],
         'styles': ['-', '--', ':'],
         'title': 'Accuracy', 'ylabel': 'Accuracy', 'ylim': (0, 1),
         'hlines': [(1/27, 'Random letter (3.7%)', 'gray')]},
        {'keys': ['meta_attn', 'sub_attn'], 'labels': ['Meta guide', 'Sub guide'],
         'colors': ['tab:green', 'tab:olive'], 'styles': ['-', '--'],
         'title': 'Attention guide (lower = fixations on content)', 'ylabel': 'Loss'},
        {'keys': ['read_void'], 'labels': ['Read void'], 'colors': ['tab:brown'],
         'title': 'Read void repulsion', 'ylabel': 'Loss'},
        {'keys': ['div'], 'labels': ['Diversity'], 'colors': ['tab:orange'],
         'title': 'Fixation diversity (lower = more spread)', 'ylabel': 'Repulsion'},
        {'keys': ['meta_content', 'sub_content'],
         'labels': ['Meta content', 'Sub content'],
         'colors': ['tab:cyan', 'tab:teal'], 'styles': ['-', '--'],
         'title': 'Content detection (meta + sub)', 'ylabel': 'BCE'},
        {'keys': ['hit_rate', 'hit_intensity'],
         'labels': ['Hit rate', 'Intensity'],
         'colors': ['tab:purple', 'tab:purple'], 'styles': ['-', '--'],
         'title': 'Fixation hit rate (on letter pixels)',
         'ylabel': 'Rate / Intensity', 'ylim': (0, 1)},
    ]
    plot_training_metrics(tracker, os.path.join(save_dir, 'training_metrics.png'), specs)

    total_time = time.time() - train_start
    total_min, total_s = divmod(int(total_time), 60)
    print(f"Training complete in {total_min}m{total_s:02d}s. "
          f"Model and graph saved in {save_dir}")


def _save_reading_checkpoint(model, epoch, save_dir, cfg, tracker,
                             n_meta, n_sub_per_meta, n_read_per_sub,
                             meta_patch_pixels, meta_blur_sigma,
                             sub_patch_pixels, sub_blur_sigma,
                             read_patch_size, latent_dim, n_scales,
                             n_letter_classes, n_read_per_group,
                             meta_x_drift, sub_x_drift,
                             name='model_final.pth'):
    save_checkpoint(model, epoch,
                    os.path.join(save_dir, name),
                    cfg=cfg, losses_dict=tracker.get_history_dict(),
                    extra={
                        'model_type': 'reading',
                        'n_meta': n_meta,
                        'n_sub_per_meta': n_sub_per_meta,
                        'n_read_per_sub': n_read_per_sub,
                        'meta_patch_pixels': list(meta_patch_pixels),
                        'meta_blur_sigma': meta_blur_sigma,
                        'sub_patch_pixels': list(sub_patch_pixels),
                        'sub_blur_sigma': sub_blur_sigma,
                        'read_patch_size': read_patch_size,
                        'latent_dim': latent_dim,
                        'n_scales': n_scales,
                        'max_count': 3,
                        'n_letter_classes': n_letter_classes,
                        'n_read_glimpses_per_group': n_read_per_group,
                        'meta_x_drift': meta_x_drift,
                        'sub_x_drift': sub_x_drift,
                    })
