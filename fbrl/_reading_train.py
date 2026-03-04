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


def train_reading_model(cfg):
    """Train a ReadingModel from an ExperimentConfig.

    Three-phase hierarchical attention: meta-scan (peripheral) ->
    sub-scan (parafoveal) -> read (foveal). Count classification
    on final latent (3-class, 1-3 letters on 192x128 canvas).
    Reuses counting data.
    """
    n_meta = cfg.n_meta_glimpses
    n_sub_per_meta = cfg.n_sub_per_meta
    n_read_per_sub = cfg.n_read_per_sub
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
    n_total = 1 + n_meta + n_meta * n_sub_per_meta + n_meta * n_sub_per_meta * n_read_per_sub
    print(f"Reading training on: {device}")
    print(f"Three-phase: meta={n_meta} ({meta_patch_pixels}, blur={meta_blur_sigma}) + "
          f"sub={n_sub_per_meta}/meta ({sub_patch_pixels}, blur={sub_blur_sigma}) + "
          f"read={n_read_per_sub}/sub ({read_patch_size}px sharp) = {n_total} GRU steps")
    print(f"Attention: meta_guide={meta_guide_weight}  sub_guide={sub_guide_weight}  "
          f"read_void={read_void_weight}  blur_sigma_ratio={blur_sigma_ratio}  "
          f"diversity={diversity_weight}  diversity_sigma={diversity_sigma}  "
          f"scan_vy={scan_vy}  batch_size={batch_size}")

    os.makedirs(save_dir, exist_ok=True)
    save_run_info(save_dir, cfg)

    dataset = CountingDataset(data_dir)
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
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"ReadingModel: {n_params:,} parameters")

    loss_names = ['count_cls', 'meta_attn', 'sub_attn', 'read_void',
                  'meta_content', 'sub_content', 'div',
                  'count_acc', 'hit_rate', 'hit_intensity']
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

    header = (f"{'epoch':>5s}  {'count_cls':>9s}  {'meta_a':>7s}  {'sub_a':>7s}  "
              f"{'rd_void':>7s}  {'meta_c':>7s}  {'sub_c':>7s}  {'div':>6s}  "
              f"{'count_acc':>9s}  {'hr':>6s}  {'hi':>6s}  {'lr':>8s}  time")
    logger = TrainingLogger(save_dir, header, start_epoch)

    for epoch in range(start_epoch, end_epoch):
        epoch_start = time.time()
        tracker.reset_epoch()
        total_correct = 0
        total_samples = 0

        for img, clean, counts_batch, _letters, _fonts in dataloader:
            img = img.to(device)
            clean = clean.to(device)
            B = img.shape[0]

            count_target = torch.tensor([c - 1 for c in counts_batch], device=device)

            count_logits, enc = model(img)

            # Primary: count classification
            count_cls_loss = F.cross_entropy(count_logits, count_target)

            # Phase-specific location lists (with dummy init for loss functions)
            init_loc = enc.locations[0]
            meta_locs = [init_loc] + enc.meta_positions
            sub_locs = [init_loc] + enc.sub_positions

            read_locs = [enc.locations[i] for i, tag in enumerate(enc.phase_tags) if tag == 'read']

            # Attention guides (meta + sub toward ink)
            meta_attn = meta_guide_weight * attention_content_loss(
                clean, meta_locs, blur_sigma_ratio=blur_sigma_ratio)
            sub_attn = sub_guide_weight * attention_content_loss(
                clean, sub_locs, blur_sigma_ratio=blur_sigma_ratio)

            # Read void repulsion (no guide, like v7)
            read_void_loss = torch.tensor(0.0, device=device)
            if read_void_weight > 0 and len(read_locs) > 0:
                read_void_loss = read_void_weight * void_repulsion(
                    clean, read_locs, read_patch_size, read_patch_size)

            # Diversity (per phase)
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

            total_loss = (count_cls_loss
                          + meta_attn + sub_attn + read_void_loss
                          + diversity_weight * div_loss
                          + meta_content_weight * meta_content_loss
                          + sub_content_weight * sub_content_loss)

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            with torch.no_grad():
                preds = count_logits.argmax(1)
                total_correct += (preds == count_target).sum().item()
                total_samples += B
                hr, hi = fixation_hit_rate(clean, enc.locations)

            tracker.update(count_cls=count_cls_loss, meta_attn=meta_attn,
                           sub_attn=sub_attn, read_void=read_void_loss,
                           meta_content=meta_content_loss, sub_content=sub_content_loss,
                           div=div_loss, count_acc=0, hit_rate=hr, hit_intensity=hi)

        avgs = tracker.end_epoch()
        count_acc = total_correct / total_samples if total_samples > 0 else 0
        avgs['count_acc'] = count_acc
        tracker.history['count_acc'][-1] = count_acc

        epoch_time = time.time() - epoch_start
        elapsed = time.time() - train_start
        done = epoch - start_epoch + 1
        eta = format_eta(elapsed, done, epochs - done)
        current_lr = scheduler.get_last_lr()[0]

        print(f"Epoch {epoch+1}/{end_epoch}: "
              f"Count {avgs['count_cls']:.4f}  MetaA {avgs['meta_attn']:.4f}  "
              f"SubA {avgs['sub_attn']:.4f}  Void {avgs['read_void']:.4f}  "
              f"Div {avgs['div']:.4f}  "
              f"Acc {count_acc:.0%}  Hit {avgs['hit_rate']:.0%}  "
              f"lr {current_lr:.6f}  [{epoch_time:.1f}s  ETA {eta}]")

        logger.write_line(f"{epoch+1:>5d}  {avgs['count_cls']:>9.4f}  {avgs['meta_attn']:>7.4f}  "
                          f"{avgs['sub_attn']:>7.4f}  {avgs['read_void']:>7.4f}  "
                          f"{avgs['meta_content']:>7.4f}  {avgs['sub_content']:>7.4f}  "
                          f"{avgs['div']:>6.4f}  "
                          f"{count_acc:>9.4f}  {avgs['hit_rate']:>6.4f}  {avgs['hit_intensity']:>6.4f}  "
                          f"{current_lr:>8.6f}  {epoch_time:.1f}s")
        scheduler.step()

        if (epoch + 1 - start_epoch) % checkpoint_interval == 0:
            _save_reading_checkpoint(model, epoch, save_dir, cfg, tracker,
                                     n_meta, n_sub_per_meta, n_read_per_sub,
                                     meta_patch_pixels, meta_blur_sigma,
                                     sub_patch_pixels, sub_blur_sigma,
                                     read_patch_size, latent_dim, n_scales,
                                     name=f'checkpoint_epoch_{epoch+1}.pth')

    logger.close()

    _save_reading_checkpoint(model, end_epoch - 1, save_dir, cfg, tracker,
                             n_meta, n_sub_per_meta, n_read_per_sub,
                             meta_patch_pixels, meta_blur_sigma,
                             sub_patch_pixels, sub_blur_sigma,
                             read_patch_size, latent_dim, n_scales,
                             name='model_final.pth')

    specs = [
        {'keys': ['count_cls'], 'labels': ['Count CE'], 'colors': ['tab:red'],
         'title': 'Count Classification (3-class)', 'ylabel': 'Cross-Entropy',
         'hlines': [(np.log(3), f'Random ({np.log(3):.2f})', 'gray')]},
        {'keys': ['count_acc'], 'labels': ['Accuracy'], 'colors': ['tab:blue'],
         'title': 'Count Accuracy', 'ylabel': 'Accuracy', 'ylim': (0, 1),
         'hlines': [(1/3, 'Random (33%)', 'gray')]},
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
                    })
