"""Counting training functions — imported by train.py."""
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import time

from fbrl import _resolve_device
from fbrl.data import CountingDataset
from fbrl.model import CountingModel
from fbrl.losses import attention_content_loss, fixation_diversity_loss, fixation_hit_rate
from fbrl.training_utils import (LossTracker, TrainingLogger, save_checkpoint,
                                  plot_training_metrics, format_eta, save_run_info)


def train_counting_model(cfg):
    """Train a CountingModel from an ExperimentConfig.

    Fixed 192x128 canvas for all counts. Single shuffled dataloader —
    all counts mixed, no cheating via batch structure.
    """
    n_scan_glimpses = cfg.n_scan_glimpses
    scan_patch_size = cfg.scan_patch_size
    latent_dim = cfg.latent_dim
    n_scales = cfg.n_scales
    scan_guide_weight = cfg.scan_guide_weight
    blur_sigma_ratio = cfg.blur_sigma_ratio
    diversity_weight = cfg.diversity_weight
    diversity_sigma = cfg.diversity_sigma
    scan_vy = cfg.scan_vy
    content_weight = cfg.content_weight
    batch_size = cfg.batch_size
    epochs = cfg.epochs
    resume = cfg.resume
    save_dir = cfg.save_dir
    data_dir = cfg.data_dir
    checkpoint_interval = cfg.checkpoint_interval

    device = _resolve_device(cfg.device)
    print(f"Counting training on: {device}")
    print(f"Scan-only: {n_scan_glimpses} glimpses ({scan_patch_size}), "
          f"latent_dim={latent_dim}, fixed 192x128 canvas")
    print(f"Attention: scan_guide_weight={scan_guide_weight}  "
          f"blur_sigma_ratio={blur_sigma_ratio}  "
          f"diversity_weight={diversity_weight}  diversity_sigma={diversity_sigma}  "
          f"scan_vy={scan_vy}  content_weight={content_weight}  "
          f"batch_size={batch_size}")

    os.makedirs(save_dir, exist_ok=True)
    save_run_info(save_dir, cfg)

    dataset = CountingDataset(data_dir)
    use_cuda = device.type == 'cuda'
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                             pin_memory=use_cuda)

    model = CountingModel(
        n_scan_glimpses=n_scan_glimpses, scan_patch_size=scan_patch_size,
        latent_dim=latent_dim, n_scales=n_scales, max_count=3,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"CountingModel: {n_params:,} parameters")

    loss_names = ['count_cls', 'content', 'attn', 'div',
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

    header = (f"{'epoch':>5s}  {'count_cls':>9s}  {'content':>7s}  {'attn':>7s}  {'div':>6s}  "
              f"{'count_acc':>9s}  {'hr':>6s}  {'hi':>6s}  {'lr':>8s}  time")
    logger = TrainingLogger(save_dir, header, start_epoch)

    for epoch in range(start_epoch, end_epoch):
        epoch_start = time.time()
        tracker.reset_epoch()
        total_correct = 0
        total_samples = 0

        for img, clean, counts_batch, _letters, _fonts, *_ in dataloader:
            img = img.to(device)
            clean = clean.to(device)
            B = img.shape[0]

            # Count target: 0-indexed (count 1 -> class 0, count 2 -> class 1, etc.)
            count_target = torch.tensor([c - 1 for c in counts_batch], device=device)

            count_logits, scan_content_logits, locations, latent = model(img)

            count_cls_loss = F.cross_entropy(count_logits, count_target)

            attn_loss = scan_guide_weight * attention_content_loss(
                clean, locations, blur_sigma_ratio=blur_sigma_ratio)

            div_loss = fixation_diversity_loss(
                locations, sigma=diversity_sigma, vy=scan_vy)

            content_loss = torch.tensor(0.0, device=device)
            if content_weight > 0 and len(scan_content_logits) > 0:
                scan_locs = locations[1:len(scan_content_logits) + 1]
                for loc, logit in zip(scan_locs, scan_content_logits):
                    grid = loc.view(B, 1, 1, 2)
                    sampled = F.grid_sample(clean, grid, align_corners=True,
                                            padding_mode='zeros')
                    label = (sampled.view(-1, 1) > 0.1).float()
                    content_loss = content_loss + bce(logit, label)
                content_loss = content_loss / len(scan_content_logits)

            total_loss = (count_cls_loss
                          + attn_loss
                          + diversity_weight * div_loss
                          + content_weight * content_loss)

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            with torch.no_grad():
                preds = count_logits.argmax(1)
                total_correct += (preds == count_target).sum().item()
                total_samples += B
                hr, hi = fixation_hit_rate(clean, locations)

            tracker.update(count_cls=count_cls_loss, content=content_loss,
                           attn=attn_loss, div=div_loss,
                           count_acc=0, hit_rate=hr, hit_intensity=hi)

        avgs = tracker.end_epoch()
        count_acc = total_correct / total_samples if total_samples > 0 else 0
        # Override the averaged count_acc with the true epoch value
        avgs['count_acc'] = count_acc
        tracker.history['count_acc'][-1] = count_acc

        epoch_time = time.time() - epoch_start
        elapsed = time.time() - train_start
        done = epoch - start_epoch + 1
        eta = format_eta(elapsed, done, epochs - done)
        current_lr = scheduler.get_last_lr()[0]

        print(f"Epoch {epoch+1}/{end_epoch}: "
              f"Count {avgs['count_cls']:.4f}  Cont {avgs['content']:.4f}  "
              f"Attn {avgs['attn']:.4f}  Div {avgs['div']:.4f}  "
              f"Acc {count_acc:.0%}  Hit {avgs['hit_rate']:.0%}  "
              f"lr {current_lr:.6f}  [{epoch_time:.1f}s  ETA {eta}]")

        logger.write_line(f"{epoch+1:>5d}  {avgs['count_cls']:>9.4f}  {avgs['content']:>7.4f}  "
                          f"{avgs['attn']:>7.4f}  {avgs['div']:>6.4f}  "
                          f"{count_acc:>9.4f}  {avgs['hit_rate']:>6.4f}  {avgs['hit_intensity']:>6.4f}  "
                          f"{current_lr:>8.6f}  {epoch_time:.1f}s")
        scheduler.step()

        if (epoch + 1 - start_epoch) % checkpoint_interval == 0:
            save_checkpoint(model, epoch,
                            os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth'),
                            cfg=cfg, losses_dict=tracker.get_history_dict(),
                            extra={
                                'model_type': 'counting',
                                'n_scan_glimpses': n_scan_glimpses,
                                'scan_patch_size': scan_patch_size,
                                'latent_dim': latent_dim,
                                'n_scales': n_scales,
                                'max_count': 3,
                            })

    logger.close()

    save_checkpoint(model, end_epoch - 1,
                    os.path.join(save_dir, 'model_final.pth'),
                    cfg=cfg, losses_dict=tracker.get_history_dict(),
                    extra={
                        'model_type': 'counting',
                        'n_scan_glimpses': n_scan_glimpses,
                        'scan_patch_size': scan_patch_size,
                        'latent_dim': latent_dim,
                        'n_scales': n_scales,
                        'max_count': 3,
                    })

    specs = [
        {'keys': ['count_cls'], 'labels': ['Count CE'], 'colors': ['tab:red'],
         'title': 'Count Classification (3-class)', 'ylabel': 'Cross-Entropy',
         'hlines': [(np.log(3), f'Random ({np.log(3):.2f})', 'gray')]},
        {'keys': ['count_acc'], 'labels': ['Accuracy'], 'colors': ['tab:blue'],
         'title': 'Count Accuracy', 'ylabel': 'Accuracy', 'ylim': (0, 1),
         'hlines': [(1/3, 'Random (33%)', 'gray')]},
        {'keys': ['content'], 'labels': ['Content BCE'], 'colors': ['tab:cyan'],
         'title': 'Content detection (scan phase)', 'ylabel': 'Loss'},
        {'keys': ['attn'], 'labels': ['Guide'], 'colors': ['tab:green'],
         'title': 'Attention guide (lower = fixations on content)', 'ylabel': 'Loss'},
        {'keys': ['div'], 'labels': ['Diversity'], 'colors': ['tab:orange'],
         'title': 'Fixation diversity (lower = more spread)', 'ylabel': 'Repulsion'},
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
