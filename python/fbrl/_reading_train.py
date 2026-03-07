"""Reading training functions (v9.2 — isolated read heads) — imported by train.py."""
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


def assign_heads_spatial(prescan_end_positions, char_positions, char_labels, n_heads):
    """Assign ground-truth letters to read heads by spatial proximity.

    For each letter (non-NaN position), find the closest head by final
    prescan-phase x-position (detached). Assign that letter's label to
    that head. Unassigned heads get void (26).

    Greedy closest: each letter claims the nearest unassigned head.

    Args:
        prescan_end_positions: list of n_heads × (B, 2) tensors
        char_positions: (B, max_count) float tensor, NaN-padded
        char_labels: (B, max_count) long tensor, 26-padded (void)
        n_heads: int

    Returns:
        head_targets: (B, n_heads) long tensor of letter indices (0-25 or 26=void)
    """
    B = char_positions.shape[0]
    device = char_positions.device

    # Single CPU transfer — avoids ~800 per-element .item() CUDA syncs
    head_xs_cpu = torch.stack(
        [p[:, 0].detach() for p in prescan_end_positions], dim=0
    ).cpu().numpy()  # (n_heads, B)
    char_pos_cpu = char_positions.cpu().numpy()  # (B, max_count)
    char_lab_cpu = char_labels.cpu().numpy()     # (B, max_count)

    # Initialize all heads as void (on CPU, move to device at end)
    targets = np.full((B, n_heads), 26, dtype=np.int64)

    max_count = char_pos_cpu.shape[1]
    for b in range(B):
        assigned = set()
        # Gather valid letters for this sample
        letters = []
        for li in range(max_count):
            pos = char_pos_cpu[b, li]
            if not np.isnan(pos):
                letters.append((pos, int(char_lab_cpu[b, li])))

        # Sort letters by position (left to right) for deterministic assignment
        letters.sort(key=lambda x: x[0])

        for pos, label in letters:
            best_dist = float('inf')
            best_h = -1
            for hi in range(n_heads):
                if hi in assigned:
                    continue
                dist = abs(head_xs_cpu[hi, b] - pos)
                if dist < best_dist:
                    best_dist = dist
                    best_h = hi
            if best_h >= 0:
                targets[b, best_h] = label
                assigned.add(best_h)

    return torch.from_numpy(targets).to(device)


def train_reading_model(cfg):
    """Train a ReadingModel (v9.2) from an ExperimentConfig.

    Fully isolated read heads: 8 heads (2 per zone), each with:
      search (blurred) → prescan (sharp wide) → read (sharp).
    No shared discovery GRU. Content probes at fixed positions for aux.
    Letter classification (27-class: a-z + void) on each head.
    Count derived from non-void predictions.
    """
    n_zones = cfg.n_zones
    n_heads_per_zone = cfg.n_heads_per_zone
    n_search_steps = cfg.n_search_steps
    n_prescan_steps = cfg.n_prescan_steps
    n_read_steps = cfg.n_read_steps
    head_offset = cfg.head_offset
    n_letter_classes = cfg.n_letter_classes
    probe_patch_pixels = cfg.probe_patch_pixels
    probe_blur_sigma = cfg.probe_blur_sigma
    search_patch_pixels = cfg.search_patch_pixels
    search_blur_sigma = cfg.search_blur_sigma
    prescan_patch_size = cfg.prescan_patch_size
    read_patch_size = cfg.read_patch_size
    latent_dim = cfg.latent_dim
    n_scales = cfg.n_scales
    search_guide_weight = cfg.search_guide_weight
    search_content_weight = cfg.search_content_weight
    probe_content_weight = cfg.probe_content_weight
    read_void_weight = cfg.read_void_weight
    blur_sigma_ratio = cfg.blur_sigma_ratio
    diversity_weight = cfg.diversity_weight
    diversity_sigma = cfg.diversity_sigma
    zone_diversity_weight = cfg.zone_diversity_weight
    zone_diversity_sigma = cfg.zone_diversity_sigma
    batch_size = cfg.batch_size
    epochs = cfg.epochs
    resume = cfg.resume
    save_dir = cfg.save_dir
    data_dir = cfg.data_dir
    checkpoint_interval = cfg.checkpoint_interval

    n_heads = n_zones * n_heads_per_zone
    n_steps_per_head = n_search_steps + n_prescan_steps + n_read_steps

    device = _resolve_device(cfg.device)
    print(f"Reading v9.2 training on: {device}")
    print(f"Isolated heads: {n_zones} zones × {n_heads_per_zone} heads = {n_heads} heads")
    print(f"Per head: {n_search_steps} search ({search_patch_pixels}, blur={search_blur_sigma}) + "
          f"{n_prescan_steps} prescan ({prescan_patch_size}) + "
          f"{n_read_steps} read ({read_patch_size}) = {n_steps_per_head} steps")
    print(f"Total GRU steps: {n_heads * n_steps_per_head} + {n_zones} probes (no GRU)")
    print(f"Letter classes: {n_letter_classes} (26 letters + void)")
    print(f"Losses: search_guide={search_guide_weight}  search_content={search_content_weight}  "
          f"probe_content={probe_content_weight}  read_void={read_void_weight}  "
          f"diversity={diversity_weight}  zone_div={zone_diversity_weight}  "
          f"batch_size={batch_size}")

    os.makedirs(save_dir, exist_ok=True)
    save_run_info(save_dir, cfg)

    dataset = CountingDataset(data_dir, train_fonts=cfg.train_fonts)
    use_cuda = device.type == 'cuda'
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                             pin_memory=use_cuda)

    model = ReadingModel(
        n_zones=n_zones, n_heads_per_zone=n_heads_per_zone,
        n_search_steps=n_search_steps, n_prescan_steps=n_prescan_steps,
        n_read_steps=n_read_steps,
        search_patch_pixels=search_patch_pixels, search_blur_sigma=search_blur_sigma,
        probe_patch_pixels=probe_patch_pixels, probe_blur_sigma=probe_blur_sigma,
        prescan_patch_size=prescan_patch_size,
        read_patch_size=read_patch_size, latent_dim=latent_dim,
        n_scales=n_scales, n_letter_classes=n_letter_classes,
        head_offset=head_offset,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"ReadingModel v9.2: {n_params:,} parameters")

    loss_names = ['letter_ce', 'search_guide', 'search_content', 'probe_content',
                  'read_void', 'zone_div', 'read_div',
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
              f"{'cnt_a':>6s}  {'s_gd':>6s}  {'s_cnt':>6s}  "
              f"{'p_cnt':>6s}  {'r_void':>6s}  {'z_div':>6s}  {'r_div':>6s}  "
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

            group_logits, enc = model(img)  # (B, n_heads, 27)

            # Assign ground-truth labels to heads by spatial proximity
            head_targets = assign_heads_spatial(
                enc.prescan_end_positions, char_pos, char_lab, n_heads)  # (B, n_heads)

            # Dynamic void class weight: inverse frequency per batch
            flat_targets = head_targets.view(-1)
            n_void = (flat_targets == 26).sum().float()
            n_nonvoid = (flat_targets != 26).sum().float()
            total_fg = n_void + n_nonvoid
            if n_void > 0 and n_nonvoid > 0:
                class_weights = torch.ones(n_letter_classes, device=device)
                class_weights[:26] = total_fg / (2 * n_nonvoid)
                class_weights[26] = total_fg / (2 * n_void)
            else:
                class_weights = None

            # Primary: letter CE on all heads
            letter_ce = F.cross_entropy(
                group_logits.view(-1, n_letter_classes),
                flat_targets,
                weight=class_weights,
            )

            # Search guide: pull search positions toward ink
            search_locs = [enc.locations[i] for i, tag in enumerate(enc.phase_tags) if tag == 'search']
            search_guide_loss = torch.tensor(0.0, device=device)
            if search_guide_weight > 0 and len(search_locs) > 0:
                # Add a dummy init location at center for attention_content_loss
                init_loc = torch.zeros(B, 2, device=device)
                search_guide_loss = search_guide_weight * attention_content_loss(
                    clean, [init_loc] + search_locs, blur_sigma_ratio=blur_sigma_ratio)

            # Search content BCE
            search_content_loss = torch.tensor(0.0, device=device)
            if search_content_weight > 0 and len(enc.search_content_logits) > 0:
                for loc, logit in zip(search_locs, enc.search_content_logits):
                    grid = loc.view(B, 1, 1, 2)
                    sampled = F.grid_sample(clean, grid, align_corners=True,
                                            padding_mode='zeros')
                    label = (sampled.view(-1, 1) > 0.1).float()
                    search_content_loss = search_content_loss + bce(logit, label)
                search_content_loss = search_content_loss / len(enc.search_content_logits)

            # Probe content BCE
            probe_content_loss = torch.tensor(0.0, device=device)
            if probe_content_weight > 0 and len(enc.probe_content_logits) > 0:
                probe_locs = [enc.locations[i] for i, tag in enumerate(enc.phase_tags) if tag == 'probe']
                for loc, logit in zip(probe_locs, enc.probe_content_logits):
                    grid = loc.view(B, 1, 1, 2)
                    sampled = F.grid_sample(clean, grid, align_corners=True,
                                            padding_mode='zeros')
                    label = (sampled.view(-1, 1) > 0.1).float()
                    probe_content_loss = probe_content_loss + bce(logit, label)
                probe_content_loss = probe_content_loss / len(enc.probe_content_logits)

            # Read void repulsion
            read_void_loss = torch.tensor(0.0, device=device)
            if read_void_weight > 0:
                read_locs = [enc.locations[i] for i, tag in enumerate(enc.phase_tags) if tag == 'read']
                if len(read_locs) > 0:
                    read_void_loss = read_void_weight * void_repulsion(
                        clean, read_locs, read_patch_size, read_patch_size)

            # Zone diversity: within-zone repulsion between paired heads
            zone_div_loss = torch.tensor(0.0, device=device)
            if zone_diversity_weight > 0:
                for zi in range(n_zones):
                    # Get prescan end positions for the 2 heads in this zone
                    h0_idx = zi * n_heads_per_zone
                    h1_idx = h0_idx + 1
                    pos0 = enc.prescan_end_positions[h0_idx]  # (B, 2)
                    pos1 = enc.prescan_end_positions[h1_idx]  # (B, 2)
                    diff = pos0 - pos1
                    dist_sq = (diff ** 2).sum(-1)  # (B,)
                    repulsion = torch.exp(-dist_sq / (2 * zone_diversity_sigma ** 2))
                    zone_div_loss = zone_div_loss + repulsion.mean()
                zone_div_loss = zone_div_loss / n_zones

            # Per-head read diversity: spread read positions within each head
            read_div_loss = torch.tensor(0.0, device=device)
            if diversity_weight > 0:
                for head_i in range(n_heads):
                    head_read_locs = [enc.locations[i] for i in range(len(enc.locations))
                                      if enc.head_ids[i] == head_i and enc.phase_tags[i] == 'read']
                    if len(head_read_locs) >= 2:
                        # Use standard fixation_diversity_loss (expects [init] + locs)
                        dummy_init = head_read_locs[0]  # reuse first as dummy
                        read_div_loss = read_div_loss + fixation_diversity_loss(
                            [dummy_init] + head_read_locs, sigma=diversity_sigma)
                read_div_loss = read_div_loss / max(n_heads, 1)

            total_loss = (letter_ce
                          + search_guide_loss
                          + search_content_weight * search_content_loss
                          + probe_content_weight * probe_content_loss
                          + read_void_loss
                          + zone_diversity_weight * zone_div_loss
                          + diversity_weight * read_div_loss)

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            # Metrics
            with torch.no_grad():
                preds = group_logits.argmax(2)  # (B, n_heads)
                targets = head_targets

                # Letter accuracy (non-void heads only)
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

            tracker.update(letter_ce=letter_ce, search_guide=search_guide_loss,
                           search_content=search_content_loss,
                           probe_content=probe_content_loss,
                           read_void=read_void_loss,
                           zone_div=zone_div_loss, read_div=read_div_loss,
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
              f"SGd {avgs['search_guide']:.4f}  SCnt {avgs['search_content']:.4f}  "
              f"PCnt {avgs['probe_content']:.4f}  Void {avgs['read_void']:.4f}  "
              f"ZDiv {avgs['zone_div']:.4f}  RDiv {avgs['read_div']:.4f}  "
              f"Hit {avgs['hit_rate']:.0%}  "
              f"lr {current_lr:.6f}  [{epoch_time:.1f}s  ETA {eta}]")

        logger.write_line(
            f"{epoch+1:>5d}  {avgs['letter_ce']:>7.4f}  {letter_acc:>6.4f}  {void_acc:>6.4f}  "
            f"{count_acc:>6.4f}  {avgs['search_guide']:>6.4f}  {avgs['search_content']:>6.4f}  "
            f"{avgs['probe_content']:>6.4f}  {avgs['read_void']:>6.4f}  "
            f"{avgs['zone_div']:>6.4f}  {avgs['read_div']:>6.4f}  "
            f"{avgs['hit_rate']:>6.4f}  {avgs['hit_intensity']:>6.4f}  "
            f"{current_lr:>8.6f}  {epoch_time:.1f}s")
        scheduler.step()

        if (epoch + 1 - start_epoch) % checkpoint_interval == 0:
            _save_reading_checkpoint(model, epoch, save_dir, cfg, tracker,
                                     name=f'checkpoint_epoch_{epoch+1}.pth')

    logger.close()

    _save_reading_checkpoint(model, end_epoch - 1, save_dir, cfg, tracker,
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
        {'keys': ['search_guide'], 'labels': ['Search guide'], 'colors': ['tab:green'],
         'title': 'Search attention guide (lower = on content)', 'ylabel': 'Loss'},
        {'keys': ['search_content', 'probe_content'],
         'labels': ['Search content', 'Probe content'],
         'colors': ['tab:cyan', 'tab:teal'], 'styles': ['-', '--'],
         'title': 'Content detection (search + probe)', 'ylabel': 'BCE'},
        {'keys': ['read_void'], 'labels': ['Read void'], 'colors': ['tab:brown'],
         'title': 'Read void repulsion', 'ylabel': 'Loss'},
        {'keys': ['zone_div', 'read_div'],
         'labels': ['Zone diversity', 'Read diversity'],
         'colors': ['tab:orange', 'tab:pink'], 'styles': ['-', '--'],
         'title': 'Diversity losses', 'ylabel': 'Repulsion'},
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


def _save_reading_checkpoint(model, epoch, save_dir, cfg, tracker, name='model_final.pth'):
    save_checkpoint(model, epoch,
                    os.path.join(save_dir, name),
                    cfg=cfg, losses_dict=tracker.get_history_dict(),
                    extra={
                        'model_type': 'reading',
                        'n_zones': cfg.n_zones,
                        'n_heads_per_zone': cfg.n_heads_per_zone,
                        'n_search_steps': cfg.n_search_steps,
                        'n_prescan_steps': cfg.n_prescan_steps,
                        'n_read_steps': cfg.n_read_steps,
                        'head_offset': cfg.head_offset,
                        'probe_patch_pixels': list(cfg.probe_patch_pixels),
                        'probe_blur_sigma': cfg.probe_blur_sigma,
                        'search_patch_pixels': list(cfg.search_patch_pixels),
                        'search_blur_sigma': cfg.search_blur_sigma,
                        'prescan_patch_size': list(cfg.prescan_patch_size),
                        'read_patch_size': cfg.read_patch_size,
                        'latent_dim': cfg.latent_dim,
                        'n_scales': cfg.n_scales,
                        'n_letter_classes': cfg.n_letter_classes,
                    })
