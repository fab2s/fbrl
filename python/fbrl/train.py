import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import time

from fbrl import _resolve_device
from fbrl.data import LetterDataset, BigramDataset
from fbrl.model import VisionModel, BigramVisionModel
from fbrl.losses import (attention_content_loss, two_phase_attention_loss,
                          fixation_diversity_loss, fixation_edge_loss,
                          fixation_hit_rate, void_repulsion)
from fbrl.training_utils import (LossTracker, TrainingLogger, save_checkpoint,
                                  apply_transfer, plot_training_metrics, format_eta,
                                  save_run_info)
from fbrl._word_train import train_word_model  # noqa: F401
from fbrl._motor_train import train_motor_model  # noqa: F401
from fbrl._counting_train import train_counting_model  # noqa: F401
from fbrl._reading_train import train_reading_model  # noqa: F401


# --- Training ---

def train_model(cfg):
    """Train a VisionModel (single-letter) from an ExperimentConfig."""
    n_glimpses = cfg.n_read_glimpses
    patch_size = cfg.read_patch_size
    n_scales = cfg.n_scales
    n_scan_glimpses = cfg.n_scan_glimpses
    scan_patch_size = cfg.scan_patch_size
    scan_vy = cfg.scan_vy
    scan_guide_weight = cfg.scan_guide_weight
    content_weight = cfg.content_weight
    guide_weight = cfg.guide_weight
    blur_sigma_ratio = cfg.blur_sigma_ratio
    diversity_weight = cfg.diversity_weight
    diversity_sigma = cfg.diversity_sigma
    diversity_vy = cfg.diversity_vy
    recode_weight = cfg.recode_weight
    void_weight = cfg.void_weight
    scan_void_weight = cfg.scan_void_weight
    batch_size = cfg.batch_size
    epochs = cfg.epochs
    resume = cfg.resume
    save_dir = cfg.save_dir
    data_dir = cfg.data_dir
    checkpoint_interval = cfg.checkpoint_interval

    device = _resolve_device(cfg.device)
    print(f"Training on: {device}")
    vy_str = f"  diversity_vy={diversity_vy}" if diversity_vy != 1.0 else ""
    scan_str = (f"\nTwo-phase: scan={n_scan_glimpses} (prescribed x, {scan_patch_size}) + "
                f"read={n_glimpses} ({patch_size}) = {n_scan_glimpses + n_glimpses} glimpses  "
                f"scan_vy={scan_vy}  content_weight={content_weight}") if n_scan_glimpses > 0 else ""
    print(f"Attention: guide_weight={guide_weight}  blur_sigma_ratio={blur_sigma_ratio}  "
          f"diversity_weight={diversity_weight}  diversity_sigma={diversity_sigma}{vy_str}  "
          f"recode_weight={recode_weight}  batch_size={batch_size}{scan_str}")

    os.makedirs(save_dir, exist_ok=True)
    save_run_info(save_dir, cfg)
    dataset = LetterDataset(data_dir)
    use_cuda = device.type == 'cuda'
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=use_cuda)

    if dataset.has_partners:
        print(f"Partner images found — recode loss enabled (weight={recode_weight})")
    else:
        print("No partner images — recode loss disabled")

    model = VisionModel(
        n_glimpses=n_glimpses, patch_size=patch_size, n_scales=n_scales,
        n_scan_glimpses=n_scan_glimpses, scan_patch_size=scan_patch_size,
        read_anchor_scan_indices=cfg.read_anchor_scan_indices,
        n_read_per_group=cfg.n_read_per_group,
        learnable_scan_x=cfg.learnable_scan_x,
    ).to(device)

    # Loss tracking
    loss_names = ['recon', 'letter_cls', 'case_cls', 'attn', 'div',
                  'recode', 'content', 'void', 'hit_rate', 'hit_intensity']
    tracker = LossTracker(loss_names)

    start_epoch = 0
    if resume:
        checkpoint = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model'], strict=False)
        start_epoch = checkpoint['epoch'] + 1
        if 'losses' in checkpoint:
            h = checkpoint['losses']
            # Map old key names
            restore = {}
            for k in loss_names:
                if k in h:
                    restore[k] = h[k]
                elif k == 'letter_cls' and 'cls' in h:
                    restore[k] = h['cls']
            tracker.restore_history(restore)
        print(f"Resumed from epoch {start_epoch} ({len(tracker.history['letter_cls'])} prior epochs of history)")

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = torch.nn.MSELoss()

    end_epoch = start_epoch + epochs
    train_start = time.time()

    # Log file
    content_hdr = f"  {'content':>7s}" if n_scan_glimpses > 0 else ""
    vram_hdr = f"  {'vram':>7s}" if use_cuda else ""
    header = (f"{'epoch':>5s}  {'recon':>6s}  {'ltr':>6s}  {'case':>6s}  {'attn':>7s}  {'div':>6s}  "
              f"{'hit':>6s}  {'recode':>6s}{content_hdr}  {'lr':>8s}{vram_hdr}  time")
    logger = TrainingLogger(save_dir, header, start_epoch)

    # --- Profiling infrastructure (epoch 0 only) ---
    prof = {k: 0.0 for k in [
        'forward', 'cls_recon', 'recode', 'content', 'attn_void_div',
        'total_sum', 'zero_grad', 'backward', 'clip', 'step', 'detach_metrics',
    ]}
    def sync():
        if use_cuda:
            torch.cuda.synchronize()

    for epoch in range(start_epoch, end_epoch):
        epoch_start = time.time()
        tracker.reset_epoch()
        if use_cuda:
            torch.cuda.reset_peak_memory_stats()
        probe = (epoch == start_epoch)
        if probe:
            for k in prof:
                prof[k] = 0.0
        n_batches = 0

        for img, clean, letters, cases, _fonts, partner_clean in dataloader:
            img = img.to(device)
            clean = clean.to(device)
            partner_clean = partner_clean.to(device)

            letter_idx = torch.tensor([ord(l) - ord('A') for l in letters], device=device)
            case_idx = torch.tensor([0 if c == 'upper' else 1 for c in cases], device=device)
            case_float = case_idx.float().unsqueeze(1)

            if probe: sync()
            t0 = time.perf_counter()

            recon, letter_logits, case_logits, locations, latent, scan_content_logits = model(img, case_float)
            actual_n_scan = len(scan_content_logits)

            if probe: sync(); prof['forward'] += time.perf_counter() - t0
            t1 = time.perf_counter()

            recon_loss = criterion(recon, img)
            letter_cls_loss = F.cross_entropy(letter_logits, letter_idx)
            case_cls_loss = F.cross_entropy(case_logits, case_idx)

            if probe: sync(); prof['cls_recon'] += time.perf_counter() - t1
            t2 = time.perf_counter()

            # Recode (measured before other losses to match Rust order).
            recode_loss_val = 0.0
            if dataset.has_partners and recode_weight > 0:
                flipped_case = 1.0 - case_float
                recode_img = model.decoder(latent, flipped_case)
                recode_loss = criterion(recode_img, partner_clean)
                recode_loss_val = recode_loss.item()

            if probe: sync(); prof['recode'] += time.perf_counter() - t2
            t3 = time.perf_counter()

            content_loss = torch.tensor(0.0, device=device)
            if content_weight > 0 and actual_n_scan > 0:
                bce = torch.nn.BCEWithLogitsLoss()
                scan_locs = locations[1:actual_n_scan + 1]
                for loc, logit in zip(scan_locs, scan_content_logits):
                    grid = loc.view(img.shape[0], 1, 1, 2)
                    sampled = F.grid_sample(clean, grid, align_corners=True,
                                            padding_mode='zeros')
                    label = (sampled.view(-1, 1) > 0.1).float()
                    content_loss = content_loss + bce(logit, label)
                content_loss = content_loss / actual_n_scan

            if probe: sync(); prof['content'] += time.perf_counter() - t3
            t4 = time.perf_counter()

            if actual_n_scan > 0:
                scan_attn = attention_content_loss(clean, locations[:actual_n_scan + 1],
                                                   blur_sigma_ratio=blur_sigma_ratio)
                read_attn = attention_content_loss(clean, locations[actual_n_scan:],
                                                   blur_sigma_ratio=blur_sigma_ratio)
                attn_loss = scan_guide_weight * scan_attn + guide_weight * read_attn
            else:
                attn_loss = guide_weight * attention_content_loss(clean, locations,
                                                                  blur_sigma_ratio=blur_sigma_ratio)

            if actual_n_scan > 0:
                scan_div = fixation_diversity_loss(locations[:actual_n_scan + 1],
                                                   sigma=diversity_sigma, vy=scan_vy)
                read_div = fixation_diversity_loss(locations[actual_n_scan:],
                                                   sigma=diversity_sigma, vy=diversity_vy)
                div_loss = scan_div + read_div
            else:
                div_loss = fixation_diversity_loss(locations, sigma=diversity_sigma,
                                                   vy=diversity_vy)

            void_loss = torch.tensor(0.0, device=device)
            if scan_void_weight > 0 and actual_n_scan > 0:
                scan_sample_locs = locations[1:actual_n_scan + 1]
                scan_ph, scan_pw = scan_patch_size
                void_loss = void_loss + scan_void_weight * void_repulsion(
                    clean, scan_sample_locs, scan_ph, scan_pw)
            if void_weight > 0:
                read_sample_locs = locations[actual_n_scan + 1:]
                void_loss = void_loss + void_weight * void_repulsion(
                    clean, read_sample_locs, patch_size, patch_size)

            if probe: sync(); prof['attn_void_div'] += time.perf_counter() - t4
            t5 = time.perf_counter()

            total_loss = (recon_loss + letter_cls_loss + case_cls_loss
                          + attn_loss + diversity_weight * div_loss
                          + content_weight * content_loss + void_loss)

            if dataset.has_partners and recode_weight > 0:
                total_loss = total_loss + recode_weight * recode_loss

            if probe: sync(); prof['total_sum'] += time.perf_counter() - t5

            if probe and n_batches == 0:
                # Count autograd nodes via BFS over grad_fn graph.
                seen = set()
                queue = [total_loss.grad_fn]
                while queue:
                    fn = queue.pop(0)
                    if fn is None or id(fn) in seen:
                        continue
                    seen.add(id(fn))
                    for child, _ in fn.next_functions:
                        queue.append(child)
                print(f"--- Autograd graph (batch 0) ---")
                print(f"  total loss: {len(seen)} nodes")

            t6 = time.perf_counter()

            optimizer.zero_grad()

            if probe: sync(); prof['zero_grad'] += time.perf_counter() - t6
            t7 = time.perf_counter()

            total_loss.backward()

            if probe: sync(); prof['backward'] += time.perf_counter() - t7
            t8 = time.perf_counter()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            if probe: sync(); prof['clip'] += time.perf_counter() - t8
            t9 = time.perf_counter()

            optimizer.step()

            if probe: sync(); prof['step'] += time.perf_counter() - t9
            t10 = time.perf_counter()

            with torch.no_grad():
                hr, hi = fixation_hit_rate(clean, locations)

            tracker.update(recon=recon_loss, letter_cls=letter_cls_loss,
                           case_cls=case_cls_loss, attn=attn_loss,
                           div=div_loss, content=content_loss,
                           void=void_loss,
                           recode=recode_loss_val, hit_rate=hr, hit_intensity=hi)

            if probe: sync(); prof['detach_metrics'] += time.perf_counter() - t10
            n_batches += 1

        if probe:
            total_probed = sum(prof.values())
            print(f"=== Epoch 0 profile ({n_batches} batches, {total_probed:.3f}s probed) ===")
            for k, v in prof.items():
                pct = v / total_probed * 100 if total_probed > 0 else 0
                print(f"  {k:16s} {v:8.3f}s  ({pct:.1f}%)")
            if use_cuda:
                alloc_mb = torch.cuda.max_memory_allocated() / 1024**2
                reserved_mb = torch.cuda.max_memory_reserved() / 1024**2
                free, total = torch.cuda.mem_get_info()
                used_mb = (total - free) / 1024**2
                total_mb = total / 1024**2
                print(f"  VRAM: alloc={alloc_mb:.0f}MB  reserved={reserved_mb:.0f}MB  "
                      f"device_used={used_mb:.0f}MB/{total_mb:.0f}MB")

        avgs = tracker.end_epoch()
        epoch_time = time.time() - epoch_start
        elapsed = time.time() - train_start
        done = epoch - start_epoch + 1
        eta = format_eta(elapsed, done, epochs - done)
        current_lr = scheduler.get_last_lr()[0]

        content_str = f"  Cont {avgs['content']:.4f}" if n_scan_glimpses > 0 else ""
        void_str = f"  Void {avgs['void']:.4f}" if (void_weight > 0 or scan_void_weight > 0) else ""
        vram_mb = torch.cuda.max_memory_allocated() / 1024**2 if use_cuda else 0
        vram_str = f"  VRAM {vram_mb:.0f}MB" if use_cuda else ""
        print(f"Epoch {epoch+1}/{end_epoch}: "
              f"Recon {avgs['recon']:.4f}  Ltr {avgs['letter_cls']:.4f}  "
              f"Case {avgs['case_cls']:.4f}  Attn {avgs['attn']:.4f}  "
              f"Div {avgs['div']:.4f}{content_str}{void_str}  Hit {avgs['hit_rate']:.0%}  "
              f"Recode {avgs['recode']:.4f}  "
              f"lr {current_lr:.6f}{vram_str}  "
              f"[{epoch_time:.1f}s  ETA {eta}]")

        content_log = f"  {avgs['content']:>7.4f}" if n_scan_glimpses > 0 else ""
        vram_log = f"  {vram_mb:>6.0f}MB" if use_cuda else ""
        logger.write_line(f"{epoch+1:>5d}  {avgs['recon']:>6.4f}  {avgs['letter_cls']:>6.4f}  "
                          f"{avgs['case_cls']:>6.4f}  {avgs['attn']:>7.4f}  {avgs['div']:>6.4f}  "
                          f"{avgs['hit_rate']:>6.4f}  {avgs['recode']:>6.4f}{content_log}  {current_lr:>8.6f}"
                          f"{vram_log}  {epoch_time:.1f}s")
        scheduler.step()

        if (epoch + 1 - start_epoch) % checkpoint_interval == 0:
            save_checkpoint(model, epoch,
                            os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth'),
                            cfg=cfg, losses_dict=tracker.get_history_dict(),
                            extra={
                                'n_glimpses': n_glimpses, 'patch_size': patch_size,
                                'n_scales': n_scales,
                                'n_scan_glimpses': n_scan_glimpses,
                                'scan_patch_size': scan_patch_size,
                                'learnable_scan_x': cfg.learnable_scan_x,
                                'image_size': 128, 'has_case': True,
                            })

    logger.close()

    save_checkpoint(model, end_epoch - 1,
                    os.path.join(save_dir, 'model_final.pth'),
                    cfg=cfg, losses_dict=tracker.get_history_dict(),
                    extra={
                        'n_glimpses': n_glimpses, 'patch_size': patch_size,
                        'n_scales': n_scales,
                        'n_scan_glimpses': n_scan_glimpses,
                        'scan_patch_size': scan_patch_size,
                        'learnable_scan_x': cfg.learnable_scan_x,
                        'image_size': 128, 'has_case': True,
                    })

    # Metrics plot
    has_content = n_scan_glimpses > 0 and any(v > 0 for v in tracker.history.get('content', []))
    specs = [
        {'keys': ['recon', 'recode'], 'labels': ['Recon', 'Recode'],
         'colors': ['tab:blue', 'tab:cyan'], 'styles': ['-', '--'],
         'title': 'Reconstruction', 'ylabel': 'MSE'},
        {'keys': ['letter_cls'], 'labels': ['Letter'], 'colors': ['tab:red'],
         'title': 'Letter Classification (26-class)', 'ylabel': 'Cross-Entropy',
         'hlines': [(np.log(26), f'Random ({np.log(26):.1f})', 'gray')]},
        {'keys': ['case_cls'], 'labels': ['Case'], 'colors': ['tab:pink'],
         'title': 'Case Classification (upper/lower)', 'ylabel': 'Cross-Entropy',
         'hlines': [(np.log(2), f'Random ({np.log(2):.2f})', 'gray')]},
        {'keys': ['attn'], 'labels': ['Guide'], 'colors': ['tab:green'],
         'title': 'Attention guide (lower = fixations on letter)', 'ylabel': 'Loss'},
        {'keys': ['div'], 'labels': ['Diversity'], 'colors': ['tab:orange'],
         'title': 'Fixation diversity (lower = more spread)', 'ylabel': 'Repulsion'},
    ]
    if has_content:
        specs.append({'keys': ['content'], 'labels': ['Content BCE'], 'colors': ['tab:cyan'],
                      'title': 'Content detection (scan phase)', 'ylabel': 'Loss'})
    specs.append({'keys': ['hit_rate', 'hit_intensity'],
                  'labels': ['Hit rate', 'Intensity'],
                  'colors': ['tab:purple', 'tab:purple'], 'styles': ['-', '--'],
                  'title': 'Fixation hit rate (on sharp letter pixels)',
                  'ylabel': 'Rate / Intensity', 'ylim': (0, 1)})
    plot_training_metrics(tracker, os.path.join(save_dir, 'training_metrics.png'), specs)

    total_time = time.time() - train_start
    total_min, total_s = divmod(int(total_time), 60)
    print(f"Training complete in {total_min}m{total_s:02d}s. "
          f"Model and graph saved in {save_dir}")


# --- Attention Pre-Check ---

def check_attention(data_dir, n_epochs=10, n_glimpses=10, patch_size=12,
                    n_scales=1, device='auto',
                    guide_weight=8.0, blur_sigma_ratio=0.16,
                    diversity_weight=1.0, diversity_sigma=0.1,
                    diversity_vy=1.0):
    """Quick diagnostic: can the attention guide pull fixations onto letter content?"""
    device = _resolve_device(device)
    dataset = LetterDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=26, shuffle=True,
                            pin_memory=device.type == 'cuda')

    model = VisionModel(
        n_glimpses=n_glimpses, patch_size=patch_size, n_scales=n_scales,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    sample_img = dataset[0][0]
    img_h, img_w = sample_img.shape[1], sample_img.shape[2]
    effective_sigma = blur_sigma_ratio * min(img_h, img_w)

    vy_str = f"  diversity_vy={diversity_vy}" if diversity_vy != 1.0 else ""
    print(f"Attention pre-check on {device}")
    print(f"Image: {img_h}x{img_w}  blur_sigma_ratio={blur_sigma_ratio} "
          f"-> {effective_sigma:.1f}px  guide_weight={guide_weight}{vy_str}")
    print(f"Running {n_epochs} diagnostic epochs (attention + diversity only)...")

    hit_rates = []
    for epoch in range(n_epochs):
        total_hr = 0
        total_attn = 0
        n = 0
        for img, clean, _letters, cases, _fonts, _partner in dataloader:
            img = img.to(device)
            clean = clean.to(device)
            case_idx = torch.tensor(
                [0 if c == 'upper' else 1 for c in cases], device=device,
            )
            case_float = case_idx.float().unsqueeze(1)

            _, _, _, locations, _, _ = model(img, case_float)

            attn_loss = attention_content_loss(
                clean, locations, blur_sigma_ratio=blur_sigma_ratio,
            )
            div_loss = fixation_diversity_loss(locations, sigma=diversity_sigma,
                                                vy=diversity_vy)
            loss = guide_weight * attn_loss + diversity_weight * div_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            with torch.no_grad():
                hr, _ = fixation_hit_rate(clean, locations)
                total_hr += hr
                total_attn += attn_loss.item()
            n += 1

        avg_hr = total_hr / n
        avg_attn = total_attn / n
        hit_rates.append(avg_hr)
        print(f"  Epoch {epoch+1}/{n_epochs}: Hit {avg_hr:.0%}  Attn {avg_attn:.4f}")

    initial_hr = hit_rates[0]
    final_hr = hit_rates[-1]
    peak_hr = max(hit_rates)
    improved = peak_hr > initial_hr + 0.05

    print()
    if peak_hr >= 0.20:
        print(f"PASS: Hit rate {initial_hr:.0%} -> {final_hr:.0%} "
              f"(peak {peak_hr:.0%}). Attention guide is working.")
        return True
    elif improved:
        print(f"WEAK: Hit rate {initial_hr:.0%} -> {final_hr:.0%} "
              f"(peak {peak_hr:.0%}). Improving but low — consider "
              f"increasing guide_weight (current: {guide_weight}).")
        return True
    else:
        print(f"FAIL: Hit rate {initial_hr:.0%} -> {final_hr:.0%} "
              f"(peak {peak_hr:.0%}). Attention guide has no effect. "
              f"Increase blur_sigma_ratio (current: {blur_sigma_ratio}) "
              f"or guide_weight (current: {guide_weight}).")
        return False


# --- Bigram Training ---

def train_bigram_model(cfg):
    """Train a BigramVisionModel from an ExperimentConfig."""
    n_scan_glimpses = cfg.n_scan_glimpses
    n_read_glimpses = cfg.n_read_glimpses
    scan_patch_size = cfg.scan_patch_size
    read_patch_size = cfg.read_patch_size
    n_scales = cfg.n_scales
    scan_vy = cfg.scan_vy
    read_vy = cfg.read_vy
    scan_guide_weight = cfg.scan_guide_weight
    guide_weight = cfg.guide_weight
    blur_sigma_ratio = cfg.blur_sigma_ratio
    diversity_weight = cfg.diversity_weight
    diversity_sigma = cfg.diversity_sigma
    batch_size = cfg.batch_size
    epochs = cfg.epochs
    resume = cfg.resume
    save_dir = cfg.save_dir
    data_dir = cfg.data_dir
    checkpoint_interval = cfg.checkpoint_interval
    scaffold_epochs = cfg.scaffold_epochs
    scaffold_floor = cfg.scaffold_floor
    transfer_from = cfg.transfer
    mask_weight = cfg.mask_weight
    edge_weight = cfg.edge_weight

    device = _resolve_device(cfg.device)
    n_glimpses = n_scan_glimpses + n_read_glimpses
    print(f"Bigram training on: {device}")
    edge_str = f"  edge_weight={edge_weight}" if edge_weight > 0 else ""
    print(f"Two-phase: scan={n_scan_glimpses} ({scan_patch_size}) + "
          f"read={n_read_glimpses} ({read_patch_size}) = {n_glimpses} glimpses")
    print(f"Attention: guide_weight={guide_weight}  scan_guide={scan_guide_weight}  "
          f"blur_sigma_ratio={blur_sigma_ratio}  "
          f"diversity_weight={diversity_weight}  diversity_sigma={diversity_sigma}  "
          f"scan_vy={scan_vy}  read_vy={read_vy}  "
          f"batch_size={batch_size}{edge_str}")
    floor_str = f", floor={scaffold_floor}" if scaffold_floor > 0 else ""
    print(f"Temporal scaffold (read phase): {scaffold_epochs} epochs "
          f"({'disabled' if scaffold_epochs == 0 else 'left→right→holistic'}){floor_str}")

    os.makedirs(save_dir, exist_ok=True)
    save_run_info(save_dir, cfg)
    dataset = BigramDataset(data_dir)
    use_cuda = device.type == 'cuda'
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=use_cuda)

    model = BigramVisionModel(
        n_scan_glimpses=n_scan_glimpses, n_read_glimpses=n_read_glimpses,
        scan_patch_size=scan_patch_size, read_patch_size=read_patch_size,
        n_scales=n_scales,
    ).to(device)

    # Transfer learning
    transfer_scaffold = False
    if transfer_from and not resume:
        n_positions = 2
        broadcast = [
            ('letter_classifier.weight', [f'classifiers.{p}.weight' for p in range(n_positions)]),
            ('letter_classifier.bias', [f'classifiers.{p}.bias' for p in range(n_positions)]),
        ]
        apply_transfer(model, transfer_from,
                        key_mappings=[
                            ('encoder.glimpse_sensor.', 'read_sensor.'),
                            ('encoder.attention_controller.', 'controller.'),
                        ],
                        device=device,
                        broadcast_keys=broadcast,
                        freeze_keys=['read_sensor.', 'classifiers.'] if scaffold_epochs > 0 else None)
        if scaffold_epochs > 0:
            transfer_scaffold = True
            print(f"Transfer scaffold: read_sensor + classifiers frozen for {scaffold_epochs} epochs")
    elif transfer_from and resume:
        print("Warning: --transfer ignored when --resume is used")

    # Loss tracking
    tracker = LossTracker(['recon', 'pos1_cls', 'pos2_cls', 'attn', 'div',
                           'hit_rate', 'hit_intensity'])
    start_epoch = 0

    if resume:
        checkpoint = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model'], strict=False)
        start_epoch = checkpoint['epoch'] + 1
        if 'losses' in checkpoint:
            tracker.restore_history(checkpoint['losses'])
        print(f"Resumed from epoch {start_epoch} ({len(tracker.history['pos1_cls'])} prior epochs of history)")

    # Optimizer
    if transfer_scaffold:
        optimizer = optim.Adam([
            {'params': list(model.scan_sensor.parameters()), 'lr': 0.001},
            {'params': list(model.controller.parameters()), 'lr': 0.0001},
            {'params': list(model.readout.parameters()) + list(model.decoder.parameters()), 'lr': 0.001},
        ])
    else:
        sensor_ctrl_params = set(
            list(model.scan_sensor.parameters()) +
            list(model.read_sensor.parameters()) +
            list(model.controller.parameters())
        )
        readout_params = [p for p in model.parameters() if p not in sensor_ctrl_params]
        optimizer = optim.Adam([
            {'params': list(sensor_ctrl_params), 'lr': 0.0001},
            {'params': readout_params, 'lr': 0.001},
        ])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = torch.nn.MSELoss()

    end_epoch = start_epoch + epochs
    train_start = time.time()

    header = (f"{'epoch':>5s}  {'recon':>6s}  {'pos1':>6s}  {'pos2':>6s}  {'s_attn':>7s}  {'r_attn':>7s}  "
              f"{'div':>6s}  {'hit':>6s}  {'lr_enc':>8s}  {'lr_read':>8s}  {'scaff':>6s}  time")
    logger = TrainingLogger(save_dir, header, start_epoch)

    for epoch in range(start_epoch, end_epoch):
        if transfer_scaffold and epoch >= scaffold_epochs:
            transfer_scaffold = False
            for p in model.read_sensor.parameters():
                p.requires_grad = True
            for clf in model.classifiers:
                for p in clf.parameters():
                    p.requires_grad = True
            optimizer = optim.Adam([
                {'params': list(model.scan_sensor.parameters()), 'lr': 0.0001},
                {'params': list(model.read_sensor.parameters()), 'lr': 0.00001},
                {'params': list(model.controller.parameters()), 'lr': 0.0001},
                {'params': [p for clf in model.classifiers for p in clf.parameters()], 'lr': 0.0001},
                {'params': list(model.readout.parameters()) + list(model.decoder.parameters()), 'lr': 0.001},
            ])
            remaining = end_epoch - epoch
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=remaining)
            print(f"Transfer: unfroze read_sensor + classifiers, 5 param groups for {remaining} epochs")

        epoch_start = time.time()
        tracker.reset_epoch()
        total_pos1_correct = 0
        total_pos2_correct = 0
        total_samples = 0

        if scaffold_epochs > 0:
            scaffold_weight = max(scaffold_floor, 1.0 - epoch / scaffold_epochs)
        else:
            scaffold_weight = scaffold_floor

        for img, clean, letter1s, letter2s, _bigrams, _fonts in dataloader:
            img = img.to(device)
            clean = clean.to(device)

            idx1 = torch.tensor([ord(l) - ord('a') for l in letter1s], device=device)
            idx2 = torch.tensor([ord(l) - ord('a') for l in letter2s], device=device)

            recon, logits_list, locations, _readout_states = model(img)

            recon_loss = criterion(recon, img)
            pos1_cls_loss = F.cross_entropy(logits_list[0], idx1)
            pos2_cls_loss = F.cross_entropy(logits_list[1], idx2)

            scan_attn, read_attn = two_phase_attention_loss(
                clean, locations, n_scan_glimpses,
                blur_sigma_ratio=blur_sigma_ratio,
                scaffold_weight=scaffold_weight,
            )

            scan_locations = locations[:n_scan_glimpses + 1]
            read_locations = locations[n_scan_glimpses:]
            scan_div = fixation_diversity_loss(scan_locations, sigma=diversity_sigma,
                                               vy=scan_vy)
            read_div = fixation_diversity_loss(read_locations, sigma=diversity_sigma,
                                               vy=read_vy)
            div_loss = scan_div + read_div

            if edge_weight > 0:
                edge_loss = fixation_edge_loss(scan_locations)
            else:
                edge_loss = torch.tensor(0.0)

            total_loss = (recon_loss + pos1_cls_loss + pos2_cls_loss
                          + scan_guide_weight * scan_attn
                          + guide_weight * read_attn
                          + edge_weight * edge_loss
                          + diversity_weight * div_loss)

            if mask_weight > 0:
                mid_w = img.shape[3] // 2
                mask_left = torch.rand(img.shape[0], device=device) < 0.5
                spatial_mask = torch.ones_like(img)
                spatial_mask[mask_left, :, :, :mid_w] = 0.0
                spatial_mask[~mask_left, :, :, mid_w:] = 0.0
                masked_img = img * spatial_mask
                _, logits_m, _, _ = model(masked_img)
                mask_cls = torch.tensor(0.0, device=device)
                if mask_left.any():
                    mask_cls = mask_cls + F.cross_entropy(logits_m[1][mask_left], idx2[mask_left])
                if (~mask_left).any():
                    mask_cls = mask_cls + F.cross_entropy(logits_m[0][~mask_left], idx1[~mask_left])
                total_loss = total_loss + mask_weight * mask_cls

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            with torch.no_grad():
                total_pos1_correct += (logits_list[0].argmax(1) == idx1).sum().item()
                total_pos2_correct += (logits_list[1].argmax(1) == idx2).sum().item()
                total_samples += img.shape[0]
                hr, hi = fixation_hit_rate(clean, locations)

            tracker.update(recon=recon_loss, pos1_cls=pos1_cls_loss,
                           pos2_cls=pos2_cls_loss,
                           attn=scan_attn.item() + read_attn.item(),
                           div=div_loss, hit_rate=hr, hit_intensity=hi)

        avgs = tracker.end_epoch()
        acc1 = total_pos1_correct / total_samples if total_samples > 0 else 0
        acc2 = total_pos2_correct / total_samples if total_samples > 0 else 0

        epoch_time = time.time() - epoch_start
        elapsed = time.time() - train_start
        done = epoch - start_epoch + 1
        eta = format_eta(elapsed, done, epochs - done)

        lrs = scheduler.get_last_lr()
        lr_enc, lr_read = lrs[0], lrs[-1]
        # Split combined attn into scan/read for display (recompute from latest batch)
        avg_scan_attn = scan_attn.item() if torch.is_tensor(scan_attn) else 0
        avg_read_attn = read_attn.item() if torch.is_tensor(read_attn) else 0
        scaff_str = f"  scaff {scaffold_weight:.2f}" if scaffold_epochs > 0 else ""
        print(f"Epoch {epoch+1}/{end_epoch}: "
              f"Recon {avgs['recon']:.4f}  Pos1 {avgs['pos1_cls']:.4f} ({acc1:.0%})  "
              f"Pos2 {avgs['pos2_cls']:.4f} ({acc2:.0%})  "
              f"Attn {avgs['attn']:.4f}  "
              f"Div {avgs['div']:.4f}  Hit {avgs['hit_rate']:.0%}  "
              f"lr {lr_enc:.6f}/{lr_read:.6f}{scaff_str}  "
              f"[{epoch_time:.1f}s  ETA {eta}]")

        logger.write_line(f"{epoch+1:>5d}  {avgs['recon']:>6.4f}  {avgs['pos1_cls']:>6.4f}  "
                          f"{avgs['pos2_cls']:>6.4f}  {avg_scan_attn:>7.4f}  {avg_read_attn:>7.4f}  "
                          f"{avgs['div']:>6.4f}  "
                          f"{avgs['hit_rate']:>6.4f}  {lr_enc:>8.6f}  {lr_read:>8.6f}  "
                          f"{scaffold_weight:>6.4f}  {epoch_time:.1f}s")
        scheduler.step()

        if (epoch + 1 - start_epoch) % checkpoint_interval == 0:
            save_checkpoint(model, epoch,
                            os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth'),
                            cfg=cfg, losses_dict=tracker.get_history_dict(),
                            extra={
                                'model_type': 'bigram',
                                'n_scan_glimpses': n_scan_glimpses,
                                'n_read_glimpses': n_read_glimpses,
                                'scan_patch_size': scan_patch_size,
                                'read_patch_size': read_patch_size,
                                'n_glimpses': n_glimpses,
                                'n_scales': n_scales,
                                'image_size': (128, 128),
                            })

    logger.close()

    save_checkpoint(model, end_epoch - 1,
                    os.path.join(save_dir, 'model_final.pth'),
                    cfg=cfg, losses_dict=tracker.get_history_dict(),
                    extra={
                        'model_type': 'bigram',
                        'n_scan_glimpses': n_scan_glimpses,
                        'n_read_glimpses': n_read_glimpses,
                        'scan_patch_size': scan_patch_size,
                        'read_patch_size': read_patch_size,
                        'n_glimpses': n_glimpses,
                        'n_scales': n_scales,
                        'image_size': (128, 128),
                    })

    specs = [
        {'keys': ['recon'], 'labels': ['Recon'], 'colors': ['tab:blue'],
         'title': 'Reconstruction (128x128)', 'ylabel': 'MSE'},
        {'keys': ['pos1_cls', 'pos2_cls'], 'labels': ['Pos 1', 'Pos 2'],
         'colors': ['tab:red', 'tab:orange'], 'styles': ['-', '--'],
         'title': 'Letter Classification (26-class, per position)', 'ylabel': 'Cross-Entropy',
         'hlines': [(np.log(26), f'Random ({np.log(26):.1f})', 'gray')]},
        {'keys': ['attn'], 'labels': ['Guide (scan+read)'], 'colors': ['tab:green'],
         'title': 'Attention guide (lower = fixations on letters)', 'ylabel': 'Loss'},
        {'keys': ['div'], 'labels': ['Diversity'], 'colors': ['tab:orange'],
         'title': 'Fixation diversity (lower = more spread)', 'ylabel': 'Repulsion'},
        {'keys': ['hit_rate', 'hit_intensity'], 'labels': ['Hit rate', 'Intensity'],
         'colors': ['tab:purple', 'tab:purple'], 'styles': ['-', '--'],
         'title': 'Fixation hit rate (on sharp letter pixels)',
         'ylabel': 'Rate / Intensity', 'ylim': (0, 1)},
    ]
    plot_training_metrics(tracker, os.path.join(save_dir, 'training_metrics.png'), specs)

    total_time = time.time() - train_start
    total_min, total_s = divmod(int(total_time), 60)
    print(f"Training complete in {total_min}m{total_s:02d}s. "
          f"Model and graph saved in {save_dir}")


# --- Bigram Attention Pre-Check ---

def check_bigram_attention(data_dir, n_epochs=10,
                           n_scan_glimpses=5, n_read_glimpses=6,
                           scan_patch_size=(12, 18), read_patch_size=12,
                           n_scales=1, device='auto',
                           guide_weight=8.0, blur_sigma_ratio=0.16,
                           diversity_weight=1.0, diversity_sigma=0.1,
                           diversity_vy=1.0):
    """Quick diagnostic: can the attention guide pull fixations onto bigram letter content?"""
    device = _resolve_device(device)
    dataset = BigramDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True,
                            pin_memory=device.type == 'cuda')

    model = BigramVisionModel(
        n_scan_glimpses=n_scan_glimpses, n_read_glimpses=n_read_glimpses,
        scan_patch_size=scan_patch_size, read_patch_size=read_patch_size,
        n_scales=n_scales,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    sample_img = dataset[0][0]
    img_h, img_w = sample_img.shape[1], sample_img.shape[2]
    effective_sigma = blur_sigma_ratio * min(img_h, img_w)

    vy_str = f"  diversity_vy={diversity_vy}" if diversity_vy != 1.0 else ""
    print(f"Bigram attention pre-check on {device}")
    print(f"Image: {img_h}x{img_w}  blur_sigma_ratio={blur_sigma_ratio} "
          f"-> {effective_sigma:.1f}px  guide_weight={guide_weight}{vy_str}")
    print(f"Running {n_epochs} diagnostic epochs (attention + diversity only)...")

    hit_rates = []
    for epoch in range(n_epochs):
        total_hr = 0
        total_attn = 0
        n = 0
        for img, clean, _l1, _l2, _bigrams, _fonts in dataloader:
            img = img.to(device)
            clean = clean.to(device)

            _, _, locations, _ = model(img)

            attn_loss = attention_content_loss(
                clean, locations, blur_sigma_ratio=blur_sigma_ratio,
            )
            div_loss = fixation_diversity_loss(locations, sigma=diversity_sigma,
                                                vy=diversity_vy)
            loss = guide_weight * attn_loss + diversity_weight * div_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            with torch.no_grad():
                hr, _ = fixation_hit_rate(clean, locations)
                total_hr += hr
                total_attn += attn_loss.item()
            n += 1

        avg_hr = total_hr / n
        avg_attn = total_attn / n
        hit_rates.append(avg_hr)
        print(f"  Epoch {epoch+1}/{n_epochs}: Hit {avg_hr:.0%}  Attn {avg_attn:.4f}")

    initial_hr = hit_rates[0]
    final_hr = hit_rates[-1]
    peak_hr = max(hit_rates)
    improved = peak_hr > initial_hr + 0.05

    print()
    if peak_hr >= 0.20:
        print(f"PASS: Hit rate {initial_hr:.0%} -> {final_hr:.0%} "
              f"(peak {peak_hr:.0%}). Attention guide is working.")
        return True
    elif improved:
        print(f"WEAK: Hit rate {initial_hr:.0%} -> {final_hr:.0%} "
              f"(peak {peak_hr:.0%}). Improving but low — consider "
              f"increasing guide_weight (current: {guide_weight}).")
        return True
    else:
        print(f"FAIL: Hit rate {initial_hr:.0%} -> {final_hr:.0%} "
              f"(peak {peak_hr:.0%}). Attention guide has no effect. "
              f"Increase blur_sigma_ratio (current: {blur_sigma_ratio}) "
              f"or guide_weight (current: {guide_weight}).")
        return False
