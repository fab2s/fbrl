"""Word training functions — imported by train.py."""
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import numpy as np
import matplotlib.pyplot as plt
import os
import time

import random
from torch.nn.utils import clip_grad_norm_

from fbrl import _resolve_device
from fbrl.data import WordDataset, IsolationLetterDataset
from fbrl.model import WordVisionModel
from fbrl.losses import (word_attention_loss, fixation_diversity_loss,
                          fixation_hit_rate)


def train_word_model(data_dir, epochs=200, resume=None, save_dir='word_models',
                      checkpoint_interval=10,
                      n_scan_glimpses=8, n_read_glimpses=12,
                      scan_patch_size=(12, 18), read_patch_size=12,
                      n_scales=1, n_positions=4, device='auto',
                      diversity_weight=1.0, diversity_sigma=0.1,
                      scan_vy=0.3, read_vy=1.5,
                      guide_weight=8.0, scan_guide_weight=None,
                      blur_sigma_ratio=0.16,
                      batch_size=32, scaffold_epochs=200,
                      scaffold_floor=0.0,
                      transfer_from=None,
                      content_weight=0.5,
                      isolation_weight=0.5,
                      edge_weight=0.0,
                      isolation_data_dir=None,
                      isolation_random_prob=0.0,
                      multi_head=False,
                      amp=False):
    """Train a WordVisionModel with prescribed x-scan + free read on 256x128 word images.

    Phase 1 — SCAN: prescribed x sweep, learned y, content detection.
      scan_vy < 1.0 makes horizontal proximity expensive (forces horizontal spread).
    Phase 2 — READ: fully free x,y on focused patches.
      read_vy > 1.0 makes vertical proximity expensive (forces vertical exploration).

    Content detection: BCE loss on scan hidden states predicting whether
    letter content exists at each scan location. Weight controlled by content_weight.
    """
    if scan_guide_weight is None:
        scan_guide_weight = guide_weight

    device = _resolve_device(device)
    n_glimpses = n_scan_glimpses + n_read_glimpses
    print(f"Word training on: {device}")
    print(f"Two-phase: scan={n_scan_glimpses} (prescribed x, {scan_patch_size}) + "
          f"read={n_read_glimpses} ({read_patch_size}) = {n_glimpses} glimpses")
    iso_mode = f"128px ({isolation_data_dir})" if isolation_data_dir else "mask"
    print(f"Attention: guide_weight={guide_weight}  scan_guide={scan_guide_weight}  "
          f"blur_sigma_ratio={blur_sigma_ratio}  "
          f"diversity_weight={diversity_weight}  diversity_sigma={diversity_sigma}  "
          f"scan_vy={scan_vy}  read_vy={read_vy}  "
          f"content_weight={content_weight}  isolation_weight={isolation_weight} ({iso_mode})  "
          f"batch_size={batch_size}  multi_head={multi_head}")
    floor_str = f", floor={scaffold_floor}" if scaffold_floor > 0 else ""
    print(f"Temporal scaffold (read phase): {scaffold_epochs} epochs "
          f"({'disabled' if scaffold_epochs == 0 else f'{n_positions}-stripe L->R'}){floor_str}")

    os.makedirs(save_dir, exist_ok=True)
    dataset = WordDataset(data_dir)
    use_cuda = device.type == 'cuda'
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=use_cuda)

    # Isolation dataset: 128x128 single-letter images
    iso_dataset = None
    if isolation_data_dir and isolation_weight > 0:
        iso_dataset = IsolationLetterDataset(isolation_data_dir)

    model = WordVisionModel(
        n_scan_glimpses=n_scan_glimpses, n_read_glimpses=n_read_glimpses,
        scan_patch_size=scan_patch_size, read_patch_size=read_patch_size,
        n_scales=n_scales, n_positions=n_positions,
    ).to(device)

    # --- Transfer learning from single-letter model ---
    transfer_scaffold = False
    if transfer_from and not resume:
        import gzip, io as _io
        if transfer_from.endswith('.gz'):
            with gzip.open(transfer_from, 'rb') as f:
                src = torch.load(_io.BytesIO(f.read()), map_location=device,
                                 weights_only=False)['model']
        else:
            src = torch.load(transfer_from, map_location=device,
                             weights_only=False)['model']

        dst = model.state_dict()
        n_transferred = 0
        for key in src:
            # encoder.glimpse_sensor.* -> read_sensor.*
            if key.startswith('encoder.glimpse_sensor.'):
                new_key = key.replace('encoder.glimpse_sensor.', 'read_sensor.')
                if new_key in dst:
                    dst[new_key] = src[key].float()
                    n_transferred += 1
            # encoder.attention_controller.* -> controller.*
            elif key.startswith('encoder.attention_controller.'):
                new_key = key.replace('encoder.attention_controller.', 'controller.')
                if new_key in dst:
                    dst[new_key] = src[key].float()
                    n_transferred += 1
            # scan_sensor.* -> scan_sensor.* (direct, if source has scan phase)
            elif key.startswith('scan_sensor.'):
                if key in dst:
                    dst[key] = src[key].float()
                    n_transferred += 1
            # content_head.* -> content_head.* (direct)
            elif key.startswith('content_head.'):
                if key in dst:
                    dst[key] = src[key].float()
                    n_transferred += 1
        # Single-letter letter_classifier -> all position classifiers
        for suffix in ('weight', 'bias'):
            sk = f'letter_classifier.{suffix}'
            if sk in src:
                for p in range(n_positions):
                    dst[f'classifiers.{p}.{suffix}'] = src[sk].float()
                    n_transferred += 1
        model.load_state_dict(dst)
        print(f"Transfer: {n_transferred} tensors from {transfer_from}")

        # Freeze read_sensor + classifiers during scaffold phase
        if scaffold_epochs > 0:
            transfer_scaffold = True
            for p in model.read_sensor.parameters():
                p.requires_grad = False
            for clf in model.classifiers:
                for p in clf.parameters():
                    p.requires_grad = False
            print(f"Transfer scaffold: read_sensor + classifiers frozen for {scaffold_epochs} epochs")
    elif transfer_from and resume:
        print("Warning: --transfer ignored when --resume is used")

    start_epoch = 0
    losses_recon = []
    losses_pos_cls = [[] for _ in range(n_positions)]
    losses_attn = []
    losses_div = []
    losses_content = []
    losses_isolation = []
    hist_hit_rate = []
    hist_hit_intensity = []

    if resume:
        checkpoint = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model'], strict=False)
        start_epoch = checkpoint['epoch'] + 1
        if 'losses' in checkpoint:
            h = checkpoint['losses']
            losses_recon = h.get('recon', [])
            for p in range(n_positions):
                losses_pos_cls[p] = h.get(f'pos{p+1}_cls', [])
            losses_attn = h.get('attn', [])
            losses_div = h.get('div', [])
            losses_content = h.get('content', [])
            losses_isolation = h.get('isolation', [])
            hist_hit_rate = h.get('hit_rate', [])
            hist_hit_intensity = h.get('hit_intensity', [])

        # Restore scaffold state from checkpoint (or infer from CLI args)
        saved_scaffold = checkpoint.get('scaffold_epochs', 0)
        if saved_scaffold == 0 and scaffold_epochs > 0 and start_epoch < scaffold_epochs:
            # Old checkpoint without scaffold_epochs field — infer from CLI
            saved_scaffold = scaffold_epochs
            print(f"Resume: scaffold_epochs not in checkpoint, using CLI value {scaffold_epochs}")
        if saved_scaffold > 0 and start_epoch < saved_scaffold:
            transfer_scaffold = True
            scaffold_epochs = saved_scaffold
            for p in model.read_sensor.parameters():
                p.requires_grad = False
            for clf in model.classifiers:
                for p in clf.parameters():
                    p.requires_grad = False
            print(f"Resume: scaffold active (read_sensor + classifiers frozen until epoch {scaffold_epochs})")
        elif saved_scaffold > 0:
            print(f"Resume: scaffold complete (ended at epoch {saved_scaffold})")

        print(f"Resumed from epoch {start_epoch} ({len(losses_recon)} prior epochs of history)")

    # --- Optimizer setup ---
    criterion = torch.nn.MSELoss()
    bce = torch.nn.BCEWithLogitsLoss()

    def _make_multi_head_optimizers(scaffold_phase, remaining_epochs):
        """Create 3 separate optimizers for attention/classification/reconstruction."""
        if scaffold_phase:
            attn_opt = optim.Adam([
                {'params': list(model.scan_sensor.parameters()), 'lr': 0.001},
                {'params': list(model.controller.parameters()), 'lr': 0.0001},
                {'params': list(model.content_head.parameters()), 'lr': 0.001},
            ])
            cls_opt = optim.Adam([
                {'params': list(model.readout.parameters()), 'lr': 0.001},
            ])
            recon_opt = optim.Adam([
                {'params': list(model.decoder.parameters()), 'lr': 0.001},
            ])
            print(f"Multi-head (scaffold): attn[scan=0.001,ctrl=0.0001,content=0.001] "
                  f"cls[readout=0.001] recon[decoder=0.001]")
        else:
            attn_opt = optim.Adam([
                {'params': list(model.scan_sensor.parameters()), 'lr': 0.0001},
                {'params': list(model.read_sensor.parameters()), 'lr': 0.00001},
                {'params': list(model.controller.parameters()), 'lr': 0.0001},
                {'params': list(model.content_head.parameters()), 'lr': 0.0001},
            ])
            cls_opt = optim.Adam([
                {'params': list(model.readout.parameters()), 'lr': 0.001},
                {'params': [p for clf in model.classifiers for p in clf.parameters()], 'lr': 0.0001},
            ])
            recon_opt = optim.Adam([
                {'params': list(model.decoder.parameters()), 'lr': 0.001},
            ])
            print(f"Multi-head (full): attn[scan=0.0001,read=0.00001,ctrl=0.0001,content=0.0001] "
                  f"cls[readout=0.001,classifiers=0.0001] recon[decoder=0.001]")
        attn_sched = optim.lr_scheduler.CosineAnnealingLR(attn_opt, T_max=remaining_epochs)
        cls_sched = optim.lr_scheduler.CosineAnnealingLR(cls_opt, T_max=remaining_epochs)
        recon_sched = optim.lr_scheduler.CosineAnnealingLR(recon_opt, T_max=remaining_epochs)
        return attn_opt, cls_opt, recon_opt, attn_sched, cls_sched, recon_sched

    def _make_single_optimizer(scaffold_phase, remaining_epochs):
        """Create single combined optimizer (original behavior)."""
        if scaffold_phase:
            opt = optim.Adam([
                {'params': list(model.scan_sensor.parameters()), 'lr': 0.001},
                {'params': list(model.controller.parameters()), 'lr': 0.0001},
                {'params': list(model.content_head.parameters()), 'lr': 0.001},
                {'params': list(model.readout.parameters()) + list(model.decoder.parameters()), 'lr': 0.001},
            ])
            print(f"Single opt (scaffold): scan=0.001, ctrl=0.0001, content+readout+decoder=0.001")
        else:
            sensor_ctrl_params = set(
                list(model.scan_sensor.parameters()) +
                list(model.read_sensor.parameters()) +
                list(model.controller.parameters()) +
                list(model.content_head.parameters())
            )
            readout_params = [p for p in model.parameters() if p not in sensor_ctrl_params]
            opt = optim.Adam([
                {'params': list(sensor_ctrl_params), 'lr': 0.0001},
                {'params': readout_params, 'lr': 0.001},
            ])
            print(f"Single opt: sensors+ctrl+content=0.0001, readout=0.001")
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=remaining_epochs)
        return opt, sched

    # Collect param lists for multi-head backward(inputs=...)
    attn_params = (list(model.scan_sensor.parameters()) +
                   list(model.read_sensor.parameters()) +
                   list(model.controller.parameters()) +
                   list(model.content_head.parameters()))
    cls_params = (list(model.readout.parameters()) +
                  [p for clf in model.classifiers for p in clf.parameters()])
    recon_params = list(model.decoder.parameters())

    # Initial optimizer creation
    if multi_head:
        attn_opt, cls_opt, recon_opt, attn_sched, cls_sched, recon_sched = \
            _make_multi_head_optimizers(transfer_scaffold, epochs)
        # Restore optimizer states from checkpoint if available
        if resume and checkpoint.get('multi_head', False):
            if 'attn_optimizer' in checkpoint:
                attn_opt.load_state_dict(checkpoint['attn_optimizer'])
                cls_opt.load_state_dict(checkpoint['cls_optimizer'])
                recon_opt.load_state_dict(checkpoint['recon_optimizer'])
                attn_sched.load_state_dict(checkpoint['attn_scheduler'])
                cls_sched.load_state_dict(checkpoint['cls_scheduler'])
                recon_sched.load_state_dict(checkpoint['recon_scheduler'])
                print(f"Resume: restored multi-head optimizer + scheduler states")
            else:
                print(f"Warning: checkpoint has no optimizer states, starting fresh")
    else:
        optimizer, scheduler = _make_single_optimizer(transfer_scaffold, epochs)

    end_epoch = start_epoch + epochs
    train_start = time.time()

    # AMP (Automatic Mixed Precision) — FP16 activations, halves memory
    use_amp = amp and device.type == 'cuda'
    scaler = GradScaler('cuda', enabled=use_amp)
    if use_amp:
        print("AMP enabled: FP16 forward passes, GradScaler for backward")

    # Training log file
    log_path = os.path.join(save_dir, 'training.log')
    if start_epoch == 0 and os.path.exists(log_path):
        from datetime import datetime
        ts = datetime.fromtimestamp(os.path.getmtime(log_path)).strftime('%Y%m%d_%H%M%S')
        os.rename(log_path, os.path.join(save_dir, f'training_{ts}.log'))
    log_file = open(log_path, 'a')
    if start_epoch == 0:
        pos_hdrs = '  '.join(f'{"pos"+str(p+1):>6s}' for p in range(n_positions))
        if multi_head:
            log_file.write(f"epoch   recon  {pos_hdrs}   s_attn   r_attn      div  content   isolat      hit   lr_attn    lr_cls  lr_rcon    scaff  time\n")
        else:
            log_file.write(f"epoch   recon  {pos_hdrs}   s_attn   r_attn      div  content   isolat      hit    lr_enc   lr_read    scaff  time\n")
        log_file.write("-" * 160 + "\n")
    log_file.flush()

    for epoch in range(start_epoch, end_epoch):
        # Transfer: unfreeze read_sensor + classifiers when scaffold phase ends
        if transfer_scaffold and epoch >= scaffold_epochs:
            transfer_scaffold = False
            for p in model.read_sensor.parameters():
                p.requires_grad = True
            for clf in model.classifiers:
                for p in clf.parameters():
                    p.requires_grad = True
            remaining = end_epoch - epoch
            if multi_head:
                attn_opt, cls_opt, recon_opt, attn_sched, cls_sched, recon_sched = \
                    _make_multi_head_optimizers(False, remaining)
            else:
                optimizer, scheduler = _make_single_optimizer(False, remaining)
            print(f"Transfer: unfroze read_sensor + classifiers for {remaining} epochs")

        epoch_start = time.time()
        total_loss_recon = 0
        total_loss_pos = [0.0] * n_positions
        total_loss_scan_attn = 0
        total_loss_read_attn = 0
        total_loss_div = 0
        total_loss_content = 0
        total_loss_isolation = 0
        total_hit_rate = 0
        total_hit_intensity = 0
        total_pos_correct = [0] * n_positions
        total_samples = 0

        # Scaffold annealing
        if scaffold_epochs > 0:
            scaffold_weight = max(scaffold_floor, 1.0 - epoch / scaffold_epochs)
        else:
            scaffold_weight = scaffold_floor

        for img, clean, l1s, l2s, l3s, l4s, _words, fonts_batch in dataloader:
            img = img.to(device)
            clean = clean.to(device)
            B = img.shape[0]

            # Labels: a=0 .. z=25
            letter_lists = [l1s, l2s, l3s, l4s]
            idx_list = [
                torch.tensor([ord(l) - ord('a') for l in letters], device=device)
                for letters in letter_lists[:n_positions]
            ]

            # Forward + losses under AMP autocast
            with autocast('cuda', enabled=use_amp):
                recon, logits_list, locations, _readout_states, scan_content_logits = model(img)

                # Actual scan count (may differ from n_scan_glimpses for non-256 images)
                actual_n_scan = len(scan_content_logits)

                # Losses
                recon_loss = criterion(recon, img)
                cls_losses = []
                for p in range(n_positions):
                    cls_losses.append(F.cross_entropy(logits_list[p], idx_list[p]))

                # Word attention loss: scan gets full-image guide, read gets stripe scaffold
                scan_attn, read_attn = word_attention_loss(
                    clean, locations, actual_n_scan,
                    n_positions=n_positions,
                    blur_sigma_ratio=blur_sigma_ratio,
                    scaffold_weight=scaffold_weight,
                )

                # Split diversity by phase
                scan_locations = locations[:actual_n_scan + 1]
                read_locations = locations[actual_n_scan:]
                scan_div = fixation_diversity_loss(scan_locations, sigma=diversity_sigma,
                                                   vy=scan_vy)
                read_div = fixation_diversity_loss(read_locations, sigma=diversity_sigma,
                                                   vy=read_vy)
                div_loss = scan_div + read_div

                # Content detection loss: sample clean image at scan locations
                content_loss = torch.tensor(0.0, device=device)
                if content_weight > 0 and len(scan_content_logits) > 0:
                    scan_locs = locations[1:actual_n_scan + 1]
                    for t, (loc, logit) in enumerate(zip(scan_locs, scan_content_logits)):
                        grid = loc.view(B, 1, 1, 2)
                        sampled = F.grid_sample(clean, grid, align_corners=True,
                                                padding_mode='zeros')
                        label = (sampled.view(-1, 1) > 0.1).float()
                        content_loss = content_loss + bce(logit, label)
                    content_loss = content_loss / len(scan_content_logits)

            # Isolation: prepare data (forward deferred in multi_head to save GPU memory)
            isolation_loss = torch.tensor(0.0, device=device)
            iso_batch = None
            iso_targets_t = None
            iso_k = None
            if isolation_weight > 0 and iso_dataset is not None:
                # 128x128 isolation: prepare single-letter image batch
                iso_k = torch.randint(0, n_positions, (1,)).item()
                iso_imgs = []
                iso_targets = []
                for b in range(B):
                    letter_idx = idx_list[iso_k][b].item()
                    if random.random() < isolation_random_prob:
                        letter_idx = random.randint(0, 25)
                    iso_font = random.choice(iso_dataset.font_list)
                    iso_imgs.append(iso_dataset.get_image(letter_idx, iso_font))
                    iso_targets.append(letter_idx)
                iso_batch = torch.stack(iso_imgs).to(device)  # (B, 1, 128, 128)
                iso_targets_t = torch.tensor(iso_targets, device=device)
                if not multi_head:
                    # Single optimizer: forward now (needed for total_loss sum)
                    with autocast('cuda', enabled=use_amp):
                        _, iso_logits, _, _, _ = model(iso_batch)
                        isolation_loss = F.cross_entropy(iso_logits[iso_k], iso_targets_t)
            elif isolation_weight > 0:
                # Legacy mask-based isolation (fallback)
                iso_k = torch.randint(0, n_positions, (1,)).item()
                W = img.shape[3]
                stripe_w = W // n_positions
                mask = torch.zeros_like(img)
                mask[:, :, :, iso_k * stripe_w:(iso_k + 1) * stripe_w] = 1.0
                masked_img = img * mask
                if not multi_head:
                    with autocast('cuda', enabled=use_amp):
                        _, iso_logits, _, _, _ = model(masked_img)
                        isolation_loss = F.cross_entropy(iso_logits[iso_k], idx_list[iso_k])
                else:
                    iso_batch = masked_img
                    iso_targets_t = idx_list[iso_k]

            # --- Backward pass ---
            if multi_head:
                # Multi-head: 3 targeted backward passes
                # Filter to trainable params only (scaffold freezes some)
                active_attn = [p for p in attn_params if p.requires_grad]
                active_cls = [p for p in cls_params if p.requires_grad]

                attn_opt.zero_grad()
                cls_opt.zero_grad()
                recon_opt.zero_grad()

                # 1. Attention losses -> controller, sensors, content_head
                attn_total = (scan_guide_weight * scan_attn
                              + guide_weight * read_attn
                              + diversity_weight * div_loss
                              + content_weight * content_loss)
                if active_attn:
                    scaler.scale(attn_total).backward(retain_graph=True, inputs=active_attn)

                # 2. Classification losses -> readout, classifiers
                cls_total = sum(cls_losses)
                if active_cls:
                    scaler.scale(cls_total).backward(retain_graph=True, inputs=active_cls)

                # 3. Reconstruction loss -> decoder (last pass, frees main graph)
                scaler.scale(recon_loss).backward(inputs=recon_params)

                # Isolation forward + backward AFTER main graph freed (saves GPU memory)
                if isolation_weight > 0 and iso_batch is not None and active_cls:
                    with autocast('cuda', enabled=use_amp):
                        _, iso_logits, _, _, _ = model(iso_batch)
                        isolation_loss = F.cross_entropy(iso_logits[iso_k], iso_targets_t)
                    scaler.scale(isolation_weight * isolation_loss).backward(inputs=active_cls)

                # Unscale, clip, step — per group
                for params, opt in [(active_attn, attn_opt), (active_cls, cls_opt),
                                     (recon_params, recon_opt)]:
                    scaler.unscale_(opt)
                    if params:
                        clip_grad_norm_(params, 5.0)
                    scaler.step(opt)
                scaler.update()
            else:
                # Single optimizer: sum all losses
                total_loss = (recon_loss + sum(cls_losses)
                              + scan_guide_weight * scan_attn
                              + guide_weight * read_attn
                              + diversity_weight * div_loss
                              + content_weight * content_loss
                              + isolation_weight * isolation_loss)
                optimizer.zero_grad()
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()

            total_loss_recon += recon_loss.item()
            for p in range(n_positions):
                total_loss_pos[p] += cls_losses[p].item()
            total_loss_scan_attn += scan_attn.item()
            total_loss_read_attn += read_attn.item()
            total_loss_div += div_loss.item()
            total_loss_content += content_loss.item()
            total_loss_isolation += isolation_loss.item()

            # Accuracy tracking
            with torch.no_grad():
                for p in range(n_positions):
                    total_pos_correct[p] += (logits_list[p].argmax(1) == idx_list[p]).sum().item()
                total_samples += img.shape[0]
                hr, hi = fixation_hit_rate(clean, locations)
                total_hit_rate += hr
                total_hit_intensity += hi

        n = len(dataloader)
        avg_recon = total_loss_recon / n
        avg_pos = [total_loss_pos[p] / n for p in range(n_positions)]
        avg_scan_attn = total_loss_scan_attn / n
        avg_read_attn = total_loss_read_attn / n
        avg_div = total_loss_div / n
        avg_content = total_loss_content / n
        avg_isolation = total_loss_isolation / n
        avg_hr = total_hit_rate / n
        avg_hi = total_hit_intensity / n
        accs = [total_pos_correct[p] / total_samples if total_samples > 0 else 0
                for p in range(n_positions)]

        losses_recon.append(avg_recon)
        for p in range(n_positions):
            losses_pos_cls[p].append(avg_pos[p])
        losses_attn.append(avg_scan_attn + avg_read_attn)
        losses_div.append(avg_div)
        losses_content.append(avg_content)
        losses_isolation.append(avg_isolation)
        hist_hit_rate.append(avg_hr)
        hist_hit_intensity.append(avg_hi)

        epoch_time = time.time() - epoch_start
        elapsed = time.time() - train_start
        done = epoch - start_epoch + 1
        remaining_ep = epochs - done
        eta_sec = remaining_ep * (elapsed / done)
        eta_min, eta_s = divmod(int(eta_sec), 60)

        if multi_head:
            lr_attn = attn_sched.get_last_lr()[0]
            lr_cls = cls_sched.get_last_lr()[0]
            lr_recon = recon_sched.get_last_lr()[0]
            lr_str = f"lr {lr_attn:.6f}/{lr_cls:.6f}/{lr_recon:.6f}"
        else:
            lrs = scheduler.get_last_lr()
            lr_attn = lrs[0]
            lr_cls = lrs[-1]
            lr_recon = lr_cls
            lr_str = f"lr {lr_attn:.6f}/{lr_cls:.6f}"

        pos_str = '  '.join(f'P{p+1} {avg_pos[p]:.4f} ({accs[p]:.0%})' for p in range(n_positions))
        scaff_str = f"  scaff {scaffold_weight:.2f}" if scaffold_epochs > 0 else ""
        iso_str = f"  Iso {avg_isolation:.4f}" if isolation_weight > 0 else ""
        print(f"Epoch {epoch+1}/{end_epoch}: "
              f"Recon {avg_recon:.4f}  {pos_str}  "
              f"SAttn {avg_scan_attn:.4f}  RAttn {avg_read_attn:.4f}  "
              f"Div {avg_div:.4f}  Cont {avg_content:.4f}{iso_str}  Hit {avg_hr:.0%}  "
              f"{lr_str}{scaff_str}  "
              f"[{epoch_time:.1f}s  ETA {eta_min}m{eta_s:02d}s]")

        pos_log = '  '.join(f'{avg_pos[p]:.4f}' for p in range(n_positions))
        log_file.write(f"{epoch+1:>5d}  {avg_recon:.4f}  {pos_log}  "
                       f"{avg_scan_attn:.4f}  {avg_read_attn:.4f}  "
                       f"{avg_div:.4f}  {avg_content:.4f}  {avg_isolation:.4f}  "
                       f"{avg_hr:.4f}  {lr_attn:.6f}  {lr_cls:.6f}  "
                       f"{lr_recon:.6f}  {scaffold_weight:.4f}  {epoch_time:.1f}s\n")
        log_file.flush()

        if multi_head:
            attn_sched.step()
            cls_sched.step()
            recon_sched.step()
        else:
            scheduler.step()

        if (epoch + 1 - start_epoch) % checkpoint_interval == 0:
            opt_states = None
            if multi_head:
                opt_states = {
                    'attn_optimizer': attn_opt.state_dict(),
                    'cls_optimizer': cls_opt.state_dict(),
                    'recon_optimizer': recon_opt.state_dict(),
                    'attn_scheduler': attn_sched.state_dict(),
                    'cls_scheduler': cls_sched.state_dict(),
                    'recon_scheduler': recon_sched.state_dict(),
                }
            _save_word_checkpoint(model, epoch, n_scan_glimpses, n_read_glimpses,
                                  scan_patch_size, read_patch_size, n_scales, n_positions,
                                  losses_recon, losses_pos_cls, losses_attn, losses_div,
                                  losses_content, losses_isolation, hist_hit_rate,
                                  hist_hit_intensity,
                                  os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth'),
                                  multi_head=multi_head, opt_states=opt_states,
                                  scaffold_epochs=scaffold_epochs)

    log_file.close()
    print(f"Training log saved to {log_path}")

    opt_states = None
    if multi_head:
        opt_states = {
            'attn_optimizer': attn_opt.state_dict(),
            'cls_optimizer': cls_opt.state_dict(),
            'recon_optimizer': recon_opt.state_dict(),
            'attn_scheduler': attn_sched.state_dict(),
            'cls_scheduler': cls_sched.state_dict(),
            'recon_scheduler': recon_sched.state_dict(),
        }
    _save_word_checkpoint(model, end_epoch - 1, n_scan_glimpses, n_read_glimpses,
                           scan_patch_size, read_patch_size, n_scales, n_positions,
                           losses_recon, losses_pos_cls, losses_attn, losses_div,
                           losses_content, losses_isolation, hist_hit_rate,
                           hist_hit_intensity,
                           os.path.join(save_dir, 'model_final.pth'),
                           multi_head=multi_head, opt_states=opt_states,
                           scaffold_epochs=scaffold_epochs)

    # Training metrics graph
    epochs_x = range(end_epoch - len(losses_recon) + 1, end_epoch + 1)
    n_plots = 7 if losses_isolation and any(v > 0 for v in losses_isolation) else 6
    fig, axes = plt.subplots(n_plots, 1, figsize=(8, 2.3 * n_plots), sharex=True)

    axes[0].plot(epochs_x, losses_recon, label='Recon', color='tab:blue')
    axes[0].set_ylabel('MSE')
    axes[0].legend(loc='upper right')
    axes[0].set_title('Reconstruction (256x128)')

    plot_colors = ['tab:red', 'tab:orange', 'tab:brown', 'tab:pink']
    for p in range(n_positions):
        style = '-' if p % 2 == 0 else '--'
        axes[1].plot(epochs_x, losses_pos_cls[p], label=f'Pos {p+1}',
                     color=plot_colors[p], linestyle=style)
    axes[1].axhline(y=np.log(26), color='gray', linestyle='--',
                    label=f'Random ({np.log(26):.1f})')
    axes[1].set_ylabel('Cross-Entropy')
    axes[1].legend(loc='upper right')
    axes[1].set_title('Letter Classification (26-class, per position)')

    axes[2].plot(epochs_x, losses_attn, label='Guide (scan+read)', color='tab:green')
    axes[2].set_ylabel('Loss')
    axes[2].legend(loc='upper right')
    axes[2].set_title('Attention guide (lower = fixations on letters)')

    axes[3].plot(epochs_x, losses_div, label='Diversity', color='tab:orange')
    axes[3].set_ylabel('Repulsion')
    axes[3].legend(loc='upper right')
    axes[3].set_title('Fixation diversity (lower = more spread)')

    axes[4].plot(epochs_x, losses_content, label='Content BCE', color='tab:cyan')
    axes[4].set_ylabel('Loss')
    axes[4].legend(loc='upper right')
    axes[4].set_title('Content detection (scan phase)')

    next_ax = 5
    if n_plots == 7:
        axes[next_ax].plot(epochs_x, losses_isolation, label='Isolation CE',
                           color='tab:olive')
        axes[next_ax].axhline(y=np.log(26), color='gray', linestyle='--',
                              label=f'Random ({np.log(26):.1f})')
        axes[next_ax].set_ylabel('Cross-Entropy')
        axes[next_ax].legend(loc='upper right')
        axes[next_ax].set_title('Isolation mask (single-letter forced fixation)')
        next_ax += 1

    axes[next_ax].plot(epochs_x, hist_hit_rate, label='Hit rate', color='tab:purple')
    axes[next_ax].plot(epochs_x, hist_hit_intensity, label='Intensity',
                       color='tab:purple', linestyle='--', alpha=0.6)
    axes[next_ax].set_xlabel('Epoch')
    axes[next_ax].set_ylabel('Rate / Intensity')
    axes[next_ax].set_ylim(0, 1)
    axes[next_ax].legend(loc='upper right')
    axes[next_ax].set_title('Fixation hit rate (on sharp letter pixels)')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_metrics.png'), dpi=150)
    plt.close()

    total_time = time.time() - train_start
    total_min, total_s = divmod(int(total_time), 60)
    print(f"Training complete in {total_min}m{total_s:02d}s. "
          f"Model and graph saved in {save_dir}")


def _save_word_checkpoint(model, epoch, n_scan_glimpses, n_read_glimpses,
                           scan_patch_size, read_patch_size, n_scales, n_positions,
                           losses_recon, losses_pos_cls, losses_attn, losses_div,
                           losses_content, losses_isolation, hist_hr, hist_hi, path,
                           multi_head=False, opt_states=None, scaffold_epochs=0):
    """Save a word model checkpoint with two-phase metadata."""
    losses = {
        'recon': losses_recon, 'attn': losses_attn,
        'div': losses_div, 'content': losses_content,
        'isolation': losses_isolation,
        'hit_rate': hist_hr, 'hit_intensity': hist_hi,
    }
    for p in range(n_positions):
        losses[f'pos{p+1}_cls'] = losses_pos_cls[p]

    ckpt = {
        'epoch': epoch,
        'model': {k: v.cpu() for k, v in model.state_dict().items()},
        'model_type': 'word',
        'n_scan_glimpses': n_scan_glimpses,
        'n_read_glimpses': n_read_glimpses,
        'scan_patch_size': scan_patch_size,
        'read_patch_size': read_patch_size,
        'n_glimpses': n_scan_glimpses + n_read_glimpses,
        'n_scales': n_scales,
        'n_positions': n_positions,
        'image_size': (128, 256),
        'losses': losses,
        'multi_head': multi_head,
        'scaffold_epochs': scaffold_epochs,
    }
    if multi_head and opt_states:
        ckpt.update(opt_states)
    torch.save(ckpt, path)

    # Save lightweight weights-only file alongside the final checkpoint
    if path.endswith('model_final.pth'):
        weights_path = os.path.join(os.path.dirname(path), 'model_weights.pth')
        torch.save({
            'model_state_dict': {k: v.cpu() for k, v in model.state_dict().items()},
            'model_type': 'word',
            'epoch': epoch + 1,
            'n_scan_glimpses': n_scan_glimpses,
            'n_read_glimpses': n_read_glimpses,
            'scan_patch_size': scan_patch_size,
            'read_patch_size': read_patch_size,
            'n_scales': n_scales,
            'n_positions': n_positions,
        }, weights_path)
        import pathlib
        full_sz = pathlib.Path(path).stat().st_size / (1024 * 1024)
        wt_sz = pathlib.Path(weights_path).stat().st_size / (1024 * 1024)
        print(f"Weights-only saved: {weights_path} ({wt_sz:.0f}MB vs {full_sz:.0f}MB full)")
