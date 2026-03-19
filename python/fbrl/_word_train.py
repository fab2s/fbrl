"""Word training functions — imported by train.py."""
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import numpy as np
import os
import time
import resource

import random
from torch.nn.utils import clip_grad_norm_

from fbrl import _resolve_device
from fbrl.data import WordDataset, IsolationLetterDataset
from fbrl.model import WordVisionModel
from fbrl.losses import (word_attention_loss, fixation_diversity_loss,
                          fixation_hit_rate)
from fbrl.training_utils import (LossTracker, TrainingLogger, save_checkpoint,
                                  apply_transfer, plot_training_metrics, format_eta,
                                  save_run_info)


def train_word_model(cfg):
    """Train a WordVisionModel from an ExperimentConfig."""
    n_scan_glimpses = cfg.n_scan_glimpses
    n_read_glimpses = cfg.n_read_glimpses
    scan_patch_size = cfg.scan_patch_size
    read_patch_size = cfg.read_patch_size
    n_scales = cfg.n_scales
    n_positions = cfg.n_positions
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
    content_weight = cfg.content_weight
    isolation_weight = cfg.isolation_weight
    edge_weight = cfg.edge_weight
    isolation_data_dir = cfg.isolation_data_dir
    isolation_random_prob = cfg.isolation_random_prob
    multi_head = cfg.multi_head
    amp = cfg.amp

    device = _resolve_device(cfg.device)
    n_glimpses = n_scan_glimpses + n_read_glimpses
    print(f"Word training on: {device}")
    if cfg.interleaved:
        total_glimpses = n_scan_glimpses * (1 + cfg.n_read_per_group)
        print(f"Interleaved: {n_scan_glimpses} positions × (1 scan + {cfg.n_read_per_group} reads) = "
              f"{total_glimpses} glimpses  scan={scan_patch_size} read={read_patch_size}")
    else:
        grouped_str = ""
        if cfg.read_anchor_scan_indices:
            grouped_str = f" [grouped: {len(cfg.read_anchor_scan_indices)} groups × {cfg.n_read_per_group}, anchors={cfg.read_anchor_scan_indices}]"
        print(f"Two-phase: scan={n_scan_glimpses} (prescribed x, {scan_patch_size}) + "
              f"read={n_read_glimpses} ({read_patch_size}) = {n_glimpses} glimpses{grouped_str}")
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
    save_run_info(save_dir, cfg)
    dataset = WordDataset(data_dir)
    use_cuda = device.type == 'cuda'
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=use_cuda)

    # Isolation dataset: 128x128 single-letter images
    iso_dataset = None
    if isolation_data_dir and isolation_weight > 0:
        iso_dataset = IsolationLetterDataset(isolation_data_dir)

    read_anchor_scan_indices = cfg.read_anchor_scan_indices
    n_read_per_group = cfg.n_read_per_group

    model = WordVisionModel(
        n_scan_glimpses=n_scan_glimpses, n_read_glimpses=n_read_glimpses,
        scan_patch_size=scan_patch_size, read_patch_size=read_patch_size,
        n_scales=n_scales, n_positions=n_positions,
        read_anchor_scan_indices=read_anchor_scan_indices,
        n_read_per_group=n_read_per_group,
        interleaved=cfg.interleaved,
    ).to(device)

    # Transfer learning
    transfer_scaffold = False
    if transfer_from and not resume:
        broadcast = []
        for suffix in ('weight', 'bias'):
            broadcast.append(
                (f'letter_classifier.{suffix}',
                 [f'classifiers.{p}.{suffix}' for p in range(n_positions)])
            )
        apply_transfer(model, transfer_from,
                        key_mappings=[
                            ('encoder.glimpse_sensor.', 'read_sensor.'),
                            ('encoder.attention_controller.', 'controller.'),
                            ('scan_sensor.', 'scan_sensor.'),
                            ('content_head.', 'content_head.'),
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
    loss_names = (['recon'] +
                  [f'pos{p+1}_cls' for p in range(n_positions)] +
                  ['attn', 'div', 'content', 'isolation', 'hit_rate', 'hit_intensity'])
    tracker = LossTracker(loss_names)
    start_epoch = 0

    if resume:
        checkpoint = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model'], strict=False)
        start_epoch = checkpoint['epoch'] + 1
        if 'losses' in checkpoint:
            tracker.restore_history(checkpoint['losses'])

        # Restore scaffold state
        saved_scaffold = checkpoint.get('scaffold_epochs', 0)
        if saved_scaffold == 0 and scaffold_epochs > 0 and start_epoch < scaffold_epochs:
            saved_scaffold = scaffold_epochs
        if saved_scaffold > 0 and start_epoch < saved_scaffold:
            transfer_scaffold = True
            scaffold_epochs = saved_scaffold
            for p in model.read_sensor.parameters():
                p.requires_grad = False
            for clf in model.classifiers:
                for p in clf.parameters():
                    p.requires_grad = False
            print(f"Resume: scaffold active (frozen until epoch {scaffold_epochs})")
        elif saved_scaffold > 0:
            print(f"Resume: scaffold complete (ended at epoch {saved_scaffold})")

        print(f"Resumed from epoch {start_epoch} ({len(tracker.history['recon'])} prior epochs of history)")

    # Optimizer setup
    criterion = torch.nn.MSELoss()
    bce = torch.nn.BCEWithLogitsLoss()

    def _make_multi_head_optimizers(scaffold_phase, remaining_epochs):
        if scaffold_phase:
            attn_groups = [
                {'params': list(model.scan_sensor.parameters()), 'lr': 0.001},
                {'params': list(model.controller.parameters()), 'lr': 0.0001},
                {'params': list(model.content_head.parameters()), 'lr': 0.001},
            ]
            if model.scan_xs is not None:
                attn_groups.append({'params': [model.scan_xs], 'lr': 0.001})
            attn_opt = optim.Adam(attn_groups)
            cls_opt = optim.Adam([
                {'params': list(model.readout.parameters()), 'lr': 0.001},
            ])
            recon_opt = optim.Adam([
                {'params': list(model.decoder.parameters()), 'lr': 0.001},
            ])
        else:
            attn_groups = [
                {'params': list(model.scan_sensor.parameters()), 'lr': 0.0001},
                {'params': list(model.read_sensor.parameters()), 'lr': 0.00001},
                {'params': list(model.controller.parameters()), 'lr': 0.0001},
                {'params': list(model.content_head.parameters()), 'lr': 0.0001},
            ]
            if model.scan_xs is not None:
                attn_groups.append({'params': [model.scan_xs], 'lr': 0.0001})
            attn_opt = optim.Adam(attn_groups)
            cls_opt = optim.Adam([
                {'params': list(model.readout.parameters()), 'lr': 0.001},
                {'params': [p for clf in model.classifiers for p in clf.parameters()], 'lr': 0.0001},
            ])
            recon_opt = optim.Adam([
                {'params': list(model.decoder.parameters()), 'lr': 0.001},
            ])
        attn_sched = optim.lr_scheduler.CosineAnnealingLR(attn_opt, T_max=remaining_epochs)
        cls_sched = optim.lr_scheduler.CosineAnnealingLR(cls_opt, T_max=remaining_epochs)
        recon_sched = optim.lr_scheduler.CosineAnnealingLR(recon_opt, T_max=remaining_epochs)
        return attn_opt, cls_opt, recon_opt, attn_sched, cls_sched, recon_sched

    def _make_single_optimizer(scaffold_phase, remaining_epochs):
        if scaffold_phase:
            scan_xs_group = [{'params': [model.scan_xs], 'lr': 0.001}] if model.scan_xs is not None else []
            opt = optim.Adam([
                {'params': list(model.scan_sensor.parameters()), 'lr': 0.001},
                {'params': list(model.controller.parameters()), 'lr': 0.0001},
                {'params': list(model.content_head.parameters()), 'lr': 0.001},
                {'params': list(model.readout.parameters()) + list(model.decoder.parameters()), 'lr': 0.001},
            ] + scan_xs_group)
        else:
            sensor_ctrl_params = set(
                list(model.scan_sensor.parameters()) +
                list(model.read_sensor.parameters()) +
                list(model.controller.parameters()) +
                list(model.content_head.parameters())
            )
            if model.scan_xs is not None:
                sensor_ctrl_params.add(model.scan_xs)
            readout_params = [p for p in model.parameters() if p not in sensor_ctrl_params]
            opt = optim.Adam([
                {'params': list(sensor_ctrl_params), 'lr': 0.0001},
                {'params': readout_params, 'lr': 0.001},
            ])
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=remaining_epochs)
        return opt, sched

    # Multi-head param lists
    scan_xs_params = [model.scan_xs] if model.scan_xs is not None else []
    attn_params = (list(model.scan_sensor.parameters()) +
                   list(model.read_sensor.parameters()) +
                   list(model.controller.parameters()) +
                   list(model.content_head.parameters()) +
                   scan_xs_params)
    cls_params = (list(model.readout.parameters()) +
                  [p for clf in model.classifiers for p in clf.parameters()])
    recon_params = list(model.decoder.parameters())

    if multi_head:
        attn_opt, cls_opt, recon_opt, attn_sched, cls_sched, recon_sched = \
            _make_multi_head_optimizers(transfer_scaffold, epochs)
        if resume and checkpoint.get('multi_head', False):
            if 'attn_optimizer' in checkpoint:
                attn_opt.load_state_dict(checkpoint['attn_optimizer'])
                cls_opt.load_state_dict(checkpoint['cls_optimizer'])
                recon_opt.load_state_dict(checkpoint['recon_optimizer'])
                attn_sched.load_state_dict(checkpoint['attn_scheduler'])
                cls_sched.load_state_dict(checkpoint['cls_scheduler'])
                recon_sched.load_state_dict(checkpoint['recon_scheduler'])
    else:
        optimizer, scheduler = _make_single_optimizer(transfer_scaffold, epochs)

    end_epoch = start_epoch + epochs
    train_start = time.time()

    use_amp = amp and device.type == 'cuda'
    scaler = GradScaler('cuda', enabled=use_amp)
    if use_amp:
        print("AMP enabled: FP16 forward passes, GradScaler for backward")

    pos_hdrs = '  '.join(f'{"pos"+str(p+1):>6s}' for p in range(n_positions))
    vram_hdr = f"  {'vram':>7s}  {'spill':>6s}  {'gpu':>4s}" if device.type == 'cuda' else ""
    header = (f"{'epoch':>5s}  {'recon':>6s}  {pos_hdrs}  {'s_attn':>7s}  {'r_attn':>7s}  "
              f"{'div':>6s}  {'content':>7s}  {'isolat':>6s}  {'hit':>6s}  "
              f"{'lr_attn':>8s}  {'lr_cls':>8s}  {'lr_rcon':>8s}  {'scaff':>6s}"
              f"{vram_hdr}  time")
    logger = TrainingLogger(save_dir, header, start_epoch)

    for epoch in range(start_epoch, end_epoch):
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
        tracker.reset_epoch()
        total_pos_correct = [0] * n_positions
        total_samples = 0

        if scaffold_epochs > 0:
            scaffold_weight = max(scaffold_floor, 1.0 - epoch / scaffold_epochs)
        else:
            scaffold_weight = scaffold_floor

        for img, clean, l1s, l2s, l3s, l4s, _words, fonts_batch in dataloader:
            img = img.to(device)
            clean = clean.to(device)
            B = img.shape[0]

            letter_lists = [l1s, l2s, l3s, l4s]
            idx_list = [
                torch.tensor([ord(l) - ord('a') for l in letters], device=device)
                for letters in letter_lists[:n_positions]
            ]

            with autocast('cuda', enabled=use_amp):
                recon, logits_list, locations, _readout_states, scan_content_logits, read_group_boundaries, phase_tags = model(img)

                recon_loss = criterion(recon, img)
                cls_losses = [F.cross_entropy(logits_list[p], idx_list[p])
                              for p in range(n_positions)]

                # Extract scan/read locations using phase_tags
                scan_locs = [loc for loc, tag in zip(locations, phase_tags) if tag == 'scan']
                read_locs = [loc for loc, tag in zip(locations, phase_tags) if tag == 'read']

                scan_attn, read_attn = word_attention_loss(
                    clean, scan_locs, read_locs,
                    n_positions=n_positions,
                    blur_sigma_ratio=blur_sigma_ratio,
                    scaffold_weight=scaffold_weight,
                    read_group_boundaries=read_group_boundaries,
                )

                scan_div = fixation_diversity_loss([locations[0]] + scan_locs,
                                                   sigma=diversity_sigma, vy=scan_vy)
                if read_group_boundaries is not None:
                    # Per-group read diversity
                    group_divs = []
                    for gi, start in enumerate(read_group_boundaries):
                        end = read_group_boundaries[gi + 1] if gi + 1 < len(read_group_boundaries) else len(read_locs)
                        group_locs = [locations[0]] + read_locs[start:end]
                        group_divs.append(fixation_diversity_loss(group_locs, sigma=diversity_sigma, vy=read_vy))
                    read_div = sum(group_divs) / len(group_divs)
                else:
                    read_div = fixation_diversity_loss([locations[0]] + read_locs,
                                                       sigma=diversity_sigma, vy=read_vy)
                div_loss = scan_div + read_div

                content_loss = torch.tensor(0.0, device=device)
                if content_weight > 0 and len(scan_content_logits) > 0:
                    for loc, logit in zip(scan_locs, scan_content_logits):
                        grid = loc.view(B, 1, 1, 2)
                        sampled = F.grid_sample(clean, grid, align_corners=True,
                                                padding_mode='zeros')
                        label = (sampled.view(-1, 1) > 0.1).float()
                        content_loss = content_loss + bce(logit, label)
                    content_loss = content_loss / len(scan_content_logits)

            # Isolation
            isolation_loss = torch.tensor(0.0, device=device)
            iso_batch = None
            iso_targets_t = None
            iso_k = None
            if isolation_weight > 0 and iso_dataset is not None:
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
                iso_batch = torch.stack(iso_imgs).to(device)
                iso_targets_t = torch.tensor(iso_targets, device=device)
                if not multi_head:
                    with autocast('cuda', enabled=use_amp):
                        _, iso_logits, _, _, _, _, _ = model(iso_batch)
                        isolation_loss = F.cross_entropy(iso_logits[iso_k], iso_targets_t)
            elif isolation_weight > 0:
                iso_k = torch.randint(0, n_positions, (1,)).item()
                W = img.shape[3]
                stripe_w = W // n_positions
                mask = torch.zeros_like(img)
                mask[:, :, :, iso_k * stripe_w:(iso_k + 1) * stripe_w] = 1.0
                masked_img = img * mask
                if not multi_head:
                    with autocast('cuda', enabled=use_amp):
                        _, iso_logits, _, _, _, _, _ = model(masked_img)
                        isolation_loss = F.cross_entropy(iso_logits[iso_k], idx_list[iso_k])
                else:
                    iso_batch = masked_img
                    iso_targets_t = idx_list[iso_k]

            # Backward
            if multi_head:
                active_attn = [p for p in attn_params if p.requires_grad]
                active_cls = [p for p in cls_params if p.requires_grad]

                attn_opt.zero_grad()
                cls_opt.zero_grad()
                recon_opt.zero_grad()

                attn_total = (scan_guide_weight * scan_attn
                              + guide_weight * read_attn
                              + diversity_weight * div_loss
                              + content_weight * content_loss)
                if active_attn:
                    scaler.scale(attn_total).backward(retain_graph=True, inputs=active_attn)

                cls_total = sum(cls_losses)
                if active_cls:
                    scaler.scale(cls_total).backward(retain_graph=True, inputs=active_cls)

                scaler.scale(recon_loss).backward(inputs=recon_params)

                if isolation_weight > 0 and iso_batch is not None and active_cls:
                    with autocast('cuda', enabled=use_amp):
                        _, iso_logits, _, _, _, _, _ = model(iso_batch)
                        isolation_loss = F.cross_entropy(iso_logits[iso_k], iso_targets_t)
                    scaler.scale(isolation_weight * isolation_loss).backward(inputs=active_cls)

                for params, opt in [(active_attn, attn_opt), (active_cls, cls_opt),
                                     (recon_params, recon_opt)]:
                    scaler.unscale_(opt)
                    if params:
                        clip_grad_norm_(params, 5.0)
                    scaler.step(opt)
                scaler.update()
            else:
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

            with torch.no_grad():
                for p in range(n_positions):
                    total_pos_correct[p] += (logits_list[p].argmax(1) == idx_list[p]).sum().item()
                total_samples += B
                hr, hi = fixation_hit_rate(clean, locations)

            losses_update = {'recon': recon_loss, 'attn': scan_attn.item() + read_attn.item(),
                             'div': div_loss, 'content': content_loss,
                             'isolation': isolation_loss, 'hit_rate': hr, 'hit_intensity': hi}
            for p in range(n_positions):
                losses_update[f'pos{p+1}_cls'] = cls_losses[p]
            tracker.update(**losses_update)

        avgs = tracker.end_epoch()
        accs = [total_pos_correct[p] / total_samples if total_samples > 0 else 0
                for p in range(n_positions)]

        epoch_time = time.time() - epoch_start
        elapsed = time.time() - train_start
        done = epoch - start_epoch + 1
        eta = format_eta(elapsed, done, epochs - done)

        # GPU/VRAM snapshot.
        use_cuda = device.type == 'cuda'
        if use_cuda:
            vram_alloc_mb = torch.cuda.max_memory_allocated() / 1024**2
            free, total_dev = torch.cuda.mem_get_info()
            vram_device_mb = (total_dev - free) / 1024**2
            vram_total_mb = total_dev / 1024**2
            vram_spill_mb = max(0, vram_alloc_mb - vram_total_mb)
            # Sample GPU utilization via nvidia-smi every 10 epochs.
            if (epoch - start_epoch) % 10 == 0:
                try:
                    import subprocess
                    nv = subprocess.check_output(
                        ['nvidia-smi', '--query-gpu=utilization.gpu',
                         '--format=csv,noheader,nounits'],
                        timeout=5).decode().strip()
                    gpu_util = int(nv.split('\n')[0])
                except Exception:
                    gpu_util = -1
            else:
                gpu_util = -1
        else:
            vram_alloc_mb = vram_used_mb = vram_total_mb = 0
            gpu_util = -1
        rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

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

        pos_str = '  '.join(f'P{p+1} {avgs[f"pos{p+1}_cls"]:.4f} ({accs[p]:.0%})'
                            for p in range(n_positions))
        scaff_str = f"  scaff {scaffold_weight:.2f}" if scaffold_epochs > 0 else ""
        iso_str = f"  Iso {avgs['isolation']:.4f}" if isolation_weight > 0 else ""
        gpu_str = f" GPU:{gpu_util}%" if gpu_util >= 0 else ""
        spill_str = f"+{vram_spill_mb:.0f}MB spill" if vram_spill_mb > 0 else ""
        vram_str = f"  VRAM {vram_device_mb:.0f}/{vram_total_mb:.0f}MB {spill_str}{gpu_str}" if use_cuda else ""
        print(f"Epoch {epoch+1}/{end_epoch}: "
              f"Recon {avgs['recon']:.4f}  {pos_str}  "
              f"Attn {avgs['attn']:.4f}  "
              f"Div {avgs['div']:.4f}  Cont {avgs['content']:.4f}{iso_str}  Hit {avgs['hit_rate']:.0%}  "
              f"{lr_str}{scaff_str}{vram_str}  "
              f"[{epoch_time:.1f}s  ETA {eta}]")

        pos_log = '  '.join(f'{avgs[f"pos{p+1}_cls"]:>6.4f}' for p in range(n_positions))
        gpu_log = f"  {gpu_util:>3d}%" if gpu_util >= 0 else "     "
        spill_log = f"  {vram_spill_mb:>5.0f}MB" if use_cuda else ""
        vram_log = f"  {vram_device_mb:>6.0f}MB{spill_log}{gpu_log}" if use_cuda else ""
        logger.write_line(f"{epoch+1:>5d}  {avgs['recon']:>6.4f}  {pos_log}  "
                          f"{scan_attn.item():>7.4f}  {read_attn.item():>7.4f}  "
                          f"{avgs['div']:>6.4f}  {avgs['content']:>7.4f}  {avgs['isolation']:>6.4f}  "
                          f"{avgs['hit_rate']:>6.4f}  {lr_attn:>8.6f}  {lr_cls:>8.6f}  "
                          f"{lr_recon:>8.6f}  {scaffold_weight:>6.4f}"
                          f"{vram_log}  {epoch_time:.1f}s")

        if multi_head:
            attn_sched.step()
            cls_sched.step()
            recon_sched.step()
        else:
            scheduler.step()

        if device.type == 'cuda':
            torch.cuda.empty_cache()

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
            extra = {
                'model_type': 'word',
                'n_scan_glimpses': n_scan_glimpses,
                'n_read_glimpses': n_read_glimpses,
                'scan_patch_size': scan_patch_size,
                'read_patch_size': read_patch_size,
                'n_glimpses': n_glimpses,
                'n_scales': n_scales,
                'n_positions': n_positions,
                'image_size': (128, 256),
                'multi_head': multi_head,
                'scaffold_epochs': scaffold_epochs,
                'read_anchor_scan_indices': read_anchor_scan_indices,
                'n_read_per_group': n_read_per_group,
                'interleaved': cfg.interleaved,
            }
            if opt_states:
                extra.update(opt_states)
            save_checkpoint(model, epoch,
                            os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth'),
                            cfg=cfg, losses_dict=tracker.get_history_dict(),
                            extra=extra)

    logger.close()

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
    extra = {
        'model_type': 'word',
        'n_scan_glimpses': n_scan_glimpses,
        'n_read_glimpses': n_read_glimpses,
        'scan_patch_size': scan_patch_size,
        'read_patch_size': read_patch_size,
        'n_glimpses': n_glimpses,
        'n_scales': n_scales,
        'n_positions': n_positions,
        'image_size': (128, 256),
        'multi_head': multi_head,
        'scaffold_epochs': scaffold_epochs,
        'read_anchor_scan_indices': read_anchor_scan_indices,
        'n_read_per_group': n_read_per_group,
    }
    if opt_states:
        extra.update(opt_states)
    final_path = os.path.join(save_dir, 'model_final.pth')
    save_checkpoint(model, end_epoch - 1, final_path,
                    cfg=cfg, losses_dict=tracker.get_history_dict(),
                    extra=extra)

    # Lightweight weights-only file
    weights_path = os.path.join(save_dir, 'model_weights.pth')
    torch.save({
        'model_state_dict': {k: v.cpu() for k, v in model.state_dict().items()},
        'model_type': 'word', 'epoch': end_epoch,
        'n_scan_glimpses': n_scan_glimpses, 'n_read_glimpses': n_read_glimpses,
        'scan_patch_size': scan_patch_size, 'read_patch_size': read_patch_size,
        'n_scales': n_scales, 'n_positions': n_positions,
        'read_anchor_scan_indices': read_anchor_scan_indices,
        'n_read_per_group': n_read_per_group,
        'interleaved': cfg.interleaved,
    }, weights_path)
    import pathlib
    full_sz = pathlib.Path(final_path).stat().st_size / (1024 * 1024)
    wt_sz = pathlib.Path(weights_path).stat().st_size / (1024 * 1024)
    print(f"Weights-only saved: {weights_path} ({wt_sz:.0f}MB vs {full_sz:.0f}MB full)")

    # Metrics plot
    plot_colors = ['tab:red', 'tab:orange', 'tab:brown', 'tab:pink']
    pos_keys = [f'pos{p+1}_cls' for p in range(n_positions)]
    pos_labels = [f'Pos {p+1}' for p in range(n_positions)]
    pos_styles = ['-' if p % 2 == 0 else '--' for p in range(n_positions)]

    has_iso = any(v > 0 for v in tracker.history.get('isolation', []))
    specs = [
        {'keys': ['recon'], 'labels': ['Recon'], 'colors': ['tab:blue'],
         'title': 'Reconstruction (256x128)', 'ylabel': 'MSE'},
        {'keys': pos_keys, 'labels': pos_labels,
         'colors': plot_colors[:n_positions], 'styles': pos_styles,
         'title': 'Letter Classification (26-class, per position)', 'ylabel': 'Cross-Entropy',
         'hlines': [(np.log(26), f'Random ({np.log(26):.1f})', 'gray')]},
        {'keys': ['attn'], 'labels': ['Guide (scan+read)'], 'colors': ['tab:green'],
         'title': 'Attention guide (lower = fixations on letters)', 'ylabel': 'Loss'},
        {'keys': ['div'], 'labels': ['Diversity'], 'colors': ['tab:orange'],
         'title': 'Fixation diversity (lower = more spread)', 'ylabel': 'Repulsion'},
        {'keys': ['content'], 'labels': ['Content BCE'], 'colors': ['tab:cyan'],
         'title': 'Content detection (scan phase)', 'ylabel': 'Loss'},
    ]
    if has_iso:
        specs.append({'keys': ['isolation'], 'labels': ['Isolation CE'], 'colors': ['tab:olive'],
                      'title': 'Isolation mask (single-letter forced fixation)', 'ylabel': 'Cross-Entropy',
                      'hlines': [(np.log(26), f'Random ({np.log(26):.1f})', 'gray')]})
    specs.append({'keys': ['hit_rate', 'hit_intensity'],
                  'labels': ['Hit rate', 'Intensity'],
                  'colors': ['tab:purple', 'tab:purple'], 'styles': ['-', '--'],
                  'title': 'Fixation hit rate (on sharp letter pixels)',
                  'ylabel': 'Rate / Intensity', 'ylim': (0, 1)})
    plot_training_metrics(tracker, os.path.join(save_dir, 'training_metrics.png'), specs)

    total_time = time.time() - train_start
    total_min, total_s = divmod(int(total_time), 60)
    avg_epoch = total_time / max(1, epochs)
    print(f"\n--- Benchmark ---")
    if device.type == 'cuda':
        gpu_name = torch.cuda.get_device_name()
        alloc_mb = torch.cuda.max_memory_allocated() / 1024**2
        free, total_dev = torch.cuda.mem_get_info()
        device_mb = (total_dev - free) / 1024**2
        total_mb = total_dev / 1024**2
        spill_mb = max(0, alloc_mb - total_mb)
        print(f"GPU:         {gpu_name}")
        print(f"VRAM:        {device_mb:.0f}/{total_mb:.0f} MB device, {alloc_mb:.0f} MB peak alloc"
              f"{f', {spill_mb:.0f} MB spill' if spill_mb > 0 else ''}")
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"RAM:         {rss_mb:.0f} MB peak RSS")
    print(f"Avg epoch:   {avg_epoch:.1f}s  ({epochs} epochs, {total_time:.1f}s total)")
    print(f"Training complete in {total_min}m{total_s:02d}s. "
          f"Model and graph saved in {save_dir}")
