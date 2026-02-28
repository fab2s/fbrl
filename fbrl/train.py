import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
import time

from fbrl import _resolve_device
from fbrl.data import LetterDataset, BigramDataset
from fbrl.model import VisionModel, BigramVisionModel
from fbrl.losses import (attention_content_loss, temporal_attention_content_loss,
                          two_phase_attention_loss,
                          fixation_diversity_loss, fixation_edge_loss,
                          fixation_hit_rate)
from fbrl._word_train import train_word_model  # noqa: F401


# --- Training ---

def train_model(data_dir, epochs=200, resume=None, save_dir='models',
                checkpoint_interval=10, n_glimpses=10, patch_size=12,
                n_scales=1, device='auto',
                diversity_weight=1.0, diversity_sigma=0.1, diversity_vy=1.0,
                recode_weight=1.0, guide_weight=8.0, blur_sigma_ratio=0.16,
                batch_size=52,
                n_scan_glimpses=0, scan_patch_size=(12, 18),
                scan_vy=0.3, scan_guide_weight=None, content_weight=0.5):
    if scan_guide_weight is None:
        scan_guide_weight = guide_weight

    device = _resolve_device(device)
    print(f"Training on: {device}")
    vy_str = f"  diversity_vy={diversity_vy}" if diversity_vy != 1.0 else ""
    scan_str = (f"\nTwo-phase: scan={n_scan_glimpses} (prescribed x, {scan_patch_size}) + "
                f"read={n_glimpses} ({patch_size}) = {n_scan_glimpses + n_glimpses} glimpses  "
                f"scan_vy={scan_vy}  content_weight={content_weight}") if n_scan_glimpses > 0 else ""
    print(f"Attention: guide_weight={guide_weight}  blur_sigma_ratio={blur_sigma_ratio}  "
          f"diversity_weight={diversity_weight}  diversity_sigma={diversity_sigma}{vy_str}  "
          f"recode_weight={recode_weight}  batch_size={batch_size}{scan_str}")

    os.makedirs(save_dir, exist_ok=True)
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
    ).to(device)

    start_epoch = 0
    losses_recon = []
    losses_letter_cls = []
    losses_case_cls = []
    losses_attn = []
    losses_div = []
    losses_recode = []
    losses_content = []
    hist_hit_rate = []
    hist_hit_intensity = []

    if resume:
        checkpoint = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model'], strict=False)
        start_epoch = checkpoint['epoch'] + 1
        if 'losses' in checkpoint:
            h = checkpoint['losses']
            losses_recon = h.get('recon', [])
            losses_letter_cls = h.get('letter_cls', h.get('cls', []))
            losses_case_cls = h.get('case_cls', [])
            losses_attn = h.get('attn', [])
            losses_div = h.get('div', [])
            losses_recode = h.get('recode', [])
            losses_content = h.get('content', [])
            hist_hit_rate = h.get('hit_rate', [])
            hist_hit_intensity = h.get('hit_intensity', [])
        print(f"Resumed from epoch {start_epoch} ({len(losses_letter_cls)} prior epochs of history)")

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # CosineAnnealingLR: starts at lr=0.001, smoothly decays to near zero by final epoch.
    # Prevents the constant-lr instability that caused catastrophic divergence at epoch 43/100.
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = torch.nn.MSELoss()  # pixel-level reconstruction loss

    end_epoch = start_epoch + epochs
    train_start = time.time()

    # Training log file — one log per run, rotate old log with timestamp suffix
    log_path = os.path.join(save_dir, 'training.log')
    if start_epoch == 0 and os.path.exists(log_path):
        from datetime import datetime
        ts = datetime.fromtimestamp(os.path.getmtime(log_path)).strftime('%Y%m%d_%H%M%S')
        os.rename(log_path, os.path.join(save_dir, f'training_{ts}.log'))
    log_file = open(log_path, 'a')
    if start_epoch == 0:
        content_hdr = "  content" if n_scan_glimpses > 0 else ""
        log_file.write(f"epoch  recon    ltr      case     attn     div      hit    recode{content_hdr}   lr         time\n")
        log_file.write("-" * (90 + len(content_hdr)) + "\n")
    log_file.flush()

    for epoch in range(start_epoch, end_epoch):
        epoch_start = time.time()
        total_loss_recon = 0
        total_loss_letter_cls = 0
        total_loss_case_cls = 0
        total_loss_attn = 0
        total_loss_div = 0
        total_loss_recode = 0
        total_loss_content = 0
        total_hit_rate = 0
        total_hit_intensity = 0

        for img, clean, letters, cases, _fonts, partner_clean in dataloader:
            # img: noisy input the model sees
            # clean: noise-free version (used for attention guide — honest signal)
            # partner_clean: same letter, opposite case, same font (recode target)
            img = img.to(device)
            clean = clean.to(device)
            partner_clean = partner_clean.to(device)

            # Convert string labels to integer indices for loss functions
            letter_idx = torch.tensor(
                [ord(l) - ord('A') for l in letters], device=device,
            )
            case_idx = torch.tensor(
                [0 if c == 'upper' else 1 for c in cases], device=device,
            )
            case_float = case_idx.float().unsqueeze(1)  # (B, 1) for decoder conditioning

            # --- Forward pass ---
            # The model: looks at noisy image through tiny windows,
            # builds a latent, then decodes/classifies from that latent
            recon, letter_logits, case_logits, locations, latent, scan_content_logits = model(img, case_float)
            actual_n_scan = len(scan_content_logits)

            # --- Compute all loss terms ---
            # 1. Reconstruction: can the decoder rebuild the image from the latent?
            recon_loss = criterion(recon, img)
            # 2. Letter classification: does the latent encode which letter this is?
            letter_cls_loss = F.cross_entropy(letter_logits, letter_idx)
            # 3. Case classification: does the latent encode upper vs lower?
            case_cls_loss = F.cross_entropy(case_logits, case_idx)
            # 4. Attention guide: are fixations landing near letter strokes?
            #    (evaluated on clean image — noisy pixels would give false signal)
            if actual_n_scan > 0:
                scan_attn = attention_content_loss(clean, locations[:actual_n_scan + 1],
                                                   blur_sigma_ratio=blur_sigma_ratio)
                read_attn = attention_content_loss(clean, locations[actual_n_scan:],
                                                   blur_sigma_ratio=blur_sigma_ratio)
                attn_loss = scan_guide_weight * scan_attn + guide_weight * read_attn
            else:
                attn_loss = guide_weight * attention_content_loss(clean, locations,
                                                                  blur_sigma_ratio=blur_sigma_ratio)
            # 5. Diversity: are fixations spread out, not clustered?
            if actual_n_scan > 0:
                scan_div = fixation_diversity_loss(locations[:actual_n_scan + 1],
                                                   sigma=diversity_sigma, vy=scan_vy)
                read_div = fixation_diversity_loss(locations[actual_n_scan:],
                                                   sigma=diversity_sigma, vy=diversity_vy)
                div_loss = scan_div + read_div
            else:
                div_loss = fixation_diversity_loss(locations, sigma=diversity_sigma,
                                                   vy=diversity_vy)
            # 6. Content detection: does scan correctly identify letter presence?
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

            # Weighted sum — guide_weight is the critical knob. Too low and the
            # decoder learns to ignore attention; too high and it dominates training.
            total_loss = (recon_loss + letter_cls_loss + case_cls_loss
                          + attn_loss
                          + diversity_weight * div_loss
                          + content_weight * content_loss)

            # 6. Recode loss: flip the case label, decode the SAME latent, compare
            #    to the partner image (e.g., encode 'a' -> decode as 'A').
            #    Forces the latent to capture letter identity separately from case.
            if dataset.has_partners and recode_weight > 0:
                flipped_case = 1.0 - case_float
                recode_img = model.decoder(latent, flipped_case)
                recode_loss = criterion(recode_img, partner_clean)
                total_loss = total_loss + recode_weight * recode_loss
                total_loss_recode += recode_loss.item()

            # --- Backward pass ---
            optimizer.zero_grad()       # clear old gradients
            total_loss.backward()       # compute gradients through entire model
            # Clip gradients to prevent exploding updates (safety net)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()            # update weights

            total_loss_recon += recon_loss.item()
            total_loss_letter_cls += letter_cls_loss.item()
            total_loss_case_cls += case_cls_loss.item()
            total_loss_attn += attn_loss.item()
            total_loss_div += div_loss.item()
            total_loss_content += content_loss.item()

            # Hit rate diagnostic (no grad needed, on clean image)
            with torch.no_grad():
                hr, hi = fixation_hit_rate(clean, locations)
                total_hit_rate += hr
                total_hit_intensity += hi

        n = len(dataloader)
        avg_recon = total_loss_recon / n
        avg_letter_cls = total_loss_letter_cls / n
        avg_case_cls = total_loss_case_cls / n
        avg_attn = total_loss_attn / n
        avg_div = total_loss_div / n
        avg_recode = total_loss_recode / n
        avg_content = total_loss_content / n
        avg_hr = total_hit_rate / n
        avg_hi = total_hit_intensity / n
        losses_recon.append(avg_recon)
        losses_letter_cls.append(avg_letter_cls)
        losses_case_cls.append(avg_case_cls)
        losses_attn.append(avg_attn)
        losses_div.append(avg_div)
        losses_recode.append(avg_recode)
        losses_content.append(avg_content)
        hist_hit_rate.append(avg_hr)
        hist_hit_intensity.append(avg_hi)

        epoch_time = time.time() - epoch_start
        elapsed = time.time() - train_start
        done = epoch - start_epoch + 1
        remaining = epochs - done
        eta_sec = remaining * (elapsed / done)
        eta_min, eta_s = divmod(int(eta_sec), 60)

        current_lr = scheduler.get_last_lr()[0]
        content_str = f"  Cont {avg_content:.4f}" if n_scan_glimpses > 0 else ""
        print(f"Epoch {epoch+1}/{end_epoch}: "
              f"Recon {avg_recon:.4f}  Ltr {avg_letter_cls:.4f}  "
              f"Case {avg_case_cls:.4f}  Attn {avg_attn:.4f}  "
              f"Div {avg_div:.4f}{content_str}  Hit {avg_hr:.0%}  "
              f"Recode {avg_recode:.4f}  "
              f"lr {current_lr:.6f}  "
              f"[{epoch_time:.1f}s  ETA {eta_min}m{eta_s:02d}s]")

        # Write to log file (machine-readable, tab-separated)
        content_log = f"  {avg_content:.4f}" if n_scan_glimpses > 0 else ""
        log_file.write(f"{epoch+1:>5d}  {avg_recon:.4f}  {avg_letter_cls:.4f}  "
                       f"{avg_case_cls:.4f}  {avg_attn:.4f}  {avg_div:.4f}  "
                       f"{avg_hr:.4f}  {avg_recode:.4f}{content_log}  {current_lr:.6f}  "
                       f"{epoch_time:.1f}s\n")
        log_file.flush()

        # Step the learning rate schedule (once per epoch, after logging)
        scheduler.step()

        if (epoch + 1 - start_epoch) % checkpoint_interval == 0:
            ckpt = {
                'epoch': epoch,
                'model': {k: v.cpu() for k, v in model.state_dict().items()},
                'n_glimpses': n_glimpses, 'patch_size': patch_size,
                'n_scales': n_scales,
                'n_scan_glimpses': n_scan_glimpses,
                'scan_patch_size': scan_patch_size,
                'image_size': 128, 'has_case': True,
                'losses': {
                    'recon': losses_recon, 'letter_cls': losses_letter_cls,
                    'case_cls': losses_case_cls, 'attn': losses_attn,
                    'div': losses_div, 'recode': losses_recode,
                    'content': losses_content,
                    'hit_rate': hist_hit_rate, 'hit_intensity': hist_hit_intensity,
                },
            }
            torch.save(ckpt, os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth'))

    log_file.close()
    print(f"Training log saved to {log_path}")

    # Save final
    torch.save({
        'epoch': end_epoch - 1,
        'model': {k: v.cpu() for k, v in model.state_dict().items()},
        'n_glimpses': n_glimpses, 'patch_size': patch_size, 'n_scales': n_scales,
        'n_scan_glimpses': n_scan_glimpses,
        'scan_patch_size': scan_patch_size,
        'image_size': 128, 'has_case': True,
        'losses': {
            'recon': losses_recon, 'letter_cls': losses_letter_cls,
            'case_cls': losses_case_cls, 'attn': losses_attn,
            'div': losses_div, 'recode': losses_recode,
            'content': losses_content,
            'hit_rate': hist_hit_rate, 'hit_intensity': hist_hit_intensity,
        },
    }, os.path.join(save_dir, 'model_final.pth'))

    # Training metrics graph
    epochs_x = range(end_epoch - len(losses_letter_cls) + 1, end_epoch + 1)
    has_content = n_scan_glimpses > 0 and any(v > 0 for v in losses_content)
    n_plots = 7 if has_content else 6
    fig, axes = plt.subplots(n_plots, 1, figsize=(8, 2.3 * n_plots), sharex=True)

    axes[0].plot(epochs_x, losses_recon, label='Recon', color='tab:blue')
    if any(v > 0 for v in losses_recode):
        axes[0].plot(epochs_x, losses_recode, label='Recode', color='tab:cyan',
                     linestyle='--')
    axes[0].set_ylabel('MSE')
    axes[0].legend(loc='upper right')
    axes[0].set_title('Reconstruction')

    axes[1].plot(epochs_x, losses_letter_cls, label='Letter', color='tab:red')
    axes[1].axhline(y=np.log(26), color='gray', linestyle='--',
                    label=f'Random ({np.log(26):.1f})')
    axes[1].set_ylabel('Cross-Entropy')
    axes[1].legend(loc='upper right')
    axes[1].set_title('Letter Classification (26-class)')

    axes[2].plot(epochs_x, losses_case_cls, label='Case', color='tab:pink')
    axes[2].axhline(y=np.log(2), color='gray', linestyle='--',
                    label=f'Random ({np.log(2):.2f})')
    axes[2].set_ylabel('Cross-Entropy')
    axes[2].legend(loc='upper right')
    axes[2].set_title('Case Classification (upper/lower)')

    axes[3].plot(epochs_x, losses_attn, label='Guide', color='tab:green')
    axes[3].set_ylabel('Loss')
    axes[3].legend(loc='upper right')
    axes[3].set_title('Attention guide (lower = fixations on letter)')

    axes[4].plot(epochs_x, losses_div, label='Diversity', color='tab:orange')
    axes[4].set_ylabel('Repulsion')
    axes[4].legend(loc='upper right')
    axes[4].set_title('Fixation diversity (lower = more spread)')

    next_ax = 5
    if has_content:
        axes[next_ax].plot(epochs_x, losses_content, label='Content BCE', color='tab:cyan')
        axes[next_ax].set_ylabel('Loss')
        axes[next_ax].legend(loc='upper right')
        axes[next_ax].set_title('Content detection (scan phase)')
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


# --- Attention Pre-Check ---

def check_attention(data_dir, n_epochs=10, n_glimpses=10, patch_size=12,
                    n_scales=1, device='auto',
                    guide_weight=8.0, blur_sigma_ratio=0.16,
                    diversity_weight=1.0, diversity_sigma=0.1,
                    diversity_vy=1.0):
    """Quick diagnostic: can the attention guide pull fixations onto letter content?

    Runs a few epochs with ONLY attention + diversity loss (no cls/recon/recode).
    If hit rate doesn't improve, the guide config is wrong for this image size.
    """
    device = _resolve_device(device)
    dataset = LetterDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=26, shuffle=True,
                            pin_memory=device.type == 'cuda')

    model = VisionModel(
        n_glimpses=n_glimpses, patch_size=patch_size, n_scales=n_scales,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Report effective blur_sigma from image dimensions
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

def train_bigram_model(data_dir, epochs=100, resume=None, save_dir='bigram_models',
                       checkpoint_interval=10,
                       n_scan_glimpses=5, n_read_glimpses=6,
                       scan_patch_size=(12, 18), read_patch_size=12,
                       n_scales=1, device='auto',
                       diversity_weight=1.0, diversity_sigma=0.1,
                       scan_vy=0.3, read_vy=1.5,
                       guide_weight=8.0, scan_guide_weight=None,
                       blur_sigma_ratio=0.16,
                       batch_size=32, scaffold_epochs=200,
                       scaffold_floor=0.0,
                       transfer_from=None, mask_weight=0.5,
                       edge_weight=0.0):
    """Train a BigramVisionModel with two-phase scan/read attention on 128x128 bigram images.

    Phase 1 — SCAN: wide patches build spatial map. Loss: full-image guide + edge.
      scan_vy < 1.0 makes horizontal proximity expensive (forces horizontal spread).
    Phase 2 — READ: focused patches read letters. Loss: temporal scaffold.
      read_vy > 1.0 makes vertical proximity expensive (forces vertical exploration).

    scaffold_epochs controls temporal scaffold annealing for the READ phase.
    Set scaffold_epochs=0 to disable scaffolding entirely.
    """
    if scan_guide_weight is None:
        scan_guide_weight = guide_weight

    device = _resolve_device(device)
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
    dataset = BigramDataset(data_dir)
    use_cuda = device.type == 'cuda'
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=use_cuda)

    model = BigramVisionModel(
        n_scan_glimpses=n_scan_glimpses, n_read_glimpses=n_read_glimpses,
        scan_patch_size=scan_patch_size, read_patch_size=read_patch_size,
        n_scales=n_scales,
    ).to(device)

    # --- Transfer learning from single-letter model ---
    # Map: encoder.glimpse_sensor.* -> read_sensor.* (same 12x12 patches)
    #       encoder.attention_controller.* -> controller.*
    #       letter_classifier.* -> classifiers.0.* + classifiers.1.*
    # scan_sensor trains from scratch (new rectangular patches).
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
        # Single-letter letter_classifier -> both bigram position classifiers
        for suffix in ('weight', 'bias'):
            sk = f'letter_classifier.{suffix}'
            if sk in src:
                dst[f'classifiers.0.{suffix}'] = src[sk].float()
                dst[f'classifiers.1.{suffix}'] = src[sk].float()
                n_transferred += 2
        model.load_state_dict(dst)
        print(f"Transfer: {n_transferred} tensors from {transfer_from}")

        # Freeze read_sensor + classifiers during scaffold phase.
        # scan_sensor is new (trains from scratch). controller gets low lr.
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
    losses_pos1_cls = []
    losses_pos2_cls = []
    losses_attn = []
    losses_div = []
    hist_hit_rate = []
    hist_hit_intensity = []

    if resume:
        checkpoint = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model'], strict=False)
        start_epoch = checkpoint['epoch'] + 1
        if 'losses' in checkpoint:
            h = checkpoint['losses']
            losses_recon = h.get('recon', [])
            losses_pos1_cls = h.get('pos1_cls', [])
            losses_pos2_cls = h.get('pos2_cls', [])
            losses_attn = h.get('attn', [])
            losses_div = h.get('div', [])
            hist_hit_rate = h.get('hit_rate', [])
            hist_hit_intensity = h.get('hit_intensity', [])
        print(f"Resumed from epoch {start_epoch} ({len(losses_pos1_cls)} prior epochs of history)")

    # Optimizer param groups depend on transfer mode.
    # Transfer scaffold: scan_sensor + controller + readout/decoder train.
    #   read_sensor + classifiers frozen.
    # Standard: sensors + controller get 10x lower lr than readout path.
    if transfer_scaffold:
        optimizer = optim.Adam([
            {'params': list(model.scan_sensor.parameters()), 'lr': 0.001},
            {'params': list(model.controller.parameters()), 'lr': 0.0001},
            {'params': list(model.readout.parameters()) + list(model.decoder.parameters()), 'lr': 0.001},
        ])
        print(f"Param groups (scaffold): scan_sensor lr=0.001, controller lr=0.0001, readout+decoder lr=0.001")
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
        print(f"Param groups: sensors+controller lr=0.0001, readout lr=0.001")
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = torch.nn.MSELoss()

    end_epoch = start_epoch + epochs
    train_start = time.time()

    # Training log file — one log per run, rotate old log with timestamp suffix
    log_path = os.path.join(save_dir, 'training.log')
    if start_epoch == 0 and os.path.exists(log_path):
        from datetime import datetime
        ts = datetime.fromtimestamp(os.path.getmtime(log_path)).strftime('%Y%m%d_%H%M%S')
        os.rename(log_path, os.path.join(save_dir, f'training_{ts}.log'))
    log_file = open(log_path, 'a')
    if start_epoch == 0:
        edge_hdr = "  edge" if edge_weight > 0 else ""
        log_file.write(f"epoch  recon    pos1     pos2     s_attn   r_attn   div      hit      lr_enc   lr_read  scaff{edge_hdr}  time\n")
        log_file.write("-" * (100 + len(edge_hdr)) + "\n")
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
        total_loss_recon = 0
        total_loss_pos1 = 0
        total_loss_pos2 = 0
        total_loss_scan_attn = 0
        total_loss_read_attn = 0
        total_loss_div = 0
        total_loss_edge = 0
        total_hit_rate = 0
        total_hit_intensity = 0
        total_pos1_correct = 0
        total_pos2_correct = 0
        total_samples = 0

        # Scaffold annealing: linearly decay from 1.0 to scaffold_floor over
        # scaffold_epochs.  A non-zero floor keeps gentle spatial pressure
        # (left/right cluster separation) throughout training.
        if scaffold_epochs > 0:
            scaffold_weight = max(scaffold_floor, 1.0 - epoch / scaffold_epochs)
        else:
            scaffold_weight = scaffold_floor

        for img, clean, letter1s, letter2s, _bigrams, _fonts in dataloader:
            img = img.to(device)
            clean = clean.to(device)

            # Labels: a=0 .. z=25
            idx1 = torch.tensor(
                [ord(l) - ord('a') for l in letter1s], device=device,
            )
            idx2 = torch.tensor(
                [ord(l) - ord('a') for l in letter2s], device=device,
            )

            # Forward
            recon, logits_list, locations, _readout_states = model(img)

            # Losses
            recon_loss = criterion(recon, img)
            pos1_cls_loss = F.cross_entropy(logits_list[0], idx1)
            pos2_cls_loss = F.cross_entropy(logits_list[1], idx2)

            # Two-phase attention: scan gets full-image guide, read gets scaffold
            scan_attn, read_attn = two_phase_attention_loss(
                clean, locations, n_scan_glimpses,
                blur_sigma_ratio=blur_sigma_ratio,
                scaffold_weight=scaffold_weight,
            )

            # Split diversity by phase: scan spreads horizontally, read explores vertically
            scan_locations = locations[:n_scan_glimpses + 1]
            read_locations = locations[n_scan_glimpses:]  # includes last scan loc as read start
            scan_div = fixation_diversity_loss(scan_locations, sigma=diversity_sigma,
                                               vy=scan_vy)
            read_div = fixation_diversity_loss(read_locations, sigma=diversity_sigma,
                                               vy=read_vy)
            div_loss = scan_div + read_div

            # Edge loss on scan locations only (encourage horizontal spread during scan)
            if edge_weight > 0:
                edge_loss = fixation_edge_loss(scan_locations)
            else:
                edge_loss = torch.tensor(0.0)

            total_loss = (recon_loss + pos1_cls_loss + pos2_cls_loss
                          + scan_guide_weight * scan_attn
                          + guide_weight * read_attn
                          + edge_weight * edge_loss
                          + diversity_weight * div_loss)

            # --- Masked-half auxiliary loss ---
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
                    mask_cls = mask_cls + F.cross_entropy(
                        logits_m[1][mask_left], idx2[mask_left])
                if (~mask_left).any():
                    mask_cls = mask_cls + F.cross_entropy(
                        logits_m[0][~mask_left], idx1[~mask_left])
                total_loss = total_loss + mask_weight * mask_cls

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss_recon += recon_loss.item()
            total_loss_pos1 += pos1_cls_loss.item()
            total_loss_pos2 += pos2_cls_loss.item()
            total_loss_scan_attn += scan_attn.item()
            total_loss_read_attn += read_attn.item()
            total_loss_div += div_loss.item()
            if edge_weight > 0:
                total_loss_edge += edge_loss.item()

            # Accuracy tracking
            with torch.no_grad():
                total_pos1_correct += (logits_list[0].argmax(1) == idx1).sum().item()
                total_pos2_correct += (logits_list[1].argmax(1) == idx2).sum().item()
                total_samples += img.shape[0]
                hr, hi = fixation_hit_rate(clean, locations)
                total_hit_rate += hr
                total_hit_intensity += hi

        n = len(dataloader)
        avg_recon = total_loss_recon / n
        avg_pos1 = total_loss_pos1 / n
        avg_pos2 = total_loss_pos2 / n
        avg_scan_attn = total_loss_scan_attn / n
        avg_read_attn = total_loss_read_attn / n
        avg_div = total_loss_div / n
        avg_hr = total_hit_rate / n
        avg_hi = total_hit_intensity / n
        acc1 = total_pos1_correct / total_samples if total_samples > 0 else 0
        acc2 = total_pos2_correct / total_samples if total_samples > 0 else 0

        losses_recon.append(avg_recon)
        losses_pos1_cls.append(avg_pos1)
        losses_pos2_cls.append(avg_pos2)
        losses_attn.append(avg_scan_attn + avg_read_attn)  # combined for history compat
        losses_div.append(avg_div)
        hist_hit_rate.append(avg_hr)
        hist_hit_intensity.append(avg_hi)

        epoch_time = time.time() - epoch_start
        elapsed = time.time() - train_start
        done = epoch - start_epoch + 1
        remaining = epochs - done
        eta_sec = remaining * (elapsed / done)
        eta_min, eta_s = divmod(int(eta_sec), 60)

        lrs = scheduler.get_last_lr()
        lr_enc = lrs[0]
        lr_read = lrs[-1]
        avg_edge = total_loss_edge / n
        scaff_str = f"  scaff {scaffold_weight:.2f}" if scaffold_epochs > 0 else ""
        edge_str = f"  Edge {avg_edge:.4f}" if edge_weight > 0 else ""
        print(f"Epoch {epoch+1}/{end_epoch}: "
              f"Recon {avg_recon:.4f}  Pos1 {avg_pos1:.4f} ({acc1:.0%})  "
              f"Pos2 {avg_pos2:.4f} ({acc2:.0%})  "
              f"SAttn {avg_scan_attn:.4f}  RAttn {avg_read_attn:.4f}  "
              f"Div {avg_div:.4f}  Hit {avg_hr:.0%}{edge_str}  "
              f"lr {lr_enc:.6f}/{lr_read:.6f}{scaff_str}  "
              f"[{epoch_time:.1f}s  ETA {eta_min}m{eta_s:02d}s]")

        edge_log = f"  {avg_edge:.4f}" if edge_weight > 0 else ""
        log_file.write(f"{epoch+1:>5d}  {avg_recon:.4f}  {avg_pos1:.4f}  "
                       f"{avg_pos2:.4f}  {avg_scan_attn:.4f}  {avg_read_attn:.4f}  "
                       f"{avg_div:.4f}  "
                       f"{avg_hr:.4f}  {lr_enc:.6f}  {lr_read:.6f}  "
                       f"{scaffold_weight:.4f}{edge_log}  {epoch_time:.1f}s\n")
        log_file.flush()

        scheduler.step()

        if (epoch + 1 - start_epoch) % checkpoint_interval == 0:
            _save_bigram_checkpoint(model, epoch, n_scan_glimpses, n_read_glimpses,
                                   scan_patch_size, read_patch_size, n_scales,
                                   losses_recon, losses_pos1_cls, losses_pos2_cls,
                                   losses_attn, losses_div, hist_hit_rate, hist_hit_intensity,
                                   os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth'))

    log_file.close()
    print(f"Training log saved to {log_path}")

    _save_bigram_checkpoint(model, end_epoch - 1, n_scan_glimpses, n_read_glimpses,
                            scan_patch_size, read_patch_size, n_scales,
                            losses_recon, losses_pos1_cls, losses_pos2_cls,
                            losses_attn, losses_div, hist_hit_rate, hist_hit_intensity,
                            os.path.join(save_dir, 'model_final.pth'))

    # Training metrics graph (5 subplots — no case/recode for bigrams)
    epochs_x = range(end_epoch - len(losses_recon) + 1, end_epoch + 1)
    fig, axes = plt.subplots(5, 1, figsize=(8, 12), sharex=True)

    axes[0].plot(epochs_x, losses_recon, label='Recon', color='tab:blue')
    axes[0].set_ylabel('MSE')
    axes[0].legend(loc='upper right')
    axes[0].set_title('Reconstruction (128x128)')

    axes[1].plot(epochs_x, losses_pos1_cls, label='Pos 1', color='tab:red')
    axes[1].plot(epochs_x, losses_pos2_cls, label='Pos 2', color='tab:orange',
                 linestyle='--')
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

    axes[4].plot(epochs_x, hist_hit_rate, label='Hit rate', color='tab:purple')
    axes[4].plot(epochs_x, hist_hit_intensity, label='Intensity',
                 color='tab:purple', linestyle='--', alpha=0.6)
    axes[4].set_xlabel('Epoch')
    axes[4].set_ylabel('Rate / Intensity')
    axes[4].set_ylim(0, 1)
    axes[4].legend(loc='upper right')
    axes[4].set_title('Fixation hit rate (on sharp letter pixels)')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_metrics.png'), dpi=150)
    plt.close()

    total_time = time.time() - train_start
    total_min, total_s = divmod(int(total_time), 60)
    print(f"Training complete in {total_min}m{total_s:02d}s. "
          f"Model and graph saved in {save_dir}")


def _save_bigram_checkpoint(model, epoch, n_scan_glimpses, n_read_glimpses,
                            scan_patch_size, read_patch_size, n_scales,
                            losses_recon, losses_pos1, losses_pos2,
                            losses_attn, losses_div, hist_hr, hist_hi, path):
    """Save a bigram model checkpoint with two-phase metadata."""
    torch.save({
        'epoch': epoch,
        'model': {k: v.cpu() for k, v in model.state_dict().items()},
        'model_type': 'bigram',
        'n_scan_glimpses': n_scan_glimpses,
        'n_read_glimpses': n_read_glimpses,
        'scan_patch_size': scan_patch_size,
        'read_patch_size': read_patch_size,
        'n_glimpses': n_scan_glimpses + n_read_glimpses,  # eval compat
        'n_scales': n_scales,
        'image_size': (128, 128),
        'losses': {
            'recon': losses_recon, 'pos1_cls': losses_pos1,
            'pos2_cls': losses_pos2, 'attn': losses_attn,
            'div': losses_div,
            'hit_rate': hist_hr, 'hit_intensity': hist_hi,
        },
    }, path)


# --- Bigram Attention Pre-Check ---

def check_bigram_attention(data_dir, n_epochs=10,
                           n_scan_glimpses=5, n_read_glimpses=6,
                           scan_patch_size=(12, 18), read_patch_size=12,
                           n_scales=1, device='auto',
                           guide_weight=8.0, blur_sigma_ratio=0.16,
                           diversity_weight=1.0, diversity_sigma=0.1,
                           diversity_vy=1.0):
    """Quick diagnostic: can the attention guide pull fixations onto bigram letter content?

    Runs a few epochs with ONLY attention + diversity loss on 128x128 bigram images.
    """
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
