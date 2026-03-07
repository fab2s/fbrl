"""Motor v5 training: frozen v8 backbone + self-supervised constrained motor.

Architecture:
  Frozen v8 backbone (all params frozen) provides encoder + classifiers.
  Trainable ConstrainedMotorDecoder: latent -> gated strokes -> render.
  Self-supervised losses: ink-on-target (spatial) + re-read (semantic).
  No GT trajectory scaffold.
"""
import math
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
import os
import time

from fbrl import _resolve_device
from fbrl.data import LetterDataset
from fbrl.model import MotorVisionModel
from fbrl.motor import render_gated_strokes
from fbrl.training_utils import (LossTracker, TrainingLogger, save_checkpoint,
                                  load_checkpoint, plot_training_metrics,
                                  format_eta, save_run_info)


def _gaussian_blur_2d(images, sigma, kernel_size=None):
    """Apply Gaussian blur to a batch of images.

    Args:
        images: (B, 1, H, W) tensor
        sigma: blur sigma in pixels
        kernel_size: int, defaults to 2*ceil(3*sigma)+1
    Returns:
        blurred: (B, 1, H, W) tensor
    """
    if kernel_size is None:
        kernel_size = 2 * int(math.ceil(3 * sigma)) + 1
    half = kernel_size // 2
    x = torch.arange(-half, half + 1, dtype=images.dtype, device=images.device)
    kernel_1d = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel_1d = kernel_1d / kernel_1d.sum()
    # Separable: blur rows then columns
    k_h = kernel_1d.view(1, 1, kernel_size, 1)
    k_w = kernel_1d.view(1, 1, 1, kernel_size)
    padded = F.pad(images, [half, half, half, half], mode='reflect')
    blurred = F.conv2d(padded, k_h.expand(1, 1, kernel_size, 1), padding=0)
    # After vertical blur, need horizontal
    blurred = F.conv2d(blurred, k_w.expand(1, 1, 1, kernel_size), padding=0)
    return blurred


def _build_canonical_lookup(dataset, canonical_font):
    """Build lookup: (letter_upper, case) -> (1, 128, 128) clean image tensor.

    Filters dataset to canonical font only.
    """
    lookup = {}
    for i in range(len(dataset)):
        _, clean, letter, case, font, _ = dataset[i]
        if font != canonical_font:
            continue
        key = (letter, case)
        if key not in lookup:
            lookup[key] = clean  # (1, 128, 128)
    return lookup


def train_motor_model(cfg):
    """Train a constrained motor decoder with frozen v8 backbone.

    Self-supervised: ink-on-target + re-read through frozen encoder.
    """
    # Unpack config
    n_glimpses = cfg.n_read_glimpses
    patch_size = cfg.read_patch_size
    n_scales = cfg.n_scales
    n_scan_glimpses = cfg.n_scan_glimpses
    scan_patch_size = cfg.scan_patch_size
    latent_dim = cfg.latent_dim
    batch_size = cfg.batch_size
    epochs = cfg.epochs
    resume = cfg.resume
    save_dir = cfg.save_dir
    data_dir = cfg.data_dir
    checkpoint_interval = cfg.checkpoint_interval

    # Motor config
    n_strokes = cfg.motor_n_strokes
    points_per_stroke = cfg.motor_points_per_stroke
    motor_hidden_dim = cfg.motor_hidden_dim
    render_sigma = cfg.render_sigma
    ink_weight = cfg.ink_weight
    ink_blur_sigma = cfg.ink_blur_sigma
    ink_blur_sigma_start = cfg.ink_blur_sigma_start
    ink_void_weight = cfg.ink_void_weight
    rr_cls_weight = cfg.rr_cls_weight
    rr_cls_warmup_ratio = cfg.rr_cls_warmup_ratio
    lr_floor = cfg.lr_floor
    canonical_font = cfg.canonical_font
    vision_checkpoint = cfg.vision_checkpoint

    device = _resolve_device(cfg.device)
    print(f"Motor v5 training on: {device}")
    print(f"Frozen backbone: {vision_checkpoint}")
    print(f"Constrained motor: {n_strokes} strokes x {points_per_stroke} points, "
          f"hidden_dim={motor_hidden_dim}, render_sigma={render_sigma}")
    print(f"Self-supervised: ink_weight={ink_weight}, ink_blur_sigma={ink_blur_sigma}, "
          f"ink_void_weight={ink_void_weight}, rr_cls_weight={rr_cls_weight}")

    rr_cls_warmup_epochs = int(rr_cls_warmup_ratio * epochs)

    os.makedirs(save_dir, exist_ok=True)
    save_run_info(save_dir, cfg)

    # Load dataset (all fonts, both cases)
    dataset = LetterDataset(data_dir, case_filter=cfg.case_filter,
                            font_filter=cfg.train_fonts)
    use_cuda = device.type == 'cuda'
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            pin_memory=use_cuda)

    # Build canonical font lookup for ink-on-target
    canonical_lookup = _build_canonical_lookup(dataset, canonical_font)
    print(f"Canonical font lookup: {len(canonical_lookup)} images ({canonical_font})")

    # Build model with constrained motor decoder
    model = MotorVisionModel(
        n_glimpses=n_glimpses, patch_size=patch_size, n_scales=n_scales,
        latent_dim=latent_dim,
        n_scan_glimpses=n_scan_glimpses, scan_patch_size=scan_patch_size,
        render_sigma=render_sigma,
        read_anchor_scan_indices=cfg.read_anchor_scan_indices,
        n_read_per_group=cfg.n_read_per_group,
        learnable_scan_x=cfg.learnable_scan_x,
        motor_n_strokes=n_strokes,
        motor_points_per_stroke=points_per_stroke,
        motor_hidden_dim=motor_hidden_dim,
    ).to(device)

    # Load frozen v8 checkpoint
    if vision_checkpoint and not resume:
        ckpt = load_checkpoint(vision_checkpoint, device)
        state = ckpt['model'] if 'model' in ckpt else ckpt
        # Load vision weights (strict=False: motor_decoder keys won't match)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"Loaded v8 checkpoint: {len(state) - len(missing)} matched, "
              f"{len(missing)} missing (motor_decoder), {len(unexpected)} unexpected")

    # Freeze everything except motor_decoder
    frozen_count = 0
    trainable_count = 0
    for name, param in model.named_parameters():
        if name.startswith('motor_decoder.'):
            param.requires_grad = True
            trainable_count += param.numel()
        else:
            param.requires_grad = False
            frozen_count += param.numel()
    print(f"Parameters: {frozen_count:,} frozen, {trainable_count:,} trainable (motor_decoder)")

    motor_params = list(model.motor_decoder.parameters())
    motor_opt = optim.Adam(motor_params, lr=0.001)
    motor_sched = optim.lr_scheduler.CosineAnnealingLR(
        motor_opt, T_max=epochs, eta_min=lr_floor)

    # Loss tracking
    loss_names = ['ink', 'rr_cls', 'rr_letter_acc', 'rr_case_acc', 'gate_active']
    tracker = LossTracker(loss_names)

    start_epoch = 0
    if resume:
        checkpoint = load_checkpoint(resume, device)
        model.load_state_dict(checkpoint['model'], strict=False)
        start_epoch = checkpoint['epoch'] + 1
        if 'losses' in checkpoint:
            tracker.restore_history(checkpoint['losses'])
        print(f"Resumed from epoch {start_epoch}")

    end_epoch = start_epoch + epochs
    train_start = time.time()

    rr_w_hdr = f"  {'rr_w':>5s}" if rr_cls_warmup_epochs > 0 else ""
    blur_hdr = f"  {'blur':>5s}" if ink_blur_sigma_start > 0 else ""
    header = (f"{'epoch':>5s}  {'ink':>7s}  {'rr_cls':>6s}  "
              f"{'rr_ltr':>6s}  {'rr_cas':>6s}  {'gates':>5s}  "
              f"{'lr':>8s}{rr_w_hdr}{blur_hdr}  time")
    logger = TrainingLogger(save_dir, header, start_epoch)

    for epoch in range(start_epoch, end_epoch):
        epoch_start = time.time()
        tracker.reset_epoch()

        # rr_cls warmup -- inverse cosine ramp 0->1
        if rr_cls_warmup_epochs > 0 and epoch < rr_cls_warmup_epochs:
            rr_cls_w = 0.5 * (1 - math.cos(math.pi * epoch / rr_cls_warmup_epochs))
        else:
            rr_cls_w = 1.0

        # Ink blur sigma: performance-driven, tied to rr_ltr accuracy
        # sigma = max(final, start * (1 - rr_ltr_acc))
        # rr_ltr=0%: sigma=start (broad attractor)
        # rr_ltr=100%: sigma=final (tight precision)
        if ink_blur_sigma_start > 0:
            prev_rr_ltr = tracker.history['rr_letter_acc'][-1] if tracker.history['rr_letter_acc'] else 0.0
            cur_blur_sigma = max(ink_blur_sigma,
                                 ink_blur_sigma_start * (1 - prev_rr_ltr))
        else:
            cur_blur_sigma = ink_blur_sigma

        for img, clean, letters, cases, fonts, partner_clean in dataloader:
            img = img.to(device)
            B = img.shape[0]

            letter_idx = torch.tensor([ord(l) - ord('A') for l in letters], device=device)
            case_idx = torch.tensor([0 if c == 'upper' else 1 for c in cases], device=device)
            case_float = case_idx.float().unsqueeze(1)

            # === Frozen forward -- no graph needed for vision ===
            with torch.no_grad():
                _, _, _, _, latent, _ = model(img, case_float)

            # === Motor decode -- graph starts here ===
            points, gates = model.motor_decoder(latent)
            rendered = render_gated_strokes(points, gates,
                                            height=128, width=128,
                                            sigma=render_sigma)

            # === Ink-on-target loss ===
            # Build canonical target for this batch
            canonical_imgs = []
            for letter, case in zip(letters, cases):
                key = (letter, case)
                if key in canonical_lookup:
                    canonical_imgs.append(canonical_lookup[key])
                else:
                    canonical_imgs.append(torch.zeros(1, 128, 128))
            canonical = torch.stack(canonical_imgs).to(device)
            blurred = _gaussian_blur_2d(canonical, sigma=cur_blur_sigma)

            # Coverage reward + void penalty
            ink_loss = (-torch.mean(rendered * blurred) +
                        ink_void_weight * torch.mean(rendered * (1 - blurred)))

            # === Re-read through frozen encoder ===
            # No torch.no_grad() -- graph must flow through rendered -> motor_decoder
            reread_enc = model._encode(rendered)
            rr_letter_logits = model.letter_classifier(reread_enc.latent)
            rr_case_logits = model.case_classifier(reread_enc.latent)
            rr_cls_loss = (F.cross_entropy(rr_letter_logits, letter_idx) +
                           F.cross_entropy(rr_case_logits, case_idx))

            # === Combined loss ===
            motor_loss = ink_weight * ink_loss + rr_cls_weight * rr_cls_w * rr_cls_loss
            motor_opt.zero_grad()
            motor_loss.backward()
            clip_grad_norm_(motor_params, 5.0)
            motor_opt.step()

            # Metrics
            with torch.no_grad():
                rr_letter_acc = (rr_letter_logits.argmax(1) == letter_idx).float().mean().item()
                rr_case_acc = (rr_case_logits.argmax(1) == case_idx).float().mean().item()
                gate_active = (gates > 0.5).float().sum(dim=1).mean().item()

            tracker.update(
                ink=ink_loss,
                rr_cls=rr_cls_loss,
                rr_letter_acc=rr_letter_acc,
                rr_case_acc=rr_case_acc,
                gate_active=gate_active,
            )

        avgs = tracker.end_epoch()
        epoch_time = time.time() - epoch_start
        elapsed = time.time() - train_start
        done = epoch - start_epoch + 1
        eta = format_eta(elapsed, done, epochs - done)
        lr = motor_sched.get_last_lr()[0]

        rr_w_str = f"  rr_w {rr_cls_w:.2f}" if rr_cls_warmup_epochs > 0 else ""
        blur_str = f"  blur {cur_blur_sigma:.1f}" if ink_blur_sigma_start > 0 else ""
        print(f"Epoch {epoch+1}/{end_epoch}: "
              f"Ink {avgs['ink']:.4f}  RR_cls {avgs['rr_cls']:.4f}  "
              f"RR_ltr {avgs['rr_letter_acc']:.0%}  RR_cas {avgs['rr_case_acc']:.0%}  "
              f"Gates {avgs['gate_active']:.1f}  lr {lr:.6f}{rr_w_str}{blur_str}  "
              f"[{epoch_time:.1f}s  ETA {eta}]")

        rr_w_log = f"  {rr_cls_w:>5.4f}" if rr_cls_warmup_epochs > 0 else ""
        blur_log = f"  {cur_blur_sigma:>5.1f}" if ink_blur_sigma_start > 0 else ""
        logger.write_line(
            f"{epoch+1:>5d}  {avgs['ink']:>7.4f}  {avgs['rr_cls']:>6.4f}  "
            f"{avgs['rr_letter_acc']:>6.4f}  {avgs['rr_case_acc']:>6.4f}  "
            f"{avgs['gate_active']:>5.1f}  {lr:>8.6f}{rr_w_log}{blur_log}  {epoch_time:.1f}s")

        if epoch - start_epoch < epochs:
            motor_sched.step()

        if (epoch + 1 - start_epoch) % checkpoint_interval == 0:
            save_checkpoint(model, epoch,
                            os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth'),
                            cfg=cfg, losses_dict=tracker.get_history_dict(),
                            extra={
                                'model_type': 'letter_motor',
                                'n_glimpses': n_glimpses, 'patch_size': patch_size,
                                'n_scales': n_scales,
                                'n_scan_glimpses': n_scan_glimpses,
                                'scan_patch_size': scan_patch_size,
                                'motor_n_strokes': n_strokes,
                                'motor_points_per_stroke': points_per_stroke,
                                'render_sigma': render_sigma,
                                'learnable_scan_x': cfg.learnable_scan_x,
                                'image_size': 128, 'has_case': True,
                            })

    logger.close()

    save_checkpoint(model, end_epoch - 1,
                    os.path.join(save_dir, 'model_final.pth'),
                    cfg=cfg, losses_dict=tracker.get_history_dict(),
                    extra={
                        'model_type': 'letter_motor',
                        'n_glimpses': n_glimpses, 'patch_size': patch_size,
                        'n_scales': n_scales,
                        'n_scan_glimpses': n_scan_glimpses,
                        'scan_patch_size': scan_patch_size,
                        'motor_n_strokes': n_strokes,
                        'motor_points_per_stroke': points_per_stroke,
                        'render_sigma': render_sigma,
                        'learnable_scan_x': cfg.learnable_scan_x,
                        'image_size': 128, 'has_case': True,
                    })

    # Metrics plot
    specs = [
        {'keys': ['ink'], 'labels': ['Ink-on-target'], 'colors': ['tab:brown'],
         'title': 'Ink-on-target loss (coverage - void penalty)', 'ylabel': 'Loss'},
        {'keys': ['rr_cls'], 'labels': ['Re-read CE'], 'colors': ['tab:red'],
         'title': 'Re-read classification', 'ylabel': 'Cross-Entropy',
         'hlines': [(3.258, 'Random letter (3.26)', 'gray')]},
        {'keys': ['rr_letter_acc', 'rr_case_acc'],
         'labels': ['RR Letter Acc', 'RR Case Acc'],
         'colors': ['tab:purple', 'tab:pink'], 'styles': ['-', '--'],
         'title': 'Re-read accuracy (from rendered strokes)',
         'ylabel': 'Accuracy', 'ylim': (0, 1)},
        {'keys': ['gate_active'], 'labels': ['Active gates'], 'colors': ['tab:blue'],
         'title': 'Stroke utilization (gates > 0.5)',
         'ylabel': 'Mean active strokes', 'ylim': (0, n_strokes + 0.5)},
    ]
    plot_training_metrics(tracker, os.path.join(save_dir, 'training_metrics.png'), specs)

    total_time = time.time() - train_start
    total_min, total_s = divmod(int(total_time), 60)
    print(f"Training complete in {total_min}m{total_s:02d}s. "
          f"Model and graph saved in {save_dir}")
