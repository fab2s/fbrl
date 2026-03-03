"""Motor training functions -- imported by train.py."""
import copy
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
import numpy as np
import os
import time

from fbrl import _resolve_device
from fbrl.data import LetterDataset
from fbrl.model import MotorVisionModel, encode_scan_read
from fbrl.motor import load_trajectory_data, batch_gt_trajectories
from fbrl.losses import (attention_content_loss, fixation_diversity_loss,
                          fixation_hit_rate, void_repulsion)
from fbrl.training_utils import (LossTracker, TrainingLogger, save_checkpoint,
                                  plot_training_metrics, format_eta,
                                  apply_transfer, save_run_info)


def train_motor_model(cfg):
    """Train a MotorVisionModel (single-letter + motor trace) from an ExperimentConfig.

    Multi-head backward with deferred motor pass for VRAM safety:
      Head 1: attn_loss -> controller + sensors
      Head 2: cls_loss -> classifiers
      Head 3: recon_loss -> decoder
      (main graph freed)
      Head 4: motor_loss -> motor_decoder (via re-read through encoder)
    """
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
    batch_size = cfg.batch_size
    epochs = cfg.epochs
    resume = cfg.resume
    save_dir = cfg.save_dir
    data_dir = cfg.data_dir
    checkpoint_interval = cfg.checkpoint_interval

    # Motor config
    n_trajectory_points = cfg.n_trajectory_points
    render_sigma = cfg.render_sigma
    traj_weight = cfg.traj_weight
    traj_scaffold_ratio = cfg.traj_scaffold_ratio
    traj_scaffold_floor = cfg.traj_scaffold_floor
    rr_cls_weight = cfg.rr_cls_weight
    trajectory_data_dir = cfg.trajectory_data_dir
    latent_match_weight = cfg.latent_match_weight
    frozen_rr_weight = cfg.frozen_rr_weight
    render_match_weight = cfg.render_match_weight
    void_weight = cfg.void_weight
    scan_void_weight = cfg.scan_void_weight
    tmax_attn_ratio = cfg.tmax_attn_ratio
    tmax_cls_ratio = cfg.tmax_cls_ratio
    tmax_recon_ratio = cfg.tmax_recon_ratio
    tmax_motor_ratio = cfg.tmax_motor_ratio
    lr_floor = cfg.lr_floor

    device = _resolve_device(cfg.device)
    print(f"Motor training on: {device}")
    scan_str = ""
    if n_scan_glimpses > 0:
        scan_str = (f"\nTwo-phase: scan={n_scan_glimpses} (prescribed x, {scan_patch_size}) + "
                    f"read={n_glimpses} ({patch_size}) = {n_scan_glimpses + n_glimpses} glimpses  "
                    f"scan_vy={scan_vy}  content_weight={content_weight}")
    print(f"Attention: guide_weight={guide_weight}  blur_sigma_ratio={blur_sigma_ratio}  "
          f"diversity_weight={diversity_weight}  diversity_sigma={diversity_sigma}  "
          f"diversity_vy={diversity_vy}  recode_weight={recode_weight}  "
          f"batch_size={batch_size}{scan_str}")
    traj_scaffold_epochs = int(traj_scaffold_ratio * epochs)
    v2_str = ""
    if latent_match_weight > 0 or frozen_rr_weight > 0 or render_match_weight > 0:
        v2_str = (f"\nMotor v2: latent_match={latent_match_weight}  "
                  f"frozen_rr={frozen_rr_weight}  render_match={render_match_weight}")
    if traj_weight > 0:
        print(f"Motor: n_points={n_trajectory_points}  render_sigma={render_sigma}  "
              f"traj_weight={traj_weight}  rr_cls_weight={rr_cls_weight}  "
              f"traj_scaffold={traj_scaffold_epochs}ep (floor={traj_scaffold_floor})"
              f"{v2_str}")
    else:
        print(f"Motor: n_points={n_trajectory_points}  render_sigma={render_sigma}  "
              f"traj_scaffold=OFF  rr_cls_weight={rr_cls_weight}"
              f"{v2_str}")

    single_case = cfg.case_filter in ('upper', 'lower')

    os.makedirs(save_dir, exist_ok=True)
    save_run_info(save_dir, cfg)
    dataset = LetterDataset(data_dir, case_filter=cfg.case_filter,
                            font_filter=cfg.train_fonts)
    use_cuda = device.type == 'cuda'
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=use_cuda)

    if dataset.has_partners:
        print(f"Partner images found -- recode loss enabled (weight={recode_weight})")
    else:
        print("No partner images -- recode loss disabled")

    # Load trajectory ground truth
    if traj_weight > 0:
        traj_data = load_trajectory_data(trajectory_data_dir)
        print(f"Loaded {len(traj_data)} trajectory templates from {trajectory_data_dir}")
    else:
        traj_data = None
        print("Trajectory scaffold disabled (traj_weight=0)")

    model = MotorVisionModel(
        n_glimpses=n_glimpses, patch_size=patch_size, n_scales=n_scales,
        n_scan_glimpses=n_scan_glimpses, scan_patch_size=scan_patch_size,
        n_trajectory_points=n_trajectory_points, render_sigma=render_sigma,
        read_anchor_scan_indices=cfg.read_anchor_scan_indices,
        n_read_per_group=cfg.n_read_per_group,
        learnable_scan_x=cfg.learnable_scan_x,
    ).to(device)

    # Transfer learning from pretrained vision model
    if cfg.transfer and not resume:
        # MotorVisionModel has same key structure as VisionModel --
        # identity mappings transfer all vision weights directly.
        # motor_decoder.* has no match in source, stays randomly initialized.
        n_transferred = apply_transfer(model, cfg.transfer,
                                        key_mappings=[
                                            ('encoder.', 'encoder.'),
                                            ('decoder.', 'decoder.'),
                                            ('letter_classifier.', 'letter_classifier.'),
                                            ('case_classifier.', 'case_classifier.'),
                                            ('scan_sensor.', 'scan_sensor.'),
                                            ('content_head.', 'content_head.'),
                                        ],
                                        device=device)
        print(f"Transferred {n_transferred} tensors from {cfg.transfer}")
    elif cfg.transfer and resume:
        print("Warning: --transfer ignored when --resume is used")

    # Frozen encoder for honest readability check (plain dict, not model attribute)
    frozen_modules = None
    if frozen_rr_weight > 0:
        frozen_modules = {
            'controller': copy.deepcopy(model.encoder.attention_controller),
            'read_sensor': copy.deepcopy(model.encoder.glimpse_sensor),
            'letter_classifier': copy.deepcopy(model.letter_classifier),
            'case_classifier': copy.deepcopy(model.case_classifier),
        }
        if n_scan_glimpses > 0:
            frozen_modules['scan_sensor'] = copy.deepcopy(model.scan_sensor)
            frozen_modules['content_head'] = copy.deepcopy(model.content_head)
        if model.scan_xs is not None:
            frozen_modules['scan_xs'] = copy.deepcopy(model.scan_xs)
        for m in frozen_modules.values():
            for p in m.parameters():
                p.requires_grad_(False)
            m.to(device)
        n_frozen = sum(p.numel() for m in frozen_modules.values() for p in m.parameters())
        print(f"Created frozen encoder ({n_frozen} params)")

    def _frozen_encode(rendered):
        """Re-read through frozen encoder — no co-adaptation."""
        return encode_scan_read(
            rendered, frozen_modules['controller'],
            frozen_modules.get('scan_sensor', frozen_modules['read_sensor']),
            frozen_modules['read_sensor'],
            n_scan=n_scan_glimpses,
            n_read=model.encoder.n_glimpses,
            content_head=frozen_modules.get('content_head'),
            prescribed_x=(model.scan_xs is None and n_scan_glimpses > 0),
            scan_xs=frozen_modules.get('scan_xs'),
            read_group_anchors=list(cfg.read_anchor_scan_indices) if cfg.read_anchor_scan_indices else None,
            n_read_per_group=cfg.n_read_per_group,
        )

    # Loss tracking
    loss_names = ['recon', 'letter_cls', 'case_cls', 'attn', 'div',
                  'recode', 'content', 'void', 'hit_rate', 'hit_intensity',
                  'rr_cls', 'rr_letter_acc', 'rr_case_acc']
    if traj_weight > 0:
        loss_names[len(loss_names):len(loss_names)] = ['traj_mse', 'pen_bce']
    if latent_match_weight > 0:
        loss_names.append('latent_match')
    if frozen_rr_weight > 0:
        loss_names.append('frozen_rr')
    if render_match_weight > 0:
        loss_names.append('render_match')
    tracker = LossTracker(loss_names)

    start_epoch = 0
    if resume:
        checkpoint = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model'], strict=False)
        start_epoch = checkpoint['epoch'] + 1
        if 'losses' in checkpoint:
            tracker.restore_history(checkpoint['losses'])
        print(f"Resumed from epoch {start_epoch} ({len(tracker.history['letter_cls'])} prior epochs of history)")

    # Multi-head optimizers: separate param groups for independent backward passes
    # Attention params: sensors + controller
    attn_params = (list(model.encoder.glimpse_sensor.parameters()) +
                   list(model.encoder.attention_controller.parameters()))
    if n_scan_glimpses > 0:
        attn_params += (list(model.scan_sensor.parameters()) +
                        list(model.content_head.parameters()))
    if model.scan_xs is not None:
        attn_params += [model.scan_xs]

    cls_params = (list(model.letter_classifier.parameters()) +
                  list(model.case_classifier.parameters()))

    recon_params = list(model.decoder.parameters())

    motor_params = list(model.motor_decoder.parameters())

    attn_opt = optim.Adam(attn_params, lr=0.001)
    cls_opt = optim.Adam(cls_params, lr=0.001)
    recon_opt = optim.Adam(recon_params, lr=0.001)
    motor_opt = optim.Adam(motor_params, lr=0.001)

    tmax_attn = max(1, int(epochs * tmax_attn_ratio))
    tmax_cls = max(1, int(epochs * tmax_cls_ratio))
    tmax_recon = max(1, int(epochs * tmax_recon_ratio))
    tmax_motor = max(1, int(epochs * tmax_motor_ratio))

    attn_sched = optim.lr_scheduler.CosineAnnealingLR(attn_opt, T_max=tmax_attn, eta_min=lr_floor)
    cls_sched = optim.lr_scheduler.CosineAnnealingLR(cls_opt, T_max=tmax_cls, eta_min=lr_floor)
    recon_sched = optim.lr_scheduler.CosineAnnealingLR(recon_opt, T_max=tmax_recon, eta_min=lr_floor)
    motor_sched = optim.lr_scheduler.CosineAnnealingLR(motor_opt, T_max=tmax_motor, eta_min=lr_floor)

    if any(r != 1.0 for r in [tmax_attn_ratio, tmax_cls_ratio, tmax_recon_ratio, tmax_motor_ratio]):
        print(f"Differential T_max: attn={tmax_attn}  cls={tmax_cls}  recon={tmax_recon}  motor={tmax_motor}  floor={lr_floor}")
    elif lr_floor > 0:
        print(f"LR floor: {lr_floor}")

    criterion = torch.nn.MSELoss()
    bce_logits = torch.nn.BCEWithLogitsLoss()

    end_epoch = start_epoch + epochs
    train_start = time.time()

    content_hdr = f"  {'content':>7s}" if n_scan_glimpses > 0 else ""
    v2_hdr = ""
    if latent_match_weight > 0:
        v2_hdr += f"  {'lat_m':>6s}"
    if frozen_rr_weight > 0:
        v2_hdr += f"  {'frz_rr':>6s}"
    if render_match_weight > 0:
        v2_hdr += f"  {'rnd_m':>6s}"
    traj_hdr = f"  {'traj':>6s}  {'pen':>6s}" if traj_weight > 0 else ""
    scaff_hdr = f"  {'scaff':>6s}" if traj_weight > 0 else ""
    void_hdr = f"  {'void':>6s}" if (void_weight > 0 or scan_void_weight > 0) else ""
    header = (f"{'epoch':>5s}  {'recon':>6s}  {'ltr':>6s}  {'case':>6s}  {'attn':>7s}  {'div':>6s}  "
              f"{'hit':>6s}  {'recode':>6s}{content_hdr}{void_hdr}"
              f"{traj_hdr}  {'rr_cls':>6s}  "
              f"{'rr_ltr':>6s}  {'rr_cas':>6s}"
              f"{v2_hdr}  "
              f"{'lr_attn':>8s}  {'lr_cls':>8s}  {'lr_rec':>8s}  {'lr_mot':>8s}{scaff_hdr}  time")
    logger = TrainingLogger(save_dir, header, start_epoch)

    for epoch in range(start_epoch, end_epoch):
        epoch_start = time.time()
        tracker.reset_epoch()

        # Trajectory scaffold weight (anneals over first traj_scaffold_ratio of training)
        if traj_scaffold_epochs > 0:
            traj_scaff_w = max(traj_scaffold_floor, 1.0 - epoch / traj_scaffold_epochs)
        else:
            traj_scaff_w = traj_scaffold_floor

        for img, clean, letters, cases, _fonts, partner_clean in dataloader:
            img = img.to(device)
            clean = clean.to(device)
            partner_clean = partner_clean.to(device)
            B = img.shape[0]

            letter_idx = torch.tensor([ord(l) - ord('A') for l in letters], device=device)
            case_idx = torch.tensor([0 if c == 'upper' else 1 for c in cases], device=device)
            case_float = case_idx.float().unsqueeze(1)

            # === MAIN FORWARD ===
            recon, letter_logits, case_logits, locations, latent, scan_content_logits = model(img, case_float)
            actual_n_scan = len(scan_content_logits)

            recon_loss = criterion(recon, img)
            letter_cls_loss = F.cross_entropy(letter_logits, letter_idx)
            case_cls_loss = F.cross_entropy(case_logits, case_idx)

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

            content_loss = torch.tensor(0.0, device=device)
            if content_weight > 0 and actual_n_scan > 0:
                bce = torch.nn.BCEWithLogitsLoss()
                scan_locs = locations[1:actual_n_scan + 1]
                for loc, logit in zip(scan_locs, scan_content_logits):
                    grid = loc.view(B, 1, 1, 2)
                    sampled = F.grid_sample(clean, grid, align_corners=True,
                                            padding_mode='zeros')
                    label = (sampled.view(-1, 1) > 0.1).float()
                    content_loss = content_loss + bce(logit, label)
                content_loss = content_loss / actual_n_scan

            void_loss = torch.tensor(0.0, device=device)
            if scan_void_weight > 0 and actual_n_scan > 0:
                scan_sample_locs = locations[1:actual_n_scan + 1]
                scan_ph, scan_pw = scan_patch_size
                void_loss = void_loss + scan_void_weight * void_repulsion(
                    clean, scan_sample_locs, scan_ph, scan_pw)
            if void_weight > 0:
                read_sample_locs = locations[actual_n_scan + 1:-1]
                void_loss = void_loss + void_weight * void_repulsion(
                    clean, read_sample_locs, patch_size, patch_size)

            recode_loss_val = 0.0
            if dataset.has_partners and recode_weight > 0:
                flipped_case = 1.0 - case_float
                recode_img = model.decoder(latent, flipped_case)
                recode_loss = criterion(recode_img, partner_clean)
                recode_loss_val = recode_loss.item()

            # === MULTI-HEAD BACKWARD (3 main passes) ===
            attn_opt.zero_grad()
            cls_opt.zero_grad()
            recon_opt.zero_grad()

            # Head 1: attention -> controller + sensors
            attn_total = attn_loss + diversity_weight * div_loss + content_weight * content_loss + void_loss
            attn_total.backward(retain_graph=True, inputs=attn_params)

            # Head 2: classification -> classifiers
            cls_total = letter_cls_loss + case_cls_loss
            cls_total.backward(retain_graph=True, inputs=cls_params)

            # Head 3: reconstruction -> decoder (frees main graph)
            recon_total = recon_loss
            if dataset.has_partners and recode_weight > 0:
                recon_total = recon_total + recode_weight * recode_loss
            recon_total.backward(inputs=recon_params)

            for params, opt in [(attn_params, attn_opt), (cls_params, cls_opt),
                                 (recon_params, recon_opt)]:
                clip_grad_norm_(params, 5.0)
                opt.step()

            # === DEFERRED MOTOR FORWARD (VRAM-safe: main graph is freed) ===
            motor_opt.zero_grad()
            latent_d = latent.detach()  # detach from freed main graph

            # Motor decode
            trajectory = model.motor_decoder(latent_d)

            # Trajectory MSE (xy only) + pen BCE — only if scaffold enabled
            if traj_weight > 0:
                gt_traj = batch_gt_trajectories(letters, cases, traj_data, device)
                traj_mse = F.mse_loss(trajectory[:, :, :2], gt_traj[:, :, :2])
                pen_bce_loss = bce_logits(trajectory[:, :, 2], gt_traj[:, :, 2])
            else:
                traj_mse = torch.tensor(0.0)
                pen_bce_loss = torch.tensor(0.0)

            # Render + re-read
            rendered = model._soft_render(trajectory, sigma=render_sigma)
            reread_enc = model._encode(rendered)
            rr_letter_logits = model.letter_classifier(reread_enc.latent)
            rr_case_logits = model.case_classifier(reread_enc.latent)
            rr_cls_loss = (F.cross_entropy(rr_letter_logits, letter_idx) +
                           F.cross_entropy(rr_case_logits, case_idx))

            # === Enhanced motor losses (v2) ===
            # Latent matching: rendered -> encoder -> latent2 must match original latent1
            latent_match_val = 0.0
            if latent_match_weight > 0:
                latent_match = F.mse_loss(reread_enc.latent, latent_d)
                latent_match_val = latent_match.item()

            # Frozen re-reader: static encoder, no co-adaptation
            frozen_rr_val = 0.0
            if frozen_rr_weight > 0:
                frozen_enc = _frozen_encode(rendered)
                frozen_ltr = frozen_modules['letter_classifier'](frozen_enc.latent)
                frozen_cls = frozen_modules['case_classifier'](frozen_enc.latent)
                frozen_rr_loss = (F.cross_entropy(frozen_ltr, letter_idx) +
                                  F.cross_entropy(frozen_cls, case_idx))
                frozen_rr_val = frozen_rr_loss.item()

            # Render matching: MSE between rendered trace and clean source image
            render_match_val = 0.0
            if render_match_weight > 0:
                render_match = F.mse_loss(rendered, clean)
                render_match_val = render_match.item()

            # Motor backward -- restrict to motor_decoder params
            motor_total = (traj_weight * traj_scaff_w * (traj_mse + pen_bce_loss) +
                           rr_cls_weight * rr_cls_loss)
            if latent_match_weight > 0:
                motor_total = motor_total + latent_match_weight * latent_match
            if frozen_rr_weight > 0:
                motor_total = motor_total + frozen_rr_weight * frozen_rr_loss
            if render_match_weight > 0:
                motor_total = motor_total + render_match_weight * render_match
            motor_total.backward(inputs=motor_params)
            clip_grad_norm_(motor_params, 5.0)
            motor_opt.step()

            # Metrics
            with torch.no_grad():
                hr, hi = fixation_hit_rate(clean, locations)
                rr_letter_acc = (rr_letter_logits.argmax(1) == letter_idx).float().mean().item()
                rr_case_acc = (rr_case_logits.argmax(1) == case_idx).float().mean().item()

            extra_metrics = {}
            if latent_match_weight > 0:
                extra_metrics['latent_match'] = latent_match_val
            if frozen_rr_weight > 0:
                extra_metrics['frozen_rr'] = frozen_rr_val
            if render_match_weight > 0:
                extra_metrics['render_match'] = render_match_val

            traj_metrics = {}
            if traj_weight > 0:
                traj_metrics = {'traj_mse': traj_mse, 'pen_bce': pen_bce_loss}
            tracker.update(
                recon=recon_loss, letter_cls=letter_cls_loss, case_cls=case_cls_loss,
                attn=attn_loss, div=div_loss, content=content_loss,
                void=void_loss,
                recode=recode_loss_val, hit_rate=hr, hit_intensity=hi,
                rr_cls=rr_cls_loss,
                rr_letter_acc=rr_letter_acc, rr_case_acc=rr_case_acc,
                **traj_metrics, **extra_metrics,
            )

        avgs = tracker.end_epoch()
        epoch_time = time.time() - epoch_start
        elapsed = time.time() - train_start
        done = epoch - start_epoch + 1
        eta = format_eta(elapsed, done, epochs - done)
        lr_attn = attn_sched.get_last_lr()[0]
        lr_cls = cls_sched.get_last_lr()[0]
        lr_recon = recon_sched.get_last_lr()[0]
        lr_motor = motor_sched.get_last_lr()[0]

        content_str = f"  Cont {avgs['content']:.4f}" if n_scan_glimpses > 0 else ""
        void_str = f"  Void {avgs['void']:.4f}" if (void_weight > 0 or scan_void_weight > 0) else ""
        traj_str = (f"  TrajMSE {avgs['traj_mse']:.4f}  PenBCE {avgs['pen_bce']:.4f}"
                    if traj_weight > 0 else "")
        scaff_str = f"  scaff {traj_scaff_w:.2f}" if traj_weight > 0 else ""
        v2_str = ""
        if latent_match_weight > 0:
            v2_str += f"  LatM {avgs['latent_match']:.4f}"
        if frozen_rr_weight > 0:
            v2_str += f"  FrzRR {avgs['frozen_rr']:.4f}"
        if render_match_weight > 0:
            v2_str += f"  RndM {avgs['render_match']:.4f}"
        case_disp = 'xxxxx' if single_case else f"{avgs['case_cls']:.4f}"
        recode_disp = 'xxxxx' if single_case else f"{avgs['recode']:.4f}"
        rr_case_disp = 'xxxxx' if single_case else f"{avgs['rr_case_acc']:.0%}"
        print(f"Epoch {epoch+1}/{end_epoch}: "
              f"Recon {avgs['recon']:.4f}  Ltr {avgs['letter_cls']:.4f}  "
              f"Case {case_disp}  Attn {avgs['attn']:.4f}  "
              f"Div {avgs['div']:.4f}{content_str}{void_str}  Hit {avgs['hit_rate']:.0%}  "
              f"Recode {recode_disp}{traj_str}  "
              f"RR_cls {avgs['rr_cls']:.4f}  RR_ltr {avgs['rr_letter_acc']:.0%}  "
              f"RR_case {rr_case_disp}{v2_str}  "
              f"lr {lr_attn:.6f}/{lr_cls:.6f}/{lr_recon:.6f}/{lr_motor:.6f}{scaff_str}  "
              f"[{epoch_time:.1f}s  ETA {eta}]")

        content_log = f"  {avgs['content']:>7.4f}" if n_scan_glimpses > 0 else ""
        void_log = f"  {avgs['void']:>6.4f}" if (void_weight > 0 or scan_void_weight > 0) else ""
        traj_log = (f"  {avgs['traj_mse']:>6.4f}  {avgs['pen_bce']:>6.4f}"
                    if traj_weight > 0 else "")
        scaff_log = f"  {traj_scaff_w:>6.4f}" if traj_weight > 0 else ""
        v2_log = ""
        if latent_match_weight > 0:
            v2_log += f"  {avgs['latent_match']:>6.4f}"
        if frozen_rr_weight > 0:
            v2_log += f"  {avgs['frozen_rr']:>6.4f}"
        if render_match_weight > 0:
            v2_log += f"  {avgs['render_match']:>6.4f}"
        case_log = ' xxxxx' if single_case else f"{avgs['case_cls']:>6.4f}"
        recode_log = ' xxxxx' if single_case else f"{avgs['recode']:>6.4f}"
        rr_case_log = ' xxxxx' if single_case else f"{avgs['rr_case_acc']:>6.4f}"
        logger.write_line(
            f"{epoch+1:>5d}  {avgs['recon']:>6.4f}  {avgs['letter_cls']:>6.4f}  "
            f"{case_log}  {avgs['attn']:>7.4f}  {avgs['div']:>6.4f}  "
            f"{avgs['hit_rate']:>6.4f}  {recode_log}{content_log}{void_log}"
            f"{traj_log}  {avgs['rr_cls']:>6.4f}  "
            f"{avgs['rr_letter_acc']:>6.4f}  {rr_case_log}"
            f"{v2_log}  "
            f"{lr_attn:>8.6f}  {lr_cls:>8.6f}  {lr_recon:>8.6f}  {lr_motor:>8.6f}{scaff_log}  {epoch_time:.1f}s")

        if epoch - start_epoch < tmax_attn:
            attn_sched.step()
        if epoch - start_epoch < tmax_cls:
            cls_sched.step()
        if epoch - start_epoch < tmax_recon:
            recon_sched.step()
        if epoch - start_epoch < tmax_motor:
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
                                'n_trajectory_points': n_trajectory_points,
                                'render_sigma': render_sigma,
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
                        'n_trajectory_points': n_trajectory_points,
                        'render_sigma': render_sigma,
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
    if traj_weight > 0:
        specs.append({'keys': ['traj_mse', 'pen_bce'], 'labels': ['Traj MSE', 'Pen BCE'],
                      'colors': ['tab:brown', 'tab:olive'], 'styles': ['-', '--'],
                      'title': 'Trajectory (scaffold anneals)', 'ylabel': 'Loss'})
    specs.extend([
        {'keys': ['rr_cls'], 'labels': ['Re-read CE'], 'colors': ['tab:red'],
         'title': 'Re-read classification', 'ylabel': 'Cross-Entropy',
         'hlines': [(np.log(26), f'Random ({np.log(26):.1f})', 'gray')]},
        {'keys': ['rr_letter_acc', 'rr_case_acc'],
         'labels': ['RR Letter Acc', 'RR Case Acc'],
         'colors': ['tab:purple', 'tab:pink'], 'styles': ['-', '--'],
         'title': 'Re-read accuracy (from rendered trajectory)',
         'ylabel': 'Accuracy', 'ylim': (0, 1)},
        {'keys': ['hit_rate', 'hit_intensity'],
         'labels': ['Hit rate', 'Intensity'],
         'colors': ['tab:purple', 'tab:purple'], 'styles': ['-', '--'],
         'title': 'Fixation hit rate (on sharp letter pixels)',
         'ylabel': 'Rate / Intensity', 'ylim': (0, 1)},
    ])
    # v2 motor loss panels
    if latent_match_weight > 0:
        specs.append({'keys': ['latent_match'], 'labels': ['Latent Match'],
                      'colors': ['tab:red'], 'title': 'Latent matching MSE',
                      'ylabel': 'MSE'})
    if frozen_rr_weight > 0:
        specs.append({'keys': ['frozen_rr'], 'labels': ['Frozen RR'],
                      'colors': ['tab:orange'], 'title': 'Frozen re-read CE',
                      'ylabel': 'Cross-Entropy',
                      'hlines': [(np.log(26), f'Random ({np.log(26):.1f})', 'gray')]})
    if render_match_weight > 0:
        specs.append({'keys': ['render_match'], 'labels': ['Render Match'],
                      'colors': ['tab:green'], 'title': 'Render matching MSE',
                      'ylabel': 'MSE'})
    plot_training_metrics(tracker, os.path.join(save_dir, 'training_metrics.png'), specs)

    total_time = time.time() - train_start
    total_min, total_s = divmod(int(total_time), 60)
    print(f"Training complete in {total_min}m{total_s:02d}s. "
          f"Model and graph saved in {save_dir}")
