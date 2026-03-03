import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import base64
import io
from PIL import Image

from fbrl import _resolve_device
from fbrl.data import LetterDataset, BigramDataset, WordDataset
from fbrl.model import VisionModel, BigramVisionModel, WordVisionModel, MotorVisionModel
from fbrl.losses import fixation_hit_rate


# --- Model Loading ---

def _load_model(model_dir, device):
    """Load a trained model from a checkpoint directory.

    Detects model_type in checkpoint ('single' or 'bigram') and instantiates
    the appropriate class. Two-phase bigram checkpoints have n_scan_glimpses
    and n_read_glimpses; legacy bigram checkpoints fall back to single-phase.

    Returns (model, n_glimpses, model_type) with the model in eval mode.
    Backwards compatible: checkpoints without model_type default to 'single'.
    """
    ckpt = torch.load(os.path.join(model_dir, 'model_final.pth'),
                      map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and 'model' in ckpt:
        n_glimpses = ckpt.get('n_glimpses', 10)
        n_scales = ckpt.get('n_scales', 1)
        model_type = ckpt.get('model_type', 'single')
        state_dict = ckpt['model']
    else:
        n_glimpses, n_scales = 10, 1
        model_type = 'single'
        state_dict = ckpt

    if model_type == 'letter_motor':
        n_scan = ckpt.get('n_scan_glimpses', 0)
        scan_ps = ckpt.get('scan_patch_size', (12, 18))
        patch_size = ckpt.get('patch_size', 12)
        n_traj = ckpt.get('n_trajectory_points', 32)
        render_sigma = ckpt.get('render_sigma', 1.5)
        model = MotorVisionModel(
            n_glimpses=n_glimpses, patch_size=patch_size, n_scales=n_scales,
            n_scan_glimpses=n_scan, scan_patch_size=scan_ps,
            n_trajectory_points=n_traj, render_sigma=render_sigma,
        ).to(device)
    elif model_type == 'word':
        n_scan = ckpt['n_scan_glimpses']
        n_read = ckpt['n_read_glimpses']
        scan_ps = ckpt.get('scan_patch_size', (12, 18))
        read_ps = ckpt.get('read_patch_size', 12)
        n_positions = ckpt.get('n_positions', 4)
        read_anchor = ckpt.get('read_anchor_scan_indices', None)
        if isinstance(read_anchor, list):
            read_anchor = tuple(read_anchor)
        n_rpg = ckpt.get('n_read_per_group', None)
        interleaved = ckpt.get('interleaved', False)
        model = WordVisionModel(
            n_scan_glimpses=n_scan, n_read_glimpses=n_read,
            scan_patch_size=scan_ps, read_patch_size=read_ps,
            n_scales=n_scales, n_positions=n_positions,
            read_anchor_scan_indices=read_anchor, n_read_per_group=n_rpg,
            interleaved=interleaved,
        ).to(device)
    elif model_type == 'bigram':
        # Two-phase checkpoint (has n_scan_glimpses)
        if 'n_scan_glimpses' in ckpt:
            n_scan = ckpt['n_scan_glimpses']
            n_read = ckpt['n_read_glimpses']
            scan_ps = ckpt.get('scan_patch_size', (12, 18))
            read_ps = ckpt.get('read_patch_size', 12)
            model = BigramVisionModel(
                n_scan_glimpses=n_scan, n_read_glimpses=n_read,
                scan_patch_size=scan_ps, read_patch_size=read_ps,
                n_scales=n_scales,
            ).to(device)
        else:
            # Legacy single-phase bigram checkpoint — not supported by new model
            raise ValueError(
                "Legacy single-phase bigram checkpoint. "
                "Re-train with two-phase architecture."
            )
    else:
        patch_size = ckpt.get('patch_size', 12)
        n_scan = ckpt.get('n_scan_glimpses', 0)
        scan_ps = ckpt.get('scan_patch_size', (12, 18))
        learnable_scan_x = ckpt.get('learnable_scan_x', False)
        model = VisionModel(
            n_glimpses=n_glimpses, patch_size=patch_size, n_scales=n_scales,
            n_scan_glimpses=n_scan, scan_patch_size=scan_ps,
            learnable_scan_x=learnable_scan_x,
        ).to(device)

    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model, n_glimpses, model_type


# --- Attention Visualization ---

def visualize_attention(img_tensor, locations, save_path):
    """Overlay fixation points and saccade arrows on image.

    Drops the last location (vestigial GRU "next" prediction that is
    never sampled) so only actually-used fixation points are shown.
    """
    img = img_tensor.squeeze(0).cpu().detach().numpy()
    H, W = img.shape

    # Drop vestigial last location (predicted but never sampled)
    locs = locations[:-1]

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.imshow(img, cmap='gray', vmin=0, vmax=1)

    colors = plt.cm.hot(np.linspace(0.2, 0.9, len(locs)))

    for i, loc in enumerate(locs):
        loc_np = loc[0].cpu().detach().numpy()
        px = (loc_np[0] + 1) / 2 * W
        py = (loc_np[1] + 1) / 2 * H

        ax.plot(px, py, 'o', color=colors[i], markersize=8,
                markeredgecolor='white', markeredgewidth=0.5)
        ax.annotate(str(i), (px, py), color='white', fontsize=6,
                    ha='center', va='center')

        if i > 0:
            prev_np = locs[i - 1][0].cpu().detach().numpy()
            prev_px = (prev_np[0] + 1) / 2 * W
            prev_py = (prev_np[1] + 1) / 2 * H
            ax.annotate('', xy=(px, py), xytext=(prev_px, prev_py),
                        arrowprops=dict(arrowstyle='->', color=colors[i], lw=1.5))

    ax.set_title(f'{len(locs)} fixations')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def _tensor_to_base64_png(tensor):
    """Convert (1, H, W) grayscale tensor to base64-encoded PNG string."""
    arr = tensor.squeeze(0).cpu().detach().numpy()
    arr = (arr * 255).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(arr, mode='L')
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('ascii')


# --- Testing ---

def test_model(model_dir, test_data_dir, output_dir='letter_results', device='auto'):
    device = _resolve_device(device)
    print(f"Testing on: {device}")

    os.makedirs(output_dir, exist_ok=True)

    model, _, _model_type = _load_model(model_dir, device)
    dataset = LetterDataset(test_data_dir)

    letter_correct = 0
    case_correct = 0
    total = 0
    mse_scores = []
    recode_mse_scores = []
    errors = []
    correct_list = []

    # Per-font tracking
    font_stats = {}  # font_name -> {'letter_ok': int, 'case_ok': int, 'total': int}

    for i in range(len(dataset)):
        img, clean, letter, case, font, partner_clean = dataset[i]
        img = img.unsqueeze(0).to(device)
        clean = clean.unsqueeze(0).to(device)
        partner_clean = partner_clean.unsqueeze(0).to(device)

        letter_idx = ord(letter) - ord('A')
        case_idx = 0 if case == 'upper' else 1
        case_float = torch.tensor([[float(case_idx)]], device=device)

        with torch.no_grad():
            recon, letter_logits, case_logits, locations, latent, _scan_logits = model(img, case_float)

        # Letter accuracy
        letter_pred = letter_logits.argmax(dim=1).item()
        letter_ok = letter_pred == letter_idx
        letter_correct += int(letter_ok)

        # Case accuracy
        case_pred = case_logits.argmax(dim=1).item()
        case_ok = case_pred == case_idx
        case_correct += int(case_ok)

        total += 1

        # Per-font stats
        if font not in font_stats:
            font_stats[font] = {'letter_ok': 0, 'case_ok': 0, 'total': 0}
        font_stats[font]['letter_ok'] += int(letter_ok)
        font_stats[font]['case_ok'] += int(case_ok)
        font_stats[font]['total'] += 1

        mse = F.mse_loss(recon, img).item()
        mse_scores.append(mse)

        # Recode: flip case, decode same latent
        with torch.no_grad():
            flipped_case = 1.0 - case_float
            recode_recon = model.decoder(latent, flipped_case)
            hr, hi = fixation_hit_rate(clean, locations)

        # Display character for output
        original_char = letter.lower() if case == 'lower' else letter
        pred_letter = chr(letter_pred + ord('A'))
        letter_mark = 'OK' if letter_ok else f'WRONG({pred_letter})'
        case_mark = 'OK' if case_ok else 'WRONG'

        # Include font in per-sample output only when multiple fonts present
        font_tag = f'  [{font}]' if len(font_stats) > 1 or font != 'default' else ''
        print(f"  {original_char}{font_tag}: Ltr={letter_mark}  Case={case_mark}  "
              f"MSE={mse:.4f}  Hit={hr:.0%}")

        # Output filenames include font when multiple fonts present
        suffix = f'_{font}' if len(set(dataset.fonts)) > 1 else ''

        # Save attention overlay
        visualize_attention(
            img.squeeze(0), locations,
            os.path.join(output_dir, f'attention_{original_char}{suffix}.png'),
        )

        # Save reconstruction
        recon_img = recon.squeeze().cpu().clamp(0, 1).detach().numpy()
        Image.fromarray((recon_img * 255).astype(np.uint8)).save(
            os.path.join(output_dir, f'recon_{original_char}{suffix}.png'),
        )

        # Save recode output + compute recode MSE against partner
        recode_img_np = recode_recon.squeeze().cpu().clamp(0, 1).detach().numpy()
        target_char = letter if case == 'lower' else letter.lower()
        Image.fromarray((recode_img_np * 255).astype(np.uint8)).save(
            os.path.join(output_dir, f'recode_{original_char}_to_{target_char}{suffix}.png'),
        )

        if dataset.has_partners:
            recode_mse = F.mse_loss(recode_recon, partner_clean).item()
            recode_mse_scores.append(recode_mse)

        # Track for summary
        if letter_ok and case_ok:
            correct_list.append((original_char, font))
        else:
            pred_display = pred_letter.lower() if case == 'lower' else pred_letter
            errors.append((original_char, font, pred_display, letter_ok, case_ok))

    letter_acc = letter_correct / total if total > 0 else 0
    case_acc = case_correct / total if total > 0 else 0
    avg_mse = np.mean(mse_scores) if mse_scores else 0
    avg_recode_mse = np.mean(recode_mse_scores) if recode_mse_scores else 0

    print(f"\nLetter accuracy: {letter_correct}/{total} ({letter_acc:.1%})")
    print(f"Case accuracy:   {case_correct}/{total} ({case_acc:.1%})")
    print(f"Avg reconstruction MSE: {avg_mse:.4f}")
    if recode_mse_scores:
        print(f"Avg recode MSE:         {avg_recode_mse:.4f}")

    # Per-font breakdown (only when multiple fonts)
    if len(font_stats) > 1:
        print(f"\nPer-font breakdown:")
        for fname in sorted(font_stats.keys()):
            s = font_stats[fname]
            lt_acc = s['letter_ok'] / s['total'] * 100
            cs_acc = s['case_ok'] / s['total'] * 100
            print(f"  {fname:<24s}: Letter {lt_acc:5.1f}%  Case {cs_acc:5.1f}%  ({s['total']} samples)")

    # Write summary file
    summary_path = os.path.join(output_dir, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"Letter: {letter_correct}/{total} ({letter_acc:.1%})\n")
        f.write(f"Case:   {case_correct}/{total} ({case_acc:.1%})\n")
        f.write(f"Avg MSE:      {avg_mse:.4f}\n")
        if recode_mse_scores:
            f.write(f"Avg recode:   {avg_recode_mse:.4f}\n")

        if errors:
            f.write(f"\nErrors ({len(errors)}):\n")
            for char, font, pred, l_ok, c_ok in sorted(errors):
                parts = []
                if not l_ok:
                    parts.append(f"ltr: {char}→{pred}")
                if not c_ok:
                    parts.append(f"case wrong")
                font_tag = f"  [{font}]" if font != 'default' else ''
                f.write(f"  {char}{font_tag}  ({', '.join(parts)})\n")

        if len(font_stats) > 1:
            f.write(f"\nPer-font:\n")
            for fname in sorted(font_stats.keys()):
                s = font_stats[fname]
                lt_acc = s['letter_ok'] / s['total'] * 100
                cs_acc = s['case_ok'] / s['total'] * 100
                f.write(f"  {fname:<24s}: Letter {lt_acc:5.1f}%  Case {cs_acc:5.1f}%  "
                        f"({s['total']})\n")

        f.write(f"\nCorrect ({len(correct_list)}):\n")
        line = '  '
        for i, (char, _font) in enumerate(sorted(correct_list)):
            line += char
            if i < len(correct_list) - 1:
                line += ', '
            if len(line) > 78:
                f.write(line + '\n')
                line = '  '
        if line.strip():
            f.write(line + '\n')

    print(f"Summary written to {summary_path}")
    print(f"Results saved in {output_dir}")


# --- Visualization ---

def visualize_model(model_dir, data_dir, output_dir='letter_visualizations', device='auto'):
    device = _resolve_device(device)
    print(f"Visualizing on: {device}")

    os.makedirs(output_dir, exist_ok=True)

    model, _, _model_type = _load_model(model_dir, device)
    dataset = LetterDataset(data_dir)

    multi_font = len(set(dataset.fonts)) > 1
    for i in range(len(dataset)):
        img, _clean, letter, case, font, _partner = dataset[i]
        img = img.unsqueeze(0).to(device)

        case_idx = 0 if case == 'upper' else 1
        case_float = torch.tensor([[float(case_idx)]], device=device)

        with torch.no_grad():
            _, _, _, locations, _, _ = model(img, case_float)

        original_char = letter.lower() if case == 'lower' else letter
        suffix = f'_{font}' if multi_font else ''
        visualize_attention(
            img.squeeze(0), locations,
            os.path.join(output_dir, f'attention_{original_char}{suffix}.png'),
        )
        print(f"Saved attention visualization for '{original_char}'{f' [{font}]' if multi_font else ''}")

    print(f"Visualizations saved in {output_dir}")


# --- Attention Atlas ---

def _atlas_html_template():
    """Return self-contained HTML/CSS/JS template for the interactive attention atlas.

    Contains a {atlas_json} placeholder to be filled with the data payload.
    All rendering happens client-side: Gaussian-splat heatmaps on Canvas,
    hot colormap, fixation path overlay, per-font drill-down on click.
    """
    return r'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Attention Atlas</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { background: #1a1a2e; color: #e0e0e0; font-family: system-ui, sans-serif; }
#controls {
  position: sticky; top: 0; z-index: 10; background: #16213e;
  padding: 10px 20px; display: flex; gap: 20px; align-items: center;
  border-bottom: 1px solid #0f3460; flex-wrap: wrap;
}
#controls label { font-size: 13px; color: #a0a0c0; }
#controls button {
  background: #0f3460; color: #e0e0e0; border: 1px solid #1a1a4e;
  padding: 5px 12px; border-radius: 4px; cursor: pointer; font-size: 13px;
}
#controls button.active { background: #e94560; border-color: #e94560; }
#controls input[type=range] { width: 120px; vertical-align: middle; }
#grid {
  display: grid; grid-template-columns: repeat(26, 1fr);
  gap: 4px; padding: 12px; max-width: 1800px; margin: 0 auto;
}
.cell {
  position: relative; cursor: pointer; border: 2px solid transparent;
  border-radius: 4px; overflow: hidden; transition: transform 0.15s;
  aspect-ratio: 1;
}
.cell:hover { transform: scale(1.3); z-index: 5; }
.cell.correct { border-color: #2ecc71; }
.cell.partial { border-color: #f39c12; }
.cell.wrong { border-color: #e74c3c; }
.cell.selected { border-color: #3498db; box-shadow: 0 0 8px #3498db; }
.cell canvas { width: 100%; height: 100%; display: block; }
.cell-label {
  position: absolute; bottom: 1px; right: 3px; font-size: 10px;
  color: #fff; text-shadow: 0 0 3px #000; pointer-events: none;
}
#detail-panel {
  background: #16213e; border-top: 2px solid #0f3460;
  padding: 16px; display: none; max-width: 1800px; margin: 0 auto;
}
#detail-title { font-size: 18px; margin-bottom: 12px; color: #e94560; }
#detail-grid {
  display: flex; gap: 8px; flex-wrap: wrap; justify-content: center;
}
.detail-cell {
  text-align: center; border: 2px solid transparent; border-radius: 4px;
  padding: 4px; background: #1a1a2e;
}
.detail-cell.correct { border-color: #2ecc71; }
.detail-cell.wrong { border-color: #e74c3c; }
.detail-cell canvas { width: 240px; height: 240px; display: block; }
.detail-cell .font-name { font-size: 11px; color: #a0a0c0; margin-top: 2px; }
.detail-cell .pred-info { font-size: 10px; color: #888; }
</style>
</head>
<body>
<div id="controls">
  <span style="font-weight:bold;color:#e94560;">Attention Atlas</span>
  <div>
    <label>View:</label>
    <button id="btn-heatmap" class="active" onclick="setView('heatmap')">Heatmap</button>
    <button id="btn-path" onclick="setView('path')">Path</button>
  </div>
  <div>
    <label>Case:</label>
    <button id="btn-both" class="active" onclick="setCase('both')">Both</button>
    <button id="btn-upper" onclick="setCase('upper')">Upper</button>
    <button id="btn-lower" onclick="setCase('lower')">Lower</button>
  </div>
  <div>
    <label>Opacity:</label>
    <input type="range" id="opacity-slider" min="0" max="100" value="60"
           oninput="setOpacity(this.value)">
    <span id="opacity-val">60%</span>
  </div>
  <div style="margin-left:auto;font-size:12px;color:#666;">
    <span id="stats"></span>
  </div>
</div>
<div id="grid"></div>
<div id="detail-panel">
  <div id="detail-title"></div>
  <div id="detail-grid"></div>
</div>

<script>
const DATA = ATLAS_JSON_PLACEHOLDER;
let viewMode = 'heatmap';
let caseFilter = 'both';
let opacity = 0.6;
let selectedIdx = -1;

// Hot colormap (matches matplotlib 'hot': black -> red -> yellow -> white)
function hotColor(t) {
  t = Math.max(0, Math.min(1, t));
  let r, g, b;
  if (t < 0.33) { r = t / 0.33; g = 0; b = 0; }
  else if (t < 0.66) { r = 1; g = (t - 0.33) / 0.33; b = 0; }
  else { r = 1; g = 1; b = (t - 0.66) / 0.34; }
  return [r * 255 | 0, g * 255 | 0, b * 255 | 0];
}

// Render Gaussian-splat heatmap into an ImageData
function renderHeatmap(fixations, size) {
  // fixations: [[x,y], ...] in normalized [-1,1] coords
  const sigma = size * 0.06;  // Gaussian splat radius
  const sigma2 = 2 * sigma * sigma;
  const field = new Float32Array(size * size);
  let maxVal = 0;

  for (const [fx, fy] of fixations) {
    const cx = (fx + 1) / 2 * size;
    const cy = (fy + 1) / 2 * size;
    const r = Math.ceil(sigma * 3);
    const x0 = Math.max(0, cx - r | 0), x1 = Math.min(size - 1, cx + r | 0);
    const y0 = Math.max(0, cy - r | 0), y1 = Math.min(size - 1, cy + r | 0);
    for (let y = y0; y <= y1; y++) {
      for (let x = x0; x <= x1; x++) {
        const dx = x - cx, dy = y - cy;
        const v = Math.exp(-(dx * dx + dy * dy) / sigma2);
        const idx = y * size + x;
        field[idx] += v;
        if (field[idx] > maxVal) maxVal = field[idx];
      }
    }
  }

  if (maxVal === 0) maxVal = 1;
  const data = new Uint8ClampedArray(size * size * 4);
  for (let i = 0; i < size * size; i++) {
    const t = field[i] / maxVal;
    const [r, g, b] = hotColor(t);
    data[i * 4] = r; data[i * 4 + 1] = g; data[i * 4 + 2] = b;
    data[i * 4 + 3] = t > 0.01 ? (t * opacity * 255) | 0 : 0;
  }
  return new ImageData(data, size, size);
}

// Render fixation path as numbered circles + arrows
function renderPath(ctx, fixations, size) {
  const n = fixations.length;
  for (let i = 0; i < n; i++) {
    const [fx, fy] = fixations[i];
    const cx = (fx + 1) / 2 * size;
    const cy = (fy + 1) / 2 * size;
    const t = i / Math.max(1, n - 1);
    const [r, g, b] = hotColor(0.2 + t * 0.7);
    const color = `rgb(${r},${g},${b})`;

    // Arrow from previous fixation
    if (i > 0) {
      const [px, py] = fixations[i - 1];
      const pcx = (px + 1) / 2 * size;
      const pcy = (py + 1) / 2 * size;
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(pcx, pcy); ctx.lineTo(cx, cy); ctx.stroke();
      // Arrowhead
      const angle = Math.atan2(cy - pcy, cx - pcx);
      const hl = 5;
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(cx - hl * Math.cos(angle - 0.4), cy - hl * Math.sin(angle - 0.4));
      ctx.moveTo(cx, cy);
      ctx.lineTo(cx - hl * Math.cos(angle + 0.4), cy - hl * Math.sin(angle + 0.4));
      ctx.stroke();
    }

    // Circle
    ctx.beginPath(); ctx.arc(cx, cy, 4, 0, Math.PI * 2);
    ctx.fillStyle = color; ctx.fill();
    ctx.strokeStyle = '#fff'; ctx.lineWidth = 0.5; ctx.stroke();

    // Number
    ctx.fillStyle = '#fff'; ctx.font = '7px sans-serif';
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillText(i.toString(), cx, cy);
  }
}

// Draw a single cell (grayscale image + heatmap/path overlay)
function drawCell(canvas, b64, fixations, size) {
  canvas.width = size; canvas.height = size;
  const ctx = canvas.getContext('2d');

  const img = new window.Image();
  img.onload = function() {
    ctx.drawImage(img, 0, 0, size, size);

    if (viewMode === 'heatmap') {
      // putImageData ignores alpha, so splat heatmap onto a temp canvas,
      // then drawImage (which respects alpha) to composite over grayscale
      const hm = renderHeatmap(fixations, size);
      const tmp = document.createElement('canvas');
      tmp.width = size; tmp.height = size;
      tmp.getContext('2d').putImageData(hm, 0, 0);
      ctx.drawImage(tmp, 0, 0);
    } else {
      renderPath(ctx, fixations, size);
    }
  };
  img.src = 'data:image/png;base64,' + b64;
}

// Aggregate fixations across all fonts for a letter
function aggregateFixations(entry) {
  const all = [];
  for (const fname of DATA.font_names) {
    const fd = entry.fonts[fname];
    if (fd) all.push(...fd.fixations);
  }
  return all;
}

// Pick a representative clean image (first available font)
function representativeImage(entry) {
  for (const fname of DATA.font_names) {
    const fd = entry.fonts[fname];
    if (fd) return fd.clean_b64;
  }
  return '';
}

// Check correctness across fonts
function correctnessClass(entry) {
  let ok = 0, total = 0;
  for (const fname of DATA.font_names) {
    const fd = entry.fonts[fname];
    if (fd) { total++; if (fd.correct) ok++; }
  }
  if (ok === total) return 'correct';
  if (ok === 0) return 'wrong';
  return 'partial';
}

function buildGrid() {
  const grid = document.getElementById('grid');
  grid.innerHTML = '';
  let shown = 0, totalCorrect = 0, totalSamples = 0;

  DATA.letters.forEach((entry, idx) => {
    if (caseFilter === 'upper' && entry.case !== 'upper') { grid.appendChild(createPlaceholder()); return; }
    if (caseFilter === 'lower' && entry.case !== 'lower') { grid.appendChild(createPlaceholder()); return; }

    const cell = document.createElement('div');
    cell.className = 'cell ' + correctnessClass(entry);
    if (idx === selectedIdx) cell.classList.add('selected');

    const canvas = document.createElement('canvas');
    const fixations = aggregateFixations(entry);
    const b64 = representativeImage(entry);
    drawCell(canvas, b64, fixations, DATA.image_size);
    cell.appendChild(canvas);

    const label = document.createElement('span');
    label.className = 'cell-label';
    label.textContent = entry.char;
    cell.appendChild(label);

    cell.onclick = () => { selectedIdx = idx; buildGrid(); showDetail(entry); };
    grid.appendChild(cell);
    shown++;

    // Stats
    for (const fname of DATA.font_names) {
      const fd = entry.fonts[fname];
      if (fd) { totalSamples++; if (fd.correct) totalCorrect++; }
    }
  });

  document.getElementById('stats').textContent =
    `${totalCorrect}/${totalSamples} correct (${(totalCorrect/totalSamples*100).toFixed(1)}%)`;
}

function createPlaceholder() {
  const div = document.createElement('div');
  div.style.visibility = 'hidden';
  return div;
}

function showDetail(entry) {
  const panel = document.getElementById('detail-panel');
  const title = document.getElementById('detail-title');
  const grid = document.getElementById('detail-grid');

  title.textContent = `${entry.char} — per-font attention (${DATA.font_names.length} fonts)`;
  grid.innerHTML = '';
  panel.style.display = 'block';

  for (const fname of DATA.font_names) {
    const fd = entry.fonts[fname];
    if (!fd) continue;

    const cell = document.createElement('div');
    cell.className = 'detail-cell ' + (fd.correct ? 'correct' : 'wrong');

    const canvas = document.createElement('canvas');
    drawCell(canvas, fd.clean_b64, fd.fixations, 120);
    cell.appendChild(canvas);

    const fnLabel = document.createElement('div');
    fnLabel.className = 'font-name';
    fnLabel.textContent = fname;
    cell.appendChild(fnLabel);

    const pred = document.createElement('div');
    pred.className = 'pred-info';
    pred.textContent = fd.correct ? 'OK' :
      `pred: ${fd.letter_pred}${fd.case_pred === 'lower' ? '.lower' : ''}`;
    cell.appendChild(pred);

    grid.appendChild(cell);
  }

  panel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function setView(mode) {
  viewMode = mode;
  document.getElementById('btn-heatmap').className = mode === 'heatmap' ? 'active' : '';
  document.getElementById('btn-path').className = mode === 'path' ? 'active' : '';
  buildGrid();
  if (selectedIdx >= 0) showDetail(DATA.letters[selectedIdx]);
}

function setCase(mode) {
  caseFilter = mode;
  document.getElementById('btn-both').className = mode === 'both' ? 'active' : '';
  document.getElementById('btn-upper').className = mode === 'upper' ? 'active' : '';
  document.getElementById('btn-lower').className = mode === 'lower' ? 'active' : '';
  selectedIdx = -1;
  document.getElementById('detail-panel').style.display = 'none';
  buildGrid();
}

function setOpacity(val) {
  opacity = val / 100;
  document.getElementById('opacity-val').textContent = val + '%';
  buildGrid();
  if (selectedIdx >= 0) showDetail(DATA.letters[selectedIdx]);
}

// Initial render
buildGrid();
</script>
</body>
</html>'''


def generate_atlas(model_dir, test_data_dir, output_path='data/letter_atlas.html', device='auto'):
    """Generate an interactive HTML attention atlas from a trained model.

    Runs inference on all test samples, collects fixation coordinates and clean
    images, then renders a self-contained HTML file with Canvas-based heatmaps
    and per-font drill-down.
    """
    device = _resolve_device(device)
    print(f"Generating attention atlas on: {device}")

    model, n_glimpses, _model_type = _load_model(model_dir, device)
    dataset = LetterDataset(test_data_dir)
    font_names = sorted(set(dataset.fonts))

    # Collect per-(letter, case) data keyed by font
    entries = {}  # (letter, case) -> {char, fonts: {font -> {fixations, clean_b64, ...}}}

    for i in range(len(dataset)):
        img, clean, letter, case, font, _partner = dataset[i]
        img_dev = img.unsqueeze(0).to(device)

        letter_idx = ord(letter) - ord('A')
        case_idx = 0 if case == 'upper' else 1
        case_float = torch.tensor([[float(case_idx)]], device=device)

        with torch.no_grad():
            _, letter_logits, case_logits, locations, _, _ = model(img_dev, case_float)

        # Collect fixation coordinates as [[x, y], ...] in normalized [-1, 1]
        # Drop last location (vestigial GRU prediction, never sampled)
        fixations = []
        for loc in locations[:-1]:
            loc_np = loc[0].cpu().detach().tolist()
            fixations.append([round(loc_np[0], 4), round(loc_np[1], 4)])

        letter_pred = chr(letter_logits.argmax(dim=1).item() + ord('A'))
        case_pred = 'upper' if case_logits.argmax(dim=1).item() == 0 else 'lower'
        correct = (letter_pred == letter) and (case_pred == case)

        key = (letter, case)
        if key not in entries:
            char = letter.lower() if case == 'lower' else letter
            entries[key] = {'letter': letter, 'case': case, 'char': char, 'fonts': {}}

        entries[key]['fonts'][font] = {
            'fixations': fixations,
            'clean_b64': _tensor_to_base64_png(clean),
            'letter_pred': letter_pred,
            'case_pred': case_pred,
            'correct': correct,
        }

        print(f"  {entries[key]['char']} [{font}]: {'OK' if correct else 'WRONG'}")

    # Build ordered list: A-Z then a-z
    letters_list = []
    for letter_char in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
        key = (letter_char, 'upper')
        if key in entries:
            letters_list.append(entries[key])
    for letter_char in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
        key = (letter_char, 'lower')
        if key in entries:
            letters_list.append(entries[key])

    atlas_data = {
        'image_size': 128,
        'n_fixations': n_glimpses + 1,  # initial center + n_glimpses saccades
        'font_names': font_names,
        'letters': letters_list,
    }

    # Inject JSON into HTML template
    atlas_json = json.dumps(atlas_data)
    html = _atlas_html_template().replace('ATLAS_JSON_PLACEHOLDER', atlas_json)

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(html)

    total = sum(len(e['fonts']) for e in letters_list)
    correct = sum(1 for e in letters_list for fd in e['fonts'].values() if fd['correct'])
    size_kb = os.path.getsize(output_path) / 1024
    print(f"\nAtlas: {len(letters_list)} letters x {len(font_names)} fonts = {total} samples")
    print(f"Accuracy: {correct}/{total} ({correct/total*100:.1f}%)")
    print(f"Written to {output_path} ({size_kb:.0f} KB)")


# --- Bigram Testing ---

def test_bigram_model(model_dir, test_data_dir, output_dir='bigram_results', device='auto'):
    """Test a trained BigramVisionModel on bigram test data.

    Reports per-position accuracy, both-correct accuracy, and reconstruction MSE.
    """
    device = _resolve_device(device)
    print(f"Bigram testing on: {device}")

    os.makedirs(output_dir, exist_ok=True)

    model, _, model_type = _load_model(model_dir, device)
    if model_type != 'bigram':
        print(f"Warning: checkpoint model_type is '{model_type}', expected 'bigram'")

    dataset = BigramDataset(test_data_dir)

    pos1_correct = 0
    pos2_correct = 0
    both_correct = 0
    total = 0
    mse_scores = []

    # Per-font tracking
    font_stats = {}
    # Per-sample results for summary file
    errors = []    # list of (bigram, font, pred1_char, pred2_char, ok1, ok2)
    correct = []   # list of (bigram, font)

    for i in range(len(dataset)):
        img, clean, letter1, letter2, bigram, font = dataset[i]
        img = img.unsqueeze(0).to(device)

        idx1 = ord(letter1) - ord('a')
        idx2 = ord(letter2) - ord('a')

        with torch.no_grad():
            recon, logits_list, locations, _ = model(img)

        pred1 = logits_list[0].argmax(dim=1).item()
        pred2 = logits_list[1].argmax(dim=1).item()
        ok1 = pred1 == idx1
        ok2 = pred2 == idx2
        ok_both = ok1 and ok2

        pos1_correct += int(ok1)
        pos2_correct += int(ok2)
        both_correct += int(ok_both)
        total += 1

        # Per-font stats
        if font not in font_stats:
            font_stats[font] = {'pos1_ok': 0, 'pos2_ok': 0, 'both_ok': 0, 'total': 0}
        font_stats[font]['pos1_ok'] += int(ok1)
        font_stats[font]['pos2_ok'] += int(ok2)
        font_stats[font]['both_ok'] += int(ok_both)
        font_stats[font]['total'] += 1

        mse = F.mse_loss(recon, img).item()
        mse_scores.append(mse)

        # Display
        pred1_char = chr(pred1 + ord('a'))
        pred2_char = chr(pred2 + ord('a'))
        mark1 = 'OK' if ok1 else f'WRONG({pred1_char})'
        mark2 = 'OK' if ok2 else f'WRONG({pred2_char})'
        font_tag = f'  [{font}]' if len(font_stats) > 1 or font != 'default' else ''
        print(f"  {bigram}{font_tag}: P1={mark1}  P2={mark2}  MSE={mse:.4f}")

        # Track for summary
        if ok_both:
            correct.append((bigram, font))
        else:
            errors.append((bigram, font, pred1_char, pred2_char, ok1, ok2))

        # Save attention overlay
        visualize_attention(
            img.squeeze(0), locations,
            os.path.join(output_dir, f'attention_{bigram}_{font}.png'),
        )

        # Save reconstruction
        recon_img = recon.squeeze().cpu().clamp(0, 1).detach().numpy()
        Image.fromarray((recon_img * 255).astype(np.uint8)).save(
            os.path.join(output_dir, f'recon_{bigram}_{font}.png'),
        )

    acc1 = pos1_correct / total if total > 0 else 0
    acc2 = pos2_correct / total if total > 0 else 0
    acc_both = both_correct / total if total > 0 else 0
    avg_mse = np.mean(mse_scores) if mse_scores else 0

    print(f"\nPos 1 accuracy:  {pos1_correct}/{total} ({acc1:.1%})")
    print(f"Pos 2 accuracy:  {pos2_correct}/{total} ({acc2:.1%})")
    print(f"Both correct:    {both_correct}/{total} ({acc_both:.1%})")
    print(f"Avg reconstruction MSE: {avg_mse:.4f}")

    # Per-font breakdown
    if len(font_stats) > 1:
        print(f"\nPer-font breakdown:")
        for fname in sorted(font_stats.keys()):
            s = font_stats[fname]
            a1 = s['pos1_ok'] / s['total'] * 100
            a2 = s['pos2_ok'] / s['total'] * 100
            ab = s['both_ok'] / s['total'] * 100
            print(f"  {fname:<24s}: P1 {a1:5.1f}%  P2 {a2:5.1f}%  Both {ab:5.1f}%  ({s['total']} samples)")

    # Write summary file
    summary_path = os.path.join(output_dir, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"Both-correct: {both_correct}/{total} ({acc_both:.1%})\n")
        f.write(f"Pos 1:        {pos1_correct}/{total} ({acc1:.1%})\n")
        f.write(f"Pos 2:        {pos2_correct}/{total} ({acc2:.1%})\n")
        f.write(f"Avg MSE:      {avg_mse:.4f}\n")

        if errors:
            f.write(f"\nErrors ({len(errors)}):\n")
            for bigram, font, p1, p2, ok1, ok2 in sorted(errors):
                parts = []
                if not ok1:
                    parts.append(f"pos1: {bigram[0]}→{p1}")
                if not ok2:
                    parts.append(f"pos2: {bigram[1]}→{p2}")
                font_tag = f"  [{font}]" if font != 'default' else ''
                f.write(f"  {bigram} → {p1}{p2}{font_tag}  ({', '.join(parts)})\n")

        if len(font_stats) > 1:
            f.write(f"\nPer-font:\n")
            for fname in sorted(font_stats.keys()):
                s = font_stats[fname]
                a1 = s['pos1_ok'] / s['total'] * 100
                a2 = s['pos2_ok'] / s['total'] * 100
                ab = s['both_ok'] / s['total'] * 100
                f.write(f"  {fname:<24s}: P1 {a1:5.1f}%  P2 {a2:5.1f}%  "
                        f"Both {ab:5.1f}%  ({s['total']})\n")

        f.write(f"\nCorrect ({len(correct)}):\n")
        line = '  '
        for i, (bigram, _font) in enumerate(sorted(correct)):
            line += bigram
            if i < len(correct) - 1:
                line += ', '
            if len(line) > 78:
                f.write(line + '\n')
                line = '  '
        if line.strip():
            f.write(line + '\n')

    print(f"Summary written to {summary_path}")
    print(f"Results saved in {output_dir}")


# --- Bigram Attention Atlas ---

def _bigram_atlas_html_template():
    """Return self-contained HTML/CSS/JS template for the bigram attention atlas.

    Adapted from the single-letter atlas with key differences:
    - Grid flows with auto-fill (200 bigrams vs 52 letters)
    - Cells have 1:1 aspect ratio (128x128 images)
    - No case filter (bigrams are all lowercase)
    - Rendering functions take (width, height) instead of single size
    - Correctness: green=both ok, yellow=one ok, red=neither
    """
    return r'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Bigram Attention Atlas</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { background: #1a1a2e; color: #e0e0e0; font-family: system-ui, sans-serif; }
#controls {
  position: sticky; top: 0; z-index: 10; background: #16213e;
  padding: 10px 20px; display: flex; gap: 20px; align-items: center;
  border-bottom: 1px solid #0f3460; flex-wrap: wrap;
}
#controls label { font-size: 13px; color: #a0a0c0; }
#controls button {
  background: #0f3460; color: #e0e0e0; border: 1px solid #1a1a4e;
  padding: 5px 12px; border-radius: 4px; cursor: pointer; font-size: 13px;
}
#controls button.active { background: #e94560; border-color: #e94560; }
#controls input[type=range] { width: 120px; vertical-align: middle; }
#grid {
  display: grid; grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
  gap: 4px; padding: 12px; max-width: 1800px; margin: 0 auto;
}
.cell {
  position: relative; cursor: pointer; border: 2px solid transparent;
  border-radius: 4px; overflow: hidden; transition: transform 0.15s;
  aspect-ratio: 1;
}
.cell:hover { transform: scale(1.3); z-index: 5; }
.cell.correct { border-color: #2ecc71; }
.cell.partial { border-color: #f39c12; }
.cell.wrong { border-color: #e74c3c; }
.cell.selected { border-color: #3498db; box-shadow: 0 0 8px #3498db; }
.cell canvas { width: 100%; height: 100%; display: block; }
.cell-label {
  position: absolute; bottom: 1px; right: 3px; font-size: 10px;
  color: #fff; text-shadow: 0 0 3px #000; pointer-events: none;
}
#detail-panel {
  background: #16213e; border-top: 2px solid #0f3460;
  padding: 16px; display: none; max-width: 1800px; margin: 0 auto;
}
#detail-title { font-size: 18px; margin-bottom: 12px; color: #e94560; }
#detail-grid {
  display: flex; gap: 8px; flex-wrap: wrap; justify-content: center;
}
.detail-cell {
  text-align: center; border: 2px solid transparent; border-radius: 4px;
  padding: 4px; background: #1a1a2e;
}
.detail-cell.correct { border-color: #2ecc71; }
.detail-cell.partial { border-color: #f39c12; }
.detail-cell.wrong { border-color: #e74c3c; }
.detail-cell canvas { width: 240px; height: 240px; display: block; }
.detail-cell .font-name { font-size: 11px; color: #a0a0c0; margin-top: 2px; }
.detail-cell .pred-info { font-size: 10px; color: #888; }
</style>
</head>
<body>
<div id="controls">
  <span style="font-weight:bold;color:#e94560;">Bigram Attention Atlas</span>
  <div>
    <label>View:</label>
    <button id="btn-heatmap" class="active" onclick="setView('heatmap')">Heatmap</button>
    <button id="btn-path" onclick="setView('path')">Path</button>
  </div>
  <div>
    <label>Opacity:</label>
    <input type="range" id="opacity-slider" min="0" max="100" value="60"
           oninput="setOpacity(this.value)">
    <span id="opacity-val">60%</span>
  </div>
  <div style="margin-left:auto;font-size:12px;color:#666;">
    <span id="stats"></span>
  </div>
</div>
<div id="grid"></div>
<div id="detail-panel">
  <div id="detail-title"></div>
  <div id="detail-grid"></div>
</div>

<script>
const DATA = ATLAS_JSON_PLACEHOLDER;
const W = DATA.image_width;
const H = DATA.image_height;
let viewMode = 'heatmap';
let opacity = 0.6;
let selectedIdx = -1;

// Hot colormap (matches matplotlib 'hot': black -> red -> yellow -> white)
function hotColor(t) {
  t = Math.max(0, Math.min(1, t));
  let r, g, b;
  if (t < 0.33) { r = t / 0.33; g = 0; b = 0; }
  else if (t < 0.66) { r = 1; g = (t - 0.33) / 0.33; b = 0; }
  else { r = 1; g = 1; b = (t - 0.66) / 0.34; }
  return [r * 255 | 0, g * 255 | 0, b * 255 | 0];
}

// Render Gaussian-splat heatmap into an ImageData
function renderHeatmap(fixations, width, height) {
  const sigma = Math.min(width, height) * 0.06;
  const sigma2 = 2 * sigma * sigma;
  const field = new Float32Array(width * height);
  let maxVal = 0;

  for (const [fx, fy] of fixations) {
    const cx = (fx + 1) / 2 * width;
    const cy = (fy + 1) / 2 * height;
    const r = Math.ceil(sigma * 3);
    const x0 = Math.max(0, cx - r | 0), x1 = Math.min(width - 1, cx + r | 0);
    const y0 = Math.max(0, cy - r | 0), y1 = Math.min(height - 1, cy + r | 0);
    for (let y = y0; y <= y1; y++) {
      for (let x = x0; x <= x1; x++) {
        const dx = x - cx, dy = y - cy;
        const v = Math.exp(-(dx * dx + dy * dy) / sigma2);
        const idx = y * width + x;
        field[idx] += v;
        if (field[idx] > maxVal) maxVal = field[idx];
      }
    }
  }

  if (maxVal === 0) maxVal = 1;
  const data = new Uint8ClampedArray(width * height * 4);
  for (let i = 0; i < width * height; i++) {
    const t = field[i] / maxVal;
    const [r, g, b] = hotColor(t);
    data[i * 4] = r; data[i * 4 + 1] = g; data[i * 4 + 2] = b;
    data[i * 4 + 3] = t > 0.01 ? (t * opacity * 255) | 0 : 0;
  }
  return new ImageData(data, width, height);
}

// Render fixation path as numbered circles + arrows
function renderPath(ctx, fixations, width, height) {
  const n = fixations.length;
  for (let i = 0; i < n; i++) {
    const [fx, fy] = fixations[i];
    const cx = (fx + 1) / 2 * width;
    const cy = (fy + 1) / 2 * height;
    const t = i / Math.max(1, n - 1);
    const [r, g, b] = hotColor(0.2 + t * 0.7);
    const color = `rgb(${r},${g},${b})`;

    if (i > 0) {
      const [px, py] = fixations[i - 1];
      const pcx = (px + 1) / 2 * width;
      const pcy = (py + 1) / 2 * height;
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(pcx, pcy); ctx.lineTo(cx, cy); ctx.stroke();
      const angle = Math.atan2(cy - pcy, cx - pcx);
      const hl = 5;
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(cx - hl * Math.cos(angle - 0.4), cy - hl * Math.sin(angle - 0.4));
      ctx.moveTo(cx, cy);
      ctx.lineTo(cx - hl * Math.cos(angle + 0.4), cy - hl * Math.sin(angle + 0.4));
      ctx.stroke();
    }

    ctx.beginPath(); ctx.arc(cx, cy, 4, 0, Math.PI * 2);
    ctx.fillStyle = color; ctx.fill();
    ctx.strokeStyle = '#fff'; ctx.lineWidth = 0.5; ctx.stroke();

    ctx.fillStyle = '#fff'; ctx.font = '7px sans-serif';
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillText(i.toString(), cx, cy);
  }
}

// Draw a single cell (grayscale image + heatmap/path overlay)
function drawCell(canvas, b64, fixations, width, height) {
  canvas.width = width; canvas.height = height;
  const ctx = canvas.getContext('2d');

  const img = new window.Image();
  img.onload = function() {
    ctx.drawImage(img, 0, 0, width, height);

    if (viewMode === 'heatmap') {
      const hm = renderHeatmap(fixations, width, height);
      const tmp = document.createElement('canvas');
      tmp.width = width; tmp.height = height;
      tmp.getContext('2d').putImageData(hm, 0, 0);
      ctx.drawImage(tmp, 0, 0);
    } else {
      renderPath(ctx, fixations, width, height);
    }
  };
  img.src = 'data:image/png;base64,' + b64;
}

// Aggregate fixations across all fonts for a bigram
function aggregateFixations(entry) {
  const all = [];
  for (const fname of DATA.font_names) {
    const fd = entry.fonts[fname];
    if (fd) all.push(...fd.fixations);
  }
  return all;
}

// Pick a representative clean image (first available font)
function representativeImage(entry) {
  for (const fname of DATA.font_names) {
    const fd = entry.fonts[fname];
    if (fd) return fd.clean_b64;
  }
  return '';
}

// Check correctness across fonts
function correctnessClass(entry) {
  let allOk = 0, total = 0;
  for (const fname of DATA.font_names) {
    const fd = entry.fonts[fname];
    if (!fd) continue;
    total++;
    if (fd.ok1 && fd.ok2) allOk++;
  }
  if (allOk === total) return 'correct';
  if (allOk === 0) return 'wrong';
  return 'partial';
}

// Correctness for a single font entry
function fontCorrectness(fd) {
  if (fd.ok1 && fd.ok2) return 'correct';
  if (fd.ok1 || fd.ok2) return 'partial';
  return 'wrong';
}

function buildGrid() {
  const grid = document.getElementById('grid');
  grid.innerHTML = '';
  let totalBoth = 0, totalSamples = 0;

  DATA.bigrams.forEach((entry, idx) => {
    const cell = document.createElement('div');
    cell.className = 'cell ' + correctnessClass(entry);
    if (idx === selectedIdx) cell.classList.add('selected');

    const canvas = document.createElement('canvas');
    const fixations = aggregateFixations(entry);
    const b64 = representativeImage(entry);
    drawCell(canvas, b64, fixations, W, H);
    cell.appendChild(canvas);

    const label = document.createElement('span');
    label.className = 'cell-label';
    label.textContent = entry.bigram;
    cell.appendChild(label);

    cell.onclick = () => { selectedIdx = idx; buildGrid(); showDetail(entry); };
    grid.appendChild(cell);

    // Stats
    for (const fname of DATA.font_names) {
      const fd = entry.fonts[fname];
      if (fd) { totalSamples++; if (fd.ok1 && fd.ok2) totalBoth++; }
    }
  });

  document.getElementById('stats').textContent =
    `${totalBoth}/${totalSamples} both-correct (${(totalBoth/totalSamples*100).toFixed(1)}%)`;
}

function showDetail(entry) {
  const panel = document.getElementById('detail-panel');
  const title = document.getElementById('detail-title');
  const grid = document.getElementById('detail-grid');

  title.textContent = `"${entry.bigram}" — per-font attention (${DATA.font_names.length} fonts)`;
  grid.innerHTML = '';
  panel.style.display = 'block';

  for (const fname of DATA.font_names) {
    const fd = entry.fonts[fname];
    if (!fd) continue;

    const cell = document.createElement('div');
    cell.className = 'detail-cell ' + fontCorrectness(fd);

    const canvas = document.createElement('canvas');
    drawCell(canvas, fd.clean_b64, fd.fixations, 180, 120);
    cell.appendChild(canvas);

    const fnLabel = document.createElement('div');
    fnLabel.className = 'font-name';
    fnLabel.textContent = fname;
    cell.appendChild(fnLabel);

    const pred = document.createElement('div');
    pred.className = 'pred-info';
    if (fd.ok1 && fd.ok2) {
      pred.textContent = 'OK';
    } else {
      pred.textContent = `pred: ${fd.pred1}${fd.pred2}`;
    }
    cell.appendChild(pred);

    grid.appendChild(cell);
  }

  panel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function setView(mode) {
  viewMode = mode;
  document.getElementById('btn-heatmap').className = mode === 'heatmap' ? 'active' : '';
  document.getElementById('btn-path').className = mode === 'path' ? 'active' : '';
  buildGrid();
  if (selectedIdx >= 0) showDetail(DATA.bigrams[selectedIdx]);
}

function setOpacity(val) {
  opacity = val / 100;
  document.getElementById('opacity-val').textContent = val + '%';
  buildGrid();
  if (selectedIdx >= 0) showDetail(DATA.bigrams[selectedIdx]);
}

// Initial render
buildGrid();
</script>
</body>
</html>'''


def generate_bigram_atlas(model_dir, test_data_dir, output_path='data/bigram_atlas.html',
                          device='auto'):
    """Generate an interactive HTML attention atlas for a trained bigram model.

    Same pattern as generate_atlas() but adapted for BigramVisionModel:
    - Uses BigramDataset for test data
    - Tracks per-position predictions (pred1, pred2, ok1, ok2)
    - Image dimensions are 128x128
    """
    device = _resolve_device(device)
    print(f"Generating bigram attention atlas on: {device}")

    model, n_glimpses, model_type = _load_model(model_dir, device)
    if model_type != 'bigram':
        print(f"Warning: checkpoint model_type is '{model_type}', expected 'bigram'")

    dataset = BigramDataset(test_data_dir)
    font_names = sorted(set(dataset.fonts))

    # Collect per-bigram data keyed by font
    entries = {}  # bigram_str -> {bigram, letter1, letter2, fonts: {font -> {...}}}

    for i in range(len(dataset)):
        img, clean, letter1, letter2, bigram, font = dataset[i]
        img_dev = img.unsqueeze(0).to(device)

        idx1 = ord(letter1) - ord('a')
        idx2 = ord(letter2) - ord('a')

        with torch.no_grad():
            recon, logits_list, locations, _ = model(img_dev)

        # Collect fixation coordinates as [[x, y], ...] in normalized [-1, 1]
        # Drop last location (vestigial GRU prediction, never sampled)
        fixations = []
        for loc in locations[:-1]:
            loc_np = loc[0].cpu().detach().tolist()
            fixations.append([round(loc_np[0], 4), round(loc_np[1], 4)])

        pred1 = logits_list[0].argmax(dim=1).item()
        pred2 = logits_list[1].argmax(dim=1).item()
        ok1 = pred1 == idx1
        ok2 = pred2 == idx2

        pred1_char = chr(pred1 + ord('a'))
        pred2_char = chr(pred2 + ord('a'))

        if bigram not in entries:
            entries[bigram] = {
                'bigram': bigram, 'letter1': letter1, 'letter2': letter2,
                'fonts': {},
            }

        entries[bigram]['fonts'][font] = {
            'fixations': fixations,
            'clean_b64': _tensor_to_base64_png(clean),
            'pred1': pred1_char,
            'pred2': pred2_char,
            'ok1': ok1,
            'ok2': ok2,
        }

        mark = 'OK' if (ok1 and ok2) else f'{pred1_char}{pred2_char}'
        print(f"  {bigram} [{font}]: {mark}")

    # Build ordered list (alphabetical by bigram)
    bigrams_list = [entries[k] for k in sorted(entries.keys())]

    # Get image dimensions from first sample
    sample_img = dataset[0][0]  # (1, H, W)
    img_height, img_width = sample_img.shape[1], sample_img.shape[2]

    atlas_data = {
        'image_width': img_width,
        'image_height': img_height,
        'n_fixations': n_glimpses + 1,
        'font_names': font_names,
        'bigrams': bigrams_list,
    }

    # Inject JSON into HTML template
    atlas_json = json.dumps(atlas_data)
    html = _bigram_atlas_html_template().replace('ATLAS_JSON_PLACEHOLDER', atlas_json)

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(html)

    total = sum(len(e['fonts']) for e in bigrams_list)
    both_ok = sum(1 for e in bigrams_list for fd in e['fonts'].values()
                  if fd['ok1'] and fd['ok2'])
    size_kb = os.path.getsize(output_path) / 1024
    print(f"\nAtlas: {len(bigrams_list)} bigrams x {len(font_names)} fonts = {total} samples")
    print(f"Both-correct: {both_ok}/{total} ({both_ok/total*100:.1f}%)")
    print(f"Written to {output_path} ({size_kb:.0f} KB)")


# --- Word evaluation (imported from separate module to keep file manageable) ---
from fbrl._word_eval import test_word_model, generate_word_atlas, test_word_isolation, generate_isolation_atlas  # noqa: E402, F401
