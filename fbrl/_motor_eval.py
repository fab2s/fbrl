"""Motor evaluation functions -- imported by evaluate.py."""
import torch
import torch.nn.functional as F
import numpy as np
import os
import json
import base64
import io
from PIL import Image

from fbrl import _resolve_device
from fbrl.data import LetterDataset
from fbrl.motor import load_trajectory_data, batch_gt_trajectories, soft_render
from fbrl.losses import fixation_hit_rate
from fbrl.evaluate import _load_model, visualize_attention, _tensor_to_base64_png


def test_motor_model(model_dir, test_data_dir, output_dir='motor_results',
                     trajectory_data_dir='data/trajectories', device='auto'):
    """Test a trained MotorVisionModel on letter test data.

    Reports standard letter/case accuracy plus motor-specific metrics:
    - Re-read accuracy (classify from rendered trajectory)
    - Trajectory MSE + pen-state F1
    - Reconstruction MSE
    - Per-font breakdown
    """
    device = _resolve_device(device)
    print(f"Motor testing on: {device}")

    os.makedirs(output_dir, exist_ok=True)

    model, _, model_type = _load_model(model_dir, device)
    if model_type != 'letter_motor':
        print(f"Warning: checkpoint model_type is '{model_type}', expected 'letter_motor'")

    dataset = LetterDataset(test_data_dir)

    # Load trajectory GT
    traj_data = load_trajectory_data(trajectory_data_dir)

    letter_correct = 0
    case_correct = 0
    rr_letter_correct = 0
    rr_case_correct = 0
    total = 0
    mse_scores = []
    traj_mse_scores = []
    pen_tp, pen_fp, pen_fn = 0, 0, 0  # for pen F1

    font_stats = {}
    errors = []
    correct_list = []
    rr_errors = []

    for i in range(len(dataset)):
        img, clean, letter, case, font, partner_clean = dataset[i]
        img = img.unsqueeze(0).to(device)

        letter_idx = ord(letter) - ord('A')
        case_idx = 0 if case == 'upper' else 1
        case_float = torch.tensor([[float(case_idx)]], device=device)

        with torch.no_grad():
            recon, letter_logits, case_logits, locations, latent, _ = model(img, case_float)

            # Motor path
            trajectory, rendered = model.motor_forward(latent)

            # Re-read
            rr_enc = model._encode(rendered)
            rr_letter_logits = model.letter_classifier(rr_enc.latent)
            rr_case_logits = model.case_classifier(rr_enc.latent)

        # Standard accuracy
        letter_pred = letter_logits.argmax(dim=1).item()
        letter_ok = letter_pred == letter_idx
        letter_correct += int(letter_ok)

        case_pred = case_logits.argmax(dim=1).item()
        case_ok = case_pred == case_idx
        case_correct += int(case_ok)

        # Re-read accuracy
        rr_letter_pred = rr_letter_logits.argmax(dim=1).item()
        rr_letter_ok = rr_letter_pred == letter_idx
        rr_letter_correct += int(rr_letter_ok)

        rr_case_pred = rr_case_logits.argmax(dim=1).item()
        rr_case_ok = rr_case_pred == case_idx
        rr_case_correct += int(rr_case_ok)

        total += 1

        # MSE
        mse = F.mse_loss(recon, img).item()
        mse_scores.append(mse)

        # Trajectory metrics
        gt_traj = batch_gt_trajectories([letter], [case], traj_data, device)
        t_mse = F.mse_loss(trajectory[:, :, :2], gt_traj[:, :, :2]).item()
        traj_mse_scores.append(t_mse)

        # Pen state F1
        pred_pen = (torch.sigmoid(trajectory[:, :, 2]) > 0.5).float()
        gt_pen = gt_traj[:, :, 2]
        pen_tp += ((pred_pen == 1) & (gt_pen == 1)).sum().item()
        pen_fp += ((pred_pen == 1) & (gt_pen == 0)).sum().item()
        pen_fn += ((pred_pen == 0) & (gt_pen == 1)).sum().item()

        # Per-font stats
        if font not in font_stats:
            font_stats[font] = {'letter_ok': 0, 'case_ok': 0,
                                'rr_letter_ok': 0, 'rr_case_ok': 0, 'total': 0}
        font_stats[font]['letter_ok'] += int(letter_ok)
        font_stats[font]['case_ok'] += int(case_ok)
        font_stats[font]['rr_letter_ok'] += int(rr_letter_ok)
        font_stats[font]['rr_case_ok'] += int(rr_case_ok)
        font_stats[font]['total'] += 1

        original_char = letter.lower() if case == 'lower' else letter
        pred_char = chr(letter_pred + ord('A'))
        rr_pred_char = chr(rr_letter_pred + ord('A'))
        letter_mark = 'OK' if letter_ok else f'WRONG({pred_char})'
        rr_mark = 'OK' if rr_letter_ok else f'WRONG({rr_pred_char})'
        font_tag = f'  [{font}]' if len(font_stats) > 1 or font != 'default' else ''
        print(f"  {original_char}{font_tag}: Ltr={letter_mark}  RR={rr_mark}  "
              f"TrajMSE={t_mse:.4f}  MSE={mse:.4f}")

        if not (letter_ok and case_ok):
            errors.append((original_char, font, pred_char, letter_ok, case_ok))
        else:
            correct_list.append((original_char, font))

        if not (rr_letter_ok and rr_case_ok):
            rr_errors.append((original_char, font, rr_pred_char, rr_letter_ok, rr_case_ok))

        # Save attention overlay
        suffix = f'_{font}' if len(set(dataset.fonts)) > 1 else ''
        visualize_attention(
            img.squeeze(0), locations,
            os.path.join(output_dir, f'attention_{original_char}{suffix}.png'),
        )

        # Save reconstruction image
        recon_np = recon.squeeze().cpu().clamp(0, 1).numpy()
        Image.fromarray((recon_np * 255).astype(np.uint8)).save(
            os.path.join(output_dir, f'recon_{original_char}{suffix}.png'),
        )

        # Save recode (opposite case) image
        with torch.no_grad():
            recode_case = torch.tensor([[1.0 - case_float.item()]], device=device)
            recode_img, _ = model.recode(img, recode_case)
        recode_np = recode_img.squeeze().cpu().clamp(0, 1).numpy()
        Image.fromarray((recode_np * 255).astype(np.uint8)).save(
            os.path.join(output_dir, f'recode_{original_char}{suffix}.png'),
        )

        # Save rendered trajectory image
        rendered_np = rendered.squeeze().cpu().clamp(0, 1).numpy()
        Image.fromarray((rendered_np * 255).astype(np.uint8)).save(
            os.path.join(output_dir, f'rendered_{original_char}{suffix}.png'),
        )

    # Summary
    letter_acc = letter_correct / total if total > 0 else 0
    case_acc = case_correct / total if total > 0 else 0
    rr_letter_acc = rr_letter_correct / total if total > 0 else 0
    rr_case_acc = rr_case_correct / total if total > 0 else 0
    avg_mse = np.mean(mse_scores) if mse_scores else 0
    avg_traj_mse = np.mean(traj_mse_scores) if traj_mse_scores else 0
    pen_precision = pen_tp / max(pen_tp + pen_fp, 1)
    pen_recall = pen_tp / max(pen_tp + pen_fn, 1)
    pen_f1 = 2 * pen_precision * pen_recall / max(pen_precision + pen_recall, 1e-8)

    print(f"\nLetter accuracy:    {letter_correct}/{total} ({letter_acc:.1%})")
    print(f"Case accuracy:      {case_correct}/{total} ({case_acc:.1%})")
    print(f"RR Letter accuracy: {rr_letter_correct}/{total} ({rr_letter_acc:.1%})")
    print(f"RR Case accuracy:   {rr_case_correct}/{total} ({rr_case_acc:.1%})")
    print(f"Avg recon MSE:      {avg_mse:.4f}")
    print(f"Avg traj MSE:       {avg_traj_mse:.4f}")
    print(f"Pen F1:             {pen_f1:.3f} (P={pen_precision:.3f} R={pen_recall:.3f})")

    if len(font_stats) > 1:
        print(f"\nPer-font breakdown:")
        for fname in sorted(font_stats.keys()):
            s = font_stats[fname]
            lt = s['letter_ok'] / s['total'] * 100
            rr = s['rr_letter_ok'] / s['total'] * 100
            print(f"  {fname:<24s}: Ltr {lt:5.1f}%  RR {rr:5.1f}%  ({s['total']})")

    # Write summary.txt — vision metrics only, matching letter model format
    summary_path = os.path.join(output_dir, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"Letter: {letter_correct}/{total} ({letter_acc:.1%})\n")
        f.write(f"Case:   {case_correct}/{total} ({case_acc:.1%})\n")
        f.write(f"Avg MSE:      {avg_mse:.4f}\n")

        if errors:
            f.write(f"\nErrors ({len(errors)}):\n")
            for char, font, pred, l_ok, c_ok in sorted(errors):
                parts = []
                if not l_ok:
                    parts.append(f"ltr: {char}\u2192{pred}")
                if not c_ok:
                    parts.append("case wrong")
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

    # Write summary_motor.txt — motor-specific metrics
    motor_summary_path = os.path.join(output_dir, 'summary_motor.txt')
    with open(motor_summary_path, 'w') as f:
        f.write(f"Re-read Letter: {rr_letter_correct}/{total} ({rr_letter_acc:.1%})\n")
        f.write(f"Re-read Case:   {rr_case_correct}/{total} ({rr_case_acc:.1%})\n")
        f.write(f"Avg Traj MSE:   {avg_traj_mse:.4f}\n")
        f.write(f"Pen F1:         {pen_f1:.3f} (P={pen_precision:.3f} R={pen_recall:.3f})\n")

        if len(font_stats) > 1:
            f.write(f"\nPer-font re-read:\n")
            for fname in sorted(font_stats.keys()):
                s = font_stats[fname]
                rr_lt = s['rr_letter_ok'] / s['total'] * 100
                rr_cs = s['rr_case_ok'] / s['total'] * 100
                f.write(f"  {fname:<24s}: Letter {rr_lt:5.1f}%  Case {rr_cs:5.1f}%  "
                        f"({s['total']})\n")

        if rr_errors:
            f.write(f"\nRe-read errors ({len(rr_errors)}):\n")
            for char, font, pred, l_ok, c_ok in sorted(rr_errors):
                parts = []
                if not l_ok:
                    parts.append(f"ltr: {char}\u2192{pred}")
                if not c_ok:
                    parts.append("case wrong")
                font_tag = f"  [{font}]" if font != 'default' else ''
                f.write(f"  {char}{font_tag}  ({', '.join(parts)})\n")

    print(f"Summary written to {summary_path}")
    print(f"Motor summary written to {motor_summary_path}")
    print(f"Results saved in {output_dir}")


def _motor_atlas_html_template():
    """Return self-contained HTML/CSS/JS template for the motor attention atlas.

    Shows: letter image with fixations, predicted trajectory, rendered image,
    and classification results for both read and re-read.
    """
    return r'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Motor Attention Atlas</title>
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
#grid {
  display: grid; grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
  gap: 4px; padding: 12px; max-width: 1800px; margin: 0 auto;
}
.cell {
  position: relative; cursor: pointer; border: 2px solid transparent;
  border-radius: 4px; overflow: hidden; transition: transform 0.15s;
  aspect-ratio: 1;
}
.cell:hover { transform: scale(1.3); z-index: 5; }
.cell.correct { border-color: #2ecc71; }
.cell.wrong { border-color: #e74c3c; }
.cell.partial { border-color: #f39c12; }
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
.detail-cell .detail-row { display: flex; gap: 4px; justify-content: center; }
.detail-cell canvas { width: 128px; height: 128px; display: block; }
.detail-cell .img-label { font-size: 9px; color: #606080; text-align: center; }
.detail-cell .font-name { font-size: 11px; color: #a0a0c0; margin-top: 2px; }
.detail-cell .pred-info { font-size: 10px; color: #888; }
</style>
</head>
<body>
<div id="controls">
  <span style="font-weight:bold;color:#e94560;">Motor Attention Atlas</span>
  <div>
    <label>View:</label>
    <button id="btn-input" class="active" onclick="setView('input')">Input</button>
    <button id="btn-recon" onclick="setView('recon')">Recon</button>
    <button id="btn-recode" onclick="setView('recode')">Recode</button>
    <button id="btn-rendered" onclick="setView('rendered')">Rendered</button>
    <button id="btn-traj" onclick="setView('traj')">Trajectory</button>
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
const SZ = 128;
let viewMode = 'input';
let selectedIdx = -1;

function drawImg(canvas, b64, w, h) {
  canvas.width = w; canvas.height = h;
  const ctx = canvas.getContext('2d');
  const img = new window.Image();
  img.onload = function() { ctx.drawImage(img, 0, 0, w, h); };
  img.src = 'data:image/png;base64,' + b64;
}

function drawTrajectory(canvas, traj, w, h) {
  canvas.width = w; canvas.height = h;
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, w, h);
  if (!traj || traj.length === 0) return;
  for (let i = 1; i < traj.length; i++) {
    const [x0, y0, p0] = traj[i-1];
    const [x1, y1, p1] = traj[i];
    const cx0 = (x0 + 1) / 2 * w;
    const cy0 = (y0 + 1) / 2 * h;
    const cx1 = (x1 + 1) / 2 * w;
    const cy1 = (y1 + 1) / 2 * h;
    ctx.strokeStyle = p1 > 0.5 ? '#4af' : 'rgba(255,50,50,0.3)';
    ctx.lineWidth = p1 > 0.5 ? 1.5 : 0.5;
    ctx.beginPath(); ctx.moveTo(cx0, cy0); ctx.lineTo(cx1, cy1); ctx.stroke();
  }
}

function representativeData(entry) {
  for (const fname of DATA.font_names) {
    const fd = entry.fonts[fname];
    if (fd) return fd;
  }
  return null;
}

function correctnessClass(entry) {
  let allOk = 0, total = 0;
  for (const fname of DATA.font_names) {
    const fd = entry.fonts[fname];
    if (!fd) continue;
    total++;
    if (fd.letter_ok && fd.rr_letter_ok) allOk++;
  }
  if (allOk === total) return 'correct';
  if (allOk === 0) return 'wrong';
  return 'partial';
}

function buildGrid() {
  const grid = document.getElementById('grid');
  grid.innerHTML = '';
  let totalOk = 0, totalSamples = 0;

  DATA.letters.forEach((entry, idx) => {
    const cell = document.createElement('div');
    cell.className = 'cell ' + correctnessClass(entry);
    if (idx === selectedIdx) cell.classList.add('selected');

    const canvas = document.createElement('canvas');
    const fd = representativeData(entry);
    if (fd) {
      if (viewMode === 'rendered') {
        drawImg(canvas, fd.rendered_b64, SZ, SZ);
      } else if (viewMode === 'recon') {
        drawImg(canvas, fd.recon_b64, SZ, SZ);
      } else if (viewMode === 'recode') {
        drawImg(canvas, fd.recode_b64, SZ, SZ);
      } else if (viewMode === 'traj') {
        drawTrajectory(canvas, fd.trajectory, SZ, SZ);
      } else {
        drawImg(canvas, fd.clean_b64, SZ, SZ);
      }
    }
    cell.appendChild(canvas);

    const label = document.createElement('span');
    label.className = 'cell-label';
    label.textContent = entry.char;
    cell.appendChild(label);

    cell.onclick = () => { selectedIdx = idx; buildGrid(); showDetail(entry); };
    grid.appendChild(cell);

    for (const fname of DATA.font_names) {
      const fd2 = entry.fonts[fname];
      if (fd2) {
        totalSamples++;
        if (fd2.letter_ok) totalOk++;
      }
    }
  });

  document.getElementById('stats').textContent =
    `${totalOk}/${totalSamples} letter-correct (${(totalOk/totalSamples*100).toFixed(1)}%)`;
}

function showDetail(entry) {
  const panel = document.getElementById('detail-panel');
  const title = document.getElementById('detail-title');
  const grid = document.getElementById('detail-grid');

  title.textContent = `"${entry.char}" -- per-font detail`;
  grid.innerHTML = '';
  panel.style.display = 'block';

  for (const fname of DATA.font_names) {
    const fd = entry.fonts[fname];
    if (!fd) continue;

    const cell = document.createElement('div');
    cell.className = 'detail-cell ' + (fd.letter_ok ? 'correct' : 'wrong');

    const row = document.createElement('div');
    row.className = 'detail-row';

    function addPanel(label, drawFn) {
      const wrap = document.createElement('div');
      const cv = document.createElement('canvas');
      drawFn(cv);
      wrap.appendChild(cv);
      const lbl = document.createElement('div');
      lbl.className = 'img-label';
      lbl.textContent = label;
      wrap.appendChild(lbl);
      row.appendChild(wrap);
    }

    addPanel('input', c => drawImg(c, fd.clean_b64, 128, 128));
    addPanel('recon', c => drawImg(c, fd.recon_b64, 128, 128));
    addPanel('recode', c => drawImg(c, fd.recode_b64, 128, 128));
    addPanel('trajectory', c => drawTrajectory(c, fd.trajectory, 128, 128));
    addPanel('rendered', c => drawImg(c, fd.rendered_b64, 128, 128));

    cell.appendChild(row);

    const fnLabel = document.createElement('div');
    fnLabel.className = 'font-name';
    fnLabel.textContent = fname;
    cell.appendChild(fnLabel);

    const pred = document.createElement('div');
    pred.className = 'pred-info';
    const ltr = fd.letter_ok ? 'OK' : `pred:${fd.letter_pred}`;
    const rr = fd.rr_letter_ok ? 'OK' : `rr:${fd.rr_letter_pred}`;
    pred.textContent = `Read: ${ltr}  Re-read: ${rr}`;
    cell.appendChild(pred);

    grid.appendChild(cell);
  }

  panel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function setView(mode) {
  viewMode = mode;
  document.querySelectorAll('#controls button').forEach(b => b.className = '');
  document.getElementById('btn-' + mode).className = 'active';
  buildGrid();
  if (selectedIdx >= 0) showDetail(DATA.letters[selectedIdx]);
}

buildGrid();
</script>
</body>
</html>'''


def generate_motor_atlas(model_dir, test_data_dir, output_path='data/motor_atlas.html',
                          trajectory_data_dir='data/trajectories', device='auto'):
    """Generate an interactive HTML motor attention atlas.

    Each cell shows the letter image, predicted trajectory, and soft-rendered output.
    Detail view shows all three side by side for each font.
    """
    device = _resolve_device(device)
    print(f"Generating motor attention atlas on: {device}")

    model, _, model_type = _load_model(model_dir, device)
    if model_type != 'letter_motor':
        print(f"Warning: checkpoint model_type is '{model_type}', expected 'letter_motor'")

    dataset = LetterDataset(test_data_dir)
    font_names = sorted(set(dataset.fonts))

    entries = {}  # char -> {char, fonts: {font -> {...}}}

    for i in range(len(dataset)):
        img, clean, letter, case, font, _partner = dataset[i]
        img_dev = img.unsqueeze(0).to(device)

        letter_idx = ord(letter) - ord('A')
        case_idx = 0 if case == 'upper' else 1
        case_float = torch.tensor([[float(case_idx)]], device=device)

        with torch.no_grad():
            recon, letter_logits, case_logits, locations, latent, _ = model(img_dev, case_float)
            trajectory, rendered = model.motor_forward(latent)

            rr_enc = model._encode(rendered)
            rr_letter_logits = model.letter_classifier(rr_enc.latent)

            recode_case = torch.tensor([[1.0 - case_float.item()]], device=device)
            recode_img, _ = model.recode(img_dev, recode_case)

        letter_pred = letter_logits.argmax(dim=1).item()
        letter_ok = letter_pred == letter_idx
        rr_letter_pred = rr_letter_logits.argmax(dim=1).item()
        rr_letter_ok = rr_letter_pred == letter_idx

        original_char = letter.lower() if case == 'lower' else letter

        # Trajectory as list of [x, y, pen_prob]
        traj_np = trajectory[0].cpu().numpy()
        traj_list = []
        for t in range(traj_np.shape[0]):
            x, y = float(traj_np[t, 0]), float(traj_np[t, 1])
            pen_prob = float(1.0 / (1.0 + np.exp(-traj_np[t, 2])))  # sigmoid
            traj_list.append([round(x, 4), round(y, 4), round(pen_prob, 3)])

        if original_char not in entries:
            entries[original_char] = {'char': original_char, 'fonts': {}}

        entries[original_char]['fonts'][font] = {
            'clean_b64': _tensor_to_base64_png(clean),
            'recon_b64': _tensor_to_base64_png(recon.squeeze(0).cpu()),
            'recode_b64': _tensor_to_base64_png(recode_img.squeeze(0).cpu()),
            'rendered_b64': _tensor_to_base64_png(rendered.squeeze(0).cpu()),
            'trajectory': traj_list,
            'letter_ok': letter_ok,
            'letter_pred': chr(letter_pred + ord('A')),
            'rr_letter_ok': rr_letter_ok,
            'rr_letter_pred': chr(rr_letter_pred + ord('A')),
        }

        mark = 'OK' if letter_ok else chr(letter_pred + ord('A'))
        rr_mark = 'OK' if rr_letter_ok else chr(rr_letter_pred + ord('A'))
        print(f"  {original_char} [{font}]: Read={mark}  RR={rr_mark}")

    letters_list = [entries[k] for k in sorted(entries.keys())]

    atlas_data = {
        'image_size': 128,
        'font_names': font_names,
        'letters': letters_list,
    }

    atlas_json = json.dumps(atlas_data)
    html = _motor_atlas_html_template().replace('ATLAS_JSON_PLACEHOLDER', atlas_json)

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(html)

    total = sum(len(e['fonts']) for e in letters_list)
    all_ok = sum(1 for e in letters_list for fd in e['fonts'].values() if fd['letter_ok'])
    rr_ok = sum(1 for e in letters_list for fd in e['fonts'].values() if fd['rr_letter_ok'])
    size_kb = os.path.getsize(output_path) / 1024
    print(f"\nAtlas: {len(letters_list)} letters x {len(font_names)} fonts = {total} samples")
    print(f"Read accuracy:    {all_ok}/{total} ({all_ok/total*100:.1f}%)")
    print(f"Re-read accuracy: {rr_ok}/{total} ({rr_ok/total*100:.1f}%)")
    print(f"Written to {output_path} ({size_kb:.0f} KB)")
