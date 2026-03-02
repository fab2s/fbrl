"""Word evaluation functions — imported by evaluate.py."""
import torch
import torch.nn.functional as F
import numpy as np
import os
import json
from PIL import Image

from fbrl import _resolve_device
from fbrl.data import WordDataset
from fbrl.losses import fixation_hit_rate
from fbrl.evaluate import _load_model, visualize_attention, _tensor_to_base64_png


def test_word_model(model_dir, test_data_dir, output_dir='word_results', device='auto'):
    """Test a trained WordVisionModel on word test data.

    Reports per-position accuracy, all-correct accuracy, and reconstruction MSE.
    """
    device = _resolve_device(device)
    print(f"Word testing on: {device}")

    os.makedirs(output_dir, exist_ok=True)

    model, _, model_type = _load_model(model_dir, device)
    if model_type != 'word':
        print(f"Warning: checkpoint model_type is '{model_type}', expected 'word'")

    n_positions = model.n_positions
    dataset = WordDataset(test_data_dir)

    pos_correct = [0] * n_positions
    all_correct = 0
    total = 0
    mse_scores = []

    # Per-font tracking
    font_stats = {}
    errors = []
    correct_list = []
    all_results = []  # (word, font, pred_chars, oks, mse) for detailed log

    for i in range(len(dataset)):
        img, clean, l1, l2, l3, l4, word, font = dataset[i]
        img = img.unsqueeze(0).to(device)

        letters = [l1, l2, l3, l4]
        idx = [ord(l) - ord('a') for l in letters[:n_positions]]

        with torch.no_grad():
            recon, logits_list, locations, _, _, _, _ = model(img)

        preds = [logits_list[p].argmax(dim=1).item() for p in range(n_positions)]
        oks = [preds[p] == idx[p] for p in range(n_positions)]
        ok_all = all(oks)

        for p in range(n_positions):
            pos_correct[p] += int(oks[p])
        all_correct += int(ok_all)
        total += 1

        # Per-font stats
        if font not in font_stats:
            font_stats[font] = {f'pos{p+1}_ok': 0 for p in range(n_positions)}
            font_stats[font].update({'all_ok': 0, 'total': 0})
        for p in range(n_positions):
            font_stats[font][f'pos{p+1}_ok'] += int(oks[p])
        font_stats[font]['all_ok'] += int(ok_all)
        font_stats[font]['total'] += 1

        mse = F.mse_loss(recon, img).item()
        mse_scores.append(mse)

        # Display
        pred_chars = [chr(preds[p] + ord('a')) for p in range(n_positions)]
        marks = ['OK' if oks[p] else f'WRONG({pred_chars[p]})' for p in range(n_positions)]
        font_tag = f'  [{font}]' if len(font_stats) > 1 or font != 'default' else ''
        marks_str = '  '.join(f'P{p+1}={marks[p]}' for p in range(n_positions))
        print(f"  {word}{font_tag}: {marks_str}  MSE={mse:.4f}")

        all_results.append((word, font, pred_chars, oks, mse))

        if ok_all:
            correct_list.append((word, font))
        else:
            errors.append((word, font, pred_chars, oks))

        # Save attention overlay
        visualize_attention(
            img.squeeze(0), locations,
            os.path.join(output_dir, f'attention_{word}_{font}.png'),
        )

        # Save reconstruction
        recon_img = recon.squeeze().cpu().clamp(0, 1).detach().numpy()
        Image.fromarray((recon_img * 255).astype(np.uint8)).save(
            os.path.join(output_dir, f'recon_{word}_{font}.png'),
        )

    accs = [pos_correct[p] / total if total > 0 else 0 for p in range(n_positions)]
    acc_all = all_correct / total if total > 0 else 0
    avg_mse = np.mean(mse_scores) if mse_scores else 0

    print(f"\nAll-correct:     {all_correct}/{total} ({acc_all:.1%})")
    for p in range(n_positions):
        print(f"Pos {p+1} accuracy:  {pos_correct[p]}/{total} ({accs[p]:.1%})")
    print(f"Avg reconstruction MSE: {avg_mse:.4f}")

    # Per-font breakdown
    if len(font_stats) > 1:
        print(f"\nPer-font breakdown:")
        for fname in sorted(font_stats.keys()):
            s = font_stats[fname]
            a_all = s['all_ok'] / s['total'] * 100
            pos_accs = [s[f'pos{p+1}_ok'] / s['total'] * 100 for p in range(n_positions)]
            pos_str = '  '.join(f'P{p+1} {pos_accs[p]:5.1f}%' for p in range(n_positions))
            print(f"  {fname:<24s}: {pos_str}  All {a_all:5.1f}%  ({s['total']} samples)")

    # Write summary file
    summary_path = os.path.join(output_dir, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"All-correct: {all_correct}/{total} ({acc_all:.1%})\n")
        for p in range(n_positions):
            f.write(f"Pos {p+1}:       {pos_correct[p]}/{total} ({accs[p]:.1%})\n")
        f.write(f"Avg MSE:     {avg_mse:.4f}\n")

        if errors:
            f.write(f"\nErrors ({len(errors)}):\n")
            for word, font, pchars, oks in sorted(errors):
                parts = []
                for p in range(n_positions):
                    if not oks[p]:
                        parts.append(f"pos{p+1}: {word[p]}->{pchars[p]}")
                pred_word = ''.join(pchars)
                font_tag = f"  [{font}]" if font != 'default' else ''
                f.write(f"  {word} -> {pred_word}{font_tag}  ({', '.join(parts)})\n")

        # Detailed per-word log
        f.write(f"\nPer-word detail ({total} samples):\n")
        for word, font, pchars, oks, mse_val in sorted(all_results):
            font_tag = f"  [{font}]" if font != 'default' else ''
            marks = '  '.join(
                f'P{p+1}=OK' if oks[p] else f'P{p+1}=WRONG({pchars[p]})'
                for p in range(n_positions)
            )
            f.write(f"  {word}{font_tag}: {marks}  MSE={mse_val:.4f}\n")

        if len(font_stats) > 1:
            f.write(f"\nPer-font:\n")
            for fname in sorted(font_stats.keys()):
                s = font_stats[fname]
                a_all = s['all_ok'] / s['total'] * 100
                pos_accs = [s[f'pos{p+1}_ok'] / s['total'] * 100 for p in range(n_positions)]
                pos_str = '  '.join(f'P{p+1} {pos_accs[p]:5.1f}%' for p in range(n_positions))
                f.write(f"  {fname:<24s}: {pos_str}  All {a_all:5.1f}%  ({s['total']})\n")

        f.write(f"\nCorrect ({len(correct_list)}):\n")
        line = '  '
        for i, (word, _font) in enumerate(sorted(correct_list)):
            line += word
            if i < len(correct_list) - 1:
                line += ', '
            if len(line) > 78:
                f.write(line + '\n')
                line = '  '
        if line.strip():
            f.write(line + '\n')

    print(f"Summary written to {summary_path}")
    print(f"Results saved in {output_dir}")


def _word_atlas_html_template():
    """Return self-contained HTML/CSS/JS template for the word attention atlas.

    Adapted from bigram atlas with:
    - 2:1 aspect ratio cells (256x128 images)
    - Detail cells: 360x180
    - Correctness: green=all 4, yellow=some, red=none
    """
    return r'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Word Attention Atlas</title>
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
  display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
  gap: 4px; padding: 12px; max-width: 1800px; margin: 0 auto;
}
.cell {
  position: relative; cursor: pointer; border: 2px solid transparent;
  border-radius: 4px; overflow: hidden; transition: transform 0.15s;
  aspect-ratio: 2;
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
.detail-cell canvas { width: 360px; height: 180px; display: block; }
.detail-cell .font-name { font-size: 11px; color: #a0a0c0; margin-top: 2px; }
.detail-cell .pred-info { font-size: 10px; color: #888; }
</style>
</head>
<body>
<div id="controls">
  <span style="font-weight:bold;color:#e94560;">Word Attention Atlas</span>
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
const NP = DATA.n_positions;
let viewMode = 'heatmap';
let opacity = 0.6;
let selectedIdx = -1;

function hotColor(t) {
  t = Math.max(0, Math.min(1, t));
  let r, g, b;
  if (t < 0.33) { r = t / 0.33; g = 0; b = 0; }
  else if (t < 0.66) { r = 1; g = (t - 0.33) / 0.33; b = 0; }
  else { r = 1; g = 1; b = (t - 0.66) / 0.34; }
  return [r * 255 | 0, g * 255 | 0, b * 255 | 0];
}

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

function aggregateFixations(entry) {
  const all = [];
  for (const fname of DATA.font_names) {
    const fd = entry.fonts[fname];
    if (fd) all.push(...fd.fixations);
  }
  return all;
}

function representativeImage(entry) {
  for (const fname of DATA.font_names) {
    const fd = entry.fonts[fname];
    if (fd) return fd.clean_b64;
  }
  return '';
}

function correctnessClass(entry) {
  let allOk = 0, total = 0;
  for (const fname of DATA.font_names) {
    const fd = entry.fonts[fname];
    if (!fd) continue;
    total++;
    let ok = true;
    for (let p = 0; p < NP; p++) { if (!fd['ok' + (p+1)]) ok = false; }
    if (ok) allOk++;
  }
  if (allOk === total) return 'correct';
  if (allOk === 0) return 'wrong';
  return 'partial';
}

function fontCorrectness(fd) {
  let nOk = 0;
  for (let p = 0; p < NP; p++) { if (fd['ok' + (p+1)]) nOk++; }
  if (nOk === NP) return 'correct';
  if (nOk === 0) return 'wrong';
  return 'partial';
}

function buildGrid() {
  const grid = document.getElementById('grid');
  grid.innerHTML = '';
  let totalAll = 0, totalSamples = 0;

  DATA.words.forEach((entry, idx) => {
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
    label.textContent = entry.word;
    cell.appendChild(label);

    cell.onclick = () => { selectedIdx = idx; buildGrid(); showDetail(entry); };
    grid.appendChild(cell);

    for (const fname of DATA.font_names) {
      const fd = entry.fonts[fname];
      if (fd) {
        totalSamples++;
        let ok = true;
        for (let p = 0; p < NP; p++) { if (!fd['ok' + (p+1)]) ok = false; }
        if (ok) totalAll++;
      }
    }
  });

  document.getElementById('stats').textContent =
    `${totalAll}/${totalSamples} all-correct (${(totalAll/totalSamples*100).toFixed(1)}%)`;
}

function showDetail(entry) {
  const panel = document.getElementById('detail-panel');
  const title = document.getElementById('detail-title');
  const grid = document.getElementById('detail-grid');

  title.textContent = `"${entry.word}" \u2014 per-font attention (${DATA.font_names.length} fonts)`;
  grid.innerHTML = '';
  panel.style.display = 'block';

  for (const fname of DATA.font_names) {
    const fd = entry.fonts[fname];
    if (!fd) continue;

    const cell = document.createElement('div');
    cell.className = 'detail-cell ' + fontCorrectness(fd);

    const canvas = document.createElement('canvas');
    drawCell(canvas, fd.clean_b64, fd.fixations, 360, 180);
    cell.appendChild(canvas);

    const fnLabel = document.createElement('div');
    fnLabel.className = 'font-name';
    fnLabel.textContent = fname;
    cell.appendChild(fnLabel);

    const pred = document.createElement('div');
    pred.className = 'pred-info';
    let allOk = true;
    for (let p = 0; p < NP; p++) { if (!fd['ok' + (p+1)]) allOk = false; }
    if (allOk) {
      pred.textContent = 'OK';
    } else {
      let predWord = '';
      for (let p = 0; p < NP; p++) predWord += fd['pred' + (p+1)];
      pred.textContent = `pred: ${predWord}`;
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
  if (selectedIdx >= 0) showDetail(DATA.words[selectedIdx]);
}

function setOpacity(val) {
  opacity = val / 100;
  document.getElementById('opacity-val').textContent = val + '%';
  buildGrid();
  if (selectedIdx >= 0) showDetail(DATA.words[selectedIdx]);
}

buildGrid();
</script>
</body>
</html>'''


def generate_word_atlas(model_dir, test_data_dir, output_path='data/word_atlas.html',
                         device='auto'):
    """Generate an interactive HTML attention atlas for a trained word model.

    Same pattern as generate_bigram_atlas() but adapted for WordVisionModel:
    - Uses WordDataset for test data
    - Tracks per-position predictions (pred1..pred4, ok1..ok4)
    - Image dimensions are 128x256 (2:1 aspect ratio)
    """
    device = _resolve_device(device)
    print(f"Generating word attention atlas on: {device}")

    model, n_glimpses, model_type = _load_model(model_dir, device)
    if model_type != 'word':
        print(f"Warning: checkpoint model_type is '{model_type}', expected 'word'")

    n_positions = model.n_positions
    dataset = WordDataset(test_data_dir)
    font_names = sorted(set(dataset.fonts))

    entries = {}  # word_str -> {word, fonts: {font -> {...}}}

    for i in range(len(dataset)):
        img, clean, l1, l2, l3, l4, word, font = dataset[i]
        img_dev = img.unsqueeze(0).to(device)

        letters = [l1, l2, l3, l4]
        idx = [ord(l) - ord('a') for l in letters[:n_positions]]

        with torch.no_grad():
            recon, logits_list, locations, _, _, _, _ = model(img_dev)

        # Collect fixation coordinates
        fixations = []
        for loc in locations:
            loc_np = loc[0].cpu().detach().tolist()
            fixations.append([round(loc_np[0], 4), round(loc_np[1], 4)])

        preds = [logits_list[p].argmax(dim=1).item() for p in range(n_positions)]
        oks = [preds[p] == idx[p] for p in range(n_positions)]
        pred_chars = [chr(preds[p] + ord('a')) for p in range(n_positions)]

        if word not in entries:
            entries[word] = {'word': word, 'fonts': {}}

        font_data = {
            'fixations': fixations,
            'clean_b64': _tensor_to_base64_png(clean),
        }
        for p in range(n_positions):
            font_data[f'pred{p+1}'] = pred_chars[p]
            font_data[f'ok{p+1}'] = oks[p]

        entries[word]['fonts'][font] = font_data

        ok_all = all(oks)
        mark = 'OK' if ok_all else ''.join(pred_chars)
        print(f"  {word} [{font}]: {mark}")

    # Build ordered list (alphabetical by word)
    words_list = [entries[k] for k in sorted(entries.keys())]

    # Get image dimensions from first sample
    sample_img = dataset[0][0]  # (1, H, W)
    img_height, img_width = sample_img.shape[1], sample_img.shape[2]

    atlas_data = {
        'image_width': img_width,
        'image_height': img_height,
        'n_fixations': n_glimpses + 1,
        'n_positions': n_positions,
        'font_names': font_names,
        'words': words_list,
    }

    atlas_json = json.dumps(atlas_data)
    html = _word_atlas_html_template().replace('ATLAS_JSON_PLACEHOLDER', atlas_json)

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(html)

    total = sum(len(e['fonts']) for e in words_list)
    all_ok = sum(1 for e in words_list for fd in e['fonts'].values()
                 if all(fd.get(f'ok{p+1}', False) for p in range(n_positions)))
    size_kb = os.path.getsize(output_path) / 1024
    print(f"\nAtlas: {len(words_list)} words x {len(font_names)} fonts = {total} samples")
    print(f"All-correct: {all_ok}/{total} ({all_ok/total*100:.1f}%)")
    print(f"Written to {output_path} ({size_kb:.0f} KB)")


def test_word_isolation(model_dir, test_data_dir, output_dir='word_results', device='auto'):
    """Test a trained WordVisionModel on single-letter 128x128 images.

    Validates that the model can read individual letters (isolation capability).
    Each letter is tested against all 4 position classifiers — reports which
    positions recognise it correctly. Only uses lowercase test images.
    """
    device = _resolve_device(device)
    print(f"Word isolation test on: {device}")

    os.makedirs(output_dir, exist_ok=True)

    model, _, model_type = _load_model(model_dir, device)
    if model_type != 'word':
        print(f"Warning: checkpoint model_type is '{model_type}', expected 'word'")

    n_positions = model.n_positions

    # Discover lowercase test images: img_<letter>_<font>.png
    test_files = []
    for fname in sorted(os.listdir(test_data_dir)):
        if not fname.startswith('img_') or not fname.endswith('.png'):
            continue
        parts = fname[4:-4].split('_', 1)  # letter, font
        if len(parts) != 2:
            continue
        letter, font = parts
        if len(letter) != 1 or not letter.islower():
            continue
        test_files.append((letter, font, os.path.join(test_data_dir, fname)))

    if not test_files:
        print(f"No lowercase test images found in {test_data_dir}")
        return

    print(f"Found {len(test_files)} lowercase test images")

    # Per-position stats
    pos_correct = [0] * n_positions
    total = 0
    any_correct = 0

    # Per-font tracking
    font_stats = {}
    errors = []

    for letter, font, img_path in test_files:
        img = Image.open(img_path).convert('L')
        tensor = torch.tensor(np.array(img) / 255.0, dtype=torch.float32)
        tensor = tensor.unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, 128, 128)

        letter_idx = ord(letter) - ord('a')

        with torch.no_grad():
            _, logits_list, locations, _, _, _, _ = model(tensor)

        preds = [logits_list[p].argmax(dim=1).item() for p in range(n_positions)]
        oks = [preds[p] == letter_idx for p in range(n_positions)]

        for p in range(n_positions):
            pos_correct[p] += int(oks[p])
        any_ok = any(oks)
        any_correct += int(any_ok)
        total += 1

        # Per-font
        if font not in font_stats:
            font_stats[font] = {f'pos{p+1}_ok': 0 for p in range(n_positions)}
            font_stats[font].update({'any_ok': 0, 'total': 0})
        for p in range(n_positions):
            font_stats[font][f'pos{p+1}_ok'] += int(oks[p])
        font_stats[font]['any_ok'] += int(any_ok)
        font_stats[font]['total'] += 1

        pred_chars = [chr(preds[p] + ord('a')) for p in range(n_positions)]
        marks = ['OK' if oks[p] else pred_chars[p] for p in range(n_positions)]
        status = 'OK' if any_ok else 'MISS'
        font_tag = f'  [{font}]' if len(font_stats) > 1 or font != 'default' else ''
        marks_str = '  '.join(f'P{p+1}={marks[p]}' for p in range(n_positions))
        print(f"  {letter}{font_tag}: {marks_str}  {status}")

        if not any_ok:
            errors.append((letter, font, pred_chars))

        # Save attention overlay
        visualize_attention(
            tensor.squeeze(0), locations,
            os.path.join(output_dir, f'iso_attention_{letter}_{font}.png'),
        )

    # Summary
    accs = [pos_correct[p] / total * 100 if total > 0 else 0 for p in range(n_positions)]
    any_acc = any_correct / total * 100 if total > 0 else 0

    print(f"\nIsolation results ({total} single-letter images):")
    for p in range(n_positions):
        print(f"  P{p+1} accuracy: {pos_correct[p]}/{total} ({accs[p]:.1f}%)")
    print(f"  Any-pos correct: {any_correct}/{total} ({any_acc:.1f}%)")

    if len(font_stats) > 1:
        print(f"\nPer-font breakdown:")
        for fname in sorted(font_stats.keys()):
            s = font_stats[fname]
            pos_str = '  '.join(
                f'P{p+1} {s[f"pos{p+1}_ok"]/s["total"]*100:5.1f}%'
                for p in range(n_positions)
            )
            any_pct = s['any_ok'] / s['total'] * 100
            print(f"  {fname:<24s}: {pos_str}  Any {any_pct:5.1f}%  ({s['total']})")

    # Write summary
    summary_path = os.path.join(output_dir, 'summary_isolation.txt')
    with open(summary_path, 'w') as f:
        f.write(f"Word model isolation test ({total} single-letter images)\n")
        f.write("=" * 60 + "\n\n")
        for p in range(n_positions):
            f.write(f"P{p+1} accuracy: {pos_correct[p]}/{total} ({accs[p]:.1f}%)\n")
        f.write(f"Any-pos correct: {any_correct}/{total} ({any_acc:.1f}%)\n")

        if len(font_stats) > 1:
            f.write(f"\nPer-font:\n")
            for fname in sorted(font_stats.keys()):
                s = font_stats[fname]
                pos_str = '  '.join(
                    f'P{p+1} {s[f"pos{p+1}_ok"]/s["total"]*100:5.1f}%'
                    for p in range(n_positions)
                )
                any_pct = s['any_ok'] / s['total'] * 100
                f.write(f"  {fname:<24s}: {pos_str}  Any {any_pct:5.1f}%  ({s['total']})\n")

        if errors:
            f.write(f"\nMisses ({len(errors)}):\n")
            for letter, font, pchars in sorted(errors):
                preds_str = '/'.join(pchars)
                f.write(f"  {letter} [{font}]: predicted {preds_str}\n")

    print(f"\nSummary written to {summary_path}")


def _isolation_atlas_html_template():
    """Return self-contained HTML/CSS/JS template for the word isolation atlas.

    Adapted from word atlas with:
    - 1:1 aspect ratio cells (128x128 images)
    - Detail cells: 180x180
    - Grid items: single letters instead of words
    - Correctness: green=all 4, yellow=some, red=none
    """
    return r'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Word Isolation Atlas</title>
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
.detail-cell canvas { width: 180px; height: 180px; display: block; }
.detail-cell .font-name { font-size: 11px; color: #a0a0c0; margin-top: 2px; }
.detail-cell .pred-info { font-size: 10px; color: #888; }
</style>
</head>
<body>
<div id="controls">
  <span style="font-weight:bold;color:#e94560;">Word Isolation Atlas</span>
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
const NP = DATA.n_positions;
let viewMode = 'heatmap';
let opacity = 0.6;
let selectedIdx = -1;

function hotColor(t) {
  t = Math.max(0, Math.min(1, t));
  let r, g, b;
  if (t < 0.33) { r = t / 0.33; g = 0; b = 0; }
  else if (t < 0.66) { r = 1; g = (t - 0.33) / 0.33; b = 0; }
  else { r = 1; g = 1; b = (t - 0.66) / 0.34; }
  return [r * 255 | 0, g * 255 | 0, b * 255 | 0];
}

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

function aggregateFixations(entry) {
  const all = [];
  for (const fname of DATA.font_names) {
    const fd = entry.fonts[fname];
    if (fd) all.push(...fd.fixations);
  }
  return all;
}

function representativeImage(entry) {
  for (const fname of DATA.font_names) {
    const fd = entry.fonts[fname];
    if (fd) return fd.clean_b64;
  }
  return '';
}

function correctnessClass(entry) {
  let allOk = 0, total = 0;
  for (const fname of DATA.font_names) {
    const fd = entry.fonts[fname];
    if (!fd) continue;
    total++;
    let ok = true;
    for (let p = 0; p < NP; p++) { if (!fd['ok' + (p+1)]) ok = false; }
    if (ok) allOk++;
  }
  if (allOk === total) return 'correct';
  if (allOk === 0) return 'wrong';
  return 'partial';
}

function fontCorrectness(fd) {
  let nOk = 0;
  for (let p = 0; p < NP; p++) { if (fd['ok' + (p+1)]) nOk++; }
  if (nOk === NP) return 'correct';
  if (nOk === 0) return 'wrong';
  return 'partial';
}

function buildGrid() {
  const grid = document.getElementById('grid');
  grid.innerHTML = '';
  let totalAll = 0, totalSamples = 0;

  DATA.letters.forEach((entry, idx) => {
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
    label.textContent = entry.letter;
    cell.appendChild(label);

    cell.onclick = () => { selectedIdx = idx; buildGrid(); showDetail(entry); };
    grid.appendChild(cell);

    for (const fname of DATA.font_names) {
      const fd = entry.fonts[fname];
      if (fd) {
        totalSamples++;
        let ok = true;
        for (let p = 0; p < NP; p++) { if (!fd['ok' + (p+1)]) ok = false; }
        if (ok) totalAll++;
      }
    }
  });

  document.getElementById('stats').textContent =
    `${totalAll}/${totalSamples} all-correct (${(totalAll/totalSamples*100).toFixed(1)}%)`;
}

function showDetail(entry) {
  const panel = document.getElementById('detail-panel');
  const title = document.getElementById('detail-title');
  const grid = document.getElementById('detail-grid');

  title.textContent = `"${entry.letter}" \u2014 per-font attention (${DATA.font_names.length} fonts)`;
  grid.innerHTML = '';
  panel.style.display = 'block';

  for (const fname of DATA.font_names) {
    const fd = entry.fonts[fname];
    if (!fd) continue;

    const cell = document.createElement('div');
    cell.className = 'detail-cell ' + fontCorrectness(fd);

    const canvas = document.createElement('canvas');
    drawCell(canvas, fd.clean_b64, fd.fixations, 180, 180);
    cell.appendChild(canvas);

    const fnLabel = document.createElement('div');
    fnLabel.className = 'font-name';
    fnLabel.textContent = fname;
    cell.appendChild(fnLabel);

    const pred = document.createElement('div');
    pred.className = 'pred-info';
    let allOk = true;
    for (let p = 0; p < NP; p++) { if (!fd['ok' + (p+1)]) allOk = false; }
    if (allOk) {
      pred.textContent = 'OK';
    } else {
      let parts = [];
      for (let p = 0; p < NP; p++) {
        parts.push('P' + (p+1) + ':' + (fd['ok' + (p+1)] ? 'OK' : fd['pred' + (p+1)]));
      }
      pred.textContent = parts.join(' ');
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
  if (selectedIdx >= 0) showDetail(DATA.letters[selectedIdx]);
}

function setOpacity(val) {
  opacity = val / 100;
  document.getElementById('opacity-val').textContent = val + '%';
  buildGrid();
  if (selectedIdx >= 0) showDetail(DATA.letters[selectedIdx]);
}

buildGrid();
</script>
</body>
</html>'''


def generate_isolation_atlas(model_dir, test_data_dir,
                             output_path='data/isolation_atlas.html',
                             device='auto'):
    """Generate an interactive HTML attention atlas for word model isolation test.

    Runs single 128x128 lowercase letter images through the WordVisionModel and
    shows fixation heatmaps/paths with per-position classification results.
    """
    device = _resolve_device(device)
    print(f"Generating isolation atlas on: {device}")

    model, n_glimpses, model_type = _load_model(model_dir, device)
    if model_type != 'word':
        print(f"Warning: checkpoint model_type is '{model_type}', expected 'word'")

    n_positions = model.n_positions

    # Discover lowercase test images (same logic as test_word_isolation)
    test_files = []
    for fname in sorted(os.listdir(test_data_dir)):
        if not fname.startswith('img_') or not fname.endswith('.png'):
            continue
        parts = fname[4:-4].split('_', 1)
        if len(parts) != 2:
            continue
        letter, font = parts
        if len(letter) != 1 or not letter.islower():
            continue
        test_files.append((letter, font, os.path.join(test_data_dir, fname)))

    if not test_files:
        print(f"No lowercase test images found in {test_data_dir}")
        return

    font_names = sorted(set(f for _, f, _ in test_files))
    entries = {}  # letter -> {letter, fonts: {font -> {...}}}

    for letter, font, img_path in test_files:
        img = Image.open(img_path).convert('L')
        tensor = torch.tensor(np.array(img) / 255.0, dtype=torch.float32)
        tensor = tensor.unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, 128, 128)

        letter_idx = ord(letter) - ord('a')

        with torch.no_grad():
            _, logits_list, locations, _, _, _, _ = model(tensor)

        fixations = []
        for loc in locations:
            loc_np = loc[0].cpu().detach().tolist()
            fixations.append([round(loc_np[0], 4), round(loc_np[1], 4)])

        preds = [logits_list[p].argmax(dim=1).item() for p in range(n_positions)]
        oks = [preds[p] == letter_idx for p in range(n_positions)]
        pred_chars = [chr(preds[p] + ord('a')) for p in range(n_positions)]

        if letter not in entries:
            entries[letter] = {'letter': letter, 'fonts': {}}

        # Clean image for display (use the raw grayscale tensor)
        clean_tensor = tensor.squeeze(0)  # (1, 128, 128)
        font_data = {
            'fixations': fixations,
            'clean_b64': _tensor_to_base64_png(clean_tensor),
        }
        for p in range(n_positions):
            font_data[f'pred{p+1}'] = pred_chars[p]
            font_data[f'ok{p+1}'] = oks[p]

        entries[letter]['fonts'][font] = font_data

        ok_all = all(oks)
        mark = 'OK' if ok_all else ' '.join(
            f'P{p+1}={pred_chars[p]}' if not oks[p] else f'P{p+1}=OK'
            for p in range(n_positions)
        )
        print(f"  {letter} [{font}]: {mark}")

    # Build ordered list (alphabetical)
    letters_list = [entries[k] for k in sorted(entries.keys())]

    atlas_data = {
        'image_width': 128,
        'image_height': 128,
        'n_fixations': n_glimpses + 1,
        'n_positions': n_positions,
        'font_names': font_names,
        'letters': letters_list,
    }

    atlas_json = json.dumps(atlas_data)
    html = _isolation_atlas_html_template().replace('ATLAS_JSON_PLACEHOLDER', atlas_json)

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(html)

    total = sum(len(e['fonts']) for e in letters_list)
    all_ok = sum(1 for e in letters_list for fd in e['fonts'].values()
                 if all(fd.get(f'ok{p+1}', False) for p in range(n_positions)))
    size_kb = os.path.getsize(output_path) / 1024
    print(f"\nAtlas: {len(letters_list)} letters x {len(font_names)} fonts = {total} samples")
    print(f"All-correct: {all_ok}/{total} ({all_ok/total*100:.1f}%)")
    print(f"Written to {output_path} ({size_kb:.0f} KB)")
