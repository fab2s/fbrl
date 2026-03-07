"""Reading evaluation functions (v9.2 — isolated read heads) — imported by evaluate.py."""
import torch
import torch.nn.functional as F
import numpy as np
import os
import json
import base64
import io

from fbrl import _resolve_device
from fbrl.data import CountingDataset
from fbrl.model import ReadingModel
from fbrl.losses import fixation_hit_rate


def _load_reading_model(model_dir, device):
    """Load a trained ReadingModel v9.2 from checkpoint."""
    ckpt = torch.load(os.path.join(model_dir, 'model_final.pth'),
                      map_location=device, weights_only=False)
    n_zones = ckpt.get('n_zones', 4)
    n_heads_per_zone = ckpt.get('n_heads_per_zone', 2)
    n_search_steps = ckpt.get('n_search_steps', 2)
    n_prescan_steps = ckpt.get('n_prescan_steps', 1)
    n_read_steps = ckpt.get('n_read_steps', 6)
    head_offset = ckpt.get('head_offset', 0.12)
    probe_patch_pixels = tuple(ckpt.get('probe_patch_pixels', [32, 96]))
    probe_blur_sigma = ckpt.get('probe_blur_sigma', 6.0)
    search_patch_pixels = tuple(ckpt.get('search_patch_pixels', [20, 28]))
    search_blur_sigma = ckpt.get('search_blur_sigma', 2.0)
    prescan_patch_size = tuple(ckpt.get('prescan_patch_size', [12, 18]))
    read_patch_size = ckpt.get('read_patch_size', 12)
    latent_dim = ckpt.get('latent_dim', 256)
    n_scales = ckpt.get('n_scales', 1)
    n_letter_classes = ckpt.get('n_letter_classes', 27)

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
    model.load_state_dict(ckpt['model'], strict=False)
    model.eval()
    n_heads = n_zones * n_heads_per_zone
    return model, n_heads, n_zones


def test_reading_model(model_dir, test_data_dir, output_dir='reading_results',
                        device='auto'):
    """Test a trained ReadingModel v9.2 on counting test data.

    Reports per-head letter accuracy, void accuracy,
    derived count accuracy, and per-font breakdown.
    """
    device = _resolve_device(device)
    print(f"Reading v9.2 testing on: {device}")

    os.makedirs(output_dir, exist_ok=True)
    model, n_heads, n_zones = _load_reading_model(model_dir, device)
    dataset = CountingDataset(test_data_dir)

    max_count = 3
    per_count_correct = {c: 0 for c in range(1, max_count + 1)}
    per_count_total = {c: 0 for c in range(1, max_count + 1)}
    total_letter_correct = 0
    total_letter_total = 0
    total_void_correct = 0
    total_void_total = 0
    confusion_count = np.zeros((max_count, max_count), dtype=int)

    font_stats = {}
    errors = []

    for i in range(len(dataset)):
        img, clean, count, letters, font, char_pos, char_lab = dataset[i]
        img_t = img.unsqueeze(0).to(device)
        char_pos_t = char_pos.unsqueeze(0).to(device)
        char_lab_t = char_lab.unsqueeze(0).to(device)

        with torch.no_grad():
            group_logits, enc = model(img_t)  # (1, n_heads, 27)
            preds = group_logits.argmax(2).squeeze(0)  # (n_heads,)

            # Derived count
            pred_count = (preds != 26).sum().item()
            pred_count = max(1, min(pred_count, max_count))

            # Assign ground-truth for letter accuracy
            from fbrl._reading_train import assign_heads_spatial
            targets = assign_heads_spatial(
                enc.prescan_end_positions, char_pos_t, char_lab_t, n_heads,
            ).squeeze(0)  # (n_heads,)

            # Letter accuracy (non-void)
            nonvoid = targets != 26
            if nonvoid.any():
                total_letter_correct += (preds[nonvoid] == targets[nonvoid]).sum().item()
                total_letter_total += nonvoid.sum().item()

            # Void accuracy
            void_mask = targets == 26
            if void_mask.any():
                total_void_correct += (preds[void_mask] == 26).sum().item()
                total_void_total += void_mask.sum().item()

        per_count_total[count] = per_count_total.get(count, 0) + 1
        pred_count_clamped = max(1, min(pred_count, max_count))
        confusion_count[count - 1, pred_count_clamped - 1] += 1

        if pred_count == count:
            per_count_correct[count] = per_count_correct.get(count, 0) + 1
        else:
            pred_letters = []
            for p in preds:
                if p.item() != 26:
                    pred_letters.append(chr(ord('a') + p.item()))
            errors.append({
                'letters': letters, 'font': font,
                'true_count': count, 'predicted_count': pred_count,
                'pred_letters': ''.join(pred_letters),
            })

        if font not in font_stats:
            font_stats[font] = {'count_correct': 0, 'total': 0}
        font_stats[font]['total'] += 1
        if pred_count == count:
            font_stats[font]['count_correct'] += 1

    total_count_correct = sum(per_count_correct.values())
    total = sum(per_count_total.values())
    count_acc = total_count_correct / total if total > 0 else 0
    letter_acc = total_letter_correct / max(total_letter_total, 1)
    void_acc = total_void_correct / max(total_void_total, 1)

    print(f"\nDerived count accuracy: {total_count_correct}/{total} = {count_acc:.1%}")
    print(f"Letter accuracy (non-void heads): {total_letter_correct}/{total_letter_total} = {letter_acc:.1%}")
    print(f"Void accuracy: {total_void_correct}/{total_void_total} = {void_acc:.1%}")

    for c in sorted(per_count_total):
        cc = per_count_correct.get(c, 0)
        tt = per_count_total[c]
        acc = cc / tt if tt > 0 else 0
        print(f"  Count {c}: {cc}/{tt} = {acc:.1%}")

    print(f"\nCount confusion matrix (rows=true, cols=predicted):")
    header = "     " + "  ".join(f"P={c}" for c in range(1, max_count + 1))
    print(header)
    for r in range(max_count):
        row = f"T={r+1}  " + "  ".join(f"{confusion_count[r, c]:4d}" for c in range(max_count))
        print(row)

    print(f"\nPer-font count accuracy:")
    for font in sorted(font_stats):
        fs = font_stats[font]
        acc = fs['count_correct'] / fs['total'] if fs['total'] > 0 else 0
        print(f"  {font}: {fs['count_correct']}/{fs['total']} = {acc:.1%}")

    if errors:
        print(f"\n{len(errors)} count errors (first 20):")
        for e in errors[:20]:
            print(f"  '{e['letters']}' ({e['font']}): true={e['true_count']}, "
                  f"pred_count={e['predicted_count']}, pred_letters='{e['pred_letters']}'")

    results = {
        'count_accuracy': count_acc,
        'letter_accuracy': letter_acc,
        'void_accuracy': void_acc,
        'per_count': {str(c): {
            'correct': per_count_correct.get(c, 0),
            'total': per_count_total[c],
            'accuracy': per_count_correct.get(c, 0) / per_count_total[c] if per_count_total[c] > 0 else 0,
        } for c in sorted(per_count_total)},
        'confusion_matrix': confusion_count.tolist(),
        'per_font': {font: {
            'count_correct': fs['count_correct'], 'total': fs['total'],
            'accuracy': fs['count_correct'] / fs['total'] if fs['total'] > 0 else 0,
        } for font, fs in sorted(font_stats.items())},
        'n_errors': len(errors),
    }
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {os.path.join(output_dir, 'results.json')}")


def generate_reading_atlas(model_dir, test_data_dir,
                            output_path='data/reading_atlas.html',
                            device='auto'):
    """Generate HTML atlas with phase-colored trajectory overlay.

    Blue circles: content probes (4 fixed positions)
    Green circles: search positions (2 per zone, 8 total)
    Yellow circles: prescan positions (refine exact center)
    Orange diamonds: prescan endpoints (read head anchors)
    Per-head colored dots: read positions
    Letter prediction labels on each active head
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    device = _resolve_device(device)
    print(f"Generating reading v9.2 atlas on: {device}")

    model, n_heads, n_zones = _load_reading_model(model_dir, device)
    dataset = CountingDataset(test_data_dir)

    max_count = 3
    by_count = {c: [] for c in range(1, max_count + 1)}
    for i in range(len(dataset)):
        img, clean, count, letters, font, _cp, _cl = dataset[i]
        by_count[count].append((i, letters, font))

    # Per-head colors (8 heads)
    head_colors = ['#ff4444', '#ff8844', '#ffcc44', '#ff44cc',
                   '#44ccff', '#44ff88', '#cc44ff', '#ffff44']

    html_parts = ["""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Reading Atlas (v9.2 — Isolated Heads)</title>
<style>
body { font-family: monospace; background: #1a1a1a; color: #ccc; padding: 20px; }
h1 { color: #fff; }
h2 { color: #aaa; margin-top: 30px; }
.legend { margin: 10px 0; }
.legend span { margin-right: 15px; }
.probe { color: #4488ff; }
.search { color: #44cc44; }
.prescan { color: #cccc44; }
.anchor { color: #ff8800; }
.read { color: #ff4444; }
.grid { display: flex; flex-wrap: wrap; gap: 10px; }
.card { background: #2a2a2a; padding: 8px; border-radius: 4px; text-align: center; }
.card img { display: block; margin: 0 auto; }
.correct { border: 2px solid #4a4; }
.wrong { border: 2px solid #a44; }
.label { font-size: 12px; margin-top: 4px; }
</style></head><body>
<h1>Reading Atlas (v9.2 — Isolated Read Heads)</h1>
<div class="legend">
  <span class="probe">&#9679; Content probes</span>
  <span class="search">&#9679; Search positions</span>
  <span class="prescan">&#9679; Prescan positions</span>
  <span class="anchor">&#9670; Prescan endpoints</span>
  <span class="read">&#9679; Read positions (per head)</span>
</div>
"""]

    n_samples_per_count = 30

    for count_val in sorted(by_count):
        items = by_count[count_val][:n_samples_per_count]
        correct = 0
        total = len(items)

        html_parts.append(f'<h2>Count = {count_val} (showing {total} samples)</h2>\n<div class="grid">\n')

        for idx, letters, font in items:
            img, clean, count, _letters, _font, char_pos, char_lab = dataset[idx]
            img_t = img.unsqueeze(0).to(device)

            with torch.no_grad():
                group_logits, enc = model(img_t)
                preds = group_logits.argmax(2).squeeze(0)  # (n_heads,)
                pred_count = (preds != 26).sum().item()
                pred_count = max(1, min(pred_count, max_count))

            is_correct = pred_count == count_val
            if is_correct:
                correct += 1

            img_np = img.squeeze(0).cpu().numpy()
            H, W = img_np.shape

            fig, ax = plt.subplots(figsize=(max(2, W / 64), 2))
            ax.imshow(img_np, cmap='gray', vmin=0, vmax=1)

            # Plot by phase
            for li, (loc, tag, head_id) in enumerate(
                    zip(enc.locations, enc.phase_tags, enc.head_ids)):
                loc_np = loc[0].cpu().numpy()
                px = (loc_np[0] + 1) / 2 * W
                py = (loc_np[1] + 1) / 2 * H

                if tag == 'probe':
                    ax.plot(px, py, 'o', color='#4488ff', markersize=8,
                            markeredgecolor='white', markeredgewidth=0.2)
                elif tag == 'search':
                    ax.plot(px, py, 'o', color='#44cc44', markersize=5,
                            markeredgecolor='white', markeredgewidth=0.2)
                elif tag == 'prescan':
                    ax.plot(px, py, 'o', color='#cccc44', markersize=6,
                            markeredgecolor='white', markeredgewidth=0.2)
                elif tag == 'read':
                    color = head_colors[head_id % len(head_colors)]
                    ax.plot(px, py, 'o', color=color, markersize=3,
                            markeredgecolor='white', markeredgewidth=0.1)

            # Plot prescan endpoints (anchors) and labels
            for hi in range(n_heads):
                anchor = enc.prescan_end_positions[hi][0].cpu().numpy()
                ax_x = (anchor[0] + 1) / 2 * W
                ax_y = (anchor[1] + 1) / 2 * H
                ax.plot(ax_x, ax_y, 'D', color='#ff8800', markersize=5,
                        markeredgecolor='white', markeredgewidth=0.3)

                pred_idx = preds[hi].item()
                if pred_idx < 26:
                    pred_letter = chr(ord('a') + pred_idx)
                    color = head_colors[hi % len(head_colors)]
                    ax.text(ax_x, -2, pred_letter, fontsize=7, color=color,
                            ha='center', va='bottom', fontweight='bold')

            ax.axis('off')
            plt.tight_layout(pad=0)

            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=100, bbox_inches='tight',
                        facecolor='#2a2a2a')
            plt.close(fig)
            b64 = base64.b64encode(buf.getvalue()).decode('ascii')

            css_class = 'correct' if is_correct else 'wrong'

            pred_letters = []
            for p in preds:
                if p.item() < 26:
                    pred_letters.append(chr(ord('a') + p.item()))
            pred_str = ''.join(pred_letters) if pred_letters else '-'

            html_parts.append(
                f'<div class="card {css_class}">'
                f'<img src="data:image/png;base64,{b64}">'
                f'<div class="label">"{letters}" {font}<br>'
                f'pred: {pred_str} (cnt={pred_count})</div>'
                f'</div>\n'
            )

        acc = correct / total if total > 0 else 0
        html_parts.append(f'</div>\n<p>Count accuracy: {correct}/{total} = {acc:.0%}</p>\n')

    html_parts.append('</body></html>')

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(''.join(html_parts))
    print(f"Atlas saved to {output_path}")
