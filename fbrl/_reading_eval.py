"""Three-phase reading evaluation functions — imported by evaluate.py."""
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
    """Load a trained ReadingModel from checkpoint."""
    ckpt = torch.load(os.path.join(model_dir, 'model_final.pth'),
                      map_location=device, weights_only=False)
    n_meta = ckpt.get('n_meta', 4)
    n_sub_per_meta = ckpt.get('n_sub_per_meta', 3)
    n_read_per_sub = ckpt.get('n_read_per_sub', 3)
    meta_patch_pixels = tuple(ckpt.get('meta_patch_pixels', [32, 96]))
    meta_blur_sigma = ckpt.get('meta_blur_sigma', 6.0)
    sub_patch_pixels = tuple(ckpt.get('sub_patch_pixels', [20, 28]))
    sub_blur_sigma = ckpt.get('sub_blur_sigma', 2.0)
    read_patch_size = ckpt.get('read_patch_size', 12)
    latent_dim = ckpt.get('latent_dim', 256)
    n_scales = ckpt.get('n_scales', 1)
    max_count = ckpt.get('max_count', 3)

    model = ReadingModel(
        n_meta=n_meta, n_sub_per_meta=n_sub_per_meta,
        n_read_per_sub=n_read_per_sub,
        meta_patch_pixels=meta_patch_pixels, meta_blur_sigma=meta_blur_sigma,
        sub_patch_pixels=sub_patch_pixels, sub_blur_sigma=sub_blur_sigma,
        read_patch_size=read_patch_size, latent_dim=latent_dim,
        n_scales=n_scales, max_count=max_count,
    ).to(device)
    model.load_state_dict(ckpt['model'], strict=False)
    model.eval()
    return model, max_count


def test_reading_model(model_dir, test_data_dir, output_dir='reading_results',
                        device='auto'):
    """Test a trained ReadingModel on counting test data.

    Reports per-count accuracy, overall accuracy, confusion matrix,
    and per-font breakdown.
    """
    device = _resolve_device(device)
    print(f"Reading testing on: {device}")

    os.makedirs(output_dir, exist_ok=True)
    model, max_count = _load_reading_model(model_dir, device)
    dataset = CountingDataset(test_data_dir)

    per_count_correct = {c: 0 for c in range(1, max_count + 1)}
    per_count_total = {c: 0 for c in range(1, max_count + 1)}
    confusion = np.zeros((max_count, max_count), dtype=int)

    font_stats = {}
    errors = []

    for i in range(len(dataset)):
        img, clean, count, letters, font = dataset[i]
        img = img.unsqueeze(0).to(device)

        with torch.no_grad():
            count_logits, enc = model(img)
            pred = count_logits.argmax(1).item() + 1

        per_count_total[count] = per_count_total.get(count, 0) + 1
        confusion[count - 1, pred - 1] += 1

        if pred == count:
            per_count_correct[count] = per_count_correct.get(count, 0) + 1
        else:
            errors.append({
                'letters': letters, 'font': font,
                'true_count': count, 'predicted': pred,
            })

        if font not in font_stats:
            font_stats[font] = {'correct': 0, 'total': 0}
        font_stats[font]['total'] += 1
        if pred == count:
            font_stats[font]['correct'] += 1

    total_correct = sum(per_count_correct.values())
    total = sum(per_count_total.values())
    overall_acc = total_correct / total if total > 0 else 0

    print(f"\nOverall: {total_correct}/{total} = {overall_acc:.1%}")
    for c in sorted(per_count_total):
        cc = per_count_correct.get(c, 0)
        tt = per_count_total[c]
        acc = cc / tt if tt > 0 else 0
        print(f"  Count {c}: {cc}/{tt} = {acc:.1%}")

    print(f"\nConfusion matrix (rows=true, cols=predicted):")
    header = "     " + "  ".join(f"P={c}" for c in range(1, max_count + 1))
    print(header)
    for r in range(max_count):
        row = f"T={r+1}  " + "  ".join(f"{confusion[r, c]:4d}" for c in range(max_count))
        print(row)

    print(f"\nPer-font accuracy:")
    for font in sorted(font_stats):
        fs = font_stats[font]
        acc = fs['correct'] / fs['total'] if fs['total'] > 0 else 0
        print(f"  {font}: {fs['correct']}/{fs['total']} = {acc:.1%}")

    if errors:
        print(f"\n{len(errors)} errors (first 20):")
        for e in errors[:20]:
            print(f"  '{e['letters']}' ({e['font']}): true={e['true_count']}, pred={e['predicted']}")

    results = {
        'overall_accuracy': overall_acc,
        'per_count': {str(c): {
            'correct': per_count_correct.get(c, 0),
            'total': per_count_total[c],
            'accuracy': per_count_correct.get(c, 0) / per_count_total[c] if per_count_total[c] > 0 else 0,
        } for c in sorted(per_count_total)},
        'confusion_matrix': confusion.tolist(),
        'per_font': {font: {
            'correct': fs['correct'], 'total': fs['total'],
            'accuracy': fs['correct'] / fs['total'] if fs['total'] > 0 else 0,
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

    Blue circles: meta-scan positions
    Green circles: sub-scan positions
    Red/hot colormap: read positions
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    device = _resolve_device(device)
    print(f"Generating reading atlas on: {device}")

    model, max_count = _load_reading_model(model_dir, device)
    dataset = CountingDataset(test_data_dir)

    by_count = {c: [] for c in range(1, max_count + 1)}
    for i in range(len(dataset)):
        img, clean, count, letters, font = dataset[i]
        by_count[count].append((i, letters, font))

    html_parts = ["""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Reading Atlas (Three-Phase)</title>
<style>
body { font-family: monospace; background: #1a1a1a; color: #ccc; padding: 20px; }
h1 { color: #fff; }
h2 { color: #aaa; margin-top: 30px; }
.legend { margin: 10px 0; }
.legend span { margin-right: 15px; }
.meta { color: #4488ff; }
.sub { color: #44cc44; }
.read { color: #ff4444; }
.grid { display: flex; flex-wrap: wrap; gap: 10px; }
.card { background: #2a2a2a; padding: 8px; border-radius: 4px; text-align: center; }
.card img { display: block; margin: 0 auto; }
.correct { border: 2px solid #4a4; }
.wrong { border: 2px solid #a44; }
.label { font-size: 12px; margin-top: 4px; }
</style></head><body>
<h1>Reading Atlas (Three-Phase Foveal Pipeline)</h1>
<div class="legend">
  <span class="meta">&#9679; Meta-scan (peripheral)</span>
  <span class="sub">&#9679; Sub-scan (parafoveal)</span>
  <span class="read">&#9679; Read (foveal)</span>
</div>
"""]

    n_samples_per_count = 30
    phase_colors = {'meta': '#4488ff', 'sub': '#44cc44', 'read': '#ff4444'}
    phase_sizes = {'meta': 8, 'sub': 6, 'read': 4}

    for count_val in sorted(by_count):
        items = by_count[count_val][:n_samples_per_count]
        correct = 0
        total = len(items)

        html_parts.append(f'<h2>Count = {count_val} (showing {total} samples)</h2>\n<div class="grid">\n')

        for idx, letters, font in items:
            img, clean, count, _letters, _font = dataset[idx]
            img_t = img.unsqueeze(0).to(device)

            with torch.no_grad():
                count_logits, enc = model(img_t)
                pred = count_logits.argmax(1).item() + 1

            is_correct = pred == count_val
            if is_correct:
                correct += 1

            img_np = img.squeeze(0).cpu().numpy()
            H, W = img_np.shape

            fig, ax = plt.subplots(figsize=(max(2, W / 64), 2))
            ax.imshow(img_np, cmap='gray', vmin=0, vmax=1)

            # Plot phase-colored trajectory
            for li, (loc, tag) in enumerate(zip(enc.locations, enc.phase_tags)):
                if tag == 'init':
                    continue
                loc_np = loc[0].cpu().numpy()
                px = (loc_np[0] + 1) / 2 * W
                py = (loc_np[1] + 1) / 2 * H
                color = phase_colors.get(tag, 'white')
                size = phase_sizes.get(tag, 4)
                ax.plot(px, py, 'o', color=color, markersize=size,
                        markeredgecolor='white', markeredgewidth=0.2)

            ax.axis('off')
            plt.tight_layout(pad=0)

            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=100, bbox_inches='tight',
                        facecolor='#2a2a2a')
            plt.close(fig)
            b64 = base64.b64encode(buf.getvalue()).decode('ascii')

            css_class = 'correct' if is_correct else 'wrong'
            pred_str = f'pred={pred}' if not is_correct else f'{pred}'
            html_parts.append(
                f'<div class="card {css_class}">'
                f'<img src="data:image/png;base64,{b64}">'
                f'<div class="label">"{letters}" {font}<br>{pred_str}</div>'
                f'</div>\n'
            )

        acc = correct / total if total > 0 else 0
        html_parts.append(f'</div>\n<p>Accuracy: {correct}/{total} = {acc:.0%}</p>\n')

    html_parts.append('</body></html>')

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(''.join(html_parts))
    print(f"Atlas saved to {output_path}")
