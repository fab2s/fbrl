"""Counting evaluation functions — imported by evaluate.py."""
import torch
import torch.nn.functional as F
import numpy as np
import os
import json
import base64
import io
from PIL import Image

from fbrl import _resolve_device
from fbrl.data import CountingDataset
from fbrl.model import CountingModel
from fbrl.losses import fixation_hit_rate
from fbrl.evaluate import _tensor_to_base64_png


def _load_counting_model(model_dir, device):
    """Load a trained CountingModel from checkpoint."""
    ckpt = torch.load(os.path.join(model_dir, 'model_final.pth'),
                      map_location=device, weights_only=False)
    n_scan = ckpt.get('n_scan_glimpses', 6)
    scan_ps = ckpt.get('scan_patch_size', (12, 18))
    latent_dim = ckpt.get('latent_dim', 256)
    n_scales = ckpt.get('n_scales', 1)
    max_count = ckpt.get('max_count', 3)

    model = CountingModel(
        n_scan_glimpses=n_scan, scan_patch_size=scan_ps,
        latent_dim=latent_dim, n_scales=n_scales, max_count=max_count,
    ).to(device)
    model.load_state_dict(ckpt['model'], strict=False)
    model.eval()
    return model, max_count


def test_counting_model(model_dir, test_data_dir, output_dir='counting_results',
                         device='auto'):
    """Test a trained CountingModel on counting test data.

    Reports per-count accuracy, overall accuracy, confusion matrix,
    and per-font breakdown.
    """
    device = _resolve_device(device)
    print(f"Counting testing on: {device}")

    os.makedirs(output_dir, exist_ok=True)
    model, max_count = _load_counting_model(model_dir, device)
    dataset = CountingDataset(test_data_dir)

    # Per-count stats
    per_count_correct = {c: 0 for c in range(1, max_count + 1)}
    per_count_total = {c: 0 for c in range(1, max_count + 1)}
    confusion = np.zeros((max_count, max_count), dtype=int)

    # Per-font stats
    font_stats = {}
    errors = []

    for i in range(len(dataset)):
        img, clean, count, letters, font, *_ = dataset[i]
        img = img.unsqueeze(0).to(device)

        with torch.no_grad():
            count_logits, _, locations, _ = model(img)
            pred = count_logits.argmax(1).item() + 1  # 0-indexed -> 1-indexed

        per_count_total[count] = per_count_total.get(count, 0) + 1
        confusion[count - 1, pred - 1] += 1

        if pred == count:
            per_count_correct[count] = per_count_correct.get(count, 0) + 1
        else:
            errors.append({
                'letters': letters, 'font': font,
                'true_count': count, 'predicted': pred,
            })

        # Per-font
        if font not in font_stats:
            font_stats[font] = {'correct': 0, 'total': 0}
        font_stats[font]['total'] += 1
        if pred == count:
            font_stats[font]['correct'] += 1

    # Summary
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


def generate_counting_atlas(model_dir, test_data_dir,
                             output_path='data/counting_atlas.html',
                             device='auto'):
    """Generate HTML atlas with sample images per count, scan trajectories overlaid."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    device = _resolve_device(device)
    print(f"Generating counting atlas on: {device}")

    model, max_count = _load_counting_model(model_dir, device)
    dataset = CountingDataset(test_data_dir)

    # Group samples by count
    by_count = {c: [] for c in range(1, max_count + 1)}
    for i in range(len(dataset)):
        img, clean, count, letters, font, *_ = dataset[i]
        by_count[count].append((i, letters, font))

    html_parts = ["""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Counting Atlas</title>
<style>
body { font-family: monospace; background: #1a1a1a; color: #ccc; padding: 20px; }
h1 { color: #fff; }
h2 { color: #aaa; margin-top: 30px; }
.grid { display: flex; flex-wrap: wrap; gap: 10px; }
.card { background: #2a2a2a; padding: 8px; border-radius: 4px; text-align: center; }
.card img { display: block; margin: 0 auto; }
.correct { border: 2px solid #4a4; }
.wrong { border: 2px solid #a44; }
.label { font-size: 12px; margin-top: 4px; }
</style></head><body>
<h1>Counting Atlas</h1>
"""]

    n_samples_per_count = 30  # show up to 30 per count

    for count_val in sorted(by_count):
        items = by_count[count_val][:n_samples_per_count]
        correct = 0
        total = len(items)

        html_parts.append(f'<h2>Count = {count_val} (showing {total} samples)</h2>\n<div class="grid">\n')

        for idx, letters, font in items:
            img, clean, count, _letters, _font, *_ = dataset[idx]
            img_t = img.unsqueeze(0).to(device)

            with torch.no_grad():
                count_logits, scan_content_logits, locations, _ = model(img_t)
                pred = count_logits.argmax(1).item() + 1

            is_correct = pred == count_val
            if is_correct:
                correct += 1

            # Render image with scan trajectory
            img_np = img.squeeze(0).cpu().numpy()
            H, W = img_np.shape

            fig, ax = plt.subplots(figsize=(max(2, W / 64), 2))
            ax.imshow(img_np, cmap='gray', vmin=0, vmax=1)

            locs = locations[:-1]  # drop vestigial last
            colors = plt.cm.hot(np.linspace(0.2, 0.9, len(locs)))
            for si, loc in enumerate(locs):
                loc_np = loc[0].cpu().numpy()
                px = (loc_np[0] + 1) / 2 * W
                py = (loc_np[1] + 1) / 2 * H
                ax.plot(px, py, 'o', color=colors[si], markersize=5,
                        markeredgecolor='white', markeredgewidth=0.3)
                if si > 0:
                    prev_np = locs[si - 1][0].cpu().numpy()
                    prev_px = (prev_np[0] + 1) / 2 * W
                    prev_py = (prev_np[1] + 1) / 2 * H
                    ax.annotate('', xy=(px, py), xytext=(prev_px, prev_py),
                                arrowprops=dict(arrowstyle='->', color=colors[si], lw=1))
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
