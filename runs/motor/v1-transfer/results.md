# Results: Motor v1-transfer, Curriculum Learning from v5-scan, 200 epochs

## Summary
- **Letter accuracy**: 560/572 (97.9%) across 11 fonts
- **Case accuracy**: 510/572 (89.2%)
- **Re-read letter accuracy**: 367/572 (64.2%) — motor traces read back through vision encoder
- **Re-read case accuracy**: 404/572 (70.6%)
- **Recon MSE**: 0.0018
- **Trajectory MSE**: 0.4379 (against font vector GT)
- **Pen F1**: 0.943 (P=0.974, R=0.914)
- **Training time**: 200 epochs, ~77s/epoch (257 min total)
- **Dataset**: 11,440 samples (52 letters × 11 fonts × 20 variants)
- **Transfer from**: `runs/letters/v5-scan/model_final.pth.gz` (scan-phase letter model)

## Key finding: curriculum learning works

Transfer from a pretrained vision model dramatically improves both vision and motor:

| Metric | From scratch | v1-transfer |
|--------|-------------|-------------|
| Letter accuracy | 84.6% | **97.9%** |
| Case accuracy | 78.8% | **89.2%** |
| RR letter accuracy | 47.6% | **64.2%** |
| RR case accuracy | — | **70.6%** |
| Pen F1 | 0.94 | 0.943 |
| Time/epoch | ~80s | ~77s |

The from-scratch model suffered from vision and motor competing for the shared latent — vision degraded from 100% (pure vision model) to 84.6%. With transfer, vision starts pretrained and barely moves (epoch 1 letter CE = 0.065, epoch 200 = 0.084), giving the motor decoder a stable, high-quality latent to learn from.

## But: re-read accuracy is misleading

Despite 64.2% re-read accuracy, the atlas reveals most rendered trajectories are not visually readable as letters. The motor decoder produces rough spatial distributions that the vision encoder can partially decode, but not actual letter shapes.

### Why the signal is too soft

1. **Fat Gaussian blobs**: render sigma=1.5 creates ~6px-wide blobs. With 32 points, even crude spatial placement covers the right general area. A bad "D" with blobs on the right side looks different enough from a bad "L" with blobs on the bottom for the encoder to discriminate.

2. **Encoder co-adaptation**: the same vision encoder reads both font images and motor output. Over time it learns to decode motor artifacts — spatial blob distributions become diagnostic shortcuts rather than letter shapes.

3. **Classification is coarse**: cross-entropy gives no gradient for "how far off" — an unreadable trace classified correctly contributes zero motor loss. A nearly-perfect trace confused with a neighbor gets the same penalty as random scribble.

### Systematic confusion patterns

Re-read errors show the motor decoder learns structural families, not letter identities:
- **O→Q** across all 11 fonts — motor adds a tail-like artifact
- **D→T** across 10 fonts — produces a top-heavy shape without the curved body
- **B→S/N/G** — can't render the double-bump structure
- **l→I, j→I, i→J** — thin vertical letters collapse to the same trace
- **K→H, R→I/L/F** — diagonal strokes lost, only verticals survive
- **Case is hard**: S, G, I, W, T wrong case across nearly all fonts. The motor decoder doesn't distinguish upper/lower trajectory scale.

Some letters work well: C, L, N (simple strokes), A (distinctive shape), and most lowercase letters with unique geometry (a, d, m, s).

## Training dynamics

### Convergence timeline
- **Epoch 1**: ltr 0.065 CE (pretrained!), rr_ltr 7%, pen BCE 0.69, traj MSE 0.44
- **Epoch 10**: rr_ltr 44%, pen BCE 0.22 — fast initial motor learning against stable vision
- **Epoch 50**: rr_ltr 37-50% (noisy), pen BCE 0.14 — motor oscillating
- **Epoch 100**: rr_ltr 15%, rr_cls 8.0 — regression as scaffold anneals
- **Epoch 134**: scaffold hits 0.1 floor — rr_cls becomes dominant motor signal
- **Epoch 142**: rr_ltr 31% — re-read signal begins recovering
- **Epoch 200**: rr_ltr 66%, rr_cls 1.79 — strong final climb

### The mid-training dip (epochs 50-134)

Re-read accuracy peaked early (~45% at epoch 10-12), then dropped to ~15% by epoch 100 before recovering. This happened because:
1. Early trajectory scaffold (weight=1.0) gave strong shape guidance, producing GT-like traces the encoder could read
2. As scaffold annealed (1.0→0.1), the motor decoder drifted from GT shapes
3. The rr_cls signal was too weak relative to the (still large) scaffold to maintain readability
4. Once scaffold hit floor (0.1), rr_cls dominated and pulled trajectories back toward readable output
5. Final recovery to 66% shows the re-read signal works — just needs to be stronger earlier

### Vision stability

Transfer preserved vision quality throughout:
- Recon MSE: 0.0065 (epoch 1) → 0.0039 (epoch 200) — improved
- Letter CE: 0.065 → 0.084 — slight drift but still 97.9% test accuracy
- Hit rate: 38% → 39% — stable attention patterns
- Recode: 0.0045 → 0.0005 — improved

## Per-font breakdown

| Font | Letter | RR Letter | Notes |
|------|--------|-----------|-------|
| default | 94.2% | 67.3% | Training font |
| dejavu-sans | 98.1% | 61.5% | |
| dejavu-sans-bold | 98.1% | 67.3% | |
| dejavu-serif | 100.0% | **73.1%** | Best RR |
| dejavu-serif-bold | 94.2% | 53.8% | Worst RR |
| liberation-mono | 100.0% | 63.5% | |
| liberation-mono-bold | 100.0% | 69.2% | |
| liberation-narrow | 96.2% | 57.7% | |
| liberation-sans | 100.0% | 71.2% | |
| liberation-sans-bold | 98.1% | 61.5% | |
| liberation-serif | 98.1% | 59.6% | |

RR accuracy is surprisingly uniform across fonts (54-73%) — the motor decoder produces a single trajectory per letter regardless of input font, so re-read accuracy reflects how well that one trajectory matches each font's encoder representations.

## Architecture

Same as from-scratch motor model:
- **Scan**: 3 glimpses, 12x18 patches, prescribed x
- **Read**: 10 glimpses, 12x12 patches, free x/y
- **Motor decoder**: GRU-based, 32 trajectory points (x, y, pen_logit)
- **Soft render**: Gaussian blobs, sigma=1.5
- **Multi-head backward**: 4 separate passes (attn, cls, recon, motor)
- **Transfer**: 57 tensors from v5-scan (encoder, decoder, classifiers, scan_sensor, content_head)

## Training command

```bash
make train-motor DEVICE=cuda TRANSFER=runs/letters/v5-scan/model_final.pth.gz
```

## Losses

| Loss | Weight | Group | Purpose |
|------|--------|-------|---------|
| Classification CE | 1.0 | cls | Letter identity (26 classes) |
| Case CE | 1.0 | cls | Upper/lower |
| Reconstruction MSE | 1.0 | recon | Forces sufficient info gathering |
| Recode MSE | 1.0 | recon | Case-invariant latent |
| Scan attention guide | 8.0 | attn | Pulls scan y onto strokes |
| Fixation diversity | 1.0 | attn | Prevents fixation collapse |
| Content detection BCE | 0.5 | attn | Scan content awareness |
| Trajectory MSE | 1.0 × scaffold | motor | Shape guidance (anneals to 0.1) |
| Pen BCE | 1.0 × scaffold | motor | Pen up/down timing |
| Re-read CE | 1.0 | motor | Readability of rendered trace |

## What to try next (→ v2)

### Uppercase-only training
Simpler motor targets: straight lines, no ascenders/descenders, fewer confusable pairs. Isolates the motor learning signal. 26 letters × 1 font, ~100 epochs.

### Enhanced motor loss stack
The current re-read classification signal is too coarse. Proposed additions:

| Signal | Weight | Role |
|--------|--------|------|
| **Latent match** MSE(latent₂, latent₁) | 2.0 | Same abstract representation — impossible to cheat |
| Re-read classification (existing) | 1.0 | Proven signal |
| **Frozen re-reader** classification | 0.5 | Static encoder copy, no co-adaptation |
| **Render match** MSE(rendered, clean) | 0.25 | Soft pixel anchor |
| Trajectory scaffold (existing) | anneals | Initial shape guidance |

**Latent matching** forces the rendered image to produce the same internal representation as the original — a dense, continuous "how far off" signal. **Frozen re-reader** prevents the encoder from learning to read motor artifacts. **Render match** at low weight provides a gentle pixel-level reality check without constraining writing style.

### Sharper rendering
Reduce render sigma from 1.5 to ~0.75. Thinner lines make the task more honest — can't pass with vague spatial blobs. Similar to the blurred guide → prescanned guide evolution in the vision pipeline.

## Files in this archive

- `atlas.html` — post-transfer motor atlas (5 panels: input | recon | recode | trajectory | rendered)
- `from_scratch_atlas.html` — atlas from the initial from-scratch training run (not archived separately). This run trained vision+motor simultaneously without transfer — it showed vision learning first while motor lagged behind (84.6% letter vs 47.6% re-read), which motivated the curriculum learning approach. The atlas reveals clean recon/recode (vision solid) alongside crude motor traces (motor catching up), making the case for "read before write" — train vision first, then add motor pressure.
- `summary.txt` — vision test metrics
- `summary_motor.txt` — motor test metrics with per-letter re-read errors
- `training.log`, `training_metrics.png` — full training history
- `config.yaml`, `info.txt` — reproducibility
