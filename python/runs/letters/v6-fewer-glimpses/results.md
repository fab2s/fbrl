# Results: v6 — Fewer Glimpses with Learnable Scan Positions

## Summary

Three experiments testing how lean the attention budget can get while maintaining 100% accuracy.

| Config | Glimpses | Letter | Case | Recode MSE | Notes |
|--------|----------|--------|------|------------|-------|
| v5-scan (baseline) | 3+10 = 13 | 100% | 100% | 0.0007 | Fixed scan x at linspace |
| 3 scan + 7 read | 3+7 = 10 | **100%** | **100%** | 0.0010 | Learnable scan x |
| 1 scan + 7 read | 1+7 = 8 | **100%** | **100%** | 0.0010 | Minimal scan, learnable x |
| Anchored read | 3+7 = 10 | 48.6% | — | — | FAILED: catastrophic overfitting |

**Best result**: 100%/100% with only 8 glimpses — 38% fewer than v5-scan, zero accuracy loss.

## Experiment 1: Anchored read (FAILED)

3 scan + 7 read, with read position reset to middle scan's (x, y) location. Learnable scan x positions.

- Training: 0.0000 letter CE, 0.0000 case CE — perfect
- Test: **48.6% letter accuracy** — catastrophic overfitting
- Scan positions collapsed, reads clustered too narrowly around anchor point
- Root cause: position reset breaks GRU spatial continuity. The hidden state carries context but the imposed location creates a discontinuity the model can't recover from on unseen data.
- X-only anchoring was considered but abandoned: scan y is already near 0 (scan_vy=0.3), so x-only vs full reset is negligible. The problem is the reset itself.

**Lesson**: never reset the read position. Flat continuation from the last scan preserves momentum.

## Experiment 2: 3 scan + 7 flat read (10 total)

Learnable scan x, flat read continuing from last scan position.

- **Letter**: 572/572 (100%) across all 11 fonts
- **Case**: 572/572 (100%)
- **Recon MSE**: 0.0006
- **Recode MSE**: 0.0010
- Learnable scan x drifted closer to letter content vs v5's fixed linspace(-0.75, 0.75)
- 23% fewer glimpses than v5, same accuracy
- Atlas: `atlas_3scan_7read.html`

## Experiment 3: 1 scan + 7 flat read (8 total)

Single learnable scan position, flat read.

- **Letter**: 572/572 (100%) across all 11 fonts
- **Case**: 572/572 (100%)
- **Recon MSE**: 0.0005
- **Recode MSE**: 0.0010
- 38% fewer glimpses than v5, same accuracy
- Recode quality slightly degraded compared to v5 — the compression edge
- Atlas: `atlas_1scan_7read.html`

## Key insights

1. **Anchoring kills generalization**: GRU spatial continuity is essential. Flat read (continuing from last scan) works; position reset (anchoring) causes overfitting.

2. **Learnable scan x > prescribed**: The model finds better horizontal positions than evenly-spaced linspace. Scans drift onto the letter content rather than wasting time on empty margins.

3. **v5's edge scans were wasted**: With 3 fixed scans at [-0.75, 0, 0.75] on centered 128x128 images, the left and right scans looked at mostly empty space. 1 learnable scan does better.

4. **Recode is the honest metric**: Classification saturates (100%) before reconstruction quality. Recode MSE reveals latent richness — with fewer glimpses the latent captures enough for "what letter" but less of "how exactly shaped."

5. **1+7 is the atomic unit**: One wide scan + seven narrow reads per letter position. This tiles directly to an interleaved word architecture: scan1 → reads → scan2 → reads → ... → scanN (end boundary).

6. **Practical design**: For OCR, the encoder is the product. Decoder, recode, and motor are training scaffolding. Inference = encoder + linear head → UTF-8.

## Training dynamics (1 scan + 7 read)

- Epoch 5: ltr 0.41 — fast initial drop
- Epoch 20: ltr 0.05 — approaching zero
- Epoch 48: ltr 0.0002 — essentially zero
- Epoch 67: ltr 0.0000 — locked in
- Epoch 100: recon 0.0038, recode 0.0004, attn -2.727
- ~49s/epoch, 100 epochs total

## Architecture

- **Scan**: 1 glimpse, 12x18 patches, learnable x, learned y
- **Read**: 7 glimpses, 12x12 patches, free x/y, flat continuation from scan
- **Config field**: `learnable_scan_x: true` (decoupled from read anchoring)
- **No anchoring, no grouped read**

## Files in this archive

- `atlas.html` — final atlas (1 scan + 7 read)
- `atlas_3scan_7read.html` — experiment 2 atlas (3 scan + 7 read)
- `config.yaml` — full expanded config (1 scan + 7 read)
- `model_final.pth.gz` — compressed model checkpoint
- `training.log` — training log (1 scan + 7 read)
- `training_metrics.png` — loss curves
- `summary.txt` — test results (3 scan + 7 read)
- `summary_1scan_7read.txt` — test results (1 scan + 7 read)
- `info.txt` — git hash and date
