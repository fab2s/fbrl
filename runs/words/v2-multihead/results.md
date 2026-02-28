# Results: Word Multi-Head Optimizer, 128px Isolation, 200 epochs

## Summary
- **Word accuracy**: 199/200 (99.5%) — 1 error: `game` → `gamd` (P4: e→d)
- **Isolation accuracy**: 144/286 (50.3% any-position) across 11 fonts
  - Default font: 100% | liberation-sans: 73.1% | dejavu-sans: 61.5%
  - Bold fonts worst: dejavu-sans-bold 26.9%, liberation-mono-bold 23.1%
- **Recon MSE**: 0.0014 (word test), 0.0034 (final training)
- **Training time**: 200 epochs, ~300s/epoch (728 min total)
- **Dataset**: 4,000 word samples (200 words × 20 variants × 1 font, default)
- **Isolation data**: 128x128 single-letter images (default font only)
- **Transfer from**: single-letter model (`data/models/model_final.pth`)

## What changed from v1

### Multi-head optimizer (3 separate backward passes)
Split `total_loss.backward()` into 3 targeted passes using `backward(inputs=...)`:

| Group | Components | Losses |
|-------|-----------|--------|
| Attention | controller, scan_sensor, read_sensor, content_head | scan_guide + read_scaffold + diversity + content |
| Classification | readout, classifiers | CE × 4 positions + isolation CE |
| Reconstruction | decoder | MSE reconstruction |

Key: classification gradients do NOT flow back to the controller. Each system learns from its own feedback signal only.

### 128x128 single-letter isolation (replacing mask-based)
V1 used canvas masking (zero 3 of 4 stripes) — never converged (~1.0 CE). V2 feeds actual 128x128 single-letter images through the word model (3 dynamic scan + 12 read). Isolation CE dropped to 0.008 — massively better.

### Dynamic scan count
Scan glimpses scale with image width: `n_scan = max(1, round(8 * (W/256)^1.5))`. Gives 3 scans for 128px isolation, 8 for 256px words. Read stays fixed at 12.

### Deferred isolation forward
In multi-head mode, isolation forward pass runs AFTER `recon_loss.backward()` frees the main computation graph — prevents two graphs coexisting in VRAM.

## Training dynamics

### Convergence timeline
- Epoch 1: P1-P4 ~2.8-3.2 CE, isolation 2.14, recon 0.044
- Epoch 50: P1 0.83, P2 0.73, P3 0.74, P4 0.57, isolation 0.18, scaffold 0.63
- Epoch 100: P1 0.21, P2 0.13, P3 0.19, P4 0.15, isolation 0.04, scaffold 0.26
- Epoch 134: scaffold reaches ~0 (last scaffold epoch)
- Epoch 135: **unfreeze spike** — P1 jumps 0.08→0.44, all positions spike. 3 LRs split: attn=0.0001, cls=0.001, recon=0.001
- Epoch 150: recovered — P1 0.08, P2 0.06, P3 0.07, P4 0.06
- Epoch 200: all positions < 0.003 CE, isolation 0.008

### Post-unfreeze memory spike
GPU memory jumped from ~6GB to 10GB at epoch 135 when read_sensor + classifiers unfroze — optimizer states for newly-trainable parameters allocated. Caused ~60s/epoch penalty from shared memory spill (dedicated VRAM is 6GB on GTX 1080 Ti).

### P4 no longer hardest
V1 showed P4 (rightmost) as the hardest position. V2 final: P4=0.0018 CE is actually the *best* position. Multi-head training may have resolved the edge bias by giving classification its own clean gradient path.

## Isolation analysis

Default font: 100% — isolation training works perfectly for the trained font. Cross-font generalization drops sharply because:
1. Word training uses only default font
2. Isolation training also uses only default font letter images
3. The pretrained single-letter model's multi-font knowledge was partially overwritten during word fine-tuning

Bold fonts are worst (~23-31%) — their thick strokes differ most from default. Thin sans-serif fonts (liberation-sans, dejavu-sans) fare better (~62-73%).

Confusions show systematic patterns: bold fonts trigger `n` predictions (bold vertical strokes resemble 'n'), thin letters (i, l, j) confuse with each other.

## Attention patterns

- **Words**: scan sweeps L→R correctly. Read glimpses cluster in a tighter horizontal band than v1 — less vertical spread, less diagonal exploration. Multi-head may have reduced read diversity since diversity loss only flows to controller.
- **Isolation** (128px): 16 fixations (1 initial + 3 scan + 12 read). Scan fixation 3 flies to x=0.75 (right edge of canvas where there's nothing). Read glimpses cluster tightly around the letter center.

## Training commands

```bash
make train-words EPOCHS=200 DEVICE=cuda BATCH=52 GUIDE=8.0 \
    SCAFFOLD_RATIO=0.67 CONTENT=0.5 ISOLATION=0.5 \
    SCAN_VY=0.3 READ_VY=1.5 \
    TRANSFER=data/models/model_final.pth \
    ISOLATION_DATA=data/letters MULTI_HEAD=1
```

## Comparison across word runs

| Metric | v1-word (single opt) | v2-word (multi-head) |
|--------|---------------------|---------------------|
| Word accuracy | 200/200 (100%) | 199/200 (99.5%) |
| Recon MSE | 0.0015 | 0.0014 |
| Isolation CE (final) | ~1.0 (mask, never converged) | 0.008 (128px images) |
| Isolation test (11 fonts) | not tested | 50.3% any-correct |
| Time/epoch | ~135s | ~300s (shared mem spill) |
| Checkpoint size | ~1GB | ~3GB (3 optimizer states) |
| P4 difficulty | Hardest position | Best position |

## Losses used

| Loss | Weight | Group | Purpose |
|------|--------|-------|---------|
| Classification CE | 1.0 ×4 | cls | Per-position letter identity |
| Reconstruction MSE | 1.0 | recon | Forces sufficient info gathering |
| Scan attention guide | 8.0 | attn | Pulls scan y onto letter strokes |
| Read temporal scaffold | 8.0→0 | attn | L→R reading order, anneals |
| Fixation diversity | 1.0 | attn | Prevents fixation collapse |
| Content detection BCE | 0.5 | attn | Scan content awareness |
| Isolation CE | 0.5 | cls | Per-position reading (128px images) |

## What to try next (→ v3)
- **Wider read patches (18x18)**: capture cross-letter context, better single-letter coverage
- **Multi-font isolation (3 fonts)**: teach font-invariant reading through isolation training
- **AMP**: halve VRAM usage, eliminate shared memory spill
- **Weights-only checkpoint**: avoid 3GB archives
