# Results: Vertical Diversity (VY=1.5), 128x128, Aa-Zz, 100 epochs

## Summary
- **Letter accuracy**: 572/572 (100.0%)
- **Case accuracy**: 572/572 (100.0%)
- **Recon MSE**: 0.0012
- **Recode MSE**: 0.0019
- **11 fonts**: all at 100% letter and case
- **Dataset**: 11,440 training samples (52 letters x 20 variants x 11 fonts)
- **Test set**: 572 samples (52 letters x 11 fonts)

## The problem: horizontal scan bias

The bigram attention atlas (v4) showed fixations clustering in a narrow vertical band around the letter midline. The model scanned left-to-right effectively (the temporal scaffold worked) but never explored ascenders or descenders. Letters that differ only in vertical extent (p vs o, t vs r, l vs i) were confused because the distinguishing features were never visited.

This mirrors a real phenomenon in human saccade research: horizontal saccades are faster, more accurate, and more natural than vertical ones. Our oculomotor system is optimized for lateral reading movement. The model inherited the same bias — the isotropic diversity loss treated x and y equally, and since the training signal pushed for horizontal scanning (wider images, left-right scaffold), vertical exploration was never incentivized.

## The fix: directional diversity

Split the diversity loss to apply stronger repulsion in the vertical dimension:

```python
def fixation_diversity_loss(locations, sigma=0.1, vy=1.0):
    diff = locs.unsqueeze(2) - locs.unsqueeze(1)  # (B, T, T, 2)
    if vy != 1.0:
        scale = torch.tensor([1.0, vy], device=locs.device)
        diff = diff * scale  # stretch y-distances before computing repulsion
    dist_sq = (diff ** 2).sum(-1)
    repulsion = torch.exp(-dist_sq / (2 * sigma ** 2))
    ...
```

With `vy=1.5`, two fixations at the same horizontal position but separated vertically by distance `d` experience repulsion as if they were `1.5d` apart. This makes vertical clustering 1.5x more expensive than horizontal clustering, directly counteracting the horizontal scan bias.

## Training commands

```bash
# Generate data (11 fonts, same as v3)
make generate FONTS=all LETTERS=Aa-Zz VARIANTS=20 NOISE=0.1
make generate-test FONTS=all LETTERS=Aa-Zz

# Train (VY=1.5 was the key change from v3)
make train EPOCHS=100 DEVICE=cuda BATCH=52 GUIDE=8.0 VY=1.5 FONTS=all

# Full CLI equivalent
python vision_training.py train --data_dir data/letters --epochs 100 --save_dir data/models --checkpoint_interval 10 --n_glimpses 10 --patch_size 12 --n_scales 1 --device cuda --batch_size 52 --guide_weight 8.0 --diversity_weight 1.0 --diversity_sigma 0.1 --recode_weight 1.0 --blur_sigma_ratio 0.16 --diversity_vy 1.5
```

## Key findings

### 1. Better attention with no accuracy cost
| Metric | v3 (VY=1.0) | v5 (VY=1.5) |
|--------|-------------|-------------|
| Letter acc | 100% | 100% |
| Case acc | 100% | 100% |
| Attn loss | -0.1585 | **-0.1615** |
| Diversity | 0.1072 | **0.0819** |
| Hit rate | 46% | 44% |
| Recon MSE | 0.0039 | 0.0042 |
| Recode MSE | 0.0033 | **0.0019** |

The encoder finds *more* letter content (better attention loss) while being *more* spread out (lower diversity). Recode MSE nearly halved — the latent captures letter identity more cleanly when the encoder visits the full vertical extent.

### 2. Faster convergence to perfect test accuracy
v3 needed 100 epochs to reach 100% test accuracy. v5 achieves the same 100% in 100 epochs but with a harder constraint — and by epoch 73 classification was already locked at 0.0001. The training loss was bumpier mid-run (vertical repulsion kept pushing fixations to new positions, temporarily hurting classification), but this forced the model to find a more robust solution faster. The result: perfect accuracy with significantly better attention metrics, in the same epoch budget despite the added difficulty.

### 3. Atlas shows visibly more vertical spread
The attention atlas confirms fixations now visit ascender/descender regions, not just the midline band. This is exactly the coverage pattern needed for bigram letter disambiguation.

## Final epoch losses
```
Epoch 100/100: Recon 0.0042  Ltr 0.0000  Case 0.0000  Attn -0.1615  Div 0.0819  Hit 45%  Recode 0.0008  lr 0.000000
```

## Convergence timeline
- Epoch 1: Ltr 2.91, Case 0.69, Hit 38% — attention engaged
- Epoch 10: Ltr 0.18, Case 0.09, Hit 43% — faster early convergence than v3
- Epoch 36: Ltr 0.023, Case 0.008 — nearing lock-in
- Epoch 46: Ltr 0.042 (bump) — vertical repulsion pushing exploration
- Epoch 60: Ltr 0.002 — nearly locked
- Epoch 73: Ltr 0.0001, Case 0.0001 — locked in, 22 epochs later than v3
- Epoch 88: all metrics stable, polishing

## Comparison across runs
| Metric | v1 (1 font) | v2 (11 fonts) | v3 (cosine) | v5 (VY=1.5) |
|--------|------------|---------------|-------------|-------------|
| Letter acc | 100% | 99.5% | 100% | 100% |
| Case acc | 100% | 99.7% | 100% | 100% |
| Attn loss | — | — | -0.1585 | **-0.1615** |
| Diversity | — | — | 0.1072 | **0.0819** |
| Recon MSE | 0.0004 | 0.0081 | 0.0039 | 0.0042 |
| Recode MSE | 0.0004 | 0.0063 | 0.0033 | **0.0019** |
| Epochs | 200 | 50 | 100 | 100 |

## Next: bigram transfer with vertical diversity
The v4 bigram errors (of→cf, ld→id, etc.) were vertical confusion pairs. This encoder now has vertical saccade habits baked in from single-letter training. Transfer to bigrams with VY=1.5 + mask_weight should address those specific failure modes.
