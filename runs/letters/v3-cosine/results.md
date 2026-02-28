# Results: Multi-Font + CosineAnnealingLR, 128x128, Aa-Zz, 100 epochs

## Summary
- **Letter accuracy**: 572/572 (100.0%)
- **Case accuracy**: 572/572 (100.0%)
- **Recon MSE**: 0.0012
- **Recode MSE**: 0.0033
- **11 fonts**: all at 100% letter and case
- **Training time**: ~80 min on GPU (100 epochs, ~48s/epoch)
- **Dataset**: 11,440 training samples (52 letters x 20 variants x 11 fonts)
- **Test set**: 572 samples (52 letters x 11 fonts)

## The cosine fix

A previous 100-epoch run with constant lr=0.001 hit catastrophic divergence at epoch 43: attention collapsed (hit rate → 0%), all losses spiked, and the model never recovered. Root cause: constant learning rate too aggressive once the attention controller, decoder, and classifier reached a tightly coupled equilibrium — a large gradient step destabilized one component, cascading to all others.

**Fix**: CosineAnnealingLR decays lr smoothly from 0.001 → 0 over the run. By epoch 43, lr was 0.000624 — gentle enough to keep the equilibrium stable. A minor perturbation at epoch 62 (Ltr jumped 0.0001 → 0.043) self-corrected within 2 epochs at lr=0.00033.

## Training commands

```bash
# Generate data (11 fonts, same as v2)
make generate FONTS=all LETTERS=Aa-Zz VARIANTS=20 NOISE=0.1
make generate-test FONTS=all LETTERS=Aa-Zz

# Train (CosineAnnealingLR was the key change from v2)
make train EPOCHS=100 DEVICE=cuda BATCH=52 GUIDE=8.0 FONTS=all

# Full CLI equivalent
python vision_training.py train --data_dir data/letters --epochs 100 --save_dir data/models --checkpoint_interval 10 --n_glimpses 10 --patch_size 12 --n_scales 1 --device cuda --batch_size 52 --guide_weight 8.0 --diversity_weight 1.0 --diversity_sigma 0.1 --recode_weight 1.0 --blur_sigma_ratio 0.16 --diversity_vy 1.0
```

## Final epoch losses
```
Epoch 100/100: Recon 0.0039  Ltr 0.0000  Case 0.0000  Attn -0.1582  Div 0.1053  Hit 46%  Recode 0.0004  lr 0.000000
```

## Per-font breakdown
```
default                 : Letter 100.0%  Case 100.0%  (52 samples)
dejavu-sans             : Letter 100.0%  Case 100.0%  (52 samples)
dejavu-sans-bold        : Letter 100.0%  Case 100.0%  (52 samples)
dejavu-serif            : Letter 100.0%  Case 100.0%  (52 samples)
dejavu-serif-bold       : Letter 100.0%  Case 100.0%  (52 samples)
liberation-mono         : Letter 100.0%  Case 100.0%  (52 samples)
liberation-mono-bold    : Letter 100.0%  Case 100.0%  (52 samples)
liberation-narrow       : Letter 100.0%  Case 100.0%  (52 samples)
liberation-sans         : Letter 100.0%  Case 100.0%  (52 samples)
liberation-sans-bold    : Letter 100.0%  Case 100.0%  (52 samples)
liberation-serif        : Letter 100.0%  Case 100.0%  (52 samples)
```

## Convergence timeline
- Epoch 1: Ltr 2.83, Case 0.67, Hit 40% — attention engaged immediately
- Epoch 10: Ltr 0.29, Case 0.16, Hit 41% — solid convergence
- Epoch 21: Ltr 0.086, Case 0.053, Hit 42% — nearing convergence
- Epoch 43: Ltr 0.026, Case 0.019, Hit 44% — **sailed through the old crash point**
- Epoch 55: Ltr 0.0001, Case 0.0000, Hit 45% — effectively converged
- Epoch 62: Ltr 0.043 (blip), recovered by epoch 64
- Epoch 100: Ltr 0.0000, Case 0.0000, Hit 46% — perfect

## Comparison across runs
| Metric | v1 (1 font, 200ep) | v2 (11 fonts, 50ep) | v3 (11 fonts, 100ep, cosine) |
|--------|-------------------|---------------------|------------------------------|
| Letter acc | 100% (52/52) | 99.5% (569/572) | 100% (572/572) |
| Case acc | 100% (52/52) | 99.7% (570/572) | 100% (572/572) |
| Recon MSE | 0.0004 | 0.0081 | 0.0012 |
| Recode MSE | 0.0004 | 0.0063 | 0.0033 |
| Hit rate | 35% | 44% | 46% |
| guide_weight | 4.0 | 8.0 | 8.0 |
| Scheduler | constant | constant | CosineAnnealingLR |

## What to try next
- **Attention atlas**: `make atlas` → `data/atlas.html` — explore per-font fixation strategies
- **Bigrams**: the attention mechanism generalizes across 11 font styles with perfect accuracy — ready to test multi-character sequences
