# Results: Multi-Font, 128x128, Aa-Zz, 50 epochs

**Archived model**: [`runs/v2-multi-font/`](../runs/v2-multi-font/)

## Summary
- **Letter accuracy**: 569/572 (99.5%)
- **Case accuracy**: 570/572 (99.7%)
- **11 fonts**: default, dejavu-serif, dejavu-serif-bold, dejavu-sans, dejavu-sans-bold, liberation-serif, liberation-sans, liberation-sans-bold, liberation-mono, liberation-mono-bold, liberation-narrow
- **Training time**: ~41 min on GPU (50 epochs, ~49.5s/epoch)
- **Dataset**: 11,440 training samples (52 letters x 20 variants x 11 fonts)
- **Test set**: 572 samples (52 letters x 11 fonts)

## Final epoch losses
```
Epoch 50/50: Recon 0.0081  Ltr 0.0350  Case 0.0139  Attn -0.1558  Div 0.1033  Hit 44%  Recode 0.0063
```

## Per-font breakdown
```
default                 : Letter 100.0%  Case 100.0%  (52 samples)
dejavu-sans             : Letter 100.0%  Case 100.0%  (52 samples)
dejavu-sans-bold        : Letter 100.0%  Case 100.0%  (52 samples)
dejavu-serif            : Letter 100.0%  Case 100.0%  (52 samples)
dejavu-serif-bold       : Letter  98.1%  Case 100.0%  (52 samples)
liberation-mono         : Letter 100.0%  Case 100.0%  (52 samples)
liberation-mono-bold    : Letter 100.0%  Case  98.1%  (52 samples)
liberation-narrow       : Letter 100.0%  Case 100.0%  (52 samples)
liberation-sans         : Letter 100.0%  Case 100.0%  (52 samples)
liberation-sans-bold    : Letter 100.0%  Case  98.1%  (52 samples)
liberation-serif        : Letter 100.0%  Case 100.0%  (52 samples)
```

8/11 fonts at 100% on both metrics. 3 single-sample errors, all on bold variants.

## Key finding: guide_weight must scale with data diversity

**guide_weight=4.0 failed.** The first attempt used guide_weight=4.0 (proven for single-font). Attention engaged initially (hit 36-39% at epoch 1-3) but collapsed by epoch 4 (hit 10-12%). The model recovered classification by routing around attention — the 33.7M param decoder compensated without good fixations. This is the same failure mode seen when scaling from 96x96 to 128x128: the decoder is powerful enough to ignore attention when the guide pressure is too low.

**guide_weight=8.0 fixed it.** Hit rate locked at 40-44% from epoch 1 onward, never dipped. The higher weight makes it too costly for the optimizer to sacrifice attention quality, even when the decoder could technically compensate.

**Pattern:** every time we increase the task complexity (image size, font diversity), the decoder's ability to route around attention grows. guide_weight must increase to compensate. The ratio so far:
- Single font, 96x96: guide_weight=2.0
- Single font, 128x128: guide_weight=4.0
- 11 fonts, 128x128: guide_weight=8.0

## Convergence timeline
- Epoch 1: Ltr 2.89 (random=3.26), Case 0.68, Hit 40% — attention engaged immediately
- Epoch 4: Ltr 0.71, Case 0.36, Hit 42% — past the danger zone where guide=4.0 collapsed
- Epoch 10: Ltr 0.25, Case 0.10, Hit 42% — solid convergence
- Epoch 32: Ltr 0.048, Case 0.030, Hit 43% — near-converged
- Epoch 50: Ltr 0.035, Case 0.014, Hit 44% — still improving slowly

## Comparison to single-font (v1)
| Metric | v1 (1 font, 200ep) | v2 (11 fonts, 50ep) |
|--------|-------------------|---------------------|
| Letter acc | 100% (52/52) | 99.5% (569/572) |
| Case acc | 100% (52/52) | 99.7% (570/572) |
| Recon MSE | 0.0004 | 0.0081 |
| Recode MSE | 0.0004 | 0.0063 |
| Hit rate | 35% | 44% |
| guide_weight | 4.0 | 8.0 |
| Training time | 19 min | 41 min |

## What to try next
- **Resume for more epochs**: losses still decreasing at epoch 50. The 3 bold-font errors will likely resolve with more training. `make resume DEVICE=cuda EPOCHS=150 GUIDE=8.0`
- **Bigrams**: the attention mechanism generalizes across font styles — ready to test multi-character sequences
