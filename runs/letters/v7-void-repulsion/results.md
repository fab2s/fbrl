# v7 — Void Repulsion

## Config
- **Architecture**: 1 scan (12×18) + 6 read (12×12) = 7 glimpses
- **Scan**: blurred guide (weight 8.0), learnable_scan_x, void repulsion (weight 1.5)
- **Reads**: NO blurred guide (weight 0.0), void repulsion (weight 0.5)
- **Diversity**: isotropic (vy=1.0), sigma=0.1, weight=1.0
- **Training**: 100 epochs, batch 52, cosine LR, 11 fonts

## Results
- **Test accuracy**: 100% letter, 100% case (572/572)
- **Recode**: 0.0005 (best yet — better latent factorization than v6's 0.0006)
- **Training time**: 82 minutes (~50s/epoch)

## What changed from v6
| | v6 (1+7=8) | v7 (1+6=7) |
|---|---|---|
| Read guide | blurred (8.0) | **none** (0.0) |
| Void repulsion | — | reads 0.5, scan 1.5 |
| diversity_vy | 1.5 | **1.0** (isotropic) |
| Read glimpses | 7 | **6** |
| Total glimpses | 8 | **7** (46% fewer than v5's 13) |

## Void repulsion design
Replaces both the blurred attention guide (for reads) and ink_reward:
- **Full patch sampling** (12×12 or 12×18) at each fixation via grid_sample
- **Saturating penalty**: `clamp(patch_max / 0.1, max=1.0)` — gradient active only when patch is pure void, zero once any ink pixel found
- Returns -1.0 (all patches on ink) to 0.0 (all patches in void)
- "Don't stare at nothing" rather than "find the most ink"

### Key property: local-only gradient
Unlike the blurred guide which creates a global gradient field, void repulsion has **zero gradient when a patch is entirely in void** far from ink. The gradient only activates at ink boundaries via bilinear interpolation. This is a feature for reads (no center-pull) but a limitation for scan (can't recover from void).

## Experiments and findings

### Run 1: void repulsion everywhere (no blurred guide at all)
- `scan_guide_weight: 0.0`, `guide_weight: 0.0`
- `void_weight: 0.5`, `scan_void_weight: 1.5`
- `diversity_vy: 1.5`, `n_read_glimpses: 7`
- **Result**: 100%/100% BUT scan fixation always in lower-left void
- **Root cause**: NOT diversity pushing scan away (scan/read diversity is computed separately). The scan drifted to void because: (1) reads already solved classification → zero gradient for scan, (2) void repulsion has no long-range signal to pull it back, (3) no blurred guide to provide global direction
- **Effective**: only 6 reads doing the work, scan wasted

### Run 1 attention analysis
- Read fixations spread too far vertically — diversity_vy=1.5 forces excessive Y spread
- The Y spread was originally needed to counteract the blurred guide's center-pull
- Without guide center-pull, vy>1.0 is counterproductive
- Despite this, classification perfect — model reads via peripheral patch vision (12×12 patches see ink even when center isn't on stroke)

### Run 2: blurred guide for scan, isotropic diversity, 6 reads
- `scan_guide_weight: 8.0`, `guide_weight: 0.0`
- `diversity_vy: 1.0`, `n_read_glimpses: 6`
- **Result**: 100%/100%, recode 0.0005 (slightly better than run 1)
- Scan now lands on ink consistently (attn = -1.417)
- Diversity half of run 1 (0.003 vs 0.007) — tighter, more natural clustering
- Attention paths show reads exploring letter features meaningfully
- Cross-font consistency: same letter → similar fixation strategy across fonts (visible for 'a')

## Self-scaffolding discovery
Classification (letter + case) converges to 0.000 by ~epoch 57, while recode is still at 0.0014 and actively dropping. The loss terms naturally sequence by difficulty:
1. **Classification** (easiest, 26 bins) → converges first, locks in attention patterns
2. **Reconstruction** (pixel-level) → converges alongside
3. **Recode** (hardest, requires identity/case factorization) → takes over as dominant gradient

No curriculum learning, no staged training — the loss landscape self-organizes. The encode-decode-recode loop acts as a self-scaffolding system, analogous to how brains learn from the same sensory stream with easy patterns consolidating before hard ones.

## Visualization fix
Discovered that `locations[-1]` is a vestigial GRU prediction ("where to look next" after the last glimpse) that is never actually sampled. Previously rendered in attention plots as a ghost point always drifting to void. Now excluded from all visualization (attention PNGs + atlas HTML).

## Remaining observations
- **P vs R**: no fixation lands on R's distinguishing leg. Works with 11 fonts but may need 7 reads when scaling to more fonts where bowl shapes vary.
- **Hit rate paradox**: 22% hit rate (center-point metric) despite perfect classification. The model reads via peripheral patch vision — the 12×12 patch sees ink even when its center isn't on a stroke. Hit rate is a misleading diagnostic for this architecture.
- **Scan content head**: converges to ~0.015 BCE. Useful training signal but not critical — the blurred guide does the heavy lifting for scan positioning.
