# Results: 128x128, Aa-Zz, 200 epochs

## Summary
- **Letter accuracy**: 52/52 (100%)
- **Case accuracy**: 52/52 (100%)
- **Reconstruction MSE**: 0.0004
- **Recode MSE**: 0.0004
- **Hit rate**: ~35% (diagnostic)
- **Training time**: ~19 min on GPU (200 epochs, ~5.7s/epoch)

## Training commands

```bash
# Generate data
make generate FONTS=default LETTERS=Aa-Zz VARIANTS=20 NOISE=0.1
make generate-test FONTS=default LETTERS=Aa-Zz

# Train
make train EPOCHS=200 DEVICE=cuda BATCH=26 GUIDE=8.0 FONTS=default

# Full CLI equivalent
python vision_training.py generate --letters Aa-Zz --num_variants 20 --noise_level 0.1 --output_dir data/letters --fonts default
python vision_training.py train --data_dir data/letters --epochs 200 --save_dir data/letter_models --checkpoint_interval 10 --n_glimpses 10 --patch_size 12 --n_scales 1 --device cuda --batch_size 26 --guide_weight 8.0 --diversity_weight 1.0 --diversity_sigma 0.1 --recode_weight 1.0 --blur_sigma_ratio 0.16 --diversity_vy 1.0
```

## Final epoch losses
```
Epoch 200/200: Recon 0.0035  Ltr 0.0001  Case 0.0001  Attn -0.1205  Div 0.0545  Hit 36%  Recode 0.0001
```

## Convergence timeline
- Epoch 5: Ltr 2.86 (random=3.26), Case 0.70 (random=0.69), Hit 28% — attention engaged from start
- Epoch 47: Ltr 0.13, Case 0.04 — past the knee
- Epoch 100: Ltr ~0.05, Case ~0.01 — near-converged (previous 100-epoch run: 94% letter, 100% case)
- Epoch 138: Ltr 0.17, Case 0.03, Recode 0.0018 — recode nearly perfect
- Epoch 179: Ltr 0.0001, Case 0.0001, Recode 0.0001 — fully converged
- Model was essentially perfect by ~180 epochs

## Attention patterns
- **A**: circles apex and crossbar area, hits left leg — structural scan of the triangle+crossbar
- **a**: traces around the bowl and stem — different strategy from A, adapted to lowercase form
- **T/t**: share a similar top-down strategy but t wraps tighter around the smaller crossbar
- **O/o**: both explore ring interior but with scale-appropriate paths
- Each letter has a distinct scan path; upper and lower case of same letter share similar overall trajectory but adapted to the specific glyph shape
- Hit rate ~30-50% per letter — selective, not exhaustive

## Recode outputs
- `recode_a_to_A`: clean capital A, indistinguishable from ground truth
- `recode_A_to_a`: clean lowercase a
- `recode_t_to_T`: sharp uppercase T
- `recode_g_to_G`: clean uppercase G
- All recode outputs are crisp — the latent space has fully factored letter identity from case
- MSE 0.0004 means the decoder produces nearly pixel-perfect case flips

## Key finding: latent factorization works
The recode loss converging to 0.0001 proves the latent space captures abstract letter identity independently of visual form. The same 256-dim latent decoded with case=0.0 produces 'A' and with case=1.0 produces 'a'. This validates the feedback recursive loop as a mechanism for learning disentangled representations.

## Hyperparameters that matter
- **guide_weight=4.0**: 2.0 was too weak at 128x128, caused attention collapse (0% hit rate)
- **blur_sigma_ratio=0.16**: auto-scales with image size. Equivalent to 20.5px at 128x128, 15px at 96x96
- **recode_weight=1.0**: no special tuning needed
- **diversity_sigma=0.1**: same as before, works at any image size (normalized coords)

## What broke along the way
1. Scaling from 96x96 to 128x128 with absolute blur_sigma=15 broke attention completely — fixations ran a fixed trajectory ignoring letters (0% hit rate). Fixed by switching to ratio-based sigma.
2. Decoder FC layer at 128x128 is 33.7M params (vs 18.9M at 96x96) — powerful enough to reconstruct from garbage latents, reducing gradient pressure on attention. Stronger guide_weight (4.0 vs 2.0) compensated.
