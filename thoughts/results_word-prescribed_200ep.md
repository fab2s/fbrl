# Results: Word Prescribed X-Scan, 256x128, 200 words, 200 epochs

**Archived model**: [`runs/v1-word-prescribed/`](../runs/v1-word-prescribed/)

## Summary
- **All-correct accuracy**: 200/200 (100.0%)
- **Pos 1 accuracy**: 200/200 (100.0%)
- **Pos 2 accuracy**: 200/200 (100.0%)
- **Pos 3 accuracy**: 200/200 (100.0%)
- **Pos 4 accuracy**: 200/200 (100.0%)
- **Recon MSE**: 0.0015
- **Training time**: 200 epochs, ~135s/epoch (27,000s total)
- **Dataset**: 4,000 training samples (200 words x 20 variants x 1 font)
- **Test set**: 200 samples (200 words x 1 font, clean)
- **Transfer from**: single-letter model (`data/models/model_final.pth`)

## Architecture: prescribed x-scan + free read

The WordVisionModel introduces a key constraint: scan-phase x-coordinates are **prescribed** as a fixed left-to-right linear sweep `torch.linspace(-0.75, 0.75, 8)`. Only y is learned during scan. The read phase (12 glimpses) is fully free.

This reflects how human reading proceeds roughly left-to-right — the model doesn't need to discover horizontal scanning order, only where to look vertically during scan and where to fixate precisely during read.

**Two-phase attention:**
1. **Scan** (8 glimpses, 12x18 wide patches): prescribed x sweep, learned y, content detection head
2. **Read** (12 glimpses, 12x12 focused patches): both x,y fully free, temporal scaffold guides left-to-right reading order

**Content detection head**: `nn.Linear(256, 1)` on each scan hidden state. BCE loss against sampled clean image intensity at the fixation location. Teaches the GRU to encode "letter content found here" in its hidden state.

**Isolation mask loss**: Random 1-of-4 letter position exposed per batch, other 3 zeroed out. Second forward pass, CE on exposed position only. Weight=0.5.

## Training dynamics

### Convergence timeline
- Epoch 1: P1-P4 ~3.2 CE (random), hit rate 33% (transfer engaged immediately)
- Epoch 50: P1 0.20, P2 0.08, P3 0.12, P4 0.35 — scaffold at ~0.63
- Epoch 100: P1 0.03, P2 0.01, P3 0.02, P4 0.05 — scaffold at ~0.25
- Epoch 134: scaffold reaches 0 (scaffold_ratio=0.67, scaffold_floor=0)
- Epoch 200: all positions < 0.002 CE — fully converged

### Position learning order
P2 converged fastest, then P3, then P1, then P4. Interior positions (P2, P3) are easier — they're surrounded by other letters and have more contextual cues. Edge positions (P1, P4) take longer, with P4 (rightmost) being the hardest. This mirrors human reading difficulty at word boundaries.

### Isolation loss behavior
The isolation mask loss stayed at ~1.0-1.3 CE throughout training and never truly converged. This is expected — the task is extremely hard: find and read a single letter on a mostly-empty 256x128 canvas through a 12x12 foveal window. The model must first locate the visible stripe, then fixate accurately within it.

**Key insight**: isolation masking tests "can you find AND read?" on a sparse canvas. A cleaner test of per-position reading would use actual single-letter images (128x128), which tests only "can you read?" This is a promising direction for future work.

### Hit rate
Stable at 33-34% throughout training. This is lower than bigram (~40%) but expected — the canvas is 2x wider (256 vs 128px) so random-equivalent hit rate is lower. The model samples diagnostic features rather than tracing letter outlines.

## GPU performance observations

- **Cold start**: ~180s/epoch (first training attempt)
- **Warm restart**: ~135s/epoch (consistent throughout 200 epochs after restart)
- **GPU utilization**: 98-99% (vs ~80% for bigram/letter — the wider canvas fully saturates the GPU)
- **GPU temperature**: 57°C (vs ~64°C on previous runs) — lower temp despite higher utilization
- **Possible explanation**: GPU memory/cache warming effects, or thermal management differences at sustained high load

## Attention patterns (from atlas)

- **Scan phase**: clean left-to-right sweep (prescribed), y tracks vertical letter center
- **Read phase**: fixations cluster on individual letter positions, roughly 3 glimpses per letter
- **Some words show potential shortcuts**: e.g., "gulf" — the fixation trajectory appears to skip the top of 'f', suggesting the model reads enough of the stroke to classify without full coverage. This isn't necessarily wrong (humans do this too) but worth watching in multi-font scenarios.

## Comparison across runs

| Metric | v5 (1-letter, 11 fonts) | v4 (bigrams, 300ep) | v1-word (4-letter, 200ep) |
|--------|------------------------|---------------------|--------------------------|
| Task | 52 letters | 200 bigrams | 200 words |
| Accuracy | 100% (572/572) | 97.0% (194/200) | 100% (200/200) |
| Recon MSE | 0.0012 | 0.0017 | 0.0015 |
| Hit rate | 46% | 40% | 33% |
| Image size | 128x128 | 192x128 | 256x128 |
| Glimpses | 10 | 11 (5+6) | 20 (8+12) |
| Transfer | no | yes (v3) | yes (single-letter) |
| Prescribed scan | no | no | yes (x only) |
| Time/epoch | ~5s | ~40s | ~135s |

## Losses used

| Loss | Weight | Purpose |
|------|--------|---------|
| Classification CE | 1.0 x4 | Per-position letter identity (26 classes) |
| Reconstruction MSE | 1.0 | Forces sufficient info gathering |
| Scan attention guide | 8.0 | Pulls scan y onto letter strokes |
| Read temporal scaffold | 8.0→0 | Teaches L→R reading order, anneals to 0 |
| Fixation diversity | 1.0 | Prevents fixation collapse |
| Content detection BCE | 0.5 | Scan content awareness |
| Isolation mask CE | 0.5 | Per-position reading capability |

## What to try next
- **Multi-font words**: validate that attention generalizes across fonts (start with 3-5 fonts)
- **Isolation via single-letter images**: cleaner signal than full-canvas masking
- **Separate optimizers**: attention losses → controller only, classification → readout only. May give cleaner gradient signals vs current total_loss sum
- **Mixed-case words**: expand from lowercase to mixed-case
- **Longer words**: 5-6 letters on wider canvas, test if the recipe scales
- **Diagnostic font pairs**: find fonts where specific letters look very different to test attention robustness
