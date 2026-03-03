# Word Experiments

4-letter word reading on a 256x128 canvas. The first task where genuine sequential reading is geometrically mandatory — a 12x12 foveal window covers 0.4% of the canvas, making holistic reading impossible.

## Architecture

```
Input (256x128 grayscale, 4-letter word)
         |
    SCAN PHASE (learnable x sweep, 12x18 wide patches)
         x: learnable positions (or prescribed linspace)
         y: GRU -> location_head -> tanh -> y component only
         Content head: predicts "is there ink here?"
         |  h carries forward
    READ PHASE (fully free x,y, 12x12 focused patches)
         GRU -> location_head -> tanh -> (x, y)
         |
    CrossAttentionReadout (4 position query tokens)
         queries initialized at [-0.75, -0.25, +0.25, +0.75]
         attends over READ hidden states
         |         |         |         |
      Pos1       Pos2      Pos3      Pos4
    Classifier  Classifier Classifier Classifier
     (26: a-z)   (26: a-z)  (26: a-z)  (26: a-z)
         |
    VisualDecoder (256x128 recon)
```

## Loss Terms

| Term | Purpose |
|------|---------|
| **Recon** (MSE) | Reconstruct the full 256x128 word image |
| **Pos cls** (CE x4) | Per-position letter classification |
| **Scan attn** | Blurred guide for scan phase only |
| **Read attn** | 4-stripe temporal scaffold — each read segment guided to a horizontal quarter |
| **Content BCE** | Binary prediction on scan states — "letter content here?" |
| **Diversity** | Split VY for scan (horizontal) and read |
| **Isolation cls** | 128x128 single-letter images through the word model — tests per-letter reading |

**Multi-head optimization** (v2+): 3 separate backward passes — attention losses -> controller/sensors, classification -> classifiers, reconstruction -> decoder. Prevents gradient cross-contamination.

## Design Journey

### Why prescribed scanning

Giving the model full x,y control during scanning led to collapse — random fixations or center-camping. Prescribing x as a left-to-right sweep teaches L-to-R mechanics while y is freely learned. The model discovers that lowercase letter bodies sit below vertical center and develops a slight diagonal scan trajectory.

### Transfer scaffold

The single-letter model already knows how to read letterforms. Naively unfreezing everything on the word task destroys this — the untrained scan sensor sends garbage through the shared GRU, corrupting pretrained representations. Solution: freeze read_sensor + classifiers for 67% of training, then gradually unfreeze with very low learning rates.

### Isolation testing

Canvas masking (zero 3 of 4 letter stripes) tests two things at once: "can you find the letter?" and "can you read it?" — too hard, stalls at ~1.0 CE. Feeding 128x128 single-letter images directly tests only reading — converges to 0.04 CE. **Test one capability at a time.**

### Interior positions learn faster

P2 and P3 (interior letters) converge before P1 and P4 (edges). Interior positions have richer context from neighboring letters during cross-attention readout, and the scan passes through their region more centrally.

## The Interleaved Failure (v4)

Instead of all scans then all reads, v4 tried alternating: scan position 1 -> read group 1 -> scan position 2 -> read group 2. Config: 4 scan + 4x5 read = 24 glimpses, transfer from v6.

**Training looked great**: all position CEs below 0.15 by epoch 200. **Test: 0% accuracy.** Pure memorization of 200 training words.

### Root cause

1. `scan_xs` initialized at +/-0.75, but word content spans roughly +/-0.5. Outer scans landed in empty space.
2. Empty space = zero gradient through grid_sample. Outer scans **stuck** — can't drift inward.
3. Read glimpses (fully free) migrated inward following the attention guide's global content gradient.
4. Outer and inner groups ended up reading the **same inner letters**.
5. Cross-attention readout confused -> model fell back to word-level pattern memorization.
6. Isolation loss stayed at 1.77 throughout (vs v2's 0.008) — never learned per-position reading.

### The lesson: scan = guide, reads = self-directed

The global attention guide creates a center-pull magnet that's toxic for interleaved reading. The fix:
- `guide_weight: 0.0` for reads (no global pull)
- `scan_guide_weight: 8.0` for scan only (long-range signal to find content)
- Void repulsion for reads (local "don't stare at nothing" — no center bias)
- Tighter scan init (+/-0.5 instead of +/-0.75)

This principle was **validated in v7 single letters**: guide-free reads + void repulsion -> 100%/100% with only 7 glimpses. Next: retry interleaved words with void repulsion reads.

## Results

| Version | Accuracy (all 4 correct) | Key detail |
|---------|--------------------------|------------|
| v1 | 100% | Prescribed x-scan, single font, 200ep |
| v2 | 99.5% | Multi-head + isolation testing |
| v3 | converging ~2x faster | v5-scan transfer + AMP |
| v4 | 0% test (FAILED) | Interleaved — memorization, see analysis above |

Detailed results in `runs/words/`.

## Next Steps

- **Interleaved v2**: retry with void repulsion reads, scan-only guide, tighter scan init
- **Multi-font words**: generalization across visual styles
- **Semantic layer**: learn letter transition probabilities or word-level priors to aid disambiguation (see [roadmap](../README.md#whats-next))
