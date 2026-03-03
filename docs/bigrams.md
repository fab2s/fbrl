# Bigram Experiments

Two-letter reading on a 128x128 canvas. Originally conceived as the step between single letters and words, bigrams revealed a fundamental lesson about task geometry that shaped everything after.

## Architecture

```
Input (128x128 grayscale, two lowercase letters)
         |
    SCAN PHASE (5 glimpses, 12x18 wide patches)
         GlimpseSensor -> GRU -> location_head -> (free x, free y)
         Purpose: map spatial layout, find letter positions
         |  h carries forward
    READ PHASE (6 glimpses, 12x12 focused patches)
         GlimpseSensor -> GRU -> location_head -> (free x, free y)
         Purpose: identify individual letters
         |
    CrossAttentionReadout (2 position query tokens)
         attends over READ hidden states only
         |                     |
    Pos1 Classifier       Pos2 Classifier
     (26: a-z)              (26: a-z)
         |
    VisualDecoder (128x128 recon)
```

Two sensors: wide scan (12x18) for spatial mapping, focused read (12x12) for identification. A single GRU controller spans both phases. CrossAttentionReadout uses 2 learned position tokens (left/right bias) to extract per-position representations from read-phase hidden states.

## Loss Terms

| Term | Purpose |
|------|---------|
| **Recon** (MSE) | Reconstruct the full bigram image |
| **Pos1/Pos2 cls** (CE, 26 classes each) | Per-position letter classification |
| **Scan attn** | Blurred guide for scan phase |
| **Read attn** | Temporal scaffold — left/right halves guide early/late read glimpses |
| **Scan/Read diversity** | Split VY: scan_vy=0.3 (horizontal spread), read_vy=1.5 (vertical) |
| **Masked cls** | Randomly mask one half, classify the visible letter |

## The Canvas Scale Finding

This is the single most important result from the bigram experiments.

Two letters on 128px is **geometrically cheatable**. The 12x12 foveal window covers ~0.9% of the canvas — enough that a few central fixations can see parts of both letters simultaneously. Even with diversity pressure forcing horizontal spread, the model finds holistic shortcuts rather than developing genuine sequential reading.

98-99% accuracy was achieved, but the attention patterns showed the model wasn't truly reading each letter independently. Masked classification loss (hide one half, read the other) helped but couldn't fully overcome the geometry.

**The lesson**: you can't force reading strategy through loss design alone. The task geometry must make cheating impossible.

This drove the transition to 4-letter words on a 256x128 canvas, where a 12x12 window covers only 0.4% of the image and holistic reading is geometrically impossible.

## Results

| Version | Accuracy (both correct) | Key detail |
|---------|------------------------|------------|
| v1 | 97-99% | Transfer from single-letter, 300 epochs |

Detailed results in `runs/bigrams/`.

## Status

Bigrams are a completed stepping stone. The two-phase scan/read architecture and CrossAttentionReadout validated here became the foundation for the word model. The canvas scale finding was the critical insight — bigrams themselves are not actively developed, but the architecture they pioneered lives on.
