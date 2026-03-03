# Single-Letter Experiments

The foundation of everything. A foveal attention model learns to recognize 52 letter classes (Aa-Zz) across 11 fonts by placing strategic fixations on a 128x128 canvas — seeing only a 12x12 pixel patch at each glance.

## Architecture

```
Input (128x128 grayscale, noisy)
         |
    SCAN PHASE (optional, 1+ glimpses, 12x18 wide patches)
         GlimpseSensor -> GRU -> location_head -> (prescribed/learnable x, learned y)
         Content head: predicts "is there ink here?"
         |  h carries forward
    READ PHASE (6-10 glimpses, 12x12 focused patches)
         GlimpseSensor -> GRU -> location_head -> (free x, free y)
         |
    Latent (256-384 dim)
    |         |          |
VisualDecoder  LetterCls   CaseCls
(128x128 recon) (26: A-Z)  (2: upper/lower)
```

The decoder is **case-conditioned**: it receives a case label (0=upper, 1=lower) concatenated to the latent. This enables **recoding** — encode lowercase 'a', decode as uppercase 'A'. If both work, the latent has factored identity from case.

## Loss Terms

| Term | Purpose |
|------|---------|
| **Recon** (MSE) | Decode latent back to full image — forces attention to gather enough information |
| **Letter cls** (CE, 26 classes) | Classify abstract letter identity (A-Z, case-invariant) |
| **Case cls** (CE, binary) | Classify upper vs lower case |
| **Attn guide** | Gaussian-blurred "scent trail" pulls fixations toward ink. Blur sigma scales with image size |
| **Diversity** | Pairwise Gaussian repulsion prevents fixation collapse |
| **Recode** (MSE) | Flip case label, decode same latent, compare to opposite-case image |
| **Void repulsion** (v7+) | Penalizes fixations on empty space. Local gradient only — no global center-pull |

## Experiment History

### v1-v3: establishing baselines

**v1** (single font, 200ep) — 100%/100%. Proved the architecture works. **v2** (11 fonts, 50ep) — 99.5%. Multi-font forces the latent to encode identity, not font style. **v3** (CosineAnnealingLR, 100ep) — 100%/100% across all 11 fonts. Became the baseline for transfer.

### v4: vertical diversity

Adding `diversity_vy=1.5` (penalize vertical clustering) produced better attention spread without hurting accuracy. Revealed that anisotropic diversity pressure shapes scan behavior.

### v5: scan phase

Added a scan phase (3 scan + 10 read = 13 glimpses) with wide rectangular patches (12x18) and a content detection head. 100%/100%. The pretrained scan_sensor and content_head became the foundation for word model transfer.

### v6: fewer glimpses

Systematic reduction experiments:
- **3 scan + 7 read (10 glimpses)**: 100%/100% — 23% fewer than v5
- **1 scan + 7 read (8 glimpses)**: 100%/100% — 38% fewer
- **Learnable scan x**: positions drift toward content, better than prescribed linspace
- **Anchored read (3+7 with position reset): CATASTROPHIC FAILURE** — 0.0000 train CE, 48.6% test. Position reset breaks GRU spatial continuity.

Key insight: **flat read preserves GRU momentum**. The hidden state carrying forward from one glimpse to the next is essential — resetting position discards spatial context.

Recode quality is the honest measure of latent richness. Classification saturates first (binary signal), but recode demands pixel-level spatial fidelity from the latent.

### v7: void repulsion (current best)

**1 scan + 6 read = 7 glimpses** — 46% fewer than v5, zero accuracy loss.

Architecture changes:
- **Scan-only guide** (`scan_guide_weight=8.0`, `guide_weight=0.0`): blurred guide for scan, nothing for reads. The v4 interleaved word failure showed that global guide pulls reads to center.
- **Void repulsion for reads**: "don't stare at nothing" — penalizes fixations on empty space via patch sampling. Local gradient only (active at ink boundaries, zero in deep void).
- **Isotropic diversity** (`diversity_vy=1.0`): earlier vy>1 was compensating for guide center-pull. Without guide on reads, isotropic produces natural clustering.

#### Self-scaffolding discovery

All losses active from epoch 1 with fixed weights. No curriculum, no freezing. Yet they sequence by difficulty:

1. **Classification** (easiest — 26 discrete bins, strong CE gradient) converges first (~epoch 30)
2. **Reconstruction** (pixel fidelity) converges alongside
3. **Recode** (hardest — requires latent factorization) takes over as dominant gradient once classification saturates

The easy tasks bootstrap representations that hard tasks need. Classification forces diagnostic fixations; those fixations provide spatial detail for recode. The encode-decode-recode loop is a self-scaffolding system.

**Implication**: when designing multi-task losses, intrinsic difficulty can replace explicit curriculum if tasks share representations and have natural difficulty ordering.

## Key Findings

### Structural reading emerges naturally

The model develops letter-specific scan strategies without being taught them. 'T' gets the crossbar junction, 'O' gets the negative space inside the ring, 'A' gets the apex and legs. Only ~40% of fixations land on letter pixels — the model samples diagnostic features rather than tracing outlines.

### For OCR, the encoder is the product

Decoder, recode, and motor are all training scaffolding. Inference is just: encoder + linear head -> UTF-8. The recode mechanism proves the latent factored identity from case. The decoder proves spatial information was captured. But at inference time, you only need the encoder pathway.

## Results Summary

| Version | Glimpses | Accuracy | Key change |
|---------|----------|----------|------------|
| v3 | 10 read | 100%/100% | CosineAnnealingLR, 11 fonts |
| v5 | 3+10=13 | 100%/100% | Scan phase, content head |
| v6 | 1+7=8 | 100%/100% | Learnable scan x, 38% fewer glimpses |
| v7 | 1+6=7 | 100%/100% | Void repulsion, scan-only guide, self-scaffolding |

Detailed per-run results in `runs/letters/`.
