# Motor Trace Experiments

The Read-Write-Render-Re-Read loop: the vision encoder produces a latent, a GRU motor decoder writes a pen trajectory, a differentiable renderer draws it, and the same encoder re-reads the result. If the rendered trace is readable, the motor decoder has learned to write.

## Architecture

```
Input (128x128 grayscale)
         |
    SCAN PHASE (1 glimpse, 12x18, learnable x)
         |  h carries forward
    READ PHASE (6 glimpses, 12x12, free x,y)
         |
    Latent (384-dim)
    |         |          |               |
VisualDecoder  LetterCls   CaseCls    MotorTraceDecoder
(128x128 recon) (26:A-Z)  (2:case)      (GRU -> 48 points)
                                              |
                                    (x, y, pen_down) trajectory
                                              |
                                     SoftRender (Gaussian blobs)
                                              |
                                    Rendered image (128x128)
                                              |
                                     Re-read through SAME encoder
                                              |
                                         Re-read latent + classification
```

**Multi-head backward** (4 passes, VRAM-safe):
1. Attention -> controller + sensors (guide, diversity, void, content)
2. Classification -> classifiers (letter CE, case CE)
3. Reconstruction -> decoder (recon MSE, recode MSE)
4. Motor -> motor_decoder only (deferred after main graph freed)

The motor decoder receives a **detached** latent — motor gradients don't interfere with vision learning through the latent. Motor gradients flow through the renderer and re-read encoder to shape the motor_decoder's output.

## Loss Terms

### Vision (same as single-letter)

| Term | Purpose |
|------|---------|
| **Recon** (MSE) | Reconstruct input from latent |
| **Letter/Case cls** (CE) | Identity classification |
| **Attn guide** | Scan-only blurred guide |
| **Void repulsion** | Read "don't stare at nothing" |
| **Diversity** | Fixation spread |
| **Recode** (MSE) | Case-flipped reconstruction (disabled in single-case mode) |

### Motor

| Term | Weight | Role |
|------|--------|------|
| **Trajectory MSE** (scaffold) | 0.5, anneals | Point-wise match to font skeleton ground truth. Shape guidance early, fades out. |
| **Pen BCE** (scaffold) | (same) | Pen up/down timing against ground truth |
| **Re-read cls** (CE) | 1.0 | Classify the rendered trajectory — forces readable output |
| **Latent matching** (MSE) | 2.0 | Re-read latent must match original latent. Dense continuous gradient. Primary motor signal. |
| **Render matching** (MSE) | 0.25 | Rendered image vs clean source. Gentle pixel anchor. |

The trajectory scaffold provides initial shape guidance (like training wheels), then anneals so classification and latent matching dominate. The motor decoder must ultimately produce traces that are *readable*, not just geometrically close to ground truth.

### Enhanced losses (v2/v3)

**Why basic re-read accuracy is misleading**: fat Gaussian blobs (sigma=1.5) cover the right general area even with bad trajectories. The same encoder learns to "read" motor artifacts (spatial blob distribution shortcuts). Classification is coarse: right/wrong gives no "how far off" signal.

The enhanced loss stack addresses this:

- **Latent matching**: forces rendered image to produce the *same internal representation* as the original. Dense, continuous gradient — can't be cheated with spatial shortcuts.
- **Render matching**: at low weight (0.25), a gentle guardrail preventing total visual divergence without constraining style.
- **Sharper rendering** (sigma=0.75 vs 1.5): thinner lines demand precision from the motor decoder.
- **Centerline trajectories**: skeleton-based ground truth instead of outline tracing — cleaner stroke paths.

Optional (not in v3): **Frozen re-reader** — static encoder copy from checkpoint, never updates. Gradients flow through it to motor_decoder but weights are fixed. Prevents encoder co-adaptation to motor artifacts.

## Experiment History

### v1: transfer from v5-scan (200ep, 11 fonts)

First motor experiment. Transferred pretrained vision from v5-scan (3 scan + 10 read), trained motor simultaneously.

- Vision: 84.6% letter, 78.8% case (degraded vs pure vision — shared latent pressure)
- Motor: 47.6% re-read letter, Pen F1 0.94, Traj MSE 0.39
- Motor decoder invented its own writing style — didn't mimic font vectors
- Serif fonts struggled most (69% letter accuracy)

**Key insight**: training vision + motor simultaneously means both compete for latent space. Motor v1 showed the read-write-re-read loop works in principle but needs curriculum learning.

### v2 prep: enhanced losses (designed, not yet trained standalone)

Designed the enhanced loss stack (latent matching, frozen re-reader, render matching, sharper sigma). Switched to 48 trajectory points, 384 latent dim, binary closing before skeletonization for cleaner targets.

### v3: from scratch with void repulsion (IN PROGRESS)

Aligned with v7 letter findings: 1 scan + 6 read = 7 glimpses, scan-only guide, void repulsion for reads, isotropic diversity. Training from scratch — no transfer. The hypothesis: self-scaffolding should let all loss terms sequence naturally, as it did for pure vision in v7.

Config: lowercase only (`case_filter: lower`), 384 latent dim, centerline trajectory targets, fast-annealing scaffold (25% of epochs).

Early observations (~80 epochs):
- Vision side converging slower than pure v7 (expected — 384 dim + motor noise through shared encoder)
- Void repulsion saturated quickly (all fixations on ink)
- Motor re-read accuracy near zero — waiting for letter classification to break through
- Self-scaffolding hypothesis being tested: will classification convergence trigger motor improvement?

## Key Concepts

### Read -> Write -> Re-Read

The motor pathway is a recode direction: instead of decoding the latent as an image with flipped case, decode it as a *pen trajectory* that draws the letter. The re-read step closes the loop — if the encoder can read back what the motor wrote, the motor has captured the essential visual structure.

### Why detached latent

Motor gradients could interfere with vision learning if they flow through the latent. By detaching, the motor decoder learns to work with whatever latent the vision system produces — it's a downstream consumer, not a co-designer of the latent space.

### Trajectory ground truth

Generated from font outlines (centerline/skeleton extraction):
1. Render the letter at high resolution
2. Binarize and extract skeleton (morphological thinning)
3. Order skeleton pixels into a continuous path
4. Resample to fixed N points (48)
5. Each point: (x, y, pen_down) in normalized [-1, 1] coordinates

The scaffold trains against these targets early, then fades so the model can develop its own writing style optimized for readability.

## Results

| Version | Vision Accuracy | Re-read Accuracy | Key detail |
|---------|----------------|-----------------|------------|
| v1 | 84.6% letter | 47.6% letter | Transfer, 11 fonts, sigma=1.5 |
| v3 | TBD | TBD | From scratch, void repulsion, enhanced losses |

Detailed results in `runs/motor/`.
