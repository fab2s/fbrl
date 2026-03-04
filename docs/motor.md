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

### v3: from scratch with void repulsion

Aligned with v7 letter findings: 1 scan + 6 read = 7 glimpses, scan-only guide, void repulsion for reads, isotropic diversity. Training from scratch — no transfer. The hypothesis: self-scaffolding should let all loss terms sequence naturally, as it did for pure vision in v7.

Config: lowercase only (`case_filter: lower`), 384 latent dim, centerline trajectory targets, fast-annealing scaffold (25% of epochs).

Results (200 epochs): letter CE 1.97, RR 23%, traj MSE 0.097. LR exhausted before classification broke through.

### v4: two-pass unified vision + coupled motor

**Hypothesis**: motor training failed because multi-head backward (4 separate passes with `inputs=` targeting) isolated the encoder from classification gradients. Fix: make vision identical to v7 (single unified backward), then add motor as a second pass where rr_cls gradients couple back into the encoder at a controlled LR.

**Architecture**: two optimizers instead of four:

```
Pass 1 (vision — identical to v7):
  vision_loss = recon + letter_CE + case_CE + attn + div + content + void + recode
  vision_opt.zero_grad() → vision_loss.backward() → clip + step

Pass 2 (motor — fresh forward through post-update encoder):
  motor_opt.zero_grad()
  latent2 = model(img)           ← fresh graph, post-vision-update weights
  trajectory = motor_decoder(latent2)
  traj scaffold + render + re-read → rr_cls
  motor_loss.backward()           ← gradients to motor_decoder + encoder
  motor_opt.step()                ← motor at 0.001, encoder at 0.0001
```

Motor optimizer holds both motor params (lr=0.001) and vision params (lr=0.0001 = `motor_coupling_lr`). Each optimizer maintains separate Adam momentum states.

**Config**: single font (dejavu-sans) for fast iteration (~8s/epoch), 8 read glimpses, 384 latent, 300 epochs, rr_cls warmup 25%.

**Training convergence** (300 epochs):

| Metric | Ep 1 | Ep 50 | Ep 100 | Ep 150 | Ep 300 |
|--------|------|-------|--------|--------|--------|
| Letter CE | 3.28 | 0.26 | 0.016 | 0.007 | 0.0002 |
| Case CE | 0.70 | 0.035 | 0.008 | 0.0005 | 0.0000 |
| RR letter acc | 4% | 28% | 55% | 68% | 96% |
| Traj MSE | 0.022 | 0.073 | 0.060 | 0.055 | 0.068 |

Vision converged exactly like v7 — the unified backward works. RR accuracy climbed to 96% despite traj MSE plateauing at ~0.07 (motor found readable output without perfect trajectory fidelity).

**Test results** (572 samples, 11 fonts):

| Font | Letter | RR |
|------|--------|----|
| dejavu-sans (train) | 34.6% | 26.9% |
| All others | 6-23% | 6-12% |
| **Overall** | **17.8%** | **11.0%** |

**Failure analysis**: massive train/test gap — 96% RR in training vs 11% on test. Even on the training font, only 35% letter accuracy (vs v7's 100%). The motor coupling at 0.0001 LR over 300 epochs was enough to corrupt the encoder's generalization. The encoder co-adapted to its own motor traces rather than learning robust letter features.

**Conclusion**: joint training of vision + motor doesn't work. Any motor→encoder gradient flow, regardless of LR scaling, causes co-adaptation that destroys generalization. The encoder must be completely frozen.

### v5: frozen v8 backbone + self-supervised constrained motor

**Hypothesis**: freeze the solved v8 backbone (100%/100%), train only a constrained motor decoder with self-supervised losses. No GT trajectory scaffold. Two signals: ink-on-target (spatial) and re-read through frozen classifier (semantic).

**Architecture**:
- Frozen v8 backbone (1 scan + 8 reads, latent_dim=256, learnable_scan_x)
- ConstrainedMotorDecoder: GRU-based, 4 gated strokes × 20 points, latent-predicted starts, sigmoid gates, 269K trainable params
- Gated rendering: per-stroke sigmoid gates control ink, Gaussian blob rendering (sigma=0.75)
- Ink-on-target: `rendered * blur(canonical)` — spatial gradient toward canonical font pixels
- Re-read: rendered → frozen encoder → frozen classifier → CE — semantic signal

**Key findings**:

1. **Gate collapse** (first run): with rr_cls warmup, only ink loss active early → model learns degenerate solution (turn off all gates = zero ink = zero void penalty). Fix: remove warmup entirely (modular design doesn't need encoder protection).

2. **Cold-start bootstrapping**: fixed blur sigma=3 too narrow for distant strokes — gradient vanishes. Fix: blur sigma annealing from 20→3. Large blur acts as smooth spatial attractor, pulling strokes toward letter pixels. This drove all the real progress.

3. **Adversarial pattern problem** (fundamental): 55.8% train rr_ltr, 13.6% test. The re-read path optimizes against the classifier's decision boundary, not visual letter similarity. The motor learns patterns that activate the correct class neurons — adversarial examples, not letters. Like simple makeup patterns that trick face recognition. The classifier is a lossy bottleneck (images → 26 classes), discarding spatial structure. The motor only needs to land in the right decision region, which is much easier than actually writing the letter.

**Conclusion**: self-supervised motor via re-read through a frozen classifier produces adversarial shortcuts, not genuine writing. The ink-on-target spatial signal is the honest signal, but alone it lacks semantic pressure. Fixing this with direct pixel matching (MSE vs canonical) would work but reduces to supervised copying — a different project that doesn't feed back into the vision/attention research. Motor experiments paused pending new ideas.

**Technical notes** (for future reference):
- Performance-driven blur annealing implemented but not fully tested: `max(ink_blur_sigma, ink_blur_sigma_start * (1 - prev_rr_ltr))`
- Code supports both ConstrainedMotorDecoder (v5) and legacy MotorTraceDecoder (v1-v4)
- `evaluate.py` detects `learnable_scan_x` from state_dict (`scan_xs` key) as fallback when not in checkpoint metadata

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
| v3 | CE 1.97 | 23% letter | From scratch, 200ep, LR exhausted |
| v4 | 17.8% letter | 11.0% letter | Two-pass coupled, co-adaptation failure |
| v5 | 100% (frozen v8) | 13.6% letter | Frozen backbone, adversarial patterns |

Detailed results in `runs/motor/`.
