# FBRL — Feedback Recursive Loop

Can a recurrent foveal attention mechanism learn to "read" text by placing strategic fixations — the way a human eye scans a page?

This project trains a vision model that sees the world through a tiny 12x12 pixel window (0.9% of the image). A GRU-based controller decides where to look next, building up a latent representation over 10 sequential glimpses. The model must learn *where* to look, not just *what* it sees.

### Why "Feedback Recursive Loop"?

The name describes the core learning mechanism: the model encodes an image into a latent, decodes it to reconstruct the input, then **recodes** it — decodes the same latent under a different condition (e.g., flipped case). Each recode path adds a constraint that feeds back into the latent space: "can I decode this as 'a' *and* as 'A'?" If both succeed, the latent has captured abstract letter identity. If not, the error reshapes everything — the encoder, the attention policy, the latent geometry.

This extends naturally as the project scales. At word level, the feedback loop becomes: encode a word, decode it, recode it (different font? different case? different modality — audio?). Each recode direction forces the latent toward more abstract, transferable representations. The loop is the learning mechanism, not just an architectural detail.

## Key Results

94% letter classification accuracy and 100% case accuracy on Aa-Zz (52 classes) after 100 epochs. The attention patterns show structural reading — not brute-force tracing:

- **T/t**: scans the crossbar junction area
- **O/o**: fixates the negative space inside the ring
- **A**: hits the apex, crossbar, and legs
- **a**: explores the bowl and stem — different pattern from A

Only ~33% of fixations land directly on letter pixels. The model is selective — it samples diagnostic features (junctions, endpoints, crossbars, negative space) rather than tracing outlines.

## What is Foveal Attention?

It's modeled on how your eye actually works.

**The biological reality:** Your retina isn't uniform. The **fovea** is a tiny spot at the center (~1-2° of visual angle — roughly your thumbnail at arm's length) packed with cone cells. That's the *only* part where you see sharp detail. Everything else (peripheral vision) is progressively blurry. You can detect motion and rough shapes in the periphery, but you can't read text or recognize fine features.

**Saccades:** To see anything in detail, your eye physically moves to aim the fovea at it. These rapid jumps are called saccades — you make 3-4 per second while reading. Between saccades, you're sampling one tiny high-res patch. Your brain stitches these samples into a coherent scene.

**The illusion:** You *feel* like you see the whole world in sharp detail. You don't. Your brain is doing a remarkable job of integrating a sparse sequence of tiny sharp snapshots into a unified percept.

### How this maps to the model

| Biology | Model |
|---------|-------|
| Fovea (tiny sharp patch) | `GlimpseSensor` — 12x12 pixel window (0.9% of 128x128 image) |
| Peripheral vision | Nothing — the model is *even more constrained* than biology, it sees zero outside the patch |
| Saccades (eye movements) | `AttentionController` — GRU decides the next (x,y) fixation |
| Sequence of fixations | 10 glimpses per image |
| Brain integrating across fixations | GRU hidden state accumulating information |
| Conscious perception | Latent vector (256-dim) — the final representation |

### Why it matters

A standard CNN sees the **entire image at once** — like having an eye where the whole retina is fovea. It doesn't need to decide where to look; everything is already there. That's computationally powerful but biologically implausible, and it doesn't scale well to very large images.

Foveal attention forces the model to develop a **strategy** for looking. It has to answer: given what I've seen so far, where should I look next? That's a fundamentally different problem than "process all pixels simultaneously."

Our model develops letter-specific scan strategies — it doesn't trace outlines, it goes straight for diagnostic features. That mirrors human reading: experienced readers don't look at every letter, they fixate on key features and fill in the rest from context.

## Architecture

```
Input (128x128 grayscale) → [noisy image, model only sees 12x12 patches]
                         ↓
          GlimpseSensor (extracts patch at fixation point)
                         ↓
          AttentionController (GRU decides next fixation)
                         ↓  × 10 glimpses
                    Latent (256-dim)
                ↓         ↓          ↓
         CNNDecoder   LetterCls   CaseCls
      (128x128 recon)  (26: A-Z)  (2: upper/lower)
```

The decoder is **case-conditioned**: it receives a case label (0.0=upper, 1.0=lower) concatenated to the latent vector. This lets the model decode the same latent into either case, enabling **recoding** — encode lowercase 'a', decode as uppercase 'A'.

## Training Loss

```
total = recon + letter_cls + case_cls + guide_weight * attn_guide
        + diversity_weight * diversity + recode_weight * recode
```

| Term | Purpose |
|------|---------|
| **Recon** (MSE) | Decode latent back to full image — forces the attention to gather enough information |
| **Letter cls** (cross-entropy, 26 classes) | Classify abstract letter identity (A-Z, case-invariant) |
| **Case cls** (cross-entropy, binary) | Classify upper vs lower case |
| **Attn guide** | Gaussian-blurred "scent trail" pulls fixations toward letter strokes. Evaluated on **clean** images. Blur sigma auto-scales with image size (ratio-based) |
| **Diversity** | Pairwise Gaussian repulsion between fixation points prevents collapse to a single spot |
| **Recode** (MSE) | Flip the case label, decode same latent, compare to partner image (e.g., encode 'a' → decode as 'A') |

## Training Output

Each epoch prints a line like:

```
Epoch 5/200: Recon 0.0173  Ltr 2.8557  Case 0.6955  Attn -0.1115  Div 0.0782  Hit 28%  Recode 0.0174  [5.6s  ETA 18m46s]
```

| Field | What it means | Good values |
|-------|--------------|-------------|
| **Recon** | MSE between decoded image and input. Lower = better reconstruction. | < 0.01 |
| **Ltr** | Cross-entropy for 26-class letter identity (A-Z). Random = ln(26) = 3.26. | < 0.1 |
| **Case** | Cross-entropy for binary case classification. Random = ln(2) = 0.69. | < 0.1 |
| **Attn** | Attention guide loss. More negative = fixations closer to letter strokes. | < -0.10 |
| **Div** | Fixation diversity (repulsion). Lower = fixation points more spread out. | 0.05–0.15 |
| **Hit** | % of fixations landing on actual letter pixels (diagnostic, not a loss). | 25–40% |
| **Recode** | MSE between case-flipped decode and partner clean image. Lower = better. | < 0.01 |
| **[time]** | Wall time per epoch + estimated time remaining. | — |

**Key benchmarks:** Ltr and Case start near their random baselines (3.26 and 0.69) and should drop steadily. Hit rate should climb to ~30% within the first 10-20 epochs — if it stays near 0%, the attention guide is misconfigured (run `make check-attention` to diagnose).

## Quick Start

```bash
# Build and start the container
make build up

# Generate training data (52 letters × 20 noisy variants)
make generate

# Generate clean test data
make generate-test

# Pre-check that attention guide works for this image size
make check-attention DEVICE=cuda

# Train (200 epochs, GPU)
make train DEVICE=cuda

# Test (prints per-letter accuracy + attention visualizations)
make test DEVICE=cuda

# Visualize attention paths on training data
make visualize
```

## CLI Reference

All commands run via `python vision_training.py <command>`:

### generate
```
--letters Aa-Zz        Letter range (A-Z, a-z, Aa-Zz, or individual chars)
--num_variants 20      Noisy copies per letter
--noise_level 0.1      Gaussian noise std (0-1 scale)
--output_dir data/letters
```

### generate_test
```
--letters Aa-Zz
--output_dir data/test
```

### train
```
--data_dir data/letters    Training data path
--epochs 200               Epochs to train
--save_dir data/models     Checkpoint output
--checkpoint_interval 10
--n_glimpses 10            Fixations per image
--patch_size 12            Glimpse window size
--n_scales 1               Resolution scales (1=foveal only)
--device auto|cpu|cuda
--resume PATH              Resume from checkpoint
--guide_weight 4.0         Attention guide loss weight
--blur_sigma_ratio 0.16    Blur sigma as fraction of image size (auto-scales)
--diversity_weight 1.0     Fixation spread pressure (0=off)
--diversity_sigma 0.1      Repulsion radius in normalized coords
--recode_weight 1.0        Case-flip reconstruction loss weight (0=off)
```

### test
```
--model_dir data/models
--test_data_dir data/test
--output_dir data/results
--device auto|cpu|cuda
```

### check_attention
```
--data_dir data/letters    Dataset to check against
--n_epochs 10              Diagnostic epochs to run
--device auto|cpu|cuda
--guide_weight 4.0         Guide weight to test
--blur_sigma_ratio 0.16    Blur ratio to test
```

### visualize
```
--model_dir data/models
--data_dir data/letters
--output_dir data/visualizations
--device auto|cpu|cuda
```

## Makefile

All targets accept overridable variables:

```bash
make train DEVICE=cuda EPOCHS=500 CKPT=100
make generate LETTERS=A-Z VARIANTS=10 NOISE=0.2
make test DEVICE=cuda
make check-attention DEVICE=cuda
```

| Variable | Default | Description |
|----------|---------|-------------|
| `DEVICE` | cpu | cpu or cuda |
| `EPOCHS` | 200 | Training epochs |
| `CKPT` | 100 | Checkpoint interval |
| `LETTERS` | Aa-Zz | Letter range for generation |
| `VARIANTS` | 20 | Noisy variants per letter |
| `NOISE` | 0.1 | Gaussian noise level |

| Target | Description |
|--------|-------------|
| `make generate` | Generate training data |
| `make generate-test` | Generate clean test data |
| `make train` | Train from scratch |
| `make resume` | Resume from last checkpoint |
| `make test` | Run test evaluation |
| `make check-attention` | Quick attention diagnostic (PASS/FAIL) |
| `make visualize` | Generate attention visualizations |
| `make build` / `up` / `down` | Docker lifecycle |
| `make shell` | Shell into container |

## Project Structure

```
fbrl/
├── vision_training.py      # All model code, training, testing, visualization
├── Dockerfile              # python:3.10-slim + torch + numpy + matplotlib + pillow
├── docker-compose.yml      # GPU-enabled container with data volume
├── Makefile                # Pipeline shortcuts (all targets accept variable overrides)
├── thoughts/               # Research notes and iteration roadmap
├── multimodal/             # Archived multimodal (vision+audio) experiment
└── data/                   # Generated data (Docker volume mount)
    ├── letters/            # Training PNGs + metadata.json (with case field)
    ├── test/               # Clean test PNGs
    ├── models/             # Checkpoints + training_metrics.png
    ├── results/            # Test output (attention overlays + reconstructions + recode)
    └── visualizations/     # Attention path visualizations
```

## Design Decisions

**Why foveal-only (no global CNN)?** The model must learn an active perception strategy. A global CNN would give it the answer for free. With a 12x12 window, the model *must* decide where to look — attention is not optional, it's the only channel.

**Why not force higher hit rate?** At ~33% hit rate the model achieves high accuracy by sampling diagnostic features. Forcing fixations onto every stroke would produce outline tracing — wasteful and unlikely to generalize. The current behavior (selective, structural reading) is more biologically plausible.

**Why evaluate attention on clean images?** With noise_level=0.1, noisy images have white speckle everywhere. The attention guide would blur those noise pixels into a diffuse glow, and hit rate would count noise hits as successes. Clean images give honest signals.

**Why ratio-based blur sigma?** The attention guide uses a Gaussian blur to create a "scent trail" around letter strokes. At 96x96, `blur_sigma=15` worked. At 128x128, the same absolute value broke attention completely (0% hit rate). Expressing sigma as a fraction of image size (0.16) makes it auto-scale. This is critical for future work where the MetaController will feed crops of varying sizes to the attention system.

**Why separate letter and case classifiers?** Letter identity (A-Z) and case (upper/lower) are orthogonal concepts. A single 52-class classifier would conflate them. Separate heads let the latent space learn abstract letter identity that generalizes across case — 'a' and 'A' share the same letter class but differ in case.

**Why recode loss?** Forces the latent space to capture letter identity independently of case. If the same latent can decode both 'a' and 'A' (conditioned on the case label), the representation is truly case-invariant. This is a stepping stone toward abstract symbol understanding.

## Roadmap

The current single-letter model is the foundation. The research goal is to scale foveal attention from character recognition toward reading:

1. **Multi-font** — Same letters, 10-15 fonts. Tests whether attention strategies generalize across visual styles or are font-specific.
2. **Bigrams/trigrams** — 2-3 letter combinations. The attention must scan left-to-right and segment characters. Output becomes a sequence.
3. **Words** — Variable-length strings with a language model prior. Tests whether the model skips predictable letters, fixates word centers, and spends more glimpses on rare words (all human reading behaviors).
4. **Meta-attention** — A coarse controller that finds word boundaries (whitespace, line breaks) and deploys the fine letter-reader within each region. Hierarchical saccade planning.
5. **Multimodal** — Integrate audio to provide top-down priors (syllable boundaries, word predictions) that guide visual attention during reading.
