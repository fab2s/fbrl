# FBRL — Feedback Recursive Loop

Can a recurrent foveal attention mechanism learn to "read" text by placing strategic fixations — the way a human eye scans a page?

This project trains a vision model that sees the world through a tiny 12x12 pixel window (0.9% of the image). A GRU-based controller decides where to look next, building up a latent representation over a sequence of glimpses. The model must learn *where* to look, not just *what* it sees.

### Why "Feedback Recursive Loop"?

The name describes the core learning mechanism: the model encodes an image into a latent, decodes it to reconstruct the input, then **recodes** it — decodes the same latent under a different condition (e.g., flipped case). Each recode path adds a constraint that feeds back into the latent space: "can I decode this as 'a' *and* as 'A'?" If both succeed, the latent has captured abstract letter identity. If not, the error reshapes everything — the encoder, the attention policy, the latent geometry.

This extends naturally as the project scales. At word level, the feedback loop becomes: encode a word, decode it, recode it (different font? different case? different modality — audio?). Each recode direction forces the latent toward more abstract, transferable representations. The loop is the learning mechanism, not just an architectural detail.

## Key Results

**Single letters** — 100% letter and case accuracy across 52 classes (Aa-Zz) and 11 fonts, with pixel-perfect case recoding (encode 'a', decode as 'A').

**Bigrams** — 97% both-correct accuracy on 200 common English bigrams (192x128 images), using transfer learning from the single-letter encoder. First successful multi-character run.

The attention patterns show structural reading — not brute-force tracing:

- **T/t**: scans the crossbar junction area
- **O/o**: fixates the negative space inside the ring
- **A**: hits the apex, crossbar, and legs
- **a**: explores the bowl and stem — different pattern from A
- **"th"**: scans left letter, then right letter — develops sequential reading

Only ~40% of fixations land directly on letter pixels. The model is selective — it samples diagnostic features (junctions, endpoints, crossbars, negative space) rather than tracing outlines.

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
| Sequence of fixations | 10 glimpses (single letter) or 15 (bigrams) |
| Brain integrating across fixations | GRU hidden state accumulating information |
| Conscious perception | Latent vector (256-dim) — the final representation |

### Why it matters

A standard CNN sees the **entire image at once** — like having an eye where the whole retina is fovea. It doesn't need to decide where to look; everything is already there. That's computationally powerful but biologically implausible, and it doesn't scale well to very large images.

Foveal attention forces the model to develop a **strategy** for looking. It has to answer: given what I've seen so far, where should I look next? That's a fundamentally different problem than "process all pixels simultaneously."

Our model develops letter-specific scan strategies — it doesn't trace outlines, it goes straight for diagnostic features. That mirrors human reading: experienced readers don't look at every letter, they fixate on key features and fill in the rest from context.

## Architecture

### Single-letter model (`VisionModel`)

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

### Bigram model (`BigramVisionModel`)

```
Input (192x128 grayscale) → [wider canvas for two letters]
                         ↓
          GlimpseSensor (same 12x12 patch — even smaller relative to image)
                         ↓
          AttentionController (GRU plans left-to-right scanning)
                         ↓  × 15 glimpses
                    Latent (256-dim)
                ↓              ↓            ↓
         CNNDecoder      Pos1Classifier  Pos2Classifier
      (192x128 recon)    (26: a-z)       (26: a-z)
```

Key differences from single-letter: wider canvas (192x128 vs 128x128), more glimpses (15 vs 10), two position classifiers instead of letter+case, no case conditioning or recode (bigrams are all lowercase). The encoder (GlimpseSensor + GRU) can be **transferred** from a trained single-letter model, giving the attention system a head start on reading letterforms.

## Training Loss

### Single-letter

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

### Bigram

```
total = recon + pos1_cls + pos2_cls + guide_weight * attn_guide
        + diversity_weight * diversity
```

Same core losses but with two position classifiers (pos1, pos2) instead of letter + case, and no recode term. Training uses a **temporal attention scaffold** that linearly anneals from 1.0 to 0.0 over the scaffold phase, gently guiding the attention controller to develop left-then-right scanning before it must discover the pattern on its own.

## Training Output

### Single-letter

Each epoch prints a line like:

```
Epoch 5/200: Recon 0.0173  Ltr 2.8557  Case 0.6955  Attn -0.1115  Div 0.0782  Hit 28%  Recode 0.0174  lr 0.000996  [5.6s  ETA 18m46s]
```

| Field | What it means | Good values |
|-------|--------------|-------------|
| **Recon** | MSE between decoded image and input. Lower = better reconstruction. | < 0.01 |
| **Ltr** | Cross-entropy for 26-class letter identity (A-Z). Random = ln(26) = 3.26. | < 0.1 |
| **Case** | Cross-entropy for binary case classification. Random = ln(2) = 0.69. | < 0.1 |
| **Attn** | Attention guide loss. More negative = fixations closer to letter strokes. | < -0.10 |
| **Div** | Fixation diversity (repulsion). Lower = fixation points more spread out. | 0.05–0.15 |
| **Hit** | % of fixations landing on actual letter pixels (diagnostic, not a loss). | 25–45% |
| **Recode** | MSE between case-flipped decode and partner clean image. Lower = better. | < 0.01 |
| **lr** | Current learning rate (CosineAnnealingLR). Decays from 0.001 → ~0 over the run. | — |
| **[time]** | Wall time per epoch + estimated time remaining. | — |

**Key benchmarks:** Ltr and Case start near their random baselines (3.26 and 0.69) and should drop steadily. Hit rate should climb to ~30% within the first 10-20 epochs — if it stays near 0%, the attention guide is misconfigured (run `make check-attention` to diagnose).

### Bigram

```
Epoch 5/300: Recon 0.0262  Pos1 2.5719  Pos2 2.5564  Attn -0.1125  Div 0.1505  Hit 40%  lr_enc 0.000100  lr_read 0.001000  scaff 0.9800  [8.2s  ETA 40m23s]
```

| Field | What it means | Good values |
|-------|--------------|-------------|
| **Pos1** | Cross-entropy for first letter (26 classes). Random = 3.26. | < 0.01 |
| **Pos2** | Cross-entropy for second letter (26 classes). Random = 3.26. | < 0.01 |
| **lr_enc** | Encoder learning rate (low when using transfer). | — |
| **lr_read** | Readout head learning rate (classifiers + decoder). | — |
| **scaff** | Temporal scaffold strength (1.0 = full guidance, 0.0 = off). | anneals → 0 |

## Results

- [128x128, Aa-Zz, single font, 200 epochs](thoughts/results_128x128_Aa-Zz.md) — 100% letter, 100% case, pixel-perfect recode
- [128x128, Aa-Zz, 11 fonts, 50 epochs](thoughts/results_multi-font_50ep.md) — 99.5% letter, 99.7% case across all fonts (guide_weight=8.0)
- [128x128, Aa-Zz, 11 fonts, 100 epochs, CosineAnnealingLR](thoughts/results_multi-font-cosine_100ep.md) — 100% letter, 100% case, all 11 fonts perfect
- [192x128, 200 bigrams, transfer learning, 300 epochs](thoughts/results_bigram-transfer_300ep.md) — 97% both-correct, first successful bigram run

## Requirements

**PyTorch 2.5.1** — Pinned to this version because training runs on a Pascal-era GPU (GTX 1080 Ti, compute capability 6.1). PyTorch 2.6+ dropped CUDA support for Pascal. If you have an Ampere or newer card, feel free to bump the version in the Dockerfile.

If anyone feels like donating a modern GPU to the cause, the latent space would be eternally grateful.

## Quick Start

### Single-letter pipeline

```bash
# Build and start the container
make build up

# Generate training data (52 letters x 20 noisy variants x 11 fonts)
make generate

# Generate clean test data (52 letters x 11 fonts)
make generate-test

# Pre-check that attention guide works for this image size
make check-attention

# Train (100 epochs, GPU)
make train DEVICE=cuda

# Test (prints per-letter accuracy + attention visualizations)
make test DEVICE=cuda

# Generate interactive attention atlas (self-contained HTML)
make atlas DEVICE=cuda

# Archive a trained model
make archive NAME=my-run EPOCHS=100
```

### Bigram pipeline

```bash
# Generate bigram training + test data (200 bigrams, default font)
make generate-bigrams
make generate-bigrams-test

# Train with transfer learning from a single-letter model
make train-bigrams EPOCHS=300 DEVICE=cuda TRANSFER=runs/v3-multi-font-cosine/model_final.pth.gz

# Test
make test-bigrams DEVICE=cuda

# Generate bigram attention atlas
make bigram-atlas DEVICE=cuda

# Archive
make archive-bigrams NAME=v4-bigram-transfer EPOCHS=300
```

## CLI Reference

All commands run via `python vision_training.py <command>`:

### Single-letter commands

#### generate
```
--letters Aa-Zz        Letter range (A-Z, a-z, Aa-Zz, or individual chars)
--num_variants 20      Noisy copies per letter
--noise_level 0.1      Gaussian noise std (0-1 scale)
--output_dir data/letters
--fonts all            Font spec: "all", "default", or comma-separated names
```

#### generate_test
```
--letters Aa-Zz
--output_dir data/test
--fonts all            Font spec: "all", "default", or comma-separated names
```

#### train
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
--guide_weight 8.0         Attention guide loss weight
--blur_sigma_ratio 0.16    Blur sigma as fraction of image size (auto-scales)
--diversity_weight 1.0     Fixation spread pressure (0=off)
--diversity_sigma 0.1      Repulsion radius in normalized coords
--recode_weight 1.0        Case-flip reconstruction loss weight (0=off)
--batch_size 52            Training batch size
```

#### test
```
--model_dir data/models
--test_data_dir data/test
--output_dir data/results
--device auto|cpu|cuda
```

#### atlas
```
--model_dir data/models
--test_data_dir data/test
--output data/atlas.html    Self-contained HTML file
--device auto|cpu|cuda
```

Generates an interactive attention atlas — a single HTML file with Canvas-based Gaussian-splat heatmaps for all 52 letters across all fonts. The grid view shows averaged attention across fonts; clicking a letter drills down into per-font fixation patterns. Controls: heatmap/path toggle, upper/lower/both filter, opacity slider. Cell borders show correctness: green = all fonts correct, yellow = some wrong, red = all wrong.

#### check_attention
```
--data_dir data/letters    Dataset to check against
--n_epochs 10              Diagnostic epochs to run
--device auto|cpu|cuda
--guide_weight 8.0         Guide weight to test
--blur_sigma_ratio 0.16    Blur ratio to test
```

#### visualize
```
--model_dir data/models
--data_dir data/letters
--output_dir data/visualizations
--device auto|cpu|cuda
```

### Bigram commands

#### generate_bigrams
```
--num_variants 20          Noisy copies per bigram
--noise_level 0.01         Gaussian noise std
--output_dir data/bigrams
--fonts default            Font spec: "all", "default", or comma-separated names
```

#### generate_bigrams_test
```
--output_dir data/bigram_test
--fonts default
```

#### train_bigrams
```
--data_dir data/bigrams
--epochs 100
--save_dir data/bigram_models
--checkpoint_interval 10
--n_glimpses 15            More glimpses for wider image
--device auto|cpu|cuda
--resume PATH              Resume from checkpoint
--guide_weight 8.0
--batch_size 32
--scaffold_epochs 200      Epochs to anneal temporal attention scaffold (0=off)
--transfer PATH            Single-letter .pth/.pth.gz for transfer learning
```

#### test_bigrams
```
--model_dir data/bigram_models
--test_data_dir data/bigram_test
--output_dir data/bigram_results
--device auto|cpu|cuda
```

#### bigram_atlas
```
--model_dir data/bigram_models
--test_data_dir data/bigram_test
--output data/bigram_atlas.html
--device auto|cpu|cuda
```

Same concept as the single-letter atlas but for 200 bigrams on 192x128 canvases. Grid flows with auto-fill layout. Cell borders show correctness: green = both letters correct, yellow = one correct, red = neither.

#### check_bigram_attention
```
--data_dir data/bigrams
--n_epochs 10
--device auto|cpu|cuda
```

## Makefile

All targets accept overridable variables:

```bash
make train EPOCHS=200 DEVICE=cuda BATCH=52
make generate LETTERS=A-Z VARIANTS=10 NOISE=0.2 FONTS=all
make train-bigrams EPOCHS=300 DEVICE=cuda TRANSFER=runs/v3-multi-font-cosine/model_final.pth.gz
```

| Variable | Default | Description |
|----------|---------|-------------|
| `DEVICE` | auto | auto, cpu, or cuda |
| `EPOCHS` | 100 | Training epochs |
| `CKPT` | 50 | Checkpoint interval |
| `LETTERS` | Aa-Zz | Letter range for generation |
| `VARIANTS` | 20 | Noisy variants per letter |
| `NOISE` | 0.1 | Gaussian noise level |
| `FONTS` | all | Font spec: all, default, or comma-separated names |
| `BATCH` | 52 | Training batch size |
| `GUIDE` | 8.0 | Attention guide loss weight |
| `SCAFFOLD` | 200 | Bigram scaffold annealing epochs |
| `TRANSFER` | *(empty)* | Path to single-letter model for bigram transfer learning |

### Single-letter targets

| Target | Description |
|--------|-------------|
| `make generate` | Generate training data |
| `make generate-test` | Generate clean test data |
| `make train` | Train from scratch |
| `make resume` | Resume from last checkpoint |
| `make test` | Run test evaluation |
| `make atlas` | Generate interactive attention atlas (HTML) |
| `make check-attention` | Quick attention diagnostic (PASS/FAIL) |
| `make visualize` | Generate attention visualizations |
| `make archive NAME=...` | Compress + archive model to `runs/` |

### Bigram targets

| Target | Description |
|--------|-------------|
| `make generate-bigrams` | Generate bigram training data |
| `make generate-bigrams-test` | Generate clean bigram test data |
| `make train-bigrams` | Train bigram model (supports `TRANSFER=`) |
| `make resume-bigrams` | Resume bigram training |
| `make test-bigrams` | Run bigram test evaluation |
| `make bigram-atlas` | Generate bigram attention atlas (HTML) |
| `make check-bigram-attention` | Quick bigram attention diagnostic |
| `make archive-bigrams NAME=...` | Compress + archive bigram model to `runs/` |

### Lifecycle targets

| Target | Description |
|--------|-------------|
| `make build` | Build Docker image |
| `make up` / `down` / `restart` | Docker container lifecycle |
| `make logs` | Follow container logs |
| `make shell` | Shell into container |

## Project Structure

```
fbrl/
├── vision_training.py      # CLI entry point — subcommand dispatch
├── fbrl/                   # Core package
│   ├── __init__.py         # Device resolution helper
│   ├── model.py            # VisionModel, BigramVisionModel, GlimpseSensor, AttentionController
│   ├── data.py             # LetterDataset, BigramDataset, data generation
│   ├── train.py            # Training loops (single-letter + bigram)
│   ├── evaluate.py         # Testing, visualization, atlas generation
│   └── losses.py           # Loss functions (attention guide, diversity, hit rate)
├── Dockerfile              # python:3.10-slim + torch 2.5.1 + fonts
├── docker-compose.yml      # GPU-enabled container with data volume
├── Makefile                # Pipeline shortcuts (all targets accept variable overrides)
├── thoughts/               # Research notes and per-run results
├── runs/                   # Archived models (Git LFS-tracked .pth.gz files)
│   ├── v1-single-font/
│   ├── v2-multi-font/
│   ├── v3-multi-font-cosine/
│   └── v4-bigram-transfer/
└── data/                   # Generated data (Docker volume mount, not in git)
    ├── letters/            # Single-letter training PNGs + metadata.json
    ├── test/               # Single-letter clean test PNGs
    ├── models/             # Single-letter checkpoints + training_metrics.png
    ├── bigrams/            # Bigram training PNGs + metadata.json
    ├── bigram_test/        # Bigram clean test PNGs
    ├── bigram_models/      # Bigram checkpoints + training_metrics.png
    ├── atlas.html          # Single-letter attention atlas (make atlas)
    └── bigram_atlas.html   # Bigram attention atlas (make bigram-atlas)
```

## Design Decisions

**Why foveal-only (no global CNN)?** The model must learn an active perception strategy. A global CNN would give it the answer for free. With a 12x12 window, the model *must* decide where to look — attention is not optional, it's the only channel.

**Why not force higher hit rate?** At ~40% hit rate the model achieves high accuracy by sampling diagnostic features. Forcing fixations onto every stroke would produce outline tracing — wasteful and unlikely to generalize. The current behavior (selective, structural reading) is more biologically plausible.

**Why evaluate attention on clean images?** With noise_level=0.1, noisy images have white speckle everywhere. The attention guide would blur those noise pixels into a diffuse glow, and hit rate would count noise hits as successes. Clean images give honest signals.

**Why ratio-based blur sigma?** The attention guide uses a Gaussian blur to create a "scent trail" around letter strokes. At 96x96, `blur_sigma=15` worked. At 128x128, the same absolute value broke attention completely (0% hit rate). Expressing sigma as a fraction of image size (0.16) makes it auto-scale. This is critical for bigrams (192x128) and future work where the MetaController will feed crops of varying sizes to the attention system.

**Why separate letter and case classifiers?** Letter identity (A-Z) and case (upper/lower) are orthogonal concepts. A single 52-class classifier would conflate them. Separate heads let the latent space learn abstract letter identity that generalizes across case — 'a' and 'A' share the same letter class but differ in case.

**Why recode loss?** Forces the latent space to capture letter identity independently of case. If the same latent can decode both 'a' and 'A' (conditioned on the case label), the representation is truly case-invariant. This is a stepping stone toward abstract symbol understanding.

**Why transfer learning for bigrams?** The single-letter encoder already knows how to read letterforms across 11 fonts. Rather than learning from scratch on a harder task (wider canvas, two letters), the bigram model inherits the encoder weights and only needs to learn sequential scanning and dual-position classification. This cut training time significantly and avoided attention collapse during early epochs.

**Why a temporal scaffold?** Without guidance, the attention controller on a 192x128 canvas tends to fixate randomly or collapse to center. The scaffold provides a gentle left-then-right bias during early training, then anneals to zero so the model must discover its own scanning strategy. This is analogous to training wheels — necessary early, removed once balance is learned.

## Roadmap

The research goal is to scale foveal attention from character recognition toward reading:

1. **Multi-font** — Same letters, 11 fonts (serif, sans, mono, narrow, bold). Tests whether attention strategies generalize across visual styles. *Done — 100% across all fonts.*
2. **Bigrams** — 200 common English bigrams on wider canvases. The attention must scan left-to-right and segment characters. *Done — 97% with transfer learning.*
3. **Multi-font bigrams** — Can the bigram model generalize across fonts like the single-letter model did?
4. **Words** — Variable-length strings with a language model prior. Tests whether the model skips predictable letters, fixates word centers, and spends more glimpses on rare words (all human reading behaviors).
5. **Meta-attention** — A coarse controller that finds word boundaries (whitespace, line breaks) and deploys the fine letter-reader within each region. Hierarchical saccade planning.
6. **Multimodal** — Integrate audio to provide top-down priors (syllable boundaries, word predictions) that guide visual attention during reading.
