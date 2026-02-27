# FBRL — Feedback Recursive Loop

Can a recurrent foveal attention mechanism learn to "read" text by placing strategic fixations — the way a human eye scans a page?

This project trains a vision model that sees the world through a tiny patch window (12x12 pixels — under 1% of the image). A GRU-based controller decides where to look next, building up a latent representation over a sequence of glimpses. The model must learn *where* to look, not just *what* it sees. The system scales from single characters through bigrams to full 4-letter words, progressively forcing genuine sequential reading behavior.

### Why "Feedback Recursive Loop"?

The name describes the core learning mechanism: the model encodes an image into a latent, decodes it to reconstruct the input, then **recodes** it — decodes the same latent under a different condition (e.g., flipped case). Each recode path adds a constraint that feeds back into the latent space: "can I decode this as 'a' *and* as 'A'?" If both succeed, the latent has captured abstract letter identity. If not, the error reshapes everything — the encoder, the attention policy, the latent geometry.

This extends naturally as the project scales. At word level, the feedback loop becomes: encode a word, decode it, recode it (different font? different case? different modality — audio?). Each recode direction forces the latent toward more abstract, transferable representations. The loop is the learning mechanism, not just an architectural detail.

## Key Results

**Single letters** — 100% letter and case accuracy across 52 classes (Aa-Zz) and 11 fonts, with pixel-perfect case recoding (encode 'a', decode as 'A').

**Bigrams** — Two-phase scan/read architecture on 128x128 canvas. Scan phase (5 wide glimpses) maps the spatial layout; read phase (6 focused glimpses) identifies individual letters. 98-99% both-correct on 200 common English bigrams. Key finding: 2 letters on a 128px canvas is too cheatable — the 12x12 foveal window can see enough from near-center to read holistically, even with split diversity forcing horizontal scan spread.

**Words** — New: 4-letter words on 256x128 canvas with prescribed left-to-right scan. At this scale, holistic cheating is geometrically impossible — the model *must* sequentially fixate on each letter. Architecture is complete and ready for training.

The attention patterns show structural reading — not brute-force tracing:

- **T/t**: scans the crossbar junction area
- **O/o**: fixates the negative space inside the ring
- **A**: hits the apex, crossbar, and legs
- **a**: explores the bowl and stem — different pattern from A
- **"th"**: scans left letter, then right letter — develops sequential reading

Only ~40% of fixations land directly on letter pixels. The model is selective — it samples diagnostic features (junctions, endpoints, crossbars, negative space) rather than tracing outlines.

## What is Foveal Attention?

It's modeled on how your eye actually works.

**The biological reality:** Your retina isn't uniform. The **fovea** is a tiny spot at the center (~1-2 degrees of visual angle — roughly your thumbnail at arm's length) packed with cone cells. That's the *only* part where you see sharp detail. Everything else (peripheral vision) is progressively blurry. You can detect motion and rough shapes in the periphery, but you can't read text or recognize fine features.

**Saccades:** To see anything in detail, your eye physically moves to aim the fovea at it. These rapid jumps are called saccades — you make 3-4 per second while reading. Between saccades, you're sampling one tiny high-res patch. Your brain stitches these samples into a coherent scene.

**The illusion:** You *feel* like you see the whole world in sharp detail. You don't. Your brain is doing a remarkable job of integrating a sparse sequence of tiny sharp snapshots into a unified percept.

### How this maps to the model

| Biology | Model |
|---------|-------|
| Fovea (tiny sharp patch) | `GlimpseSensor` — 12x12 pixel window (0.9% of 128x128; 0.4% of 256x128) |
| Peripheral vision | Nothing — the model is *even more constrained* than biology, it sees zero outside the patch |
| Saccades (eye movements) | `AttentionController` — GRU decides the next (x,y) fixation |
| Sequence of fixations | 10 glimpses (single letter), 11 (bigrams: 5 scan + 6 read), or 20 (words: 8 scan + 12 read) |
| Brain integrating across fixations | GRU hidden state accumulating information |
| Conscious perception | Latent vector (256-dim) — the final representation |

### Why it matters

A standard CNN sees the **entire image at once** — like having an eye where the whole retina is fovea. It doesn't need to decide where to look; everything is already there. That's computationally powerful but biologically implausible, and it doesn't scale well to very large images.

Foveal attention forces the model to develop a **strategy** for looking. It has to answer: given what I've seen so far, where should I look next? That's a fundamentally different problem than "process all pixels simultaneously."

Our model develops letter-specific scan strategies — it doesn't trace outlines, it goes straight for diagnostic features. That mirrors human reading: experienced readers don't look at every letter, they fixate on key features and fill in the rest from context.

## Architecture

### Single-letter model (`VisionModel`)

```
Input (128x128 grayscale) -> [noisy image, model only sees 12x12 patches]
                         |
          GlimpseSensor (extracts patch at fixation point)
                         |
          AttentionController (GRU decides next fixation)
                         |  x 10 glimpses
                    Latent (256-dim)
                |         |          |
         CNNDecoder   LetterCls   CaseCls
      (128x128 recon)  (26: A-Z)  (2: upper/lower)
```

The decoder is **case-conditioned**: it receives a case label (0.0=upper, 1.0=lower) concatenated to the latent vector. This lets the model decode the same latent into either case, enabling **recoding** — encode lowercase 'a', decode as uppercase 'A'.

### Bigram model (`BigramVisionModel`) — two-phase scan/read

```
Input (128x128 grayscale) -> [two lowercase letters, natural kerning]
                         |
    SCAN PHASE (5 glimpses, 12x18 wide patches)
          GlimpseSensor -> GRU -> location_head -> next (x,y)
          Purpose: map spatial layout, find letter positions
                         |  h carries forward
    READ PHASE (6 glimpses, 12x12 focused patches)
          GlimpseSensor -> GRU -> location_head -> next (x,y)
          Purpose: identify individual letters with sharp fovea
                         |
          CrossAttentionReadout (2 position query tokens)
              attends over READ hidden states only
                |                     |
         Pos1 Classifier       Pos2 Classifier
          (26: a-z)              (26: a-z)
                         |
                   BigramDecoder
                  (128x128 recon)
```

Key design: the scan phase uses **wide rectangular patches** (12x18) to capture more horizontal context, while the read phase uses **focused square patches** (12x12) for precise letter identification. A single GRU controller is shared across both phases — the hidden state carries spatial knowledge from scan into read. `CrossAttentionReadout` uses 2 learned position query tokens (initialized with left/right spatial bias) to attend over read-phase hidden states and produce per-position representations.

Split diversity pressure: `scan_vy=0.3` makes horizontal proximity expensive (forces horizontal spread during scan), while `read_vy=1.5` makes vertical proximity expensive (forces vertical exploration during read).

### Word model (`WordVisionModel`) — prescribed x-scan + free read

```
Input (256x128 grayscale) -> [4-letter word, centered]
                         |
    SCAN PHASE (8 glimpses, 12x18 wide patches)
          x PRESCRIBED: linear sweep [-0.75, +0.75] (8 evenly spaced)
          y LEARNED: GRU -> location_head -> tanh -> y component only
          Content head: nn.Linear(256, 1) -> "content here?" (BCE)
          Purpose: forced L->R scanning, learns vertical positioning
                         |  h carries forward
    READ PHASE (12 glimpses, 12x12 focused patches)
          Both x and y fully learned (GRU -> tanh -> full location)
          Purpose: fixate on individual letters, identify each
                         |
          CrossAttentionReadout (4 position query tokens)
              queries initialized at [-0.75, -0.25, +0.25, +0.75]
                |         |         |         |
             Pos1       Pos2      Pos3      Pos4
           Classifier  Classifier Classifier Classifier
            (26: a-z)   (26: a-z)  (26: a-z)  (26: a-z)
                         |
                    WordDecoder
                   (256x128 recon)
```

At 256x128, a 12x12 foveal window covers only 0.4% of the image — holistic cheating is geometrically impossible. The model *must* sequentially attend to each letter position.

**Prescribed x-scan**: During scan, x-coordinates follow a fixed left-to-right linear sweep. Only y is learned. This reflects real reading mechanics — humans sweep roughly left-to-right and only adjust vertically for line tracking. The GRU still processes each glimpse and updates its hidden state, so it builds a spatial map of where content is.

**Content detection**: A `nn.Linear(256, 1)` head on each scan hidden state predicts whether letter content exists at that location (BCE loss against sampled image intensity). This teaches the GRU to explicitly encode "I found a letter here" in its hidden state, so the read phase inherits content-aware representations.

**Isolation mask**: During training, randomly picks 1 of 4 letter positions per batch, masks out the other 3 letter stripes, runs the masked image through the model, and computes classification loss only on the exposed position. This forces the model to genuinely fixate on individual letters rather than relying on cross-position context. Conceptually similar to BERT's masked language modeling — random masking per batch averages out over many batches.

**Temporal scaffold**: Read-phase glimpses are divided into 4 equal temporal segments, each guided toward a horizontal stripe of the image (one per letter position). Anneals from full guidance to zero over the scaffold phase.

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
| **Recode** (MSE) | Flip the case label, decode same latent, compare to partner image (e.g., encode 'a' -> decode as 'A') |

### Bigram (two-phase)

```
total = recon + pos1_cls + pos2_cls
        + scan_guide * scan_attn + guide_weight * read_attn
        + diversity_weight * (scan_div + read_div)
        + mask_weight * masked_cls
```

| Term | Purpose |
|------|---------|
| **Scan attn** | Full-image Gaussian guide for scan-phase fixations (pulls y onto strokes) |
| **Read attn** | Temporal scaffold — divides read glimpses into temporal segments guided by left/right image halves |
| **Scan div** | Fixation diversity with `vy=0.3` (penalizes horizontal clustering) |
| **Read div** | Fixation diversity with `vy=1.5` (penalizes vertical clustering) |
| **Masked cls** | Randomly mask one half of the image, classify the visible letter. Forces per-position reading |

### Word (prescribed scan + free read)

```
total = recon + sum(pos_cls[1..4])
        + scan_guide * scan_attn + guide_weight * read_attn
        + diversity_weight * (scan_div + read_div)
        + content_weight * content_bce
        + isolation_weight * isolation_cls
```

| Term | Purpose |
|------|---------|
| **Pos cls** (x4) | Cross-entropy for each of 4 letter positions (26 classes each) |
| **Scan attn** | Full-image guide for scan phase (only affects learned y since x is prescribed) |
| **Read attn** | 4-stripe temporal scaffold — divides read glimpses into 4 segments, each guided by a horizontal quarter-stripe |
| **Content BCE** | Binary cross-entropy on scan hidden states — predicts whether letter content exists at each scan location |
| **Isolation cls** | Random 1-of-4 position masking — zeros 3 letter stripes, classifies exposed position only. Forces genuine per-letter fixation |

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
| **Div** | Fixation diversity (repulsion). Lower = fixation points more spread out. | 0.05-0.15 |
| **Hit** | % of fixations landing on actual letter pixels (diagnostic, not a loss). | 25-45% |
| **Recode** | MSE between case-flipped decode and partner clean image. Lower = better. | < 0.01 |
| **lr** | Current learning rate (CosineAnnealingLR). Decays from 0.001 to ~0 over the run. | -- |
| **[time]** | Wall time per epoch + estimated time remaining. | -- |

**Key benchmarks:** Ltr and Case start near their random baselines (3.26 and 0.69) and should drop steadily. Hit rate should climb to ~30% within the first 10-20 epochs — if it stays near 0%, the attention guide is misconfigured (run `make check-attention` to diagnose).

### Bigram

```
Epoch 5/150: Recon 0.0262  Pos1 2.5719 (4%)  Pos2 2.5564 (4%)  SAttn -0.1125  RAttn -0.0500  Div 0.1505  Hit 40%  lr 0.000100/0.001000  scaff 0.98  [8.2s  ETA 40m23s]
```

| Field | What it means | Good values |
|-------|--------------|-------------|
| **Pos1 / Pos2** | Cross-entropy + accuracy for each position. Random = 3.26 / 4%. | < 0.01 / > 98% |
| **SAttn** | Scan-phase attention guide loss. | < -0.10 |
| **RAttn** | Read-phase attention guide loss (scaffold-weighted). | < -0.10 |
| **Div** | Fixation diversity (scan + read combined). | 0.05-0.15 |
| **Hit** | % of fixations on letter pixels (diagnostic). | 25-45% |
| **lr** | Encoder / readout learning rates (separate param groups). | -- |
| **scaff** | Temporal scaffold strength (1.0 = full guidance, anneals to 0). | anneals to 0 |

### Word

```
Epoch 5/200: Recon 0.0312  P1 3.1000 (5%)  P2 3.0800 (5%)  P3 3.1200 (4%)  P4 3.0900 (5%)  SAttn -0.0800  RAttn -0.0200  Div 0.1800  Cont 0.6500  Iso 3.1500  Hit 35%  lr 0.000100/0.001000  scaff 0.97  [12.0s  ETA 39m00s]
```

| Field | What it means | Good values |
|-------|--------------|-------------|
| **P1-P4** | Cross-entropy + accuracy for each of 4 letter positions. Random = 3.26 / 4%. | < 0.1 / > 95% |
| **SAttn** | Scan attention guide (only affects y since x is prescribed). | < -0.10 |
| **RAttn** | Read attention guide (4-stripe temporal scaffold). | < -0.10 |
| **Div** | Fixation diversity (scan + read combined). | 0.05-0.20 |
| **Cont** | Content detection BCE loss (scan phase). | < 0.3 |
| **Iso** | Isolation mask CE — single-letter classification on masked image. Random = 3.26. | < 0.5 |
| **Hit** | % of fixations on letter pixels (diagnostic). | 25-45% |
| **lr** | Encoder / readout learning rates (separate param groups). | -- |
| **scaff** | Temporal scaffold strength (1.0 = full guidance, anneals to 0). | anneals to 0 |

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
make train-bigrams EPOCHS=150 DEVICE=cuda TRANSFER=data/models/model_final.pth

# Test
make test-bigrams DEVICE=cuda

# Generate bigram attention atlas
make bigram-atlas DEVICE=cuda

# Archive
make archive-bigrams NAME=v4-bigram-transfer EPOCHS=150
```

### Word pipeline

```bash
# Generate 4-letter word training + test data (200 words, default font)
make generate-words
make generate-words-test

# Train with transfer learning from single-letter model
make train-words EPOCHS=200 DEVICE=cuda TRANSFER=data/models/model_final.pth

# Test (per-position + all-correct accuracy)
make test-words DEVICE=cuda

# Generate word attention atlas
make word-atlas DEVICE=cuda

# Archive
make archive-words NAME=v1-word-prescribed EPOCHS=200
```

Tunable parameters:

```bash
# Disable isolation mask (saves ~2x forward pass compute)
make train-words ISOLATION=0

# Adjust content detection weight
make train-words CONTENT=0.5

# Custom glimpse counts
make train-words WORD_SCAN_GLIMPSES=10 WORD_READ_GLIMPSES=16
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
--n_scan_glimpses 5          Scan-phase glimpses (wide patches)
--n_read_glimpses 6          Read-phase glimpses (focused patches)
--scan_patch_size 12,18      Scan patch H,W
--read_patch_size 12         Read patch size (square)
--device auto|cpu|cuda
--resume PATH                Resume from checkpoint
--guide_weight 8.0           Read-phase guide weight
--scan_guide_weight 8.0      Scan-phase guide weight (defaults to --guide_weight)
--batch_size 32
--scaffold_ratio 0.67        Scaffold phase as fraction of total epochs
--scaffold_floor 0.0         Minimum scaffold weight after annealing
--transfer PATH              Single-letter .pth/.pth.gz for transfer learning
--mask_weight 0.5            Masked-half auxiliary loss (0=disabled)
--scan_vy 0.3                Scan diversity VY (<1 = horizontal spread)
--read_vy 1.5                Read diversity VY (>1 = vertical exploration)
--edge_weight 0.0            Edge exploration weight (pushes scan to image sides)
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

Same concept as the single-letter atlas but for 200 bigrams on 128x128 canvases. Grid flows with auto-fill layout. Cell borders show correctness: green = both letters correct, yellow = one correct, red = neither.

#### check_bigram_attention
```
--data_dir data/bigrams
--n_epochs 10
--device auto|cpu|cuda
```

### Word commands

#### generate_words
```
--num_variants 20          Noisy copies per word
--noise_level 0.01         Gaussian noise std
--output_dir data/words
--fonts default            Font spec: "all", "default", or comma-separated names
```

#### generate_words_test
```
--output_dir data/word_test
--fonts default
```

#### train_words
```
--data_dir data/words
--epochs 200
--save_dir data/word_models
--checkpoint_interval 10
--n_scan_glimpses 8          Prescribed x-scan glimpses (wide patches)
--n_read_glimpses 12         Free read-phase glimpses (focused patches)
--scan_patch_size 12,18      Scan patch H,W
--read_patch_size 12         Read patch size (square)
--n_positions 4              Letter positions in each word
--device auto|cpu|cuda
--resume PATH                Resume from checkpoint
--guide_weight 8.0           Read-phase guide weight
--scan_guide_weight 8.0      Scan-phase guide weight (defaults to --guide_weight)
--batch_size 32
--scaffold_ratio 0.67        Scaffold phase as fraction of total epochs
--scaffold_floor 0.0         Minimum scaffold weight after annealing
--transfer PATH              Single-letter .pth/.pth.gz for transfer learning
--content_weight 0.5         Content detection BCE loss (0=disabled)
--isolation_weight 0.5       Isolation mask loss — masks 3 of 4 letters (0=disabled)
--scan_vy 0.3                Scan diversity VY (<1 = horizontal spread)
--read_vy 1.5                Read diversity VY (>1 = vertical exploration)
--edge_weight 0.0            Edge exploration weight (unnecessary with prescribed x)
```

#### test_words
```
--model_dir data/word_models
--test_data_dir data/word_test
--output_dir data/word_results
--device auto|cpu|cuda
```

#### word_atlas
```
--model_dir data/word_models
--test_data_dir data/word_test
--output data/word_atlas.html
--device auto|cpu|cuda
```

Same concept as the bigram atlas but for 200 four-letter words on 256x128 canvases. 2:1 aspect ratio cells. Cell borders show correctness: green = all 4 letters correct, yellow = some correct, red = none correct.

## Makefile

All targets accept overridable variables:

```bash
make train EPOCHS=200 DEVICE=cuda BATCH=52
make generate LETTERS=A-Z VARIANTS=10 NOISE=0.2 FONTS=all
make train-bigrams EPOCHS=150 DEVICE=cuda TRANSFER=data/models/model_final.pth
make train-words EPOCHS=200 DEVICE=cuda TRANSFER=data/models/model_final.pth ISOLATION=0.5
```

| Variable | Default | Description |
|----------|---------|-------------|
| `DEVICE` | auto | auto, cpu, or cuda |
| `EPOCHS` | 100 | Training epochs |
| `CKPT` | 50 | Checkpoint interval |
| `LETTERS` | Aa-Zz | Letter range for generation |
| `VARIANTS` | 20 | Noisy variants per letter/bigram/word |
| `NOISE` | 0.1 | Gaussian noise level |
| `FONTS` | all | Font spec: all, default, or comma-separated names |
| `BATCH` | 52 | Training batch size |
| `GUIDE` | 8.0 | Attention guide loss weight |
| `SCAFFOLD_RATIO` | 0.67 | Scaffold phase as fraction of total epochs |
| `SCAFFOLD_FLOOR` | 0.0 | Minimum scaffold weight after annealing |
| `TRANSFER` | *(empty)* | Path to single-letter model for transfer learning |
| `SCAN_GLIMPSES` | 5 | Bigram scan glimpses |
| `READ_GLIMPSES` | 6 | Bigram read glimpses |
| `SCAN_PATCH` | 12,18 | Scan patch size (H,W) |
| `READ_PATCH` | 12 | Read patch size (square) |
| `SCAN_GUIDE` | *(empty)* | Scan-phase guide weight (defaults to GUIDE) |
| `MASK` | 0.5 | Bigram masked-half loss weight |
| `SCAN_VY` | 0.3 | Scan diversity VY |
| `READ_VY` | 1.5 | Read diversity VY |
| `VY` | 1.0 | Single-letter diversity VY |
| `EDGE` | 0.0 | Edge exploration weight |
| `CONTENT` | 0.5 | Word content detection weight |
| `ISOLATION` | 0.5 | Word isolation mask weight |
| `WORD_SCAN_GLIMPSES` | 8 | Word scan glimpses (prescribed x) |
| `WORD_READ_GLIMPSES` | 12 | Word read glimpses (free) |

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

### Word targets

| Target | Description |
|--------|-------------|
| `make generate-words` | Generate 4-letter word training data (256x128) |
| `make generate-words-test` | Generate clean word test data |
| `make train-words` | Train word model (prescribed scan, supports `TRANSFER=`) |
| `make resume-words` | Resume word training |
| `make test-words` | Run word test evaluation (per-position + all-correct) |
| `make word-atlas` | Generate word attention atlas (HTML) |
| `make archive-words NAME=...` | Compress + archive word model to `runs/` |

### Lifecycle targets

| Target | Description |
|--------|-------------|
| `make build` | Build Docker image |
| `make up` / `down` / `restart` | Docker container lifecycle |
| `make logs` | Follow container logs |
| `make shell` | Shell into container |

**Important:** Always `make restart` after code changes — Docker caches old code.

## Project Structure

```
fbrl/
+-- vision_training.py      # CLI entry point -- subcommand dispatch
+-- fbrl/                   # Core package
|   +-- __init__.py         # Device resolution helper
|   +-- model.py            # VisionModel, BigramVisionModel, WordVisionModel,
|   |                       #   GlimpseSensor, AttentionController, CrossAttentionReadout
|   +-- data.py             # LetterDataset, BigramDataset, WordDataset, data generation
|   +-- train.py            # Training loops (single-letter + bigram)
|   +-- _word_train.py      # Word training loop (train_word_model)
|   +-- evaluate.py         # Testing, visualization, atlas generation (single-letter + bigram)
|   +-- _word_eval.py       # Word evaluation + atlas (test_word_model, generate_word_atlas)
|   +-- losses.py           # Loss functions (attention guide, diversity, hit rate,
|                           #   two_phase_attention_loss, word_attention_loss)
+-- Dockerfile              # python:3.10-slim + torch 2.5.1 + fonts
+-- docker-compose.yml      # GPU-enabled container with data volume
+-- Makefile                # Pipeline shortcuts (all targets accept variable overrides)
+-- thoughts/               # Research notes and per-run results
+-- runs/                   # Archived models (Git LFS-tracked .pth.gz files)
|   +-- v1-single-font/
|   +-- v2-multi-font/
|   +-- v3-multi-font-cosine/
|   +-- v4-bigram-transfer/
+-- data/                   # Generated data (Docker volume mount, not in git)
    +-- letters/            # Single-letter training PNGs + metadata.json
    +-- test/               # Single-letter clean test PNGs
    +-- models/             # Single-letter checkpoints + training_metrics.png
    +-- bigrams/            # Bigram training PNGs + metadata.json
    +-- bigram_test/        # Bigram clean test PNGs
    +-- bigram_models/      # Bigram checkpoints + training_metrics.png
    +-- words/              # Word training PNGs + metadata.json (256x128)
    +-- word_test/          # Word clean test PNGs
    +-- word_models/        # Word checkpoints + training_metrics.png
    +-- atlas.html          # Single-letter attention atlas
    +-- bigram_atlas.html   # Bigram attention atlas
    +-- word_atlas.html     # Word attention atlas
```

## Design Decisions

**Why foveal-only (no global CNN)?** The model must learn an active perception strategy. A global CNN would give it the answer for free. With a 12x12 window, the model *must* decide where to look — attention is not optional, it's the only channel.

**Why not force higher hit rate?** At ~40% hit rate the model achieves high accuracy by sampling diagnostic features. Forcing fixations onto every stroke would produce outline tracing — wasteful and unlikely to generalize. The current behavior (selective, structural reading) is more biologically plausible.

**Why evaluate attention on clean images?** With noise_level=0.1, noisy images have white speckle everywhere. The attention guide would blur those noise pixels into a diffuse glow, and hit rate would count noise hits as successes. Clean images give honest signals.

**Why ratio-based blur sigma?** The attention guide uses a Gaussian blur to create a "scent trail" around letter strokes. At 96x96, `blur_sigma=15` worked. At 128x128, the same absolute value broke attention completely (0% hit rate). Expressing sigma as a fraction of image size (0.16) makes it auto-scale. This is critical for bigrams (128x128) and words (256x128) where image dimensions vary.

**Why separate letter and case classifiers?** Letter identity (A-Z) and case (upper/lower) are orthogonal concepts. A single 52-class classifier would conflate them. Separate heads let the latent space learn abstract letter identity that generalizes across case — 'a' and 'A' share the same letter class but differ in case.

**Why recode loss?** Forces the latent space to capture letter identity independently of case. If the same latent can decode both 'a' and 'A' (conditioned on the case label), the representation is truly case-invariant. This is a stepping stone toward abstract symbol understanding.

**Why transfer learning for bigrams and words?** The single-letter encoder already knows how to read letterforms across 11 fonts. Rather than learning from scratch on a harder task, the bigram/word model inherits the encoder weights and only needs to learn sequential scanning and multi-position classification. The transfer scaffold freezes read_sensor + classifiers during early training so the scan sensor and controller can adapt without destroying pretrained weights.

**Why a temporal scaffold?** Without guidance, the attention controller on a wide canvas tends to fixate randomly or collapse to center. The scaffold provides gentle spatial guidance during early training, then anneals to zero so the model must discover its own scanning strategy. This is analogous to training wheels — necessary early, removed once balance is learned.

**Why two phases (scan + read)?** Humans use two distinct eye movement strategies: fast saccades to survey a scene (scan) and slower fixations to process detail (read). Wide rectangular patches during scan capture more horizontal context to map letter positions. Focused square patches during read enable precise character identification. The shared GRU hidden state carries spatial knowledge from scan into read.

**Why prescribed x-scan for words?** Two letters on 128px (bigrams) is cheatable — the model can read holistically from near-center. Four letters on 256px makes holistic reading geometrically impossible, but the scan phase still needs to cover the full width. Prescribing x as a linear sweep guarantees complete coverage and mirrors how human reading proceeds roughly left-to-right. Only y is learned (for line tracking). The read phase is fully free.

**Why isolation masking?** Even with prescribed scan, the model could learn to classify letters using cross-position context leaking through the readout. Isolation masking removes this shortcut by zeroing out 3 of 4 letter positions and forcing classification of only the exposed letter. This is the same principle as BERT's masked language modeling — random masking per batch averages out over many batches and produces robust per-position representations.

**Why content detection?** The scan phase sweeps left-to-right but needs to know where letters actually are. The content detection head (BCE loss on scan hidden states) teaches the GRU to encode "letter content is here" in its hidden state. The read phase inherits this content-aware state and can target its free fixations more efficiently.

## Roadmap

The research goal is to scale foveal attention from character recognition toward reading:

1. **Multi-font single letters** — Same letters, 11 fonts (serif, sans, mono, narrow, bold). Tests whether attention strategies generalize across visual styles. *Done — 100% across all fonts.*
2. **Bigrams** — 200 common English bigrams with two-phase scan/read. Tests sequential reading and per-position classification. *Done — 98-99% with transfer learning. Key finding: 128px canvas is too small for 2 letters to force genuine sequential reading.*
3. **4-letter words** — 200 common words on 256x128 canvas with prescribed x-scan, content detection, and isolation masking. Forces genuine sequential reading at a scale where holistic cheating is impossible. *Architecture complete, ready for training.*
4. **Multi-font words** — Can the word model generalize across fonts?
5. **Variable-length words** — Mixed word lengths with a language model prior. Tests whether the model skips predictable letters, fixates word centers, and spends more glimpses on rare words (all human reading behaviors).
6. **Meta-attention** — A coarse controller that finds word boundaries (whitespace, line breaks) and deploys the fine letter-reader within each region. Hierarchical saccade planning.
7. **Multimodal** — Integrate audio to provide top-down priors (syllable boundaries, word predictions) that guide visual attention during reading.
