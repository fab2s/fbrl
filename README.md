# FBRL — Feedback Recursive Loop

Can a recurrent foveal attention mechanism learn to "read" text by placing strategic fixations — the way a human eye scans a page?

This project trains a vision model that sees the world through a tiny patch window (12x12 pixels — under 1% of the image). A GRU-based controller decides where to look next, building up a latent representation over a sequence of glimpses. The model must learn *where* to look, not just *what* it sees. The system scales from single characters through bigrams to full 4-letter words, progressively forcing genuine sequential reading behavior.

This is an active research project, built from scratch in a few days by someone new to deep learning and Python — driven by curiosity about whether a biologically-inspired attention mechanism can develop reading strategies that mirror human eye movements. The entire codebase, architecture decisions, and training pipeline were designed from first principles.

### Why "Feedback Recursive Loop"?

The name describes the core learning mechanism: the model encodes an image into a latent, decodes it to reconstruct the input, then **recodes** it — decodes the same latent under a different condition (e.g., flipped case). Each recode path adds a constraint that feeds back into the latent space: "can I decode this as 'a' *and* as 'A'?" If both succeed, the latent has captured abstract letter identity. If not, the error reshapes everything — the encoder, the attention policy, the latent geometry.

This extends naturally as the project scales. At word level, the feedback loop becomes: encode a word, decode it, recode it (different font? different case? different modality — audio?). Each recode direction forces the latent toward more abstract, transferable representations. The loop is the learning mechanism, not just an architectural detail.

## Key Results

**Single letters** — 100% letter and case accuracy across 52 classes (Aa-Zz) and 11 fonts, with pixel-perfect case recoding (encode 'a', decode as 'A').

**Bigrams** — Two-phase scan/read architecture on 128x128 canvas. 98-99% both-correct on 200 common English bigrams with transfer learning.

**Words** — 4-letter words on 256x128 canvas with prescribed left-to-right scan. 100% accuracy on all 4 positions with single-font training. Multi-head optimization (v2, in progress) showing faster convergence with separate gradient paths for attention, classification, and reconstruction.

The attention patterns show structural reading — not brute-force tracing:

- **T/t**: scans the crossbar junction area
- **O/o**: fixates the negative space inside the ring
- **A**: hits the apex, crossbar, and legs
- **a**: explores the bowl and stem — different pattern from A
- **"th"**: scans left letter, then right letter — develops sequential reading

Only ~40% of fixations land directly on letter pixels. The model is selective — it samples diagnostic features (junctions, endpoints, crossbars, negative space) rather than tracing outlines.

## Research Findings

Each stage of the project revealed something that shaped the next. These aren't just results — they're the iterative reasoning that drove the architecture.

### Structural reading emerges naturally

The model develops letter-specific scan strategies without being taught them. It doesn't trace outlines or visit every pixel — it goes straight for diagnostic features. 'T' gets the crossbar junction, 'O' gets the negative space inside the ring, 'A' gets the apex and legs. Different scan patterns for 'a' versus 'A' despite sharing the same letter identity. This mirrors human reading: experienced readers fixate on key features and fill in the rest from context.

### Canvas scale determines reading strategy

Two letters on 128px (bigrams) is geometrically cheatable: the 12x12 foveal window, covering ~0.9% of the image, can see enough from near-center to read both letters holistically. Even with diversity pressure forcing horizontal spread, the model finds holistic shortcuts. Scaling to 4 letters on 256px makes holistic reading impossible — the foveal window covers only 0.4% of the canvas, and the model *must* sequentially fixate on each letter. This finding drove the transition from bigrams to the word architecture.

### Isolation testing requires the right abstraction

Masking 3 of 4 letter stripes on the full 256px canvas (testing "can you find AND read a single letter?") stalls at ~1.0 cross-entropy — barely better than random. The task is too hard: mostly-empty canvas, tiny fovea, and the model has to both locate and identify. Feeding 128x128 single-letter images directly to the word model (testing only "can you read?") drops isolation CE to 0.04. The lesson: decompose compound capabilities and test them independently.

### Gradient separation accelerates convergence

Summing all losses into a single backward pass blurs the gradient signal — the attention controller gets mixed feedback from classification, reconstruction, and attention losses simultaneously. Splitting into 3 separate backward passes with `backward(inputs=...)` gives each component clean gradients: the controller learns *where* to look from attention losses only, the readout learns *what* to read from classification only, the decoder learns to reconstruct from MSE only. Multi-head optimization shows faster convergence in early experiments.

### Transfer scaffold prevents catastrophic forgetting

The single-letter model already knows how to read letterforms. Naively unfreezing everything on the word task destroys this knowledge — the untrained scan sensor sends garbage through the shared GRU, corrupting the pre-trained read sensor's representations. The solution: freeze read_sensor + classifiers for 67% of training while the scan system learns left-to-right mechanics. Then gradually unfreeze with very low learning rates. The scaffold is training wheels — necessary early, removed once balance is learned.

### Scan trajectories reveal learned geometry

The prescribed x-scan (fixed left-to-right sweep with learned y) develops a slight diagonal trajectory: upper-left to lower-right. This isn't emergent intelligence — it's the model learning that lowercase letter bodies sit below the vertical center of the canvas. The y-component adapts to where the content actually is. This suggests that for mixed-case training, initializing scan y above center could accelerate convergence.

### Interior positions learn faster than edges

In 4-letter word training, P2 and P3 (interior letters) converge before P1 and P4 (edges). Likely because interior positions have richer context from neighboring letters during cross-attention readout, and the prescribed scan passes through their region more centrally.

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

## Requirements

**PyTorch 2.5.1** — Pinned to this version because training runs on a Pascal-era GPU (GTX 1080 Ti, compute capability 6.1). PyTorch 2.6+ dropped CUDA support for Pascal. If you have an Ampere or newer card, feel free to bump the version in the Dockerfile.

If anyone feels like donating a modern GPU to the cause, the latent space would be eternally grateful.

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
       VisualDecoder  LetterCls   CaseCls
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
                  VisualDecoder
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
                   VisualDecoder
                   (256x128 recon)
```

At 256x128, a 12x12 foveal window covers only 0.4% of the image — holistic cheating is geometrically impossible. The model *must* sequentially attend to each letter position.

**Dynamic scan count**: When the model processes narrower images (e.g., 128x128 for isolation testing), the scan glimpse count scales automatically: `n_scan = max(1, round(base * (width/256)^1.5))`. This gives 3 scan glimpses at 128px instead of 8 at 256px — matching the intuition that wider images need disproportionately more scanning.

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
| **Isolation cls** | 128x128 single-letter images fed to the word model — forces genuine per-letter reading ability independent of multi-letter context |

**Multi-head optimization** (v2): Instead of summing all losses, 3 separate backward passes target specific components: attention losses train only the controller and sensors, classification losses train only the readout and classifiers, reconstruction loss trains only the decoder. This prevents gradient cross-contamination and accelerates convergence.

## Design Journey

Each design decision was driven by something that didn't work in the previous stage.

**Single letters worked immediately** — 10 glimpses, 12x12 patches, encode-decode-recode. The surprise: only 40% of fixations land on letter pixels, yet the model reaches 100% accuracy. It learned to sample diagnostic features rather than trace outlines. This set the expectation: don't force high hit rates, let the model develop its own strategy.

**Bigrams revealed cheating** — Scaling to two letters on 128px should require sequential reading. It didn't. The 12x12 foveal window on a 128px canvas covers enough area that the model can read both letters from a few central fixations. Even split diversity pressure (horizontal scan spread, vertical read spread) wasn't enough to force genuine sequential behavior. This was the critical finding: the *geometry* of the task has to make cheating impossible. You can't force reading strategy through loss design alone.

**Words demanded prescribed scanning** — Four letters on 256px finally makes holistic reading geometrically impossible. But giving the model full control over both x and y during scanning led to collapse — random fixations or center-camping. The solution: prescribe x as a left-to-right sweep (mimicking how humans read) and only let y be learned. This teaches L-to-R mechanics while the model learns vertical positioning on its own. The read phase is then fully free — by then, the GRU hidden state has a spatial map of where content is.

**Transfer learning needed protection** — The single-letter model's read sensor and classifiers already know how to read. Loading them into the word model and training everything at once destroyed that knowledge within a few epochs. The scaffold strategy: freeze pretrained components, let the new scan sensor learn, then gradually unfreeze. This is analogous to how human learning works — you don't unlearn reading when you learn speed-reading.

**Isolation masking failed, then succeeded differently** — Canvas masking (zero 3 of 4 stripes) tests two things at once: "can you find the letter?" and "can you read it?" The model couldn't converge. Feeding single-letter 128x128 images directly to the word model tests only reading — and converges to 0.04 CE. Test one thing at a time.

**Gradient mixing slowed convergence** — The attention controller was getting conflicting signals: "move here for better classification" vs "move here for better reconstruction" vs "spread out for diversity." Splitting into 3 separate backward passes, each targeting only its relevant parameters, gives each system a clean learning signal.

## Results

Detailed analysis for each training run:

**Letters** (`runs/letters/`):
- [v1: single font, 200 epochs](runs/letters/v1-single-font/results.md) — 100% letter, 100% case, pixel-perfect recode
- [v2: 11 fonts, 50 epochs](runs/letters/v2-multi-font/results.md) — 99.5% letter, 99.7% case (guide_weight=8.0)
- [v3: 11 fonts, CosineAnnealingLR, 100 epochs](runs/letters/v3-cosine/results.md) — 100% letter, 100% case, all 11 fonts
- [v4: vertical diversity (VY=1.5), 100 epochs](runs/letters/v4-vertical-diversity/results.md) — 100% with better attention spread
- [v5: scan phase, 100 epochs](runs/letters/v5-scan/results.md) — 100%, pretrained scan_sensor + content_head

**Bigrams** (`runs/bigrams/`):
- [v1: transfer learning, 300 epochs](runs/bigrams/v1-transfer/results.md) — 97% both-correct

**Words** (`runs/words/`):
- [v1: prescribed x-scan, 200 epochs](runs/words/v1-prescribed/results.md) — 100% all 4 positions
- [v2: multi-head + isolation, 200 epochs](runs/words/v2-multihead/results.md) — 99.5%, 128px isolation

## Quick Start

```bash
# Build and start the container
make build up

# === Single-letter pipeline ===
make generate                    # Training data (52 letters x 20 variants x 11 fonts)
make generate-test               # Clean test data
make train DEVICE=cuda           # Train (configs/letter.yaml defaults)
make test DEVICE=cuda            # Evaluate
make atlas DEVICE=cuda           # Interactive attention atlas (HTML)

# === Bigram pipeline ===
make generate-bigrams
make train-bigrams DEVICE=cuda TRANSFER=data/models/model_final.pth
make bigram-atlas DEVICE=cuda

# === Word pipeline ===
make generate-words
make train-words DEVICE=cuda TRANSFER=data/models/model_final.pth
make word-atlas DEVICE=cuda

# Override any config value at the command line
make train-words EPOCHS=300 BATCH=64 DEVICE=cuda

# Archive any trained model
make archive-words NAME=v2-multi-head

# Run unit tests (no GPU needed, fast)
make test-unit
```

Training parameters live in YAML config files (`configs/letter.yaml`, `configs/bigram.yaml`, `configs/word.yaml`). CLI args override config values. See [docs/usage.md](docs/usage.md) for full reference.

**Important:** Always `make restart` after code changes — Docker caches old code.

## Project Structure

```
fbrl/
+-- vision_training.py      # CLI entry point — subcommand dispatch
+-- configs/                # YAML training configs (one per model type)
|   +-- letter.yaml         #   Single-letter defaults (no scan phase)
|   +-- letter_scan.yaml    #   Single-letter with scan phase (3 scan + 10 read)
|   +-- bigram.yaml         #   Bigram defaults (5 scan + 6 read)
|   +-- word.yaml           #   Word defaults (6 scan + 20 read, grouped, multi-head, AMP)
|   +-- letter_motor.yaml   #   Motor letter defaults (3 scan + 10 read, trajectory decoder)
+-- fbrl/                   # Core package
|   +-- __init__.py         # Device resolution helper
|   +-- config.py           # ExperimentConfig dataclass, YAML load/save
|   +-- model.py            # VisionModel, BigramVisionModel, WordVisionModel,
|   |                       #   VisualDecoder, GlimpseSensor, AttentionController,
|   |                       #   CrossAttentionReadout, encode_scan_read()
|   +-- data.py             # LetterDataset, BigramDataset, WordDataset, data generation
|   +-- train.py            # Training loops (single-letter + bigram)
|   +-- _word_train.py      # Word training loop (train_word_model)
|   +-- training_utils.py   # Shared training infrastructure: LossTracker, TrainingLogger,
|   |                       #   checkpoint save/load, transfer learning, metrics plotting
|   +-- evaluate.py         # Testing, visualization, atlas generation (single-letter + bigram)
|   +-- _word_eval.py       # Word evaluation + atlas (test_word_model, generate_word_atlas)
|   +-- losses.py           # Loss functions (attention guide, diversity, hit rate,
|   |                       #   two_phase_attention_loss, word_attention_loss)
|   +-- motor.py            # MotorTraceDecoder, soft_render, trajectory extraction from TTF
|   +-- _motor_train.py     # Motor letter training loop (train_motor_model)
|   +-- _motor_eval.py      # Motor evaluation + atlas (test_motor_model, generate_motor_atlas)
+-- tests/                  # Unit tests (pytest, runs inside Docker)
|   +-- conftest.py         #   Shared fixtures (device, batch size, synthetic images)
|   +-- test_model.py       #   Forward pass shapes for all model types
|   +-- test_losses.py      #   All loss functions with synthetic tensors
|   +-- test_config.py      #   Config load/save/roundtrip
|   +-- test_data.py        #   Dataset word lists, font discovery
|   +-- test_motor.py       #   Trajectory extraction, soft render, motor decoder
|   +-- test_utils.py       #   LossTracker, checkpoint save/load, ETA formatting
+-- Dockerfile              # python:3.10-slim + torch 2.5.1 + fonts
+-- docker-compose.yml      # GPU-enabled container with data volume
+-- Makefile                # Pipeline shortcuts (all targets accept variable overrides)
+-- thoughts/               # Research notes (non-run-specific)
+-- runs/                   # Archived models + results, organized by type
|   +-- letters/            #   Single-letter experiments (v1-v5)
|   +-- bigrams/            #   Bigram experiments (v1)
|   +-- words/              #   Word experiments (v1-v2)
+-- data/                   # Generated data (Docker volume mount, not in git)
+-- docs/
    +-- glossary.md         # Deep learning terms and concepts
    +-- usage.md            # Full CLI reference and Makefile documentation
```

## Roadmap

The research goal is to scale foveal attention from character recognition toward reading:

1. **Multi-font single letters** — Same letters, 11 fonts (serif, sans, mono, narrow, bold). Tests whether attention strategies generalize across visual styles. *Done — 100% across all fonts.*
2. **Bigrams** — 200 common English bigrams with two-phase scan/read. Tests sequential reading and per-position classification. *Done — 98-99% with transfer learning. Key finding: 128px canvas is too small for 2 letters to force genuine sequential reading.*
3. **4-letter words** — 200 common words on 256x128 canvas with prescribed x-scan, content detection, and isolation testing. *Done — 100% all 4 positions (v1, single optimizer). v2 (multi-head optimization + 128px isolation) in progress.*
4. **Multi-font words** — Can the word model generalize across fonts?
5. **Variable-length words** — Mixed word lengths with a language model prior. Tests whether the model skips predictable letters, fixates word centers, and spends more glimpses on rare words (all human reading behaviors).
6. **Meta-attention** — A coarse controller that finds word boundaries (whitespace, line breaks) and deploys the fine letter-reader within each region. Hierarchical saccade planning.
7. **Multimodal** — Integrate audio to provide top-down priors (syllable boundaries, word predictions) that guide visual attention during reading.

## Reference

- [Research Hypotheses](thoughts/hypotheses.md) — Core intuitions, testable predictions, and the multimodal roadmap
- [Word Read Phase](thoughts/word_read_phase.md) — Why free read fixations degenerate and how scan-anchored groups fix it
- [Glossary](docs/glossary.md) — Deep learning terms, acronyms, and concepts as they appear in this project
- [Usage Reference](docs/usage.md) — Full CLI reference, training output format, Makefile documentation
