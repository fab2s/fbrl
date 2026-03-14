# FBRL — Feedback Recursive Loop

Can a recurrent foveal attention mechanism learn to "read" text by placing strategic fixations — the way a human eye scans a page?

This project trains a vision model that sees the world through a tiny patch window (12x12 pixels — under 1% of the image). A GRU-based controller decides where to look next, building up a latent representation over a sequence of glimpses. The model must learn *where* to look, not just *what* it sees.

This is an active research project — driven by curiosity about whether a biologically-inspired attention mechanism can develop reading strategies that mirror human eye movements.

The project has two implementations: the original **Python/PyTorch** prototype that proved the architecture (100% letter accuracy), and a **Rust/floDl** port that serves as the first real-world benchmark for [floDl](https://github.com/fab2s/flodl), a graph-native deep learning framework built on libtorch. A Go implementation (goDl) validated the graph API but [hit fundamental GC/VRAM limits](docs/go-retrospective.md).

### Why "Feedback Recursive Loop"?

The name describes the core learning mechanism: the model encodes an image into a latent, decodes it to reconstruct the input, then **recodes** it — decodes the same latent under a different condition (flipped case, pen trajectory, different modality). Each recode path adds a constraint that feeds back into the latent space: "can I decode this as 'a' *and* as 'A'?" If both succeed, the latent has captured abstract letter identity. If not, the error reshapes everything.

This extends naturally as the project scales. At word level, the feedback loop becomes: encode a word, decode it, recode it. Each recode direction forces the latent toward more abstract, transferable representations. The loop is the learning mechanism, not just an architectural detail.

## What is Foveal Attention?

Modeled on how your eye actually works. The **fovea** is a tiny spot at the center of your retina (~1-2 degrees of visual angle) — the only part where you see sharp detail. To see anything clearly, your eye physically jumps to aim the fovea at it (saccades — 3-4 per second while reading). Your brain stitches these sparse snapshots into a coherent scene. You *feel* like you see everything in sharp detail. You don't.

| Biology | Model |
|---------|-------|
| Fovea (tiny sharp patch) | `GlimpseSensor` — 12x12 pixel window |
| Peripheral vision | Nothing — even more constrained than biology |
| Saccades (eye movements) | `AttentionController` — GRU decides next (x,y) fixation |
| Brain integrating fixations | GRU hidden state accumulating information |
| Conscious perception | Latent vector — the final representation |

A standard CNN sees the entire image at once. Foveal attention forces the model to develop a **strategy** for looking — given what I've seen so far, where should I look next?

## Research Lines

The project has four experiment tracks, each building on discoveries from the previous:

### [Single Letters](docs/letters.md) — the foundation

100% accuracy across 52 classes (Aa-Zz) and 11 fonts with only **7 glimpses** (1 scan + 6 read) — 46% fewer than the initial 13-glimpse architecture. The model develops letter-specific scan strategies: 'T' gets the crossbar junction, 'O' gets the negative space, 'A' gets the apex and legs. Only ~40% of fixations land on letter pixels — it samples diagnostic features, not outlines.

### [Bigrams](docs/bigrams.md) — the geometry lesson

Two letters on 128x128. Achieved 98-99% accuracy but revealed a fundamental truth: **you can't force reading strategy through loss design alone**. The foveal window covers enough of the canvas for holistic shortcuts. The task geometry must make cheating impossible. This drove the transition to words.

### [Words](docs/words.md) — genuine sequential reading

4-letter words on 256x128. The foveal window covers 0.4% of the canvas — holistic reading is geometrically impossible. 100% accuracy on all 4 positions. Multi-head optimization (separate gradient paths for attention, classification, reconstruction) accelerates convergence. Interleaved scan-read failed due to global attention guide center-pull — led to the void repulsion insight.

### [Motor Traces](docs/motor.md) — learning to write

Read-Write-Render-Re-Read: the encoder produces a latent, a motor decoder writes a pen trajectory, a renderer draws it, the encoder re-reads it. If the re-read matches the original, the motor has learned to write. Currently training v3 from scratch with void repulsion and an enhanced loss stack (latent matching, render matching, centerline trajectories).

## Key Discoveries

These insights emerged iteratively — each shaped the next experiment.

**Self-scaffolding** — All losses active from epoch 1, no curriculum. Yet they sequence by difficulty: classification converges first (strong CE gradient), reconstruction follows, recode takes over last (requires latent factorization). Easy tasks bootstrap representations that hard tasks need. Intrinsic difficulty can replace explicit curriculum design. ([details](docs/letters.md#self-scaffolding-discovery))

**Scan = guide, reads = self-directed** — A global attention guide creates center-pull that's toxic for multi-position reading. The fix: blurred guide for scan only (long-range "find the content" signal), void repulsion for reads (local "don't stare at nothing" — zero gradient in deep void, active only at ink boundaries). ([details](docs/words.md#the-lesson-scan--guide-reads--self-directed))

**Flat read preserves GRU momentum** — Resetting the GRU's position between read groups causes catastrophic overfitting. The hidden state carrying spatial context from one glimpse to the next is essential. Position reset discards it. ([details](docs/letters.md#v6-fewer-glimpses))

**Gradient separation accelerates convergence** — Summing all losses blurs the gradient signal. Splitting into separate backward passes gives each component clean gradients: the controller learns *where* to look from attention losses only, the readout learns *what* to read from classification only. ([details](docs/words.md))

**Canvas scale determines reading strategy** — The geometry of the task must make cheating impossible. No amount of loss engineering can force sequential reading if the foveal window can see enough from center. ([details](docs/bigrams.md#the-canvas-scale-finding))

## What's Next

The architecture is converging toward a **semantic OCR** system:

- **Interleaved words v2**: retry with void repulsion reads, now validated on single letters
- **Multi-font words**: generalization across visual styles
- **Semantic layer**: learn letter transition probabilities and word-level priors. The cross-attention readout already produces per-position representations — a small model on their concatenation could learn "is this a real word?" as an auxiliary loss. Visual ambiguity resolved by language structure, the way human reading works.
- **Variable-length words**: mixed lengths with language model priors. Does the model skip predictable letters and spend more glimpses on rare words?
- **Meta-attention**: hierarchical saccade planning — a coarse controller finds word boundaries, deploys the fine letter-reader within each region

## Quick Start

### Rust/floDl (active development)

```bash
# All commands run inside Docker (libtorch is container-only)
make build                           # Build Docker image
make test                            # Run unit tests (CUDA)
make train-letter DATA=../python/data/letters EPOCHS=100
make train-letter SYNTHETIC=64 EPOCHS=2  # Quick smoke test
make shell                           # Interactive container shell
```

### Python/PyTorch (reference implementation)

```bash
make build up                        # Build and start Docker container

# Single-letter pipeline
make generate && make generate-test  # Training + test data
make train DEVICE=cuda               # Train (configs/letter.yaml)
make test DEVICE=cuda                # Evaluate
make atlas DEVICE=cuda               # Interactive attention atlas (HTML)

# Word pipeline
make generate-words && make generate-words-test
make train-words DEVICE=cuda TRANSFER=data/letter_models/model_final.pth

# Override any config value
make train-words EPOCHS=300 BATCH=64 DEVICE=cuda
```

Training parameters live in YAML configs (`python/configs/*.yaml`). CLI args override config values.

See [docs/usage.md](docs/usage.md) for full CLI reference and Makefile documentation.

## Requirements

**Rust/floDl**: Docker with NVIDIA GPU. libtorch 2.10 (cu126) is installed in the container — no local torch installation needed. Rust stable toolchain (also in-container).

**Python**: PyTorch 2.5.1 — pinned for Pascal-era GPU compatibility (GTX 1080 Ti). PyTorch 2.6+ dropped CUDA support for Pascal.

## Project Structure

```
fbrl/
+-- letter/                      # Rust/floDl implementation (active)
|   +-- src/
|   |   +-- lib.rs               #   Crate root
|   |   +-- main.rs              #   CLI entry point
|   |   +-- letter/
|   |       +-- model.rs         #     LetterModel (FlowBuilder graph)
|   |       +-- glimpse.rs       #     GlimpseSensor (multi-scale grid_sample + CNN)
|   |       +-- decoder.rs       #     VisualDecoder (deconv reconstruction)
|   |       +-- modules.rs       #     Identity, H0Init, AttentionStep
|   |       +-- train.rs         #     Training loop, config, profiling
|   |       +-- loss.rs          #     Batched attention/diversity losses
|   |       +-- data.rs          #     PNG loader, batched data pipeline
|   |       +-- synthetic.rs     #     Random images for smoke tests
|   +-- Cargo.toml               #   Depends on flodl (path = "../rdl/flodl")
+-- python/                      # Python/PyTorch implementation (reference)
|   +-- fbrl/                    #   Core package (model, losses, training)
|   +-- configs/                 #   YAML training configs
|   +-- runs/                    #   Archived models + results
+-- goDl/                        # Go/goDl implementation (archived)
+-- docs/                        # Research documentation (shared)
|   +-- letters.md               #   Single-letter experiments
|   +-- words.md                 #   Word experiments
|   +-- motor.md                 #   Motor trace experiments
|   +-- trajectory-thesis.md     #   Trajectory-native framework vision
|   +-- go-retrospective.md      #   Go→Rust pivot: lessons and results
|   +-- glossary.md              #   Deep learning terms and concepts
+-- thoughts/                    # Research notes and hypotheses
+-- Dockerfile                   # nvidia/cuda + libtorch + Rust
+-- docker-compose.yml           # Dev container
+-- Makefile                     # Build, test, train (all Docker-based)
```

## Reference

- [Trajectory Thesis](docs/trajectory-thesis.md) — Why neural networks are trajectory generators, and why the tools matter
- [Go Retrospective](docs/go-retrospective.md) — Lessons from Go/goDl, what we did differently in Rust
- [Research Hypotheses](thoughts/hypotheses.md) — Core intuitions and testable predictions
- [Word Read Phase](thoughts/word_read_phase.md) — Why free read fixations degenerate and how to fix it
- [Glossary](docs/glossary.md) — Deep learning terms as they appear in this project
- [Usage Reference](docs/usage.md) — Full CLI reference and Makefile documentation
