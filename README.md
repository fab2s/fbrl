# FBRL — Feedback Recursive Loop

Can a recurrent foveal attention mechanism learn to "read" text by placing strategic fixations — the way a human eye scans a page?

This project trains a vision model that sees the world through a tiny patch window (12x12 pixels — under 1% of the image). A GRU-based controller decides where to look next, building up a latent representation over a sequence of glimpses. The model must learn *where* to look, not just *what* it sees.

Built on [floDl](https://github.com/fab2s/flodl), a graph-native deep learning framework in Rust. The architecture was first proven in a [Python/PyTorch prototype](python/README.md) that reached 100% accuracy on letters, bigrams, and words — then ported to Rust as floDl's first real-world benchmark. A Go attempt (goDl) validated the graph API but [hit fundamental GC/VRAM limits](docs/go-retrospective.md).

### Why "Feedback Recursive Loop"?

The name describes the core learning mechanism: the model encodes an image into a latent, decodes it to reconstruct the input, then **recodes** it — decodes the same latent under a different condition (flipped case, pen trajectory, different modality). Each recode path adds a constraint that feeds back into the latent space: "can I decode this as 'a' *and* as 'A'?" If both succeed, the latent has captured abstract letter identity. If not, the error reshapes everything.

This extends naturally as the project scales. At word level, the feedback loop becomes: encode a word, decode it, recode it. Each recode direction forces the latent toward more abstract, transferable representations. The loop is the learning mechanism, not just an architectural detail.

## What is Foveal Attention?

Modeled on how your eye actually works. The **fovea** is a tiny spot at the center of your retina (~1-2 degrees of visual angle) — the only part where you see sharp detail. To see anything clearly, your eye physically jumps to aim the fovea at it (saccades — 3-4 per second while reading). Your brain stitches these sparse snapshots into a coherent scene. You *feel* like you see everything in sharp detail. You don't.

| Biology | Model |
|---------|-------|
| Fovea (tiny sharp patch) | `GlimpseSensor` — 12x12 pixel window |
| Peripheral vision | Nothing — even more constrained than biology |
| Saccades (eye movements) | `Controller` — GRU decides next (x,y) fixation |
| Brain integrating fixations | GRU hidden state accumulating information |
| Conscious perception | Latent vector — the final representation |

A standard CNN sees the entire image at once. Foveal attention forces the model to develop a **strategy** for looking — given what I've seen so far, where should I look next?

## Research Lines

Four experiment tracks, each building on discoveries from the previous:

### [Single Letters](docs/letters.md) — the foundation

100% accuracy across 52 classes (Aa-Zz) and 11 fonts with only **7 glimpses** (1 scan + 6 read) — 46% fewer than the initial 13-glimpse architecture. The model develops letter-specific scan strategies: 'T' gets the crossbar junction, 'O' gets the negative space, 'A' gets the apex and legs. Only ~40% of fixations land on letter pixels — it samples diagnostic features, not outlines.

### [Bigrams](docs/bigrams.md) — the geometry lesson

Two letters on 128x128. Achieved 98-99% accuracy but revealed a fundamental truth: **you can't force reading strategy through loss design alone**. The foveal window covers enough of the canvas for holistic shortcuts. The task geometry must make cheating impossible. This drove the transition to words.

### [Words](docs/words.md) — genuine sequential reading

4-letter words on 256x128. The foveal window covers 0.4% of the canvas — holistic reading is geometrically impossible. 100% accuracy on all 4 positions. Multi-head optimization (separate gradient paths for attention, classification, reconstruction) accelerates convergence. Interleaved scan-read failed due to global attention guide center-pull — led to the void repulsion insight.

### [Motor Traces](docs/motor.md) — learning to write

Read-Write-Render-Re-Read: the encoder produces a latent, a motor decoder writes a pen trajectory, a renderer draws it, the encoder re-reads it. If the re-read matches the original, the motor has learned to write.

## Key Discoveries

These insights emerged iteratively — each shaped the next experiment.

**Self-scaffolding** — All losses active from epoch 1, no curriculum. Yet they sequence by difficulty: classification converges first (strong CE gradient), reconstruction follows, recode takes over last (requires latent factorization). Easy tasks bootstrap representations that hard tasks need. Intrinsic difficulty can replace explicit curriculum design. ([details](docs/letters.md#self-scaffolding-discovery))

**Scan = guide, reads = self-directed** — A global attention guide creates center-pull that's toxic for multi-position reading. The fix: blurred guide for scan only (long-range "find the content" signal), void repulsion for reads (local "don't stare at nothing" — zero gradient in deep void, active only at ink boundaries). ([details](docs/words.md#the-lesson-scan--guide-reads--self-directed))

**Flat read preserves GRU momentum** — Resetting the GRU's position between read groups causes catastrophic overfitting. The hidden state carrying spatial context from one glimpse to the next is essential. Position reset discards it. ([details](docs/letters.md#v6-fewer-glimpses))

**Gradient separation accelerates convergence** — Summing all losses blurs the gradient signal. Splitting into separate backward passes gives each component clean gradients: the controller learns *where* to look from attention losses only, the readout learns *what* to read from classification only. ([details](docs/words.md))

**Canvas scale determines reading strategy** — The geometry of the task must make cheating impossible. No amount of loss engineering can force sequential reading if the foveal window can see enough from center. ([details](docs/bigrams.md#the-canvas-scale-finding))

## Architecture

The Rust implementation lives in `letter/` and is built on floDl's `FlowBuilder` graph engine.

```
Input: image [B, 1, 128, 128] + case_label [B, 1]
  |
  H0Init (learnable initial hidden state)
  |
  ScanStep (1x) -- wide patch (12x18), learnable x, free y
  |                 shared Controller (GRU + loc_head)
  |
  AttentionStep (6x) -- fine patch (12x12), free (x,y)
  |                      same shared Controller
  |
  +-> letterHead (Linear -> 26 classes)
  +-> caseHead   (Linear -> 2 classes)
  +-> VisualDecoder (deconv reconstruction -> [B, 1, 128, 128])
```

Key modules: `GlimpseSensor` (grid_sample + CNN), `Controller` (shared GRU + location head via `Rc`), `VisualDecoder` (transposed convolutions + BatchNorm).

## Quick Start

All commands run inside Docker — libtorch and Rust toolchain are container-only.

```bash
make build                                              # Build Docker image
make test                                               # Run unit tests (CUDA)
make train-letter DATA=../python/data/letters MONITOR=3000  # Train with live dashboard
make train-letter SYNTHETIC=64 EPOCHS=2                 # Quick smoke test
make eval-letter RUN_DIR=runs/v1                        # Evaluate a trained model
make shell                                              # Interactive container shell
```

The live monitor dashboard is served at `localhost:3000` during training.

## Requirements

Docker with NVIDIA GPU runtime. The container includes libtorch 2.10 (cu126) and the Rust toolchain — nothing to install on the host.

All results achieved on a single **GTX 1060 6GB** (Pascal, 2016). floDl works out of the box on Pascal-era hardware via libtorch — no version pinning required. The Python/PyTorch prototype needed PyTorch 2.5.1 specifically because 2.6+ dropped Pascal CUDA support.

## Project Structure

```
fbrl/
+-- letter/                      # Rust/floDl implementation (active)
|   +-- src/letter/
|   |   +-- model.rs             #   LetterModel (FlowBuilder graph)
|   |   +-- modules.rs           #   Controller, ScanStep, AttentionStep, H0Init
|   |   +-- glimpse.rs           #   GlimpseSensor (grid_sample + CNN)
|   |   +-- decoder.rs           #   VisualDecoder (deconv reconstruction)
|   |   +-- train.rs             #   Training loop, config, Monitor integration
|   |   +-- eval.rs              #   Inference, accuracy report, HTML attention atlas
|   |   +-- loss.rs              #   Attention guide, diversity, hit rate
|   |   +-- data.rs              #   PNG loader, batched pipeline
|   +-- runs/                    #   Training runs + eval results
|   +-- Cargo.toml               #   Depends on flodl (path dependency)
+-- python/                      # PyTorch reference implementation (archived)
|   +-- README.md                #   Python-specific docs + experiment history
|   +-- runs/                    #   Archived models: letters v1-v8, bigrams, words, motor
+-- goDl/                        # Go/goDl implementation (archived)
+-- docs/                        # Research documentation
|   +-- letters.md               #   Single-letter experiments
|   +-- bigrams.md               #   Bigram experiments
|   +-- words.md                 #   Word experiments
|   +-- motor.md                 #   Motor trace experiments
|   +-- trajectory-thesis.md     #   Why neural networks are trajectory generators
|   +-- go-retrospective.md     #   Go->Rust pivot: lessons learned
+-- thoughts/                    # Research notes and hypotheses
+-- Dockerfile                   # nvidia/cuda:12.6.3 + libtorch 2.10 + Rust
+-- docker-compose.yml           # GPU dev container
+-- Makefile                     # Build, test, train (all Docker-based)
```

## Reference

- [Trajectory Thesis](docs/trajectory-thesis.md) — Why neural networks are trajectory generators, and why the tools matter
- [Go Retrospective](docs/go-retrospective.md) — Lessons from Go/goDl, what we did differently in Rust
- [Research Hypotheses](thoughts/hypotheses.md) — Core intuitions and testable predictions
- [Word Read Phase](thoughts/word_read_phase.md) — Why free read fixations degenerate and how to fix it
- [Glossary](docs/glossary.md) — Deep learning terms as they appear in this project
- [Python Reference](python/README.md) — PyTorch prototype, experiment history, archived runs
