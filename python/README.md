# FBRL — Python/PyTorch Reference Implementation

The original prototype that proved the foveal attention architecture. Active development has moved to [Rust/floDl](../README.md) — this codebase is archived as a reference and for its training data generators.

## Quick Start

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

Training parameters live in YAML configs (`configs/*.yaml`). CLI args override config values.

See [docs/usage.md](../docs/usage.md) for full CLI reference and Makefile documentation.

## Requirements

PyTorch 2.5.1 — pinned for Pascal-era GPU compatibility (GTX 1060). PyTorch 2.6+ dropped CUDA support for Pascal.

## Experiment History

Eight iterations on single letters, then bigrams, words, and motor traces. Each version tested a specific hypothesis — failures were as informative as successes.

### Letters (v1-v8)

| Version | What changed | Result | Key insight |
|---------|-------------|--------|-------------|
| **v1** single-font | Baseline: 1 font, 200 epochs | 100% / 100% | Encode-decode-recode factorization works; recode MSE 0.0004 |
| **v2** multi-font | 11 fonts, guide_weight 4.0 | 99.5% / 99.7% | Guide weight must scale with complexity — decoder bypasses attention when guidance is weak |
| **v3** cosine LR | CosineAnnealingLR | 100% / 100% | Constant LR causes catastrophic divergence at epoch 43; cosine scheduling essential |
| **v4** vertical diversity | Directional diversity, VY=1.5 | 100% / 100% | Horizontal scan bias mirrors human saccades; VY scaling enables vertical exploration |
| **v5** scan phase | 3 scan + 10 read glimpses | 100% / 100% | Zero cost to add scan phase; content detection transfers to word model |
| **v6** fewer glimpses | 1 scan + 7 read = 8 total | 100% / 100% | 38% fewer glimpses, no accuracy loss. Position reset kills generalization (48.6%) |
| **v7** void repulsion | 1 scan + 6 read = 7 total | 100% / 100% | Self-scaffolding: classification -> reconstruction -> recode converges by natural difficulty |
| **v8** 9-glimpse | 1 scan + 8 read, latent_dim=256 | 100% / 100% | Baseline for Rust port |

### Beyond Letters

| Experiment | Setup | Result | Key insight |
|-----------|-------|--------|-------------|
| **Bigrams v1** | Transfer from v3, 192x128, 300 epochs | 97% both-correct | Temporal scaffold + transfer works; 6 errors on confusable pairs (o/c, u/i) |
| **Words v1** prescribed | Prescribed x-scan, 256x128, 200 epochs | 100% all 4 positions | Prescribing scan x removes discovery cost; P4 (rightmost) hardest |
| **Words v2** multihead | Split backward passes, 200 epochs | 99.5% (1 error) | Gradient separation eliminates position bias; isolation loss (128x128) >> canvas masking |
| **Motor v1** | Read-Write-Render-Re-Read from v5 | 97.9% vision, 64.2% re-read | Curriculum learning works; blob co-adaptation limits re-read; sharper rendering needed |

Detailed results for each run are in `runs/<experiment>/results.md`.

## Structure

```
python/
+-- fbrl/                    # Core package
|   +-- model.py             #   VisionModel, BigramVisionModel, WordVisionModel
|   +-- losses.py            #   Attention guide, diversity, void repulsion
|   +-- training.py          #   Training loops (letter, bigram, word, motor)
|   +-- config.py            #   ExperimentConfig + YAML loading
+-- configs/                 # YAML training configs
|   +-- letter.yaml          #   Single-letter (batch=52, 10 reads, no scan)
|   +-- letter_scan.yaml     #   With scan phase (3 scan + 10 read)
|   +-- bigram.yaml          #   Bigram (5 scan + 6 read, scaffold)
|   +-- word.yaml            #   Word (8 scan + 12 read, multi-head, AMP)
+-- runs/                    # Archived models + results
|   +-- letters/v1-v8/       #   Eight letter iterations
|   +-- bigrams/v1-transfer/ #   Bigram transfer learning
|   +-- words/v1-v2/         #   Word experiments
|   +-- motor/v1-transfer/   #   Motor trace experiment
+-- tests/                   # Unit tests (pytest, CPU-only)
+-- data/                    # Training data (generated, not committed)
+-- multimodal/              # Bidirectional audio-visual POC (experimental)
```
