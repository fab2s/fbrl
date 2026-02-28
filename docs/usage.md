# Usage Reference

Full CLI reference, training output format, and Makefile documentation.

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
| **Iso** | Isolation CE — single-letter classification. Random = 3.26. | < 0.5 |
| **Hit** | % of fixations on letter pixels (diagnostic). | 25-45% |
| **lr** | Learning rates (multi-head: attn/cls/recon; single: encoder/readout). | -- |
| **scaff** | Temporal scaffold strength (1.0 = full guidance, anneals to 0). | anneals to 0 |

## CLI Reference

All commands run via `python vision_training.py <command>`.

### Config-based training

Training commands (`train`, `train_bigrams`, `train_words`) use YAML config files for all hyperparameters. A config file is required; CLI args override individual values.

```bash
# Basic usage — all params from config
python vision_training.py train --config configs/letter.yaml

# Override specific values
python vision_training.py train_words --config configs/word.yaml --epochs 300 --device cuda

# Resume from checkpoint
python vision_training.py train_words --config configs/word.yaml --resume data/word_models/model_final.pth

# Transfer learning
python vision_training.py train_words --config configs/word.yaml --transfer data/models/model_final.pth
```

#### Config files

Each YAML file mirrors the fields of `ExperimentConfig` in `fbrl/config.py`:

| Config | Model type | Key settings |
|--------|-----------|--------------|
| `configs/letter.yaml` | letter | batch_size=52, n_read_glimpses=10, no scan phase |
| `configs/letter_scan.yaml` | letter | n_scan_glimpses=3, diversity_vy=1.5, for pretrained scan+content_head |
| `configs/bigram.yaml` | bigram | n_scan=5, n_read=6, scaffold_ratio=0.67 |
| `configs/word.yaml` | word | n_scan=8, n_read=12, multi_head=true, amp=true |

#### Training CLI overrides

All training subcommands accept these optional overrides (non-None values replace the config):

```
--config PATH             YAML config file (required)
--device auto|cpu|cuda    Compute device
--resume PATH             Resume from checkpoint
--transfer PATH           Source model for transfer learning
--epochs N                Training epochs
--batch_size N            Batch size
--data_dir PATH           Training data directory
--save_dir PATH           Checkpoint output directory
--checkpoint_interval N   Save checkpoint every N epochs
--guide_weight F          Attention guide loss weight
--scan_guide_weight F     Scan-phase guide weight (defaults to guide_weight)
--scaffold_ratio F        Scaffold phase as fraction of total epochs
--scaffold_floor F        Minimum scaffold weight after annealing
--scaffold_epochs N       Explicit scaffold epoch count (overrides ratio)
--diversity_weight F      Fixation spread pressure
--diversity_sigma F       Repulsion radius
--scan_vy F               Scan diversity VY (<1 = horizontal spread)
--read_vy F               Read diversity VY (>1 = vertical exploration)
--content_weight F        Content detection BCE weight
--edge_weight F           Edge exploration weight
--blur_sigma_ratio F      Blur sigma as fraction of image size
--n_scan_glimpses N       Scan-phase glimpse count
--n_read_glimpses N       Read-phase glimpse count
--n_positions N           Letter positions per word
--scan_patch_size H,W     Scan patch dimensions (e.g. 12,18)
--read_patch_size N       Read patch size (square)
--n_scales N              Resolution scales
--recode_weight F         Case-flip reconstruction weight (letter only)
--diversity_vy F          Single-letter diversity VY
--mask_weight F           Masked-half auxiliary loss (bigram only)
--isolation_weight F      Isolation loss weight (word only)
--isolation_data_dir PATH 128x128 single-letter data for isolation
--isolation_random_prob F Random letter substitution probability in isolation
--multi_head              Multi-head optimization (3 separate backward passes)
--amp                     Automatic mixed precision
```

### Non-training commands

These commands keep their own argument sets (no config file needed).

#### Single-letter

**generate**
```
--letters Aa-Zz        Letter range (A-Z, a-z, Aa-Zz, or individual chars)
--num_variants 20      Noisy copies per letter
--noise_level 0.1      Gaussian noise std (0-1 scale)
--output_dir data/letters
--fonts all            Font spec: "all", "default", or comma-separated names
```

**generate_test**
```
--letters Aa-Zz
--output_dir data/test
--fonts all
```

**test**
```
--model_dir data/models
--test_data_dir data/test
--output_dir data/results
--device auto|cpu|cuda
```

**atlas**
```
--model_dir data/models
--test_data_dir data/test
--output data/atlas.html    Self-contained HTML file
--device auto|cpu|cuda
```

Generates an interactive attention atlas — a single HTML file with Canvas-based Gaussian-splat heatmaps for all 52 letters across all fonts. The grid view shows averaged attention across fonts; clicking a letter drills down into per-font fixation patterns. Controls: heatmap/path toggle, upper/lower/both filter, opacity slider. Cell borders show correctness: green = all fonts correct, yellow = some wrong, red = all wrong.

**check_attention**
```
--data_dir data/letters    Dataset to check against
--n_epochs 10              Diagnostic epochs to run
--device auto|cpu|cuda
--guide_weight 8.0         Guide weight to test
--blur_sigma_ratio 0.16    Blur ratio to test
```

**visualize**
```
--model_dir data/models
--data_dir data/letters
--output_dir data/visualizations
--device auto|cpu|cuda
```

#### Bigram

**generate_bigrams**
```
--num_variants 20          Noisy copies per bigram
--noise_level 0.1          Gaussian noise std
--output_dir data/bigrams
--fonts default            Font spec: "all", "default", or comma-separated names
```

**generate_bigrams_test**
```
--output_dir data/bigram_test
--fonts default
```

**test_bigrams**
```
--model_dir data/bigram_models
--test_data_dir data/bigram_test
--output_dir data/bigram_results
--device auto|cpu|cuda
```

**bigram_atlas**
```
--model_dir data/bigram_models
--test_data_dir data/bigram_test
--output data/bigram_atlas.html
--device auto|cpu|cuda
```

Same concept as the single-letter atlas but for 200 bigrams on 128x128 canvases. Grid flows with auto-fill layout. Cell borders show correctness: green = both letters correct, yellow = one correct, red = neither.

**check_bigram_attention**
```
--data_dir data/bigrams
--n_epochs 10
--device auto|cpu|cuda
```

#### Word

**generate_words**
```
--num_variants 20          Noisy copies per word
--noise_level 0.1          Gaussian noise std
--output_dir data/words
--fonts default            Font spec: "all", "default", or comma-separated names
```

**generate_words_test**
```
--output_dir data/word_test
--fonts default
```

**test_words**
```
--model_dir data/word_models
--test_data_dir data/word_test
--output_dir data/word_results
--device auto|cpu|cuda
```

**word_atlas**
```
--model_dir data/word_models
--test_data_dir data/word_test
--output data/word_atlas.html
--device auto|cpu|cuda
```

Same concept as the bigram atlas but for 200 four-letter words on 256x128 canvases. 2:1 aspect ratio cells. Cell borders show correctness: green = all 4 letters correct, yellow = some correct, red = none correct.

## Makefile

Training targets use YAML config files for hyperparameters. Most parameters live in the config; the Makefile exposes a few key runtime overrides.

```bash
make train DEVICE=cuda                    # Uses configs/letter.yaml
make train-bigrams DEVICE=cuda TRANSFER=data/models/model_final.pth
make train-words EPOCHS=300 BATCH=64 DEVICE=cuda
make generate LETTERS=A-Z VARIANTS=10 NOISE=0.2 FONTS=all
```

### Makefile variables

**Training overrides** (passed as CLI args, override config values):

| Variable | Default | Description |
|----------|---------|-------------|
| `CONFIG` | configs/letter.yaml | YAML config file for `make train` / `make resume` |
| `BIGRAM_CONFIG` | configs/bigram.yaml | YAML config file for `make train-bigrams` / `make resume-bigrams` |
| `WORD_CONFIG` | configs/word.yaml | YAML config file for `make train-words` / `make resume-words` |
| `DEVICE` | auto | auto, cpu, or cuda |
| `EPOCHS` | *(empty)* | Training epochs (overrides config) |
| `BATCH` | *(empty)* | Batch size (overrides config) |
| `TRANSFER` | *(empty)* | Path to source model for transfer learning |
| `CKPT` | *(empty)* | Checkpoint interval (overrides config) |
| `RESUME_FROM` | model_final.pth | Checkpoint filename for resume targets |

**Generation variables:**

| Variable | Default | Description |
|----------|---------|-------------|
| `LETTERS` | Aa-Zz | Letter range for generation |
| `VARIANTS` | 20 | Noisy variants per letter/bigram/word |
| `NOISE` | 0.1 | Gaussian noise level |
| `FONTS` | all | Font spec: all, default, or comma-separated names |

Each pipeline has its own config variable with a sensible default. Override to use a custom config:
```bash
make train-words WORD_CONFIG=configs/word_experimental.yaml
```

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
| `make train-bigrams` | Train bigram model (uses `configs/bigram.yaml`, supports `TRANSFER=`) |
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
| `make train-words` | Train word model (uses `configs/word.yaml`, supports `TRANSFER=`) |
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
