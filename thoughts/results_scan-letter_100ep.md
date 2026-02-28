# v6-scan-letter: Scan Phase on Single-Letter Model (100 epochs, 11 fonts)

## Experiment

Add an optional two-phase scan->read architecture to the single-letter VisionModel.
Goal: pretrain scan_sensor and content_head on cheap 128x128 data so they transfer
directly to WordVisionModel, instead of training from scratch during word training.

### Architecture
- **Scan phase**: 3 glimpses, prescribed x at [-0.75, 0.0, +0.75], learned y
  - GlimpseSensor(12x18) wide patches (same as word scan_sensor)
  - content_head: nn.Linear(256, 1) — BCE on whether scan location has letter content
- **Read phase**: 10 glimpses, free x,y with existing 12x12 sensor
  - Identical to v5 baseline behavior
- **Total**: 14 fixations (1 start + 3 scan + 10 read) vs v5's 11 (1 start + 10 read)
- Shared GRU controller across both phases (h carries scan->read)

### Training config
```
make train LETTER_SCAN_GLIMPSES=3 DEVICE=cuda GUIDE=8.0 VY=1.5
```
- guide_weight=8.0, scan_guide_weight=8.0 (same for both phases)
- content_weight=0.5, scan_vy=0.3, diversity_vy=1.5
- CosineAnnealingLR, batch_size=52, 100 epochs
- 11 fonts, 52 letters (Aa-Zz), 20 variants = 11,440 training samples

## Results

### Accuracy
- **Letter: 100.0%** (572/572) across all 11 fonts
- **Case: 100.0%** (572/572) across all 11 fonts
- Matches v5-vertical-diversity baseline exactly

### Final metrics (epoch 100)
| Metric | v6 (scan) | v5 (no scan) |
|--------|-----------|--------------|
| Ltr CE | 0.0000 | 0.0000 |
| Case CE | 0.0000 | 0.0000 |
| Recon MSE | 0.0039 | ~0.004 |
| Recode MSE | 0.0004 | ~0.0005 |
| Content BCE | 0.0011 | n/a |
| Hit rate | 38% | ~38% |

### Convergence speed
- Letter CE < 0.1 by epoch 15, < 0.001 by epoch 45
- Content BCE < 0.01 by epoch 50, stabilized at 0.001
- ~52s/epoch on GTX 1080 Ti (vs ~52s for v5 — no measurable overhead)
- Total training time: ~87 minutes

### Attention patterns
The scan phase produces a clear 3-step L->R sweep visible in all attention plots:
- Fix 0: center start (0, 0)
- Fix 1: far left (-0.75, learned_y) — empty space, content_head learns "nothing here"
- Fix 2: center (0.0, learned_y) — lands on letter content
- Fix 3: far right (+0.75, learned_y) — empty space again

Read phase (fix 4-13) then clusters tightly on letter strokes. The scan->read
transition is clean — GRU hidden state carries the spatial overview into focused reading.

## Transfer path to word model (v3)

The scan-trained checkpoint now has these keys ready for direct copy:
```
scan_sensor.*    -> scan_sensor.*     (NEW: pretrained wide-patch sensor)
content_head.*   -> content_head.*    (NEW: pretrained content detection)
encoder.glimpse_sensor.*  -> read_sensor.*    (existing mapping)
encoder.attention_controller.*  -> controller.*  (existing mapping)
letter_classifier.*  -> classifiers.*.*  (existing mapping)
```

Key insight: on 128x128 single letters, the scan sees content at center (x=0) and
empty space at edges (x=-0.75, +0.75). This is exactly the content detection skill
needed for word images — where letters occupy some positions and others are empty.
The content_head BCE converged to 0.001, meaning it reliably distinguishes content
from background even through the wide 12x18 scan patches.

## Key finding

Adding a scan phase to the single-letter model has **zero cost** to convergence quality
and **zero measurable overhead** in training time (same ~52s/epoch). The 3 extra
scan glimpses are essentially free because:
1. The scan patches are processed by a lightweight CNN (same architecture as read sensor)
2. Content BCE adds negligible computation (single grid_sample + BCE per scan step)
3. The prescribed x positions require no gradient for the x component

This makes scan-augmented single-letter training strictly better than baseline when
word training transfer is planned — you get identical letter accuracy plus pretrained
scan mechanics for free.

## Archive
- `runs/v6-scan-letter/` — model, training log, metrics graph, atlas
