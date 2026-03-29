# Word Model Lab Log

Reverse chronology. The journey from first composition attempt to recursive composed training.

## 2026-03-27/28: REINFORCE experiments → triangle SubScan redesign

### What we tried

**Exp A — REINFORCE single-phase** (σ=0.03, fail_penalty=-0.1, target_bonus=0.2):
Pure REINFORCE retry loop. Oracle reads from SubScan output, reward on success.
Result: 99.9% success rate but target_acc=4.6% (≈1/26 = random). Mean attempts=1.1.
Everything succeeds immediately because ANY position near a letter gets a confident read.
Reward differential between correct (1.2) and wrong (1.0) letter is negligible.

**Exp B — REINFORCE two-phase** (same σ, added wrong_letter_penalty=-0.05, settle_reward=0.1):
Phase 1: retry until oracle reads something. Phase 2: 8 refinement attempts to find correct letter.
Result: mean_attempts=8.6 (refinement fires), target_acc=6.1%, flat across 3 epochs. Not learning.

**Exp C — REINFORCE two-phase, stronger** (σ=0.12, wrong_letter_penalty=-0.25):
4x bigger exploration, 5x stronger penalty.
Result: mean_attempts=8.6 (unchanged), target_acc=7.0%, still flat. Same wall.

### Why REINFORCE fails for SubScan positioning

The oracle says "wrong letter" but gives no spatial direction. The noise ε determines gradient direction, which is random. Each retry's backward+step is a random walk in weight space. The model cannot extract "move left" from "you read the wrong letter."

With 2 horizontal glimpses, SubScan senses only 1D ink density (left-right). It can't distinguish letter boundaries — "O" and adjacent "IL" look identical from the side.

### The insight: triangle glimpses

SubScan needs to learn **where letters start and end**, not which letter it is (that's the oracle's job). Two horizontal glimpses are insufficient. Three glimpses in a triangle pattern provide:
- Left/right base: bracket the letter horizontally → sense ink edges
- Apex (above/below): sense vertical structure → distinguish O (arc) from IL (two strokes)

The triangle has constrained geometry:
- Base width: bounded min (I-width) to max (<MM-width), learned via tanh
- Height: fixed (known from line height)
- Center: midpoint of base, y fixed to line center

SubScan outputs center_x. Trained with MSE(center_x, gt_center_x). No oracle, no REINFORCE. Pure supervised on known letter positions. The ink density is the INPUT (through glimpse features), not the loss.

**Key difference from earlier MSE attempts**: the architecture. 3 structured triangle glimpses vs 2 free horizontal ones. The triangle constraint is the inductive bias that was missing.

**Composition strategy**: SubScan trained independently on word images (no letter model). Letter model trained independently on single letters. Compose at inference: SubScan → center_x → LetterModel. Neither has seen the other during training.

### Next: Experiment D — Triangle SubScan with supervised MSE

---

## 2026-03-26 (night): Two origin bugs and relative glimpse sampling

### Bug 1: Shared origin consumed by scan, invisible to read

`ScanStep::step()` used `.take()` on the shared `Rc<RefCell<Option<Variable>>>` origin slot,
emptying it before `AttentionStep` could read it. The read phase operated without origin —
always reading from absolute (0,0). On isolated 128×128 letters this was invisible: the
letter IS at (0,0). On word images, the read phase looked between letters.

**Diagnostic trace (before fix):**
```
pos 0: origin=(-0.75,0)  scan_last=-0.375 (correct)  read_last=0.017 (WRONG — no origin)
```

**Fix:** `.take()` → `.clone()`. One character of intent.

### Bug 2: Absolute position in location embedding — the deeper issue

After fixing bug 1, the read phase received origin but STILL drifted to (0,0):
```
Step 1: loc_x=-0.092 + origin=-0.750 = sample_x=-0.842  (starts near letter)
Step 2: loc_x=+0.330 + origin=-0.750 = sample_x=-0.420  (drifting to center)
Step 6: loc_x=+0.833 + origin=-0.750 = sample_x=+0.083  (at image center, NOT at letter)
```

The loc_head actively fought the origin. Why?

**Root cause:** The `GlimpseSensor` fed the ABSOLUTE `sample_pos` (relative + origin) into
the location embedding (`location_fc`). The GRU received "where I am in the image" as
absolute coordinates. During training on 128×128 images where letters are always at (0,0),
the model learned: "navigate to absolute (0,0)." On word images with origin at (-0.75, 0),
it tried to reach (0,0) — the wrong place.

**Fix: Split the location into two signals:**
1. **grid_sample** gets `sample_pos` (absolute) — extract patches from the right image location
2. **location_fc** gets `relative_loc` (local frame) — the model's "where am I" belief stays relative

```rust
// Before: sensor sees absolute position
refs.insert("location", sample_pos);

// After: sensor sees both
refs.insert("location", sample_pos);           // for grid_sample
refs.insert("relative_location", loc.clone()); // for location embedding
```

The GlimpseSensor uses `location` for patch extraction and `relative_location` for the
position embedding. In standalone mode (no origin), they're identical. In composed mode,
the embedding stays in [-0.3, 0.3] (training range) while extraction reaches anywhere.

### The principle: relative glimpse sampling

The foveal attention mechanism must separate two uses of position:

| Purpose | Coordinate frame | Signal |
|---------|-----------------|--------|
| "What do I see" (grid_sample) | Global / absolute | `sample_pos = relative + origin` |
| "Where am I" (location embedding → GRU) | Local / relative | `relative_loc` (near origin) |

This is analogous to relative position encodings in transformers, but for spatial attention
over actual images. The origin is a pure geometric constant — invisible to learning, applied
only at the sampling boundary.

**Why this matters for composition:** the model's trajectory policy (GRU + loc_head) is
trained once in its local frame and works at any absolute position. The word model sets
the origin, the letter model doesn't know or care. Same principle at every level of the
hierarchy: SubScan sets origin for letter, word model sets origin for SubScan, sentence
model sets origin for word model.

**The journey to get here:** 5 attempts at origin translation (backprop through frozen model,
MSE regression, word-image retraining, crop, coordinate origin) → discovered shared-state bug
→ discovered absolute-embedding bug → arrived at relative glimpse sampling. Each wrong turn
revealed a deeper assumption about how position flows through the system.

### Convergence with biological vision

The relative glimpse sampling solution maps directly to a known neuroscience observation:
retinotopic maps in early visual cortex encode position relative to the current fixation
point, not absolute position in the visual field. The brain's "location embedding" IS relative.

We arrived at the same design constraint independently, from a pure engineering failure mode:
the loc_head fighting the origin to reach (0,0). The fix — separate "where to sample"
(absolute, for patch extraction) from "where I think I am" (relative, for the position
embedding) — is not a design choice. It's a constraint. Any system that composes spatial
attention hierarchically HAS to do this, or it breaks at composition time.

The letter model's robustness was also telling: it masked two origin bugs across training,
eval, and test by compensating for broken coordination. If biological vision modules are
similarly robust within their local frame, this explains how the visual system degrades
gracefully through noisy saccade targeting, imprecise coordinate handoffs, and partial
signals — each module just works anyway, because it's good at its task regardless of what's
happening above it. A coupled end-to-end system would fail immediately. Isolated modules
compensate silently.

This is perhaps the strongest signal yet that the recursive foveal attention architecture
is on the right track. The constraints we're discovering from engineering necessity
match the constraints biology solved. Not because we copied biology — because the math
doesn't work any other way.

---

## 2026-03-26: The composition day

### Run 5 — Recursive composed training (RUNNING)

The idea: train like you infer. The training loop mirrors the architecture.

```
Epoch loop (generalization — sentence level)
  Position loop (curriculum — word level)
    Retry loop (learn to position — foveal read level)
      forward → CE loss → backward → step
      loop until letter model recognizes the letter
```

Each level must succeed before the next proceeds.

**Config**: `max_attempts=0` (unlimited, safety cap 500), `attempt_threshold=0.9`,
`noise_start=0.02`, `noise_ramp=100 epochs`, `recon_weight=0.0` (pure CE),
curriculum threshold 80%, 200 epochs, noise2 checkpoint.

**Key insight**: the letter model IS the loss function. Its frozen read module is the quality gate.
SubScan must learn to position well enough that the read trajectory converges.
Every failed attempt updates SubScan's weights — the next attempt benefits.

**Metric to watch**: `att` (average attempts per batch). If it drops from 100+ to single digits
over training, composition works. If it stays at safety cap, the problem is structural.

**Results** (partial — killed at epoch 15):
- Epoch 1: att=74.7, acc=89.9%, 4m29s — IT WORKS. Model solves letters with many tries.
- Epoch 4: att=15.4, acc=91.6%, 58s — learning fast, speeding up naturally.
- Epoch 10: att=10.4, acc=92.0%, 41s — curriculum unlocked position 2 at epoch 11.
- Epoch 11: att=52.7, acc=91.4%, 6m06s — position 2 spike, expected.
- Epoch 15: att=250.5, acc=25.1%, 29m — **catastrophic interference between positions.**

**Root cause**: positions share SubScan weights. Backward+step for position 0 shifts weights,
position 1 fails with updated weights. Positions fighting each other.

**Fix**: one random position per batch. Each letter read is fully independent.

---

### Run 6 — Independent position per batch (RUNNING)

One random position per batch. No cross-position interference. Each batch is a
self-contained foveal episode — exactly like inference.

Over an epoch (125 batches), each position gets ~62 batches (2 active) or ~31 (4 active).
Positions learn without sabotaging each other.

**Key principle discovered**: isolation is the feature, not the limitation.
Each module trained in isolation BECAUSE it forces clear interfaces, makes failures
diagnosable, and allows exotic per-module training patterns.

This is the Unix philosophy for neural networks: do one thing well, compose via clean
interfaces. The "pipes" are learned position vectors, the "programs" are foveal attention modules.

Standard ML: "throw everything in, end-to-end, let backprop figure it out."
This approach: isolation → verification → composition → verification. Harder to start,
straightforward to debug. You're never lost — you always know what works and what doesn't.

---

### Run 10 — Letter model word-image training (RUNNING)

**The realization**: the letter model has never seen a word image. Run 9 proved SubScan can
position perfectly (MSE=0, err<0.001) but the frozen letter model gets 6% accuracy on
128×256 word images — it only knows 128×128 isolated letters.

**Root cause clarified**: the problem was never positioning. It was distribution mismatch.
The letter model's scan/read GRUs, glimpse sensors, and classifier were trained on isolated
letters. Word images have different content (neighboring letters visible in patches).

**The principle**: each layer operates in its own coordinate frame. Upper layers hand off
position constants. No cropping, no image manipulation — pure coordinate translation.

```
Word model  →  (x, y) origin  →  SubScan  →  (x, y) start  →  Letter model
                constant              constant
```

**The fractal pattern**: every layer trains against the frozen stack below it.
```
Letter (training)         → can I read from this position?
SubScan (training)        → frozen Letter        → can it read?
Word model (training)     → frozen SubScan+Letter → can they read?
```
Bottom-up: train, freeze, become the oracle for the layer above.
The retry loop is the universal mechanism at every level.

**Step 1b**: Retrain letter model on word images (128×256) with `set_scan_start(gt_pos + noise)`.
CE + case loss only (skip recon — decoder outputs 128×128). Fine-tune from v2_noise2 checkpoint.
LR=0.0003 (fine-tuning). The model learns to focus on one letter in a multi-letter image.

**Prediction**: accuracy should climb quickly (model already reads letters, just needs to
adapt to word-image context). Once trained, this becomes the frozen oracle for SubScan.

---

### Run 9 — Position regression (completed, 200 epochs)

**Rewrite #2**: removed retry loop entirely. Direct MSE on positions.
SubScan learns to point at letter centers via regression.

**Results**: MSE converges to ~0 by epoch 4. Position error < 0.001.
Training: ~2s/epoch (no letter model in the loop).

**But**: validation accuracy (frozen letter model) stuck at 5.9-6.0% across all 200 epochs.
SubScan points perfectly but the letter model can't read word images.

**Key insight**: the positioning problem was always solved. The letter model distribution
mismatch was the real blocker. This run proved it conclusively.

---

### Run 8 — Clean rewrite: frozen oracle pattern (killed)

**Deep code review** revealed several issues in the composition training loop. Instead of
patching, rewrote `subscan_train.rs` from scratch.

**Key discovery**: The ScanStep x-channel "cold start" is a non-issue for noise-trained
checkpoints. When `set_scan_start()` is called during noise training, `from_external=true`
activates free (x,y) mode — **both loc_head channels are trained**. The v2_noise2 checkpoint
has a properly trained scan loc_head.

**Design: frozen oracle pattern**
- ALL letter model params frozen + eval mode (fixes BN drift)
- Frozen scan step enriches h with one wide glimpse (value-add, not trainable)
- SubScan is the ONLY module learning
- Single optimizer group, CE only, no recon/diversity
- Noise training defines interface contract: SubScan within ±0.3x / ±0.15y tolerance

**Fixes in the rewrite:**
- BN drift: `letter.eval()` instead of `letter.train()`
- Tensor allocation: region_half_w/h/case_var hoisted out of retry loop
- Two optimizer groups → one (SubScan only)
- Removed dead code: recon, diversity, unused imports

**Also added** `set_read_start()` to LetterModel for future n_scan=0 experiments —
AttentionStep checks external_start > scan handoff > zeros.

**Scale mismatch hypothesis debunked**: `build_base_grid` math preserves physical pixel
coverage regardless of image width. 12px patch = 12px on both 128×128 and 128×256.

**Prediction**: att should drop. If not, the h-pathway gradient (frozen scan → read → CE)
may be too weak → fall back to set_read_start (direct position gradient, n_scan=0).

---

### Run 7 — Referential translation (killed for rewrite)

**The insight**: SubScan shouldn't know where in the image it is. Every letter read is the
same task: "there's a letter near (0,0) in my local frame, find it."

**Referential translation**: instead of cropping, shift the coordinate frame. SubScan works
in local coords where (0,0) = estimated letter center. The center offset is a constant added
only at glimpse extraction time. SubScan's loc_head outputs pure local offsets. The caller
translates to global: `global_pos = local_offset + center`.

```
Word model: "letter at x=-0.75"
  → center = [-0.75, 0.0] (constant, not learned)
  → SubScan works at (0,0), glimpses extracted at (0,0) + center
  → SubScan outputs local offset, e.g., (+0.02, -0.01)
  → Global position = (-0.73, -0.01)
  → Letter model reads from there
```

**Why this matters**:
- All positions are identical — train once, works everywhere
- No curriculum needed for positions
- No position spikes when unlocking (the problem doesn't change)
- The only learned skill: "given a letter near center, refine the position"
- Word model noise shows up as the letter being slightly off-center — exactly what noise
  training prepared the system for

**Prediction**: att should stay flat when new positions unlock. If confirmed, this proves
that true isolation (position-agnostic coordinate frame) eliminates the need for
position-specific learning entirely.

---

### Run 4 — Retry loop with 5 max attempts

**What**: per-attempt backward+step, 5 max retries, CE-only (dropped recon).
**Result**: ~25% accuracy, `att=5.0` (always maxing out). Not enough attempts.
**Killed to iterate**.

---

### Run 3 — Curriculum + y-bounded regions + noise2 checkpoint

**What**: sequential curriculum (1 position, unlock at 80%), `region_half_h=0.15`,
letter checkpoint trained with y=0.15 noise.
**Result**: 24.6% accuracy, 200 epochs, never unlocked position 2.
**Learning**: noise range and curriculum alone don't solve composition.

---

### Run 2 — First noise-trained checkpoint

**What**: letter model trained with `scan_noise_x=0.3, y=0.05`.
SubScan with 4 simultaneous positions, `region_half_w=0.5`.
**Result**: 21.8% accuracy, 100 epochs. Better than run 1 (14%) but still plateau.
**Learning**: noise helps but doesn't solve the fundamental issue.

---

### Run 1 — Clean checkpoint (overnight, from 2026-03-25)

**What**: SubScan + frozen letter read, 4 positions simultaneously, clean letter checkpoint.
**Result**: 14.3% accuracy, CE 3.03 (near random over 26 letters).
**Root cause**: frozen read module trained from (0,0) center. When SubScan places scan
elsewhere, the trajectory changes completely. Read has never seen this regime.
**Key insight**: each layer must be pre-adapted for the context the layer above provides.

---

## Architecture evolution through the day

### Letter model noise training

The letter model proved remarkably robust to noise:

| Run | Noise (x, y) | Epochs | Letter acc | Case acc | Eval MSE |
|-----|-------------|--------|-----------|---------|----------|
| v2_retrain | 0, 0 | 100 | 100% @ ep69 | 100% @ ep69 | 0.0010 |
| v2_noise | 0.3, 0.05 | 150 | 100% @ ep56 | 100% @ ep35 | 0.0010 |
| v2_noise2 | 0.3, 0.15 | 150 | 100% @ ep56 | 100% @ ep35 | 0.0008 |

3x more y noise and the model doesn't care. Harder training actually improved recon quality.
The foveal read mechanism is fundamentally noise-invariant — the GRU policy just adapts.

### SubScan changes

**subscan.rs**: y now bounded — `y = region_half_h * tanh(raw_y)` instead of free `tanh(raw_y)`.
Rationale: vertical position is given by sentence → word model. SubScan refines horizontally.

**subscan_train.rs** — accumulated changes:
- Sequential curriculum: `active_positions` starts at 1, unlocks on accuracy threshold
- Retry-until-success: inner `loop` with per-attempt backward+step
- `max_attempts=0` = unlimited (safety cap prevents infinite loop)
- `recon_weight=0.0` — letter model CE is the only signal
- Tracks `avg_attempts` as key progress metric
- `region_half_h` parameter for vertical bounding

---

## Insights crystallized

### Attention IS inference

The foveal trajectory IS the computation. No separate thinking step.
- Each glimpse: a projection that collapses uncertainty in belief space
- GRU hidden state IS the belief state
- First glimpse bootstraps the entire chain — almost determines everything
- Deterministic at inference: one forward pass, one trajectory, no search

Example narrowing trajectory:
1. "tall letter, left-leaning ink" → eliminates 80% of candidates
2. "closed loop at top" → b, d, p, or q
3. "descender" → p or q
4. "loop opens right" → p

### Efficiency vs brute force

- Foveal: 7 glimpses x 12x12 = ~1,000 pixels per letter (6% of image)
- CNN: processes all 16,384 pixels
- Cost: O(glimpses) not O(pixels) — scales to any image size
- Trade-off: harder to train (must learn WHAT to compute), but each layer is verifiable

### Recursive composed training

The training strategy mirrors the architecture:
- **Inner loop** (retry until success) = foveal read trajectory
- **Position loop** (curriculum) = word-level scan
- **Epoch loop** (generalization) = sentence-level view

The thesis isn't just the architecture — it's the training strategy too.

### Fractal composition: train, freeze, become oracle

The training strategy is fractal — the same pattern at every level of the hierarchy:

1. Train the layer to succeed at its task
2. Freeze it
3. The frozen layer becomes the oracle (loss function) for the layer above
4. The layer above retries until the oracle succeeds

The frozen stack below IS the loss function. No separate loss design needed per level.
Each layer only learns one thing: "how good does my output need to be for the next layer to work?"

The retry loop is the universal training mechanism:
```
loop {
    propose → evaluate with frozen oracle → update if failed → break if succeeded
}
```

This works because each level defines a clear interface contract (a position vector) and
the oracle below tells you if your contract is met.

### The paradigm resistance

It took five attempts to arrive at coordinate origin translation — a ~30 line change —
despite having an AI assistant trained on the entire ML literature. The sequence:

1. **Backprop through frozen model** — gradient through h-pathway. Doesn't work:
   frozen scan GRU gates close on word images, zero useful gradient. (Run 8: att=500)
2. **Direct position regression** — MSE on positions. Works perfectly (MSE=0 by epoch 4),
   but the frozen letter model can't read word images. Proves positioning was never the problem.
3. **Retrain letter model on word images** — 300+ line word-image training loop.
   Accuracy climbing (56% at epoch 45) but solving the wrong problem.
4. **Crop 128×128 from word image** — feed letter model its native format. The AI's
   suggestion. Rejected: violates isolation. Layers shouldn't manipulate each other's images.
5. **Origin translation** — each layer operates in its own coordinate frame. Upper layer
   provides an origin constant. ~30 lines in modules.rs. Letter model trains on isolated
   letters, works on word images unchanged. "Nothing has changed."

Every wrong turn was the same instinct: **couple the layers**. Make them see each other's
data. Retrain on the downstream distribution. Crop to bridge the gap. Let end-to-end
optimization sort it out.

The correct answer was the opposite: **decouple harder**. The letter model doesn't know
it's in a word image. It doesn't know word images exist. It operates at (0,0) in its own
frame. The origin is a constant added at the glimpse extraction boundary — invisible to the
model's internal state.

This is not just a technical observation. The dominant ML paradigm is end-to-end coupling:
throw everything in, let backprop figure it out. This reflex is so deeply embedded that an
AI trained on the field's literature actively resisted the simpler, decoupled solution —
repeatedly reaching for complex coupled alternatives before arriving at the principled one.

The field's instinct is coupling. The correct instinct here is isolation. That gap is the
paradigm itself.

### Isolation is the feature

Standard ML: "throw everything in, end-to-end, let backprop figure it out."
This approach inverts it: **isolation is not a limitation, it's the methodology.**

Each module is trained in isolation BECAUSE:
- Forces clear interfaces (position vectors, letter identities)
- Failures are diagnosable (which interface broke?)
- Each module is verifiable before composition
- Training can be exotic per-module (retry loops, curriculum) without affecting others
- Composed system is debuggable because each piece was proven independently

The irony: looks harder upfront (more modules, more stages, more design). Actually easier
because you're never lost. End-to-end is "easy to start, impossible to debug." This is
"harder to start, straightforward to debug."

Run 5 proved this viscerally: position 1 in isolation → 92% with attempts dropping fast.
Two positions sharing gradients → catastrophic interference, back to 25%. The fix was
more isolation (one position per batch), not less.

The hierarchy of concerns:
```
SubScan: "given a position, find and read the letter" — pure spatial, no semantics
Word model: "these letters form a word" — sequence assembly + semantic validation
Semantic layer: "this word makes sense in context" — language prior
Region gate: "is there a letter here?" — binary filter before expensive SubScan
```

Each independently trainable, each verifiable. The semantic layer doesn't know HOW letters
are read. SubScan doesn't know what a word is. Clean contracts between concerns.

Future: if semantic layer says "no English word starts with xq", it can request a re-read
of specific positions. Top-down feedback through the hierarchy — like human double-takes.

### Why Rust/flodl matters here

This session: 4 different training loop structures, 5 runs, nested retry with per-attempt
backward+step, frozen subgraphs, optimizer groups, loop-until-correct with safety cap.
Zero runtime bugs. Every run either worked or revealed a training dynamics insight.

In Python/PyTorch: memory leaks, double-backward errors, detach nightmares, 10x slower epochs.
The framework makes exotic training patterns cheap to try.

---

## Scale mismatch hypothesis (unconfirmed)

If run 5 also hits a wall, the leading theory becomes image scale mismatch:
- Letter model trained on 128x128 — glimpse covers 9.4% of width
- Word image is 128x256 — same glimpse covers 4.7% of width
- grid_sample in normalized coords → patches see wider spatial extent
- Neighboring letters visible in patches → distribution shift

**Fix if confirmed**: crop 128x128 window from word image at SubScan's position before
feeding to letter model. Letter model always operates at native scale.

---

## GPU-resident training (future, flodl wishlist)

Full model stack (~15GB) fits in H100 80GB HBM3 with room for:
- Adam state (30GB), activations (10GB), entire dataset (<1GB)
- Zero CPU-GPU transfer — everything lives in VRAM
- CUDA Graphs: capture full batch cycle, eliminate kernel launch overhead
- Natural multi-GPU: one layer per device, [B, 2] position handoff
