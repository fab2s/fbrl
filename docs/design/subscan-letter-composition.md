# SubScan + Letter Composition (Step 2)

Progressive training step 2: compose a SubScan localization module with the
proven letter model. SubScan learns to locate letters within bounded regions
of a word image. The letter model's scan adapts to word context while its
read phase stays frozen.

This is the first hierarchical composition in FBRL -- the moment the trajectory
thesis either holds or doesn't. Can a pre-trained read module, frozen, correctly
classify letters when a different module positions it?

**Prerequisite:** Trained letter model checkpoint (step 1, done).
**Output:** `subscan_v1.fdl.gz` -- SubScan + LetterModel, ready for word model
composition in step 3.

---

## Architecture

```
Word image [B, 1, 128, 256]
         |
    SubScan (bounded to region)
         short wide blurred glimpses within region bounds
         2 free glimpses -> infer letter center (x, y)
         |
    LetterModel "letter" (loaded from checkpoint)
         |
         +-- Scan "scan" [trainable]
         |   starts from SubScan position, free on full image
         |   refines toward letter center
         |
         +-- Read "read" [frozen]
             proven letter classification
             -> 26-class logits + reconstruction
```

### Data flow per position

For each letter position p in [0, N):

1. **Region assignment.** Compute a noisy region around the ground truth letter
   center: `(center_x + noise, half_width)`. The noise prevents center-cheating
   and simulates MetaScan imprecision. The region is a constraint on SubScan's
   location head, not a crop.

2. **SubScan forward.** SubScan receives the full word image. Its location head
   output is reparameterized to the region:
   ```
   subscan_x = region_center_x + region_half_width * tanh(raw_x)
   subscan_y = tanh(raw_y)  // full vertical range
   ```
   Two short, wide, blurred glimpses sense horizontal ink density. SubScan
   freely chooses where to look within the region, then infers a letter center
   position -- which does not need to coincide with either glimpse location.

3. **Letter scan forward.** Letter scan receives the full word image and starts
   from SubScan's position. It is NOT bounded to the region -- if the letter
   bleeds past the region edge, scan can follow it. The step budget (1-2 steps)
   keeps it local naturally.

4. **Letter read forward.** Frozen. Takes over from scan, classifies the letter.
   Same architecture and weights as the standalone letter model.

5. **Classification.** Read outputs 26-class logits. CE loss per position.

### Why no crop

Both SubScan and LetterModel operate on the same full word image tensor. The
region is encoded as a reparameterization of SubScan's location head, not as a
data operation. This means:

- One image tensor flows through the entire pipeline
- No crop/resize artifacts at region boundaries
- SubScan sees full-image context through its blurred glimpses (neighboring
  letters are visible but unreadable at SubScan's blur level)
- Letter scan can reach slightly outside the region to capture edge letters
- The same mechanism scales to sentence models: MetaScan outputs word regions,
  WordScan outputs letter regions, SubScan outputs letter centers -- all
  operating on the same image tensor with nested reparameterized bounds

---

## SubScan module

### Purpose

Given a bounded region of a word image, find a suitable starting position for
the letter model. SubScan is a localization primitive -- it finds ink, not
reads it.

### Glimpse design

- **Short and wide.** Each glimpse is ~2/3 letter width and much shorter in
  height (e.g., ~8x28 pixels, h x w). On a 256px canvas with 4 letters
  (~64px each), 2/3 width is ~42px. The short height limits vertical letter
  structure visibility -- SubScan sees horizontal ink density, not letterforms.
  The wide aspect ratio emphasizes horizontal localization, which is the
  primary axis for finding letter centers in a word.

- **Two free glimpses.** SubScan freely chooses where to look within its
  bounded region. Two 2/3-width looks cover the region with overlap. The GRU
  integrates both partial views to triangulate the letter center. This is the
  minimum needed for spatial reasoning from partial observations: "ink density
  is here on the left, there on the right, so the center must be between them."

- **Blurred.** Aggressive blur on top of the short/wide patch. Even the partial
  view shows density, not structure. Triple insurance against SubScan learning
  to read: short height + partial width + blur.

- **Free output position.** SubScan's output (the position handed to letter
  scan) does not need to be at either glimpse location. The two glimpses are
  observations; the output is an inference. SubScan looks left-ish and
  right-ish, then points to where the letter center must be. This is genuine
  spatial reasoning, not position passthrough.

### Region and neighbor visibility

The region is centered near the target letter (with noise, see below) and
extends ~half a letter width on each side. This means SubScan sees:
- The target letter (partially, through two blurred 2/3-width glimpses)
- Fragments of neighboring letters (or void at word boundaries)

Neighbor visibility teaches SubScan about letter spacing -- where one letter
ends and another begins. At word edges, one side is void, which is also
informative. This context transfers to the word model where MetaScan will
provide similarly-sized regions with neighbor overlap.

### Why this sizing scales

In the word model, MetaScan will provide coarser, noisier regions. A SubScan
trained on partial views with neighbor fragments is pre-adapted for imprecise
regions. If SubScan trained on full-letter-width glimpses, it would overfit to
clean centered views and struggle with MetaScan's less precise positioning.
The 2/3 width and short height build in robustness from day one.

### Architecture

```
SubScan
    GlimpseSensor (~8x28, blurred, n_scales=1)
    GRUCell (hidden_dim -> hidden_dim)
    location_head (hidden_dim -> 2, reparameterized to region)
    h0: learned initial hidden state
    2 glimpse steps -> output final position
```

Minimal. Two steps, one GRU, partial views. The complexity budget goes to
localization quality, not module size.

### Region bounding

The location head output is constrained to the region:

```rust
let raw = self.loc_head.forward(&h)?;       // [B, 2]
let xy = raw.tanh()?;                        // [-1, 1]
let x = &region_center + &region_half_w * xy.narrow(1, 0, 1)?;
let y = xy.narrow(1, 1, 1)?;                // full vertical range
```

This ensures SubScan stays in its assigned region. It cannot drift to a
neighboring letter. Whatever it finds in the region IS the target letter --
the classification loss signal is unambiguous.

### Why bounding matters

Without bounding, SubScan could drift to the next letter. If it does, the
letter model correctly classifies *that* letter, but the loss says "wrong."
The system can't distinguish "SubScan pointed wrong" from "classification
failed." Bounding removes the ambiguity and matches the real operating
conditions -- in the word model, MetaScan will provide bounded regions.

---

## Region noise curriculum

### The center-cheating problem

If the region is always perfectly centered on the letter, SubScan's optimal
policy is `tanh(raw) -> 0` -- always output the center. Zero localization
learned. The model gets a free ride and never develops real spatial reasoning.
When MetaScan later provides imprecise regions, SubScan fails because it
never learned to actually find ink.

### Solution: noise from the start, increasing over training

Region centers are offset from ground truth by horizontal noise. The noise
prevents the center-cheating degenerate solution and simulates the imprecision
SubScan will face in the word model.

**Noise schedule:**

```
noise_range = noise_start + (noise_end - noise_start) * min(1, epoch / noise_ramp_epochs)
region_center = gt_center + uniform(-noise_range, +noise_range)
```

| Phase | Noise range | Purpose |
|-------|-------------|---------|
| **Start** | ~15% of letter width (~10px) | Moderate: center-cheating doesn't always work, SubScan must look. But the letter is always well within the region -- both modules can learn. |
| **End** | ~35-40% of letter width (~22-25px) | Approaches MetaScan's realistic imprecision. SubScan is now robust to significant centering error. |
| **Ramp** | Over ~50% of training epochs | Gradual increase. SubScan's localization skill develops alongside the growing challenge. |

### Why noise from the start

Starting with zero noise and scaffolding it in risks SubScan learning the
center-cheating habit early. Unlearning a degenerate solution is harder than
never forming it. Moderate noise from epoch 1 ensures SubScan always needs to
do real work, even if the work is easy at first.

### Why not full noise from the start

Letter scan is also adapting simultaneously. If SubScan gives poor positions
because noise is extreme, scan can't learn a useful starting-position-to-
refinement mapping. Both modules need a curriculum -- moderate challenge that
they can jointly solve, increasing as they get better.

### Region width vs noise interaction

The region half-width is ~1 letter width (covering the target plus half a
neighbor on each side). With 35-40% noise, the target letter can shift
significantly within the region, but is always contained. The SubScan's 2/3
letter-width glimpses combined with two free looks are enough to find the
letter even at maximum noise -- but only through actual localization, not
centering.

---

## Letter model adaptation

The existing letter model (v2, separate ScanStep + AttentionStep GRUs) needs
minimal changes for composition:

### 1. Accept initial scan position

Currently scan starts from `(0, 0)` (image center). In composed mode, scan
starts from SubScan's output position. This requires:

- A method to set the initial location: `scan.set_initial_location(pos)`
- The scan GRU's h0 remains learned (not coupled to SubScan state)
- Only the starting position changes -- the scan loop is otherwise identical

### 2. Respect `is_composed()`

When `letter_graph.is_composed()` is true:
- Skip standalone loss computation (parent handles loss)
- Skip standalone observation/monitoring
- The read module's classification output flows to the parent

### 3. Scan step count

In standalone training, scan runs 1+ steps from center. In composed mode,
SubScan already found the approximate position. Scan may need fewer steps
(or the same count but converging faster). This is tunable -- start with the
same count and observe.

---

## Training

### Data

Use the existing word dataset (128x256, 4 letters per word). For each letter
position, we know the ground truth position and identity from the dataset
metadata.

Region assignment per position: compute from ground truth letter positions.
The region center (with noise) and half-width are derived from the data --
not learned.

### Progressive difficulty

The noise curriculum (above) provides a natural difficulty progression. On top
of that, three experimental levels:

**Level 1: Two SubScan glimpses, moderate noise.**
The baseline configuration. Validates the core pipeline: SubScan localizes
within a noisy region, letter scan refines, frozen read classifies.

**Level 2: Increased noise + more fonts.**
Push noise toward the upper range. Add font diversity if the word dataset
supports it. Tests robustness of the localization + classification pipeline.

**Level 3: Retry logic.**
SubScan proposes -> letter model attempts -> confidence check -> if low,
SubScan adjusts within the same region. This is a learned halt pattern.
The interesting research question: does the model learn to recognize its own
uncertainty and self-correct?

### Loss

Simpler than the flat word model. The hierarchical structure removes the need
for most scaffolding:

| Term | Purpose | Weight |
|------|---------|--------|
| **Per-letter CE** | From frozen read's classification. The primary signal. Flows back through scan and SubScan. | 1.0 |
| **Reconstruction MSE** | From frozen read's decoder. Forces scan to gather enough information. | 1.0 |
| **SubScan diversity** | Prevent all SubScan positions from collapsing to the same point. | 1.0 |
| **Scan guide** | Optional. May not be needed -- classification loss may suffice. | 0.0 (experiment) |

**No scaffold loss.** The flat model needed stripe scaffolding because reads
had no position guidance. Here, SubScan provides the position and the region
bound prevents ambiguity.

**No content head.** SubScan's blur-level glimpses serve the same purpose --
finding ink. A separate content detection head may be redundant.

**No isolation loss.** The letter read module is frozen and already proven on
isolated letters. It doesn't need additional per-letter supervision.

### Optimizer groups

Two parameter groups via graph tree paths:

```rust
// SubScan learns to locate (higher LR, new module)
let subscan_params = graph.parameters_at("subscan")?;

// Letter scan adapts to word context (lower LR, pretrained)
let scan_params = graph.parameters_at("letter.scan")?;

let optimizer = Adam::with_groups()
    .group(&subscan_params, 0.001)
    .group(&scan_params, 0.0001)
    .build();
```

Letter read is frozen -- not in any optimizer group.

### Freeze/thaw via graph tree

```rust
graph.freeze("letter.read")?;
// letter.scan stays trainable (default)
// subscan stays trainable (default)
```

### Training loop (per batch)

```
for each epoch:
    noise_range = compute_noise(epoch)
    for each batch:
        for each position p in 0..4:
            region = compute_noisy_region(p, batch, noise_range)
            subscan_pos = subscan.forward(image, region)
            letter_result = letter.forward(image, subscan_pos)
            loss_p = ce(letter_result.logits, target[p]) + recon(letter_result.recon, image)

        total_loss = sum(loss_p) + diversity(subscan_positions)
        total_loss.backward()
        optimizer.step()
```

The 4 position calls share LetterModel weights. Gradients accumulate across
positions before the optimizer step.

---

## Graph structure

```rust
// Step 1 output: load proven letter model
let letter = build_letter_model(&letter_cfg)?;
letter_graph.load_checkpoint("letter_v8.fdl.gz")?;

// Build subscan
let subscan = SubScan::new(hidden_dim, glimpse_cfg)?;

// The composition uses graph tree for freeze/thaw and parameter groups,
// but the 4x position loop lives in the training code (not in the graph)
// because each invocation needs different region bounds.

// Freeze read phase
letter_graph.freeze("read")?;

// After training, save the composed unit
// (SubScan params + letter.scan adapted params + letter.read frozen params)
graph.save_checkpoint("subscan_v1.fdl.gz")?;
```

### Why the position loop is outside the graph

Each position needs different region bounds (SubScan reparameterization) and
maps to a different ground truth label. This is data-dependent routing that
the training loop handles naturally. The graph handles the per-position
forward pass; the training loop handles the iteration and loss aggregation.

When `each(N)` lands in flodl (future work), this loop could move into the
graph. But it's not a blocker -- the training loop approach works and is
explicit.

---

## Scaling path

The design is fractal. Each level uses the same pattern: bounded region +
localization + handoff to the next level.

```
Sentence model (future):
    MetaWordScan (bounded to line region)
        -> word centers
    WordModel (bounded to word region)
        MetaLetterScan (bounded to word region)
            -> letter centers
        SubScan (bounded to letter region)
            -> refined letter center
        LetterModel
            Scan (free) -> Read (frozen) -> classify
```

The region reparameterization composes naturally: each level's output position
becomes the next level's region center. The bounds narrow at each level. The
full image tensor flows through unchanged. No crops at any level.

Each level is trained independently and checkpointed. A better letter model
drops into any word model without retraining. A better word model drops into
any sentence model. The composition is structural, not entangled.

---

## Success criteria

**Minimum viable result:** Frozen read classifies correctly when SubScan +
trainable scan position it, on at least one word image. This proves the
composition principle works.

**Target:** Per-position accuracy comparable to the flat Python word model
(~99%+) on the same word dataset. If this holds, the hierarchical approach
is validated and step 3 (full word model with MetaScan) is justified.

**Research questions:**
1. Does the classification loss alone teach SubScan to localize, or does it
   need explicit position supervision?
2. Does letter scan need attention guidance in word context, or does the
   classification signal suffice?
3. What is the optimal noise ramp schedule? How much centering imprecision
   can SubScan handle before accuracy degrades?
4. What region width gives the best trade-off between SubScan difficulty
   and position ambiguity?
5. Is 2/3 letter width the right glimpse size, or does SubScan benefit from
   seeing more (risking reading) or less (risking too little information)?
6. Does the retry mechanism (level 3) improve accuracy, and does the model
   learn meaningful confidence-gated behavior?

---

## Relationship to other documents

- **Graph tree design:** `../rdl/docs/design/graph-tree.md` -- the infrastructure
  this composition is built on. Step 2 is the driving use case from that doc.
- **Trajectory thesis:** `docs/trajectory-thesis.md` -- this is the first
  hierarchical composition test. Part IV (mixture of strategies, hierarchical
  composition) predicts this should work.
- **Letter experiments:** `docs/letters.md` -- the proven letter model that
  provides the frozen read module.
- **Word experiments:** `docs/words.md` -- the flat model this replaces. The
  lessons (prescribed scan, scaffold, isolation) informed the design but the
  architecture is fundamentally different.
