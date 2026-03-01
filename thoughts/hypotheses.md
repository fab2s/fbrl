# Research Hypotheses & Brainstorming

Raw ideas and testable hypotheses driving the project direction. Not proven — just the thinking behind the roadmap.

## Core Intuition: The Biological Feedback Loop

The project name (Feedback Recursive Loop) comes from a motor learning analogy. When a leg moves, it produces a sensory signal that the brain encodes. To reproduce the movement, the brain sends back a similar signal. The feedback loop IS the learning mechanism — the discrepancy between intended and actual sensation refines the internal model over thousands of iterations until motor control becomes precise.

This maps directly to the model: fixate on a letter, extract a patch, encode into a latent, decode back to an image. Does the reconstruction match? If yes, the latent captured what matters. The recode step (decode the same latent as a different case, font, or modality) forces further abstraction. The loop refines itself — each cycle of encode-decode-recode reshapes the latent geometry toward more abstract, transferable representations.

## Hypothesis 1: Early Multimodal Fusion > Late Fusion

**Claim**: Training all modalities jointly from the start (vision + audio + motor) produces qualitatively better representations than training specialists and combining them later, even if initial convergence is slower.

**Biological basis**: A newborn's brain doesn't have separate "vision" and "audio" modules that later connect. Vision, sound, and motor signals arrive simultaneously in a plastic substrate and compete for the same neural territory. The representations that emerge are multimodal from the ground up. That's why a human can hear a word they've never seen written and still have an intuition about how it looks — the modalities were never separate.

**Analogy to child learning**: A child learning to read the old way — the teacher speaks out what they write on the board while children reproduce the writing — mixes mechanical movement with sounds and vision from the ground up. Three modalities constraining the same internal representation simultaneously.

**In the architecture**: The GRU controller would receive gradients not just from "did you reconstruct the image?" but simultaneously from "did you produce the right sound?" and "did you trace the right motor path?" A much harder optimization landscape — but the controller that survives it has learned something deeper. It finds features that are diagnostic *across modalities*, not just within vision.

**Existing evidence (small scale)**: The single-letter model already trains with 3 simultaneous loss signals (reconstruction + classification + recode). This miniature version of joint training produced abstract letter identity that transfers across 11 fonts and both cases with only a 256-dim latent. None of the 3 losses alone would achieve this — it's the combination from the start that forces genuine abstraction. If 3 constraints produce font-invariant, case-invariant identity, what would 5 constraints produce?

**The bet**: Longer initial training → fundamentally more robust latent → better generalization to unseen conditions. Not just "more training = more accuracy" but "more simultaneous constraints = qualitatively different representations."

## Hypothesis 2: Motor Trace as Low-Cost Multimodal Entry Point

**Claim**: A pen trajectory decoder (sequence of x, y, pen_up/pen_down) is the cheapest multimodal signal to add, and could demonstrate the generalization benefit of early fusion even on a single GPU.

**Why motor specifically**: Writing is inherently sequential — you trace a path through time. This temporal structure mirrors the sequential nature of foveal attention. The model would learn: "the sequence of fixations that *reads* 'b' relates to the sequence of strokes that *writes* 'b'." Both are temporal policies over spatial actions. The shared GRU controller is already designed for exactly this.

**Practical advantages**: A 2D pen trajectory is tiny — just a sequence of (x, y, pen_up/pen_down) tuples. No spectrogram processing, no audio encoder. A small trajectory decoder head on the existing word model, trained with an additional "recode as handwriting" loss alongside reconstruction.

**Testable prediction**: Train the word model with and without a trajectory decoder head. Test on a novel font never seen during training. If the trajectory-augmented model generalizes better to the unseen font, that's concrete evidence: early multimodal fusion produces more robust representations than unimodal training, even when the test is purely visual.

**Data source**: Handwriting trajectory data exists in public datasets (IAM On-Line Handwriting Database, UNIPEN). Or generate synthetic trajectories from font bezier curves — less naturalistic but fully controlled.

### Motor v1 results and next directions

Motor trace decoding is now implemented and tested. Key findings from v1-transfer (200 epochs, curriculum learning from v5-scan):

**What worked**: Curriculum learning — transferring pretrained vision weights and training motor on top — dramatically outperformed training from scratch (97.9% vs 84.6% letter accuracy, 64.2% vs 47.6% re-read). Vision stays stable while motor learns against a meaningful latent.

**What didn't work**: Despite 64% re-read accuracy, most rendered trajectories aren't visually readable. The motor decoder learns structural families (D→T shapes, O→Q shapes, l/I/j collapse) but not true letter shapes. The re-read signal is too soft: fat Gaussian blobs (sigma=1.5) let vague spatial distributions pass, and the same encoder co-adapts to motor artifacts.

**Next: enhanced motor loss stack** — four complementary signals, weighted by abstraction level:
1. **Latent matching** (MSE between original and re-read latents, weight=2.0) — forces cross-modal convergence to the same abstract concept. This IS the multimodal fusion signal from Hypothesis 1.
2. **Frozen re-reader** (static encoder copy, never updates, weight=0.5) — prevents co-adaptation, honest readability check
3. **Rendered image matching** (MSE to clean image, weight=0.25) — gentle pixel anchor, prevents total divergence
4. **Sharper rendering** (sigma 1.5→0.75) — thinner lines, less forgiving blobs

**Why latent matching matters for the broader hypothesis**: if latent₂ (from rendered motor output) must match latent₁ (from font image), the motor decoder is forced to produce output that evokes the same abstract representation — regardless of visual style. This is exactly the "early fusion" mechanism: two modalities constrained to the same latent geometry. The motor trace doesn't need to look like the font; it needs to carry the same information.

**Uppercase-only first**: 26 simpler shapes (straight lines, no ascenders/descenders) to isolate the motor learning signal before tackling the full 52-letter mixed-case task.

## Hypothesis 3: Canvas Geometry as Anti-Cheating Mechanism

**Confirmed experimentally**: The ratio of foveal window size to canvas size determines whether the model can cheat (read holistically) or must develop genuine sequential reading.

- 12x12 on 128x128 (0.9%) — single letters: fine, only one thing to read
- 12x12 on 128x128 (0.9%) — bigrams: cheatable, model reads holistically from center
- 12x12 on 256x128 (0.4%) — 4-letter words: not cheatable, genuine sequential fixation required

**Implication**: When scaling to sentences or paragraphs, the canvas-to-fovea ratio will naturally enforce increasingly sophisticated reading strategies. The model won't need explicit loss engineering to read sequentially — the geometry will force it. This is analogous to how human reading develops: the visual system doesn't change, but the task demands (longer words, more complex layouts) force more efficient eye movement strategies.

## Hypothesis 4: Constraint Count Drives Representation Quality

**Claim**: There's a relationship between the number of simultaneous constraints on a latent space and the abstractness/robustness of the resulting representation. More constraints → fewer "cheating" solutions → more genuinely abstract encoding.

Current constraint progression:
- Single letter: recon + letter_cls + case_cls + recode = 4 constraints → font-invariant, case-invariant identity
- Bigram: recon + 2x pos_cls + mask_cls + attention + diversity = 7 constraints → per-position reading
- Word: recon + 4x pos_cls + attention + content + isolation = 9 constraints → sequential scanning

Each stage added constraints because the previous stage found a shortcut. The pattern suggests: the right number of constraints is "enough that no shortcut survives." Every surviving shortcut is a signal that a constraint is missing.

**Extension**: Adding motor trace and phoneme decoding would bring the word model to ~11+ constraints. The hypothesis predicts that this would not just add capabilities but would *improve existing capabilities* (better letter recognition, more efficient scanning) because the latent has to be more abstract to satisfy all constraints simultaneously.

## Future Directions

### Near-term (single GPU feasible)
- **Enhanced motor losses**: Latent matching + frozen re-reader + sharper rendering on uppercase-only, then scale to full alphabet
- **Scan-anchored grouped read for letters**: Apply the word model's grouped read strategy to single-letter models. 3 scan glimpses with learnable x → inner scan(s) anchor read groups. Forces systematic spatial examination (left/center/right strokes). Particularly valuable for motor training — gives the motor decoder better structural information about letter anatomy. Could become a standard pattern: learnable scan x → anchored grouped read at every scale.
- **Variable-length words**: Language model prior, test if model skips predictable letters (human-like)
- **Mixed case**: Uppercase + lowercase in same word, test if scan y adapts

### Medium-term (needs more compute)
- **Phoneme recode**: Add small audio decoder, test if vision+audio representations outperform vision-only on novel fonts
- **Meta-attention**: Hierarchical controller — coarse saccades find word boundaries, fine saccades read within words
- **Cross-lingual transfer**: Train on English, test if attention strategies (not letter identity) transfer to French or German

### Long-term (the bet)
- **Full multimodal from scratch**: Vision + audio + motor + language model, all training simultaneously from epoch 1. Slower convergence, but the hypothesis predicts fundamentally different (more abstract, more robust) representations than any staged or late-fusion approach.
