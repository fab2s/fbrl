# Word Model Read Phase: From Free to Anchored

Notes from v3 word training (multi-head + v5-scan transfer + AMP). The model hit 100% accuracy but the *how* revealed a structural problem in the read phase design.

## What v3 showed

The atlas made the issue clear: read fixations cluster on the discriminative letter rather than revisiting all positions. For head/hear/heat, the model learns to check only the distinguishing letter (d/r/t) and infers the rest from context. 100% accuracy but degenerate behavior — it's exploiting cross-attention's global access to skip positions it considers "easy."

This is rational optimization given the architecture. The read phase has 12 free glimpses with no structural constraint on where they go. The model discovers that a few well-placed fixations plus cross-attention readout suffice. Why visit all 4 letter positions when 2 are enough to disambiguate?

## Why this matters

The model passes the accuracy test but fails the reading test. A system that skips letters isn't reading — it's pattern matching. When words share more structure or fonts change, this strategy becomes fragile. More concretely:

- **Multi-font generalization** will suffer because the discriminative-letter shortcut depends on specific visual features that vary across fonts
- **Longer words** will break the strategy — you can't skip 6 of 8 letters
- **Isolation accuracy** stays high because it tests single-letter reading in isolation, not whether the model actually reads all positions during word reading

## The fix: scan-anchored grouped read

The insight is that the scan phase already solves the "where are the letters?" problem. The read phase should use that answer rather than re-deciding from scratch.

**Current**: scan prescribes left-to-right x, read is fully free. No connection between scan positions and read behavior. Read starts from the last scan position and wanders.

**New**: scan positions become anchors for read groups. Each read group starts at its corresponding scan position (reset location, keep h). Within each group, fixations are free. This mirrors how the single-letter model worked — scan finds the letter, read examines it in detail.

Key design choices:
- **h carries forward** across groups (left-to-right context accumulates, earlier letters inform later reading)
- **Location resets** at each group start (forces the model to revisit each position)
- **Per-group cross-attention** (query token i only attends to its group's states, cleaner than global)
- **Learnable scan x** (initialized at letter centers + boundaries, then refines via gradient from read losses flowing back through the anchor)

## Gradient flow is the mechanism

The scan positions aren't just starting points — they're differentiable anchors. Gradients from read classification flow back through the anchor into the scan x parameter. This teaches the scan: "place your fixation where reading will be most productive." The scan learns to serve the read, not just detect content.

## What to watch for in v4

- Read fixations should form 4 visible clusters, each near its letter position
- Per-group diversity should keep fixations spread within each cluster
- Scan x positions may drift from their initial letter-center values toward slightly different optima
- The model should no longer be able to skip positions — each group forces engagement with its letter
