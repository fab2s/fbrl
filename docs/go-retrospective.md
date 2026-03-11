# Go/goDl Retrospective

Lessons learned from the Go implementation (archived in `goDl/`), carried forward into Rust/rDl.

## Architecture (validated, carry forward)

- **Graph-based recurrent attention works.** 22 epochs, letter accuracy 3% to 27%, clearly converging. Python reference (v8) reached 100% with 9 glimpses.
- **9 glimpses** is the sweet spot for letter recognition.
- **Hyperparameters that matter:**
  - `scan_guide_weight=8.0`, `guide_weight=0.0` for reads
  - Isotropic diversity (`diversity_vy=1.0`)
  - `blur_sigma_ratio=0.16`
  - CosineAnnealingLR scheduler
- **Motor must not touch encoder.** v4 proved co-adaptation: the motor learns to exploit encoder quirks instead of developing genuine reading strategy.
- **Observation system is essential.** collect/flush/trend pattern proved critical for debugging convergence. Don't treat metrics as an afterthought.

## VRAM and GC (why we pivoted)

- **GC has zero visibility into VRAM behind FFI wrappers.** A tiny Go struct can hide megabytes of GPU memory. The GC sees a few bytes, has no pressure to collect, and VRAM fills silently. This is fundamental to any GC language with FFI tensor bindings, not a goDl-specific bug.
- **4-phase cleanup was a band-aid.** nil gradFn, refcounting, saved tensor Release, CUDA OOM callback -- each solved one leak path but the root cause (GC ignorance of true memory cost) remained.
- **CGo overhead adds up.** ~150ns per tensor op. In tight training loops with thousands of small ops per batch, this becomes measurable.
- **autograd.Scope got 98% GPU utilization** but caused 1.4GB spill on a 6GB card. Aggressive scoping helped but couldn't prevent the GC from hoarding stale tensors.

## What to do differently in Rust

- **RAII solves cleanup in one phase.** Drop trait on Tensor/Variable frees VRAM deterministically. Don't over-engineer memory management -- the language handles it.
- **Batch transfer is the only remaining concern.** Loader creates CPU tensors, single `to_device()` per batch. No per-op device juggling needed.
- **Build observation in from the start.** The trend analysis (slope, convergence detection, stall detection) saved hours of manual log reading in Go. Wire it up early in the Rust training loop.
- **Test on real data early.** Synthetic smoke tests catch crashes but not convergence issues. The real training run (22 epochs) revealed things unit tests never would.
