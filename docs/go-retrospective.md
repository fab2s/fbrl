# Go/goDl Retrospective

Lessons learned from the Go implementation (archived in `goDl/`), and how they shaped the Rust/floDl rewrite.

## Architecture (validated, carried forward)

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
- **4-phase cleanup was a band-aid.** nil gradFn, refcounting, saved tensor Release, CUDA OOM callback — each solved one leak path but the root cause (GC ignorance of true memory cost) remained.
- **CGo overhead adds up.** ~150ns per tensor op. In tight training loops with thousands of small ops per batch, this becomes measurable.
- **autograd.Scope got 98% GPU utilization** but caused 1.4GB spill on a 6GB card. Aggressive scoping helped but couldn't prevent the GC from hoarding stale tensors.

## What we did differently in Rust

These were recommendations coming out of the Go experience. All have been implemented in floDl.

- **RAII solves cleanup in one phase.** Drop trait on Tensor/Variable frees VRAM deterministically. No GC, no finalizers, no 4-phase workarounds. VRAM usage is stable at ~2.1GB and *decreasing* over epochs — something that never happened with Go or Python.
- **Native libtorch autograd.** Instead of reimplementing backward passes in Rust, floDl delegates to libtorch's C++ autograd engine. Same kernels, same accumulation order, same numerical behavior as PyTorch. This required careful lifecycle management — `reset()` and `detach_state()` on Module to prevent stale grad_fns from leaking across batches — but gives numerical parity by construction.
- **Observation built in from the start.** The collect/flush/trend system was ported early. Graph-native `record_scalar`, `flush`, `trend` with slope detection and convergence analysis. Live HTML plots and CSV export. This paid for itself immediately during the autograd migration.
- **Testing on real data early.** The first real training run caught device mismatch bugs and autograd lifecycle issues that synthetic smoke tests never would.

## Performance

Early Rust training is at practical parity with the Python reference:

| Metric | Python | Rust/floDl |
|--------|--------|------------|
| Epoch time | ~50s | ~52s |
| GPU utilization | ~80% | ~82% |
| VRAM (stable) | ~3.5GB | ~2.1GB |

The remaining 2s gap is mostly in the backward pass — libtorch autograd overhead from the Rust→C++ FFI boundary. VRAM is the clear win: deterministic Drop cleanup keeps memory tight without any manual management.

Full training results will follow once the 100-epoch run completes.
