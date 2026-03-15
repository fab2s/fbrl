# floDl Performance Report: FBRL Letter Training

## Context

FBRL trains a foveal attention model (1 scan + 6 read glimpses) on 128x128 letter images.
The architecture is identical in Python/PyTorch and Rust/floDl — same model, same loss stack, same data.

| | Python/PyTorch | Rust/floDl |
|---|---|---|
| **latent_dim=256** | **~50s/epoch** | **~71s/epoch** |
| latent_dim=128 | not tested | ~52s/epoch |
| VRAM | ~3.5GB | ~2.2GB |
| GPU util | ~80% | 73-87% |

**The question: why is floDl 42% slower at dim=256?**

Training setup: batch_size=32, 7 glimpses/forward (1 scan + 6 read), Adam + CosineAnnealingLR, CUDA (same GPU), same Docker container with libtorch 2.10 cu126.

---

## 1. GRUCell Kernel Launch Overhead (HIGH impact)

**The single biggest factor.**

PyTorch's `nn.GRUCell` on CUDA concatenates all weight matrices and uses a **fused kernel** — roughly **2 kernel launches** per call (one GEMM for all gates, one fused gate computation).

floDl's `GRUCell` decomposes into 6 separate `Linear` modules (xr, xz, xn, hr, hz, hn). Each `Linear.forward()` does:
1. `transpose()` — 1 kernel
2. `matmul()` — 1 kernel
3. `add(bias)` — 1 kernel (when bias present)

Then gate computations add sigmoid, tanh, mul, add, sub operations as separate kernels.

**Total per GRUCell.forward():**
- floDl: **~25 kernel launches**
- PyTorch: **~2 kernel launches**

With 7 glimpses per forward pass, that's **175 vs 14 kernel launches per batch** just for the GRU. At dim=256 the matmuls are still small enough that kernel launch overhead (~5-10µs each) dominates over compute time.

**Mitigation options (for floDl):**
- Add a fused GRUCell FFI that calls libtorch's native `torch::gru_cell()` — this is a single C++ call that already exists in libtorch's ATen
- Alternatively, concatenate weight matrices in the Rust GRUCell: pack W_ir/W_iz/W_in into one `[3*hidden, input]` matrix, do one matmul, split. Same for hidden weights. Reduces 6 matmuls → 2.

---

## 2. cuDNN Benchmark Mode (MEDIUM impact)

floDl does **not** enable `cudnn.benchmark`. PyTorch also doesn't set it explicitly in FBRL, but default behavior may differ.

When enabled, cuDNN auto-tunes convolution algorithms for the specific input sizes. Since FBRL uses fixed-size convolutions (always 128x128 input, fixed patch sizes), the first-epoch warmup cost amortizes quickly.

**What's needed in floDl:**
```cpp
// In shim.cpp — add an FFI function:
extern "C" void flodl_set_cudnn_benchmark(int enable) {
    at::globalContext().setBenchmarkCuDNN(enable != 0);
}
```
Then call it once on startup from Rust. Single line of C++, single FFI binding.

---

## 3. Redundant Attention Guide Computation (MEDIUM impact)

In `fbrl/letter/train.rs`, the attention guide loss is computed for **both scan and read locations**, even though `read_guide_weight = 0.0`:

```rust
let scan_guide = attention_guide_loss(&clean_var, &result.scan_locations, ...)?;
let read_guide = attention_guide_loss(&clean_var, &result.read_locations, ...)?;
// read_guide is multiplied by 0.0 — wasted computation
```

`attention_guide_loss` calls `gaussian_blur_2d` (expensive separable convolution on [B, 1, 128, 128]) plus `grid_sample`. Computing it twice when one result is discarded wastes ~15% of loss computation time.

**Fix (in fbrl):** Skip the call when weight is zero. Trivial.

---

## 4. GPU-CPU Synchronization in Metrics (MEDIUM impact)

Per batch, the training loop forces **10 `.item()` calls** (one per recorded metric) plus the `fixation_hit_rate` function which does **7 `to_f32_vec()` transfers** (one per glimpse location).

Each `.item()` is a GPU→CPU sync that stalls the pipeline until all queued GPU work completes.

**Python does the same** (similar sync pattern), so this isn't a Rust-specific regression — but it's still a bottleneck in both.

**Mitigation options:**
- **Batch metrics**: Record raw tensors, defer `.item()` to epoch end. floDl's `record_scalar` could accept a `&Variable` and call `.item()` lazily during `flush()`.
- **Vectorize hit_rate**: Stack all 7 locations, one `grid_sample` call, threshold on-device, `.item()` once for the count. Eliminates 6 of 7 sync points.

---

## 5. Grid Creation in GlimpseSensor (LOW-MEDIUM impact)

Both Python and Rust create a fresh base grid (`linspace` + `meshgrid` + `stack` + `unsqueeze` + `expand`) on **every glimpse** — 7 times per forward pass. The base grid shape is deterministic (only the location offset varies).

**Mitigation:** Cache the base grid tensor per (scale, patch_h, patch_w, img_h, img_w, device) tuple. Only add the location offset per call. Saves ~35 tensor allocations per forward pass.

---

## 6. Linear.forward() Transpose Overhead (LOW impact)

floDl's `Linear.forward()` does `weight.transpose(0,1)` every call, which may allocate a new contiguous tensor. PyTorch caches the transposed weight or uses `F.linear()` which calls BLAS with transpose flags (no data copy).

**Mitigation:** Store weight as `[out, in]` and use `addmm` or pass a transpose flag to matmul rather than materializing the transpose.

---

## Summary: Estimated Impact

| Issue | Est. impact | Fix location | Difficulty |
|---|---|---|---|
| GRUCell decomposition (25 vs 2 kernels) | **15-25%** | floDl | Medium — FFI to `torch::gru_cell()` or weight concat |
| cuDNN benchmark mode | **5-10%** | floDl | Easy — one FFI function |
| Redundant attention_guide when weight=0 | **3-5%** | fbrl | Trivial |
| Metric .item() sync points | **3-5%** | floDl + fbrl | Medium — lazy recording API |
| Grid caching in GlimpseSensor | **2-3%** | fbrl | Easy |
| Linear transpose overhead | **1-2%** | floDl | Medium |

**Cumulative potential improvement: 29-50%**, which would bring Rust from ~71s down to ~47-50s — matching Python.

The GRU fusion and cuDNN benchmark together likely account for the bulk of the gap. The rest are optimizations that both implementations could benefit from.

---

## What Python Does Right (implicitly)

1. **PyTorch's nn.GRUCell** uses libtorch's `torch::gru_cell()` under the hood — a single fused ATen op. No Python overhead in the hot path.
2. **F.linear()** passes transpose flags to BLAS instead of materializing transposed weight tensors.
3. **pin_memory=True** on DataLoader when using CUDA — enables async CPU→GPU transfer via page-locked memory. floDl's data loader doesn't use pinned memory.
4. **Default cuDNN heuristics** — even without explicit `benchmark=True`, PyTorch's cuDNN integration may select better algorithms via heuristic mode.

## What Rust Does Right

1. **Vectorized attention_guide_loss** — stacks all locations, one `grid_sample` call. Python loops per-location.
2. **Lower VRAM** (~3.0GB at dim=256 vs 3.5GB) — tighter memory management.
3. **Explicit state management** — `detach_state()` is cleaner than Python's implicit detach.
4. **Live Monitor is free** — SSE server, real-time dashboard, HTML archiving, epoch metrics — runs every epoch with no measurable perf cost. Python has nothing comparable.
