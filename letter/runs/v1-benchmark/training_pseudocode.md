# FBRL Letter v1 — Training Pseudo Code (flodl)

## Architecture: LetterModel (identical to Python v7)

- **GlimpseSensor**: `grid_sample` foveal patches → CNN → latent embedding
- **Controller**: shared GRU + location head (predicts next fixation)
- **ScanStep**: 1 scan glimpse (learnable x, wide 12x18 patch)
- **AttentionStep**: 6 read glimpses (free x,y, 12x12 patch)
- **VisualDecoder**: deconv reconstruction from latent (BatchNorm2d)
- **Letter classifier**: FC → 26 classes
- **Case classifier**: FC → 2 classes (upper/lower)
- **Content classifier**: FC → binary (ink at scan location)

## Config

| Parameter | Value |
|-----------|-------|
| latent_dim | 256 |
| batch_size | 52 |
| epochs | 100 |
| optimizer | Adam (lr=0.001) |
| scheduler | CosineAnnealing (min_lr=0.0) |
| max_grad_norm | 5.0 |

## Data

11,440 grayscale 128x128 letter images (11 fonts, A-Z upper+lower).
Each sample: image, clean mask, letter, case, font, partner_clean.

## Training Loop

```
for epoch in 0..100:
    lr = cosine_schedule(epoch)

    for batch in shuffled_loader:
        img, clean, letter_idx, case_float, partner_clean = batch

        # ── Forward ──────────────────────────────────────────────
        # Combined scan+read loop (7 glimpses total):
        #   h0 = learned initial hidden state
        #   loc0 = center (0, 0)
        #   for step in 0..7:
        #     if step < 1: scan (wide patch, learnable x)
        #     else:        read (square patch, free x,y)
        #     glimpse = grid_sample(img, loc) → CNN → embedding
        #     h = GRU(glimpse, h)
        #     loc_next = tanh(linear(h))
        #   letter_logits = classifier(h)
        #   case_logits = case_classifier(h)
        #   latent = h
        #   recon = decoder(latent, case)
        result = model.forward(img, case)

        # ── Losses (fused single-node ops in flodl) ──────────────
        recon_loss  = MSE(result.recon, img)
        letter_loss = cross_entropy(result.letter_logits, letter_idx)
        case_loss   = cross_entropy(result.case_logits, case_idx)

        # Recode: decode with flipped case, compare to partner
        flipped_case = 1.0 - case_float
        recode_img   = decoder(latent, flipped_case)
        recode_loss  = MSE(recode_img, partner_clean)

        # Content: BCE on whether scan fixation has ink
        for loc, logit in scan_locations:
            sampled = grid_sample(clean, loc)
            label = (sampled > 0.1).float()
            content_loss += BCE_with_logits(logit, label)
        content_loss /= n_scan

        # Attention guide: gaussian blur clean → sample at fixations
        scan_guide = attention_guide(clean, scan_locations, sigma_ratio=0.16)
        read_guide = attention_guide(clean, read_locations, sigma_ratio=0.16)

        # Diversity: repel fixations from each other
        scan_div = diversity(scan_locations, sigma=0.1, vy=0.3)
        read_div = diversity(read_locations, sigma=0.1, vy=1.0)

        # Void repulsion: penalize fixations on empty space
        scan_void = void_repulsion(clean, scan_locations, patch=12x18)
        read_void = void_repulsion(clean, read_locations, patch=12x12)

        # ── Total ────────────────────────────────────────────────
        total = letter_loss + case_loss
              + recon_loss
              + 1.0 * recode_loss
              + 0.5 * content_loss
              + 8.0 * scan_guide + 0.0 * read_guide
              + 1.0 * (scan_div + read_div)
              + 1.5 * scan_void + 0.5 * read_void

        # ── Backward + step ──────────────────────────────────────
        zero_grad(set_to_none=true)
        total.backward()
        clip_grad_norm(params, max_norm=5.0)
        optimizer.step()            # batched Adam (single C++ call)
        graph.detach_state()        # break gradient chain

    # End of epoch
    graph.flush()                   # batch means → epoch history
    # Checkpoint every 50 epochs (async: snapshot_cpu → gzip → CpuWorker)
    # Live monitor updated per-epoch (dashboard at :3000)
```

## Implementation Details

| Aspect | Detail |
|--------|--------|
| Framework | flodl 0.1.1 (Rust on libtorch 2.10+cu126), native autograd |
| Checkpoints | `snapshot_cpu` → gzip → background thread (non-blocking) |
| Logging | Streaming to file per-epoch + live monitor dashboard |
| Data loading | Native PNG decode → Tensor, pre-batched in memory |
| Fused ops | Losses (MSE/CE/BCE → 1 node each), BatchNorm, GRU, Linear |
