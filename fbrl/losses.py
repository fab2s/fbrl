import torch
import torch.nn.functional as F


# --- Attention Guide Loss ---

def attention_content_loss(image, locations, blur_sigma_ratio=0.16):
    """Guide fixations toward letter strokes using a blurred guidance field.

    blur_sigma is computed as a fraction of min(H, W), so the guidance field
    auto-scales to any image/crop size. Default ratio 0.16 matches the proven
    recipe (15px at 96x96, 20px at 128x128).
    """
    B, C, H, W = image.shape
    blur_sigma = blur_sigma_ratio * min(H, W)

    # Build a 1D Gaussian kernel, then apply it as separable H/W convolutions.
    # This creates a soft "scent field" around letter strokes — bright where
    # strokes are, fading smoothly into the background. The blur radius
    # determines how far from a stroke a fixation can land and still get reward.
    k = int(4 * blur_sigma) | 1  # kernel size (odd, covers ~4 sigma each side)
    x = torch.arange(k, device=image.device, dtype=image.dtype) - k // 2
    gauss = torch.exp(-x ** 2 / (2 * blur_sigma ** 2))
    gauss = gauss / gauss.sum()
    # Separable blur: convolve rows then columns (faster than 2D kernel)
    guide = F.conv2d(image, gauss.view(1, 1, k, 1), padding=(k // 2, 0))
    guide = F.conv2d(guide, gauss.view(1, 1, 1, k), padding=(0, k // 2))

    # Sample the guide field at each fixation point. Higher value = fixation
    # landed near a stroke. We maximize this (hence return negative = loss).
    total = 0
    for loc in locations[1:]:  # skip locations[0] which is the fixed start point
        grid = loc.view(B, 1, 1, 2)
        sampled = F.grid_sample(guide, grid, align_corners=True, padding_mode='zeros')
        total = total + sampled.mean()
    # Negate: we want to MAXIMIZE guide values (fixations on strokes),
    # but optimizers MINIMIZE loss. So loss = -average_guide_value.
    return -total / len(locations[1:])


# --- Temporal Attention Guide (left-to-right scaffold for bigrams) ---

def temporal_attention_content_loss(image, locations, blur_sigma_ratio=0.16,
                                     scaffold_weight=1.0):
    """Position-aware attention guide that teaches left-to-right scanning.

    Divides glimpses into three equal phases:
      Phase 1 (first third):  guide from LEFT half of image only
      Phase 2 (middle third): guide from RIGHT half of image only
      Phase 3 (final third):  guide from FULL image (holistic)

    scaffold_weight controls the blend between position-specific and full guides:
      1.0 = full scaffolding (position-specific guides for phases 1-2)
      0.0 = no scaffolding (equivalent to standard attention_content_loss)

    The scaffold is designed to be annealed over training: teach the scanning
    pattern early, then remove the training wheels once classification reward
    sustains the behavior on its own.
    """
    B, C, H, W = image.shape
    blur_sigma = blur_sigma_ratio * min(H, W)

    # Build Gaussian blur kernel (same as attention_content_loss)
    k = int(4 * blur_sigma) | 1
    x = torch.arange(k, device=image.device, dtype=image.dtype) - k // 2
    gauss = torch.exp(-x ** 2 / (2 * blur_sigma ** 2))
    gauss = gauss / gauss.sum()

    def _blur(img):
        out = F.conv2d(img, gauss.view(1, 1, k, 1), padding=(k // 2, 0))
        return F.conv2d(out, gauss.view(1, 1, 1, k), padding=(0, k // 2))

    # Full guide field (used for holistic phase and as the non-scaffold baseline)
    guide_full = _blur(image)

    if scaffold_weight > 0:
        # Create masked images: left half and right half
        # Mask before blurring so the guide field fades naturally at the boundary
        mid = W // 2
        left_img = image.clone()
        left_img[:, :, :, mid:] = 0   # zero out right half
        right_img = image.clone()
        right_img[:, :, :, :mid] = 0  # zero out left half

        guide_left = _blur(left_img)
        guide_right = _blur(right_img)

    # Divide glimpses into three phases (equal thirds)
    n_locs = len(locations) - 1  # skip locations[0] (fixed start point)
    phase1_end = n_locs // 3           # end of left-letter phase
    phase2_end = 2 * n_locs // 3       # end of right-letter phase

    total = 0
    for t, loc in enumerate(locations[1:]):
        grid = loc.view(B, 1, 1, 2)

        if scaffold_weight > 0 and t < phase1_end:
            # Phase 1: left letter — blend left guide with full guide
            guide = scaffold_weight * guide_left + (1 - scaffold_weight) * guide_full
        elif scaffold_weight > 0 and t < phase2_end:
            # Phase 2: right letter — blend right guide with full guide
            guide = scaffold_weight * guide_right + (1 - scaffold_weight) * guide_full
        else:
            # Phase 3 (holistic) or scaffold disabled: full guide
            guide = guide_full

        sampled = F.grid_sample(guide, grid, align_corners=True, padding_mode='zeros')
        total = total + sampled.mean()

    return -total / n_locs


# --- Two-Phase Attention Loss (scan + read) ---

def two_phase_attention_loss(image, locations, n_scan, blur_sigma_ratio=0.16,
                              scaffold_weight=1.0):
    """Separate attention guides for scan and read phases.

    Scan phase (locations[1:n_scan+1]): full-image guide — land on any content.
    Read phase (locations[n_scan+1:]): temporal scaffold — left/right/full thirds.

    Returns (scan_attn_loss, read_attn_loss) separately for independent weighting.
    """
    B, C, H, W = image.shape
    blur_sigma = blur_sigma_ratio * min(H, W)

    k = int(4 * blur_sigma) | 1
    x = torch.arange(k, device=image.device, dtype=image.dtype) - k // 2
    gauss = torch.exp(-x ** 2 / (2 * blur_sigma ** 2))
    gauss = gauss / gauss.sum()

    def _blur(img):
        out = F.conv2d(img, gauss.view(1, 1, k, 1), padding=(k // 2, 0))
        return F.conv2d(out, gauss.view(1, 1, 1, k), padding=(0, k // 2))

    guide_full = _blur(image)

    # --- Scan loss: full-image guide for all scan locations ---
    scan_locs = locations[1:n_scan + 1]  # skip locations[0] (fixed start)
    scan_total = 0
    for loc in scan_locs:
        grid = loc.view(B, 1, 1, 2)
        sampled = F.grid_sample(guide_full, grid, align_corners=True, padding_mode='zeros')
        scan_total = scan_total + sampled.mean()
    scan_loss = -scan_total / max(len(scan_locs), 1)

    # --- Read loss: temporal scaffold (left/right/full thirds) ---
    read_locs = locations[n_scan + 1:]
    n_read = len(read_locs)
    if n_read == 0:
        return scan_loss, torch.tensor(0.0, device=image.device)

    if scaffold_weight > 0:
        mid = W // 2
        left_img = image.clone()
        left_img[:, :, :, mid:] = 0
        right_img = image.clone()
        right_img[:, :, :, :mid] = 0
        guide_left = _blur(left_img)
        guide_right = _blur(right_img)

    phase1_end = n_read // 3
    phase2_end = 2 * n_read // 3

    read_total = 0
    for t, loc in enumerate(read_locs):
        grid = loc.view(B, 1, 1, 2)
        if scaffold_weight > 0 and t < phase1_end:
            guide = scaffold_weight * guide_left + (1 - scaffold_weight) * guide_full
        elif scaffold_weight > 0 and t < phase2_end:
            guide = scaffold_weight * guide_right + (1 - scaffold_weight) * guide_full
        else:
            guide = guide_full
        sampled = F.grid_sample(guide, grid, align_corners=True, padding_mode='zeros')
        read_total = read_total + sampled.mean()
    read_loss = -read_total / n_read

    return scan_loss, read_loss


# --- Fixation Diversity Loss ---

def fixation_diversity_loss(locations, sigma=0.1, vy=1.0):
    """Pairwise repulsion between fixation points.

    Gaussian RBF kernel: fixations closer than ~sigma (in [-1,1] coords)
    repel each other strongly. At sigma=0.1, that's ~10% of image width.

    vy > 1.0 makes vertical proximity more expensive, encouraging fixations
    to spread vertically (useful for letters with ascenders/descenders).
    """
    # Stack all fixation points into (B, T, 2) tensor
    locs = torch.stack(locations[1:], dim=1)
    B, T, _ = locs.shape
    # Compute pairwise distances between all fixation pairs
    diff = locs.unsqueeze(2) - locs.unsqueeze(1)  # (B, T, T, 2)
    # Scale y-component: vy > 1 makes vertical proximity more expensive
    if vy != 1.0:
        scale = torch.tensor([1.0, vy], device=locs.device)
        diff = diff * scale
    dist_sq = (diff ** 2).sum(-1)                  # (B, T, T)
    # Gaussian RBF: close pairs -> repulsion near 1.0, far pairs -> near 0.0
    # This penalizes fixations that cluster together (forces spatial spread)
    repulsion = torch.exp(-dist_sq / (2 * sigma ** 2))
    # Zero out self-pairs (distance of a point to itself is always 0)
    mask = 1 - torch.eye(T, device=locs.device)
    return (repulsion * mask).mean()


# --- Fixation Edge Loss (bigrams only) ---

def fixation_edge_loss(locations):
    """Penalize fixations near the horizontal center of the image.

    In [-1,1] coords: center=0, edges=±1.
    Loss = mean(1 - |x|): 0 at edges, 1 at center.
    Gives a soft gradient pushing fixations toward the left and right
    sides of the image — useful for bigrams where the model needs to
    visit both letters' outer edges, not just the inner halves.
    """
    locs = torch.stack(locations[1:], dim=1)  # (B, T, 2)
    return (1 - locs[:, :, 0].abs()).mean()


# --- Fixation Hit Rate (diagnostic) ---

def fixation_hit_rate(image, locations, threshold=0.3):
    """Fraction of fixations that land on actual letter pixels (sharp image).

    Samples the raw (unblurred) image at each fixation point.
    A hit is when the sampled intensity exceeds threshold.
    Returns hit_rate (0-1) and mean sampled intensity.
    """
    B, C, H, W = image.shape
    hits = 0
    total_intensity = 0
    n = 0
    for loc in locations[1:]:
        grid = loc.float().view(B, 1, 1, 2)
        sampled = F.grid_sample(image, grid, align_corners=True, padding_mode='zeros')
        intensity = sampled.mean(dim=0).squeeze()  # average over batch
        total_intensity += intensity.item()
        hits += (sampled.squeeze() > threshold).float().sum().item()
        n += B
    return hits / max(n, 1), total_intensity / len(locations[1:])


# --- Word Attention Loss (scan + read with N-position scaffold) ---

def word_attention_loss(image, scan_locations, read_locations, n_positions=4,
                         blur_sigma_ratio=0.16, scaffold_weight=1.0,
                         read_group_boundaries=None):
    """Separate attention guides for scan and read phases of word reading.

    Scan phase: full-image guide — pulls y onto strokes.
    Read phase: divide into n_positions equal temporal segments.
      Segment k gets guide from horizontal stripe
      [k*W/n_positions, (k+1)*W/n_positions] of the image.

    Callers extract scan/read locations using phase_tags (works for both
    interleaved and non-interleaved modes).

    When read_group_boundaries is provided, group assignment is structural
    (from boundaries list) rather than temporal (t * n_positions // n_read).

    Returns (scan_attn_loss, read_attn_loss) separately for independent weighting.
    """
    B, C, H, W = image.shape
    blur_sigma = blur_sigma_ratio * min(H, W)

    k = int(4 * blur_sigma) | 1
    x = torch.arange(k, device=image.device, dtype=image.dtype) - k // 2
    gauss = torch.exp(-x ** 2 / (2 * blur_sigma ** 2))
    gauss = gauss / gauss.sum()

    def _blur(img):
        out = F.conv2d(img, gauss.view(1, 1, k, 1), padding=(k // 2, 0))
        return F.conv2d(out, gauss.view(1, 1, 1, k), padding=(0, k // 2))

    guide_full = _blur(image)

    # --- Scan loss: full-image guide for all scan locations ---
    scan_total = 0
    for loc in scan_locations:
        grid = loc.view(B, 1, 1, 2)
        sampled = F.grid_sample(guide_full, grid, align_corners=True, padding_mode='zeros')
        scan_total = scan_total + sampled.mean()
    scan_loss = -scan_total / max(len(scan_locations), 1)

    # --- Read loss: n_positions horizontal stripe scaffold ---
    n_read = len(read_locations)
    if n_read == 0:
        return scan_loss, torch.tensor(0.0, device=image.device)

    # Build per-position stripe guides (mask image to horizontal stripe, then blur)
    stripe_guides = []
    if scaffold_weight > 0:
        stripe_w = W // n_positions
        for p in range(n_positions):
            stripe_img = image.clone()
            # Zero out everything outside this stripe
            if p > 0:
                stripe_img[:, :, :, :p * stripe_w] = 0
            if p < n_positions - 1:
                stripe_img[:, :, :, (p + 1) * stripe_w:] = 0
            stripe_guides.append(_blur(stripe_img))

    # Divide read glimpses into n_positions segments.
    # When read_group_boundaries provided: structural assignment from boundaries.
    # Otherwise: equal temporal segments (t * n_positions // n_read).
    if read_group_boundaries is not None:
        # Build per-timestep group mapping from boundaries
        _seg_map = {}
        for gi, start in enumerate(read_group_boundaries):
            end = read_group_boundaries[gi + 1] if gi + 1 < len(read_group_boundaries) else n_read
            for tt in range(start, end):
                _seg_map[tt] = gi

    read_total = 0
    for t, loc in enumerate(read_locations):
        grid = loc.view(B, 1, 1, 2)
        if scaffold_weight > 0:
            if read_group_boundaries is not None:
                seg = _seg_map.get(t, n_positions - 1)
            else:
                seg = min(t * n_positions // n_read, n_positions - 1)
            guide = scaffold_weight * stripe_guides[seg] + (1 - scaffold_weight) * guide_full
        else:
            guide = guide_full
        sampled = F.grid_sample(guide, grid, align_corners=True, padding_mode='zeros')
        read_total = read_total + sampled.mean()
    read_loss = -read_total / n_read

    return scan_loss, read_loss
