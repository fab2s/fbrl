import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional


# --- Glimpse Sensor ---

class GlimpseSensor(nn.Module):
    """Extracts multi-resolution patches from the RAW IMAGE at a given location.

    Strict foveal-only field of view: each glimpse sees ONLY patch_h x patch_w
    raw pixels. No peripheral context. patch_size can be int (square) or (h, w)
    tuple (rectangular — e.g. (12, 18) for wide scan patches).
    Default patch_size=12, n_scales=1:
      Scale 1: 12x12 pixels (0.9% of 128x128 image)
    With 10 glimpses: 8.8% max coverage.
    """
    def __init__(self, patch_size=12, n_scales=1, latent_dim=256):
        super().__init__()
        if isinstance(patch_size, int):
            self.patch_h = patch_size
            self.patch_w = patch_size
        else:
            self.patch_h, self.patch_w = patch_size
        self.patch_size = patch_size  # keep original for serialization
        # Each scale doubles the crop area: scale 0 = 1x, scale 1 = 2x, etc.
        # With n_scales=1, only the raw foveal patch is used (no peripheral).
        self.scales = [2**i for i in range(n_scales)]

        # Small CNN that digests the extracted patch(es) into a feature vector.
        # Input channels = n_scales (one channel per resolution scale).
        # Two stride-2 convs reduce 12x12 -> 6x6 -> 3x3, then pool to 1x1.
        self.patch_cnn = nn.Sequential(
            nn.Conv2d(n_scales, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),  # -> (B, 128, 1, 1)
            nn.Flatten(),             # -> (B, 128)
        )
        # Project patch features to latent dim
        self.glimpse_fc = nn.Sequential(
            nn.Linear(128, latent_dim),
            nn.ReLU(),
        )
        # Encode the (x, y) fixation location so the model knows WHERE it looked
        self.location_fc = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(),
        )
        # Fuse "what I see" (glimpse_fc) with "where I am" (location_fc)
        self.combine_fc = nn.Sequential(
            nn.Linear(latent_dim + 128, latent_dim),
            nn.ReLU(),
        )

    def forward(self, image, location):
        """Extract patch at location, encode it, fuse with location info.

        Returns a single vector (B, latent_dim) representing "what + where".
        """
        B, C, H, W = image.shape
        # Extract one patch per resolution scale via differentiable grid_sample
        patches = []
        for scale in self.scales:
            grid = self._make_grid(location, scale, H, W)
            patch = F.grid_sample(image, grid, align_corners=True, padding_mode='zeros')
            patches.append(patch)

        # Stack scales as channels and run through CNN
        combined = torch.cat(patches, dim=1)  # (B, n_scales, patch_size, patch_size)
        feat = self.patch_cnn(combined)        # (B, 128)

        # Fuse visual content with spatial position
        glimpse_feat = self.glimpse_fc(feat)       # "what I see"
        loc_feat = self.location_fc(location)      # "where I am"
        return self.combine_fc(torch.cat([glimpse_feat, loc_feat], dim=1))

    def _make_grid(self, location, scale, H, W):
        """Build a sampling grid for grid_sample, centered on `location`.

        Coordinates are in [-1, 1] (PyTorch grid_sample convention).
        The grid covers patch_h x patch_w pixels, scaled by `scale`.
        Adding `location` shifts the grid to the fixation point.
        """
        B = location.shape[0]
        # Convert patch extent from pixels to normalized [-1, 1] coords
        delta_h = scale * self.patch_h / H
        delta_w = scale * self.patch_w / W

        grid_y = torch.linspace(-delta_h, delta_h, self.patch_h, device=location.device)
        grid_x = torch.linspace(-delta_w, delta_w, self.patch_w, device=location.device)
        grid_y, grid_x = torch.meshgrid(grid_y, grid_x, indexing='ij')

        # grid shape: (1, patch_h, patch_w, 2) — a centered patch
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)
        # Shift the centered grid to the fixation location
        loc = location.view(B, 1, 1, 2)
        return grid + loc


# --- Attention Controller ---

class AttentionController(nn.Module):
    """GRU-based saccade planner — decides where to look next.

    The GRU accumulates information across glimpses. After each glimpse:
      1. Feed the glimpse vector into the GRU (updates hidden state)
      2. Predict next (x, y) fixation from hidden state (tanh -> [-1, 1])
      3. After all glimpses, project final hidden state to the latent vector

    The hidden state is the model's "working memory" — it integrates what
    was seen at each fixation to decide where to look next and what the
    letter is.
    """
    def __init__(self, glimpse_dim=256, hidden_dim=256, latent_dim=256):
        super().__init__()
        self.gru = nn.GRUCell(glimpse_dim, hidden_dim)
        self.location_head = nn.Linear(hidden_dim, 2)   # predict next (x, y)
        self.latent_head = nn.Linear(hidden_dim, latent_dim)  # final representation
        # Learned initial hidden state (the model learns where to start looking)
        self.h0 = nn.Parameter(torch.zeros(1, hidden_dim))

    def forward(self, image, glimpse_sensor, n_glimpses):
        B = image.shape[0]
        h = self.h0.expand(B, -1).contiguous()
        # Start at image center (0, 0) in normalized coords
        location = torch.zeros(B, 2, device=image.device)

        locations = [location]  # locations[0] = starting point
        for t in range(n_glimpses):
            # Look: extract patch at current fixation
            glimpse = glimpse_sensor(image, location)
            # Think: update working memory with what we just saw
            h = self.gru(glimpse, h)
            # Move: decide where to look next (tanh clamps to [-1, 1])
            location = torch.tanh(self.location_head(h))
            locations.append(location)

        # After all glimpses, compress hidden state into final latent
        latent = self.latent_head(h)
        return latent, locations


# --- Visual Attention Encoder ---

class VisualAttentionEncoder(nn.Module):
    """Recurrent spatial attention on raw pixels.

    No global CNN preprocessing — the model only sees what the attention
    decides to look at.
    """
    def __init__(self, n_glimpses=10, patch_size=12, n_scales=1, latent_dim=256):
        super().__init__()
        self.glimpse_sensor = GlimpseSensor(
            patch_size=patch_size, n_scales=n_scales, latent_dim=latent_dim,
        )
        self.attention_controller = AttentionController(
            glimpse_dim=latent_dim, hidden_dim=latent_dim, latent_dim=latent_dim,
        )
        self.n_glimpses = n_glimpses

    def forward(self, x):
        latent, locations = self.attention_controller(
            x, self.glimpse_sensor, self.n_glimpses,
        )
        return latent, locations


# --- Shared Encode Loop ---

@dataclass
class EncodeResult:
    """Output of encode_scan_read — everything downstream needs."""
    read_states: torch.Tensor    # (B, n_read, D) hidden states from read phase
    locations: list              # list of (B, 2) fixation points (all phases)
    latent: torch.Tensor         # (B, D) from latent_head(h_final)
    scan_content_logits: list    # list of (B, 1), empty if no content_head
    actual_n_scan: int           # scan glimpses used (may differ if dynamic)
    read_group_boundaries: list = None  # e.g. [0, 5, 10, 15] for 4 groups of 5
    phase_tags: list = None      # ['init', 'scan', 'read', ...] per location entry


def encode_scan_read(image, controller, scan_sensor, read_sensor,
                     n_scan, n_read,
                     content_head=None,
                     prescribed_x=False,
                     dynamic_width=None,
                     scan_xs=None,
                     read_group_anchors=None,
                     n_read_per_group=None,
                     interleaved=False):
    """Two-phase or interleaved GRU attention loop. Shared by all model types.

    Phase 1 — SCAN: n_scan glimpses with scan_sensor.
      If scan_xs provided: x = tanh(scan_xs) (learnable parameter), y learned.
      Elif prescribed_x: x = linspace(-0.75, 0.75, n_scan), y learned.
      If dynamic_width set: n_scan scaled by (width/256)^1.5.
    Phase 2 — READ: n_read glimpses with read_sensor, fully free x,y.
      h carries forward from scan.
      If read_group_anchors set: read is grouped; each group resets location
      to the scan position at the anchor index. h carries forward across groups.

    Interleaved mode (interleaved=True, n_read_per_group set):
      For each scan position: 1 scan glimpse + n_read_per_group flat reads.
      No position reset between scan and reads — GRU momentum preserved.
      scan_xs provides learnable x for each scan; y is learned.

    Args:
        image: (B, C, H, W) input image
        controller: AttentionController (has gru, location_head, latent_head, h0)
        scan_sensor: GlimpseSensor for scan phase (or same as read_sensor)
        read_sensor: GlimpseSensor for read phase
        n_scan: base number of scan glimpses (0 = read-only)
        n_read: number of read glimpses (ignored when read_group_anchors set)
        content_head: nn.Linear(D, 1) or None — content detection on scan states
        prescribed_x: if True, scan x positions are prescribed linspace
        dynamic_width: if set (int), scale n_scan by (width/256)^1.5
        scan_xs: Tensor or nn.Parameter of raw scan x positions (tanh applied)
        read_group_anchors: list of scan step indices for group anchoring
        n_read_per_group: glimpses per group (required when read_group_anchors set)
        interleaved: if True, alternate scan/read per position (requires n_read_per_group)
    """
    B = image.shape[0]
    h = controller.h0.expand(B, -1).contiguous()
    location = torch.zeros(B, 2, device=image.device)
    locations = [location]
    scan_content_logits = []

    # Dynamic scan count
    actual_n_scan = n_scan
    if dynamic_width is not None and n_scan > 0:
        actual_n_scan = max(1, round(n_scan * (dynamic_width / 256) ** 1.5))

    # --- Interleaved mode: scan1 → reads1 → scan2 → reads2 → ... ---
    if interleaved and n_read_per_group is not None:
        phase_tags = ['init']
        read_states = []
        read_group_boundaries = []

        if scan_xs is not None:
            effective_scan_xs = torch.tanh(scan_xs)
        else:
            effective_scan_xs = None

        for pos in range(actual_n_scan):
            # SCAN glimpse (wide sensor)
            glimpse = scan_sensor(image, location)
            h = controller.gru(glimpse, h)
            raw_loc = torch.tanh(controller.location_head(h))
            if effective_scan_xs is not None:
                location = torch.stack([effective_scan_xs[pos].expand(B), raw_loc[:, 1]], dim=1)
            else:
                location = raw_loc
            locations.append(location)
            phase_tags.append('scan')
            if content_head is not None:
                scan_content_logits.append(content_head(h))

            # READ glimpses (flat continuation from scan)
            read_group_boundaries.append(len(read_states))
            for _r in range(n_read_per_group):
                glimpse = read_sensor(image, location)
                h = controller.gru(glimpse, h)
                location = torch.tanh(controller.location_head(h))
                locations.append(location)
                read_states.append(h)
                phase_tags.append('read')

        read_states_t = torch.stack(read_states, dim=1) if read_states else None
        latent = controller.latent_head(h)

        return EncodeResult(
            read_states=read_states_t,
            locations=locations,
            latent=latent,
            scan_content_logits=scan_content_logits,
            actual_n_scan=actual_n_scan,
            read_group_boundaries=read_group_boundaries,
            phase_tags=phase_tags,
        )

    # --- Non-interleaved: Phase 1 (scan) then Phase 2 (read) ---
    phase_tags = ['init']

    # Phase 1: SCAN
    scan_locs_by_step = {}
    if actual_n_scan > 0:
        if scan_xs is not None:
            effective_scan_xs = torch.tanh(scan_xs)
        elif prescribed_x:
            effective_scan_xs = torch.linspace(-0.75, 0.75, actual_n_scan, device=image.device)
        else:
            effective_scan_xs = None
        for t in range(actual_n_scan):
            glimpse = scan_sensor(image, location)
            h = controller.gru(glimpse, h)
            raw_loc = torch.tanh(controller.location_head(h))
            if effective_scan_xs is not None:
                location = torch.stack([effective_scan_xs[t].expand(B), raw_loc[:, 1]], dim=1)
            else:
                location = raw_loc
            locations.append(location)
            scan_locs_by_step[t] = location
            phase_tags.append('scan')
            if content_head is not None:
                scan_content_logits.append(content_head(h))

    # Phase 2: READ
    read_states = []
    read_group_boundaries = None

    if read_group_anchors is not None and n_read_per_group is not None:
        # Grouped read: each group resets x to scan anchor, y starts at center (0)
        read_group_boundaries = []
        for anchor_idx in read_group_anchors:
            read_group_boundaries.append(len(read_states))
            anchor_loc = scan_locs_by_step[anchor_idx]
            # x from scan anchor (hint, not constraint — GRU can move freely after)
            # y starts at 0 (center) — full vertical freedom
            location = torch.stack([anchor_loc[:, 0], torch.zeros(B, device=image.device)], dim=1)
            for _g in range(n_read_per_group):
                glimpse = read_sensor(image, location)
                h = controller.gru(glimpse, h)
                location = torch.tanh(controller.location_head(h))
                locations.append(location)
                read_states.append(h)
                phase_tags.append('read')
    else:
        # Flat read (backward compat)
        for t in range(n_read):
            glimpse = read_sensor(image, location)
            h = controller.gru(glimpse, h)
            location = torch.tanh(controller.location_head(h))
            locations.append(location)
            read_states.append(h)
            phase_tags.append('read')

    read_states_t = torch.stack(read_states, dim=1) if read_states else None
    latent = controller.latent_head(h)

    return EncodeResult(
        read_states=read_states_t,
        locations=locations,
        latent=latent,
        scan_content_logits=scan_content_logits,
        actual_n_scan=actual_n_scan,
        read_group_boundaries=read_group_boundaries,
        phase_tags=phase_tags,
    )


# --- Visual Decoder ---

class VisualDecoder(nn.Module):
    """Generates images from latent/readout vectors using transposed convolutions.

    Unified decoder for all model types — differs only in input_dim and output_shape.
    FC projects to spatial feature map, then two stride-2 deconvs to output_shape.

    Usage:
      VisionModel:  VisualDecoder(input_dim=257, output_shape=(128, 128))  # latent + case label
      BigramModel:  VisualDecoder(input_dim=512, output_shape=(128, 128))  # 2 * latent_dim
      WordModel:    VisualDecoder(input_dim=1024, output_shape=(128, 256)) # 4 * latent_dim
    """
    def __init__(self, input_dim, output_shape=(128, 128)):
        super().__init__()
        self.spatial_h = output_shape[0] // 4  # 32
        self.spatial_w = output_shape[1] // 4  # 32 or 64
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128 * self.spatial_h * self.spatial_w),
            nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 1, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z, condition=None):
        if condition is not None:
            z = torch.cat([z, condition], dim=1)
        x = self.fc(z)
        x = x.view(-1, 128, self.spatial_h, self.spatial_w)
        return self.deconv(x)


# Backward compat aliases
CNNVisualDecoder = VisualDecoder


# --- Vision Model ---

class VisionModel(nn.Module):
    """Full model: encoder (attention) + decoder (reconstruction) + classifiers.

    The latent vector (256-dim) is the bottleneck. Everything flows through it:
      - Encoder produces it from sequential glimpses
      - Decoder reconstructs the full image from it (proves it captured enough info)
      - Letter classifier reads letter identity from it (A-Z, 26 classes)
      - Case classifier reads upper/lower from it (2 classes)
      - Recode: same latent decoded with flipped case -> tests factorization

    Optional scan phase (n_scan_glimpses > 0):
      Prescribed x sweep with wide patches before the free read phase.
      Mirrors the word model's two-phase architecture on cheap 128x128 data.
      scan_sensor and content_head can later transfer directly to WordVisionModel.
    """
    def __init__(self, n_classes=26, latent_dim=256, n_glimpses=10,
                 patch_size=12, n_scales=1,
                 n_scan_glimpses=0, scan_patch_size=(12, 18),
                 read_anchor_scan_indices=None, n_read_per_group=None,
                 learnable_scan_x=False):
        super().__init__()
        self.n_scan_glimpses = n_scan_glimpses
        self.read_anchor_scan_indices = read_anchor_scan_indices
        self.n_read_per_group = n_read_per_group
        self.encoder = VisualAttentionEncoder(
            n_glimpses=n_glimpses, patch_size=patch_size,
            n_scales=n_scales, latent_dim=latent_dim,
        )
        # input_dim = latent + 1 case label; output 128x128
        self.decoder = VisualDecoder(input_dim=latent_dim + 1, output_shape=(128, 128))
        self.letter_classifier = nn.Linear(latent_dim, n_classes)  # 26: A-Z identity
        self.case_classifier = nn.Linear(latent_dim, 2)            # upper/lower

        # Optional scan phase: wide patches + content detection head
        if n_scan_glimpses > 0:
            self.scan_sensor = GlimpseSensor(
                patch_size=scan_patch_size, n_scales=n_scales, latent_dim=latent_dim,
            )
            self.content_head = nn.Linear(latent_dim, 1)

        # Learnable scan x positions
        if learnable_scan_x and n_scan_glimpses > 0:
            if n_scan_glimpses == 1:
                init_xs = torch.zeros(1)
            else:
                init_xs = torch.linspace(-0.75, 0.75, n_scan_glimpses)
            self.scan_xs = nn.Parameter(torch.atanh(init_xs))
        else:
            self.scan_xs = None

    def _encode(self, img):
        """Run encode loop — shared scan/read or legacy encoder."""
        if self.n_scan_glimpses > 0:
            return encode_scan_read(
                img, self.encoder.attention_controller,
                self.scan_sensor, self.encoder.glimpse_sensor,
                n_scan=self.n_scan_glimpses,
                n_read=self.encoder.n_glimpses,
                content_head=self.content_head,
                prescribed_x=(self.scan_xs is None),
                scan_xs=self.scan_xs,
                read_group_anchors=list(self.read_anchor_scan_indices) if self.read_anchor_scan_indices else None,
                n_read_per_group=self.n_read_per_group,
            )
        else:
            enc = encode_scan_read(
                img, self.encoder.attention_controller,
                self.encoder.glimpse_sensor, self.encoder.glimpse_sensor,
                n_scan=0,
                n_read=self.encoder.n_glimpses,
            )
            return enc

    def forward(self, img, case_label):
        """Forward pass with case-conditioned decoding.

        Args:
            img: (B, 1, 128, 128) input image
            case_label: (B, 1) float — 0.0=upper, 1.0=lower
        Returns:
            recon, letter_logits, case_logits, locations, latent, scan_content_logits
        """
        enc = self._encode(img)
        recon = self.decoder(enc.latent, case_label)
        letter_logits = self.letter_classifier(enc.latent)
        case_logits = self.case_classifier(enc.latent)
        return recon, letter_logits, case_logits, enc.locations, enc.latent, enc.scan_content_logits

    def recode(self, img, target_case):
        """Encode image, decode with target case -> capitalize/uncapitalize."""
        enc = self._encode(img)
        recon = self.decoder(enc.latent, target_case)
        return recon, enc.locations


# --- Motor Vision Model ---

class MotorVisionModel(nn.Module):
    """VisionModel + motor trace decoder for Read->Write->Render->Re-Read.

    Identical vision components to VisionModel (encoder, decoder, classifiers,
    optional scan phase). Adds MotorTraceDecoder for the motor pathway.
    Supports transfer from pretrained VisionModel (e.g. v5-scan).

    The motor pathway is called separately via motor_forward() so the main
    graph can be freed before allocating the motor+re-read graph (VRAM-safe).
    """
    def __init__(self, n_classes=26, latent_dim=256, n_glimpses=10,
                 patch_size=12, n_scales=1,
                 n_scan_glimpses=0, scan_patch_size=(12, 18),
                 n_trajectory_points=32, render_sigma=1.5,
                 read_anchor_scan_indices=None, n_read_per_group=None,
                 learnable_scan_x=False):
        super().__init__()
        self.n_scan_glimpses = n_scan_glimpses
        self.read_anchor_scan_indices = read_anchor_scan_indices
        self.n_read_per_group = n_read_per_group
        self.encoder = VisualAttentionEncoder(
            n_glimpses=n_glimpses, patch_size=patch_size,
            n_scales=n_scales, latent_dim=latent_dim,
        )
        self.decoder = VisualDecoder(input_dim=latent_dim + 1, output_shape=(128, 128))
        self.letter_classifier = nn.Linear(latent_dim, n_classes)
        self.case_classifier = nn.Linear(latent_dim, 2)

        if n_scan_glimpses > 0:
            self.scan_sensor = GlimpseSensor(
                patch_size=scan_patch_size, n_scales=n_scales, latent_dim=latent_dim,
            )
            self.content_head = nn.Linear(latent_dim, 1)

        # Learnable scan x positions
        if learnable_scan_x and n_scan_glimpses > 0:
            if n_scan_glimpses == 1:
                init_xs = torch.zeros(1)
            else:
                init_xs = torch.linspace(-0.75, 0.75, n_scan_glimpses)
            self.scan_xs = nn.Parameter(torch.atanh(init_xs))
        else:
            self.scan_xs = None

        # Motor pathway
        from fbrl.motor import MotorTraceDecoder, soft_render
        self.motor_decoder = MotorTraceDecoder(latent_dim, latent_dim, n_trajectory_points)
        self._render_sigma = render_sigma
        self._soft_render = soft_render

    def _encode(self, img):
        """Run encode loop -- shared scan/read or legacy encoder."""
        if self.n_scan_glimpses > 0:
            return encode_scan_read(
                img, self.encoder.attention_controller,
                self.scan_sensor, self.encoder.glimpse_sensor,
                n_scan=self.n_scan_glimpses,
                n_read=self.encoder.n_glimpses,
                content_head=self.content_head,
                prescribed_x=(self.scan_xs is None),
                scan_xs=self.scan_xs,
                read_group_anchors=list(self.read_anchor_scan_indices) if self.read_anchor_scan_indices else None,
                n_read_per_group=self.n_read_per_group,
            )
        else:
            return encode_scan_read(
                img, self.encoder.attention_controller,
                self.encoder.glimpse_sensor, self.encoder.glimpse_sensor,
                n_scan=0,
                n_read=self.encoder.n_glimpses,
            )

    def forward(self, img, case_label):
        """Standard forward (same signature as VisionModel).

        Motor path is called separately via motor_forward() for VRAM management.
        """
        enc = self._encode(img)
        recon = self.decoder(enc.latent, case_label)
        letter_logits = self.letter_classifier(enc.latent)
        case_logits = self.case_classifier(enc.latent)
        return recon, letter_logits, case_logits, enc.locations, enc.latent, enc.scan_content_logits

    def motor_forward(self, latent):
        """Deferred motor path: latent -> trajectory -> rendered image.

        Call this AFTER freeing the main forward graph.
        """
        trajectory = self.motor_decoder(latent)
        rendered = self._soft_render(trajectory, sigma=self._render_sigma)
        return trajectory, rendered

    def recode(self, img, target_case):
        """Encode image, decode with target case -> capitalize/uncapitalize."""
        enc = self._encode(img)
        recon = self.decoder(enc.latent, target_case)
        return recon, enc.locations


# --- Bigram (Letter-Pair) Reading ---

BigramDecoder = VisualDecoder  # backward compat alias


class CrossAttentionReadout(nn.Module):
    """Cross-attention readout: position tokens query the visual glimpse history.

    Instead of a sequential GRU that processes positions one after another
    (where pos2 depends on pos1's output), each position token independently
    attends to all glimpse hidden states in parallel.

    The left token naturally learns to attend to glimpses that fixated on the
    left letter, and the right token to right-letter glimpses. Both positions
    get equal access to the full visual memory — no asymmetry.
    """
    def __init__(self, latent_dim=256, n_positions=2):
        super().__init__()
        self.scale = latent_dim ** -0.5

        # Learned position query tokens — initialized with left/right bias
        self.query_tokens = nn.Parameter(torch.zeros(n_positions, latent_dim))
        with torch.no_grad():
            self.query_tokens[0, 0] = -1.0   # left bias
            self.query_tokens[1, 0] = +1.0   # right bias

        # Projections for scaled dot-product attention
        self.query_proj = nn.Linear(latent_dim, latent_dim)
        self.key_proj = nn.Linear(latent_dim, latent_dim)
        self.value_proj = nn.Linear(latent_dim, latent_dim)
        self.out_proj = nn.Linear(latent_dim, latent_dim)

    def forward(self, glimpse_states, group_boundaries=None):
        """Cross-attend position tokens over glimpse history.

        Args:
            glimpse_states: (B, T, latent_dim) hidden states from all visual glimpses
            group_boundaries: list of start indices for per-group attention.
                When provided, query token i attends only to its group's states.
                When None: global attention over all states (original behavior).
        Returns:
            readout_states: (B, n_positions, latent_dim) per-position outputs
        """
        B, T, D = glimpse_states.shape
        n_pos = self.query_tokens.shape[0]

        Q = self.query_proj(
            self.query_tokens.unsqueeze(0).expand(B, -1, -1)
        )                                        # (B, n_positions, D)
        K = self.key_proj(glimpse_states)         # (B, T, D)
        V = self.value_proj(glimpse_states)       # (B, T, D)

        # Scaled dot-product attention
        attn = torch.bmm(Q, K.transpose(1, 2)) * self.scale  # (B, n_pos, T)

        if group_boundaries is not None:
            # Per-group masking: query i only attends to its group's keys
            mask = torch.full((n_pos, T), float('-inf'), device=glimpse_states.device)
            for i, start in enumerate(group_boundaries):
                end = group_boundaries[i + 1] if i + 1 < len(group_boundaries) else T
                mask[i, start:end] = 0.0
            attn = attn + mask.unsqueeze(0)  # broadcast over batch

        attn = F.softmax(attn, dim=-1)

        out = torch.bmm(attn, V)      # (B, n_positions, D)
        return self.out_proj(out)


class BigramVisionModel(nn.Module):
    """Full bigram model with two-phase scan/read attention.

    Phase 1 — SCAN (wide rectangular patches, coarse spatial map):
      scan_sensor extracts 12h x 18w patches. GRU builds a spatial map of
      where letters are. Loss: content guide (full image) + edge loss.

    Phase 2 — READ (focused square patches, precise letter identification):
      read_sensor extracts 12x12 patches. GRU continues from scan's final
      hidden state, carrying spatial knowledge. Loss: temporal scaffold.

    CrossAttentionReadout attends to READ states only → classify + reconstruct.

    Same GRU + location_head for both phases — different visual inputs
    naturally produce different saccade patterns.
    """
    def __init__(self, n_classes=26, latent_dim=256,
                 n_scan_glimpses=5, n_read_glimpses=6,
                 scan_patch_size=(12, 18), read_patch_size=12,
                 n_scales=1, n_positions=2):
        super().__init__()
        self.n_positions = n_positions
        self.latent_dim = latent_dim
        self.n_scan_glimpses = n_scan_glimpses
        self.n_read_glimpses = n_read_glimpses

        # Two sensors: wide scan, focused read
        self.scan_sensor = GlimpseSensor(
            patch_size=scan_patch_size, n_scales=n_scales, latent_dim=latent_dim,
        )
        self.read_sensor = GlimpseSensor(
            patch_size=read_patch_size, n_scales=n_scales, latent_dim=latent_dim,
        )

        # Shared GRU controller for both phases
        self.controller = AttentionController(
            glimpse_dim=latent_dim, hidden_dim=latent_dim, latent_dim=latent_dim,
        )

        self.decoder = VisualDecoder(
            input_dim=latent_dim * n_positions, output_shape=(128, 128),
        )

        # Cross-attention readout — position tokens query READ states only
        self.readout = CrossAttentionReadout(
            latent_dim=latent_dim, n_positions=n_positions,
        )

        # Per-position letter classifiers (26-class each: a-z)
        self.classifiers = nn.ModuleList([
            nn.Linear(latent_dim, n_classes) for _ in range(n_positions)
        ])

    def forward(self, img):
        """Forward pass with two-phase scan/read attention.

        Args:
            img: (B, 1, 128, 128) bigram input image
        Returns:
            recon: (B, 1, 128, 128) reconstructed image
            logits_list: [logits_pos1, logits_pos2] each (B, 26)
            locations: list of (B, 2) fixation coords (all phases)
            readout_states: (B, n_positions, latent_dim) per-position hidden states
        """
        B = img.shape[0]
        enc = encode_scan_read(
            img, self.controller, self.scan_sensor, self.read_sensor,
            n_scan=self.n_scan_glimpses, n_read=self.n_read_glimpses,
        )

        readout_states = self.readout(enc.read_states)
        logits_list = [self.classifiers[i](readout_states[:, i])
                       for i in range(self.n_positions)]
        recon = self.decoder(readout_states.view(B, -1))

        return recon, logits_list, enc.locations, readout_states


# --- Word (4-Letter) Reading ---

WordDecoder = VisualDecoder  # backward compat alias


class WordVisionModel(nn.Module):
    """Full word model with prescribed x-scan + free read attention.

    Phase 1 — SCAN (wide patches, prescribed x sweep):
      x is PRESCRIBED: linear sweep [-0.75, +0.75] across 8 positions.
      y is LEARNED: GRU -> location_head -> tanh -> y component only.
      Content head: nn.Linear(hidden, 1) on each scan h -> content detection.
      Purpose: force L->R scanning, learn vertical positioning, detect content.

    Phase 2 — READ (focused patches, fully free):
      Both x and y fully learned. h carries forward from scan.
      Loss: temporal scaffold with 4 horizontal stripes (one per letter).

    CrossAttentionReadout with 4 position tokens -> 4 classifiers + reconstruct.
    """
    def __init__(self, n_classes=26, latent_dim=256,
                 n_scan_glimpses=8, n_read_glimpses=12,
                 scan_patch_size=(12, 18), read_patch_size=12,
                 n_scales=1, n_positions=4,
                 read_anchor_scan_indices=None, n_read_per_group=None,
                 interleaved=False):
        super().__init__()
        self.n_positions = n_positions
        self.latent_dim = latent_dim
        self.n_scan_glimpses = n_scan_glimpses
        self.n_read_glimpses = n_read_glimpses
        self.read_anchor_scan_indices = read_anchor_scan_indices
        self.n_read_per_group = n_read_per_group
        self.interleaved = interleaved

        # Two sensors: wide scan, focused read
        self.scan_sensor = GlimpseSensor(
            patch_size=scan_patch_size, n_scales=n_scales, latent_dim=latent_dim,
        )
        self.read_sensor = GlimpseSensor(
            patch_size=read_patch_size, n_scales=n_scales, latent_dim=latent_dim,
        )

        # Shared GRU controller for both phases
        self.controller = AttentionController(
            glimpse_dim=latent_dim, hidden_dim=latent_dim, latent_dim=latent_dim,
        )

        # Content detection head: predicts whether scan location has letter content
        self.content_head = nn.Linear(latent_dim, 1)

        # Learnable scan x positions
        if interleaved:
            # Interleaved: one scan per letter center, tighter init to land on content
            init_xs = torch.linspace(-0.5, 0.5, n_positions)
            self.scan_xs = nn.Parameter(torch.atanh(init_xs))
        elif read_anchor_scan_indices is not None:
            # Grouped read: boundary scans at ±0.99, inner at letter centers
            letter_centers = torch.linspace(-0.75, 0.75, n_positions)
            init_xs = torch.cat([
                torch.tensor([-0.99]),
                letter_centers,
                torch.tensor([0.99]),
            ])
            self.scan_xs = nn.Parameter(torch.atanh(init_xs))
        else:
            self.scan_xs = None

        self.decoder = VisualDecoder(
            input_dim=latent_dim * n_positions, output_shape=(128, 256),
        )

        # Cross-attention readout — position tokens query READ states only
        self.readout = CrossAttentionReadout(
            latent_dim=latent_dim, n_positions=n_positions,
        )
        # Override query token initialization for 4-position reading
        with torch.no_grad():
            positions = torch.linspace(-0.75, 0.75, n_positions)
            for i in range(n_positions):
                self.readout.query_tokens.data[i, 0] = positions[i]

        # Per-position letter classifiers (26-class each: a-z)
        self.classifiers = nn.ModuleList([
            nn.Linear(latent_dim, n_classes) for _ in range(n_positions)
        ])

    def forward(self, img):
        """Forward pass with prescribed x-scan + free read (or interleaved).

        Args:
            img: (B, 1, 128, 256) word input image
        Returns:
            recon: (B, 1, 128, 256) reconstructed image
            logits_list: [logits_pos1..pos4] each (B, 26)
            locations: list of (B, 2) fixation coords (all phases)
            readout_states: (B, n_positions, latent_dim)
            scan_content_logits: list of (B, 1) content predictions per scan step
            read_group_boundaries: list of int or None
            phase_tags: list of str ('init', 'scan', 'read') per location
        """
        B = img.shape[0]
        if self.interleaved:
            enc = encode_scan_read(
                img, self.controller, self.scan_sensor, self.read_sensor,
                n_scan=self.n_scan_glimpses, n_read=self.n_read_glimpses,
                content_head=self.content_head,
                scan_xs=self.scan_xs,
                n_read_per_group=self.n_read_per_group,
                interleaved=True,
            )
        else:
            enc = encode_scan_read(
                img, self.controller, self.scan_sensor, self.read_sensor,
                n_scan=self.n_scan_glimpses, n_read=self.n_read_glimpses,
                content_head=self.content_head, prescribed_x=True,
                dynamic_width=img.shape[3] if self.scan_xs is None else None,
                scan_xs=self.scan_xs,
                read_group_anchors=list(self.read_anchor_scan_indices) if self.read_anchor_scan_indices else None,
                n_read_per_group=self.n_read_per_group,
            )

        readout_states = self.readout(enc.read_states,
                                       group_boundaries=enc.read_group_boundaries)
        logits_list = [self.classifiers[i](readout_states[:, i])
                       for i in range(self.n_positions)]
        recon = self.decoder(readout_states.view(B, -1))

        return recon, logits_list, enc.locations, readout_states, enc.scan_content_logits, enc.read_group_boundaries, enc.phase_tags
