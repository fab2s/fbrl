import torch
import torch.nn as nn
import torch.nn.functional as F


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


# --- CNN Visual Decoder ---

class CNNVisualDecoder(nn.Module):
    """Generates 128x128 images from latent vectors using transposed convolutions.

    Optionally conditioned on a case label (condition_dim floats concatenated to latent).
    FC projects to 32x32, then two stride-2 deconvs: 32->64->128.
    """
    def __init__(self, latent_dim=256, condition_dim=0):
        super().__init__()
        # FC expands the latent vector into a spatial feature map (128 channels x 32x32).
        # This is the most parameter-heavy layer (~33.7M params at 128x128) — it gives
        # the decoder enough capacity to render fine details, but also means it can
        # sometimes reconstruct without good attention (hence the need for guide_weight).
        self.fc = nn.Sequential(
            nn.Linear(latent_dim + condition_dim, 128 * 32 * 32),
            nn.ReLU(),
        )
        # Transposed convolutions upsample 32x32 -> 64x64 -> 128x128
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),  # 32->64
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 5, stride=2, padding=2, output_padding=1),  # 64->128
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 1, 3, padding=1),  # final 1-channel output
            nn.Sigmoid(),                      # clamp to [0, 1] (pixel intensity)
        )

    def forward(self, z, condition=None):
        # Concatenate condition (e.g., case label) to latent before decoding
        if condition is not None:
            z = torch.cat([z, condition], dim=1)
        x = self.fc(z)
        x = x.view(-1, 128, 32, 32)  # reshape flat vector into spatial feature map
        return self.deconv(x)


# --- Vision Model ---

class VisionModel(nn.Module):
    """Full model: encoder (attention) + decoder (reconstruction) + classifiers.

    The latent vector (256-dim) is the bottleneck. Everything flows through it:
      - Encoder produces it from sequential glimpses
      - Decoder reconstructs the full image from it (proves it captured enough info)
      - Letter classifier reads letter identity from it (A-Z, 26 classes)
      - Case classifier reads upper/lower from it (2 classes)
      - Recode: same latent decoded with flipped case -> tests factorization
    """
    def __init__(self, n_classes=26, latent_dim=256, n_glimpses=10,
                 patch_size=12, n_scales=1):
        super().__init__()
        self.encoder = VisualAttentionEncoder(
            n_glimpses=n_glimpses, patch_size=patch_size,
            n_scales=n_scales, latent_dim=latent_dim,
        )
        # condition_dim=1: the decoder receives the case label (0.0 or 1.0)
        # concatenated to the latent, so it knows which case to render
        self.decoder = CNNVisualDecoder(latent_dim=latent_dim, condition_dim=1)
        self.letter_classifier = nn.Linear(latent_dim, n_classes)  # 26: A-Z identity
        self.case_classifier = nn.Linear(latent_dim, 2)            # upper/lower

    def forward(self, img, case_label):
        """Forward pass with case-conditioned decoding.

        Args:
            img: (B, 1, 128, 128) input image
            case_label: (B, 1) float — 0.0=upper, 1.0=lower
        Returns:
            recon, letter_logits, case_logits, locations, latent
        """
        latent, locations = self.encoder(img)
        recon = self.decoder(latent, case_label)
        letter_logits = self.letter_classifier(latent)
        case_logits = self.case_classifier(latent)
        return recon, letter_logits, case_logits, locations, latent

    def recode(self, img, target_case):
        """Encode image, decode with target case -> capitalize/uncapitalize."""
        latent, locations = self.encoder(img)
        recon = self.decoder(latent, target_case)
        return recon, locations


# --- Bigram (Letter-Pair) Reading ---

class BigramDecoder(nn.Module):
    """Generates 128x128 images from concatenated readout states.

    Input: concatenated readout states (2 * latent_dim = 512 by default).
    FC projects to 128 channels x 32x32, then two stride-2 deconvs.
    Output: (B, 1, 128, 128).
    """
    def __init__(self, latent_dim=256, n_positions=2):
        super().__init__()
        input_dim = latent_dim * n_positions  # 512 by default
        # FC expands to spatial feature map: 128 channels x 32x32
        # Two stride-2 deconvs: 32x32 -> 64x64 -> 128x128
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128 * 32 * 32),
            nn.ReLU(),
        )
        # Transposed convolutions: 32x32 -> 64x64 -> 128x128
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),  # 32->64
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 5, stride=2, padding=2, output_padding=1),  # 64->128
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 1, 3, padding=1),  # final 1-channel output
            nn.Sigmoid(),
        )

    def forward(self, readout_states):
        """Decode concatenated readout states into a 128x128 image.

        Args:
            readout_states: (B, n_positions * latent_dim) — concatenated readout hidden states
        Returns:
            (B, 1, 128, 128) reconstructed bigram image
        """
        x = self.fc(readout_states)
        x = x.view(-1, 128, 32, 32)  # (B, 128, 32, 32)
        return self.deconv(x)


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

    def forward(self, glimpse_states):
        """Cross-attend position tokens over glimpse history.

        Args:
            glimpse_states: (B, T, latent_dim) hidden states from all visual glimpses
        Returns:
            readout_states: (B, n_positions, latent_dim) per-position outputs
        """
        B = glimpse_states.shape[0]

        Q = self.query_proj(
            self.query_tokens.unsqueeze(0).expand(B, -1, -1)
        )                                        # (B, n_positions, D)
        K = self.key_proj(glimpse_states)         # (B, T, D)
        V = self.value_proj(glimpse_states)       # (B, T, D)

        # Scaled dot-product attention
        attn = torch.bmm(Q, K.transpose(1, 2)) * self.scale  # (B, n_pos, T)
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

        self.decoder = BigramDecoder(latent_dim=latent_dim, n_positions=n_positions)

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
        h = self.controller.h0.expand(B, -1).contiguous()
        location = torch.zeros(B, 2, device=img.device)  # center start
        all_locations = [location]

        # --- Phase 1: SCAN (wide patches, build spatial map) ---
        for t in range(self.n_scan_glimpses):
            glimpse = self.scan_sensor(img, location)
            h = self.controller.gru(glimpse, h)
            location = torch.tanh(self.controller.location_head(h))
            all_locations.append(location)

        # --- Phase 2: READ (focused patches, precise identification) ---
        # h carries forward from scan — spatial knowledge persists
        read_states = []
        for t in range(self.n_read_glimpses):
            glimpse = self.read_sensor(img, location)
            h = self.controller.gru(glimpse, h)
            location = torch.tanh(self.controller.location_head(h))
            all_locations.append(location)
            read_states.append(h)

        # Stack READ states for cross-attention (B, n_read, latent_dim)
        read_states = torch.stack(read_states, dim=1)

        # --- Cross-attention readout (READ states only) ---
        readout_states = self.readout(read_states)  # (B, n_positions, latent_dim)

        # --- Classify each position ---
        logits_list = []
        for i in range(self.n_positions):
            logits_list.append(self.classifiers[i](readout_states[:, i]))

        # --- Reconstruct from concatenated readout states ---
        concat = readout_states.view(B, -1)  # (B, n_positions * latent_dim)
        recon = self.decoder(concat)

        return recon, logits_list, all_locations, readout_states
