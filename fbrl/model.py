import torch
import torch.nn as nn
import torch.nn.functional as F


# --- Glimpse Sensor ---

class GlimpseSensor(nn.Module):
    """Extracts multi-resolution patches from the RAW IMAGE at a given location.

    Strict foveal-only field of view: each glimpse sees ONLY patch_size x patch_size
    raw pixels. No peripheral context. Default patch_size=12, n_scales=1:
      Scale 1: 12x12 pixels (0.9% of 128x128 image)
    With 10 glimpses: 8.8% max coverage.
    """
    def __init__(self, patch_size=12, n_scales=1, latent_dim=256):
        super().__init__()
        self.patch_size = patch_size
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
        The grid covers patch_size x patch_size pixels, scaled by `scale`.
        Adding `location` shifts the grid to the fixation point.
        """
        B = location.shape[0]
        # Convert patch extent from pixels to normalized [-1, 1] coords
        delta_h = scale * self.patch_size / H
        delta_w = scale * self.patch_size / W

        grid_y = torch.linspace(-delta_h, delta_h, self.patch_size, device=location.device)
        grid_x = torch.linspace(-delta_w, delta_w, self.patch_size, device=location.device)
        grid_y, grid_x = torch.meshgrid(grid_y, grid_x, indexing='ij')

        # grid shape: (1, patch_size, patch_size, 2) — a centered patch
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
