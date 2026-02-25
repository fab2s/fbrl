import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import json
import time
from PIL import Image, ImageDraw, ImageFont


# --- Data Generation ---

def generate_dataset(letters, output_dir, noise_level=0.01, num_variants=1):
    os.makedirs(output_dir, exist_ok=True)
    metadata = {}
    for letter in letters:
        # Render clean image
        img = Image.new('L', (128, 128), color=0)
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default(size=60)
        bbox = draw.textbbox((0, 0), letter, font=font)
        x = (128 - bbox[2] - bbox[0]) / 2
        y = (128 - bbox[3] - bbox[1]) / 2
        draw.text((x, y), letter, fill=255, font=font)

        clean_path = os.path.join(output_dir, f'clean_{letter}.png')
        img.save(clean_path)

        img_array = np.array(img) / 255.0

        for v in range(num_variants):
            noisy = img_array + np.random.normal(0, noise_level, img_array.shape)
            noisy = np.clip(noisy, 0, 1)
            noisy_img = Image.fromarray((noisy * 255).astype(np.uint8))
            img_path = os.path.join(output_dir, f'img_{letter}_{v}.png')
            noisy_img.save(img_path)
            key = f'{letter}_{v}'
            metadata[key] = {
                'image': img_path,
                'clean': clean_path,
                'letter': letter.upper(),
                'case': 'upper' if letter.isupper() else 'lower',
            }

    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f)
    print(f"Dataset generated in {output_dir}: {len(metadata)} samples "
          f"({num_variants} variants per letter)")


def generate_test(letters, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    metadata = {}
    for letter in letters:
        img = Image.new('L', (128, 128), color=0)
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default(size=60)
        bbox = draw.textbbox((0, 0), letter, font=font)
        x = (128 - bbox[2] - bbox[0]) / 2
        y = (128 - bbox[3] - bbox[1]) / 2
        draw.text((x, y), letter, fill=255, font=font)

        img_path = os.path.join(output_dir, f'img_{letter}.png')
        img.save(img_path)
        metadata[letter] = {
            'image': img_path,
            'letter': letter.upper(),
            'case': 'upper' if letter.isupper() else 'lower',
        }

    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f)
    print(f"Test data generated in {output_dir}: {len(metadata)} samples")


# --- Dataset ---

class LetterDataset(Dataset):
    def __init__(self, data_dir):
        with open(os.path.join(data_dir, 'metadata.json'), 'r') as f:
            metadata = json.load(f)

        print(f"Loading {len(metadata)} samples into memory...", end=' ', flush=True)
        t0 = time.time()
        self.images = []
        self.clean_images = []
        self.letters = []
        self.cases = []

        # Cache clean images (one per letter+case, shared across variants)
        clean_cache = {}

        for key in metadata:
            entry = metadata[key]
            img = Image.open(entry['image']).convert('L')
            img_tensor = torch.tensor(
                np.array(img) / 255.0, dtype=torch.float32,
            ).unsqueeze(0)  # (1, H, W)
            self.images.append(img_tensor)
            self.letters.append(entry['letter'])
            self.cases.append(entry.get('case', 'upper'))

            # Load clean reference (if available, else use the image itself)
            clean_path = entry.get('clean')
            if clean_path:
                if clean_path not in clean_cache:
                    cimg = Image.open(clean_path).convert('L')
                    clean_cache[clean_path] = torch.tensor(
                        np.array(cimg) / 255.0, dtype=torch.float32,
                    ).unsqueeze(0)
                self.clean_images.append(clean_cache[clean_path])
            else:
                self.clean_images.append(img_tensor)

        # Build partner index: for each sample, find clean image of same letter, opposite case
        clean_by_letter_case = {}
        for i, (letter, case) in enumerate(zip(self.letters, self.cases)):
            if (letter, case) not in clean_by_letter_case:
                clean_by_letter_case[(letter, case)] = self.clean_images[i]

        self.partner_clean = []
        self.has_partners = True
        for letter, case in zip(self.letters, self.cases):
            opposite = 'lower' if case == 'upper' else 'upper'
            partner = clean_by_letter_case.get((letter, opposite))
            if partner is None:
                self.has_partners = False
                self.partner_clean.append(torch.zeros_like(self.images[0]))
            else:
                self.partner_clean.append(partner)

        print(f"done ({time.time() - t0:.1f}s)")

    def __len__(self):
        return len(self.letters)

    def __getitem__(self, idx):
        return (self.images[idx], self.clean_images[idx],
                self.letters[idx], self.cases[idx],
                self.partner_clean[idx])


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
        self.scales = [2**i for i in range(n_scales)]

        self.patch_cnn = nn.Sequential(
            nn.Conv2d(n_scales, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.glimpse_fc = nn.Sequential(
            nn.Linear(128, latent_dim),
            nn.ReLU(),
        )
        self.location_fc = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(),
        )
        self.combine_fc = nn.Sequential(
            nn.Linear(latent_dim + 128, latent_dim),
            nn.ReLU(),
        )

    def forward(self, image, location):
        B, C, H, W = image.shape
        patches = []
        for scale in self.scales:
            grid = self._make_grid(location, scale, H, W)
            patch = F.grid_sample(image, grid, align_corners=True, padding_mode='zeros')
            patches.append(patch)

        combined = torch.cat(patches, dim=1)
        feat = self.patch_cnn(combined)

        glimpse_feat = self.glimpse_fc(feat)
        loc_feat = self.location_fc(location)
        return self.combine_fc(torch.cat([glimpse_feat, loc_feat], dim=1))

    def _make_grid(self, location, scale, H, W):
        B = location.shape[0]
        delta_h = scale * self.patch_size / H
        delta_w = scale * self.patch_size / W

        grid_y = torch.linspace(-delta_h, delta_h, self.patch_size, device=location.device)
        grid_x = torch.linspace(-delta_w, delta_w, self.patch_size, device=location.device)
        grid_y, grid_x = torch.meshgrid(grid_y, grid_x, indexing='ij')

        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)
        loc = location.view(B, 1, 1, 2)
        return grid + loc


# --- Attention Controller ---

class AttentionController(nn.Module):
    """GRU that decides where to look next."""
    def __init__(self, glimpse_dim=256, hidden_dim=256, latent_dim=256):
        super().__init__()
        self.gru = nn.GRUCell(glimpse_dim, hidden_dim)
        self.location_head = nn.Linear(hidden_dim, 2)
        self.latent_head = nn.Linear(hidden_dim, latent_dim)
        self.h0 = nn.Parameter(torch.zeros(1, hidden_dim))

    def forward(self, image, glimpse_sensor, n_glimpses):
        B = image.shape[0]
        h = self.h0.expand(B, -1).contiguous()
        location = torch.zeros(B, 2, device=image.device)

        locations = [location]
        for t in range(n_glimpses):
            glimpse = glimpse_sensor(image, location)
            h = self.gru(glimpse, h)
            location = torch.tanh(self.location_head(h))
            locations.append(location)

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
        self.fc = nn.Sequential(
            nn.Linear(latent_dim + condition_dim, 128 * 32 * 32),
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
        x = x.view(-1, 128, 32, 32)
        return self.deconv(x)


# --- Vision Model ---

class VisionModel(nn.Module):
    def __init__(self, n_classes=26, latent_dim=256, n_glimpses=10,
                 patch_size=12, n_scales=1):
        super().__init__()
        self.encoder = VisualAttentionEncoder(
            n_glimpses=n_glimpses, patch_size=patch_size,
            n_scales=n_scales, latent_dim=latent_dim,
        )
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


# --- Attention Visualization ---

def visualize_attention(img_tensor, locations, save_path):
    """Overlay fixation points and saccade arrows on image."""
    img = img_tensor.squeeze(0).cpu().detach().numpy()
    H, W = img.shape

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.imshow(img, cmap='gray', vmin=0, vmax=1)

    colors = plt.cm.hot(np.linspace(0.2, 0.9, len(locations)))

    for i, loc in enumerate(locations):
        loc_np = loc[0].cpu().detach().numpy()
        px = (loc_np[0] + 1) / 2 * W
        py = (loc_np[1] + 1) / 2 * H

        ax.plot(px, py, 'o', color=colors[i], markersize=8,
                markeredgecolor='white', markeredgewidth=0.5)
        ax.annotate(str(i), (px, py), color='white', fontsize=6,
                    ha='center', va='center')

        if i > 0:
            prev_np = locations[i - 1][0].cpu().detach().numpy()
            prev_px = (prev_np[0] + 1) / 2 * W
            prev_py = (prev_np[1] + 1) / 2 * H
            ax.annotate('', xy=(px, py), xytext=(prev_px, prev_py),
                        arrowprops=dict(arrowstyle='->', color=colors[i], lw=1.5))

    ax.set_title(f'{len(locations)} fixations')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# --- Attention Guide Loss ---

def attention_content_loss(image, locations, blur_sigma_ratio=0.16):
    """Guide fixations toward letter strokes using a blurred guidance field.

    blur_sigma is computed as a fraction of min(H, W), so the guidance field
    auto-scales to any image/crop size. Default ratio 0.16 matches the proven
    recipe (15px at 96x96, 20px at 128x128).
    """
    B, C, H, W = image.shape
    blur_sigma = blur_sigma_ratio * min(H, W)

    k = int(4 * blur_sigma) | 1
    x = torch.arange(k, device=image.device, dtype=image.dtype) - k // 2
    gauss = torch.exp(-x ** 2 / (2 * blur_sigma ** 2))
    gauss = gauss / gauss.sum()
    guide = F.conv2d(image, gauss.view(1, 1, k, 1), padding=(k // 2, 0))
    guide = F.conv2d(guide, gauss.view(1, 1, 1, k), padding=(0, k // 2))

    total = 0
    for loc in locations[1:]:
        grid = loc.view(B, 1, 1, 2)
        sampled = F.grid_sample(guide, grid, align_corners=True, padding_mode='zeros')
        total = total + sampled.mean()
    return -total / len(locations[1:])


# --- Fixation Diversity Loss ---

def fixation_diversity_loss(locations, sigma=0.1):
    """Pairwise repulsion between fixation points.

    Gaussian RBF kernel: fixations closer than ~sigma (in [-1,1] coords)
    repel each other strongly. At sigma=0.1, that's ~10% of image width.
    """
    locs = torch.stack(locations[1:], dim=1)  # (B, T, 2)
    B, T, _ = locs.shape
    diff = locs.unsqueeze(2) - locs.unsqueeze(1)  # (B, T, T, 2)
    dist_sq = (diff ** 2).sum(-1)  # (B, T, T)
    repulsion = torch.exp(-dist_sq / (2 * sigma ** 2))
    mask = 1 - torch.eye(T, device=locs.device)
    return (repulsion * mask).mean()


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
        grid = loc.view(B, 1, 1, 2)
        sampled = F.grid_sample(image, grid, align_corners=True, padding_mode='zeros')
        intensity = sampled.mean(dim=0).squeeze()  # average over batch
        total_intensity += intensity.item()
        hits += (sampled.squeeze() > threshold).float().sum().item()
        n += B
    return hits / max(n, 1), total_intensity / len(locations[1:])


# --- Training ---

def _resolve_device(device_str):
    if device_str == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device_str)


def train_model(data_dir, epochs=200, resume=None, save_dir='models',
                checkpoint_interval=10, n_glimpses=10, patch_size=12,
                n_scales=1, device='auto',
                diversity_weight=1.0, diversity_sigma=0.1,
                recode_weight=1.0, guide_weight=4.0, blur_sigma_ratio=0.16):
    device = _resolve_device(device)
    print(f"Training on: {device}")
    print(f"Attention: guide_weight={guide_weight}  blur_sigma_ratio={blur_sigma_ratio}  "
          f"diversity_weight={diversity_weight}  diversity_sigma={diversity_sigma}  "
          f"recode_weight={recode_weight}")

    os.makedirs(save_dir, exist_ok=True)
    dataset = LetterDataset(data_dir)
    use_cuda = device.type == 'cuda'
    dataloader = DataLoader(dataset, batch_size=26, shuffle=True, pin_memory=use_cuda)

    if dataset.has_partners:
        print(f"Partner images found — recode loss enabled (weight={recode_weight})")
    else:
        print("No partner images — recode loss disabled")

    model = VisionModel(
        n_glimpses=n_glimpses, patch_size=patch_size, n_scales=n_scales,
    ).to(device)

    start_epoch = 0
    losses_recon = []
    losses_letter_cls = []
    losses_case_cls = []
    losses_attn = []
    losses_div = []
    losses_recode = []
    hist_hit_rate = []
    hist_hit_intensity = []

    if resume:
        checkpoint = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model'], strict=False)
        start_epoch = checkpoint['epoch'] + 1
        if 'losses' in checkpoint:
            h = checkpoint['losses']
            losses_recon = h.get('recon', [])
            losses_letter_cls = h.get('letter_cls', h.get('cls', []))
            losses_case_cls = h.get('case_cls', [])
            losses_attn = h.get('attn', [])
            losses_div = h.get('div', [])
            losses_recode = h.get('recode', [])
            hist_hit_rate = h.get('hit_rate', [])
            hist_hit_intensity = h.get('hit_intensity', [])
        print(f"Resumed from epoch {start_epoch} ({len(losses_letter_cls)} prior epochs of history)")

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    end_epoch = start_epoch + epochs
    train_start = time.time()

    for epoch in range(start_epoch, end_epoch):
        epoch_start = time.time()
        total_loss_recon = 0
        total_loss_letter_cls = 0
        total_loss_case_cls = 0
        total_loss_attn = 0
        total_loss_div = 0
        total_loss_recode = 0
        total_hit_rate = 0
        total_hit_intensity = 0

        for img, clean, letters, cases, partner_clean in dataloader:
            img = img.to(device)
            clean = clean.to(device)
            partner_clean = partner_clean.to(device)

            letter_idx = torch.tensor(
                [ord(l) - ord('A') for l in letters], device=device,
            )
            case_idx = torch.tensor(
                [0 if c == 'upper' else 1 for c in cases], device=device,
            )
            case_float = case_idx.float().unsqueeze(1)  # (B, 1)

            recon, letter_logits, case_logits, locations, latent = model(img, case_float)

            recon_loss = criterion(recon, img)
            letter_cls_loss = F.cross_entropy(letter_logits, letter_idx)
            case_cls_loss = F.cross_entropy(case_logits, case_idx)
            attn_loss = attention_content_loss(clean, locations, blur_sigma_ratio=blur_sigma_ratio)
            div_loss = fixation_diversity_loss(locations, sigma=diversity_sigma)

            total_loss = (recon_loss + letter_cls_loss + case_cls_loss
                          + guide_weight * attn_loss
                          + diversity_weight * div_loss)

            # Recode loss: flip case, decode same latent, compare to partner clean image
            if dataset.has_partners and recode_weight > 0:
                flipped_case = 1.0 - case_float
                recode_img = model.decoder(latent, flipped_case)
                recode_loss = criterion(recode_img, partner_clean)
                total_loss = total_loss + recode_weight * recode_loss
                total_loss_recode += recode_loss.item()

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss_recon += recon_loss.item()
            total_loss_letter_cls += letter_cls_loss.item()
            total_loss_case_cls += case_cls_loss.item()
            total_loss_attn += attn_loss.item()
            total_loss_div += div_loss.item()

            # Hit rate diagnostic (no grad needed, on clean image)
            with torch.no_grad():
                hr, hi = fixation_hit_rate(clean, locations)
                total_hit_rate += hr
                total_hit_intensity += hi

        n = len(dataloader)
        avg_recon = total_loss_recon / n
        avg_letter_cls = total_loss_letter_cls / n
        avg_case_cls = total_loss_case_cls / n
        avg_attn = total_loss_attn / n
        avg_div = total_loss_div / n
        avg_recode = total_loss_recode / n
        avg_hr = total_hit_rate / n
        avg_hi = total_hit_intensity / n
        losses_recon.append(avg_recon)
        losses_letter_cls.append(avg_letter_cls)
        losses_case_cls.append(avg_case_cls)
        losses_attn.append(avg_attn)
        losses_div.append(avg_div)
        losses_recode.append(avg_recode)
        hist_hit_rate.append(avg_hr)
        hist_hit_intensity.append(avg_hi)

        epoch_time = time.time() - epoch_start
        elapsed = time.time() - train_start
        done = epoch - start_epoch + 1
        remaining = epochs - done
        eta_sec = remaining * (elapsed / done)
        eta_min, eta_s = divmod(int(eta_sec), 60)

        print(f"Epoch {epoch+1}/{end_epoch}: "
              f"Recon {avg_recon:.4f}  Ltr {avg_letter_cls:.4f}  "
              f"Case {avg_case_cls:.4f}  Attn {avg_attn:.4f}  "
              f"Div {avg_div:.4f}  Hit {avg_hr:.0%}  "
              f"Recode {avg_recode:.4f}  "
              f"[{epoch_time:.1f}s  ETA {eta_min}m{eta_s:02d}s]")

        if (epoch + 1 - start_epoch) % checkpoint_interval == 0:
            ckpt = {
                'epoch': epoch,
                'model': {k: v.cpu() for k, v in model.state_dict().items()},
                'n_glimpses': n_glimpses, 'patch_size': patch_size,
                'n_scales': n_scales,
                'image_size': 128, 'has_case': True,
                'losses': {
                    'recon': losses_recon, 'letter_cls': losses_letter_cls,
                    'case_cls': losses_case_cls, 'attn': losses_attn,
                    'div': losses_div, 'recode': losses_recode,
                    'hit_rate': hist_hit_rate, 'hit_intensity': hist_hit_intensity,
                },
            }
            torch.save(ckpt, os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth'))

    # Save final
    torch.save({
        'epoch': end_epoch - 1,
        'model': {k: v.cpu() for k, v in model.state_dict().items()},
        'n_glimpses': n_glimpses, 'patch_size': patch_size, 'n_scales': n_scales,
        'image_size': 128, 'has_case': True,
        'losses': {
            'recon': losses_recon, 'letter_cls': losses_letter_cls,
            'case_cls': losses_case_cls, 'attn': losses_attn,
            'div': losses_div, 'recode': losses_recode,
            'hit_rate': hist_hit_rate, 'hit_intensity': hist_hit_intensity,
        },
    }, os.path.join(save_dir, 'model_final.pth'))

    # Training metrics graph (6 subplots)
    epochs_x = range(end_epoch - len(losses_letter_cls) + 1, end_epoch + 1)
    fig, axes = plt.subplots(6, 1, figsize=(8, 14), sharex=True)

    axes[0].plot(epochs_x, losses_recon, label='Recon', color='tab:blue')
    if any(v > 0 for v in losses_recode):
        axes[0].plot(epochs_x, losses_recode, label='Recode', color='tab:cyan',
                     linestyle='--')
    axes[0].set_ylabel('MSE')
    axes[0].legend(loc='upper right')
    axes[0].set_title('Reconstruction')

    axes[1].plot(epochs_x, losses_letter_cls, label='Letter', color='tab:red')
    axes[1].axhline(y=np.log(26), color='gray', linestyle='--',
                    label=f'Random ({np.log(26):.1f})')
    axes[1].set_ylabel('Cross-Entropy')
    axes[1].legend(loc='upper right')
    axes[1].set_title('Letter Classification (26-class)')

    axes[2].plot(epochs_x, losses_case_cls, label='Case', color='tab:pink')
    axes[2].axhline(y=np.log(2), color='gray', linestyle='--',
                    label=f'Random ({np.log(2):.2f})')
    axes[2].set_ylabel('Cross-Entropy')
    axes[2].legend(loc='upper right')
    axes[2].set_title('Case Classification (upper/lower)')

    axes[3].plot(epochs_x, losses_attn, label='Guide', color='tab:green')
    axes[3].set_ylabel('Loss')
    axes[3].legend(loc='upper right')
    axes[3].set_title('Attention guide (lower = fixations on letter)')

    axes[4].plot(epochs_x, losses_div, label='Diversity', color='tab:orange')
    axes[4].set_ylabel('Repulsion')
    axes[4].legend(loc='upper right')
    axes[4].set_title('Fixation diversity (lower = more spread)')

    axes[5].plot(epochs_x, hist_hit_rate, label='Hit rate', color='tab:purple')
    axes[5].plot(epochs_x, hist_hit_intensity, label='Intensity',
                 color='tab:purple', linestyle='--', alpha=0.6)
    axes[5].set_xlabel('Epoch')
    axes[5].set_ylabel('Rate / Intensity')
    axes[5].set_ylim(0, 1)
    axes[5].legend(loc='upper right')
    axes[5].set_title('Fixation hit rate (on sharp letter pixels)')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_metrics.png'), dpi=150)
    plt.close()

    total_time = time.time() - train_start
    total_min, total_s = divmod(int(total_time), 60)
    print(f"Training complete in {total_min}m{total_s:02d}s. "
          f"Model and graph saved in {save_dir}")


# --- Testing ---

def test_model(model_dir, test_data_dir, output_dir='results', device='auto'):
    device = _resolve_device(device)
    print(f"Testing on: {device}")

    os.makedirs(output_dir, exist_ok=True)

    ckpt = torch.load(os.path.join(model_dir, 'model_final.pth'), map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and 'model' in ckpt:
        n_glimpses = ckpt.get('n_glimpses', 10)
        patch_size = ckpt.get('patch_size', 12)
        n_scales = ckpt.get('n_scales', 1)
        state_dict = ckpt['model']
    else:
        n_glimpses, patch_size, n_scales = 10, 12, 1
        state_dict = ckpt

    model = VisionModel(
        n_glimpses=n_glimpses, patch_size=patch_size, n_scales=n_scales,
    ).to(device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    dataset = LetterDataset(test_data_dir)

    letter_correct = 0
    case_correct = 0
    total = 0
    mse_scores = []
    recode_mse_scores = []

    for i in range(len(dataset)):
        img, clean, letter, case, partner_clean = dataset[i]
        img = img.unsqueeze(0).to(device)
        clean = clean.unsqueeze(0).to(device)
        partner_clean = partner_clean.unsqueeze(0).to(device)

        letter_idx = ord(letter) - ord('A')
        case_idx = 0 if case == 'upper' else 1
        case_float = torch.tensor([[float(case_idx)]], device=device)

        with torch.no_grad():
            recon, letter_logits, case_logits, locations, latent = model(img, case_float)

        # Letter accuracy
        letter_pred = letter_logits.argmax(dim=1).item()
        letter_ok = letter_pred == letter_idx
        letter_correct += int(letter_ok)

        # Case accuracy
        case_pred = case_logits.argmax(dim=1).item()
        case_ok = case_pred == case_idx
        case_correct += int(case_ok)

        total += 1

        mse = F.mse_loss(recon, img).item()
        mse_scores.append(mse)

        # Recode: flip case, decode same latent
        with torch.no_grad():
            flipped_case = 1.0 - case_float
            recode_recon = model.decoder(latent, flipped_case)
            hr, hi = fixation_hit_rate(clean, locations)

        # Display character for output
        original_char = letter.lower() if case == 'lower' else letter
        pred_letter = chr(letter_pred + ord('A'))
        letter_mark = 'OK' if letter_ok else f'WRONG({pred_letter})'
        case_mark = 'OK' if case_ok else 'WRONG'

        print(f"  {original_char}: Ltr={letter_mark}  Case={case_mark}  "
              f"MSE={mse:.4f}  Hit={hr:.0%}")

        # Save attention overlay
        visualize_attention(
            img.squeeze(0), locations,
            os.path.join(output_dir, f'attention_{original_char}.png'),
        )

        # Save reconstruction
        recon_img = recon.squeeze().cpu().clamp(0, 1).detach().numpy()
        Image.fromarray((recon_img * 255).astype(np.uint8)).save(
            os.path.join(output_dir, f'recon_{original_char}.png'),
        )

        # Save recode output + compute recode MSE against partner
        recode_img_np = recode_recon.squeeze().cpu().clamp(0, 1).detach().numpy()
        target_char = letter if case == 'lower' else letter.lower()
        Image.fromarray((recode_img_np * 255).astype(np.uint8)).save(
            os.path.join(output_dir, f'recode_{original_char}_to_{target_char}.png'),
        )

        if dataset.has_partners:
            recode_mse = F.mse_loss(recode_recon, partner_clean).item()
            recode_mse_scores.append(recode_mse)

    letter_acc = letter_correct / total if total > 0 else 0
    case_acc = case_correct / total if total > 0 else 0
    avg_mse = np.mean(mse_scores) if mse_scores else 0
    avg_recode_mse = np.mean(recode_mse_scores) if recode_mse_scores else 0

    print(f"\nLetter accuracy: {letter_correct}/{total} ({letter_acc:.1%})")
    print(f"Case accuracy:   {case_correct}/{total} ({case_acc:.1%})")
    print(f"Avg reconstruction MSE: {avg_mse:.4f}")
    if recode_mse_scores:
        print(f"Avg recode MSE:         {avg_recode_mse:.4f}")
    print(f"Results saved in {output_dir}")


# --- Visualization ---

def visualize_model(model_dir, data_dir, output_dir='visualizations', device='auto'):
    device = _resolve_device(device)
    print(f"Visualizing on: {device}")

    os.makedirs(output_dir, exist_ok=True)

    ckpt = torch.load(os.path.join(model_dir, 'model_final.pth'), map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and 'model' in ckpt:
        n_glimpses = ckpt.get('n_glimpses', 10)
        patch_size = ckpt.get('patch_size', 12)
        n_scales = ckpt.get('n_scales', 1)
        state_dict = ckpt['model']
    else:
        n_glimpses, patch_size, n_scales = 10, 12, 1
        state_dict = ckpt

    model = VisionModel(
        n_glimpses=n_glimpses, patch_size=patch_size, n_scales=n_scales,
    ).to(device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    dataset = LetterDataset(data_dir)
    for i in range(len(dataset)):
        img, _clean, letter, case, _partner = dataset[i]
        img = img.unsqueeze(0).to(device)

        case_idx = 0 if case == 'upper' else 1
        case_float = torch.tensor([[float(case_idx)]], device=device)

        with torch.no_grad():
            _, _, _, locations, _ = model(img, case_float)

        original_char = letter.lower() if case == 'lower' else letter
        visualize_attention(
            img.squeeze(0), locations,
            os.path.join(output_dir, f'attention_{original_char}.png'),
        )
        print(f"Saved attention visualization for '{original_char}'")

    print(f"Visualizations saved in {output_dir}")


# --- Attention Pre-Check ---

def check_attention(data_dir, n_epochs=10, n_glimpses=10, patch_size=12,
                    n_scales=1, device='auto',
                    guide_weight=4.0, blur_sigma_ratio=0.16,
                    diversity_weight=1.0, diversity_sigma=0.1):
    """Quick diagnostic: can the attention guide pull fixations onto letter content?

    Runs a few epochs with ONLY attention + diversity loss (no cls/recon/recode).
    If hit rate doesn't improve, the guide config is wrong for this image size.
    """
    device = _resolve_device(device)
    dataset = LetterDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=26, shuffle=True,
                            pin_memory=device.type == 'cuda')

    model = VisionModel(
        n_glimpses=n_glimpses, patch_size=patch_size, n_scales=n_scales,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Report effective blur_sigma from image dimensions
    sample_img = dataset[0][0]
    img_h, img_w = sample_img.shape[1], sample_img.shape[2]
    effective_sigma = blur_sigma_ratio * min(img_h, img_w)

    print(f"Attention pre-check on {device}")
    print(f"Image: {img_h}x{img_w}  blur_sigma_ratio={blur_sigma_ratio} "
          f"-> {effective_sigma:.1f}px  guide_weight={guide_weight}")
    print(f"Running {n_epochs} diagnostic epochs (attention + diversity only)...")

    hit_rates = []
    for epoch in range(n_epochs):
        total_hr = 0
        total_attn = 0
        n = 0
        for img, clean, _letters, cases, _partner in dataloader:
            img = img.to(device)
            clean = clean.to(device)
            case_idx = torch.tensor(
                [0 if c == 'upper' else 1 for c in cases], device=device,
            )
            case_float = case_idx.float().unsqueeze(1)

            _, _, _, locations, _ = model(img, case_float)

            attn_loss = attention_content_loss(
                clean, locations, blur_sigma_ratio=blur_sigma_ratio,
            )
            div_loss = fixation_diversity_loss(locations, sigma=diversity_sigma)
            loss = guide_weight * attn_loss + diversity_weight * div_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            with torch.no_grad():
                hr, _ = fixation_hit_rate(clean, locations)
                total_hr += hr
                total_attn += attn_loss.item()
            n += 1

        avg_hr = total_hr / n
        avg_attn = total_attn / n
        hit_rates.append(avg_hr)
        print(f"  Epoch {epoch+1}/{n_epochs}: Hit {avg_hr:.0%}  Attn {avg_attn:.4f}")

    initial_hr = hit_rates[0]
    final_hr = hit_rates[-1]
    peak_hr = max(hit_rates)
    improved = peak_hr > initial_hr + 0.05

    print()
    if peak_hr >= 0.20:
        print(f"PASS: Hit rate {initial_hr:.0%} -> {final_hr:.0%} "
              f"(peak {peak_hr:.0%}). Attention guide is working.")
        return True
    elif improved:
        print(f"WEAK: Hit rate {initial_hr:.0%} -> {final_hr:.0%} "
              f"(peak {peak_hr:.0%}). Improving but low — consider "
              f"increasing guide_weight (current: {guide_weight}).")
        return True
    else:
        print(f"FAIL: Hit rate {initial_hr:.0%} -> {final_hr:.0%} "
              f"(peak {peak_hr:.0%}). Attention guide has no effect. "
              f"Increase blur_sigma_ratio (current: {blur_sigma_ratio}) "
              f"or guide_weight (current: {guide_weight}).")
        return False


# --- CLI ---

def _parse_letters(letters_str):
    if letters_str == 'A-Z':
        return [chr(i) for i in range(65, 91)]
    if letters_str == 'a-z':
        return [chr(i) for i in range(97, 123)]
    if letters_str in ('Aa-Zz', 'A-Za-z'):
        return [chr(i) for i in range(65, 91)] + [chr(i) for i in range(97, 123)]
    return list(letters_str)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Vision-Only Letter Training Pipeline')
    subparsers = parser.add_subparsers(dest='command', required=True)

    gen_parser = subparsers.add_parser('generate')
    gen_parser.add_argument('--letters', default='Aa-Zz')
    gen_parser.add_argument('--num_variants', type=int, default=20)
    gen_parser.add_argument('--noise_level', type=float, default=0.01)
    gen_parser.add_argument('--output_dir', default='data/letters')

    gentest_parser = subparsers.add_parser('generate_test')
    gentest_parser.add_argument('--letters', default='Aa-Zz')
    gentest_parser.add_argument('--output_dir', default='data/test')

    train_parser = subparsers.add_parser('train')
    train_parser.add_argument('--data_dir', required=True)
    train_parser.add_argument('--epochs', type=int, default=200)
    train_parser.add_argument('--save_dir', default='models')
    train_parser.add_argument('--checkpoint_interval', type=int, default=10)
    train_parser.add_argument('--n_glimpses', type=int, default=10)
    train_parser.add_argument('--patch_size', type=int, default=12)
    train_parser.add_argument('--n_scales', type=int, default=1)
    train_parser.add_argument('--device', default='auto',
                              choices=['auto', 'cpu', 'cuda'])
    train_parser.add_argument('--resume', default=None)
    train_parser.add_argument('--diversity_weight', type=float, default=1.0,
                              help='Weight for fixation diversity loss (0=off)')
    train_parser.add_argument('--diversity_sigma', type=float, default=0.1,
                              help='Repulsion radius in normalized coords (0.1=10%% of image)')
    train_parser.add_argument('--recode_weight', type=float, default=1.0,
                              help='Weight for recode loss (0=off)')
    train_parser.add_argument('--guide_weight', type=float, default=4.0,
                              help='Weight for attention guide loss')
    train_parser.add_argument('--blur_sigma_ratio', type=float, default=0.16,
                              help='Blur sigma as fraction of image size (0.16=proven default)')

    test_parser = subparsers.add_parser('test')
    test_parser.add_argument('--model_dir', required=True)
    test_parser.add_argument('--test_data_dir', required=True)
    test_parser.add_argument('--output_dir', default='results')
    test_parser.add_argument('--device', default='auto',
                             choices=['auto', 'cpu', 'cuda'])

    viz_parser = subparsers.add_parser('visualize')
    viz_parser.add_argument('--model_dir', required=True)
    viz_parser.add_argument('--data_dir', required=True)
    viz_parser.add_argument('--output_dir', default='visualizations')
    viz_parser.add_argument('--device', default='auto',
                            choices=['auto', 'cpu', 'cuda'])

    chk_parser = subparsers.add_parser('check_attention')
    chk_parser.add_argument('--data_dir', required=True)
    chk_parser.add_argument('--n_epochs', type=int, default=10)
    chk_parser.add_argument('--n_glimpses', type=int, default=10)
    chk_parser.add_argument('--patch_size', type=int, default=12)
    chk_parser.add_argument('--n_scales', type=int, default=1)
    chk_parser.add_argument('--device', default='auto',
                            choices=['auto', 'cpu', 'cuda'])
    chk_parser.add_argument('--guide_weight', type=float, default=4.0)
    chk_parser.add_argument('--blur_sigma_ratio', type=float, default=0.16)
    chk_parser.add_argument('--diversity_weight', type=float, default=1.0)
    chk_parser.add_argument('--diversity_sigma', type=float, default=0.1)

    args = parser.parse_args()

    if args.command == 'generate':
        letters = _parse_letters(args.letters)
        generate_dataset(letters, args.output_dir, args.noise_level, args.num_variants)

    elif args.command == 'generate_test':
        letters = _parse_letters(args.letters)
        generate_test(letters, args.output_dir)

    elif args.command == 'train':
        train_model(args.data_dir, args.epochs, args.resume, args.save_dir,
                    args.checkpoint_interval, n_glimpses=args.n_glimpses,
                    patch_size=args.patch_size, n_scales=args.n_scales,
                    device=args.device,
                    diversity_weight=args.diversity_weight,
                    diversity_sigma=args.diversity_sigma,
                    recode_weight=args.recode_weight,
                    guide_weight=args.guide_weight,
                    blur_sigma_ratio=args.blur_sigma_ratio)

    elif args.command == 'test':
        test_model(args.model_dir, args.test_data_dir, args.output_dir,
                   device=args.device)

    elif args.command == 'visualize':
        visualize_model(args.model_dir, args.data_dir, args.output_dir,
                        device=args.device)

    elif args.command == 'check_attention':
        check_attention(args.data_dir, n_epochs=args.n_epochs,
                        n_glimpses=args.n_glimpses, patch_size=args.patch_size,
                        n_scales=args.n_scales, device=args.device,
                        guide_weight=args.guide_weight,
                        blur_sigma_ratio=args.blur_sigma_ratio,
                        diversity_weight=args.diversity_weight,
                        diversity_sigma=args.diversity_sigma)
