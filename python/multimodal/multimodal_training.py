# Note: To run this script on your local machine, install required packages:
# pip install torch torchaudio numpy matplotlib pillow gtts moviepy librosa
# gTTS requires internet for realistic speech synthesis.
# MoviePy for video generation/loading, Librosa for MFCC inversion to audio.
# If no internet or gTTS not desired, set --use_real_tts False in generate command (uses synthetic sine waves).
# If moviepy/librosa not installed, script falls back to separate image/audio files.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import json
import shutil
import time
from PIL import Image, ImageDraw, ImageFont
try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False
    print("gTTS not installed; using synthetic audio fallback.")

try:
    import moviepy.editor as mpy
    from moviepy.video.io.bindings import mplfig_to_npimage
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False
    print("MoviePy not installed; falling back to separate image/audio files.")

try:
    import librosa
    import soundfile as sf
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    print("Librosa not installed; test outputs MFCC npy instead of wav.")

# --- Clean Video Generation (text + sound in .mp4) ---

def generate_clean_video(letters, output_dir, use_real_tts=True, lang='en', duration=1.0):
    if not MOVIEPY_AVAILABLE:
        print("MoviePy not available; generating separate clean audio instead.")
        generate_clean_tts(letters, output_dir, use_real_tts, lang)
        return

    os.makedirs(output_dir, exist_ok=True)
    for letter in letters:
        video_path = os.path.join(output_dir, f'video_{letter}.mp4')

        # Generate text image with PIL for MoviePy
        vid_size = 96
        img = Image.new('L', (vid_size, vid_size), color=0)
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default(size=60)
        bbox = draw.textbbox((0, 0), letter, font=font)
        x = (vid_size - bbox[2] - bbox[0]) / 2
        y = (vid_size - bbox[3] - bbox[1]) / 2
        draw.text((x, y), letter, fill=255, font=font)
        frame_rgb = np.stack([np.array(img)] * 3, axis=-1)  # Grayscale to RGB

        def make_frame(t):
            return frame_rgb

        clip = mpy.VideoClip(make_frame, duration=duration)

        # Audio
        if use_real_tts and GTTS_AVAILABLE:
            temp_audio_path = 'temp.mp3'
            tts = gTTS(letter, lang=lang)
            tts.save(temp_audio_path)
            audio_clip = mpy.AudioFileClip(temp_audio_path)
        else:
            waveform, sr = generate_letter_audio_fallback(letter, duration=duration)
            temp_audio_path = 'temp.wav'
            torchaudio.save(temp_audio_path, waveform, sr)
            audio_clip = mpy.AudioFileClip(temp_audio_path)

        # Pad or trim audio to match video duration
        if audio_clip.duration < duration:
            silence = mpy.AudioClip(lambda t: [0], duration=duration - audio_clip.duration, fps=audio_clip.fps or 22050)
            audio_clip = mpy.concatenate_audioclips([audio_clip, silence])
        else:
            audio_clip = audio_clip.subclip(0, duration)

        # Save canonical WAV (lossless reference for MFCC extraction)
        wav_path = os.path.join(output_dir, f'audio_{letter}.wav')
        audio_clip.write_audiofile(wav_path, fps=audio_clip.fps or 22050, nbytes=2, codec='pcm_s16le')

        clip = clip.set_audio(audio_clip)
        clip.write_videofile(video_path, fps=1, codec='libx264', audio_codec='aac')
        os.remove(temp_audio_path)

    print(f"Clean videos generated in {output_dir}")

# --- Fallback Clean TTS if no MoviePy ---

def generate_clean_tts(letters, output_dir, use_real_tts=True, lang='en'):
    os.makedirs(output_dir, exist_ok=True)
    for letter in letters:
        audio_ext = '.mp3' if use_real_tts else '.wav'
        audio_path = os.path.join(output_dir, f'audio_{letter}{audio_ext}')

        if use_real_tts and GTTS_AVAILABLE:
            tts = gTTS(letter, lang=lang)
            tts.save(audio_path)
        else:
            waveform, sr = generate_letter_audio_fallback(letter)
            torchaudio.save(audio_path, waveform, sr)

    print(f"Clean TTS generated in {output_dir}")

# --- Dataset Generation (add noise to clean video) ---

def generate_dataset(letters, output_dir, noise_level=0.01, use_real_tts=True, clean_video_dir=None, num_variants=1):
    if not MOVIEPY_AVAILABLE:
        print("MoviePy not available; generating with separate files.")
        generate_dataset_fallback(letters, output_dir, noise_level, use_real_tts, clean_video_dir)
        return

    os.makedirs(output_dir, exist_ok=True)
    samples = []
    for letter in letters:
        if clean_video_dir:
            clean_path = os.path.join(clean_video_dir, f'video_{letter}.mp4')
            if not os.path.exists(clean_path):
                raise ValueError(f"Clean video for {letter} not found in {clean_video_dir}")
        else:
            raise ValueError("clean_video_dir required for video mode")

        clean_wav_path = os.path.join(clean_video_dir, f'audio_{letter}.wav')
        clean_waveform, clean_sr = torchaudio.load(clean_wav_path)

        for v in range(num_variants):
            suffix = f'{letter}_{v}'
            noisy_path = os.path.join(output_dir, f'video_{suffix}.mp4')

            clip = mpy.VideoFileClip(clean_path)

            # Add noise to video frames (unique per variant)
            nl = noise_level
            def add_video_noise(frame):
                frame = frame.astype(float) / 255.0
                noisy_frame = frame + np.random.normal(0, nl, frame.shape)
                noisy_frame = np.clip(noisy_frame, 0, 1) * 255
                return noisy_frame.astype(np.uint8)

            noisy_clip = clip.fl_image(add_video_noise)

            # Add noise to audio from clean WAV (lossless source)
            noisy_waveform = clean_waveform + torch.randn_like(clean_waveform) * noise_level
            noisy_waveform = noisy_waveform.clamp(-1, 1)

            # Save noisy audio as lossless WAV
            noisy_wav_path = os.path.join(output_dir, f'audio_{suffix}.wav')
            torchaudio.save(noisy_wav_path, noisy_waveform, clean_sr)

            # Attach noisy audio to video for playback
            noisy_audio_clip = mpy.AudioFileClip(noisy_wav_path)
            noisy_clip = noisy_clip.set_audio(noisy_audio_clip)

            noisy_clip.write_videofile(noisy_path, codec='libx264', audio_codec='aac')

            samples.append({'letter': letter, 'video': noisy_path, 'audio': noisy_wav_path})

    metadata = {s['letter'] + '_' + str(i): {'video': s['video'], 'audio': s['audio'], 'letter': s['letter']}
                for i, s in enumerate(samples)}
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f)
    print(f"Video dataset generated in {output_dir}: {len(samples)} samples ({num_variants} variants per letter)")

# --- Fallback Dataset Gen if no MoviePy ---

def generate_dataset_fallback(letters, output_dir, noise_level=0.01, use_real_tts=True, clean_audio_dir=None):
    os.makedirs(output_dir, exist_ok=True)
    metadata = {}
    for letter in letters:
        img = generate_letter_image(letter)
        noisy_img = add_noise(img, noise_level)

        img_path = os.path.join(output_dir, f'img_{letter}.png')

        audio_ext = '.mp3' if use_real_tts else '.wav'
        audio_path = os.path.join(output_dir, f'audio_{letter}{audio_ext}')

        # Save noisy image
        img_pil = Image.fromarray((noisy_img.squeeze(0).numpy() * 255).astype(np.uint8))
        img_pil.save(img_path)

        # Audio: Use clean reference if provided, else generate
        if clean_audio_dir:
            clean_audio_path = os.path.join(clean_audio_dir, f'audio_{letter}{audio_ext}')
            if not os.path.exists(clean_audio_path):
                raise ValueError(f"Clean audio for {letter} not found in {clean_audio_dir}")
            waveform, sr = torchaudio.load(clean_audio_path)
        else:
            if use_real_tts and GTTS_AVAILABLE:
                temp_audio = BytesIO()
                tts = gTTS(letter, lang='en')
                tts.save(temp_audio)
                temp_audio.seek(0)
                waveform, sr = torchaudio.load(temp_audio)
            else:
                waveform, sr = generate_letter_audio_fallback(letter)

        # Add noise to waveform
        noisy_waveform = add_noise(waveform, noise_level)

        # Save noisy audio
        torchaudio.save(audio_path, noisy_waveform, sr)

        metadata[letter] = {'img': img_path, 'audio': audio_path}

    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f)
    print(f"Fallback dataset (separate files) generated in {output_dir}")

# --- MFCC Extraction ---

def extract_mfcc(waveform, sr, n_mfcc=13, n_frames=50):
    """Extract MFCC features from a waveform tensor. Returns tensor of shape (n_mfcc, n_frames)."""
    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=sr, n_mfcc=n_mfcc,
        melkwargs={'n_fft': 400, 'hop_length': 160, 'n_mels': 23}
    )
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    mfcc = mfcc_transform(waveform).squeeze(0)  # (n_mfcc, time)
    # Pad or trim to fixed n_frames
    if mfcc.shape[1] < n_frames:
        mfcc = torch.nn.functional.pad(mfcc, (0, n_frames - mfcc.shape[1]))
    else:
        mfcc = mfcc[:, :n_frames]
    return mfcc

# --- Dataset Loading ---

class LetterDataset(Dataset):
    def __init__(self, data_dir):
        with open(os.path.join(data_dir, 'metadata.json'), 'r') as f:
            metadata = json.load(f)
        keys = list(metadata.keys())
        first = metadata[keys[0]]
        video_mode = 'video' in first

        # Preload entire dataset into memory
        print(f"Loading {len(keys)} samples into memory...", end=' ', flush=True)
        t0 = time.time()
        self.images = []
        self.mfccs = []
        self.letters = []

        for key in keys:
            entry = metadata[key]
            letter = entry.get('letter', key)
            if video_mode:
                if not MOVIEPY_AVAILABLE:
                    raise RuntimeError("MoviePy required for video mode.")
                clip = mpy.VideoFileClip(entry['video'])
                frame = clip.get_frame(clip.duration / 2)
                img_array = frame[:,:,0] / 255.0 if frame.ndim == 3 else frame / 255.0
                img_tensor = torch.tensor(img_array, dtype=torch.float32).unsqueeze(0)
                clip.close()
                waveform, sr = torchaudio.load(entry['audio'])
            else:
                img = Image.open(entry['img']).convert('L')
                img_tensor = torch.tensor(np.array(img) / 255.0, dtype=torch.float32).unsqueeze(0)
                waveform, sr = torchaudio.load(entry['audio'])

            self.images.append(img_tensor)
            self.mfccs.append(extract_mfcc(waveform, sr))
            self.letters.append(letter)

        print(f"done ({time.time() - t0:.1f}s)")

    def __len__(self):
        return len(self.letters)

    def __getitem__(self, idx):
        return self.images[idx], self.mfccs[idx], self.letters[idx]

def generate_letter_image(letter, size=(32, 32)):
    img = Image.new('L', size, color=0)  # Grayscale
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), letter, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((size[0] - w) / 2, (size[1] - h) / 2), letter, fill=255, font=font)
    img_array = np.array(img) / 255.0
    return torch.tensor(img_array, dtype=torch.float32).unsqueeze(0)  # Channel dim

def generate_letter_audio_fallback(letter, sample_rate=16000, duration=0.5):
    freq = 400 + 20 * (ord(letter) - 65)  # Unique freq per letter A-Z
    t = torch.linspace(0, duration, int(sample_rate * duration))
    waveform = torch.sin(2 * np.pi * freq * t).unsqueeze(0)  # Mono
    return waveform, sample_rate

# --- Glimpse Sensor (multi-resolution raw-pixel patches via grid_sample) ---

class GlimpseSensor(nn.Module):
    """Extracts multi-resolution patches from the RAW IMAGE at a given location.

    The model only sees what the attention decides to look at — no global CNN
    preprocessing. Each glimpse is processed by a small per-patch CNN.

    Strict foveal-only field of view: each glimpse sees ONLY patch_size×patch_size
    raw pixels. No peripheral context. If the fixation misses the letter,
    the model sees black — attention MUST hit the letter for any useful
    information to reach the latent space. Default patch_size=12, n_scales=1:
      Scale 1: 12x12 pixels (1.6% of 96x96 image)
    With 10 glimpses: 15.6% max coverage — model must place fixations carefully.
    """
    def __init__(self, patch_size=12, n_scales=1, latent_dim=256):
        super().__init__()
        self.patch_size = patch_size
        self.scales = [2**i for i in range(n_scales)]  # e.g. [1, 2] for n_scales=2

        # Per-glimpse CNN: multi-scale patches stacked as channels
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

        # Stack scales as channels: (B, n_scales, patch_size, patch_size)
        combined = torch.cat(patches, dim=1)
        feat = self.patch_cnn(combined)

        glimpse_feat = self.glimpse_fc(feat)
        loc_feat = self.location_fc(location)
        return self.combine_fc(torch.cat([glimpse_feat, loc_feat], dim=1))

    def _make_grid(self, location, scale, H, W):
        """Create a sampling grid centered at location with given scale."""
        B = location.shape[0]
        delta_h = scale * self.patch_size / H
        delta_w = scale * self.patch_size / W

        grid_y = torch.linspace(-delta_h, delta_h, self.patch_size, device=location.device)
        grid_x = torch.linspace(-delta_w, delta_w, self.patch_size, device=location.device)
        grid_y, grid_x = torch.meshgrid(grid_y, grid_x, indexing='ij')

        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)
        loc = location.view(B, 1, 1, 2)
        return grid + loc

# --- Attention Controller (GRU-based) ---

class AttentionController(nn.Module):
    """GRU that decides where to look next. Stores fixation locations for visualization."""
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

# --- Visual Attention Encoder (composes CNN + Glimpse + Controller) ---

class VisualAttentionEncoder(nn.Module):
    """Recurrent spatial attention on raw pixels.

    No global CNN preprocessing — the model only sees what the attention
    decides to look at. Each glimpse extracts multi-resolution patches
    from the raw image and processes them with a small per-patch CNN.
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

# --- CNN Visual Decoder (transposed CNN) ---

class CNNVisualDecoder(nn.Module):
    """Generates 96x96 images from latent vectors using transposed convolutions."""
    def __init__(self, latent_dim=256):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 128 * 24 * 24),
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

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 128, 24, 24)
        return self.deconv(x)

# --- Bidirectional Model ---

class BidirectionalModel(nn.Module):
    def __init__(self, mfcc_dim=13*50, latent_dim=256, n_glimpses=10, patch_size=12, n_scales=1):
        super().__init__()
        self.visual_encoder = VisualAttentionEncoder(
            n_glimpses=n_glimpses, patch_size=patch_size,
            n_scales=n_scales, latent_dim=latent_dim,
        )
        self.audio_encoder = nn.Sequential(
            nn.Linear(mfcc_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
        self.audio_decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, mfcc_dim)
        )
        self.visual_decoder = CNNVisualDecoder(latent_dim=latent_dim)
        self.classifier = nn.Linear(latent_dim, 26)
        self._last_attention_info = None
        self._last_visual_latent = None

    def visual_to_audio(self, img):
        latent, locations = self.visual_encoder(img)
        self._last_attention_info = {'locations': locations}
        self._last_visual_latent = latent
        pred_mfcc_flat = self.audio_decoder(latent)
        return pred_mfcc_flat.view(-1, 13, 50)

    def audio_to_visual(self, mfcc):
        mfcc_flat = mfcc.flatten(1)
        latent = self.audio_encoder(mfcc_flat)
        self._last_audio_latent = latent
        return self.visual_decoder(latent)

# --- Attention Visualization ---

def visualize_attention(img_tensor, attention_info, save_path):
    """Overlay fixation points and saccade arrows on image."""
    locations = attention_info['locations']
    img = img_tensor.squeeze(0).cpu().detach().numpy()  # (H, W)
    H, W = img.shape

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.imshow(img, cmap='gray', vmin=0, vmax=1)

    colors = plt.cm.hot(np.linspace(0.2, 0.9, len(locations)))

    for i, loc in enumerate(locations):
        loc_np = loc[0].cpu().detach().numpy()  # (2,) first item in batch
        # Convert from [-1, 1] normalized coords to pixel coords
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

def attention_content_loss(image, locations, blur_sigma=15.0):
    """Guide fixations toward letter strokes using a blurred guidance field.

    The sharp image has zero spatial gradient in dark areas, so fixations
    there receive no gradient signal. A Gaussian-blurred guide extends
    the influence of bright pixels across the entire image — a smooth
    'scent trail' pulling fixations toward letter strokes from anywhere.

    Only affects WHERE the model looks (location gradients).
    Does NOT feed image information to the model — the model still sees
    only its tiny foveal patches through the GlimpseSensor.
    """
    B, C, H, W = image.shape

    # Gaussian blur: separable 2D convolution
    k = int(4 * blur_sigma) | 1  # kernel covers ~4sigma, rounded to odd
    x = torch.arange(k, device=image.device, dtype=image.dtype) - k // 2
    gauss = torch.exp(-x ** 2 / (2 * blur_sigma ** 2))
    gauss = gauss / gauss.sum()
    guide = F.conv2d(image, gauss.view(1, 1, k, 1), padding=(k // 2, 0))
    guide = F.conv2d(guide, gauss.view(1, 1, 1, k), padding=(0, k // 2))

    # Sample guide value at each fixation (single point — not a patch)
    total = 0
    for loc in locations[1:]:
        grid = loc.view(B, 1, 1, 2)
        sampled = F.grid_sample(guide, grid, align_corners=True, padding_mode='zeros')
        total = total + sampled.mean()
    return -total / len(locations[1:])

# --- Training ---

def _resolve_device(device_str):
    if device_str == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device_str)

def train_model(data_dir, epochs=200, resume=None, save_dir='models',
                checkpoint_interval=10, n_glimpses=10, patch_size=12, n_scales=1,
                device='auto', mode='vision'):
    device = _resolve_device(device)
    vision_only = (mode == 'vision')
    print(f"Training on: {device}  mode: {mode}")

    os.makedirs(save_dir, exist_ok=True)
    dataset = LetterDataset(data_dir)
    use_cuda = device.type == 'cuda'
    dataloader = DataLoader(dataset, batch_size=26, shuffle=True, pin_memory=use_cuda)

    model = BidirectionalModel(
        n_glimpses=n_glimpses, patch_size=patch_size, n_scales=n_scales,
    ).to(device)

    start_epoch = 0
    losses_recon = []
    losses_cls = []
    losses_attn = []
    # Multimodal-only histories
    losses_v2a = []
    losses_a2v = []
    losses_cycle = []
    losses_align = []

    if resume:
        checkpoint = torch.load(resume, map_location=device)
        model.load_state_dict(checkpoint['model'], strict=False)
        start_epoch = checkpoint['epoch'] + 1
        # Restore loss history for continuous graphs across resumes
        if 'losses' in checkpoint:
            h = checkpoint['losses']
            losses_recon = h.get('recon', [])
            losses_cls = h.get('cls', [])
            losses_attn = h.get('attn', [])
            losses_v2a = h.get('v2a', [])
            losses_a2v = h.get('a2v', [])
            losses_cycle = h.get('cycle', [])
            losses_align = h.get('align', [])
        print(f"Resumed from epoch {start_epoch} ({len(losses_cls)} prior epochs of history)")

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    end_epoch = start_epoch + epochs
    train_start = time.time()

    for epoch in range(start_epoch, end_epoch):
        epoch_start = time.time()
        total_loss_recon = 0
        total_loss_attn = 0
        total_loss_cls = 0
        total_loss_v2a = 0
        total_loss_a2v = 0
        total_loss_cycle = 0
        total_loss_align = 0
        for img, mfcc, letters in dataloader:
            img = img.to(device)
            letter_idx = torch.tensor([ord(l) - ord('A') for l in letters], device=device)

            # Visual encode (attention) → latent
            latent, locations = model.visual_encoder(img)
            model._last_attention_info = {'locations': locations}
            model._last_visual_latent = latent

            # Visual classification
            cls_loss = F.cross_entropy(model.classifier(latent), letter_idx)
            total_loss_cls += cls_loss.item()

            # Attention guide: blurred scent trail toward letter strokes
            attn_loss = attention_content_loss(img, locations)
            total_loss_attn += attn_loss.item()

            # Visual reconstruction: encode → decode → reconstruct
            pred_img = model.visual_decoder(latent)
            recon_loss = criterion(pred_img, img)
            total_loss_recon += recon_loss.item()

            if vision_only:
                total_loss = recon_loss + cls_loss + 2.0 * attn_loss
            else:
                mfcc = mfcc.to(device)

                # V2A: visual latent → audio
                pred_mfcc = model.audio_decoder(latent).view(-1, 13, 50)
                loss_v2a = criterion(pred_mfcc, mfcc)
                total_loss_v2a += loss_v2a.item()

                # A2V: audio → visual
                mfcc_flat = mfcc.flatten(1)
                audio_latent = model.audio_encoder(mfcc_flat)
                model._last_audio_latent = audio_latent
                pred_img_a2v = model.visual_decoder(audio_latent)
                loss_a2v = criterion(pred_img_a2v, img)
                total_loss_a2v += loss_a2v.item()

                # Audio classification
                cls_loss_a = F.cross_entropy(model.classifier(audio_latent), letter_idx)
                cls_loss = cls_loss + cls_loss_a
                total_loss_cls += cls_loss_a.item()

                # Mutual alignment
                align_loss = (
                    F.mse_loss(audio_latent, latent.detach()) +
                    F.mse_loss(latent, audio_latent.detach())
                )
                total_loss_align += align_loss.item()

                # Cycle consistency
                cycle_mfcc_latent, _ = model.visual_encoder(pred_img_a2v)
                cycle_mfcc = model.audio_decoder(cycle_mfcc_latent).view(-1, 13, 50)
                cycle_img = model.visual_decoder(model.audio_encoder(pred_mfcc.flatten(1)))
                cycle_loss = 0.5 * (criterion(cycle_mfcc, mfcc) + criterion(cycle_img, img))
                total_loss_cycle += cycle_loss.item()

                total_loss = (recon_loss + loss_v2a + loss_a2v + 0.5 * cycle_loss
                              + 2.0 * attn_loss + cls_loss + align_loss)

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        n = len(dataloader)
        avg_recon = total_loss_recon / n
        avg_cls = total_loss_cls / n
        avg_attn = total_loss_attn / n
        losses_recon.append(avg_recon)
        losses_cls.append(avg_cls)
        losses_attn.append(avg_attn)

        epoch_time = time.time() - epoch_start
        elapsed = time.time() - train_start
        done = epoch - start_epoch + 1
        remaining = epochs - done
        eta_sec = remaining * (elapsed / done)
        eta_min, eta_s = divmod(int(eta_sec), 60)

        if vision_only:
            print(f"Epoch {epoch+1}/{end_epoch}: Recon {avg_recon:.4f}  Cls {avg_cls:.4f}  Attn {avg_attn:.4f}  [{epoch_time:.1f}s/epoch  ETA {eta_min}m{eta_s:02d}s]")
        else:
            avg_v2a = total_loss_v2a / n
            avg_a2v = total_loss_a2v / n
            avg_cycle = total_loss_cycle / n
            avg_align = total_loss_align / n
            losses_v2a.append(avg_v2a)
            losses_a2v.append(avg_a2v)
            losses_cycle.append(avg_cycle)
            losses_align.append(avg_align)
            print(f"Epoch {epoch+1}/{end_epoch}: Recon {avg_recon:.4f}  V2A {avg_v2a:.4f}  A2V {avg_a2v:.4f}  Cycle {avg_cycle:.4f}  Cls {avg_cls:.4f}  Attn {avg_attn:.4f}  Align {avg_align:.4f}  [{epoch_time:.1f}s/epoch  ETA {eta_min}m{eta_s:02d}s]")

        if (epoch + 1 - start_epoch) % checkpoint_interval == 0:
            ckpt = {
                'epoch': epoch, 'mode': mode,
                'model': {k: v.cpu() for k, v in model.state_dict().items()},
                'n_glimpses': n_glimpses, 'patch_size': patch_size, 'n_scales': n_scales,
                'losses': {
                    'recon': losses_recon, 'cls': losses_cls, 'attn': losses_attn,
                    'v2a': losses_v2a, 'a2v': losses_a2v, 'cycle': losses_cycle,
                    'align': losses_align,
                },
            }
            torch.save(ckpt, os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth'))

    # Save final
    torch.save({
        'epoch': end_epoch - 1, 'mode': mode,
        'model': {k: v.cpu() for k, v in model.state_dict().items()},
        'n_glimpses': n_glimpses, 'patch_size': patch_size, 'n_scales': n_scales,
        'losses': {
            'recon': losses_recon, 'cls': losses_cls, 'attn': losses_attn,
            'v2a': losses_v2a, 'a2v': losses_a2v, 'cycle': losses_cycle,
            'align': losses_align,
        },
    }, os.path.join(save_dir, 'model_final.pth'))

    # --- Training metrics graph ---
    epochs_x = range(end_epoch - len(losses_cls) + 1, end_epoch + 1)

    if vision_only:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 7), sharex=True)

        ax1.plot(epochs_x, losses_recon, label='Visual recon', color='tab:blue')
        ax1.set_ylabel('MSE')
        ax1.legend(loc='upper right')
        ax1.set_title('Reconstruction (attention → latent → decode)')

        ax2.plot(epochs_x, losses_cls, label='Cls', color='tab:red')
        ax2.axhline(y=np.log(26), color='gray', linestyle='--', label=f'Random ({np.log(26):.1f})')
        ax2.set_ylabel('Cross-Entropy')
        ax2.legend(loc='upper right')
        ax2.set_title('Classification (visual)')

        ax3.plot(epochs_x, losses_attn, label='Guide', color='tab:green')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Loss')
        ax3.legend(loc='upper right')
        ax3.set_title('Attention guide (lower = fixations on letter)')
    else:
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(8, 10), sharex=True)

        ax1.plot(epochs_x, losses_recon, label='Visual recon')
        ax1.plot(epochs_x, losses_v2a, label='V2A')
        ax1.plot(epochs_x, losses_a2v, label='A2V')
        ax1.plot(epochs_x, losses_cycle, label='Cycle')
        ax1.set_ylabel('MSE')
        ax1.legend(loc='upper right')
        ax1.set_title('Reconstruction')

        ax2.plot(epochs_x, losses_cls, label='Cls (V+A)', color='tab:red')
        ax2.axhline(y=2 * np.log(26), color='gray', linestyle='--', label=f'Random ({2 * np.log(26):.1f})')
        ax2.set_ylabel('Cross-Entropy')
        ax2.legend(loc='upper right')
        ax2.set_title('Classification (visual + audio)')

        ax3.plot(epochs_x, losses_attn, label='Guide', color='tab:green')
        ax3.set_ylabel('Loss')
        ax3.legend(loc='upper right')
        ax3.set_title('Attention guide (lower = fixations on letter)')

        ax4.plot(epochs_x, losses_align, label='Align (mutual)', color='tab:purple')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('MSE')
        ax4.legend(loc='upper right')
        ax4.set_title('Latent alignment (V↔A)')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_metrics.png'), dpi=150)
    plt.close()
    total_time = time.time() - train_start
    total_min, total_s = divmod(int(total_time), 60)
    print(f"Training complete in {total_min}m{total_s:02d}s. Model and graph saved in {save_dir}")

# --- Testing (detect missing modality in video) ---

def test_model(model_dir, test_data_dir, output_dir='results',
               n_glimpses=10, patch_size=12, n_scales=1, device='auto'):
    device = _resolve_device(device)
    print(f"Testing on: {device}")

    os.makedirs(output_dir, exist_ok=True)

    # Load checkpoint (handles both new dict format and legacy raw state_dict)
    ckpt = torch.load(os.path.join(model_dir, 'model_final.pth'), map_location=device)
    if isinstance(ckpt, dict) and 'model' in ckpt:
        n_glimpses = ckpt.get('n_glimpses', n_glimpses)
        patch_size = ckpt.get('patch_size', patch_size)
        n_scales = ckpt.get('n_scales', n_scales)
        state_dict = ckpt['model']
    else:
        state_dict = ckpt

    model = BidirectionalModel(
        n_glimpses=n_glimpses, patch_size=patch_size, n_scales=n_scales,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    # Load test manifest
    manifest_path = os.path.join(test_data_dir, 'metadata.json')
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    scores = []
    for letter, entry in manifest.items():
        video_path = entry['video']
        audio_path = entry['audio']
        mask = entry['mask']

        if not MOVIEPY_AVAILABLE:
            print("MoviePy required for video test; skipping.")
            continue

        # Extract frame from video
        clip = mpy.VideoFileClip(video_path)
        frame = clip.get_frame(clip.duration / 2)
        img_array = frame[:,:,0] / 255.0 if frame.ndim == 3 else frame / 255.0
        img_tensor = torch.tensor(img_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Batch, channel
        clip.close()

        # Load audio from lossless WAV
        waveform, sr = torchaudio.load(audio_path)
        mfcc = extract_mfcc(waveform, sr)

        if mask == 'silent':
            # Visual-only input: Generate audio
            with torch.no_grad():
                pred_mfcc = model.visual_to_audio(img_tensor.to(device))

            # Save attention visualization
            if model._last_attention_info:
                visualize_attention(
                    img_tensor.squeeze(0), model._last_attention_info,
                    os.path.join(output_dir, f'attention_{letter}.png'),
                )

            if LIBROSA_AVAILABLE:
                pred_audio = librosa.feature.inverse.mfcc_to_audio(pred_mfcc.squeeze(0).cpu().detach().numpy())
                pred_audio = np.clip(pred_audio, -1.0, 1.0).astype(np.float32)
                sf.write(os.path.join(output_dir, f'pred_audio_{letter}.wav'), pred_audio, 16000, subtype='FLOAT')
            else:
                np.save(os.path.join(output_dir, f'pred_audio_{letter}.npy'), pred_mfcc.cpu().detach().numpy())
            print(f"Generated audio for visual-only input {letter}")

        elif mask == 'black':
            # Audio-only input: Generate visual
            with torch.no_grad():
                pred_img = model.audio_to_visual(mfcc.unsqueeze(0).to(device))
            pred_img_cpu = pred_img.cpu()
            pred_img_pil = Image.fromarray((pred_img_cpu.squeeze().clamp(0, 1).detach().numpy() * 255).astype(np.uint8))
            pred_img_pil.save(os.path.join(output_dir, f'pred_img_{letter}.png'))
            print(f"Generated image for audio-only input {letter}")

            score = nn.MSELoss()(pred_img_cpu, img_tensor).item()
            scores.append(score)
            print(f"MSE to original image: {score:.4f}")

        else:
            print(f"Full video input for {letter}; no generation needed (both modalities present).")

    if scores:
        avg_score = np.mean(scores)
        print(f"Average MSE Score: {avg_score:.4f}")
        plt.figure()
        plt.bar(range(len(scores)), scores)
        plt.xlabel('Test Sample')
        plt.ylabel('MSE')
        plt.savefig(os.path.join(output_dir, 'test_scores.png'))
        plt.close()
    print(f"Test results saved in {output_dir}")

# --- Visualization (run on dataset, save attention paths) ---

def visualize_model(model_dir, data_dir, output_dir='visualizations',
                    n_glimpses=10, patch_size=12, n_scales=1, device='auto'):
    device = _resolve_device(device)
    print(f"Visualizing on: {device}")

    os.makedirs(output_dir, exist_ok=True)

    ckpt = torch.load(os.path.join(model_dir, 'model_final.pth'), map_location=device)
    if isinstance(ckpt, dict) and 'model' in ckpt:
        n_glimpses = ckpt.get('n_glimpses', n_glimpses)
        patch_size = ckpt.get('patch_size', patch_size)
        n_scales = ckpt.get('n_scales', n_scales)
        state_dict = ckpt['model']
    else:
        state_dict = ckpt

    model = BidirectionalModel(
        n_glimpses=n_glimpses, patch_size=patch_size, n_scales=n_scales,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    dataset = LetterDataset(data_dir)
    for i in range(len(dataset)):
        img, mfcc, letter = dataset[i]
        img = img.unsqueeze(0).to(device)

        with torch.no_grad():
            model.visual_to_audio(img)

        if model._last_attention_info:
            visualize_attention(
                img.squeeze(0), model._last_attention_info,
                os.path.join(output_dir, f'attention_{letter}.png'),
            )
            print(f"Saved attention visualization for '{letter}'")

    print(f"Visualizations saved in {output_dir}")

# --- Test Data Generation (from clean videos + lossless WAVs) ---

def generate_test_data(letters, clean_video_dir, output_dir):
    if not MOVIEPY_AVAILABLE:
        raise RuntimeError("MoviePy required for test data generation.")

    os.makedirs(output_dir, exist_ok=True)

    # Split letters into 3 groups: silent, black-screen, full
    n = len(letters)
    third = n // 3
    silent_letters = letters[:third]
    black_letters = letters[third:2*third]
    full_letters = letters[2*third:]

    metadata = {}

    for letter in silent_letters:
        src = os.path.join(clean_video_dir, f'video_{letter}.mp4')
        dst = os.path.join(output_dir, f'video_{letter}.mp4')
        clip = mpy.VideoFileClip(src)
        clip.without_audio().write_videofile(dst, fps=1, codec='libx264')
        clip.close()
        # Silent: zero-length WAV
        wav_path = os.path.join(output_dir, f'audio_{letter}.wav')
        sr = 22050
        torchaudio.save(wav_path, torch.zeros(1, sr), sr)  # 1s of silence
        metadata[letter] = {'video': dst, 'audio': wav_path, 'mask': 'silent'}

    for letter in black_letters:
        src = os.path.join(clean_video_dir, f'video_{letter}.mp4')
        dst = os.path.join(output_dir, f'video_{letter}.mp4')
        clip = mpy.VideoFileClip(src)
        black_frame = np.zeros_like(clip.get_frame(0))
        black_clip = mpy.VideoClip(lambda t: black_frame, duration=clip.duration)
        black_clip = black_clip.set_audio(clip.audio)
        black_clip.write_videofile(dst, fps=1, codec='libx264', audio_codec='aac')
        clip.close()
        # Copy clean WAV (lossless audio source)
        clean_wav = os.path.join(clean_video_dir, f'audio_{letter}.wav')
        wav_path = os.path.join(output_dir, f'audio_{letter}.wav')
        shutil.copy2(clean_wav, wav_path)
        metadata[letter] = {'video': dst, 'audio': wav_path, 'mask': 'black'}

    for letter in full_letters:
        src = os.path.join(clean_video_dir, f'video_{letter}.mp4')
        dst = os.path.join(output_dir, f'video_{letter}.mp4')
        clip = mpy.VideoFileClip(src)
        clip.write_videofile(dst, fps=1, codec='libx264', audio_codec='aac')
        clip.close()
        clean_wav = os.path.join(clean_video_dir, f'audio_{letter}.wav')
        wav_path = os.path.join(output_dir, f'audio_{letter}.wav')
        shutil.copy2(clean_wav, wav_path)
        metadata[letter] = {'video': dst, 'audio': wav_path, 'mask': 'none'}

    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f)

    print(f"Test data generated in {output_dir}: "
          f"{len(silent_letters)} silent, {len(black_letters)} black, {len(full_letters)} full")

# --- Main CLI ---

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multimodal Letter Training Pipeline')
    subparsers = parser.add_subparsers(dest='command', required=True)

    # Generate Clean Video
    clean_parser = subparsers.add_parser('generate_clean')
    clean_parser.add_argument('--letters', default='A-Z', help='Letters range, e.g., A-Z')
    clean_parser.add_argument('--output_dir', default='data/clean_video')
    clean_parser.add_argument('--use_real_tts', action='store_true', help='Use gTTS for realistic audio')

    # Generate Dataset (noisy video)
    gen_parser = subparsers.add_parser('generate')
    gen_parser.add_argument('--letters', default='A-Z', help='Letters range, e.g., A-Z')
    gen_parser.add_argument('--noise_level', type=float, default=0.01)
    gen_parser.add_argument('--num_variants', type=int, default=1, help='Number of noisy variants per letter')
    gen_parser.add_argument('--output_dir', default='data/letters')
    gen_parser.add_argument('--clean_video_dir', required=True, help='Path to clean video dir')

    # Train
    train_parser = subparsers.add_parser('train')
    train_parser.add_argument('--data_dir', required=True, help='Path to generated dataset')
    train_parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train')
    train_parser.add_argument('--resume', default=None, help='Path to checkpoint for resume')
    train_parser.add_argument('--save_dir', default='models')
    train_parser.add_argument('--checkpoint_interval', type=int, default=10)
    train_parser.add_argument('--n_glimpses', type=int, default=10, help='Number of attention glimpses per image')
    train_parser.add_argument('--patch_size', type=int, default=12, help='Glimpse patch size (smaller = tighter FoV)')
    train_parser.add_argument('--n_scales', type=int, default=1, help='Number of glimpse resolution scales (1=foveal only)')
    train_parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'], help='Device: auto (GPU if available), cpu, or cuda')
    train_parser.add_argument('--mode', default='vision', choices=['vision', 'multimodal'], help='Training mode: vision (visual attention only) or multimodal (V+A)')

    # Generate Test Data
    gentest_parser = subparsers.add_parser('generate_test')
    gentest_parser.add_argument('--letters', default='A-Z', help='Letters range, e.g., A-Z')
    gentest_parser.add_argument('--clean_video_dir', required=True, help='Path to clean video dir')
    gentest_parser.add_argument('--output_dir', default='data/test_videos')

    # Test
    test_parser = subparsers.add_parser('test')
    test_parser.add_argument('--model_dir', required=True, help='Path to saved model')
    test_parser.add_argument('--test_data_dir', required=True, help='Path to test videos (craft silent or black for partial input)')
    test_parser.add_argument('--output_dir', default='results')
    test_parser.add_argument('--n_glimpses', type=int, default=10, help='Number of attention glimpses (overridden by checkpoint)')
    test_parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'], help='Device: auto (GPU if available), cpu, or cuda')

    # Visualize
    viz_parser = subparsers.add_parser('visualize')
    viz_parser.add_argument('--model_dir', required=True, help='Path to saved model')
    viz_parser.add_argument('--data_dir', required=True, help='Path to dataset')
    viz_parser.add_argument('--output_dir', default='visualizations')
    viz_parser.add_argument('--n_glimpses', type=int, default=10, help='Number of attention glimpses (overridden by checkpoint)')
    viz_parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'], help='Device: auto (GPU if available), cpu, or cuda')

    args = parser.parse_args()

    if args.command == 'generate_clean':
        if args.letters == 'A-Z':
            letters = [chr(i) for i in range(65, 91)]
        else:
            letters = list(args.letters.upper())
        generate_clean_video(letters, args.output_dir, args.use_real_tts)

    elif args.command == 'generate':
        if args.letters == 'A-Z':
            letters = [chr(i) for i in range(65, 91)]
        else:
            letters = list(args.letters.upper())
        generate_dataset(letters, args.output_dir, args.noise_level, clean_video_dir=args.clean_video_dir, num_variants=args.num_variants)

    elif args.command == 'generate_test':
        if args.letters == 'A-Z':
            letters = [chr(i) for i in range(65, 91)]
        else:
            letters = list(args.letters.upper())
        generate_test_data(letters, args.clean_video_dir, args.output_dir)

    elif args.command == 'train':
        train_model(args.data_dir, args.epochs, args.resume, args.save_dir,
                    args.checkpoint_interval, n_glimpses=args.n_glimpses,
                    patch_size=args.patch_size, n_scales=args.n_scales,
                    device=args.device, mode=args.mode)

    elif args.command == 'test':
        test_model(args.model_dir, args.test_data_dir, args.output_dir,
                   n_glimpses=args.n_glimpses, device=args.device)

    elif args.command == 'visualize':
        visualize_model(args.model_dir, args.data_dir, args.output_dir,
                        n_glimpses=args.n_glimpses, device=args.device)
