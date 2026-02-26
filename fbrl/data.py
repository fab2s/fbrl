import torch
import numpy as np
import os
import json
import time
from torch.utils.data import Dataset
from PIL import Image, ImageDraw, ImageFont


# --- Top 200 English Bigrams (Norvig / Google Books corpus) ---
# Last two entries swapped from corpus order ("rk","ys") to ("ju","ze")
# so that all 26 letters appear at least once across the bigram set.
BIGRAMS_200 = [
    "th", "he", "in", "er", "an", "re", "on", "at", "en", "nd",
    "ti", "es", "or", "te", "of", "ed", "is", "it", "al", "ar",
    "st", "to", "nt", "ng", "se", "ha", "as", "ou", "io", "le",
    "ve", "co", "me", "de", "hi", "ri", "ro", "ic", "ne", "ea",
    "ra", "ce", "li", "ch", "ll", "be", "ma", "si", "om", "ur",
    "ca", "el", "ta", "la", "ns", "di", "fo", "ho", "pe", "ec",
    "pr", "no", "ct", "us", "ac", "ot", "il", "tr", "ly", "nc",
    "et", "ut", "ss", "so", "rs", "un", "lo", "wa", "ge", "ie",
    "wh", "ee", "wi", "em", "ad", "ol", "rt", "po", "we", "na",
    "ul", "ni", "ts", "mo", "ow", "pa", "im", "mi", "ai", "sh",
    "ir", "su", "id", "os", "iv", "ia", "am", "fi", "ci", "vi",
    "pl", "ig", "tu", "ev", "ld", "ry", "mp", "fe", "bl", "ab",
    "gh", "ty", "op", "wo", "sa", "ay", "ex", "ke", "fr", "oo",
    "av", "ag", "if", "ap", "gr", "od", "bo", "sp", "rd", "do",
    "uc", "bu", "ei", "ov", "by", "rm", "ep", "tt", "oc", "fa",
    "ef", "cu", "rn", "sc", "gi", "da", "yo", "cr", "cl", "du",
    "ga", "qu", "ue", "ff", "ba", "ey", "ls", "va", "um", "pp",
    "ua", "up", "lu", "go", "ht", "ru", "ug", "ds", "lt", "pi",
    "rc", "rr", "eg", "au", "ck", "ew", "mu", "br", "bi", "pt",
    "ak", "pu", "ui", "rg", "ib", "tl", "ny", "ki", "ju", "ze",
]


# --- Font Registry ---

# Short name -> TTF path (None = Pillow built-in default)
FONT_REGISTRY = {
    'default':              None,
    'dejavu-serif':         '/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf',
    'dejavu-serif-bold':    '/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf',
    'dejavu-sans':          '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
    'dejavu-sans-bold':     '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
    'liberation-serif':     '/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf',
    'liberation-sans':      '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf',
    'liberation-sans-bold': '/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf',
    'liberation-mono':      '/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf',
    'liberation-mono-bold': '/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf',
    'liberation-narrow':    '/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Regular.ttf',
}


def discover_fonts(font_spec='all'):
    """Resolve font spec to list of (short_name, ttf_path|None) tuples.

    font_spec: 'all', 'default', or comma-separated names like 'dejavu-sans,liberation-mono'
    Probes filesystem; skips fonts whose TTF files are missing.
    """
    if font_spec == 'default':
        return [('default', None)]

    if font_spec == 'all':
        names = list(FONT_REGISTRY.keys())
    else:
        names = [n.strip() for n in font_spec.split(',')]

    found = []
    for name in names:
        if name not in FONT_REGISTRY:
            print(f"  Warning: unknown font '{name}', skipping")
            continue
        path = FONT_REGISTRY[name]
        if path is not None and not os.path.isfile(path):
            print(f"  Warning: font file not found for '{name}' ({path}), skipping")
            continue
        found.append((name, path))

    if not found:
        print("  No fonts found, falling back to default")
        return [('default', None)]
    return found


def load_font(font_path, size):
    """Load a PIL ImageFont — default or from TTF path."""
    if font_path is None:
        return ImageFont.load_default(size=size)
    return ImageFont.truetype(font_path, size=size)


# --- Data Generation ---

def generate_dataset(letters, output_dir, noise_level=0.01, num_variants=1,
                     font_spec='all'):
    os.makedirs(output_dir, exist_ok=True)
    fonts = discover_fonts(font_spec)
    print(f"Generating with {len(fonts)} font(s): {', '.join(n for n, _ in fonts)}")

    metadata = {}
    for font_name, font_path in fonts:
        font = load_font(font_path, size=60)
        for letter in letters:
            # Render clean image
            img = Image.new('L', (128, 128), color=0)
            draw = ImageDraw.Draw(img)
            bbox = draw.textbbox((0, 0), letter, font=font)
            x = (128 - bbox[2] - bbox[0]) / 2
            y = (128 - bbox[3] - bbox[1]) / 2
            draw.text((x, y), letter, fill=255, font=font)

            clean_path = os.path.join(output_dir, f'clean_{letter}_{font_name}.png')
            img.save(clean_path)

            img_array = np.array(img) / 255.0

            for v in range(num_variants):
                noisy = img_array + np.random.normal(0, noise_level, img_array.shape)
                noisy = np.clip(noisy, 0, 1)
                noisy_img = Image.fromarray((noisy * 255).astype(np.uint8))
                img_path = os.path.join(output_dir, f'img_{letter}_{font_name}_{v}.png')
                noisy_img.save(img_path)
                key = f'{letter}_{font_name}_{v}'
                metadata[key] = {
                    'image': img_path,
                    'clean': clean_path,
                    'letter': letter.upper(),
                    'case': 'upper' if letter.isupper() else 'lower',
                    'font': font_name,
                }

    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f)
    print(f"Dataset generated in {output_dir}: {len(metadata)} samples "
          f"({len(fonts)} fonts x {len(letters)} letters x {num_variants} variants)")


def generate_test(letters, output_dir, font_spec='all'):
    os.makedirs(output_dir, exist_ok=True)
    fonts = discover_fonts(font_spec)
    print(f"Generating test with {len(fonts)} font(s): {', '.join(n for n, _ in fonts)}")

    metadata = {}
    for font_name, font_path in fonts:
        font = load_font(font_path, size=60)
        for letter in letters:
            img = Image.new('L', (128, 128), color=0)
            draw = ImageDraw.Draw(img)
            bbox = draw.textbbox((0, 0), letter, font=font)
            x = (128 - bbox[2] - bbox[0]) / 2
            y = (128 - bbox[3] - bbox[1]) / 2
            draw.text((x, y), letter, fill=255, font=font)

            img_path = os.path.join(output_dir, f'img_{letter}_{font_name}.png')
            img.save(img_path)
            key = f'{letter}_{font_name}'
            metadata[key] = {
                'image': img_path,
                'letter': letter.upper(),
                'case': 'upper' if letter.isupper() else 'lower',
                'font': font_name,
            }

    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f)
    print(f"Test data generated in {output_dir}: {len(metadata)} samples "
          f"({len(fonts)} fonts x {len(letters)} letters)")


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
        self.fonts = []

        # Cache clean images (one per letter+case+font, shared across variants)
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
            self.fonts.append(entry.get('font', 'default'))

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

        # Build partner index: for each sample, find clean image of same letter+font, opposite case
        clean_by_key = {}
        for i, (letter, case, font) in enumerate(
                zip(self.letters, self.cases, self.fonts)):
            key = (letter, case, font)
            if key not in clean_by_key:
                clean_by_key[key] = self.clean_images[i]

        self.partner_clean = []
        self.has_partners = True
        for letter, case, font in zip(self.letters, self.cases, self.fonts):
            opposite = 'lower' if case == 'upper' else 'upper'
            partner = clean_by_key.get((letter, opposite, font))
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
                self.fonts[idx], self.partner_clean[idx])


# --- Bigram Data Generation ---

def generate_bigram_dataset(output_dir, noise_level=0.01, num_variants=1,
                            font_spec='default'):
    """Render 192x128 images with natural bigram strings.

    The two-character string is rendered with natural kerning (as real text),
    centered on a 192x128 canvas. This is much tighter than the old 256x128
    layout where each letter sat in its own 128px half.
    Uses BIGRAMS_200 as the bigram set.
    """
    os.makedirs(output_dir, exist_ok=True)
    fonts = discover_fonts(font_spec)
    print(f"Generating bigrams with {len(fonts)} font(s): {', '.join(n for n, _ in fonts)}")

    metadata = {}
    for font_name, font_path in fonts:
        font = load_font(font_path, size=60)
        for bigram in BIGRAMS_200:
            letter1, letter2 = bigram[0], bigram[1]

            # Render clean 192x128 image: bigram as a natural 2-char string, centered
            img = Image.new('L', (192, 128), color=0)
            draw = ImageDraw.Draw(img)
            bbox = draw.textbbox((0, 0), bigram, font=font)
            x = (192 - bbox[2] - bbox[0]) / 2
            y = (128 - bbox[3] - bbox[1]) / 2
            draw.text((x, y), bigram, fill=255, font=font)

            clean_path = os.path.join(output_dir, f'clean_{bigram}_{font_name}.png')
            img.save(clean_path)

            img_array = np.array(img) / 255.0

            for v in range(num_variants):
                noisy = img_array + np.random.normal(0, noise_level, img_array.shape)
                noisy = np.clip(noisy, 0, 1)
                noisy_img = Image.fromarray((noisy * 255).astype(np.uint8))
                img_path = os.path.join(output_dir, f'img_{bigram}_{font_name}_{v}.png')
                noisy_img.save(img_path)
                key = f'{bigram}_{font_name}_{v}'
                metadata[key] = {
                    'image': img_path,
                    'clean': clean_path,
                    'bigram': bigram,
                    'letter1': letter1,
                    'letter2': letter2,
                    'font': font_name,
                }

    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f)
    print(f"Bigram dataset generated in {output_dir}: {len(metadata)} samples "
          f"({len(fonts)} fonts x {len(BIGRAMS_200)} bigrams x {num_variants} variants)")


def generate_bigram_test(output_dir, font_spec='default'):
    """Generate clean (no noise) bigram test set — one image per bigram per font."""
    os.makedirs(output_dir, exist_ok=True)
    fonts = discover_fonts(font_spec)
    print(f"Generating bigram test with {len(fonts)} font(s): {', '.join(n for n, _ in fonts)}")

    metadata = {}
    for font_name, font_path in fonts:
        font = load_font(font_path, size=60)
        for bigram in BIGRAMS_200:
            letter1, letter2 = bigram[0], bigram[1]

            img = Image.new('L', (192, 128), color=0)
            draw = ImageDraw.Draw(img)
            bbox = draw.textbbox((0, 0), bigram, font=font)
            x = (192 - bbox[2] - bbox[0]) / 2
            y = (128 - bbox[3] - bbox[1]) / 2
            draw.text((x, y), bigram, fill=255, font=font)

            img_path = os.path.join(output_dir, f'img_{bigram}_{font_name}.png')
            img.save(img_path)
            key = f'{bigram}_{font_name}'
            metadata[key] = {
                'image': img_path,
                'bigram': bigram,
                'letter1': letter1,
                'letter2': letter2,
                'font': font_name,
            }

    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f)
    print(f"Bigram test data generated in {output_dir}: {len(metadata)} samples "
          f"({len(fonts)} fonts x {len(BIGRAMS_200)} bigrams)")


# --- Bigram Dataset ---

class BigramDataset(Dataset):
    """Loads bigram images. Returns (img, clean, letter1, letter2, bigram, font)."""
    def __init__(self, data_dir):
        with open(os.path.join(data_dir, 'metadata.json'), 'r') as f:
            metadata = json.load(f)

        print(f"Loading {len(metadata)} bigram samples into memory...", end=' ', flush=True)
        t0 = time.time()
        self.images = []
        self.clean_images = []
        self.letter1s = []
        self.letter2s = []
        self.bigrams = []
        self.fonts = []

        # Cache clean images (one per bigram+font, shared across variants)
        clean_cache = {}

        for key in metadata:
            entry = metadata[key]
            img = Image.open(entry['image']).convert('L')
            img_tensor = torch.tensor(
                np.array(img) / 255.0, dtype=torch.float32,
            ).unsqueeze(0)  # (1, H, W) = (1, 128, 256)
            self.images.append(img_tensor)
            self.letter1s.append(entry['letter1'])
            self.letter2s.append(entry['letter2'])
            self.bigrams.append(entry['bigram'])
            self.fonts.append(entry.get('font', 'default'))

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

        print(f"done ({time.time() - t0:.1f}s)")

    def __len__(self):
        return len(self.bigrams)

    def __getitem__(self, idx):
        return (self.images[idx], self.clean_images[idx],
                self.letter1s[idx], self.letter2s[idx],
                self.bigrams[idx], self.fonts[idx])
