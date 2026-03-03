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


# --- Common 4-Letter English Words (200) ---
# Covers all 26 letters. Used for word-level reading experiments.
WORDS_200 = [
    "that", "with", "have", "this", "will", "your", "from", "they",
    "been", "call", "find", "long", "down", "come", "made", "back",
    "only", "good", "year", "them", "some", "time", "very", "when",
    "work", "each", "make", "over", "such", "more", "most", "must",
    "name", "said", "also", "does", "done", "even", "full", "give",
    "gone", "hand", "help", "here", "high", "home", "hope", "into",
    "just", "keep", "kind", "know", "last", "left", "life", "like",
    "line", "live", "look", "many", "much", "next", "once", "open",
    "part", "plan", "play", "read", "rest", "same", "side", "sign",
    "size", "step", "stop", "sure", "take", "tell", "than", "then",
    "true", "turn", "upon", "used", "want", "were", "what", "word",
    "zero", "able", "area", "away", "best", "body", "book", "both",
    "came", "case", "city", "dark", "data", "date", "days", "deal",
    "deep", "door", "draw", "drop", "drug", "duty", "east", "edge",
    "else", "eyes", "face", "fact", "fail", "fair", "fall", "farm",
    "fast", "fear", "feel", "feet", "fill", "film", "fire", "firm",
    "fish", "five", "flat", "flow", "food", "foot", "form", "four",
    "free", "fund", "gain", "game", "girl", "gold", "grew", "grow",
    "gulf", "hair", "half", "hall", "hard", "hate", "head", "hear",
    "heat", "held", "hide", "hill", "hold", "hole", "hour", "huge",
    "idea", "iron", "item", "jack", "join", "jump", "jury", "keen",
    "king", "lack", "laid", "lake", "land", "late", "lead", "less",
    "link", "list", "loan", "lock", "lord", "lose", "love", "luck",
    "mark", "mass", "meal", "mile", "mind", "miss", "mode", "move",
    "near", "need", "news", "nine", "nose", "note", "quiz", "zone",
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
    def __init__(self, data_dir, case_filter=None, font_filter=None):
        with open(os.path.join(data_dir, 'metadata.json'), 'r') as f:
            metadata = json.load(f)

        # Filter by case if requested
        if case_filter == 'upper':
            metadata = {k: v for k, v in metadata.items()
                        if v.get('case', 'upper') == 'upper'}
        elif case_filter == 'lower':
            metadata = {k: v for k, v in metadata.items()
                        if v.get('case', 'upper') == 'lower'}

        # Filter by font if requested (comma-separated names)
        if font_filter:
            allowed = {f.strip() for f in font_filter.split(',')}
            metadata = {k: v for k, v in metadata.items()
                        if v.get('font', 'default') in allowed}

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
    """Render 128x128 images with natural bigram strings.

    The two-character string is rendered with natural kerning (as real text),
    centered on a 128x128 canvas. Tighter canvas eliminates dead space
    (widest bigram text ~96px on 128px canvas).
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

            # Render clean 128x128 image: bigram as a natural 2-char string, centered
            img = Image.new('L', (128, 128), color=0)
            draw = ImageDraw.Draw(img)
            bbox = draw.textbbox((0, 0), bigram, font=font)
            x = (128 - bbox[2] - bbox[0]) / 2
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
    """Generate clean (no noise) bigram test set — one image per bigram per font.

    128x128 canvas with centered text.
    """
    os.makedirs(output_dir, exist_ok=True)
    fonts = discover_fonts(font_spec)
    print(f"Generating bigram test with {len(fonts)} font(s): {', '.join(n for n, _ in fonts)}")

    metadata = {}
    for font_name, font_path in fonts:
        font = load_font(font_path, size=60)
        for bigram in BIGRAMS_200:
            letter1, letter2 = bigram[0], bigram[1]

            img = Image.new('L', (128, 128), color=0)
            draw = ImageDraw.Draw(img)
            bbox = draw.textbbox((0, 0), bigram, font=font)
            x = (128 - bbox[2] - bbox[0]) / 2
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
            ).unsqueeze(0)  # (1, H, W) = (1, 128, 128)
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


# --- Isolation Letter Dataset (128x128 single letters for word isolation testing) ---

class IsolationLetterDataset:
    """Loads 128x128 single-letter clean images for isolation testing.

    Lightweight lookup by (letter_index, font) -> random variant tensor.
    Uses existing letter data directory (from `make generate-letters`).
    Only loads lowercase letters (a-z) since word training uses lowercase.
    """
    def __init__(self, data_dir):
        with open(os.path.join(data_dir, 'metadata.json'), 'r') as f:
            metadata = json.load(f)

        # by_letter_font[(letter_idx, font)] -> list of (1, 128, 128) tensors
        self.by_letter_font = {}
        self.fonts = set()
        n_loaded = 0

        for key, entry in metadata.items():
            letter = entry.get('letter', '')
            case = entry.get('case', 'upper')
            font = entry.get('font', 'default')
            if case != 'lower':
                continue
            letter_idx = ord(letter.lower()) - ord('a')
            if letter_idx < 0 or letter_idx >= 26:
                continue

            # Prefer clean image if available
            img_path = entry.get('clean', entry['image'])
            img = Image.open(img_path).convert('L')
            tensor = torch.tensor(
                np.array(img) / 255.0, dtype=torch.float32,
            ).unsqueeze(0)  # (1, 128, 128)

            lf_key = (letter_idx, font)
            if lf_key not in self.by_letter_font:
                self.by_letter_font[lf_key] = []
            self.by_letter_font[lf_key].append(tensor)
            self.fonts.add(font)
            n_loaded += 1

        self.font_list = sorted(self.fonts)
        print(f"IsolationLetterDataset: {n_loaded} images, "
              f"{len(self.fonts)} font(s), {len(self.by_letter_font)} (letter, font) combos")

    def get_image(self, letter_idx, font='default'):
        """Return a clean image tensor for the given letter index and font.

        Args:
            letter_idx: 0-25 (a-z)
            font: font name string
        Returns:
            (1, 128, 128) tensor
        """
        variants = self.by_letter_font.get((letter_idx, font))
        if variants is None:
            # Fallback: try any font for this letter
            for f in self.fonts:
                variants = self.by_letter_font.get((letter_idx, f))
                if variants:
                    break
        if variants is None:
            # Last resort: return zeros
            return torch.zeros(1, 128, 128)
        idx = torch.randint(0, len(variants), (1,)).item()
        return variants[idx]


# --- Word Data Generation (4-letter words, 256x128 canvas) ---

def generate_word_dataset(output_dir, noise_level=0.01, num_variants=1,
                           font_spec='default'):
    """Render 256x128 images with 4-letter words.

    Wide canvas (256x128) forces genuine saccades — holistic reading is
    geometrically impossible with a 12x12 foveal window.
    Uses WORDS_200 as the word set.
    """
    os.makedirs(output_dir, exist_ok=True)
    fonts = discover_fonts(font_spec)
    print(f"Generating words with {len(fonts)} font(s): {', '.join(n for n, _ in fonts)}")

    metadata = {}
    for font_name, font_path in fonts:
        font = load_font(font_path, size=60)
        for word in WORDS_200:
            # Render clean 256x128 image: word centered
            img = Image.new('L', (256, 128), color=0)
            draw = ImageDraw.Draw(img)
            bbox = draw.textbbox((0, 0), word, font=font)
            x = (256 - bbox[2] - bbox[0]) / 2
            y = (128 - bbox[3] - bbox[1]) / 2
            draw.text((x, y), word, fill=255, font=font)

            clean_path = os.path.join(output_dir, f'clean_{word}_{font_name}.png')
            img.save(clean_path)

            img_array = np.array(img) / 255.0

            for v in range(num_variants):
                noisy = img_array + np.random.normal(0, noise_level, img_array.shape)
                noisy = np.clip(noisy, 0, 1)
                noisy_img = Image.fromarray((noisy * 255).astype(np.uint8))
                img_path = os.path.join(output_dir, f'img_{word}_{font_name}_{v}.png')
                noisy_img.save(img_path)
                key = f'{word}_{font_name}_{v}'
                metadata[key] = {
                    'image': img_path,
                    'clean': clean_path,
                    'word': word,
                    'letter1': word[0],
                    'letter2': word[1],
                    'letter3': word[2],
                    'letter4': word[3],
                    'font': font_name,
                }

    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f)
    print(f"Word dataset generated in {output_dir}: {len(metadata)} samples "
          f"({len(fonts)} fonts x {len(WORDS_200)} words x {num_variants} variants)")


def generate_word_test(output_dir, font_spec='default'):
    """Generate clean (no noise) word test set — one image per word per font.

    256x128 canvas with centered text.
    """
    os.makedirs(output_dir, exist_ok=True)
    fonts = discover_fonts(font_spec)
    print(f"Generating word test with {len(fonts)} font(s): {', '.join(n for n, _ in fonts)}")

    metadata = {}
    for font_name, font_path in fonts:
        font = load_font(font_path, size=60)
        for word in WORDS_200:
            img = Image.new('L', (256, 128), color=0)
            draw = ImageDraw.Draw(img)
            bbox = draw.textbbox((0, 0), word, font=font)
            x = (256 - bbox[2] - bbox[0]) / 2
            y = (128 - bbox[3] - bbox[1]) / 2
            draw.text((x, y), word, fill=255, font=font)

            img_path = os.path.join(output_dir, f'img_{word}_{font_name}.png')
            img.save(img_path)
            key = f'{word}_{font_name}'
            metadata[key] = {
                'image': img_path,
                'word': word,
                'letter1': word[0],
                'letter2': word[1],
                'letter3': word[2],
                'letter4': word[3],
                'font': font_name,
            }

    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f)
    print(f"Word test data generated in {output_dir}: {len(metadata)} samples "
          f"({len(fonts)} fonts x {len(WORDS_200)} words)")


# --- Word Dataset ---

class WordDataset(Dataset):
    """Loads word images. Returns (img, clean, letter1, letter2, letter3, letter4, word, font)."""
    def __init__(self, data_dir):
        with open(os.path.join(data_dir, 'metadata.json'), 'r') as f:
            metadata = json.load(f)

        print(f"Loading {len(metadata)} word samples into memory...", end=' ', flush=True)
        t0 = time.time()
        self.images = []
        self.clean_images = []
        self.letter1s = []
        self.letter2s = []
        self.letter3s = []
        self.letter4s = []
        self.words = []
        self.fonts = []

        clean_cache = {}

        for key in metadata:
            entry = metadata[key]
            img = Image.open(entry['image']).convert('L')
            img_tensor = torch.tensor(
                np.array(img) / 255.0, dtype=torch.float32,
            ).unsqueeze(0)  # (1, 128, 256)
            self.images.append(img_tensor)
            self.letter1s.append(entry['letter1'])
            self.letter2s.append(entry['letter2'])
            self.letter3s.append(entry['letter3'])
            self.letter4s.append(entry['letter4'])
            self.words.append(entry['word'])
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
        return len(self.words)

    def __getitem__(self, idx):
        return (self.images[idx], self.clean_images[idx],
                self.letter1s[idx], self.letter2s[idx],
                self.letter3s[idx], self.letter4s[idx],
                self.words[idx], self.fonts[idx])
