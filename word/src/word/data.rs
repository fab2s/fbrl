//! Word dataset and batched loader for training.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use flodl::tensor::{cuda_available, Device, Result, Tensor, TensorError};
use serde::Deserialize;

/// One rendered word with metadata.
pub struct WordSample {
    pub image: Tensor,       // [1, 128, 256] noisy image
    pub clean: Tensor,       // [1, 128, 256] clean image
    pub letters: [i64; 4],   // letter indices 0-25 per position
    pub word: String,
}

/// Stacked mini-batch ready for training.
pub struct WordBatch {
    pub image: Tensor,               // [B, 1, 128, 256]
    pub clean: Tensor,               // [B, 1, 128, 256]
    pub letter_idx: [Tensor; 4],     // 4 × [B] int64 — per-position targets
}

/// Holds all word samples in memory.
pub struct WordDataset {
    pub samples: Vec<WordSample>,
}

impl WordDataset {
    pub fn len(&self) -> usize { self.samples.len() }
    pub fn is_empty(&self) -> bool { self.samples.is_empty() }
}

/// Isolation letter sample (128×128 single letter for isolation loss).
pub struct IsolationSample {
    pub image: Tensor,       // [1, 128, 128]
    pub letter_idx: i64,     // 0-25
}

/// Isolation dataset: single-letter images indexed by (letter_idx, font).
/// Supports random font selection matching Python's IsolationLetterDataset.
pub struct IsolationDataset {
    pub samples: Vec<IsolationSample>,
    /// Sample indices grouped by (letter_idx, font_idx) for random font selection.
    by_letter_font: HashMap<(i64, usize), Vec<usize>>,
    /// Sorted font list for random selection.
    pub font_list: Vec<String>,
}

impl IsolationDataset {
    pub fn len(&self) -> usize { self.samples.len() }
    pub fn is_empty(&self) -> bool { self.samples.is_empty() }

    /// Get a random image for the given letter, choosing a random font.
    /// Matches Python's `iso_font = random.choice(font_list); get_image(letter_idx, iso_font)`.
    pub fn get_random_image(&self, letter_idx: i64, rng: &mut impl FnMut(usize) -> usize) -> &Tensor {
        // Pick a random font.
        let font_idx = rng(self.font_list.len());
        if let Some(indices) = self.by_letter_font.get(&(letter_idx, font_idx)) {
            let vi = rng(indices.len());
            return &self.samples[indices[vi]].image;
        }
        // Fallback: try any font for this letter.
        for fi in 0..self.font_list.len() {
            if let Some(indices) = self.by_letter_font.get(&(letter_idx, fi)) {
                let vi = rng(indices.len());
                return &self.samples[indices[vi]].image;
            }
        }
        // Last resort: first sample (should not happen with well-formed data).
        &self.samples[0].image
    }
}

// --- PNG loading ---

/// Load a grayscale PNG as a [1, H, W] float32 tensor in [0, 1].
pub fn load_gray_png(path: &Path) -> Result<Tensor> {
    let file = fs::File::open(path)
        .map_err(|e| TensorError::new(&format!("open {}: {e}", path.display())))?;
    let decoder = png::Decoder::new(file);
    let mut reader = decoder.read_info()
        .map_err(|e| TensorError::new(&format!("decode {}: {e}", path.display())))?;
    let mut buf = vec![0u8; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf)
        .map_err(|e| TensorError::new(&format!("read {}: {e}", path.display())))?;

    let (h, w) = (info.height as usize, info.width as usize);
    let bytes = &buf[..info.buffer_size()];

    let data: Vec<f32> = match info.color_type {
        png::ColorType::Grayscale => {
            bytes.iter().map(|&b| b as f32 / 255.0).collect()
        }
        png::ColorType::GrayscaleAlpha => {
            bytes.chunks(2).map(|px| px[0] as f32 / 255.0).collect()
        }
        png::ColorType::Rgb => {
            bytes.chunks(3).map(|px| {
                (0.299 * px[0] as f32 + 0.587 * px[1] as f32 + 0.114 * px[2] as f32) / 255.0
            }).collect()
        }
        png::ColorType::Rgba => {
            bytes.chunks(4).map(|px| {
                (0.299 * px[0] as f32 + 0.587 * px[1] as f32 + 0.114 * px[2] as f32) / 255.0
            }).collect()
        }
        _ => return Err(TensorError::new(&format!(
            "{}: unsupported color type {:?}", path.display(), info.color_type
        ))),
    };

    Tensor::from_f32(&data, &[1, h as i64, w as i64], Device::CPU)
}

// --- Metadata ---

#[derive(Deserialize)]
#[allow(dead_code)]
struct WordMetaEntry {
    image: String,
    clean: String,
    word: String,
    letter1: String,
    letter2: String,
    letter3: String,
    letter4: String,
    #[serde(default)]
    font: String,
}

#[derive(Deserialize)]
struct LetterMetaEntry {
    image: String,
    #[serde(default)]
    clean: String,
    letter: String,
    #[serde(default)]
    case: String,
    #[serde(default)]
    font: String,
}

fn letter_to_idx(s: &str) -> Option<i64> {
    let ch = s.to_uppercase().as_bytes().first().copied()?;
    if ch.is_ascii_uppercase() {
        Some((ch - b'A') as i64)
    } else {
        None
    }
}

/// Load word dataset from a Python-generated directory with metadata.json.
pub fn load_word_dataset(dir: &str) -> Result<WordDataset> {
    let dir_path = Path::new(dir);
    let meta_path = dir_path.join("metadata.json");
    let meta_str = fs::read_to_string(&meta_path)
        .map_err(|e| TensorError::new(&format!("read metadata: {e}")))?;
    let meta: HashMap<String, WordMetaEntry> = serde_json::from_str(&meta_str)
        .map_err(|e| TensorError::new(&format!("parse metadata: {e}")))?;

    let mut clean_cache: HashMap<PathBuf, Tensor> = HashMap::new();
    let mut samples = Vec::with_capacity(meta.len());

    for entry in meta.values() {
        let img_path = resolve_data_path(dir_path, &entry.image);
        let img_tensor = load_gray_png(&img_path)?;

        let clean_path = resolve_data_path(dir_path, &entry.clean);
        let clean_tensor = if let Some(cached) = clean_cache.get(&clean_path) {
            cached.clone()
        } else {
            let t = load_gray_png(&clean_path)?;
            clean_cache.insert(clean_path, t.clone());
            t
        };

        let l1 = letter_to_idx(&entry.letter1);
        let l2 = letter_to_idx(&entry.letter2);
        let l3 = letter_to_idx(&entry.letter3);
        let l4 = letter_to_idx(&entry.letter4);
        let (Some(l1), Some(l2), Some(l3), Some(l4)) = (l1, l2, l3, l4) else { continue };

        samples.push(WordSample {
            image: img_tensor,
            clean: clean_tensor,
            letters: [l1, l2, l3, l4],
            word: entry.word.clone(),
        });
    }

    if samples.is_empty() {
        return Err(TensorError::new(&format!("no valid word samples in {dir}")));
    }
    eprintln!("Loaded {} word samples from {}", samples.len(), dir);
    Ok(WordDataset { samples })
}

/// Load isolation dataset from a letter data directory.
///
/// Matches Python's IsolationLetterDataset: indexes images by (letter_idx, font)
/// for random font selection during training.
pub fn load_isolation_dataset(dir: &str) -> Result<IsolationDataset> {
    let dir_path = Path::new(dir);
    let meta_path = dir_path.join("metadata.json");
    let meta_str = fs::read_to_string(&meta_path)
        .map_err(|e| TensorError::new(&format!("read isolation metadata: {e}")))?;
    let meta: HashMap<String, LetterMetaEntry> = serde_json::from_str(&meta_str)
        .map_err(|e| TensorError::new(&format!("parse isolation metadata: {e}")))?;

    // Collect all fonts and build (letter_idx, font) → images map.
    let mut font_set = std::collections::BTreeSet::new();
    let mut raw_entries: Vec<(i64, String, Tensor)> = Vec::new();

    for entry in meta.values() {
        // Only lowercase (matching Python).
        if entry.case != "lower" { continue; }

        let Some(letter_idx) = letter_to_idx(&entry.letter) else { continue };

        // Use clean image for isolation (no noise).
        let path = if !entry.clean.is_empty() {
            resolve_data_path(dir_path, &entry.clean)
        } else {
            resolve_data_path(dir_path, &entry.image)
        };

        if !path.exists() { continue; }

        let font = if entry.font.is_empty() { "default".to_string() } else { entry.font.clone() };
        font_set.insert(font.clone());

        let img = load_gray_png(&path)?;
        raw_entries.push((letter_idx, font, img));
    }

    let font_list: Vec<String> = font_set.into_iter().collect();
    let font_idx_map: HashMap<String, usize> = font_list.iter()
        .enumerate().map(|(i, f)| (f.clone(), i)).collect();

    let mut by_letter_font: HashMap<(i64, usize), Vec<usize>> = HashMap::new();
    let mut samples = Vec::new();

    for (letter_idx, font, img) in raw_entries {
        let fi = font_idx_map[&font];
        let sample_idx = samples.len();
        by_letter_font.entry((letter_idx, fi))
            .or_default()
            .push(sample_idx);
        samples.push(IsolationSample { image: img, letter_idx });
    }

    eprintln!("IsolationDataset: {} images, {} (letter, font) combos, {} font(s) from {}",
        samples.len(), by_letter_font.len(), font_list.len(), dir);
    Ok(IsolationDataset { samples, by_letter_font, font_list })
}

/// Resolve a possibly-relative image path from metadata.json.
fn resolve_data_path(dir: &Path, path: &str) -> PathBuf {
    let p = Path::new(path);
    if p.is_absolute() { return p.to_path_buf(); }
    let joined = dir.join(p);
    if joined.exists() { return joined; }
    if let Some(name) = p.file_name() {
        let by_name = dir.join(name);
        if by_name.exists() { return by_name; }
    }
    let mut parent = dir.to_path_buf();
    for _ in 0..5 {
        if let Some(p) = parent.parent() {
            parent = p.to_path_buf();
        } else { break; }
        let candidate = parent.join(path);
        if candidate.exists() { return candidate; }
    }
    joined
}

// --- Batched loader ---

pub struct WordLoader<'a> {
    ds: &'a WordDataset,
    batch_size: usize,
    shuffle: bool,
    device: Option<Device>,
    perm: Vec<usize>,
    pos: usize,
}

impl<'a> WordLoader<'a> {
    pub fn new(ds: &'a WordDataset, batch_size: usize, shuffle: bool) -> Self {
        let mut loader = WordLoader {
            ds, batch_size, shuffle,
            device: None, perm: Vec::new(), pos: 0,
        };
        loader.init_perm();
        loader
    }

    pub fn set_device(&mut self, device: Device) { self.device = Some(device); }

    fn init_perm(&mut self) {
        let n = self.ds.len();
        self.perm = (0..n).collect();
        if self.shuffle {
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};
            use std::time::SystemTime;
            let mut seed = {
                let mut h = DefaultHasher::new();
                SystemTime::now().hash(&mut h);
                h.finish()
            };
            for i in (1..n).rev() {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                let j = (seed >> 33) as usize % (i + 1);
                self.perm.swap(i, j);
            }
        }
        self.pos = 0;
    }

    pub fn next_batch(&mut self) -> Result<Option<WordBatch>> {
        let n = self.ds.len();
        if self.pos >= n { return Ok(None); }
        let end = (self.pos + self.batch_size).min(n);
        if end - self.pos < self.batch_size { return Ok(None); } // drop last

        let indices = &self.perm[self.pos..end];
        self.pos = end;
        let b = indices.len();

        let mut images = Vec::with_capacity(b);
        let mut cleans = Vec::with_capacity(b);
        let mut letters: [Vec<i64>; 4] = [
            Vec::with_capacity(b), Vec::with_capacity(b),
            Vec::with_capacity(b), Vec::with_capacity(b),
        ];

        for &idx in indices {
            let s = &self.ds.samples[idx];
            images.push(&s.image);
            cleans.push(&s.clean);
            for (p, letter_vec) in letters.iter_mut().enumerate() {
                letter_vec.push(s.letters[p]);
            }
        }

        let img_batch = Tensor::stack(&images, 0)?;
        let clean_batch = Tensor::stack(&cleans, 0)?;
        let letter_idx = [
            Tensor::from_i64(&letters[0], &[b as i64], Device::CPU)?,
            Tensor::from_i64(&letters[1], &[b as i64], Device::CPU)?,
            Tensor::from_i64(&letters[2], &[b as i64], Device::CPU)?,
            Tensor::from_i64(&letters[3], &[b as i64], Device::CPU)?,
        ];

        let batch = if let Some(device) = self.device {
            let use_pin = device.is_cuda() && cuda_available();
            let pin = |t: Tensor| -> Result<Tensor> {
                if use_pin { t.pin_memory() } else { Ok(t) }
            };
            WordBatch {
                image: pin(img_batch)?.to_device(device)?,
                clean: pin(clean_batch)?.to_device(device)?,
                letter_idx: [
                    pin(letter_idx[0].clone())?.to_device(device)?,
                    pin(letter_idx[1].clone())?.to_device(device)?,
                    pin(letter_idx[2].clone())?.to_device(device)?,
                    pin(letter_idx[3].clone())?.to_device(device)?,
                ],
            }
        } else {
            WordBatch { image: img_batch, clean: clean_batch, letter_idx }
        };

        Ok(Some(batch))
    }

    pub fn reset(&mut self) { self.init_perm(); }
}
