//! Word dataset and batched loader for training.
//!
//! Supports variable-length words (1-4 letters) with per-letter
//! ground-truth centers from proportional glyph spacing.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use flodl::tensor::{cuda_available, Device, Result, Tensor, TensorError};
use serde::Deserialize;

/// One rendered word with metadata.
pub struct WordSample {
    pub image: Tensor,        // [1, H, W] noisy image
    pub clean: Tensor,        // [1, H, W] clean image
    pub word: String,         // the rendered word (with case)
    pub letters: Vec<i64>,    // letter indices 0-25, length = word length
    pub centers: Vec<f64>,    // GT x-center per letter (normalized [-1, 1])
}

/// Stacked mini-batch ready for training.
pub struct WordBatch {
    pub image: Tensor,            // [B, 1, H, W]
    pub clean: Tensor,            // [B, 1, H, W]
    pub word_len: usize,          // uniform within this batch
    pub letter_idx: Vec<Tensor>,  // word_len × [B] int64
    pub centers: Vec<Tensor>,     // word_len × [B] f32
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
    pub fn get_random_image(&self, letter_idx: i64, rng: &mut impl FnMut(usize) -> usize) -> &Tensor {
        let font_idx = rng(self.font_list.len());
        if let Some(indices) = self.by_letter_font.get(&(letter_idx, font_idx)) {
            let vi = rng(indices.len());
            return &self.samples[indices[vi]].image;
        }
        for fi in 0..self.font_list.len() {
            if let Some(indices) = self.by_letter_font.get(&(letter_idx, fi)) {
                let vi = rng(indices.len());
                return &self.samples[indices[vi]].image;
            }
        }
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

/// Legacy fixed centers for backward compat with old 4-position datasets.
const LEGACY_CENTERS: [f64; 4] = [-0.75, -0.25, 0.25, 0.75];

#[derive(Deserialize)]
#[allow(dead_code)]
struct WordMetaEntry {
    image: String,
    #[serde(default)]
    clean: String,
    #[serde(default)]
    word: String,
    // New format: variable-length arrays
    #[serde(default)]
    letters: Vec<String>,
    #[serde(default)]
    centers: Vec<f64>,
    // Legacy format: fixed 4 positions
    #[serde(default)]
    letter1: String,
    #[serde(default)]
    letter2: String,
    #[serde(default)]
    letter3: String,
    #[serde(default)]
    letter4: String,
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

/// Load word dataset from a directory with metadata.json.
///
/// Supports both new format (variable-length `letters`/`centers` arrays)
/// and legacy format (`letter1`..`letter4` with fixed grid centers).
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

        let clean_src = if entry.clean.is_empty() { &entry.image } else { &entry.clean };
        let clean_path = resolve_data_path(dir_path, clean_src);
        let clean_tensor = if let Some(cached) = clean_cache.get(&clean_path) {
            cached.clone()
        } else {
            let t = load_gray_png(&clean_path)?;
            clean_cache.insert(clean_path, t.clone());
            t
        };

        // Parse letters and centers — new format or legacy
        let (letters, centers) = if !entry.letters.is_empty() {
            // New format
            let letters: Vec<i64> = entry.letters.iter()
                .filter_map(|s| letter_to_idx(s))
                .collect();
            let centers = entry.centers.clone();
            if letters.len() != centers.len() || letters.is_empty() {
                continue; // malformed entry
            }
            (letters, centers)
        } else {
            // Legacy format: letter1..letter4
            let legacy = [&entry.letter1, &entry.letter2, &entry.letter3, &entry.letter4];
            let mut letters = Vec::new();
            let mut centers = Vec::new();
            for (i, s) in legacy.iter().enumerate() {
                if let Some(idx) = letter_to_idx(s) {
                    letters.push(idx);
                    centers.push(LEGACY_CENTERS[i]);
                }
            }
            if letters.is_empty() { continue; }
            (letters, centers)
        };

        samples.push(WordSample {
            image: img_tensor,
            clean: clean_tensor,
            word: entry.word.clone(),
            letters,
            centers,
        });
    }

    if samples.is_empty() {
        return Err(TensorError::new(&format!("no valid word samples in {dir}")));
    }
    eprintln!("Loaded {} word samples from {}", samples.len(), dir);
    Ok(WordDataset { samples })
}

/// Load isolation dataset from a letter data directory.
pub fn load_isolation_dataset(dir: &str) -> Result<IsolationDataset> {
    let dir_path = Path::new(dir);
    let meta_path = dir_path.join("metadata.json");
    let meta_str = fs::read_to_string(&meta_path)
        .map_err(|e| TensorError::new(&format!("read isolation metadata: {e}")))?;

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

    let meta: HashMap<String, LetterMetaEntry> = serde_json::from_str(&meta_str)
        .map_err(|e| TensorError::new(&format!("parse isolation metadata: {e}")))?;

    let mut font_set = std::collections::BTreeSet::new();
    let mut raw_entries: Vec<(i64, String, Tensor)> = Vec::new();

    for entry in meta.values() {
        if entry.case != "lower" { continue; }
        let Some(letter_idx) = letter_to_idx(&entry.letter) else { continue };

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

// --- Batched loader with length-grouped batching ---

pub struct WordLoader<'a> {
    ds: &'a WordDataset,
    batch_size: usize,
    shuffle: bool,
    device: Option<Device>,
    /// Indices grouped by word length (bucket 0 = length 1, etc.)
    buckets: Vec<Vec<usize>>,
    /// Current position within each bucket.
    bucket_pos: Vec<usize>,
    /// Which bucket to serve next (round-robin among non-exhausted).
    next_bucket: usize,
}

impl<'a> WordLoader<'a> {
    pub fn new(ds: &'a WordDataset, batch_size: usize, shuffle: bool) -> Self {
        let mut loader = WordLoader {
            ds, batch_size, shuffle,
            device: None,
            buckets: Vec::new(),
            bucket_pos: Vec::new(),
            next_bucket: 0,
        };
        loader.init_buckets();
        loader
    }

    pub fn set_device(&mut self, device: Device) { self.device = Some(device); }

    fn init_buckets(&mut self) {
        // Group by word length (1-indexed: bucket[0] = 1-letter words, etc.)
        let max_len = self.ds.samples.iter().map(|s| s.letters.len()).max().unwrap_or(0);
        self.buckets = vec![Vec::new(); max_len];
        for (i, s) in self.ds.samples.iter().enumerate() {
            let len = s.letters.len();
            if len > 0 && len <= max_len {
                self.buckets[len - 1].push(i);
            }
        }

        if self.shuffle {
            let mut seed = {
                use std::collections::hash_map::DefaultHasher;
                use std::hash::{Hash, Hasher};
                use std::time::SystemTime;
                let mut h = DefaultHasher::new();
                SystemTime::now().hash(&mut h);
                h.finish()
            };
            for bucket in &mut self.buckets {
                for i in (1..bucket.len()).rev() {
                    seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                    let j = (seed >> 33) as usize % (i + 1);
                    bucket.swap(i, j);
                }
            }
        }

        self.bucket_pos = vec![0; self.buckets.len()];
        self.next_bucket = 0;
    }

    pub fn next_batch(&mut self) -> Result<Option<WordBatch>> {
        // Find next non-exhausted bucket with enough samples (round-robin).
        let n_buckets = self.buckets.len();
        if n_buckets == 0 { return Ok(None); }

        for _ in 0..n_buckets {
            let bi = self.next_bucket % n_buckets;
            self.next_bucket = bi + 1;

            let remaining = self.buckets[bi].len().saturating_sub(self.bucket_pos[bi]);
            if remaining >= self.batch_size {
                let start = self.bucket_pos[bi];
                let end = start + self.batch_size;
                self.bucket_pos[bi] = end;

                let indices = &self.buckets[bi][start..end];
                let word_len = bi + 1; // bucket 0 = 1-letter words
                return self.build_batch(indices, word_len).map(Some);
            }
        }

        Ok(None) // all buckets exhausted
    }

    fn build_batch(&self, indices: &[usize], word_len: usize) -> Result<WordBatch> {
        let b = indices.len();

        let mut images = Vec::with_capacity(b);
        let mut cleans = Vec::with_capacity(b);
        let mut letters_per_pos: Vec<Vec<i64>> = (0..word_len).map(|_| Vec::with_capacity(b)).collect();
        let mut centers_per_pos: Vec<Vec<f32>> = (0..word_len).map(|_| Vec::with_capacity(b)).collect();

        for &idx in indices {
            let s = &self.ds.samples[idx];
            images.push(&s.image);
            cleans.push(&s.clean);
            for pos in 0..word_len {
                letters_per_pos[pos].push(s.letters[pos]);
                centers_per_pos[pos].push(s.centers[pos] as f32);
            }
        }

        let img_batch = Tensor::stack(&images, 0)?;
        let clean_batch = Tensor::stack(&cleans, 0)?;

        let letter_idx: Vec<Tensor> = letters_per_pos.iter()
            .map(|v| Tensor::from_i64(v, &[b as i64], Device::CPU))
            .collect::<Result<_>>()?;

        let centers: Vec<Tensor> = centers_per_pos.iter()
            .map(|v| Tensor::from_f32(v, &[b as i64], Device::CPU))
            .collect::<Result<_>>()?;

        let batch = if let Some(device) = self.device {
            let use_pin = device.is_cuda() && cuda_available();
            let pin = |t: Tensor| -> Result<Tensor> {
                if use_pin { t.pin_memory() } else { Ok(t) }
            };
            WordBatch {
                image: pin(img_batch)?.to_device(device)?,
                clean: pin(clean_batch)?.to_device(device)?,
                word_len,
                letter_idx: letter_idx.into_iter()
                    .map(|t| pin(t)?.to_device(device))
                    .collect::<Result<_>>()?,
                centers: centers.into_iter()
                    .map(|t| pin(t)?.to_device(device))
                    .collect::<Result<_>>()?,
            }
        } else {
            WordBatch { image: img_batch, clean: clean_batch, word_len, letter_idx, centers }
        };

        Ok(batch)
    }

    pub fn reset(&mut self) { self.init_buckets(); }
}
