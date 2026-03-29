//! Letter dataset and batched loader for training.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use flodl::tensor::{cuda_available, Device, Result, Tensor, TensorError};
use serde::Deserialize;

/// One rendered letter with metadata.
pub struct LetterSample {
    pub image: Tensor,           // [1, H, W] noisy image
    pub clean: Tensor,           // [1, H, W] clean image (for attention guide)
    pub letter_idx: i64,         // 0-25 (A=0, B=1, ...)
    pub case_label: f32,         // 0.0=upper, 1.0=lower
    pub partner_clean: Tensor,   // [1, H, W] opposite-case clean image (for recode loss)
    pub font: String,            // font name (for eval/atlas)
}

/// Stacked mini-batch ready for training.
pub struct LetterBatch {
    pub image: Tensor,           // [B, 1, H, W]
    pub clean: Tensor,           // [B, 1, H, W]
    pub letter_idx: Tensor,      // [B] int64
    pub case_label: Tensor,      // [B, 1] float32
    pub partner_clean: Tensor,   // [B, 1, H, W] opposite-case clean images
}

/// Holds all samples in memory.
pub struct LetterDataset {
    pub samples: Vec<LetterSample>,
    pub has_partners: bool,
}

impl LetterDataset {
    pub fn len(&self) -> usize { self.samples.len() }
    pub fn is_empty(&self) -> bool { self.samples.is_empty() }
}

// --- Data loading from Python-generated directories ---

/// Metadata entry from Python's metadata.json.
#[derive(Deserialize)]
struct MetaEntry {
    image: String,
    clean: String,
    letter: String,
    case: String,
    #[serde(default)]
    font: String,
}

/// Load a dataset from a Python-generated data directory.
/// The directory must contain `metadata.json` and referenced PNG files.
pub fn load_letter_dataset(dir: &str) -> Result<LetterDataset> {
    let dir_path = Path::new(dir);
    let meta_path = dir_path.join("metadata.json");
    let meta_str = fs::read_to_string(&meta_path)
        .map_err(|e| TensorError::new(&format!("read metadata: {e}")))?;
    let meta: HashMap<String, MetaEntry> = serde_json::from_str(&meta_str)
        .map_err(|e| TensorError::new(&format!("parse metadata: {e}")))?;

    // Cache clean images (shared across noisy variants).
    let mut clean_cache: HashMap<PathBuf, Tensor> = HashMap::new();

    // First pass: load all samples, build clean-by-identity lookup for partners.
    struct RawSample {
        image: Tensor,
        clean: Tensor,
        letter_idx: i64,
        case_label: f32,
        font: String,
    }
    let mut raw_samples = Vec::with_capacity(meta.len());
    let mut clean_by_id: HashMap<(i64, bool, String), Tensor> = HashMap::new(); // (letter_idx, is_lower, font)

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

        // Letter index: first char uppercase → 0-25.
        let letter = entry.letter.to_uppercase();
        let ch = match letter.as_bytes().first() {
            Some(&b) if b.is_ascii_uppercase() => b,
            _ => continue,
        };
        let letter_idx = (ch - b'A') as i64;
        let is_lower = entry.case == "lower";
        let case_label = if is_lower { 1.0f32 } else { 0.0f32 };

        let font = entry.font.clone();
        clean_by_id.entry((letter_idx, is_lower, font.clone()))
            .or_insert_with(|| clean_tensor.clone());

        raw_samples.push(RawSample { image: img_tensor, clean: clean_tensor, letter_idx, case_label, font });
    }

    // Second pass: resolve partner clean images (opposite case, same letter).
    let mut has_partners = true;
    let mut samples = Vec::with_capacity(raw_samples.len());
    for raw in raw_samples {
        let is_lower = raw.case_label > 0.5;
        let partner_clean = if let Some(p) = clean_by_id.get(&(raw.letter_idx, !is_lower, raw.font.clone())) {
            p.clone()
        } else {
            has_partners = false;
            Tensor::zeros(&raw.clean.shape(), Default::default())?
        };
        samples.push(LetterSample {
            image: raw.image,
            clean: raw.clean,
            letter_idx: raw.letter_idx,
            case_label: raw.case_label,
            partner_clean,
            font: raw.font,
        });
    }

    if samples.is_empty() {
        return Err(TensorError::new(&format!("no valid samples found in {dir}")));
    }

    eprintln!("Loaded {} samples from {}{}", samples.len(), dir,
        if has_partners { "" } else { " (some missing partners)" });
    Ok(LetterDataset { samples, has_partners })
}

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

    // Convert to float32 grayscale.
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

/// Resolve a possibly-relative image path from metadata.json.
fn resolve_data_path(dir: &Path, path: &str) -> PathBuf {
    let p = Path::new(path);
    if p.is_absolute() {
        return p.to_path_buf();
    }
    // Try dir-relative first.
    let joined = dir.join(p);
    if joined.exists() {
        return joined;
    }
    // Try basename only (metadata sometimes stores full paths).
    if let Some(name) = p.file_name() {
        let by_name = dir.join(name);
        if by_name.exists() {
            return by_name;
        }
    }
    // Walk up parent directories.
    let mut parent = dir.to_path_buf();
    for _ in 0..5 {
        if let Some(p) = parent.parent() {
            parent = p.to_path_buf();
        } else {
            break;
        }
        let candidate = parent.join(path);
        if candidate.exists() {
            return candidate;
        }
    }
    // Return dir-relative (will produce a clear error on open).
    joined
}

/// Iterates over a LetterDataset in shuffled batches.
///
/// ```ignore
/// let mut loader = LetterLoader::new(&ds, 32, true);
/// while let Some(batch) = loader.next_batch()? {
///     // ... training step ...
/// }
/// loader.reset(); // new epoch
/// ```
pub struct LetterLoader<'a> {
    ds: &'a LetterDataset,
    batch_size: usize,
    shuffle: bool,
    drop_last: bool,
    device: Option<Device>,
    perm: Vec<usize>,
    pos: usize,
}

impl<'a> LetterLoader<'a> {
    pub fn new(ds: &'a LetterDataset, batch_size: usize, shuffle: bool) -> Self {
        let mut loader = LetterLoader {
            ds,
            batch_size,
            shuffle,
            drop_last: true,
            device: None,
            perm: Vec::new(),
            pos: 0,
        };
        loader.init_perm();
        loader
    }

    /// Set the target device for batch tensors.
    pub fn set_device(&mut self, device: Device) {
        self.device = Some(device);
    }

    fn init_perm(&mut self) {
        let n = self.ds.len();
        self.perm = (0..n).collect();
        if self.shuffle {
            // Simple Fisher-Yates shuffle
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

    /// Advance to the next batch. Returns `Ok(None)` when the epoch is exhausted.
    pub fn next_batch(&mut self) -> Result<Option<LetterBatch>> {
        let n = self.ds.len();
        if self.pos >= n {
            return Ok(None);
        }
        let end = self.pos + self.batch_size;
        if end > n && self.drop_last {
            return Ok(None);
        }
        let end = end.min(n);

        let indices = &self.perm[self.pos..end];
        self.pos = end;
        let b = indices.len();

        // Collect per-sample data.
        let mut images = Vec::with_capacity(b);
        let mut cleans = Vec::with_capacity(b);
        let mut partners = Vec::with_capacity(b);
        let mut letter_data = Vec::with_capacity(b);
        let mut case_data = Vec::with_capacity(b);

        for &idx in indices {
            let s = &self.ds.samples[idx];
            images.push(&s.image);
            cleans.push(&s.clean);
            partners.push(&s.partner_clean);
            letter_data.push(s.letter_idx);
            case_data.push(s.case_label);
        }

        // Stack into batch tensors.
        let img_batch = Tensor::stack(&images, 0)?;
        let clean_batch = Tensor::stack(&cleans, 0)?;
        let partner_batch = Tensor::stack(&partners, 0)?;
        let letter_idx = Tensor::from_i64(&letter_data, &[b as i64], Device::CPU)?;
        let case_label = Tensor::from_f32(&case_data, &[b as i64, 1], Device::CPU)?;

        // Move to target device if set. Pin memory first for async transfer on CUDA.
        let batch = if let Some(device) = self.device {
            let use_pin = device.is_cuda() && cuda_available();
            let pin = |t: Tensor| -> Result<Tensor> {
                if use_pin { t.pin_memory() } else { Ok(t) }
            };
            LetterBatch {
                image: pin(img_batch)?.to_device(device)?,
                clean: pin(clean_batch)?.to_device(device)?,
                partner_clean: pin(partner_batch)?.to_device(device)?,
                letter_idx: pin(letter_idx)?.to_device(device)?,
                case_label: pin(case_label)?.to_device(device)?,
            }
        } else {
            LetterBatch {
                image: img_batch,
                clean: clean_batch,
                partner_clean: partner_batch,
                letter_idx,
                case_label,
            }
        };

        Ok(Some(batch))
    }

    /// Reset for a new epoch.
    pub fn reset(&mut self) {
        self.init_perm();
    }
}

