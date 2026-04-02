//! Letter dataset and batched loader for training.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use flodl::tensor::{cuda_available, Device, Result, Tensor, TensorError};
use serde::Deserialize;

/// One rendered letter with metadata.
#[derive(Clone)]
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
#[derive(Clone)]
pub struct LetterDataset {
    pub samples: Vec<LetterSample>,
    pub has_partners: bool,
}

impl LetterDataset {
    pub fn len(&self) -> usize { self.samples.len() }
    pub fn is_empty(&self) -> bool { self.samples.is_empty() }
}

/// Adapter for flodl's DataLoader with origin noise injection.
///
/// Returns [image, clean, partner_clean, letter_idx, case, origin] per batch.
/// Noise parameters are updated per-epoch via `set_noise()` (atomic, thread-safe).
/// Batch field names match graph inputs: "case" (not "case_label"), "origin".
pub struct LetterBatchAdapter {
    pub dataset: LetterDataset,
    noise_x_bits: std::sync::atomic::AtomicU64,
    noise_y_bits: std::sync::atomic::AtomicU64,
}

// Safety: LetterDataset contains Tensor (internally reference-counted) and primitives.
// AtomicU64 is Send+Sync by definition.
unsafe impl Send for LetterBatchAdapter {}
unsafe impl Sync for LetterBatchAdapter {}

impl LetterBatchAdapter {
    pub fn new(dataset: LetterDataset) -> Self {
        LetterBatchAdapter {
            dataset,
            noise_x_bits: std::sync::atomic::AtomicU64::new(0f64.to_bits()),
            noise_y_bits: std::sync::atomic::AtomicU64::new(0f64.to_bits()),
        }
    }

    /// Update origin noise parameters (called per-epoch from training loop).
    pub fn set_noise(&self, x: f64, y: f64) {
        self.noise_x_bits.store(x.to_bits(), std::sync::atomic::Ordering::Relaxed);
        self.noise_y_bits.store(y.to_bits(), std::sync::atomic::Ordering::Relaxed);
    }

    fn noise_x(&self) -> f64 {
        f64::from_bits(self.noise_x_bits.load(std::sync::atomic::Ordering::Relaxed))
    }

    fn noise_y(&self) -> f64 {
        f64::from_bits(self.noise_y_bits.load(std::sync::atomic::Ordering::Relaxed))
    }
}

impl flodl::BatchDataSet for LetterBatchAdapter {
    fn len(&self) -> usize { self.dataset.samples.len() }

    fn get_batch(&self, indices: &[usize]) -> Result<Vec<Tensor>> {
        let b = indices.len();
        let images: Vec<&Tensor> = indices.iter().map(|&i| &self.dataset.samples[i].image).collect();
        let cleans: Vec<&Tensor> = indices.iter().map(|&i| &self.dataset.samples[i].clean).collect();
        let partners: Vec<&Tensor> = indices.iter().map(|&i| &self.dataset.samples[i].partner_clean).collect();
        let letter_data: Vec<i64> = indices.iter().map(|&i| self.dataset.samples[i].letter_idx).collect();
        let case_data: Vec<f32> = indices.iter().map(|&i| self.dataset.samples[i].case_label).collect();

        // Origin noise: generated per-batch with current noise params.
        let nx = self.noise_x();
        let ny = self.noise_y();
        let n = b as i64;
        let origin = if nx > 0.0 || ny > 0.0 {
            let noise = Tensor::randn(&[n, 2], Default::default())?;
            let scale = Tensor::from_f32(&[nx as f32, ny as f32], &[1, 2], Device::CPU)?;
            noise.mul(&scale)?
        } else {
            Tensor::zeros(&[n, 2], Default::default())?
        };

        Ok(vec![
            Tensor::stack(&images, 0)?,
            Tensor::stack(&cleans, 0)?,
            Tensor::stack(&partners, 0)?,
            Tensor::from_i64(&letter_data, &[n], Device::CPU)?,
            Tensor::from_f32(&case_data, &[n, 1], Device::CPU)?,
            origin,
        ])
    }
}

/// Batch field names matching graph inputs. Use with DataLoader `.names()`.
pub const BATCH_NAMES: &[&str] = &["image", "clean", "partner_clean", "letter_idx", "case", "origin"];

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

// --- VRAM-resident data (pre-stacked on GPU) ---

/// All dataset tensors stacked contiguously on a device.
struct ResidentData {
    images: Tensor,      // [N, 1, H, W]
    cleans: Tensor,      // [N, 1, H, W]
    partners: Tensor,    // [N, 1, H, W]
    letters: Tensor,     // [N]
    cases: Tensor,       // [N, 1]
}

impl ResidentData {
    /// Stack all samples and move to device. Reports VRAM usage.
    fn from_dataset(ds: &LetterDataset, device: Device) -> Result<Self> {
        let n = ds.len();
        let images_ref: Vec<&Tensor> = ds.samples.iter().map(|s| &s.image).collect();
        let cleans_ref: Vec<&Tensor> = ds.samples.iter().map(|s| &s.clean).collect();
        let partners_ref: Vec<&Tensor> = ds.samples.iter().map(|s| &s.partner_clean).collect();
        let letter_data: Vec<i64> = ds.samples.iter().map(|s| s.letter_idx).collect();
        let case_data: Vec<f32> = ds.samples.iter().map(|s| s.case_label).collect();

        eprintln!("Stacking {} samples to {}...", n, if device.is_cuda() { "VRAM" } else { "RAM" });

        let images = Tensor::stack(&images_ref, 0)?.to_device(device)?;
        let cleans = Tensor::stack(&cleans_ref, 0)?.to_device(device)?;
        let partners = Tensor::stack(&partners_ref, 0)?.to_device(device)?;
        let letters = Tensor::from_i64(&letter_data, &[n as i64], Device::CPU)?.to_device(device)?;
        let cases = Tensor::from_f32(&case_data, &[n as i64, 1], Device::CPU)?.to_device(device)?;

        let data_bytes = images.nbytes() + cleans.nbytes() + partners.nbytes()
            + letters.nbytes() + cases.nbytes();
        eprintln!("Resident data: {:.1} MB on {:?}", data_bytes as f64 / 1024.0 / 1024.0, device);

        Ok(ResidentData { images, cleans, partners, letters, cases })
    }

    /// Extract a batch via index_select — single GPU kernel per tensor.
    fn batch(&self, idx: &Tensor) -> Result<LetterBatch> {
        Ok(LetterBatch {
            image: self.images.index_select(0, idx)?,
            clean: self.cleans.index_select(0, idx)?,
            partner_clean: self.partners.index_select(0, idx)?,
            letter_idx: self.letters.index_select(0, idx)?,
            case_label: self.cases.index_select(0, idx)?,
        })
    }
}

// --- Batched loader ---

/// Iterates over a LetterDataset in shuffled batches.
///
/// Supports two modes:
/// - **Resident** (CUDA): all data pre-stacked on GPU, batches via `index_select`.
/// - **Transfer** (fallback): per-batch `pin_memory` + `to_device`.
///
/// The mode is chosen automatically when `set_device()` is called.
pub struct LetterLoader<'a> {
    ds: &'a LetterDataset,
    batch_size: usize,
    shuffle: bool,
    drop_last: bool,
    device: Option<Device>,
    resident: Option<ResidentData>,
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
            resident: None,
            perm: Vec::new(),
            pos: 0,
        };
        loader.init_perm();
        loader
    }

    /// Set the target device for batch tensors.
    /// On CUDA, automatically enables VRAM-resident mode if data fits.
    pub fn set_device(&mut self, device: Device) {
        self.device = Some(device);
    }

    /// Pre-load all data to device. Called after set_device().
    /// On failure, falls back to per-batch transfer silently.
    pub fn make_resident(&mut self) -> Result<()> {
        let Some(device) = self.device else { return Ok(()); };
        match ResidentData::from_dataset(self.ds, device) {
            Ok(data) => {
                self.resident = Some(data);
                Ok(())
            }
            Err(e) => {
                eprintln!("warning: resident mode failed ({e}), using per-batch transfer");
                Ok(())
            }
        }
    }

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

        if let Some(ref resident) = self.resident {
            // VRAM-resident path: index_select on GPU tensors.
            let device = self.device.unwrap_or(Device::CPU);
            let idx_data: Vec<i64> = indices.iter().map(|&i| i as i64).collect();
            let idx = Tensor::from_i64(&idx_data, &[idx_data.len() as i64], device)?;
            return resident.batch(&idx).map(Some);
        }

        // Fallback: per-batch stack + transfer.
        let b = indices.len();
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

        let img_batch = Tensor::stack(&images, 0)?;
        let clean_batch = Tensor::stack(&cleans, 0)?;
        let partner_batch = Tensor::stack(&partners, 0)?;
        let letter_idx = Tensor::from_i64(&letter_data, &[b as i64], Device::CPU)?;
        let case_label = Tensor::from_f32(&case_data, &[b as i64, 1], Device::CPU)?;

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
