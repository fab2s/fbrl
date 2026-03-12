//! Letter dataset and batched loader for training.

use flodl::tensor::{Device, Result, Tensor};

/// One rendered letter with metadata.
pub struct LetterSample {
    pub image: Tensor,       // [1, H, W] noisy image
    pub clean: Tensor,       // [1, H, W] clean image (for attention guide)
    pub letter_idx: i64,     // 0-25 (A=0, B=1, ...)
    pub case_label: f32,     // 0.0=upper, 1.0=lower
}

/// Stacked mini-batch ready for training.
pub struct LetterBatch {
    pub image: Tensor,       // [B, 1, H, W]
    pub clean: Tensor,       // [B, 1, H, W]
    pub letter_idx: Tensor,  // [B] int64
    pub case_label: Tensor,  // [B, 1] float32
}

/// Holds all samples in memory.
pub struct LetterDataset {
    pub samples: Vec<LetterSample>,
}

impl LetterDataset {
    pub fn len(&self) -> usize { self.samples.len() }
    pub fn is_empty(&self) -> bool { self.samples.is_empty() }
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
        let mut letter_data = Vec::with_capacity(b);
        let mut case_data = Vec::with_capacity(b);

        for &idx in indices {
            let s = &self.ds.samples[idx];
            images.push(&s.image);
            cleans.push(&s.clean);
            letter_data.push(s.letter_idx);
            case_data.push(s.case_label);
        }

        // Stack into batch tensors via unsqueeze + cat.
        let img_batch = stack_tensors(&images)?;
        let clean_batch = stack_tensors(&cleans)?;
        let letter_idx = Tensor::from_i64(&letter_data, &[b as i64])?;
        let case_label = Tensor::from_f32(&case_data, &[b as i64, 1], Device::CPU)?;

        // Move to target device if set.
        let batch = if let Some(device) = self.device {
            LetterBatch {
                image: img_batch.to_device(device)?,
                clean: clean_batch.to_device(device)?,
                letter_idx: letter_idx.to_device(device)?,
                case_label: case_label.to_device(device)?,
            }
        } else {
            LetterBatch {
                image: img_batch,
                clean: clean_batch,
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

/// Stack tensors along a new leading dimension via unsqueeze(0) + cat.
fn stack_tensors(tensors: &[&Tensor]) -> Result<Tensor> {
    assert!(!tensors.is_empty(), "cannot stack empty tensor list");
    let mut result = tensors[0].unsqueeze(0)?;
    for t in &tensors[1..] {
        let expanded = t.unsqueeze(0)?;
        result = result.cat(&expanded, 0)?;
    }
    Ok(result)
}
