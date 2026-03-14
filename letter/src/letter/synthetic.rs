//! Synthetic data generation for testing (no font rendering dependency).

use flodl::tensor::{Result, Tensor};

use super::data::{LetterDataset, LetterSample};

/// Create a dataset of random images for testing.
/// Each sample gets a random 128x128 image with a random letter/case label.
/// Not useful for real training — only for testing the pipeline.
pub fn new_synthetic_dataset(n: usize) -> Result<LetterDataset> {
    let mut samples = Vec::with_capacity(n);
    for i in 0..n {
        let img = Tensor::rand(&[1, 128, 128], Default::default())?;
        let letter_idx = (i % 26) as i64;
        let case_label = if i % 2 == 1 { 1.0 } else { 0.0 };

        samples.push(LetterSample {
            clean: img.clone(),
            image: img,
            letter_idx,
            case_label,
        });
    }
    Ok(LetterDataset { samples })
}
