//! fbrl — foveal attention letter recognition.
//!
//! Single-letter recognition via a recurrent foveal attention mechanism.
//! The model learns to direct its gaze across the input image, extracting
//! multi-resolution glimpses at each fixation point, then classifies the
//! letter identity and case from the accumulated representation.
//!
//! Built on [flodl]'s computation graph framework.

pub mod letter;
