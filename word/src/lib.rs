//! fbrl-word — foveal attention word recognition.
//!
//! 4-letter word recognition via a two-phase foveal attention mechanism:
//! 8 wide-patch scans with prescribed x-positions discover letter locations,
//! then 12 focused read glimpses extract per-letter features. A cross-attention
//! readout produces per-position latents for independent classification.
//!
//! Built on [flodl]'s computation graph framework.

pub mod word;
