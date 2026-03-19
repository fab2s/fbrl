//! Word recognition domain — model, training, data, and losses.
//!
//! Two-phase foveal attention pipeline:
//!
//! ```text
//! image → H0Init → Loop(ScanStep, 8).Using("image").Tag("scan")
//!       → Loop(ReadStep, 12).Using("image").Tag("read")
//!       → CrossAttentionReadout → 4 × classifier + decoder
//! ```
//!
//! Scan discovers letter positions via prescribed x-coordinates.
//! Read extracts per-letter features with free attention.
//! CrossAttentionReadout produces per-position latents from grouped reads.

pub mod data;
mod decoder;
mod glimpse;
pub mod loss;
mod model;
mod modules;
mod readout;
mod synthetic;
mod train;

pub use data::*;
pub use decoder::*;
pub use glimpse::*;
pub use loss::*;
pub use model::*;
pub use modules::*;
pub use readout::*;
pub use synthetic::*;
pub use train::*;
