//! fbrl-word — hierarchical foveal attention word recognition.
//!
//! Composed from independently trained modules:
//!
//! - **SubScan**: bounded-region localization (wide, blurred, partial glimpses)
//! - **LetterModel**: scan (trainable) + read (frozen) from letter checkpoint
//! - **MetaScan** (future): word-level position estimation
//!
//! Training is progressive:
//! 1. Letter model trained standalone (separate crate)
//! 2. SubScan + Letter composition — SubScan locates, letter scan adapts, read stays frozen
//! 3. Full word model — MetaScan assigns regions, SubScan + Letter per position
//!
//! Built on [flodl]'s graph tree for hierarchical composition.

pub mod word;
