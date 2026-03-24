//! SubScan: bounded-region letter localization.
//!
//! Given a region of a word image (defined by center + half-width), SubScan
//! takes two short, wide, blurred glimpses within the bounded region and
//! infers a letter center position for handoff to the letter model.
//!
//! The output position is free — it does not need to coincide with either
//! glimpse location. SubScan observes partial ink structure and infers
//! where the letter center must be.
//!
//! ## Region bounding
//!
//! SubScan's location head is reparameterized to stay within its region:
//! ```text
//! x = region_center + region_half_width * tanh(raw_x)
//! y = tanh(raw_y)   // full vertical range
//! ```
//!
//! ## Glimpse design
//!
//! - **Short and wide** (~8x28 pixels): horizontal localization emphasis
//! - **2/3 letter width**: one glimpse cannot see the whole letter
//! - **Blurred**: density only, no letterform detail
//! - **Two free glimpses**: minimum for triangulating the letter center

// TODO: implement SubScan module
// - SubScanConfig: patch dimensions, blur, hidden_dim, region sizing
// - SubScanStep: GRU + region-bounded location head + blurred GlimpseSensor
// - SubScan graph: H0Init → Loop(SubScanStep, 2) → output position
