# Iteration Roadmap: From Letters to Reading

## Progression

### Level 0 (current): Single letters, single font, 128x128
- Model learns foveal attention extracts identity from local features
- Fixation strategies are letter-specific
- Case work adds first taste of abstraction (same identity, different form)

### Level 1: Single letters, multiple fonts
- Critical step before sequences
- Forces structural feature learning — "horizontal stroke near top connected to vertical stroke = T" regardless of serif, weight, proportions
- Attention controller must develop font-invariant fixation strategies
- If fixation patterns generalize across fonts, the approach has real substance
- Practical: bundle 10-15 TTF files (DejaVu, Liberation, Noto — all open license), rotate during generation

### Level 2: Bigrams/trigrams — the bridge
- 2-3 letter combinations ("ab", "th", "ing") on 128x128 canvas
- Each letter ~40-60px wide, still readable at current patch size
- Model must: plan scan path, segment letters, accumulate identity across fixations
- Output becomes a sequence (CTC loss or fixed-length prediction)
- Key advantage: sequential fixation naturally produces temporal ordering aligned with spatial character order — CNN sees everything at once and must learn reading direction from supervision alone

### Level 3: Words
- Full words, variable length, multiple fonts
- Wider images (128x384 or 128x512), more glimpses needed
- Interesting emergent behaviors to measure:
  - Does it skip predictable letters? (humans do)
  - Does it fixate word centers first? (optimal landing position — humans do this)
  - Does it spend more glimpses on rare/long words? (humans do)
- Language model component (even bigram/trigram prior) could modulate attention: "after T-H, next is probably E or A, don't need to look carefully"
- Top-down prediction meets bottom-up perception — central question in reading research

### Level 4: Lines and paragraphs (requires meta-attention)
- See "Hierarchical Attention" section below

## What Carries Forward

- GlimpseSensor and AttentionController architectures reuse directly
- Learned letter-feature detectors in glimpse CNN transfer (a serif is a serif)
- Fixation diversity loss and attention guide loss apply at every level
- Pre-training encoder on isolated letters, fine-tuning on bigrams = legitimate transfer
- CNNVisualDecoder does NOT transfer to sequences — reconstruction becomes optional, sequence prediction takes over

## Where Syllables Fit

Syllables emerge naturally rather than being an explicit training target. In Level 2-3, if the model chunks "th", "ing", "tion" as units processed with fewer fixations than random trigrams, that's syllable-like behavior from efficiency pressure. Measurable without explicit training.

The multimodal angle makes syllables explicit — syllables are fundamentally phonological units. Vision-only model has no reason to prefer "str-ong" over "st-ron-g". A model that also hears the word has natural syllable boundary signals. Strong argument for revisiting multimodal at Level 3.

## Hierarchical / Meta Attention

### The Problem
At word/sentence scale, the model needs to allocate fixation budget across two problems:
1. Where are the words? (whitespace, line breaks)
2. What does each word say? (letter-level reading)

### Human Reading Hierarchy
1. **Page level**: eyes jump between lines (return sweeps)
2. **Line level**: eyes jump between words (inter-word saccades, guided by spaces)
3. **Word level**: eyes fixate within a word (current model)

### Proposed Architecture
```
MetaController (coarse GRU, large receptive field)
  → picks region of interest (word-sized bounding box)
  → AttentionController (fine GRU, current letter reader)
    → N glimpses within that region
    → returns word latent
  → MetaController integrates word latent, picks next region
```

The meta controller doesn't need to read — it needs to find structure. Its glimpses can be larger and lower-resolution than the letter reader's.

### Options for Meta-Level Perception

**Downsampled overview glimpse**: before fine reading, take glimpses at very low resolution (whole line at 16x-32x downscale). Enough to see word-shaped blobs and gaps. Analogous to parafoveal vision — can't read letters in periphery but can see word shapes and spaces.

**Density profile**: collapse image vertically (sum columns) → 1D signal. Peaks = ink, valleys = spaces. Meta controller scans this to plan word-level jumps. Cheap, and close to what human visual system does for saccade targeting. Clean separation: meta = 1D sequence model over density, word reader = 2D attention model.

### Training Curriculum
- Level 0-1: train word-level reader in isolation
- Level 2-3: freeze/fine-tune word reader, train meta controller to deploy it
- Level 4: end-to-end fine-tuning of both levels

### Why the Hierarchy is Natural
Not arbitrary — forced by the foveal constraint. Full-image models don't need meta-attention (process everything in parallel). Foveal model MUST allocate limited fixation budget, and that allocation naturally factorizes into "where are words" and "what does this word say." Problem structure imposes solution structure.

### Multimodal Reconnection
Audio provides strong prior for meta controller:
- How many words to expect
- Roughly how long each is (syllable count → letter count)
- What the next word might be
Meta controller could integrate this when planning where to look — very close to what literate humans do during read-aloud tasks.
