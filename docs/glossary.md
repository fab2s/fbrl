# Glossary

Deep learning terms, acronyms, and concepts as they appear in this project. Organized from general foundations to FBRL-specific details.

---

## Neural Network Foundations

**Neural Network** — A function composed of layers of learnable parameters (weights and biases) that transforms inputs into outputs. Each layer applies a linear transformation followed by a non-linear activation function.

**Weight** — A learnable parameter in a neural network. In a linear layer, weights form a matrix that multiplies the input. In a convolutional layer, weights form small kernels that slide over the input.

**Bias** — A learnable scalar added after the linear transformation. Shifts the output independently of the input.

**Activation Function** — A non-linear function applied after a linear transformation. Without non-linearity, stacking layers would collapse into a single linear operation. Common activations:

- **ReLU** (Rectified Linear Unit) — `max(0, x)`. The most common activation in hidden layers. Simple, fast, but outputs are unbounded above.
- **Tanh** — Squashes output to `[-1, 1]`. Used in FBRL's `location_head` to keep fixation coordinates in normalized image bounds.
- **Sigmoid** — Squashes output to `[0, 1]`. Used in decoder output layers to match pixel intensity range (0=black, 1=white).
- **Softmax** — Converts a vector of raw scores (logits) into a probability distribution that sums to 1. Used before classification losses.

**Logits** — Raw unnormalized scores output by a network before softmax. Higher logit = higher confidence for that class. Cross-entropy loss operates on logits directly.

**Feature Map** — The 2D or 3D tensor output of a convolutional layer. Each channel represents a different learned feature (edges, textures, shapes). Early layers detect simple features; deeper layers combine them into complex patterns.

**Embedding** — A learned dense vector representation of a discrete input (e.g., a position index, a letter category). Converts categorical data into continuous space where similar items can be nearby.

**Forward Pass** — Computing the model's output given an input by passing data through all layers in sequence.

**Backward Pass** — Computing gradients of the loss with respect to all model parameters, using the chain rule (backpropagation). These gradients tell each parameter how to change to reduce the loss.

**Gradient** — The partial derivative of the loss with respect to a parameter. Points in the direction of steepest loss increase; optimization steps in the opposite direction.

**Backpropagation** — Algorithm that efficiently computes all gradients in a neural network by applying the chain rule backwards through the computation graph.

---

## Layer Types

**Linear Layer** (`nn.Linear`) — Multiplies input by a weight matrix and adds bias: `output = input @ W + b`. The fundamental building block. Also called "fully connected" or "dense" layer.

**Conv2d** (2D Convolution) — Slides small learnable kernels across a 2D input (like an image), computing dot products at each position. Detects local patterns (edges, textures) regardless of position. Parameters: `in_channels`, `out_channels`, `kernel_size`, `stride`, `padding`.

**ConvTranspose2d** (Transposed Convolution, "Deconvolution") — The approximate inverse of convolution. Upsamples spatial dimensions — used in decoders to go from small feature maps back to full image resolution. With `stride=2`, each spatial dimension doubles.

**BatchNorm** (Batch Normalization) — Normalizes activations across the batch dimension to have zero mean and unit variance. Stabilizes training, allows higher learning rates, acts as mild regularization.

**AdaptiveAvgPool2d** — Average pooling that outputs a fixed spatial size regardless of input size. `AdaptiveAvgPool2d(1)` collapses spatial dimensions to 1x1, producing a single value per channel.

**Flatten** — Reshapes a multi-dimensional tensor into a 1D vector. Bridges between convolutional layers (which produce 3D feature maps) and linear layers (which expect 1D vectors).

---

## Recurrent Networks

**RNN** (Recurrent Neural Network) — Network that processes sequences by maintaining a hidden state that updates at each timestep. The hidden state is a compressed memory of everything seen so far.

**GRU** (Gated Recurrent Unit) — A type of RNN cell with gating mechanisms that control what information to keep and what to discard. Simpler than LSTM but similarly effective. In FBRL, the GRU processes one glimpse per timestep and accumulates spatial knowledge across fixations.

**Hidden State** (`h`) — The internal memory vector of an RNN/GRU, updated at each timestep. In FBRL, `h` carries forward from the scan phase into the read phase, so spatial knowledge from scanning persists during reading.

**Timestep** (`t`) — One step in a sequence. In FBRL, each timestep corresponds to one fixation/glimpse.

---

## Convolutional Architectures

**CNN** (Convolutional Neural Network) — A neural network primarily composed of convolutional layers. Standard approach for image processing — but in FBRL, the CNN only processes tiny patches, not the full image.

**Stride** — Step size of the convolution kernel. `stride=1` moves one pixel at a time (preserves spatial size). `stride=2` skips every other position (halves spatial dimensions).

**Padding** — Adding zeros around the input border so the convolution kernel can operate at the edges without shrinking the output.

**Kernel Size** — The spatial dimensions of a convolution filter (e.g., 3x3, 5x5). Larger kernels see more context but have more parameters.

---

## Attention Mechanisms

**Attention** — A mechanism that lets the network selectively focus on relevant parts of its input. Different from "attention" in transformers — FBRL uses *spatial* attention (choosing where to look in an image).

**Cross-Attention** — Attention where queries come from one source and keys/values come from another. In FBRL's `CrossAttentionReadout`, learned position tokens (queries) attend over glimpse hidden states (keys/values) to extract per-position representations.

**Query, Key, Value** — The three components of attention. The query asks "what am I looking for?", keys are "what's available?", values are "what to return?". Attention weight = how well each query matches each key. Output = weighted sum of values.

**Scaled Dot-Product Attention** — Standard attention formula: `softmax(Q @ K^T / sqrt(d_k)) @ V`. The scaling factor `sqrt(d_k)` prevents dot products from growing too large.

**Position Tokens** — Learned vectors in `CrossAttentionReadout` that represent spatial positions (one per letter position). Initialized with spatial bias (e.g., left-to-right spread) and refined during training.

---

## Loss Functions

**Loss Function** — A scalar value measuring how wrong the model's output is. Training minimizes this value. The total loss is typically a weighted sum of multiple terms.

**MSE** (Mean Squared Error) — Average squared difference between prediction and target: `mean((pred - target)^2)`. Used for reconstruction loss — measures pixel-level accuracy.

**Cross-Entropy Loss** — Standard loss for classification. Measures how different the predicted probability distribution is from the true label. For 26 letters, random guessing gives `ln(26) = 3.26`. Lower is better; 0 means perfect confidence on the correct class.

**BCE** (Binary Cross-Entropy) — Cross-entropy for binary (two-class) predictions. Used for content detection ("is there a letter here? yes/no") and case classification.

**BCEWithLogitsLoss** — BCE that takes raw logits (before sigmoid) as input. Numerically more stable than applying sigmoid first then BCE.

**Regularization** — Any technique that prevents overfitting by adding constraints. In FBRL, diversity loss acts as regularization by preventing fixation collapse.

---

## Optimization

**Optimizer** — Algorithm that updates model parameters using gradients to minimize the loss.

**Adam** — Adaptive Moment Estimation. The most popular optimizer — maintains per-parameter learning rates based on first and second moments of gradients. Converges faster than plain SGD for most problems. Used throughout FBRL.

**SGD** (Stochastic Gradient Descent) — The simplest optimizer: `param -= lr * gradient`. "Stochastic" because it uses random mini-batches rather than the full dataset.

**Learning Rate** (`lr`) — Step size for parameter updates. Too high = unstable training, too low = slow convergence. Typical range: 0.0001 to 0.001.

**Learning Rate Scheduler** — Adjusts the learning rate during training. FBRL uses `CosineAnnealingLR`.

**CosineAnnealingLR** — Decays learning rate following a cosine curve from initial value to near-zero over `T_max` epochs. Smooth decay avoids sudden drops.

**Param Groups** — Optimizer configuration with different learning rates for different model components. In FBRL: sensors/controller at 0.0001 (slow, preserve pretrained features), readout/decoder at 0.001 (fast, learn from scratch).

**Gradient Clipping** (`clip_grad_norm_`) — Caps the total magnitude of gradients to a maximum norm (e.g., 5.0). Prevents exploding gradients that can destabilize training.

**Batch Size** — Number of images processed in one gradient update step. Larger batches = smoother gradients but more memory. FBRL uses 32-52.

**Epoch** — One complete pass through the entire training dataset. At 200 words x 20 variants = 4000 samples with batch_size=52, one epoch is ~77 gradient steps.

**Convergence** — When training loss stabilizes at a low value and the model has learned its task. Further training yields diminishing returns.

---

## Training Concepts

**Overfitting** — Model memorizes training data but fails on unseen test data. Signs: training loss drops but test accuracy doesn't improve.

**Generalization** — Model performs well on data it hasn't seen during training. The real goal.

**Transfer Learning** — Initializing a model with weights from a previously trained model on a related task. In FBRL: single-letter encoder weights transfer to bigram/word models, giving the attention system a head start on reading letterforms.

**Fine-tuning** — Training a transferred model on the new task, typically with lower learning rates on pretrained components to avoid destroying useful features.

**Freezing** — Setting `requires_grad = False` on parameters so they aren't updated during training. FBRL's transfer scaffold freezes read_sensor + classifiers during early training while the scan sensor learns from scratch.

**Scaffold** — Training assistance that's gradually removed. FBRL's temporal scaffold guides attention to specific image regions early on, then anneals to zero so the model must discover its own strategy. Analogous to training wheels.

**Annealing** — Gradually reducing a value during training. Used for scaffold weight (1.0 to 0.0), learning rate (cosine decay), and other hyperparameters.

**Warmup** — Running a few initial steps at low learning rate before ramping up. Helps avoid early instability when model parameters are random.

**Checkpoint** — Saved snapshot of model weights + training state at a specific epoch. Allows resuming training and selecting the best model.

**Hyperparameter** — A setting chosen before training starts (learning rate, batch size, guide_weight, etc.). Unlike model parameters (weights), hyperparameters are not learned — they're set by the practitioner.

---

## Data & Preprocessing

**Dataset** (`torch.utils.data.Dataset`) — A Python class that holds training data and returns individual samples by index. FBRL has `LetterDataset`, `BigramDataset`, and `WordDataset`.

**DataLoader** — PyTorch utility that wraps a Dataset to provide batching, shuffling, and parallel loading. Feeds mini-batches to the training loop.

**Tensor** — A multi-dimensional array (PyTorch's core data type). Images are 4D tensors: `(batch, channels, height, width)`.

**Normalization** — Scaling input values to a standard range. FBRL images are 0-1 (pixel intensity).

**Gaussian Noise** — Random noise sampled from a normal distribution, added to training images. Forces the model to be robust to imperfections. Controlled by `noise_level`.

**Clean Image** — Noise-free version of a training image. Used for attention guide loss evaluation (honest signal without noise artifacts).

**Canvas** — The full image dimensions. 128x128 for single letters and bigrams, 256x128 for 4-letter words.

---

## Evaluation

**Accuracy** — Percentage of correctly classified samples. 100% = perfect.

**Per-Position Accuracy** — Separate accuracy for each letter position in a word. Shows whether the model reads all positions equally well.

**All-Correct Accuracy** — Percentage of samples where every letter is classified correctly. More stringent than per-position accuracy.

**Test Set** — Held-out data never seen during training. Performance on test data measures generalization.

**Atlas** — FBRL-specific: an interactive HTML visualization showing fixation patterns (heatmaps and paths) for all test samples. Reveals how the model's attention strategy varies across letters, fonts, and words.

---

## Numerical Concepts

**Normalized Coordinates** `[-1, 1]` — PyTorch's `grid_sample` convention for spatial coordinates. `(-1, -1)` = top-left corner, `(0, 0)` = center, `(1, 1)` = bottom-right. All fixation locations use this space.

**Grid Sample** (`F.grid_sample`) — Differentiable image sampling at continuous coordinates. Extracts patches at arbitrary (x, y) locations with bilinear interpolation. The key operation that makes foveal attention differentiable (gradients flow back through the sampling location).

**Differentiable** — An operation that supports gradient computation. Critical for end-to-end training — if fixation placement weren't differentiable, the model couldn't learn where to look via backpropagation.

**RBF** (Radial Basis Function) — A function whose value depends only on distance from a center. FBRL uses Gaussian RBFs for fixation diversity repulsion: `exp(-dist^2 / (2 * sigma^2))`.

**Separable Convolution** — Applying a 2D operation as two sequential 1D operations (horizontal then vertical). Used for efficient Gaussian blur in the attention guide.

---

## FBRL Architecture

**FBRL** — Feedback Recursive Loop. The core idea: encode an image, decode it, then recode it under a different condition. Each recode direction forces the latent toward more abstract representations.

**Foveal Attention** — The model sees only a tiny patch (12x12 pixels) at each fixation. No peripheral vision — even more constrained than biology. Forces the model to develop an active perception strategy.

**GlimpseSensor** — Extracts a small patch from the image at a given fixation location using `grid_sample`. The extracted patch is processed by a small CNN to produce a feature vector. Accepts patch sizes as int (square) or (h, w) tuple (rectangular).

**AttentionController** — GRU-based module that processes glimpse features sequentially and predicts the next fixation location via a `location_head` (linear layer + tanh). Accumulates information across all glimpses.

**Latent Vector** — The 256-dimensional representation produced after all glimpses. Contains everything the model knows about the image, compressed from a sequence of tiny patch observations.

**VisionModel** — Single-letter model. 10 glimpses, 12x12 patches, 128x128 canvas. Classifies letter identity (26 classes) and case (upper/lower). Case-conditioned decoder enables recoding.

**BigramVisionModel** — Two-letter model with two-phase attention. Scan phase: 5 glimpses, 12x18 wide patches. Read phase: 6 glimpses, 12x12 focused patches. Shared GRU controller. CrossAttentionReadout with 2 position tokens.

**WordVisionModel** — Four-letter model with prescribed x-scan. Scan phase: 8 glimpses, prescribed left-to-right x sweep, learned y. Read phase: 12 glimpses, fully free. Content detection head on scan states. CrossAttentionReadout with 4 position tokens.

**Two-Phase Attention** — Scan + Read architecture. Scan uses wide rectangular patches to map the spatial layout. Read uses focused square patches to identify individual letters. Hidden state carries forward between phases.

**Prescribed X-Scan** — Word model constraint: scan-phase x-coordinates follow a fixed linear sweep `[-0.75, +0.75]` across the image width. Only y is learned. Reflects how human reading proceeds roughly left-to-right.

**Content Detection Head** — `nn.Linear(256, 1)` on each scan hidden state. Predicts whether letter content exists at the current fixation location. BCE loss against sampled image intensity. Teaches the GRU to explicitly encode "letter found here" in its hidden state.

---

## FBRL Loss Terms

**Reconstruction Loss** — MSE between the decoded image and the input. Forces the attention system to gather enough information for the decoder to rebuild the full image.

**Classification Loss** — Cross-entropy for letter identity at each position. The primary task objective.

**Attention Guide Loss** — A Gaussian-blurred version of the clean image creates a "scent trail" around letter strokes. Fixations are evaluated against this blurred map — locations on bright areas (near letters) score high. Maximized as a loss (negated). `guide_weight=8.0` controls its influence.

**Fixation Diversity Loss** — Pairwise Gaussian repulsion between all fixation points. Without it, fixations collapse to a single spot. The `sigma` parameter controls the repulsion radius; the `vy` parameter makes it anisotropic (directional).

**Split Diversity** — Different VY values for scan and read phases. `scan_vy=0.3` (< 1) penalizes horizontal clustering, forcing the scan to spread across the image width. `read_vy=1.5` (> 1) penalizes vertical clustering, encouraging the read phase to explore vertically within each letter.

**Temporal Scaffold** — Divides read-phase glimpses into N equal temporal segments (one per letter position). Each segment is guided toward a horizontal stripe of the image. Anneals from full strength to zero over the scaffold phase. Teaches left-to-right reading order before the model must discover it independently.

**Recode Loss** — Single-letter model only. Flip the case label, decode the same latent, compare to the partner image (e.g., encode 'a', decode as 'A'). Forces case-invariant latent representations.

**Isolation Mask Loss** — Word model: randomly picks 1 of 4 letter positions per batch, zeros out the other 3 letter stripes, runs the masked image through the model, and computes classification loss only on the exposed position. Conceptually similar to BERT's masked language modeling — forces per-position reading capability.

**Edge Loss** — Pushes scan fixations toward image edges: `(1 - |x|).mean()`. Encourages full-width scanning coverage. Less useful with prescribed x-scan.

**Hit Rate** — Diagnostic metric (not a loss term). Fraction of fixations that land on actual letter pixels in the clean image. ~30-40% indicates selective, structural reading — the model samples diagnostic features rather than tracing outlines.

---

## Infrastructure

**Docker** — Containerization platform ensuring reproducible environments. FBRL runs in a container with PyTorch 2.5.1, system fonts, and GPU support.

**AMP** (Automatic Mixed Precision) — Training technique that uses float16 for most operations and float32 only where needed (loss scaling, accumulation). Halves VRAM usage and speeds up computation on GPUs with hardware float16 support. In FBRL, enabled via `torch.amp.autocast('cuda')` with a `GradScaler` to prevent underflow. ~17% speedup on Pascal (GTX 1080 Ti). Controlled by the `amp` config flag.

**CUDA** — NVIDIA's parallel computing platform for GPU acceleration. Training on GPU is ~10-100x faster than CPU.

**GPU** (Graphics Processing Unit) — Hardware accelerator optimized for parallel matrix operations. FBRL trains on a GTX 1080 Ti (Pascal architecture, compute capability 6.1).

**PyTorch** — The deep learning framework used by FBRL. Provides tensors, automatic differentiation, neural network layers, and GPU support.

**Makefile** — Build automation file defining targets (like `make train-words`) with overridable variables. Provides reproducible, one-command pipeline execution.

**Git LFS** — Git Large File Storage. Used for archived model files (.pth.gz) that are too large for regular Git.

**pin_memory** — DataLoader optimization that pre-stages CPU tensors in pinned (page-locked) memory for faster GPU transfer.
