# The Trajectory Thesis

Neural networks are trajectory generators. An input enters a high-dimensional
space, and the network's weights define a landscape that guides it along a path
through that space. Training shapes the landscape. Inference lets the input
follow its natural trajectory. Everything else is implementation detail.

This document traces how that idea shaped the FBRL project, from the initial
problem through two framework implementations to a broader vision for how
neural networks could be built.

---

## Part I: The Problem

### Trajectories, not layers

The standard mental model of a neural network is a stack of layers: input →
hidden → hidden → output. This is a useful abstraction for building software,
but it obscures what's actually happening geometrically.

Each layer transforms a point in activation space to a new point. A forward pass
is a trajectory, a sequence of positions through a high-dimensional manifold.
The weights define the vector field that determines where each point moves next.

This isn't a metaphor. Residual networks (ResNet) made it literal: the skip
connection `x + f(x)` means each layer computes a *delta*, a small step from
the current position. He et al. (2015) showed this dramatically improves
training. Chen et al. (2018) took it further with Neural ODEs, replacing
discrete residual steps with a continuous differential equation:

```
dx/dt = f(x, t, θ)
```

The forward pass becomes solving an ODE, following a continuous trajectory
through activation space. The "layers" are just discretization steps along
that trajectory.

### Unified vocabulary

Most DL concepts have clean trajectory interpretations:

| Standard framing | Trajectory framing |
|---|---|
| Training | Shaping the landscape so trajectories converge correctly |
| Inference | Letting an input follow its natural trajectory |
| Loss function | Measuring how far the trajectory's endpoint is from the target region |
| Gradient descent | Adjusting the landscape to pull trajectories toward targets |
| Overfitting | Trajectories that are too narrow, only work for training inputs |
| Generalization | Wide valleys, nearby inputs follow similar paths |
| Regularization | Smoothing the landscape to prevent sharp, narrow valleys |
| Attention | Dynamically choosing which dimensions matter at each trajectory step |
| Residual connections | Making trajectory steps incremental (continuous-like flow) |
| Adaptive computation | Letting the trajectory decide its own length |
| Transfer learning | A landscape from one task has valleys useful for another |
| Dropout | Randomly blocking dimensions, forcing trajectories to be robust |
| Batch normalization | Re-centering the trajectory distribution at each step |

The trajectory frame doesn't replace the math, it provides geometric intuition
for why the math works.

### The selection bias

If your framework makes certain trajectory structures expensive, researchers
avoid them. This isn't a conscious choice, it's selection pressure:

**Well-explored** (cheap trajectories in Python):
- Fixed-depth feedforward (ResNet, ViT)
- Single-pass attention (Transformer)
- Constant-width parallel heads

**Under-explored** (expensive trajectories in Python):
- Recurrent attention with variable fixation count
- Tree search during training (MCTS-style)
- Iterative hypothesis refinement
- Adaptive computation depth
- Multi-scale processing with feedback loops

Biological cognition is firmly in the second category. Vision involves
sequential fixations. Reasoning involves iterative refinement. Memory recall
involves variable-depth search. The architectures that most closely model
human cognition are the ones Python punishes most.

Fixed-architecture networks force every input through the same trajectory
length and structure. A 50-layer ResNet always takes 50 steps. A 12-head
transformer always runs 12 parallel sub-trajectories. But adaptive
architectures let the input *choose its trajectory*:

- **Adaptive depth**: iterate until confidence is high enough → variable-length
  trajectory, short for easy inputs, long for hard ones.
- **Conditional branches**: route different inputs through different sub-networks
  → trajectories fork based on the input's position in activation space.
- **Recurrent attention**: each step chooses where to look next → the trajectory
  is literally a sequence of positions in the input space.
- **Early exit**: stop when a criterion is met → the trajectory terminates when
  it reaches a confident region.

These are the architectures Python/PyTorch penalizes most. Every branch
evaluation, every loop iteration, every early-exit check is a Python `if`
statement with ~3-5μs overhead plus a CUDA synchronization point. The framework
*discourages* trajectory branching through performance pressure.

Training adaptive architectures requires backpropagation through the trajectory
structure itself:

- **Variable-length loops**: if input A takes 3 steps and input B takes 7, the
  backward pass unrolls 3 and 7 steps respectively. The gradient signal teaches
  the network both *what to compute at each step* and *when to stop*.
- **Conditional branches**: only the taken branch receives gradients. Over many
  training samples, each branch's weights specialize for the inputs that route
  to them.
- **Parallel paths**: independent trajectories (e.g., multiple attention heads)
  get independent gradients. The heads can specialize without interference.

This selection bias may be steering the entire field away from architectures
that would work better for certain problems — not because the ideas are wrong,
but because the tools make them impractical to explore.

---

## Part II: The Journey

### goDl — proving the concept

FBRL (Feedback Recursive Loop) needed exactly the kind of architecture that
Python punishes: a recurrent attention mechanism that takes multiple glimpses
of an image, choosing where to look next based on what it has seen so far.
Each glimpse feeds back into the next, the trajectory *is* the model.

To make this tractable, we built goDl: a Go-based deep learning framework with
a graph-native execution engine. The key idea was that branching, looping, and
adaptive depth should be composition primitives, declared in the graph
topology, not implemented as imperative code inside a `forward()` method.

The design layered naturally onto the trajectory thesis:

1. **Tensor API** — the coordinate system. Points in activation space are
   tensors. Operations move points.
2. **Autograd** — trajectory analysis. Given a trajectory (forward pass),
   compute how changing the landscape (weights) would change where the
   trajectory ends up (gradients).
3. **Layers & Optimizers** — standard landscape components. Linear
   transformations, activation functions, gradient-based landscape shaping.
4. **Graph Engine** — trajectory orchestration. Branching, looping, parallel
   paths, and adaptive depth as first-class constructs. The graph engine
   manages trajectories through a dynamic computation structure.

goDl validated the architecture. The FBRL letter model, a recurrent foveal
attention network learning to recognize letters through strategic fixation
placement, went from 3% to 27% letter accuracy in 22 training epochs and was
clearly converging. The graph API expressed the model naturally: attention loops,
skip connections, tagged observation points, all declared as topology.

But goDl hit a wall. Go's garbage collector has no visibility into VRAM behind
CGo wrappers. A tiny Go struct can hide megabytes of GPU memory. The GC sees a
few bytes, feels no pressure to collect, and VRAM fills silently. We built a
4-phase cleanup system (nil gradFn, refcounting, saved tensor Release, CUDA OOM
callback), each phase solved one leak path, but the root cause was
fundamental: a garbage-collected language cannot manage foreign memory it cannot
see. At 98% GPU utilization the model spilled 1.4GB over the card's 6GB budget.

The architecture was validated. The runtime was not.

### floDl — the Rust rewrite

Rust solves the memory problem at the language level. The Drop trait gives
deterministic cleanup: when a Tensor goes out of scope, its VRAM is freed
immediately. No GC, no finalizers, no 4-phase workarounds. The borrow checker
prevents use-after-free at compile time. FFI to libtorch is zero-cost.

floDl ports the graph engine, observation system, and flow constructs from goDl.
What changes is everything underneath:

- **libtorch's native autograd** handles tensor operations and backward passes.
  The same kernels, the same accumulation order, the same numerical behavior
  as PyTorch. A model expressed in floDl and an equivalent model in PyTorch
  produce the same numbers, no framework discrepancy to debug.
- **RAII replaces garbage collection.** One-phase cleanup instead of four.
  Tensor drop frees VRAM. Variable drop frees its gradient function and saved
  tensors. No OOM callbacks, no manual Release calls.
- **The graph API is preserved.** FlowBuilder, loops (for_n, while_cond,
  until_cond), switch, gate, map, tag/using, observation, the constructs
  that make FBRL expressible carry over unchanged.

The split is clean: libtorch owns the math (and must match PyTorch exactly),
floDl owns the trajectory orchestration (and has no equivalent in PyTorch).

---

## Part III: Observable Trajectories

Expressing complex trajectories is necessary but not sufficient. If you can't
see a trajectory evolving, you can't understand it. And if you can't understand
it, you can't improve it.

Current observability in deep learning is primitive: log a scalar loss every N
steps, plot it after training, squint at the curve. This is like debugging a
program by printing its exit code. You know *whether* it worked, not *why*.

### Structural observability

When the computation graph is explicit, observation becomes structural. Any
node in the graph can be tagged as an observation point. A tagged node doesn't
just produce a tensor, it produces a *named measurement* at a known position
in the trajectory. This is fundamentally different from sprinkling
`print(loss.item())` through imperative code:

- **Observation points are part of the graph topology**, not afterthoughts.
  When you read the graph, you see what's being measured and where.
- **Metrics flow through the same pipeline as data.** Collection, aggregation,
  and analysis are built into the execution model, not bolted on via external
  tools.
- **The graph structure itself is visible.** Graphviz rendering shows loops,
  branches, skip connections, and routing, the trajectory's shape, not just
  its outputs.

### From logging to understanding

Scalar logging tells you the loss went down. Structural observation tells you
*how the trajectory changed*:

- Which branch is firing more often as training progresses?
- Is attention diversifying across the input or collapsing to a single point?
- Is a particular head stalling while others improve?
- Did the routing pattern shift after the learning rate decay?

These questions require per-node, per-step metrics, not just endpoint
summaries. When observation is structural, answering them is a query, not a
research project.

### Trend analysis as first-class feedback

Raw metrics are data. Trends are information. A graph-native observation system
can compute slopes, detect convergence, identify stalls, and flag regressions
automatically. This turns the training loop into a feedback system where the
researcher responds to *analyzed signals*, not raw numbers.

Combined with live visualization, real-time charts updating as the graph
topology is rendered alongside its flowing metrics, the gap between "run an
experiment" and "understand what happened" collapses. You watch the trajectory
evolve, see a head stalling, adjust a loss weight, and see the effect in the
next epoch. This is what fast iteration on research ideas actually requires.

---

## Part IV: Where This Leads

### Beyond single-strategy training

Current large models are trained with essentially one strategy: predict the next
token, then refine with RLHF. All knowledge, physics, poetry, reasoning,
perception, must be acquired through that single lens. This works at scale,
but it's brute force. It's like teaching someone everything through
multiple-choice tests.

### Mixture of Experts is shallow routing

Mixture of Experts (MoE) as deployed in current models (GPT-4, Mixtral) is a
step toward structured computation: a router sends each input to a subset of
expert sub-networks. But the routing is shallow, one decision at the input
boundary, and all experts share the same architecture, the same training
objective, and the same loss function. It's "which expert computes this" not
"which strategy solves this."

### Mixture of Strategies

A deeper approach: different sub-networks trained with fundamentally different
learning strategies, composed into a single system.

- A perception module trained with supervised learning on labeled data
- A reasoning module trained with reinforcement learning on reward signals
- A memory module trained with contrastive learning on similarity
- A consistency module trained to detect contradictions in intermediate results
- A meta-controller that learns when to invoke which module

Each component has its own loss function, learning rate, and update schedule.
Some are frozen (pretrained knowledge bases), others are actively learning.
Gradients flow between components where strategies should reinforce each other,
and are blocked where they shouldn't interfere.

In the trajectory frame: each module defines a different kind of landscape, and
the meta-controller learns to compose trajectories across these landscapes.
An input might start in the perception landscape, branch into reasoning, loop
through memory retrieval, pass a consistency check, and exit, all as a single
adaptive trajectory.

### Hierarchical composition

The key insight: strategies can be composed hierarchically. A trained mixture of
experts is itself a component that can be placed inside a larger graph:

```
Level 0: Individual modules (Linear, GRU, attention heads)
Level 1: Trained sub-networks (perception, reasoning, memory)
Level 2: Strategy mixtures (MoE with learned routing)
Level 3: Meta-graph that learns to compose strategy mixtures
```

Each level is trained independently, then composed. The meta-graph at level 3
doesn't need to learn perception, it learns *when to use the perception
strategy mixture versus the reasoning one*, and how to route intermediate
results between them.

This is Graph-as-Module composition: a trained graph is a Module, which is a
node in a parent graph. The same principle that lets you nest a Linear layer
inside a Transformer block lets you nest an entire trained model inside a
meta-learning graph.

### The training challenge

Multi-strategy training raises hard questions:

**Gradient interference.** When two strategies optimize different objectives,
their gradients can conflict in shared parameters. The graph engine must support
selective gradient flow: blocking, scaling, or rerouting gradients at strategy
boundaries.

**Catastrophic forgetting.** When the meta-controller trains, it must not
destroy what the sub-modules already learned. This requires freezing, elastic
weight consolidation, or explicit memory mechanisms.

**Credit assignment.** When a composed trajectory produces a good result, which
strategy deserves the credit? The meta-controller's routing decision? The
reasoning module's computation? Proper credit assignment through branching
trajectories is an open research problem.

**Curriculum design.** What order do you train the components? Bottom-up
(modules first, then composition)? Top-down (end-to-end with structure)?
Interleaved? The training curriculum itself becomes a design decision.

These are research problems, not engineering problems. But they require a
framework where multi-strategy composition is a natural primitive — not a
fragile collection of custom training loops and manual gradient hacks.

---

## Modular intelligence

The current AI development paradigm is monolithic. One team trains one model
with one loss function in one massive run. Everything is entangled: fixing
math reasoning risks degrading language ability, retraining visual perception
requires a full run costing millions. The organizational structure mirrors the
architecture: everyone must understand everything because everything affects
everything.

This is not how any other complex engineering discipline works.

### How complex systems are actually built

Nobody builds an airplane as one monolithic piece. The engine team builds
engines. The avionics team builds avionics. The airframe team builds structure.
Integration engineers compose them. Each team is world-class at their piece.
An engine upgrade doesn't require rebuilding the wings.

The same principle applies to intelligence. The human brain is not a single
homogeneous network. It is a composition of specialized modules, visual cortex,
motor cortex, hippocampus, prefrontal cortex, each with different architecture,
different learning rules, different connectivity patterns. They were "trained"
on different objectives over evolutionary timescales. They compose through
well-defined interfaces (neural pathways). Damage to one module impairs specific
abilities without destroying others.

### Modular AI development

Graph-as-Module composition enables the same structure for AI:

**Architecture level.** Independent modules with clear interfaces. A perception
graph doesn't know or care about the reasoning graph. They communicate through
typed tensor connections, not shared weights. Each module can have different
architecture, CNNs for vision, GRUs for sequential reasoning, transformers
for language, composed in a single executable graph.

**Training level.** Each module has its own training strategy, its own data, its
own loss function, its own optimizer. The math module is trained on mathematical
reasoning with RL rewards. The vision module is trained on images with
supervised labels. The orchestrator is trained on how to compose them. Retraining
one doesn't touch the others.

**Team level.** A small team owns the vision module end-to-end. They understand
its architecture, its failure modes, its training data. They don't need to
understand reinforcement learning, that's another team's module. The graph
designer composes their work. This scales: ten specialized teams of five
outperform one team of fifty trying to hold the entire system in their heads.

**Deployment level.** Update one module without redeploying the whole system. The
vision module improved? Swap it in. The reasoning module has a regression? Roll
it back. The orchestrator stays the same. Version each module independently.
A/B test individual components.

### The meta-learning layer

The most powerful implication: a graph that orchestrates pre-trained specialized
modules is itself a Module. It can be trained. Its training objective is not
"solve the task", it's "learn how to compose the available capabilities to
solve the task."

This separates two fundamentally different kinds of learning:

1. **Capability learning** — teaching a module to do something (perceive,
   reason, remember). Requires large data, specialized training, deep domain
   expertise. Done once, reused everywhere.

2. **Composition learning** — teaching the orchestrator when and how to invoke
   capabilities. Requires much less data (routing decisions, not raw
   computation). Can be retrained quickly. Can be task-specific while the
   capabilities remain general.

This mirrors how human expertise works. A doctor doesn't re-learn visual
perception for each patient. They compose pre-existing capabilities, vision,
memory, reasoning, pattern matching, through a learned orchestration strategy
specific to medical diagnosis. The capabilities are general; the composition
is specialized.

### Why the tools matter

The reason AI hasn't adopted modular development isn't that it's a bad idea.
It's that the tools didn't support it:

- **You can't modularize what you can't compose.** If the framework has no
  concept of sub-graphs, you can't build independent modules.
- **You can't independently retrain what you can't independently differentiate.**
  If gradients must flow through the entire system, you can't freeze one module
  while training another.
- **You can't parallelize what your framework serializes.** If every module
  dispatch goes through Python's GIL, you can't run independent modules
  concurrently.
- **You can't iterate quickly on composition if composition is expensive.** If
  the orchestrator's branching decisions each cost 3-5μs of Python overhead,
  complex routing becomes impractical.

A graph-native engine makes all of this structural: sub-graphs with independent
training contexts, selective gradient flow, parallel module execution, and
zero-overhead routing decisions. Not because the engine solves the research
problems, but because it removes the engineering barriers that prevent the
research from happening.

---

## References

- He et al. (2015) — *Deep Residual Learning for Image Recognition*. Skip
  connections as incremental trajectory steps.
- Chen et al. (2018) — *Neural Ordinary Differential Equations*. Continuous-depth
  networks as ODE trajectories.
- Graves (2016) — *Adaptive Computation Time for Recurrent Neural Networks*.
  Variable-length trajectories with a halting mechanism.
- Bengio et al. (2015) — *Conditional Computation in Neural Networks*. Gating
  and routing as trajectory branching.
- Amari (1998) — *Natural Gradient Works Efficiently in Learning*. Information
  geometry — the manifold structure of parameter space.
- Shazeer et al. (2017) — *Outrageously Large Neural Networks: The
  Sparsely-Gated Mixture-of-Experts Layer*. Learned routing to expert
  sub-networks.
- Kirkpatrick et al. (2017) — *Overcoming Catastrophic Forgetting in Neural
  Networks*. Elastic weight consolidation for multi-task learning without
  destroying prior knowledge.
- Jacobs et al. (1991) — *Adaptive Mixtures of Local Experts*. The original
  mixture of experts — competitive learning between specialized modules.
