# Inference Pushback: Continuous Learning Through Predictive Error Residuals

## Origin

Design concept for the Autonet native model. The core question: can an architecture allow inference to write back into weights, enabling continuous learning without a separate training phase?

Current base: VL-JEPA with local decoder.

## Core Insight

Separate training and inference phases are an engineering convenience, not a biological reality. Biological neural networks are always live, always writable. Experience that produces sufficiently large prediction error permanently modifies the network that produced it. The experience and the learning are the same event.

## Rejected Approaches

### Bolt-on modular design (too much overhead)

The initial decomposition suggested three additional systems on top of JEPA:

1. **Dual weight system** -- frozen base weights + parallel fast weight matrix at attention layers
2. **Gating mechanism** -- a meta-predictive layer maintaining a self-model, filtering which errors qualify for write-back
3. **Consolidation pipeline** -- offline folding of fast weights into slow weights (analogous to sleep)

This was rejected because it imposes an engineering topology that doesn't match the biological reality. The brain doesn't carry separate adversarial systems for error detection, gating, and consolidation. It's one architecture doing everything at once.

### Formal novelty scoring as a precondition (unnecessary abstraction)

Defining "novelty" upfront as a precondition for surprise requires building a novelty detector, which reintroduces a separate evaluative system. This separates knowing from experiencing, which is exactly the split we're trying to eliminate.

However, novelty can be formally derived after the fact from the architecture's own outputs. See "Derived Novelty Definition" below.

## Proposed Architecture: Predictive Error Residuals

### Single rule, applied everywhere

Each layer in the network operates in predictive coding mode:

1. Each layer generates a prediction of the representation it expects from the next layer
2. It receives the actual representation
3. The difference (prediction error) is a vector
4. The magnitude of that vector is the surprise -- not defined, not scored, just computed as a byproduct of subtraction
5. A small **writable local residual** at each layer updates proportionally to this prediction error during inference

That's it. One modification to VL-JEPA layers. No gating module, no meta-model, no consolidation pipeline.

### Properties this gives you for free

**Self-selecting threshold.** Most prediction errors are small. Local residual updates are tiny and wash out. When prediction error is massive, the update is proportionally large and leaves a lasting trace. The magnitude of the error IS the gate. No separate gating mechanism needed.

**Contextual surprise.** The same input produces different surprise levels depending on context, because prediction error depends on what was predicted, which depends on everything processed so far. Identical stimulus, different history, different update. This is how biological surprise works.

**No formal novelty definition.** Surprise is just the distance between expectation and reality. The architecture computes it as a natural byproduct of prediction. Adding a formal novelty metric on top would be redundant and would reintroduce the separation between experience and evaluation.

### The biological analog

This maps to the observation that shame (or any overwhelming corrective experience) isn't gated by a committee. There's no internal system that scores the event and decides whether it qualifies for permanent storage. The signal is simply too large to be absorbed by the forward pass alone. It pushes back into the weights because the error magnitude demands it. The intensity IS the selection criterion.

## Derived Novelty Definition

The rejection of formal novelty scoring as a precondition leads to an inversion: if surprise is just prediction error magnitude, and that's sufficient to drive the whole system, then novelty isn't a precondition for surprise. It's a description of surprise after the fact.

You don't compute novelty to detect surprise. You read surprise to derive novelty.

**Novelty(x) = aggregate prediction error magnitude across layers for input x given current model state**

Key properties of this definition:

- **Fully derived.** Requires no separate detector or scoring system. Falls out of the architecture naturally as a readable quantity.
- **Subjective by construction.** Novelty is not a property of the input. It's a property of the relationship between the input and the current state of the model. The same stimulus has different novelty for different model states. This is correct, because novelty IS subjective. (A cockroach is unremarkable in Bucharest and unprecedented on a cockroach-free ocean fleet.)
- **Layer distribution carries information.** An input that produces large prediction error at a single layer is surprising in a narrow way. An input that produces large error across many layers simultaneously is deeply novel, structurally unexpected at every level of abstraction. The distribution of error across layers is itself a meaningful signal about the *kind* of novelty.
- **Retrospective, not predictive.** This definition doesn't tell you in advance what will be novel. It tells you after processing what was novel. This is the right direction of explanation. Defining novelty in advance requires a model of "what is normal," which is just another way of smuggling in a separate evaluative system.

## Open Questions

- **Residual persistence.** Do local residuals reset per session (ephemeral learning) or persist across sessions (continuous identity)? This is the difference between "remembers within a conversation" and "grows over a lifetime." Both are interesting. The latter is harder.

- **Catastrophic drift.** Without explicit gating, what prevents accumulation of residual updates from gradually corrupting the base representations? Possible that the proportionality rule handles this naturally (small errors dominate in frequency, wash each other out, only extreme events leave permanent marks) but this needs empirical validation.

- **Residual capacity.** How large does the writable residual at each layer need to be? Too small and large errors can't be captured. Too large and the model has too many degrees of freedom at inference time.

- **Interaction with JEPA joint embedding.** VL-JEPA predicts in representation space, not pixel space. The prediction errors are already abstract. Does this make the residual updates more semantically meaningful by default, or does the abstraction introduce noise?

- **Self-model emergence.** In the rejected modular design, a self-model was an explicit component. In this architecture, does something like a self-model emerge naturally from the accumulation of residuals? The pattern of what has surprised the system is, in a sense, a record of what the system expected to be. That's a self-model defined negatively: not what I am, but what I was wrong about.

- **Consolidation without a pipeline.** Is there a lightweight analog to sleep-based consolidation? Periodic inference on null input where residuals settle? Or is the accumulation itself sufficient?

## Summary

One rule: surprise changes you, proportional to how surprising it was.

Everything else is either already present in the JEPA forward pass or emerges from accumulation over time.
