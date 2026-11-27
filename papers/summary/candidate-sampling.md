# Candidate Sampling

Summary of [Candidate Sampling Algorithms Reference](https://www.tensorflow.org/extras/candidate_sampling.pdf) (TensorFlow).

## Problem Setting

Given a multiclass or multilabel problem where each training example $(x_i, T_i)$ consists of a context $x_i$ and a small (multi)set of target classes $T_i$ from a large universe $\mathcal{L}$, we want to learn a compatibility function $F(x, y)$ between classes and contexts.

**Exhaustive** methods (full softmax, full logistic) compute $F(x, y)$ for every $y \in \mathcal{L}$ on every training example. When $|\mathcal{L}|$ is large, this is prohibitively expensive.

**Candidate sampling** methods construct a training task where we only evaluate $F(x, y)$ for a small candidate set $C_i = T_i \cup S_i$, where $S_i \subset \mathcal{L}$ is a randomly chosen sample. The sampling distribution $Q(y \mid x)$ gives the probability (or expected count) of class $y$ appearing in $S_i$.

### Music Recommendation Framing

| Concept | Notation | Music Example |
|---|---|---|
| Context | $x_i$ | User profile: listening history, demographics, time of day |
| Target class(es) | $T_i$ | Artist(s) the user actually streamed |
| Class universe | $\mathcal{L}$ | All ~10M artists on the platform |
| Compatibility fn | $F(x, y)$ | Dot product of user embedding and artist embedding |
| Sampled negatives | $S_i$ | Small set of artists the user did *not* stream (sampled) |

A training example is a (user, artist, stream) tuple. Computing $F$ against all 10M artists per example is infeasible — candidate sampling makes it tractable by evaluating only the streamed artist plus a handful of negatives.

## Algorithms at a Glance

| Algorithm | Positives $\text{POS}_i$ | Negatives $\text{NEG}_i$ | Training input $G(x,y)$ | Loss | $F(x,y)$ learns to approximate |
|---|---|---|---|---|---|
| **NCE** | $T_i$ | $S_i$ | $F(x,y) - \log Q(y \mid x)$ | Logistic | $\log P(y \mid x)$ |
| **Negative Sampling** | $T_i$ | $S_i$ | $F(x,y)$ | Logistic | $\log \frac{P(y \mid x)}{Q(y \mid x)}$ |
| **Sampled Logistic** | $T_i$ | $S_i - T_i$ | $F(x,y) - \log Q(y \mid x)$ | Logistic | $\log \text{odds}(y \mid x)$ |
| **Sampled Softmax** | $\{t_i\}$ | $S_i - \{t_i\}$ | $F(x,y) - \log Q(y \mid x)$ | Softmax | $\log P(y \mid x) + K(x)$ |
| **Full Logistic** | $T_i$ | $\mathcal{L} - T_i$ | $F(x,y)$ | Logistic | $\log \text{odds}(y \mid x)$ |
| **Full Softmax** | $\{t_i\}$ | $\mathcal{L} - \{t_i\}$ | $F(x,y)$ | Softmax | $\log P(y \mid x) + K(x)$ |

The **logistic training loss** is:

$$\mathcal{L} = \sum_i \left( \sum_{y \in \text{POS}_i} \log(1 + e^{-G(x_i, y)}) + \sum_{y \in \text{NEG}_i} \log(1 + e^{G(x_i, y)}) \right)$$

The **softmax training loss** is:

$$\mathcal{L} = \sum_i \left( -G(x_i, t_i) + \log \sum_{y \in \text{POS}_i \cup \text{NEG}_i} e^{G(x_i, y)} \right)$$

The key insight across all sampled methods: subtract $\log Q(y \mid x)$ from the raw logits before feeding them into the loss. This corrects for the sampling bias — without it, frequently-sampled classes would be penalised too harshly.

## Sampled Softmax

**Reference**: [Jean et al., 2015](http://arxiv.org/abs/1412.2007)

For single-label problems. Each example $(x_i, \{t_i\})$ has one target class. We want $F(x, y) \leftarrow \log P(y \mid x) + K(x)$.

**Procedure**:

1. Sample a small set $S_i \subset \mathcal{L}$ according to $Q(y \mid x)$
2. Form candidates $C_i = S_i \cup \{t_i\}$
3. For each $y \in C_i$, compute corrected logits: $G(x, y) = F(x, y) - \log Q(y \mid x)$
4. Apply softmax loss: predict which candidate is the true target

**Derivation via Bayes' rule**: the posterior probability that $y$ is the target given the candidate set $C_i$ works out to:

$$\log P(t_i = y \mid x_i, C_i) = \log P(y \mid x_i) - \log Q(y \mid x_i) + K'(x_i, C_i)$$

The $-\log Q$ correction falls out naturally — it is not an approximation but an exact bias correction.

### Music Example: Learning Artist Embeddings with Sampled Softmax

**Setup**: learn a user embedding $u \in \mathbb{R}^d$ and per-artist embedding $v_a \in \mathbb{R}^d$ such that $F(u, a) = u^\top v_a \approx \log P(\text{artist} = a \mid \text{user})$.

```
Training example:
  context  x = user_42  (featurised as the hidden-layer output of a user tower)
  target   t = "Radiohead"

Universe L = {Radiohead, Drake, Taylor Swift, ..., 10M artists}

Full softmax: score all 10M artists, normalise, backprop → infeasible

Sampled softmax (k=1000):
  1. Sample S = {Drake, Billie Eilish, ..., 999 other artists} ~ Q
  2. Candidates C = S ∪ {Radiohead}
  3. For each artist a ∈ C, compute:
       G(user_42, a) = u₄₂ᵀ vₐ  −  log Q(a)
  4. Softmax loss: push Radiohead's corrected logit above the rest
```

The $-\log Q(a)$ correction is essential: without it, popular artists like Drake (high $Q$) would be penalised disproportionately as negatives, biasing the embeddings.

## Noise Contrastive Estimation (NCE)

**Reference**: [Gutmann & Hyvärinen, 2010](http://www.jmlr.org/proceedings/papers/v9/gutmann10a/gutmann10a.pdf)

Generalises to multilabel problems (multiple target classes per example). Trains $F(x, y) \leftarrow \log P(y \mid x)$.

**Procedure**:

1. Sample a multiset $S_i$ (may or may not depend on $x_i$, must not depend on $T_i$)
2. Form candidates $C_i = T_i + S_i$ (multiset union)
3. Task: distinguish targets from sampled classes via logistic regression
4. Training input: $G(x, y) = F(x, y) - \log Q(y \mid x)$

**Core identity**:

$$\log \text{odds}(y \text{ came from } T \text{ vs } S \mid x) = \log \frac{P(y \mid x)}{Q(y \mid x)} = \log P(y \mid x) - \log Q(y \mid x)$$

The logistic classifier learns to separate real targets from noise. Backpropagation through the $-\log Q$ corrected input trains $F$ to approximate $\log P(y \mid x)$.

### Music Example: NCE for Multi-Artist Sessions

A user streams multiple artists in a session — this is a multilabel setting where NCE naturally applies.

```
Training example:
  context  x = user_42's session context (time, recent history, ...)
  targets  T = {Radiohead, Portishead}       ← artists actually streamed

Noise sample:
  S = {Drake, BTS, Adele, Bad Bunny, Beyoncé}  ~ Q  (5 noise artists)

Candidates: C = T + S

For each artist a ∈ C:
  G(x, a) = u₄₂ᵀ vₐ  −  log Q(a)
  label   = 1 if a ∈ T, else 0

Logistic loss per candidate:
  Radiohead:  log σ(G(x, Radiohead))       ← push up (positive)
  Portishead: log σ(G(x, Portishead))      ← push up (positive)
  Drake:      log σ(−G(x, Drake))          ← push down (negative)
  BTS:        log σ(−G(x, BTS))            ← push down (negative)
  ...
```

NCE handles the fact that a user can stream *multiple* artists in one session — each is a positive, and the loss is per-candidate logistic.

## Negative Sampling

**Reference**: [Mikolov et al., 2013](http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)

A simplified variant of NCE that **omits the $-\log Q(y \mid x)$ correction**. The training input is just $G(x,y) = F(x,y)$ with no bias correction.

As a result, $F(x,y)$ is trained to approximate:

$$F(x,y) \leftarrow \log P(y \mid x) - \log Q(y \mid x)$$

This means the learned function is entangled with the sampling distribution $Q$. Changing $Q$ changes what $F$ converges to — unlike all other algorithms described here. This makes negative sampling highly sensitive to the choice of noise distribution.

### Music Example: Why the Correction Matters

```
Suppose Q is proportional to artist stream counts (unigram distribution):
  Q(Drake)     = 0.05   (very popular)
  Q(Radiohead) = 0.001  (moderately popular)

With NCE (corrected):
  G(x, Drake)     = F(x, Drake)     − log(0.05)  = F(x, Drake) + 3.0
  G(x, Radiohead) = F(x, Radiohead) − log(0.001) = F(x, Radiohead) + 6.9

  → The correction compensates for Drake appearing as a negative far more often.
    F learns log P(y \mid x) for both artists on the same scale.

With Negative Sampling (uncorrected):
  G(x, Drake)     = F(x, Drake)
  G(x, Radiohead) = F(x, Radiohead)

  → F converges to log P(y \mid x) − log Q(y \mid x).
    Drake's embedding is pushed down extra by his high Q.
    The embeddings are only meaningful relative to the specific Q used during training.
```

For artist embeddings that need to transfer to downstream tasks (playlist generation, similar-artist retrieval), NCE or sampled softmax are preferable because the learned $F$ has a clean probabilistic interpretation independent of $Q$.

## Sampled Logistic

A variant of NCE that discards sampled classes that happen to also be target classes (sampling without replacement from $\mathcal{L} \setminus T_i$). Requires $T_i$ to be a set (no duplicates).

Trains $F(x,y) \leftarrow \log \text{odds}(y \mid x) = \log \frac{P(y \mid x)}{1 - P(y \mid x)}$.

$$\log \text{odds}(y \text{ from } T \text{ vs } (S - T) \mid x) = \log \frac{P(y \mid x)}{Q(y \mid x)(1 - P(y \mid x))} = \log \frac{P(y \mid x)}{1 - P(y \mid x)} - \log Q(y \mid x)$$

Training input is the same as NCE: $G(x,y) = F(x,y) - \log Q(y \mid x)$.

### Music Example

Sampled logistic is natural when you want to learn: "given a user, what are the odds they'd stream artist $a$ vs. not?" — a useful signal for binary relevance scoring in recommendation. Removing accidental hits (target artists that also appear in the negative sample) avoids contradictory gradients on the same example.

## Context-Specific vs. Generic Sampling

The sampling distribution $Q$ may depend on the context $x$. Context-specific sampling could generate **hard negatives** — artists similar to the user's taste but not streamed — providing a stronger training signal.

In practice, most implementations use **generic** (context-independent) sampling:

- **Uniform**: $Q(y) = 1/|\mathcal{L}|$
- **Unigram**: $Q(y) \propto \text{count}(y)^{\alpha}$ (e.g., $\alpha = 0.75$ in Word2Vec)

The reason is computational: generic sampling enables **batchwise sampling**.

## Batchwise Sampling

A key practical optimisation. The same set $S$ of sampled negatives is shared across all examples in a batch.

Since $F(x, y) = x^\top v_y$ in many models, computing logits for $B$ examples against $|S|$ candidates is a single matrix multiplication of shape $(B, d) \times (d, |S|)$ — highly efficient on GPUs.

```
Batch of 512 (user, artist) training examples
  + 1000 shared sampled negative artists

Without batchwise sampling:
  512 × 1000 = 512K independent embedding lookups

With batchwise sampling:
  1000 embedding lookups + one (512 × d) @ (d × 1000) matmul
  → orders of magnitude faster on GPU
```

This means we can afford hundreds or thousands of negatives per batch with negligible overhead, because fetching an artist embedding across devices costs more than computing its dot product with all 512 user vectors in the batch.

## Summary: Choosing an Algorithm

| Consideration | Recommendation |
|---|---|
| Single-label (one artist per example) | Sampled softmax |
| Multilabel (session with multiple artists) | NCE or sampled logistic |
| Need clean $\log P(y \mid x)$ for downstream use | NCE or sampled softmax (not negative sampling) |
| Want log-odds for binary relevance | Sampled logistic |
| Simplicity, don't need calibrated probabilities | Negative sampling |
| GPU efficiency with large negative sets | Use batchwise sampling with any method |

All methods except negative sampling produce an $F(x,y)$ that is independent of the sampling distribution $Q$ — making them more robust and transferable.
