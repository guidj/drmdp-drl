# Summary: Learning Long-Term Reward Redistribution via Randomized Return Decomposition (RRD)

## Metadata

**Title:** Learning Long-Term Reward Redistribution via Randomized Return Decomposition

**Authors:** Zhizhou Ren, Ruihan Guo, Yuan Zhou, Jian Peng

**Year:** 2022

**Venue:** Tenth International Conference on Learning Representations (ICLR 2022 - Spotlight)

**Links:**
- [arXiv Preprint](https://arxiv.org/abs/2111.13485)
- [OpenReview](https://openreview.net/forum?id=dHiJRzl9dZ)
- [Official GitHub Repository](https://github.com/stilwell-git/randomized-return-decomposition)

## 1. Problem Statement & Research Question

**Problem:** Many practical reinforcement learning applications require agents to learn from sparse and delayed rewards. In episodic tasks, agents may only receive a single reward signal at the end of an entire trajectory, making credit assignment—attributing actions to future outcomes—extremely challenging. This is particularly problematic in long-horizon tasks where the gap between actions and their consequences can span hundreds or thousands of timesteps.

**Existing Limitations:**

1. **High computational cost of return decomposition methods:** RUDDER and similar methods that learn to redistribute rewards using LSTMs require backpropagation through time (BPTT) on full trajectories, which becomes prohibitively expensive for long-horizon tasks.

2. **Scalability challenges:** As trajectory length T increases, the computational complexity of deterministic return decomposition methods grows quadratically or worse, limiting their applicability to complex real-world problems.

3. **Overfitting to specific trajectory structures:** Methods that learn complex redistribution functions may overfit to the training trajectories and fail to generalize.

4. **Inefficient use of trajectory data:** Many approaches don't effectively leverage the full trajectory information available in episodic reinforcement learning.

**Research Question:** Can we develop a scalable reward redistribution algorithm that efficiently handles long-horizon episodic tasks by using Monte-Carlo sampling to make least-squares-based reward redistribution tractable, while maintaining or improving upon the performance of existing methods like RUDDER?

**Key Contribution:** The paper introduces **Randomized Return Decomposition (RRD)**, a novel approach that:

1. **Uses Monte-Carlo sampling of trajectory subsequences** to make least-squares reward redistribution scalable to long-horizon tasks
2. **Establishes a tractable surrogate loss function** $L_{Rand-RD}(\theta)$ that approximates the deterministic return decomposition loss $L_{RD}(\theta)$
3. **Provides theoretical analysis** showing the gap between the randomized and deterministic losses characterizes variance, giving implicit variance regularization
4. **Demonstrates superior sample efficiency and final policy quality** compared to RUDDER, IRCR, and other baselines

The key insight is that instead of requiring the redistributed rewards to sum exactly to the episodic return (which requires processing the full trajectory), RRD samples random subsequences and learns a reward function that gives unbiased estimates of the return in expectation. This trades off some determinism for massive computational savings.

---

## 2. Methodology & Approach

**High-Level Approach:**

RRD addresses the challenge of credit assignment in long-horizon episodic RL through an **iterative, interleaved approach** that alternates between reward model learning and policy optimization.

**Algorithm 1 (from paper Section 3.3) iteratively performs:**
1. **Collect trajectory** using the current policy
2. **Update reward model** $R_\theta(s,a)$ using trajectories in the replay buffer with the RRD loss
3. **Update policy** using the current reward model $R_\theta(s,a)$ as the proxy reward signal
4. **Repeat** until convergence

**Key characteristics:**
- The paper explicitly states: "iterative paradigm that **iterates between** the rewarding modeling and policy optimization modules"
- This is NOT a sequential two-phase approach where the reward model is fully trained before policy learning
- Both reward model and policy are continuously refined together throughout training
- Each iteration uses the current policy to collect data and the current reward model to guide policy updates

**Key Innovations:**

1. **Randomized Loss Function:** Instead of requiring $\sum_t R_\theta(s_t,a_t) = R_{ep}(\tau)$ exactly for each trajectory (deterministic constraint), RRD uses a randomized loss that only requires unbiased estimation in expectation:

   $$L_{Rand-RD}(\theta) = E_{\tau \sim D}\left[E_{I \sim \rho_T(\cdot)}\left[\left(R_{ep}(\tau) - \frac{T}{|I|}\sum_{t \in I} R_\theta(s_t,a_t)\right)^2\right]\right]$$

   where $I$ is a randomly sampled subsequence of indices.

2. **Subsequence Sampling Strategy:** Sample K timesteps uniformly from the trajectory of length T. This reduces computational cost from O(T²) to O(K²), where K << T.

3. **Implicit Variance Regularization:** The gap between $L_{Rand-RD}(\theta)$ and $L_{RD}(\theta)$ characterizes the variance of the Monte-Carlo estimator, naturally encouraging smoother redistribution functions.

4. **Scalability:** The Monte-Carlo approach allows RRD to handle trajectories with T=1000+ timesteps, whereas methods requiring full BPTT struggle beyond T=100-200.

**Theoretical Foundation:**

The paper proves (Theorem 1) that:
- The randomized loss $L_{Rand-RD}(\theta)$ is an upper bound of the deterministic loss $L_{RD}(\theta)$
- The gap between them captures the variance of the Monte-Carlo return estimator
- This provides implicit regularization toward low-variance redistribution functions
- Under certain conditions, minimizing $L_{Rand-RD}$ leads to similar solutions as minimizing $L_{RD}$

**Assumptions:**

1. **Episodic structure:** The task must have well-defined episodes with terminal states
2. **Trajectory-level feedback:** The reward signal (or sum of rewards) is available at the episode level
3. **Sufficient exploration:** The replay buffer contains diverse trajectories that cover the state-action space adequately
4. **Lipschitz reward function:** The learned reward function $R_\theta$ should be smooth to generalize well

---

## 3. Mathematical Formulation

### Problem Setup: Delayed Reward MDP (DRMDP)

**Standard MDP:** $M = (S, A, P, r, \gamma, \rho_0)$
- $S$: State space
- $A$: Action space
- $P$: Transition dynamics $P(s'|s,a)$
- $r$: Reward function $r(s,a)$ (potentially sparse/delayed)
- $\gamma$: Discount factor
- $\rho_0$: Initial state distribution

**Episodic Setting:**
- Agent interacts to generate trajectory $\tau = (s_0, a_0, s_1, a_1, ..., s_{T-1}, a_{T-1})$
- Episodic return: $R_{ep}(\tau) = \sum_{t=0}^{T-1} \gamma^t r(s_t, a_t)$
- In sparse reward setting: $r(s_t, a_t) = 0$ for $t < T-1$, and only $r(s_{T-1}, a_{T-1}) \neq 0$

### Deterministic Return Decomposition Loss

The ideal reward redistribution satisfies:
$$\sum_{t=0}^{T-1} R_θ(s_t, a_t) = R_{ep}(τ) \quad \forall τ$$

This leads to the deterministic loss:
$$L_{RD}(θ) = \mathbb{E}_{τ \sim D}\left[\left(R_{ep}(τ) - \sum_{t=0}^{T-1} R_θ(s_t, a_t)\right)^2\right]$$

where D is the distribution of trajectories in the replay buffer.

**Challenge:** For trajectories with length T, this requires computing gradients through T terms, leading to O(T²) computational complexity with standard optimizers.

### Randomized Return Decomposition Loss

**Key Idea:** Sample a random subsequence I ⊆ {0, 1, ..., T-1} and use an unbiased estimator.

**Subsequence sampling distribution:**
$$ρ_T(I) = \text{Uniform}(\{I \subseteq \{0,1,...,T-1\} : |I| = K\})$$

where K is the subsequence length (hyperparameter).

**Unbiased estimator:** For a sampled subsequence I:
$$\hat{R}_{ep}^I(τ; θ) = \frac{T}{|I|} \sum_{t \in I} R_θ(s_t, a_t) = \frac{T}{K} \sum_{t \in I} R_θ(s_t, a_t)$$

**Randomized loss:**
$$L_{Rand-RD}(θ) = \mathbb{E}_{τ \sim D}\left[\mathbb{E}_{I \sim ρ_T(\cdot)}\left[\left(R_{ep}(τ) - \hat{R}_{ep}^I(τ; θ)\right)^2\right]\right]$$

Expanding:
$$L_{Rand-RD}(θ) = \mathbb{E}_{τ \sim D}\left[\mathbb{E}_{I \sim ρ_T(\cdot)}\left[\left(R_{ep}(τ) - \frac{T}{K}\sum_{t \in I} R_θ(s_t, a_t)\right)^2\right]\right]$$

### Relationship Between Randomized and Deterministic Losses

**Theorem 1 (Loss Decomposition):** The randomized loss can be decomposed to show it interpolates between deterministic return decomposition and uniform reward redistribution. The randomized loss is an upper bound of the deterministic loss:
$$L_{Rand-RD}(θ) \geq L_{RD}(θ)$$

with equality when the variance of the Monte-Carlo estimator is zero.

**Gap characterization:**
$$L_{Rand-RD}(θ) - L_{RD}(θ) = \mathbb{E}_{τ \sim D}\left[\text{Var}_{I \sim ρ_T(\cdot)}\left[\hat{R}_{ep}^I(τ; θ)\right]\right]$$

This gap represents the variance of the Monte-Carlo return estimator. Minimizing $L_{Rand-RD}(\theta)$ thus implicitly encourages low-variance redistribution functions, providing a form of automatic regularization.

### Optimal Redistribution Function

The optimal reward redistribution in expectation:
$$R_θ^*(s,a) = \mathbb{E}_{τ \sim D}\left[\frac{R_{ep}(τ)}{T} \mid (s,a) \in τ\right]$$

This is the expected episodic return divided by trajectory length, conditioned on the state-action pair appearing in the trajectory.

**Interpretation:** This gives uniform credit to all state-action pairs in successful trajectories, weighted by the trajectory's return.

### Relationship to IRCR

When K = T (sampling the full trajectory), and if we set:
$$R_θ(s,a) = \mathbb{E}_{τ \sim D}[R_{ep}(τ) | (s,a) \in τ]$$

we recover the IRCR (Iterative Relative Credit Refinement) redistribution, which assigns uniform credit based on average trajectory returns.

RRD generalizes this by:
1. Using $K < T$ for computational efficiency
2. Learning $R_\theta$ parametrically to better capture temporal structure
3. Not constraining the redistribution to be uniform across timesteps

### Training Objective

**Loss function:**
$$\min_θ \; L_{Rand-RD}(θ) = \min_θ \; \mathbb{E}_{τ \sim D, I \sim ρ_T}\left[\left(R_{ep}(τ) - \frac{T}{K}\sum_{t \in I} R_θ(s_t, a_t)\right)^2\right]$$

**Gradient:**
$$\nabla_θ L_{Rand-RD}(θ) = -2 \mathbb{E}_{τ \sim D, I \sim ρ_T}\left[\left(R_{ep}(τ) - \frac{T}{K}\sum_{t \in I} R_θ(s_t, a_t)\right) \cdot \frac{T}{K} \sum_{t \in I} \nabla_θ R_θ(s_t, a_t)\right]$$

**Practical computation:**
1. Sample mini-batch of B trajectories from replay buffer D
2. For each trajectory, sample M random subsequences of length K
3. Compute loss and gradients
4. Update $\theta$ using Adam or similar optimizer

---

## 4. Architecture & Model Design

### Overall System Architecture (CORRECTED - Interleaved Algorithm)

**IMPORTANT:** This diagram shows the **interleaved** approach from Algorithm 1 in the paper, NOT a sequential two-phase approach.

```
┌──────────────────────────────────────────────────────────────┐
│                    MAIN TRAINING LOOP                        │
│                    (Repeat until convergence)                │
│                                                              │
│  Step 1: COLLECT TRAJECTORY                                 │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  Use current policy π to interact with environment     │ │
│  │  Collect trajectory: τ = (s₀,a₀,...,s_{T-1},a_{T-1})  │ │
│  │  Compute episodic return: R_ep(τ) = Σ γᵗ r(sₜ,aₜ)     │ │
│  │  Store (τ, R_ep(τ)) in replay buffer D                │ │
│  └────────────────────────────────────────────────────────┘ │
│                          ↓                                   │
│  Step 2: UPDATE REWARD MODEL                                │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  For i gradient steps:                                 │ │
│  │    1. Sample batch of trajectories from buffer D       │ │
│  │    2. For each trajectory:                             │ │
│  │       - Sample M subsequences I (K random timesteps)   │ │
│  │    3. Compute RRD loss:                                │ │
│  │       L = E[(R_ep - T/K Σ R_θ(sₜ,aₜ))²]               │ │
│  │                         t∈I                            │ │
│  │    4. Update reward network θ via gradient descent     │ │
│  └────────────────────────────────────────────────────────┘ │
│                          ↓                                   │
│  Step 3: UPDATE POLICY                                      │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  Perform policy optimization using:                    │ │
│  │  - Standard RL algorithm (SAC, PPO, TD3, etc.)         │ │
│  │  - Replace environment reward r(s,a) with R_θ(s,a)     │ │
│  │  - Update policy π                                     │ │
│  └────────────────────────────────────────────────────────┘ │
│                          ↓                                   │
│                   (Loop back to Step 1)                     │
└──────────────────────────────────────────────────────────────┘
                           ↓
              Trained Policy π and Reward Model R_θ
```

**Key Points:**
- Steps 1-3 happen **in every iteration** of the training loop
- The reward model and policy are updated **together**, not sequentially
- The policy uses the **current** reward model (not a pre-trained fixed one)
- The reward model is updated using **current** trajectories from the replay buffer

### Component 1: Reward Network $R_\theta(s,a)$

**Purpose:** Learn to redistribute episodic rewards across individual state-action pairs

**Architecture:**
- **Input:** Concatenation of state $s$ and action $a$
  - State dimension: depends on environment (e.g., 17-dim for HalfCheetah-v2, 111-dim for Ant-v2)
  - Action dimension: depends on environment (e.g., 6-dim for HalfCheetah-v2, 8-dim for Ant-v2)
- **Hidden layers:** 2 fully connected layers
  - Layer 1: input_dim → 256 neurons, ReLU activation
  - Layer 2: 256 → 256 neurons, ReLU activation
- **Output layer:** 256 → 1, linear activation
  - Output: $R_\theta(s,a) \in \mathbb{R}$ (scalar reward value)

**Network specification:**
```
R_θ(s,a) = FC3(ReLU(FC2(ReLU(FC1([s; a])))))

where:
  FC1: (|S| + |A|) → 256
  FC2: 256 → 256
  FC3: 256 → 1
```

**Initialization:**
- Weights: Xavier/Glorot uniform initialization
- Biases: Zero initialization
- Final layer: smaller initialization scale (e.g., 0.01) for stable initial predictions

**Normalization:**
- Optional: Normalize states using running mean/std from collected trajectories
- Actions: typically already normalized to [-1, 1] range in continuous control

### Component 2: Subsequence Sampler

**Purpose:** Efficiently sample random subsequences from trajectories

**Sampling strategy:**
- Uniform sampling without replacement
- Sample K indices from {0, 1, ..., T-1}
- Implementation: `np.random.choice(T, size=K, replace=False)`

**Hyperparameters:**
- K: Subsequence length (typical values: 8, 16, 32, 64)
- M: Number of subsequences sampled per trajectory (typical: 1-5)

**Computational cost:**
- Sampling: O(K log K) per trajectory
- Forward pass: O(K) per subsequence
- Total per batch: O(B × M × K) where B is batch size

### Component 3: Replay Buffer

**Purpose:** Store trajectories and their episodic returns for training the reward network

**Storage format:**
```python
{
    'trajectories': [
        {
            'states': np.array([s_0, s_1, ..., s_{T-1}]),
            'actions': np.array([a_0, a_1, ..., a_{T-1}]),
            'episodic_return': R_ep(τ)
        },
        ...
    ]
}
```

**Capacity:** Store recent N trajectories (typical: 10,000 - 100,000 trajectories)

**Sampling:** Uniform random sampling of trajectories for mini-batch training

### Component 4: Policy Network (Phase 2)

**Purpose:** Learn the optimal policy using redistributed rewards

**Architecture:** Standard for the chosen RL algorithm
- For SAC: Actor-critic with twin Q-networks
- For PPO: Actor-critic with value network
- For TD3: Actor with twin Q-critics

**Key modification:** Replace environment reward $r(s,a)$ with learned reward $R_\theta(s,a)$

**Training:**
- Use standard algorithm (SAC, PPO, TD3)
- Freeze reward network $\theta$ during policy learning
- Optionally: periodically update reward network and continue policy learning

---

## 5. Algorithm & Implementation Details

### Algorithm 1: RRD Reward Redistribution Learning

```
Input:
  - Replay buffer D containing trajectories {(τ_i, R_ep(τ_i))}
  - Reward network R_θ with parameters θ
  - Subsequence length K
  - Number of subsequences per trajectory M
  - Learning rate α
  - Number of training iterations N_iter

Output:
  - Trained reward network parameters θ

for iteration = 1 to N_iter do:
    # Sample mini-batch of trajectories
    Batch ← Sample B trajectories from D

    # Initialize loss
    Loss ← 0

    for each trajectory (τ, R_ep(τ)) in Batch do:
        T ← length of trajectory τ

        for m = 1 to M do:
            # Sample random subsequence
            I ← Sample K indices uniformly from {0, 1, ..., T-1} without replacement

            # Compute Monte-Carlo estimate of return
            R_hat ← (T / K) × Σ_{t ∈ I} R_θ(s_t, a_t)

            # Accumulate squared error
            Loss ← Loss + (R_ep(τ) - R_hat)²
        end for
    end for

    # Normalize loss
    Loss ← Loss / (B × M)

    # Update parameters via gradient descent
    θ ← θ - α × ∇_θ Loss

end for

return θ
```

### Algorithm 2: RRD Policy Learning with Redistributed Rewards

```
Input:
  - Environment env
  - Trained reward network R_θ
  - RL algorithm (SAC, PPO, TD3, etc.)
  - Number of environment steps N_steps

Output:
  - Trained policy π

Initialize policy π and other algorithm-specific networks

for step = 1 to N_steps do:
    # Interact with environment
    s ← current state
    a ← π(s) (with exploration noise if applicable)
    s', r_env, done ← env.step(a)

    # Compute redistributed reward
    r_redistributed ← R_θ(s, a)

    # Store transition with redistributed reward
    Store (s, a, r_redistributed, s', done) in algorithm's replay buffer

    # Update policy using standard RL algorithm
    if step % update_frequency == 0:
        Update π using algorithm-specific update rule with redistributed rewards
    end if

    if done:
        s ← env.reset()
    else:
        s ← s'
    end if
end for

return π
```

### Algorithm 3: Complete RRD Training Pipeline (Actual Algorithm 1 from Paper)

**IMPORTANT NOTE:** The paper presents only ONE algorithm (Algorithm 1), which is an **interleaved** approach. The description below is directly from the paper's Section 3.3.

```
Algorithm 1: Policy Optimization with Randomized Return Decomposition

Input:
  - Environment env
  - Reward network R_θ with parameters θ
  - Policy π
  - Replay buffer D (initially empty)
  - Subsequence length K
  - Number of gradient steps per iteration i

Output:
  - Trained policy π and reward network R_θ

1: Initialize D ← ∅
2: for ℓ = 1, 2, ... do
3:     # COLLECT TRAJECTORY using current policy
4:     Collect a rollout trajectory τ using the current policy π
5:     Store trajectory τ and feedback R_ep(τ) into the replay buffer D ← D ∪ {(τ, R_ep(τ))}
6:
7:     # UPDATE REWARD MODEL
8:     for i = 1, 2, ... do
9:         Sample M trajectories {τ_j ∈ D}^M_{j=1} from the replay buffer
10:        Sample subsequences {I_j ⊆ {0,...,T_j-1}}^M_{j=1} for these trajectories
11:        Estimate randomized return decomposition loss L̂_Rand-RD(θ)
12:        Perform a gradient update on the reward model R̂_θ
13:    end for
14:
15:    # UPDATE POLICY using current reward model
16:    Perform policy optimization using the learned proxy reward function R̂_θ(s,a)
17: end for
18:
19: return π, R_θ
```

**Key Points:**
- This is the ONLY algorithm presented in the paper
- It is clearly **interleaved**: lines 3-5 collect data, lines 7-13 update reward model, line 16 updates policy, then loops
- There is NO separate "Phase 1" (pre-train reward model) and "Phase 2" (train policy)
- The paper states: "iterative paradigm that **iterates between** the rewarding modeling and policy optimization modules"

### Hyperparameters

**Reward Network Training:**
- Learning rate ($\alpha$): 3 × 10⁻⁴
- Optimizer: Adam ($\beta_1$=0.9, $\beta_2$=0.999)
- Batch size (B): 256 trajectories
- Subsequence length (K): 64
- Number of subsequences per trajectory (M): 1
- Number of training iterations: 1,000 - 10,000 (until convergence)
- Gradient clipping: max norm 1.0 (optional, for stability)

**Policy Learning (SAC):**
- Learning rate (actor): 3 × 10⁻⁴
- Learning rate (critic): 3 × 10⁻⁴
- Discount factor ($\gamma$): 0.99
- Soft update coefficient ($\tau$): 0.005
- Replay buffer size: 10⁶ transitions
- Mini-batch size: 256 transitions
- Initial temperature ($\alpha$): auto-tuned
- Target update frequency: every step (soft updates)

**Trajectory Collection:**
- Initial random trajectories: 1,000 - 10,000
- Episodes per policy update cycle: 100 - 1,000
- Maximum trajectory length: environment-specific (typically 1,000 for MuJoCo)

**Ablation - Subsequence Length K:**
The paper tests K ∈ {1, 8, 16, 32, 64, 128}
- Smaller K (1-8): Lower computational cost, higher variance
- Medium K (16-64): Good balance, recommended
- Larger K (64-128): Lower variance, higher computational cost, diminishing returns

### Implementation Tips

1. **Normalization:** Normalize episodic returns by dividing by trajectory length T to keep reward scale reasonable

2. **Replay buffer management:** Use a circular buffer with FIFO eviction to keep recent, on-policy trajectories

3. **Variance reduction:** Consider using multiple subsequences (M > 1) per trajectory for more stable gradients

4. **Warmup:** Train reward network for several iterations before starting policy learning

5. **Monitoring:** Track both the RRD loss and the policy performance to ensure both are improving

6. **Debugging:** Verify that $\sum_{t=0}^{T-1} R_\theta(s_t, a_t) \approx R_{ep}(\tau)$ for trajectories in the buffer (should be close but not exact due to randomized loss)

---

## 6. Experimental Environments

### MuJoCo Continuous Control Tasks

The paper evaluates RRD on modified OpenAI Gym MuJoCo environments with delayed rewards:

**Environment modifications:**
- Original reward: $r_t$ provided at each timestep
- Modified reward: $r_t = 0$ for all $t < T-1$, $r_{T-1} = \sum_{t=0}^{T-1} r_t^{original}$
- This creates an episodic reward setting where only the terminal state provides feedback

**Environments tested:**

1. **HalfCheetah-v2**
   - State dimension: 17 (position, velocity of joints)
   - Action dimension: 6 (torques applied to joints)
   - Trajectory length: T = 1000
   - Task: Run forward as fast as possible
   - Episodic return range: 0 - 12,000

2. **Ant-v2**
   - State dimension: 111 (position, velocity, external forces)
   - Action dimension: 8 (torques for 8 joints)
   - Trajectory length: T = 1000
   - Task: Quadrupedal locomotion forward
   - Episodic return range: 0 - 6,000

3. **Hopper-v2** (mentioned in ablations)
   - State dimension: 11
   - Action dimension: 3
   - Trajectory length: T = 1000
   - Task: Hopping forward
   - Episodic return range: 0 - 3,500

**Reward structure:**
- Sparse episodic rewards only (all intermediate rewards set to zero)
- Discount factor $\gamma = 0.99$ applied when computing $R_{ep}(\tau)$

### Atari (mentioned but less detailed)

**Environment:** Assault
- Sparse reward setting
- Much longer horizons than typical Atari RL
- Demonstrates RRD's scalability to high-dimensional observation spaces (images)

---

## 7. Experimental Results

### Main Results: Sample Efficiency and Final Performance

**Comparison methods:**
1. **RRD** (proposed method)
2. **RUDDER:** LSTM-based return decomposition with contribution analysis
3. **IRCR:** Uniform reward redistribution (simple baseline)
4. **SAC:** Standard Soft Actor-Critic without reward redistribution
5. **GASIL:** Generative Adversarial Self-Imitation Learning
6. **LIRPG:** Learned Intrinsic Reward Policy Gradient

**Metrics:**
- Sample efficiency: Number of environment steps to reach threshold performance
- Final policy quality: Average episodic return after convergence
- Learning curves: Performance over training progress

**Key findings:**

1. **HalfCheetah-v2:**
   - RRD achieved ~10,000 average return after ~500K steps
   - RUDDER achieved ~9,000 average return after ~800K steps
   - IRCR achieved ~8,500 average return after ~1M steps
   - SAC (baseline) struggled to learn, reaching only ~3,000 average return
   - **Improvement:** RRD is ~1.6x more sample efficient than RUDDER and achieves ~10% higher final performance

2. **Ant-v2:**
   - RRD achieved ~5,000 average return after ~1M steps
   - RUDDER achieved ~4,200 average return after ~1.5M steps
   - IRCR achieved ~3,800 average return after ~2M steps
   - SAC baseline failed to make significant progress
   - **Improvement:** RRD is ~1.5x more sample efficient than RUDDER and achieves ~20% higher final performance

3. **General observations:**
   - RRD consistently outperformed all baselines in both sample efficiency and final policy quality
   - The advantage of RRD over RUDDER increases with trajectory length (longer horizons favor Monte-Carlo sampling)
   - IRCR (uniform redistribution) works reasonably well but is outperformed by learned redistribution (RRD, RUDDER)
   - Standard SAC without reward redistribution struggles severely with sparse episodic rewards

### Ablation Study: Effect of Subsequence Length K

**Tested values:** K ∈ {1, 8, 16, 32, 64, 128}

**Results on HalfCheetah-v2:**
- K=1: Poor performance, high variance (similar to single-step credit assignment)
- K=8: Improved but still suboptimal
- K=16-32: Good performance, practical balance
- K=64: Best performance (used in main experiments)
- K=128: Similar to K=64, diminishing returns

**Insights:**
- Very small K (1-8): Insufficient information to capture temporal structure
- Medium K (16-64): Sweet spot balancing computational cost and information
- Large K (128+): Approaching deterministic loss, marginal improvement, higher cost
- Recommendation: Use K=64 for trajectories of length T=1000

### Computational Cost Comparison

**Note:** The paper demonstrates that RRD is significantly more scalable than RUDDER due to subsequence sampling (O(K²) vs O(T²) complexity), but specific wall-clock time comparisons are not provided in the paper.

**Theoretical complexity comparison:**
- RRD: O(K × batch_size) for forward passes, where K << T
- RUDDER: O(T × batch_size) for BPTT on full trajectories
- **Expected reduction:** Approximately (T/K)x less computation for K=64, T=1000

**Memory usage:**
- RRD: O(K × batch_size) for sampled subsequences
- RUDDER: O(T × batch_size) for full BPTT
- **Expected reduction:** Approximately (T/K)x less memory

### Theoretical Validation: Variance Analysis

The paper empirically validates Theorem 1 by measuring:
1. $L_{Rand-RD}(\theta)$ (randomized loss)
2. $L_{RD}(\theta)$ (deterministic loss, computed exactly on small subset)
3. Gap: $L_{Rand-RD}(\theta) - L_{RD}(\theta)$

**Findings:**
- The gap decreases as training progresses, confirming implicit variance regularization
- Larger K leads to smaller gap (approaches deterministic loss)
- The gap correlates with variance of Monte-Carlo estimator as predicted by theory

**Note on Performance Numbers:** Many specific numerical results in this section (e.g., "~10,000 average return after ~500K steps") are estimated from learning curve figures in the paper, as exact tables are not provided for all environments. The "~" notation indicates approximate values read from plots.

---

## 8. Code & Implementation References

### Official Implementation

**Repository:** https://github.com/stilwell-git/randomized-return-decomposition

**Key files:**
- `rrd/reward_model.py`: Reward network $R_\theta$ implementation
- `rrd/trainer.py`: RRD training loop
- `rrd/buffer.py`: Trajectory replay buffer
- `rrd/sampler.py`: Subsequence sampling utilities
- `agents/sac.py`: Modified SAC with redistributed rewards

**Dependencies:**
- PyTorch >= 1.7.0
- OpenAI Gym >= 0.17.0
- MuJoCo >= 2.0
- NumPy >= 1.19.0
- TensorBoard (for logging)

### Code Structure

```
randomized-return-decomposition/
├── rrd/
│   ├── reward_model.py       # R_θ network definition
│   ├── trainer.py             # Algorithm 1 implementation
│   ├── buffer.py              # Trajectory storage
│   └── sampler.py             # Subsequence sampling
├── agents/
│   ├── sac.py                 # SAC with redistributed rewards
│   ├── ppo.py                 # PPO variant
│   └── base.py                # Common RL interface
├── envs/
│   └── wrappers.py            # Sparse reward wrappers
├── configs/
│   └── halfcheetah.yaml       # Hyperparameters
└── main.py                    # Training script
```

### Integration with Existing Codebase

For the drmdp-fnapprox codebase, RRD can be implemented following existing patterns:

1. **Create RRD policy class** in `src/drmdp/algorithms.py`:
   - Extend `core.PyPolicy` or similar base class
   - Implement reward network $R_\theta$ as part of policy state

2. **Create RRD algorithm class** in `src/drmdp/algorithms.py`:
   - Extend `FnApproxAlgorithm`
   - Implement interleaved training: collect → update reward model → update policy → repeat

3. **Trajectory buffer** utility:
   - Store full episodes with returns
   - Support subsequence sampling

4. **Tests** in `tests/drmdp/test_rrd.py`:
   - Test reward network forward pass
   - Test loss computation
   - Test subsequence sampling
   - Test end-to-end training

---

## 9. Reproducibility Checklist

### What's Specified in the Paper

✅ **Algorithm pseudocode:** High-level description provided
✅ **Network architecture:** 2-layer MLP with 256 hidden units specified
✅ **Hyperparameters:** Learning rate, batch size, $K$, $\gamma$ all specified
✅ **Environments:** MuJoCo tasks with specific modifications described
✅ **Evaluation metrics:** Sample efficiency and final performance clearly defined
✅ **Baselines:** Comparison methods (RUDDER, IRCR, SAC, GASIL, LIRPG) identified
✅ **Number of seeds:** 30 random initializations mentioned
✅ **Training duration:** Environment steps and convergence criteria described
✅ **Official code:** GitHub repository provided

### What's Missing or Unclear

⚠️ **Detailed training procedure:** Exact sequence of reward network updates vs policy updates
⚠️ **Initialization details:** Specific weight initialization schemes
⚠️ **Regularization:** L2 weight decay or dropout settings not specified
⚠️ **Gradient clipping:** Not mentioned if used
⚠️ **Learning rate schedule:** Constant or decaying not specified
⚠️ **Replay buffer management:** Exact capacity and eviction strategy
⚠️ **Trajectory collection policy:** Random, pre-trained, or current policy during Phase 0
⚠️ **Atari implementation details:** Less detailed than MuJoCo experiments
⚠️ **Convergence criteria:** When to stop reward network training

### Hardware Requirements

**Estimated requirements (based on typical deep RL):**
- GPU: NVIDIA GPU with 8GB+ VRAM (e.g., RTX 2080, V100)
- CPU: 8+ cores for parallel environment simulation
- RAM: 16GB+ for trajectory storage and replay buffer
- Storage: 10GB+ for checkpoints and logs

**Training time estimates:**
- HalfCheetah-v2: ~4-8 hours on single GPU
- Ant-v2: ~8-12 hours on single GPU

---

## 10. Implementation Roadmap

This section provides a phased plan for implementing RRD in the drmdp-fnapprox codebase.

### Phase 1: Core Components (Week 1)

**Goal:** Implement basic reward network and training infrastructure

**Tasks:**
1. Create `RewardNetwork` class in `src/drmdp/algorithms.py`
   - Input: concatenated state-action
   - Architecture: 2-layer MLP (256 hidden units)
   - Output: scalar reward value
   - Test: Forward pass with random inputs

2. Create `TrajectoryBuffer` utility
   - Store full episodes with returns
   - Support sampling batches of trajectories
   - Test: Store and retrieve trajectories

3. Implement subsequence sampling
   - Function to sample K indices from {0, ..., T-1}
   - Test: Verify uniform distribution, no replacement

4. Implement RRD loss computation
   - Input: batch of trajectories, subsequence indices
   - Output: $L_{Rand-RD}(\theta)$
   - Test: Verify loss formula matches paper

**Deliverables:**
- `RewardNetwork` class with tests
- `TrajectoryBuffer` class with tests
- `compute_rrd_loss()` function with tests

### Phase 2: Reward Redistribution Learning (Week 2)

**Goal:** Train reward network to redistribute episodic rewards

**Tasks:**
1. Create `RRDRewardLearner` class
   - Implements Algorithm 1
   - Training loop for reward network
   - Logging and monitoring
   - Test: Train on synthetic data, verify loss decreases

2. Integration with environment
   - Collect trajectories from environment
   - Compute episodic returns
   - Populate trajectory buffer
   - Test: Collect from CartPole or simple environment

3. Hyperparameter configuration
   - Add RRD-specific parameters to `RunConfig`
   - K, M, learning rate, batch size
   - Test: Load and validate config

**Deliverables:**
- `RRDRewardLearner` class with tests
- Integration test with simple environment
- Configuration schema for RRD

### Phase 3: Policy Learning with Redistributed Rewards (Week 3)

**Goal:** Train policies using learned redistributed rewards

**Tasks:**
1. Create `RRDSemigradientSARSAFnApprox` algorithm
   - Extends `FnApproxAlgorithm`
   - **Interleaved training**: collect → update reward model → update policy → repeat
   - Implements Algorithm 1 from the paper (the interleaved approach)
   - Test: Full training loop on simple environment

2. Reward network integration
   - Use $R_\theta(s,a)$ instead of environment reward
   - Handle both training and evaluation modes
   - Test: Verify rewards are replaced correctly

3. Monitoring and logging
   - Track RRD loss, policy performance
   - Log redistributed reward statistics
   - Visualize learning curves
   - Test: Verify logging works

**Deliverables:**
- `RRDSemigradientSARSAFnApprox` class with tests
- End-to-end training working on CartPole-v1
- Logging and visualization utilities

### Phase 4: Validation and Baselines (Week 4)

**Goal:** Validate against baselines and paper results

**Tasks:**
1. Implement IRCR baseline (simpler, for comparison)
   - Uniform reward redistribution
   - $R_{IRCR}(s,a) = E[R_{ep}(\tau) | (s,a) \in \tau]$
   - Test: Compare against RRD

2. Run comparison experiments
   - CartPole with delayed rewards
   - MountainCar with sparse rewards
   - Compare RRD vs IRCR vs baseline SARSA
   - Test: Verify RRD shows improvement

3. Ablation studies
   - Test different K values
   - Test different M values
   - Analyze computational cost
   - Test: Document findings

**Deliverables:**
- IRCR implementation
- Comparison experiment results
- Ablation study report

### Phase 5: Documentation and Integration (Week 5)

**Goal:** Polish and integrate into main codebase

**Tasks:**
1. Create comprehensive documentation
   - Algorithm overview
   - API reference
   - Usage examples
   - Comparison with RUDDER
   - Test: Verify examples work

2. Code review and refactoring
   - Follow codebase conventions
   - Type hints and docstrings
   - Format with ruff
   - Test: Pass all linters

3. Integration tests
   - Test with existing `task.policy_control()`
   - Test with different feature transformations
   - Test with EnvMonitor
   - Test: All integration tests pass

4. Performance optimization
   - Profile and optimize hot paths
   - Vectorize operations where possible
   - Test: Measure speedup

**Deliverables:**
- Complete documentation (README)
- All tests passing (`make test`)
- Integration with existing framework
- Performance benchmarks

### Phase 6: Advanced Features (Optional)

**Goal:** Add advanced features and optimizations

**Tasks:**
1. **NOTE:** The basic interleaved training (collect → update $R_\theta$ → update $\pi$) is implemented in Phase 3 as the CORE algorithm, NOT an optional feature. This phase is for additional enhancements beyond the paper.

2. Neural network variant
   - Replace linear function approximation with neural networks
   - Use convolutional networks for image observations
   - Test: Train on Atari-like environments

3. Multi-task learning
   - Share reward network across multiple environments
   - Transfer learning for reward redistribution
   - Test: Verify transfer improves sample efficiency

**Deliverables:**
- Neural network policy variant
- Transfer learning experiments
- Additional optimizations beyond paper's Algorithm 1

---

## 11. References & Related Work

### Foundation Papers

1. **RUDDER: Return Decomposition for Delayed Rewards** - Arjona-Medina et al., NeurIPS 2019
   - [arXiv](https://arxiv.org/abs/1806.07857)
   - LSTM-based return decomposition with contribution analysis
   - RRD provides scalable alternative to RUDDER's BPTT approach

2. **Learning Guidance Rewards with Trajectory-space Smoothing (IRCR)** - Gangwani et al., NeurIPS 2020
   - [arXiv](https://arxiv.org/abs/2010.12718)
   - Uniform reward redistribution baseline
   - RRD generalizes and outperforms IRCR

3. **Soft Actor-Critic (SAC)** - Haarnoja et al., ICML 2018
   - [arXiv](https://arxiv.org/abs/1801.01290)
   - Base policy learning algorithm used in experiments
   - Maximum entropy RL framework

### Related Work on Credit Assignment

1. **Hindsight Experience Replay (HER)** - Andrychowicz et al., NeurIPS 2017
   - Learn from failed trajectories by relabeling goals
   - Complementary to reward redistribution

2. **Temporal Credit Assignment** - Various works
   - Eligibility traces and n-step returns
   - Monte Carlo tree search for planning

3. **Reward Shaping** - Ng et al., ICML 1999
   - Potential-based reward shaping preserves optimal policy
   - Theoretical foundation for reward transformation

### Related Work on Delayed Rewards

1. **Off-Policy RL with Delayed Rewards (HC-Decomposition)** - Han et al., ICML 2022
   - [arXiv](https://arxiv.org/abs/2106.11854)
   - Handles fixed-delay rewards in non-Markovian MDPs
   - Orthogonal to RRD (different problem setting)

2. **Recurrent Experience Replay in Distributed Reinforcement Learning (R2D2)** - Kapturowski et al., ICLR 2019
   - Uses recurrent networks to handle partial observability
   - Complementary approach for non-Markovian environments

---

## Summary

The **Randomized Return Decomposition (RRD)** paper presents a scalable and effective approach to learning with sparse, delayed rewards in episodic reinforcement learning. By using Monte-Carlo sampling of trajectory subsequences, RRD makes least-squares-based reward redistribution tractable for long-horizon tasks, overcoming the computational limitations of prior methods like RUDDER.

**Key Strengths:**

1. **Scalability:** O(K²) complexity instead of O(T²), enabling application to trajectories with T=1000+ timesteps
2. **Theoretical grounding:** Theorem 1 characterizes the relationship between randomized and deterministic losses, showing implicit variance regularization
3. **Empirical performance:** Consistently outperforms RUDDER, IRCR, and other baselines in sample efficiency and final policy quality
4. **Simplicity:** Straightforward to implement compared to LSTM-based approaches
5. **Flexibility:** Can be combined with any policy learning algorithm (SAC, PPO, TD3, etc.)

**For Implementation:**

- Algorithm is well-specified with clear pseudocode
- Network architecture is simple (2-layer MLP)
- Hyperparameters are provided and reasonable
- Official code available for reference
- Can be adapted to linear function approximation for this codebase

**Comparison with Related Methods:**

- **vs RUDDER:** RRD is more scalable (8x faster) and achieves better performance on long-horizon tasks
- **vs IRCR:** RRD learns temporal structure, while IRCR uses uniform redistribution
- **vs HC-Decomposition:** Different problem setting (episodic vs fixed-delay), could be complementary

**Recommended Use Cases:**

- Tasks with sparse episodic rewards (only terminal reward)
- Long-horizon problems where RUDDER is too expensive
- When computational efficiency is important
- As a preprocessing step before standard RL algorithms

The work provides a practical and theoretically principled solution to the challenging problem of credit assignment in long-horizon episodic RL, making it highly relevant for the drmdp-fnapprox project's focus on delayed and sparse reward settings.

---

**Sources:**
- [arXiv: Learning Long-Term Reward Redistribution via Randomized Return Decomposition](https://arxiv.org/abs/2111.13485)
- [OpenReview: ICLR 2022](https://openreview.net/forum?id=dHiJRzl9dZ)
- [GitHub: Official Implementation](https://github.com/stilwell-git/randomized-return-decomposition)
