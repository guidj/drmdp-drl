# Summary: Learning Guidance Rewards with Trajectory-space Smoothing (IRCR)

## Metadata

**Title:** Learning Guidance Rewards with Trajectory-space Smoothing

**Authors:** Tanmay Gangwani, Yuan Zhou, Jian Peng

**Year:** 2020

**Venue:** 34th Conference on Neural Information Processing Systems (NeurIPS 2020)

**Links:**
- [arXiv Preprint](https://arxiv.org/abs/2010.12718)
- [NeurIPS Proceedings](https://proceedings.neurips.cc/paper/2020/hash/0912d0f15f1394268c66639e39b26215-Abstract.html)
- [NeurIPS Reviews](https://proceedings.neurips.cc/paper/2020/file/0912d0f15f1394268c66639e39b26215-Review.html)

---

## 1. Problem Statement & Research Question

**Problem:** Long-term temporal credit assignment is one of the most important and challenging problems in deep reinforcement learning. When the consequences of actions are delayed by many timesteps, agents struggle to determine which actions were responsible for eventual success or failure. This problem is exacerbated in tasks with:
- Sparse rewards (reward only at episode termination)
- Long horizons (hundreds to thousands of timesteps)
- Complex state-action spaces (high-dimensional continuous control)

**Existing Limitations:**

1. **Sample inefficiency:** Standard RL algorithms (DQN, DDPG, SAC, TD3) require enormous amounts of experience when rewards are sparse or delayed, often failing to learn at all.

2. **Variance in policy gradients:** When using episodic returns for credit assignment, the variance grows exponentially with episode length, making gradient estimates unreliable.

3. **Bias in temporal difference methods:** TD methods with bootstrapping accumulate bias over long horizons, leading to inaccurate value estimates.

4. **Computational cost of sophisticated methods:** Advanced credit assignment methods (e.g., RUDDER with LSTM) require training auxiliary networks and complex computations, adding significant overhead.

5. **Difficulty of reward shaping:** Manual reward shaping requires domain expertise and can accidentally change the optimal policy.

**Research Question:** Can we develop a simple, computationally lightweight algorithm that learns dense "guidance" rewards to replace sparse environmental rewards, enabling efficient credit assignment without requiring auxiliary neural networks or complex training procedures?

**Key Contribution:** The paper introduces **IRCR (Iterative Relative Credit Refinement)**, a novel algorithm that:

1. **Learns guidance rewards** r_g(s,a) through trajectory-space smoothing, providing dense feedback in place of sparse environmental rewards
2. **Uses a non-parametric approach** based on Monte Carlo estimation, avoiding the need for auxiliary neural networks
3. **Provides uniform credit assignment** by redistributing episodic returns evenly across all state-action pairs in successful trajectories
4. **Integrates easily with existing off-policy RL algorithms** (SAC, TD3, DDPG) without algorithmic modifications
5. **Demonstrates strong empirical performance** on sparse reward benchmarks including MuJoCo locomotion tasks and multi-agent environments

The key insight is that a simple, uniform redistribution of trajectory returns can be surprisingly effective, providing a strong baseline that often matches or exceeds more complex methods while being trivial to implement.

---

## 2. Methodology & Approach

**High-Level Approach:**

IRCR addresses sparse reward credit assignment through a simple two-component framework:

**Component 1: Guidance Reward Computation**
- Maintain a database of trajectories and their returns
- For each state-action pair (s,a), compute the expected return of all trajectories containing (s,a)
- Use this expected return (normalized) as the guidance reward r_g(s,a)
- This provides uniform credit across all state-action pairs in successful trajectories

**Component 2: Policy Learning with Guidance Rewards**
- Replace sparse environmental rewards with dense guidance rewards
- Train policy using any standard off-policy RL algorithm (SAC, TD3, etc.)
- As the policy improves, collect new trajectories and update guidance rewards
- Iterate until convergence

**Key Innovations:**

1. **Trajectory-space Smoothing Objective:**

   Instead of the standard RL objective $\eta(\pi_\theta) = E_{\tau \sim \pi_\theta}[R(\tau)]$, IRCR optimizes:

   $$\eta_{smooth}(\pi_\theta) = E_{\hat{\tau} \sim \pi_\theta}[E_{\tau \sim M_{\hat{\tau}}}[R(\tau)]]$$

   where $M_{\hat{\tau}}$ is a smoothing distribution over trajectories "similar" to $\hat{\tau}$.

   This smoothing has the effect of spreading credit uniformly across timesteps.

2. **Non-parametric Credit Assignment:**

   The guidance reward is computed via simple averaging:

   $$r_g(s,a; \beta) = E_{\tau \sim p_\beta(\tau; s,a)}[R(\tau)]$$

   where $p_\beta(\tau; s,a)$ is the distribution of trajectories containing $(s,a)$ under policy $\beta$.

   **Implementation:** Maintain running statistics:
   - Count(s,a): number of trajectories containing (s,a)
   - Sum(s,a): sum of returns of trajectories containing (s,a)
   - $r_g(s,a) = \text{Sum}(s,a) / \text{Count}(s,a)$

3. **Normalization to [0,1]:**

   To keep rewards on a consistent scale:

   $$\tilde{r}_g(s,a) = (r_g(s,a) - r_{min}) / (r_{max} - r_{min})$$

   where $r_{min}$ and $r_{max}$ are the minimum and maximum trajectory returns observed.

4. **Iterative Refinement:**

   - Start with random policy $\beta_0$
   - Collect trajectories, compute initial guidance rewards $r_g^0$
   - Train policy $\pi_1$ using $r_g^0$
   - Collect new trajectories with $\pi_1$, update guidance rewards to $r_g^1$
   - Repeat: $\pi_{k+1}$ learned using $r_g^k$

**Theoretical Foundation:**

The paper provides theoretical analysis showing:
- The smoothed objective $\eta_{smooth}(\pi_\theta)$ is a well-defined relaxation of the original objective
- Gradient ascent on $\eta_{smooth}$ leads to policy improvement under certain conditions
- The smoothing distribution can be viewed as an infinite mixture model
- Uniform credit assignment emerges naturally from trajectory-space smoothing

**Assumptions:**

1. **Trajectory database accessibility:** Ability to store and query past trajectories
2. **Episodic tasks:** Clear episode boundaries with well-defined returns
3. **Sufficient exploration:** Policy must explore enough to visit relevant state-action pairs
4. **Stationarity:** Environment dynamics don't change over time
5. **Reasonable trajectory length:** Episodes not so long that every (s,a) is unique

---

## 3. Mathematical Formulation

### Standard RL Objective

**MDP:** $M = (S, A, P, r, \gamma, \rho_0)$

**Policy objective:**
$$\eta(\pi_\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)] = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T-1} \gamma^t r(s_t, a_t)\right]$$

where $\tau = (s_0, a_0, s_1, a_1, ..., s_{T-1}, a_{T-1})$ is a trajectory.

**Challenge:** When $r(s_t, a_t)$ is sparse (e.g., only non-zero at $t=T-1$), the gradient estimator:
$$\nabla_\theta \eta(\pi_\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t|s_t) R(\tau)\right]$$

has variance that grows exponentially with $T$.

### Smoothed Objective

**Key idea:** Smooth the objective over trajectories to enable better credit assignment.

**Smoothed objective:**
$$\eta_{smooth}(\pi_\theta) = \mathbb{E}_{\hat{\tau} \sim \pi_\theta}\left[\mathbb{E}_{\tau \sim M_{\hat{\tau}}}[R(\tau)]\right]$$

where:
- $\hat{\tau}$ is a trajectory sampled from policy $\pi_\theta$
- $M_{\hat{\tau}}$ is a "smoothing distribution" over trajectories
- The inner expectation averages returns over trajectories similar to $\hat{\tau}$

**Smoothing distribution:**

The paper defines the smoothing distribution as a discount-weighted mixture:
$$M_{\hat{\tau},\beta}(\tau) = (1-\gamma) \sum_{t=0}^{\infty} \gamma^t p_\beta(\tau; \hat{s}_t, \hat{a}_t)$$

where:
- $p_\beta(\tau; \hat{s}_t, \hat{a}_t)$ is the distribution of trajectories from policy $\beta$ containing state-action pair $(\hat{s}_t, \hat{a}_t)$ from trajectory $\hat{\tau}$
- $\gamma$ is the discount factor
- The sum creates a weighted mixture over all timesteps in $\hat{\tau}$

**Interpretation:** This smoothing distribution creates a mixture over trajectories that share state-action pairs with the reference trajectory $\hat{\tau}$, with more recent timesteps weighted higher due to the discount factor.

**Practical instantiation:**

In practice, guidance rewards are computed from the replay buffer by averaging returns of all trajectories containing each (s,a) pair, which approximates this smoothing objective.

### Guidance Reward Function

**Definition:**

The guidance reward for a state-action pair $(s,a)$ is:
$$r_g(s,a; \beta) = \mathbb{E}_{\tau \sim p_\beta(\tau; s,a)}[R(\tau)]$$

where $p_\beta(\tau; s,a)$ is the distribution of trajectories from policy $\beta$ that contain the state-action pair $(s,a)$.

**Monte Carlo estimation:**

Given a replay buffer $D = \{\tau_1, \tau_2, ..., \tau_N\}$ with returns $\{R(\tau_1), R(\tau_2), ..., R(\tau_N)\}$:

$$r_g(s,a) = \frac{\sum_{i=1}^N R(\tau_i) \cdot \mathbb{1}[(s,a) \in \tau_i]}{\sum_{i=1}^N \mathbb{1}[(s,a) \in \tau_i]}$$

**Interpretation:** Average return of all trajectories containing (s,a).

**Normalization:**

To keep rewards in [0,1]:
$$\tilde{r}_g(s,a) = \frac{r_g(s,a) - r_{min}}{r_{max} - r_{min}}$$

where:
- $r_{min} = \min_i R(\tau_i)$
- $r_{max} = \max_i R(\tau_i)$

### Connection to Uniform Credit Assignment

**Observation:** For a trajectory $\tau$ with return $R(\tau)$, if we use guidance rewards:

Expected credit assigned to $\tau$:
$$\mathbb{E}\left[\sum_{t=0}^{T-1} r_g(s_t, a_t)\right] \approx R(\tau)$$

This means IRCR redistributes the episodic return $R(\tau)$ approximately uniformly across all $T$ timesteps in the trajectory.

**Comparison with RUDDER/RRD:**
- RUDDER/RRD: Learn temporal structure, assign higher credit to important timesteps
- IRCR: Uniform redistribution, all timesteps in successful trajectories get equal credit

**Trade-off:**
- IRCR is simpler and faster (no neural network training)
- RUDDER/RRD can be more sample-efficient when temporal structure is informative

### Policy Gradient with Guidance Rewards

Using guidance rewards as the reward signal, the policy gradient becomes:
$$\nabla_\theta \eta_{smooth}(\pi_\theta) \approx \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t|s_t) \sum_{t'=t}^{T-1} \gamma^{t'-t} \tilde{r}_g(s_{t'}, a_{t'})\right]$$

This has much lower variance than using $R(\tau)$ because guidance rewards are dense (non-zero at every timestep).

### Algorithm Convergence

**Iterative Update:**

At iteration $k$:
1. Policy $\pi_k$ learned using guidance rewards $r_g^{k-1}$
2. Collect trajectories $\mathcal{D}_k$ using $\pi_k$
3. Update guidance rewards: $r_g^k$ computed from $\mathcal{D}_0 \cup \mathcal{D}_1 \cup ... \cup \mathcal{D}_k$

**Convergence property:**

Under certain regularity conditions, the sequence $\{\pi_k\}$ converges to a policy that is a local optimum of the smoothed objective $\eta_{smooth}$.

**Note:** The smoothed objective may have different optima than the original objective $\eta$, but empirically the solutions are similar and the smoothed problem is much easier to optimize.

---

## 4. Architecture & Model Design

### Overall System Architecture

```
Environment
    ↓
Trajectory Collection
    τ = (s₀,a₀,...,s_{T-1},a_{T-1})
    R(τ) = Σ γᵗ r(sₜ,aₜ)
    ↓
Trajectory Database D
    {(τ₁, R(τ₁)), (τ₂, R(τ₂)), ...}
    ↓
┌──────────────────────────────────────┐
│  Guidance Reward Computation         │
│                                      │
│  For each (s,a):                     │
│    Count(s,a) ← # trajectories with  │
│                 (s,a)                │
│    Sum(s,a) ← Σ R(τ) for τ with     │
│               (s,a)                  │
│    r_g(s,a) ← Sum(s,a)/Count(s,a)   │
│                                      │
│  Normalize:                          │
│    r̃_g(s,a) ← (r_g - r_min) /       │
│               (r_max - r_min)        │
└──────────────────────────────────────┘
    ↓
Dense Guidance Rewards r̃_g(s,a)
    ↓
┌──────────────────────────────────────┐
│  Off-Policy RL Algorithm             │
│  (SAC, TD3, DDPG, etc.)              │
│                                      │
│  Use r̃_g(s,a) instead of r(s,a)     │
│  as reward signal                    │
│                                      │
│  Train actor π_θ and critic Q_φ      │
└──────────────────────────────────────┘
    ↓
Improved Policy π_θ
    ↓
Collect New Trajectories → Add to Database D
    ↓
Iterate (Update Guidance Rewards)
```

### Component 1: Trajectory Database

**Purpose:** Store past trajectories and their returns for computing guidance rewards

**Suggested implementation (not from paper):**
```python
class TrajectoryDatabase:
    def __init__(self):
        self.trajectories = []  # List of (states, actions, return)
        self.state_action_map = {}  # Map (s,a) → list of trajectory indices

    def add_trajectory(self, states, actions, return_value):
        traj_idx = len(self.trajectories)
        self.trajectories.append((states, actions, return_value))

        # Update state-action map
        for s, a in zip(states, actions):
            key = (s, a)  # or hash(s, a) for continuous spaces
            if key not in self.state_action_map:
                self.state_action_map[key] = []
            self.state_action_map[key].append(traj_idx)

    def get_guidance_reward(self, s, a):
        key = (s, a)
        if key not in self.state_action_map:
            return 0.0  # or default value

        traj_indices = self.state_action_map[key]
        returns = [self.trajectories[i][2] for i in traj_indices]
        return np.mean(returns)
```

**Handling continuous state-action spaces:**

**Note:** The paper does not explicitly describe how to compute guidance rewards for continuous state-action spaces. The paper uses continuous control environments (MuJoCo) with neural network policies and critics, but the guidance reward computation method for continuous (s,a) pairs is not detailed.

Possible implementation approaches (not from paper):
1. **Discretization:** Bin states and actions into discrete cells
2. **k-NN:** Use k-nearest neighbors to find similar (s,a) pairs (see Algorithm 3)
3. **Function approximation:** Learn r_g(s,a) with a neural network (hybrid approach)
4. **Replay buffer lookup:** Rely on the RL algorithm's replay buffer to provide experience, with guidance rewards computed from stored transitions

The actual implementation likely relies on the replay buffer mechanism of SAC/TD3, where guidance rewards are computed from trajectories stored as transitions.

### Component 2: Guidance Reward Normalizer

**Purpose:** Normalize guidance rewards to [0,1] range for stability

**Implementation:**
```python
class GuidanceRewardNormalizer:
    def __init__(self):
        self.r_min = float('inf')
        self.r_max = float('-inf')

    def update_statistics(self, trajectory_return):
        self.r_min = min(self.r_min, trajectory_return)
        self.r_max = max(self.r_max, trajectory_return)

    def normalize(self, r_g):
        if self.r_max == self.r_min:
            return 0.5
        return (r_g - self.r_min) / (self.r_max - self.r_min)
```

### Component 3: Off-Policy RL Algorithm

**Purpose:** Learn policy using guidance rewards

**Algorithms tested in paper:**
1. **Soft Actor-Critic (SAC)** - primary algorithm
2. **Twin Delayed DDPG (TD3)** - also tested
3. **Deep Deterministic Policy Gradient (DDPG)** - baseline

**Modification:** Replace environment reward r(s,a) with guidance reward r̃_g(s,a)

**No architectural changes needed:**
- Same actor network: $\pi_\theta(a|s)$
- Same critic network(s): $Q_\phi(s,a)$
- Same update rules

**Key integration point:** When storing transitions in replay buffer:
```python
# Original RL:
replay_buffer.add(s, a, r_env, s', done)

# IRCR:
r_guidance = guidance_reward_func(s, a)
replay_buffer.add(s, a, r_guidance, s', done)
```

### No Auxiliary Networks Required

**Key advantage of IRCR:** Unlike RUDDER (requires LSTM) or RRD (requires reward network), IRCR needs no additional neural networks.

**Computational cost:**
- Storage: O(N × T) for N trajectories of length T
- Lookup: O(1) with hashing, or O(log N) with k-NN
- Update: O(T) per new trajectory

This makes IRCR extremely lightweight compared to methods requiring auxiliary network training.

---

## 5. Algorithm & Implementation Details

### Algorithm 1: IRCR Complete Training Procedure

```
Input:
  - Environment env
  - Off-policy RL algorithm (SAC, TD3, etc.)
  - Number of initial random episodes N_init
  - Number of training episodes per iteration N_train
  - Number of iterations K

Output:
  - Trained policy π

# Initialize
trajectory_database ← empty
guidance_reward_normalizer ← GuidanceRewardNormalizer()
policy π ← random initialization
RL_agent ← initialize algorithm (SAC/TD3) with policy π

# Phase 1: Collect initial trajectories with random policy
for episode = 1 to N_init:
    τ ← collect_trajectory(env, random_policy)
    R(τ) ← compute_return(τ)
    trajectory_database.add(τ, R(τ))
    guidance_reward_normalizer.update_statistics(R(τ))
end for

# Phase 2: Iterative policy learning
for iteration = 1 to K:
    # Compute guidance rewards from current database
    guidance_reward_func ← build_guidance_reward_function(
        trajectory_database,
        guidance_reward_normalizer
    )

    # Train policy using guidance rewards
    for episode = 1 to N_train:
        # Collect trajectory
        τ ← collect_trajectory(env, π, exploration_noise)
        R(τ) ← compute_return(τ)

        # Store trajectory
        trajectory_database.add(τ, R(τ))
        guidance_reward_normalizer.update_statistics(R(τ))

        # Update policy with guidance rewards
        for step (s,a,s',done) in τ:
            r_g ← guidance_reward_func(s, a)
            RL_agent.store_transition(s, a, r_g, s', done)
        end for

        # Perform RL updates
        RL_agent.update()
    end for
end for

return π
```

### Algorithm 2: Build Guidance Reward Function

```
Input:
  - trajectory_database
  - guidance_reward_normalizer

Output:
  - guidance_reward_func: (s,a) → r̃_g

# Build state-action to return mapping
sa_to_returns ← empty dictionary

for each (τ, R(τ)) in trajectory_database:
    for each (s,a) in τ:
        if (s,a) not in sa_to_returns:
            sa_to_returns[(s,a)] ← []
        sa_to_returns[(s,a)].append(R(τ))
    end for
end for

# Compute average return for each (s,a)
sa_to_guidance ← empty dictionary
for each (s,a) in sa_to_returns:
    r_g ← mean(sa_to_returns[(s,a)])
    r̃_g ← guidance_reward_normalizer.normalize(r_g)
    sa_to_guidance[(s,a)] ← r̃_g
end for

# Create lookup function
def guidance_reward_func(s, a):
    if (s,a) in sa_to_guidance:
        return sa_to_guidance[(s,a)]
    else:
        return 0.0  # Default for unseen (s,a)
    end if
end def

return guidance_reward_func
```

### Algorithm 3 (Suggested Extension): Handle Continuous Spaces (k-NN Variant)

**Note:** This algorithm is not described in the original paper. The paper uses neural network function approximators for the policy and critic but does not explicitly describe how to handle continuous spaces for guidance reward computation. The following k-NN approach is a reasonable extension for implementation:

For continuous state-action spaces, use k-nearest neighbors:

```
Input:
  - trajectory_database
  - query state s, action a
  - number of neighbors k

Output:
  - guidance reward r̃_g(s,a)

# Collect all (s,a) pairs from database with their returns
sa_pairs ← []
for each (τ, R(τ)) in trajectory_database:
    for each (s',a') in τ:
        sa_pairs.append(((s',a'), R(τ)))
    end for
end for

# Find k nearest neighbors to (s,a)
distances ← [distance((s,a), (s',a')) for ((s',a'), R) in sa_pairs]
k_nearest_indices ← indices of k smallest distances
k_nearest_returns ← [sa_pairs[i][1] for i in k_nearest_indices]

# Average returns of k-nearest neighbors
r_g ← mean(k_nearest_returns)
r̃_g ← guidance_reward_normalizer.normalize(r_g)

return r̃_g
```

Where distance can be Euclidean distance on normalized states and actions:
```
distance((s,a), (s',a')) = ||normalize(s) - normalize(s')||₂ + ||normalize(a) - normalize(a')||₂
```

### Hyperparameters

**IRCR-specific:**
- N_init: Number of initial random episodes (100-1000)
- N_train: Episodes per iteration (10-100)
- K: Number of iterations (10-50)
- k: Number of neighbors for k-NN variant (5-50)
- Database capacity: Maximum trajectories to store (10,000-100,000)

**RL algorithm (SAC):**
- Actor learning rate: 3 × 10⁻⁴
- Critic learning rate: 3 × 10⁻⁴
- Discount factor $\gamma$: 0.99
- Soft update $\tau$: 0.005
- Replay buffer size: 10⁶ transitions
- Batch size: 256
- Network architecture: 2-layer MLP, 256 hidden units per layer

**Normalization:**
- State normalization: Running mean and std
- Action normalization: [-1, 1] clipping
- Reward normalization: [0, 1] via min-max scaling

### Implementation Tips

1. **Warm-start:** Collect diverse initial trajectories before starting policy learning

2. **Database management:** Use FIFO eviction when database reaches capacity to keep recent trajectories

3. **Default rewards:** For unseen (s,a) pairs, use:
   - Option 1: 0.0 (conservative)
   - Option 2: Mean of all guidance rewards (optimistic)
   - Option 3: k-NN even for seen pairs (smooth)

4. **Continuous spaces:**
   - Discretize if state-action space is low-dimensional
   - Use k-NN for high-dimensional spaces
   - Consider learned function approximation (hybrid IRCR) for very high dimensions

5. **Computational efficiency:**
   - Pre-compute guidance rewards before each training iteration
   - Store in hash table for O(1) lookup
   - Only recompute when database is updated

6. **Stability:**
   - Normalize states and actions before distance computation
   - Clip guidance rewards to [0,1] to prevent outliers
   - Use running statistics for normalization (update incrementally)

---

## 6. Experimental Environments

### MuJoCo Continuous Control (Primary Experiments)

**Environments tested:**

1. **HalfCheetah-v2**
   - State: 17-dim (positions and velocities)
   - Action: 6-dim continuous (joint torques)
   - Horizon: 1000 steps
   - Reward: Dense (reward at every step)
   - **Sparse variant:** Only terminal reward = sum of all step rewards

2. **Hopper-v2**
   - State: 11-dim
   - Action: 3-dim continuous
   - Horizon: 1000 steps
   - **Sparse variant:** Episode return given only at end

3. **Walker2d-v2**
   - State: 17-dim
   - Action: 6-dim continuous
   - Horizon: 1000 steps
   - **Sparse variant:** Episodic reward only

4. **Ant-v2**
   - State: 111-dim
   - Action: 8-dim continuous
   - Horizon: 1000 steps
   - **Sparse variant:** Terminal reward only

**Sparse reward modification:**
- Original: r_t provided at each timestep t
- Sparse: r_t = 0 for all t < T-1, r_{T-1} = Σ_{t=0}^{T-1} r_t^{original}

### Grid World (Simple Baseline)

**Environment:**
- 10×10 or 20×20 grid
- Goal location that provides +1 reward
- All other locations give 0 reward
- Actions: {up, down, left, right}
- Horizon: 50-100 steps

**Purpose:** Validate algorithm on simple discrete environment

### Multi-Agent Rover Domain

**Environment:**
- Multiple agents (rovers) must coordinate
- Sparse reward when all agents reach designated locations
- Continuous state and action spaces
- Demonstrates IRCR on multi-agent settings

**Challenges:**
- Credit assignment across agents
- Coordination requirements
- Exponentially large joint action space

---

## 7. Experimental Results

### Main Results: IRCR vs Baselines on Sparse MuJoCo

**Note:** The specific numerical values below are approximations estimated from learning curve figures in the paper. The paper does not provide comprehensive numerical tables for all environments. One confirmed result: HalfCheetah with TD3 achieves 6844 ± 918 average return.

**Baselines compared:**
1. **SAC** (standard, no modifications)
2. **GASIL** (Generative Adversarial Self-Imitation Learning)
3. **Reward Regression** (learn reward predictor)
4. **Evolution Strategies** (ES, gradient-free optimization)

**Metrics:**
- Sample efficiency: Steps to reach threshold performance
- Final performance: Average return after convergence
- Success rate: Percentage of random seeds that learn successfully

**Key Findings (approximate values from figures):**

**1. HalfCheetah-Sparse:**
- IRCR: Reaches ~8000 return after ~500K steps
- SAC: Struggles, reaches ~2000 return after 1M steps
- GASIL: Reaches ~6000 return after ~800K steps
- **Result:** IRCR is 1.6x more sample efficient than GASIL, 5x more than SAC

**2. Hopper-Sparse:**
- IRCR: Reaches ~2500 return after ~400K steps
- SAC: Fails to learn (stays near 0)
- GASIL: Reaches ~1500 return after ~800K steps
- **Result:** IRCR succeeds where SAC fails, 2x more efficient than GASIL

**3. Walker2d-Sparse:**
- IRCR: Reaches ~3500 return after ~600K steps
- SAC: Reaches ~1000 return after 1M steps
- **Result:** IRCR achieves 3.5x higher final performance

**4. Ant-Sparse:**
- IRCR: Reaches ~4000 return after ~1M steps
- SAC: Reaches ~500 return after 1M steps
- **Result:** IRCR achieves 8x higher final performance

**General observations:**
- IRCR consistently outperforms baselines on sparse reward tasks
- Larger improvements on harder tasks (Walker2d, Ant)
- SAC alone struggles or fails completely with sparse rewards
- IRCR provides competitive results with minimal computational overhead

### Comparison with RUDDER and RRD (from RRD paper)

**Note:** The following comparisons are from the RRD paper (Ren et al., ICLR 2022), which used IRCR as a baseline. Specific performance improvements are approximate.

When RRD paper compared against IRCR:
- **RRD outperforms IRCR** in most tasks (better final performance)
- **RRD is more sample efficient** (faster convergence)
- **IRCR is computationally cheaper** (no neural network training for reward redistribution)

**Trade-off:**
- IRCR: Simpler, faster to compute, but less effective at capturing temporal structure
- RRD/RUDDER: More sophisticated, better performance, but higher computational cost

**When to use IRCR:**
- Quick baseline for sparse reward problems
- Computational resources are limited
- Temporal structure is less important (rewards roughly uniform across trajectory)
- Simplicity and ease of implementation are priorities

### Ablation Studies

**Note:** The paper includes some ablation studies, but the specific numerical ablations below may be inferred or from supplementary materials. The main confirmed finding is that iterative refinement (updating guidance rewards with new trajectories) is important.

**Confirmed from paper:**

**1. Iterative Refinement vs One-shot:**
- Iterative refinement: Significantly better than using fixed initial database
- **Insight:** Updating guidance rewards with on-policy data is crucial for performance

**Suggested ablations for implementation (not explicitly in paper):**

**2. Effect of Database Size:**
- The paper specifies replay buffer sizes (e.g., 3×10⁵ transitions + 10 episodes for SAC)
- Larger databases generally provide better guidance rewards up to a point
- **Recommendation:** Follow paper's buffer specifications for each algorithm

**3. Effect of k (for k-NN variant):**
- Since k-NN is not in the original paper, these are suggested experimental values
- k=5-20: Likely good balance between variance and bias
- **Recommendation:** Tune k empirically if using k-NN approach

**4. Effect of Normalization:**
- The paper uses min-max normalization to [0,1]
- **Recommendation:** Use min-max [0,1] normalization as specified in paper

### Multi-Agent Rover Domain Results

- IRCR successfully coordinates multiple agents
- Baseline SAC fails to discover coordination strategy
- IRCR enables credit assignment across agents
- **Result:** Demonstrates applicability beyond single-agent RL

---

## 8. Code & Implementation References

### Official Implementation

**Note:** The paper does not provide an official GitHub repository.

**Referenced Implementations:**
- Algorithm described in sufficient detail to implement from scratch
- Can be integrated into existing SAC/TD3 codebases with minimal changes

### Pseudo-Implementation in Python

```python
import numpy as np
from collections import defaultdict

class IRCR:
    def __init__(self, rl_algorithm):
        self.rl_algorithm = rl_algorithm
        self.trajectory_db = []
        self.r_min = float('inf')
        self.r_max = float('-inf')

    def add_trajectory(self, states, actions, trajectory_return):
        """Add trajectory to database and update statistics."""
        self.trajectory_db.append({
            'states': states,
            'actions': actions,
            'return': trajectory_return
        })
        self.r_min = min(self.r_min, trajectory_return)
        self.r_max = max(self.r_max, trajectory_return)

    def compute_guidance_rewards(self):
        """Compute guidance reward for each (s,a) pair."""
        sa_returns = defaultdict(list)

        # Aggregate returns for each (s,a)
        for traj in self.trajectory_db:
            for s, a in zip(traj['states'], traj['actions']):
                key = self._discretize(s, a)  # Hash or discretize
                sa_returns[key].append(traj['return'])

        # Compute average and normalize
        guidance_rewards = {}
        for key, returns in sa_returns.items():
            r_g = np.mean(returns)
            r_g_norm = self._normalize(r_g)
            guidance_rewards[key] = r_g_norm

        return guidance_rewards

    def _normalize(self, r_g):
        """Normalize reward to [0,1]."""
        if self.r_max == self.r_min:
            return 0.5
        return (r_g - self.r_min) / (self.r_max - self.r_min)

    def _discretize(self, s, a):
        """Convert continuous (s,a) to discrete key."""
        # Simple binning example
        s_binned = tuple(np.floor(s).astype(int))
        a_binned = tuple(np.floor(a).astype(int))
        return (s_binned, a_binned)

    def train(self, env, n_iterations, n_episodes_per_iter):
        """Main IRCR training loop."""
        for iteration in range(n_iterations):
            # Compute current guidance rewards
            guidance_rewards = self.compute_guidance_rewards()

            # Collect episodes and train policy
            for episode in range(n_episodes_per_iter):
                states, actions, trajectory_return = self._collect_episode(
                    env, guidance_rewards
                )
                self.add_trajectory(states, actions, trajectory_return)

        return self.rl_algorithm.get_policy()

    def _collect_episode(self, env, guidance_rewards):
        """Collect one episode using current policy and guidance rewards."""
        states, actions = [], []
        s = env.reset()
        done = False

        while not done:
            a = self.rl_algorithm.select_action(s)
            s_next, r_env, done, _ = env.step(a)

            # Use guidance reward instead of environment reward
            key = self._discretize(s, a)
            r_g = guidance_rewards.get(key, 0.0)

            # Store transition with guidance reward
            self.rl_algorithm.store_transition(s, a, r_g, s_next, done)
            self.rl_algorithm.update()

            states.append(s)
            actions.append(a)
            s = s_next

        # Compute true trajectory return for database
        trajectory_return = env.get_episode_return()

        return states, actions, trajectory_return
```

### Integration with drmdp-fnapprox Codebase

**Recommended structure:**

1. **Create IRCR policy class** in `src/drmdp/algorithms.py`:
   - `IRCRLinearFnApproxPolicy` extending `core.PyPolicy`
   - Maintain trajectory database
   - Compute guidance rewards

2. **Create IRCR algorithm class** in `src/drmdp/algorithms.py`:
   - `IRCRSemigradientSARSAFnApprox` extending `FnApproxAlgorithm`
   - Implement iterative training loop

3. **Utility classes:**
   - `TrajectoryDatabase`: Store and query trajectories
   - `GuidanceRewardComputer`: Compute r_g from database

4. **Tests** in `tests/drmdp/test_ircr.py`:
   - Test trajectory storage
   - Test guidance reward computation
   - Test normalization
   - Test end-to-end training

---

## 9. Reproducibility Checklist

### What's Specified

✅ **Algorithm description:** Clear pseudocode and mathematical formulation
✅ **Environments:** MuJoCo tasks with sparse reward modifications described
✅ **Hyperparameters:** SAC parameters provided
✅ **Evaluation metrics:** Sample efficiency and final performance specified
✅ **Baselines:** Comparison methods clearly identified
✅ **Normalization:** Min-max to [0,1] described

### What's Missing or Unclear

⚠️ **Exact discretization scheme:** How to discretize continuous (s,a) for exact matching
⚠️ **k-NN details:** Exact distance metric and normalization for k-NN variant
⚠️ **Database capacity:** Exact number of trajectories stored not specified
⚠️ **Iteration schedule:** How many iterations and episodes per iteration
⚠️ **Initialization policy:** Random vs pre-trained for initial trajectory collection
⚠️ **Code availability:** No official implementation released
⚠️ **Default reward value:** What to use for unseen (s,a) pairs
⚠️ **Computational cost:** Wall-clock time and memory usage not reported

### Hardware Requirements

**Estimated requirements:**
- GPU: Not strictly required (no neural network for reward redistribution)
- CPU: 4-8 cores sufficient
- RAM: 8-16GB (for trajectory database storage)
- Storage: 5-10GB

**Training time estimates:**
- Simpler than RRD/RUDDER (no auxiliary network training)
- HalfCheetah: ~2-4 hours on CPU
- Ant: ~6-10 hours on CPU

---

## 10. Implementation Roadmap

### Phase 1: Core Components (Week 1)

**Goal:** Implement trajectory database and guidance reward computation

**Tasks:**
1. Create `TrajectoryDatabase` class
   - Store trajectories with returns
   - Query by (s,a) pair
   - Test: Add and retrieve trajectories

2. Create `GuidanceRewardComputer` class
   - Compute r_g(s,a) from database
   - Normalization to [0,1]
   - Test: Verify correct averaging

3. Handle discrete and continuous spaces
   - Discretization for continuous (s,a)
   - k-NN variant for better generalization
   - Test: Both variants work correctly

**Deliverables:**
- `TrajectoryDatabase` with tests
- `GuidanceRewardComputer` with tests
- Discretization utilities with tests

### Phase 2: IRCR Algorithm Integration (Week 2)

**Goal:** Integrate IRCR with existing RL algorithm

**Tasks:**
1. Create `IRCRLinearFnApproxPolicy`
   - Extends `core.PyPolicy`
   - Includes guidance reward lookup
   - Test: Policy initialization and action selection

2. Create `IRCRSemigradientSARSAFnApprox`
   - Extends `FnApproxAlgorithm`
   - Implements iterative IRCR training
   - Test: Single iteration completes successfully

3. Replace environment rewards with guidance rewards
   - Modify reward signal in training loop
   - Test: Verify guidance rewards are used

**Deliverables:**
- `IRCRLinearFnApproxPolicy` with tests
- `IRCRSemigradientSARSAFnApprox` with tests
- Integration tests

### Phase 3: Validation on Simple Environments (Week 3)

**Goal:** Validate IRCR on simple sparse reward tasks

**Tasks:**
1. Create sparse reward environments
   - CartPole with terminal reward only
   - GridWorld with goal reward
   - Test: Environments work correctly

2. Train IRCR on simple tasks
   - CartPole-sparse
   - GridWorld-sparse
   - Test: IRCR learns successfully

3. Compare against baseline SARSA
   - Run same tasks with standard SARSA
   - Verify IRCR outperforms on sparse rewards
   - Test: Statistical significance

**Deliverables:**
- Sparse reward environments
- IRCR training successful on simple tasks
- Comparison results documented

### Phase 4: Comparison with RRD (Week 4)

**Goal:** Compare IRCR against RRD implementation

**Tasks:**
1. Run both IRCR and RRD on same tasks
   - CartPole-sparse
   - MountainCar-sparse
   - Test: Both complete training

2. Analyze performance differences
   - Sample efficiency
   - Final performance
   - Computational cost
   - Test: Collect metrics

3. Document trade-offs
   - When to use IRCR vs RRD
   - Computational vs performance trade-off
   - Test: Create comparison guide

**Deliverables:**
- IRCR vs RRD comparison results
- Performance analysis report
- Usage recommendations

### Phase 5: Documentation and Polish (Week 5)

**Goal:** Finalize implementation and documentation

**Tasks:**
1. Create comprehensive documentation
   - Algorithm overview
   - API reference
   - Usage examples
   - Test: Examples run successfully

2. Code cleanup
   - Follow codebase style
   - Add type hints
   - Format with ruff
   - Test: Linting passes

3. Integration with existing framework
   - Work with `task.policy_control()`
   - Work with feature transformations
   - Test: Integration tests pass

**Deliverables:**
- Complete documentation
- Clean, tested code
- Full integration with codebase

---

## 11. References & Related Work

### Foundation Papers

1. **Soft Actor-Critic (SAC)** - Haarnoja et al., ICML 2018
   - [arXiv](https://arxiv.org/abs/1801.01290)
   - Off-policy RL algorithm used with IRCR

2. **Twin Delayed DDPG (TD3)** - Fujimoto et al., ICML 2018
   - [arXiv](https://arxiv.org/abs/1802.09477)
   - Alternative off-policy algorithm compatible with IRCR

### Related Work on Credit Assignment

1. **RUDDER: Return Decomposition for Delayed Rewards** - Arjona-Medina et al., NeurIPS 2019
   - [arXiv](https://arxiv.org/abs/1806.07857)
   - More sophisticated but computationally expensive alternative

2. **Randomized Return Decomposition (RRD)** - Ren et al., ICLR 2022
   - [arXiv](https://arxiv.org/abs/2111.13485)
   - Scalable alternative to RUDDER, outperforms IRCR
   - Uses IRCR as baseline comparison

3. **Hindsight Experience Replay (HER)** - Andrychowicz et al., NeurIPS 2017
   - Different approach to sparse rewards via goal relabeling
   - Complementary to IRCR

### Related Work on Reward Shaping

1. **Potential-Based Reward Shaping** - Ng et al., ICML 1999
   - Theoretical foundation for reward transformation
   - IRCR can be viewed as learned reward shaping

2. **Curiosity-Driven Exploration** - Pathak et al., ICML 2017
   - Intrinsic motivation for exploration
   - Orthogonal to IRCR's credit assignment

---

## Summary

**Learning Guidance Rewards with Trajectory-space Smoothing (IRCR)** provides a simple, effective approach to sparse reward credit assignment through uniform redistribution of trajectory returns. The key insight is that assigning average trajectory returns to all state-action pairs in successful trajectories often works surprisingly well, despite ignoring temporal structure.

**Key Strengths:**

1. **Simplicity:** No auxiliary networks, just averaging and normalization
2. **Ease of implementation:** ~100 lines of code to integrate with existing RL
3. **Computational efficiency:** Minimal overhead compared to standard RL
4. **Broad applicability:** Works with any off-policy algorithm (SAC, TD3, DDPG)
5. **Strong baseline:** Often matches or exceeds more complex methods

**Limitations:**

1. **Ignores temporal structure:** Uniform redistribution may be suboptimal when some timesteps are clearly more important
2. **Outperformed by RRD/RUDDER:** More sophisticated methods achieve better sample efficiency
3. **Continuous space challenges:** Requires discretization or k-NN, which can be brittle
4. **Database storage:** Requires storing many trajectories

**For Implementation:**

- Extremely straightforward to implement
- Good starting point for sparse reward problems
- Provides strong baseline for comparison with more complex methods (RRD, RUDDER)
- Can be implemented in linear FA setting for drmdp-fnapprox

**Comparison with Related Methods:**

- **vs RRD:** IRCR is simpler and faster but less sample-efficient
- **vs RUDDER:** IRCR is much simpler but less effective at capturing temporal credit
- **vs Standard RL:** IRCR dramatically improves performance on sparse reward tasks

**Recommended Use Cases:**

- Initial baseline for sparse reward problems
- When computational resources are limited
- When simplicity is prioritized over optimal performance
- As a debugging tool (if IRCR works but RRD doesn't, issue may be in RRD implementation)

The work demonstrates that simple, uniform credit assignment can be remarkably effective, providing a valuable baseline and practical tool for sparse reward reinforcement learning problems.

---

**Sources:**
- [arXiv: Learning Guidance Rewards with Trajectory-space Smoothing](https://arxiv.org/abs/2010.12718)
- [NeurIPS 2020 Proceedings](https://proceedings.neurips.cc/paper/2020/hash/0912d0f15f1394268c66639e39b26215-Abstract.html)
- [NeurIPS 2020 Reviews](https://proceedings.neurips.cc/paper/2020/file/0912d0f15f1394268c66639e39b26215-Review.html)
