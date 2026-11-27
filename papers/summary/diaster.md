# Summary: Episodic Return Decomposition by Difference of Implicitly Assigned Sub-trajectory Reward (Diaster)

## Metadata

**Title**: Episodic Return Decomposition by Difference of Implicitly Assigned Sub-trajectory Reward

**Authors**: Haoxin Lin, Hongqiu Wu, Jiaji Zhang, Yihao Sun, Junyin Ye, Yang Yu

**Year**: 2024

**Venue**: The Thirty-Eighth AAAI Conference on Artificial Intelligence (AAAI-24)

**Repository**: https://github.com/HxLyn3/Diaster

**Paper Pages**: 13808-13816

---

## Problem Statement & Research Question

**Problem**: Real-world reinforcement learning (RL) applications often involve **episodic rewards** where feedback is only available at the end of an episode (extremely delayed case). This creates two critical challenges:
- High bias in Temporal-Difference (TD) learning
- High variance in Monte-Carlo (MC) learning
- Severe sample inefficiency when learning value functions

**Existing Limitations**:
- Hand-crafted reward shaping requires domain knowledge and is task-specific
- Potential-based reward shaping (PBRS) couples reshaped rewards with original rewards, making them ineffective when only episodic rewards are available
- Previous return decomposition methods suffer from either:
  - **Lack of attribution**: Methods like RUDDER focus on predicting long-term returns without proper reward redistribution to timesteps
  - **Lack of representation capacity**: Methods focusing on step-wise decomposition without considering temporal structure
- These limitations make decomposition inefficient for long-horizon episodes

**Research Question**: How can we design an episodic return decomposition method that combines both **temporal structural representation** and **proper attribution** of rewards to trajectory segments?

**Key Contribution**: Diaster proposes a novel approach that decomposes episodic returns into rewards of **two sub-trajectories at any cut point**, extracting step-wise proxy rewards via expectation differences. This balances attribution (why different parts matter) and representation (learning temporal structure through RNNs).

---

## Methodology & Approach

**High-Level Approach**: 

Instead of directly decomposing episodic rewards into step-wise rewards, Diaster:
1. Learns a **sub-trajectory reward function** $\hat{R}_{\text{sub}}$ that assigns a single reward value to any sub-trajectory
2. For any cut point $c$ in a trajectory, decomposes the episodic return as: $\hat{R}_{\text{sub}}(\tau_{0:c}) + \hat{R}_{\text{sub}}(\tau_{c:T}) \approx R_{\text{ep}}(\tau_{0:T})$
3. Derives step-wise proxy rewards as **differences** of expected sub-trajectory rewards: $\hat{r}(s_t, a_t) = \mathbb{E}[\hat{R}_{\text{sub}}(\tau_{0:t+1}) - \hat{R}_{\text{sub}}(\tau_{0:t})]$

**Key Innovations**:
1. **Sub-trajectory level decomposition**: Rather than step-level decomposition, operates at the sub-trajectory level, providing more training signals (T cut points) and easing credit assignment
2. **Two-component decomposition**: Only two sub-trajectory rewards need assignment at each cut point, reducing complexity
3. **RNN-based temporal representation**: Uses RNNs (specifically GRU cells) to encode temporal structure within sub-trajectories, capturing sequential information
4. **Difference-based proxy rewards**: Step-wise rewards derived via expectation differences ensure Markovian, step-wise guidance for policy optimization
5. **Flexible sub-trajectory reward definition**: Unlike step-wise methods that assume rewards sum over steps, allows sub-trajectory rewards to be arbitrary functions of trajectories

**Theoretical Foundation**: 
- Based on the principle that the sum of step-wise rewards should decompose episodic returns: $R_{\text{ep}}(\tau_{0:T}) \approx \sum_{t=0}^{T-1} \hat{r}(s_t, a_t)$
- Extends this to sub-trajectories: $R_{\text{ep}}(\tau_{0:T}) \approx \hat{R}_{\text{sub}}(\tau_{0:c}) + \hat{R}_{\text{sub}}(\tau_{c:T})$ for any cut point $c$
- Conditions reward aggregation to states to ensure policy optimality (Lemma 1)

**Assumptions Made**:
1. Episodic reward is the sum of unobservable step-wise rewards: $R_{\text{ep}}(\tau_{0:T}) = \sum_{t=0}^{T-1} r(s_t, a_t)$ (satisfied in most environments)
2. Finite-horizon episodic MDPs with bounded horizon $T$
3. Access to collected trajectories with their episodic rewards
4. Can learn sub-trajectory reward function through supervised learning

---

## Mathematical Formulation

### Core Problem Setup

For a finite-horizon MDP with episodic-reward setting $\langle S, A, P, \rho_0, r, R_{\text{ep}}, T \rangle$:

$$
Q^\pi(s_t, a_t) = \mathbb{E}_{P,\pi}\left[\sum_{i=0}^{T-t-1} r(s_{t+i}, a_{t+i}) \bigg| s_t, a_t\right]
$$

Cannot be estimated directly since immediate reward $r(s_t, a_t)$ is unobservable. The goal is finding proxy reward function $\hat{r}(s, a)$ satisfying:

$$
R_{\text{ep}}(\tau_{0:T}) \approx \hat{R}_{\text{ep}}(\tau_{0:T}) = \sum_{t=0}^{T-1} \hat{r}(s_t, a_t)
$$

### Sub-trajectory Reward Decomposition

Learn a sub-trajectory reward function $\hat{R}_{\text{sub}}$ satisfying three conditions:

$$
\hat{R}_{\text{sub}}(\emptyset) = 0
$$

$$
\hat{R}_{\text{sub}}(\tau_{0:T}) \approx R_{\text{ep}}(\tau_{0:T})
$$

$$
\hat{R}_{\text{sub}}(\tau_{0:c}) + \hat{R}_{\text{sub}}(\tau_{c:T}) \approx R_{\text{ep}}(\tau_{0:T}), \quad \forall c \in \{1, \ldots, T-1\}
$$

Where:
- $\tau_{0:c}$ = former $c$-step sub-trajectory
- $\tau_{c:T}$ = latter $(T-c)$-step sub-trajectory
- $\tau_{0:0} = \emptyset$ = empty trajectory

### Step-wise Proxy Reward Definition

From differences of sub-trajectory rewards:

$$
\hat{r}_d(s_0, a_0) = \hat{R}_{\text{sub}}(\tau_{0:1}) - \hat{R}_{\text{sub}}(\emptyset)
$$

$$
\hat{r}_d(\tau_{0:t}, s_t, a_t) = \hat{R}_{\text{sub}}(\tau_{0:t+1}) - \hat{R}_{\text{sub}}(\tau_{0:t})
$$

This ensures decomposition preserves episodic return:

$$
\sum_{t=0}^{T-1} \hat{r}_d(\tau_{0:t}, s_t, a_t) = \hat{R}_{\text{sub}}(\tau_{0:T}) \approx R_{\text{ep}}(\tau_{0:T})
$$

However, $\hat{r}_d$ is still delayed and non-Markovian. Define the **Markovian step-wise proxy reward**:

$$
\hat{r}^\pi(s_t, a_t) = \mathbb{E}_{\tau_{0:t} \sim \Gamma^\pi_t(\cdot|s_t)} \left[\hat{r}_d(\tau_{0:t}, s_t, a_t)\right]
$$

Where the conditioning distribution is:

$$
\Gamma^\pi_t(\tau_{0:t}|s_t) = \frac{\mathcal{T}^\pi_t(\tau_{0:t})P(s_t|s_{t-1}, a_{t-1})}{\rho^\pi(s_t)}
$$

And the unconditioned distribution:

$$
\mathcal{T}^\pi_t(\tau_{0:t}) = \rho_0(s_0)\pi(a_0|s_0) \prod_{i=1}^{t-1} P(s_i|s_{i-1}, a_{i-1})\pi(a_i|s_i)
$$

### Proxy Value Function

Define proxy state-action value function based on proxy rewards:

$$
\hat{Q}^\pi(s_t, a_t) = \mathbb{E}_{P,\pi}\left[\sum_{i=0}^{T-t-1} \hat{r}^\pi(s_{t+i}, a_{t+i}) \bigg| s_t, a_t\right]
$$

---

## Architecture & Model Design

### Architecture Overview

Diaster uses a **two-network architecture**:

```
Trajectory τ₀:T with episodic return Rₑₚ(τ₀:T)
    ↓
[RNN with GRU cells (ψ parameters)]
    ↓
Sub-trajectory Reward Function R̂ψsub(τ₀:c)
    ↓ [Differences at each cut point c]
    ↓
[Neural Network (φ parameters)]
    ↓
Step-wise Proxy Reward r̂φ(st, at)
    ↓ [Feed to RL algorithm]
Policy πθ
```

### Component Descriptions

**Component 1: Sub-trajectory Reward Network $\hat{R}^\psi_{\text{sub}}$**

- **Purpose**: Learn to assign a scalar reward value to any sub-trajectory
- **Architecture**: 
  - RNN with GRU (Gated Recurrent Unit) cells
  - Input: Sequence of (state, action) pairs forming a sub-trajectory
  - Hidden state dimension: User-defined (typically 64-256)
  - Output: Single scalar representing cumulative sub-trajectory reward
  - Parameters: $\psi$ (RNN weights)
- **Key feature**: Captures temporal structure and dependencies within sub-trajectories through recurrent processing
- **Training**: Minimize episodic return decomposition loss (Equation 19)

**Component 2: Step-wise Reward Prediction Network $\hat{r}^\phi$**

- **Purpose**: Learn to predict step-wise proxy rewards as differences of sub-trajectory rewards
- **Architecture**:
  - Multi-layer perceptron or simple function
  - Input: Current state $s_t$ and action $a_t$
  - Output: Single scalar representing step-wise proxy reward
  - Parameters: $\phi$ (neural network weights)
- **Key feature**: Creates Markovian (state-action dependent only), step-wise guidance for policy optimization
- **Training**: Minimize prediction loss against differences of sub-trajectory rewards (Equation 21)

### Model Specifications

| Specification | Value | Notes |
|--------------|-------|-------|
| Input (Sub-trajectory network) | Variable-length trajectory | Processed sequentially by RNN |
| Hidden dimension (RNN) | Problem-dependent | Typically 64-256 for MuJoCo tasks |
| Output (Sub-trajectory) | Scalar (1D) | Single reward value per sub-trajectory |
| Input (Step-wise network) | State + Action | Concatenated features |
| Output (Step-wise) | Scalar (1D) | Single reward per step |
| Total parameters | Problem-dependent | Typically thousands for continuous control |

### Data Flow

1. **Forward pass on trajectory**:
   - Encode trajectory $\tau_{0:T}$ through RNN to get $\hat{R}^\psi_{\text{sub}}(\tau_{0:T})$
   - For each cut point $c$: encode $\tau_{0:c}$ and $\tau_{c:T}$ separately
   - Compute loss: $({\hat{R}^\psi_{\text{sub}}(\tau_{0:c}) + \hat{R}^\psi_{\text{sub}}(\tau_{c:T}) - R_{\text{ep}}(\tau_{0:T})})^2$

2. **Step-wise reward prediction**:
   - For each transition $(s_t, a_t)$: predict $\hat{r}^\phi(s_t, a_t)$
   - Target: $\hat{R}^\psi_{\text{sub}}(\tau_{0:t+1}) - \hat{R}^\psi_{\text{sub}}(\tau_{0:t})$
   - Compute loss: $(\hat{r}^\phi(s_t, a_t) - (\hat{R}^\psi_{\text{sub}}(\tau_{0:t+1}) - \hat{R}^\psi_{\text{sub}}(\tau_{0:t})))^2$

3. **RL optimization**:
   - Use proxy rewards $\hat{r}^\phi(s_t, a_t)$ with any off-policy RL algorithm (e.g., SAC)

---

## Algorithm & Implementation Details

### Main Algorithm

```
Algorithm: Diaster

Input:
  - Neural parameters ψ (sub-trajectory network), φ (step-wise network)
  - Replay buffer D
  - Learning rates λR̂, λr̂
  - Batch size B
  - RL algorithm RLALGO (e.g., SAC with policy πθ)

For N episodes:
  1. Collect trajectory τ₀:T using policy πθ
  2. Store τ₀:T and episodic return Rₑₚ(τ₀:T) in D
  
  For M batches:
    3. Sample B trajectories {τᵢ₀:T} from D
    4. Sample B random cut points {cᵢ} from {0, ..., T-1}
    
    5. Update sub-trajectory network ψ:
       ψ ← ψ - ∇ψ (1/B) Σ lR̂(ψ, τᵢ₀:T, cᵢ)
       where lR̂(ψ, τ₀:T, c) = (R̂ψsub(τ₀:c) + R̂ψsub(τc:T) - Rₑₚ(τ₀:T))²
    
    6. Update step-wise network φ:
       φ ← φ - ∇φ (1/B) Σ lr̂(φ, τᵢ₀:cᵢ₊₁)
       where lr̂(φ, τ₀:t+1) = (r̂φ(st, at) - (R̂ψsub(τ₀:t+1) - R̂ψsub(τ₀:t)))²
    
    7. Sample B transitions {(sᵢ, aᵢ, sᵢ₊₁)} from D
    8. Optimize policy πθ with RL-ALGO using proxy rewards {r̂φ(sᵢ, aᵢ)}
```

### Training Procedure

1. **Data Collection**: 
   - Use current policy $\pi_\theta$ to collect episodes
   - Store complete trajectories and their episodic rewards (only feedback available)

2. **Sub-trajectory Reward Function Learning**:
   - Randomly sample trajectories from replay buffer
   - For each trajectory, randomly sample a cut point
   - Forward pass: encode sub-trajectories through RNN
   - Compute decomposition loss (Equation 19)
   - Backpropagate and update network $\psi$ with learning rate $\lambda_{\hat{R}}$

3. **Step-wise Reward Function Learning**:
   - For each trajectory, sample a time step
   - Predict step-wise reward with network $\phi$
   - Target: difference of sub-trajectory rewards (requires forward passes on $\tau_{0:t}$ and $\tau_{0:t+1}$)
   - Backpropagate and update network $\phi$ with learning rate $\lambda_{\hat{r}}$

4. **Policy Optimization**:
   - Sample transitions from replay buffer
   - Replace environment rewards with learned proxy rewards $\hat{r}^\phi(s_t, a_t)$
   - Apply standard RL algorithm (SAC in experiments) using these proxy rewards
   - Update policy network $\theta$

5. **Repeat**: Continue collecting data with updated policy

### Loss Functions

**Sub-trajectory Decomposition Loss**:

$$
\mathcal{L}_{\hat{R}}(\psi) = \mathbb{E}_{\tau_{0:T} \in D} \left[\mathbb{E}_{c \sim \mathcal{U}(\{0,\ldots,T-1\})} \left[l_{\hat{R}}(\psi, \tau_{0:T}, c)\right]\right]
$$

$$
l_{\hat{R}}(\psi, \tau_{0:T}, c) = \left(\hat{R}^\psi_{\text{sub}}(\tau_{0:c}) + \hat{R}^\psi_{\text{sub}}(\tau_{c:T}) - R_{\text{ep}}(\tau_{0:T})\right)^2
$$

**Step-wise Reward Prediction Loss**:

$$
\mathcal{L}_{\hat{r}}(\phi) = \mathbb{E}_{\tau_{0:T} \in D} \left[\mathbb{E}_{t \sim \mathcal{U}(\{0,\ldots,T-1\})} \left[l_{\hat{r}}(\phi, \tau_{0:t+1})\right]\right]
$$

$$
l_{\hat{r}}(\phi, \tau_{0:t+1}) = \left(\hat{r}^\phi(s_t, a_t) - \left(\hat{R}^\psi_{\text{sub}}(\tau_{0:t+1}) - \hat{R}^\psi_{\text{sub}}(\tau_{0:t})\right)\right)^2
$$

### Hyperparameters

| Hyperparameter | Value | Notes |
|---------------|-------|-------|
| Number of cut points per trajectory | 1 (random) | For efficiency; can sample multiple |
| RNN hidden dimension | 64-256 | Task and trajectory length dependent |
| Learning rate $\lambda_{\hat{R}}$ | 3e-4 | For sub-trajectory network (Adam optimizer) |
| Learning rate $\lambda_{\hat{r}}$ | 3e-4 | For step-wise network (Adam optimizer) |
| Batch size | 256 | For neural network training |
| Replay buffer capacity | 1M steps | Store entire episodes |
| RL algorithm | SAC | Soft Actor-Critic (off-policy) |
| RL learning rate | 3e-4 | For policy and value networks in SAC |

### Implementation Notes

**Architecture Choices**:
- **GRU over LSTM**: GRU used for computational efficiency while capturing temporal dependencies
- **Single vs. Multiple Cut Points**: Paper samples one random cut point per batch update; could sample multiple for more training signal but increases computation
- **Separate Networks**: Two separate networks $\hat{R}^\psi_{\text{sub}}$ and $\hat{r}^\phi$ allow independent optimization and interpretation

**Stability Techniques**:
- **Expectation through Conditioning**: Taking expectation over sub-trajectories (Equation 9) ensures Markovian property, enabling stable policy optimization
- **Simple Loss Functions**: MSE losses are stable compared to divergence-based losses
- **Off-policy Training**: Using SAC allows learning from past trajectories without policy constraints

**Computational Complexity**:
- **Time Complexity per Batch**:
  - Sub-trajectory forward pass: $O(T \cdot H)$ where $H$ is hidden dimension
  - Step-wise learning: $O(T \cdot D)$ where $D$ is network width
  - Overall: $O(B \cdot T \cdot \max(H, D))$ for batch size $B$
- **Space Complexity**: $O(T \cdot B)$ for storing trajectories in replay buffer

**Key Design Decisions**:
1. **Why two networks**: Separates learning sub-trajectory structure (RNN) from creating Markovian step-wise guidance (MLP)
2. **Why random cut points**: Provides diverse training signals; systematic cut points could overfit patterns
3. **Why expectation over sub-trajectories**: Ensures step-wise rewards are Markovian, critical for policy optimization stability

---

## Datasets Used

The paper evaluates Diaster on **continuous control benchmarks**, not traditional ML datasets.

### MuJoCo Benchmark

**Source**: MuJoCo physics simulator (Todorov et al. 2012)

**Task**: Continuous control with episodic-reward setting

**Environments Tested**:
1. **InvertedPendulum**: Balance an inverted pendulum
2. **Hopper**: 3D hopper locomotion
3. **Swimmer**: 3D swimming locomotion
4. **Walker2d**: 2D bipedal walking
5. **Ant**: 4-legged locomotion
6. **Humanoid**: 17-degree-of-freedom humanoid walking
7. **HumanoidStandup**: Humanoid standing up from ground

**Modification for Episodic-Reward Setting**:
- Original environments provide step-wise rewards $r(s_t, a_t)$ at each step
- Modified to: Agent receives zero signals at all non-terminal states
- Only feedback: Episodic return $R_{\text{ep}}(\tau_{0:T})$ at episode end
- All other environment settings remain at defaults

**Episode Horizon**: $T = 1000$ steps for all tasks

**Evaluation Protocol**:
- Training: Multiple episodes collected over time steps
- Evaluation: Average return over 10 episodes, computed every 1,000 environment steps
- Reported: Mean and standard error over 5 random seeds

### PointMaze Benchmark

**Source**: D4RL (Data-Driven Deep RL) benchmark suite (Fu et al. 2020)

**Task**: Navigation in 2D mazes using force control

**Environments**:
- **PointMaze-UMaze**: Simple U-shaped maze
- **PointMaze-MediumMaze**: Medium complexity maze
- **PointMaze-LargeMaze**: Large complexity maze

**Agent**: 2-degree-of-freedom ball actuated in Cartesian directions (x, y)

**Objective**: Reach target goal position in closed maze

**Evaluation**: Heatmaps of expected cumulative proxy values to visualize learned value function

### No Traditional Datasets

Unlike supervised learning papers, Diaster does not use standard ML datasets. Environments are **procedurally simulated** rather than collected datasets.

### Why These Benchmarks

- **MuJoCo**: Gold standard for continuous control RL, standard for comparing episodic reward methods
- **PointMaze**: Simpler domain for visualizing learned value functions and understanding behavior
- **Episodic-reward modification**: Matches the problem setting addressed by the paper (only feedback at episode end)

---

## Experimental Results

### Main Results: Performance Comparison

**Setup**: Seven MuJoCo continuous control tasks with episodic-reward setting

**Baselines**:
1. **RUDDER** (Arjona-Medina et al. 2019): LSTM-based return prediction with contribution analysis
2. **IRCR** (Gangwani et al. 2020): Uniform reward redistribution for optimal reward problem
3. **RRD** (Ren et al. 2022): Random return decomposition with trade-off between complexity and accuracy

**Key Findings** (Figure 3):

| Task | Metric | RUDDER | IRCR | RRD | Diaster | Winner |
|------|--------|--------|------|-----|---------|--------|
| InvertedPendulum | Final Return | Poor | ~1k | ~1k | ~1k | Tie (all converge) |
| Hopper | Sample Efficiency | Very Poor | ~3k | ~3k | ~4k | Diaster (faster convergence) |
| Swimmer | Sample Efficiency | Very Poor | ~200 | ~300 | ~300 | RRD/Diaster |
| Walker2d | Sample Efficiency | Very Poor | ~2k | ~4k | ~4k | Diaster/RRD |
| Ant | Final Performance | Poor | ~50k | ~100k | ~150k | **Diaster (best final)** |
| Humanoid | Final Performance | Poor | ~50k | ~100k | ~150k | **Diaster (best final)** |
| HumanoidStandup | Final Performance | Poor | ~4k | ~6k | ~5k | RRD (slight edge) |

**Quantitative Improvements**:
- **RUDDER**: Diaster outperforms in all 7 tasks (RUDDER consistently underperforms)
- **IRCR vs Diaster**: Comparable sample efficiency on most tasks; Diaster better final performance on Ant, Humanoid
- **RRD vs Diaster**: Similar sample efficiency; competitive on most tasks; only HumanoidStandup favors RRD slightly

**Interpretation**: 
- Diaster shows **consistent competitive or superior performance** across diverse continuous control tasks
- Particularly strong on high-dimensional tasks (Humanoid)
- Sample efficiency gains suggest effective proxy reward learning

### Proxy Reward Visualization

**Objective**: Verify learned proxy rewards correlate with task objectives

**Methodology** (Figure 2):
- Sample state-action pairs from trained agent
- Plot learned proxy reward $\hat{r}^\phi(s, a)$ (blue) vs. original reward $r(s, a)$ (orange)
- X-axis: Forward displacement (action consequence)
- Y-axis: Reward values

**Results**:
- **Hopper**: Strong linear correlation between forward motion and proxy reward
- **Swimmer**: Proxy rewards align with swimming velocity
- **Walker2d**: Clear monotonic relationship with forward progress
- **Humanoid**: Proxy rewards correlate with movement magnitude

**Insight**: Diaster successfully discovers latent reward attribution in forward-motion tasks. The nearly monotonic relationships demonstrate that learned proxy rewards properly attribute credit to actions that advance the task goal.

### Ablation Study 1: Number of Cut Points

**Question**: What happens with more than two sub-trajectories?

**Methodology**: Extend decomposition to $m$ cut points:

$$
\hat{R}_{\text{sub}}(\tau_{0:c_0}) + \sum_{i=1}^{m-1} \hat{R}_{\text{sub}}(\tau_{c_{i-1}:c_i}) + \hat{R}_{\text{sub}}(\tau_{c_{m-1}:T}) \approx R_{\text{ep}}(\tau_{0:T})
$$

**Tested Values**: $m \in \{0, 1, 5, 10, 100, T-1\}$

**Results** (Figure 5):

| Setting | Trade-off | Performance |
|---------|-----------|-------------|
| $m = 0$ | No cut points; learns to fit return directly; no attribution | Poor (converges slowly) |
| $m = 1$ | One cut point; **balanced** | **Best for most tasks** |
| $m = 5$ | Five cut points; still good balance | **Best for most tasks** |
| $m = 10$ | Many cut points; more representation | Good but slightly worse than $m=1,5$ |
| $m = 100$ | Many cut points | Worse performance (overfitting to representation) |
| $m = T-1$ | Step-wise decomposition; no temporal structure | Very poor (no representation) |

**Insights**:
- **Trade-off between attribution and representation**: 
  - $m=0$: Only learns return prediction (no attribution capability)
  - $m=T-1$: Each step independent (no temporal structure)
  - Optimal $m \in [1, 5]$ balances both aspects
- **Task sensitivity**: Optimal $m$ may vary by task
- **Heatmap visualization** (Figure 4): PointMaze with $m=10$ shows clearest value gradient toward goal

**Conclusion**: Relatively small number of cut points ($m=1$ or $5$) provides best performance, suggesting attribution is more important than fine-grained temporal structure.

### Ablation Study 2: Step-wise Proxy Reward Function Learning

**Question**: Is learning the separate step-wise network $\hat{r}^\phi$ necessary?

**Comparison**:
- **With $\hat{r}^\phi$**: Use learned step-wise proxy rewards to guide policy
- **Without $\hat{r}^\phi$**: Directly use difference of $\hat{R}^\psi_{\text{sub}}$ for policy optimization

**Results** (Figure 6):

| Task | With $\hat{r}^\phi$ | Without $\hat{r}^\phi$ | Difference |
|------|------------------|----------------------|-----------|
| InvertedPendulum | ~1k (converges fast) | ~0.5k (poor convergence) | **50% improvement** |
| Hopper | ~4k (smooth learning) | ~1.5k (high variance) | **2.7x improvement** |
| Swimmer | ~300 (stable) | ~50 (barely learns) | **6x improvement** |
| Walker2d | ~5k (smooth) | ~1.5k (noisy) | **3.3x improvement** |

**Technical Explanation**:
- $\hat{R}^\psi_{\text{sub}}(\tau_{0:t})$ is **conditioned on the sub-trajectory** $\tau_{0:t}$ (not Markovian)
- Difference $\hat{R}^\psi_{\text{sub}}(\tau_{0:t+1}) - \hat{R}^\psi_{\text{sub}}(\tau_{0:t})$ is **non-Markovian** (depends on history)
- Without learning $\hat{r}^\phi$: Policy optimization becomes **non-stationary** and unstable

**Insight**: The step-wise proxy reward network is **critical** for creating Markovian rewards necessary for stable policy optimization. Direct use of sub-trajectory differences fails because they are history-dependent.

### Statistical Significance

- **Multiple seeds**: All experiments run with 5 random seeds
- **Error bands**: Shaded areas in figures show standard error over seeds
- **Metrics**: Learning curves track average return over evaluation episodes
- **Note**: Paper does not report confidence intervals or statistical tests, but error bands show Diaster has comparable or lower variance than baselines

---

## Code & Implementation References

### Official Implementation

**Repository**: https://github.com/HxLyn3/Diaster

**Framework**: PyTorch

**License**: Not explicitly stated in paper; typically research code

**Language**: Python

### Framework & Dependencies

**Core Dependencies**:
- PyTorch (neural networks)
- MuJoCo (physics simulation)
- D4RL (benchmark environments)
- NumPy (numerical operations)

### RL Algorithm Integration

**Tested with**: Soft Actor-Critic (SAC)
- Can integrate with any off-policy RL algorithm
- Diaster produces proxy rewards that replace environment rewards

### Key Implementation Components

**Sub-trajectory Reward Network**:
```python
# Pseudocode structure
class SubTrajectoryRewardNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        self.gru = nn.GRU(state_dim + action_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, 1)
    
    def forward(self, trajectory):
        # trajectory: sequence of (s, a) pairs
        hidden, _ = self.gru(trajectory)
        return self.output_layer(hidden[-1])  # Use final hidden state
```

**Step-wise Reward Network**:
```python
# Pseudocode structure
class StepwiseRewardNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, s, a):
        return self.net(torch.cat([s, a], dim=-1))
```

### Training Loop Integration

```python
# Integration with SAC
for episode in range(num_episodes):
    trajectory = collect_episode(env, policy)
    replay_buffer.add(trajectory)
    
    for batch in range(num_batches):
        # Update sub-trajectory network
        batch_trajectories = replay_buffer.sample_trajectories(batch_size)
        cut_points = sample_cut_points(batch_size, T)
        loss_R = compute_decomposition_loss(sub_traj_net, batch_trajectories, cut_points)
        update_network(sub_traj_net, loss_R, lr_R)
        
        # Update step-wise reward network
        for t in range(T):
            target = sub_traj_net(traj[:t+1]) - sub_traj_net(traj[:t])
            pred = stepwise_reward_net(s[t], a[t])
            loss_r = (pred - target.detach()) ** 2
            update_network(stepwise_reward_net, loss_r, lr_r)
        
        # RL update with proxy rewards
        proxy_rewards = stepwise_reward_net(states, actions)
        sac_update(policy, states, actions, proxy_rewards, next_states)
```

---

## Reproducibility Checklist

### Fully Specified

- [x] **Model architecture**: GRU-based RNN for sub-trajectory rewards, MLP for step-wise rewards (Section: Architecture & Model Design)
- [x] **Loss functions**: MSE losses for sub-trajectory decomposition and step-wise prediction (Section: Algorithm & Implementation Details)
- [x] **Optimizer settings**: Adam optimizer with learning rates $\lambda_{\hat{R}} = 3e-4$, $\lambda_{\hat{r}} = 3e-4$ (Section: Algorithm & Implementation Details)
- [x] **Dataset preprocessing**: Episodes collected from environments; no additional preprocessing (Section: Datasets Used)
- [x] **Hyperparameters**: Batch size 256, RNN hidden dimension 64-256, RL algorithm SAC (Section: Algorithm & Implementation Details)
- [x] **Training procedure**: Clear algorithm description with update steps (Section: Algorithm & Implementation Details)
- [x] **Evaluation metrics**: Average return over 10 episodes, computed every 1,000 steps (Section: Datasets Used)

### Partially Specified

- [ ] **Network initialization**: Paper doesn't specify weight initialization scheme (PyTorch defaults assumed)
- [ ] **RNN variant details**: Hidden state dimension for each task not explicitly provided (varies by task)
- [ ] **Batch sampling strategy**: Random sampling from replay buffer; exact buffer management not detailed
- [ ] **RL algorithm details**: SAC hyperparameters not fully specified (assumed standard settings)

### Not Specified

- [ ] **Random seed specification**: Five seeds used but specific values not provided
- [ ] **Exact SAC configuration**: Temperature learning, target network update frequency not detailed
- [ ] **Computational resources**: GPU/CPU, memory usage, training time per task not reported
- [ ] **Data collection policy**: Exploration strategy during data collection not specified

### Hardware Requirements

**Reported in Paper**: Not provided

**Estimated for Reproduction**:
- **Minimum**: Single modern GPU (RTX 2080 or equivalent), 16 GB RAM
- **Recommended**: Single modern GPU (RTX 3090 or equivalent), 32 GB RAM
- **Training Time**: Likely 2-6 hours per task (not reported, estimated from standard RL experiments)

### Known Challenges for Reproduction

1. **Missing RNN dimensionality**: Paper uses task-dependent hidden dimensions without specification
2. **SAC hyperparameter details**: Full SAC configuration (temperature, value function architecture) not provided
3. **Initialization sensitivity**: RNN initialization may affect convergence (standard PyTorch defaults assumed)
4. **Environment randomness**: Even with fixed seeds, MuJoCo simulations may have minor variance
5. **Baseline implementations**: Implementing baselines (RUDDER, IRCR, RRD) requires careful attention to their papers

### Reproducibility Strengths

- Official code repository available on GitHub
- Clear algorithm description with pseudocode
- Multiple baselines for comparison
- Extensive ablation studies
- Multiple random seeds reported with error bands

---

## Implementation Roadmap

### Phase 1: Core Infrastructure Setup (Week 1)

**Goal**: Set up project structure and basic data collection pipeline

**Tasks**:
1. [ ] Initialize project structure
   ```
   diaster/
   ├── models/
   │   ├── sub_trajectory_network.py    # RNN for R̂sub
   │   └── stepwise_reward_network.py   # Network for r̂φ
   ├── training/
   │   ├── losses.py                    # Loss functions
   │   └── trainer.py                   # Main training loop
   ├── environments/
   │   └── episodic_wrapper.py          # MuJoCo episodic reward wrapper
   ├── utils/
   │   ├── replay_buffer.py             # Trajectory replay buffer
   │   └── visualization.py             # Plotting utilities
   └── main.py                          # Entry point
   ```

2. [ ] Install dependencies
   - PyTorch (latest version)
   - MuJoCo and dm_control
   - D4RL benchmark
   - NumPy, matplotlib for visualization

3. [ ] Implement episodic-reward environment wrapper
   - Modify MuJoCo environments to suppress step rewards
   - Only return episodic reward at episode end
   - Verify wrapper works on InvertedPendulum

4. [ ] Create replay buffer for trajectory storage
   - Store full trajectories and episodic returns
   - Support efficient sampling of random cut points
   - Implement batch sampling with trajectory re-use

5. [ ] Implement network architectures
   - GRU-based sub-trajectory reward network (test with 256-dim hidden state)
   - Simple MLP for step-wise reward prediction
   - Verify forward passes work

**Checkpoint**: Model can process a single batch (1 trajectory) without errors; sub-trajectory network outputs correct shape (scalar)

### Phase 2: Loss Functions & Training Loop (Week 2)

**Goal**: Implement training procedures for both networks

**Tasks**:
1. [ ] Implement sub-trajectory decomposition loss
   - MSE between sum of sub-trajectory rewards and episodic return
   - Random cut point sampling
   - Batched computation for efficiency

2. [ ] Implement step-wise reward prediction loss
   - MSE between predicted step-wise reward and actual difference
   - Requires running sub-trajectory network on $\tau_{0:t}$ and $\tau_{0:t+1}$
   - Detach targets to avoid unnecessary gradients

3. [ ] Create main training loop
   ```python
   for episode in num_episodes:
       trajectory = collect_episode(env, policy)
       replay_buffer.add(trajectory)
       
       for batch in num_batches:
           # Train sub-trajectory network
           train_sub_traj_network()
           
           # Train step-wise network
           train_stepwise_network()
   ```

4. [ ] Integrate with SAC (Soft Actor-Critic)
   - Use learned proxy rewards in SAC's value/policy updates
   - Verify SAC converges with proxy rewards on simple task

5. [ ] Add logging and visualization
   - Track loss values over training
   - Log policy returns during training
   - Create plots comparing proxy vs. original rewards

**Checkpoint**: Model trains for 1 epoch on InvertedPendulum; losses decrease; proxy rewards show some correlation with task progress

### Phase 3: Baseline Implementation & Comparison (Week 2-3)

**Goal**: Implement baselines for fair comparison

**Tasks**:
1. [ ] Implement RUDDER baseline
   - LSTM-based return prediction
   - Contribution analysis for reward assignment
   - Reference: Arjona-Medina et al. 2019

2. [ ] Implement IRCR baseline
   - Uniform reward redistribution
   - Reference: Gangwani et al. 2020

3. [ ] Implement RRD baseline
   - Random subsequence return decomposition
   - Reference: Ren et al. 2022

4. [ ] Create evaluation harness
   - Run all methods with same random seeds
   - Collect learning curves
   - Compute final performance metrics

5. [ ] Benchmark on MuJoCo tasks
   - Test on InvertedPendulum, Hopper, Swimmer first
   - Verify Diaster outperforms RUDDER
   - Compare with IRCR and RRD

**Checkpoint**: Diaster matches paper's performance on at least 2 MuJoCo tasks; outperforms RUDDER consistently

### Phase 4: Full Evaluation & Ablations (Week 3-4)

**Goal**: Reproduce full experimental results and ablation studies

**Tasks**:
1. [ ] Run on all 7 MuJoCo tasks
   - InvertedPendulum, Hopper, Swimmer, Walker2d, Ant, Humanoid, HumanoidStandup
   - Train with 5 random seeds each
   - 500k-1M environment steps per task

2. [ ] Implement ablation: number of cut points
   - Test $m \in \{0, 1, 5, 10, 100, T-1\}$
   - Generate performance comparison plots
   - Verify $m=1$ or $5$ is optimal

3. [ ] Implement ablation: step-wise network necessity
   - Compare performance with and without $\hat{r}^\phi$
   - Verify performance drops significantly without step-wise network
   - Document why (non-Markovian rewards)

4. [ ] Proxy reward visualization
   - Sample state-action pairs during evaluation
   - Plot proxy rewards vs. forward motion
   - Verify correlation on MuJoCo locomotion tasks

5. [ ] Create comprehensive results table
   - Compile final returns across all tasks
   - Compare means and standard errors with paper
   - Document any discrepancies

6. [ ] Write up reproducibility notes
   - Document any deviations from paper specifications
   - Note any hyperparameter tuning required
   - Suggest best practices

**Checkpoint**: Results within 10% of published numbers; ablations confirm key design choices

### Phase 5: Extensions & Improvements (Optional, Week 4+)

**Potential enhancements**:

1. **Multiple cut points per trajectory**
   - Sample $k$ cut points instead of 1 per trajectory
   - Provides more training signal
   - Trade-off with computational cost

2. **Adaptive cut point selection**
   - Learn where to place cut points
   - Focus on temporal structure boundaries
   - May improve efficiency

3. **Alternative architectures**
   - Test Transformer-based sub-trajectory encoding
   - Compare Transformer vs. GRU representations
   - Evaluate on very long-horizon tasks

4. **Theoretical analysis**
   - Prove convergence rates under function approximation
   - Analyze approximation error bounds
   - Study sample complexity

5. **Real-world applications**
   - Apply to robot control tasks
   - Test on actual delayed reward scenarios
   - Compare with other credit assignment methods

### Critical Implementation Details

**Pay special attention to**:

1. **RNN Handling of Variable-Length Trajectories**
   - Use padding or masking for batch processing
   - Properly handle sequences of different lengths
   - Ensure GRU only processes actual trajectory steps

2. **Gradient Flow in Sub-trajectory Loss**
   - Ensure both $R̂_{\text{sub}}(\tau_{0:c})$ and $R̂_{\text{sub}}(\tau_{c:T})$ contribute to gradients
   - Avoid common gotchas where one path gets blocked

3. **Cut Point Sampling**
   - Sample cut points uniformly from $\{1, ..., T-1\}$
   - Avoid endpoints ($c=0$ or $c=T$) that trivialize decomposition
   - Ensure all cut points are used during training

4. **Detaching Targets in Step-wise Loss**
   - Target difference $R̂_{\text{sub}}(\tau_{0:t+1}) - R̂_{\text{sub}}(\tau_{0:t})$ should be detached
   - Only network $\phi$ (step-wise predictor) should have gradients
   - Prevents circular gradient dependencies

### Common Pitfalls

1. **Forgetting to Detach Sub-trajectory Network Outputs**
   - When training step-wise network, detach $R̂_{\text{sub}}$ outputs
   - Otherwise gradients flow back into sub-trajectory network unintentionally

2. **Incorrect Episodic Return in Modified Environments**
   - Must verify episodic return equals sum of original step rewards
   - Easy to introduce bugs when modifying environments
   - Sanity check: $\sum r(s_t, a_t) = R_{\text{ep}}$ before modification

3. **Mixing Batches with Different Trajectory Lengths**
   - Ensure padding/masking works correctly in RNN
   - Test on trajectories of different lengths
   - Verify loss doesn't include padding steps

4. **Policy Optimization with Wrong Reward Shape**
   - Proxy rewards must be single scalars per (s, a) pair
   - RL algorithm expects shape [batch_size] or [batch_size, 1]
   - Dimension mismatches will cause gradient errors

### Testing Strategy

**Unit tests**:
- Test sub-trajectory network on fixed sequences
- Test step-wise network on single (s, a) pairs
- Verify loss computation on toy problems

**Integration tests**:
- Train on InvertedPendulum for 10k steps
- Verify losses decrease
- Check proxy rewards make sense

**Sanity checks**:
- Overfit on single trajectory
- Verify network can memorize episode rewards
- Check gradient flow with small network

**Validation**:
- Train on several seeds
- Verify mean and std of returns
- Compare with published numbers
- Check sensitivity to hyperparameters

---

## Theoretical Analysis

### Key Theoretical Results

**Theorem 1** establishes that the sum of consecutive step-wise rewards equals the difference of sub-trajectory rewards in expectation:

$$
\mathbb{E}_{\rho^\pi}\left[\sum_{i=t}^{t+h-1} \hat{r}^\pi(s_i, a_i)\right] = \mathbb{E}_{\rho^\pi}\left[\hat{R}_{\text{sub}}(\tau_{0:t+h}) - \hat{R}_{\text{sub}}(\tau_{0:t})\right]
$$

**Importance**: This shows that the decomposition constraint (Equation 5) is compatible with creating return-equivalent step-wise rewards.

**Theorem 2** (conditional form) shows that the proxy value function $\hat{Q}^\pi$ relates to the true value function $Q^\pi$ by a state-dependent offset:

$$
\hat{Q}^\pi(s_t, a_t) = \hat{r}^\pi(s_t, a_t) + \mathbb{E}_{P,\pi}\left[\hat{R}_{\text{sub}}(\tau_{0:T}) - \hat{R}_{\text{sub}}(\tau_{0:t+1})\bigg|s_t, a_t\right]
$$

**Lemma 1** provides the sufficient condition for policy optimality: if $\hat{Q}^\pi(s, a) = Q^\pi(s, a) + \delta(s)$ where $\delta$ depends only on state, then the same optimal policy results.

**Significance**: Together, these results prove that:
1. Step-wise rewards decompose the episodic return correctly
2. The proxy value function has the right structure for policy optimization
3. Using proxy rewards maintains optimal policy consistency

### Why Attribution Matters

**Credit assignment problem**: Which actions/steps deserve credit for the final return?

- **No attribution** ($m=0$): Learn to predict return but provide no step-level guidance → poor policy learning
- **Full attribution** ($m=T-1$): Each step independent, no temporal structure → can't capture how sequence matters
- **Balanced attribution** ($m=1$ to $5$): Some sub-trajectory grouping acknowledges temporal dependencies → better policy learning

The ablation studies empirically validate this trade-off.

---

## References

[Arjona-Medina et al. 2019] RUDDER: Return Decomposition for Delayed Rewards. NeurIPS 19.

[Barto et al. 2009] Where Do Rewards Come From. CogSci 09.

[Gangwani et al. 2020] Learning Guidance Rewards with Trajectory-space Smoothing. NeurIPS 20.

[Cho et al. 2014] Learning Phrase Representations using RNN Encoder-Decoder. EMNLP 14.

[Efroni et al. 2021] Reinforcement Learning with Trajectory Feedback. AAAI 21.

[Fu et al. 2020] D4RL: Datasets for Deep Data-Driven Reinforcement Learning. ICML 20.

[Haarnoja et al. 2018] Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning. ICML 18.

[Hochreiter & Schmidhuber 1997] Long Short-Term Memory. Neural Computation 9.

[Liu et al. 2019] Sequence Modeling of Temporal Credit Assignment. ICLR 19.

[Ng et al. 1999] Policy Invariance Under Reward Transformations. ICML 99.

[Ren et al. 2022] Learning Long-Term Reward Redistribution via Randomized Return Decomposition. ICLR 22.

[Todorov et al. 2012] MuJoCo: A Physics Engine for Model-Based Control. IROS 12.

[Wiewiora et al. 2003] Principled Methods for Advising Reinforcement Learning Agents. ICML 03.

