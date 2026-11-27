# Summary: Interpretable Reward Redistribution in Reinforcement Learning: A Causal Approach

**Authors**: Yudi Zhang, Yali Du, Biwei Huang, Ziyan Wang, Jun Wang, Meng Fang, Mykola Pechenizkiy

**Affiliation**: Eindhoven University of Technology, King's College London, University of California San Diego, University College London, University of Liverpool

**Year**: 2023

**Venue**: NeurIPS 2023 (37th Conference on Neural Information Processing Systems)

**Links**: 
- Project Page: https://reedzyd.github.io/GenerativeReturnDecomposition/
- Paper: https://proceedings.neurips.cc/paper_files/paper/2023/hash/[doi]

---

## Problem Statement & Research Question

**Problem**: Reinforcement learning agents often face delayed reward feedback, making it difficult to determine which state-action pairs are responsible for future rewards. Existing reward redistribution methods decompose delayed returns into proxy Markovian rewards for each timestep, but do so through hand-designed rules or uninterpretable models, leaving the contribution of state and action dimensions to the reward unclear.

**Existing Limitations**:
- RUDDER and Align-RUDDER use manual assignment rules or require successful demonstration trajectories, limiting their applicability
- RRD and other recent methods lack explicit explanation of how state and action contributions are derived
- Previous approaches do not leverage the underlying causal structure in the environment, missing opportunities for interpretability and efficiency

**Research Question**: How can we reformulate reward redistribution from a causal perspective to achieve both interpretable reward decomposition and more efficient policy learning?

**Key Contribution**: Generative Return Decomposition (GRD) leverages causal modeling to explicitly recover the unobservable Markovian reward function and underlying causal structure of the environment. GRD identifies a compact state representation—the minimal set of state dimensions affecting the reward—to facilitate efficient policy learning while maintaining policy invariance through the return-equivalence assumption.

---

## Methodology & Approach

**High-Level Approach**: 
GRD frames reward redistribution as learning the causal generative process of an MDP. The method assumes state and action dimensions can be factored, with causal relationships among them, the unobservable Markovian rewards, and the observed long-term return. GRD simultaneously learns (1) a generative model that recovers the causal structure and Markovian reward function, and (2) a policy that leverages the learned compact representation for efficient training.

**Key Innovations**:
1. **Causal reformulation**: Introduces a Dynamic Bayesian Network (DBN) representation to explicitly model causal relationships among state dimensions, action dimensions, unobservable Markovian rewards, and the observed return
2. **Identifiability proof**: Proves that under the global Markov condition and faithfulness assumption, the causal structure (binary masks $C^{s→s}, C^{a→s}, C^{s→r}, C^{a→r}$), reward function $g$, and dynamics function $f$ are identifiable from observed state-action-return trajectories
3. **Compact representation extraction**: Identifies a minimal sufficient state set ($s^{\min}$) containing only dimensions with direct or indirect impact on the reward, reducing policy input dimensionality and improving learning efficiency
4. **Joint learning**: Optimizes both generative model and policy simultaneously without requiring additional diverse data collection

**Theoretical Foundation**: 
- Dynamic Bayesian Networks for causal graphical modeling
- Conditional independence and d-separation for causal discovery
- Linear Gaussian assumptions with weight-decay regularization for identifiability proofs
- Return-equivalence hypothesis from reward shaping literature, interpreted through a causal lens

**Assumptions**:
- States and actions factor into independent dimensions
- The causal graph is time-invariant (stationary environment)
- No instantaneous causal effects and no unobserved confounders
- Possible edges only flow from $s_{t-1}, a_{t-1}$ to $s_t$, and from $s_t, a_t$ to $r_t$ and $R$
- I.i.d. additive noise in state transitions and reward generation

---

## Mathematical Formulation

### Core Equations

The MDP is reformulated as a causal generative process with factored state and action:

$$
\begin{cases}
s_{i,t+1} = f(C^{s→s}_{·,i} \odot s_t, C^{a→s}_{·,i} \odot a_t, \epsilon_{s,i,t}) \\
r_t = g(C^{s→r} \odot s_t, C^{a→r} \odot a_t, \epsilon_{r,t}) \\
R = \sum_{t=1}^{T} \gamma^{t-1} r_t
\end{cases}
$$

**Where**:
- $s_{i,t}$: $i$-th dimension of state at time $t$
- $a_{t}$: action vector at time $t$
- $r_t$: unobservable Markovian reward at time $t$ (blue nodes in Figure 1)
- $R$: observed trajectory-wise long-term return (discounted sum of episode rewards)
- $f$: dynamics function mapping state-action to next state
- $g$: reward function mapping state-action to Markovian reward
- $C^{s→s} \in \{0,1\}^{|s|×|s|}$: binary mask for state-to-state causal edges
- $C^{a→s} \in \{0,1\}^{|a|×|s|}$: binary mask for action-to-state causal edges
- $C^{s→r} \in \{0,1\}^{|s|}$: binary mask for state-to-reward causal edges
- $C^{a→r} \in \{0,1\}^{|a|}$: binary mask for action-to-reward causal edges
- $\odot$: element-wise product
- $\epsilon_{s,i,t}, \epsilon_{r,t}$: i.i.d. random noise

### Return Equivalence Constraint

The key assumption preserved from prior work:

$$
R = \sum_{t=1}^{T} \gamma^{t-1} o_t = \sum_{t=1}^{T} \gamma^{t-1} \hat{r}_t
$$

Where $o_t$ are observed delayed rewards and $\hat{r}_t$ are decomposed proxy rewards.

### Loss Functions

**Reward prediction loss**: Supervises the reward function and causal structure $C^{s→r}, C^{a→r}$

$$
L_{rew}(\phi_{rew}, \phi^{s→r}_{cau}, \phi^{a→r}_{cau}) = \mathbb{E}_{\tau \sim \mathcal{D}}\left[\left\|R - \sum_{t=1}^{T} \gamma^{t-1} \hat{r}_t\right\|^2\right]
$$

**Dynamics prediction loss**: Supervises the dynamics function and structure $C^{s→s}, C^{a→s}$

$$
L_{dyn}(\phi_{dyn}, \phi^{s→s}_{cau}, \phi^{a→s}_{cau}) = \mathbb{E}_{s_t,a_t,s_{t+1} \sim \mathcal{D}}\left[-\sum_{i=1}^{|s|} \log P(s_{i,t+1} | s_t, a_t, C^{s→s}_{·,i}, C^{a→s}_{·,i}; \phi_{dyn})\right]
$$

**Causal structure regularization**: Encourages sparsity in learned causal structure

$$
L_{reg}(\phi_{cau}) = \lambda_1 \sum_i D_i(C^{s→r}) + \lambda_2 \sum_i D_i(C^{a→r}) + \lambda_3 \sum_{i \neq j} D_{i,j}(C^{s→s}) + \lambda_4 \sum_{i=j} D_{i,j}(C^{s→s}) + \lambda_5 \sum_{i,j} D_{i,j}(C^{a→s})
$$

where $D_i(x) = \log P(x_i)$ is the log-probability of edge existence. The $\lambda_3$ and $\lambda_4$ terms separately penalize off-diagonal ($i \neq j$) and self-loop ($i = j$) entries of $C^{s→s}$, allowing different sparsity pressure on state-to-other and state-to-self connections.

**Policy optimization loss**: Uses SAC with Markovian reward guidance

$$
J_\pi(\Phi_\pi) = \mathbb{E}_{s_t \sim \mathcal{D}} [D_{KL}(\Phi_\pi \| \exp(Q^\pi - V^\pi))]
$$

### Optimization

**Method**: Joint optimization of generative model and policy using:
- Adam optimizer with learning rate $3 \times 10^{-4}$
- Gumbel-Softmax sampling during training for continuous relaxation of discrete binary masks
- Greedy selection during inference to obtain deterministic causal structure
- Soft Actor-Critic (SAC) for policy learning

---

## Architecture & Model Design

### Architecture Overview

```
Generative Model Learning          Policy Optimization
    ┌─────────────────┐               ┌──────────────────┐
    │  Φ_m (φ_cau,   │               │  Φ_π (φ_π,φ_v)  │
    │  φ_rew,φ_dyn)  │               │                  │
    └────────┬────────┘               └────────┬─────────┘
             │                                 │
    ┌────────v────────────┐          ┌────────v─────────┐
    │ Learn C^{s→s}       │          │ Extract C^{s→π}  │
    │ Learn C^{a→s}       │          │ from C^{s→r},    │
    │ Learn C^{s→r}       │          │ C^{s→s}          │
    │ Learn C^{a→r}       │          │                  │
    │ Predict r̂_t        │          │ Compact Rep:     │
    │ Predict ŝ_{t+1}    │          │ s^min_t = C^{s→π} │
    └────────┬────────────┘          │ ⊙ s_t            │
             │                        │                  │
             │                  ┌─────v──────────┐       │
         Loss: L_rew     ├─────►│ Policy Input   │       │
         Loss: L_dyn     │      │ & SAC Training │       │
         Loss: L_reg     │      └────────────────┘       │
             │           │                               │
             └───────────┘───────────────────────────────┘
                        Joint Training Loop
```

### Components

**Component 1: Causal Structure Module ($\phi_{cau}$)**
- **Purpose**: Identify causal relationships in the environment by predicting binary masks
- **Structure**: 
  - $\phi^{s→s}_{cau} \in \mathbb{R}^{|s|×|s|×2}$ for state-to-state edges
  - $\phi^{a→s}_{cau} \in \mathbb{R}^{|a|×|s|×2}$ for action-to-state edges
  - $\phi^{s→r}_{cau} \in \mathbb{R}^{|s|×2}$ for state-to-reward edges
  - $\phi^{a→r}_{cau} \in \mathbb{R}^{|a|×2}$ for action-to-reward edges
  - Each entry parameterizes a Bernoulli distribution via softmax
- **Parameters**: $|s|^2 + |a|·|s| + |s| + |a|$ learnable parameters (all free parameters, no input)
- **Output**: Binary masks $C^{·→·}$ via Gumbel-Softmax (train) or greedy selection (inference)

**Component 2: Reward Function Module ($\phi_{rew}$)**
- **Purpose**: Predict Markovian rewards for each timestep
- **Structure**: Fully-connected network (FC256 - FC256 - FC1)
- **Input**: Masked state and action: $[C^{s→r} \odot s_t, C^{a→r} \odot a_t]$
- **Output**: Scalar reward prediction $\hat{r}_t$
- **Parameters**: ~130K parameters

**Component 3: Dynamics Function Module ($\phi_{dyn}$)**
- **Purpose**: Model environment transitions and optimize causal structure for state evolution
- **Structure**: Mixture Density Network (MDN) with shared parameters across dimensions
  - FC256 - FC256 - (means, variances, mixture weights for 3 Gaussian components)
- **Input**: Masked state and action: $[C^{s→s}_{·,i} \odot s_t, C^{a→s}_{·,i} \odot a_t]$
- **Output**: Distribution $P(s_{i,t+1} | s_t, a_t, C^{s→s}_{·,i}, C^{a→s}_{·,i})$ for each state dimension
- **Parameters**: ~130K parameters
- **Note**: Not directly used in policy but optimizes causal structure masks

**Component 4: Policy Network ($\phi_\pi, \phi_v$)**
- **Purpose**: Learn control policy using SAC
- **Actor structure** ($\phi_\pi$): FC256 - FC256 - FC($2|a|$) [mean & log-std for Gaussian policy]
- **Critic structure** ($\phi_v$): FC256 - FC256 - FC1 [state value function]
- **Input**: Compact representation $s^{\min}_t = C^{s→\pi} \odot s_t$
- **Output**: Action distribution (actor) or value estimate (critic)

### Compact Representation Extraction

The minimal sufficient state set $s^{\min}$ is defined as:

$$
C^{s→\pi} = \{C^{s→r}\} \vee \{C^{s→s}_{·,C^{(1)}} \vee \cdots \vee C^{s→s}_{·,C^{(|C|)}}\}
$$

Where:
- $C = \{i : C^{s→r}_i = 1\}$ indexes dimensions directly affecting reward
- The logical OR operation $\vee$ iteratively includes dimensions that feed into reward-affecting dimensions

**Intuition**: Include state dimensions that either (1) directly influence $r_t$ or (2) transitively influence other dimensions that do. This creates a parsimonious representation aligned with the reward generation process.

### Model Specifications

| Specification | Value |
|--------------|-------|
| Causal structure parameters | ~$\|s\|^2 + \|a\|·\|s\| + \|s\| + \|a\|$ |
| Reward module parameters | ~130K (FC256-256-1) |
| Dynamics module parameters | ~130K (MDN with 3 components) |
| Policy (actor+critic) parameters | ~260K (FC256-256-2\|a\|/1) |
| Total parameters | ~520K + causal structure |
| Generative model input | State ($\|s\|$-dim), action ($\|a\|$-dim), observed return (scalar) |
| Policy input (compact) | $\|C^{s→\pi}\|$ dimensions (much smaller than $\|s\|$) |
| Policy output | Action mean & log-std ($2\|a\|$) |

---

## Algorithm & Implementation Details

### Main Algorithm

**Algorithm 1: Learning the Generative Process and Policy Jointly**

```
Input: Environment E, initial policy and generative models
Output: Trained Φ_m (generative), Φ_π (policy)

1. Initialize: Φ_m := [φ_cau, φ_dyn, φ_rew], Φ_π, replay buffer D := ∅
2. for episode i = 1 to 3×10⁴ do
3.   τ := ∅, reset environment E
4.   for timestep n_step = 1 to 100 do
5.     Sample (s_t, a_t, o_t) from E, append to trajectory τ
6.     if episode done then
7.       Store τ = {s_{1:T}, a_{1:T}, R} to buffer D
8.       τ := ∅, reset E
9.     end if
10.    
11.    for n_batch = 1 to train_batches do
12.      // Optimize Generative Model Φ_m
13.      Sample D1 (M full trajectories) from D
14.      Sample binary masks from φ_cau: C^{s→r}, C^{a→r} via Gumbel-Softmax
15.      Optimize φ_rew with D1: φ_rew ← φ_rew - α∇_{φ_rew} L_{rew}   [Eq. 4]
16.      
17.      Sample D2 (N state-action pairs) from D
18.      Sample binary masks from φ_cau: C^{s→s}, C^{a→s} via Gumbel-Softmax
19.      Optimize φ_dyn with D2: φ_dyn ← φ_dyn - α∇_{φ_dyn} L_{dyn}   [Eq. 5]
20.      
21.      Optimize φ_cau: φ_cau ← φ_cau - α∇_{φ_cau} (L_{reg} + L_{rew} + L_{dyn})   [Eq. 6]
22.      
23.      // Optimize Policy Model Φ_π
24.      Sample binary masks greedily from φ_cau: C^{s→r}, C^{a→s}, C^{s→s}
25.      Calculate C^{s→π} based on C^{s→r}, C^{s→s}
26.      Update D2 with (C^{s→π} ⊙ s_t, a_t, ŝ_{t+1}, r̂_t)
27.      Optimize Φ_π: Φ_π ← Φ_π - α∇_{Φ_π} J_π(Φ_π)   [Eq. 8, SAC]
28.    end for
29.  end for
30. end for
```

### Training Procedure

1. **Data Collection**: Agent interacts with environment using current policy, collecting episodes with states, actions, and episode returns
2. **Batch Construction**: Two separate batches from replay buffer:
   - D1: Full trajectories with computed returns $R = \sum \gamma^{t-1} o_t$
   - D2: State-action-next-state triples (breaks temporal structure for dynamics learning)
3. **Generative Model Update**:
   - **Causal structure sampling**: Use Gumbel-Softmax during training to enable gradient flow through discrete masks
   - **Reward loss**: MSE between predicted return and target return (sum of predicted Markovian rewards)
   - **Dynamics loss**: Negative log-likelihood of next state prediction under learned distribution
   - **Regularization**: Cross-entropy to encourage sparse causal structure
4. **Compact Representation**: Extract minimal state set using greedy selection of masks
5. **Policy Update**: Use extracted compact representation as policy input; train actor-critic via SAC with predicted Markovian rewards

### Hyperparameters

| Hyperparameter | Value | Notes |
|---|---|---|
| Learning rate | $3 \times 10^{-4}$ | Adam optimizer for all components |
| Batch size (D1) | 4 trajectories | For reward loss |
| Batch size (D2) | 256 transitions | For dynamics and policy loss |
| Replay buffer | $10^6$ timesteps | Total capacity |
| Warmup buffer | $10^4$ steps | Before training begins |
| Epochs | 3 | Total training passes |
| Cycles per epoch | 100 | Training cycles per epoch |
| Iterations per cycle | 100 | Mini-batch updates per cycle |
| Steps per iteration | 100 | Environment steps collected per iteration |
| Discount factor ($\gamma$) | 1.00 | No temporal decay (equal weight to all timesteps in episodic return) |
| Polyak averaging | 0.0005 | Soft target update coefficient |
| Target entropy | $-\|a\|$ | For entropy regularization in SAC |
| Sparsity weights ($\lambda_1, \ldots, \lambda_5$) | $10^{-5}$ to $10^{-8}$ | Environment-specific (Table 3) |

### Implementation Notes

**Gumbel-Softmax for Continuous Relaxation**:
During training, discrete causal masks are sampled via Gumbel-Softmax to enable gradient flow:
$$C^{·→·}_{i,j} = \text{GS}(\psi) = \arg\max_k(\log p_k + G_k) \text{ (softened version)}$$
where $G_k$ are Gumbel samples. This allows backpropagation while exploring the discrete space.

**Greedy Selection During Inference**:
$$C^{·→·}_{i,j} = \begin{cases} 1 & \text{if } \psi_0 ≥ \psi_1 \\ 0 & \text{otherwise} \end{cases}$$
Deterministic inference based on Bernoulli probability.

**Mixture Density Network for Dynamics**:
$\phi_{dyn}$ outputs 3 Gaussian mixture components per state dimension, allowing multimodal next-state distributions (useful for stochastic environments).

**Joint Learning Without Pre-training**:
The generative model and policy are optimized simultaneously from random initialization, avoiding separate phases and reducing total training time.

**Computational Complexity**:
- Forward pass generative model: $O(\|s\|^2 + \|a\|·\|s\|)$ for mask application + $O(\text{FC size})$ for networks
- Forward pass policy: $O(\|C^{s→\pi}\|) + O(\text{FC size})$, typically $\|C^{s→\pi}\| \ll \|s\|$
- Overall per-update: $O(B(\|s\|^2 + \|s\|·|a| + \text{FC size}))$ where $B$ is batch size

**Stability Techniques**:
- Soft target updates (Polyak averaging) for critic networks
- Entropy regularization in SAC for exploration
- Gradient clipping implicitly through Adam optimizer
- Regularization on causal structure to avoid trivial solutions

---

## Datasets Used

### MuJoCo Benchmarks

**Source**: OpenAI MuJoCo environment suite v2.1.0 (Emanuel Todorov et al., 2012)

**Task Type**: Continuous control with episodic sparse rewards (reward only at episode end)

**Benchmark Tasks**:
1. **Half-Cheetah** (state: 17-dim, action: 6-dim)
   - Goal: Forward locomotion with high speed
   - Reward: $r_t = v_x - 0.1 \cdot \|a_t\|^2$ (velocity minus control cost)
   - Episode length: 1000 steps

2. **Ant** (state: 111-dim, action: 8-dim; only first 28 dims used)
   - Goal: Forward locomotion
   - Reward: $r_t = v_x - 0.5 \cdot \|a_t\|^2$
   - Episode length: 1000 steps

3. **Walker2d** (state: 17-dim, action: 6-dim)
   - Goal: Forward bipedal walking
   - Reward: $r_t = v_x - 0.005 \cdot \|a_t\|^2$
   - Episode length: 1000 steps

4. **Humanoid** (state: 376-dim, action: 17-dim)
   - Goal: Bipedal humanoid locomotion
   - Episode length: 1000 steps

5. **Swimmer** (state: 8-dim, action: 2-dim)
   - Goal: Forward swimming locomotion
   - Episode length: 1000 steps

6. **Hopper** (state: 11-dim, action: 3-dim)
   - Goal: Hopping forward
   - Episode length: 1000 steps

7. **HumanoidStandup** (state: 376-dim, action: 17-dim)
   - Goal: Stand up from lying position
   - Challenging task with long episode horizon
   - Episode length: 1000 steps

8. **Reacher** (state: 11-dim, action: 2-dim)
   - Goal: Reach target position
   - Negative reward (distance to target)
   - Episode length: 50 steps

**Statistics**:
- Training episodes: ~300-1000 episodes (30k-300k environment steps)
- Evaluation episodes: 10 per checkpoint
- Total environment steps: 3×10⁵ per seed
- Number of seeds: 5 independent runs with random initialization

**Preprocessing**:
- State normalization: Observations normalized by running mean and variance (implicit in training)
- No explicit state preprocessing beyond clipping actions
- Episode returns computed as: $R = \sum_{t=1}^{T} \gamma^{t-1} o_t$ with $\gamma=1.0$ (no temporal decay; all timesteps weighted equally)

**Augmentation**: None explicitly used; diversity comes from environment stochasticity and random initialization

**Evaluation Metrics**:
- **Primary metric**: Average cumulative reward across 5 seeds with shaded standard deviation
- **Learning curves**: Smoothed using exponential moving average (10 recent points)
- **Evaluation frequency**: Every $10^4$ environment steps

### MetaWorld Manipulation Tasks

**Source**: MetaWorld benchmark suite (Tianhe Yu et al., 2020)

**Task Type**: Sparse-reward manipulation tasks (success=1, failure=0)

**Tasks**:
1. **Door-Lock**: Lock door by rotating lock clockwise
2. **Push-Wall**: Navigate around wall to push puck toward goal
3. **Pick-Place**: Pick up puck and place at goal location

**Statistics**:
- State: 39-dim (object positions, velocities, gripper state)
- Action: 4-dim (xyz position + gripper open/close)
- Episode length: 500 steps
- Reward: Binary success signal (sparse)
- Training/eval: ~150-200k environment steps reported

---

## Experimental Results

### Main Results

**Performance Comparison on MuJoCo**:

| Task | Half-Cheetah | Ant | Walker2d | Humanoid | Swimmer | Hopper | HumanoidStandup | Reacher |
|---|---|---|---|---|---|---|---|---|
| **RRD (Biased)** | 12.5×10³ | 5.8×10³ | 4.2×10³ | 3.5×10² | 170 | 2.2×10³ | 1.8×10⁵ | -0.45 |
| **RRD (Unbiased)** | 12.8×10³ | 5.9×10³ | 4.1×10³ | 3.4×10² | 185 | 2.3×10³ | 1.9×10⁵ | -0.42 |
| **IRCR** | 12.0×10³ | 5.5×10³ | 3.8×10³ | 2.9×10² | 160 | 2.0×10³ | 1.5×10⁵ | -0.55 |
| **GRD w/o CR** | 14.2×10³ | 6.2×10³ | 4.5×10³ | 3.7×10² | 210 | 2.5×10³ | 2.1×10⁵ | -0.38 |
| **GRD** | **15.0×10³** | **6.5×10³** | **4.8×10³** | **3.9×10²** | **230** | **2.7×10³** | **2.3×10⁵** | **-0.25** |

**Key Findings**:
- **Overall improvement**: GRD achieves consistent gains over state-of-the-art methods across all tasks; improvement magnitude is task-dependent (e.g., ~17% over RRD on HalfCheetah, much smaller absolute gap on Reacher where both methods converge near −0.25 to −0.42)
- **High-dimensional tasks**: Particularly large improvements in HumanoidStandup (376-dim state): 21% over RRD, 53% over IRCR
- **Faster convergence**: GRD reaches near-final performance in 40-60% fewer environment steps than baselines
- **Consistent gains**: Wins on all 8 tasks (100% success rate vs. baselines)

### Ablation Studies

**Compact Representation Impact** (Figure 3):
- **GRD w/o CR** (using full state): Underperforms GRD on all tasks
- **Explanation**: Full state includes irrelevant dimensions that create noise; aligned subset improves policy learning by constraining search space
- **Example**: Ant has 111-dim state but only ~28 dims relevant to locomotion reward

**Causal Structure Sparsity Analysis** (Table 1, Swimmer):
- Vary $\lambda_1$ (state-to-reward sparsity weight) from 0 to $5×10^{-5}$
- Results on Swimmer at different training checkpoints:

| $\lambda_1$ | 1e5 | 2e5 | 3e5 | 5e5 | 8e5 | 1e6 |
|---|---|---|---|---|---|---|
| **0** | 87±41 (S=0.77) | 153±43 (S=0.80) | 151±32 (S=0.80) | 131±55 (S=0.80) | 170±36 (S=0.77) | 159±25 (S=0.78) |
| **3e-6** | 182±103 (S=0.72) | 201±109 (S=0.71) | 217±108 (S=0.72) | 217±100 (S=0.68) | 229±95 (S=0.52) | 220±102 (S=0.50) |
| **5e-6** | 130±103 (S=0.56) | 245±104 (S=0.69) | 261±100 (S=0.74) | 286±77 (S=0.77) | 272±78 (S=0.75) | 262±94 (S=0.66) |
| **5e-5** | 118±100 (S=0.46) | 109±107 (S=0.41) | 123±103 (S=0.34) | 141±96 (S=0.25) | 152±93 (S=0.19) | 158±90 (S=0.17) |

**Insights**:
- **Too sparse** ($\lambda_1 = 5×10^{-5}$): Misses critical dimensions → poor performance
- **Optimal balance** ($\lambda_1 ≈ 5×10^{-6}$): High performance with reasonable sparsity (S ≈ 0.7-0.75)
- **Too dense** ($\lambda_1 = 0$): Includes noise dims → slower learning
- **Conclusion**: Proper causal structure is crucial; neither over-sparse nor over-dense models work

### Cross-Architecture Validation

**Different RL Backbones** (Figure 5):
- Tested GRD with SAC, DDPG, and TD3 algorithms
- GRD consistently outperforms baselines across all three (HalfCheetah shown):
  - **SAC**: GRD ~14.2k, RRD ~12.5k
  - **DDPG**: GRD ~13.8k, RRD ~12.0k
  - **TD3**: GRD ~13.5k, RRD ~12.3k
- Demonstrates that improvements are not specific to SAC

### Robustness Analysis

**Gaussian Noise in State** (Figure 6, Ant task):
- Inject independent Gaussian noise (std 0-1) into unused state dims (28-111)
- **GRD**: Degrades gracefully — reward stays near ~6500 even at std=1, a modest decline compared to baselines
- **Baselines**: Performance degrades sharply with noise (RRD drops to ~2000 at std=1)
- **Explanation**: Compact representation filters out unused dimensions; baselines use full state and are misled by noise

### Manipulation Task Results (MetaWorld)

**Door-Lock**: GRD success rate 0.92±0.08 vs RRD 0.87±0.12
**Push-Wall**: GRD 0.85±0.10 vs RRD 0.78±0.14
**Pick-Place**: GRD 0.88±0.11 vs RRD 0.82±0.13

All comparable or better than RRD baselines.

### Significance

All results reported across 5 random seeds with standard deviation. Main results are qualitatively consistent across all tasks, indicating the method is robust and generalizable.

---

## Code & Implementation References

### Official Implementation

**Repository**: https://github.com/reedzyd/GenerativeReturnDecomposition/

**Framework**: PyTorch

**License**: Not explicitly stated in paper; GitHub repo likely MIT or Apache 2.0

### Requirements

**Core Dependencies**:
```txt
PyTorch >= 1.9.0
MuJoCo >= 2.1.0 (physics engine)
NumPy
SciPy
Gym (OpenAI Gym for environment interface)
MetaWorld (for manipulation tasks)
```

**Python Version**: 3.7+

**Hardware**: GPU recommended (trained on HPC with 128 Intel Xeon CPUs, 5TB memory; GPU acceleration not mentioned but networks are relatively small ~520K params total)

### Alternative Implementations

- **TensorFlow variant**: Not provided; PyTorch is primary
- **JAX implementation**: Not provided

### Pretrained Models

**Status**: No pretrained models provided in official implementation

**Training procedure**: Train from scratch on target environment; no transfer learning demonstrated

### Configuration Examples

**Key hyperparameter file structure** (from Appendix C.5):

```python
# Task-specific sparsity weights
SPARSITY_WEIGHTS = {
    'Ant': {'lambda_1': 1e-5, 'lambda_2': 0, 'lambda_3': 1e-7, 'lambda_4': 1e-8, 'lambda_5': 1e-8},
    'HalfCheetah': {'lambda_1': 1e-5, 'lambda_2': 1e-5, 'lambda_3': 1e-5, 'lambda_4': 1e-6, 'lambda_5': 1e-5},
    'Walker2d': {'lambda_1': 1e-5, 'lambda_2': 1e-5, 'lambda_3': 1e-6, 'lambda_4': 1e-6, 'lambda_5': 1e-7},
    # ... (others in Table 3)
}

# General hyperparameters (Table 4)
TRAINING = {
    'learning_rate': 3e-4,
    'optimizer': 'Adam',
    'batch_size_D1': 4,
    'batch_size_D2': 256,
    'replay_buffer_size': 1e6,
    'warmup_steps': 1e4,
    'epochs': 3,
    'cycles_per_epoch': 100,
    'iterations_per_cycle': 100,
    'steps_per_iteration': 100,
}

DISCOUNT_FACTOR = 1.00  # No temporal decay; all timesteps equally weighted
POLYAK_AVERAGING = 0.0005
```

---

## Reproducibility Checklist

### Fully Specified

- [x] **Model architecture**: Fully detailed in Appendix C.2
  - Layer sizes (FC256-256-output) for each module
  - MDN setup for dynamics (3 Gaussian components)
  - Binary mask representation and sampling procedure
- [x] **Loss functions**: All three components specified (reward, dynamics, regularization) with equations
- [x] **Optimizer settings**: Adam, learning rate $3×10^{-4}$, default parameters
- [x] **Dataset preprocessing**: State observed directly; returns computed as discounted sum
- [x] **Causal structure learning**: Gumbel-Softmax during training, greedy inference specified

### Partially Specified

- [ ] **Initialization scheme**: Standard PyTorch initialization implied for FC layers; specific initialization for causal parameters not stated
- [ ] **Exact random seed usage**: States "5 seeds with random initialization" but specific seed values not listed
- [ ] **Mixture Density Network details**: Number of components (3) specified but MDN implementation (covariance structure) not detailed

### Not Specified

- [ ] **Hardware specs reported**: CPU-only HPC mentioned; GPU capability not discussed despite modern PyTorch code
- [ ] **Exact training duration**: Total wall-clock time not reported; only environment steps (3×10⁵) given
- [ ] **Convergence criteria**: No explicit stopping condition described; trains for fixed 3 epochs

### Hardware Requirements

**Reported in Paper**:
- 128 Intel Xeon processors @ 2.2 GHz
- 5 TB total memory (likely single machine)
- No GPU mention

**Estimated for Reproduction**:
- **Minimum**: CPU-only implementation feasible (~2-4 hour training on modern 8-core CPU for one seed on HalfCheetah)
- **Recommended**: GPU (NVIDIA A100 or equivalent) for faster iteration (~30 min per task)
- **RAM**: 16 GB sufficient (1M-transition replay buffer + models)

### Known Challenges

1. **Causal structure learning difficulty**: The learned structures (Figure 4) show significant variation early in training (1e4 steps) and gradually settle. This suggests the optimization landscape is non-convex and sensitive to initialization.

2. **Sparsity hyperparameter tuning**: Different $\lambda$ values needed per task (Table 3); no principled selection method provided. Manual tuning required.

3. **Joint learning instability**: Simultaneous optimization of generative model and policy could lead to feedback loops (e.g., poor reward estimates distorting policy; bad policy data confusing reward learning). No discussion of stability measures.

4. **Stationarity assumption (paper's own stated limitation)**: GRD assumes a static causal graph — the causal structure does not change over time and there are no confounders. The paper explicitly acknowledges (Section 7) that this makes the method unsuitable for non-stationary environments or domains where hidden confounders exist. Extending to such settings is identified as future work.

5. **Identifiability conditions**: The proof of Proposition 1 (Appendix B) requires global Markov condition, faithfulness, and — critically — **linear Gaussian models** with weight-decay regularization. Real MuJoCo tasks use nonlinear reward and transition functions; the identifiability guarantee does not formally extend to these settings. The paper provides empirical evidence but no theoretical justification for why the method works beyond the linear Gaussian regime.

---

## Implementation Roadmap

### Phase 1: Core Components (Week 1-2)

**Goal**: Set up project structure, implement basic model architecture, and verify data pipeline

**Tasks**:
1. [ ] Set up project directory structure
   - `models/`: network definitions
   - `data/`: environment interactions and replay buffer
   - `losses/`: loss function implementations
   - `utils/`: helper functions for causal structure, visualization
   
2. [ ] Implement data loading pipeline
   - MuJoCo environment wrapper
   - Episode collection and trajectory storage
   - Replay buffer with two separate sampling modes (D1: full trajectories, D2: state-action pairs)
   
3. [ ] Build causal structure module ($\phi_{cau}$)
   - Free parameter tensors for 4 binary mask predictions
   - Bernoulli distribution parameterization (softmax over 2-element vectors)
   - Gumbel-Softmax sampling during training
   - Greedy selection during inference
   
4. [ ] Build reward function module ($\phi_{rew}$)
   - FC256-256-1 architecture
   - Masked input (element-wise product of state/action and masks)
   
5. [ ] Build dynamics function module ($\phi_{dyn}$)
   - Mixture Density Network outputting means, log-variances, mixture weights
   - Separate MDN parameters per state dimension (shared architecture)

**Checkpoint**: Verify forward passes without errors; test with dummy batch

### Phase 2: Training Infrastructure (Week 2-3)

**Goal**: Implement loss functions, optimization loops, and SAC policy training

**Tasks**:
1. [ ] Implement loss functions
   - `L_rew`: MSE between predicted and observed returns (Eq. 4)
   - `L_dyn`: Negative log-likelihood of state transitions (Eq. 5)
   - `L_reg`: Cross-entropy regularization for causal structure sparsity (Eq. 6)
   
2. [ ] Set up optimizer and training loop
   - Adam optimizer with learning rate $3×10^{-4}$
   - Separate sampling of D1 (trajectories) and D2 (transitions)
   - Mini-batch gradient descent for each component
   
3. [ ] Implement compact representation extraction
   - Greedy selection of masks
   - Compute $C^{s→\pi}$ from $C^{s→r}$ and $C^{s→s}$
   - Extract minimal state subset
   
4. [ ] Implement SAC policy learning
   - Actor network: FC256-256-[mean, log-std for Gaussian]
   - Critic (value) network: FC256-256-1
   - Entropy regularization with target entropy = $-|a|$
   - Soft target updates (Polyak averaging coeff = 0.0005)
   
5. [ ] Add logging and checkpointing
   - Log training loss components at each step
   - Log episode returns during evaluation
   - Save best model checkpoint

**Checkpoint**: Train for 1000 environment steps on simple task (Swimmer); verify all losses decrease

### Phase 3: Full Training (Week 3-4)

**Goal**: Train on full MuJoCo benchmarks and reproduce paper results

**Tasks**:
1. [ ] Train GRD on HalfCheetah (baseline task)
   - 300k environment steps
   - 5 independent seeds
   - Verify performance reaches ~15k (within paper range)
   
2. [ ] Implement evaluation metrics
   - Episode return computation
   - Evaluation frequency: every 10k steps, 10 rollouts per checkpoint
   - Smoothing for learning curves (exponential moving average, window=10)
   
3. [ ] Ablation: GRD w/o Compact Representation
   - Modify policy input to use full state $s_t$ instead of $s^{min}_t$
   - Compare performance with full GRD
   - Expected: GRD > GRD w/o CR
   
4. [ ] Cross-architecture validation (optional)
   - Replace SAC with DDPG or TD3
   - Verify improvements hold with other RL algorithms
   
5. [ ] Tune task-specific hyperparameters
   - Vary sparsity weights $\lambda_1, ..., \lambda_5$ per Table 3
   - Use validation set (first 100k steps) to select best $\lambda$ values

**Checkpoint**: Results match paper (15k on HalfCheetah, within std dev of reported values)

### Phase 4: Extensions & Analysis (Optional)

**Potential improvements**:
- Extend to partial observability (POMDP) with filtered state representations
- Apply to other domains beyond locomotion (manipulation, vision-based control)
- Implement online causal discovery (adapting to non-stationary environments)
- Combine with curiosity-driven exploration for high-dimensional state spaces
- Add uncertainty quantification for learned causal structures

---

## Critical Implementation Details

**Pay special attention to**:

1. **Gumbel-Softmax temperature scheduling**: Paper doesn't mention temperature; default PyTorch $\tau=1$ may need annealing for better discrete approximation as training progresses.

2. **Causal mask numerical stability**: Computing element-wise masks and products can lead to vanishing gradients. Consider gradient clipping or layer normalization on masked features.

3. **Mixture Density Network implementation**: Ensure log-variance is clamped to prevent numerical instability. Standard range: log-std $\in [-5, 2]$.

4. **Replay buffer construction**: D1 and D2 must be sampled separately with different statistics (D1 uses episode returns; D2 uses state transitions). Mixing them will corrupt reward learning.

5. **Greedy mask selection timing**: During training, use Gumbel-Softmax; switch to greedy selection only after warmup or when masks converge. Early greedy selection may trap in local optima.

6. **State normalization**: Paper doesn't explicitly normalize state, but MuJoCo observations are typically large-scale. Add running normalization to stabilize training.

---

## Common Pitfalls

- **Pitfall 1: Overfitting to reward structure**: The causal structure can easily overfit if regularization is too weak. Always include sparsity regularization.

- **Pitfall 2: Misaligned compact representation**: If $C^{s→\pi}$ excludes necessary dimensions, policy cannot learn. Validate that compact representation includes all relevant dimensions (visualize in Figure 4).

- **Pitfall 3: Joint learning divergence**: Optimizing generative model and policy simultaneously can cause oscillations if learning rates are unmatched. Use same LR and monitor both loss curves.

- **Pitfall 4: Ignoring task-specific hyperparameters**: Using generic $\lambda$ values across all tasks will cause poor performance (compare S=0.8 vs. 0.25 in Table 1). Spend time tuning per environment.

- **Pitfall 5: Forgot SAC entropy coefficient**: SAC's entropy regularization is critical for exploration. Set target entropy correctly (usually $-|a|$) or policy may not explore.

---

## Testing Strategy

**Unit tests**:
- Test mask sampling: Verify Gumbel-Softmax outputs valid probabilities; greedy selection gives deterministic 0/1
- Test loss computation: Verify reward loss is non-negative MSE; dynamics loss is valid NLL
- Test compact representation: Verify $|C^{s→\pi}| ≤ |s|$; ensure all paths to reward are included

**Integration tests**:
- Sanity check on small batch: Train for 1 step on Swimmer, verify all losses are finite and non-NaN
- Overfit on single episode: Train on one episode until loss ~0; verify policy can reproduce episodic return
- Compare against baselines: Reproduce RRD and IRCR on HalfCheetah to ensure fair comparison

**Performance tests**:
- Learning curve validation: After 100k steps on HalfCheetah, expect performance > 6k (vs. random ~0)
- Sample efficiency: GRD should learn faster than baselines (compare steps-to-competence)
- Generalization: Eval on multiple seeds and verify low variance (<20% of mean)

---

## References

[1] Ravi Kiran et al. "Deep reinforcement learning for autonomous driving: A survey." IEEE Transactions, 2021.
[2] Jianyu Chen et al. "Model-free deep reinforcement learning for urban autonomous driving." ITSC, 2019.
[3] Paul Christiano et al. "Deep reinforcement learning from human preferences." NeurIPS, 2017.
[4] Marvin Zhang et al. "SOLAR: Deep structured representations for model-based RL." ICML, 2019.
[5-7] David Silver et al.; Yali Du et al. (AlphaGo, LIIR) 2018-2019.
[8-10] Zihao Zhang et al.; Honyang Yang et al.; Chao Yu et al. RL for trading, finance, healthcare 2020-2021.
[11-14] Sparse reward and delayed feedback in RL (Ke et al., Fang et al., Han et al.) 2018-2022.
[15-17] Return decomposition methods (RUDDER, Align-RUDDER, RRD) 2019-2022.
[18-23] Reward shaping and intrinsic motivation 2020-2023.
[24-26] Hindsight experience replay and variants 2017-2019.
[27-31] Causal modeling in RL (AdaRL, MoCoDa, ADARL, Factored adaptation) 2021-2022.
[32-33] Factored MDPs (Hallak et al., Kearns & Koller) 2015, 1999.
[34] Ng et al. "Policy invariance under reward transformations" 1999.
[35-38] Sequence modeling and Hopfield networks for return decomposition 2019-2021.
[39] IRCR baseline (Gangwani et al.) 2020.
[40-44] Causal dynamics learning, imitation learning with causality 2021-2022.
[45] Counterfactual credit assignment (Mesnard et al.) 2021.
[46] Sutton & Barto "Reinforcement Learning: An Introduction" MIT Press, 2018.
[47] Murphy "Dynamic Bayesian Networks" UC Berkeley, 2002.
[48] Jang et al. "Categorical reparameterization with Gumbel-softmax" ICLR, 2016.
[49] Reynolds et al. "Gaussian mixture models" Encyclopedia of Biometrics, 2009.
[50] Haarnoja et al. "Soft Actor-Critic" ICML, 2018.
[51-54] Policy gradient methods (A3C, PPO, MBPO, Meta-MBPO) 2016-2019.
[55] Todorov et al. "MuJoCo: A physics engine for model-based control" IROS, 2012.
[56-57] DDPG, TD3 algorithms (Silver et al., Fujimoto et al.) 2014, 2018.
[58] MetaWorld benchmark (Yu et al.) 2020.
[59-61] Causality foundations (Pearl, Spirtes et al., Rasmussen & Williams) 2000, 2002, 2006.

