# Summary: Off-Policy Reinforcement Learning with Delayed Rewards

## Metadata

**Title:** Off-Policy Reinforcement Learning with Delayed Rewards

**Authors:** Beining Han, Zhizhou Ren, Zuofan Wu, Yuan Zhou, Jian Peng

**Venue:** 39th International Conference on Machine Learning (ICML 2022)

**Citation:** Proceedings of Machine Learning Research, Volume 162, Pages 8280-8303, 2022

**Links:**
- [PMLR Proceedings](https://proceedings.mlr.press/v162/han22e.html)
- [arXiv](https://arxiv.org/abs/2106.11854)
- [OpenReview](https://openreview.net/forum?id=nsjkNB2oKsQ)
- [ICML Virtual Poster](https://icml.cc/virtual/2022/poster/16161)

---

## 1. Problem Statement & Research Question

**Problem:**
Many real-world reinforcement learning environments exhibit **delayed rewards**, where agents cannot observe immediate reward feedback after performing actions. Instead, rewards are delayed by multiple timesteps or accumulated over future episodes, making the environment inherently **non-Markovian**. This violates the standard Markov assumption in traditional RL algorithms.

**Existing Limitations:**
- Standard off-policy RL algorithms (SAC, DQN, TD3) assume immediate reward feedback and Markovian transitions
- These algorithms lack theoretical guarantees when applied to delayed reward settings
- Naïve application of standard RL to delayed reward problems results in suboptimal performance
- Existing approaches do not provide principled ways to handle temporal reward dependencies
- Current methods struggle with long reward delays in high-dimensional state spaces

**Research Question:**
How can we develop an off-policy reinforcement learning framework that:
1. Handles non-Markovian delayed reward environments with theoretical guarantees?
2. Scales efficiently to high-dimensional state spaces?
3. Outperforms standard RL algorithms in delayed reward settings?

**Key Contribution:**
The paper introduces the **Past-Invariant Delayed Reward MDP (PI-DRMDP)** framework, providing formal definitions and theoretical analysis of delayed reward environments. It proposes a novel **off-policy RL algorithm with a decomposed Q-function** that explicitly separates historical and current information, along with the **HC-decomposition rule** for practical scalability to high-dimensional problems.

---

## 2. Methodology & Approach

**High-Level Approach:**

The paper addresses delayed rewards through three main components:

1. **Problem Formalization:** Formally define environments with delayed rewards as PI-DRMDPs where the reward function depends on the history of states and actions
2. **Theoretical Framework:** Develop convergence guarantees in the tabular (discrete) setting
3. **Practical Algorithm:** Design a scalable algorithm for continuous control using Q-function decomposition

**Key Innovations:**

1. **PI-DRMDP Definition:** A novel MDP formulation that handles environments where rewards depend on the state-action trajectory within a *signal interval*, not just the current state-action pair
2. **New Q-Function Formulation:** Instead of standard $Q(s, a)$ functions, the framework defines $Q^\pi(\tau_{t_i:t+1})$ over the trajectory segment from the start of the current signal interval $t_i$ to the current step $t+1$
3. **HC-Decomposition:** Decomposes the Q-function (Eq. 6 in paper) into **History (H) components** $H_\phi(\tau_{t_i:t})$ (over previous steps in the current interval) and **Current (C) components** $C_\phi(s_t, a_t)$ (current step only), making optimization easier and reducing policy gradient variance
4. **Non-Markovian Analysis:** Explicitly addresses challenges from breaking the Markov assumption while maintaining convergence guarantees

**Theoretical Foundation:**

- **Markov Decision Process Extension:** Extends standard MDPs to handle non-Markovian reward structures
- **Value Function Decomposition:** Leverages decomposition of value functions into history-dependent and state-dependent components
- **Policy Gradient Methods:** Uses actor-critic framework adapted for delayed rewards

**Policy Classes:**

The paper considers two policy classes beyond the standard Markov policy:
- $\Pi_s = \{\pi = \pi(a_t | s_t, t-t_i)\}$: standard policy augmented with relative position within the current signal interval; this is what the paper optimises in practice
- $\Pi_\tau = \{\pi = \pi(a_t | \tau_{t_i:t} \circ s_t)\}$: full history-conditioned policy

For sum-form PI-DRMDPs (reward = sum of per-step rewards), optimal policies exist in $\Pi_s$. In general, there exist PI-DRMDPs where the optimal policy is not in $\Pi_s$, but this does not arise in practice for the tested environments.

**Assumptions:**

- Environment transition dynamics remain Markovian; only reward feedback is non-Markovian
- Rewards satisfy the Past-Invariant (PI) condition: relative action credit within a signal interval is independent of the history before the interval began
- Signal interval lengths $n_i$ are drawn i.i.d. from a fixed distribution $q_n(\cdot)$
- The reward $r(\tau_i)$ is revealed only at the *end* of the $i$-th signal interval

---

## 3. Mathematical Formulation

**Core Concept - DRMDP:**

Time steps are divided into consecutive *signal intervals* $[t_i, t_{i+1})$ of random length $n_i \sim q_n(\cdot)$. The environment transition is still Markovian, but the reward $r(\tau_i)$ depends on the entire state-action trajectory $\tau_i = (s_{t_i}, a_{t_i}, \ldots, s_{t_i+n_i-1}, a_{t_i+n_i-1})$ and is revealed only at step $t_{i+1} - 1$.

The discounted cumulative reward of a trajectory is:
$$R(\tau, n) := \sum_{i=1}^{\infty} \gamma^{t_{i+1}-1} r(\tau_i)$$

**The New Q-Function (Eq. 3):**

Because rewards are non-Markovian, the standard $Q(s_t, a_t)$ is ambiguous. The paper defines:
$$Q^{\pi}(\tau_{t_i:t+1}) := \mathbb{E}_{(\tau,n)\sim\pi}\!\left[\sum_{j=i}^{\infty} \gamma^{t_{j+1}-t-1} r(\tau_j)\;\Big|\;\tau_{t_i:t+1}\right]$$

where $\tau_{t_i:t+1}$ is the trajectory segment from the **start of the current signal interval** $t_i$ through step $t$. This Q-function takes the full within-interval history as input.

**HC-Decomposition (Eq. 6):**

To make approximation practical, the paper decomposes:
$$Q_\phi(\tau_{t_i:t+1}) = H_\phi(\tau_{t_i:t}) + C_\phi(s_t, a_t)$$

- $H_\phi(\tau_{t_i:t})$: **History** component — a function of *all previous steps within the current signal interval* (not a fixed sliding window; resets at each new interval)
- $C_\phi(s_t, a_t)$: **Current** component — depends only on the current state-action pair

The key motivation: under Markovian dynamics, the current step $(s_t, a_t)$ has more future influence than past steps, so separating it simplifies learning $C_\phi$.

**Policy Gradient with HC-Decomposition (Eq. 7):**

The policy gradient depends **only on** $C_\phi$:
$$\nabla_\theta J^{\mathrm{HC}}(\pi_\theta) = \mathbb{E}_D\!\left[\nabla_{a_t} C_\phi(s_t, a_t)\Big|_{\pi_\theta(s_t, t-t_i)} \nabla_\theta \pi_\theta(s_t, t-t_i)\right]$$

This means the gradient is not affected by the history trajectory $\tau_{t_i:t}$, reducing variance compared to the full Q-gradient in Eq. (5).

**HC Training Loss (Eq. 8):**

$$\mathcal{L}_\phi^{\mathrm{HC}} = \mathbb{E}_D\!\left[\bigl(R_t + \gamma(\hat{H}_\phi(\tau_{t_j:t+1}) + \hat{C}_\phi(s_{t+1}, a'_{t+1})) - (H_\phi(\tau_{t_i:t}) + C_\phi(s_t, a_t))\bigr)^2\right] + \lambda \mathcal{L}_{\mathrm{reg}}(H_\phi)$$

where $\tau_{t_j:t+1}$ extends the current interval if $t$ is not the last step, or starts a new one otherwise.

**Regulariser on $H_\phi$ (Eq. 9):**

$$\mathcal{L}_{\mathrm{reg}}(H_\phi) = \mathbb{E}_D\!\left[(H_\phi(\tau_i) - r(\tau_i))^2\right]$$

This prevents $H_\phi$ from absorbing too much credit away from $C_\phi$, which would degrade the policy gradient. $\lambda$ is searched over $\{0.0, 0.05, 0.5, 5.0\}$ per variant.

**Convergence Guarantee (Tabular Case):**

Fact 3.1 proves that $Q^\pi(\tau_{t_i:t+1})$ is the **unique fixed point** of the MSE objective in Eq. (4). Proposition 3.3 proves the Policy Improvement Theorem: policy iteration w.r.t. $Q^{\pi_k}$ produces $\pi_{k+1}$ with $J(\pi_{k+1}) \geq J(\pi_k)$. These guarantees hold for any PI-DRMDP despite the non-Markovian rewards.

---

## 4. Architecture & Model Design

**Overall System Architecture:**

```
Current signal interval history: τ_{t_i:t} (resets each interval)
    ↓
    ├─→ H_φ(τ_{t_i:t})  ──→ History component (scalar)
    │   (one of 3 variants below)
    │
    ├─→ C_φ(s_t, a_t)   ──→ Current component (scalar)
    │   (2-layer MLP, 256 units)
    │
    └─→ H_φ + C_φ ──→ Total Q-value for critic update
         ↓
    Policy π_θ(a_t | s_t, t-t_i) ──→ uses ∇C_φ only (lower variance)
         ↓
    Environment: Markovian transitions, delayed aggregate reward R_t
```

**H_φ Implementations (three variants evaluated):**

**Q-HC-RNN**
- GRU over $(s, a)$ pairs within the current signal interval
- Each step's $(s_{t'}, a_{t'})$ first projected through FC(48 units, ReLU), then fed to GRU(48-unit hidden state)
- $H_\phi(\tau_{t_i:t})$ = GRU output at step $t$
- $\lambda = 5.0$

**Q-HC-Singleton**
- $H_\phi(\tau_{t_i:t}) = \sum_{j \in [t_i:t]} b_\phi(\tau_{j:j+1})$
- $b_\phi$: 2-layer MLP, 64 hidden units each
- Representation power: lowest; optimization: easiest
- $\lambda = 0.05$

**Q-HC-Pairwise-K** (K=1 or K=3)
- $H_\phi(\tau_{t_i:t}) = \sum_{k \in [0:K+1]} \sum_{j \in [t_i:t-k]} c_\phi^k(\tau_{j:j+1} \circ \tau_{j+k:j+k+1})$
- K=1: $c_\phi^0$, $c_\phi^1$ are 2-layer MLPs, 64 units each; $\lambda = 0.5$
- K=3: $c_\phi^0, \ldots, c_\phi^3$ are 2-layer MLPs, 48 units each; $\lambda = 5.0$
- Representation power: between Singleton and RNN

**C_φ (Current component, shared across variants):**
- Input: concatenation of $(s_t, a_t)$
- 2-layer MLP, 256 units each, ReLU
- Output: scalar $C_\phi(s_t, a_t)$

**Actor (Policy) Network:**
- Input: current state $s_t$ with appended normalized $(t - t_i)$ (position in current interval)
- 2-layer MLP, 256 units, SAC-style squashed Gaussian
- Gradient uses only $\nabla_{a_t} C_\phi(s_t, a_t)$, not $H_\phi$

**Model Specifications (from Table 2):**

| Component | Specification |
|-----------|---------------|
| $Q_\phi$, $\pi_\theta$ hidden layers | 2 layers, 256 units each |
| Non-linearity | ReLU |
| History window | Current signal interval (length $n_i \sim q_n(\cdot)$, resets each interval) |
| Q-HC-RNN encoder | FC(48) → GRU(48) |
| Q-HC-Singleton $b_\phi$ | 2-layer MLP, 64 units |
| Q-HC-Pairwise-K networks | 2-layer MLP, 64 or 48 units per $c_\phi^k$ |
| Action distribution | Squashed Gaussian (tanh) |
| Target network update | Soft, $\tau = 0.005$ |
| State augmentation | Append normalised $(t - t_i)$ to all inputs |

---

## 5. Algorithm & Implementation Details

**Main Algorithm (Algorithm 1 from paper):**

```
Choose Q_φ structure: Q-RNN, Q-HC-Pairwise-K, Q-HC-Singleton, etc.
Initialize π_θ, Q_φ, target networks Q̂_φ, D ← ∅

for each env step:
    a_t ~ π_θ(·| s_t, t - t_i)           # policy conditioned on interval position
    Take a_t, observe s_{t+1}, R_t         # R_t = 0 unless t = t_{i+1}-1
    Add (s_t, a_t, R_t, s_{t+1}) to D

    for each gradient step:
        Sample batch (τ_{t_i:t+1}, R_t, s_{t+1}) from D
        # τ_{t_i:t+1}: full history within current interval up to this step
        Update Q_φ by minimising Eq. (8): HC loss + λ·L_reg(H_φ)
        Update π_θ using Eq. (7): gradient through C_φ only
        Soft-update target networks: Q̂_φ ← (1-τ)Q̂_φ + τ Q_φ
```

**Training Procedure:**

1. **Signal Interval Tracking:**
   - Track current interval start $t_i$ and step position $t - t_i$ throughout
   - At each step, store the trajectory segment $\tau_{t_i:t+1}$ (all $(s, a)$ pairs since interval start) with the transition
   - When the interval ends (reward $R_t$ is revealed), that reward labels the full window

2. **History $H_\phi$ at Sample Time:**
   - For each sampled transition, the history input is $\tau_{t_i:t}$ — the trajectory from interval start to the step *before* the current action
   - This window is **not fixed-size**: it grows during the interval and resets to empty at each new interval start
   - In practice, the replay buffer stores these interval-aligned trajectory segments

3. **Target Computation:**
   - Sample $a'_{t+1} \sim \pi_\theta(\cdot | s_{t+1}, 0)$ if $t+1$ starts a new interval, else $\pi_\theta(\cdot | s_{t+1}, t+1-t_i)$
   - Target $= R_t + \gamma (\hat{H}_\phi(\tau_{t_j:t+1}) + \hat{C}_\phi(s_{t+1}, a'_{t+1}))$

4. **Network Updates:**
   - **$H_\phi$ and $C_\phi$:** Updated jointly by minimising Eq. (8); regulariser $\mathcal{L}_{\mathrm{reg}}$ constrains $H_\phi$
   - **$\pi_\theta$:** SAC-style gradient using only $\nabla_{a_t} C_\phi(s_t, a_t)$

**Hyperparameters (Table 2 from paper):**

| Hyperparameter | Value |
|---|---|
| $Q_\phi$, $\pi_\theta$ architecture | 2-layer MLPs, 256 units each |
| Non-linearity | ReLU |
| Batch size | 128 |
| Discount factor $\gamma$ | 0.99 |
| Optimizer | Adam, lr = $3 \times 10^{-4}$ |
| Entropy target | $-|A|$ (auto-tuned in SAC) |
| Target smoothing $\tau$ | 0.005 |
| Replay buffer | $\geq 1\mathrm{M}$ transitions |
| Target update interval | 1 |
| Gradient steps per env step | 1 |
| Regularisation $\lambda$ | Searched in $\{0.0, 0.05, 0.5, 5.0\}$ per variant |
| Signal interval length $q_n$ | $U(15, 20)$ in main experiments; $\delta(10), \delta(20), \delta(40), \delta(60)$ also tested |

**Implementation Notes:**

- The normalised interval position $(t - t_i)$ is appended to the state input for both $Q_\phi$ and $\pi_\theta$
- The Q-HC-RNN variant uses CDQ (clipped double Q-learning from TD3/SAC) with two critics; Q-HC-Singleton and Q-HC-Pairwise use a single critic
- Experiments use SAC (maximum entropy objective); TD3-based variants are also tested in Appendix C.1 with consistent results
- Hardware: GeForce GTX 1080 Ti + Intel Xeon E5-2630; each single run completes within 36 hours

---

## 6. Datasets Used

**Experiment Environment:**

**1. PI-DRMDP Locomotion Tasks**

**Source:** OpenAI Gym continuous control environments

**Task Type:** Continuous control locomotion (movement control)

**Task Specifications:**

| Task Name | Environment | Notes |
|-----------|------------|-------|
| Walker2d-v2 | MuJoCo | Primary locomotion benchmark |
| HalfCheetah-v2 | MuJoCo | Primary locomotion benchmark |
| Hopper-v2 | MuJoCo | Primary locomotion benchmark |
| Ant-v2 | MuJoCo | Primary locomotion benchmark |
| Reacher-v2 | MuJoCo | Included in comparative evaluation |

**Reward Delay Configuration:**
- The environment is wrapped so that per-step rewards are accumulated over a signal interval
- Signal interval length $n_i$ drawn from $q_n(\cdot)$; main experiments use $q_n = U(15, 20)$
- $R_t = r(\tau_i)$ emitted only at the end of interval $i$; zero at all intermediate steps
- Sum-form reward: $r(\tau_i) = \sum_{t=t_i}^{t_i+n_i-1} \hat{r}(s_t, a_t)$ where $\hat{r}$ is the standard per-step reward
- Appendix C.1 also tests $\delta(10), \delta(20), \delta(40), \delta(60)$ and non-sum-form rewards (Max, Square)

**Training Setup:**
- Online, off-policy learning; replay buffer large enough for 1M samples
- Training: 1M environment steps per task (Reacher-v2: 200k steps)
- 6–7 random seeds per experiment

**Dataset Characteristics:**

| Characteristic | Details |
|---|---|
| Continuous vs Discrete | Continuous action spaces |
| State representation | Raw MuJoCo observations (position, velocity) |
| Action space | Box(-1, 1) continuous |
| Reward signal | Delayed aggregate (sum over $n_i$ steps), emitted once per interval |
| Trajectory diversity | Multiple seeds, various $q_n$ distributions |

---

## 7. Experimental Results

**Main Results - Performance Comparison (Figure 2, 3):**

**Metric:** Relative Average Performance (RAP) — episodic return after 1M steps normalised by Oracle SAC (trained on dense rewards for 1M steps). Higher is better; 1.0 = matches Oracle SAC.

**Baselines:**
- **Vanilla SAC:** Standard off-policy actor-critic; cannot handle non-Markovian rewards
- **Q-RNN:** SAC with GRU-based Q-function $Q^\pi(\tau_{t_i:t+1})$ approximated by a GRU (128-unit hidden state); naive sequence-to-Q approach
- **LIRPG** (Zheng et al., 2018): On-policy (PPO-based); uses meta-learned intrinsic rewards
- **RUDDER** (Arjona-Medina et al., 2019): On-policy (PPO-based); RNN-based reward decomposition
- **SAC-IRCR** (Gangwani et al., 2020): Off-policy; trajectory-space smoothed guidance rewards

**Design Evaluation (Figure 2a, $q_n = U(15,20)$, 7 seeds):**

All Q-HC variants outperform both vanilla SAC and Q-RNN. HC-decomposition is more sample-efficient and stable than Q-RNN, which suffers from high gradient variance when approximating the full sequential Q-function.

**Scaling with interval length (Table 1 — RAP averaged over 4 tasks: Hopper, HalfCheetah, Walker2d, Ant):**

| $q_n$ | SAC | Q-RNN | Q-HC-Singleton | Q-HC-Pairwise-1 | Q-HC-Pairwise-3 | Q-HC-RNN |
|---|---|---|---|---|---|---|
| $\delta(10)$ | 0.62 | 0.61 | 0.91 | 0.89 | 0.92 | **0.98** |
| $U(10,20)$ | 0.48 | 0.52 | 0.95 | 0.84 | 0.95 | **0.96** |
| $U(15,20)$ | 0.48 | 0.61 | 0.89 | **0.99** | 0.95 | 0.93 |
| $\delta(20)$ | 0.51 | 0.53 | 0.91 | 0.93 | 0.98 | **0.99** |
| $\delta(40)$ | 0.30 | 0.38 | 0.90 | **0.94** | 0.85 | 0.87 |
| $\delta(60)$ | 0.17 | N.A. | **0.87** | 0.81 | 0.88 | N.A. |

**Comparative Evaluation (Figure 3, $q_n = \delta(20)$, 7 seeds):**

Q-HC-Pairwise-1 and Q-HC-RNN both strongly outperform all baselines on all 5 tasks (Reacher-v2, HalfCheetah-v2, Hopper-v2, Walker2d-v2, Ant-v2). LIRPG and RUDDER perform poorly (on-policy). SAC-IRCR is inconsistent (good on Hopper-v2, poor on Ant-v2, Walker2d-v2, Reacher-v2). Q-HC achieves near-Oracle-SAC performance.

**Non-sum-form tasks (Appendix C.1, Figure 5):**
Q-HC variants still outperform Q-RNN and SAC on Max-form and Square-form reward tasks, demonstrating that HC-decomposition is not limited to sum-form PI-DRMDPs.

### Ablation Studies

**Regularisation $\mathcal{L}_{\mathrm{reg}}$ (Figure 2b, 7b):**
- Regularisation is critical for Q-HC-RNN (complex $H_\phi$ can deprive $C_\phi$ of credit without it)
- Q-HC-Singleton benefits less from regularisation; simple structure acts as implicit regulariser

**HC-Decomposition ablation (Figure 8):**
- Q-Singleton (same sum structure, but no H/C separation) and Q-Pairwise-1 (no decomposition) perform worse than Q-RNN, confirming that the *separation* of H and C (not just the sum structure) is essential

**Policy gradient variance (Figure 1):**
- HC-decomposition produces significantly lower policy gradient variance than Q-RNN across all tasks, directly validating the theoretical motivation

### Analysis

**What Works Well:**
- Handles long reward delays (up to $\delta(60)$ steps) where SAC degrades severely (RAP 0.17 vs 0.87)
- Consistent across diverse tasks and interval length distributions
- Near-Oracle-SAC performance achievable with Q-HC on 1M steps (rewards every 20 steps)
- Lower gradient variance than Q-RNN; more stable training

**Limitations:**
- Storing full within-interval trajectories in the replay buffer increases memory usage
- Q-HC-RNN requires regularisation ($\lambda = 5.0$) to avoid ill-optimised $C_\phi$
- Performance of SAC-IRCR is inconsistent; the paper notes this as a known failure mode of smoothing-based guidance rewards

---

## 8. Code & Implementation References

**Official Implementation Status:**

Unfortunately, the official code for this paper is **not publicly available** as of 2022. The paper was published at ICML 2022 with the focus on theoretical contributions and experimental validation, but no code repository was released.

**Related Official Resources:**

**Authors:**
- Beining Han: [Personal Website](https://beininghan.github.io/)
- Yuan Zhou: [BIMSA Faculty Page](https://www.bimsa.cn/detail/YuanZhou.html)

**Publication Repositories:**
- [PMLR Paper Page](https://proceedings.mlr.press/v162/han22e.html)
- [arXiv Preprint](https://arxiv.org/abs/2106.11854)
- [OpenReview Forum](https://openreview.net/forum?id=nsjkNB2oKsQ)

**Framework & Dependencies:**

While no official code is released, implementation would require:

```txt
Python 3.7+
PyTorch 1.8+
NumPy
MuJoCo (mjpro150 or later)
OpenAI Gym
```

**Referenced Base Implementations:**

The paper builds on established RL frameworks:

1. **Soft Actor-Critic (SAC):** [SAC Paper](https://arxiv.org/abs/1801.01290)
   - Base algorithm that Q-HC extends
   - PyTorch SAC implementations available on GitHub

2. **PyTorch RL Libraries with SAC:**
   - [PyTorchRL](https://pytorch.org/rl/) - Official PyTorch reinforcement learning library
   - [denisyarats/pytorch_sac](https://github.com/denisyarats/pytorch_sac) - Clean SAC implementation
   - [quantumiracle/Popular-RL-Algorithms](https://github.com/quantumiracle/Popular-RL-Algorithms) - Multiple algorithms including SAC

**Implementation Components (from pseudocode):**

The paper describes sufficient detail for implementation:

1. **$H_\phi$ — Q-HC-Singleton (simplest variant):**
   ```python
   # H_φ(τ_{t_i:t}) = sum of b_φ(s_j, a_j) for j in [t_i, t)
   class H_Singleton(nn.Module):
       def __init__(self, state_dim, action_dim, hidden_dim=64):
           self.b = nn.Sequential(
               nn.Linear(state_dim + action_dim, hidden_dim), nn.ReLU(),
               nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
               nn.Linear(hidden_dim, 1)
           )
       def forward(self, interval_obs, interval_acts):
           # interval_obs: (T, B, state_dim), interval_acts: (T, B, action_dim)
           x = torch.cat([interval_obs, interval_acts], dim=-1)
           return self.b(x).sum(dim=0)  # sum over steps in interval
   ```

2. **$H_\phi$ — Q-HC-RNN:**
   ```python
   # H_φ(τ_{t_i:t}): GRU over within-interval (s, a) pairs
   class H_RNN(nn.Module):
       def __init__(self, state_dim, action_dim, hidden_dim=48):
           self.proj = nn.Linear(state_dim + action_dim, hidden_dim)
           self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=False)
           self.out = nn.Linear(hidden_dim, 1)
       def forward(self, interval_obs, interval_acts):
           x = F.relu(self.proj(torch.cat([interval_obs, interval_acts], -1)))
           _, h = self.gru(x)          # h: (1, B, hidden_dim)
           return self.out(h.squeeze(0))
   ```

3. **$C_\phi$ — Current component:**
   ```python
   # C_φ(s_t, a_t): standard 2-layer MLP
   class C_Current(nn.Module):
       def __init__(self, state_dim, action_dim, hidden_dim=256):
           self.net = nn.Sequential(
               nn.Linear(state_dim + action_dim, hidden_dim), nn.ReLU(),
               nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
               nn.Linear(hidden_dim, 1)
           )
       def forward(self, state, action):
           return self.net(torch.cat([state, action], dim=-1))
   ```

**Reproducibility Resources:**

- MuJoCo environment setup: [OpenAI Gym](https://gym.openai.com/docs/environments/mujoco/)
- SAC algorithm details: [SAC Paper & Code](https://github.com/rail-berkeley/softlearning)
- Continuous control benchmarks: [Benchmarking DRL Paper](https://arxiv.org/pdf/1604.06778)

---

## 9. Reproducibility Checklist

### Fully Specified

- [x] **Model architecture:** Three $H_\phi$ variants (RNN, Singleton, Pairwise-K) with exact hidden dims (Section D.1)
- [x] **Loss function:** Eq. (8) — TD error + $\lambda \mathcal{L}_{\mathrm{reg}}(H_\phi)$; regulariser Eq. (9)
- [x] **Optimizer settings:** Adam, lr = $3 \times 10^{-4}$, batch size 128 (Table 2)
- [x] **Network dimensions:** All $H_\phi$ and $C_\phi$ dims given in Appendix D.1; shared SAC in Table 2
- [x] **Activation functions:** ReLU for MLPs, GRU for Q-HC-RNN, squashed Gaussian actor
- [x] **Baseline algorithms:** Q-RNN, LIRPG, RUDDER, SAC-IRCR, vanilla SAC — all described
- [x] **Evaluation metric:** Relative Average Performance (RAP) w.r.t. Oracle SAC at 1M steps (Appendix D.5)
- [x] **HC-Decomposition structure:** Eq. (6), three variants, with policy gradient Eq. (7)
- [x] **History:** Within-interval trajectory $\tau_{t_i:t}$ (not a fixed sliding window); resets at interval boundaries
- [x] **Hardware:** GeForce GTX 1080 Ti + Intel Xeon E5-2630; ≤36 hours per run
- [x] **Number of seeds:** 6–7 seeds per experiment

### Partially Specified

- [ ] **Exact $\lambda$ per task:** Searched in $\{0.0, 0.05, 0.5, 5.0\}$ but chosen value for each task not listed
- [ ] **Training duration (Reacher-v2):** Shorter than other tasks (200k steps from figures) but not stated

### Not Specified

- [ ] **Random seeds:** Exact seed values not provided
- [ ] **Source code:** No official repository released

### Known Implementation Challenges

1. **Signal interval tracking:** Must correctly maintain $t_i$ (interval start) and $t - t_i$ (position within interval) throughout both data collection and replay sampling.

2. **Replay buffer design:** Each stored transition needs the full $\tau_{t_i:t+1}$ (variable-length within-interval history), which requires careful storage (e.g., storing interval boundaries alongside transitions).

3. **Reward timing:** $R_t = r(\tau_i)$ is non-zero only at $t = t_{i+1} - 1$; zero at all other steps. Misassigning rewards breaks learning.

4. **MuJoCo setup:** MuJoCo200 used; modern replacements (dm_control, mujoco-mjx) should work.

5. **Variance in results:** Paper reports 6–7 seeds; continuous control RL has high variance; reproducing exact curves requires many seeds.

6. **Regulariser input (Eq. 9) differs from HC decomposition input (Eq. 6):**
   In the HC decomposition, $H_\phi$ receives $\tau_{t_i:t}$ — the interval history *excluding* the current step $(s_t, a_t)$, which goes to $C_\phi$. At the last step of an $n$-step interval, $H_\phi$ sees $n-1$ pairs. However, the regulariser (Eq. 9) evaluates $H_\phi(\tau_i)$ where $\tau_i$ is the *complete* interval — all $n$ pairs. This is a separate forward pass on a different input than what $H_\phi$ receives during Q evaluation. The implementation must construct the full interval (e.g. by appending the current $(s_t, a_t)$ to the stored history) for the regulariser, not reuse the $n-1$-step history from the Q decomposition. No official code exists to confirm this interpretation, but the notation is unambiguous: $\tau_i$ is defined in Definition 2.1 as the full interval trajectory.

---

## 10. Implementation Roadmap

### Phase 1: Environment & Data Pipeline (Days 1-3)

**Goal:** Build MuJoCo environment wrapper with delayed reward mechanism

**Tasks:**

1. [ ] Set up MuJoCo environment with gym
   - Install mujoco-py and OpenAI Gym
   - Verify MuJoCo license/installation
   - Load Walker2d-v2 and test forward pass

2. [ ] Implement delayed reward wrapper
   - Create custom Gym wrapper that accumulates rewards
   - Implement delay window (uniform sample 15-20 steps)
   - Track reward sums correctly across episodes

3. [ ] Build replay buffer with trajectory history
   - Store (state, action, accumulated_reward, next_state) tuples with interval start index $t_i$
   - Reconstruct within-interval trajectory $\tau_{t_i:t}$ at sample time from stored transitions
   - Support efficient batching with sequence padding (variable-length interval histories)

4. [ ] Implement state normalization
   - Running mean/variance of observations
   - Normalize on-the-fly during collection
   - Ensure consistency between train and eval

**Checkpoint:** Can collect 1000-step trajectories from MuJoCo with correct delayed rewards displayed

### Phase 2: Core Model Components (Days 4-7)

**Goal:** Implement neural network architecture ($H_\phi$, $C_\phi$, Actor)

**Tasks:**

1. [ ] Implement $H_\phi$ (start with Q-HC-Singleton, easiest to validate)
   - Input: variable-length sequence $\tau_{t_i:t}$ of $(s, a)$ pairs within current interval
   - Q-HC-Singleton: sum $b_\phi(\tau_{j:j+1})$ over $j \in [t_i:t]$; $b_\phi$ is 2-layer MLP, 64 units
   - Q-HC-RNN: FC(48) → GRU(48) over the interval; output at step $t$
   - Validate that output is zero for empty interval (start of new interval)

2. [ ] Implement $C_\phi$ (Current component)
   - Input: concatenation of $(s_t, a_t)$ and normalised $(t - t_i)$
   - 2-layer MLP, 256 units each, ReLU
   - Output: single scalar; same architecture as standard SAC critic

3. [ ] Implement Actor Network
   - Input: $(s_t, t - t_i)$ — current state plus interval position
   - 2-layer MLP, 256 units, SAC-style squashed Gaussian
   - Policy gradient uses only $\nabla_{a_t} C_\phi(s_t, a_t)$

**Checkpoint:** Forward pass on batch with $\tau_{t_i:t}$ of varying length produces valid $H_\phi + C_\phi$ Q-values and valid action samples

### Phase 3: Training Loop & Loss Functions (Days 8-11)

**Goal:** Implement SAC training with HC-decomposition updates

**Tasks:**

1. [ ] Implement Q-learning target computation
   - Sample next actions from policy: $a' \sim \pi_\theta(\cdot | s_{t+1}, t+1-t_j)$
   - Compute target $= R_t + \gamma(\hat{H}_\phi(\tau_{t_j:t+1}) + \hat{C}_\phi(s_{t+1}, a'))$
   - Note: $\tau_{t_j:t+1}$ may start a new interval if current step was last in interval

2. [ ] Implement HC loss (Eq. 8) and regulariser (Eq. 9)
   - Joint update of $H_\phi$ and $C_\phi$ via MSE on TD error
   - Add $\lambda \cdot \mathbb{E}[(H_\phi(\tau_i) - R_t)^2]$ regulariser term
   - Verify $\lambda$ controls credit balance between $H_\phi$ and $C_\phi$

3. [ ] Implement actor loss and update
   - Policy gradient: only $\nabla_{a_t} C_\phi(s_t, a_t)$ — verify no gradient through $H_\phi$
   - SAC-style entropy bonus with auto-tuned $\alpha$
   - Entropy target: $-|A|$

4. [ ] Add optimization details
   - Adam optimizer for all networks
   - Learning rate scheduling (if needed)
   - Gradient clipping
   - Soft target updates (tau=0.005)

**Checkpoint:** Single training step produces updated networks without NaN/Inf; losses decrease over iterations

### Phase 4: Full Training & Evaluation (Days 12-15)

**Goal:** Train complete algorithm and reproduce baseline results

**Tasks:**

1. [ ] Implement main training loop
   - Environment interaction loop
   - Replay buffer sampling
   - Periodic network updates (every 2 steps)
   - Logging (loss, return, sample efficiency)

2. [ ] Implement evaluation protocol
   - Run 10 deterministic policy rollouts per evaluation
   - Track average return and std dev
   - Save checkpoint when improving
   - Plot learning curves

3. [ ] Implement SAC baseline
   - Standard SAC without HC-decomposition
   - Use vanilla Q-network (not $H_\phi + C_\phi$ split)
   - For comparison with Q-HC results

4. [ ] Implement Q-RNN baseline
   - GRU-based Q-function (naive approach)
   - No decomposition, single Q(h, s, a) output
   - For comparison with Q-HC results

5. [ ] Run experiments
   - Train Q-HC on Walker2d-v2 for 2M steps
   - Compare with SAC baseline
   - Compare with Q-RNN baseline
   - Run 3-5 random seeds for variance estimation

**Checkpoint:** Q-HC achieves 80%+ performance relative to paper results on Walker2d-v2

### Phase 5: Ablation Studies & Analysis (Days 16-18)

**Goal:** Validate design choices through ablation studies

**Tasks:**

1. [ ] Ablate $H_\phi$ component
   - Train $C_\phi$-only (no history component) — equivalent to vanilla SAC critic
   - Measure performance degradation vs full $H_\phi + C_\phi$

2. [ ] Ablate HC-decomposition structure
   - Train Q-Singleton (same sum-over-steps structure, but no $H/C$ separation — output is $Q(\tau_{t_i:t+1})$ directly)
   - Should see worse performance than Q-HC-Singleton (matches Figure 8)

3. [ ] Ablate regulariser $\mathcal{L}_{\mathrm{reg}}$
   - Set $\lambda = 0$ for Q-HC-RNN; compare with regularised version
   - Should see degradation, especially on HalfCheetah-v2 and Ant-v2

4. [ ] Analyse interval length scaling
   - Train with $\delta(10), \delta(20), \delta(40), \delta(60)$
   - Plot RAP vs interval length; compare with Table 1 targets

5. [ ] Generate analysis plots
   - Learning curves (return vs steps)
   - Baseline comparisons
   - Ablation results summary
   - Sample efficiency analysis

**Checkpoint:** Ablation results match paper intuitions; head+critic decomposition clearly beneficial

### Phase 6: Scaling & Extensions (Optional, Days 19-20)

**Goal:** Extend to additional environments and tasks

**Tasks:**

1. [ ] Test on additional MuJoCo tasks
   - HalfCheetah-v2
   - Hopper-v2
   - Reacher-v2
   - Ant-v2
   - Verify consistent improvements

2. [ ] Test with longer delays
   - Vary reward delay from 15-20 to 30-40 steps
   - Verify algorithm scales to longer delays
   - Plot performance vs delay length

3. [ ] Explore alternative history encodings
   - Compare GRU vs LSTM vs Transformer
   - Test fixed vs learned history window
   - Analyze encoder computational cost

4. [ ] Performance optimization
   - Profile code to find bottlenecks
   - Optimize replay buffer indexing
   - Batch RNN processing for speed

**Checkpoint:** Trained on 5 different tasks; clear improvements over baselines

### Critical Implementation Details

**Pay Special Attention To:**

1. **Signal Interval Tracking:**
   - Track $t_i$ (start of current interval) throughout both collection and replay
   - Append normalised $t - t_i$ to all state inputs (policy and Q-networks)
   - Reset interval history $\tau_{t_i:t}$ at each new interval start; never let it span intervals

2. **Decomposition Correctness:**
   - $H_\phi$ and $C_\phi$ are trained jointly via Eq. (8); there is no separate per-network target
   - Policy gradient uses ONLY $\nabla_{a_t} C_\phi(s_t, a_t)$ — stop gradients through $H_\phi$
   - Regulariser $\lambda \mathcal{L}_{\mathrm{reg}}$ is essential to prevent $H_\phi$ from absorbing all reward credit

3. **Reward Signal Handling:**
   - $R_t = r(\tau_i)$ is non-zero only at the last step of each interval; zero elsewhere
   - Episode boundaries must not cross interval boundaries (truncate intervals at episode end)

4. **Target Network Synchronization:**
   - $H_\phi$ and $C_\phi$ share a single set of target network parameters; update with soft coefficient $\tau = 0.005$

5. **Entropy Coefficient Tuning:**
   - SAC auto-tunes $\alpha$; default entropy target = $-|A|$
   - Monitor entropy during training; entropy maximisation helps avoid local optima in delayed reward tasks

### Common Pitfalls

1. **History Encoder Gradient Flow:**
   - Q-HC-RNN uses GRU; watch for vanishing gradients through long within-interval sequences
   - The regulariser $\mathcal{L}_{\mathrm{reg}}$ also helps stabilise GRU training by anchoring $H_\phi$

2. **Delayed Reward Timing:**
   - Off-by-one error in interval end assignment causes reward to be attributed to wrong step
   - Carefully verify: $R_t$ should be non-zero only at step $t = t_{i+1} - 1$
   - Test on simple $\delta(2)$ delay first

3. **Q-Function Explosion:**
   - Without proper target networks, Q-values can diverge
   - Use soft updates (tau ~ 0.005) and clipped loss
   - Monitor Q-value statistics during training

4. **Policy Collapse:**
   - If entropy bonus too small, policy collapses to deterministic
   - If too large, policy becomes too exploratory
   - Auto-tuning via SAC should handle, but verify

5. **Batch Normalization Issues:**
   - BN during training/eval mismatch causes problems
   - Use layer norm instead for RL (more stable)
   - Or ensure eval uses correct BN statistics

### Testing Strategy

**Unit Tests:**
```
- Test H_φ (Singleton) on empty interval → output 0
- Test H_φ (Singleton) on 1-step interval → b_φ(s, a)
- Test H_φ (RNN) reset at interval boundary → same output as fresh GRU init
- Test C_φ produces valid Q-values for (s, a) inputs
- Test Actor produces valid bounded actions and correct log-prob
- Test reward accumulation: R_t non-zero only at interval end
```

**Integration Tests:**
```
- One env step: interval tracking increments t-t_i correctly
- Interval end: R_t assigned; t_i resets; H_φ history cleared
- Replay buffer sampling: τ_{t_i:t} correctly reconstructed from stored data
- HC loss computation without NaN/Inf; regulariser term positive
- Policy gradient: verify no gradient flows through H_φ parameters
```

**Sanity Checks:**
```
1. Overfit on single interval trajectory → loss → 0
2. Q-HC-Singleton on δ(3) delay → should learn faster than Q-RNN
3. RAP of Q-HC-Singleton ≥ 0.87 on δ(60) delay (Table 1 target)
4. Policy gradient variance lower for Q-HC than Q-RNN (Figure 1 pattern)
```

**Validation:**
```
Target (Table 1, U(15,20), 4-task average):
- Q-HC-Singleton: RAP ≥ 0.85
- Q-HC-Pairwise-1: RAP ≥ 0.80
- Strongly outperforms SAC (RAP 0.48) and Q-RNN (RAP 0.61)
```

---

## References & Related Work

**Foundation Papers:**

1. [Soft Actor-Critic](https://arxiv.org/abs/1801.01290) - Haarnoja et al., 2018
   - Base algorithm extended with HC-decomposition

2. [Twin Delayed DDPG (TD3)](https://arxiv.org/abs/1802.09477) - Fujimoto et al., 2018
   - Double Q-network trick borrowed for stability

3. [Deep Deterministic Policy Gradient (DDPG)](https://arxiv.org/abs/1509.02971) - Lillicrap et al., 2015
   - Actor-critic foundation

**Related Work on Delayed/Sparse Rewards:**

- Reinforcement learning with reward delay and partial observability
- Recurrent architectures for non-Markovian environments
- Value function decomposition and multi-task learning

**Cited by Han et al. 2022:**

- Papers on non-Markovian MDPs and partial observability (POMDPs)
- Works on off-policy RL and stability
- Research on reward shaping and curriculum learning

---

## Summary

The Han et al. 2022 paper presents a principled approach to **off-policy reinforcement learning with delayed rewards**, addressing a critical gap in real-world RL applications. The key innovation—the **HC-decomposition of Q-functions into History and Current components**—separates learning the value of the trajectory *within the current signal interval* ($H_\phi$) from learning the value of the *current action* ($C_\phi$). This reduces policy gradient variance and stabilises training.

**Key Strengths:**
- Solid theoretical foundation (PI-DRMDP formalism with convergence guarantees in tabular case)
- History component resets at each signal interval boundary — not a sliding window
- Near-Oracle-SAC performance achievable even with delays of 20–60 steps
- Three $H_\phi$ variants with different complexity/optimisation trade-offs

**For Implementation:**
- Algorithm 1 and Appendix D.1 give exact architecture specs
- Main implementation challenge: correctly tracking signal intervals and storing $\tau_{t_i:t}$ in the replay buffer
- Start with Q-HC-Singleton (simplest $H_\phi$) to validate the control loop before adding GRU complexity
- The regulariser $\lambda \mathcal{L}_{\mathrm{reg}}(H_\phi)$ is essential — do not omit it for RNN variants

The work is suitable for practitioners needing to handle real-world systems with inherent reward delays, and provides a strong foundation for further research in non-Markovian RL settings.

---

Sources:
- [PMLR Proceedings](https://proceedings.mlr.press/v162/han22e.html)
- [arXiv](https://arxiv.org/abs/2106.11854)
- [OpenReview](https://openreview.net/forum?id=nsjkNB2oKsQ)
- [ICML Virtual Poster](https://icml.cc/virtual/2022/poster/16161)
- [Author: Beining Han](https://beininghan.github.io/)
- [Author: Yuan Zhou](https://www.bimsa.cn/detail/YuanZhou.html)
