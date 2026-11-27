"""
Tests for drmdp.control.dsa: HistoryEncoder, DSAReplayBuffer, wrappers, callback.
"""

from typing import Any, Optional

import gymnasium as gym
import numpy as np
import pytest
import torch

from drmdp import rewdelay
from drmdp.control import base, dsa, hc

# ---------------------------------------------------------------------------
# TestLearnedPositionalEncoding
# ---------------------------------------------------------------------------


class TestLearnedPositionalEncoding:
    def test_output_shape(self):
        enc = dsa._LearnedPositionalEncoding(max_len=10, d_model=16)
        out = enc(seq_len=5)
        assert out.shape == (1, 5, 16)

    def test_different_lengths(self):
        enc = dsa._LearnedPositionalEncoding(max_len=10, d_model=8)
        out3 = enc(seq_len=3)
        out7 = enc(seq_len=7)
        assert out3.shape == (1, 3, 8)
        assert out7.shape == (1, 7, 8)
        torch.testing.assert_close(out3[0], out7[0, :3])


# ---------------------------------------------------------------------------
# TestTransformerHistoryEncoder
# ---------------------------------------------------------------------------


class TestTransformerHistoryEncoder:
    def test_output_shape(self):
        encoder = dsa._TransformerHistoryEncoder(
            sa_dim=6, d_model=16, nhead=2, num_layers=1, latent_dim=8, max_len=5
        )
        history = torch.randn(4, 5, 6)
        padding_mask = torch.zeros(4, 5, dtype=torch.bool)
        z = encoder(history, padding_mask)
        assert z.shape == (4, 8)

    def test_padding_mask_affects_output(self):
        """Masking more positions changes the output."""
        torch.manual_seed(42)
        encoder = dsa._TransformerHistoryEncoder(
            sa_dim=6, d_model=16, nhead=2, num_layers=1, latent_dim=8, max_len=5
        )
        encoder.eval()
        history = torch.randn(1, 5, 6)

        no_mask = torch.zeros(1, 5, dtype=torch.bool)
        with_mask = torch.tensor([[True, True, True, False, False]])

        z_full = encoder(history, no_mask)
        z_partial = encoder(history, with_mask)
        assert not torch.allclose(z_full, z_partial, atol=1e-4)

    def test_deterministic_with_seed(self):
        torch.manual_seed(0)
        enc1 = dsa._TransformerHistoryEncoder(
            sa_dim=4, d_model=8, nhead=2, num_layers=1, latent_dim=4, max_len=3
        )
        enc1.eval()
        history = torch.randn(2, 3, 4)
        mask = torch.zeros(2, 3, dtype=torch.bool)
        z1 = enc1(history, mask)

        torch.manual_seed(0)
        enc2 = dsa._TransformerHistoryEncoder(
            sa_dim=4, d_model=8, nhead=2, num_layers=1, latent_dim=4, max_len=3
        )
        enc2.eval()
        z2 = enc2(history, mask)
        torch.testing.assert_close(z1, z2)


# ---------------------------------------------------------------------------
# TestNextStatePredictor
# ---------------------------------------------------------------------------


class TestNextStatePredictor:
    def test_output_shape(self):
        pred = dsa._NextStatePredictor(
            latent_dim=8, obs_dim=3, action_dim=1, hidden_dim=32
        )
        z = torch.randn(4, 8)
        obs = torch.randn(4, 3)
        actions = torch.randn(4, 1)
        out = pred(z, obs, actions)
        assert out.shape == (4, 3)


# ---------------------------------------------------------------------------
# TestHistoryEncoder
# ---------------------------------------------------------------------------


class TestHistoryEncoder:
    def test_encode_returns_detached(self):
        enc = dsa.HistoryEncoder(
            sa_dim=4,
            obs_dim=3,
            action_dim=1,
            d_model=8,
            nhead=2,
            num_encoder_layers=1,
            latent_dim=4,
            max_len=3,
        )
        history = torch.randn(2, 3, 4)
        mask = torch.zeros(2, 3, dtype=torch.bool)
        z = enc.encode(history, mask)
        assert not z.requires_grad
        assert z.shape == (2, 4)

    def test_prediction_loss_computes_gradient(self):
        torch.manual_seed(42)
        enc = dsa.HistoryEncoder(
            sa_dim=4,
            obs_dim=3,
            action_dim=1,
            d_model=8,
            nhead=2,
            num_encoder_layers=1,
            latent_dim=4,
            max_len=3,
        )
        history = torch.randn(4, 3, 4)
        mask = torch.zeros(4, 3, dtype=torch.bool)
        obs = torch.randn(4, 3)
        actions = torch.randn(4, 1)
        next_obs = torch.randn(4, 3)
        terminals = torch.zeros(4)

        loss = enc.prediction_loss(history, mask, obs, actions, next_obs, terminals)
        loss.backward()

        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in enc.parameters()
        )
        assert has_grad

    def test_prediction_loss_skips_terminal_steps(self):
        enc = dsa.HistoryEncoder(
            sa_dim=4,
            obs_dim=3,
            action_dim=1,
            d_model=8,
            nhead=2,
            num_encoder_layers=1,
            latent_dim=4,
            max_len=3,
        )
        history = torch.randn(4, 3, 4)
        mask = torch.zeros(4, 3, dtype=torch.bool)
        obs = torch.randn(4, 3)
        actions = torch.randn(4, 1)
        next_obs = torch.randn(4, 3)
        all_terminal = torch.ones(4)

        loss = enc.prediction_loss(history, mask, obs, actions, next_obs, all_terminal)
        assert loss.item() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# TestExtractHistoryWindows
# ---------------------------------------------------------------------------


class TestExtractHistoryWindows:
    def test_per_interval_resets_at_boundary(self):
        """History resets after interval_end in per_interval mode."""
        traj = _make_trajectory(
            obs=np.arange(12, dtype=np.float32).reshape(6, 2),
            actions=np.zeros((6, 1), dtype=np.float32),
            env_rewards=np.array([0, 0, 1.5, 0, 0, 2.5], dtype=np.float32),
            interval_ends=np.array([False, False, True, False, False, True]),
        )
        windows = dsa._extract_history_windows(traj, max_delay=3, mode="per_interval")
        assert len(windows) == 6

        # Step 3 (first step of second interval): history should be empty
        # because interval_end at step 2 cleared the deque.
        np.testing.assert_allclose(windows[3].history, 0.0)

    def test_sliding_window_does_not_reset_at_boundary(self):
        """History carries across interval boundaries in sliding_window mode."""
        traj = _make_trajectory(
            obs=np.arange(12, dtype=np.float32).reshape(6, 2),
            actions=np.ones((6, 1), dtype=np.float32),
            env_rewards=np.array([0, 0, 1.5, 0, 0, 2.5], dtype=np.float32),
            interval_ends=np.array([False, False, True, False, False, True]),
        )
        windows = dsa._extract_history_windows(traj, max_delay=3, mode="sliding_window")
        # Step 3: history should NOT be empty in sliding_window mode,
        # since step 2 was added before the interval_end cleared.
        # Step 2 is added because is_interval_end doesn't clear in sliding mode.
        assert windows[3].history.any()

    def test_episode_boundary_clears_history(self):
        """Both modes clear history at episode termination."""
        traj = _make_trajectory(
            obs=np.arange(6, dtype=np.float32).reshape(3, 2),
            actions=np.zeros((3, 1), dtype=np.float32),
            env_rewards=np.array([0, 0, 1.0], dtype=np.float32),
            interval_ends=np.array([False, False, True]),
            terminal_at=-1,
        )
        for mode in ("per_interval", "sliding_window"):
            windows = dsa._extract_history_windows(traj, max_delay=3, mode=mode)
            assert len(windows) == 3

    def test_left_zero_padded(self):
        """Short histories are left-zero-padded to max_delay."""
        traj = _make_trajectory(
            obs=np.ones((3, 2), dtype=np.float32),
            actions=np.ones((3, 1), dtype=np.float32),
            env_rewards=np.array([0, 0, 1.0], dtype=np.float32),
            interval_ends=np.array([False, False, True]),
        )
        windows = dsa._extract_history_windows(traj, max_delay=5, mode="per_interval")
        # Step 0: no history yet — all zeros
        np.testing.assert_allclose(windows[0].history, 0.0)
        # Step 1: one (s,a) pair, right-aligned
        assert windows[1].history[-1].any()
        assert not windows[1].history[:-1].any()

    def test_window_has_correct_transition(self):
        obs = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
        actions = np.array([[0.1], [0.2], [0.3]], dtype=np.float32)
        traj = _make_trajectory(
            obs=obs,
            actions=actions,
            env_rewards=np.array([0, 0, 1.0], dtype=np.float32),
            interval_ends=np.array([False, False, True]),
        )
        windows = dsa._extract_history_windows(traj, max_delay=3, mode="per_interval")
        np.testing.assert_allclose(windows[1].obs, [3, 4])
        np.testing.assert_allclose(windows[1].action, [0.2])
        np.testing.assert_allclose(windows[1].next_obs, [5, 6])


# ---------------------------------------------------------------------------
# TestLatentAugmentedObsWrapper
# ---------------------------------------------------------------------------


class TestLatentAugmentedObsWrapper:
    def test_observation_space_expanded(self):
        env = gym.make("Pendulum-v1")
        base_dim = env.observation_space.shape[0]
        wrapped = dsa.LatentAugmentedObsWrapper(env, latent_dim=8)
        assert wrapped.observation_space.shape[0] == base_dim + 8

    def test_reset_pads_with_zeros(self):
        env = gym.make("Pendulum-v1")
        wrapped = dsa.LatentAugmentedObsWrapper(env, latent_dim=4)
        obs, _ = wrapped.reset()
        np.testing.assert_allclose(obs[-4:], 0.0)

    def test_step_pads_with_zeros(self):
        env = gym.make("Pendulum-v1")
        wrapped = dsa.LatentAugmentedObsWrapper(env, latent_dim=4)
        wrapped.reset()
        obs, _, _, _, _ = wrapped.step(env.action_space.sample())
        np.testing.assert_allclose(obs[-4:], 0.0)


# ---------------------------------------------------------------------------
# TestDSAEvalWrapper
# ---------------------------------------------------------------------------


class TestDSAEvalWrapper:
    def _make_eval_env(
        self, latent_dim: int = 4, max_delay: int = 3
    ) -> dsa.DSAEvalWrapper:
        env = gym.make("Pendulum-v1")
        env = rewdelay.DelayedRewardWrapper(env, rewdelay.FixedDelay(max_delay))
        env = rewdelay.ImputeMissingRewardWrapper(env, impute_value=0.0)
        env = hc.IntervalPositionWrapper(env, max_delay=max_delay)
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        env = dsa.LatentAugmentedObsWrapper(env, latent_dim=latent_dim)

        torch.manual_seed(42)
        sa_dim = obs_dim + action_dim
        history_encoder = dsa.HistoryEncoder(
            sa_dim=sa_dim,
            obs_dim=obs_dim,
            action_dim=action_dim,
            d_model=8,
            nhead=2,
            num_encoder_layers=1,
            latent_dim=latent_dim,
            max_len=max_delay,
        )
        return dsa.DSAEvalWrapper(
            env,
            history_encoder=history_encoder,
            max_delay=max_delay,
            latent_dim=latent_dim,
            obs_dim_before_augmentation=obs_dim,
            action_dim=action_dim,
            history_mode="per_interval",
        )

    def test_reset_returns_correct_dim(self):
        env = self._make_eval_env(latent_dim=4)
        obs, _ = env.reset()
        base_obs_dim = gym.make("Pendulum-v1").observation_space.shape[0]
        expected_dim = base_obs_dim + 1 + 4  # +1 interval_pos, +4 latent
        assert obs.shape == (expected_dim,)

    def test_z_changes_with_history(self):
        """Mid-interval steps should have non-zero z from accumulated history."""
        env = self._make_eval_env(latent_dim=4, max_delay=5)
        env.reset()

        z_values = []
        for _ in range(4):
            obs, _, _, _, _ = env.step(env.action_space.sample())
            z_values.append(obs[-4:].copy())

        # At least one mid-interval step should have non-zero z.
        any_nonzero = any(np.abs(zv).sum() > 1e-6 for zv in z_values)
        assert any_nonzero

    def test_reset_clears_history(self):
        env = self._make_eval_env(latent_dim=4)
        env.reset()
        for _ in range(5):
            env.step(env.action_space.sample())
        obs_after_reset, _ = env.reset()
        obs_fresh, _ = env.reset()
        np.testing.assert_allclose(obs_after_reset[-4:], obs_fresh[-4:])


# ---------------------------------------------------------------------------
# TestDSAReplayBuffer
# ---------------------------------------------------------------------------


class TestDSAReplayBuffer:
    def test_add_stores_history(self):
        """After add(), history_sa is populated for the position."""
        obs_space = gym.spaces.Box(-1, 1, (3,), dtype=np.float32)
        action_space = gym.spaces.Box(-1, 1, (1,), dtype=np.float32)
        buf = dsa.DSAReplayBuffer(
            buffer_size=10,
            observation_space=obs_space,
            action_space=action_space,
            max_delay=3,
            raw_obs_dim=3,
        )
        obs = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        next_obs = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        action = np.array([0.5], dtype=np.float32)
        buf.add(
            obs,
            next_obs,
            action,
            np.array([1.0]),
            np.array([False]),
            [{"interval_end": False}],
        )
        buf.add(
            obs,
            next_obs,
            action,
            np.array([1.0]),
            np.array([False]),
            [{"interval_end": False}],
        )

        # Second add should have history from first step
        assert buf._history_sa[1, 0].any()

    def test_per_interval_clears_at_boundary(self):
        obs_space = gym.spaces.Box(-1, 1, (2,), dtype=np.float32)
        action_space = gym.spaces.Box(-1, 1, (1,), dtype=np.float32)
        buf = dsa.DSAReplayBuffer(
            buffer_size=10,
            observation_space=obs_space,
            action_space=action_space,
            max_delay=3,
            raw_obs_dim=2,
            history_mode="per_interval",
        )
        obs = np.array([1.0, 2.0], dtype=np.float32)
        next_obs = np.zeros(2, dtype=np.float32)
        action = np.array([0.5], dtype=np.float32)

        buf.add(
            obs,
            next_obs,
            action,
            np.array([0.0]),
            np.array([False]),
            [{"interval_end": False}],
        )
        buf.add(
            obs,
            next_obs,
            action,
            np.array([1.0]),
            np.array([False]),
            [{"interval_end": True}],
        )
        buf.add(
            obs,
            next_obs,
            action,
            np.array([0.0]),
            np.array([False]),
            [{"interval_end": False}],
        )

        # Step 2 (index 2): history should be empty because interval_end at step 1
        np.testing.assert_allclose(buf._history_sa[2, 0], 0.0)

    def test_sliding_window_keeps_across_boundary(self):
        obs_space = gym.spaces.Box(-1, 1, (2,), dtype=np.float32)
        action_space = gym.spaces.Box(-1, 1, (1,), dtype=np.float32)
        buf = dsa.DSAReplayBuffer(
            buffer_size=10,
            observation_space=obs_space,
            action_space=action_space,
            max_delay=3,
            raw_obs_dim=2,
            history_mode="sliding_window",
        )
        obs = np.array([1.0, 2.0], dtype=np.float32)
        next_obs = np.zeros(2, dtype=np.float32)
        action = np.array([0.5], dtype=np.float32)

        buf.add(
            obs,
            next_obs,
            action,
            np.array([0.0]),
            np.array([False]),
            [{"interval_end": False}],
        )
        buf.add(
            obs,
            next_obs,
            action,
            np.array([1.0]),
            np.array([False]),
            [{"interval_end": True}],
        )
        buf.add(
            obs,
            next_obs,
            action,
            np.array([0.0]),
            np.array([False]),
            [{"interval_end": False}],
        )

        # Step 2: history should NOT be empty in sliding_window mode
        assert buf._history_sa[2, 0].any()

    def test_sample_returns_augmented_obs(self):
        """When encoder is set, sampled obs are augmented with z."""
        torch.manual_seed(42)
        obs_dim = 3
        action_dim = 1
        latent_dim = 4
        # Augmented obs_space (as if LatentAugmentedObsWrapper was applied)
        obs_space = gym.spaces.Box(
            -np.inf, np.inf, (obs_dim + latent_dim,), dtype=np.float32
        )
        action_space = gym.spaces.Box(-1, 1, (action_dim,), dtype=np.float32)

        history_encoder = dsa.HistoryEncoder(
            sa_dim=obs_dim + action_dim,
            obs_dim=obs_dim,
            action_dim=action_dim,
            d_model=8,
            nhead=2,
            num_encoder_layers=1,
            latent_dim=latent_dim,
            max_len=3,
        )

        buf = dsa.DSAReplayBuffer(
            buffer_size=100,
            observation_space=obs_space,
            action_space=action_space,
            max_delay=3,
            history_encoder=history_encoder,
            raw_obs_dim=obs_dim,
        )

        for _ in range(20):
            obs = np.concatenate(
                [
                    np.random.randn(obs_dim).astype(np.float32),
                    np.zeros(latent_dim, dtype=np.float32),
                ]
            )
            next_obs = np.concatenate(
                [
                    np.random.randn(obs_dim).astype(np.float32),
                    np.zeros(latent_dim, dtype=np.float32),
                ]
            )
            action = np.random.randn(action_dim).astype(np.float32)
            buf.add(
                obs,
                next_obs,
                action,
                np.array([1.0]),
                np.array([False]),
                [{"interval_end": False}],
            )

        batch = buf.sample(4)
        assert batch.observations.shape[-1] == obs_dim + latent_dim


# ---------------------------------------------------------------------------
# TestDSACallback
# ---------------------------------------------------------------------------


class TestDSACallback:
    def test_encoder_loss_decreases(self):
        """Training encoder on repeated data should decrease loss."""
        torch.manual_seed(42)
        np.random.seed(42)

        obs_dim = 3
        action_dim = 1
        sa_dim = obs_dim + action_dim
        max_delay = 3

        history_encoder = dsa.HistoryEncoder(
            sa_dim=sa_dim,
            obs_dim=obs_dim,
            action_dim=action_dim,
            d_model=16,
            nhead=2,
            num_encoder_layers=1,
            latent_dim=8,
            max_len=max_delay,
            predictor_hidden_dim=32,
        )

        windows = []
        for _ in range(50):
            obs = np.random.randn(obs_dim).astype(np.float32)
            action = np.random.randn(action_dim).astype(np.float32)
            next_obs = obs + action * 0.1  # simple dynamics
            history = np.random.randn(max_delay, sa_dim).astype(np.float32)
            windows.append(
                dsa._HistoryWindow(
                    history=history,
                    obs=obs,
                    action=action,
                    next_obs=next_obs,
                    terminal=False,
                )
            )

        callback = dsa.DSACallback(
            history_encoder=history_encoder,
            reward_model=None,
            max_delay=max_delay,
            raw_obs_dim=obs_dim,
            encoder_train_epochs=3,
            encoder_batch_size=16,
            encoder_learning_rate=1e-3,
        )
        callback._history_buffer = windows

        first_metrics = callback._update_encoder()
        for _ in range(5):
            metrics = callback._update_encoder()

        assert metrics["encoder_loss"] < first_metrics["encoder_loss"]


# ---------------------------------------------------------------------------
# TestTransformerHistoryEncoderAllPadded
# ---------------------------------------------------------------------------


class TestTransformerHistoryEncoderAllPadded:
    def test_all_padded_returns_zeros(self):
        """Fully padded input returns a zero latent vector."""
        encoder = dsa._TransformerHistoryEncoder(
            sa_dim=4, d_model=8, nhead=2, num_layers=1, latent_dim=4, max_len=3
        )
        history = torch.zeros(2, 3, 4)
        padding_mask = torch.ones(2, 3, dtype=torch.bool)
        z = encoder(history, padding_mask)
        assert z.shape == (2, 4)
        torch.testing.assert_close(z, torch.zeros(2, 4))

    def test_mixed_batch_some_all_padded(self):
        """Batch where some samples are fully padded and others are not."""
        torch.manual_seed(42)
        encoder = dsa._TransformerHistoryEncoder(
            sa_dim=4, d_model=8, nhead=2, num_layers=1, latent_dim=4, max_len=3
        )
        encoder.eval()
        history = torch.randn(3, 3, 4)
        history[1] = 0.0
        padding_mask = torch.tensor(
            [
                [True, False, False],
                [True, True, True],
                [False, False, False],
            ]
        )
        z = encoder(history, padding_mask)
        assert z.shape == (3, 4)
        torch.testing.assert_close(z[1], torch.zeros(4))
        assert z[0].abs().sum() > 0
        assert z[2].abs().sum() > 0


# ---------------------------------------------------------------------------
# TestDSAReplayBufferNextHistory
# ---------------------------------------------------------------------------


class TestDSAReplayBufferNextHistory:
    def _make_buffer_with_encoder(
        self,
        obs_dim: int = 3,
        action_dim: int = 1,
        latent_dim: int = 4,
        max_delay: int = 3,
        history_mode: str = "per_interval",
    ) -> dsa.DSAReplayBuffer:
        torch.manual_seed(42)
        obs_space = gym.spaces.Box(
            -np.inf, np.inf, (obs_dim + latent_dim,), dtype=np.float32
        )
        action_space = gym.spaces.Box(-1, 1, (action_dim,), dtype=np.float32)
        history_encoder = dsa.HistoryEncoder(
            sa_dim=obs_dim + action_dim,
            obs_dim=obs_dim,
            action_dim=action_dim,
            d_model=8,
            nhead=2,
            num_encoder_layers=1,
            latent_dim=latent_dim,
            max_len=max_delay,
        )
        return dsa.DSAReplayBuffer(
            buffer_size=100,
            observation_space=obs_space,
            action_space=action_space,
            max_delay=max_delay,
            history_encoder=history_encoder,
            raw_obs_dim=obs_dim,
            history_mode=history_mode,
        )

    def _fill_buffer(
        self,
        buf: dsa.DSAReplayBuffer,
        obs_dim: int,
        latent_dim: int,
        num_steps: int,
        done_at: Optional[int] = None,
        interval_end_at: Optional[int] = None,
    ) -> None:
        for step_idx in range(num_steps):
            obs = np.concatenate(
                [
                    np.random.randn(obs_dim).astype(np.float32),
                    np.zeros(latent_dim, dtype=np.float32),
                ]
            )
            next_obs = np.concatenate(
                [
                    np.random.randn(obs_dim).astype(np.float32),
                    np.zeros(latent_dim, dtype=np.float32),
                ]
            )
            action = np.random.randn(1).astype(np.float32)
            done = step_idx == done_at if done_at is not None else False
            interval_end = (
                step_idx == interval_end_at if interval_end_at is not None else False
            )
            buf.add(
                obs,
                next_obs,
                action,
                np.array([1.0]),
                np.array([done]),
                [{"interval_end": interval_end}],
            )

    def test_next_history_zeroed_at_done(self):
        """next_history must be zeroed when the transition is terminal."""
        obs_dim, latent_dim = 3, 4
        buf = self._make_buffer_with_encoder(
            obs_dim=obs_dim, latent_dim=latent_dim, history_mode="sliding_window"
        )
        self._fill_buffer(buf, obs_dim, latent_dim, num_steps=10, done_at=5)

        batch_inds = np.array([5])
        np.random.seed(0)
        samples = buf._get_samples(batch_inds)
        z_next = samples.next_observations[0, obs_dim:]
        np.testing.assert_allclose(z_next.cpu().numpy(), 0.0, atol=1e-6)

    def test_next_history_zeroed_at_interval_end_per_interval(self):
        """In per_interval mode, next_history is zeroed at interval boundaries."""
        obs_dim, latent_dim = 3, 4
        buf = self._make_buffer_with_encoder(
            obs_dim=obs_dim, latent_dim=latent_dim, history_mode="per_interval"
        )
        self._fill_buffer(buf, obs_dim, latent_dim, num_steps=10, interval_end_at=4)

        batch_inds = np.array([4])
        np.random.seed(0)
        samples = buf._get_samples(batch_inds)
        z_next = samples.next_observations[0, obs_dim:]
        np.testing.assert_allclose(z_next.cpu().numpy(), 0.0, atol=1e-6)

    def test_next_history_not_zeroed_at_interval_end_sliding_window(self):
        """In sliding_window mode, interval boundaries do NOT reset next_history."""
        obs_dim, latent_dim = 3, 4
        buf = self._make_buffer_with_encoder(
            obs_dim=obs_dim, latent_dim=latent_dim, history_mode="sliding_window"
        )
        self._fill_buffer(buf, obs_dim, latent_dim, num_steps=10, interval_end_at=4)

        batch_inds = np.array([4])
        np.random.seed(0)
        samples = buf._get_samples(batch_inds)
        z_next = samples.next_observations[0, obs_dim:]
        assert np.abs(z_next.cpu().numpy()).sum() > 1e-6


# ---------------------------------------------------------------------------
# TestDSAReplayBufferRewardRelabeling
# ---------------------------------------------------------------------------


class TestDSAReplayBufferRewardRelabeling:
    def test_reward_relabeling_with_encoder(self):
        """Rewards are relabeled via the reward model when both encoder and model are set."""
        torch.manual_seed(42)
        obs_dim, action_dim, latent_dim = 3, 1, 4
        obs_space = gym.spaces.Box(
            -np.inf, np.inf, (obs_dim + latent_dim,), dtype=np.float32
        )
        action_space = gym.spaces.Box(-1, 1, (action_dim,), dtype=np.float32)

        history_encoder = dsa.HistoryEncoder(
            sa_dim=obs_dim + action_dim,
            obs_dim=obs_dim,
            action_dim=action_dim,
            d_model=8,
            nhead=2,
            num_encoder_layers=1,
            latent_dim=latent_dim,
            max_len=3,
        )

        class _ConstantRewardModel(base.RewardModel):
            def predict(
                self,
                observations: np.ndarray,
                actions: np.ndarray,
                terminals: np.ndarray,
            ) -> np.ndarray:
                return np.full(len(observations), 42.0, dtype=np.float32)

            def update(self, trajectories: Any) -> dict:
                return {}

        reward_model = _ConstantRewardModel()
        buf = dsa.DSAReplayBuffer(
            buffer_size=100,
            observation_space=obs_space,
            action_space=action_space,
            max_delay=3,
            history_encoder=history_encoder,
            reward_model=reward_model,
            raw_obs_dim=obs_dim,
        )

        for _ in range(20):
            obs = np.concatenate(
                [
                    np.random.randn(obs_dim).astype(np.float32),
                    np.zeros(latent_dim, dtype=np.float32),
                ]
            )
            next_obs = np.concatenate(
                [
                    np.random.randn(obs_dim).astype(np.float32),
                    np.zeros(latent_dim, dtype=np.float32),
                ]
            )
            action = np.random.randn(action_dim).astype(np.float32)
            buf.add(
                obs,
                next_obs,
                action,
                np.array([0.0]),
                np.array([False]),
                [{"interval_end": False}],
            )

        batch = buf.sample(4)
        np.testing.assert_allclose(batch.rewards.cpu().numpy(), 42.0, atol=1e-6)


# ---------------------------------------------------------------------------
# TestDSAEvalWrapperHistorySemantics
# ---------------------------------------------------------------------------


class TestDSAEvalWrapperHistorySemantics:
    def _make_eval_env(
        self,
        latent_dim: int = 4,
        max_delay: int = 5,
        history_mode: str = "per_interval",
    ) -> dsa.DSAEvalWrapper:
        env = gym.make("Pendulum-v1")
        env = rewdelay.DelayedRewardWrapper(env, rewdelay.FixedDelay(max_delay))
        env = rewdelay.ImputeMissingRewardWrapper(env, impute_value=0.0)
        env = hc.IntervalPositionWrapper(env, max_delay=max_delay)
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        env = dsa.LatentAugmentedObsWrapper(env, latent_dim=latent_dim)

        torch.manual_seed(42)
        sa_dim = obs_dim + action_dim
        history_encoder = dsa.HistoryEncoder(
            sa_dim=sa_dim,
            obs_dim=obs_dim,
            action_dim=action_dim,
            d_model=8,
            nhead=2,
            num_encoder_layers=1,
            latent_dim=latent_dim,
            max_len=max_delay,
        )
        return dsa.DSAEvalWrapper(
            env,
            history_encoder=history_encoder,
            max_delay=max_delay,
            latent_dim=latent_dim,
            obs_dim_before_augmentation=obs_dim,
            action_dim=action_dim,
            history_mode=history_mode,
        )

    def test_z_includes_prior_but_not_current(self):
        """z at step t includes (s_{t-1}, a_{t-1}) but not (s_t, a_t)."""
        env = self._make_eval_env(latent_dim=4, max_delay=10)
        obs0, _ = env.reset()
        z_at_reset = obs0[-4:].copy()

        obs1, _, _, _, _ = env.step(env.action_space.sample())
        z_step1 = obs1[-4:].copy()

        obs2, _, _, _, _ = env.step(env.action_space.sample())
        z_step2 = obs2[-4:].copy()

        # Step 1: z is computed before appending (s_0, a_0), so history
        # is empty — z should equal the reset z.
        np.testing.assert_allclose(z_step1, z_at_reset, atol=1e-6)

        # Step 2: z is computed after appending (s_0, a_0) but before
        # appending (s_1, a_1), so history has one entry — z should
        # differ from reset.
        assert not np.allclose(z_step2, z_at_reset, atol=1e-4)

    def test_per_interval_clears_at_interval_end(self):
        """In per_interval mode, z resets to zero-history encoding at interval boundaries."""
        max_delay = 3
        env = self._make_eval_env(latent_dim=4, max_delay=max_delay)
        obs_reset, _ = env.reset()
        z_reset = obs_reset[-4:].copy()

        for _ in range(max_delay + 2):
            obs, _, _, _, _ = env.step(env.action_space.sample())

        obs_after_boundary, _ = env.reset()
        z_after_boundary = obs_after_boundary[-4:].copy()
        np.testing.assert_allclose(z_after_boundary, z_reset, atol=1e-6)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_trajectory(
    obs: np.ndarray,
    actions: np.ndarray,
    env_rewards: np.ndarray,
    interval_ends: np.ndarray,
    terminal_at: int = -1,
) -> base.Trajectory:
    """Build a Trajectory for testing."""
    terminals = np.zeros(len(obs), dtype=bool)
    if terminal_at == -1:
        terminals[-1] = True
    else:
        terminals[terminal_at] = True
    infos = tuple({"interval_end": bool(ie)} for ie in interval_ends)
    return base.Trajectory(
        observations=obs.astype(np.float32),
        actions=actions.astype(np.float32),
        env_rewards=env_rewards.astype(np.float32),
        terminals=terminals.astype(np.float32),
        infos=infos,
        episode_return=float(env_rewards.sum()),
    )
