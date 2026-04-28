"""
Tests for drmdp.control.runner: RewardModelUpdateCallback and run().
"""

import dataclasses
import json
import os
import tempfile
import unittest.mock
from typing import Any, Dict, List, Mapping, Optional, Sequence

import gymnasium as gym
import numpy as np
import pytest

from drmdp.control import base, dgra, ircr, runner

# ---------------------------------------------------------------------------
# TestRewardModelUpdateCallback
# ---------------------------------------------------------------------------


class TestRewardModelUpdateCallback:
    def test_trajectory_built_correctly_across_episode(self):
        """Completed Trajectory has correct observations, actions, episode_return."""
        model = _TrackingRewardModel()
        callback = _build_callback(reward_model=model, update_every=1000)
        sac = _make_mock_sac(obs_dim=3, act_dim=1)

        rng = np.random.default_rng(0)
        obs_seq = [rng.uniform(size=3).astype(np.float32) for _ in range(4)]
        actions = [rng.uniform(size=1).astype(np.float32) for _ in range(4)]
        rewards = [1.0, 2.0, 3.0, 4.0]

        for step_idx in range(4):
            done = step_idx == 3
            _step_callback(
                callback,
                sac,
                obs_before=obs_seq[step_idx],
                action=actions[step_idx],
                reward=rewards[step_idx],
                done=done,
                num_timesteps=step_idx + 1,
            )

        assert len(callback._pending_trajectories) == 1
        traj = callback._pending_trajectories[0]
        np.testing.assert_allclose(traj.observations, np.stack(obs_seq), atol=1e-6)
        np.testing.assert_allclose(traj.actions, np.stack(actions), atol=1e-6)
        assert traj.episode_return == pytest.approx(sum(rewards))

    def test_update_called_at_configured_interval(self):
        """reward_model.update() is called when num_timesteps is a multiple of update_every."""
        model = _TrackingRewardModel()
        update_every = 5
        callback = _build_callback(reward_model=model, update_every=update_every)
        sac = _make_mock_sac()
        obs = np.zeros(3, dtype=np.float32)
        action = np.zeros(1, dtype=np.float32)

        for step_idx in range(1, 6):
            done = step_idx == 3
            _step_callback(callback, sac, obs, action, 1.0, done, step_idx)

        assert model.update_calls == 1

    def test_update_not_called_before_interval(self):
        """No update fires if num_timesteps has not reached the interval yet."""
        model = _TrackingRewardModel()
        callback = _build_callback(reward_model=model, update_every=100)
        sac = _make_mock_sac()
        obs = np.zeros(3, dtype=np.float32)
        action = np.zeros(1, dtype=np.float32)

        for step_idx in range(1, 5):
            done = step_idx == 2
            _step_callback(callback, sac, obs, action, 1.0, done, step_idx)

        assert model.update_calls == 0

    def test_clear_buffer_on_update(self):
        """replay_buffer.reset() is called after the reward model update when flag is set."""
        model = _TrackingRewardModel()
        callback = _build_callback(
            reward_model=model, update_every=3, clear_buffer=True
        )
        sac = _make_mock_sac()
        obs = np.zeros(3, dtype=np.float32)
        action = np.zeros(1, dtype=np.float32)

        for step_idx in range(1, 4):
            done = step_idx == 2
            _step_callback(callback, sac, obs, action, 1.0, done, step_idx)

        assert sac.replay_buffer.reset_count == 1

    def test_clear_buffer_not_called_when_flag_off(self):
        """replay_buffer.reset() is NOT called when clear_buffer_on_update=False."""
        model = _TrackingRewardModel()
        callback = _build_callback(
            reward_model=model, update_every=3, clear_buffer=False
        )
        sac = _make_mock_sac()
        obs = np.zeros(3, dtype=np.float32)
        action = np.zeros(1, dtype=np.float32)

        for step_idx in range(1, 4):
            done = step_idx == 2
            _step_callback(callback, sac, obs, action, 1.0, done, step_idx)

        assert sac.replay_buffer.reset_count == 0

    def test_logger_called_at_log_frequency(self):
        """ExperimentLogger.log() fires at every log_step_frequency env steps.

        4 episodes × 3 steps = 12 total steps; with log_step_frequency=6
        the threshold is crossed at step 6 (end of ep 2) and step 12 (end of ep 4).
        """
        exp_logger = _MockLogger()
        callback = _build_callback(log_step_freq=6, exp_logger=exp_logger)
        sac = _make_mock_sac()
        obs = np.zeros(3, dtype=np.float32)
        action = np.zeros(1, dtype=np.float32)

        timestep = 0
        for _episode_idx in range(4):
            for step_idx in range(3):
                timestep += 1
                done = step_idx == 2
                _step_callback(callback, sac, obs, action, 1.0, done, timestep)

        assert len(exp_logger.log_calls) == 2

    def test_training_end_flushes_pending_trajectories(self):
        """_on_training_end() passes remaining pending trajectories to the model."""
        model = _TrackingRewardModel()
        callback = _build_callback(reward_model=model, update_every=1000)
        sac = _make_mock_sac()
        obs = np.zeros(3, dtype=np.float32)
        action = np.zeros(1, dtype=np.float32)

        for step_idx in range(1, 4):
            done = step_idx == 2
            _step_callback(callback, sac, obs, action, 1.0, done, step_idx)

        assert model.update_calls == 0

        callback._on_training_end()

        assert model.update_calls == 1

    def test_elapsed_seconds_logged(self):
        """elapsed_seconds is present and non-negative in each logged info dict."""
        exp_logger = _MockLogger()
        callback = _build_callback(log_step_freq=3, exp_logger=exp_logger)
        callback._on_training_start()
        sac = _make_mock_sac()
        obs = np.zeros(3, dtype=np.float32)
        action = np.zeros(1, dtype=np.float32)

        for step_idx in range(1, 4):
            done = step_idx == 3
            _step_callback(callback, sac, obs, action, 1.0, done, step_idx)

        assert len(exp_logger.log_calls) == 1
        assert "elapsed_seconds" in exp_logger.log_calls[0]["info"]
        assert exp_logger.log_calls[0]["info"]["elapsed_seconds"] >= 0.0


# ---------------------------------------------------------------------------
# TestRun (integration)
# ---------------------------------------------------------------------------


class TestRun:
    @pytest.fixture()
    def output_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            yield tmp

    def test_run_completes_without_error(self, output_dir):
        """run() exits without raising for a short Pendulum-v1 experiment."""
        args = self._base_args(output_dir)
        runner.run(args)

    def test_output_files_created(self, output_dir):
        """After run(), experiment-logs.jsonl and experiment-params.json exist."""
        args = self._base_args(output_dir)
        runner.run(args)

        assert os.path.isfile(os.path.join(output_dir, "experiment-logs.jsonl"))
        assert os.path.isfile(os.path.join(output_dir, "experiment-params.json"))

    def test_sac_model_saved(self, output_dir):
        """After run(), the SAC model checkpoint file exists."""
        args = self._base_args(output_dir)
        runner.run(args)

        assert os.path.isfile(os.path.join(output_dir, "sac_model.zip"))

    def test_clear_buffer_on_update_does_not_crash(self, output_dir):
        """run() with clear_buffer_on_update=True completes without error."""
        args = dataclasses.replace(
            self._base_args(output_dir), clear_buffer_on_update=True
        )
        runner.run(args)

    def test_hc_run_completes_without_error(self, output_dir):
        """run() with agent_type='hc' exits without raising."""
        args = dataclasses.replace(
            self._base_args(output_dir), agent_type="hc", num_steps=200
        )
        runner.run(args)

    def test_hc_model_saved(self, output_dir):
        """After an HC run, hc_model.zip exists in output_dir."""
        args = dataclasses.replace(
            self._base_args(output_dir), agent_type="hc", num_steps=200
        )
        runner.run(args)
        assert os.path.isfile(os.path.join(output_dir, "hc_model.zip"))

    def test_run_delayed_baseline(self, output_dir):
        """run() with reward_model_type='none' completes and produces output files."""
        args = dataclasses.replace(
            self._base_args(output_dir),
            reward_model_type="none",
            reward_model_kwargs={},
        )
        runner.run(args)
        assert os.path.isfile(os.path.join(output_dir, "experiment-logs.jsonl"))
        assert os.path.isfile(os.path.join(output_dir, "sac_model.zip"))

    def test_exp_name_and_run_id_in_params_file(self, output_dir):
        """experiment-params.json contains exp_name and run_id fields."""
        args = dataclasses.replace(
            self._base_args(output_dir), exp_name="exp-007", run_id=3
        )
        runner.run(args)
        params_path = os.path.join(output_dir, "experiment-params.json")
        with open(params_path, encoding="utf-8") as params_file:
            params = json.load(params_file)
        assert params["exp_name"] == "exp-007"
        assert params["run_id"] == 3

    def _base_args(self, output_dir: str) -> runner.TrainingArgs:
        return runner.TrainingArgs(
            env="Pendulum-v1",
            delay=1,
            env_kwargs={"max_episode_steps": 50},
            reward_model_type="ircr",
            update_every_n_steps=50,
            clear_buffer_on_update=False,
            reward_model_kwargs={"max_buffer_size": 20, "k_neighbors": 3},
            num_steps=200,
            sac_learning_rate=3e-4,
            sac_buffer_size=1000,
            sac_batch_size=32,
            sac_gradient_steps=1,
            log_step_frequency=50,
            output_dir=output_dir,
            exp_name="exp-000",
            run_id=0,
            seed=42,
        )


# ---------------------------------------------------------------------------
# TestRewardObsLoggingCallback
# ---------------------------------------------------------------------------


class TestRewardObsLoggingCallback:
    def test_on_step_returns_true(self):
        callback = self._build_ro_callback()
        result = self._step_ro(callback, done=False, timestep=1)
        assert result is True

    def test_episode_count_increments_on_done(self):
        callback = self._build_ro_callback()
        self._step_ro(callback, done=True, timestep=1)
        assert callback._episode_count == 1

    def test_episode_steps_reset_after_done(self):
        callback = self._build_ro_callback()
        self._step_ro(callback, done=False, timestep=1)
        self._step_ro(callback, done=True, timestep=2)
        assert callback._episode_steps == 0

    def test_episode_rewards_reset_after_done(self):
        callback = self._build_ro_callback()
        self._step_ro(callback, done=False, timestep=1, reward=2.0)
        self._step_ro(callback, done=True, timestep=2, reward=3.0)
        assert callback._episode_rewards == []

    def test_logger_called_at_log_frequency(self):
        """Logs only when enough steps have elapsed since the last log.

        4 steps total, done at steps 2 and 4; with log_step_frequency=4
        only step 4 crosses the threshold.
        """
        log_mock = _MockLogger()
        callback = runner._RewardObsLoggingCallback(
            log_step_frequency=4, exp_logger=log_mock
        )
        for timestep in range(1, 5):
            done = timestep % 2 == 0
            self._step_ro(
                callback,
                done=done,
                timestep=timestep,
                return_val=1.0,
                reward=1.0,
            )
        assert len(log_mock.log_calls) == 1

    def test_delayed_returns_logged_as_sum_of_rewards(self):
        """delayed_returns in logged info equals sum of per-step rewards for the episode."""
        log_mock = _MockLogger()
        callback = runner._RewardObsLoggingCallback(
            log_step_frequency=3, exp_logger=log_mock
        )
        rewards = [1.5, 2.5, 0.5]
        for idx, reward in enumerate(rewards):
            done = idx == len(rewards) - 1
            callback.num_timesteps = idx + 1
            callback.locals = {
                "dones": np.array([done]),
                "infos": [{"true_episode_return": 10.0}],
                "rewards": np.array([reward]),
            }
            callback._on_step()

        assert len(log_mock.log_calls) == 1
        logged_info = log_mock.log_calls[0]["info"]
        assert "delayed_returns" in logged_info
        assert logged_info["delayed_returns"] == 4.5

    def test_delayed_returns_reset_between_episodes(self):
        """Each episode's delayed_returns reflects only that episode's rewards."""
        log_mock = _MockLogger()
        callback = runner._RewardObsLoggingCallback(
            log_step_frequency=2, exp_logger=log_mock
        )
        for idx, reward in enumerate([1.0, 2.0]):
            callback.num_timesteps = idx + 1
            callback.locals = {
                "dones": np.array([idx == 1]),
                "infos": [{"true_episode_return": 0.0}],
                "rewards": np.array([reward]),
            }
            callback._on_step()

        for idx, reward in enumerate([4.0, 6.0]):
            callback.num_timesteps = idx + 3
            callback.locals = {
                "dones": np.array([idx == 1]),
                "infos": [{"true_episode_return": 0.0}],
                "rewards": np.array([reward]),
            }
            callback._on_step()

        assert log_mock.log_calls[0]["info"]["delayed_returns"] == 3.0
        assert log_mock.log_calls[1]["info"]["delayed_returns"] == 10.0

    def test_elapsed_seconds_logged(self):
        """elapsed_seconds is present and non-negative in each logged info dict."""
        log_mock = _MockLogger()
        callback = runner._RewardObsLoggingCallback(
            log_step_frequency=3, exp_logger=log_mock
        )
        callback._on_training_start()
        rewards = [1.0, 2.0, 0.5]
        for idx, reward in enumerate(rewards):
            done = idx == len(rewards) - 1
            callback.num_timesteps = idx + 1
            callback.locals = {
                "dones": np.array([done]),
                "infos": [{"true_episode_return": 5.0}],
                "rewards": np.array([reward]),
            }
            callback._on_step()

        assert len(log_mock.log_calls) == 1
        assert "elapsed_seconds" in log_mock.log_calls[0]["info"]
        assert log_mock.log_calls[0]["info"]["elapsed_seconds"] >= 0.0

    def _build_ro_callback(self, log_freq: int = 1) -> runner._RewardObsLoggingCallback:
        exp_logger = _MockLogger()
        return runner._RewardObsLoggingCallback(
            log_step_frequency=log_freq,
            exp_logger=exp_logger,  # type: ignore[arg-type]
        )

    def _step_ro(
        self,
        callback: runner._RewardObsLoggingCallback,
        done: bool,
        timestep: int,
        return_val: float = 1.0,
        reward: float = 0.0,
    ) -> bool:
        callback.num_timesteps = timestep
        callback.locals = {
            "dones": np.array([done]),
            "infos": [{"true_episode_return": return_val}],
            "rewards": np.array([reward]),
        }
        return callback._on_step()


class TestMakeRewardModel:
    @pytest.fixture()
    def pendulum_env(self):
        env = gym.make("Pendulum-v1")
        yield env
        env.close()

    def test_ircr_type_returns_ircr_instance(self, tmp_path, pendulum_env):
        args = runner.TrainingArgs(
            env="Pendulum-v1",
            delay=1,
            env_kwargs={"max_episode_steps": 50},
            reward_model_type="ircr",
            reward_model_kwargs={"max_buffer_size": 10, "k_neighbors": 1},
            update_every_n_steps=100,
            clear_buffer_on_update=False,
            num_steps=100,
            sac_learning_rate=3e-4,
            sac_buffer_size=100,
            sac_batch_size=32,
            sac_gradient_steps=1,
            log_step_frequency=50,
            output_dir=str(tmp_path),
            exp_name="exp-000",
            run_id=0,
            seed=None,
        )
        model = runner._make_reward_model(args, pendulum_env)
        assert isinstance(model, ircr.IRCRRewardModel)

    def test_none_type_returns_none(self, tmp_path, pendulum_env):
        args = runner.TrainingArgs(
            env="Pendulum-v1",
            delay=1,
            env_kwargs={"max_episode_steps": 50},
            reward_model_type="none",
            reward_model_kwargs={},
            update_every_n_steps=100,
            clear_buffer_on_update=False,
            num_steps=100,
            sac_learning_rate=3e-4,
            sac_buffer_size=100,
            sac_batch_size=32,
            sac_gradient_steps=1,
            log_step_frequency=50,
            output_dir=str(tmp_path),
            exp_name="exp-000",
            run_id=0,
            seed=None,
        )
        assert runner._make_reward_model(args, pendulum_env) is None

    def test_unknown_type_raises_value_error(self, tmp_path, pendulum_env):
        args = runner.TrainingArgs(
            env="Pendulum-v1",
            delay=1,
            env_kwargs={"max_episode_steps": 50},
            reward_model_type="unknown_model",
            reward_model_kwargs={},
            update_every_n_steps=100,
            clear_buffer_on_update=False,
            num_steps=100,
            sac_learning_rate=3e-4,
            sac_buffer_size=100,
            sac_batch_size=32,
            sac_gradient_steps=1,
            log_step_frequency=50,
            output_dir=str(tmp_path),
            exp_name="exp-000",
            run_id=0,
            seed=None,
        )
        with pytest.raises(ValueError):
            runner._make_reward_model(args, pendulum_env)

    def test_dgra_type_returns_dgra_instance(self, tmp_path, pendulum_env):
        args = runner.TrainingArgs(
            env="Pendulum-v1",
            delay=1,
            env_kwargs={"max_episode_steps": 50},
            reward_model_type="dgra",
            reward_model_kwargs={},
            update_every_n_steps=100,
            clear_buffer_on_update=False,
            num_steps=100,
            sac_learning_rate=3e-4,
            sac_buffer_size=100,
            sac_batch_size=32,
            sac_gradient_steps=1,
            log_step_frequency=50,
            output_dir=str(tmp_path),
            exp_name="exp-000",
            run_id=0,
            seed=None,
        )
        model = runner._make_reward_model(args, pendulum_env)
        assert isinstance(model, dgra.DGRARewardModel)

    def test_dgra_predicts_with_pendulum_dimensions(self, tmp_path, pendulum_env):
        """DGRARewardModel constructed from Pendulum env accepts its obs/action dims."""
        args = runner.TrainingArgs(
            env="Pendulum-v1",
            delay=1,
            env_kwargs={"max_episode_steps": 50},
            reward_model_type="dgra",
            reward_model_kwargs={},
            update_every_n_steps=100,
            clear_buffer_on_update=False,
            num_steps=100,
            sac_learning_rate=3e-4,
            sac_buffer_size=100,
            sac_batch_size=32,
            sac_gradient_steps=1,
            log_step_frequency=50,
            output_dir=str(tmp_path),
            exp_name="exp-000",
            run_id=0,
            seed=None,
        )
        model = runner._make_reward_model(args, pendulum_env)
        pendulum_obs_dim = int(np.prod(pendulum_env.observation_space.shape))
        pendulum_action_dim = int(np.prod(pendulum_env.action_space.shape))
        obs = np.zeros((4, pendulum_obs_dim), dtype=np.float32)
        actions = np.zeros((4, pendulum_action_dim), dtype=np.float32)
        terminals = np.zeros(4, dtype=bool)

        result = model.predict(obs, actions, terminals)

        assert result.shape == (4,)


class TestParseArgs:
    def test_defaults_are_valid_mapping(self, tmp_path):
        with unittest.mock.patch("sys.argv", ["prog", "--output-dir", str(tmp_path)]):
            args = runner.parse_single_cli()
        assert isinstance(args, dict)
        assert args["env"] == "MountainCarContinuous-v0"
        assert args["delay"] == 3

    def test_custom_env_and_delay(self, tmp_path):
        with unittest.mock.patch(
            "sys.argv",
            [
                "prog",
                "--env",
                "Pendulum-v1",
                "--delay",
                "5",
                "--output-dir",
                str(tmp_path),
            ],
        ):
            args = runner.parse_single_cli()
        assert args["env"] == "Pendulum-v1"
        assert args["delay"] == 5

    def test_reward_model_kwarg_parsed(self, tmp_path):
        with unittest.mock.patch(
            "sys.argv",
            [
                "prog",
                "--reward-model-kwarg",
                "max_buffer_size=50",
                "--output-dir",
                str(tmp_path),
            ],
        ):
            args = runner.parse_single_cli()
        assert args["reward_model_kwargs"]["max_buffer_size"] == 50

    def test_agent_type_hc(self, tmp_path):
        with unittest.mock.patch(
            "sys.argv",
            ["prog", "--agent-type", "hc", "--output-dir", str(tmp_path)],
        ):
            args = runner.parse_single_cli()
        assert args["agent_type"] == "hc"

    def test_clear_buffer_flag(self, tmp_path):
        with unittest.mock.patch(
            "sys.argv",
            ["prog", "--clear-buffer-on-update", "--output-dir", str(tmp_path)],
        ):
            args = runner.parse_single_cli()
        assert args["clear_buffer_on_update"] is True

    def test_none_reward_model_type_accepted(self, tmp_path):
        with unittest.mock.patch(
            "sys.argv",
            ["prog", "--reward-model-type", "none", "--output-dir", str(tmp_path)],
        ):
            args = runner.parse_single_cli()
        assert args["reward_model_type"] == "none"


class TestGenerateConfigs:
    def test_single_run_produces_one_config(self, tmp_path):
        """num_runs=1 produces exactly one TrainingArgs."""
        single_cli = self._make_single_cli(tmp_path)
        configs = runner._generate_configs(single_cli, exec_kwargs={"num_runs": 1})
        assert len(configs) == 1
        assert isinstance(configs[0], runner.TrainingArgs)

    def test_multi_run_produces_correct_count(self, tmp_path):
        """num_runs=3 produces three TrainingArgs instances."""
        single_cli = self._make_single_cli(tmp_path)
        configs = runner._generate_configs(single_cli, exec_kwargs={"num_runs": 3})
        assert len(configs) == 3

    def test_multi_run_produces_unique_output_dirs(self, tmp_path):
        """Each run gets a distinct output_dir containing its run index."""
        single_cli = self._make_single_cli(tmp_path)
        configs = runner._generate_configs(single_cli, exec_kwargs={"num_runs": 3})
        dirs = [cfg.output_dir for cfg in configs]
        assert len(set(dirs)) == 3
        assert "run-000" in dirs[0]
        assert "run-001" in dirs[1]
        assert "run-002" in dirs[2]

    def test_multi_run_produces_unique_seeds(self, tmp_path):
        """Each run receives a distinct seed when num_runs > 1."""
        single_cli = self._make_single_cli(tmp_path)
        configs = runner._generate_configs(single_cli, exec_kwargs={"num_runs": 3})
        seeds = [cfg.seed for cfg in configs]
        assert len(set(seeds)) == 3

    def test_cli_args_override_defaults(self, tmp_path):
        """Custom env, delay, and reward_model_type are preserved in produced configs."""
        single_cli = {
            **self._make_single_cli(tmp_path),
            "env": "Pendulum-v1",
            "delay": 7,
            "reward_model_type": "none",
        }
        configs = runner._generate_configs(single_cli, exec_kwargs={"num_runs": 1})
        assert configs[0].env == "Pendulum-v1"
        assert configs[0].delay == 7
        assert configs[0].reward_model_type == "none"

    def test_exp_name_and_run_id_set_in_single_cli_mode(self, tmp_path):
        """Single-CLI mode always uses exp-000; run_id increments per run."""
        single_cli = self._make_single_cli(tmp_path)
        configs = runner._generate_configs(single_cli, exec_kwargs={"num_runs": 3})
        assert all(cfg.exp_name == "exp-000" for cfg in configs)
        assert [cfg.run_id for cfg in configs] == [0, 1, 2]

    def _make_single_cli(self, tmp_path: Any) -> Mapping[str, Any]:
        """Minimal single_cli dict equivalent to a parsed CLI invocation.

        Mirrors what parse_single_cli() returns: max_episode_steps has been
        folded into env_kwargs, and output_dir / seed are present as top-level keys.
        """
        return {
            "env": "MountainCarContinuous-v0",
            "delay": 3,
            "env_kwargs": {"max_episode_steps": 2500},
            "reward_model_type": "ircr",
            "update_every_n_steps": 1000,
            "clear_buffer_on_update": False,
            "reward_model_kwargs": {},
            "agent_type": "sac",
            "agent_kwargs": {},
            "seed": None,
            "num_steps": 50000,
            "sac_learning_rate": 3e-4,
            "sac_buffer_size": 100000,
            "sac_batch_size": 256,
            "sac_gradient_steps": -1,
            "log_step_frequency": 10000,
            "output_dir": str(tmp_path),
        }


class TestLoadConfigs:
    def test_single_experiment_single_run(self, tmp_path):
        """One experiment with num_runs=1 produces exactly one TrainingArgs."""
        config = self._single_env(
            extra_env_fields={"delay": 2},
            output_dir=str(tmp_path),
            num_runs=1,
        )
        configs = runner._load_configs(self._write_config(tmp_path, config))
        assert len(configs) == 1
        assert configs[0].env == "Pendulum-v1"
        assert configs[0].delay == 2

    def test_num_runs_expands_entries(self, tmp_path):
        """num_runs=3 expands one experiment into three TrainingArgs."""
        config = self._single_env(output_dir=str(tmp_path), num_runs=3)
        configs = runner._load_configs(self._write_config(tmp_path, config))
        assert len(configs) == 3

    def test_multiple_experiments_expanded(self, tmp_path):
        """Two experiments in one env each with num_runs=2 produces four configs total."""
        config = {
            "output_dir": str(tmp_path),
            "num_runs": 2,
            "environments": [
                {
                    "env": "Pendulum-v1",
                    "experiments": [{}, {}],
                }
            ],
        }
        configs = runner._load_configs(self._write_config(tmp_path, config))
        assert len(configs) == 4

    def test_multiple_environments_expanded(self, tmp_path):
        """Two environments each with one experiment and num_runs=2 gives four configs."""
        config = {
            "output_dir": str(tmp_path),
            "num_runs": 2,
            "environments": [
                {"env": "Pendulum-v1", "experiments": [{}]},
                {"env": "MountainCarContinuous-v0", "experiments": [{}]},
            ],
        }
        configs = runner._load_configs(self._write_config(tmp_path, config))
        assert len(configs) == 4

    def test_explicit_seed_passed_through_for_single_run(self, tmp_path):
        """An explicit seed with num_runs=1 is kept verbatim, not run through Seeder."""
        config = self._single_env(
            extra_exp_fields={"seed": 99},
            output_dir=str(tmp_path),
            num_runs=1,
        )
        configs = runner._load_configs(self._write_config(tmp_path, config))
        assert configs[0].seed == 99

    def test_seeds_differ_across_runs(self, tmp_path):
        """Multiple runs for the same experiment receive different seeds."""
        config = self._single_env(
            extra_exp_fields={"seed": 7},
            output_dir=str(tmp_path),
            num_runs=3,
        )
        configs = runner._load_configs(self._write_config(tmp_path, config))
        seeds = [cfg.seed for cfg in configs]
        assert len(set(seeds)) == 3, "All seeds should be distinct across runs"

    def test_seeds_unique_across_environments(self, tmp_path):
        """The global exp offset ensures seeds do not repeat across different environments."""
        config = {
            "output_dir": str(tmp_path),
            "num_runs": 2,
            "environments": [
                {"env": "Pendulum-v1", "experiments": [{}]},
                {"env": "Pendulum-v1", "experiments": [{}]},
            ],
        }
        configs = runner._load_configs(self._write_config(tmp_path, config))
        seeds = [cfg.seed for cfg in configs]
        assert len(set(seeds)) == len(seeds), "Seeds must be unique across environments"

    def test_output_dirs_include_env_label(self, tmp_path):
        """Output dirs embed the gym env ID and then exp/run indices."""
        config = self._single_env(output_dir=str(tmp_path), num_runs=2)
        configs = runner._load_configs(self._write_config(tmp_path, config))
        assert configs[0].output_dir == str(
            tmp_path / "Pendulum-v1" / "exp-000" / "run-000"
        )
        assert configs[1].output_dir == str(
            tmp_path / "Pendulum-v1" / "exp-000" / "run-001"
        )

    def test_output_dirs_use_env_id_directly(self, tmp_path):
        """The gym env ID is used as-is as the directory label."""
        config = {
            "output_dir": str(tmp_path),
            "environments": [{"env": "Pendulum-v1", "experiments": [{}]}],
        }
        configs = runner._load_configs(self._write_config(tmp_path, config))
        assert "Pendulum-v1" in configs[0].output_dir

    def test_missing_env_field_raises(self, tmp_path):
        """A config entry without 'env' raises KeyError at load time."""
        config = {
            "output_dir": str(tmp_path),
            "environments": [{"experiments": [{}]}],
        }
        with pytest.raises(KeyError):
            runner._load_configs(self._write_config(tmp_path, config))

    def test_experiment_output_dir_overrides_top_level(self, tmp_path):
        """An experiment-level output_dir is used verbatim for all its runs."""
        exp_dir = str(tmp_path / "custom")
        config = self._single_env(
            extra_exp_fields={"output_dir": exp_dir},
            output_dir=str(tmp_path),
            num_runs=2,
        )
        configs = runner._load_configs(self._write_config(tmp_path, config))
        assert all(cfg.output_dir == exp_dir for cfg in configs)

    def test_top_level_shared_fields_applied_to_all(self, tmp_path):
        """Top-level fields serve as defaults for all environments and experiments."""
        config = {
            "output_dir": str(tmp_path),
            "delay": 10,
            "environments": [
                {"env": "Pendulum-v1", "experiments": [{}, {}]},
            ],
        }
        configs = runner._load_configs(self._write_config(tmp_path, config))
        assert all(cfg.delay == 10 for cfg in configs)

    def test_env_field_overrides_top_level_default(self, tmp_path):
        """An environment entry can override a top-level shared field."""
        config = {
            "output_dir": str(tmp_path),
            "delay": 5,
            "environments": [
                {"env": "Pendulum-v1", "delay": 2, "experiments": [{}]},
            ],
        }
        configs = runner._load_configs(self._write_config(tmp_path, config))
        assert configs[0].delay == 2

    def test_experiment_field_overrides_env_default(self, tmp_path):
        """An experiment entry can override an environment-level field."""
        config = {
            "output_dir": str(tmp_path),
            "environments": [
                {
                    "env": "Pendulum-v1",
                    "delay": 5,
                    "experiments": [{"delay": 1}],
                }
            ],
        }
        configs = runner._load_configs(self._write_config(tmp_path, config))
        assert configs[0].delay == 1

    def test_returns_training_args_instances(self, tmp_path):
        """All returned objects are TrainingArgs dataclass instances."""
        config = self._single_env(output_dir=str(tmp_path))
        configs = runner._load_configs(self._write_config(tmp_path, config))
        assert all(isinstance(cfg, runner.TrainingArgs) for cfg in configs)

    def test_exp_name_and_run_id_set_correctly(self, tmp_path):
        """exp_name is shared across runs; run_id is unique within an experiment."""
        config = self._single_env(output_dir=str(tmp_path), num_runs=3)
        configs = runner._load_configs(self._write_config(tmp_path, config))
        assert all(cfg.exp_name == "exp-000" for cfg in configs)
        assert [cfg.run_id for cfg in configs] == [0, 1, 2]

    def _write_config(self, tmp_path, data: Mapping[str, Any]) -> str:
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(data))
        return str(config_file)

    def _single_env(
        self,
        extra_env_fields: Optional[Mapping[str, Any]] = None,
        extra_exp_fields: Optional[Mapping[str, Any]] = None,
        **top: Any,
    ) -> Mapping[str, Any]:
        """Helper to build a minimal multi-env config with one env and one experiment."""
        env_entry: Dict[str, Any] = {"env": "Pendulum-v1"}
        if extra_env_fields:
            env_entry.update(extra_env_fields)
        exp_entry: Dict[str, Any] = {}
        if extra_exp_fields:
            exp_entry.update(extra_exp_fields)
        env_entry["experiments"] = [exp_entry]
        return {**top, "environments": [env_entry]}


class TestLoadConfigsPrecedence:
    """Argument precedence: code defaults < batch file < CLI args."""

    def test_cli_num_runs_overrides_file_num_runs(self, tmp_path):
        """CLI --num-runs overrides the top-level num_runs in the batch file."""
        config = self._single_env(output_dir=str(tmp_path), num_runs=3)
        configs = runner._load_configs(
            self._write_config(tmp_path, config),
            exec_kwargs={"num_runs": 5},
        )
        assert len(configs) == 5

    def test_cli_num_runs_overrides_env_level_num_runs(self, tmp_path):
        """CLI --num-runs overrides num_runs set at the environment level."""
        config = {
            "output_dir": str(tmp_path),
            "environments": [
                {"env": "Pendulum-v1", "num_runs": 2, "experiments": [{}]},
            ],
        }
        configs = runner._load_configs(
            self._write_config(tmp_path, config),
            exec_kwargs={"num_runs": 4},
        )
        assert len(configs) == 4

    def test_file_num_steps_not_overridden_by_absent_cli_arg(self, tmp_path):
        """File num_steps is preserved when the user did not pass --num-steps (None)."""
        config = {
            "output_dir": str(tmp_path),
            "num_steps": 99999,
            "environments": [{"env": "Pendulum-v1", "experiments": [{}]}],
        }
        configs = runner._load_configs(
            self._write_config(tmp_path, config),
            common_kwargs={"num_steps": None},
        )
        assert configs[0].num_steps == 99999

    def test_cli_num_steps_overrides_file(self, tmp_path):
        """CLI --num-steps overrides num_steps set in the batch file."""
        config = {
            "output_dir": str(tmp_path),
            "num_steps": 99999,
            "environments": [{"env": "Pendulum-v1", "experiments": [{}]}],
        }
        configs = runner._load_configs(
            self._write_config(tmp_path, config),
            common_kwargs={"num_steps": 1000},
        )
        assert configs[0].num_steps == 1000

    def test_cli_output_dir_overrides_file(self, tmp_path):
        """CLI --output-dir overrides the output_dir in the batch file."""
        cli_dir = str(tmp_path / "cli")
        config = {
            "output_dir": str(tmp_path / "file"),
            "environments": [{"env": "Pendulum-v1", "experiments": [{}]}],
        }
        configs = runner._load_configs(
            self._write_config(tmp_path, config),
            common_kwargs={"output_dir": cli_dir},
        )
        assert configs[0].output_dir.startswith(cli_dir)

    def _write_config(self, tmp_path: Any, data: Mapping[str, Any]) -> str:
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(data))
        return str(config_file)

    def _single_env(
        self,
        extra_env_fields: Optional[Mapping[str, Any]] = None,
        extra_exp_fields: Optional[Mapping[str, Any]] = None,
        **top: Any,
    ) -> Mapping[str, Any]:
        env_entry: Dict[str, Any] = {"env": "Pendulum-v1"}
        if extra_env_fields:
            env_entry.update(extra_env_fields)
        exp_entry: Dict[str, Any] = {}
        if extra_exp_fields:
            exp_entry.update(extra_exp_fields)
        env_entry["experiments"] = [exp_entry]
        return {**top, "environments": [env_entry]}


class TestRunBatch:
    def test_debug_mode_calls_run_for_each_config(self, tmp_path):
        """In debug mode, run() is called once for each config, sequentially."""
        configs = [
            runner.TrainingArgs(
                env="Pendulum-v1",
                delay=1,
                env_kwargs={"max_episode_steps": 50},
                reward_model_type="ircr",
                update_every_n_steps=50,
                clear_buffer_on_update=False,
                reward_model_kwargs={},
                num_steps=100,
                sac_learning_rate=3e-4,
                sac_buffer_size=1000,
                sac_batch_size=32,
                sac_gradient_steps=1,
                log_step_frequency=50,
                output_dir=str(tmp_path),
                exp_name="exp-000",
                run_id=idx,
                seed=idx,
            )
            for idx in range(3)
        ]
        with unittest.mock.patch("drmdp.control.runner.run") as mock_run:
            runner.run_batch(configs, mode="debug")
        assert mock_run.call_count == 3

    def test_local_mode_submits_one_future_per_config(self, tmp_path):
        """In local mode, ProcessPoolExecutor.submit() is called once per config."""
        configs = [
            runner.TrainingArgs(
                env="Pendulum-v1",
                delay=1,
                env_kwargs={"max_episode_steps": 50},
                reward_model_type="ircr",
                update_every_n_steps=50,
                clear_buffer_on_update=False,
                reward_model_kwargs={},
                num_steps=100,
                sac_learning_rate=3e-4,
                sac_buffer_size=1000,
                sac_batch_size=32,
                sac_gradient_steps=1,
                log_step_frequency=50,
                output_dir=str(tmp_path),
                exp_name="exp-000",
                run_id=idx,
                seed=idx,
            )
            for idx in range(2)
        ]
        mock_future = unittest.mock.MagicMock()
        mock_future.result.return_value = None
        mock_executor = unittest.mock.MagicMock()
        mock_executor.__enter__ = unittest.mock.MagicMock(return_value=mock_executor)
        mock_executor.__exit__ = unittest.mock.MagicMock(return_value=False)
        mock_executor.submit.return_value = mock_future

        with (
            unittest.mock.patch(
                "drmdp.control.runner.concurrent.futures.ProcessPoolExecutor",
                return_value=mock_executor,
            ),
            unittest.mock.patch(
                "drmdp.control.runner.concurrent.futures.as_completed",
                return_value=iter([mock_future, mock_future]),
            ),
        ):
            runner.run_batch(configs, mode="local", max_workers=2)

        assert mock_executor.submit.call_count == 2

    def test_unknown_mode_falls_through_to_sequential(self, tmp_path):
        """An unrecognised mode string falls through to the sequential else branch."""
        config = runner.TrainingArgs(
            env="Pendulum-v1",
            delay=1,
            env_kwargs={"max_episode_steps": 50},
            reward_model_type="ircr",
            update_every_n_steps=50,
            clear_buffer_on_update=False,
            reward_model_kwargs={},
            num_steps=100,
            sac_learning_rate=3e-4,
            sac_buffer_size=1000,
            sac_batch_size=32,
            sac_gradient_steps=1,
            log_step_frequency=50,
            output_dir=str(tmp_path),
            exp_name="exp-000",
            run_id=0,
        )
        with unittest.mock.patch("drmdp.control.runner.run") as mock_run:
            runner.run_batch([config], mode="unknown_mode")
        assert mock_run.call_count == 1


class TestParseRewardModelKwargs:
    def test_empty_list_returns_empty_mapping(self):
        """An empty input list produces an empty mapping."""
        result = runner.parse_reward_model_kwargs([])
        assert result == {}

    def test_int_value_parsed(self):
        """Integer strings are coerced to int."""
        result = runner.parse_reward_model_kwargs(["max_buffer_size=200"])
        assert result["max_buffer_size"] == 200
        assert isinstance(result["max_buffer_size"], int)

    def test_float_value_parsed(self):
        """Float strings are coerced to float."""
        result = runner.parse_reward_model_kwargs(["alpha=0.5"])
        assert result["alpha"] == pytest.approx(0.5)
        assert isinstance(result["alpha"], float)

    def test_bool_value_parsed(self):
        """Boolean literals True/False are coerced to bool."""
        result = runner.parse_reward_model_kwargs(["flag=True"])
        assert result["flag"] is True

    def test_string_fallback_for_non_literal(self):
        """Values that are not valid Python literals are kept as plain strings."""
        result = runner.parse_reward_model_kwargs(["mode=cosine"])
        assert result["mode"] == "cosine"
        assert isinstance(result["mode"], str)

    def test_multiple_pairs(self):
        """Multiple key=value pairs are all parsed into the same mapping."""
        result = runner.parse_reward_model_kwargs(
            ["max_buffer_size=200", "k_neighbors=5"]
        )
        assert result == {"max_buffer_size": 200, "k_neighbors": 5}


# ---------------------------------------------------------------------------
# Stubs and helpers
# ---------------------------------------------------------------------------


class _TrackingRewardModel(base.RewardModel):
    """Reward model that tracks update calls and returns a fixed constant."""

    def __init__(self, constant: float = 0.0):
        self._constant = constant
        self.update_calls: int = 0
        self.trajectories_received: List[base.Trajectory] = []

    def predict(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        terminals: np.ndarray,
    ) -> np.ndarray:
        return np.full(len(observations), self._constant, dtype=np.float32)

    def update(self, trajectories: Sequence[base.Trajectory]) -> Mapping[str, float]:
        self.update_calls += 1
        self.trajectories_received.extend(trajectories)
        return {"buffer_size": float(len(self.trajectories_received))}


class _MockReplayBuffer:
    """Minimal replay buffer stub that records reset() calls."""

    def __init__(self):
        self.reset_count = 0

    def reset(self) -> None:
        self.reset_count += 1


class _MockLogger:
    """Minimal ExperimentLogger stub."""

    def __init__(self):
        self.log_calls: List[Mapping[str, Any]] = []

    def log(
        self,
        episode: int,
        steps: int,
        global_steps: int,
        returns: float,
        info: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.log_calls.append(
            {
                "episode": episode,
                "steps": steps,
                "global_steps": global_steps,
                "returns": returns,
                "info": info,
            }
        )


def _build_callback(
    reward_model: Optional[base.RewardModel] = None,
    update_every: int = 100,
    clear_buffer: bool = False,
    log_step_freq: int = 6,
    exp_logger: Any = None,
) -> runner.RewardModelUpdateCallback:
    """Construct a callback with sensible defaults for unit testing."""
    if reward_model is None:
        reward_model = _TrackingRewardModel()
    if exp_logger is None:
        exp_logger = _MockLogger()
    return runner.RewardModelUpdateCallback(
        reward_model=reward_model,
        update_every_n_steps=update_every,
        clear_buffer_on_update=clear_buffer,
        log_step_frequency=log_step_freq,
        exp_logger=exp_logger,
    )


def _make_mock_sac(obs_dim: int = 3, act_dim: int = 1) -> unittest.mock.MagicMock:
    """Return a minimal SAC mock with _last_obs and replay_buffer."""
    sac = unittest.mock.MagicMock()
    sac._last_obs = np.zeros((1, obs_dim), dtype=np.float32)
    sac.replay_buffer = _MockReplayBuffer()
    return sac


def _step_callback(
    callback: runner.RewardModelUpdateCallback,
    sac_model: Any,
    obs_before: np.ndarray,
    action: np.ndarray,
    reward: float,
    done: bool,
    num_timesteps: int,
) -> bool:
    """Simulate one SB3 callback step by setting required attributes/locals."""
    sac_model._last_obs = obs_before[np.newaxis]  # (1, obs_dim)
    callback.model = sac_model
    callback.locals = {
        "actions": action[np.newaxis],  # (1, act_dim)
        "rewards": np.array([reward]),
        "dones": np.array([done]),
        "infos": [{}],
    }
    callback.num_timesteps = num_timesteps
    return callback._on_step()
