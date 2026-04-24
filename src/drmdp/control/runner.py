"""
Off-policy SAC control with interleaved reward model learning or HC decomposition.

Two agent types are supported:

* ``sac`` (default): standard SAC with an optional pluggable reward model
  (e.g. IRCRRewardModel) that relabels replay buffer rewards at sample time.
  Pass ``--reward-model-type none`` to train SAC directly on delayed rewards
  without any reward model (delayed-rewards baseline).
* ``hc``: HC-decomposition SAC (Han et al., ICML 2022) that learns a
  decomposed Q-function Q^H(h_t) + Q^C(s_t, a_t) directly from the delayed
  environment rewards.  No separate reward model is used.

Usage (SAC + IRCR):
    python src/drmdp/control/runner.py --env MountainCarContinuous-v0 \\
        --delay 3 --num-steps 50000 --reward-model-type ircr \\
        --update-every-n-steps 1000 --output-dir /tmp/control-ircr

Usage (SAC delayed-rewards baseline, no reward model):
    python src/drmdp/control/runner.py --env MountainCarContinuous-v0 \\
        --delay 3 --num-steps 50000 --reward-model-type none \\
        --output-dir /tmp/control-delayed

Usage (HC):
    python src/drmdp/control/runner.py --env MountainCarContinuous-v0 \\
        --delay 3 --num-steps 50000 --agent-type hc \\
        --output-dir /tmp/control-hc
"""

import argparse
import ast
import concurrent.futures
import dataclasses
import json
import logging
import os
import tempfile
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence

import gymnasium as gym
import numpy as np
import ray
import stable_baselines3
from stable_baselines3.common import callbacks

from drmdp import core, logger, ray_utils, rewdelay
from drmdp.control import base, dgra, grd, hc, ircr


@dataclasses.dataclass(frozen=True)
class TrainingArgs:
    """Configuration for the off-policy control loop.

    Attributes:
        env: Gymnasium environment name.
        delay: Reward delay (number of steps).
        env_kwargs: Keyword arguments forwarded to ``gym.make()`` (e.g.
            ``{"max_episode_steps": 2500}``).
        reward_model_type: Reward model identifier. Supports "ircr", "dgra", "grd",
            and "none" (train directly on delayed rewards; no model instantiated).
            Only used when ``agent_type="sac"``.
        update_every_n_steps: Call reward_model.update() every N env steps.
            Only used when ``agent_type="sac"``.
        clear_buffer_on_update: Reset SAC's replay buffer after each reward
            model update. Only used when ``agent_type="sac"``.
        reward_model_kwargs: Keyword arguments forwarded to the reward model
            constructor. Only used when ``agent_type="sac"``.
        num_steps: Total environment steps to train for.
        sac_learning_rate: Learning rate for SAC actor and critic networks.
        sac_buffer_size: Capacity of SAC's replay buffer.
        sac_batch_size: Mini-batch size for SAC gradient updates.
        sac_gradient_steps: Gradient steps per env step (-1 = match collected).
        log_step_frequency: Log to ExperimentLogger every N environment steps.
        output_dir: Directory for logs and the saved SAC model.
        seed: Random seed for reproducibility.
        agent_type: Agent algorithm. "sac" uses standard SAC with a reward
            model; "hc" uses HC-decomposition SAC without a reward model.
        agent_kwargs: Keyword arguments forwarded to the agent constructor.
            Used when ``agent_type="hc"`` (e.g. ``history_hidden_size``,
            ``reg_lambda``). ``max_delay`` is derived from the delay
            distribution and cannot be overridden here.
    """

    env: str
    delay: int
    reward_model_type: str
    update_every_n_steps: int
    clear_buffer_on_update: bool
    reward_model_kwargs: Mapping[str, Any]
    num_steps: int
    sac_learning_rate: float
    sac_buffer_size: int
    sac_batch_size: int
    sac_gradient_steps: int
    log_step_frequency: int
    output_dir: str
    seed: Optional[int] = None
    agent_type: str = "sac"
    agent_kwargs: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    env_kwargs: Mapping[str, Any] = dataclasses.field(default_factory=dict)


class RewardModelUpdateCallback(callbacks.BaseCallback):
    """SB3 callback that builds episode trajectories and updates the reward model.

    Builds trajectories step-by-step from SB3 callback locals, appends
    completed episodes to a pending buffer, and flushes that buffer to
    the reward model every `update_every_n_steps` env steps.  Episode
    returns and reward model metrics are recorded via ExperimentLogger.
    """

    def __init__(
        self,
        reward_model: base.RewardModel,
        update_every_n_steps: int,
        clear_buffer_on_update: bool,
        log_step_frequency: int,
        exp_logger: logger.ExperimentLogger,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self._reward_model = reward_model
        self._update_every_n_steps = update_every_n_steps
        self._clear_buffer_on_update = clear_buffer_on_update
        self._log_step_frequency = log_step_frequency
        self._exp_logger = exp_logger

        self._episode_obs: List[np.ndarray] = []
        self._episode_actions: List[np.ndarray] = []
        self._episode_rewards: List[float] = []
        self._episode_terminals: List[bool] = []
        self._episode_steps: int = 0
        self._episode_count: int = 0
        self._last_log_steps: int = 0

        self._pending_trajectories: List[base.Trajectory] = []
        self._last_model_metrics: Mapping[str, float] = {}
        self._reward_model_total_steps: int = 0
        self._start_time: float = 0.0

    def _on_training_start(self) -> None:
        self._start_time = time.time()

    def _on_step(self) -> bool:
        obs_before = self.model._last_obs[0]
        action = self.locals["actions"][0]
        reward = float(self.locals["rewards"][0])
        done = bool(self.locals["dones"][0])

        self._episode_obs.append(obs_before.copy())
        self._episode_actions.append(action.copy())
        self._episode_rewards.append(reward)
        self._episode_terminals.append(done)
        self._episode_steps += 1

        if done:
            self._on_episode_end()

        if (
            self.num_timesteps % self._update_every_n_steps == 0
            and len(self._pending_trajectories) > 0
        ):
            self._flush_pending_trajectories()

        return True

    def _on_training_end(self) -> None:
        if len(self._pending_trajectories) > 0:
            self._flush_pending_trajectories()

    def _on_episode_end(self) -> None:
        """Finalise the current episode and log if on schedule."""
        trajectory = base.Trajectory(
            observations=np.stack(self._episode_obs),
            actions=np.stack(self._episode_actions),
            env_rewards=np.array(self._episode_rewards, dtype=np.float32),
            terminals=np.array(self._episode_terminals),
            episode_return=float(sum(self._episode_rewards)),
        )
        self._pending_trajectories.append(trajectory)
        self._episode_count += 1

        if self.num_timesteps - self._last_log_steps >= self._log_step_frequency:
            true_episode_return = float(
                self.locals["infos"][0].get("true_episode_return", 0.0)
            )
            est_rewards = self._reward_model.predict(
                trajectory.observations,
                trajectory.actions,
                trajectory.terminals,
            )
            self._exp_logger.log(
                episode=self._episode_count,
                steps=self._episode_steps,
                global_steps=self.num_timesteps,
                returns=true_episode_return,
                info={
                    "sac_total_steps": self.num_timesteps,
                    "reward_model_total_steps": self._reward_model_total_steps,
                    "delayed_returns": trajectory.episode_return,
                    "estimated_return": float(est_rewards.sum()),
                    "reward_model": dict(self._last_model_metrics),
                    "elapsed_seconds": time.time() - self._start_time,
                },
            )
            self._last_log_steps = self.num_timesteps

        self._episode_obs = []
        self._episode_actions = []
        self._episode_rewards = []
        self._episode_terminals = []
        self._episode_steps = 0

    def _flush_pending_trajectories(self) -> None:
        """Pass pending trajectories to the reward model and optionally clear the buffer."""
        self._last_model_metrics = self._reward_model.update(self._pending_trajectories)
        self._reward_model_total_steps += int(
            self._last_model_metrics.get("training_steps", 0)
        )
        self._pending_trajectories = []
        if self._clear_buffer_on_update:
            self.model.replay_buffer.reset()


class _RewardObsLoggingCallback(callbacks.BaseCallback):
    """Lightweight episode logger for the Vanilla SAC and HC agents.

    Logs per-episode returns and step counts to ExperimentLogger at a
    configurable frequency.  Unlike RewardModelUpdateCallback, this callback
    does not interact with any reward model.
    """

    def __init__(
        self,
        log_step_frequency: int,
        exp_logger: logger.ExperimentLogger,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self._log_step_frequency = log_step_frequency
        self._exp_logger = exp_logger
        self._episode_steps: int = 0
        self._episode_count: int = 0
        self._episode_rewards: List[float] = []
        self._last_log_steps: int = 0
        self._start_time: float = 0.0

    def _on_training_start(self) -> None:
        self._start_time = time.time()

    def _on_step(self) -> bool:
        self._episode_steps += 1
        self._episode_rewards.append(float(self.locals["rewards"][0]))
        if bool(self.locals["dones"][0]):
            self._on_episode_end()
        return True

    def _on_episode_end(self) -> None:
        self._episode_count += 1
        if self.num_timesteps - self._last_log_steps >= self._log_step_frequency:
            true_episode_return = float(
                self.locals["infos"][0].get("true_episode_return", 0.0)
            )
            delayed_returns = float(sum(self._episode_rewards))
            self._exp_logger.log(
                episode=self._episode_count,
                steps=self._episode_steps,
                global_steps=self.num_timesteps,
                returns=true_episode_return,
                info={
                    "total_steps": self.num_timesteps,
                    "delayed_returns": delayed_returns,
                    "elapsed_seconds": time.time() - self._start_time,
                },
            )
            self._last_log_steps = self.num_timesteps
        self._episode_steps = 0
        self._episode_rewards = []


def run(args: TrainingArgs) -> None:
    """Build environment and launch the configured agent to completion.

    The environment is wrapped with DelayedRewardWrapper (to aggregate rewards
    over windows) and ImputeMissingRewardWrapper (to replace None rewards with
    0.0 so SAC always receives a numeric value).
    """
    logging.basicConfig(level=logging.INFO)
    logging.info("Training args: %s", args)

    delay = rewdelay.ClippedPoissonDelay(args.delay)
    env = gym.make(args.env, **args.env_kwargs)
    env = core.EnvMonitorWrapper(env)
    env = rewdelay.DelayedRewardWrapper(env, delay)
    env = rewdelay.ImputeMissingRewardWrapper(env, impute_value=0.0)

    with logger.ExperimentLogger(args.output_dir, params=args) as exp_logger:
        if args.agent_type == "hc":
            _run_hc(args, env, exp_logger, delay=delay)
        else:
            _run_sac(args, env, exp_logger)


def _run_sac(
    args: TrainingArgs,
    env: Any,
    exp_logger: logger.ExperimentLogger,
) -> None:
    """Train standard SAC with a reward model or directly on delayed rewards."""
    reward_model = _make_reward_model(args, env)
    sac = stable_baselines3.SAC(
        "MlpPolicy",
        env,
        learning_rate=args.sac_learning_rate,
        buffer_size=args.sac_buffer_size,
        batch_size=args.sac_batch_size,
        gradient_steps=args.sac_gradient_steps,
        replay_buffer_class=base.RelabelingReplayBuffer,
        replay_buffer_kwargs={"reward_model": reward_model},
        seed=args.seed,
        ent_coef="auto_0.1",
        verbose=1,
    )
    callback = (
        RewardModelUpdateCallback(
            reward_model=reward_model,
            update_every_n_steps=args.update_every_n_steps,
            clear_buffer_on_update=args.clear_buffer_on_update,
            log_step_frequency=args.log_step_frequency,
            exp_logger=exp_logger,
        )
        if reward_model
        else _RewardObsLoggingCallback(
            log_step_frequency=args.log_step_frequency,
            exp_logger=exp_logger,
        )
    )
    sac.learn(
        total_timesteps=args.num_steps,
        log_interval=4,
        callback=callback,
        progress_bar=True,
    )
    sac.save(os.path.join(args.output_dir, "sac_model"))
    logging.info("Model saved to %s/sac_model", args.output_dir)


def _run_hc(
    args: TrainingArgs,
    env: Any,
    exp_logger: logger.ExperimentLogger,
    delay: rewdelay.RewardDelay,
) -> None:
    """Train HC-decomposition SAC."""
    # IntervalPositionWrapper depends on info["interval_end"] from ImputeMissingRewardWrapper,
    # so it must be applied after that wrapper.
    _, max_delay = delay.range()
    env = hc.IntervalPositionWrapper(env, max_delay=max_delay)
    # Runner owns max_delay; strip from agent_kwargs so both IntervalPositionWrapper
    # and HCSAC (and its replay buffer) use the same value from the delay distribution.
    remaining_kwargs = {
        key: val for key, val in args.agent_kwargs.items() if key != "max_delay"
    }
    agent = hc.HCSAC(
        env,
        max_delay=max_delay,
        learning_rate=args.sac_learning_rate,
        buffer_size=args.sac_buffer_size,
        batch_size=args.sac_batch_size,
        gradient_steps=args.sac_gradient_steps,
        seed=args.seed,
        ent_coef="auto_0.1",
        verbose=1,
        **remaining_kwargs,
    )
    callback = _RewardObsLoggingCallback(
        log_step_frequency=args.log_step_frequency,
        exp_logger=exp_logger,
    )
    agent.learn(
        total_timesteps=args.num_steps,
        log_interval=4,
        callback=callback,
        progress_bar=True,
    )
    agent.save(os.path.join(args.output_dir, "hc_model"))
    logging.info("Model saved to %s/hc_model", args.output_dir)


def _make_reward_model(args: TrainingArgs, env: Any) -> Optional[base.RewardModel]:
    """Instantiate the reward model specified in args.

    Args:
        args: Training configuration.
        env: Wrapped Gymnasium environment; used to extract obs_dim and
            action_dim for parametric models such as DGRA.
    """
    if args.reward_model_type == "ircr":
        return ircr.IRCRRewardModel(**args.reward_model_kwargs)
    if args.reward_model_type == "dgra":
        obs_dim = int(np.prod(env.observation_space.shape))
        action_dim = int(np.prod(env.action_space.shape))
        return dgra.DGRARewardModel(
            obs_dim=obs_dim,
            action_dim=action_dim,
            **args.reward_model_kwargs,
        )
    if args.reward_model_type == "grd":
        obs_dim = int(np.prod(env.observation_space.shape))
        action_dim = int(np.prod(env.action_space.shape))
        return grd.GRDRewardModel(
            obs_dim=obs_dim,
            action_dim=action_dim,
            **args.reward_model_kwargs,
        )
    if args.reward_model_type == "none":
        return None
    raise ValueError(f"Unknown reward_model_type: {args.reward_model_type!r}")


def run_batch(
    configs: Sequence[TrainingArgs],
    mode: str = "debug",
    max_workers: Optional[int] = None,
    ray_address: Optional[str] = None,
) -> None:
    """Launch multiple experiments under one of three execution backends.

    Args:
        configs: One TrainingArgs per experiment run (already expanded for num_runs).
        mode: "debug" runs sequentially in-process; "local" uses a ProcessPoolExecutor;
            "ray" submits Ray remote tasks.
        max_workers: Maximum parallel workers for local mode. None uses os.cpu_count().
        ray_address: Ray cluster address for ray mode. None starts a local Ray instance.
    """
    logging.info("Launching %d experiment(s) in %s mode.", len(configs), mode)
    if mode == "ray":
        ray.init(address=ray_address)
        task_refs = [_run_remote.remote(args) for args in configs]
        ray_utils.wait_till_completion(task_refs)
    elif mode == "local":
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers
        ) as executor:
            futures = [executor.submit(run, args) for args in configs]
            for future in concurrent.futures.as_completed(futures):
                future.result()
    else:
        for args in configs:
            run(args)


@ray.remote
def _run_remote(args: TrainingArgs) -> None:
    run(args)


def _load_configs(config_path: str) -> List[TrainingArgs]:
    """Load and expand experiment configs from a JSON file.

    The config must have a top-level "environments" list. Top-level fields
    (except "environments", "output_dir", "num_runs") are global defaults.
    Environment-level fields override globals; experiment-level fields override
    environment-level. Seeds are generated via core.Seeder using a global counter
    that is unique across all (environment, experiment) pairs.
    """
    with open(config_path, encoding="utf-8") as config_file:
        raw = json.load(config_file)

    top_level_output_dir: Optional[str] = raw.get("output_dir")
    top_level_num_runs: int = raw.get("num_runs", 1)
    shared_fields = {
        key: value
        for key, value in raw.items()
        if key not in ("environments", "output_dir", "num_runs")
    }
    global_defaults = {**_default_training_args(), **shared_fields}

    return _expand_environments(
        raw["environments"], global_defaults, top_level_num_runs, top_level_output_dir
    )


def _expand_environments(
    environments: Sequence[Mapping[str, Any]],
    global_defaults: Mapping[str, Any],
    top_level_num_runs: int,
    top_level_output_dir: Optional[str],
) -> List[TrainingArgs]:
    configs: List[TrainingArgs] = []
    global_exp_offset: int = 0
    for env_entry in environments:
        env_label: str = env_entry.get("name", env_entry.get("env", ""))
        env_num_runs: int = env_entry.get("num_runs", top_level_num_runs)
        env_output_dir: Optional[str] = (
            env_entry.get("output_dir") or top_level_output_dir
        )
        env_fields = {
            key: value
            for key, value in env_entry.items()
            if key not in ("name", "experiments", "output_dir", "num_runs")
        }
        env_defaults = {**global_defaults, **env_fields}
        experiments = env_entry.get("experiments", [])

        configs.extend(
            _expand_experiments(
                experiments,
                env_defaults,
                env_num_runs,
                env_output_dir,
                env_label=env_label,
                global_exp_offset=global_exp_offset,
            )
        )
        global_exp_offset += len(experiments)
    return configs


def _expand_experiments(
    experiments: Sequence[Mapping[str, Any]],
    defaults: Mapping[str, Any],
    top_level_num_runs: int,
    top_level_output_dir: Optional[str],
    env_label: Optional[str] = None,
    global_exp_offset: int = 0,
) -> List[TrainingArgs]:
    configs: List[TrainingArgs] = []
    for exp_idx, entry in enumerate(experiments):
        num_runs: int = entry.get("num_runs", top_level_num_runs)
        base_seed: Optional[int] = entry.get("seed")
        experiment_fields = {
            key: value
            for key, value in entry.items()
            if key not in ("num_runs", "output_dir", "seed")
        }
        merged_base = {**defaults, **experiment_fields}
        global_idx = global_exp_offset + exp_idx

        for run_idx in range(num_runs):
            merged = {**merged_base}
            merged["seed"] = _resolve_seed(num_runs, base_seed, global_idx, run_idx)
            merged["output_dir"] = _resolve_output_dir(
                entry, exp_idx, run_idx, top_level_output_dir, env_label
            )
            configs.append(TrainingArgs(**merged))
    return configs


def _resolve_seed(
    num_runs: int,
    base_seed: Optional[int],
    exp_idx: int,
    run_idx: int,
) -> Optional[int]:
    if num_runs == 1:
        return base_seed
    seeder_instance = base_seed if base_seed is not None else exp_idx
    return core.Seeder(instance=seeder_instance).get_seed(run_idx)


def _resolve_output_dir(
    entry: Mapping[str, Any],
    exp_idx: int,
    run_idx: int,
    top_level_output_dir: Optional[str],
    env_label: Optional[str] = None,
) -> str:
    if "output_dir" in entry:
        return str(entry["output_dir"])
    base = top_level_output_dir or os.path.join(tempfile.gettempdir(), "drmdp-batch")
    parts: List[str] = []
    if env_label:
        parts.append(env_label)
    parts.extend([f"exp-{exp_idx:03d}", f"run-{run_idx:03d}"])
    return os.path.join(base, *parts)


def _default_training_args() -> Mapping[str, Any]:
    return {
        "env": "MountainCarContinuous-v0",
        "delay": 3,
        "env_kwargs": {"max_episode_steps": 2500},
        "reward_model_type": "ircr",
        "update_every_n_steps": 1000,
        "clear_buffer_on_update": False,
        "reward_model_kwargs": {},
        "num_steps": 50000,
        "sac_learning_rate": 3e-4,
        "sac_buffer_size": 100_000,
        "sac_batch_size": 256,
        "sac_gradient_steps": -1,
        "log_step_frequency": 10_000,
        "output_dir": tempfile.gettempdir(),
        "seed": None,
        "agent_type": "sac",
        "agent_kwargs": {},
    }


def main() -> None:
    """Parse command-line arguments and launch single or batch control loop."""
    batch_cli = _parse_batch_cli()
    if batch_cli.config_file is not None:
        configs = _load_configs(batch_cli.config_file)
        run_batch(
            configs,
            mode=batch_cli.mode,
            max_workers=batch_cli.max_workers,
            ray_address=batch_cli.ray_address,
        )
    else:
        args = parse_args()
        run(args)


def parse_args() -> TrainingArgs:
    """Parse command-line arguments into a TrainingArgs instance."""
    parser = argparse.ArgumentParser(
        description="Off-policy SAC control with interleaved reward model learning"
    )
    parser.add_argument(
        "--env",
        type=str,
        default="MountainCarContinuous-v0",
        help="Gymnasium environment name",
    )
    parser.add_argument(
        "--delay",
        type=int,
        default=3,
        help="Reward delay (number of steps)",
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=2500,
        help="Maximum steps per episode (None = environment default)",
    )
    parser.add_argument(
        "--reward-model-type",
        type=str,
        default="dgra",
        choices=["ircr", "dgra", "grd", "none"],
        help="Reward model to use ('none' trains directly on delayed rewards)",
    )
    parser.add_argument(
        "--update-every-n-steps",
        type=int,
        default=1000,
        help="Update the reward model every N environment steps",
    )
    parser.add_argument(
        "--clear-buffer-on-update",
        action="store_true",
        default=False,
        help="Reset SAC replay buffer after each reward model update",
    )
    parser.add_argument(
        "--reward-model-kwarg",
        action="append",
        dest="reward_model_kwargs",
        default=[],
        metavar="KEY=VALUE",
        help="Reward-model-specific keyword argument (repeatable). "
        "Values are parsed via ast.literal_eval; unrecognised literals "
        "are kept as strings. E.g. --reward-model-kwarg max_buffer_size=200",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=50000,
        help="Total environment steps to train for",
    )
    parser.add_argument(
        "--sac-learning-rate",
        type=float,
        default=3e-4,
        help="Learning rate for SAC actor and critic",
    )
    parser.add_argument(
        "--sac-buffer-size",
        type=int,
        default=100000,
        help="Capacity of SAC's replay buffer",
    )
    parser.add_argument(
        "--sac-batch-size",
        type=int,
        default=256,
        help="Mini-batch size for SAC gradient updates",
    )
    parser.add_argument(
        "--sac-gradient-steps",
        type=int,
        default=-1,
        help="Gradient steps per env step (-1 = match collected steps)",
    )
    parser.add_argument(
        "--log-step-frequency",
        type=int,
        default=10000,
        help="Log to ExperimentLogger every N environment steps",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=tempfile.gettempdir(),
        help="Directory for logs and saved model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--agent-type",
        type=str,
        default="sac",
        choices=["sac", "hc"],
        help="Agent algorithm: 'sac' (default) or 'hc' (HC-decomposition SAC)",
    )
    parser.add_argument(
        "--agent-kwarg",
        action="append",
        dest="agent_kwargs",
        default=[],
        metavar="KEY=VALUE",
        help="Agent-specific keyword argument (repeatable). "
        "Values are parsed via ast.literal_eval; unrecognised literals "
        "are kept as strings. E.g. --agent-kwarg key=value",
    )

    args, _ = parser.parse_known_args()
    args_dict = vars(args)
    args_dict["reward_model_kwargs"] = parse_reward_model_kwargs(
        args_dict["reward_model_kwargs"]
    )
    args_dict["agent_kwargs"] = parse_reward_model_kwargs(args_dict["agent_kwargs"])
    args_dict["env_kwargs"] = {"max_episode_steps": args_dict.pop("max_episode_steps")}
    return TrainingArgs(**args_dict)


def parse_reward_model_kwargs(
    pairs: Sequence[str],
) -> Mapping[str, Any]:
    """Convert a list of 'key=value' strings into a keyword-argument mapping.

    Values are parsed with ast.literal_eval so that integers, floats, and
    booleans are returned with their native types.  Strings that cannot be
    parsed as literals are kept as-is.
    """
    kwargs: Dict[str, Any] = {}
    for pair in pairs:
        key, _, raw = pair.partition("=")
        try:
            value: Any = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            value = raw
        kwargs[key] = value
    return kwargs


def _parse_batch_cli() -> argparse.Namespace:
    """Parse only batch-mode flags; single-run flags are ignored here."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config-file", type=str, default=None)
    parser.add_argument(
        "--mode", type=str, default="debug", choices=["debug", "local", "ray"]
    )
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--ray-address", type=str, default=None)
    args, _ = parser.parse_known_args()
    return args


if __name__ == "__main__":
    main()
