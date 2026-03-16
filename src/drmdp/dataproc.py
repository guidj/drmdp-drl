import copy
import uuid
from typing import Any, Mapping, Optional, Sequence

import gymnasium as gym
import pandas as pd
import ray
import ray.data

MAPPERS_NAMES = {}

POLICY_TYPES = {}

ORDERED_METHODS = []


def collection_traj_data(
    env: gym.Env, steps: int, seed: Optional[int] = None, include_term: bool = False
):
    """
    Collects sample trajectory data.

    Args:
        env: Gymnasium environment
        steps: Number of steps to collect
        seed: Random seed for environment reset
        include_term: If True, include term flag in buffer tuples

    Returns:
        List of tuples: (obs, action, next_obs, rew) if include_term=False,
                       (obs, action, next_obs, rew, term) if include_term=True
    """
    obs, _ = env.reset(seed=seed)
    step = 0
    buffer = []
    while step < steps:
        action = env.action_space.sample()
        (
            next_obs,
            rew,
            term,
            trunc,
            _,
        ) = env.step(action)
        step += 1
        done = term or trunc
        if include_term:
            buffer.append((obs, action, next_obs, rew, term))  # type: ignore
        else:
            buffer.append((obs, action, next_obs, rew))  # type: ignore
        obs = next_obs
        if done:
            obs, _ = env.reset()
    return buffer


def process_data(df_raw):
    """
    Applies data filfers, and defines the correct
    policy identifier.
    """

    def simplify_meta(meta):
        new_meta = dict(**meta, **meta["experiment"])
        new_meta["reward_mapper"] = MAPPERS_NAMES[
            new_meta["problem_spec"]["reward_mapper"]["name"]
        ]
        new_meta["policy_type"] = POLICY_TYPES[new_meta["problem_spec"]["policy_type"]]
        del new_meta["experiment"]
        return new_meta

    def get_method(meta: Mapping[str, Any]):
        if meta["policy_type"] != "PP":
            return meta["policy_type"]
        else:
            return meta["reward_mapper"]

    def filter_method(method):
        return method not in (["L-TDD[CV]", "L-TDD[CV-R]", "LEAST", "OP-S"])

    df_proc = copy.deepcopy(df_raw)
    df_proc["meta"] = df_proc["meta"].apply(simplify_meta)
    df_proc["method"] = df_proc["meta"].apply(get_method)
    df_proc = df_proc[df_proc["method"].apply(filter_method)]
    return df_proc


def read_data(files, reader: str = "ray"):
    """
    Reads experiment results data, using either
    `ray` or `pandas`.
    """
    if reader == "ray":
        with ray.init():
            ds_metrics = ray.data.read_parquet(files)
            df_metrics = ds_metrics.to_pandas()
    elif reader == "pd":
        dfs = []
        for file in files:
            dfs.append(pd.read_parquet(file))
        df_metrics = pd.concat(dfs)
    else:
        raise ValueError(reader)

    return process_data(df_metrics)


def wide_metrics(df_metrics):
    """
    Drops the column `metrics`, and explodes
    the `returns` column.
    """
    df_raw = df_metrics.drop(["metrics"], axis=1, inplace=False)
    return df_raw.explode("returns")


def get_distinct_envs(df_data: pd.DataFrame):
    """
    Returns a set of unique environments.
    """
    envs = {}
    for row in df_data.to_dict("records"):
        env_spec = row["meta"]["env_spec"]
        env_name = env_spec["name"]
        env_args = env_spec["args"]
        envs[env_name] = env_args
    return envs


def drop_duplicate_sets(df_data: pd.DataFrame, keys: Sequence[str]):
    """
    Drops rows that have the same value
    for columns specied with `keys`.
    """
    col = str(uuid.uuid4())
    rows = []
    for row in df_data.to_dict("records"):
        col_set = sorted([row[key] for key in keys])
        new_row = copy.deepcopy(row)
        new_row[col] = col_set
        rows.append(new_row)
    df_raw = pd.DataFrame(rows)
    return df_raw.drop_duplicates(col).drop([col], axis=1)
