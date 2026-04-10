"""
Utilities for logging experiments.
"""

import contextlib
import copy
import dataclasses
import json
import os.path
import types
from typing import (
    Any,
    Callable,
    Mapping,
    Optional,
    Type,
)

import numpy as np
import tensorflow as tf

from drmdp import core


class ExperimentLogger(contextlib.AbstractContextManager):
    """
    Logs info for an experiment for given episodes.
    """

    LOG_FILE_NAME = "experiment-logs.jsonl"
    PARAM_FILE_NAME = "experiment-params.json"

    def __init__(
        self,
        log_dir: str,
        experiment_instance: core.ExperimentInstance,
    ):
        self.log_file = os.path.join(log_dir, self.LOG_FILE_NAME)
        self.param_file = os.path.join(log_dir, self.PARAM_FILE_NAME)
        if not tf.io.gfile.exists(log_dir):
            tf.io.gfile.makedirs(log_dir)

        with tf.io.gfile.GFile(self.param_file, "w") as writer:
            writer.write(json.dumps(dataclasses.asdict(experiment_instance)))

        self._writer: Optional[tf.io.gfile.GFile] = None

    def open(self) -> None:
        """
        Opens the log file for writing.
        """
        self._writer = tf.io.gfile.GFile(self.log_file, "w")

    def close(self) -> None:
        """
        Closes the log file.
        """
        if self._writer is None:
            raise RuntimeError("File is not opened")
        self._writer.close()

    def __enter__(self) -> "ExperimentLogger":
        self.open()
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[types.TracebackType],
    ) -> None:
        self.close()
        super().__exit__(exc_type, exc_value, traceback)

    def log(
        self,
        episode: int,
        steps: int,
        returns: float,
        info: Optional[Mapping[str, Any]] = None,
    ):
        """
        Logs an experiment entry for an episode.
        """
        entry = {
            "episode": episode,
            "steps": steps,
            "returns": returns,
            "info": info,
        }

        if self._writer is None:
            raise RuntimeError("File is not opened")
        self._writer.write(f"{json.dumps(entry)}\n")


def dataclass_from_dict(clazz: Callable, data: Mapping[str, Any]):  # type: ignore [arg-type]
    """
    Creates an instance of a dataclass from a dictionary.
    """
    if not (dataclasses.is_dataclass(clazz) and isinstance(clazz, type)):
        raise ValueError(f"Expecting a dataclass class. Got {clazz}")
    fields = list(dataclasses.fields(clazz))
    return clazz(**{field.name: data[field.name] for field in fields})


def json_from_dict(
    obj: Mapping[str, Any], dict_encode_level: Optional[int] = None
) -> Mapping[str, Any]:
    """
    Converts a dict into a json object.
    If `level` is set, then nested fields at depth `level + 1`
    are converted to strings.
    """
    mapping = copy.deepcopy(obj)
    if dict_encode_level is None:
        return mapping

    def go(data: Any, level: int):
        """Recursively encode nested dicts to JSON strings at the threshold depth."""
        if isinstance(data, Mapping):
            if level >= dict_encode_level:
                return json.dumps(data)
            return {key: go(value, level + 1) for key, value in data.items()}
        return data

    return {key: go(value, level=0) for key, value in mapping.items()}


def save_model(weights: np.ndarray, name: str, model_dir: str) -> None:
    """
    Saves a model to a given path.
    """
    name = name if name.endswith(".npz") else f"{name}.npz"
    output_path = os.path.join(model_dir, name)
    if not tf.io.gfile.exists(model_dir):
        tf.io.gfile.makedirs(model_dir)
    with tf.io.gfile.GFile(output_path, "wb") as writable:
        np.save(writable, arr=weights, allow_pickle=False)
