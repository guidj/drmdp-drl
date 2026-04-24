import dataclasses
import json

import numpy as np
import pytest

from drmdp import logger


@dataclasses.dataclass
class SampleConfig:
    name: str
    value: int


@dataclasses.dataclass(frozen=True)
class FrozenConfig:
    alpha: float
    beta: str


class TestDataclassFromDict:
    def test_basic_round_trip(self):
        original = SampleConfig(name="test", value=42)
        data = dataclasses.asdict(original)
        restored = logger.dataclass_from_dict(SampleConfig, data)
        assert restored.name == "test"
        assert restored.value == 42

    def test_raises_on_non_dataclass_class(self):
        class PlainClass:
            pass

        with pytest.raises(ValueError):
            logger.dataclass_from_dict(PlainClass, {"x": 1})

    def test_raises_on_dataclass_instance(self):
        instance = SampleConfig(name="x", value=1)
        # Passing an instance instead of the class should raise
        with pytest.raises(ValueError):
            logger.dataclass_from_dict(instance, {"name": "x", "value": 1})


class TestJsonFromDict:
    def test_none_level_returns_deep_copy(self):
        original = {"a": {"b": 1}, "c": 2}
        result = logger.json_from_dict(original, dict_encode_level=None)
        assert result == original
        # Verify it's a copy, not the same object
        result["a"]["b"] = 999
        assert original["a"]["b"] == 1

    def test_level_0_encodes_top_level_nested_dicts_as_strings(self):
        original = {"a": {"b": 1}, "c": 2}
        result = logger.json_from_dict(original, dict_encode_level=0)
        # Value at level 0 that is a dict gets JSON-encoded
        assert isinstance(result["a"], str)
        assert json.loads(result["a"]) == {"b": 1}
        # Non-dict value unchanged
        assert result["c"] == 2

    def test_level_1_encodes_depth_2_dicts(self):
        original = {"outer": {"inner": {"deep": 1}}}
        result = logger.json_from_dict(original, dict_encode_level=1)
        # Level 0: outer → dict, recurse
        # Level 1: inner → dict at level 1 >= 1, encode as string
        assert isinstance(result["outer"]["inner"], str)
        assert json.loads(result["outer"]["inner"]) == {"deep": 1}

    def test_non_dict_values_preserved_at_any_depth(self):
        original = {"a": 42, "b": [1, 2, 3], "c": "hello"}
        result = logger.json_from_dict(original, dict_encode_level=0)
        assert result["a"] == 42
        assert result["b"] == [1, 2, 3]
        assert result["c"] == "hello"


class TestSaveModel:
    def test_creates_file_with_npz_extension(self, tmp_path):
        weights = np.array([1.0, 2.0, 3.0])
        logger.save_model(weights, "model.npz", str(tmp_path))
        assert (tmp_path / "model.npz").exists()

    def test_appends_npz_extension_when_missing(self, tmp_path):
        weights = np.array([1.0, 2.0])
        logger.save_model(weights, "model", str(tmp_path))
        assert (tmp_path / "model.npz").exists()

    def test_creates_output_directory_if_missing(self, tmp_path):
        new_dir = tmp_path / "nested" / "model_dir"
        weights = np.array([1.0])
        logger.save_model(weights, "weights.npz", str(new_dir))
        assert new_dir.exists()
        assert (new_dir / "weights.npz").exists()

    def test_saved_array_matches_input(self, tmp_path):
        weights = np.array([1.5, 2.5, 3.5], dtype=np.float64)
        logger.save_model(weights, "weights.npz", str(tmp_path))
        output_path = tmp_path / "weights.npz"
        with open(str(output_path), "rb") as readable:
            loaded = np.load(readable)
        np.testing.assert_array_equal(loaded, weights)


class TestExperimentLogger:
    def test_params_json_written_on_init(self, tmp_path):
        params = {"lr": 0.001, "steps": 100}
        logger.ExperimentLogger(str(tmp_path), params)
        param_file = tmp_path / logger.ExperimentLogger.PARAM_FILE_NAME
        assert param_file.exists()
        with open(str(param_file), "r", encoding="UTF-8") as readable:
            saved = json.load(readable)
        assert saved == params

    def test_dataclass_params_serialized_as_dict(self, tmp_path):
        params = SampleConfig(name="run1", value=10)
        logger.ExperimentLogger(str(tmp_path), params)
        param_file = tmp_path / logger.ExperimentLogger.PARAM_FILE_NAME
        with open(str(param_file), "r", encoding="UTF-8") as readable:
            saved = json.load(readable)
        assert saved == {"name": "run1", "value": 10}

    def test_context_manager_creates_log_file(self, tmp_path):
        params = {"x": 1}
        with logger.ExperimentLogger(str(tmp_path), params) as exp_logger:
            exp_logger.log(episode=0, steps=10, returns=5.0)
        log_file = tmp_path / logger.ExperimentLogger.LOG_FILE_NAME
        assert log_file.exists()

    def test_log_appends_jsonl_entry(self, tmp_path):
        params = {}
        with logger.ExperimentLogger(str(tmp_path), params) as exp_logger:
            exp_logger.log(episode=1, steps=50, returns=3.14, info={"extra": "data"})

        log_file = tmp_path / logger.ExperimentLogger.LOG_FILE_NAME
        with open(str(log_file), "r", encoding="UTF-8") as readable:
            lines = [line.strip() for line in readable if line.strip()]

        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["episode"] == 1
        assert entry["steps"] == 50
        assert entry["returns"] == 3.14
        assert entry["info"] == {"extra": "data"}

    def test_close_before_open_raises_runtime_error(self, tmp_path):
        params = {}
        exp_logger = logger.ExperimentLogger(str(tmp_path), params)
        with pytest.raises(RuntimeError):
            exp_logger.close()

    def test_log_before_open_raises_runtime_error(self, tmp_path):
        params = {}
        exp_logger = logger.ExperimentLogger(str(tmp_path), params)
        with pytest.raises(RuntimeError):
            exp_logger.log(episode=0, steps=1, returns=0.0)
