"""Tester for loqs.tools.multiprogramrunner.CheckpointConfig"""

from pathlib import Path

import pytest

from loqs.tools.multiprogramrunner import CheckpointConfig


class TestCheckpointConfigDefaults:
    """Test default values of CheckpointConfig."""

    def test_defaults(self):
        cfg = CheckpointConfig()
        assert cfg.item_checkpoint_dir is None
        assert cfg.item_checkpoint is False
        assert cfg.shot_checkpoint_dir is None
        assert cfg.shot_checkpoint is False
        assert cfg.resume is False
        assert cfg.force_resume is False
        assert cfg.lazy_loading is True
        assert cfg.keep_shot_results is False
        assert cfg.poll_interval == 1.0
        assert cfg.show_progress is True
        assert cfg.runner_filename == "runner.h5"
        assert cfg.results_filename == "results.h5"


class TestCheckpointConfigPathCoercion:
    """Test string-to-Path coercion."""

    def test_item_checkpoint_dir_str_coerced_to_path(self):
        cfg = CheckpointConfig(item_checkpoint_dir="/tmp/some/dir")
        assert isinstance(cfg.item_checkpoint_dir, Path)
        assert cfg.item_checkpoint_dir == Path("/tmp/some/dir")

    def test_shot_checkpoint_dir_str_coerced_to_path(self):
        cfg = CheckpointConfig(shot_checkpoint_dir="/tmp/other/dir")
        assert isinstance(cfg.shot_checkpoint_dir, Path)
        assert cfg.shot_checkpoint_dir == Path("/tmp/other/dir")

    def test_both_dirs_str_coerced_to_path(self):
        cfg = CheckpointConfig(
            item_checkpoint_dir="/tmp/item",
            shot_checkpoint_dir="/tmp/shot",
        )
        assert isinstance(cfg.item_checkpoint_dir, Path)
        assert isinstance(cfg.shot_checkpoint_dir, Path)
        assert cfg.item_checkpoint_dir == Path("/tmp/item")
        assert cfg.shot_checkpoint_dir == Path("/tmp/shot")


class TestCheckpointConfigValidation:
    """Test validation rules."""

    def test_resume_true_without_item_checkpoint_dir_raises(self):
        with pytest.raises(ValueError, match="resume"):
            CheckpointConfig(resume=True)

    def test_keep_shot_results_without_item_checkpoint_dir_raises(self):
        with pytest.raises(ValueError, match="keep_shot_results"):
            CheckpointConfig(keep_shot_results=True)

    def test_keep_shot_results_with_item_but_without_shot_checkpoint_dir_raises(
        self,
    ):
        with pytest.raises(ValueError, match="keep_shot_results"):
            CheckpointConfig(
                item_checkpoint_dir="/tmp/x",
                keep_shot_results=True,
            )

    def test_keep_shot_results_with_both_dirs_succeeds(self):
        cfg = CheckpointConfig(
            item_checkpoint_dir="/tmp/x",
            shot_checkpoint_dir="/tmp/y",
            keep_shot_results=True,
        )
        assert cfg.item_checkpoint is True
        assert cfg.shot_checkpoint is True
        assert cfg.keep_shot_results is True


class TestCheckpointConfigStrOutput:
    """Test __str__ output."""

    def test_str_with_no_checkpointing(self):
        cfg = CheckpointConfig()
        expected = (
            "CheckpointConfig:\n"
            "\tItem checkpointing:\toff\n"
            "\tShot checkpointing:\toff\n"
            "\tPoll interval:\t1.0s\n"
            "\tShow progress:\tTrue"
        )
        assert str(cfg) == expected

    def test_str_with_item_checkpointing_enabled(self):
        ckpt_dir = Path("/tmp/ckpt")
        cfg = CheckpointConfig(
            item_checkpoint_dir=ckpt_dir,
            resume=True,
        )
        expected = (
            "CheckpointConfig:\n"
            "\tItem checkpointing:\ton\n"
            f"\t\tCheckpoint directory:\t{ckpt_dir}\n"
            "\t\tResume from checkpoint:\tTrue\n"
            "\t\tForce resume:\tFalse\n"
            "\t\tRunner file:\trunner.h5\n"
            "\tShot checkpointing:\toff\n"
            "\tPoll interval:\t1.0s\n"
            "\tShow progress:\tTrue"
        )
        assert str(cfg) == expected

    def test_str_with_both_checkpointing_enabled(self):
        item_ckpt_dir = Path("/tmp/item")
        shot_ckpt_dir = Path("/tmp/shot")
        cfg = CheckpointConfig(
            item_checkpoint_dir=item_ckpt_dir,
            shot_checkpoint_dir=shot_ckpt_dir,
            resume=True,
            keep_shot_results=True,
            lazy_loading=False,
        )
        expected = (
            "CheckpointConfig:\n"
            "\tItem checkpointing:\ton\n"
            f"\t\tCheckpoint directory:\t{item_ckpt_dir}\n"
            "\t\tResume from checkpoint:\tTrue\n"
            "\t\tForce resume:\tFalse\n"
            "\t\tRunner file:\trunner.h5\n"
            "\tShot checkpointing:\ton\n"
            f"\t\tCheckpoint directory:\t{shot_ckpt_dir}\n"
            "\t\tKeep shot results:\tTrue\n"
            "\t\tLazy loading:\tFalse\n"
            "\t\tResults file:\tresults.h5\n"
            "\tPoll interval:\t1.0s\n"
            "\tShow progress:\tTrue"
        )
        assert str(cfg) == expected


class TestCheckpointConfigSerialization:
    """Test JSON serialization round-trip."""

    def test_round_trip_with_paths_and_non_defaults(self, tmp_path):
        cfg = CheckpointConfig(
            item_checkpoint_dir="/tmp/item",
            shot_checkpoint_dir="/tmp/shot",
            keep_shot_results=True,
            poll_interval=2.5,
            runner_filename="custom.h5",
        )
        path = tmp_path / "config.json"
        cfg.write(path)
        loaded = CheckpointConfig.read(path)

        assert isinstance(loaded.item_checkpoint_dir, Path)
        assert loaded.item_checkpoint_dir == Path("/tmp/item")
        assert isinstance(loaded.shot_checkpoint_dir, Path)
        assert loaded.shot_checkpoint_dir == Path("/tmp/shot")
        assert loaded.keep_shot_results is True
        assert loaded.poll_interval == 2.5
        assert loaded.runner_filename == "custom.h5"
        assert loaded.item_checkpoint is True
        assert loaded.shot_checkpoint is True

    def test_round_trip_with_none_values(self, tmp_path):
        cfg = CheckpointConfig(
            item_checkpoint_dir=None,
            shot_checkpoint_dir=None,
            resume=False,
            force_resume=False,
            lazy_loading=True,
            keep_shot_results=False,
            poll_interval=1.0,
            show_progress=True,
            runner_filename="runner.h5",
            results_filename="results.h5",
        )
        path = tmp_path / "config.json"
        cfg.write(path)
        loaded = CheckpointConfig.read(path)

        assert loaded.item_checkpoint_dir is None
        assert loaded.shot_checkpoint_dir is None
        assert loaded.resume is False
        assert loaded.force_resume is False
        assert loaded.lazy_loading is True
        assert loaded.keep_shot_results is False
        assert loaded.poll_interval == 1.0
        assert loaded.show_progress is True
        assert loaded.runner_filename == "runner.h5"
        assert loaded.results_filename == "results.h5"
