"""Tester for loqs.tools.multiprogramrunner.CheckpointConfig"""

from pathlib import Path

import pytest

from loqs.tools.multiprogramrunner import CheckpointConfig


class TestCheckpointConfigPathCoercion:
    """Test string-to-Path coercion."""

    def test_dirs_str_coerced_to_path(self):
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
        with pytest.raises(ValueError):
            CheckpointConfig(resume=True)

    def test_keep_shot_results_with_item_but_without_shot_checkpoint_dir_raises(
        self,
    ):
        with pytest.raises(ValueError):
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
        text = str(cfg)
        assert str(item_ckpt_dir) in text
        assert str(shot_ckpt_dir) in text


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
