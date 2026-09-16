import pytest

from pipeline.checkpoints import resolve_checkpoint


def make_run(root, steps):
    for step in steps:
        (root / "checkpoints" / f"{step:06d}" / "pretrained_model").mkdir(parents=True)
    return root


def test_picks_the_highest_step_not_the_lexically_largest(tmp_path):
    run = make_run(tmp_path, [20000, 100000, 40000])
    assert resolve_checkpoint(run) == run / "checkpoints" / "100000" / "pretrained_model"


def test_follows_the_last_symlink_when_present(tmp_path):
    run = make_run(tmp_path, [20000, 40000])
    link = run / "checkpoints" / "last"
    link.symlink_to(run / "checkpoints" / "020000", target_is_directory=True)
    assert resolve_checkpoint(run) == run / "checkpoints" / "020000" / "pretrained_model"


def test_explicit_step_wins(tmp_path):
    run = make_run(tmp_path, [20000, 40000])
    assert resolve_checkpoint(run, "020000") == run / "checkpoints" / "020000" / "pretrained_model"


def test_missing_step_lists_what_is_available(tmp_path):
    run = make_run(tmp_path, [20000, 40000])
    with pytest.raises(FileNotFoundError, match="020000, 040000"):
        resolve_checkpoint(run, "999999")


def test_no_checkpoints_at_all_is_a_clear_error(tmp_path):
    (tmp_path / "checkpoints").mkdir()
    with pytest.raises(FileNotFoundError, match="No checkpoints"):
        resolve_checkpoint(tmp_path)


def test_dangling_last_symlink_falls_back_to_highest_step(tmp_path):
    run = make_run(tmp_path, [20000, 40000])
    link = run / "checkpoints" / "last"
    link.symlink_to(run / "checkpoints" / "999999", target_is_directory=True)
    assert resolve_checkpoint(run) == run / "checkpoints" / "040000" / "pretrained_model"


def test_last_symlink_without_pretrained_model_falls_back_to_highest_step(tmp_path):
    run = make_run(tmp_path, [20000, 40000])
    # Create a directory for last to point to, but without pretrained_model
    (run / "checkpoints" / "incomplete").mkdir()
    link = run / "checkpoints" / "last"
    link.symlink_to(run / "checkpoints" / "incomplete", target_is_directory=True)
    assert resolve_checkpoint(run) == run / "checkpoints" / "040000" / "pretrained_model"


def test_non_numeric_entries_ignored(tmp_path):
    run = make_run(tmp_path, [20000, 40000])
    # Add non-numeric entries that should be ignored
    (run / "checkpoints" / "tmp").mkdir()
    (run / "checkpoints" / "README.txt").touch()
    assert resolve_checkpoint(run) == run / "checkpoints" / "040000" / "pretrained_model"


def test_missing_output_dir_is_a_clear_error(tmp_path):
    nonexistent = tmp_path / "does_not_exist"
    with pytest.raises(FileNotFoundError, match="No checkpoints directory"):
        resolve_checkpoint(nonexistent)


def test_unpadded_step_resolves_to_padded_directory(tmp_path):
    run = make_run(tmp_path, [20000, 40000])
    assert resolve_checkpoint(run, "20000") == run / "checkpoints" / "020000" / "pretrained_model"


def test_non_numeric_step_is_a_clear_error(tmp_path):
    run = make_run(tmp_path, [20000, 40000])
    with pytest.raises(ValueError, match="not a valid checkpoint step"):
        resolve_checkpoint(run, "invalid")
