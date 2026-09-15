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
