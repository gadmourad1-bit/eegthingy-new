"""NPZ pose-trace writer/loader and CLI parser behavior."""

from datetime import datetime

import numpy as np
import pytest

from tiago_maze import trace
from tiago_maze.__main__ import build_params


def test_trace_filename():
    when = datetime(2026, 7, 1, 20, 15, 30)
    assert trace.trace_filename(42, when) == "tiago_maze_20260701_201530_seed42.npz"


def test_write_and_load_roundtrip(tmp_path):
    samples = {
        "t": [0.0, 0.05, 0.10],
        "x": [0.0, 0.01, 0.02],
        "y": [0.0, 0.0, 0.0],
        "yaw": [0.0, 0.1, 0.2],
        "pitch": [0.0, 0.0, 0.0],
    }
    meta = {"seed": 42, "n_turns": 40, "cell": 1.6, "wall_height": 2.0,
            "min_gap": 1, "max_gap": 3, "trace_period": 0.05,
            "forward_speed": 0.2, "turn_speed": 0.6, "wall_thickness": 0.05}
    path = trace.write_trace(tmp_path, samples, meta)
    assert path.exists() and path.suffix == ".npz"

    got, gmeta = trace.load_trace(path)
    for k in ("t", "x", "y", "yaw", "pitch"):
        np.testing.assert_allclose(got[k], samples[k])
    assert gmeta["seed"] == 42
    assert gmeta["n_turns"] == 40
    assert gmeta["cell"] == pytest.approx(1.6)


def test_matching_report_and_trace_names():
    from tiago_maze import report
    when = datetime(2026, 7, 1, 20, 15, 30)
    base = "tiago_maze_20260701_201530_seed7"
    assert report.report_filename(7, when) == base + ".csv"
    assert trace.trace_filename(7, when) == base + ".npz"


# --- CLI parser ---------------------------------------------------------

def test_view_and_speed_args():
    p = build_params(["--view", "first", "--turn-speed", "1.2", "--forward-speed", "0.5"])
    assert p.view == "first"
    assert p.control.turn_speed == 1.2
    assert p.control.forward_speed == 0.5


def test_replay_and_fps_args():
    p = build_params(["--replay", "foo.npz", "--fps", "144"])
    assert p.replay == "foo.npz"
    assert p.fps == 144


def test_unknown_arg_triggers_help_exit(capsys):
    with pytest.raises(SystemExit) as exc:
        build_params(["--totally-unknown"])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "usage:" in err          # full help was printed
    assert "--view" in err


def test_help_flag_exits_zero(capsys):
    with pytest.raises(SystemExit) as exc:
        build_params(["--help"])
    assert exc.value.code == 0
    assert "TIAGo websocket maze game" in capsys.readouterr().out


# --- CLI metadata (--meta) ---------------------------------------------

def test_meta_json_string():
    p = build_params(["--meta", '{"subject": "S01", "condition": "A", "trial": 3}'])
    assert p.extra_meta == {"subject": "S01", "condition": "A", "trial": 3}


def test_meta_from_file(tmp_path):
    f = tmp_path / "m.json"
    f.write_text('{"subject": "S02", "notes": "left-handed"}')
    p = build_params(["--meta", f"@{f}"])
    assert p.extra_meta == {"subject": "S02", "notes": "left-handed"}


def test_meta_default_empty():
    assert build_params([]).extra_meta == {}


def test_meta_bad_json_triggers_help(capsys):
    with pytest.raises(SystemExit) as exc:
        build_params(["--meta", "{not valid json"])
    assert exc.value.code == 2
    assert "usage:" in capsys.readouterr().err


def test_meta_non_object_rejected(capsys):
    with pytest.raises(SystemExit) as exc:
        build_params(["--meta", "[1, 2, 3]"])
    assert exc.value.code == 2
    assert "usage:" in capsys.readouterr().err


def test_meta_written_into_report(tmp_path):
    """User metadata ends up as summary rows, colliding keys get user_ prefix."""
    from tiago_maze import report
    meta = {"seed": 7, "completed": True, "subject": "S01", "condition": "A"}
    path = report.write_report(tmp_path, meta, [], when=datetime(2026, 7, 1, 0, 0, 0))
    text = path.read_text()
    assert "subject,S01" in text
    assert "condition,A" in text


# --- seed round-trips for replay ---------------------------------------

def test_saved_seed_regenerates_identical_maze():
    """The trace seed must reproduce the exact maze (walls) on replay."""
    from tiago_maze import maze as maze_mod
    from tiago_maze.params import MazeParams

    original = maze_mod.generate(MazeParams(n_turns=40, seed=None))  # random seed
    saved_seed = original.seed
    assert isinstance(saved_seed, int)

    # What replay does: rebuild from the saved seed.
    rebuilt = maze_mod.generate(MazeParams(n_turns=40, seed=saved_seed))
    assert rebuilt.cells == original.cells
    assert [(w.cx, w.cy, w.sx, w.sy) for w in rebuilt.walls] == \
           [(w.cx, w.cy, w.sx, w.sy) for w in original.walls]
