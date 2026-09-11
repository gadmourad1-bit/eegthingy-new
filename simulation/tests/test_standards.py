"""The four historical standard mazes must never silently change."""

from tiago_maze.maze import generate
from tiago_maze.__main__ import build_params
from tiago_maze.params import MazeParams
from tiago_maze.standards import STANDARD_MAZES, get_standard_maze


def test_four_original_fixed_mazes_are_registered():
    assert set(STANDARD_MAZES) == {1, 2, 3, 4}
    assert [STANDARD_MAZES[i].seed for i in range(1, 5)] == [11, 12, 13, 14]


def test_frozen_routes_match_original_seeded_generator():
    for standard in STANDARD_MAZES.values():
        generated = generate(MazeParams(n_turns=40, seed=standard.seed))
        assert tuple(turn.direction for turn in generated.turn_points) == standard.directions


def test_each_standard_has_exactly_forty_corners():
    for standard in STANDARD_MAZES.values():
        assert len(standard.route) == 40
        assert set(standard.route) == {"L", "R"}


def test_invalid_standard_number_is_rejected():
    try:
        get_standard_maze(5)
    except ValueError as error:
        assert "1, 2, 3, or 4" in str(error)
    else:
        raise AssertionError("invalid standard maze was accepted")


def test_cli_standard_maze_uses_frozen_route():
    params = build_params(["--standard-maze", "3"])
    standard = get_standard_maze(3)
    assert params.maze.seed == standard.seed
    assert params.maze.n_turns == 40
    assert params.maze.turn_sequence == standard.directions
    assert params.extra_meta["standard_maze"] == 3
