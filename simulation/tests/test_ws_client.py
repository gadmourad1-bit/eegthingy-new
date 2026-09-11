"""Fresh-window gating for the live EEG maze."""

from tiago_maze.ws_client import fresh_committed_decision


def _payload(timestamps, *, decision=1, final=1, n=5):
    return {
        "smoothed": {"decision": decision, "final": final, "n": n},
        "buffer": [{"timestamp": value} for value in timestamps],
    }


def test_rejects_consensus_carried_over_from_before_robot_stopped():
    data = _payload([8.0, 9.0, 10.0, 11.0, 12.0])
    assert fresh_committed_decision(data, after_timestamp=10.5) == 0


def test_requires_final_commit_not_early_consensus():
    data = _payload([11.0, 12.0, 13.0, 14.0, 15.0], decision=2, final=0)
    assert fresh_committed_decision(data, after_timestamp=10.5) == 0


def test_accepts_commit_when_entire_vote_buffer_is_post_stop():
    data = _payload([11.0, 12.0, 13.0, 14.0, 15.0], decision=2, final=2)
    assert fresh_committed_decision(data, after_timestamp=10.5) == 2
