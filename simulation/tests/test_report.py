"""CSV report writer."""

import csv
from datetime import datetime

from tiago_maze import report


def sample_records():
    return [
        {
            "decision": 1, "turn": 1, "corner_x": 9.6, "corner_y": 0.0,
            "intended_dir": "LEFT", "chosen_dir": "LEFT", "result": "correct",
            "wait_s": 1.23, "sim_time_s": 5.0, "robot_x": 9.4, "robot_y": 0.0,
            "robot_yaw_deg": 0.0, "front_m": 0.79,
        },
        {
            "decision": 2, "turn": 2, "corner_x": 9.6, "corner_y": 3.2,
            "intended_dir": "RIGHT", "chosen_dir": "LEFT", "result": "wrong",
            "wait_s": 2.5, "sim_time_s": 12.0, "robot_x": 9.6, "robot_y": 3.0,
            "robot_yaw_deg": 90.0, "front_m": 0.78,
        },
    ]


def test_filename_has_timestamp_and_seed():
    when = datetime(2026, 7, 1, 20, 15, 30)
    name = report.report_filename(1234, when)
    assert name == "tiago_maze_20260701_201530_seed1234.csv"


def test_write_report_creates_file(tmp_path):
    meta = {"seed": 1234, "completed": True, "decisions": 2}
    when = datetime(2026, 7, 1, 20, 15, 30)
    path = report.write_report(tmp_path, meta, sample_records(), when=when)
    assert path.exists()
    assert path.name == "tiago_maze_20260701_201530_seed1234.csv"

    text = path.read_text()
    # summary section
    assert "field,value" in text
    assert "seed,1234" in text
    # blank separator then the decision header
    assert "decision,turn,corner_x" in text
    # both decisions present
    assert "correct" in text and "wrong" in text


def test_decision_table_roundtrips(tmp_path):
    meta = {"seed": 7}
    path = report.write_report(tmp_path, meta, sample_records())

    lines = path.read_text().splitlines()
    blank = lines.index("")
    table = lines[blank + 1:]
    rows = list(csv.DictReader(table))
    assert len(rows) == 2
    assert rows[0]["intended_dir"] == "LEFT"
    assert rows[0]["result"] == "correct"
    assert rows[1]["chosen_dir"] == "LEFT"
    assert rows[1]["result"] == "wrong"
    assert [r["decision"] for r in rows] == ["1", "2"]
    # every declared column is present
    assert set(report.DECISION_FIELDS) <= set(rows[0].keys())


def test_creates_missing_dir(tmp_path):
    out = tmp_path / "nested" / "reports"
    path = report.write_report(out, {"seed": 1}, sample_records())
    assert path.exists()
    assert path.parent == out
