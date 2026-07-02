"""Entry point: ``uv run tiago-maze`` or ``python -m tiago_maze``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .params import GameParams


class _HelpOnErrorParser(argparse.ArgumentParser):
    """Print full help on any parse error (unknown arg, bad value, ...)."""

    def error(self, message: str):  # noqa: D401
        sys.stderr.write(f"error: {message}\n\n")
        self.print_help(sys.stderr)
        sys.exit(2)


def build_parser() -> argparse.ArgumentParser:
    ap = _HelpOnErrorParser(
        prog="tiago-maze",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "TIAGo websocket maze game — Python port of the Gazebo/ROS 2 "
            "wall-course sim (safe_wall_teleop_webSocket.py). The robot drives "
            "forward, stops at walls, and turns left/right on websocket (or "
            "arrow-key) decisions through a randomized 40-corner maze."
        ),
        epilog="Running with no options shows this help. To start the game with "
               "defaults pass any option, e.g. `tiago-maze --view third`. Any "
               "unknown argument also prints this help. See README.md for details.",
    )
    ap.add_argument("--seed", type=int, default=None, help="maze seed (random if omitted)")
    ap.add_argument("--turns", type=int, default=40, help="number of left/right turn points")
    ap.add_argument("--view", choices=["third", "first", "top"], default="third",
                    help="initial camera: third-person, first-person, or top-down "
                         "(cycle in-game with [v])")
    ap.add_argument("--fps", type=int, default=60,
                    help="frame-rate cap (physics is fixed-step, so this only "
                         "limits rendering)")
    ap.add_argument("--ws-url", default="ws://127.0.0.1:8765", help="decision server URL")
    ap.add_argument("--no-ws", action="store_true", help="disable the websocket client")
    ap.add_argument("--no-manual-keys", action="store_true",
                    help="disable arrow-key decisions (websocket only)")
    ap.add_argument("--quiet", action="store_true", help="suppress the node-style console log")
    ap.add_argument("--no-report", action="store_true",
                    help="do not write the CSV run report")
    ap.add_argument("--no-trace", action="store_true",
                    help="do not write the NPZ pose trace")
    ap.add_argument("--report-dir", default="reports",
                    help="directory for CSV reports and NPZ traces")
    ap.add_argument("--replay", default=None, metavar="TRACE.npz",
                    help="replay a recorded NPZ pose trace instead of running live")
    ap.add_argument("--subject", default="", metavar="ID",
                    help="subject id, recorded in the report/trace metadata and filename")
    ap.add_argument("--test", default="", metavar="ID",
                    help="test id, recorded in the report/trace metadata and filename")
    ap.add_argument("--meta", default=None, metavar="JSON",
                    help='extra run metadata saved into the CSV summary; a JSON '
                         'object (e.g. \'{"subject":"S01","condition":"A"}\') or '
                         '@path.json to read it from a file')
    # Controller parameter overrides (same names as the ROS 2 parameters).
    ap.add_argument("--safe-dist", type=float, default=0.8, help="stop distance (m)")
    ap.add_argument("--forward-speed", type=float, default=0.60, help="drive speed (m/s)")
    ap.add_argument("--turn-speed", type=float, default=0.75, help="turn rate (rad/s)")
    ap.add_argument("--front-half-angle-deg", type=float, default=15.0,
                    help="front laser cone half-angle (deg)")
    ap.add_argument("--yaw-tol-deg", type=float, default=2.0,
                    help="turn completion tolerance (deg)")
    ap.add_argument("--heading-kp", type=float, default=2.0,
                    help="forward heading-hold gain (rad/s per rad; 0 disables, "
                         "reverting to the original straight-line drive)")
    # Test/CI hooks.
    ap.add_argument("--offscreen", action="store_true", help="render offscreen")
    ap.add_argument("--frames", type=int, default=None, help="exit after N frames")
    ap.add_argument("--screenshot", default=None, help="write a screenshot on exit")
    return ap


def _parse_meta(raw: str | None) -> dict:
    """Parse the --meta value: a JSON object string, or @path.json."""
    if raw is None:
        return {}
    text = raw
    if raw.startswith("@"):
        text = Path(raw[1:]).read_text()
    obj = json.loads(text)  # may raise json.JSONDecodeError
    if not isinstance(obj, dict):
        raise ValueError("metadata must be a JSON object (e.g. {\"k\": \"v\"})")
    return obj


def build_params(argv: list[str] | None = None) -> GameParams:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        extra_meta = _parse_meta(args.meta)
    except (OSError, ValueError, json.JSONDecodeError) as e:
        parser.error(f"--meta: {e}")   # prints full help, exits 2

    p = GameParams()
    p.extra_meta = extra_meta
    p.maze.seed = args.seed
    p.maze.n_turns = args.turns
    p.view = args.view
    p.fps = args.fps
    p.control.ws_url = args.ws_url
    p.control.enable_ws = not args.no_ws
    p.control.safe_dist = args.safe_dist
    p.control.forward_speed = args.forward_speed
    p.control.turn_speed = args.turn_speed
    p.control.front_half_angle_deg = args.front_half_angle_deg
    p.control.yaw_tol_deg = args.yaw_tol_deg
    p.control.heading_kp = args.heading_kp
    p.subject_id = args.subject
    p.test_id = args.test
    p.manual_keys = not args.no_manual_keys
    p.report_enabled = not args.no_report
    p.record_trace = not args.no_trace
    p.report_dir = args.report_dir
    p.replay = args.replay
    p.offscreen = args.offscreen
    p.max_frames = args.frames
    p.screenshot = args.screenshot
    p.quiet = args.quiet
    return p


def main(argv: list[str] | None = None) -> None:
    effective = sys.argv[1:] if argv is None else argv
    if not effective:
        # No options given: show the help menu instead of launching.
        build_parser().print_help()
        return

    params = build_params(argv)

    from .game import MazeGame  # deferred: importing panda3d opens a display

    game = MazeGame(params)
    game.run()


if __name__ == "__main__":
    main()
