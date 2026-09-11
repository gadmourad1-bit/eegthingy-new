"""Bridge MIRepNet JSON test output to the bundled artifact-tool Excel exporter."""

from __future__ import annotations

import os
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _find_artifact_module():
    configured = os.environ.get("OAI_ARTIFACT_TOOL_MODULE")
    candidates = [Path(configured)] if configured else []
    candidates.extend((Path.home() / ".cache" / "codex-runtimes").glob(
        "*/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs"
    ))
    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise RuntimeError(
        "the bundled @oai/artifact-tool runtime was not found; set "
        "OAI_ARTIFACT_TOOL_MODULE to artifact_tool.mjs"
    )


def _find_node(artifact_module):
    configured = os.environ.get("RUNTIME_NODE")
    if configured and Path(configured).is_file():
        return str(Path(configured).resolve())
    for parent in artifact_module.parents:
        candidate = parent / "bin" / ("node.exe" if os.name == "nt" else "node")
        if candidate.is_file():
            return str(candidate)
    found = shutil.which("node")
    if found:
        return found
    raise RuntimeError("Node.js was not found; set RUNTIME_NODE to the Node executable")


def _run_exporter(script_name, json_path, xlsx_path, preview_path=None):
    module = _find_artifact_module()
    node = _find_node(module)
    command = [
        node,
        str(ROOT / "scripts" / script_name),
        str(module),
        str(json_path),
        str(xlsx_path),
    ]
    if preview_path:
        command.append(str(Path(preview_path).resolve()))
    subprocess.run(command, cwd=ROOT, check=True)
    return xlsx_path


def export_test_workbook(json_path, xlsx_path=None, preview_path=None):
    json_path = Path(json_path).resolve()
    xlsx_path = Path(xlsx_path or json_path.with_suffix(".xlsx")).resolve()
    return _run_exporter("export_mirepnet_excel.mjs", json_path, xlsx_path, preview_path)


def export_online_workbook(rows, xlsx_path, metadata=None):
    """Save every MIRepNet live decision window to a portable Excel workbook."""
    xlsx_path = Path(xlsx_path).resolve()
    json_path = xlsx_path.with_suffix(".json")
    windows = []
    for index, row in enumerate(rows, 1):
        windows.append({
            "window": index,
            "timestamp": row[0],
            "log_odds": float(row[1]),
            "adaptive_center": float(row[2]),
            "recentered_margin": float(row[3]),
            "prediction": int(row[4]),
            "confidence": float(row[5]),
            "committed_class": int(row[6]),
        })
    payload = {
        "format": "mirepnet-online-session-v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "metadata": metadata or {},
        "windows": windows,
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return _run_exporter("export_mirepnet_online_excel.mjs", json_path, xlsx_path)
