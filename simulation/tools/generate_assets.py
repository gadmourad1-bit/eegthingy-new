#!/usr/bin/env python3
"""Generate the flattened TIAGo (no-arm) URDF and copy its meshes.

Dev-time tool: processes the exact same xacro sources that the Gazebo/ROS 2
app uses (src/tiago_robot, src/pmb2_robot, src/pal_urdf_utils) with the same
arguments that scripts/run_tiago_gui.sh passes to the launch file:

    arm_type:=no-arm  ft_sensor:=no-ft-sensor  end_effector:=no-end-effector
    camera_model:=no-camera  (base_type=pmb2, laser_model=sick-571 defaults)

Output:
    assets/tiago/tiago_no_arm.urdf   - flattened URDF, package:// URIs
                                       rewritten to relative mesh paths,
                                       gazebo/ros2_control tags stripped
    assets/tiago/meshes/...          - every referenced mesh file

Run from pythonport/:  uv run python tools/generate_assets.py
"""

from __future__ import annotations

import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from xacrodoc import XacroDoc, packages

PORT_ROOT = Path(__file__).resolve().parent.parent
WS_SRC = PORT_ROOT.parent / "src"
ASSETS = PORT_ROOT / "assets" / "tiago"

XACRO_FILE = (
    WS_SRC / "tiago_robot" / "tiago_description" / "robots" / "tiago.urdf.xacro"
)

# Mirror of the arguments used by scripts/run_tiago_gui.sh
SUBARGS = {
    "base_type": "pmb2",
    "laser_model": "sick-571",
    "arm_type": "no-arm",
    "ft_sensor": "no-ft-sensor",
    "end_effector": "no-end-effector",
    "camera_model": "no-camera",
    "has_screen": "false",
    "use_sim_time": "true",
    "is_public_sim": "True",
    "namespace": "",
}

STRIP_TAGS = {"gazebo", "transmission", "ros2_control"}


def resolve_package_uri(uri: str) -> Path:
    assert uri.startswith("package://"), uri
    rel = uri[len("package://") :]
    pkg, _, path = rel.partition("/")
    pkg_root = Path(packages.get_path(pkg))
    return pkg_root / path


def main() -> None:
    packages.look_in([str(WS_SRC)])

    doc = XacroDoc.from_file(str(XACRO_FILE), subargs=SUBARGS, resolve_packages=False)
    urdf_text = doc.to_urdf_string()

    root = ET.fromstring(urdf_text)

    # Strip simulation/control tags that the game does not consume.
    for tag in STRIP_TAGS:
        for el in root.findall(tag):
            root.remove(el)

    # Copy every referenced mesh, rewriting package:// URIs to relative paths.
    ASSETS.mkdir(parents=True, exist_ok=True)
    (ASSETS / "meshes").mkdir(exist_ok=True)
    copied: dict[str, str] = {}
    for mesh in root.iter("mesh"):
        uri = mesh.get("filename", "")
        if not uri.startswith("package://"):
            continue
        if uri not in copied:
            src = resolve_package_uri(uri)
            if not src.is_file():
                sys.exit(f"mesh not found: {uri} -> {src}")
            rel = Path("meshes") / uri[len("package://") :]
            dst = ASSETS / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied[uri] = rel.as_posix()
        mesh.set("filename", copied[uri])

    out = ASSETS / "tiago_no_arm.urdf"
    ET.indent(root)
    ET.ElementTree(root).write(out, encoding="unicode", xml_declaration=True)

    n_links = len(root.findall("link"))
    n_joints = len(root.findall("joint"))
    print(f"wrote {out.relative_to(PORT_ROOT)}")
    print(f"  links={n_links} joints={n_joints} meshes copied={len(copied)}")
    for uri, rel in sorted(copied.items()):
        print(f"  {rel}")


if __name__ == "__main__":
    main()
