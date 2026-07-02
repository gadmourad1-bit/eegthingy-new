"""Load the generated TIAGo no-arm URDF into a Panda3D node tree.

Uses the exact meshes from the ROS 2 description packages (copied by
tools/generate_assets.py). Visual geometry only; STL meshes are loaded with
trimesh and converted to flat-shaded Panda3D geometry, colored with the URDF
material palette (pal_urdf_utils materials.urdf.xacro).

Movable joints exposed for animation:
  wheel_left_joint / wheel_right_joint (continuous) — spun from the drive
  command, torso_lift_joint (prismatic) and head joints are left at 0, the
  same configuration the Gazebo sim spawns with (no arm, untucked).
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import trimesh
from panda3d.core import (
    Geom,
    GeomNode,
    GeomTriangles,
    GeomVertexData,
    GeomVertexFormat,
    GeomVertexWriter,
    LMatrix3,
    LMatrix4,
    LQuaternion,
    LVector3,
    Material,
    NodePath,
    TransformState,
)

ASSETS_DIR = Path(__file__).resolve().parent.parent.parent / "assets" / "tiago"
URDF_FILE = ASSETS_DIR / "tiago_no_arm.urdf"


def rpy_to_mat3(r: float, p: float, y: float) -> LMatrix3:
    """URDF fixed-axis RPY: R = Rz(y) @ Ry(p) @ Rx(r)."""
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    m = np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )
    # Panda3D matrices are row-vector convention (v' = v * M): store transpose.
    return LMatrix3(*m.T.flatten().tolist())


def make_transform(xyz: tuple[float, float, float], rpy: tuple[float, float, float]) -> TransformState:
    m3 = rpy_to_mat3(*rpy)
    m4 = LMatrix4(m3, LVector3(*xyz))
    return TransformState.makeMat(m4)


def _parse_floats(s: str | None, default: tuple[float, ...]) -> tuple[float, ...]:
    if not s:
        return default
    return tuple(float(v) for v in s.split())


@dataclass
class UrdfJoint:
    name: str
    jtype: str
    parent: str
    child: str
    xyz: tuple[float, float, float]
    rpy: tuple[float, float, float]
    axis: tuple[float, float, float]


@dataclass
class RobotModel:
    root: NodePath
    joints: dict[str, "JointHandle"] = field(default_factory=dict)


class JointHandle:
    """Rotates/translates a child link about its URDF joint axis."""

    def __init__(self, np_rot: NodePath, axis: tuple[float, float, float], jtype: str):
        self.np = np_rot
        self.axis = LVector3(*axis)
        if self.axis.length() > 0:
            self.axis.normalize()
        self.jtype = jtype
        self.position = 0.0

    def set_position(self, q: float) -> None:
        self.position = q
        if self.jtype in ("revolute", "continuous"):
            quat = LQuaternion()
            quat.setFromAxisAngleRad(q, self.axis)
            self.np.setQuat(quat)
        elif self.jtype == "prismatic":
            self.np.setPos(self.axis * q)


def _mesh_to_geomnode(path: Path, name: str) -> GeomNode:
    """STL -> flat-shaded Panda3D GeomNode (per-face normals, like Gazebo)."""
    mesh = trimesh.load(str(path), force="mesh")
    tris = np.asarray(mesh.faces, dtype=np.int64)
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    fnorm = np.asarray(mesh.face_normals, dtype=np.float64)

    # Expand to per-face vertices for crisp flat shading.
    v = verts[tris.reshape(-1)]                       # (F*3, 3)
    n = np.repeat(fnorm, 3, axis=0)                   # (F*3, 3)

    fmt = GeomVertexFormat.getV3n3()
    vdata = GeomVertexData(name, fmt, Geom.UHStatic)
    vdata.setNumRows(len(v))
    vw = GeomVertexWriter(vdata, "vertex")
    nw = GeomVertexWriter(vdata, "normal")
    for (vx, vy, vz), (nx, ny, nz) in zip(v, n):
        vw.addData3(vx, vy, vz)
        nw.addData3(nx, ny, nz)

    prim = GeomTriangles(Geom.UHStatic)
    prim.reserveNumVertices(len(v))
    for i in range(0, len(v), 3):
        prim.addVertices(i, i + 1, i + 2)
    prim.closePrimitive()

    geom = Geom(vdata)
    geom.addPrimitive(prim)
    node = GeomNode(name)
    node.addGeom(geom)
    return node


def _box_geomnode(sx: float, sy: float, sz: float, name: str) -> GeomNode:
    hx, hy, hz = sx / 2.0, sy / 2.0, sz / 2.0
    corners = [
        (-hx, -hy, -hz), (hx, -hy, -hz), (hx, hy, -hz), (-hx, hy, -hz),
        (-hx, -hy, hz), (hx, -hy, hz), (hx, hy, hz), (-hx, hy, hz),
    ]
    faces = [
        ((0, 3, 2, 1), (0, 0, -1)),
        ((4, 5, 6, 7), (0, 0, 1)),
        ((0, 1, 5, 4), (0, -1, 0)),
        ((2, 3, 7, 6), (0, 1, 0)),
        ((1, 2, 6, 5), (1, 0, 0)),
        ((3, 0, 4, 7), (-1, 0, 0)),
    ]
    fmt = GeomVertexFormat.getV3n3()
    vdata = GeomVertexData(name, fmt, Geom.UHStatic)
    vw = GeomVertexWriter(vdata, "vertex")
    nw = GeomVertexWriter(vdata, "normal")
    prim = GeomTriangles(Geom.UHStatic)
    idx = 0
    for quad, normal in faces:
        for qi in quad:
            vw.addData3(*corners[qi])
            nw.addData3(*normal)
        prim.addVertices(idx, idx + 1, idx + 2)
        prim.addVertices(idx, idx + 2, idx + 3)
        idx += 4
    prim.closePrimitive()
    geom = Geom(vdata)
    geom.addPrimitive(prim)
    node = GeomNode(name)
    node.addGeom(geom)
    return node


def _cylinder_geomnode(radius: float, length: float, name: str, segments: int = 32) -> GeomNode:
    fmt = GeomVertexFormat.getV3n3()
    vdata = GeomVertexData(name, fmt, Geom.UHStatic)
    vw = GeomVertexWriter(vdata, "vertex")
    nw = GeomVertexWriter(vdata, "normal")
    prim = GeomTriangles(Geom.UHStatic)
    hz = length / 2.0
    idx = 0
    for i in range(segments):
        a0 = 2 * math.pi * i / segments
        a1 = 2 * math.pi * (i + 1) / segments
        p0 = (radius * math.cos(a0), radius * math.sin(a0))
        p1 = (radius * math.cos(a1), radius * math.sin(a1))
        n0 = (math.cos(a0), math.sin(a0), 0)
        n1 = (math.cos(a1), math.sin(a1), 0)
        # side quad
        for (px, py), nrm, z in (
            (p0, n0, -hz), (p1, n1, -hz), (p1, n1, hz), (p0, n0, hz),
        ):
            vw.addData3(px, py, z)
            nw.addData3(*nrm)
        prim.addVertices(idx, idx + 1, idx + 2)
        prim.addVertices(idx, idx + 2, idx + 3)
        idx += 4
        # caps
        for z, nz, flip in ((hz, 1, False), (-hz, -1, True)):
            tri = [(0.0, 0.0), p0, p1] if not flip else [(0.0, 0.0), p1, p0]
            for px, py in tri:
                vw.addData3(px, py, z)
                nw.addData3(0, 0, nz)
            prim.addVertices(idx, idx + 1, idx + 2)
            idx += 3
    prim.closePrimitive()
    geom = Geom(vdata)
    geom.addPrimitive(prim)
    node = GeomNode(name)
    node.addGeom(geom)
    return node


class UrdfLoader:
    def __init__(self, urdf_path: Path = URDF_FILE):
        self.urdf_path = urdf_path
        self.assets_dir = urdf_path.parent
        self._geom_cache: dict[str, GeomNode] = {}

    def load(self, parent: NodePath) -> RobotModel:
        root_el = ET.parse(self.urdf_path).getroot()

        materials: dict[str, tuple[float, float, float, float]] = {}
        for mat in root_el.findall("material"):
            color = mat.find("color")
            if color is not None:
                rgba = _parse_floats(color.get("rgba"), (1, 1, 1, 1))
                materials[mat.get("name", "")] = rgba  # type: ignore[assignment]

        links = {link.get("name"): link for link in root_el.findall("link")}
        joints: list[UrdfJoint] = []
        for j in root_el.findall("joint"):
            origin = j.find("origin")
            axis_el = j.find("axis")
            joints.append(
                UrdfJoint(
                    name=j.get("name", ""),
                    jtype=j.get("type", "fixed"),
                    parent=j.find("parent").get("link"),  # type: ignore[union-attr]
                    child=j.find("child").get("link"),  # type: ignore[union-attr]
                    xyz=_parse_floats(origin.get("xyz") if origin is not None else None, (0, 0, 0)),  # type: ignore[arg-type]
                    rpy=_parse_floats(origin.get("rpy") if origin is not None else None, (0, 0, 0)),  # type: ignore[arg-type]
                    axis=_parse_floats(axis_el.get("xyz") if axis_el is not None else None, (0, 0, 1)),  # type: ignore[arg-type]
                )
            )

        children_of: dict[str, list[UrdfJoint]] = {}
        child_links = set()
        for j in joints:
            children_of.setdefault(j.parent, []).append(j)
            child_links.add(j.child)
        roots = [name for name in links if name not in child_links]
        if len(roots) != 1:
            raise ValueError(f"expected one root link, got {roots}")

        model = RobotModel(root=parent.attachNewNode("tiago"))
        link_nps: dict[str, NodePath] = {roots[0]: model.root.attachNewNode(roots[0])}

        # Breadth-first attach.
        queue = [roots[0]]
        while queue:
            link_name = queue.pop(0)
            link_np = link_nps[link_name]
            self._add_link_visuals(link_np, links[link_name], materials)
            for j in children_of.get(link_name, []):
                joint_np = link_np.attachNewNode(j.name)
                joint_np.setTransform(make_transform(j.xyz, j.rpy))
                if j.jtype in ("revolute", "continuous", "prismatic"):
                    motion_np = joint_np.attachNewNode(j.name + "_motion")
                    model.joints[j.name] = JointHandle(motion_np, j.axis, j.jtype)
                    child_np = motion_np.attachNewNode(j.child)
                else:
                    child_np = joint_np.attachNewNode(j.child)
                link_nps[j.child] = child_np
                queue.append(j.child)

        return model

    # ------------------------------------------------------------------
    def _add_link_visuals(
        self,
        link_np: NodePath,
        link_el: ET.Element,
        materials: dict[str, tuple[float, float, float, float]],
    ) -> None:
        for i, vis in enumerate(link_el.findall("visual")):
            geo = vis.find("geometry")
            if geo is None or len(geo) == 0:
                continue
            shape = geo[0]
            name = f"{link_el.get('name')}_vis{i}"

            node = None
            if shape.tag == "mesh":
                rel = shape.get("filename", "")
                if rel.lower().endswith(".stl"):
                    key = rel
                    if key not in self._geom_cache:
                        self._geom_cache[key] = _mesh_to_geomnode(self.assets_dir / rel, key)
                    node = self._geom_cache[key]
                else:
                    continue  # collision-only DAE meshes are not visuals
            elif shape.tag == "box":
                sx, sy, sz = _parse_floats(shape.get("size"), (0.1, 0.1, 0.1))
                node = _box_geomnode(sx, sy, sz, name)
            elif shape.tag == "cylinder":
                node = _cylinder_geomnode(
                    float(shape.get("radius", 0.05)), float(shape.get("length", 0.1)), name
                )
            if node is None:
                continue

            vis_np = link_np.attachNewNode(name)
            # Cached GeomNodes are copied per use (Geom data is still shared
            # internally; copying avoids multi-parent scene-graph warnings).
            NodePath(node).copyTo(vis_np)

            origin = vis.find("origin")
            xyz = _parse_floats(origin.get("xyz") if origin is not None else None, (0, 0, 0))
            rpy = _parse_floats(origin.get("rpy") if origin is not None else None, (0, 0, 0))
            vis_np.setTransform(make_transform(xyz, rpy))  # type: ignore[arg-type]

            rgba = (0.8, 0.8, 0.8, 1.0)
            mat_el = vis.find("material")
            if mat_el is not None:
                inline = mat_el.find("color")
                if inline is not None:
                    rgba = _parse_floats(inline.get("rgba"), rgba)  # type: ignore[assignment]
                else:
                    rgba = materials.get(mat_el.get("name", ""), rgba)

            m = Material()
            m.setDiffuse((*rgba[:3], rgba[3]))
            m.setAmbient((*rgba[:3], rgba[3]))
            m.setSpecular((0.3, 0.3, 0.3, 1))
            m.setShininess(32)
            vis_np.setMaterial(m, 1)
            vis_np.setColor(*rgba)
