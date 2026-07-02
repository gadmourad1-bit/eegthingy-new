"""TIAGo maze game (Panda3D).

Replicates the Gazebo wall-course setup: the exact TIAGo no-arm robot drives
forward, stops at walls, waits for a left/right decision over websocket and
performs 90-degree in-place turns — inside a randomized corridor maze with a
fixed number of decision corners, watched by a smooth follow camera.
"""

from __future__ import annotations

import json
import math
import random
import time

import numpy as np
from direct.gui.OnscreenText import OnscreenText
from direct.showbase.ShowBase import ShowBase
from panda3d.core import (
    AmbientLight,
    AntialiasAttrib,
    CardMaker,
    ClockObject,
    DirectionalLight,
    Geom,
    GeomNode,
    GeomTriangles,
    GeomVertexData,
    GeomVertexFormat,
    GeomVertexWriter,
    LineSegs,
    LVector3,
    Material,
    NodePath,
    PNMImage,
    SamplerState,
    TextNode,
    Texture,
    TransparencyAttrib,
    loadPrcFileData,
)

from . import lidar as lidar_mod
from . import maze as maze_mod
from .controller import Logger, SafeWallTeleop, wrap_pi
from .params import GameParams
from .robot_model import UrdfLoader, _box_geomnode
from .ws_client import DecisionClient


def _textured_wall_node(sx: float, sy: float, sz: float, name: str) -> GeomNode:
    """Axis-aligned wall box with UVs: V = height fraction (0 floor, 1 top),
    U = horizontal position in metres. Lets the hospital wall texture put its
    skirting / handrail band at a fixed height on every wall."""
    hx, hy, hz = sx / 2.0, sy / 2.0, sz / 2.0
    corners = [
        (-hx, -hy, -hz), (hx, -hy, -hz), (hx, hy, -hz), (-hx, hy, -hz),
        (-hx, -hy, hz), (hx, -hy, hz), (hx, hy, hz), (-hx, hy, hz),
    ]
    # (quad indices, outward normal, horizontal u-axis in local coords)
    faces = [
        ((0, 3, 2, 1), (0, 0, -1), (1, 0)),
        ((4, 5, 6, 7), (0, 0, 1), (1, 0)),
        ((0, 1, 5, 4), (0, -1, 0), (1, 0)),
        ((2, 3, 7, 6), (0, 1, 0), (1, 0)),
        ((1, 2, 6, 5), (1, 0, 0), (0, 1)),
        ((3, 0, 4, 7), (-1, 0, 0), (0, 1)),
    ]
    fmt = GeomVertexFormat.getV3n3t2()
    vdata = GeomVertexData(name, fmt, Geom.UHStatic)
    vw = GeomVertexWriter(vdata, "vertex")
    nw = GeomVertexWriter(vdata, "normal")
    tw = GeomVertexWriter(vdata, "texcoord")
    prim = GeomTriangles(Geom.UHStatic)
    idx = 0
    for quad, normal, uaxis in faces:
        for ci in quad:
            x, y, z = corners[ci]
            vw.addData3(x, y, z)
            nw.addData3(*normal)
            tw.addData2(x * uaxis[0] + y * uaxis[1], (z + hz) / sz)
        prim.addVertices(idx, idx + 1, idx + 2)
        prim.addVertices(idx, idx + 2, idx + 3)
        idx += 4
    prim.closePrimitive()
    geom = Geom(vdata)
    geom.addPrimitive(prim)
    node = GeomNode(name)
    node.addGeom(geom)
    return node


class MazeGame(ShowBase):
    def __init__(self, params: GameParams | None = None):
        self.params = params or GameParams()
        p = self.params

        loadPrcFileData("", "window-title TIAGo Maze — websocket wall course")
        loadPrcFileData("", "win-size 1280 800")
        loadPrcFileData("", "sync-video true")
        if not p.offscreen:
            # Cap the frame rate so a 144/240 Hz display doesn't spin the render
            # loop needlessly. Physics is fixed-step regardless (see _update).
            loadPrcFileData("", "clock-mode limited")
            loadPrcFileData("", f"clock-frame-rate {int(p.fps)}")

        ShowBase.__init__(self, windowType="offscreen" if p.offscreen else None)
        self.disableMouse()
        self.setBackgroundColor(0.85, 0.86, 0.88)  # soft indoor grey
        self.render.setAntialias(AntialiasAttrib.MMultisample)
        self.render.setShaderAuto()

        # Pull the near clip plane in close: the corridor walls are only ~0.8 m
        # away, so the default ~1 m near plane clips them (in first-person the
        # camera would appear to "see through" the wall to the floor beyond).
        self.camLens.setNear(0.04)
        self.camLens.setFar(400.0)

        self.clock = ClockObject.getGlobalClock()
        self.logger = Logger(enabled=not p.quiet)

        # ---------------- replay trace (optional) ----------------
        # In replay mode the maze must match the recording, so pull the maze
        # parameters out of the trace metadata before generating it.
        self.replay_mode = p.replay is not None
        if self.replay_mode:
            from . import trace as trace_mod
            self._replay_samples, rmeta = trace_mod.load_trace(p.replay)
            p.maze.seed = int(rmeta.get("seed", p.maze.seed or 0))
            for key, cast in (("n_turns", int), ("cell", float),
                              ("wall_thickness", float), ("wall_height", float),
                              ("min_gap", int), ("max_gap", int)):
                if key in rmeta:
                    setattr(p.maze, key, cast(rmeta[key]))

        # ---------------- world ----------------
        self.maze = maze_mod.generate(p.maze)
        self.lidar = lidar_mod.Lidar(self.maze.walls, p.robot)
        self.world_np = NodePath("world")
        self.world_np.reparentTo(self.render)
        self._build_world()

        # ---------------- robot ----------------
        self.robot_model = UrdfLoader().load(self.render)
        self.x, self.y = self.maze.start_xy
        self.yaw = self.maze.start_yaw
        self.wheel_q = [0.0, 0.0]  # left, right joint angles

        # ---------------- shared state ----------------
        self.completed = False
        self._flash_until = 0.0
        self._frame_count = 0
        self._sim_time = 0.0          # deterministic accumulated physics time
        self._phys_acc = 0.0          # fixed-timestep integrator accumulator

        # ---------------- camera ----------------
        # Corridor-level follow camera: kept BELOW the 2.0 m wall tops so it
        # can never see over a wall into a neighbouring corridor.
        self.view_mode = p.view if p.view in ("third", "first", "top") else "third"
        self.cam_dist = 3.0
        self.cam_height = 1.55
        self._cam_pos: LVector3 | None = None  # snapped on first frame

        if self.replay_mode:
            self._setup_replay()
        else:
            self._setup_live()

        self._build_lights()
        self._build_hud()
        self._build_minimap()
        self._bind_keys()

        self.taskMgr.add(
            self._update_replay if self.replay_mode else self._update, "game-update"
        )

    # ------------------------------------------------------------------
    def _setup_live(self) -> None:
        p = self.params

        # Controller (ported node).
        self.controller = SafeWallTeleop(p.control, logger=self.logger)
        self.controller.on_turn_complete = self._on_turn_complete
        self.controller.on_state_change = self._on_state_change
        self.controller.log.info("Safe wall teleop node started.")

        self.ws: DecisionClient | None = None
        if p.control.enable_ws:
            self.ws = DecisionClient(self.controller, p.control.ws_url)
            self.ws.start()

        # Gameplay state.
        self.progress = 0          # correct turns completed
        self.wrong_turns = 0
        self.decisions = 0
        self.start_time = time.monotonic()
        self.finish_time: float | None = None
        self._control_acc = 0.0
        self._status_acc = 0.0
        self._trace_acc = 0.0

        # Decision timing / report.
        self._wait_start: float | None = None
        self.decision_times: list[float] = []
        self.last_decision_time: float | None = None
        self.decision_records: list[dict] = []   # full per-decision rows (CSV)
        self._pending_record: dict | None = None
        self._report_written = False
        self._trace_written = False

        # Pose trace (NPZ). Record the initial pose at t=0.
        self._traj: dict[str, list[float]] = {k: [] for k in ("t", "x", "y", "yaw", "pitch")}
        self._sample_trace()

        if not p.offscreen:
            # Write a (possibly partial) report + trace if the window is closed.
            self.exitFunc = self._finalize

    def _setup_replay(self) -> None:
        self.controller = None
        self.ws = None
        s = self._replay_samples
        self._rt = np.asarray(s.get("t", []), dtype=np.float64)
        self._rx = np.asarray(s.get("x", []), dtype=np.float64)
        self._ry = np.asarray(s.get("y", []), dtype=np.float64)
        self._ryaw = np.asarray(s.get("yaw", []), dtype=np.float64)
        self._replay_time = 0.0
        self._replay_paused = False
        self._replay_len = float(self._rt[-1]) if len(self._rt) else 0.0
        if len(self._rt):
            self.x = float(self._rx[0])
            self.y = float(self._ry[0])
            self.yaw = float(self._ryaw[0])
        self.logger.info(
            f"Replay: {len(self._rt)} samples, {self._replay_len:.1f}s, "
            f"seed {self.maze.seed}"
        )

    # ==================================================================
    # World construction
    # ==================================================================

    def _build_world(self) -> None:
        # Ground: hospital linoleum tile. Snap the extent to the cell grid so
        # tile grout lines up with corridor walls.
        c = self.params.maze.cell
        xmin, ymin, xmax, ymax = self._maze_extent(margin=8.0)
        xmin = c / 2 - c * math.ceil((c / 2 - xmin) / c)
        ymin = c / 2 - c * math.ceil((c / 2 - ymin) / c)
        xmax = c / 2 + c * math.ceil((xmax - c / 2) / c)
        ymax = c / 2 + c * math.ceil((ymax - c / 2) / c)
        cm = CardMaker("ground")
        cm.setFrame(xmin, xmax, ymin, ymax)
        ground = self.world_np.attachNewNode(cm.generate())
        ground.setP(-90)
        ground.setTexture(self._floor_texture())
        # 4 floor tiles per corridor cell -> ~0.4 m tiles, grout on cell edges.
        ground.setTexScale(ground.findTextureStage("default"),
                           (xmax - xmin) / c, (ymax - ymin) / c)
        fmat = Material()
        fmat.setDiffuse((1, 1, 1, 1))
        fmat.setAmbient((1, 1, 1, 1))
        fmat.setSpecular((0.05, 0.05, 0.06, 1))
        fmat.setShininess(4)
        ground.setMaterial(fmat, 1)

        # Walls: hospital corridor panels (skirting + handrail band + white).
        walls_np = self.world_np.attachNewNode("walls")
        wtex = self._wall_texture()
        # Merge collinear runs so straight walls have no overlapping coplanar
        # faces (those z-fight); the lidar still uses the per-edge self.maze.walls.
        render_walls = maze_mod.merge_collinear_walls(
            self.maze.walls, self.params.maze.wall_thickness)
        for i, w in enumerate(render_walls):
            node = _textured_wall_node(w.sx, w.sy, w.height, f"wall{i}")
            np_w = walls_np.attachNewNode(node)
            np_w.setPos(w.cx, w.cy, w.height / 2.0)
        wmat = Material()
        wmat.setDiffuse((1, 1, 1, 1))
        wmat.setAmbient((1, 1, 1, 1))
        wmat.setSpecular((0.08, 0.08, 0.09, 1))
        wmat.setShininess(8)
        walls_np.setMaterial(wmat, 1)
        walls_np.setTexture(wtex)
        walls_np.setTexScale(walls_np.findTextureStage("default"), 1.0, 1.0)
        walls_np.flattenStrong()

        # Ceiling + recessed light fixtures (enclose the corridor).
        self._build_ceiling(xmin, ymin, xmax, ymax)

        # Start (blue) and goal (green) floor discs.
        for (gx, gy), rgba, name in (
            (self.maze.start_xy, (0.2, 0.4, 0.9, 1), "start-disc"),
            (self.maze.goal_xy, (0.1, 0.8, 0.1, 1), "goal-disc"),
        ):
            disc = self._make_disc(0.45, rgba, name)
            disc.reparentTo(self.world_np)
            disc.setPos(gx, gy, 0.012)

    # ------------------------------------------------------------------
    def _floor_texture(self) -> Texture:
        """Light linoleum tile: 4x4 tiles with subtle grout + speckle."""
        n = 128
        img = PNMImage(n, n)
        rng = random.Random(7)
        tiles = 4
        grout = max(1, n // 64)
        for yy in range(n):
            for xx in range(n):
                on_grout = (xx % (n // tiles)) < grout or (yy % (n // tiles)) < grout
                if on_grout:
                    r, g, b = 0.66, 0.68, 0.71
                else:
                    s = (rng.random() - 0.5) * 0.03
                    r, g, b = 0.82 + s, 0.83 + s, 0.85 + s
                img.setXel(xx, yy, r, g, b)
        tex = Texture("floor")
        tex.load(img)
        tex.setWrapU(Texture.WM_repeat)
        tex.setWrapV(Texture.WM_repeat)
        tex.setMinfilter(SamplerState.FT_linear_mipmap_linear)
        tex.setAnisotropicDegree(4)
        return tex

    def _wall_texture(self) -> Texture:
        """Vertical hospital-wall profile (V = height fraction 0..1):
        dark skirting, muted handrail band, white above, faint panel seams."""
        w, h = 24, 256
        img = PNMImage(w, h)
        # bands as (v_start, v_end, (r,g,b))
        bands = [
            (0.00, 0.075, (0.42, 0.45, 0.49)),   # skirting / base
            (0.075, 0.095, (0.28, 0.30, 0.33)),  # skirting trim
            (0.095, 0.46, (0.87, 0.89, 0.90)),   # lower wall
            (0.46, 0.475, (0.30, 0.42, 0.48)),   # trim under band
            (0.475, 0.55, (0.36, 0.60, 0.62)),   # handrail accent band (teal)
            (0.55, 0.565, (0.30, 0.42, 0.48)),   # trim over band
            (0.565, 1.00, (0.91, 0.93, 0.95)),   # upper wall
        ]

        def color_at(v: float):
            for v0, v1, col in bands:
                if v0 <= v < v1:
                    return col
            return bands[-1][2]

        for yy in range(h):
            # PNMImage row 0 is the image top; texture V=0 samples the bottom
            # row, so invert to keep the skirting (v_intended 0) on the floor.
            v = 1.0 - yy / (h - 1)
            r, g, b = color_at(v)
            for xx in range(w):
                # faint vertical panel seam once per metre (U repeats per metre)
                seam = 0.90 if xx < 1 else 1.0
                img.setXel(xx, yy, r * seam, g * seam, b * seam)
        tex = Texture("wall")
        tex.load(img)
        tex.setWrapU(Texture.WM_repeat)
        tex.setWrapV(Texture.WM_clamp)
        tex.setMinfilter(SamplerState.FT_linear_mipmap_linear)
        tex.setMagfilter(SamplerState.FT_linear)
        tex.setAnisotropicDegree(4)
        return tex

    def _build_ceiling(self, xmin: float, ymin: float, xmax: float, ymax: float) -> None:
        """Downward-facing ceiling plane (acoustic tile) plus emissive
        recessed light panels running along the corridor centre-line. Stored
        on self.ceiling_np so it can be hidden in the top-down view."""
        c = self.params.maze.cell
        h = self.params.maze.wall_height
        self.ceiling_np = self.world_np.attachNewNode("ceiling")

        cm = CardMaker("ceiling-plane")
        cm.setFrame(xmin, xmax, ymin, ymax)
        ceil = self.ceiling_np.attachNewNode(cm.generate())
        ceil.setP(90)          # lay flat with the visible face pointing down
        ceil.setZ(h)
        ceil.setTwoSided(True)  # visible from below regardless of winding
        ceil.setTexture(self._ceiling_texture())
        tile = c / 2.0          # ~2 ceiling tiles per corridor cell
        ceil.setTexScale(ceil.findTextureStage("default"),
                         (xmax - xmin) / tile, (ymax - ymin) / tile)
        cmat = Material()
        cmat.setDiffuse((1, 1, 1, 1))
        cmat.setAmbient((1, 1, 1, 1))
        cmat.setSpecular((0.02, 0.02, 0.02, 1))
        cmat.setShininess(2)
        ceil.setMaterial(cmat, 1)

        # Recessed fluorescent panels at each corridor cell centre.
        panels = self.ceiling_np.attachNewNode("light-panels")
        for i, cell in enumerate(self.maze.cells):
            cx, cy = self.maze.cell_center(cell)
            node = _box_geomnode(0.7, 0.7, 0.05, f"lightpanel{i}")
            pn = panels.attachNewNode(node)
            pn.setPos(cx, cy, h - 0.06)  # just below the ceiling, recessed
        lmat = Material()
        lmat.setEmission((0.95, 0.97, 1.0, 1))  # self-illuminated -> glows
        lmat.setDiffuse((0.85, 0.88, 0.92, 1))
        lmat.setAmbient((0.85, 0.88, 0.92, 1))
        panels.setMaterial(lmat, 1)
        panels.setColor(0.97, 0.98, 1.0, 1)
        panels.flattenStrong()

    def _ceiling_texture(self) -> Texture:
        """Light acoustic-tile ceiling: pale panels with thin seams."""
        n = 128
        img = PNMImage(n, n)
        tiles = 2
        seam = max(1, n // 64)
        for yy in range(n):
            for xx in range(n):
                on_seam = (xx % (n // tiles)) < seam or (yy % (n // tiles)) < seam
                if on_seam:
                    r, g, b = 0.70, 0.71, 0.74
                else:
                    r, g, b = 0.85, 0.86, 0.88
                img.setXel(xx, yy, r, g, b)
        tex = Texture("ceiling")
        tex.load(img)
        tex.setWrapU(Texture.WM_repeat)
        tex.setWrapV(Texture.WM_repeat)
        tex.setMinfilter(SamplerState.FT_linear_mipmap_linear)
        tex.setAnisotropicDegree(4)
        return tex

    def _make_disc(self, radius: float, rgba, name: str, segments: int = 40) -> NodePath:
        ls = LineSegs(name)
        ls.setThickness(4.0)
        ls.setColor(*rgba)
        for k in range(segments + 1):
            a = 2 * math.pi * k / segments
            fn = ls.moveTo if k == 0 else ls.drawTo
            fn(radius * math.cos(a), radius * math.sin(a), 0)
        np_ = NodePath(ls.create())
        return np_

    def _maze_extent(self, margin: float = 0.0) -> tuple[float, float, float, float]:
        xs = [w.cx for w in self.maze.walls]
        ys = [w.cy for w in self.maze.walls]
        return (min(xs) - margin, min(ys) - margin, max(xs) + margin, max(ys) + margin)

    def _build_lights(self) -> None:
        # Bright, even, slightly cool — like overhead fluorescent lighting.
        alight = AmbientLight("ambient")
        alight.setColor((0.62, 0.63, 0.66, 1))
        self.render.setLight(self.render.attachNewNode(alight))

        # Main near-overhead light so both floor and wall faces stay bright
        # (kept modest so the white robot body doesn't blow out).
        key = DirectionalLight("key")
        key.setColor((0.58, 0.59, 0.62, 1))
        key_np = self.render.attachNewNode(key)
        key_np.setHpr(25, -62, 0)
        self.render.setLight(key_np)

        # Two shallow fills from opposite sides so no wall face goes dark.
        for name, hpr, col in (
            ("fill_l", (115, -18, 0), (0.30, 0.31, 0.34, 1)),
            ("fill_r", (-115, -18, 0), (0.28, 0.29, 0.33, 1)),
        ):
            fill = DirectionalLight(name)
            fill.setColor(col)
            fnp = self.render.attachNewNode(fill)
            fnp.setHpr(*hpr)
            self.render.setLight(fnp)

        # Upward "bounce" fill so the (downward-facing) ceiling underside is
        # softly shaded instead of flat-dark now that the corridor is enclosed.
        bounce = DirectionalLight("bounce")
        bounce.setColor((0.30, 0.31, 0.35, 1))
        bnp = self.render.attachNewNode(bounce)
        bnp.setHpr(0, 72, 0)
        self.render.setLight(bnp)

    # ==================================================================
    # HUD / minimap
    # ==================================================================

    def _build_hud(self) -> None:
        self.hud_status = OnscreenText(
            text="", parent=self.a2dTopLeft, align=TextNode.ALeft,
            pos=(0.04, -0.08), scale=0.042, fg=(1, 1, 1, 1),
            shadow=(0, 0, 0, 0.8), mayChange=True, font=None,
        )
        self.hud_prompt = OnscreenText(
            text="", parent=self.aspect2d, align=TextNode.ACenter,
            pos=(0, 0.72), scale=0.09, fg=(1, 0.95, 0.4, 1),
            shadow=(0, 0, 0, 0.9), mayChange=True,
        )
        self.hud_flash = OnscreenText(
            text="", parent=self.aspect2d, align=TextNode.ACenter,
            pos=(0, 0.58), scale=0.07, fg=(1, 1, 1, 1),
            shadow=(0, 0, 0, 0.9), mayChange=True,
        )
        help_text = (
            "[space] pause   [r] restart   [v] view   [esc] quit"
            if self.replay_mode
            else "[left]/[right] decide   [v] view   [r] new maze   [esc] quit"
        )
        self.hud_help = OnscreenText(
            text=help_text,
            parent=self.a2dBottomCenter, align=TextNode.ACenter,
            pos=(0, 0.05), scale=0.038, fg=(1, 1, 1, 0.75),
            shadow=(0, 0, 0, 0.8), mayChange=False,
        )

    def _build_minimap(self) -> None:
        if hasattr(self, "minimap_np"):
            self.minimap_np.removeNode()
        # Anchor to the top-right corner so it stays out of the play area.
        self.minimap_np = self.a2dTopRight.attachNewNode("minimap")

        xmin, ymin, xmax, ymax = self._maze_extent()
        span = max(xmax - xmin, ymax - ymin, 1e-6)
        half = 0.24                      # half-size of the map panel
        self._mm_scale = (2 * half) / span
        self._mm_origin = ((xmin + xmax) / 2.0, (ymin + ymax) / 2.0)
        margin = 0.06
        self.minimap_np.setPos(-(half + margin), 0, -(half + margin))

        # Backing panel (contained, low-distraction).
        bg = CardMaker("mm-bg")
        pad = 0.035
        bg.setFrame(-half - pad, half + pad, -half - pad, half + pad)
        bg_np = self.minimap_np.attachNewNode(bg.generate())
        bg_np.setColor(0.07, 0.08, 0.10, 0.55)
        bg_np.setTransparency(TransparencyAttrib.MAlpha)

        ls = LineSegs("mm-walls")
        ls.setThickness(1.5)
        ls.setColor(0.85, 0.88, 0.95, 0.95)
        for w in self.maze.walls:
            x0, z0 = self._to_mm(w.cx - w.sx / 2, w.cy - w.sy / 2)
            x1, z1 = self._to_mm(w.cx + w.sx / 2, w.cy + w.sy / 2)
            # draw the long axis center-line of each wall
            if w.sx >= w.sy:
                ls.moveTo(x0, 0, (z0 + z1) / 2)
                ls.drawTo(x1, 0, (z0 + z1) / 2)
            else:
                ls.moveTo((x0 + x1) / 2, 0, z0)
                ls.drawTo((x0 + x1) / 2, 0, z1)
        self.minimap_np.attachNewNode(ls.create())

        sx, sz = self._to_mm(*self.maze.start_xy)
        start = LineSegs("mm-start")
        start.setThickness(6.0)
        start.setColor(0.3, 0.55, 1.0, 1)
        start.moveTo(sx - 0.006, 0, sz)
        start.drawTo(sx + 0.006, 0, sz)
        self.minimap_np.attachNewNode(start.create())

        gx, gz = self._to_mm(*self.maze.goal_xy)
        goal = LineSegs("mm-goal")
        goal.setThickness(6.0)
        goal.setColor(0.2, 1.0, 0.2, 1)
        goal.moveTo(gx - 0.006, 0, gz)
        goal.drawTo(gx + 0.006, 0, gz)
        self.minimap_np.attachNewNode(goal.create())

        self._mm_robot = self.minimap_np.attachNewNode("mm-robot")

    def _to_mm(self, x: float, y: float) -> tuple[float, float]:
        return ((x - self._mm_origin[0]) * self._mm_scale,
                (y - self._mm_origin[1]) * self._mm_scale)

    def _update_minimap(self) -> None:
        self._mm_robot.node().removeAllChildren()
        ls = LineSegs("mm-bot")
        ls.setThickness(5.0)
        ls.setColor(1.0, 0.35, 0.1, 1)
        px, pz = self._to_mm(self.x, self.y)
        hx = px + 0.035 * math.cos(self.yaw)
        hz = pz + 0.035 * math.sin(self.yaw)
        ls.moveTo(px, 0, pz)
        ls.drawTo(hx, 0, hz)
        self._mm_robot.attachNewNode(ls.create())

    # ==================================================================
    # Input
    # ==================================================================

    def _bind_keys(self) -> None:
        self.accept("escape", self.userExit)
        self.accept("q", self.userExit)
        self.accept("v", self._cycle_view)
        self.accept("wheel_up", self._zoom, [-0.4])
        self.accept("wheel_down", self._zoom, [+0.4])
        if self.replay_mode:
            self.accept("r", self._restart_replay)
            self.accept("space", self._toggle_pause)
        else:
            self.accept("r", self._new_maze)
            if self.params.manual_keys:
                self.accept("arrow_left", self._manual_decision, [1])
                self.accept("arrow_right", self._manual_decision, [2])

    def _manual_decision(self, val: int) -> None:
        if self.controller is not None:
            self.controller.submit_decision(val)

    def _cycle_view(self) -> None:
        order = ("third", "first", "top")
        self.view_mode = order[(order.index(self.view_mode) + 1) % len(order)]
        self._cam_pos = None  # re-snap smoothing on the new view

    def _restart_replay(self) -> None:
        self._replay_time = 0.0

    def _toggle_pause(self) -> None:
        self._replay_paused = not self._replay_paused

    def _zoom(self, d: float) -> None:
        self.cam_dist = min(12.0, max(1.6, self.cam_dist + d))

    def _new_maze(self) -> None:
        # Save the run we're leaving (if it had any decisions) before resetting.
        self._finalize()
        self.params.maze.seed = None  # fresh random maze
        self.maze = maze_mod.generate(self.params.maze)
        self.lidar.set_walls(self.maze.walls)
        self.world_np.removeNode()
        self.world_np = NodePath("world")
        self.world_np.reparentTo(self.render)
        self._build_world()
        self._build_minimap()
        self.x, self.y = self.maze.start_xy
        self.yaw = self.maze.start_yaw
        self.controller.set_state("FORWARD")
        self.controller.clear_for_next_cycle()
        self.controller.target_yaw = None
        self.progress = 0
        self.wrong_turns = 0
        self.decisions = 0
        self.completed = False
        self.finish_time = None
        self.start_time = time.monotonic()
        self._sim_time = 0.0
        self._phys_acc = 0.0
        self._control_acc = 0.0
        self._status_acc = 0.0
        self._trace_acc = 0.0
        self._wait_start = None
        self.decision_times = []
        self.last_decision_time = None
        self.decision_records = []
        self._pending_record = None
        self._report_written = False
        self._trace_written = False
        self._traj = {k: [] for k in ("t", "x", "y", "yaw", "pitch")}
        self._sample_trace()
        self._cam_pos = None  # snap the camera behind the new spawn
        self.logger.info(f"New maze generated. seed = {self.maze.seed}")

    def _sample_trace(self) -> None:
        self._traj["t"].append(self._sim_time)
        self._traj["x"].append(self.x)
        self._traj["y"].append(self.y)
        self._traj["yaw"].append(self.yaw)
        self._traj["pitch"].append(0.0)  # planar base; kept for 3D reconstruction

    # ==================================================================
    # Game events
    # ==================================================================

    def _on_state_change(self, old: str, new: str) -> None:
        now = time.monotonic()
        if new == "STOPPED_WAITING_EEG":
            # Start the decision timer when the robot stops at a wall.
            self._wait_start = now
        elif old == "STOPPED_WAITING_EEG" and new in ("TURN_L", "TURN_R"):
            # A decision was consumed: record how long it took to arrive and
            # open a per-decision record (finalized once the turn completes).
            if self._wait_start is not None:
                dt = now - self._wait_start
                self.last_decision_time = dt
                self.decision_times.append(dt)
                self._wait_start = None
                chosen = "LEFT" if new == "TURN_L" else "RIGHT"
                tp = self._nearest_turn_point()
                fd = self.controller.front_dist
                self._pending_record = {
                    "decision": len(self.decision_times),
                    "turn": (tp.index + 1) if tp else "",
                    "corner_x": round(tp.x, 3) if tp else "",
                    "corner_y": round(tp.y, 3) if tp else "",
                    "intended_dir": tp.direction if tp else "",
                    "chosen_dir": chosen,
                    "result": "",  # filled at turn completion
                    "wait_s": round(dt, 3),
                    "sim_time_s": round(now - self.start_time, 3),
                    "robot_x": round(self.x, 3),
                    "robot_y": round(self.y, 3),
                    "robot_yaw_deg": round(math.degrees(self.yaw), 1),
                    "front_m": round(fd, 3) if math.isfinite(fd) else "",
                }
                self.logger.info(
                    f"Decision {len(self.decision_times)} ({chosen}) took {dt:.2f} s"
                )

    def _on_turn_complete(self, direction: str) -> None:
        self.decisions += 1
        tp = self._nearest_turn_point()
        result = "unknown"
        if tp is not None:
            heading = round(self.yaw / (math.pi / 2)) % 4
            if heading == tp.out_heading:
                result = "correct"
                self.progress = max(self.progress, tp.index + 1)
                self._flash(
                    f"turn {tp.index + 1}/{len(self.maze.turn_points)}: {direction} - correct!",
                    (0.4, 1.0, 0.4, 1),
                )
            else:
                result = "wrong"
                self.wrong_turns += 1
                self._flash(f"wrong way! ({direction})", (1.0, 0.35, 0.3, 1))

        # Finalize the per-decision record opened in _on_state_change.
        if self._pending_record is not None:
            self._pending_record["result"] = result
            self.decision_records.append(self._pending_record)
            self._pending_record = None

    def _finalize(self, when=None) -> None:
        """Write the CSV report and NPZ pose trace (once each). Called on
        completion, on starting a new maze, and on window close."""
        if self.replay_mode:
            return
        from datetime import datetime
        when = when or datetime.now()
        self._write_report(when)
        self._write_trace(when)

    def _write_report(self, when=None) -> None:
        """Write the CSV run report (once). Includes the run's parameters."""
        if self._report_written or not self.params.report_enabled:
            return
        if not self.decision_records:
            return

        from datetime import datetime
        from . import report

        p = self.params
        waits = [r["wait_s"] for r in self.decision_records]
        elapsed = self.finish_time if self.finish_time is not None else self._sim_time
        when = when or datetime.now()
        meta = {
            "seed": self.maze.seed,
            "timestamp": when.strftime("%Y-%m-%d %H:%M:%S"),
            "completed": self.completed,
            # --- run parameters ---
            "forward_speed_mps": p.control.forward_speed,
            "turn_speed_radps": p.control.turn_speed,
            "safe_dist_m": p.control.safe_dist,
            "front_half_angle_deg": p.control.front_half_angle_deg,
            "yaw_tol_deg": p.control.yaw_tol_deg,
            "heading_kp_radps_per_rad": p.control.heading_kp,
            "view": self.view_mode,
            "fps_cap": p.fps,
            "phys_dt_s": p.phys_dt,
            "ws_enabled": p.control.enable_ws,
            "ws_url": p.control.ws_url,
            "manual_keys": p.manual_keys,
            # --- results ---
            "n_turns": len(self.maze.turn_points),
            "decisions": len(self.decision_records),
            "correct": sum(1 for r in self.decision_records if r["result"] == "correct"),
            "wrong_turns": self.wrong_turns,
            "total_time_s": round(elapsed, 2),
            "total_wait_s": round(sum(waits), 2),
            "avg_wait_s": round(sum(waits) / len(waits), 3),
            "min_wait_s": round(min(waits), 3),
            "max_wait_s": round(max(waits), 3),
        }
        # Merge user-supplied metadata (--meta). Namespace collisions with
        # computed fields under a user_ prefix; JSON-encode non-scalar values.
        for k, v in p.extra_meta.items():
            key = k if k not in meta else f"user_{k}"
            meta[key] = v if isinstance(v, (str, int, float, bool)) or v is None else json.dumps(v)
        try:
            path = report.write_report(p.report_dir, meta, self.decision_records, when=when)
            self._report_written = True
            self.logger.info(f"Wrote CSV report: {path}")
        except OSError as e:
            self.logger.warn(f"Could not write CSV report: {e}")

    def _write_trace(self, when=None) -> None:
        """Write the NPZ pose trace (once)."""
        if self._trace_written or not self.params.record_trace:
            return
        if len(self._traj["t"]) < 2:
            return

        from datetime import datetime
        from . import trace as trace_mod

        p = self.params
        when = when or datetime.now()
        meta = {
            # Resolved seed (an int even when --seed was omitted) so replay can
            # regenerate the exact maze this trace was recorded in.
            "seed": int(self.maze.seed),
            "n_turns": p.maze.n_turns,
            "cell": p.maze.cell,
            "wall_thickness": p.maze.wall_thickness,
            "wall_height": p.maze.wall_height,
            "min_gap": p.maze.min_gap,
            "max_gap": p.maze.max_gap,
            "trace_period": p.trace_period,
            "forward_speed": p.control.forward_speed,
            "turn_speed": p.control.turn_speed,
            "heading_kp": p.control.heading_kp,
        }
        try:
            path = trace_mod.write_trace(p.report_dir, self._traj, meta, when=when)
            self._trace_written = True
            self.logger.info(f"Wrote pose trace: {path}")
        except OSError as e:
            self.logger.warn(f"Could not write pose trace: {e}")

    def _nearest_turn_point(self):
        best, best_d = None, float("inf")
        for tp in self.maze.turn_points:
            d = math.hypot(self.x - tp.x, self.y - tp.y)
            if d < best_d:
                best, best_d = tp, d
        if best_d <= self.params.maze.cell:
            return best
        return None

    def _flash(self, text: str, color) -> None:
        self.hud_flash.setText(text)
        self.hud_flash["fg"] = color
        self._flash_until = time.monotonic() + 2.0

    # ==================================================================
    # Main loop
    # ==================================================================

    def _update(self, task):
        p = self.params
        # Fixed-timestep integration: identical behavior at any display rate.
        frame_dt = p.phys_dt if p.offscreen else min(self.clock.getDt(), 0.1)
        self._phys_acc += frame_dt
        steps = 0
        while self._phys_acc >= p.phys_dt and steps < 8:  # cap: avoid spiral
            self._phys_acc -= p.phys_dt
            self._step_physics(p.phys_dt)
            steps += 1

        # --- completion check ---
        if not self.completed and self.controller.state == "STOPPED_WAITING_EEG":
            gd = math.hypot(self.x - self.maze.goal_xy[0], self.y - self.maze.goal_xy[1])
            if gd < 0.9:
                self.completed = True
                self.finish_time = self._sim_time
                self._sample_trace()  # capture the final pose
                self.logger.info(
                    f"MAZE COMPLETE in {self.finish_time:.1f}s — "
                    f"{self.progress} turns, {self.wrong_turns} wrong."
                )
                self._finalize()

        self._apply_robot_pose()
        self._update_camera(frame_dt)
        self._update_hud()
        self._update_minimap()
        return self._maybe_exit(task)

    def _step_physics(self, dt: float) -> None:
        """One fixed physics tick: sense+decide at the control rate, then act."""
        p = self.params
        # Control loop at 20 Hz (sense -> decide -> act).
        self._control_acc += dt
        while self._control_acc >= p.control.control_period:
            self._control_acc -= p.control.control_period
            scan = self.lidar.scan(self.x, self.y, self.yaw)
            self.controller.on_scan(scan)
            self.controller.on_odom(self.yaw)
            self.controller.control_loop()

        # Integrate differential-drive kinematics with the current command.
        vx, wz = self.controller.cmd
        self.x += vx * math.cos(self.yaw) * dt
        self.y += vx * math.sin(self.yaw) * dt
        self.yaw = wrap_pi(self.yaw + wz * dt)
        self._resolve_collisions()

        rp = p.robot
        self.wheel_q[0] += (vx - wz * rp.wheel_separation / 2.0) / rp.wheel_radius * dt
        self.wheel_q[1] += (vx + wz * rp.wheel_separation / 2.0) / rp.wheel_radius * dt

        self._sim_time += dt

        # Periodic debug status (the original 1 Hz status timer).
        self._status_acc += dt
        if self._status_acc >= p.control.status_period:
            self._status_acc = 0.0
            self.controller.print_debug_status()

        # Pose trace sampling (every trace_period of sim time).
        self._trace_acc += dt
        if self._trace_acc >= p.trace_period:
            self._trace_acc -= p.trace_period
            self._sample_trace()

    # ------------------------------------------------------------------
    def _update_replay(self, task):
        p = self.params
        frame_dt = p.phys_dt if p.offscreen else min(self.clock.getDt(), 0.1)
        if not self._replay_paused:
            self._replay_time = min(self._replay_time + frame_dt, self._replay_len)
        self.x, self.y, self.yaw = self._interp_replay(self._replay_time)
        self._sim_time = self._replay_time

        self._apply_robot_pose(spin_wheels=False)
        self._update_camera(frame_dt)
        self._update_hud()
        self._update_minimap()
        return self._maybe_exit(task)

    def _interp_replay(self, t: float) -> tuple[float, float, float]:
        ts = self._rt
        if len(ts) == 0:
            return self.x, self.y, self.yaw
        if t <= ts[0]:
            return float(self._rx[0]), float(self._ry[0]), float(self._ryaw[0])
        if t >= ts[-1]:
            return float(self._rx[-1]), float(self._ry[-1]), float(self._ryaw[-1])
        i = int(np.searchsorted(ts, t))
        t0, t1 = ts[i - 1], ts[i]
        a = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
        x = self._rx[i - 1] + a * (self._rx[i] - self._rx[i - 1])
        y = self._ry[i - 1] + a * (self._ry[i] - self._ry[i - 1])
        # shortest-path angle interpolation
        yaw = wrap_pi(self._ryaw[i - 1] + a * wrap_pi(self._ryaw[i] - self._ryaw[i - 1]))
        return float(x), float(y), float(yaw)

    # ------------------------------------------------------------------
    def _apply_robot_pose(self, spin_wheels: bool = True) -> None:
        self.robot_model.root.setPos(self.x, self.y, 0)
        self.robot_model.root.setH(math.degrees(self.yaw))
        if spin_wheels:
            j = self.robot_model.joints
            if "wheel_left_joint" in j:
                j["wheel_left_joint"].set_position(self.wheel_q[0])
            if "wheel_right_joint" in j:
                j["wheel_right_joint"].set_position(self.wheel_q[1])

    def _maybe_exit(self, task):
        self._frame_count += 1
        if self.params.max_frames is not None and self._frame_count >= self.params.max_frames:
            if self.params.screenshot:
                self.screenshot(namePrefix=self.params.screenshot, defaultFilename=False)
            self.userExit()
        return task.cont

    # ------------------------------------------------------------------
    def _resolve_collisions(self) -> None:
        """Push the robot's footprint circle out of wall boxes (what the
        Gazebo physics engine would do on contact)."""
        r = self.params.robot.base_radius
        for w in self.maze.walls:
            hx, hy = w.sx / 2.0, w.sy / 2.0
            # closest point on AABB to circle center
            cx = min(max(self.x, w.cx - hx), w.cx + hx)
            cy = min(max(self.y, w.cy - hy), w.cy + hy)
            dx, dy = self.x - cx, self.y - cy
            d2 = dx * dx + dy * dy
            if d2 >= r * r or d2 == 0.0:
                continue
            d = math.sqrt(d2)
            push = (r - d) / d
            self.x += dx * push
            self.y += dy * push

    # ------------------------------------------------------------------
    def _update_camera(self, dt: float) -> None:
        # Robot body hidden only in first-person; ceiling hidden only in top-down.
        (self.robot_model.root.hide if self.view_mode == "first"
         else self.robot_model.root.show)()
        (self.ceiling_np.hide if self.view_mode == "top"
         else self.ceiling_np.show)()

        if self.view_mode == "top":
            self.camLens.setFov(70)
            target = LVector3(self.x, self.y, 14.0)
            if self._cam_pos is None:
                self._cam_pos = target
            k = 1.0 - math.exp(-6.0 * dt)
            self._cam_pos = self._cam_pos + (target - self._cam_pos) * k
            self.camera.setPos(self._cam_pos)
            self.camera.setHpr(0, -90, 0)  # straight down, north-up
            return

        if self.view_mode == "first":
            # Onboard view from roughly the head-camera height, looking ahead.
            # (The robot body is hidden above so it can't occlude the view.)
            # Wide FOV so the side walls stay visible.
            self.camLens.setFov(90)
            fwd = LVector3(math.cos(self.yaw), math.sin(self.yaw), 0)
            eye = LVector3(self.x, self.y, 0) + fwd * 0.16 + LVector3(0, 0, 1.30)
            self.camera.setPos(eye)
            self.camera.lookAt(eye + fwd * 4.0 - LVector3(0, 0, 0.20))
            return

        self.camLens.setFov(74)
        fwd = LVector3(math.cos(self.yaw), math.sin(self.yaw), 0)
        # Trail behind the robot, but pull in if a wall sits close behind
        # (e.g. just after a turn) so the camera never clips through it. The
        # camera height is fixed below wall-top, so shortening never leaks.
        dist = self.cam_dist
        hit = self.lidar.raycast(self.x, self.y, -fwd.x, -fwd.y)
        if math.isfinite(hit) and hit < dist + 0.3:
            dist = min(dist, max(0.8, hit - 0.25))

        desired = LVector3(self.x, self.y, 0) - fwd * dist + LVector3(0, 0, self.cam_height)
        if self._cam_pos is None:
            self._cam_pos = desired
        k = 1.0 - math.exp(-4.0 * dt)
        self._cam_pos = self._cam_pos + (desired - self._cam_pos) * k
        self.camera.setPos(self._cam_pos)
        look = LVector3(self.x, self.y, 0) + fwd * 1.4 + LVector3(0, 0, 0.85)
        self.camera.lookAt(look)

    # ------------------------------------------------------------------
    def _update_hud(self) -> None:
        if self.replay_mode:
            self._update_hud_replay()
            return

        c = self.controller
        elapsed = self.finish_time if self.finish_time is not None else self._sim_time
        fd = c.front_dist if math.isfinite(c.front_dist) else float("inf")
        tgt = "none" if c.target_yaw is None else f"{math.degrees(c.target_yaw):7.1f} deg"
        ws_status = self.ws.status if self.ws else "off (manual keys)"
        n_turns = len(self.maze.turn_points)
        last = "  -  " if self.last_decision_time is None else f"{self.last_decision_time:5.2f}s"
        if self.decision_times:
            avg = f"{sum(self.decision_times) / len(self.decision_times):5.2f}s"
        else:
            avg = "  -  "
        self.hud_status.setText(
            f"state   {c.state}\n"
            f"front   {fd:6.2f} m\n"
            f"yaw     {math.degrees(c.yaw):7.1f} deg   target {tgt}\n"
            f"cmd     v={c.cmd[0]:.2f} m/s  w={c.cmd[1]:+.2f} rad/s\n"
            f"ws      {ws_status}\n"
            f"turns   {self.progress}/{n_turns}   wrong {self.wrong_turns}\n"
            f"decide  last {last}  avg {avg}  (n={len(self.decision_times)})\n"
            f"time    {elapsed:6.1f} s   seed {self.maze.seed}   view {self.view_mode}"
        )

        if self.completed:
            total = sum(self.decision_times)
            avg_v = total / len(self.decision_times) if self.decision_times else 0.0
            self.hud_prompt.setText("MAZE COMPLETE!")
            self.hud_prompt["fg"] = (0.35, 1.0, 0.45, 1)
            self.hud_flash.setText(
                f"{len(self.decision_times)} decisions   "
                f"avg {avg_v:.2f} s   total wait {total:.1f} s"
            )
            self.hud_flash["fg"] = (0.9, 0.95, 1.0, 1)
            self._flash_until = time.monotonic() + 1.0  # keep it shown
        elif c.state == "STOPPED_WAITING_EEG":
            # Steady prompt (no flashing) with a live decision timer.
            waited = 0.0 if self._wait_start is None else time.monotonic() - self._wait_start
            self.hud_prompt.setText(
                f"waiting for decision:  LEFT or RIGHT?   ({waited:4.1f} s)"
            )
            self.hud_prompt["fg"] = (1, 0.95, 0.4, 1)
        else:
            self.hud_prompt.setText("")

        if not self.completed and time.monotonic() > self._flash_until:
            self.hud_flash.setText("")

    def _update_hud_replay(self) -> None:
        at_end = self._replay_time >= self._replay_len
        paused = "   [PAUSED]" if self._replay_paused else ""
        self.hud_status.setText(
            f"REPLAY{paused}\n"
            f"seed    {self.maze.seed}\n"
            f"time    {self._replay_time:6.2f} / {self._replay_len:6.2f} s\n"
            f"pos     x={self.x:6.2f}  y={self.y:6.2f}\n"
            f"yaw     {math.degrees(self.yaw):7.1f} deg\n"
            f"view    {self.view_mode}"
        )
        self.hud_prompt.setText("")
        self.hud_flash.setText(
            "replay finished  —  [r] restart   [space] play"
            if at_end else "[space] pause   [r] restart   [v] view"
        )
        self.hud_flash["fg"] = (0.85, 0.92, 1.0, 1)
