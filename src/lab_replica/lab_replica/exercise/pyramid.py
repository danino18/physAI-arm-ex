"""A pyramid of cubes: three in a row, two on top of them, one on the very top.

    top
    mid-left  mid-right
    base-left base-mid  base-right

A cube of a middle or the top slot rests on two cubes: the middle of them.
The planner (task_planning/domain.pddl: place-in-slot) chooses which cube
goes to which slot, and the order; this file knows where the slots are, what
the robot has put where, and what MoveIt must keep clear of:

    Layout(origin, axis_yaw, size, gap)   where the slots are
    Layout.position(slot, placed)         (x, y, z_support) of a slot: the middle of the cubes it rests on
    facts(layout, cubes)                  the problem's facts: free slots, which rests on which, the cubes
    Pyramid                               what was put where (the positions the robot reached), the boxes
                                          of those cubes for MoveIt, and whether the cubes are still there
    clear_of(layout, ...)                 what is in the way of the pyramid's area, or nothing

The cubes the robot has put are kept here and not read from the camera:
a cube on a cube is not recognised as an item (state_estimation.
estimate_placement), and the top view hides the cubes below.
"""

import math
from dataclasses import dataclass

import numpy as np

from ..toolkit import Box, polygons_intersect

BASE = ("base-left", "base-mid", "base-right")
RESTS_ON = {"mid-left": ("base-left", "base-mid"), "mid-right": ("base-mid", "base-right"),
            "top": ("mid-left", "mid-right")}
SLOTS = BASE + tuple(RESTS_ON)
LEVEL = {**{s: 0 for s in BASE}, "mid-left": 1, "mid-right": 1, "top": 2}

PAD = 0.005                 # the boxes of the cubes are this much larger on every side (motion_planning.ITEM_PAD)
ZONE_MARGIN = 0.06          # the pyramid's area: its cubes' outline grown by this (room for the fingers)
SIZE_TOLERANCE = 0.001      # a cube is as large as the pyramid's when its edge is within this
CHECK_TOLERANCE = 0.02      # a cube is still there when the map's height at its slot is at most this much lower
CIRCLE_SIDES = 24


@dataclass
class Layout:
    """origin   (x, y) of the middle cube of the base row
    axis_yaw the direction of the row, in the world (pi / 2: along y)
    size     the cubes' edge [m]
    gap      between neighbours in the base row [m]
    table_z  the table's height"""
    origin: tuple
    axis_yaw: float = math.pi / 2
    size: float = 0.05
    gap: float = 0.015
    table_z: float = 0.0

    @property
    def axis(self):
        return np.array([math.cos(self.axis_yaw), math.sin(self.axis_yaw)])

    @property
    def pitch(self):
        return self.size + self.gap

    def nominal(self, slot):
        """(x, y) of a slot when the cubes are where the layout says."""
        if slot in BASE:
            c = np.asarray(self.origin, float) + (BASE.index(slot) - 1) * self.pitch * self.axis
            return float(c[0]), float(c[1])
        a, b = (self.nominal(s) for s in RESTS_ON[slot])
        return (a[0] + b[0]) / 2, (a[1] + b[1]) / 2

    def z_support(self, slot):
        """The height of what a cube in `slot` rests on."""
        return self.table_z + LEVEL[slot] * self.size

    def position(self, slot, placed=None):
        """(x, y, z_support) of `slot`. A base slot is where the layout says;
        a slot on others is the middle of the two it rests on, where the
        robot put them (`placed`: {slot: (x, y)}), or where the layout says."""
        if slot in BASE:
            x, y = self.nominal(slot)
        else:
            pts = [(placed or {}).get(s) or self.nominal(s) for s in RESTS_ON[slot]]
            x, y = (pts[0][0] + pts[1][0]) / 2, (pts[0][1] + pts[1][1]) / 2
        return x, y, self.z_support(slot)

    def zone(self, margin=ZONE_MARGIN):
        """The pyramid's area: the polygon (4, 2) round the base row, grown by `margin`."""
        half_long = (3 * self.size + 2 * self.gap) / 2 + margin
        half_short = self.size / 2 + margin
        u = self.axis
        v = np.array([-u[1], u[0]])
        c = np.asarray(self.origin, float)
        return np.array([c + s * half_long * u + t * half_short * v for s, t in ((-1, -1), (1, -1), (1, 1), (-1, 1))])

    def in_zone(self, xy, margin=ZONE_MARGIN):
        """Whether the point (x, y) is inside the pyramid's area."""
        u = self.axis
        v = np.array([-u[1], u[0]])
        d = np.asarray(xy, float) - np.asarray(self.origin, float)
        return abs(d @ u) <= (3 * self.size + 2 * self.gap) / 2 + margin and abs(d @ v) <= self.size / 2 + margin

    def box(self, x, y, level, pad=PAD):
        """The Box of a cube of this level at (x, y), aligned with the row."""
        c, s = math.cos(self.axis_yaw), math.sin(self.axis_yaw)
        R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        h = self.size / 2 + pad
        return Box("pyramid", [x, y, self.table_z + (level + 0.5) * self.size], R, [h, h, h])


def is_cube(model, size, tolerance=SIZE_TOLERANCE):
    """Whether an item's model (objects/block.py) is a cube of edge `size`."""
    spec = getattr(model, "spec", {})
    return spec.get("kind") == "block" and abs(float(spec.get("size", 0.0)) - size) <= tolerance


def cubes_of(items, size):
    """The names of the cubes of edge `size` among `items` ({name: model}), sorted."""
    return sorted(k for k, m in items.items() if is_cube(m, size))


def facts(cubes):
    """The facts a pyramid problem adds to the state: every slot free, which
    slots are a base and which rests on which, and the cubes that may fill them."""
    out = {("slot-free", s) for s in SLOTS} | {("base", s) for s in BASE}
    out |= {("on-slots", s, a, b) for s, (a, b) in RESTS_ON.items()}
    return frozenset(out | {("cube", c) for c in cubes})


def goal():
    """The goal: every slot taken (a formula as tamp/goal.py reads one)."""
    return ["and"] + [["not", ["slot-free", s]] for s in SLOTS]


def filled(state):
    """{slot: cube} of the facts (in-slot cube slot) of a state."""
    return {f[2]: f[1] for f in state if f[0] == "in-slot"}


class Pyramid:
    """What the robot put where: for each filled slot, the cube and the
    position it was put at (the robot's, not the layout's)."""

    def __init__(self, layout):
        self.layout = layout
        self.placed = {}            # {slot: (cube, x, y)}

    def items(self):
        return {c for c, _, _ in self.placed.values()}

    def supports(self, slot):
        return RESTS_ON.get(slot, ())

    def xy(self):
        return {s: (x, y) for s, (_, x, y) in self.placed.items()}

    def target(self, slot):
        """(x, y, z_support) for a cube to be put in `slot`."""
        return self.layout.position(slot, self.xy())

    def put(self, slot, cube, x, y):
        self.placed[slot] = (cube, float(x), float(y))

    def boxes(self, skip=()):
        """MoveIt's boxes of the cubes in the pyramid, but those of the slots `skip`
        (the ones the cube being put rests on: it touches them)."""
        return [self.layout.box(x, y, LEVEL[s]) for s, (_, x, y) in self.placed.items() if s not in skip]

    def expected_top(self, slot):
        return self.layout.table_z + (LEVEL[slot] + 1) * self.layout.size

    def missing(self, heights):
        """The filled slots whose cube is gone: `heights` {slot: the map's top height
        at the slot, NaN: nothing}."""
        return [s for s in self.placed
                if not heights.get(s, float("nan")) >= self.expected_top(s) - CHECK_TOLERANCE]


def check_cubes(pyramid, heights_at):
    """The filled slots whose cube is gone: `heights_at(X, Y)` gives the map's
    tops at points (DepthWorld.heights_at); a slot is checked at its centre."""
    heights = {}
    for s, (_, x, y) in pyramid.placed.items():
        heights[s] = float(heights_at(np.array([x]), np.array([y]))[0])
    return pyramid.missing(heights)


def box_polygon(box):
    """The outline (4, 2) in the world of a Box: its footprint, grown to fit the box's rotation."""
    corners = np.array([[sx, sy] for sx in (-1, 1) for sy in (-1, 1)]) * box.h[:2]
    corners = corners @ np.asarray(box.R)[:2, :2].T + np.asarray(box.c)[:2]
    c = corners.mean(axis=0)
    order = np.argsort(np.arctan2(corners[:, 1] - c[1], corners[:, 0] - c[0]))
    return corners[order]


def circle_polygon(centre, radius, sides=CIRCLE_SIDES):
    a = np.linspace(0, 2 * math.pi, sides, endpoint=False)
    return np.c_[centre[0] + radius * np.cos(a), centre[1] + radius * np.sin(a)]


def clear_of(layout, region, barrier=(), basket=None, others=None):
    """What is in the way of the pyramid's area (`layout.zone()`), as a list of
    sentences; empty when nothing is.

    region   {"x_range", "y_range"} the board: the area must be inside it
    barrier  the fixed obstacle's Boxes (scene_model.py)
    basket   (x, y, radius) of the basket's rim, or None
    others   {name: polygon (N, 2)}: things on the table (items, unknown obstacles)"""
    zone = layout.zone()
    problems = []
    x0, x1 = region["x_range"]
    y0, y1 = region["y_range"]
    if not (np.all(zone[:, 0] >= x0) and np.all(zone[:, 0] <= x1) and np.all(zone[:, 1] >= y0)
            and np.all(zone[:, 1] <= y1)):
        problems.append(f"the pyramid's area reaches out of the board (x {x0:.2f} to {x1:.2f}, y {y0:.2f} to {y1:.2f})")
    for b in barrier:
        if polygons_intersect(zone, box_polygon(b), 0.0):
            problems.append(f"the obstacle ({b.name}) is within {ZONE_MARGIN * 100:.0f} cm of the pyramid")
    if basket is not None and polygons_intersect(zone, circle_polygon(basket[:2], basket[2]), 0.0):
        problems.append("the basket is within the pyramid's area")
    for name, poly in (others or {}).items():
        if polygons_intersect(zone, poly, 0.0):
            problems.append(f"{name} stands within the pyramid's area")
    return problems
