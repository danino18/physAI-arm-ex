"""A fixed barrier (a wall, a partition) in the cell, scanned once and kept
clear of by every motion.

The obstacles of a survey (`scene.obstacles`, motion_planning.scene_obstacles)
are what one view of the board shows, and they are replaced by the next
survey. A barrier that stands there for good is scanned instead from several
fixed views, turned into a few boxes, and kept in MoveIt's planning scene
for as long as the console runs (or, saved to a file, for the next run):

    scan_environment(robot)      the robot looks from OBSERVATION_JOINTS and the SCAN_VIEWS, and
                                 returns the boxes of the tallest thing it measured that is no item,
                                 no basket and not the table or the wall (or of everything it
                                 measured that is none of them)
    static_obstacles(world)      the same boxes from a depth_world.DepthWorld
    boxes_from_height_map(...)   the boxes of the cells of a height map (what static_obstacles does)
    barrier_top(boxes)           the height of the tallest box: motions rise above it
    near_barrier(polygon, boxes) whether a survey's obstacle is (part of) the barrier
    save_boxes, load_boxes       the boxes in a JSON file

The boxes are toolkit.Box (centre, rotation about the vertical, half sizes)
that stand on the table up to the top of what was measured, grown by
BARRIER_PAD. A slanted wall gets boxes along it, not one box around it, and a
wall of different heights gets one box per height; there are at most
MAX_BOXES unless the barrier is made of more separate pieces.
"""

import json
import math

import cv2
import numpy as np

from ..toolkit import Box, PlanError, down_view, look_at
from . import state_estimation as se

BARRIER_PAD = 0.02          # the boxes are this much larger on every side
MIN_CELLS = 40              # a piece of fewer map cells (2.5 cm^2) is noise
MIN_HEIGHT = 0.03           # a cell is a barrier when it is this much above its support
MAX_BOXES = 16              # a box is one more thing every motion is checked against
MAX_SPLITS = 6              # a piece is cut in two at most this many times
TALL_WARNING = 0.45         # scan_environment warns about an obstacle taller than this [m]
OBSTACLE_MIN_CELLS = 256    # the tallest piece is chosen among those of this many cells (16 cm^2): not a depth stray
# What a box may be: the heights of its cells within `height`, its cells at least
# `fill` of its area. When there would be more than MAX_BOXES, the next row is used.
BOX_TOLERANCES = ((0.03, 0.75), (0.06, 0.60), (0.12, 0.45), (math.inf, 0.0))
SCAN_MIN_VIEWS = 2          # fewer scanned views than this: no barrier model

# The views to scan from: each a list of alternatives (the first that the arm
# can reach is used), at the centre of the board (SCAN_CENTRE), from above and
# from the front, the back (the wall's side) and the right. Find the ones that
# see the barrier with run/workspace_view.sh, as for OBSERVATION_JOINTS.
SCAN_CENTRE = (0.43, 0.0)
SCAN_VIEWS = [
    [down_view(SCAN_CENTRE[0], SCAN_CENTRE[1], h) for h in (0.85, 0.75)],
    [se.SIDE_VIEW],
    [look_at((SCAN_CENTRE[0] + dx, 0.30, h), (SCAN_CENTRE[0], 0.0, 0.0)) for dx in (0.0, 0.2) for h in (0.65, 0.75)],
    [look_at((0.88, dy, h), (SCAN_CENTRE[0], 0.0, 0.0)) for dy in (0.0, -0.2) for h in (0.60, 0.70)],
]


# ---- the boxes of a height map ---------------------------------------------------------------


def _min_rect(xy, cell):
    """(centre (2,), u (2,), half (2,)): the smallest rectangle round the cells
    centred at xy (N, 2), u the direction of its longer side."""
    corners = cv2.boxPoints(cv2.minAreaRect(np.asarray(xy, np.float32))).astype(float)
    e0, e1 = corners[1] - corners[0], corners[2] - corners[1]
    long_, short = (e0, e1) if np.linalg.norm(e0) >= np.linalg.norm(e1) else (e1, e0)
    u = long_ / max(np.linalg.norm(long_), 1e-9)
    return corners.mean(axis=0), u, np.array([np.linalg.norm(long_), np.linalg.norm(short)]) / 2 + cell / 2


def _split(pts, height, fill, cell, depth=0):
    """pts (N, 3) cells (x, y, top) cut into pieces that a box fits: its
    cells within `height` of each other and at least `fill` of its area.
    A piece is cut in two across its longer side."""
    centre, u, half = _min_rect(pts[:, :2], cell)
    tight = len(pts) * cell ** 2 >= fill * 4 * half[0] * half[1] and np.ptp(pts[:, 2]) <= height
    if tight or depth >= MAX_SPLITS or len(pts) < 2 * MIN_CELLS:
        return [pts]
    first = (pts[:, :2] - centre) @ u < 0
    if first.all() or not first.any():
        return [pts]
    return _split(pts[first], height, fill, cell, depth + 1) + _split(pts[~first], height, fill, cell, depth + 1)


def _box(pts, base_z, cell, pad, name):
    centre, u, half = _min_rect(pts[:, :2], cell)
    top = float(pts[:, 2].max()) + pad
    R = np.array([[u[0], -u[1], 0.0], [u[1], u[0], 0.0], [0.0, 0.0, 1.0]])
    h = (top - base_z) / 2
    return Box(name, [centre[0], centre[1], base_z + h], R, [half[0] + pad, half[1] + pad, h])


def pieces_of(X, Y, H, cells, min_cells=MIN_CELLS):
    """The connected pieces (8-connected) of `cells` (a boolean map) of a
    height map, each as its cells' (x, y, top) (N, 3)."""
    n, label = cv2.connectedComponents(np.asarray(cells, np.uint8), connectivity=8)
    pieces = []
    for i in range(1, n):
        m = label == i
        if m.sum() >= min_cells:
            pieces.append(np.c_[X[m], Y[m], H[m]])
    return pieces


def piece_top(piece):
    """The height of a piece: its cells' 90th percentile (a stray high cell does not count)."""
    return float(np.percentile(piece[:, 2], 90))


def tallest_piece(pieces, min_cells=OBSTACLE_MIN_CELLS):
    """The tallest of `pieces` among those of at least `min_cells` cells, or None."""
    big = [p for p in pieces if len(p) >= min_cells]
    return max(big, key=piece_top) if big else None


def boxes_from_height_map(X, Y, H, cells, base_z, cell, pad=BARRIER_PAD, min_cells=MIN_CELLS, max_boxes=MAX_BOXES,
                          tallest_only=False):
    """The boxes of the connected pieces of `cells` (a boolean map; 8-connected)
    in a height map: X, Y the cells' centres, H their tops, each box standing
    on base_z up to the highest cell it holds, plus `pad` on every side.
    `tallest_only`: only the boxes of the tallest piece (tallest_piece)."""
    pieces = pieces_of(X, Y, H, cells, min_cells)
    if tallest_only:
        tall = tallest_piece(pieces)
        pieces = [] if tall is None else [tall]
    parts = []
    for height, fill in BOX_TOLERANCES:
        parts = [p for pts in pieces for p in _split(pts, height, fill, cell)]
        if len(parts) <= max_boxes:
            break
    return [_box(p, base_z, cell, pad, f"barrier {k}") for k, p in enumerate(parts)]


def static_obstacles(world, pad=BARRIER_PAD, tallest_only=True):
    """The boxes of what a depth_world.DepthWorld measured (built with
    fill_unseen=False over the region of the barrier) that nothing explains: no
    item, not the basket, and at least MIN_HEIGHT above the table or the
    basket floor. `tallest_only`: of those, the tallest thing only (the obstacle)."""
    from ..autonomy.depth_world import CELL
    X, Y = world.centres()
    cells = world.occupied & ~world.labelled & ~world.unseen & ~world.blind & (world.H > world.support + MIN_HEIGHT)
    return boxes_from_height_map(X, Y, world.H, cells, world.table_z, CELL, pad, tallest_only=tallest_only)


def describe_things(world):
    """One line per thing the map holds that nothing explains (tallest first), for the log."""
    X, Y = world.centres()
    cells = world.occupied & ~world.labelled & ~world.unseen & ~world.blind & (world.H > world.support + MIN_HEIGHT)
    return [f"at ({p[:, 0].mean():.3f}, {p[:, 1].mean():.3f}), {100 * piece_top(p) - 100 * world.table_z:.0f} cm high, "
            f"{len(p) * 0.0025 ** 2 * 1e4:.0f} cm^2" for p in sorted(pieces_of(X, Y, world.H, cells), key=piece_top,
                                                                      reverse=True)]


# ---- using them -------------------------------------------------------------------------


def barrier_top(boxes):
    """The height of the tallest of `boxes` (the table's surface when there are none)."""
    return max((float(b.c[2] + b.h[2]) for b in boxes), default=None)


def near_barrier(polygon, boxes, margin=0.03):
    """Whether the centre of a survey's obstacle (polygon (N, 2)) is within
    `margin` of one of the barrier's boxes: it is the barrier, seen again."""
    c = np.asarray(polygon, float).mean(axis=0)
    return any(b.inside(np.array([[c[0], c[1], b.c[2]]]), margin)[0] for b in boxes)


def save_boxes(path, boxes):
    with open(path, "w") as f:
        json.dump([{"name": b.name, "c": np.asarray(b.c).tolist(), "R": np.asarray(b.R).tolist(),
                    "h": np.asarray(b.h).tolist()} for b in boxes], f, indent=1)


def load_boxes(path):
    with open(path) as f:
        return [Box(d["name"], d["c"], d["R"], d["h"]) for d in json.load(f)]


# ---- scanning ----------------------------------------------------------------------------


def scan_environment(robot, views=None, tallest_only=True):
    """The boxes of the obstacle: the robot looks from OBSERVATION_JOINTS and
    then from each of `views` (SCAN_VIEWS; each a list of alternatives, the first
    reachable is used; a station none of whose alternatives is reachable is
    skipped), adding every view to the same scene, and models what it measured
    (static_obstacles, from robot.map_barrier(scene)): the tallest thing that is
    no item, no basket and not the table, or with `tallest_only` False every
    such thing.

    The motions keep clear of what the robot knows, and the barrier is not
    known yet: scan slowly, with a hand on the emergency stop. Raises PlanError
    when fewer than SCAN_MIN_VIEWS views could be reached."""
    scene = robot.survey(se.OBSERVATION_JOINTS)
    used = 1
    for k, alternatives in enumerate(SCAN_VIEWS if views is None else views):
        robot.phase(f"scanning: view {k + 1}")
        try:
            scene = robot.survey(alternatives, scene)
            used += 1
        except PlanError as e:
            robot.log(f"  scan view {k + 1} skipped: {e}")
    if used < SCAN_MIN_VIEWS:
        raise PlanError(f"only {used} view could be reached: no model of the barrier")
    world = robot.map_barrier(scene)
    things = describe_things(world)
    robot.log(f"  seen from {used} views, no item, basket or table: " + ("; ".join(things) if things else "nothing"))
    boxes = static_obstacles(world, tallest_only=tallest_only)
    for b in boxes:
        if b.c[2] + b.h[2] > TALL_WARNING:
            robot.log(f"  warning: {b.name} is {100 * (b.c[2] + b.h[2]):.0f} cm high: the robot's own arm, "
                      f"seen by the camera, rather than an obstacle? (see the list above)")
    robot.log(f"  {'obstacle' if tallest_only else 'barrier'}: {len(boxes)} boxes"
              + "".join(f"\n    {b.name}: centre {np.round(b.c, 3)}, half sizes {np.round(b.h, 3)}" for b in boxes))
    return boxes
