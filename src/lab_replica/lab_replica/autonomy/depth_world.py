"""What the wrist camera saw, as the obstacle model of the motion planner.

The item models are NOT obstacles. A fitted item pose is a top-down position
and yaw in one of the item's resting poses: it says where to grasp, but not
whether the real item is exactly there, leans, or has a part the model lacks.
So the planner keeps clear of what the depth camera actually measured:

  * The depth of every view goes into a height map over the table (2.5 mm
    cells): per cell, the top of what was seen there (the second-highest
    point; a cell with one point that has no occupied neighbour is taken as
    a stray depth pixel). Each cell is a
    column from the table (or the basket floor) up to that top.
  * Cells nobody saw (behind a tall item, in the camera's blind spots, holes
    in the depth) are filled conservatively: anything may be there up to the
    lowest height at which a camera ray passed through the cell's column
    (space in front of a measured surface is empty, and things stand on the
    table, so nothing in that column reaches higher). A cell no ray passed
    through (out of view, or a surface without depth) counts as occupied up
    to `max_item_height`. Tiny holes take their neighbours' height.
  * While an item is in the gripper it hides part of the table from the
    camera, which looks along the gripper axis. Those unseen cells are also
    bounded by what the survey before its pick-up saw there (`prior`):
    since then the robot only took the item away, which lowers nothing
    that was measured.
  * The robot itself, the held item and the basket's wall are removed from
    the depth, and so is the unseen space just outside the basket that is
    lower than its rim (its shadow); the basket is the one modelled obstacle
    (with the table).

The cells are also labelled with the item they belong to: the pixels of the
item's SAM 3 mask, grown by a centimetre, plus the seen occupied cells
joined to them within 2 cm of the fitted outline (the mug's handle, a part
the mask missed), and the unseen cells within 1.5 cm of those that are no
higher than the item (its own shadow: the fingers close in there); but only
cells nearer its own mask than any other item's. Every cell inside its
fitted outline, grown by a centimetre, that no other item's mask covers is
its own as well (it stood there: a tall item's own unseen base would
otherwise block its lift). The labels only decide which cells to leave
out: the item being picked (the fingers close on it, and it leaves with the
gripper) and the item in the gripper. Everything else seen stays an
obstacle.
"""

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .obstacles import Obstacle, remove_robot

CELL = 0.0025           # [m]
MIN_HEIGHT = 0.010      # a cell is occupied this much above its support (depth noise below)
STRIDE = 1              # depth pixels used (1: all)
FREE_EPS = 0.010        # a point this much in front of the measured surface is empty space
LEVEL_STEP = 0.010      # vertical resolution of the unseen-cell bound
SMALL_HOLE = 48         # unseen patches up to this many cells (3 cm^2) take their neighbours' height
ITEM_GROW = 0.010       # an item's cells: its mask grown by this ...
ITEM_ATTACH = 0.020     # ... plus occupied cells joined to it within this of its fitted outline
ITEM_SHADOW = 0.015     # ... plus unseen cells within this of it, no higher than it (its own shadow)
HELD_ENVELOPE = 0.010   # the held item's box grown by this was empty but for the item
BASKET_SHADOW = 0.05    # unseen space this far outside the rim, below it, is the basket's shadow
NEAR_BASE = 0.35        # near the base, points higher than the tallest item belong to the robot
FLOOR = -1e3            # "nothing here" in the maps (instead of -inf, which cv2.dilate mishandles)


@dataclass
class HeightLayer:
    """One obstacle for check_path: the cells of the items in `items` (or,
    with `unrecognised`, the cells of no item) in the height map. Same
    interface as robot.collision.Box (name, inside)."""
    name: str
    items: tuple
    unrecognised: bool
    world: "DepthWorld"

    def inside(self, P, margin):
        return self.world.below_top(P, margin, self.items, self.unrecognised)


class DepthWorld:
    """Height map of everything seen, labelled by item (see the module doc).

    frames      the rgbd.Frames (with .q and .T_world_cam) of this action
    detections  {item: perception.scene.Detection} (their masks label cells)
    held_box    function q -> robot.collision.Box of the held item (or None)
    prior       the DepthWorld of the survey before the held item was picked up (or None)
    region      (x_range, y_range) the map covers (each grown by 5 cm), or None for the
                placement region of cell.yaml: a map of the surroundings of the board
                (a fixed barrier) takes a larger one
    fill_unseen False: cells no view saw stay empty instead of being bounded by the camera
                rays (the map then holds only what was measured)
    """

    def __init__(self, frames, cell, items, basket, basket_xy, detections, max_item_height=0.30,
                 robot_radius=0.17, held_box=None, mask_erode_px=3, prior=None, region=None, fill_unseen=True):
        self.cell = cell
        self.items = items
        self.basket = basket
        self.basket_xy = None if basket_xy is None else np.asarray(basket_xy, float)
        self.table_z = cell["table"]["surface_z"]
        self.z_max = self.table_z + max_item_height
        pl = cell["placement"]
        x_range, y_range = region if region is not None else (pl["x_range"], pl["y_range"])
        self.x0, x1 = x_range[0] - 0.05, x_range[1] + 0.05
        self.y0, y1 = y_range[0] - 0.05, y_range[1] + 0.05
        self.nx = int(math.ceil((x1 - self.x0) / CELL))
        self.ny = int(math.ceil((y1 - self.y0) / CELL))
        X, Y = self.centres()
        self.keepout = np.hypot(X, Y) < max(robot_radius, pl["robot_keepout_radius"])
        self.support, self.wall = self._supports(X, Y)
        self.frames = [f for f in frames if f is not None]
        self.mask_erode_px = mask_erode_px

        top = self._seen_tops(self.frames, robot_radius, held_box)
        seen = np.isfinite(top)
        H = np.where(seen & (top > self.support + MIN_HEIGHT), top, FLOOR)
        self.unseen = ~seen & ~self.keepout & ~self.wall
        if fill_unseen:
            H = self._fill_unseen(H, X, Y)
        else:
            self.carved = np.zeros(H.shape, bool)
        if held_box is not None:
            H = self._below_held(H, held_box)
        if prior is not None and prior.H.shape == H.shape:
            # The prior's heights still bound everything (only an item was
            # taken away since): the lower of the two bounds holds.
            H = np.where(self.carved, np.minimum(H, prior.H), H)
        if self.basket_xy is not None:
            # Hidden by the basket and lower than its rim: the basket's shadow
            # (the basket itself is modelled exactly).
            b = self.basket
            r = np.hypot(X - self.basket_xy[0], Y - self.basket_xy[1])
            shadow = self.carved & (r <= b.r_top + BASKET_SHADOW) & (H <= self.table_z + b.height + 0.01)
            self.wall |= shadow
            self.carved &= ~shadow
        H[self.keepout | self.wall] = FLOOR
        self.H = H.astype(np.float32)
        self.occupied = self.H > FLOOR / 2
        self.blind = self.carved & (self.H >= self.z_max)     # no ray reached it

        masks = {it: self._mask_cells(d) for it, d in detections.items()}
        names = [it for it in masks if masks[it].any()]
        # Every cell goes to the item whose mask is nearest (cells between two
        # items are split between them).
        if names:
            dist = np.stack([cv2.distanceTransform((~masks[it]).astype(np.uint8), cv2.DIST_L2, 5)
                             for it in names])
            nearest = np.argmin(dist, axis=0)
            mine = {it: (nearest == k) & (dist[k] * CELL <= max(ITEM_GROW, ITEM_SHADOW) + 0.03)
                    for k, it in enumerate(names)}
        self.regions = {}
        for it, d in detections.items():
            own = mine[it] if it in mine else np.ones(self.H.shape, bool)
            foreign = np.zeros(self.H.shape, bool)
            for other in names:
                if other != it:
                    foreign |= masks[other]
            r = self._region(d, masks[it], own, foreign)
            if r.any():
                self.regions[it] = r
        self.labelled = np.zeros(self.H.shape, bool)
        for r in self.regions.values():
            self.labelled |= r
        self._cache = {}

    # ---- grid helpers ------------------------------------------------------------------

    def centres(self):
        xs = self.x0 + (np.arange(self.nx) + 0.5) * CELL
        ys = self.y0 + (np.arange(self.ny) + 0.5) * CELL
        return np.meshgrid(xs, ys)

    def index(self, x, y):
        """(ix, iy, inside-map mask) of world points."""
        ix = np.floor((np.asarray(x) - self.x0) / CELL).astype(int)
        iy = np.floor((np.asarray(y) - self.y0) / CELL).astype(int)
        ok = (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny)
        return np.clip(ix, 0, self.nx - 1), np.clip(iy, 0, self.ny - 1), ok

    def _supports(self, X, Y):
        """Support height per cell (table or basket floor) and the basket wall cells."""
        support = np.full(X.shape, self.table_z)
        wall = np.zeros(X.shape, bool)
        if self.basket_xy is not None:
            b = self.basket
            r = np.hypot(X - self.basket_xy[0], Y - self.basket_xy[1])
            support[r < b.inner_radius(b.height) - 0.01] = self.table_z + b.floor
            wall = (r >= b.inner_radius(0) - 0.012) & (r <= b.r_top + 0.02)
        return support, wall

    # ---- building the map --------------------------------------------------------------

    def _frame_points(self, f, robot_radius, held_box):
        P = f.points_world(stride=STRIDE)
        P = P[(P[:, 2] > self.table_z - 0.02) & (P[:, 2] < self.z_max + 0.05)]
        r = np.hypot(P[:, 0], P[:, 1])
        P = P[(r > robot_radius) & ((r > NEAR_BASE) | (P[:, 2] < self.z_max - 0.06))]
        if f.q is not None:
            P = remove_robot(P, f.q, self.cell)
            if held_box is not None:
                box = held_box(f.q)
                if box is not None:
                    P = P[~box.inside(P, 0.02)]
        return P

    def _seen_tops(self, frames, robot_radius, held_box):
        """Per cell the second-highest point over all frames, or the only
        one (NaN: not seen); a lone point high above an empty neighbourhood
        is dropped (a stray depth pixel)."""
        top = np.full(self.ny * self.nx, -np.inf)
        single = np.zeros(self.ny * self.nx, bool)
        for f in frames:
            P = self._frame_points(f, robot_radius, held_box)
            ix, iy, ok = self.index(P[:, 0], P[:, 1])
            idx, z = (iy * self.nx + ix)[ok], P[ok, 2]
            if len(idx) == 0:
                continue
            order = np.lexsort((z, idx))
            idx, z = idx[order], z[order]
            last = np.r_[idx[1:] != idx[:-1], True]
            first = np.r_[True, idx[1:] != idx[:-1]]
            ends = np.nonzero(last)[0]
            starts = np.nonzero(first)[0]
            two = ends - starts >= 1
            cells, z2 = idx[ends[two]], z[ends[two] - 1]
            better = z2 > top[cells]
            top[cells[better]] = z2[better]
            single[cells[better]] = False
            one = ~two
            cells, z1 = idx[ends[one]], z[ends[one]]
            better = z1 > top[cells]
            top[cells[better]] = z1[better]
            single[cells[better]] = True
        top = top.reshape(self.ny, self.nx)
        single = single.reshape(self.ny, self.nx)
        high = np.isfinite(top) & (top > self.support + MIN_HEIGHT)
        n_high = cv2.filter2D(high.astype(np.float32), -1, np.ones((3, 3), np.float32),
                              borderType=cv2.BORDER_CONSTANT) - high
        stray = high & single & (n_high < 0.5)
        top[stray] = self.support[stray]
        return np.where(np.isfinite(top), top, np.nan)

    def _fill_unseen(self, H, X, Y):
        """Heights for the unseen cells: small holes take their seen
        neighbours' height; larger patches are bounded by the camera rays
        that passed through their column."""
        n, lab, stats, _ = cv2.connectedComponentsWithStats(self.unseen.astype(np.uint8), connectivity=8)
        small = np.zeros(H.shape, bool)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] <= SMALL_HOLE:
                small |= lab == i
        if small.any():
            near = cv2.dilate(H.astype(np.float32), np.ones((9, 9), np.uint8))
            H = np.where(small, near, H)
        big = self.unseen & ~small
        self.carved = np.zeros(H.shape, bool)
        if not big.any():
            return H
        iy, ix = np.nonzero(big)
        x, y = X[iy, ix], Y[iy, ix]
        z0 = self.support[iy, ix]
        levels = np.arange(MIN_HEIGHT, self.z_max - self.table_z + 1e-9, LEVEL_STEP)
        # free[k, l]: some ray passed through column k at level l, in front of what it hit.
        free = np.zeros((len(ix), len(levels)), bool)
        for f in self.frames:
            Hh, Ww = f.depth.shape
            for k, dz in enumerate(levels):
                P = np.c_[x, y, z0 + dz]
                uv, d = f.project(P)
                u, v = np.round(uv[:, 0]).astype(int), np.round(uv[:, 1]).astype(int)
                ok = (d > 0.05) & (u >= 0) & (u < Ww) & (v >= 0) & (v < Hh)
                meas = np.full(len(ix), np.nan)
                meas[ok] = f.depth[v[ok], u[ok]]
                with np.errstate(invalid="ignore"):
                    free[:, k] |= meas > d + FREE_EPS
        # Bound: the lowest level a ray has shown to be empty (nothing stands
        # higher), none: max_item_height. Free at the lowest level: nothing there.
        any_free = free.any(axis=1)
        lowest = np.argmax(free, axis=1)
        bound = np.where(any_free, z0 + levels[lowest], self.z_max + LEVEL_STEP)
        H = H.copy()
        H[iy, ix] = np.where(any_free & (lowest == 0), FLOOR, bound)
        self.carved[iy, ix] = ~(any_free & (lowest == 0))
        return H

    def _below_held(self, H, held_box):
        """The held item was where its box is when each frame was taken, so
        nothing in an unseen cell's column reaches up to it there (the same
        bound as a camera ray's): the bound is lowered to the bottom of the
        box grown by HELD_ENVELOPE (depth points within 2 cm of the box
        already count as the held item)."""
        for f in self.frames:
            box = held_box(f.q) if f.q is not None else None
            if box is None:
                continue
            P = box.c + _box_surface(box.h + HELD_ENVELOPE, 0.002) @ box.R.T
            ix, iy, ok = self.index(P[:, 0], P[:, 1])
            low = np.full(H.shape, np.inf)
            np.minimum.at(low, (iy[ok], ix[ok]), P[ok, 2])
            cells = self.carved & np.isfinite(low)
            H = np.where(cells, np.minimum(H, low), H)
        return H

    def _mask_cells(self, det):
        """Cells under the points of a detection's mask (shrunk by a few
        pixels: depth at an item's silhouette is smeared along the ray)."""
        m = np.zeros(self.H.shape, bool)
        if det.mask is not None and det.frame is not None:
            mask = det.mask
            k = self.mask_erode_px
            if k > 0:
                mask = cv2.erode(mask.astype(np.uint8), np.ones((2 * k + 1, 2 * k + 1), np.uint8)) > 0
            P = det.frame.points_world(mask, stride=STRIDE)
            P = P[(P[:, 2] > self.table_z + 0.003) & (P[:, 2] < self.z_max + 0.05)]
            ix, iy, ok = self.index(P[:, 0], P[:, 1])
            m[iy[ok], ix[ok]] = True
        return m

    def _region(self, det, mask, own, foreign):
        """Cells of one detected item (see the module doc): its mask cells,
        grown, and the seen occupied cells joined to them near its outline;
        all of them among `own` (the cells nearer its mask than another's).
        Plus the cells inside its fitted outline (grown by ITEM_GROW) that no
        other item's mask covers: the item stood there, so it lifts its own
        model out of them."""
        k = int(round(ITEM_GROW / CELL))
        grown = cv2.dilate(mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                                           (2 * k + 1, 2 * k + 1))) > 0
        others = ~own
        m = mask | (grown & own & ~self.carved)
        # Occupied cells joined to the mask, within ITEM_ATTACH of the fitted outline.
        p = det.estimate.placement
        fp = self.items[det.item].footprint_world(p.rest, p.x, p.y, p.yaw)
        near = np.zeros(self.H.shape, np.uint8)
        cv2.fillConvexPoly(near, self._cells(fp), 1)
        # Inside the outline grown by ITEM_GROW: the item's own unless another's mask is there.
        inside = cv2.dilate(near, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))) > 0
        inside &= self.occupied & ~foreign
        a = int(round(ITEM_ATTACH / CELL))
        near = cv2.dilate(near, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * a + 1, 2 * a + 1))) > 0
        cand = (self.occupied & ~self.carved & near & ~others) | m
        n, lab = cv2.connectedComponents(cand.astype(np.uint8), connectivity=8)
        keep = np.unique(lab[m & (lab > 0)])
        if len(keep) == 0:
            # Nothing of the mask landed on the map (e.g. no depth on it): the outline then.
            region = near & self.occupied & ~self.carved & ~others
        else:
            region = np.isin(lab, keep)
        seen = region & ~self.carved & self.occupied
        if seen.any():
            s = int(math.ceil(ITEM_SHADOW / CELL))
            disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * s + 1, 2 * s + 1))
            ring = cv2.dilate(region.astype(np.uint8), disk) > 0
            region |= ring & self.carved & ~others & (self.H <= self.H[seen].max() + 0.01)
        return region | inside

    def _cells(self, poly):
        return np.round((np.asarray(poly, float) - (self.x0, self.y0)) / CELL - 0.5).astype(np.int32)

    # ---- queries -------------------------------------------------------------------------

    def has(self, item):
        return item in self.regions

    def _key(self, items):
        return tuple(sorted(i for i in items if i in self.regions))

    def _map(self, items, unrecognised=True):
        """Height map of the cells of `items` (a sorted tuple), and of every
        cell that belongs to no item if `unrecognised`; FLOOR elsewhere."""
        k = ("map", items, unrecognised)
        if k not in self._cache:
            keep = ~self.labelled if unrecognised else np.zeros(self.H.shape, bool)
            for it in items:
                keep = keep | self.regions[it]
            self._cache[k] = np.where(keep, self.H, FLOOR).astype(np.float32)
        return self._cache[k]

    def _dilated(self, items, unrecognised, r):
        k = ("dil", items, unrecognised, r)
        if k not in self._cache:
            M = self._map(items, unrecognised)
            if r > 0:
                M = cv2.dilate(M, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
            self._cache[k] = M
        return self._cache[k]

    def below_top(self, P, margin, items, unrecognised):
        """Which points (N, 3) are within `margin` of the columns of the cells
        of `items` (and the unrecognised ones)."""
        r = int(math.ceil(margin / CELL - 1e-9))
        M = self._dilated(items, unrecognised, r)
        ix, iy, ok = self.index(P[:, 0], P[:, 1])
        return ok & (P[:, 2] < M[iy, ix] + margin)

    def layers(self, items):
        """Obstacles for check_path: one per item in `items` that was seen,
        and one for everything seen that is no item ('unrecognised')."""
        out = [HeightLayer(it, (it,), False, self) for it in self._key(items)]
        out.append(HeightLayer("an unrecognised or unseen object", (), True, self))
        return out

    def top_under(self, poly, margin, items):
        """Highest top (world z) of the cells within `margin` of the convex
        polygon (centres within margin + half a cell), counting `items` and
        everything unrecognised; FLOOR if none is occupied. Space no camera
        ray reached is left out: these are quick checks of where a grasp or
        an item might fit, and the motion checks (below_top) still keep
        clear of it."""
        M = self._map(self._key(items))
        poly = np.asarray(poly, float)
        reach = margin + CELL / 2
        lo = np.floor((poly.min(axis=0) - reach - (self.x0, self.y0)) / CELL).astype(int)
        hi = np.ceil((poly.max(axis=0) + reach - (self.x0, self.y0)) / CELL).astype(int) + 1
        lo, hi = np.maximum(lo, 0), np.minimum(hi, (self.nx, self.ny))
        if np.any(hi <= lo):
            return FLOOR
        sub = M[lo[1]:hi[1], lo[0]:hi[0]]
        occ = (sub > FLOOR / 2) & ~self.blind[lo[1]:hi[1], lo[0]:hi[0]]
        if not occ.any():
            return FLOOR
        jj, ii = np.nonzero(occ)
        x = self.x0 + (ii + lo[0] + 0.5) * CELL
        y = self.y0 + (jj + lo[1] + 0.5) * CELL
        d = _distance_to_convex(poly, x, y)
        near = d <= reach
        return float(sub[jj[near], ii[near]].max()) if near.any() else FLOOR

    def heights_at(self, X, Y, radius=0.0, items=()):
        """Top heights (world z) at points, the max within `radius`; NaN where
        nothing is (or outside the map)."""
        M = self._dilated(self._key(items), True, int(math.ceil(radius / CELL - 1e-9)))
        ix, iy, ok = self.index(X, Y)
        h = M[iy, ix].astype(float)
        return np.where(ok & (h > FLOOR / 2), h, np.nan)

    def image(self, scale=2):
        """Top view of the map (BGR, x to the right, y up): free white, seen
        tops grey (darker = higher), unseen bounds purple, item cells
        outlined in colour and named, keep-out (robot) and basket wall green."""
        H, ny, nx = self.H, self.ny, self.nx
        h = np.clip((H - self.table_z) / (self.z_max - self.table_z), 0, 1)
        g = (235 - 185 * h).astype(np.uint8)
        img = np.full((ny, nx, 3), 255, np.uint8)
        img[self.occupied] = np.stack([g, g, g], -1)[self.occupied]
        unseen = self.occupied & self.carved
        img[unseen] = np.stack([200 - 120 * h, 60 + 0 * h, 170 - 60 * h], -1)[unseen].astype(np.uint8)
        img[self.keepout | self.wall] = (190, 230, 190)
        img = cv2.resize(img[::-1], (nx * scale, ny * scale), interpolation=cv2.INTER_NEAREST)
        rng = np.random.default_rng(7)
        for it, r in sorted(self.regions.items()):
            col = tuple(int(c) for c in rng.integers(0, 200, 3))
            m = cv2.resize(r[::-1].astype(np.uint8), (nx * scale, ny * scale), interpolation=cv2.INTER_NEAREST)
            cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, cnts, -1, col, 2)
            ys, xs = np.nonzero(m)
            cv2.putText(img, it, (int(xs.min()), int(ys.min()) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1,
                        cv2.LINE_AA)
        cv2.putText(img, "depth map: free white, seen grey (darker = higher), unseen purple, robot/basket green",
                    (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        return img

    def view_top(self):
        """Per cell the height [world z] up to which its column is inside the
        view of some frame (ignoring what blocks the view); -inf if none.
        Computed on a 1 cm grid."""
        if "view_top" not in self._cache:
            k = 4
            X, Y = self.centres()
            Xs, Ys = X[::k, ::k], Y[::k, ::k]
            top = np.full(Xs.shape, -np.inf)
            for dz in np.arange(0.0, self.z_max - self.table_z + 1e-9, LEVEL_STEP):
                P = np.c_[Xs.ravel(), Ys.ravel(), np.full(Xs.size, self.table_z + dz)]
                seen = np.zeros(Xs.size, bool)
                for f in self.frames:
                    uv, d = f.project(P)
                    h, w = f.depth.shape
                    seen |= (d > 0.05) & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
                top = np.where(seen.reshape(Xs.shape), self.table_z + dz, top)
            full = np.repeat(np.repeat(top, k, axis=0), k, axis=1)[:self.ny, :self.nx]
            self._cache["view_top"] = full
        return self._cache["view_top"]

    def in_view(self, poly, height, margin=0.01):
        """Whether the camera views covered the polygon (grown by `margin`)
        up to `height` [world z] (so that an item put there can be seen whole)."""
        V = self.view_top()
        poly = np.asarray(poly, float)
        lo = np.floor((poly.min(axis=0) - margin - (self.x0, self.y0)) / CELL).astype(int)
        hi = np.ceil((poly.max(axis=0) + margin - (self.x0, self.y0)) / CELL).astype(int) + 1
        if np.any(lo < 0) or hi[0] > self.nx or hi[1] > self.ny:
            return False
        jj, ii = np.mgrid[lo[1]:hi[1], lo[0]:hi[0]]
        x = self.x0 + (ii.ravel() + 0.5) * CELL
        y = self.y0 + (jj.ravel() + 0.5) * CELL
        near = _distance_to_convex(poly, x, y) <= margin
        return bool(np.all(V[jj.ravel()[near], ii.ravel()[near]] >= height))

    def blobs(self, min_cells=32):
        """The unrecognised occupied cells as obstacles (for logs and
        pictures), without the space out of every view (the map's border)."""
        occ = (self.occupied & ~self.labelled & ~self.blind).astype(np.uint8)
        n, lab = cv2.connectedComponents(occ, connectivity=8)
        X, Y = self.centres()
        out = []
        for i in range(1, n):
            cells = lab == i
            if cells.sum() < min_cells:
                continue
            cx, cy = X[cells], Y[cells]
            h = CELL / 2
            pts = np.concatenate([np.c_[cx - h, cy - h], np.c_[cx + h, cy - h], np.c_[cx + h, cy + h],
                                  np.c_[cx - h, cy + h]])
            hull = cv2.convexHull(pts.astype(np.float32)).reshape(-1, 2).astype(float)
            in_basket = self.support[cells].mean() > self.table_z + 1e-6
            out.append(Obstacle(hull, float(self.H[cells].max()), "basket" if in_basket else "table",
                                unseen=bool(self.carved[cells].mean() > 0.5)))
        return out


def _box_surface(half, step):
    """Points on the surface of a box (half sizes `half`, its own frame) at most `step` apart."""
    axes = [np.linspace(-h, h, max(2, int(math.ceil(2 * h / step)) + 1)) for h in half]
    out = []
    for k in range(3):
        a, b = [axes[j] for j in range(3) if j != k]
        A, B = np.meshgrid(a, b)
        for side in (-half[k], half[k]):
            P = np.empty((A.size, 3))
            P[:, k] = side
            P[:, [j for j in range(3) if j != k]] = np.c_[A.ravel(), B.ravel()]
            out.append(P)
    return np.concatenate(out)


def _distance_to_convex(poly, x, y):
    """Distance of points (x, y) to a convex polygon (0 inside)."""
    P = np.c_[x, y]
    a = poly
    b = np.roll(poly, -1, axis=0)
    e = b - a
    L2 = np.maximum((e ** 2).sum(axis=1), 1e-12)
    # Distance to each edge segment.
    t = np.clip(((P[:, None, :] - a[None]) * e[None]).sum(-1) / L2, 0, 1)
    proj = a[None] + t[..., None] * e[None]
    d = np.sqrt(((P[:, None, :] - proj) ** 2).sum(-1)).min(axis=1)
    # Inside: on the same side of every edge.
    cross = e[None, :, 0] * (P[:, None, 1] - a[None, :, 1]) - e[None, :, 1] * (P[:, None, 0] - a[None, :, 0])
    inside = np.all(cross >= 0, axis=1) | np.all(cross <= 0, axis=1)
    return np.where(inside, 0.0, d)
