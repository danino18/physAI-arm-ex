"""State estimation: from the wrist camera's images to where the items are,
and how to grasp one.

The executive (autonomy/executive.py) takes RGB-D frames from the wrist
camera and asks SAM 3 for every item by name. Each answer is a mask (a
boolean image) with a score. These functions turn them into the scene:

    OBSERVATION_JOINTS   the arm's joints from which the camera sees the whole workspace
    SIDE_VIEW            a camera pose looking at the workspace at an angle
    close_views          camera poses looking straight down at a spot from closer
    pixels_to_world      depth pixels -> 3D points in the world frame
    estimate_placement   one item's points and those around it -> where and how it rests
                         (height_map, pose_cost, fit_pose: its model matched to the points)
    assign_masks         the candidate masks of one frame -> at most one per item
    fuse_views           the detections of several frames -> one per item
    grasp_in_image       an item's mask -> top-down grasps in the image
    grasp_pose           a grasp in the image -> the gripper's 3D pose and opening

Conventions (see lab_replica/toolkit.py for the types):
  * The world frame has z up, z = 0 on the table top (cell.yaml).
  * The camera's optical frame has z along the view, x to the right in the
    image and y down; a pixel (u, v) is column u, row v.
  * K is the 3x3 intrinsic matrix, T_world_cam the 4x4 pose of the optical
    frame in the world.
"""

import math

import cv2
import numpy as np

from ..toolkit import (Estimate, Grasp, ImageGrasp, Placement, down_view, gauss, gripper_pose, hand_eye, look_at,
                       mask_overlap, model_height_map, rect_fit, rg6, robust_top, wrap)

# ---- where the camera looks from ----------------------------------------------------------

# The camera looks with robot.survey(view, scene=None, ...): `view` is either
# the arm's six joints or a pose of the camera (4x4, its optical frame in
# the world), or a list of them (the first reachable is used). Given the
# scene of earlier views, survey adds what the new view sees to it: an item
# is recognised from any view, and its position comes from the view that sees
# it best (straight down and close).
#
# The observation configuration: the arm's joints [rad] (shoulder_pan,
# shoulder_lift, elbow, wrist_1, wrist_2, wrist_3) from which the wrist
# camera sees the whole workspace. Every action starts by looking from here
# (robot.survey(OBSERVATION_JOINTS)): SAM 3 finds the items and the basket
# in this one image, so from here the camera must see all of the blue board
# (0.70 x 0.59 m), with a basket anywhere on it and the items in it, without
# the gripper or the arm in the way. Seen from too far away, the items get small and SAM 3 misses
# them; from too close, the board's edges are out of the image. The arm must
# also be able to stand here holding an item (the put actions look from here
# too) without touching the table or the wall.
#
# Find it on the robot: run/workspace_view.sh real shows the camera live,
# with SAM 3's masks of the items, and prints the arm's joints as they
# change. Move the arm by hand (freedrive on the teach pendant) until the
# camera sees everything, and copy the joints here.
#
# One view may not be enough: seen from straight above, some items are hard
# for SAM 3 to name, and from 80 cm away the depth is only so accurate. Other
# views can be found by hand like this one (more joints), or computed (camera
# poses: toolkit.look_at, toolkit.down_view); the skills decide when to look
# again (skills.py).
OBSERVATION_JOINTS = [2.984, -1.614, 0.294, -0.520, -1.416, 1.357]

# From the front at an angle: SAM 3 names items it does not recognise from
# straight above (the blue cup's yellow stripe is on its side, the spray
# bottle is a grey head on a red blob from above).
SIDE_VIEW = look_at(eye=(0.40, -0.25, 0.70), target=(0.40, 0.03, 0.0))

CLOSE_HEIGHT = 0.50     # a closer look: the camera this high above the table [m]


def close_views(x, y, height=CLOSE_HEIGHT):
    """Camera poses straight above (x, y), `height` above the table, looking
    down, turned about the vertical in steps of 45 degrees: the camera is
    beside the gripper, so turning moves the gripper around the spot, and
    some turns put the arm where it cannot be. The turns that put the gripper
    farthest from the robot's base axis come first: near the base the arm
    folds over its own shoulder, low over the table."""
    T_cam_grip = np.linalg.inv(hand_eye())
    poses = [down_view(x, y, height, (math.cos(a), math.sin(a), 0.0)) for a in np.arange(8) * math.pi / 4]
    return sorted(poses, key=lambda T: -math.hypot(*(T @ T_cam_grip)[:2, 3]))


# ---- from pixels to the world -------------------------------------------------------------


def pixels_to_world(u, v, depth, K, T_world_cam):
    """World points (N, 3) of the pixels (u[i], v[i]) (arrays of length N)
    whose depth along the camera's view (z of the optical frame, metres) is
    depth[i]."""
    u, v, z = (np.asarray(a, float) for a in (u, v, depth))
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    P = np.stack([(u - cx) / fx * z, (v - cy) / fy * z, z], axis=1)
    return P @ T_world_cam[:3, :3].T + T_world_cam[:3, 3]


# ---- one item: where it rests ---------------------------------------------------------------
#
# The item's model is matched to what the camera saw, both seen from above as
# height maps: the model's (toolkit.model_height_map, from its collision
# shapes) and the points' (height_map). The pose (resting pose, x, y, yaw)
# is the one where the model's heights best explain the points' heights
# (pose_cost), found by trying many poses (fit_pose). The same for every
# kind of item: nothing here knows what a can or a mug is.
#
# The points are those around the item, the table included: where the
# camera sees the table, the item is not. Its mask alone would let the model
# slide and turn as long as it covers the mask.

MAP_CELL = 0.002        # cell of the model's height map [m]
TRUNCATE = 0.015        # a height difference counts up to this much [m]: bits of other things
BELOW_WEIGHT = 0.25     # weight of the model standing higher than the points: the camera sees
                        # into a cup or a tape roll, the collision shapes are solid


def estimate_placement(item, pts, around, table_z, basket=None, basket_xy=None):
    """How `item` rests: an Estimate (its Placement, how plausible the points
    are for this item, ...), or None when there are too few points.

    pts        the world points (N, 3) of the item's mask
    around     the world points (M, 3) around the item (its mask, the table
               near it, ...) without those of the other items and the basket
    item       the item's model (objects/*.py): .kind, .spec (items.yaml),
               .poses {resting pose: RestPose (.top: its height, .footprint)}
    basket     the basket's model (objects/basket.py), basket_xy its centre
               (None: not known; then everything is on the table)

    The support is the basket when the points are inside its rim, else the
    table. In the basket the item may lie on others: its support is then
    as high as its top less its height. The resting pose, x, y and yaw come
    from fit_pose, the best of the item's resting poses."""
    pts = np.asarray(pts, float)
    if len(pts) < 30:
        return None
    support, z_floor = "table", table_z
    if basket is not None and basket_xy is not None:
        c = np.median(pts[:, :2], axis=0)
        if math.hypot(*(c - basket_xy)) < basket.inner_radius(basket.height) - 0.005:
            support, z_floor = "basket", table_z + basket.floor
    pts = pts[pts[:, 2] > z_floor + 0.004]
    if len(pts) < 30:
        return None
    top_abs = robust_top(pts[:, 2])
    centre = np.median(pts[:, :2], axis=0)
    around = np.asarray(around, float)
    if len(around) < len(pts):        # fewer than its own: use its own
        around = pts
    best = None
    for rest in item.spec["stable_poses"]:
        z_support = z_floor if support == "table" else max(z_floor, top_abs - item.poses[rest].top)
        cost, x, y, yaw = fit_pose(item, rest, around, z_support, centre)
        if best is None or cost < best[0]:
            best = (cost, rest, x, y, yaw, z_support)
    _, rest, x, y, yaw, z_support = best
    rp = item.poses[rest]
    # Plausibility: height of the top above the support and the size of the
    # outline must match the model. The top is the mask's, or that of the
    # points around it that the model covers there if that matches better:
    # the mask may have missed part of the item (the spray bottle's grey
    # head). The outline is the mask's: a mask much larger than the item (the
    # whole basket) is not the item.
    c, s = math.cos(yaw), math.sin(yaw)
    dx, dy = around[:, 0] - x, around[:, 1] - y
    covered = model_height_map(item, rest, MAP_CELL).at(c * dx + s * dy, -s * dx + c * dy) > 0
    covered &= around[:, 2] > z_floor + 0.004
    tops = [top_abs] + ([robust_top(around[covered, 2])] if np.count_nonzero(covered) >= 30 else [])
    if support == "table":
        dh = min(abs(rp.top - (t - z_support)) for t in tops)
    else:
        # In the basket an item may rest on others (taller is fine), but it
        # cannot be shorter than its model.
        dh = min(max(0.0, rp.top - (t - z_floor)) for t in tops)
    fp = rp.footprint
    ext_model = max(np.ptp(fp[:, 0]), np.ptp(fp[:, 1]))
    _, (w, _), _ = rect_fit(pts[:, :2])
    plaus = gauss(dh, 0.025) * gauss(max(0.0, w - ext_model * 1.15), 0.03)
    p = Placement(item.name, rest, float(x), float(y), float(yaw), support=support, z_support=float(z_support))
    return Estimate(p, plaus, len(pts), top_abs, [])


def height_map(pts, z_support, cell):
    """The points seen from above: the centres (N, 2) of the cells of a grid
    `cell` wide that hold points, and the height above the support of the
    highest point in each (N,). The highest, because from above the camera
    sees an item's top; lower points in a cell are its sides."""
    ij = np.floor(pts[:, :2] / cell).astype(np.int64)
    order = np.lexsort((pts[:, 2], ij[:, 1], ij[:, 0]))      # by cell, highest last
    ij, z = ij[order], pts[order, 2]
    last = np.r_[np.any(ij[1:] != ij[:-1], axis=1), True]
    return ((ij[last] + 0.5) * cell).astype(np.float32), (z[last] - z_support).astype(np.float32)


def pose_cost(model, cells, heights, x, y, yaw):
    """How badly the model, at (x[k], y[k]) turned by yaw, explains the
    points' heights: per pose k, the mean over the cells of the squared
    difference between the points' height and the model's there, at most
    TRUNCATE (a cell may hold bits of something else), and weighted by
    BELOW_WEIGHT where the model is the higher. Returns (K,).

    model      toolkit.model_height_map(item, rest): model.at(u, v), the
               heights at points (u, v) of the item's rested frame"""
    c, s = math.cos(yaw), math.sin(yaw)
    dx = cells[None, :, 0] - np.asarray(x, np.float32)[:, None]
    dy = cells[None, :, 1] - np.asarray(y, np.float32)[:, None]
    r = heights[None] - model.at(c * dx + s * dy, -s * dx + c * dy)      # cells into the rested frame
    w = np.where(r > 0, np.float32(1.0), np.float32(BELOW_WEIGHT))
    return (w * np.minimum(r * r, np.float32(TRUNCATE ** 2))).mean(axis=1)


# The search of fit_pose, coarse to fine. Per stage: the cell of the points'
# height map [m], the steps of yaw [deg] and of x, y [m], how far from the
# poses kept by the stage before it looks (yaw [deg], x, y [m]) and how many
# poses it keeps (each the best at its yaw).
SEARCH = [(0.006, 10.0, 0.006, 180.0, 0.030, 6),
          (0.003, 2.0, 0.002, 6.0, 0.004, 3),
          (0.002, 0.5, 0.001, 2.0, 0.002, 1)]


def fit_pose(item, rest, around, z_support, centre):
    """(cost, x, y, yaw): the pose of `item` resting as `rest` on a support
    at z_support whose model best explains the heights of the points
    `around` it, near `centre` (the middle of its mask): every pose on a
    grid, coarse to fine (SEARCH). Symmetric items (a cube, a can standing)
    fit equally well at several yaws: any of them is right."""
    model = model_height_map(item, rest, MAP_CELL)
    kept = [(np.inf, centre[0], centre[1], 0.0)]
    for cell, yaw_step, step, yaw_span, span, n_keep in SEARCH:
        cells, heights = height_map(around, z_support, cell)
        g = np.arange(-span, span + 1e-9, step)
        gx, gy = [a.ravel() for a in np.meshgrid(g, g)]
        found = []
        for _, x0, y0, yaw0 in kept:
            for yaw in yaw0 + np.radians(np.arange(-yaw_span, yaw_span + 1e-9, yaw_step)):
                cost = pose_cost(model, cells, heights, x0 + gx, y0 + gy, yaw)
                k = int(np.argmin(cost))
                found.append((float(cost[k]), x0 + gx[k], y0 + gy[k], yaw))
        kept = sorted(found, key=lambda t: t[0])[:n_keep]
    cost, x, y, yaw = kept[0]
    return cost, x, y, wrap(yaw)


# ---- which mask is which item -------------------------------------------------------------


def assign_masks(candidates, min_score, max_overlap=0.5):
    """At most one mask per item and one item per mask, from the candidates
    of one frame (a list of Detection: .item, .score, .mask, ...). A name may
    match several objects, and one object several names.

    Greedy, best score first: a candidate is taken unless its item already
    has a mask, its score is below min_score, or its mask overlaps a taken
    one by more than max_overlap (toolkit.mask_overlap). Returns {item: Detection}."""
    chosen = {}
    for d in sorted(candidates, key=lambda d: -d.score):
        if d.item in chosen or d.score < min_score:
            continue
        if any(mask_overlap(d.mask, o.mask) > max_overlap for o in chosen.values()):
            continue
        chosen[d.item] = d
    return chosen


def fuse_views(per_view, weights):
    """One detection per item from several frames: per_view[k] is the
    {item: Detection} of frame k, weights[k] how much that frame counts.

    Per item, the detection with the highest score x weight, unless another
    item already stands there (same support, centres within 2.5 cm, support
    heights within 3 cm: two items cannot be in one place); the items with
    the best detections choose first. Returns {item: Detection}."""
    ranked = {}
    for k, dets in enumerate(per_view):
        for item, d in dets.items():
            ranked.setdefault(item, []).append((d.score * weights[k], d))
    fused = {}
    order = sorted(((max(r, key=lambda t: t[0])[0], it) for it, r in ranked.items()), reverse=True)
    for _, item in order:
        for w, d in sorted(ranked[item], key=lambda t: -t[0]):
            p = d.estimate.placement
            clash = any(o.estimate.placement.support == p.support and
                        np.hypot(o.estimate.placement.x - p.x, o.estimate.placement.y - p.y) < 0.025 and
                        abs(o.estimate.placement.z_support - p.z_support) < 0.03
                        for o in fused.values())
            if not clash:
                fused[item] = d
                break
    return fused


# ---- grasping ---------------------------------------------------------------------------


GRASP_BAND = 0.012      # the top of an item: points this far below its highest
ROUND_ASPECT = 1.25     # a top this little longer than wide is not elongated
SQUARE_FILL = 0.83      # a top that fills this much of its rectangle is square (a disk fills pi / 4)


def grasp_in_image(frame, mask):
    """Top-down grasps of the item whose mask this is, in the image: each
    the pixel the fingers close around and the direction they close along.
    A list of ImageGrasp, best first ([] without enough depth in the mask).

    frame  the rgbd.Frame (.rgb, .depth in metres along the view, NaN where
           unknown, .K, .T_world_cam), taken looking straight down
    mask   the item's boolean mask in that frame

    The fingers close on the top of the item: the pixels of the mask that
    are nearest to the camera, around their centre.
      * An elongated top (a can lying down, the spray bottle's head): across
        its narrow side.
      * A square top (a block): across one pair of sides, then the other.
      * A round top (a cup, a can standing, a tape roll, the mug's rim): any
        direction; first across the narrow side of the whole mask, which
        keeps the fingers off the mug's handle, then turned in steps of
        22.5 degrees (only up to 45 degrees when the whole mask is
        elongated: the mug's handle)."""
    d = np.where(mask & np.isfinite(frame.depth) & (frame.depth > 0), frame.depth, np.nan)
    if np.count_nonzero(np.isfinite(d)) < 30:
        return []
    near = np.nanpercentile(d, 3)
    top = np.isfinite(d) & (d < near + GRASP_BAND)
    vv, uu = np.nonzero(top)
    pts = np.stack([uu, vv], axis=1).astype(np.float32)
    (cu, cv), (w, h), ang = cv2.minAreaRect(pts)
    # minAreaRect: `ang` (degrees) is the direction of the side of length w.
    across = math.radians(ang) + (math.pi / 2 if w > h else 0.0)     # across the shorter side
    narrow = float(min(w, h))
    if max(w, h) >= ROUND_ASPECT * min(w, h):
        angles = [across]
    elif cv2.contourArea(cv2.convexHull(pts)) >= SQUARE_FILL * w * h:
        angles = [across, across + math.pi / 2]
    else:
        va, ua = np.nonzero(mask)
        _, (W, H), A = cv2.minAreaRect(np.stack([ua, va], axis=1).astype(np.float32))
        across = math.radians(A) + (math.pi / 2 if W > H else 0.0)
        steps = [0, 1, -1, 2, -2] if max(W, H) >= 1.15 * min(W, H) else [0, 1, -1, 2, -2, 3, -3, 4]
        angles = [across + k * math.pi / 8 for k in steps]
    return [ImageGrasp(u=float(cu), v=float(cv), angle=wrap(a), width_px=narrow) for a in angles]


TIP_CLEARANCE = 0.008   # the fingertips stay this far above the table
KNUCKLE_MARGIN = 0.004  # the item's top stays this far below the knuckles


def grasp_pose(frame, image_grasp, mask, table_z):
    """The 3D grasp for a grasp in the image: a Grasp with the pose of the
    gripper's mounting face (rg6_base_link) at the grasp, the item's width
    between the pads and the fingertips' height.

    The centre pixel and a pixel on each side along the closing direction go
    back into the world at the depth of the item's top; the closing
    direction is their horizontal direction. The fingertips go as deep as
    the knuckles allow (the item's top must stay below them:
    rg6.free_height(width)), but not lower than TIP_CLEARANCE above the
    table. The width is how far the item's points between the fingertips
    and its top extend along the closing direction. The gripper points
    straight down, its fingers closing along its y axis."""
    g = image_grasp
    d = np.where(mask & np.isfinite(frame.depth) & (frame.depth > 0), frame.depth, np.nan)
    near = np.nanpercentile(d, 3)
    top = np.isfinite(d) & (d < near + GRASP_BAND)
    z_top_depth = float(np.nanmedian(d[top]))
    du, dv = math.cos(g.angle), math.sin(g.angle)
    r = max(g.width_px / 2, 5.0)
    P = pixels_to_world([g.u, g.u - r * du, g.u + r * du], [g.v, g.v - r * dv, g.v + r * dv],
                        [z_top_depth] * 3, frame.K, frame.T_world_cam)
    centre = P[0]
    axis = P[2, :2] - P[1, :2]
    closing = math.atan2(axis[1], axis[0])
    n = np.array([math.cos(closing), math.sin(closing)])
    # The item's 3D points, and how far they extend along the closing direction.
    vv, uu = np.nonzero(np.isfinite(d))
    Q = pixels_to_world(uu, vv, d[vv, uu], frame.K, frame.T_world_cam)
    z_top = float(np.percentile(Q[:, 2], 97))

    def extent(z_min):
        t = (Q[Q[:, 2] >= z_min, :2] - centre[:2]) @ n
        lo, hi = np.percentile(t, [0.5, 99.5])      # a stray pixel or two do not count
        return float(hi - lo), float(hi + lo) / 2

    # First the width of the top, then the width between the fingertips'
    # height and the top (the pads close there: a can lying down is wider
    # below its top).
    width, _ = extent(z_top - GRASP_BAND)
    tips = max(table_z + TIP_CLEARANCE, z_top - rg6.free_height(width) + KNUCKLE_MARGIN)
    width, mid = extent(tips)
    tips = max(table_z + TIP_CLEARANCE, z_top - rg6.free_height(width) + KNUCKLE_MARGIN)
    centre[:2] += n * mid
    yaw = wrap(closing - math.pi / 2)          # the gripper's x axis; the fingers close along its y axis
    pose = gripper_pose(centre[0], centre[1], tips + rg6.depth_from_width(width), yaw)
    return Grasp(pose=pose, width=width, tips_z=float(tips), top_z=z_top)
