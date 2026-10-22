"""Motion planning: the robot's motions for the camera views and the three
PDDL actions, planned with MoveIt.

`moveit` is a lab_replica.moveit_client.MoveIt: it plans with MoveIt's
move_group, collision-free with respect to what is in its planning scene
(the table and the wall, plus whatever set_obstacles added):

    moveit.plan_to_joints(q_start, q_goal, width, held=None)       -> joint path
    moveit.plan_to_pose(q_start, T_goal, width, held=None)          -> joint path
    moveit.cartesian_path(q_start, [T, ...], width, held=None)      -> joint path (straight lines)
    moveit.is_valid(q, width, held=None)                            -> bool
    moveit.contacts(q, width, held=None)                            -> what touches what there
    moveit.set_obstacles([Box, ...])

A joint path is a list of arrays of the six arm joints. Poses T are 4x4
world poses of rg6_base_link, the RG6's mounting face; `width` is the
gripper's opening (the fingers are part of the robot), `held` the box of
the item in the gripper in the gripper's frame (held_box), which then
moves with it. Everything raises toolkit.PlanError when no motion is found.

A plan for the robot is a list of steps (toolkit.trajectory,
toolkit.gripper_step), which the executive runs in order.

Conventions: the gripper points straight down for grasping and placing;
its yaw is the direction of its x axis in the world, the fingers close
along its y axis (toolkit.gripper_pose). The fingertips are
rg6.depth_from_width(width) below the mounting face.
"""

import math

import numpy as np

from ..toolkit import (ARM_JOINTS, Box, Cylinder, PlanError, Placement, finger_footprints, fk, gripper_pose, gripper_step,
                       ik_all, joint_limits, object_box, placement_pose, polygon_inside_rect, polygons_intersect,
                       rg6, rpy_to_matrix, trajectory, wrap)

OPEN_MARGIN = 0.016         # open this much wider than the item on each side before closing
SQUEEZE = 0.025             # close this much below the item's width (the force limits the grip)
APPROACH = 0.12             # the gripper comes down to a grasp (and leaves) along this last stretch
PLACE_APPROACH = 0.10       # ... and to a place on the table
DROP_APPROACH = 0.06        # ... and to a drop into the basket (a tall item is dropped from high up)
RELEASE_GAP = 0.004         # an item put on the table is released this far above it
SLOW = 0.06                 # the last (first) part of a descent (lift) that is slow
RISE_Z = 0.60               # plan_path: the height the mounting face rises to before a straight line
RISE_CLEARANCE = 0.10       # ... and this much above a barrier (scene_model.py) taller than that
RRT_RETRY_TIME = 15.0       # plan_path: planning time [s] of RRT-Connect's second try
JOINT_WEIGHTS = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.3])   # turning the last wrist joint is harmless


# ---- inverse kinematics and frames ----------------------------------------------------------


def choose_solution(solutions, q_ref):
    """Of the inverse kinematics solutions (angles in [-pi, pi]), the one
    nearest to q_ref, or None. A joint angle and the same angle plus or minus
    2 pi put the arm in the same place: use the one nearest to q_ref within
    the joint limits (toolkit.joint_limits). Distance: the largest weighted
    joint change (JOINT_WEIGHTS)."""
    q_ref = np.asarray(q_ref, float)
    limits = joint_limits()
    best, best_d = None, math.inf
    for q in solutions:
        q = np.array(q, float)
        for i, name in enumerate(ARM_JOINTS):
            lo, hi = limits[name]
            options = [q[i] + k * 2 * math.pi for k in (-2, -1, 0, 1, 2)]
            options = [a for a in options if lo - 1e-9 <= a <= hi + 1e-9]
            if not options:
                break
            q[i] = min(options, key=lambda a: abs(a - q_ref[i]))
        else:
            d = float((np.abs(q - q_ref) * JOINT_WEIGHTS).max())
            if d < best_d:
                best, best_d = q, d
    return best


def camera_to_gripper(T_world_cam, T_grip_cam):
    """The pose of rg6_base_link that puts the camera's optical frame at
    T_world_cam; T_grip_cam is the camera's pose in the gripper's frame."""
    return T_world_cam @ np.linalg.inv(T_grip_cam)


def held_box(held, item):
    """The box of a held item in the gripper's frame (a Box: centre c,
    rotation R, half sizes h), for MoveIt to move with the gripper.

    held   how it is held (toolkit.Held): its resting pose `rest`, `width`
           between the pads, `tip_z` the fingertips' height above its support
           when grasped, `offset_obj` the grasp centre in the item's rested
           frame, `yaw_rel` the item's yaw minus the gripper's
    item   the item's model: item.poses[rest] (.rpy, .z: the item frame on
           its support), and toolkit.object_box(item, xyz, R) its box for a pose

    It was grasped top-down; in the gripper's frame (z pointing down from the
    mounting face), the item's support was at depth + tip_z."""
    rest = item.poses[held.rest]
    depth = rg6.depth_from_width(held.width)
    c, s = math.cos(held.yaw_rel), math.sin(held.yaw_rel)
    ox, oy = held.offset_obj
    # The item frame in a frame at the grasp centre with z up and x along the gripper's x.
    p_up = np.array([-(c * ox - s * oy), -(s * ox + c * oy), -depth - held.tip_z + rest.z])
    R_up = rpy_to_matrix(0, 0, held.yaw_rel) @ rpy_to_matrix(*rest.rpy)
    G = rpy_to_matrix(math.pi, 0, 0)          # that frame -> the gripper's (pointing down)
    return object_box(item, G.T @ p_up, G.T @ R_up, held.item)


# ---- moving -----------------------------------------------------------------------------


def joint_line(q0, q1, step=0.02):
    """Configurations along the straight line in joint space from q0 to q1,
    no joint moving more than `step` (rad) between two of them."""
    q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
    n = max(2, int(math.ceil(np.abs(q1 - q0).max() / step)) + 1)
    return [q0 + (q1 - q0) * k / (n - 1) for k in range(n)]


def straight_path(moveit, q0, q1, width, held=None):
    """The straight line in joint space from q0 to q1 (joint_line) when no
    configuration on it touches anything (moveit.is_valid), else None. The
    shortest way when nothing is in it; MoveIt's planner is for the others."""
    path = joint_line(q0, q1)
    if all(moveit.is_valid(q, width, held) for q in path[1:]):
        return path
    return None


def rise_z(moveit):
    """The height plan_path rises to: RISE_Z, or RISE_CLEARANCE above the
    barrier (scene_model.py; `moveit.barrier_top`) when that is higher."""
    top = getattr(moveit, "barrier_top", None)
    return RISE_Z if top is None else max(RISE_Z, top + RISE_CLEARANCE)


def _note(moveit, text):
    log = getattr(moveit, "log", None)
    if log is not None:
        log(f"  motion: {text}")


def plan_path(moveit, q0, q1, width, held=None):
    """A collision-free joint path from q0 to q1, the simplest that is free:
    the straight line in joint space; else straight up first (to rise_z, a
    Cartesian path), then the straight line; else MoveIt's planner
    (RRT-Connect), which finds a way around but not always a short one, tried
    a second time for longer. Which of them it was is logged."""
    path = straight_path(moveit, q0, q1, width, held)
    if path is not None:
        _note(moveit, "straight line in joint space")
        return path
    T0 = fk(q0)
    z = rise_z(moveit)
    if T0[2, 3] < z - 0.01:
        try:
            up = moveit.cartesian_path(q0, [lifted(T0, z - T0[2, 3])], width, held)
            rest = straight_path(moveit, up[-1], q1, width, held)
            if rest is not None:
                _note(moveit, f"straight up to {z:.2f} m, then a straight line")
                return up + rest[1:]
        except PlanError:
            pass
    try:
        path = moveit.plan_to_joints(q0, q1, width, held)
        _note(moveit, "MoveIt's planner (RRT-Connect)")
    except PlanError as e:
        _note(moveit, f"RRT-Connect found no way ({e}); again, for longer")
        path = moveit.plan_to_joints(q0, q1, width, held, planning_time=RRT_RETRY_TIME)
        _note(moveit, "MoveIt's planner (RRT-Connect, second try)")
    return path


def move_to_pose(moveit, q0, T, width, held=None):
    """A collision-free joint path from q0 to the gripper pose T. The goal is
    an inverse kinematics solution: the one nearest to q0 first (the arm
    keeps its elbow and wrist the way they are), then the others, skipping
    those in collision. Raises PlanError at once when T is out of reach."""
    solutions = ik_all(T)
    if not solutions:
        raise PlanError(f"the pose at {np.round(T[:3, 3], 3)} is out of the arm's reach")
    touching, errors = set(), []
    while solutions:
        q_goal = choose_solution(solutions, q0)
        if q_goal is None:
            break
        # The others (the same angles up to multiples of 2 pi are the same solution).
        solutions = [q for q in solutions if np.abs((q - q_goal + math.pi) % (2 * math.pi) - math.pi).max() > 1e-6]
        if not moveit.is_valid(q_goal, width, held):
            touching |= set(moveit.contacts(q_goal, width, held))
            continue
        try:
            return plan_path(moveit, q0, q_goal, width, held)
        except PlanError as e:
            errors.append(str(e))
    if touching:
        errors.append("the robot would touch " + ", ".join(sorted(touching)))
    raise PlanError(f"no motion to the pose at {np.round(T[:3, 3], 3)}: " + "; ".join(sorted(set(errors))))


def plan_view(moveit, q0, T_cam, T_grip_cam, width, held=None, label="look"):
    """The steps that move the camera to T_cam."""
    path = move_to_pose(moveit, q0, camera_to_gripper(T_cam, T_grip_cam), width, held)
    return [trajectory(label, path)]


def plan_observation(moveit, q0, q_obs, width, held=None, label="look"):
    """The steps that move the arm to the configuration q_obs a survey looks
    from (state_estimation.OBSERVATION_JOINTS, ...). Raises PlanError saying
    what the arm would touch there."""
    if not moveit.is_valid(q_obs, width, held):
        raise PlanError("at the configuration to look from the robot would touch "
                        + (", ".join(moveit.contacts(q_obs, width, held)) or "something"))
    return [trajectory(label, plan_path(moveit, q0, q_obs, width, held))]


def plan_home(moveit, q0, q_home, width, held=None):
    """The steps back to the home configuration. When what the gripper holds
    would touch something there (a tall item hangs low), to the home pose
    raised (2 cm steps, up to 30 cm) instead."""
    if moveit.is_valid(q_home, width, held):
        return [trajectory("home", plan_path(moveit, q0, q_home, width, held))]
    T = fk(q_home)
    for dz in np.arange(0.02, 0.301, 0.02):
        q = choose_solution(ik_all(lifted(T, dz)), q_home)
        if q is not None and moveit.is_valid(q, width, held):
            return [trajectory("home (raised: the held item would touch something)",
                               plan_path(moveit, q0, q, width, held))]
    raise PlanError("no free home pose for what the gripper holds")


def lifted(T, dz):
    T = np.array(T, float)
    T[2, 3] += dz
    return T


def turned(T):
    """The same gripper pose turned half a turn about the vertical: the
    RG6 is symmetric, so it grasps the same way."""
    return T @ np.diag([-1.0, -1.0, 1.0, 1.0])


def plan_pick(moveit, q0, grasps, force):
    """The steps of a top-down grasp: to above the grasp, open, straight down,
    close, straight up. `grasps` are toolkit.Grasp (the pose, the item's
    width), best first: the first one that can be planned is taken, each
    with the gripper facing both ways (it is symmetric), the way the arm
    turns less to first. Returns (steps, q_end, the grasp, the gripper pose
    used)."""
    def turn(T):
        q = choose_solution(ik_all(lifted(T, APPROACH)), q0)
        return math.inf if q is None else float(np.abs(q - q0).max())

    errors = []
    for grasp in grasps:
        open_w = min(grasp.width + 2 * OPEN_MARGIN, rg6.MAX_WIDTH)
        close_w = max(0.0, grasp.width - SQUEEZE)
        for T in sorted((grasp.pose, turned(grasp.pose)), key=turn):
            # Open, the fingertips are higher than at the grasp; closing, they
            # go deeper (the RG6's parallelogram): the arm goes down to the
            # grasp pose itself and stays there while the fingers close.
            T_above = lifted(T, APPROACH)
            try:
                to_above = move_to_pose(moveit, q0, T_above, open_w)
                down = moveit.cartesian_path(to_above[-1], [T], open_w)
                up = moveit.cartesian_path(down[-1], [T_above], grasp.width)
            except PlanError as e:
                errors.append(str(e))
                continue
            steps = [trajectory("move above grasp", to_above),
                     gripper_step("open", open_w, force),
                     trajectory("descend", down, slow_last=SLOW),
                     gripper_step("grasp", close_w, force, expect_grip=True),
                     trajectory("lift", up, slow_first=SLOW)]
            return steps, up[-1], grasp, T
    raise PlanError("cannot pick: " + ("; ".join(sorted(set(errors))) or "no grasp"))


def gripper_for(held, x, y, yaw):
    """The gripper's (x, y, yaw) that puts the held item at (x, y, yaw)."""
    c, s = math.cos(yaw), math.sin(yaw)
    ox, oy = held.offset_obj
    return x + c * ox - s * oy, y + s * ox + c * oy, wrap(yaw - held.yaw_rel)


def plan_place(moveit, q0, held, box, x, y, yaw, z_support, gap, force, approach=PLACE_APPROACH, label="lower"):
    """The steps that put the held item down at (x, y, yaw), its bottom `gap`
    above z_support (the height of what it will rest on): carry it to
    `approach` above there, straight down, open, straight up. `box` is its
    held_box.

    At the edge of the arm's reach (a tall item held high, over the far
    side of the basket) the pose above may be out of reach: then the item
    is carried to the release pose directly, and the gripper rises after
    the release as far as the arm reaches."""
    gx, gy, psi = gripper_for(held, x, y, yaw)
    release_w = min(held.width + 2 * OPEN_MARGIN, rg6.MAX_WIDTH)
    T = gripper_pose(gx, gy, z_support + gap + held.tip_z + rg6.depth_from_width(held.width), psi)
    above = lifted(T, approach)
    steps = []
    if ik_all(above):
        carry = move_to_pose(moveit, q0, above, held.width, box)
        down = moveit.cartesian_path(carry[-1], [T], held.width, box)
        steps += [trajectory("carry", carry), trajectory(label, down, slow_last=SLOW)]
    else:
        down = move_to_pose(moveit, q0, T, held.width, box)
        steps += [trajectory("carry", down, slow_last=SLOW)]
    if not moveit.is_valid(down[-1], release_w):
        raise PlanError("the fingers cannot open there")
    steps.append(gripper_step("release", release_w, force, drop=round(gap, 3)))
    up = moveit.cartesian_path(down[-1], [above], release_w, min_fraction=0.0)
    if len(up) > 1:
        steps.append(trajectory("retreat", up, slow_first=0.05))
    return steps


# ---- where to put an item down ------------------------------------------------------------


def sample_spots(region, prefer, step=0.01):
    """Candidate centres (x, y) on a grid of `step` over the region's
    x_range and y_range, nearest to `prefer` (x, y) first."""
    xs = np.arange(region["x_range"][0], region["x_range"][1] + 1e-9, step)
    ys = np.arange(region["y_range"][0], region["y_range"][1] + 1e-9, step)
    spots = [(float(x), float(y)) for x in xs for y in ys]
    spots.sort(key=lambda c: (c[0] - prefer[0]) ** 2 + (c[1] - prefer[1]) ** 2)
    return spots


def is_free(footprint, region, obstacles, margin):
    """Whether an item with this footprint (polygon (N, 2), world) may be
    put there: inside the region's x_range and y_range, outside its keepout
    circles [(x, y, r)], accepted by its `accept` (if any), and more than
    `margin` away from every obstacle (polygons (M, 2))."""
    if not polygon_inside_rect(footprint, *region["x_range"], *region["y_range"]):
        return False
    if any(math.hypot(v[0] - cx, v[1] - cy) < cr for v in footprint for cx, cy, cr in region.get("keepout", [])):
        return False
    return not any(polygons_intersect(footprint, o, margin) for o in obstacles)


def find_table_spot(moveit, q0, held, item, box, region, obstacles, prefer, table_z, force, margin=0.02,
                    max_tries=80):
    """Put the held item on the table: the free spot nearest to `prefer`
    that the robot can reach. Spots (sample_spots) and the gripper's eight
    yaws (45 degrees apart) are tried; a spot is free when the item's
    footprint and the fingers' (toolkit.finger_footprints, open to release)
    are (is_free; the fingers only with the obstacles). The first that
    plan_place plans is taken; at most max_tries are planned. `region` may
    have accept(footprint, top z) -> bool. Returns (Placement, steps)."""
    rest = item.poses[held.rest]
    release_w = min(held.width + 2 * OPEN_MARGIN, rg6.MAX_WIDTH)
    tries, last = 0, "no free spot"
    for x, y in sample_spots(region, prefer):
        for k in range(-4, 4):
            yaw = wrap(k * math.pi / 4 + held.yaw_rel)
            fp = item.footprint_world(held.rest, x, y, yaw)
            if not is_free(fp, region, obstacles, margin):
                continue
            accept = region.get("accept")
            if accept is not None and not accept(fp, table_z + rest.top):
                continue
            gx, gy, psi = gripper_for(held, x, y, yaw)
            if any(polygons_intersect(f, o, margin) for f in finger_footprints(gx, gy, psi, release_w)
                   for o in obstacles):
                continue
            tries += 1
            if tries > max_tries:
                raise PlanError(f"no place on the table for {held.item}: {last}")
            try:
                steps = plan_place(moveit, q0, held, box, x, y, yaw, table_z, RELEASE_GAP, force)
            except PlanError as e:
                last = str(e)
                continue
            return Placement(held.item, held.rest, x, y, yaw, support="table", z_support=table_z), steps
    raise PlanError(f"no place on the table for {held.item}: {last}")


def plan_place_at(moveit, q0, held, box, x, y, z_support, yaws, force, gap=RELEASE_GAP):
    """The first of `yaws` (the item's, best first) at which the held item can
    be put down at (x, y) on a support z_support high, as plan_place plans it.
    Returns (yaw, steps); raises PlanError saying why none could be."""
    errors = []
    for yaw in yaws:
        try:
            return yaw, plan_place(moveit, q0, held, box, x, y, yaw, z_support, gap, force)
        except PlanError as e:
            errors.append(str(e))
    raise PlanError("no way to put it down there: " + "; ".join(sorted(set(errors))[:3]))


# ---- what MoveIt keeps clear of -------------------------------------------------------------


ITEM_PAD = 0.005            # the items' boxes are this much larger on every side
BASKET_PAD = 0.01


def scene_obstacles(detections, items, basket, basket_xy, table_z, unknown=(), exclude=()):
    """Boxes (and a Cylinder: the basket) in the world frame for MoveIt's
    planning scene, besides the table and the wall: every detected item but
    those in `exclude` (the one to pick, the one held), the basket, and what
    the depth camera saw that no item explains.

    detections  {item: Detection}: d.estimate.placement says where it rests
    items       {item: model}; toolkit.placement_pose(model, placement) -> (xyz, R)
    unknown     depth_world.Obstacle: .polygon (N, 2), .z_top, .unseen. An
                unknown thing, or space no view saw (behind a tall item:
                maybe an item nobody recognised), up to the height the
                camera's rays bound it to; either may be in the way."""
    boxes = []
    for name, d in detections.items():
        if name in exclude:
            continue
        model = items[name]
        xyz, R = placement_pose(model, d.estimate.placement)
        b = object_box(model, xyz, R, name)
        boxes.append(Box(name, b.c, b.R, b.h + ITEM_PAD))
    if basket_xy is not None:
        boxes.append(Cylinder("basket", [basket_xy[0], basket_xy[1], table_z + basket.height / 2],
                              basket.r_top + BASKET_PAD, basket.height / 2 + BASKET_PAD / 2))
    for k, o in enumerate(unknown):
        lo, hi = o.polygon.min(axis=0), o.polygon.max(axis=0)
        z0 = table_z
        boxes.append(Box(f"unknown {k}", [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (z0 + o.z_top) / 2], np.eye(3),
                         [(hi[0] - lo[0]) / 2 + ITEM_PAD, (hi[1] - lo[1]) / 2 + ITEM_PAD, (o.z_top - z0) / 2]))
    return boxes
