"""The skills: the PDDL actions of the domain, performed by the robot.

The action (name ?x ...) is performed by the function of the same name with
'-' replaced by '_', given the robot and the action's parameters:

    (pick-up red-can)        ->  pick_up(robot, "red-can")
    (put-on-table red-can)   ->  put_on_table(robot, "red-can")
    (put-in-basket red-can)  ->  put_in_basket(robot, "red-can")
    (place-in-slot red-block top)
                             ->  place_in_slot(robot, "red-block", "top")    (the pyramid)

A skill returns a message saying what it did, or raises ActionError (from
the toolkit: the action cannot be performed) or PlanError (no motion found).
It checks the action's preconditions on the gripper, looks at the scene,
decides with the state estimation (state_estimation.py) and the motion
planning (motion_planning.py), moves, and checks the action's effects on
the gripper. The robot does the rest by itself: after a skill succeeds it
returns home; after one fails having moved the arm (robot.run) with nothing
left in the gripper, it opens the gripper and rises.

The robot (autonomy/executive.py):
    robot.items, robot.basket       {name: item model}, the basket's model
    robot.cell, robot.cfg           config/cell.yaml, config/perception.yaml
    robot.table_z, robot.force      the table's height (world z), the grip force [N]
    robot.mv                        MoveIt (moveit_client.MoveIt), for motion_planning.py
    robot.held                      what the gripper holds as far as the skills know: None, or
                                    .item, .held (Held), .source (the Placement it was picked up
                                    from, or None), .assumed (not seen being picked up)
    robot.log(text)                 a line in the log
    robot.phase(text)               the current phase (in the log and the camera window)
    robot.holding()                 the gripper detects a grip (a fresh reading)
    robot.width()                   the gripper's opening now [m]
    robot.current()                 the arm's joints now
    robot.wait(seconds)             let the robot's state come in for a while
    robot.survey(view, scene=None, held=None, exclude=())
                                    look from `view` (the arm's joints, a 4x4 camera pose, or a list
                                    of them: the first reachable; state_estimation.py) and recognise
                                    what is seen, added to `scene` when given: a Scene with
                                    .detections {item: Detection}, .basket_xy, .frames (the last is
                                    this view's), .depth, .obstacles. Raises PlanError when no view
                                    is reachable
    robot.best_frame(scene, item=None)
                                    the scene's frame looking straight down from nearest, or the one
                                    `item`'s estimate comes from
    robot.grasp_mask(frame, scene, item)
                                    the pixels of `item` in `frame` to propose a grasp on
    robot.set_obstacles(scene, exclude=(), skip_slots=())
                                    MoveIt's planning scene: the table, the wall, the fixed obstacle,
                                    the pyramid's cubes (but those in the slots `skip_slots`) and what
                                    `scene` saw (scene_obstacles), without the items in `exclude`
    robot.pyramid                   the pyramid being built (pyramid.Pyramid), or None
    robot.table_region(basket_xy, depth=None)
                                    where items may be put on the table (for find_table_spot)
    robot.drop_spots(scene, held, yaws)
                                    the basket's drop spots for the held item: .spots best first,
                                    each (cost, x, y, yaw, z_pile), and .z_floor
    robot.run(steps)                perform a plan's steps (raises ActionError when one fails)
    robot.now_holding(held, source=None, assumed=False), robot.now_empty()
                                    what the gripper holds from now on (survives restarts)
    robot.placed(item, placement)   an item just put down (kept clear of on the way home)
    robot.show(frame, draw, title)  a picture in the camera window (the toolkit's show_* draw the
                                    usual ones)
After every look the robot shows by itself what it recognised.
"""

import math

from ..toolkit import (ActionError, Held, PlanError, Placement, rg6, show_drop, show_grasp, show_image_grasps,
                       show_table_spot, spread, wrap)
from . import motion_planning as mp
from . import pyramid as pyr
from . import state_estimation as se

DROP_CLEARANCE = 0.02    # dropping into the basket: the item's bottom this far above the rim or the pile
RECOGNISED = 0.5         # an item found with a lower score (SAM 3 x plausibility) is looked at again


# ---- the skills ------------------------------------------------------------------------------


def pick_up(robot, item):
    """(pick-up item): look (from the observation configuration; also from
    the side if `item` is not recognised from above), find `item`, look at
    it from closer, propose grasps in the image, plan a collision-free
    top-down grasp, grasp, lift.
    Applicable when the gripper is empty."""
    if robot.holding():
        raise ActionError(f"not applicable: the gripper is holding "
                          f"{robot.held.item if robot.held else 'something'}")
    robot.now_empty()
    scene = robot.survey(se.OBSERVATION_JOINTS)
    d = scene.detections.get(item)
    if d is None or d.score < RECOGNISED:
        robot.log(f"  {item} {'not recognised' if d is None else f'recognised with score {d.score:.2f}'}"
                  f" from above; looking from the side too")
        scene = robot.survey(se.SIDE_VIEW, scene)
    if item not in scene.detections:
        raise ActionError(f"{item} not found")
    # From closer, straight down: the item's pose is more accurate there.
    a = scene.detections[item].estimate.placement
    try:
        scene = robot.survey(se.close_views(a.x, a.y), scene)
    except PlanError as e:
        robot.log(f"  no closer look at {item} ({e}); using the survey")
    p = scene.detections[item].estimate.placement
    robot.log(f"  {item}: {p.rest} on the {p.support} at ({p.x:.3f}, {p.y:.3f}), yaw "
              f"{math.degrees(p.yaw):.0f} deg ({1000 * math.hypot(p.x - a.x, p.y - a.y):.0f} mm from the survey's)")
    # The grasps, proposed in the top-down view the item's pose comes from.
    robot.phase(f"finding a grasp of {item}")
    frame = robot.best_frame(scene, item)
    mask = robot.grasp_mask(frame, scene, item)
    image_grasps = se.grasp_in_image(frame, mask)
    if not image_grasps:
        raise ActionError(f"no grasp of {item} in the image: too few depth pixels in its mask")
    grasps = [se.grasp_pose(frame, g, mask, robot.table_z) for g in image_grasps]
    show_image_grasps(robot, frame, item, mask, image_grasps)
    g = grasps[0]
    robot.log(f"  {len(grasps)} grasp(s); the first across {g.width * 1000:.0f} mm at ({g.pose[0, 3]:.3f}, "
              f"{g.pose[1, 3]:.3f}), fingertips {1000 * (g.tips_z - robot.table_z):.0f} mm above the table")
    grasps = [g for g in grasps if g.width <= rg6.MAX_WIDTH - 0.01]
    if not grasps:
        raise ActionError(f"{item} is too wide for the RG6 across every grasp")
    robot.phase(f"planning the grasp of {item}")
    robot.set_obstacles(scene, exclude=(item,))
    steps, _, grasp, T = mp.plan_pick(robot.mv, robot.current(), grasps, robot.force)
    k = next(k for k, g in enumerate(grasps) if g is grasp)
    robot.log(f"  grasp {k + 1}: across {grasp.width * 1000:.0f} mm, fingertips "
              f"{1000 * (grasp.tips_z - robot.table_z):.0f} mm above the table")
    held = _held_from_grasp(robot, item, p, grasp, T)
    open_w = next(s["width"] for s in steps if s.get("label") == "open")
    show_grasp(robot, frame, p, held, open_w)
    robot.phase(f"picking up {item}")
    try:
        robot.run(steps)
    except ActionError as e:
        raise ActionError(f"grasp failed: {e}")
    if not robot.holding():
        if abs(robot.width() - held.width) < 0.012:
            robot.log(f"  grip flag off but the fingers are {robot.width() * 1000:.0f} mm apart: holding {item}")
        else:
            raise ActionError(f"{item} slipped out of the gripper")
    robot.now_holding(held, source=p)
    return f"holding {item} (width {robot.width() * 1000:.0f} mm)"


def put_on_table(robot, item, target_xy=None):
    """(put-on-table item): look, put the held `item` down on the free table
    spot nearest to where it was picked up (or to target_xy). Applicable
    when the gripper holds something."""
    held = _held(robot, item)
    scene = robot.survey(se.OBSERVATION_JOINTS, held=held, exclude=(item,))
    robot.phase(f"planning where to put {item}")
    robot.set_obstacles(scene, exclude=(item,))
    target, steps = _plan_table_spot(robot, robot.current(), held, scene, target_xy)
    return _put(robot, item, target, steps)


def put_in_basket(robot, item):
    """(put-in-basket item): look, find the basket and how high things are
    piled in it, carry the held `item` above its emptiest spot and drop it
    there (it may land on other items). Applicable when the gripper holds
    something."""
    held = _held(robot, item)
    scene = robot.survey(se.OBSERVATION_JOINTS, held=held, exclude=(item,))
    robot.phase(f"planning where to put {item}")
    robot.set_obstacles(scene, exclude=(item,))
    target, steps = _plan_drop(robot, robot.current(), held, scene)
    return _put(robot, item, target, steps)


def place_in_slot(robot, item, slot):
    """(place-in-slot item slot): look, check that the cubes put in the pyramid
    before are still there, and put the held cube in `slot` of the pyramid
    (exercise/pyramid.py), on the cubes it rests on, aligned with the row.
    Applicable when the gripper holds a cube."""
    pyramid = robot.pyramid
    if pyramid is None:
        raise ActionError("no pyramid: start it with pyramid: in the goal console")
    if slot not in pyr.SLOTS:
        raise ActionError(f"{slot} is not a slot of the pyramid ({', '.join(pyr.SLOTS)})")
    if slot in pyramid.placed:
        raise ActionError(f"not applicable: {slot} is taken by {pyramid.placed[slot][0]}")
    missing = [s for s in pyramid.supports(slot) if s not in pyramid.placed]
    if missing:
        raise ActionError(f"not applicable: {slot} rests on {' and '.join(missing)}, which are empty")
    held = _held(robot, item)
    scene = robot.survey(se.OBSERVATION_JOINTS, held=held, exclude=(item,))
    gone = pyr.check_cubes(pyramid, lambda X, Y: scene.depth.heights_at(X, Y, radius=0.01, items=tuple(scene.detections)))
    if gone:
        raise ActionError(f"the pyramid is not as it was left: nothing at {', '.join(gone)} (a cube fell?)")
    robot.phase(f"planning where to put {item} in the pyramid")
    robot.set_obstacles(scene, exclude=(item,), skip_slots=pyramid.supports(slot))
    x, y, z_support = pyramid.target(slot)
    # The fingers close across the row, so that they do not meet the neighbours: the gripper's yaw
    # along the row (either way) first; the cube is as turned as it was grasped.
    axis = pyramid.layout.axis_yaw
    yaws = [wrap(psi + held.yaw_rel) for psi in (axis, axis + math.pi)]
    yaws += [wrap(axis + k * math.pi / 2) for k in range(4)]
    model = robot.items[item]
    yaw, steps = mp.plan_place_at(robot.mv, robot.current(), held, mp.held_box(held, model), x, y, z_support, yaws,
                                  robot.force)
    robot.log(f"  {item} into {slot}: ({x:.3f}, {y:.3f}), on a support {z_support * 100:.1f} cm high")
    target = Placement(item, held.rest, float(x), float(y), float(yaw), support="table", z_support=float(z_support))
    msg = _put(robot, item, target, steps)
    pyramid.put(slot, item, x, y)
    return f"{msg} (the {slot} slot)"


# ---- picking up --------------------------------------------------------------------------------


def _held_from_grasp(robot, item, p, grasp, T):
    """How `item`, resting as placement p, is held after the grasp with the
    gripper at pose T (Held)."""
    gx, gy = T[0, 3], T[1, 3]
    psi = math.atan2(T[1, 0], T[0, 0])
    c, s = math.cos(p.yaw), math.sin(p.yaw)
    dx, dy = gx - p.x, gy - p.y
    offset = (c * dx + s * dy, -s * dx + c * dy)       # in the item's rested frame
    return Held(item, p.rest, float(grasp.width), float(grasp.tips_z - p.z_support),
                (float(offset[0]), float(offset[1])), float(wrap(p.yaw - psi)),
                float(robot.items[item].poses[p.rest].top))


# ---- putting down -------------------------------------------------------------------------------


def _held(robot, item):
    """How the gripper holds `item` (Held), checking that it holds something.
    When the robot did not see it being picked up, its nominal grasp is assumed."""
    if not robot.holding():
        raise ActionError("not applicable: the gripper is empty")
    if robot.held is None or robot.held.item != item:
        if robot.held is not None and not robot.held.assumed:
            raise ActionError(f"the gripper holds {robot.held.item}, not {item}")
        robot.now_holding(_assumed_held(robot, item), assumed=True)
        robot.log(f"  assuming {item} is held at its nominal grasp")
    return robot.held.held


def _assumed_held(robot, item):
    """How `item` is probably held when the robot did not pick it up: its
    model's grasp (of any resting pose) nearest to the gripper's width."""
    obj = robot.items[item]
    w = robot.width()
    best = None
    for rest in obj.spec["stable_poses"]:
        for g in obj.poses[rest].grasps:
            d = abs(g.width - w)
            if best is None or d < best[0]:
                best = (d, rest, g)
    _, rest, g = best
    yaw_rel = wrap(math.pi / 2 - (g.yaws[0] if not g.continuous else 0.0))
    return Held(item, rest, g.width, g.tip_z, tuple(g.center), yaw_rel, obj.poses[rest].top)


def _put(robot, item, target, steps):
    """Perform the put planned (target placement, steps) and check that the
    gripper let go."""
    robot.log(f"  putting {item} at ({target.x:.3f}, {target.y:.3f}) on the {target.support}")
    robot.phase(f"putting {item} {'in the basket' if target.support == 'basket' else 'on the table'}")
    robot.run(steps)
    robot.wait(0.3)
    if robot.holding():
        raise ActionError(f"{item} is still in the gripper after release")
    robot.now_empty()
    robot.placed(item, target)
    return f"put {item} at ({target.x:.3f}, {target.y:.3f}) on the {target.support}"


def _plan_table_spot(robot, q0, held, scene, target_xy=None):
    """(target placement, steps): put the held item down on the free table
    spot nearest to target_xy, else to where it was picked up."""
    region = robot.table_region(scene.basket_xy, getattr(scene, "depth", None))
    source = robot.held.source
    prefer = target_xy if target_xy is not None and len(target_xy) == 2 else (
        (source.x, source.y) if source is not None else (0.45, -0.10))
    model = robot.items[held.item]
    # What the item must not be put on: the other items, the unknown things seen.
    others = [robot.items[k].footprint_world(d.estimate.placement.rest, d.estimate.placement.x,
                                             d.estimate.placement.y, d.estimate.placement.yaw)
              for k, d in scene.detections.items() if k != held.item]
    others += [o.polygon for o in getattr(scene, "obstacles", []) if o.support == "table"]
    target, steps = mp.find_table_spot(robot.mv, q0, held, model, mp.held_box(held, model), region, others,
                                       tuple(prefer), robot.table_z, robot.force,
                                       margin=robot.cell["placement"]["object_clearance"],
                                       max_tries=robot.cfg.get("put_max_tries", 80))
    show_table_spot(robot, robot.best_frame(scene), scene, region, target, source)
    return target, steps


def _plan_drop(robot, q0, held, scene):
    """(target placement, steps): drop the held item above the emptiest spot
    of the basket (it may land on what is there). Only the motion can fail,
    so the next best spots are tried, then higher releases."""
    yaws = [wrap(k * math.pi / 4 + held.yaw_rel) for k in range(-4, 4)]
    drops = robot.drop_spots(scene, held, yaws)
    rim = robot.table_z + robot.basket.height
    z_floor = robot.table_z + robot.basket.floor
    box = mp.held_box(held, robot.items[held.item])
    errors = []
    # The best spots with the yaw that fits there, then any yaw; then higher.
    tries = [(s, [s[3]]) for s in spread(drops.spots)] + [(s, yaws) for s in spread(drops.spots, 4)]
    for raise_ in (0.0, 0.04):
        for (cost, x, y, yaw0, z_pile), only in tries:
            z_support = max(z_floor, z_pile)
            gap = max(rim, z_pile) + DROP_CLEARANCE + raise_ - z_support
            for yaw in only:
                try:
                    steps = mp.plan_place(robot.mv, q0, held, box, x, y, yaw, z_support, gap, robot.force,
                                          approach=mp.DROP_APPROACH)
                except PlanError as e:
                    errors.append(str(e))
                    continue
                target = Placement(held.item, held.rest, float(x), float(y), float(yaw), support="basket",
                                   z_support=float(z_support))
                z_drop = z_support + gap
                pile = z_pile - drops.z_floor
                robot.log(f"  dropping from {z_drop * 100:.0f} cm onto "
                          + ("the floor" if pile < 0.01 else f"a pile {pile * 100:.0f} cm high"))
                show_drop(robot, robot.best_frame(scene), scene, drops, target, z_drop)
                return target, steps
    if errors and all("out of the arm's reach" in e for e in errors):
        raise PlanError(f"every drop spot is out of the arm's reach with {held.item} (the basket is too far "
                        f"for it)")
    raise PlanError("no drop into the basket: " + ("; ".join(sorted(set(errors))[:3]) or "no drop spot"))
