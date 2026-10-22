"""Bonus: items that are not in config/items.yaml, added while a console runs.

    new-item: purple-block prompt="purple block"   ->  new_item(robot, "purple-block", "purple block")
    new-item: lid shape=round                      ->  new_item(robot, "lid", "lid", "round")
    new-item: white-block size=0.05                ->  new_item(robot, "white-block", "white block", None, 0.05)
    forget-item: purple-block                      ->  forget_item(robot, "purple-block")

new_item returns (spec, message) or raises ActionError / PlanError, like
the skills (skills.py); after it succeeds the robot returns home by itself.
Added items last until the console quits.

Besides what skills.py describes, the robot (autonomy/executive.py) has:
    robot.segment(frame, prompt)     SAM 3's masks of a text prompt in `frame`, best first: each
                                     with .score and .mask (shown in the camera window)
    robot.points(frame, mask, erode_px=None)
                                     the world points (N, 3) of a mask's pixels, above the table and
                                     away from the robot (eroded by perception.yaml's mask_erode_px)
    robot.points_around(frame, mask, model, others=())
                                     the world points around a mask, table included (what
                                     estimate_item fits the model to), without the pixels of the
                                     masks `others`
    robot.add_item(name, spec)       know `name` from now on: its model and its SAM 3 prompts
    robot.remove_item(name)          forget it
"""

import math
import time

import numpy as np

from ..toolkit import (ActionError, PlanError, estimate_item, make_item, mask_overlap, measure_item, rg6,
                       show_new_item)
from . import state_estimation as se

MIN_POINTS = 30          # depth points a mask needs to count as an item


def new_item(robot, name, prompt, shape=None, size=None):
    """Find an item that is not known yet on the table by its text prompt,
    measure it (shape "box" or "round", or None to tell from its points) and
    add it as `name`. Returns (spec, message); the spec is an
    objects/measured.py spec.

    With `size` (the edge of a wooden cube, as the block items of items.yaml)
    it is a cube of exactly that size: it is only looked for, not measured
    (a few mm of error in a measurement would add up in a pyramid of
    cubes); the spec is an objects/block.py spec."""
    if robot.holding():
        raise ActionError("the gripper is holding something: put it down first, then measure the new item")
    scene = robot.survey(se.OBSERVATION_JOINTS)
    robot.phase(f"looking for {prompt!r} (SAM 3)")
    found = None
    for f in scene.frames:
        hit = _find_new(robot, f, prompt, scene.detections, scene.basket_xy)
        if hit is not None and (found is None or hit[0] > found[0]):
            found = hit + (f,)
    if found is None:
        raise ActionError(f"nothing on the table looks like {prompt!r} to SAM 3 "
                          f"(try another description: prompt=\"...\")")
    score, mask, pts, frame = found
    c = np.median(pts[:, :2], axis=0)
    robot.log(f"  {prompt!r} found with score {score:.2f} at ({c[0]:.3f}, {c[1]:.3f})")
    notes = []
    if size is not None:
        color = np.median(frame.rgb[mask], axis=0) / 255.0
        spec = {"kind": "block", "size": float(size), "mass": 0.075,
                "color": [round(float(v), 3) for v in color], "stable_poses": ["flat"], "prompts": [prompt]}
        robot.add_item(name, spec)
        robot.log(f"  {name}: a cube of {size * 1000:.0f} mm (given, not measured)")
        return spec, f"{name} is a {size * 1000:.0f} mm cube, SAM 3 score {score:.2f}"
    try:
        close = robot.survey(se.close_views(c[0], c[1]), scene).frames[-1]
    except PlanError:
        close = None
    if close is not None:
        robot.phase(f"measuring {name} from closer")
        hit = _find_new(robot, close, prompt, {}, scene.basket_xy, near=c)
        if hit is not None:
            score, mask, pts = hit
            frame = close
        else:
            notes.append("not found again from closer: measured from the survey")
    else:
        notes.append("no closer view reachable: measured from the survey")
    m = measure_item(pts, robot.table_z, shape)
    if m is None:
        raise ActionError(f"too few depth points on {prompt!r} to measure it")
    shape, size, more = m
    notes += more
    width, height = (size[0] if shape == "round" else size[1]), size[-1]
    if width > rg6.MAX_WIDTH - 0.01:
        raise ActionError(f"{name} is {width * 1000:.0f} mm wide, too wide for the RG6 "
                          f"(it opens {rg6.MAX_WIDTH * 1000:.0f} mm)")
    if height > robot.cfg["max_item_height"]:
        raise ActionError(f"{name} is {height * 100:.0f} cm tall; items may be at most "
                          f"{robot.cfg['max_item_height'] * 100:.0f} cm (perception.yaml max_item_height)")
    if height < 0.015:
        notes.append(f"only {height * 1000:.0f} mm tall: the pads stop 8 mm above the table "
                     f"(objects/base.py TIP_CLEARANCE) and hold little of it")
    color = np.median(frame.rgb[mask], axis=0) / 255.0
    spec = {"kind": "measured", "shape": shape, "size": [float(v) for v in size], "mass": 0.1,
            "color": [round(float(v), 3) for v in color],
            "stable_poses": ["upright" if shape == "round" else "flat"], "prompts": [prompt],
            "measured": time.strftime("%Y-%m-%d %H:%M")}
    # The new model must explain the points it was made from.
    model = make_item(name, spec)
    others = [d.mask for d in scene.detections.values() if d.frame is frame]
    est = estimate_item(model, pts, robot.points_around(frame, mask, model, others), robot.table_z, robot.basket,
                        scene.basket_xy)
    if est is None or est.plausibility < 0.5:
        notes.append(f"its model fits its own points badly (plausibility "
                     f"{0 if est is None else est.plausibility:.2f})")
    dims = " x ".join(f"{v * 1000:.0f}" for v in size) + " mm"
    what = f"{shape} {dims}" + (" (diameter x height)" if shape == "round" else " (length x width x height)")
    show_new_item(robot, frame, mask, model, est.placement if est else None, dims, f"new item {name}: {what}",
                  f"{prompt!r}, SAM 3 score {score:.2f}; magenta: its mask, green: the model made from it")
    robot.log(f"  {name}: {what}; " + "; ".join(notes))
    robot.add_item(name, spec)
    msg = f"{name} is a {what}, SAM 3 score {score:.2f}"
    return spec, msg + "".join(f"\n    note: {n}" for n in notes)


def forget_item(robot, name):
    """Forget the item `name` added with new_item."""
    robot.remove_item(name)


def _find_new(robot, frame, prompt, known, basket_xy, near=None):
    """(SAM score, mask, world points) of the best mask of `prompt` in
    `frame` on the table, or None. Masks of known items ({item: Detection})
    that SAM 3 named better are skipped; `near` (x, y) keeps only masks
    within 5 cm of it."""
    best = None
    for inst in robot.segment(frame, prompt):
        if any(d.frame is frame and d.sam_score >= inst.score and mask_overlap(d.mask, inst.mask) > 0.5
               for d in known.values()):
            continue
        # Not eroded: the size is read from the mask's outline (eroding
        # 1 px made a 60 x 30 mm block 56 x 27 mm in simulation).
        pts = robot.points(frame, inst.mask, erode_px=0)
        pts = pts[pts[:, 2] > robot.table_z + 0.004]
        if len(pts) < MIN_POINTS:
            continue
        c = np.median(pts[:, :2], axis=0)
        if basket_xy is not None and math.hypot(*(c - basket_xy)) < robot.basket.r_top + 0.01:
            robot.log(f"  {prompt!r} (score {inst.score:.2f}) is in the basket: skipped")
            continue
        if near is not None and math.hypot(*(c - np.asarray(near))) > 0.05:
            continue
        if best is None or inst.score > best[0]:
            best = (inst.score, inst.mask, pts)
    return best
