#!/usr/bin/env python3
"""Check the exercise's functions (lab_replica/exercise/) offline: no robot,
no simulation, no MoveIt. Each check builds a case whose answer is known
(a synthetic depth image of a block or a can seen from above, a goal with a
known plan, an arm configuration, the board seen from the observation
configuration) and compares.

    python3 src/lab_replica/tools/check_exercise.py              # everything
    python3 src/lab_replica/tools/check_exercise.py task state   # some parts: task, state, observation, motion, barrier

(In the clone's top directory, after `source /opt/ros/humble/setup.bash &&
source install/setup.bash && source .venv/bin/activate`.) The motions
themselves need MoveIt: try them in simulation (run/actions.sh sim).
Exits 1 if a check fails.
"""

import math
import os
import sys
import traceback

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))

from lab_replica import config, objects  # noqa: E402
from lab_replica.planning.placement import Placement, placement_pose  # noqa: E402

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append(ok)
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))


def run(name, fn):
    try:
        fn()
    except Exception as e:  # noqa: BLE001
        check(name, False, f"{type(e).__name__}: {e}")
        traceback.print_exc(limit=3)


# ---- a synthetic camera ----------------------------------------------------------------------


class Frame:
    """A depth image of the table seen straight down from `height`, with
    vertical prisms standing on it: (footprint polygon (N, 2), top z)."""

    def __init__(self, x0, y0, height, prisms, W=640, H=360, f=462.0):
        from lab_replica.autonomy.views import down_view
        self.K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])
        self.T_world_cam = down_view(x0, y0, height)
        self.rgb = np.zeros((H, W, 3), np.uint8)
        vv, uu = np.mgrid[0:H, 0:W]
        # The ray of every pixel, in the world.
        d_cam = np.stack([(uu - W / 2) / f, (vv - H / 2) / f, np.ones_like(uu, float)], -1)
        R, t = self.T_world_cam[:3, :3], self.T_world_cam[:3, 3]
        d = d_cam @ R.T
        depth = (0.0 - t[2]) / d[..., 2]                  # the table, z = 0
        self.masks = []
        for fp, top in prisms:
            s = (top - t[2]) / d[..., 2]                  # depth along the view where the ray reaches the top
            P = t + d * s[..., None]
            inside = _inside(fp, P[..., 0], P[..., 1])
            depth = np.where(inside & (s < depth), s, depth)
            self.masks.append(inside)
        self.depth = depth.astype(np.float32)


def _inside(poly, X, Y):
    """Points (X, Y) inside the convex polygon (counter-clockwise or not)."""
    poly = np.asarray(poly, float)
    ok = np.ones_like(X, bool)
    n = len(poly)
    area = sum(poly[i, 0] * poly[(i + 1) % n, 1] - poly[(i + 1) % n, 0] * poly[i, 1] for i in range(n))
    sign = 1.0 if area > 0 else -1.0
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        ok &= sign * ((b[0] - a[0]) * (Y - a[1]) - (b[1] - a[1]) * (X - a[0])) >= 0
    return ok


def _circle(cx, cy, r, n=64):
    a = np.linspace(0, 2 * math.pi, n, endpoint=False)
    return np.c_[cx + r * np.cos(a), cy + r * np.sin(a)]


# ---- task planning -------------------------------------------------------------------------------


def check_task():
    print("task planning (exercise/task_planning)")
    from lab_replica.planning import pddl
    from lab_replica.tamp import planner, task
    from lab_replica.tamp.goal import holds, parse_goal
    from lab_replica.exercise.task_planning import planning as tp
    planner.check()

    def problem_file():
        items = task.default_items()
        known = objects.load_items()
        problem = tp.load_problem(open(tp.PROBLEM_FILE).read())
        check("problem.pddl: read by UPF with domain.pddl; its objects are the robot's items; it has a goal",
              sorted(items) == sorted(known) and len(problem.goals) > 0,
              f"{len(items)} objects" + (f"; missing {sorted(set(known) - set(items))}" if set(known) - set(items)
                                         else "") + (f"; unknown {sorted(set(items) - set(known))}"
                                                     if set(items) - set(known) else ""))
    run("problem.pddl", problem_file)
    items = list(pddl.ITEMS)
    start = task.most_permissive_init(items)
    cases = [("(in-basket green-mug)", 2),
             ("(and (in-basket red-can) (robot-gripping green-mug))", 3),
             ("(and (in-basket red-block) (or (in-basket blue-block) (on-table green-block)))", 2),
             ("(forall (?x - item) (imply (in-basket ?x) (not (on-table red-can))))", 0),
             ("(and (exists (?x - item) (robot-gripping ?x)) (in-basket yellow-block))", 3)]
    for goal, n in cases:
        def one(goal=goal, n=n):
            formula, _ = parse_goal(goal)
            text = task.problem_text(start, formula, "check", items)
            problem = tp.load_problem(text)
            plan = tp.solve(problem)
            states = tp.expected_states(problem, plan)
            acts = [planner.action_text(a) for a in plan]
            # Each state must follow from the one before by the domain's rules.
            s, legal = start, True
            for a, nxt in zip(acts, states):
                legal &= dict(pddl.successors(s, items)).get(a) == frozenset(nxt)
                s = frozenset(nxt)
            check(f"plan for {goal}", legal and holds(formula, s) and len(acts) == n,
                  f"{len(acts)} actions: {' '.join(acts) or 'none'}")
        run(f"plan for {goal}", one)

    def unsolvable():
        formula, _ = parse_goal("(and (in-basket red-can) (on-table red-can))")
        text = task.problem_text(start, formula, "check", items)
        try:
            tp.solve(tp.load_problem(text))
            check("an unreachable goal raises Unsolvable", False, "a plan was returned")
        except planner.Unsolvable:
            check("an unreachable goal raises Unsolvable", True)
    run("an unreachable goal raises Unsolvable", unsolvable)

    def execute():
        acts = ["(pick-up red-can)", "(put-in-basket red-can)", "(pick-up green-mug)"]
        states = [frozenset({("x", str(k))}) for k in range(3)]
        done = []
        state, k = tp.execute_plan(start, acts, states, lambda a: (done.append(a) or (a != acts[1]), "msg"))
        check("execute_plan stops at the first failed action", k == 1 and state == states[0] and done == acts[:2],
              f"failed_at {k}, performed {len(done)}")
        state, k = tp.execute_plan(start, acts, states, lambda a: (True, "msg"))
        check("execute_plan performs a whole plan", k is None and state == states[-1])
        state, k = tp.execute_plan(start, acts, states, lambda a: (True, "msg"), before=lambda k, a: k < 2)
        check("execute_plan stops when before() says so", k == 2 and state == states[1])
    run("execute_plan", execute)


# ---- state estimation ------------------------------------------------------------------------------


def check_state():
    print("state estimation (exercise/state_estimation.py)")
    from lab_replica.exercise import state_estimation as se
    from lab_replica.robot import rg6
    cell = config.load_cell()
    items = objects.load_items()
    table_z = cell["table"]["surface_z"]

    block, can = items["red-block"], items["red-can"]
    pb = Placement("red-block", "flat", 0.45, -0.05, math.radians(30))
    pc = Placement("red-can", "upright", 0.55, 0.08, 0.0)
    fb = block.footprint_world(pb.rest, pb.x, pb.y, pb.yaw)
    frame = Frame(0.5, 0.0, 0.5, [(fb, block.poses["flat"].top), (_circle(pc.x, pc.y, can.spec["diameter"] / 2),
                                                                     can.poses["upright"].top)])

    def points():
        v, u = np.array([180, 10, 350]), np.array([320, 600, 30])
        P = se.pixels_to_world(u, v, frame.depth[v, u], frame.K, frame.T_world_cam)
        # The centre pixel looks straight down at (0.5, 0) on the table.
        c_ok = np.allclose(P[0], [0.5, 0.0, 0.0], atol=1e-6)
        z_ok = np.allclose(P[:, 2], 0.0, atol=1e-6)
        check("pixels_to_world: pixels of the table land on the table, the centre below the camera",
              c_ok and z_ok, f"centre {np.round(P[0], 4)}")
    run("pixels_to_world", points)

    def placement():
        for model, p, mask, other in ((block, pb, *frame.masks), (can, pc, *frame.masks[::-1])):
            v, u = np.nonzero(mask)
            pts = se.pixels_to_world(u, v, frame.depth[v, u], frame.K, frame.T_world_cam)
            # Around it: the pixels within 10 cm of its middle, but those of the other item.
            v, u = np.nonzero(~other)
            P = se.pixels_to_world(u, v, frame.depth[v, u], frame.K, frame.T_world_cam)
            c = np.median(pts[:, :2], axis=0)
            around = P[np.hypot(P[:, 0] - c[0], P[:, 1] - c[1]) < 0.10]
            est = se.estimate_placement(model, pts, around, table_z)
            q = est.placement
            dyaw = abs((q.yaw - p.yaw + math.pi / 4) % (math.pi / 2) - math.pi / 4) if model is block else 0.0
            check(f"estimate_placement: {p.item}",
                  q.rest == p.rest and q.support == "table" and math.hypot(q.x - p.x, q.y - p.y) < 0.003
                  and dyaw < math.radians(2) and est.plausibility > 0.5,
                  f"{q.rest} at ({q.x:.3f}, {q.y:.3f}), yaw {math.degrees(q.yaw):.0f} deg, "
                  f"plausibility {est.plausibility:.2f}")
    run("estimate_placement", placement)

    def masks():
        from lab_replica.perception.scene import Detection
        a = np.zeros((10, 10), bool)
        a[:5] = True
        b = np.zeros((10, 10), bool)
        b[3:] = True
        c = np.zeros((10, 10), bool)
        c[8:] = True
        cands = [Detection("x", 0.9, 0.9, None, a), Detection("y", 0.8, 0.8, None, a.copy()),
                 Detection("y", 0.6, 0.6, None, b), Detection("z", 0.05, 0.05, None, c),
                 Detection("x", 0.7, 0.7, None, c)]
        got = se.assign_masks(cands, min_score=0.1)
        check("assign_masks: best first, one mask per item, one item per mask, scores above the minimum",
              set(got) == {"x", "y"} and got["x"] is cands[0] and got["y"] is cands[2],
              ", ".join(f"{k}: {d.score}" for k, d in got.items()))
    run("assign_masks", masks)

    def fuse():
        from lab_replica.perception.estimate import Estimate
        from lab_replica.perception.scene import Detection

        def det(item, score, x, y):
            return Detection(item, score, score, Estimate(Placement(item, "flat", x, y, 0.0), 1.0, 100, 0.05), None)
        views = [{"a": det("a", 0.9, 0.3, 0.0), "b": det("b", 0.5, 0.5, 0.0)},
                 {"a": det("a", 0.6, 0.3, 0.0), "b": det("b", 0.9, 0.3, 0.01)}]
        got = se.fuse_views(views, [1.0, 0.8])
        # b's best (0.9 x 0.8 = 0.72) stands where a (0.9) is: b takes its other detection.
        check("fuse_views: best score x weight per item, two items never in one place",
              got["a"] is views[0]["a"] and got["b"] is views[0]["b"])
    run("fuse_views", fuse)

    def grasps():
        for model, p, mask, width, across in ((block, pb, frame.masks[0], 0.050, "a pair of its faces"),
                                              (can, pc, frame.masks[1], 0.055, "any direction")):
            gs = se.grasp_in_image(frame, mask)
            g3 = se.grasp_pose(frame, gs[0], mask, table_z)
            T = g3.pose
            closing = math.atan2(T[1, 0], T[0, 0]) + math.pi / 2     # the fingers close along the gripper's y
            top = p.z_support + model.poses[p.rest].top
            tips = max(table_z + se.TIP_CLEARANCE, top - rg6.free_height(g3.width) + se.KNUCKLE_MARGIN)
            ok = (math.hypot(T[0, 3] - p.x, T[1, 3] - p.y) < 0.004 and abs(g3.width - width) < 0.006
                  and abs(g3.tips_z - tips) < 0.004 and abs(T[2, 3] - g3.tips_z - rg6.depth_from_width(g3.width)) < 1e-6
                  and T[2, 2] < -0.999)
            if model is block:
                ok &= abs((closing - p.yaw + math.pi / 4) % (math.pi / 2) - math.pi / 4) < math.radians(4)
            check(f"grasp_in_image + grasp_pose: {p.item} (across {across})", ok,
                  f"{len(gs)} grasp(s); the first at ({T[0, 3]:.3f}, {T[1, 3]:.3f}), {g3.width * 1000:.0f} mm, "
                  f"fingertips {1000 * g3.tips_z:.0f} mm up, closing at {math.degrees(closing) % 180:.0f} deg")
    run("grasps", grasps)


# ---- motion planning (the parts without MoveIt) ------------------------------------------------------


def check_motion():
    print("motion planning (exercise/motion_planning.py; the motions need MoveIt: try them in simulation)")
    from lab_replica.autonomy.views import hand_eye
    from lab_replica.exercise import motion_planning as mp
    from lab_replica.toolkit import Held
    from lab_replica.robot.collision import object_box
    from lab_replica.toolkit import box_in_world, fk, gripper_pose, ik_all
    cell = config.load_cell()
    items = objects.load_items()
    rng = np.random.default_rng(1)
    home = np.array(cell["robot"]["home_joints"])

    def ik():
        worst = 0.0
        for _ in range(20):
            q = home + rng.uniform(-0.6, 0.6, 6)
            got = mp.choose_solution(ik_all(fk(q)), q)
            worst = max(worst, np.abs(got - q).max())
        check("choose_solution: of the solutions for fk(q), q itself", worst < 1e-5, f"worst {worst:.1e} rad")
        q = home.copy()
        q[5] += 2 * math.pi - 0.1
        got = mp.choose_solution(ik_all(fk(q)), home)
        check("choose_solution: the same angle up to 2 pi, nearest the reference", abs(got[5] - (home[5] - 0.1)) < 1e-5)
    run("choose_solution", ik)

    def camera():
        T_grip_cam = hand_eye(cell["robot"]["ur_type"])
        T = fk(home)
        check("camera_to_gripper", np.allclose(mp.camera_to_gripper(T @ T_grip_cam, T_grip_cam), T, atol=1e-9))
    run("camera_to_gripper", camera)

    def held():
        # A can standing at p, grasped with the gripper's x along yaw 0.4, fingertips 87 mm up:
        # its held box, carried by the arm at the grasp, must be where the can stands.
        model = items["red-can"]
        p = Placement("red-can", "upright", 0.5, 0.05, 0.3)
        psi, width, tip_z = 0.4, 0.055, 0.087
        gx, gy = p.x + 0.004, p.y - 0.003
        c, s = math.cos(p.yaw), math.sin(p.yaw)
        off = (c * (gx - p.x) + s * (gy - p.y), -s * (gx - p.x) + c * (gy - p.y))
        h = Held("red-can", "upright", width, tip_z, off, (p.yaw - psi), model.poses["upright"].top)
        from lab_replica.robot import rg6
        T = gripper_pose(gx, gy, p.z_support + tip_z + rg6.depth_from_width(width), psi)
        q = mp.choose_solution(ik_all(T), home)
        got = box_in_world(mp.held_box(h, model), q)
        want = object_box(model, *placement_pose(model, p))
        check("held_box: carried at the grasp, the box is where the item stood",
              np.allclose(got.c, want.c, atol=1e-6) and np.allclose(np.abs(got.R.T @ want.R), np.eye(3), atol=1e-6),
              f"centre off by {1000 * np.linalg.norm(got.c - want.c):.2f} mm")
    run("held_box", held)

    def spots():
        region = dict(x_range=(0.2, 0.3), y_range=(-0.1, 0.1), keepout=[(0.25, 0.05, 0.03)])
        s = mp.sample_spots(region, (0.21, -0.09))
        first_ok = math.hypot(s[0][0] - 0.21, s[0][1] + 0.09) < 1e-9
        d = [math.hypot(x - 0.21, y + 0.09) for x, y in s]
        check("sample_spots: the grid, nearest to the preferred spot first", first_ok and d == sorted(d), f"{len(s)} spots")
        sq = lambda x, y, r=0.01: np.array([[x - r, y - r], [x + r, y - r], [x + r, y + r], [x - r, y + r]])  # noqa: E731
        obstacles = [sq(0.25, -0.05)]
        cases = [(sq(0.215, 0.0), True), (sq(0.25, -0.05), False), (sq(0.25, 0.05), False), (sq(0.31, 0.0), False),
                 (sq(0.274, -0.05), False), (sq(0.25, -0.005), True)]
        got = [mp.is_free(fp, region, obstacles, margin=0.02) == want for fp, want in cases]
        check("is_free: inside the region, off the keepouts, `margin` from obstacles", all(got),
              f"{sum(got)}/{len(got)} cases")
    run("sample_spots, is_free", spots)


# ---- the fixed barrier (exercise/scene_model.py) and how a path is chosen around it ----------------


def check_barrier():
    print("fixed barrier (exercise/scene_model.py; plan_path with a stand-in for MoveIt)")
    from lab_replica.exercise import motion_planning as mp
    from lab_replica.exercise import scene_model as sm
    cell_size = 0.0025
    xs = -0.1 + (np.arange(440) + 0.5) * cell_size
    ys = -0.4 + (np.arange(320) + 0.5) * cell_size
    X, Y = np.meshgrid(xs, ys)

    def wall(cx, cy, length, thick, angle):
        c, s = math.cos(angle), math.sin(angle)
        u, v = (X - cx) * c + (Y - cy) * s, -(X - cx) * s + (Y - cy) * c
        return (abs(u) <= length / 2) & (abs(v) <= thick / 2)

    def covered(boxes, m, top):
        P = np.c_[X[m], Y[m], np.full(m.sum(), top - 0.01)]
        inside = np.zeros(len(P), bool)
        for b in boxes:
            inside |= b.inside(P, 0.0)
        return inside.mean()

    def boxes_of_walls():
        for name, m in (("straight wall", wall(0.4, 0.2, 0.5, 0.03, 0.0)),
                        ("slanted wall", wall(0.4, 0.0, 0.6, 0.03, math.radians(35)))):
            boxes = sm.boxes_from_height_map(X, Y, np.where(m, 0.25, -1e3), m, 0.0, cell_size)
            area = sum(4 * b.h[0] * b.h[1] for b in boxes) / (m.sum() * cell_size ** 2)
            check(f"boxes_from_height_map: a {name} is covered, by few boxes that hug it",
                  covered(boxes, m, 0.25) > 0.999 and len(boxes) <= sm.MAX_BOXES and area < 3.0,
                  f"{len(boxes)} boxes, {area:.1f}x the wall's area")
        m = wall(0.55, 0.0, 0.6, 0.03, 0.0)
        H = np.where(m & (X < 0.55), 0.1, np.where(m, 0.3, -1e3))
        boxes = sm.boxes_from_height_map(X, Y, H, m, 0.0, cell_size)
        tops = sorted(round(float(b.c[2] + b.h[2]), 2) for b in boxes)
        check("boxes_from_height_map: a wall of two heights gets a box per height",
              len(boxes) == 2 and abs(tops[0] - 0.12) < 0.011 and abs(tops[1] - 0.32) < 0.011, f"tops {tops}")
        noise = np.zeros(X.shape, bool)
        noise[100:105, 100:105] = True
        check("boxes_from_height_map: a few cells are noise, not a barrier",
              sm.boxes_from_height_map(X, Y, np.where(noise, 0.2, -1e3), noise, 0.0, cell_size) == [])
        far = np.array([[0.4, 0.0, 0.1]])
        check("a point away from the barrier is outside its boxes",
              not any(b.inside(far, 0.0)[0] for b in sm.boxes_from_height_map(
                  X, Y, np.where(wall(0.4, 0.2, 0.5, 0.03, 0.0), 0.25, -1e3), wall(0.4, 0.2, 0.5, 0.03, 0.0), 0.0,
                  cell_size)))
    run("boxes_from_height_map", boxes_of_walls)

    def helpers():
        boxes = sm.boxes_from_height_map(X, Y, np.where(wall(0.4, 0.2, 0.5, 0.03, 0.0), 0.25, -1e3),
                                         wall(0.4, 0.2, 0.5, 0.03, 0.0), 0.0, cell_size)
        check("barrier_top: the tallest box (and none without boxes)",
              abs(sm.barrier_top(boxes) - 0.27) < 0.011 and sm.barrier_top([]) is None)
        check("near_barrier: an obstacle on the barrier is it, one far off is not",
              sm.near_barrier(np.array([[0.3, 0.2], [0.5, 0.2]]), boxes) and
              not sm.near_barrier(np.array([[0.3, -0.1], [0.5, -0.1]]), boxes))
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            sm.save_boxes(os.path.join(d, "b.json"), boxes)
            back = sm.load_boxes(os.path.join(d, "b.json"))
        check("save_boxes, load_boxes: the same boxes",
              len(back) == len(boxes) and all(np.allclose(a.c, b.c) and np.allclose(a.R, b.R) and np.allclose(a.h, b.h)
                                              for a, b in zip(boxes, back)))
    run("barrier helpers", helpers)

    def tallest():
        cube = wall(0.40, 0.00, 0.10, 0.10, 0.3)             # the wooden cube: 10 x 10 cm, 12 cm high
        low = wall(0.60, -0.15, 0.50, 0.06, 0.0)             # something longer, 5 cm high
        spike = wall(0.20, 0.20, 0.02, 0.02, 0.0)            # a tall stray: 2 x 2 cm, 50 cm high
        H = np.where(cube, 0.12, np.where(low, 0.05, np.where(spike, 0.5, -1e3)))
        cells = cube | low | spike
        every = sm.boxes_from_height_map(X, Y, H, cells, 0.0, cell_size, min_cells=10)
        only = sm.boxes_from_height_map(X, Y, H, cells, 0.0, cell_size, min_cells=10, tallest_only=True)
        near_cube = all(abs(b.c[0] - 0.40) < 0.08 and abs(b.c[1]) < 0.08 for b in only)
        check("tallest_only: of a cube, something longer but lower and a tall stray, the cube only",
              len(every) >= 3 and len(only) >= 1 and near_cube and abs(sm.barrier_top(only) - 0.14) < 0.011,
              f"{len(every)} boxes without, {len(only)} with; top {sm.barrier_top(only)}")
        check("tallest_only: nothing big enough is no obstacle",
              sm.boxes_from_height_map(X, Y, np.where(spike, 0.5, -1e3), spike, 0.0, cell_size, min_cells=10,
                                       tallest_only=True) == [])
    run("tallest_only", tallest)

    def commands():
        from lab_replica.console import parse_barrier_command as parse
        got = [parse(t) for t in ("scan:", "scan", "scan: all", "forget-barrier:", "scan: x", "scanner",
                                  "(pick-up red-can)")]
        want = [("scan", {"tallest": True}), ("scan", {"tallest": True}), ("scan", {"tallest": False}),
                ("forget-barrier", {}), None, None, None]
        check("parse_barrier_command: scan:, scan: all, forget-barrier:, nothing else", got == want, str(got))
    run("parse_barrier_command", commands)

    def strategies():
        # A stand-in for MoveIt, and for the kinematics: the "height" of a configuration is its joint 1.
        # The barrier blocks joint 0 in (0.3, 0.5) below a height of 0.5.
        class Fake:
            barrier_top = None

            def __init__(self, rrt_fails=0):
                self.notes, self.rrt_fails, self.times = [], rrt_fails, []
                self.log = self.notes.append

            def is_valid(self, q, width, held=None):
                return not (0.3 < q[0] < 0.5 and q[1] < 0.5)

            def cartesian_path(self, q0, poses, width, held=None):
                q = np.array(q0, float)
                q[1] = poses[-1][2, 3]
                return [np.array(q0, float), q]

            def plan_to_joints(self, q0, q1, width, held=None, planning_time=None):
                self.times.append(planning_time)
                if len(self.times) <= self.rrt_fails:
                    raise sm.PlanError("no way")
                return [np.array(q0, float), np.array(q1, float)]

        def fake_fk(q):
            T = np.eye(4)
            T[2, 3] = q[1]
            return T
        real_fk, mp.fk = mp.fk, fake_fk
        try:
            def go(moveit, q0, q1):
                mp.plan_path(moveit, np.array(q0, float), np.array(q1, float), 0.08)
                return " | ".join(moveit.notes)
            free = go(Fake(), [0.0, 0.2, 0, 0, 0, 0], [0.2, 0.2, 0, 0, 0, 0])
            check("plan_path: a free straight line is taken", "straight line" in free, free)
            blocked = go(Fake(), [0.0, 0.2, 0, 0, 0, 0], [0.8, 0.6, 0, 0, 0, 0])
            check("plan_path: a blocked line goes up first, then straight", "straight up to 0.60" in blocked, blocked)
            m = Fake()
            m.barrier_top = 0.9
            check("rise_z: RISE_CLEARANCE above a barrier taller than RISE_Z, RISE_Z otherwise",
                  abs(mp.rise_z(m) - 1.0) < 1e-9 and mp.rise_z(Fake()) == mp.RISE_Z)
            m.rrt_fails = 0
            tall = go(m, [0.0, 0.2, 0, 0, 0, 0], [0.8, 0.6, 0, 0, 0, 0])
            check("plan_path: it rises as high as the barrier needs", "straight up to 1.00" in tall, tall)
            high = Fake()
            noisy = go(high, [0.0, 0.7, 0, 0, 0, 0], [0.8, 0.7, 0, 0, 0, 0])
            check("plan_path: no rise needed when already up: the straight line (valid at that height)",
                  "straight line" in noisy, noisy)
            retry = Fake(rrt_fails=1)
            retry.is_valid = lambda q, width, held=None: False if 0.3 < q[0] < 0.5 else True
            note = go(retry, [0.0, 0.2, 0, 0, 0, 0], [0.8, 0.2, 0, 0, 0, 0])
            check("plan_path: when RRT-Connect finds nothing, it tries again for longer",
                  retry.times == [None, mp.RRT_RETRY_TIME] and "second try" in note, note)
        finally:
            mp.fk = real_fk
    run("plan_path strategies", strategies)


def check_observation():
    print("observation configuration (exercise/state_estimation.py OBSERVATION_JOINTS)")
    from lab_replica.exercise import state_estimation as se
    from lab_replica.problems.generator import SurveyViews
    from lab_replica.toolkit import ARM_JOINTS, fk, joint_limits

    def joints():
        q = np.asarray(se.OBSERVATION_JOINTS, float)
        if q.shape != (6,):
            check("OBSERVATION_JOINTS: the arm's six joints", False, f"got {se.OBSERVATION_JOINTS!r}")
            return
        limits = joint_limits()
        out = [j for j, v in zip(ARM_JOINTS, q) if not limits[j][0] <= v <= limits[j][1]]
        check("OBSERVATION_JOINTS: six joints within their limits", not out, ", ".join(out) or "")
        cell = config.load_cell()
        bx, by = cell["table"]["board_x"], cell["table"]["board_y"]
        # The corners 2 cm in: no item stands at the very edge (the current
        # configuration's image ends just short of one corner).
        corners = [(x + d, y + e, cell["table"]["surface_z"]) for x, d in zip(bx, (0.02, -0.02))
                   for y, e in zip(by, (0.02, -0.02))]
        cam = SurveyViews(cell)
        seen = [cam.visible([c], margin=0) for c in corners]
        check("from there the camera sees the whole board (its corners, 2 cm in)", all(seen),
              f"{sum(seen)}/4 corners in the image; mounting face {fk(q)[2, 3]:.2f} m above the table")
    run("OBSERVATION_JOINTS", joints)


def main():
    parts = sys.argv[1:] or ["task", "state", "observation", "motion", "barrier"]
    for p in parts:
        {"task": check_task, "state": check_state, "observation": check_observation, "motion": check_motion,
         "barrier": check_barrier}[p]()
    n_fail = RESULTS.count(False)
    print(f"{len(RESULTS) - n_fail}/{len(RESULTS)} checks passed")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
