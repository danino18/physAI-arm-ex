"""The robot that performs the PDDL actions of the order-assembly domain on
demand, closed loop: each action is performed by its skill in the exercise
(exercise/skills.py: (pick-up X) by pick_up(robot, X), ...), given this
executive as the robot; the bonus new-item: / forget-item: by
exercise/misc.py. The executive checks the skills against the domain
(exercise/task_planning/domain.pddl) when it starts, and after a skill:
returns home when it succeeded, opens the gripper and rises when it failed
having moved the arm with nothing left in the gripper.

What the skills use (listed in exercise/skills.py and misc.py): the gripper
(the RG6's grip-detected flag and width), the wrist camera's views (item
poses from perception/ with the exercise's state_estimation.py, the basket,
the depth seen: depth_world.py), MoveIt (moveit_client.py, with the table,
the wall and what the camera saw in its planning scene), running a plan's
steps, and what the gripper holds (kept across restarts). After every look
and every segmentation the camera window shows what was recognised.

The same code runs in simulation and on the real robot; it talks to the arm
and gripper through execution.executor.Executor and to the camera through
the RealSense topics.
"""

import inspect
import math
import os
import re
import time
from dataclasses import dataclass

import cv2
import numpy as np
import rclpy
from std_msgs.msg import Bool, Float64, String

from .. import config, objects
from ..exercise import misc
from ..exercise import motion_planning as mp
from ..exercise import pyramid as pyr
from ..exercise import scene_model as sm
from ..exercise import skills
from ..exercise import state_estimation as se
from ..exercise.task_planning.planning import DOMAIN_FILE
from ..moveit_client import MoveIt
from ..perception.rgbd import RGBDCamera
from ..perception.scene import TOP_DOWN_TILT, ScenePerception, load_perception_config, view_rank
from ..planning.motion import PlanError
from ..planning.placement import Held, Placement
from ..robot import rg6
from ..toolkit import ActionError, box_in_world, fk, show_detections, show_masks
from . import annotate
from .depth_world import DepthWorld
from .drop import drop_spots
from .views import hand_eye


@dataclass
class HeldItem:
    item: str
    held: Held              # how it is held
    source: Placement = None  # where it was picked up
    assumed: bool = False   # not seen being picked: nominal grasp assumed
    depth_before: object = None  # DepthWorld of the survey before the pick-up (not saved)


def describe_view(frame, table_z):
    """Where a frame was taken from, in words."""
    T = frame.T_world_cam
    tilt = math.degrees(math.acos(float(np.clip(-T[2, 2], -1.0, 1.0))))
    how = "looking straight down" if tilt <= TOP_DOWN_TILT else f"looking {tilt:.0f} deg from straight down"
    return f"camera at ({T[0, 3]:.2f}, {T[1, 3]:.2f}), {100 * (T[2, 3] - table_z):.0f} cm above the table, {how}"


def domain_actions(domain_file=DOMAIN_FILE):
    """{action name: number of parameters} of a PDDL domain file."""
    with open(domain_file) as f:
        text = re.sub(r";[^\n]*", "", f.read())
    return {name: params.count("?")
            for name, params in re.findall(r"\(:action\s+([\w-]+)\s+:parameters\s*\(([^)]*)\)", text)}


def skill_functions(actions):
    """{action name: its skill in exercise/skills.py}; raises ActionError
    naming the actions without a skill that takes their parameters."""
    found, problems = {}, []
    for name, n in actions.items():
        fn = getattr(skills, name.replace("-", "_"), None)
        if not callable(fn):
            problems.append(f"no function {name.replace('-', '_')}(robot, ...) for ({name} ...)")
            continue
        try:
            inspect.signature(fn).bind(None, *range(n))
        except TypeError:
            problems.append(f"{fn.__name__} does not take the robot and the {n} parameter(s) of ({name} ...)")
            continue
        found[name] = fn
    if problems:
        raise ActionError("exercise/skills.py: " + "; ".join(problems))
    return found


def parse_action(text, actions):
    """'(pick-up green-mug)' -> ('pick-up', ['green-mug']), checked against
    `actions` ({name: number of parameters})."""
    parts = text.strip().strip("()").split()
    if not parts or parts[0] not in actions or len(parts) - 1 != actions[parts[0]]:
        forms = " | ".join(f"({a}{' <item>' * n})" for a, n in actions.items())
        raise ActionError(f"not an action of the domain: {text!r} (expected {forms})")
    return parts[0], parts[1:]


class ActionExecutive:
    """The robot the skills act with. `node` is an execution.executor.Executor
    (arm and gripper clients); it is spun from here with spin_once."""

    def __init__(self, node, log=print, perception=None, run_dir=None, return_home=True, save_frames=False):
        self.node = node
        self.log = log
        self.cell = config.load_cell()
        self.items = objects.load_items()
        self.basket = objects.load_basket(self.cell)
        self.cfg = load_perception_config()
        ur_type = self.cell["robot"]["ur_type"]
        self.cam = RGBDCamera(node, ur_type, self.cfg["topics"])
        self.perception = perception or ScenePerception(self.cell, self.items, self.basket, cfg=self.cfg)
        self.T_grip_cam = hand_eye(ur_type)
        self.force = float(self.cell["gripper"]["grip_force"])
        self.table_z = self.cell["table"]["surface_z"]
        # MoveIt (move_group): started here unless it runs already.
        self.mv = MoveIt(node, self.cell, log=log)
        sim = node.has_parameter("use_sim_time") and node.get_parameter("use_sim_time").value
        self.mv.start(sim=bool(sim))
        self.obstacles = []      # the boxes in MoveIt's planning scene besides the table and the wall
        self.barrier = []        # the fixed barrier's boxes (exercise/scene_model.py): always among them
        self.pyramid = None      # the pyramid being built (exercise/pyramid.py), whose cubes are always among them
        self.grip = {"detected": None, "width": None}
        node.create_subscription(Bool, "/rg6/grip_detected", lambda m: self.grip.__setitem__("detected", m.data), 10)
        node.create_subscription(Float64, "/rg6/width", lambda m: self.grip.__setitem__("width", m.data), 10)
        self.status_pub = node.create_publisher(String, "/lab_skills/status", 10)
        from sensor_msgs.msg import Image
        self.det_pub = node.create_publisher(Image, "/lab_skills/detections", 1)
        self.held = None
        self.held_file = os.path.join(os.path.expanduser("~"), ".ros", "lab_skills_held.json")
        self._load_held()
        self.scene = None
        self.return_home = return_home
        self.run_dir = run_dir
        self.save_frames = save_frames
        self.on_phase = None
        self.on_observation = None   # f(colour BGR, depth BGR, title) for each annotated view
        self.action = ""
        self.home = np.array(self.cell["robot"]["home_joints"], float)
        self.actions = domain_actions()
        self.skills = skill_functions(self.actions)
        self._moved = False      # the skill ran steps (robot.run) in this action
        self._placed = {}        # {item: Placement} put down in this action

    # ---- what is in the gripper (survives restarts) ---------------------------------------

    def _save_held(self):
        import json
        from dataclasses import asdict
        os.makedirs(os.path.dirname(self.held_file), exist_ok=True)
        data = None if self.held is None else {
            "item": self.held.item, "held": asdict(self.held.held), "assumed": self.held.assumed,
            "source": self.held.source.as_dict() if self.held.source else None}
        with open(self.held_file, "w") as f:
            json.dump(data, f)

    def _load_held(self):
        import json
        try:
            with open(self.held_file) as f:
                d = json.load(f)
        except (OSError, ValueError):
            return
        if d and d["item"] not in self.items:
            # An item added in an earlier console session (new-item:) is not
            # known any more; the gripper's own flag still says it is not empty.
            self.log(f"remembered holding {d['item']}, which is not an item now: forgotten")
            return
        if d:
            h = d["held"]
            h["offset_obj"] = tuple(h["offset_obj"])
            self.held = HeldItem(d["item"], Held(**h), Placement(**d["source"]) if d["source"] else None,
                                 d.get("assumed", False))
            self.log(f"remembered: holding {self.held.item} (checked against the gripper at the next action)")

    def set_held(self, held):
        self.held = held
        self._save_held()

    def now_holding(self, held, source=None, assumed=False):
        """The gripper holds held.item from now on, as `held` (Held), picked
        up from `source` (Placement, or None); `assumed`: not seen being picked up."""
        depth = self.scene.depth if self.scene is not None and not assumed else None
        self.set_held(HeldItem(held.item, held, source, assumed, depth_before=depth))

    def now_empty(self):
        """The gripper holds nothing from now on."""
        self.set_held(None)

    def placed(self, item, placement):
        """`item` was just put down at `placement`: kept clear of on the way home."""
        self._placed[item] = placement

    # ---- small helpers ------------------------------------------------------------------

    def phase(self, text):
        self.log(f"  [{text}]")
        self.status_pub.publish(String(data=text))
        if self.on_phase:
            self.on_phase(text)

    def spin(self, seconds):
        t0 = time.time()
        while time.time() - t0 < seconds:
            rclpy.spin_once(self.node, timeout_sec=0.02)

    def wait(self, seconds):
        """Let the robot's state (gripper, joints) come in for `seconds`."""
        self.spin(seconds)

    def holding(self):
        """Grip detected by the gripper (after a fresh reading)."""
        self.grip["detected"] = None
        t0 = time.time()
        # 15 s: on a network of several computers, a new node may hear the
        # gripper driver only ~5 s after it starts (discovery).
        while self.grip["detected"] is None and time.time() - t0 < 15.0:
            rclpy.spin_once(self.node, timeout_sec=0.05)
        if self.grip["detected"] is None:
            raise ActionError("no reading of /rg6/grip_detected: is the gripper driver running?")
        return bool(self.grip["detected"])

    def width(self):
        w = self.grip["width"]
        return float(w) if w is not None and math.isfinite(w) else rg6.width_from_q(-0.3)

    def now(self):
        return self.node.get_clock().now().nanoseconds * 1e-9

    def current(self):
        """The arm's joints now."""
        return self.node.current()

    def run(self, steps):
        """Perform a plan's steps (a skill's motions); ActionError when one fails."""
        self._moved = True
        try:
            self._run(steps)
        except ActionError:
            raise
        except Exception as e:  # noqa: BLE001
            raise ActionError(f"{type(e).__name__}: {e}") from e

    def _run(self, steps):
        self.node.run_steps(steps, log=lambda s: self.log(s))

    def held_box(self, held):
        """The box of the held item in the gripper's frame (the exercise's held_box), or None."""
        return None if held is None else mp.held_box(held, self.items[held.item])

    def set_obstacles(self, scene, exclude=(), placed=None, skip_slots=()):
        """MoveIt's planning scene: the table and the wall, the fixed barrier,
        the cubes of the pyramid (but those in the slots `skip_slots`: the
        ones a cube is being put on) and the exercise's scene_obstacles for
        `scene` (none without a scene) without the items in `exclude`;
        `placed` {item: Placement} are items put down since. What a look
        saw of the pyramid's cubes is left out: they are the pyramid's boxes."""
        boxes = []
        pyramid = self.pyramid if self.pyramid is not None and self.pyramid.placed else None
        if scene is not None:
            # (an item forgotten since the look, forget-item:, has no model any more)
            dets = {k: d for k, d in scene.detections.items() if k not in (placed or {}) and k in self.items}
            unknown = [o for o in getattr(scene, "obstacles", ()) if not sm.near_barrier(o.polygon, self.barrier)]
            if pyramid is not None:
                dets = {k: d for k, d in dets.items() if k not in pyramid.items()}
                unknown = [o for o in unknown if not pyramid.layout.in_zone(o.polygon.mean(axis=0))]
            boxes = mp.scene_obstacles(dets, self.items, self.basket, scene.basket_xy, self.table_z,
                                       unknown=unknown, exclude=exclude)
        for k, p in (placed or {}).items():
            if pyramid is not None and k in pyramid.items():
                continue
            from ..planning.placement import placement_pose
            from ..robot.collision import Box, object_box
            b = object_box(self.items[k], *placement_pose(self.items[k], p), k)
            boxes.append(Box(k, b.c, b.R, b.h + mp.ITEM_PAD))
        if pyramid is not None:
            boxes += pyramid.boxes(skip=skip_slots)
        boxes += self.barrier
        self.mv.set_obstacles(boxes)
        self.obstacles = boxes

    # ---- looking --------------------------------------------------------------------------

    def capture(self, settle=0.4):
        """A frame taken with the arm at rest, with the camera pose."""
        self.spin(0.1)
        t = self.now() + settle
        frame = self.cam.capture(after=t)
        q = self.node.current()
        frame.q = q
        frame.T_world_cam = self.cam.pose(q)
        if self.save_frames and self.run_dir:
            os.makedirs(self.run_dir, exist_ok=True)
            np.savez_compressed(os.path.join(self.run_dir, f"frame_{time.strftime('%H%M%S')}_{frame.stamp:.1f}.npz"),
                                rgb=frame.rgb, depth=frame.depth, K=frame.K, T=frame.T_world_cam, q=q)
        return frame

    def _move_to_view(self, alternatives, held, exclude, label):
        """Move the camera to the first reachable of `alternatives` (joints
        or camera poses) and take a frame. The motion keeps clear of what the
        camera saw last (the last scene, without the items in `exclude`),
        unless that blocks every way: it may have changed since."""
        box = self.held_box(held)

        def plan(a):
            q = self.node.current()
            if a.shape == (6,):
                return mp.plan_observation(self.mv, q, a, self.width(), box, label)
            return mp.plan_view(self.mv, q, a, self.T_grip_cam, self.width(), box, label)

        def first(alternatives):
            errors = []
            for a in alternatives:
                try:
                    return plan(a), errors
                except PlanError as e:
                    errors.append(str(e))
            return None, errors

        self.set_obstacles(self.scene, exclude=exclude)
        steps, errors = first(alternatives)
        if steps is None and len(self.obstacles) > len(self.barrier):
            self.log(f"  {errors[0] if len(errors) == 1 else 'no view reachable'}; looking without what was "
                     f"seen before")
            self.set_obstacles(None)
            steps, errors = first(alternatives)
        if steps is None:
            if len(errors) == 1:
                raise PlanError(errors[0])
            raise PlanError(f"none of the {len(errors)} views is reachable: " + "; ".join(sorted(set(errors))[:3]))
        self._run(steps)
        return self.capture()

    def survey(self, view, scene=None, held=None, exclude=()):
        """Look from `view` and recognise what the camera sees there (SAM 3
        and the depth: perception/scene.py with the exercise's
        state_estimation.py); returns the scene (perception.scene.Scene)
        with .depth (depth_world.DepthWorld) and .obstacles (the unrecognised
        things in it) set.

        view     the arm's six joints (the exercise's plan_observation moves
                 there), a camera pose (4x4, the colour optical frame in the
                 world: plan_view), or a list of them: the first reachable
        scene    a scene to add this view to (it is updated and returned), or
                 None for a new one
        held     the held item (Held), kept clear of on the way
        exclude  items not to look for (the held one)

        Raises PlanError when no view is reachable, ActionError when the
        scene has no basket."""
        alternatives = self._views(view)
        t0 = time.time()
        k = len(scene.frames) if scene is not None else 0
        self.phase(f"observing: view {k + 1}")
        frame = self._move_to_view(alternatives, held, exclude, f"look (view {k + 1})")
        self.log(f"  view {k + 1}: {describe_view(frame, self.table_z)}")
        self.phase("recognising the items (SAM 3)")
        scene = self.perception.add_view(scene, frame, exclude=exclude, log=self.log)
        if scene.basket_xy is None:
            raise ActionError("the basket is not in view")
        self.phase("mapping what the camera saw")
        self.map_depth(scene, scene.frames, held)
        found = sorted(scene.detections)
        self.log(f"  found {len(found)} items in {time.time() - t0:.1f} s: {', '.join(found)}"
                 + (f"; not found: {', '.join(scene.missing([i for i in self.items if i not in exclude]))}"
                    if len(found) < len(self.items) - len(exclude) else "")
                 + (f"; {len(scene.obstacles)} unknown obstacle(s)" if scene.obstacles else ""))
        for o in scene.obstacles:
            c = o.polygon.mean(axis=0)
            self.log(f"    {'unseen space' if o.unseen else 'unknown obstacle'} at ({c[0]:.3f}, {c[1]:.3f}), "
                     f"{np.ptp(o.polygon[:, 0]) * 100:.0f} x {np.ptp(o.polygon[:, 1]) * 100:.0f} cm, "
                     f"{o.z_top * 100:.0f} cm high, on the {o.support}")
        # The new view, annotated (a put then also shows its plan).
        self.publish_detections(frame, scene, f"view {k + 1}: SAM 3 + depth")
        self.save_scene(scene, "survey")
        self.scene = scene
        return scene

    @staticmethod
    def _views(view):
        """The views a survey may look from, as arrays: (6,) joints or (4, 4) camera poses."""
        try:
            a = np.asarray(view, float)
        except (TypeError, ValueError):
            a = None
        if a is not None and a.shape in ((6,), (4, 4)):
            return [a]
        try:
            alternatives = [np.asarray(v, float) for v in view]
        except (TypeError, ValueError):
            alternatives = []
        if not alternatives or any(v.shape not in ((6,), (4, 4)) for v in alternatives):
            raise ActionError(f"a view is the arm's six joints or a 4x4 camera pose, or a list of them; "
                              f"not {view!r}")
        return alternatives

    def grasp_mask(self, frame, scene, item):
        """The pixels of `item` in `frame` for proposing a grasp: its SAM 3
        mask there joined with the silhouette of its estimated model (SAM 3
        may miss a part, e.g. the spray bottle's grey head), or without a
        mask there the silhouette grown by 5 mm (the estimate's error); not
        within 3 pixels of other items' masks, and only what stands above
        the table."""
        from ..perception.scene import height_image, model_mask
        d = scene.detections[item]
        m = model_mask(frame, self.items[item], d.estimate.placement)
        if d.frame is frame:
            m |= d.mask
        else:
            grow = int(round(frame.K[0, 0] * 0.005 / max(0.2, float(np.nanmedian(frame.depth)))))
            m = cv2.dilate(m.astype(np.uint8), np.ones((2 * grow + 1, 2 * grow + 1), np.uint8)) > 0
        others = np.zeros(m.shape, np.uint8)
        for k, o in scene.detections.items():
            if k != item and o.frame is frame:
                others |= o.mask.astype(np.uint8)
        m &= ~(cv2.dilate(others, np.ones((7, 7), np.uint8)) > 0)
        return m & (height_image(frame) > self.table_z + 0.005)

    def map_depth(self, scene, frames, held=None):
        """scene.depth: the obstacle model from these frames (depth_world.py)."""
        box = self.held_box(held)
        held_box = (lambda q: box_in_world(box, q)) if box is not None else None
        scene.depth_frames = list(frames)
        prior = self.held.depth_before if held is not None and self.held is not None else None
        scene.depth = DepthWorld(frames, self.cell, self.items, self.basket, scene.basket_xy, scene.detections,
                                 max_item_height=self.cfg["max_item_height"], robot_radius=self.cfg["robot_radius"],
                                 held_box=held_box, mask_erode_px=int(self.cfg["sam3"]["mask_erode_px"]),
                                 prior=prior)
        scene.obstacles = scene.depth.blobs()
        if self.run_dir:
            os.makedirs(self.run_dir, exist_ok=True)
            cv2.imwrite(os.path.join(self.run_dir, f"{time.strftime('%H%M%S')}_depth_map.png"), scene.depth.image())

    def publish_detections(self, frame, scene, title):
        """The frame annotated with what was found in it (masks, the items'
        estimated models, names and scores, the basket, obstacles)."""
        show_detections(self, frame, scene, title)

    def segment(self, frame, prompt):
        """SAM 3's masks of a text prompt in `frame`, best first (each with
        .score and .mask); shown in the camera window."""
        found = self.perception.sam.segment(frame.rgb, [prompt], threshold=self.cfg["sam3"]["threshold"])[prompt]
        show_masks(self, frame, [(f"{prompt} {inst.score:.2f}", inst.mask) for inst in found],
                   f"SAM 3: {prompt!r}")
        return found

    def points(self, frame, mask, erode_px=None):
        """The world points (N, 3) of a mask's pixels, above the table and away
        from the robot; the mask eroded by erode_px (default perception.yaml's)."""
        return self.perception._points(frame, mask, erode_px=erode_px)

    def points_around(self, frame, mask, model, others=()):
        """The world points around a mask (as far as the item `model` reaches),
        table included, without the pixels of the masks `others`."""
        return self.perception.points_around(frame, mask, model, others)

    def show(self, frame, draw, title):
        """An annotated observation: draw(view) -> BGR image, drawn on the
        frame's colour image and on its depth (annotate.DepthLook)."""
        from ..viewer import depth_legend
        vis = draw(frame)
        look = annotate.DepthLook(frame)
        self.publish_image(vis, title, depth_legend(draw(look), look.lo, look.hi))

    def publish_image(self, vis, title, depth_vis=None):
        """A BGR image on /lab_skills/detections; into run_dir (beside its
        depth twin) and to on_observation (the console's camera window)."""
        from sensor_msgs.msg import Image
        msg = Image(height=vis.shape[0], width=vis.shape[1], encoding="bgr8", step=vis.shape[1] * 3,
                    data=vis.tobytes())
        msg.header.stamp = self.node.get_clock().now().to_msg()
        self.det_pub.publish(msg)
        if self.on_observation is not None:
            try:
                self.on_observation(vis, depth_vis, title)
            except Exception as e:  # noqa: BLE001
                self.log(f"  camera window: {e}")
        if self.run_dir:
            os.makedirs(self.run_dir, exist_ok=True)
            name = "".join(c if c.isalnum() or c in "-_" else "_" for c in title)
            both = vis if depth_vis is None or depth_vis.shape != vis.shape else np.hstack([vis, depth_vis])
            cv2.imwrite(os.path.join(self.run_dir, f"{time.strftime('%H%M%S')}_{name}.jpg"), both)

    def best_frame(self, scene, item=None):
        """The scene's best frame for geometry (looking straight down, the
        nearest; perception.scene.view_rank), or the one `item`'s estimate
        comes from."""
        if item is not None:
            d = scene.detections[item]
            return d.geometry_frame if d.geometry_frame is not None else d.frame
        return min(scene.frames, key=lambda f: view_rank(f, self.table_z))

    def save_scene(self, scene, title):
        """The perceived placements as YAML in run_dir (for debugging)."""
        if not self.run_dir:
            return
        import yaml
        os.makedirs(self.run_dir, exist_ok=True)
        data = {"basket_xy": [float(v) for v in scene.basket_xy],
                "items": {k: dict(d.estimate.placement.as_dict(), score=round(float(d.score), 3),
                                  notes=d.estimate.notes) for k, d in scene.detections.items()},
                "obstacles": [{"polygon": o.polygon.round(3).tolist(), "z_top": round(o.z_top, 3),
                               "support": o.support, "unseen": o.unseen} for o in getattr(scene, "obstacles", [])]}
        with open(os.path.join(self.run_dir, f"{time.strftime('%H%M%S')}_{title}.yaml"), "w") as f:
            yaml.safe_dump(data, f, sort_keys=False)

    # ---- planning context -------------------------------------------------------------

    def table_region(self, basket_xy, depth=None):
        """Where an item may be put on the table: the placement region of
        cell.yaml, and with `depth` only where the camera's views cover the
        item up to its top. Nearer the edge of the view the item would cast
        its shadow into space no view sees, and could not be picked up again."""
        pl, t = self.cell["placement"], self.cell["table"]
        m = pl["edge_margin"]
        region = dict(x_range=(max(pl["x_range"][0], t["x_min"] + m), min(pl["x_range"][1], t["x_max"] - m)),
                      y_range=(max(pl["y_range"][0], t["y_min"] + m), min(pl["y_range"][1], t["y_max"] - m)),
                      keepout=[(0.0, 0.0, pl["robot_keepout_radius"]),
                               (basket_xy[0], basket_xy[1], self.basket.r_top + pl["basket_margin"])])
        if depth is not None:
            region["accept"] = lambda fp, top: depth.in_view(fp, top + 0.01)
        return region

    def drop_spots(self, scene, held, yaws):
        """Where the held item (Held) can be dropped into the basket with the
        item yaws `yaws` (drop.DropSpots: .spots best first, each (cost, x, y,
        yaw, z_pile); .heights, .z_floor), from the depth `scene` saw."""
        fp = self.items[held.item].poses[held.rest].footprint
        return drop_spots(scene.frames, self.cell, self.basket, scene.basket_xy, fp, yaws, depth=scene.depth,
                          items=[k for k in scene.detections if k != held.item])

    # ---- the actions --------------------------------------------------------------------------

    def execute(self, action, target_xy=None):
        """Perform one action by its skill (exercise/skills.py). target_xy
        goes to a skill that takes it (put_on_table). Returns (success, message)."""
        name, args = parse_action(action, self.actions)
        for item in args:
            if item not in self.items and item not in pyr.SLOTS:
                raise ActionError(f"unknown item {item!r}")
        skill = self.skills[name]
        kwargs = {"target_xy": target_xy} if "target_xy" in inspect.signature(skill).parameters else {}
        self.log(f"== {action}")
        self.action = action
        return self._perform(lambda: skill(self, *args, **kwargs), action)

    def _perform(self, act, title):
        """Run act() (a skill): (success, its message, or the error). After
        it succeeds return home; after it fails having moved the arm with
        nothing left in the gripper, open and rise."""
        t0 = time.time()
        self._moved = False
        self._placed = {}
        try:
            msg = act()
        except (ActionError, PlanError) as e:
            if self._moved and not self._holding_or_unknown():
                self.recover()
            self.phase(f"failed: {e}")
            return False, str(e)
        if self.return_home:
            self.go_home()
        self.phase(f"done: {title} ({time.time() - t0:.0f} s)")
        return True, msg

    def _holding_or_unknown(self):
        try:
            return self.holding()
        except ActionError:
            return True

    # ---- the fixed barrier ------------------------------------------------------------------

    def set_barrier(self, boxes):
        """The fixed barrier (exercise/scene_model.py): `boxes` are in MoveIt's
        planning scene from now on, with every set of obstacles, and motions
        rise above them (mp.rise_z)."""
        self.barrier = list(boxes)
        self.mv.barrier_top = sm.barrier_top(self.barrier)
        self.set_obstacles(self.scene)

    def load_barrier(self, path):
        """The barrier saved in `path` by an earlier scan."""
        boxes = sm.load_boxes(path)
        self.set_barrier(boxes)
        self.log(f"barrier: {len(boxes)} boxes from {path}")

    def map_barrier(self, scene):
        """The depth map of the barrier's surroundings from every frame of
        `scene` (the region and the ceiling: perception.yaml's `barrier`), with
        only what was measured in it (depth_world.py)."""
        b = self.cfg.get("barrier", {})
        pl = self.cell["placement"]
        region = (b.get("x_range", pl["x_range"]), b.get("y_range", pl["y_range"]))
        return DepthWorld(scene.depth_frames, self.cell, self.items, self.basket, scene.basket_xy, scene.detections,
                          max_item_height=b.get("max_height", self.cfg["max_item_height"]),
                          robot_radius=self.cfg["robot_radius"], mask_erode_px=int(self.cfg["sam3"]["mask_erode_px"]),
                          region=region, fill_unseen=False, base_height=self.cfg["max_item_height"] - 0.06)

    def scan_barrier(self, tallest_only=True):
        """`scan:` scan the surroundings from several views and keep the
        obstacle they show: the tallest thing that is no item (`scan: all`:
        everything that is none; exercise/scene_model.py scan_environment).
        Returns (success, message)."""
        if self.pyramid is not None and self.pyramid.placed:
            return False, "a pyramid stands on the table (its upper cubes would be taken for the obstacle): " \
                          "forget-pyramid: first"
        self.log("== scan: the obstacle")
        self.action = "scan"

        def act():
            seen = []
            boxes = sm.scan_environment(self, tallest_only=tallest_only, report=seen)
            self.set_barrier(boxes)
            msg = f"{len(boxes)} boxes of the obstacle"
            if self.run_dir:
                path = os.path.join(self.run_dir, "barrier.json")
                sm.save_boxes(path, boxes)
                msg += f" (saved in {path})"
            if seen:
                msg += "\n    seen, no item, basket or table (tallest first): " + "; ".join(seen[:4])
            if len(boxes) > 4:
                msg += ("\n    warning: that is a lot of boxes for one obstacle: it may be joined to something "
                        "else; move whatever touches it, or check the list above (forget-barrier: and scan: again)")
            return msg

        return self._perform(act, "scan")

    def start_pyramid(self, layout):
        """`pyramid:` the pyramid to be built in `layout` (exercise/pyramid.py Layout), whose
        cubes are kept clear of from now on; the one before is forgotten."""
        self.pyramid = pyr.Pyramid(layout)

    def check_pyramid_area(self, layout, search=False):
        """`pyramid:` look at the table and say what is in the way of the
        pyramid's area: (success, message, layout). Success when nothing is
        (the obstacle, the basket and the items are what it is checked
        against), and `layout` is the one to build. With `search`, when the
        area is not free, the nearest place on the board where it is
        (pyramid.search) is chosen."""
        self.log("== pyramid: is its area free?")
        self.action = "pyramid area"
        found = {"layout": layout}

        def act():
            scene = self.survey(se.OBSERVATION_JOINTS)
            others = {k: self.items[k].footprint_world(d.estimate.placement.rest, d.estimate.placement.x,
                                                       d.estimate.placement.y, d.estimate.placement.yaw)
                      for k, d in scene.detections.items()}
            for o in getattr(scene, "obstacles", ()):
                c = o.polygon.mean(axis=0)
                if o.support == "table" and not sm.near_barrier(o.polygon, self.barrier):
                    others[f"an unknown obstacle at ({c[0]:.2f}, {c[1]:.2f})"] = o.polygon
            basket = None
            if scene.basket_xy is not None:
                basket = (float(scene.basket_xy[0]), float(scene.basket_xy[1]), self.basket.r_top + 0.01)
            region = self.cell["placement"]
            problems = pyr.clear_of(layout, region, self.barrier, basket, others)
            if not problems:
                return "the pyramid's area is free"
            if search:
                reach = (float(region["min_reach"]), float(region["max_reach"]))
                moved = pyr.search(layout, region, reach, self.barrier, basket, others)
                if moved is not None:
                    found["layout"] = moved
                    return (f"({layout.origin[0]:.2f}, {layout.origin[1]:.2f}) is not free ({'; '.join(problems)}): "
                            f"the pyramid goes at ({moved.origin[0]:.2f}, {moved.origin[1]:.2f})")
                raise ActionError("no free place for the pyramid on the board (" + "; ".join(problems) + ")")
            raise ActionError("; ".join(problems))

        ok, msg = self._perform(act, "pyramid area")
        return ok, msg, found["layout"]

    def forget_pyramid(self):
        """`forget-pyramid:` the cubes put in the pyramid are no obstacle in the planning scene any more."""
        self.pyramid = None
        self.set_obstacles(self.scene)

    def forget_barrier(self):
        """`forget-barrier:` the motions need not keep clear of the barrier any more."""
        self.set_barrier([])

    # ---- items added at run time ------------------------------------------------------------

    def add_item(self, name, spec):
        """Know `name` from now on: its model (objects/measured.py) and its SAM 3 prompts."""
        self.items[name] = objects.make_item(name, spec)
        self.perception.add_prompts(name, spec["prompts"])

    def remove_item(self, name):
        """Forget the item `name`: its model and its SAM 3 prompts."""
        self.items.pop(name, None)
        self.perception.remove_prompts(name)
        if self.scene is not None:
            self.scene.detections.pop(name, None)       # the last look no longer holds it

    def new_item(self, name, prompt, shape=None, size=None):
        """Find an item that is not known yet by its text prompt, measure it
        and add it (exercise/misc.py new_item). Returns (success, message,
        spec); the spec is an objects/measured.py spec."""
        self.log(f"== new item {name}: {prompt!r}")
        self.action = f"new-item {name}"
        out = {}

        def act():
            out["spec"], msg = misc.new_item(self, name, prompt, shape, size)
            return msg

        ok, msg = self._perform(act, f"new item {name}")
        return ok, msg, out.get("spec")

    def forget_item(self, name):
        """Forget an item added with new_item (exercise/misc.py forget_item)."""
        misc.forget_item(self, name)

    def go_home(self):
        """Back to the home configuration (the exercise's plan_home), keeping
        clear of what the last survey saw and of the items put down since."""
        self.phase("returning home")
        held = self.held.held if self.held is not None else None
        try:
            self.set_obstacles(self.scene, exclude=(held.item,) if held else (), placed=self._placed)
            steps = mp.plan_home(self.mv, self.node.current(), self.home, self.width(), self.held_box(held))
            self._run(steps)
        except PlanError as e:
            self.log(f"  no path home ({e}); staying here")

    def recover(self):
        """After a failed action (e.g. a grasp): open, rise straight up."""
        self.phase("recovering: open and rise")
        try:
            w = min(self.width() + 0.03, rg6.MAX_WIDTH)
            self.node.gripper(w, self.force, "open")
            q = self.node.current()
            T = fk(q)
            T[2, 3] += 0.12
            path = self.mv.cartesian_path(q, [T], w, avoid_collisions=False)
            self._run([mp.trajectory("rise", path, slow=True)])
        except Exception as e:  # noqa: BLE001
            self.log(f"  recovery failed: {e}")
