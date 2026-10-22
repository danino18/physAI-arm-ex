"""MoveIt for the motion planning exercise, without ROS in its interface.

`MoveIt` talks to MoveIt's move_group (launch/moveit.launch.py) through its
services. Its methods take and return plain numpy values:

    q       the six arm joints (shoulder_pan ... wrist_3), radians
    T       a 4x4 pose in the world frame (cell.yaml: z = 0 on the table)
    width   the RG6's opening (m): the fingers are part of the robot model
    held    a Box held in the gripper (exercise/motion_planning.py held_box), or None

Planning frames: the planning group `arm` ends at rg6_base_link, the RG6's
mounting face. Pose goals and Cartesian paths are for that frame.

The planning scene holds the table and the wall from the start (cell.yaml);
more boxes (robot.collision.Box) and vertical cylinders (Cylinder) can be
added with set_obstacles (they replace the previous ones).

All methods raise PlanError when MoveIt finds nothing.
"""

import os
import shutil
import signal
import subprocess
import time

import numpy as np

from . import config
from .planning.motion import PlanError
from .robot import rg6
from .robot.collision import Box
from .robot.kinematics import ARM_JOINTS
from .transforms import matrix_to_quat

GROUP = "arm"
TIP = "rg6_base_link"
FRAME = "world"
FINGER_LINKS = ["rg6_base_link", "left_inner_finger", "right_inner_finger", "left_outer_knuckle",
                "right_outer_knuckle", "left_inner_knuckle", "right_inner_knuckle"]
HELD_ID = "held_item"
SERVICES = ("/plan_kinematic_path", "/compute_cartesian_path", "/compute_ik", "/apply_planning_scene",
            "/check_state_validity")


def _pose_msg(T):
    from geometry_msgs.msg import Pose
    T = np.asarray(T, float)
    q = matrix_to_quat(T[:3, :3])        # (x, y, z, w)
    p = Pose()
    p.position.x, p.position.y, p.position.z = (float(v) for v in T[:3, 3])
    p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = (float(v) for v in q)
    return p


class Cylinder:
    """A vertical cylinder obstacle: name, centre c (x, y, z), radius r, half height h."""

    def __init__(self, name, center, radius, half_height):
        self.name = name
        self.c = np.asarray(center, float)
        self.r = float(radius)
        self.h = float(half_height)


def _box_object(box, frame, op_add=True):
    """A CollisionObject of a Box (or a Cylinder)."""
    from moveit_msgs.msg import CollisionObject
    from shape_msgs.msg import SolidPrimitive
    o = CollisionObject()
    o.header.frame_id = frame
    o.id = box.name
    if op_add:
        T = np.eye(4)
        T[:3, 3] = box.c
        if isinstance(box, Cylinder):
            prim = SolidPrimitive(type=SolidPrimitive.CYLINDER, dimensions=[2 * box.h, box.r])
        else:
            prim = SolidPrimitive(type=SolidPrimitive.BOX, dimensions=[float(2 * v) for v in box.h])
            T[:3, :3] = box.R
        o.primitives = [prim]
        o.primitive_poses = [_pose_msg(T)]
        o.operation = CollisionObject.ADD
    else:
        o.operation = CollisionObject.REMOVE
    return o


def static_boxes(cell):
    """The table top and the wall behind it, as Boxes (world frame)."""
    t, room = cell["table"], cell["room"]
    I = np.eye(3)
    table = Box("table", [(t["x_min"] + t["x_max"]) / 2, (t["y_min"] + t["y_max"]) / 2, t["surface_z"] - 0.05],
                I, [(t["x_max"] - t["x_min"]) / 2, (t["y_max"] - t["y_min"]) / 2, 0.05])
    wy = float(room["wall_y"])
    wall = Box("wall", [(t["x_min"] + t["x_max"]) / 2, wy + 0.05, t["surface_z"] + 1.0],
               I, [(t["x_max"] - t["x_min"]) / 2 + 0.5, 0.05, 1.0])
    return [table, wall]


class MoveIt:
    """Planning requests to move_group. `node` is an rclpy node (spun here
    with spin_once, like the rest of the executive)."""

    def __init__(self, node, cell=None, log=print, planning_time=5.0, attempts=3):
        from moveit_msgs.srv import (ApplyPlanningScene, GetCartesianPath, GetMotionPlan, GetPositionIK,
                                     GetStateValidity)
        self.node = node
        self.log = log
        self.cell = cell or config.load_cell()
        self.planning_time = planning_time
        self.attempts = attempts
        self.clients = {
            "plan": node.create_client(GetMotionPlan, "/plan_kinematic_path"),
            "cartesian": node.create_client(GetCartesianPath, "/compute_cartesian_path"),
            "ik": node.create_client(GetPositionIK, "/compute_ik"),
            "scene": node.create_client(ApplyPlanningScene, "/apply_planning_scene"),
            "valid": node.create_client(GetStateValidity, "/check_state_validity"),
        }
        self.obstacles = []          # names of the boxes added with set_obstacles
        self.barrier_top = None      # the height of the fixed barrier's tallest box (motion_planning.rise_z), if any
        self.process = None          # move_group started by start()

    # ---- move_group ---------------------------------------------------------------------

    def available(self, timeout=1.0):
        return all(c.wait_for_service(timeout_sec=timeout) for c in self.clients.values())

    def start(self, sim=False, timeout=60.0):
        """Use the running move_group, or start one (launch/moveit.launch.py)
        and wait for its services. Then set up the planning scene."""
        if not self.available(timeout=2.0):
            self.log("starting MoveIt (move_group) ...")
            ros2 = shutil.which("ros2") or "ros2"
            log_dir = os.path.join(os.path.expanduser("~"), ".ros")
            os.makedirs(log_dir, exist_ok=True)
            self.log_file = open(os.path.join(log_dir, "lab_move_group.log"), "w")
            self.process = subprocess.Popen(
                [ros2, "launch", "lab_replica", "moveit.launch.py", f"use_sim_time:={str(sim).lower()}"],
                stdout=self.log_file, stderr=subprocess.STDOUT, start_new_session=True)
            t0 = time.time()
            while not self.available(timeout=1.0):
                if self.process.poll() is not None:
                    raise RuntimeError(f"move_group exited (log: {self.log_file.name})")
                if time.time() - t0 > timeout:
                    raise RuntimeError(f"move_group did not come up in {timeout:.0f} s (log: {self.log_file.name})")
        self.reset_scene()

    def stop(self):
        if self.process is not None and self.process.poll() is None:
            os.killpg(self.process.pid, signal.SIGINT)
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
        self.process = None

    def _call(self, name, req, timeout=30.0):
        client = self.clients[name]
        if not client.wait_for_service(timeout_sec=5.0):
            raise PlanError(f"MoveIt is not running (no {client.srv_name})")
        import rclpy
        fut = client.call_async(req)
        t0 = time.time()
        while not fut.done():
            rclpy.spin_once(self.node, timeout_sec=0.02)
            if time.time() - t0 > timeout:
                raise PlanError(f"MoveIt: no answer from {client.srv_name} in {timeout:.0f} s")
        return fut.result()

    # ---- the planning scene ------------------------------------------------------------

    def _apply(self, world=(), remove=(), attach=None, detach=False):
        from moveit_msgs.msg import AttachedCollisionObject, PlanningScene
        from moveit_msgs.srv import ApplyPlanningScene
        scene = PlanningScene(is_diff=True)
        scene.robot_state.is_diff = True
        scene.world.collision_objects = [_box_object(b, FRAME) for b in world] + \
                                        [_box_object(Box(n, 0, np.eye(3), 0), FRAME, False) for n in remove]
        if attach is not None or detach:
            a = AttachedCollisionObject(link_name=TIP, touch_links=FINGER_LINKS)
            a.object = _box_object(attach if attach is not None else Box(HELD_ID, 0, np.eye(3), 0), TIP,
                                   attach is not None)
            a.object.id = HELD_ID
            scene.robot_state.attached_collision_objects = [a]
        r = self._call("scene", ApplyPlanningScene.Request(scene=scene))
        if not r.success:
            raise PlanError("MoveIt did not accept the planning scene")

    def reset_scene(self):
        """Only the table and the wall."""
        self._apply(world=static_boxes(self.cell), remove=self.obstacles)
        self.obstacles = []

    def set_obstacles(self, boxes):
        """Replace the obstacles added before by `boxes` (Box or Cylinder, world frame)."""
        import copy
        named = []
        for k, b in enumerate(boxes):
            b = copy.copy(b)
            b.name = f"obstacle {k}: {b.name}"
            named.append(b)
        boxes = named
        names = [b.name for b in boxes]
        self._apply(world=boxes, remove=[n for n in self.obstacles if n not in names])
        self.obstacles = names

    # ---- robot states -----------------------------------------------------------------

    def _state(self, q, width, held=None):
        """A full RobotState: the arm at q, the fingers at `width`, and the
        held box attached to the gripper (or nothing attached)."""
        from moveit_msgs.msg import AttachedCollisionObject, RobotState
        from sensor_msgs.msg import JointState
        fq = rg6.q_from_width(min(max(width, 0.0), rg6.MAX_WIDTH))
        js = JointState(name=list(ARM_JOINTS) + ["left_finger_slide", "right_finger_slide", "left_finger_lift",
                                                 "right_finger_lift", "finger_joint", "left_inner_knuckle_joint",
                                                 "right_outer_knuckle_joint", "right_inner_knuckle_joint"],
                        position=[float(v) for v in q] + [float(v) for v in rg6.joint_command(fq)])
        s = RobotState(joint_state=js, is_diff=False)
        if held is not None:
            a = AttachedCollisionObject(link_name=TIP, touch_links=FINGER_LINKS, object=_box_object(held, TIP))
            a.object.id = HELD_ID
            s.attached_collision_objects = [a]
        return s

    def is_valid(self, q, width, held=None):
        """Whether the robot at q (and what it holds) touches nothing."""
        from moveit_msgs.srv import GetStateValidity
        r = self._call("valid", GetStateValidity.Request(robot_state=self._state(q, width, held), group_name=GROUP))
        return bool(r.valid)

    def contacts(self, q, width, held=None):
        """The pairs in contact at q, as strings (for messages)."""
        from moveit_msgs.srv import GetStateValidity
        r = self._call("valid", GetStateValidity.Request(robot_state=self._state(q, width, held), group_name=GROUP))
        return sorted({f"{c.contact_body_1}-{c.contact_body_2}" for c in r.contacts})

    # ---- planning ----------------------------------------------------------------------

    def _path(self, joint_trajectory):
        idx = [joint_trajectory.joint_names.index(j) for j in ARM_JOINTS]
        return [np.array([p.positions[i] for i in idx]) for p in joint_trajectory.points]

    def plan_to_joints(self, q_start, q_goal, width, held=None, planning_time=None):
        """A collision-free joint path from q_start to q_goal (OMPL, RRT-Connect)."""
        from moveit_msgs.msg import Constraints, JointConstraint
        goal = Constraints(joint_constraints=[
            JointConstraint(joint_name=j, position=float(v), tolerance_above=1e-3, tolerance_below=1e-3, weight=1.0)
            for j, v in zip(ARM_JOINTS, q_goal)])
        return self._plan(q_start, goal, width, held, planning_time, f"joint goal {np.round(q_goal, 2)}")

    def plan_to_pose(self, q_start, T_goal, width, held=None, planning_time=None, tolerance=(0.002, 0.01)):
        """A collision-free joint path to any arm configuration that puts
        rg6_base_link at T_goal (within `tolerance`: m, rad). MoveIt picks the
        configuration (inverse kinematics inside the planner); to choose it
        yourself, plan_to_joints to a solution of ik_all."""
        from moveit_msgs.msg import BoundingVolume, Constraints, OrientationConstraint, PositionConstraint
        from shape_msgs.msg import SolidPrimitive
        T_goal = np.asarray(T_goal, float)
        pose = _pose_msg(T_goal)
        pc = PositionConstraint(link_name=TIP, weight=1.0)
        pc.header.frame_id = FRAME
        pc.constraint_region = BoundingVolume(
            primitives=[SolidPrimitive(type=SolidPrimitive.SPHERE, dimensions=[float(tolerance[0])])],
            primitive_poses=[pose])
        oc = OrientationConstraint(link_name=TIP, orientation=pose.orientation, weight=1.0,
                                   absolute_x_axis_tolerance=float(tolerance[1]),
                                   absolute_y_axis_tolerance=float(tolerance[1]),
                                   absolute_z_axis_tolerance=float(tolerance[1]))
        oc.header.frame_id = FRAME
        goal = Constraints(position_constraints=[pc], orientation_constraints=[oc])
        return self._plan(q_start, goal, width, held, planning_time, f"pose {np.round(T_goal[:3, 3], 3)}")

    def _plan(self, q_start, goal, width, held, planning_time, what):
        from moveit_msgs.msg import MotionPlanRequest, MoveItErrorCodes, WorkspaceParameters
        from moveit_msgs.srv import GetMotionPlan
        req = MotionPlanRequest(group_name=GROUP, start_state=self._state(q_start, width, held),
                                goal_constraints=[goal], pipeline_id="ompl", planner_id="RRTConnectkConfigDefault",
                                num_planning_attempts=self.attempts,
                                allowed_planning_time=float(planning_time or self.planning_time),
                                max_velocity_scaling_factor=1.0, max_acceleration_scaling_factor=1.0)
        req.workspace_parameters = WorkspaceParameters()
        req.workspace_parameters.header.frame_id = FRAME
        req.workspace_parameters.min_corner.x = req.workspace_parameters.min_corner.y = -2.0
        req.workspace_parameters.min_corner.z = -1.0
        req.workspace_parameters.max_corner.x = req.workspace_parameters.max_corner.y = 2.0
        req.workspace_parameters.max_corner.z = 2.0
        r = self._call("plan", GetMotionPlan.Request(motion_plan_request=req),
                       timeout=float(planning_time or self.planning_time) * self.attempts + 30.0)
        res = r.motion_plan_response
        if res.error_code.val != MoveItErrorCodes.SUCCESS:
            raise PlanError(f"MoveIt found no path to the {what} ({_error_name(res.error_code.val)})")
        return self._path(res.trajectory.joint_trajectory)

    def cartesian_path(self, q_start, poses, width, held=None, step=0.005, avoid_collisions=True, min_fraction=1.0):
        """Joint path that moves rg6_base_link along straight lines through
        `poses` (4x4, world). Raises PlanError if MoveIt gets less than
        `min_fraction` of the way (IK fails, or a collision with
        avoid_collisions)."""
        from moveit_msgs.srv import GetCartesianPath
        req = GetCartesianPath.Request(start_state=self._state(q_start, width, held), group_name=GROUP,
                                       link_name=TIP, waypoints=[_pose_msg(T) for T in poses], max_step=float(step),
                                       jump_threshold=0.0, avoid_collisions=bool(avoid_collisions),
                                       max_velocity_scaling_factor=1.0, max_acceleration_scaling_factor=1.0)
        req.header.frame_id = FRAME
        r = self._call("cartesian", req)
        path = self._path(r.solution.joint_trajectory) if r.solution.joint_trajectory.points else [q_start]
        if r.fraction < min_fraction - 1e-6:
            why = self._why_stopped(path[-1], poses, step, width, held) if avoid_collisions else ""
            raise PlanError(f"MoveIt's Cartesian path gets only {100 * max(r.fraction, 0):.0f}% of the way"
                            + (f": {why}" if why else " (no inverse kinematics on the line)"))
        # A straight line must not flip a joint (a jump of the IK branch).
        for a, b in zip(path, path[1:]):
            if np.abs(b - a).max() > 0.3:
                raise PlanError("MoveIt's Cartesian path jumps between arm configurations")
        return path

    def _why_stopped(self, q_last, poses, step, width, held):
        """What the robot would touch one step beyond where a Cartesian path
        stopped ('' if nothing: then the inverse kinematics failed)."""
        from .robot.kinematics import URKinematics
        kin = getattr(self, "_kin", None) or URKinematics(self.cell["robot"]["ur_type"])
        self._kin = kin
        T = kin.fk(q_last)
        # The waypoint the path was heading to: the first one not reached.
        goal = next((np.asarray(P, float) for P in poses
                     if np.linalg.norm(np.asarray(P)[:3, 3] - T[:3, 3]) > step / 2), None)
        if goal is None:
            return ""
        d = goal[:3, 3] - T[:3, 3]
        T_next = np.array(goal)
        T_next[:3, 3] = T[:3, 3] + d * min(1.0, 2 * step / max(np.linalg.norm(d), 1e-9))
        q = kin.ik_near(T_next, q_last)
        if q is None:
            return ""
        c = self.contacts(q, width, held)
        return ("the robot would touch " + ", ".join(c) + f" at {np.round(T_next[:3, 3], 3)}") if c else ""

    def ik(self, T, q_seed, width, held=None, avoid_collisions=True, timeout=0.2):
        """MoveIt's inverse kinematics (KDL, from q_seed) for rg6_base_link at T, or None."""
        from geometry_msgs.msg import PoseStamped
        from moveit_msgs.msg import MoveItErrorCodes, PositionIKRequest
        from moveit_msgs.srv import GetPositionIK
        ps = PoseStamped(pose=_pose_msg(T))
        ps.header.frame_id = FRAME
        req = PositionIKRequest(group_name=GROUP, robot_state=self._state(q_seed, width, held),
                                avoid_collisions=bool(avoid_collisions), ik_link_name=TIP, pose_stamped=ps)
        req.timeout.sec = int(timeout)
        req.timeout.nanosec = int((timeout % 1) * 1e9)
        r = self._call("ik", GetPositionIK.Request(ik_request=req))
        if r.error_code.val != MoveItErrorCodes.SUCCESS:
            return None
        d = dict(zip(r.solution.joint_state.name, r.solution.joint_state.position))
        return np.array([d[j] for j in ARM_JOINTS])


def _error_name(val):
    from moveit_msgs.msg import MoveItErrorCodes
    for k in dir(MoveItErrorCodes):
        if k.isupper() and getattr(MoveItErrorCodes, k) == val:
            return k.lower().replace("_", " ")
    return f"error {val}"

