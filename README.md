# physAI-arm-ex

A **UR5e** robot arm with an **OnRobot RG6** gripper and a **RealSense**
camera on its wrist handles the twelve items of the lab's order-assembly
task: it picks them up and puts them on the table or into the basket. You
tell it what to do in one of two ways:

* **one action at a time:** you type `(pick-up green-mug)` and the robot does it;
* **a goal:** you type `(in-basket green-mug) (in-basket red-can)` and the
  robot plans the actions itself (the Unified Planning Framework with Fast
  Downward) and does them one after
  the other.

Either way the robot works out the details itself. It looks with the wrist
camera, recognises the items (SAM 3), finds a grasp, plans a collision-free
motion (MoveIt) and moves. The same programs run the real robot and a Gazebo
simulation of the cell.

The task planning, the state estimation, the motion planning and the skills
(the PDDL actions performed by the robot) are in one folder of plain Python,
without ROS:
[`src/lab_replica/lab_replica/exercise/`](src/lab_replica/lab_replica/exercise/)
([The exercise](#the-exercise-task-planning-state-estimation-motion-planning-skills)).

Repository: <https://github.com/guyazran/physAI-arm-ex> (it continues the
earlier `lab-tamp` workspace; see [Reference](#reference)).

**Contents**

1. [Run it](#run-it)
2. [The exercise](#the-exercise-task-planning-state-estimation-motion-planning-skills):
   task planning, state estimation, motion planning, skills
3. [Build (once)](#build-once)
4. [Run it in simulation](#run-it-in-simulation)
5. [Run it on the real robot](#run-it-on-the-real-robot): the
   [UR driver](#1-the-ur-driver-on-air),
   [the RealSense](#2-the-realsense-on-guest) and
   [the RG6 driver](#3-the-rg6-gripper-driver-on-air)
6. [The action console](#the-action-console)
7. [The goal console](#the-goal-console)
8. [What the robot keeps clear of](#what-the-robot-keeps-clear-of)
9. [Watching it, and when something goes wrong](#watching-it-and-when-something-goes-wrong)
10. [Reference](#reference): how the skills work, `solve.py`, the
   measurements, the simulation, where everything is

---

## Run it

Once, before the first run: `run/setup.sh` ([Build (once)](#build-once)).

**First, the robot.** In simulation, start it on this PC and leave it
running:

```console
run/sim.sh                  # Gazebo: the arm, the gripper and the wrist camera; ready at "RG6 driver ready (sim)"
```

On the real robot, its drivers and the camera run on other computers of the
lab (air and guest): [Run it on the real robot](#run-it-on-the-real-robot).

**Then, in another terminal, one of these.** The first argument says where:
**`sim`** (the Gazebo simulation) or **`real`** (the robot in the lab).

| Command | What it starts | Ready when it prints |
|---|---|---|
| `run/actions.sh sim` or `real` | the **action console**: you type one action at a time | `Ready.` and the prompt `action>` |
| `run/goals.sh sim` or `real` | the **goal console**: you type a goal | `Ready.` and the prompt `goal>` |
| `run/workspace_view.sh sim` or `real` | the **workspace view**: the camera with SAM 3's masks, and the arm's joints as they change, to find the [observation configuration](#the-observation-configuration) (it never moves the arm) | `OBSERVATION_JOINTS = [...]`, updating |

* Run only **one** of them at a time: each console drives the robot. To
  switch, quit one (Ctrl-D) and start the other; the robot keeps running.
* All terminals must use the same `ROS_DOMAIN_ID`. Each script prints the
  one it uses; the lab PC sets 131 in `~/.bashrc`. Never run the simulation
  and the real robot on the same `ROS_DOMAIN_ID` at the same time.
* The scripts set up everything they need themselves (ROS 2, this workspace,
  the Python environment of SAM 3), so they work from any directory, e.g.
  `~/physAI-arm-ex/run/sim.sh`.
* `--help` after `run/actions.sh sim` or `run/goals.sh sim` lists the
  options; `run/sim.sh --help` shows how to start another scene.
* How the scripts were tested (simulation, and the real-robot mode without
  the robot): [verification/README.md](verification/README.md#2026-10-14-the-consoles-and-the-depth-obstacle-model).

## The exercise: task planning, state estimation, motion planning, skills

Everything the robot decides is in
[`src/lab_replica/lab_replica/exercise/`](src/lab_replica/lab_replica/exercise/).
These files are plain Python: no ROS. The rest of the code (the consoles,
the camera, SAM 3, MoveIt's `move_group`, the drivers) calls them, so a
change there shows up the next time a console runs, in simulation or on the
robot (after `colcon build`, which `run/setup.sh` did; the files are
linked, so editing them needs no new build).

| File | Function | What it does | Called by |
|---|---|---|---|
| `task_planning/domain.pddl` | | the PDDL domain: the actions' preconditions and effects (their names and parameters are fixed: the robot runs exactly these) | |
| `task_planning/problem.pddl` | | the PDDL problem: its objects (the items) and a goal. The goal console keeps the objects (adding items added with `new-item:`) and replaces `:init` with the current state and `:goal` with the goal typed (`tamp/problem_file.py`) | the goal console, `solve.py` |
| `task_planning/planning.py` | `load_problem` | the domain and the problem read by UPF (`PDDLReader`) | |
| | `solve` | the plan: UPF's `OneshotPlanner(name="fast-downward")` | |
| | `expected_states`, `facts_of` | the state after each action (UPF's `SequentialSimulator`) | |
| | `execute_plan` | the robot performs the actions one by one, stopping at the first failure | the goal console |
| `state_estimation.py` | `OBSERVATION_JOINTS` | the [observation configuration](#the-observation-configuration): the arm's joints from which the wrist camera sees the whole workspace | every action (`robot.survey(OBSERVATION_JOINTS)`) |
| | `SIDE_VIEW`, `close_views` | [other views](#more-views): a camera pose looking at the workspace from the front at an angle; camera poses straight down at a spot from 0.50 m | `(pick-up X)` |
| | `pixels_to_world` | depth pixels to 3D points in the world frame (pinhole camera) | every look |
| | `estimate_placement` | an item's 3D points, and those around it, to where and how it rests (support, resting pose, x, y, yaw) | every look |
| | `height_map`, `pose_cost`, `fit_pose` | its model seen from above (`toolkit.model_height_map`) matched to the heights of the points: the same for every kind of item | |
| | `assign_masks` | SAM 3's masks in one image to at most one per item | every look |
| | `fuse_views` | the items found in several images to one estimate per item | every look |
| | `grasp_in_image` | top-down grasps proposed in the image: a pixel and a closing direction | `(pick-up X)` |
| | `grasp_pose` | a grasp in the image to the gripper's 3D pose and opening | `(pick-up X)` |
| `motion_planning.py` | `choose_solution` | of the inverse kinematics solutions, the nearest to the arm's configuration | every motion |
| | `camera_to_gripper`, `held_box` | frames: the gripper pose for a camera pose; the held item's box in the gripper's frame | |
| | `joint_line`, `straight_path`, `plan_path` | the straight line in joint space when nothing is in its way; else straight up first, then the line; else MoveIt's planner | every motion |
| | `move_to_pose`, `plan_view`, `plan_home` | collision-free motions to a gripper pose, a camera view, home | every look, the way home |
| | `plan_observation` | the collision-free motion to a configuration to look from (the observation configuration, ...) | every survey from joints (`robot.survey`) |
| | `plan_pick` | above the grasp, open, straight down, close, straight up | `(pick-up X)` |
| | `plan_place` | carry, straight down, open, straight up | `(put-on-table X)`, `(put-in-basket X)` |
| | `sample_spots`, `is_free`, `find_table_spot` | where on the table an item can be put | `(put-on-table X)` |
| | `scene_obstacles` | what MoveIt keeps clear of besides the table and the wall | every motion |
| `skills.py` | `pick_up` | `(pick-up X)`: check the gripper is empty, look (also from the side when X is not recognised from above, then from closer), find X, propose grasps in the image, plan the grasp, grasp, check the grip | every action |
| | `put_on_table` | `(put-on-table X)`: check the gripper holds X, look, find a free table spot, put X there, check it let go | every action |
| | `put_in_basket` | `(put-in-basket X)`: check the gripper holds X, look, find the basket's emptiest spot, drop X there, check it let go | every action |
| `pyramid.py` | `Layout`, `Pyramid`, `facts`, `clear_of` | `pyramid:` in the goal console: where the six slots of a 3-2-1 pyramid are, what was put where, what is in the way ([A pyramid of cubes](#a-pyramid-of-cubes)); `skills.py` `place_in_slot` puts a cube in a slot | `pyramid:` in the goal console |
| `scene_model.py` | `scan_environment`, `static_obstacles`, `boxes_from_height_map` | `scan:` in a console: the views a fixed barrier is scanned from, and its boxes from the height map of those views (see [What the robot keeps clear of](#what-the-robot-keeps-clear-of)) | `scan:` in a console |
| `misc.py` (bonus) | `new_item` | `new-item: NAME`: find an unknown item by its text prompt (SAM 3), measure it, add it | `new-item:` in a console |
| | `forget_item` | `forget-item: NAME`: forget an added item | `forget-item:` in a console |

The robot performs the action `(name X)` with the function of the same name
in `skills.py` (`-` becomes `_`), given the robot and the action's
parameters: `(pick-up red-can)` is `pick_up(robot, "red-can")`. The console
checks when it starts that every action of `domain.pddl` has its function.
A skill returns a message or raises `ActionError` (or `PlanError`); after
it succeeds the robot returns home, after it fails having moved the arm
with nothing left in the gripper it opens the gripper and rises. Only the
skill functions are public in `skills.py`; its helpers start with `_`.

What these files may use from the rest of the code is in two files:

* [`lab_replica/toolkit.py`](src/lab_replica/lab_replica/toolkit.py): the
  data types (`Placement`, `Estimate`, `Detection`, `ImageGrasp`, `Grasp`,
  `Held`, `Box`, `ActionError`), small geometric fits, the robot's
  kinematics (`ik_all`, `fk`, `joint_limits`, the RG6's `depth_from_width`
  and `free_height`), the steps of a robot plan (`trajectory`,
  `gripper_step`) and the pictures of what the robot decided, for the
  camera window (`show_image_grasps`, `show_grasp`, `show_table_spot`,
  `show_drop`, `show_new_item`; after every look and every segmentation the
  robot shows the masks and the items' names by itself).
* [`lab_replica/moveit_client.py`](src/lab_replica/lab_replica/moveit_client.py):
  MoveIt without ROS: `plan_to_joints`, `plan_to_pose`, `cartesian_path`,
  `is_valid`, `set_obstacles`. Poses are 4x4 numpy matrices of the gripper's
  mounting face (`rg6_base_link`) in the world frame; joint paths are lists
  of the six joint angles.

The skills also get the robot itself (`autonomy/executive.py`): looking
(`robot.survey`, from the arm's joints or a camera pose, adding to the
scene of earlier views), the gripper (`robot.holding`,
`robot.width`), MoveIt's planning scene (`robot.set_obstacles`), running a
plan's steps (`robot.run`) and what the gripper holds; the top of
`skills.py` (and of `misc.py`) lists everything.

**Checking the exercise offline** (no robot, no simulation, no MoveIt), in
the clone's top directory: each function is compared with cases whose answer
is known (a synthetic depth image of a block and a can seen from above,
goals with known plans, arm configurations):

```console
source /opt/ros/humble/setup.bash && source install/setup.bash && source .venv/bin/activate
python3 src/lab_replica/tools/check_exercise.py            # or: ... check_exercise.py task state observation motion
```

The motions themselves need MoveIt: try them in simulation
([Run it in simulation](#run-it-in-simulation)).

### The observation configuration

Every action starts by looking at the whole workspace from one arm
configuration, `OBSERVATION_JOINTS` in `exercise/state_estimation.py` (the
six joint angles in radians, from `shoulder_pan` to `wrist_3`). SAM 3 finds the
items and the basket in that one image, so from there the camera must see
the whole blue board, with the basket anywhere on it, close enough for SAM 3
to recognise every item, and the arm must stand there (also holding an item)
without touching the table or the wall.

To find it, with the robot and the camera running ([Run it](#run-it)):

```console
run/workspace_view.sh real
```

A window shows the camera live (colour and depth) and below it the same
images with SAM 3's masks of the items and the basket, as often as SAM 3
keeps up (about once a second), with the items it does not find listed at
the bottom. The terminal shows the arm's joints, updating as the arm moves,
as a line to copy, and where the camera is. Move the arm by hand (freedrive
on the teach pendant) until every item and the basket are found, wherever
they stand on the board, then copy the `OBSERVATION_JOINTS = [...]` line
into `exercise/state_estimation.py`. `s` in the window prints the joints on a line
of their own (to compare a few candidates); `q` quits and prints them once
more. `--min-score` sets the SAM 3 score from which masks are shown
(default 0.3). The workspace view only reads the joints: it never moves the
arm. In simulation (`run/workspace_view.sh sim`) the arm cannot be moved by
hand.

`check_exercise.py observation` checks offline that the configuration is
within the joint limits and that the camera sees the board's corners from
there; the next action in a console moves there (the motion is MoveIt's,
`plan_observation`; a configuration where the arm would touch something is
refused, naming what it touches).

**MoveIt** runs as `move_group`, started by the console (or the skill
server) when none is running, with the robot model of
`description/urdf/lab_robot.urdf.xacro` and `config/moveit/`
(`lab_robot.srdf`: the planning group `arm` from `base_link` to
`rg6_base_link`; OMPL's RRT-Connect; KDL inverse kinematics). Its log is
`~/.ros/lab_move_group.log`. It only plans: the console executes the
trajectories, timed by `toolkit.trajectory`. To start it by hand (e.g. with
RViz): `ros2 launch lab_replica moveit.launch.py rviz:=true`.

### More views

`robot.survey(view, scene=None, held=None, exclude=())` is the robot's only
way of looking. `view` is the arm's six joints, a camera pose (a 4x4 matrix:
the camera's optical frame in the world; `toolkit.look_at` and
`toolkit.down_view` make one), or a list of them, of which the first
reachable is used. Given the `scene` of earlier views, it adds what the new
view sees: an item is recognised from any view (each frame's detections are
merged by `fuse_views`), and its pose comes from the best view that sees all
of it: looking straight down before looking at an angle, and nearer before
farther. An item a view finds more than 5 cm from where the scene has it is
taken to be something else there.

The observation configuration may not be enough: from straight above some
items are hard for SAM 3 to name, and from 0.8 m the depth is only so
accurate. Other views can be found by hand like the observation
configuration (more joints), or computed (camera poses); the skills decide
when to use them. The solution's `pick_up` looks from the front
(`SIDE_VIEW`, a camera pose) when the top view did not recognise the item
with a score of at least 0.5, and then straight down at it from 0.50 m
(`close_views(x, y)`: eight turns of the camera about the vertical, since
not every one is reachable).

## Build (once)

On the lab PC this is done. On another computer that runs the consoles
(Ubuntu 22.04 with [ROS 2 Humble](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html),
`ros-humble-desktop`, and an NVIDIA GPU), first, once, by an administrator
(the only step that needs `sudo`):

```console
sudo apt install ros-humble-control-msgs ros-humble-controller-manager-msgs ros-humble-xacro \
                 ros-humble-ur-description python3.10-venv python3-colcon-common-extensions \
                 ros-humble-moveit-ros-move-group ros-humble-moveit-planners-ompl \
                 ros-humble-moveit-kinematics ros-humble-moveit-simple-controller-manager
```

Then, as any user:

```console
git clone https://github.com/guyazran/physAI-arm-ex.git
cd physAI-arm-ex
run/setup.sh
```

Clone it into any directory, under any name: the code and the scripts find
the workspace from their own location. The commands in this README run in
the clone's top directory (the one with `run/` and `src/`). Moving it after
the setup breaks `.venv` and the build (they hold absolute paths): delete
`.venv/`, `build/`, `install/` and `log/` there and run `run/setup.sh` again.

`run/setup.sh` does the rest, without `sudo`, and skips what is done
already:

1. checks the system packages above (if some are missing, it prints the
   `apt` command and stops);
2. makes the Python environment `.venv` and installs
   [`requirements.txt`](requirements.txt) into it: torch and transformers
   for SAM 3, the Unified Planning Framework with Fast Downward for the goal
   console (about 5 GB, mostly torch);
3. asks for your Hugging Face token (saved in `.env`) and downloads SAM 3;
4. builds the ROS workspace (`colcon build --symlink-install`);
5. checks that ROS, torch with the GPU and the planner work.

* SAM 3 (`facebook/sam3`, 3.4 GB) is a gated model: accept its licence on
  [Hugging Face](https://huggingface.co/facebook/sam3) and make a read token
  (Settings, Access Tokens) before running the script. It needs about 3 GB
  of GPU memory.
* Options: `--no-model` (download SAM 3 on first use instead),
  `--fresh-venv` (make `.venv` again).
  The token can also come from `$HF_TOKEN`.
* The `.venv` sees the ROS and Ubuntu Python packages
  (`--system-site-packages`); numpy stays below 2, which ROS Humble needs.
* This sets up a PC for the consoles. The robot's drivers and the camera
  node run on other computers ([the real robot](#run-it-on-the-real-robot)).
  The simulation needs more packages:
  `sudo apt install ros-humble-ros-gz-sim ros-humble-ros-gz-bridge ros-humble-ign-ros2-control ros-humble-controller-manager ros-humble-joint-state-broadcaster ros-humble-joint-trajectory-controller ros-humble-position-controllers`.
* `.venv/`, `.env`, `build/`, `install/`, `log/` and the
  outputs (`runs/`, `videos/`, logs) are git-ignored.
* The `run/` scripts source ROS 2, `install/setup.bash` and the `.venv`
  themselves; nothing has to go into `~/.bashrc`.
* **After pulling new code,** run `run/setup.sh` again (new packages,
  added or renamed files). Edits to the YAML files and the Python
  code take effect without rebuilding (`--symlink-install`).
* The other two computers (air and guest) need **no build**: see
  [the real robot](#run-it-on-the-real-robot).

## Run it in simulation

In the clone's top directory:

```console
run/sim.sh                  # terminal 1: wait for "RG6 driver ready (sim)"
run/actions.sh sim          # terminal 2: the action console ...
run/goals.sh sim            #             ... or the goal console
```

* The Gazebo window shows the cell. `run/sim.sh gui:=false` runs without
  it.
* The simulated cell starts with **all twelve items on the table**, the
  basket empty and the gripper empty (the scene
  [`problems/all_on_table`](problems/all_on_table)). That is what the goal
  console assumes when it starts.
* Another scene: `run/sim.sh t_newcell_1` (any directory name in
  `problems/`, `problems_tamp/` or `problems_gen/`).
* **To start again from the beginning,** stop all terminals (Ctrl-C) and
  start them again from terminal 1. The simulation then starts from its scene
  again.
* The simulation runs slower than real time on the lab PC: an action takes
  30 to 60 s.

## Run it on the real robot

**The lab's set-up (2026-10-18).** Three computers on ROS_DOMAIN_ID 131 (the
lab's allocation for this project):

| Computer | Runs | Needs |
|---|---|---|
| **air** (`ssh air@10.10.34.14`) | 1. the UR driver | `ros-humble-ur` (apt) |
| **air**, second terminal | 3. the RG6 gripper driver | this repository (no build) |
| **guest** (`ssh guest@132.68.35.80`) | 2. the RealSense (plugged in there) | `ros-humble-realsense2-camera` (apt) |
| **this PC** | the consoles (SAM 3 needs its GPU) | this repository, set up with [`run/setup.sh`](#build-once) |

**Before starting any of the drivers,** in every terminal on air and on
guest (their login shells set up neither ROS nor the domain):

```console
source /opt/ros/humble/setup.bash && export ROS_DOMAIN_ID=131
```

### 1. The UR driver (on air)

```console
ssh air@10.10.34.14
source /opt/ros/humble/setup.bash && export ROS_DOMAIN_ID=131
ros2 launch ur_robot_driver ur_control.launch.py ur_type:=ur5e robot_ip:=10.240.10.4 initial_joint_controller:=passthrough_trajectory_controller
```

Then press **Play** on the pendant (program `guy.urp`). Ready when it prints
`Robot connected to reverse interface. Ready to receive control commands.`

* It is the stock driver from apt (`ros-humble-ur`); it does not use this
  repository.
* `passthrough_trajectory_controller`: the robot interpolates each motion
  itself. With the default (scaled) controller air has to stream a setpoint
  every 2 ms; it misses some (no real-time kernel), the robot drops the
  connection and the pendant program stops (driver log: `Connection to
  reverse interface dropped`). Press Play again after that.

### 2. The RealSense (on guest)

```console
ssh guest@132.68.35.80
source /opt/ros/humble/setup.bash && export ROS_DOMAIN_ID=131
ros2 launch realsense2_camera rs_launch.py align_depth.enable:=true rgb_camera.color_profile:=640x360x6 depth_module.depth_profile:=640x360x6
```

Ready when it prints `RealSense Node Is Up!`.

* It is the stock driver from apt (`ros-humble-realsense2-camera`); it does
  not use this repository.
* **640 x 360 at 6 Hz** (instead of the default 1280 x 720 at 30 Hz):
  full-size images do not get across the network. Even these arrive only
  after this PC's UDP receive buffer was raised (needed after every reboot,
  unless made permanent in `/etc/sysctl.d/`):
  ```console
  sudo sysctl -w net.core.rmem_max=2147483647 net.core.rmem_default=33554432 net.ipv4.ipfrag_time=3 net.ipv4.ipfrag_high_thresh=134217728
  ```
  Check: `ros2 topic hz /camera/camera/color/image_raw` on this PC shows
  about 6. 640 x 360 keeps the full field of view (16:9).

### 3. The RG6 gripper driver (on air)

The only driver that needs **this repository**. Once, on air:

```console
git clone https://github.com/guyazran/physAI-arm-ex.git          # any directory
```

Then, each time, in a second terminal on air:

```console
ssh air@10.10.34.14
source /opt/ros/humble/setup.bash && export ROS_DOMAIN_ID=131
cd physAI-arm-ex && git pull          # the clone; same branch as this PC
python3 src/lab_replica/scripts/rg6_driver.py --ros-args -p backend:=onrobot -p ip:=10.240.10.4 -p port:=41414 -p tool_index:=0 -r /joint_states:=/rg6/joint_states
```

Ready when it prints `RG6 driver ready (onrobot)`.

* **No `colcon build` and no `.venv`:** the script puts `src/lab_replica` on
  the Python path and runs from the source tree. It needs only ROS 2 Humble
  (`rclpy`, `sensor_msgs`, `std_msgs`, `control_msgs`; the last comes with
  `ros-humble-ur`).
* The RG6 works through the OnRobot URCap on the robot (XML-RPC on port
  41414 of the robot's address); nothing has to be installed there.
* Its driver runs on air because air is wired to the robot's network. This
  PC reaches it only over Wi-Fi, where the XML-RPC calls time out and the
  console then finds no gripper reading.
* Run only one RG6 driver: not also `real.launch.py` on this PC.

### Then, on this PC

* The three computers' clocks differ by up to two hours. The programs here
  judge whether a camera frame is new by when it arrives, not by its time
  stamp, so this does not matter to them.

**The cell** ([`config/cell.yaml`](src/lab_replica/config/cell.yaml)): the
slatted table is 1.26 m x 0.85 m, the robot near its back-left corner, the
blue board (0.70 m x 0.59 m, the workspace) to its right. The world frame's
origin is on the table below the robot's base axis, x towards the board, y
towards the wall (the robot's `base_link` is turned 180 degrees from it).
The wall behind the table is an obstacle (`room.wall_y`); the robot may move
anywhere else above and around the table. The basket may stand anywhere on
the board: the robot finds it in every survey.

**Camera pose (hand-eye calibration).** `wrist_camera` in `cell.yaml` was
calibrated on 2026-10-18. Calibrate again if the camera is moved on the
gripper (the scene must stay still; the camera stays 0.55 m or more above
the table):
```console
python3 src/lab_replica/tools/calibrate_wrist_camera.py capture runs/calib_<date>   # moves the arm through 10 views
python3 src/lab_replica/tools/calibrate_wrist_camera.py solve runs/calib_<date>     # prints the new optical_xyz / optical_rpy
```

**Each time** (in the clone's top directory, with `ROS_DOMAIN_ID=131`):

1. Start the UR driver and the RG6 driver on air and the camera on guest
   ([1](#1-the-ur-driver-on-air), [2](#2-the-realsense-on-guest),
   [3](#3-the-rg6-gripper-driver-on-air) above), press **Play** on the pendant.
2. **The scene.** For the goal console: every item on the board, the
   basket empty (anywhere on the board), the gripper empty. Leave at least
   2 cm between items.
3. **On this PC:** the action console **or** the goal console, slowly at
   first:
   ```console
   run/actions.sh real --speed-scale 0.3                  # the action console, or ...
   run/goals.sh real --speed-scale 0.3 --step             # ... the goal console (--step: Enter before every action)
   ```

**Safety.** Keep a hand on the emergency stop. **Ctrl-C** in the console terminal
while an action runs stops the arm and returns to the prompt. Ctrl-C at the
prompt quits. The pendant's speed slider scales every motion, on top of
`--speed-scale`.

## The action console

```text
$ run/actions.sh sim
lab-tamp: action console  (simulation)
  connecting to the arm and the gripper ... ok (controller joint_trajectory_controller)
  loading SAM 3 (the segmentation model) ... ok (7 s)
  checking the gripper ... ok (empty)
  waiting for the wrist camera ... ok (1280 x 720)
Ready. Log and camera views: /home/guy/Workspace/physAI-arm-ex/runs/20261013_225357

Actions: (pick-up <item>)   (put-in-basket <item>)   (put-on-table <item>)
Items:   red-can  red-spray-bottle  roll-of-black-tape  roll-of-green-tape  roll-of-blue-tape
         blue-cup-with-yellow-stripe  green-mug  yellow-block  orange-block  blue-block  red-block
         green-block
Example: (pick-up green-mug)   then   (put-in-basket green-mug)
Tab completes names; Ctrl-D quits.
action> (pick-up gren-mug)
invalid action string: unknown item 'gren-mug' (did you mean green-mug?)
            ... (the hints again)
action> (pick-up green-mug)
performing action (pick-up green-mug)
  ... picking up green-mug                          <- one line that shows what it is doing
action success: holding green-mug (width 70 mm)
            ... (the hints again)
action> (put-in-basket green-mug)
performing action (put-in-basket green-mug)
action success: put green-mug at (0.390, 0.070) on the basket
action> (put-on-table red-can)
performing action (put-on-table red-can)
action failed: not applicable: the gripper is empty
```

**Typing an action.** An action is its name and one item, as in
`(pick-up green-mug)`. The parentheses and capital letters do not matter.
Tab completes names, and the arrow keys bring back earlier lines. If the
text is not an action, the answer is `invalid action string: <why>`, and
nothing is sent to the robot.

**The actions:**

| Action | Possible when | What the robot does |
|---|---|---|
| `(pick-up X)` | the gripper is empty | looks from above (and from the front if it does not recognise X from above), finds X, looks at it from closer, grasps it, lifts it, returns home |
| `(put-in-basket X)` | the gripper holds something | looks, finds the emptiest spot in the basket, drops X there from just above the rim (it may land on other items), returns home |
| `(put-on-table X)` | the gripper holds something | looks, puts X down on the free table spot nearest to where it was picked up, returns home |

**The answer.** `performing action` is printed at once. One line then shows
what the robot is doing, and when it has finished:

* `action success: <what it did>`, or
* `action failed: <reason>`. A refused action (not possible now, item not
  found, no collision-free motion) has not moved anything. A failure during
  a motion says what happened, e.g. that the item slipped out of the gripper.

The robot checks for itself whether the gripper holds something (the RG6's
grip sensor). It assumes nothing else about the state: it looks before
every action.

**Options:** `--no-window` (no [camera window](#the-camera-window)), `--speed-scale S` (slower or faster motions), `--no-home`
(stay where the action ended), `--verbose` (print every step, not only the
current phase), `--save-frames` (also save every camera image, for
[debugging](#watching-it-and-when-something-goes-wrong)).

### The camera window

Both consoles, the action console and the [goal console](#the-goal-console), open a window
(`lab_replica/viewer.py`) with:

* **top:** the wrist camera live, colour and depth side by side (the depth's
  colours run from near, red, to a little beyond the table, blue; the bar
  in the corner gives the range in metres);
* **bottom:** every view the robot takes during an action, annotated, again
  colour and depth side by side. Survey and closer views show the SAM 3
  masks, each item's model where it is estimated (wire frame) with its
  name and score, the basket's rim and the obstacles (seen red, unseen
  purple). A pick-up adds the planned grasp: the pads open (white) and
  where they close (green), the grasp width and the force. A put adds its
  plan: the free spot on the table, or the drop spot in the basket. A new
  item shows its mask and the model measured from it;
* **the status bar:** what the robot is doing now (in the goal console
  also: planning, the plan's outcome, waiting for Enter with `--step`).

Keys in the window: `a` / `d` (or the arrow keys) go to earlier and later
views, `l` back to the newest. The views are also saved in the run
directory (colour and depth side by side). Closing the window does not stop
the console.

The window needs a display: on the lab PC's screen, or `ssh -X`. Without one
the console says so and carries on. `--no-window` leaves it out. The window
gets its images from the console itself, so the camera's images cross the
network once (they barely fit on the real robot's network).

### Add a new item

Put the item on the table, where the survey view sees it and away from the
other items, then type its name after `new-item:`:

```console
action> new-item: purple-block
measuring new item purple-block: looking for 'purple block'
item added: purple-block is a box 58 x 29 x 28 mm (length x width x height), SAM 3 score 0.96
    note: top outline fills 0.99 of its rectangle: box
action> (pick-up purple-block)
```

The robot looks from above, asks SAM 3 for the name with the dashes as
spaces (`purple block`), looks at what it found from closer, and measures its
width and height from the SAM 3 mask and the depth. The new item is a box,
or a round item when the outline of its top is round. The planner grasps it
like the other items, with the same 40 N (`gripper.grip_force` in
`cell.yaml`): a box across its short side, or across its long side when the
gripper opens wide enough; a round item from any side. The picture of what
was measured is in the run directory (`..._new_item_<name>.jpg`).

| Command | What it does |
|---|---|
| `new-item: NAME` | measure the item and add it; NAME is lowercase words joined by `-` |
| `new-item: NAME prompt="small purple cube"` | look for it by another description (when SAM 3 does not find it by its name) |
| `new-item: NAME shape=box` (or `round`) | do not guess the shape |
| `forget-item: NAME` | remove an added item |

The colon may be left out (`new-item purple-block`). The console's hints
show the command, and an action with an unknown item suggests it.

**An added item is temporary.** It lasts until the console quits and is not
written anywhere: the next console starts with the twelve items again.
The domain (`exercise/task_planning/domain.pddl`) and the problem files are unchanged. In the goal
console an added item is also an object of the planning problem for the
rest of the session ([The goal console](#the-goal-console)). Quitting while
the gripper holds an added item asks first (quit again to leave it in the
gripper); the next session then knows only that the gripper is not empty.

The code: `lab_replica/objects/measured.py` (the model),
`perception/estimate.py` `measure_item`, `autonomy/executive.py` `new_item`,
`console.py` `parse_item_command` and `item_command`.

Limits: the gripper must be empty; the item must stand on the table (not in
the basket), be at most 141 mm wide and 30 cm tall. The fingertips stop
8 mm above the table, so items lower than about 15 mm are not held well. In
simulation (tested 2026-10-20 with a 60 x 30 x 28 mm box and a 45 x 40 mm
cylinder spawned into Gazebo) the sizes came out 1-3 mm small; pick-up,
put-in-basket and put-on-table worked for both.

## The goal console

```text
$ run/goals.sh sim
lab-tamp: goal console  (simulation)
  connecting to the arm and the gripper ... ok (controller joint_trajectory_controller)
  loading SAM 3 (the segmentation model) ... ok (5 s)
  checking the gripper ... ok (empty)
  waiting for the wrist camera ... ok (1280 x 720)
Ready. Log and camera views: /home/guy/Workspace/physAI-arm-ex/runs/20261013_231313

State:   on the table: red-can  red-spray-bottle  roll-of-black-tape  roll-of-green-tape
                       roll-of-blue-tape  blue-cup-with-yellow-stripe  green-mug  yellow-block
                       orange-block  blue-block  red-block  green-block
         in the basket: nothing
         gripper: empty
Goal:    facts that must all hold at the end, for example
           (in-basket green-mug)
           (in-basket red-can) (robot-gripping green-mug)
           (and (in-basket red-block) (or (in-basket blue-block) (on-table green-block)))
Facts:   (on-table <item>)  (in-basket <item>)  (robot-gripping <item>)  (robot-gripper-empty)
         combined with and, or, not, imply, forall, exists  (e.g. (not (on-table red-can)))
Items:   red-can  red-spray-bottle  roll-of-black-tape  roll-of-green-tape  roll-of-blue-tape
         blue-cup-with-yellow-stripe  green-mug  yellow-block  orange-block  blue-block  red-block
         green-block
Commands: state (show the state), reset (all items on the table, gripper empty), quit. Tab completes names.
goal> (in-basket red-can) (in-basket gren-block)
invalid goal: (in-basket gren-block): unknown item 'gren-block' (did you mean green-block or red-block?)
            ... (the state and the hints again)
goal> (in-basket red-can) (robot-gripping yellow-block)
goal: (and (in-basket red-can) (robot-gripping yellow-block))
planning ...
plan (3 actions):
  1. (pick-up red-can)
  2. (put-in-basket red-can)
  3. (pick-up yellow-block)
performing action 1/3: (pick-up red-can)
action success: holding red-can (width 56 mm)
performing action 2/3: (put-in-basket red-can)
action success: put red-can at (0.370, 0.040) on the basket
performing action 3/3: (pick-up yellow-block)
action success: holding yellow-block (width 50 mm)
goal reached (3 actions; the state is what they should have produced)
            ... (the state, now with red-can in the basket and yellow-block in the gripper)
goal> (on-table red-can)
goal: (on-table red-can)
planning ...
the goal cannot be reached from this state: (on-table red-can) can never hold (items in the basket never leave it; the gripper holds one item at most)
```

**Typing a goal.** A goal is one or more facts that must all hold at the
end. A fact is one of these, with one of the twelve item names (or an
[added item](#add-a-new-item)):

| Fact | Means |
|---|---|
| `(on-table X)` | X is on the table |
| `(in-basket X)` | X is in the basket |
| `(robot-gripping X)` | the robot holds X |
| `(robot-gripper-empty)` | the robot holds nothing |

Facts can be combined with `and`, `or`, `not`, `imply`, `forall` and
`exists`, as in PDDL:

```text
(in-basket green-mug)                                    one fact
(in-basket red-can) (robot-gripping green-mug)           several facts: all of them
(and (in-basket red-block) (not (on-table blue-block)))  and, not
(or (in-basket red-can) (in-basket green-mug))           either one
(forall (?x - item) (in-basket ?x))                      every item in the basket
```

A misspelt name is reported with a suggestion (`invalid goal: ...`); for a
name that may be a new item, the console also shows how to add it. A path
to a `.pddl` problem file also works: its `(:goal ...)` is used.

**What happens then.** The console plans a sequence of actions from the
current state (UPF with Fast Downward, [T.3](#t3-how-it-works)), prints it, and performs the actions
one by one, as the action console would: `performing action 1/3: ...`,
then `action success: ...` or `action failed: ...`. At the end it prints
`goal reached`. If an action fails, it stops there and prints
`goal not reached`.

**The state.** The console starts from *every item on the table and the
gripper empty*. After each successful action it updates the state with that
action's effect. It does not look to check facts such as `in-basket`, so:

* After a failed action, the state is the one before that action. A refused
  action has not moved anything. If the gripper sensor disagrees with the
  state, the console says so.
* If you move items by hand, or an item bounces out of the basket, the state
  is wrong. Type `reset` (every item on the table, the gripper empty) after
  putting everything back on the table, or restart the console.
* Items never leave the basket (the PDDL domain has no action for it). A
  goal such as `(on-table red-can)` after the can went into the basket is
  answered with `the goal cannot be reached from this state`.

**New items.** `new-item: NAME` and `forget-item: NAME` work here as in the
[action console](#add-a-new-item). An added item is then one of the items
for the rest of the session: it is on the table in the state, goals may name
it (also through `forall` and `exists`), and the planner gets it as one more
object:

```text
goal> new-item: purple-block
measuring new item purple-block: looking for 'purple block'
item added: purple-block is a box 58 x 29 x 28 mm (length x width x height), SAM 3 score 0.96
goal> (in-basket purple-block)
```

`forget-item:` removes it from the state again; `reset` puts every item,
the added ones too, on the table.

**Commands at the prompt:** `state` (show the state), `reset`, `new-item:`,
`forget-item:`, `quit` (or Ctrl-D).

**The camera window** is the [same as in the action console](#the-camera-window):
the camera live, and every view of every action of the plan, annotated.

**Options:** `--step` (press Enter before every action of a plan; `n` stops),
and the same options as the action console (`--no-window` too).

### A pyramid of cubes

`pyramid:` in the goal console builds a pyramid of six cubes of the same
edge (three in a row, two on them, one on the top), out of any cubes on the
table, without touching the [obstacle](#what-the-robot-keeps-clear-of):

```text
new-item: white-block size=0.05     # cubes that are not in config/items.yaml: once per cube and session
new-item: black-block size=0.05
new-item: pink-block size=0.05
scan:                               # the obstacle, before the pyramid
pyramid:                            # the place in config/cell.yaml (pyramid:), or  pyramid: 0.60 0.05
pyramid: auto                       # the nearest place on the board where the area is free
```

The cubes are the items of kind `block` of the pyramid's edge (`size`, 5 cm: the
orange block is 4 cm and is left out; `new-item: ... size=` makes a cube of an
exact edge without measuring it: a few millimetres of error would add up in
the layers). The slots are in
[`exercise/pyramid.py`](src/lab_replica/lab_replica/exercise/pyramid.py);
the place is the base row's middle cube (`origin`), its direction (`axis_yaw_deg`,
90: along y) and the gap between neighbours. The area round the row (4 cm more on
every side) must be free of the basket, the obstacle and other items (also the cubes
that are to be picked up: they stand in the way of the base); `pyramid:` looks
first and says what is in the way. `pyramid: auto` then looks for the nearest place
on the board where nothing is and the arm reaches every slot, and says where it is.

The planner (UPF and Fast Downward) chooses which cube goes into which slot
and the order, with the domain's one more action
`(place-in-slot ?cube ?slot)` (`domain.pddl`: a base slot, or one whose two
supporting slots are taken): twelve actions, a `pick-up` and a `place-in-slot`
for each cube. `place_in_slot` (`skills.py`) looks, checks that the cubes put
before are still there (their height in the depth map), puts the cube at the middle of
the two it rests on as the robot put them (not as the layout says, so an error of
a few millimetres in the base does not add up), turned along the row, with the
fingers closing across it, and releases it 4 mm above them. The pyramid's cubes are
kept by the robot (a cube on a cube is not recognised as an item) and are boxes in
MoveIt's planning scene for every motion, but the ones the cube rests on.

If an action fails, `pyramid:` again continues from the state (`--step` waits
for Enter before every action). `forget-pyramid:` (and `reset`) forget it;
`scan:` refuses while a pyramid stands (its upper cubes would be taken for the
obstacle).
Check it offline with `python3 src/lab_replica/tools/check_exercise.py pyramid`.

## What the robot keeps clear of

Every motion is planned by MoveIt (`exercise/motion_planning.py`), which
keeps the robot, and the item in its gripper, clear of what is in its
planning scene:

* **always:** the table top and the wall behind the table (`room.wall_y` in
  `cell.yaml`), and the robot itself;
* **what `scene_obstacles` adds** for each action, from the last look: a box
  around every recognised item (its estimated model, 5 mm larger on every
  side; not the item being picked up, nor the one in the gripper), the
  basket (a box around it up to its rim), every unknown obstacle the depth
  camera saw (depth no recognised item explains), and the space the camera
  could not see (behind a tall item, which may be an item SAM 3 missed), up
  to the height the camera's rays bound it to;
* **the item in the gripper:** its model's box, held where it was grasped
  (`held_box`), moves with the gripper;
* **a fixed barrier** (a wall, a partition), when one was scanned (below).

The straight moves down to a grasp or a place and back up are MoveIt's
Cartesian paths, checked against the same scene: the fingers may not touch
a neighbouring item on the way down. Each look still turns the depth into a
height map of the table (`autonomy/depth_world.py`, 2.5 mm cells; space the
camera could not see counts as occupied), which finds the unknown obstacles,
chooses the drop spot in the basket, and limits where an item may be put on
the table (only where the camera's views cover it, so that it can be seen,
and picked up, again). Each look saves a top view of this map in the run
directory. Here the robot is holding the red can. Grey cells are what the
camera saw (darker = higher), purple cells were not seen (shadows behind tall
items, strips outside the view), and the coloured outlines are the recognised
items:

![depth map](docs/depth_map_example.png)

**An obstacle that stays** (a big wooden cube, a wall, a partition). What a
look shows is replaced by the next look, so an obstacle that stays is scanned
once instead (`exercise/scene_model.py`). In a console (either), with a hand on
the emergency stop, since the obstacle is not known to MoveIt yet:

```text
scan:              # looks from OBSERVATION_JOINTS and the SCAN_VIEWS, keeps clear of the tallest thing that is no item
scan: all          # ... of everything that is no item, the basket or the table
forget-barrier:    # the motions no longer keep clear of it
```

The depth of all the views goes into the same height map, over the larger
region of `barrier:` in `config/perception.yaml` (only what was measured, not
the space no view saw); what no item, the basket or the table explains is
the candidate (only what at least two views saw: the arm, or a hand, in one view is not an obstacle), and the tallest piece of at least 16 cm^2 (`tallest_piece`) is
the obstacle. It becomes a few boxes (oriented along a slanted wall, one per
height of a stepped one; `static_obstacles`), which are in MoveIt's planning
scene with every motion from then on, and which the motions rise above
(`rise_z`). The log lists every such thing with its height (`seen from N
views, no item, basket or table: ...`). The scan is saved as `barrier.json` in
the run directory; `--barrier-file FILE` loads it in a later run without
scanning. A motion is planned the simplest way that is free (a straight line
in joint space, else straight up and then a line, else RRT-Connect, twice);
the console log says which one (`motion: ...`).

To pick up the yellow block and put it in the basket without touching the
obstacle, after `scan:`: `(pick-up yellow-block)` then `(put-in-basket
yellow-block)` in the action console, or `(in-basket yellow-block)` in the goal
console (the same two actions, planned).

Check it offline with `python3 src/lab_replica/tools/check_exercise.py barrier`.

## Watching it, and when something goes wrong

* **What the robot saw:** the action console's
  [camera window](#the-camera-window), or
  `ros2 run rqt_image_view rqt_image_view /lab_skills/detections` (also for
  the goal console): what SAM 3 found in every view, the planned grasp, and
  for a put, where the item goes.
* **The run directory** (printed at `Ready.`, under `runs/`) holds, for
  every look:
  * `console.log`: every step of every action;
  * `*_depth_map.png`: the obstacle map above;
  * `*.jpg`: the annotated camera views (colour and depth side by side), the
    grasps and the put plans;
  * `*_survey.yaml`: the perceived item positions;
  * `frame_*.npz`: the raw camera images, with `--save-frames`.
* **Replaying saved frames** offline, without the robot. In
  the clone's top directory, with the frames of one action (e.g. its top view,
  front view and closer look):
  ```console
  source /opt/ros/humble/setup.bash && source install/setup.bash && source .venv/bin/activate
  python3 src/lab_replica/tools/depth_map.py runs/<run>/frame_A.npz runs/<run>/frame_B.npz --pick red-can
  ```
  It prints what SAM 3 finds and why a grasp is refused, and writes the
  depth map.

| Message | What to do |
|---|---|
| `no arm: ... Is the robot running?` | start `run/sim.sh` (real: the UR driver on air, and press Play on the pendant); check `ROS_DOMAIN_ID` |
| `no camera images on /camera/camera/...` | start `run/sim.sh` (real: the RealSense on guest); check `ros2 topic hz /camera/camera/color/image_raw` |
| `No RealSense devices were found!` (on guest) | the camera is not plugged in, or not into a USB 3 port; replug it, then restart the RealSense node |
| `invalid action string: ...`, `invalid goal: ...` | a typo: the message says what is wrong |
| `action failed: X not found` | SAM 3 did not recognise X (e.g. hidden behind a taller item); look at the camera view in the run directory; the prompts are in `config/perception.yaml` |
| `action failed: cannot pick: MoveIt's Cartesian path gets only N% of the way` | the fingers would touch a neighbouring item on the way down, for every grasp tried; move items a little apart |
| `... is out of the arm's reach` | the pose is too far from the robot's base (a tall item held high over the far side of the basket); move the basket closer |
| `MoveIt is not running`, `move_group did not come up` | see `~/.ros/lab_move_group.log` |
| `action failed: not applicable: the gripper is empty` (or `... is holding ...`) | the gripper sensor says the action is not possible now |
| `the goal cannot be reached from this state` | items never leave the basket, and the gripper holds one item at most |

---

## Reference

> **The cell changed on 2026-10-18** (slatted 1.26 x 0.85 m table, 0.70 x
> 0.59 m board, robot base turned 180 degrees, basket anywhere on the board,
> calibrated wrist camera looking along the gripper axis, wall as an
> obstacle; see [Run it on the real robot](#run-it-on-the-real-robot) and
> `config/cell.yaml`). The sections below were written for the earlier cell:
> their commands still apply, but numbers, photos and scene names
> (`hardcoded_problem`, `problem_006xx`, the video tasks) refer to it. Those
> scenes are kept in `problems*/old_cell_2026-10-13/`; their worlds no
> longer match `cell.yaml`.

The rest of this README is the reference. Most of it was written before the
consoles: it describes the skills in detail (section A), the all-in-one
`solve.py` (section T), and the simulated cell, its measurements and the code
(sections 1-6).

The robot code grew in three steps: **lab-replica** (the Gazebo replica of
the cell with a hard-coded solution, a problem generator and every
measurement), then **lab-skills** (single PDDL actions executed autonomously
with the wrist camera and SAM 3), then **lab-tamp** (goals planned with Fast
Downward). All of it is still here. ROS 2 Humble, Ignition Gazebo Fortress,
`ur_robot_driver`, PyTorch and Transformers (SAM 3) in a virtual environment,
the Unified Planning Framework with Fast Downward.

| I want to ... | Go to |
|---|---|
| plan and execute a goal in one command, in a new random scene, with a ground-truth check | [T.2](#t2-solve-a-goal) |
| know how the planning and execution fit together | [T.3](#t3-how-it-works) |
| execute single actions from my own program (ROS 2 action) | [A.2](#a2-execute-actions-in-simulation) |
| know how the skills work and what they assume | [A.3](#a3-how-it-works), [A.4](#a4-assumptions) |
| see the verification of the skills and the videos | [A.6](#a6-verification-and-videos) |
| run the hard-coded solution in simulation | [Run in simulation](#2-run-in-simulation) |
| fix a dimension so the simulation matches the lab | [Make the simulation match reality](#4-make-the-simulation-match-reality) |
| find the code of an object | [Where everything is](#5-where-everything-is) |

---

## T. Task and motion planning: solve a goal

### T.1 Install (once, on top of sections 1 and A.1)

Nothing more: [`run/setup.sh`](#build-once) installs the
[Unified Planning Framework](https://github.com/aiplan4eu/unified-planning)
(UPF) and its Fast Downward engine into the `.venv` (the engine's wheel
brings Fast Downward with it, nothing to build). `requirements.txt` pins
ConfigSpace (a UPF dependency) below 1.0: newer releases need a newer scipy
than Ubuntu 22.04's, which the `.venv` uses.

### T.2 Solve a goal

One command (with the workspace sourced and `source .venv/bin/activate`):

```console
ros2 run lab_replica solve.py "<goal>" [--scene <problem>] [--headless]
```

It writes the task to `problems_tamp/<name>/`, plans (UPF with Fast Downward),
starts the simulation (with the Gazebo window unless `--headless`) and the
skill server, executes the plan action by action, checks the goal against the
simulator's ground truth, writes `result.json` and shuts everything down
(`--keep-running` keeps the simulation up until Ctrl-C).

**The goal** is a PDDL goal formula over the domain's predicates
(`on-table`, `in-basket`, `robot-gripping`, `robot-gripper-empty`) and the
twelve items, or a `.pddl` problem file whose `(:goal ...)` is taken (its
`:objects` must be the twelve items; its `:init` is ignored):

```console
ros2 run lab_replica solve.py "(in-basket green-mug)"
ros2 run lab_replica solve.py "(in-basket green-mug) (robot-gripping red-can)"          # facts: a conjunction
ros2 run lab_replica solve.py "(and (in-basket roll-of-blue-tape) (or (in-basket red-can) (in-basket green-mug)))"
ros2 run lab_replica solve.py "(forall (?x - item) (not (on-table ?x)))"                # and, or, not, imply, forall, exists
ros2 run lab_replica solve.py reference/000001_problem.pddl
```

A misspelt item or predicate is reported with a suggestion before anything
starts.

**The initial state** comes from the scene:

| | Scene |
|---|---|
| (default) | a new random scene: a stable layout like `generate_problem.py` makes (settled and checked in Gazebo, every table item graspable), some items in the basket, maybe one in the gripper; sampled until the goal is reachable from it (about 30 s). Options: `--seed S`, `--basket-items K` (default random 0-3), `--p-holding P` (default 0.3), `--randomize-basket`, `--no-render` |
| `--scene <problem>` | the scene of an existing problem or task directory: a name in `problems/`, `problems_tamp/` or `problems_gen/`, or a path. Its `world.sdf` and its initial state |

A goal that cannot be reached from the scene's initial state is refused
(items never leave the basket in this domain, and the gripper holds one item
at most), and so is a goal that can never hold.

More options: `--plan-only` (write the task and the plan, execute nothing),
`--name NAME` (default `task_<date>_<time>`), `--step` (Enter before every
action), `--domain ID` (ROS_DOMAIN_ID for everything started here),
`--speed-scale S`, `--time-limit` (the planner's, default 60 s); see `--help`.

**Run a task again** (same scene, same goal; it is replanned):

```console
ros2 run lab_replica solve.py --task <name>
```

**Or step by step**, with the simulation and the skill server in their own
terminals (to watch them, or to keep the simulation between tasks; the
scene must be the task's):

```console
ros2 run lab_replica solve.py "<goal>" --plan-only --name mytask     # prints the plan
ros2 launch lab_replica sim.launch.py problem:=mytask
ros2 run lab_replica skill_server.py --sim                           # in the .venv
ros2 run lab_replica solve.py --task mytask --attach
```

A task directory (`problems_tamp/<name>/`):

| File | |
|---|---|
| `problem.pddl`, `domain.pddl` | the task as PDDL: exercise/task_planning/problem.pddl with the scene's initial state and the goal; the domain (always exercise/task_planning/domain.pddl) |
| `plan.txt`, `planner.log` | the plan and the planner's output |
| `world.sdf`, `layout.yaml`, `layout.png`, `render.png` | the scene: the Gazebo world, where every item stands (and the init, goal, plan), a top view, the lab camera's view |
| `README.md` | the task, the plan, the commands, where to put the items on the real table |
| `result.json` | the last execution: every action with the skill server's answer and duration, the true state afterwards and whether the goal holds |
| `logs/` | output of the simulation and the skill server started by `solve.py` |

A task directory is a scene too (`--scene <task>`), for another goal in the
same layout.

### T.3 How it works

```
 goal (text or .pddl) ──► check against domain + items ─┐
 scene (generated / existing) ──► initial state ────────┴──► UPF problem ──► Fast Downward ──► plan
                                                                                                 │
 for each action: /lab_skills/execute ──► skill server: look, recognise, plan motion, act ◄──────┘
                                         (section A; stops at the first failed action)
 afterwards (simulation): the true state from Gazebo ──► does the goal hold?
```

* **Task planning** (`exercise/task_planning/planning.py`, called by
  `lab_replica/tamp/task.py`): the problem is `exercise/task_planning/problem.pddl`
  with its `:init` replaced by the initial state and its `:goal` by the goal
  formula (`tamp/problem_file.py`; items added at run time join its objects),
  read together with `exercise/task_planning/domain.pddl` by UPF's
  `PDDLReader`, and solved with
  `OneshotPlanner(name="fast-downward")`; UPF's `SequentialSimulator` gives
  the state each action should produce. That is Fast Downward's LAMA
  (first iteration): satisficing, so a plan is not guaranteed to be the
  shortest, though in this small domain it usually is. UPF compiles
  disjunctions and quantifiers in the goal away; a plan takes well under a
  second.
* **Scenes** (`lab_replica/tamp/task.py`): a generated scene uses the problem
  generator's layout sampler (section 5) with one addition: an initial state
  is only accepted if the planner can reach the goal from it, checked
  before the expensive steps (graspability, settling in Gazebo).
* **Execution** (`lab_replica/tamp/execute.py`): the actions go one by one to
  the skill server of section A, which knows nothing of the plan or the
  layout: every action looks at the scene itself. `solve.py` tracks the PDDL
  state the plan should produce, compares the gripper contents the skill
  server reports with it, and stops at the first action that fails (a refused
  action leaves the scene unchanged; the task can be continued by hand with
  `do_action.py`).
* **An item held in the initial state** stands on the table in the simulated
  world (it cannot start in the gripper), so it is picked up first with a
  setup `(pick-up X)`, unless the gripper already holds something
  (`--no-setup` skips it).
* **Ground truth** (`lab_replica/tamp/sim_truth.py`, same rules as
  `tools/verify_actions.py`): after the plan, the item poses and the grip
  from Gazebo give the true PDDL state, which is checked against the goal.
  The robot never sees it.

### T.4 On the real robot

For goals on the real robot, the [goal console](#the-goal-console) is the
simpler way. `solve.py` can drive it too: set the items up as in the task's
README.md (generated scenes give every item's place), start the robot and
the camera ([Run it on the real robot](#run-it-on-the-real-robot), the
drivers on air and the camera on guest), then:

```console
ros2 run lab_replica skill_server.py --speed-scale 0.3       # in the .venv
ros2 run lab_replica solve.py --task <name> --attach --step
```

Without the simulator there is no ground truth: `solve.py` reports the
actions only. If the initial state has an item in the gripper, put it in the
gripper yourself and add `--no-setup`, or leave it on the table at its place
and the robot picks it up first.

### T.5 Videos

(`videos/` is git-ignored: the recordings are only in the lab PC's copy.)

[`videos/tamp/`](videos/tamp) shows tasks solved from start to end, each with a
`.json` of every action's result and the ground-truth check. The video shows
the lab camera's view; top left the goal and the plan (done / running /
failed); insets: the wrist camera live and the robot's last annotated view;
caption: the current action and what the robot is doing; at the end, whether
the goal holds in the simulator. A setup pick-up (an item in the gripper in
the initial state) is not recorded: the video starts with the item held.
Recorded with:

```console
ros2 run lab_replica solve.py "<goal>" --plan-only --name <task>
python3 src/lab_replica/tools/record_task.py <task> videos/tamp/<name>.mp4      # in the .venv; about 3 min per action
```

| Video | Task | Goal | Scene | Result |
|---|---|---|---|---|
| `01_two_items_into_basket` (101 s) | video_05 | `(and (in-basket red-can) (in-basket roll-of-black-tape) (robot-gripper-empty))` | generated, seed 901, empty basket | 4/4 actions, goal holds |
| `02_negation_and_disjunction` (116 s) | video_03 | `(and (not (on-table red-spray-bottle)) (or (in-basket green-block) (in-basket yellow-block)) (robot-gripper-empty))` | generated, seed 903 | 4/4, goal holds (blind A*: the `or` makes axioms); the 27 cm spray bottle is dropped into the basket |
| `03_cup_into_basket_ends_holding_can` (80 s) | video_04 | `(and (in-basket blue-cup-with-yellow-stripe) (in-basket green-mug) (robot-gripping red-can))` | generated, seed 904, the mug already in the basket | 3/3, goal holds |
| `04_drop_held_roll_then_hold_mug` (96 s) | video_06 | `(and (on-table blue-block) (in-basket orange-block) (robot-gripping green-mug))` | problem_00600, starts holding the black tape roll | 4/4 (first frees the gripper), goal holds |
| `05_green_tape_into_basket_then_hold_mug` (97 s) | video_02 | `(and (on-table blue-block) (in-basket roll-of-green-tape) (robot-gripping green-mug))` | problem_00600 | 4/4, goal holds |
| `06_stopped_held_roll_would_hit_mug` (68 s) | video_01 | as 01 | generated, seed 917, empty basket | stopped at action 3 (a refusal, nothing moved): the black tape roll (10.5 cm) lies next to the mug; the planner checks the held roll as a box, whose corners would touch the mug on the lift |

Recorded again on 2026-10-13 with the measured cell (the scenes are new: the
same goals and seeds give other layouts with the new sizes;
`videos/tamp/before_measurements/`: the earlier recordings). Two fixes came
out of it: the highest transit height is now 0.74 m (was 0.66; the 27 cm
spray bottle, held at its head, has to be carried at 0.69 m to clear the
basket rim), and the wall-clock timeout of a trajectory allows a simulation
running at 1/4 of real time (a 14 s motion at 0.3 x real time timed out).

### T.6 Known limitations

* No replanning: the plan is followed open loop at the task level, and the
  first failed action ends the run. The PDDL state afterwards is the one
  the actions *should* have produced; nothing perceives facts such as
  `in-basket` (an item bouncing out of the basket goes unnoticed until the
  ground-truth check, in simulation only).
* The domain is used unchanged, so its limits are the task's: items are
  never taken out of the basket, and `put-in-basket` always drops (section
  A.3), whatever is already in there.
* Motion feasibility is only known when an action runs (the skill server
  plans each motion from what it sees); a plan whose actions are all
  applicable in PDDL can still have an action refused (e.g. an item the
  camera does not recognise, section A.7).

## A. Autonomous PDDL actions

### A.1 Install (once, on top of section 1)

Nothing more: [`run/setup.sh`](#build-once) makes the Python environment
(`.venv`, with `--system-site-packages` so that it also sees the ROS
packages; [`requirements.txt`](requirements.txt)), takes the Hugging Face
token (`.env`, git-ignored) and downloads SAM 3. Use the `.venv` for
everything that perceives: `source .venv/bin/activate`.

### A.2 Execute actions in simulation

Three terminals (each: `source install/setup.bash`; the second one also
`source .venv/bin/activate`):

```console
ros2 launch lab_replica sim.launch.py problem:=problem_00600      # any problem, or none for the empty cell
ros2 run lab_replica skill_server.py --sim                         # loads SAM 3, then: ready: /lab_skills/execute
ros2 run lab_replica do_action.py "(pick-up green-mug)" "(put-on-table green-mug)"
```

`do_action.py` prints what the robot is doing and `OK: ...` or
`FAILED: <reason>`, and stops at the first failure. The same action can be
sent from any ROS 2 program (this is what a plan executor uses):

```console
ros2 action send_goal /lab_skills/execute lab_skills_msgs/action/ExecuteAction "{action: '(put-in-basket green-mug)'}"
```

| Action | Applicable when | What the robot does |
|---|---|---|
| `(pick-up X)` | the gripper is empty | survey (top view; also the front view if X is not recognised from above), find X, look at X from closer, grasp, lift, return home |
| `(put-in-basket X)` | the gripper holds something | survey, measure how high things are piled in the basket, drop X from just above the rim over its emptiest spot (it may land on other items), return home |
| `(put-on-table X)` | the gripper holds something | survey, put X on the free table spot nearest to where it was picked up (or `--at X Y`), return home |

"Empty" and "holds something" come from the gripper's grip-detected flag;
nothing else of the PDDL state is assumed. An action is refused (and nothing
moves) when it is not applicable, when X is not found, or when no
collision-free grasp or free table spot exists (put-in-basket is not refused
for lack of room: items are never taken out of the basket, so X is dropped
onto whatever is there). While running, the server publishes
`/lab_skills/status` (the current phase) and `/lab_skills/detections` (the
annotated camera view: for a pick-up what SAM 3 found; for a put the plan,
i.e. the basket's pile heights and the drop spot, or the free table spot and
what it keeps clear of), and saves every annotated view and the perceived
scene (YAML) under `runs/<date>/` (`--save-frames` also saves the raw RGB-D
frames).

### A.3 How it works

```
 survey (top view) ──► SAM 3: every item by name ──► masks + depth ──► item poses ┐
 front view (pick-up, if the target is not recognised from above) ────────────────┤
 close look at the target (pick-up only) ─────────────────────────────────────────┤
 depth not explained by any item ──► unknown obstacles ───────────────────────────┤
                                                                                  ▼
   grasp proposed in the close look's image ──► 3D grasp ──► MoveIt plans ──► arm + gripper ──► grip check
```

* **Camera.** The RealSense sits at the wrist, level with the joint between
  the robot and the gripper, its lens 110 mm off the gripper axis (measured),
  its view tilted 30 deg away from the fingers so they never appear in the
  image (the tilt is assumed, not measured) (cell.yaml `wrist_camera`; the camera is part of the
  robot's collision model). Its pixels go to the world through the arm's joint
  angles and this hand-eye transform; frames are only taken with the arm at
  rest.
* **Survey.** One view straight down from 0.70 m covers the whole placement
  region: the best geometry, and it sees into the basket, which is all a put
  needs. Some items are hard to name from straight above (the spray bottle is
  a grey head on a red blob), so before a pick-up whose target the top view
  did not recognise (score below 0.5) the camera also looks at the same
  region at an angle from the front, where SAM 3 recognises them; the
  geometry still comes from the top view. Before a pick the camera also looks
  straight down at the target from 0.50 m. Every look is `robot.survey`
  from a view of `exercise/state_estimation.py`, added to the scene of the
  earlier ones (see [More views](#more-views)); the skills decide which.
* **Recognition.** SAM 3 is asked for every item by name (one image encoding,
  13 prompts, about 0.7 s on the RTX 3070 Ti). A name can match several objects
  and an object several names (from some angles the spray bottle's body is "a
  red block"), so every mask is turned into 3D points, the item's pose is
  fitted with the item's known shape (perception/estimate.py), and the mask is
  scored by SAM's confidence times how well height and size match. Masks are
  then assigned to items best first, one each. A mask that covers only part of
  an item (the bottle without its grey head, the mug without its handle) is
  grown to the connected depth points around it.
* **Pose fitting** (items.yaml dimensions): the resting pose from the height;
  round items by a circle of the known radius through the top outline; blocks
  and a lying can by a rectangle; the mug's yaw from its handle (found in the
  depth around the body); the spray bottle by its trigger head (grasped) and
  its nozzle side from the body. In simulation: 0.1-2 mm and under 1 deg from
  the straight-down view.
* **Obstacles: what the camera saw.** Depth that no recognised item explains
  is an unknown obstacle (`autonomy/depth_world.py`): MoveIt avoids it, and
  nothing is put down on it (see
  [What the robot keeps clear of](#what-the-robot-keeps-clear-of)). A put on the table only uses spots that the
  camera's view covers up to the item's top, so that the item can be seen,
  and picked up, again.
* **Dropping into the basket.** The depth inside the basket becomes a height
  map (so items nobody recognised count too). The drop spot is the one with
  the lowest pile under the item's footprint, then the most room around it,
  with the footprint over the basket floor (not leaning on the wall). The
  item is released with its bottom 2 cm above the rim (or above the pile
  there, if higher) and falls; only the robot and the item on its way down
  are kept clear of the basket's contents. If the motion to one spot is
  blocked, the next best spots are tried. A drop can tip or bounce items,
  which is intended: the state after a put-in-basket is for perception to
  find out.
* **Grasp.** Proposed in the top-down image the item's pose comes from (the
  close look, else the top view): the item's pixels are its SAM 3 mask joined with the silhouette of
  its estimated model (SAM 3 may miss a part, e.g. the spray bottle's head).
  The fingers close on the top of the item (the pixels nearest to the
  camera): across the narrow side of an elongated top, across either pair of
  sides of a square one, and in any direction around a round one (first
  across the whole mask's narrow side, which keeps them off the mug's
  handle). Each proposal goes back into the world with the depth: the
  closing direction, the width between the pads, and the fingertips' height
  (as deep as the RG6's knuckles allow above the item's top, at least 8 mm
  above the table) (`exercise/state_estimation.py`).
* **Motion.** To the inverse kinematics solution nearest to the arm's
  configuration: the straight line in joint space when nothing is in its
  way (checked by MoveIt); else straight up first, then that line; else
  MoveIt's planner (OMPL's RRT-Connect); and
  MoveIt's Cartesian paths for the straight moves down and up
  (`exercise/motion_planning.py`); see
  [What the robot keeps clear of](#what-the-robot-keeps-clear-of). A held
  item moves rigidly with the gripper (it leans when the gripper leans for a
  camera view).

Code: `lab_replica/perception/` (camera, SAM 3, pose fitting, scene),
`lab_replica/autonomy/` (the executive, camera views, unknown obstacles),
`scripts/skill_server.py`, `scripts/do_action.py`, `src/lab_skills_msgs/`
(the action). Tools: `tools/verify_actions.py` (A.6), `tools/record_action.py`
(videos).

### A.4 Assumptions

Made so that the simulation and the real robot differ as little as possible:

* The camera is a RealSense publishing colour and depth **aligned to colour**
  under the realsense2_camera names (`/camera/camera/color/image_raw`,
  `/camera/camera/aligned_depth_to_color/image_raw`,
  `/camera/camera/color/camera_info`); depth as 16UC1 mm (the driver) or 32FC1 m
  (Gazebo). The simulated one is D435-like: 1280 x 720, 69 deg wide, depth from
  0.17 m. The views keep every item at least 0.28 m from the camera, inside a
  real D435's depth range.
* The camera is mounted as in cell.yaml `wrist_camera` (a hand-eye
  calibration measures exactly that pose).
* The twelve items are the known ones of items.yaml, resting in one of their
  stable poses, at least 2 cm apart (as the generator makes them). An item
  leaning on another (e.g. in the basket) is not given a pose: it is an
  unknown obstacle. The item models give the boxes MoveIt keeps clear of,
  the free spots on the table, and the item's pixels for a grasp (with its
  SAM 3 mask).
* Things stand on the table (or in the basket): the space above a measured
  surface, or above a height where a camera ray passed, is empty.
* Grasps are top-down, proposed in a top-down image.

### A.5 On the real robot

See [Run it on the real robot](#run-it-on-the-real-robot) for the drivers
and the consoles. For the skill server of A.2: the same commands without
`sim.launch.py` (the drivers run on air and guest) and `skill_server.py`
without `--sim` (start with `--speed-scale 0.3`). Before that:

1. **Mount the camera** on the RG6 and run the RealSense driver with aligned
   depth (`ros2 launch realsense2_camera rs_launch.py align_depth.enable:=true`
   on any machine of the ROS domain, as on guest in
   [2](#2-the-realsense-on-guest); check the topic names against
   config/perception.yaml).
2. **Hand-eye calibration**: put the measured pose of the colour optical frame
   in rg6_base_link into cell.yaml `wrist_camera.optical_xyz/optical_rpy` (the
   values there are the design of the simulated bracket). Check it with
   `--save-frames` and `tools/depth_map.py`: the table must come out at z = 0
   (white in the depth map) and the items where they are.
3. **Check the observation configuration** with the real lens
   (`run/workspace_view.sh real`, [The observation configuration](#the-observation-configuration)):
   the camera must see the whole board from there.
4. Measure the items (items.yaml) and the table height: the pose fitting uses
   the item dimensions, the planner the table.

### A.6 Verification and videos

`tools/verify_actions.py` runs actions in freshly generated problems and checks
each one against the simulator's ground truth, which the robot never sees: the
PDDL facts afterwards are exactly the action's effect, and no other item moved
more than 2 cm or tilted more than 10 deg (items already in the basket that a
drop pushes around are listed, not counted as a failure, unless knocked out).

Current version (drops into the basket, top-view survey, 16-sided round
items): 41 actions in 4 problems:

| action | executed | verified | refused (nothing moved) | failed |
|---|---|---|---|---|
| pick-up | 25 | 17 | 7 | 1 |
| put-in-basket | 10 | 10 | 0 | 0 |
| put-on-table | 6 | 6 | 0 | 0 |

Mean time per action 33 s (was 64 s). Seven refusals are one spray bottle that
SAM 3 does not recognise from either view in two problems (nozzle towards the
camera), one a block against the basket wall; the "failure" is a refused pick
during which a pile in the basket (a spray bottle dropped onto a 12 cm pile,
toppled and leaning over the rim) shifted. Details:
[verification/README.md](verification/README.md).

The numbers below are for the previous version (careful placement in the basket).

78 actions in 20 generated problems (6 for the main run, 3 with an empty
basket for the tall items, 11 for the videos), every one checked:

| action | executed | verified | refused (nothing moved) | failed | items verified |
|---|---|---|---|---|---|
| pick-up | 40 | 37 | 1 | 2 | 12 of 12 |
| put-in-basket | 25 | 21 | 4 | 0 | 12 of 12 |
| put-on-table | 13 | 13 | 0 | 0 | 10 of 12 |

The 5 refusals are correct: no room in the basket for a tall item (4), and
an item that had fallen out of view (1). Both failures come from one bug in
the *simulated* gripper driver, since fixed (details and every single action:
[verification/README.md](verification/README.md)). Mean time per action:
64 s: roughly 25 s of survey motions, 3-5 s of recognition, 8 s for the
close look, the rest the pick or placement.

(`videos/` is git-ignored: the recordings are only in the lab PC's copy.)

The ten videos in [`videos/`](videos) each show one action in a newly generated
scene (the lab camera's view; insets: the wrist camera live and the robot's
last annotated view of this action: what SAM 3 found for a pick-up, the plan
for a put; caption: the action and the robot's current phase). Each has a
`.json` with the ground-truth check of that run. They were recorded again on
2026-10-13 with the measured cell (scenes `problems_gen/problem_00706`-`00716`,
made with `solve.py "(robot-gripping <item>)" --plan-only --seed <n>
--basket-items 3 --p-holding 0`): all ten verified, 18-34 s each.
`videos/before_measurements/`: the same actions with the photo-estimated
geometry; `videos/previous/`: the version before that. Recorded with
`tools/record_action.py <problem> "<action>" out.mp4 [--setup "<action>"]`.

### A.7 Known limitations

* Items dropped into the basket land where physics puts them: they may tip
  over, rest on others or (rarely) bounce out. Nothing checks this after the
  drop.
* SAM 3 recognises some items only from some angles (the spray bottle from
  the front, not from above, and in some poses from neither). The top view plus the front view on demand
  cover the simulated items; the real items look different and the prompts
  (config/perception.yaml) may need adjusting.
* Not tested on the real robot. With real images only one lab photo was tried
  (reference/000047_image.png, not in git; an oblique view without depth): SAM 3's best
  mask was right for most items, but it confused the tape rolls with each other
  and took the blue foam board for the blue block, which is what the depth-based
  size check and the one-mask-per-item assignment are there to resolve.
* An action takes about 33 s in simulation (pick-up about 40 s).
* A tall item dropped onto a pile topples and may end up leaning over the
  rim; what the camera sees of it (columns from the table up) can then block
  the arm for later picks near the basket.
* **The observation view only just covers the table.** From its height of 0.70 m
  it sees the table from y = -0.25 to 0.29 m and up to x = 0.86 m, and less
  higher up; the placement region is y = -0.24 to 0.27 m. Space outside
  every view counts as occupied for the motions, so an item at the edge of
  the view (the front row, or far right) can be harder to grasp, and items
  are only put down where the view covers them up to their top. An item the
  robot cannot see whole may not be recognised. A second survey view, or a
  camera with a wider lens, would help.
* Space hidden behind a tall item (from the survey view) counts as occupied
  up to where the camera's rays passed over it. The close look straight
  down at the item to pick usually sees into the gaps; a pick that still
  needs the fingers in a hidden gap is refused.
* Transit motions only vary their height. In a tight layout (a tall item in
  the basket in the way and another item close to the robot base) no height
  works and the pick is refused, although a sideways detour would.
* Holding the 27 cm spray bottle, the arm cannot go to its home pose (the
  bottle would dip into the basket); it stays where it lifted the bottle,
  and the next action starts from there.
* A held round item is checked as a box: its corners can make a pick next
  to a tall neighbour fail (the 10.5 cm black tape roll beside the mug,
  tamp video 06). The item in the gripper is the one thing still checked
  with its model (the camera cannot see it there).
* put-on-table was verified for 10 of the 12 items (not yet the red block and
  the black tape); put-in-basket and pick-up for all 12.


---

## 1. Install and build (once)

```console
cd physAI-arm-ex                     # the clone
run/setup.sh                         # see Build (once) at the top
source install/setup.bash            # in each terminal for the commands below
```

Run `run/setup.sh` again when you add or rename files. Edits to the YAML
files and the Python code take effect without rebuilding (`--symlink-install`).

Every terminal must use the same `ROS_DOMAIN_ID` (the lab PC sets 131 in
`~/.bashrc`).

## 2. Run in simulation

> **Out of date (2026-10-13):** `hardcoded_problem` does not fit the measured
> 35 cm basket (see [How the replica was built](#how-the-replica-was-built));
> measure where the items stand, put that into its `layout.yaml` and run
> `replan_problem.py hardcoded_problem` before using it. Meanwhile use
> `problem_00600` (or any problem in `problems/`) in the commands below.

Terminal 1, the simulated cell with the problem's items in place:

```console
ros2 launch lab_replica sim.launch.py problem:=hardcoded_problem
```

Terminal 2, once the first terminal prints `RG6 driver ready (sim)`:

```console
ros2 run lab_replica run_solution.py hardcoded_problem
ros2 run lab_replica check_goal.py hardcoded_problem     # optional: is the goal satisfied?
```

The solution takes about 2.5 minutes and prints each PDDL action as it runs
it, and `holding, width 58.0 mm` after every grasp. Options of
`sim.launch.py`: `gui:=false` (no Gazebo window), `rviz:=true`. Without a
`problem:=` it shows the empty cell.

## 3. Run on the real robot

**Once, on the robot:** the teach pendant needs a program containing the
*External Control* node whose *Host IP* is this PC's address, and the RG6
must work from the OnRobot URCap (it is cabled to the tool flange). Set the
payload (RG6 + Quick Changer, 1.31 kg) in the installation. Use a wired
network: the driver stops the robot when the link stalls (it did over Wi-Fi).

**Each time:**

1. Robot address: `ip:` in [`src/lab_replica/config/real_robot.yaml`](src/lab_replica/config/real_robot.yaml)
   (currently 10.240.10.2). Or add `robot_ip:=<address>` to the next command.
2. Place the items as listed in [`problems/hardcoded_problem/README.md`](problems/hardcoded_problem/README.md).
3. Start the drivers (UR arm and RG6):
   ```console
   ros2 launch lab_replica real.launch.py
   ```
4. On the teach pendant, load the External Control program and press **Play**.
   The terminal prints `Robot connected to reverse interface. Ready to receive control commands.`
5. Run the solution, slowly and one action at a time:
   ```console
   ros2 run lab_replica run_solution.py hardcoded_problem --speed-scale 0.3 --step
   ```
   If the arm is not at the solution's start pose, it first offers a slow
   joint move there (it refuses if the arm would come too close to the
   table). Then it waits for Enter before each action.

Keep a hand on the emergency stop. Ctrl-C in the terminal stops the arm; the
pendant's speed slider scales every motion on top of `--speed-scale`. More
options: `--step-motions` (Enter before every single motion),
`--from-action N`, `--skip-setup`, `--dry-run`; see `--help`.

*Without the robot*, the same commands can be tried against a mock arm and a
fake gripper:

```console
python3 src/lab_replica/tools/fake_onrobot_xmlrpc.py --port 41499 &
ros2 launch lab_replica real.launch.py use_fake_hardware:=true robot_ip:=127.0.0.1 gripper_port:=41499
```

## 4. Make the simulation match reality

Part of the geometry is **measured** (2026-10-13: the table's length and the
robot's place on it, the white board, the basket's diameters, eleven of the
twelve items, the wrist camera's offset); the rest is still **estimated from
photos** (see [How the replica was built](#how-the-replica-was-built)):
positions may be a few centimetres off, sizes 10-20 %. The solution is open
loop, so the measurements matter.

1. Measure, and write the value into the file below. All lengths in metres.
2. Rebuild the simulation and replan the solution:
   ```console
   ros2 run lab_replica replan_problem.py hardcoded_problem
   ```
   It rebuilds the Gazebo models, settles the items in Gazebo (and warns if
   one tips over or overlaps another), plans every motion again with
   collision checks, and only if that succeeds rewrites the problem's
   `world.sdf`, `plan.json`, `solution.py`, `layout.png` and `README.md`
   (about 15-30 s). If no collision-free plan exists, it says which item and
   obstacle are in the way and changes nothing.
3. Check the new solution in simulation ([section 2](#2-run-in-simulation)),
   then on the robot.

**Coordinates.** Measure positions in the world frame: origin on the table
top directly below the robot's base axis, **x** to the right as seen from the
camera (along the table), **y** away from the camera towards the wall, **z**
up. A yaw is a rotation about the vertical, counter-clockwise from +x seen
from above.

| What you measured | Where it goes |
|---|---|
| Height of the robot's mounting face above the table top (the fingertips go down to 8 mm above the table) | `robot.base_xyz[2]` in [`config/cell.yaml`](src/lab_replica/config/cell.yaml) |
| Rotation of the robot base on the table | `robot.base_yaw` in `cell.yaml` |
| Table edges, board boundary, table height | `table` section of `cell.yaml` |
| Basket position and size | `basket` section of `cell.yaml` |
| Camera position, angle and field of view | `camera` section of `cell.yaml` |
| Wall, sockets, window (seen by the camera only) | `room` section of `cell.yaml` |
| Quick Changer thickness, gripper rotation on the flange, grip force | `gripper` section of `cell.yaml` |
| Size, mass or colour of an item | its entry in [`config/items.yaml`](src/lab_replica/config/items.yaml) |
| Where an item stands for this task | its `xyz` (x and y; z is computed) and `yaw_deg` in [`problems/hardcoded_problem/layout.yaml`](problems/hardcoded_problem/layout.yaml) |
| Robot IP and driver settings | [`config/real_robot.yaml`](src/lab_replica/config/real_robot.yaml) (no replanning needed) |

(`config/` is `src/lab_replica/config/`.) What is *not* in the YAML files:
the RG6 itself (the manufacturer's CAD geometry, in
`description/urdf/rg6.xacro` and `lab_replica/robot/rg6.py`) and the UR5e's
link lengths (the nominal ones from `ur_description`).

**The shape of an item**, beyond the numbers in `items.yaml` (for example how
the mug's handle or the spray bottle's head is modelled), is in the item's
own file, [`lab_replica/objects/<kind>.py`](src/lab_replica/lab_replica/objects):
its collision shapes, how it can rest on the table, where it is grasped, and
its visual mesh. To look at changed models without replanning, run
`ros2 run lab_replica build_models.py` and `ros2 launch lab_replica sim.launch.py`.

The hard-coded layout is tight: moving an item by a few centimetres can make
a grasp impossible (the fingers need room on both sides). `replan_problem.py`
then reports the obstacle; move the item in `layout.yaml` (and on the table)
and run it again.

## 5. Where everything is

```
physAI-arm-ex/
  run/                     THE SCRIPTS: setup.sh (once), sim.sh (the simulated robot), actions.sh, goals.sh,
                           workspace_view.sh (find the observation configuration); env.sh: shared set-up
  docs/                    pictures for this README
  problems/
    all_on_table/          the default simulated scene of run/sim.sh: all twelve items on the table
    hardcoded_problem/     the task with the hard-coded solution (README.md: task, commands, table set-up)
    problem_006xx/         three more generated problems
  problems_gen/            generated problems (the old cell's, of the verification and the videos, are not in git)
  problems_tamp/           the tasks of solve.py: scene + goal + plan + result (section T.2)
  verification/            the verification reports (ground-truth check of every action)
  videos/                  ten videos of autonomous actions, each with its ground-truth check (.json) (not in git)
  runs/                    what the robot saw, one directory per console or skill server run: log,
                           depth maps, annotated views, perceived scenes, frames (not in git)
  reference/               the lab's original PDDL files (its photos, *_image.png, are not in git)
  .venv/, .env             the Python environment for SAM 3 and the Hugging Face token (not in git)
  src/lab_skills_msgs/     the ExecuteAction action
  src/lab_replica/         the ROS 2 package
    config/
      cell.yaml            MEASUREMENTS of the cell: table, robot mount, gripper, wrist camera, basket, camera, wall
      items.yaml           MEASUREMENTS of the twelve items
      perception.yaml      wrist camera topics, SAM 3 prompts and thresholds
      real_robot.yaml      robot IP and driver settings
      controllers.yaml     ros2_control controllers of the simulation
      domain.pddl          the lab's PDDL domain (the planner uses exercise/task_planning/domain.pddl)
      moveit/              MoveIt: lab_robot.srdf (planning group, collision pairs), kinematics, OMPL, limits
    lab_replica/           the Python library
      exercise/            THE EXERCISE (no ROS): task_planning/ (domain.pddl, problem.pddl, planning.py),
                           state_estimation.py, motion_planning.py, skills.py, misc.py (bonus)
      toolkit.py           what the exercise may use: data types, fits, kinematics, plan steps
      moveit_client.py     MoveIt (move_group's services) without ROS in its interface
      objects/             one file per physical object: can.py, spray_bottle.py, tape_roll.py,
                           cup.py, mug.py, block.py, basket.py, table.py, room.py, camera.py
      robot/               UR kinematics, RG6 linkage, robot collision points, the RG6 ROS driver
      planning/            PDDL planner, pick-and-place planner (pick_place.py: generator, tools), motion planning
      problems/            problem generator, replanning, writing a problem's files
      execution/           running a solution through ROS (runner, executor, goal check)
      perception/          wrist RealSense frames, SAM 3, item pose fitting, scene (section A.3)
      autonomy/            executing PDDL actions on demand: executive.py (the robot the skills use), camera views,
                           depth_world.py (the depth seen: unknown obstacles, drop spots), annotate.py
      console.py           the action and goal consoles (prompt, start-up checks, running an action)
      tamp/                solving a goal: goal formulas, tasks, plan execution (section T.3; the planning is in exercise/)
      gazebo/              writing models and worlds, headless Gazebo runs
    launch/
      sim.launch.py        the simulated cell
      real.launch.py       the real cell (ur_robot_driver + RG6 driver)
      moveit.launch.py     MoveIt's move_group (started by the consoles)
    scripts/               the `ros2 run lab_replica ...` commands (below)
    tools/                 development and calibration tools (below)
    description/urdf/      robot description: UR + Quick Changer + RG6 + wrist RealSense
    models/, worlds/       GENERATED Gazebo models and the empty world: do not edit, they are
                           rebuilt from the YAML files by replan_problem.py / build_models.py
```

**Commands** (`ros2 run lab_replica <command> --help` for details; a
`<problem>` is a name in `problems/` or a path):

| Command | What it does |
|---|---|
| `action_console.py [--sim]` | the action console (normally started by `run/actions.sh`) |
| `goal_console.py [--sim]` | the goal console (normally started by `run/goals.sh`) |
| `workspace_view.py [--sim]` | the workspace view: camera + SAM 3 masks, the arm's joints (normally started by `run/workspace_view.sh`) |
| `skill_server.py [--sim]` | the skills as a ROS 2 action, `/lab_skills/execute` (section A.2) |
| `do_action.py "<action>" ...` | sends actions to the skill server |
| `run_solution.py <problem>` | runs the problem's hard-coded `solution.py` (simulation or real robot) |
| `check_goal.py <problem>` | reads the simulated scene back as PDDL facts and checks the goal |
| `replan_problem.py <problem>` | after a measurement change: rebuilds models, world, plan and `solution.py` |
| `solve.py "<goal>"` | plans the goal (UPF + Fast Downward) and executes the plan autonomously (section T) |
| `generate_problem.py [--seed S]` | a new random problem in `problems/` |
| `build_models.py` | rebuilds the Gazebo models from the YAML files |
| `capture_camera.py out.png` | saves the simulated camera's view |

**Tools** (`python3 src/lab_replica/tools/<tool> ...`):

| Tool | What it does |
|---|---|
| `probe_cell.py <robot ip>` | read-only check of the real cell: robot model and software version, how the RG6 is connected |
| `fake_onrobot_xmlrpc.py` | stand-in for the RG6 on the robot, for testing `real.launch.py` offline |
| `render_preview.py out.png --photo <photo>` | renders the replica from the camera to compare with a lab photo |
| `depth_map.py runs/<run>/frame_*.npz [--pick X]` | replays saved camera frames: what SAM 3 finds, the depth map, why a grasp is refused |
| `record_task.py <task> out.mp4` | records a task being solved (T.5): lab camera at 1080p, goal and plan progress, insets, ground-truth result |
| `record_action.py <problem> "<action>" out.mp4` | records one autonomous action (A.6) |
| `record_video.py out.mp4` | records the simulated camera to a video (on simulation time) |
| `record_camera.py out.gif` | records the simulated camera to a GIF |
| `grasp_test.py`, `single_grasp.py` | physics tests of the grasps of every item |

## 6. How it works

**A problem directory** (`problems/<name>/`) holds the task (`problem.pddl`),
where every item stands (`layout.yaml`, `layout.png`), the Gazebo world with
the items at their settled poses (`world.sdf`), the motion plan (`plan.json`)
and the same plan written out as the hard-coded `solution.py`: joint
waypoints with time stamps for every arm motion, a width and a force for
every gripper command. `run_solution.py` sends these to the arm's
`FollowJointTrajectory` action and the gripper's `GripperCommand` action;
both exist with the same names in simulation and on the robot, and the
runner asks the controller manager which trajectory controller is active.

**Planning** (`lab_replica/planning/`): an optimal PDDL plan by breadth-first
search, then for every action a top-down grasp or placement: transit around
the base at a safe height, open to the item width + 32 mm, descend (slowly
for the last 6 cm), close with the force limit, lift; into the basket the
item is lowered until something would touch and released (it may drop a few
cm); on the table it is lowered to 4 mm above the surface. Every waypoint is
checked for collisions of the whole robot and the held item with the table,
the basket and the other items. Joint speed at most 0.8 rad/s, transit
0.2 m/s, approach 0.05 m/s.

**Simulation specifics.** Grasping is by friction only (no attach plugins).
Round items collide as 16-sided prisms: DART gives a flat pad pressed on a
cylinder almost no torsional friction, so a held cylinder would swing. The
real RG6 moves its fingertips on a parallelogram linkage (the pads stay
parallel and reach 5.5 cm deeper when closed); Gazebo Fortress has no mimic
joints, so each simulated fingertip rides on a force-limited slide and a
lift that the driver keeps at the linkage's depth, and the knuckles follow.
The knuckles collide too: tall items are held near their top with at least
4 mm below the knuckles. The camera is rendered on the GPU (EGL); the GUI
and RViz run under VirtualGL when `/opt/VirtualGL/bin/vglrun` exists.

**Real gripper.** The RG6 is cabled to the UR's tool flange and driven by the
OnRobot URCap; its daemon on the robot controller serves XML-RPC on port
41414 (`rg_grip`, `rg_get_width`, ...). `rg6_driver.py` talks to it, so
nothing has to be installed on the robot.

### How the replica was built

Nothing in the lab photos had a known size except standard products. The
rims of the basket, the mug and the tape rolls are circles; their ellipses
give the camera's focal length (about 700 px at 1280 x 720, an 85 deg wide
angle) and pitch (30 deg down). The 330 ml sleek Coca-Cola can (58 x 146 mm)
and a 48 mm duct-tape roll set the scale: the camera is then 0.39 m above
the table and 0.58 m in front of the robot base, and the table, basket, wall
and item sizes follow from their pixels. Items near the far edge measure up
to 1.5x larger than near the front (barrel distortion of the real lens), so
sizes were taken from the middle of the table. `tools/render_preview.py`
renders the replica from the same camera for side-by-side comparison with
photo 000047. The robot model (UR5e, PolyScope 5.12) was read from its
controller with `tools/probe_cell.py`.

On 2026-10-13 these were replaced by measurements from the lab: the table
(1.85 m long, the robot's base axis 0.20 m from its left edge, the white
board up to 0.93 m right of the base axis), the basket (0.25 m across the
floor, 0.35 m at the rim), the can, the black and blue tape rolls, the spray
bottle's height and width, the cup, the mug's body and the blocks (four 50 mm
cubes, the orange one 40 mm), and the wrist camera (lens 0.11 m off the
gripper axis, level with the gripper's mounting face). Still estimated: the
heights of the robot base and the camera stand, the table's depth, the
basket's position and height, the green tape roll, the spray bottle's
proportions and the mug's handle, the wrist camera's tilt. The larger basket
had two consequences. Its centre was moved from x = 0.35 to 0.39 (at 0.35 the
UR's upper arm, 13 cm above the table, came within the planner's 1.5 cm
clearance of the rim). And the photo-based layout of `hardcoded_problem` no
longer fits: the spray bottle and the green tape roll would stand 5 cm inside
the basket and there is no free place for them on the table, so that problem
was **not** replanned (its `world.sdf` and solution are from the old
measurements) until its item positions are measured in the lab. The generated
problems (problem_00600-00602, the video scenes) were generated again.

### Known limitations

* Some measurements are still estimates (above); the simulated camera has no lens distortion.
* Items are rigid; masses are guesses.
* Released items drop a short distance into the basket and may tip over in there.
* In about 3 of 7 earlier simulated runs of the hard-coded solution, every
  action succeeded but the final goal check failed; the cause was not
  investigated.
* The real-robot path (passthrough controller, RG6 width/force units of the
  URCap daemon) has been tested offline only (mock arm, fake gripper).
