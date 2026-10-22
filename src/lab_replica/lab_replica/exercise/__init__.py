"""The exercise: task planning, state estimation, motion planning and the
skills for the UR5e + RG6 + wrist RealSense cell. Nothing in here touches ROS.

    task_planning/domain.pddl   the PDDL domain (the actions' preconditions and effects)
    task_planning/problem.pddl  the PDDL problem (the objects and a goal; the goal console puts in
                                the current state and the goal typed)
    task_planning/planning.py   UPF + Fast Downward, the expected states, executing a plan
    state_estimation.py         where the camera looks from (the observation configuration: the
                                arm's joints from which the camera sees the whole workspace; a
                                side view; closer views), pixels -> world points, where
                                each item rests, which mask is which item, a grasp proposed in the
                                image and its 3D pose
    motion_planning.py          inverse kinematics choice, the held item's box, the motions of the
                                camera views, to the observation configuration and of pick-up /
                                put-on-table / put-in-basket with MoveIt, where to put an item
                                down, the obstacles MoveIt avoids
    skills.py                   the PDDL actions performed by the robot: (pick-up X) by
                                pick_up(robot, X), put_on_table, put_in_basket
    scene_model.py              a fixed barrier scanned from several views and kept clear of (the
                                scan: command of the consoles)
    misc.py                     bonus: new-item: / forget-item: (an item measured by the camera)

What these files may use from the rest of the code is in lab_replica/toolkit.py
(data types, geometric helpers, the robot's kinematics, the steps of a plan,
pictures for the camera window) and lab_replica/moveit_client.py (MoveIt,
without ROS in its interface); skills.py and misc.py also get the robot
(autonomy/executive.py: looking, the gripper, running steps), whose methods
they list at their top.

Who calls what:
    scripts/goal_console.py, tamp/task.py     -> task_planning/ (tamp/problem_file.py fills problem.pddl)
    perception/rgbd.py, perception/scene.py   -> state_estimation.py (points, placements, masks)
    autonomy/executive.py                     -> skills.py, misc.py (with itself as the robot)
    skills.py, misc.py                        -> state_estimation.py (grasps), motion_planning.py
    skills.py, misc.py                        -> robot.survey(view) with the views of
                                                 state_estimation.py (OBSERVATION_JOINTS, ...)
    autonomy/executive.py (robot.survey)      -> motion_planning.py (plan_observation, plan_view)
"""
