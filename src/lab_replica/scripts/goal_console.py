#!/usr/bin/env python3
"""Type a goal; the robot plans the actions (UPF + Fast Downward) and performs them.

    run/goals.sh sim|real                        (the usual way: sets up the environment)
    ros2 run lab_replica goal_console.py [--sim] [--speed-scale S] [--step]

Needs the arm, the gripper and the wrist camera (run/sim.sh, or on the real
robot their drivers: README, "Run it on the real robot"). Loads SAM 3 and,
once the arm, the gripper and the camera answer, shows the state, how to write
a goal, and a prompt.

The items are the objects of the exercise's problem file
(src/lab_replica/lab_replica/exercise/task_planning/problem.pddl); each goal
is planned with that file, its :init replaced by the current state and its
:goal by the goal typed. The state starts as: every item on the table, the
gripper empty. After each
goal it is the state the performed actions should have produced (nothing
perceives facts such as in-basket); `reset` sets it back to the start. A
goal is planned from that state, and the plan's actions are performed one by
one as in the action console, stopping at the first that fails.

New items, as in the action console (README "Add a new item"), for this
session only:

    new-item: purple-block           the robot finds "purple block" on the table,
                                     measures it and adds it to the items: on the
                                     table in the state, an object of the planning
                                     problem, usable in goals
    forget-item: purple-block        remove it again

A window shows the wrist camera live (colour and depth side by side) and
every view the robot takes, annotated (viewer.py), as in the action
console; --no-window leaves it out.

Ctrl-C during an action stops the arm; Ctrl-C or Ctrl-D at the prompt quits.
"""

import argparse
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))

from lab_replica.console import (FORGET_BARRIER, FORGET_ITEM, NEW_ITEM, PREDICATES, SCAN, InvalidAction,  # noqa: E402
                                 Robot, Terminal, banner, barrier_command, barrier_hints, common_arguments,
                                 item_command, item_hints, parse_barrier_command, parse_item_command,
                                 unknown_item_hint, warn_holding_added, wrap_words)
from lab_replica import objects  # noqa: E402
from lab_replica.planning import pddl  # noqa: E402
from lab_replica.tamp import planner, task  # noqa: E402
from lab_replica.exercise.task_planning import planning as tp  # noqa: E402
from lab_replica.tamp.goal import CONNECTIVES, GoalError, holds, parse_goal, to_text  # noqa: E402

ITEMS = []      # the objects of the exercise's problem.pddl (read in main)


def all_items(robot):
    """The objects of problem.pddl and the items added in this session (new-item:)."""
    return list(ITEMS) + list(robot.added)


def describe(state, items):
    table = [i for i in items if ("on-table", i) in state]
    basket = [i for i in items if ("in-basket", i) in state]
    held = pddl.held(state)
    return table, basket, held


def show_state(term, state, items):
    table, basket, held = describe(state, items)
    term.say(term.bold("State:   ") + "on the table: " + (wrap_words(table, 23) if table else "nothing"))
    term.say("         in the basket: " + (", ".join(basket) if basket else "nothing"))
    term.say("         gripper: " + (f"holding {held}" if held else "empty"))


def hints(term, state, added):
    term.say()
    show_state(term, state, list(ITEMS) + list(added))
    term.say(term.bold("Goal:    ") + "facts that must all hold at the end, for example")
    term.say("           (in-basket green-mug)")
    term.say("           (in-basket red-can) (robot-gripping green-mug)")
    term.say("           (and (in-basket red-block) (or (in-basket blue-block) (on-table green-block)))")
    facts = [f"({p})" if p == "robot-gripper-empty" else f"({p} <item>)" for p in PREDICATES]
    term.say(term.bold("Facts:   ") + "  ".join(facts))
    term.say("         combined with " + ", ".join(CONNECTIVES) + "  (e.g. (not (on-table red-can)))")
    term.say(term.bold("Items:   ") + wrap_words(list(ITEMS) + [i + "*" for i in added], 9))
    item_hints(term, added)
    barrier_hints(term)
    term.say(term.dim("Commands: state (show the state), reset (all items on the table, gripper empty), quit. "
                      "Tab completes names."))


def actions(n):
    return f"{n} action" + ("" if n == 1 else "s")


def plan_for(term, state, formula, items):
    """The planner's plan over `items`, or None (said why)."""
    if holds(formula, state, items=items):
        term.say("the goal already holds: nothing to do")
        return None
    term.say("planning ...")
    try:
        return task.solve(state, formula, log=lambda s: None, items=items)
    except planner.Unsolvable:
        parts = formula[1:] if formula[0] == "and" else [formula]
        bad = [to_text(x) for x in parts if not task.reachable(state, x, items=items)]
        term.say(term.colour("the goal cannot be reached from this state", False)
                 + (f": {' '.join(bad)} can never hold" if bad else "")
                 + " (items in the basket never leave it; the gripper holds one item at most)")
    except planner.PlannerError as e:
        term.say(term.colour("planner error", False) + f": {e}")
    return None


def main():
    sys.stdout.reconfigure(line_buffering=True)
    ap = common_arguments(argparse.ArgumentParser(description=__doc__,
                                                  formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--step", action="store_true", help="wait for Enter before every action of a plan")
    ap.add_argument("--no-window", action="store_true",
                    help="do not open the camera window (live colour and depth, annotated observations)")
    args, _ = ap.parse_known_args()
    try:
        planner.check()
        ITEMS[:] = task.default_items()       # the objects of the exercise's problem.pddl
    except planner.PlannerError as e:
        return f"{e}"
    words = list(PREDICATES) + list(CONNECTIVES) + list(ITEMS) + ["state", "reset", "quit", NEW_ITEM, FORGET_ITEM, SCAN, FORGET_BARRIER]
    term = Terminal(words, "goals")
    banner(term, "goal console", args.sim)
    unknown = [i for i in ITEMS if i not in objects.load_items()]
    if unknown:
        term.say(term.colour("warning", False) + f": {', '.join(unknown)} in problem.pddl "
                 "is not an item the robot knows (config/items.yaml): the camera cannot find it")
    robot = Robot(term, sim=args.sim, speed_scale=args.speed_scale, verbose=args.verbose,
                  return_home=not args.no_home, save_frames=args.save_frames, window=not args.no_window,
                  barrier_file=args.barrier_file)
    state = task.most_permissive_init(ITEMS)      # every item on the table, gripper empty
    if robot.holding:
        term.say(term.colour("warning", False) + ": the gripper holds something, but the goal console starts "
                 "from every item on the table and the gripper empty. Take the item out of the gripper "
                 "(or run (put-on-table <item>) in the action console), then restart.")
    else:
        robot.ex.set_held(None)
    term.say(f"Ready. Log and camera views: {robot.run_dir}")
    warned = False
    try:
        while True:
            hints(term, state, robot.added)
            with robot.idle():
                line = term.ask("goal> ")
            if line is None or line.strip().lower() in ("quit", "exit", "q"):
                if not warned and warn_holding_added(term, robot):
                    warned = True
                    continue
                break
            cmd = line.strip().lower()
            if not cmd:
                continue
            warned = False
            if cmd == "state":
                continue            # the hints show it
            if cmd == "reset":
                state = task.most_permissive_init(all_items(robot))
                term.say("state reset: every item on the table, the gripper empty")
                if robot.grip():
                    term.say(term.colour("warning", False) + ": the gripper holds something")
                continue
            barrier_cmd = parse_barrier_command(line)
            if barrier_cmd is not None:
                barrier_command(term, robot, barrier_cmd)
                continue
            try:
                item_cmd = parse_item_command(line, robot.added)
            except InvalidAction as e:
                term.say(term.colour("invalid item command", False) + f": {e}")
                continue
            if item_cmd is not None:
                name = item_cmd[1]
                if item_command(term, robot, *item_cmd):
                    # An added item stands on the table (it was just measured
                    # there); a forgotten one leaves the state.
                    state = frozenset(f for f in state if name not in f[1:])
                    if item_cmd[0] == "new-item":
                        state |= {("on-table", name)}
                continue
            items = all_items(robot)
            try:
                formula, _ = parse_goal(line, items)
            except (GoalError, OSError) as e:
                # The hints list the items and predicates already.
                why = str(e).split("; the items are")[0].split("; the predicates are")[0]
                term.say(term.colour("invalid goal", False) + f": {why}")
                m = re.search(r"unknown item '([a-z-]+)'", why)
                if m:
                    unknown_item_hint(term, m.group(1))
                continue
            term.say(f"goal: {to_text(formula)}")
            robot.window_status(f"planning {to_text(formula)}")
            with robot.idle():
                plan = plan_for(term, state, formula, items)
            if not plan:
                robot.window_status("no plan (see the console)")
                continue
            term.say(f"plan ({actions(len(plan))}):")
            for k, a in enumerate(plan):
                term.say(f"  {k + 1}. {a}")
            n = len(plan)
            failed = []

            def before(k, action):
                if args.step:
                    robot.window_status(f"waiting for Enter: action {k + 1}/{n} {action}")
                    with robot.idle():
                        answer = term.ask(f"Enter performs action {k + 1}/{n} {action} (n + Enter stops) ")
                    if answer is None or answer.strip().lower().startswith("n"):
                        term.say("stopped before the action")
                        robot.window_status(f"stopped before action {k + 1}/{n} {action}")
                        return False
                term.say(f"performing action {k + 1}/{n}: {action}")
                return True

            def after(k, action, ok, msg):
                if ok:
                    term.say(term.colour("action success", True) + f": {msg}")
                else:
                    failed.append(k)
                    term.say(term.colour("action failed", False) + f": {msg}")
                    robot.window_status(f"goal not reached: action {k + 1}/{n} {action} failed")

            # The exercise's plan execution (exercise/task_planning/planning.py).
            state, failed_at = tp.execute_plan(state, list(plan), plan.states, robot.execute, before, after)
            if failed_at is None:
                robot.window_status(f"goal reached ({actions(n)})")
                term.say(term.colour("goal reached", True)
                         + f" ({actions(n)}; the state is what they should have produced)")
            elif failed:
                term.say(term.colour("goal not reached", False)
                         + f": stopped at action {failed_at + 1} of {n}; the state below is the one "
                           "before that action")
                grip, held = robot.grip(), pddl.held(state)
                if grip is not None and grip != (held is not None):
                    term.say(term.colour("warning", False) + ": the gripper is "
                             + ("holding something" if grip else "empty") + ", unlike the state below: "
                             "check the scene, then use `reset` or the action console")
    except KeyboardInterrupt:
        term.say()
    finally:
        robot.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
