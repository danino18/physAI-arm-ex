#!/usr/bin/env python3
"""Type a PDDL action, and the robot performs it autonomously.

    run/actions.sh sim|real                      (the usual way: sets up the environment)
    ros2 run lab_replica action_console.py [--sim] [--speed-scale S]

Needs the arm, the gripper and the wrist camera (run/sim.sh, or on the real
robot their drivers: README, "Run it on the real robot"). Loads SAM 3 and,
once the arm, the gripper and the camera answer, shows the actions, the items
and a prompt. Each action is perceived, planned and executed by the skills
(README section A.3); the answer is

    invalid action string: <why>     nothing was sent to the robot
    performing action                then, when it is over:
    action success: <what was done>
    action failed: <reason>          a refused action leaves the scene as it was

New items (README "Add a new item"), for this session only:

    new-item: purple-block           the robot looks for "purple block" on the
                                     table (SAM 3), measures its width and height
                                     from the depth and adds it to the items (a
                                     box or a round model)
    new-item: purple-block prompt="small purple cube" shape=box
                                     another text prompt; force the shape
    forget-item: purple-block        remove an added item

Added items are not PDDL objects (config/domain.pddl and the problem files
are unchanged) and are gone when the console quits.

A window shows the wrist camera live (colour and depth side by side) and
every view the robot takes, annotated (viewer.py); --no-window leaves it out.

Ctrl-C during an action stops the arm; Ctrl-C or Ctrl-D at the prompt quits.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))

from lab_replica.console import (ACTIONS, FORGET_BARRIER, FORGET_ITEM, NEW_ITEM, SCAN, InvalidAction, Robot,  # noqa: E402
                                 Terminal, banner, barrier_command, barrier_hints, common_arguments,
                                 item_command, item_hints, parse_action, parse_barrier_command,
                                 parse_item_command, unknown_item_hint, warn_holding_added, wrap_words)
from lab_replica.planning import pddl  # noqa: E402


def hints(term, added):
    term.say()
    term.say(term.bold("Actions: ") + "   ".join(f"({a} <item>)" for a in ACTIONS))
    term.say(term.bold("Items:   ") + wrap_words(pddl.ITEMS + [i + "*" for i in added], 9))
    term.say(term.bold("Example: ") + "(pick-up green-mug)   then   (put-in-basket green-mug)")
    item_hints(term, added)
    barrier_hints(term)
    term.say(term.dim("Tab completes names; Ctrl-D quits."))


def main():
    sys.stdout.reconfigure(line_buffering=True)
    ap = common_arguments(argparse.ArgumentParser(description=__doc__,
                                                  formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--no-window", action="store_true",
                    help="do not open the camera window (live colour and depth, annotated observations)")
    args, _ = ap.parse_known_args()
    term = Terminal(list(ACTIONS) + list(pddl.ITEMS) + [NEW_ITEM, FORGET_ITEM, SCAN, FORGET_BARRIER], "actions")
    banner(term, "action console", args.sim)
    robot = Robot(term, sim=args.sim, speed_scale=args.speed_scale, verbose=args.verbose,
                  return_home=not args.no_home, save_frames=args.save_frames, window=not args.no_window,
                  barrier_file=args.barrier_file)
    term.say(f"Ready. Log and camera views: {robot.run_dir}")
    warned = False
    try:
        while True:
            hints(term, robot.added)
            with robot.idle():
                line = term.ask("action> ")
            if line is None or line.strip().lower() in ("quit", "exit", "q"):
                if not warned and warn_holding_added(term, robot):
                    warned = True
                    continue
                break
            if not line.strip():
                continue
            warned = False
            barrier_cmd = parse_barrier_command(line)
            if barrier_cmd is not None:
                barrier_command(term, robot, barrier_cmd)
                continue
            try:
                cmd = parse_item_command(line, robot.added)
            except InvalidAction as e:
                term.say(term.colour("invalid item command", False) + f": {e}")
                continue
            if cmd is not None:
                item_command(term, robot, *cmd)
                continue
            try:
                action = parse_action(line, robot.added)
            except InvalidAction as e:
                term.say(term.colour("invalid action string", False) + f": {e}")
                if str(e).startswith("unknown item"):
                    unknown_item_hint(term, line.lower().replace("(", " ").replace(")", " ").split()[-1])
                continue
            term.say(f"performing action {action}")
            ok, msg = robot.execute(action)
            term.say(term.colour("action success", True) + f": {msg}" if ok else
                     term.colour("action failed", False) + f": {msg}")
    except KeyboardInterrupt:
        term.say()
    finally:
        robot.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
