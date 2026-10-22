#!/usr/bin/env python3
"""Task planning end to end, offline: goal -> UPF problem -> Fast Downward plan
-> expected states -> a (simulated) robot performs the plan.

Uses only the exercise's components (exercise/task_planning/) and the problem
template filler the goal console uses (tamp/problem_file.py). No ROS, no robot.

    python3 examples/task_planning_demo.py                        # a few example goals
    python3 examples/task_planning_demo.py "(in-basket red-can)"  # your goal
    python3 examples/task_planning_demo.py --fail 2 "(in-basket red-can) (in-basket green-mug)"
        # action number 2 of the plan fails, as a real skill could
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "src", "lab_replica"))

from lab_replica.exercise.task_planning import planning  # noqa: E402
from lab_replica.tamp import planner, problem_file  # noqa: E402
from lab_replica.tamp.planner import PlannerError, Unsolvable, action_text  # noqa: E402

EXAMPLE_GOALS = [
    "(in-basket green-mug)",
    "(in-basket red-can) (robot-gripping green-mug)",
    "(and (in-basket red-block) (or (in-basket blue-block) (on-table green-block)))",
    "(forall (?x - item) (in-basket ?x))",
    "(and (in-basket red-can) (on-table red-can))",  # can never hold
]


def initial_state(items):
    """What the goal console assumes at the start: every item on the table,
    the gripper empty."""
    return {("on-table", i) for i in items} | {("robot-gripper-empty",)}


def parse_goal(text):
    """'(a) (b)' is the conjunction of the facts; one formula is used as it is."""
    text = text.strip()
    depth, top = 0, 0
    for ch in text:
        depth += ch == "("
        depth -= ch == ")"
        if ch == ")" and depth == 0:
            top += 1
    return text if top <= 1 else f"(and {text})"


def make_robot(fail_at):
    """A stand-in for the robot: performs an action by doing nothing, except
    that action number `fail_at` (1-based) fails."""
    count = {"n": 0}

    def perform(action):
        count["n"] += 1
        if count["n"] == fail_at:
            return False, "the item slipped out of the gripper (simulated)"
        return True, "done"

    return perform


def run(goal_text, items, fail_at=None):
    print(f"\ngoal: {goal_text}")
    state = initial_state(items)
    goal = parse_goal(goal_text)
    problem = planning.load_problem(problem_file.fill(state, goal, items))
    try:
        plan = planning.solve(problem)
    except Unsolvable as e:
        print(f"  unsolvable: {e}")
        return None
    except PlannerError as e:
        print(f"  planner error: {e}")
        return None

    actions = [action_text(a) for a in plan]
    states = planning.expected_states(problem, plan)
    print(f"  plan ({len(actions)} actions):")
    for k, a in enumerate(actions, 1):
        print(f"    {k}. {a}")

    final, failed_at = planning.execute_plan(
        frozenset(state),
        actions,
        states,
        make_robot(fail_at),
        before=lambda k, a: print(f"  performing {k + 1}/{len(actions)}: {a}") or True,
        after=lambda k, a, ok, msg: print(f"    {'success' if ok else 'FAILED'}: {msg}"),
    )
    if failed_at is None:
        print("  goal reached; the state is what the actions should have produced:")
    else:
        print(f"  goal not reached: stopped at action {failed_at + 1}; the state is the one before it:")
    held = [f[1] for f in final if f[0] == "robot-gripping"]
    print(f"    in basket: {sorted(f[1] for f in final if f[0] == 'in-basket') or 'nothing'}")
    print(f"    gripper: {'holds ' + held[0] if held else 'empty'}")
    return failed_at


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("goal", nargs="*", help="a PDDL goal (default: a few examples)")
    p.add_argument("--fail", type=int, default=None, metavar="K", help="action number K of each plan fails")
    args = p.parse_args()

    planner.check()  # also silences the planner's banner
    items = problem_file.default_objects()
    goals = args.goal or EXAMPLE_GOALS
    for g in goals:
        run(g, items, args.fail)


if __name__ == "__main__":
    main()
