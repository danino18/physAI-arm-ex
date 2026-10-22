"""A task: a scene (where the items are, hence the initial state) plus the
user's goal, written as a PDDL problem over the fixed domain and objects.

A task directory (problems_tamp/<name>/) holds

    problem.pddl   the twelve items, the scene's initial state, the goal
    domain.pddl    exercise/task_planning/domain.pddl (always the same)
    plan.txt       the plan (UPF + Fast Downward);  planner.log  its errors
    layout.yaml    where every item stands (with the init, goal and plan)
    layout.png     top view of the layout;  render.png  the lab camera's view
    world.sdf      the Gazebo world of the scene (items at their settled poses)
    README.md      the task, the plan, the commands, the table set-up
    result.json    what happened when the plan was executed (solve.py)

The scene is either generated (a random stable layout, as generate_problem.py
makes them, sampled until the goal is reachable from its initial state) or
taken from an existing problem or task directory (its world.sdf, layout and
initial state). A task directory is itself a scene for further goals.
"""

import os
import shutil
import time

import numpy as np
import yaml

from .. import config
from ..planning import pddl
from ..exercise.task_planning import planning as tp
from . import planner, problem_file
from .goal import to_text

TASKS_DIR = "problems_tamp"
SCENE_FILES = ("world.sdf", "layout.png", "render.png")


def tasks_root():
    return os.path.join(config.workspace_root(), TASKS_DIR)


def task_dir(name_or_path):
    """A task directory from its name (in problems_tamp/) or a path."""
    d = config.problem_dir(name_or_path)
    if not os.path.exists(os.path.join(d, "problem.pddl")):
        raise FileNotFoundError(f"{d} is not a task: it has no problem.pddl")
    return d


def scene_dir(name_or_path):
    """A scene: a problem or task directory (a name in problems/,
    problems_tamp/, problems_gen/, or a path) with world.sdf and problem.pddl."""
    d = config.problem_dir(name_or_path)
    for f in ("world.sdf", "problem.pddl", "layout.yaml"):
        if not os.path.exists(os.path.join(d, f)):
            raise FileNotFoundError(f"{d} is not a scene: it has no {f}")
    return d


class Plan(list):
    """A plan: its actions as text, e.g. ['(pick-up red-can)', ...], with
    .states, the state each action should produce (the exercise's
    expected_states, simulated by UPF)."""

    def __init__(self, actions, states):
        super().__init__(actions)
        self.states = list(states)


def default_items():
    """The objects of the exercise's problem file (exercise/task_planning/problem.pddl)."""
    return problem_file.default_objects()


def problem_text(init, formula, name, items=None, slots=()):
    """The exercise's problem file with `items` (default: its own objects) as
    the objects (and `slots`, the pyramid's), `init` as :init and the goal
    formula as :goal."""
    return problem_file.fill(init, to_text(formula), list(items or default_items()), name=name, slots=tuple(slots))


def solve(init, formula, out_dir=None, log=print, time_limit=60, items=None, slots=()):
    """The plan from `init` to the goal over `items` (default: the objects of
    the exercise's problem.pddl) (a Plan), with the exercise's task planner
    (exercise/task_planning: the problem file, UPF, Fast Downward, the
    expected states). `slots`: the pyramid's slots (exercise/pyramid.py), objects
    of the problem besides the items. Raises planner.Unsolvable. With out_dir,
    problem.pddl, plan.txt and planner.log are written there."""
    planner.check()
    name = (os.path.basename(out_dir.rstrip("/")) if out_dir else "") or "task"
    text = problem_text(init, formula, name, items, slots)
    if out_dir:
        with open(os.path.join(out_dir, "problem.pddl"), "w") as f:
            f.write(text)
    try:
        problem = tp.load_problem(text)
        actions = tp.solve(problem, time_limit=time_limit)
        states = tp.expected_states(problem, actions)
    except planner.PlannerError as e:
        if out_dir:
            with open(os.path.join(out_dir, "planner.log"), "a") as f:
                f.write(f"{type(e).__name__}: {e}\n")
        raise
    except Exception as e:  # noqa: BLE001  (UPF's own errors: a bad domain or problem text)
        if out_dir:
            with open(os.path.join(out_dir, "planner.log"), "a") as f:
                f.write(f"{type(e).__name__}: {e}\n")
        raise planner.PlannerError(f"{type(e).__name__}: {e}") from e
    plan = Plan([planner.action_text(a) for a in actions], states)
    if out_dir:
        with open(os.path.join(out_dir, "plan.txt"), "w") as f:
            f.write("".join(a + "\n" for a in plan) + f"; cost = {len(plan)} (unit cost)\n")
    return plan


def reachable(init, formula, items=None):
    """Whether the goal can be reached from `init` at all (the planner)."""
    try:
        solve(init, formula, log=lambda s: None, items=items)
        return True
    except planner.Unsolvable:
        return False


def most_permissive_init(items=None):
    """Every item (default: the objects of the exercise's problem.pddl) on the
    table, gripper empty: every state of the domain is reachable from here
    (items never leave the basket)."""
    return frozenset({("on-table", i) for i in (items or default_items())} | {pddl.EMPTY})


# ---- scenes --------------------------------------------------------------------------

def generate_scene(formula, out_dir, seed=None, n_basket=None, p_holding=0.3, randomize_basket=False,
                   render=True, log=print):
    """Generate a random scene whose initial state lets the goal be reached;
    write its files to out_dir. Returns (init, held item or None, seed)."""
    from ..gazebo.world import problem_world
    from ..planning.placement import placement_at
    from ..problems.drawing import draw_layout
    from ..problems.files import render_camera_view, write_layout_yaml
    from ..problems.generator import GenerationError, Generator

    if seed is None:
        seed = int(time.time()) % 100000
    gen = Generator(seed=seed, randomize_basket=randomize_basket, log=log)
    rng = np.random.default_rng(seed + 7919)
    last = None
    for attempt in range(4):
        nb = n_basket if n_basket is not None else int(rng.integers(0, 4))
        holding = bool(rng.uniform() < p_holding)
        log(f"  scene (seed {seed}): {nb} items in the basket, "
            f"{'holding an item' if holding else 'gripper empty'} at the start")
        try:
            placed, settled, held = gen.make_layout(nb, holding, accept=lambda init: reachable(init, formula))
            break
        except GenerationError as e:
            last = e
            log(f"  {e}")
    else:
        raise GenerationError(f"no scene from which the goal can be reached ({last}); "
                              f"try fewer --basket-items or another --p-holding")
    placements = [placement_at(gen.items[p.item], p, *settled[p.item]) for p in placed]
    init = gen.init_state(placements, held)
    prob = dict(placements=placements, settled=settled, held=held, init=init, goal=to_text(formula),
                plan=[], basket_xy=gen.basket_xy.tolist())
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "world.sdf"), "w") as f:
        f.write(problem_world(gen.cell, gen.items, poses=settled, basket_xy=prob["basket_xy"]))
    write_layout_yaml(out_dir, prob, os.path.basename(out_dir), gen.cell, seed)
    draw_layout(os.path.join(out_dir, "layout.png"), gen.cell, gen.items, prob, title=os.path.basename(out_dir))
    if render:
        log("  rendering the lab camera's view ...")
        try:
            render_camera_view(gen.cell, gen.items, prob, os.path.join(out_dir, "render.png"))
        except Exception as e:  # noqa: BLE001  (only a picture)
            log(f"  warning: no render.png ({e})")
    return init, held, seed


def copy_scene(src, out_dir):
    """Take the scene of an existing problem or task directory. Returns
    (init, held item or None, seed or None)."""
    init, _ = pddl.read_problem(os.path.join(src, "problem.pddl"))
    os.makedirs(out_dir, exist_ok=True)
    for f in SCENE_FILES:
        if os.path.exists(os.path.join(src, f)):
            shutil.copy(os.path.join(src, f), os.path.join(out_dir, f))
    layout = _read_yaml(os.path.join(src, "layout.yaml"))
    layout["scene_from"] = os.path.relpath(src, config.workspace_root())
    with open(os.path.join(out_dir, "layout.yaml"), "w") as f:
        yaml.safe_dump(layout, f, sort_keys=False, width=110)
    return init, pddl.held(init), layout.get("seed")


# ---- the task's files ---------------------------------------------------------------------

def write_task(out_dir, init, formula, plan, source):
    """problem.pddl, domain.pddl, the goal and plan in layout.yaml, README.md."""
    name = os.path.basename(out_dir.rstrip("/"))
    with open(os.path.join(out_dir, "problem.pddl"), "w") as f:
        f.write(problem_text(init, formula, name))
    shutil.copy(tp.DOMAIN_FILE, os.path.join(out_dir, "domain.pddl"))
    lp = os.path.join(out_dir, "layout.yaml")
    layout = _read_yaml(lp)
    layout.update(name=name, init=sorted(pddl.fact_str(f) for f in init), goal=to_text(formula),
                  goal_from=source, plan=plan)
    layout.pop("optimal_plan", None)
    with open(lp, "w") as f:
        yaml.safe_dump(layout, f, sort_keys=False, width=110)
    write_readme(out_dir, layout)


README = """# {name}

<!-- Written by solve.py. Edits are overwritten. -->

A task for the autonomous skills: the goal below (from {source}), in the
scene {scene}.

**Initial state:** {init_text}

**Goal:** `{goal}`

**Plan (UPF + Fast Downward, {n} actions):**

{plan}

## Run it

All in one (starts the simulation and the skill server, executes the plan,
checks the goal in the simulator):

```console
ros2 run lab_replica solve.py --task {name}
```

With the simulation (or the real robot) and the skill server already
running (README section T.2):

```console
ros2 launch lab_replica sim.launch.py problem:={name}
ros2 run lab_replica skill_server.py --sim
ros2 run lab_replica solve.py --task {name} --attach
```
{held_note}
## Setting it up on the real table

Place the items as in [`layout.png`](layout.png) and the table below (world
frame: origin on the table surface under the robot base axis, x to the right
as seen from the camera, y towards the wall; yaw about the vertical from +x).
The basket's centre is at ({bx:.1f}, {by:.1f}) cm. The robot finds the items
itself, so a few centimetres off is fine.

| # | Item | Where | Resting pose | x [cm] | y [cm] | yaw [deg] |
|---|---|---|---|---|---|---|
{rows}
"""


def write_readme(out_dir, layout):
    held = layout.get("held_at_start")
    rows = []
    for k, (item, o) in enumerate(sorted(layout["objects"].items()), start=1):
        where = "basket" if o["support"] == "basket" else "table"
        if item == held:
            where = "table (picked up before the plan: it starts in the gripper)"
        rows.append(f"| {k} | {item} | {where} | {o['resting_pose']} | {o['xyz'][0] * 100:.1f} | "
                    f"{o['xyz'][1] * 100:.1f} | {o['yaw_deg']:.0f} |")
    held_note = ""
    if held:
        held_note = (f"\nThe initial state has the robot holding the **{held}**. Before the plan, the robot "
                     f"picks it up from its place on the table (skipped when the gripper already holds "
                     f"something).\n")
    special = [f for f in layout["init"] if f.startswith("(robot") or f.startswith("(in-basket")]
    on_table = sum(f.startswith("(on-table") for f in layout["init"])
    seed = layout.get("seed")
    scene = (f"of `{layout['scene_from']}`" if layout.get("scene_from")
             else f"generated with seed {seed}" if seed is not None else "of this directory")
    plan = layout["plan"]
    bx, by = layout["basket"]["center_xy"]
    text = README.format(
        name=layout["name"], source=layout.get("goal_from", "the command line"), scene=scene,
        init_text=f"{on_table} items on the table" + "".join(f", {f}" for f in special),
        goal=layout["goal"], n=len(plan),
        plan="\n".join(f"{k + 1}. {a}" for k, a in enumerate(plan)) or "(the goal already holds)",
        held_note=held_note,
        bx=bx * 100, by=by * 100, rows="\n".join(rows))
    with open(os.path.join(out_dir, "README.md"), "w") as f:
        f.write(text)


def read_task(out_dir):
    """(init, goal formula text, plan) of a task directory."""
    from .goal import goal_from_problem_file
    init, _ = pddl.read_problem(os.path.join(out_dir, "problem.pddl"))
    formula, _ = goal_from_problem_file(os.path.join(out_dir, "problem.pddl"))
    layout = _read_yaml(os.path.join(out_dir, "layout.yaml"))
    return init, formula, layout.get("plan")


def _read_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def new_name():
    return time.strftime("task_%Y%m%d_%H%M%S")


def describe_init(init):
    held = pddl.held(init)
    basket = sorted(f[1] for f in init if f[0] == "in-basket")
    return (f"{sum(f[0] == 'on-table' for f in init)} items on the table"
            + (f", in the basket: {', '.join(basket)}" if basket else ", basket empty")
            + (f", holding {held}" if held else ", gripper empty"))

