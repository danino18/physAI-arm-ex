"""The exercise's problem file (exercise/task_planning/problem.pddl) as the
template of every planning problem.

The goal console (and solve.py) read it, keep its objects (the default
items), and replace its :init with the current state and its :goal with the
goal typed; items added at run time join its objects. The result goes to
the exercise's load_problem as PDDL text.
"""

import functools
import re

from ..exercise.task_planning import planning as tp
from .planner import PlannerError


class ProblemFileError(PlannerError):
    pass


def read(path=None):
    with open(path or tp.PROBLEM_FILE) as f:
        return f.read()


def _strip_comments(text):
    return "\n".join(line.split(";", 1)[0] for line in text.splitlines())


def _section(text, key):
    """(start, end) of the section `(:key ...)` in text (comments removed)."""
    m = re.search(r"\(\s*:" + key + r"\b", text, re.IGNORECASE)
    if m is None:
        raise ProblemFileError(f"{tp.PROBLEM_FILE} has no (:{key} ...) section")
    depth = 0
    for k in range(m.start(), len(text)):
        if text[k] == "(":
            depth += 1
        elif text[k] == ")":
            depth -= 1
            if depth == 0:
                return m.start(), k + 1
    raise ProblemFileError(f"{tp.PROBLEM_FILE}: the (:{key} ...) section is not closed")


@functools.lru_cache(maxsize=1)
def _default_objects(text):
    try:
        problem = tp.load_problem(text)
    except Exception as e:  # noqa: BLE001  (UPF's parse errors)
        raise ProblemFileError(f"{tp.PROBLEM_FILE} cannot be read with domain.pddl: {type(e).__name__}: {e}") from e
    names = [o.name for o in problem.all_objects]
    if not names:
        raise ProblemFileError(f"{tp.PROBLEM_FILE} has no objects")
    return tuple(names)


def default_objects():
    """The objects of the problem file (read by UPF, with the domain)."""
    return list(_default_objects(read()))


def fill(state, goal, items, name="task", slots=()):
    """The problem file with `items` as its objects (and `slots`, the pyramid's,
    of type slot), `state` (a set of facts) as :init and `goal` (PDDL text) as :goal."""
    text = _strip_comments(read())
    objects = "(:objects\n        " + "\n        ".join(items) + "\n        - item"
    if slots:
        objects += "\n        " + "\n        ".join(slots) + "\n        - slot"
    objects += "\n    )"
    init = "(:init\n" + "".join(f"        ({' '.join(f)})\n" for f in sorted(state)) + "    )"
    for key, new in (("goal", f"(:goal {goal})"), ("init", init), ("objects", objects)):
        a, b = _section(text, key)
        text = text[:a] + new + text[b:]
    text = re.sub(r"\(\s*problem\s+[^\s()]+\s*\)", f"(problem {name})", text, count=1, flags=re.IGNORECASE)
    return "\n".join(line for line in text.splitlines() if line.strip()) + "\n"
