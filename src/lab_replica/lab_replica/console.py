"""Interactive consoles: type a PDDL action, or a goal, and the robot does it.

Both run the autonomous skills (autonomy/executive.py) in this process: one
ROS node holds the arm and gripper clients and the wrist camera
subscriptions, and SAM 3 is loaded once at start. The prompt appears only
when the arm, the gripper and the camera answer and SAM 3 is loaded.

    scripts/action_console.py   one grounded action at a time
    scripts/goal_console.py     a goal: UPF + Fast Downward plan, the actions run in turn

Neither knows whether it drives the simulation or the real robot, except
for the clock (--sim: simulation time).
"""

import difflib
import os
import re
import shlex
import sys
import time

from .planning import pddl

ACTIONS = ("pick-up", "put-in-basket", "put-on-table")
PREDICATES = ("on-table", "in-basket", "robot-gripping", "robot-gripper-empty")


class InvalidAction(ValueError):
    pass


def parse_action(text, extra_items=()):
    """'(pick-up green-mug)' (parentheses and case optional) -> '(pick-up green-mug)'.
    extra_items: items added in this session (new-item:) besides the PDDL ones.
    Raises InvalidAction with the reason."""
    words = text.strip().lower().replace("(", " ").replace(")", " ").split()
    if len(words) != 2:
        raise InvalidAction("an action is a name and one item, e.g. (pick-up green-mug)")
    name, item = words
    if name not in ACTIONS:
        close = difflib.get_close_matches(name, ACTIONS, n=1, cutoff=0.5)
        raise InvalidAction(f"unknown action {name!r}" + (f" (did you mean {close[0]}?)" if close else ""))
    items = list(pddl.ITEMS) + list(extra_items)
    if item not in items:
        close = difflib.get_close_matches(item, items, n=1, cutoff=0.4)
        raise InvalidAction(f"unknown item {item!r}" + (f" (did you mean {close[0]}?)" if close else ""))
    return f"({name} {item})"


# ---- items added for one session (new-item:) ---------------------------------------------------

NEW_ITEM, FORGET_ITEM = "new-item:", "forget-item:"
ITEM_SHAPES = ("box", "round")
NAME_RE = re.compile(r"^[a-z]+(-[a-z]+)*$")   # what planning/pddl.py parses
# Not item names: goal connectives (tamp/goal.py) and PDDL words.
RESERVED = ("and", "or", "not", "imply", "forall", "exists", "item", "object", "either", "define", "domain",
            "problem")


def parse_item_command(text, added=()):
    """'new-item: purple-block [prompt="small purple cube"] [shape=box|round]'
    -> ("new-item", name, {"prompt": ..., "shape": ...}); 'forget-item:
    purple-block' -> ("forget-item", name, {}); None for any other line.
    `added`: the items added so far in this session. Raises InvalidAction
    with the reason."""
    t = text.strip()
    # The colon may be left out: "new-item purple-block".
    cmd = next((c for c in (NEW_ITEM, FORGET_ITEM) if t.lower().startswith(c[:-1])
                and t[len(c) - 1:len(c)] in (":", " ", "")), None)
    if cmd is None:
        return None
    try:
        words = shlex.split(t[len(cmd) - 1:].lstrip(":"))
    except ValueError as e:
        raise InvalidAction(str(e))
    if not words:
        raise InvalidAction(f"give the item's name, e.g. {cmd} purple-block")
    name, opts = words[0].lower(), {}
    if cmd == FORGET_ITEM:
        if len(words) > 1:
            raise InvalidAction(f"{cmd} takes only the item's name")
        if name not in added:
            raise InvalidAction(f"{name!r} was not added with {NEW_ITEM}"
                                + (f"; the added items are {', '.join(sorted(added))}" if added else ""))
        return cmd[:-1], name, opts
    if not NAME_RE.match(name):
        raise InvalidAction(f"{name!r}: a name is lowercase words joined by '-', e.g. purple-block")
    if name in pddl.ITEMS or name in added:
        raise InvalidAction(f"{name} is already an item" +
                            (f" ({FORGET_ITEM} {name} first to measure it again)" if name in added else ""))
    if name in ACTIONS or name in PREDICATES or name in RESERVED:
        raise InvalidAction(f"{name} is a word of PDDL or of the domain")
    for w in words[1:]:
        key, eq, val = w.partition("=")
        key = key.lower()
        if not eq or key not in ("prompt", "shape"):
            raise InvalidAction(f"unknown option {w!r} (options: prompt=\"...\", shape=box|round)")
        if key == "prompt":
            if not val.strip():
                raise InvalidAction("prompt is empty")
            opts["prompt"] = val.strip()
        else:
            if val.lower() not in ITEM_SHAPES:
                raise InvalidAction(f"shape is {' or '.join(ITEM_SHAPES)}")
            opts["shape"] = val.lower()
    opts.setdefault("prompt", name.replace("-", " "))
    return cmd[:-1], name, opts


# ---- terminal ------------------------------------------------------------------------------

class Terminal:
    """Prompt with history and Tab completion; a status line that is
    rewritten in place while an action runs."""

    def __init__(self, words, history_name):
        self.tty = sys.stdout.isatty()
        self.status_shown = False
        try:
            import readline
        except ImportError:
            return
        self.words = sorted(set(words))
        readline.set_completer_delims(" \t\n()")
        readline.set_completer(self._complete)
        readline.parse_and_bind("tab: complete")
        self.history = os.path.join(os.path.expanduser("~"), ".ros", f"lab_{history_name}_history")
        try:
            readline.read_history_file(self.history)
        except OSError:
            pass
        readline.set_history_length(500)
        import atexit
        atexit.register(self._save_history, readline)

    def _save_history(self, readline):
        try:
            os.makedirs(os.path.dirname(self.history), exist_ok=True)
            readline.write_history_file(self.history)
        except OSError:
            pass

    def add_words(self, words):
        if hasattr(self, "words"):
            self.words = sorted(set(self.words) | set(words))

    def remove_word(self, word):
        if hasattr(self, "words"):
            self.words = [w for w in self.words if w != word]

    def _complete(self, text, state):
        hits = [w for w in self.words if w.startswith(text.lower())]
        return hits[state] if state < len(hits) else None

    def bold(self, s):
        return f"\033[1m{s}\033[0m" if self.tty else s

    def dim(self, s):
        return f"\033[2m{s}\033[0m" if self.tty else s

    def colour(self, s, ok):
        return f"\033[1;{32 if ok else 31}m{s}\033[0m" if self.tty else s

    def status(self, text):
        """The current phase of the running action."""
        if self.tty:
            width = max(20, os.get_terminal_size().columns - 1) if sys.stdout.isatty() else 100
            sys.stdout.write("\r\033[K" + self.dim(f"  ... {text}"[:width]))
            sys.stdout.flush()
            self.status_shown = True
        else:
            print(f"  ... {text}", flush=True)

    def end_status(self):
        if self.status_shown:
            sys.stdout.write("\r\033[K")
            sys.stdout.flush()
            self.status_shown = False

    def say(self, text=""):
        self.end_status()
        print(text, flush=True)

    def ask(self, prompt):
        """A line from the user; None on Ctrl-D."""
        self.end_status()
        try:
            return input(prompt)
        except EOFError:
            print()
            return None


def wrap_words(words, indent, width=100):
    """Words joined into lines of at most `width` characters, each but the
    first indented by `indent` spaces."""
    lines, cur = [], ""
    for w in words:
        if cur and len(cur) + 2 + len(w) > width - indent:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur}  {w}" if cur else w
    lines.append(cur)
    return ("\n" + " " * indent).join(lines)


# ---- the robot -------------------------------------------------------------------------------

def _start_hint(sim, what):
    """Where `what` (a driver or the camera) is started."""
    return "run/sim.sh" if sim else f"{what}: README, \"Run it on the real robot\""


class Robot:
    """The autonomous skills in this process: arm, gripper, wrist camera, SAM 3.

    log_file receives the executive's detailed log (the terminal shows the
    phases only, unless verbose)."""

    def __init__(self, term, sim=False, speed_scale=1.0, run_dir=None, verbose=False, return_home=True,
                 save_frames=False, window=False, barrier_file=None):
        import rclpy
        from rclpy.parameter import Parameter
        from rclpy.signals import SignalHandlerOptions

        from . import config
        from .execution.executor import Executor, active_trajectory_controller

        self.term = term
        self.rclpy = rclpy
        self.run_dir = run_dir or os.path.join(config.workspace_root(), "runs", time.strftime("%Y%m%d_%H%M%S"))
        os.makedirs(self.run_dir, exist_ok=True)
        self.log_file = open(os.path.join(self.run_dir, "console.log"), "a", buffering=1)
        self.verbose = verbose
        # Ctrl-C must reach this program (to stop the arm), not shut ROS down.
        rclpy.init(signal_handler_options=SignalHandlerOptions.NO)

        step = self._step
        step("connecting to the arm and the gripper")
        probe = rclpy.create_node("lab_console_probe")
        if sim:
            probe.set_parameters([Parameter("use_sim_time", value=True)])
        try:
            controller = active_trajectory_controller(probe, timeout=30.0)
        except Exception as e:  # noqa: BLE001
            raise SystemExit(f"\nno arm: {e}\n  Is the robot running? ({_start_hint(sim, 'the UR driver')})")
        finally:
            probe.destroy_node()
        self.node = Executor(controller=controller, speed_scale=speed_scale)
        if sim:
            self.node.set_parameters([Parameter("use_sim_time", value=True)])
        try:
            self.node.wait_ready(timeout=30.0)
        except Exception as e:  # noqa: BLE001
            raise SystemExit(f"\nno arm or gripper: {e}\n  Is the robot running? "
                             f"({_start_hint(sim, 'the UR and RG6 drivers')})")
        self._done(f"controller {controller}")

        step("loading SAM 3 (the segmentation model), starting MoveIt")
        t0 = time.time()
        os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        import warnings
        warnings.filterwarnings("ignore", category=FutureWarning)
        from .autonomy.executive import ActionExecutive
        self.ex = ActionExecutive(self.node, log=self._log, run_dir=self.run_dir, return_home=return_home,
                                  save_frames=save_frames)
        self._done(f"{time.time() - t0:.0f} s")
        if barrier_file:
            self.ex.load_barrier(barrier_file)

        step("checking the gripper")
        try:
            self.holding = self.ex.holding()
        except Exception as e:  # noqa: BLE001
            raise SystemExit(f"\n{e}\n  Is the gripper driver running? ({_start_hint(sim, 'the RG6 driver')})")
        self._done("holding something" if self.holding else "empty")

        step("waiting for the wrist camera")
        try:
            f = self.ex.cam.capture(after=self.ex.now() - 1e6, timeout=20.0)
        except TimeoutError:
            raise SystemExit(f"\nno camera images on {self.ex.cfg['topics']['color']} and "
                             f"{self.ex.cfg['topics']['depth']}\n  Is the camera running? "
                             f"({_start_hint(sim, 'the RealSense node')})")
        self._done(f"{f.rgb.shape[1]} x {f.rgb.shape[0]}")
        self.ex.on_phase = self._phase
        self.added = []   # items added in this session (new-item:)
        self.viewer = None
        if window:
            self._start_window()

    # ---- the camera window (viewer.py) -----------------------------------------------------------

    def _start_window(self):
        step = self._step
        step("opening the camera window")
        if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
            self._done("skipped: no display (ssh -X, or run on the lab PC's screen)")
            return
        from .viewer import Viewer
        self.viewer = Viewer()
        self._last_live = 0.0
        self.ex.cam.on_image = self._live
        self.ex.on_observation = self.viewer.observation
        self._done("live colour and depth; every observation, annotated")

    def _live(self, color_msg, depth_msg):
        now = time.time()
        if now - self._last_live < 0.1 or not self.viewer.alive():
            return
        self._last_live = now
        from .perception.rgbd import decode_image
        try:
            self.viewer.live(decode_image(color_msg), decode_image(depth_msg))
        except Exception as e:  # noqa: BLE001
            self._log(f"camera window: {e}")

    def idle(self):
        """Context: keep the camera images coming (for the window) while the
        console waits for input. Nothing else may spin the node meanwhile."""
        import contextlib
        import threading

        if self.viewer is None:
            return contextlib.nullcontext()
        stop = threading.Event()

        def spin():
            while not stop.is_set() and self.rclpy.ok():
                try:
                    self.rclpy.spin_once(self.node, timeout_sec=0.05)
                except Exception:  # noqa: BLE001
                    return

        @contextlib.contextmanager
        def ctx():
            t = threading.Thread(target=spin, daemon=True, name="idle-spin")
            t.start()
            try:
                yield
            finally:
                stop.set()
                t.join()
        return ctx()

    def window_status(self, text):
        """The camera window's status bar (when there is a window)."""
        if self.viewer is not None:
            self.viewer.status(text)

    def _step(self, text):
        sys.stdout.write(f"  {text} ... ")
        sys.stdout.flush()

    def _done(self, text):
        print(f"ok ({text})" if text else "ok", flush=True)

    def _log(self, text):
        self.log_file.write(f"{time.strftime('%H:%M:%S')} {text}\n")
        if self.verbose:
            self.term.say(self.term.dim(text))

    def _phase(self, text):
        self.window_status(text if text.startswith(("done:", "failed:")) else f"{self.ex.action}: {text}")
        if text.startswith(("done:", "failed:")):
            return
        if not self.verbose:
            self.term.status(text)

    def grip(self):
        """The gripper's grip-detected flag, or None without a reading."""
        try:
            return self.ex.holding()
        except Exception:  # noqa: BLE001
            return None

    def execute(self, action):
        """Run one action. Returns (success, message). Ctrl-C stops the arm
        and fails the action."""
        return self._guarded(lambda: self.ex.execute(action))

    def _guarded(self, run):
        """run() -> (success, message, ...), with Ctrl-C stopping the arm."""
        try:
            out = run()
        except KeyboardInterrupt:
            n = self.node.cancel_active()
            out = (False, "interrupted (Ctrl-C)" + ("; the arm was stopped" if n else ""))
        except Exception as e:  # noqa: BLE001
            out = (False, f"{type(e).__name__}: {e}")
        self._log(f"   {'OK' if out[0] else 'FAILED'}: {out[1]}")
        self.term.end_status()
        return out

    # ---- the fixed barrier (scan:) -----------------------------------------------------------

    def scan_barrier(self):
        """Scan the surroundings from several views and keep the fixed barrier
        they show. Returns (success, message)."""
        return self._guarded(self.ex.scan_barrier)[:2]

    def forget_barrier(self):
        self.ex.forget_barrier()

    # ---- items added for this session (new-item:) ------------------------------------------

    def new_item(self, name, opts):
        """Measure an item with the camera and add it for this session only:
        it is not added to the PDDL items, nor kept for the next session.
        Returns (success, message)."""
        out = self._guarded(lambda: self.ex.new_item(name, opts["prompt"], opts.get("shape")))
        if out[0]:
            self.added.append(name)
        return out[:2]

    def forget_item(self, name):
        if self.holding_added() == name:
            return False, f"the gripper holds {name}: put it down first"
        self.ex.forget_item(name)
        self.added.remove(name)
        return True, f"{name} is no longer an item"

    def holding_added(self):
        """The added item the gripper holds (as far as the skills know), or None."""
        held = self.ex.held
        return held.item if held is not None and held.item in self.added else None

    def shutdown(self):
        try:
            self.node.cancel_active()
        except Exception:  # noqa: BLE001
            pass
        try:
            self.ex.mv.stop()       # the move_group this console started, if any
        except Exception:  # noqa: BLE001
            pass
        if self.viewer is not None:
            self.ex.cam.on_image = None
            self.viewer.close()
        self.rclpy.try_shutdown()
        self.log_file.close()


# ---- scan: / forget-barrier: in a console --------------------------------------------------

SCAN, FORGET_BARRIER = "scan:", "forget-barrier:"


def parse_barrier_command(text):
    """'scan:' -> "scan", 'forget-barrier:' -> "forget-barrier" (the colon may
    be left out); None for any other line."""
    t = text.strip().lower().rstrip(":").strip()
    return t if t in ("scan", "forget-barrier") else None


def barrier_hints(term):
    term.say(term.bold("Barrier:") + f"  {SCAN} looks at the surroundings from several views and keeps clear of "
             f"what it finds that is no item (slow: a hand on the emergency stop)   {FORGET_BARRIER}")


def barrier_command(term, robot, cmd):
    """Run a scan: / forget-barrier: command (parse_barrier_command). Returns
    whether it succeeded."""
    if cmd == "forget-barrier":
        robot.forget_barrier()
        term.say(term.colour("barrier forgotten", True) + ": the motions no longer keep clear of it")
        return True
    term.say("scanning the surroundings from several views ...")
    ok, msg = robot.scan_barrier()
    term.say(term.colour("barrier scanned", True) + f": {msg}" if ok else
             term.colour("barrier not scanned", False) + f": {msg}")
    return ok


# ---- new-item: / forget-item: in a console -------------------------------------------------

def item_hints(term, added):
    """The lines of the hints that tell how to add an item."""
    term.say(term.bold("New item:") + f" put it on the table, then e.g.  {NEW_ITEM} purple-block  "
             "(the camera finds and measures it)")
    term.say(" " * 10 + term.dim(f'options: prompt="..." shape=box|round   {FORGET_ITEM} <name>   '
                                 "added items" + (" (*)" if added else "") + " last until you quit"))


def item_command(term, robot, cmd, name, opts):
    """Run a new-item: / forget-item: command (parse_item_command). Returns
    whether it succeeded."""
    if cmd == "forget-item":
        ok, msg = robot.forget_item(name)
        if ok:
            term.remove_word(name)
        term.say(term.colour("item forgotten", True) + f": {msg}" if ok else
                 term.colour("item not forgotten", False) + f": {msg}")
        return ok
    term.say(f"measuring new item {name}: looking for {opts['prompt']!r}")
    ok, msg = robot.new_item(name, opts)
    if ok:
        term.add_words([name])
    term.say(term.colour("item added", True) + f": {msg}" if ok else
             term.colour("item not added", False) + f": {msg}")
    return ok


def unknown_item_hint(term, item):
    """After 'unknown item': how to add it, if it may be a new item (not a typo)."""
    if NAME_RE.match(item) and item not in RESERVED:
        term.say(term.dim(f"  if {item} is a new item: put it on the table and type  {NEW_ITEM} {item}"))


def warn_holding_added(term, robot):
    """On quitting: whether the gripper holds an added item, which is
    forgotten when the console quits (a later session could not put it
    down); if so, says so."""
    held = robot.holding_added()
    if held is not None:
        term.say(term.colour("the gripper holds " + held, False) + ", which is forgotten when the console "
                 "quits; put it down first, or quit again to leave it in the gripper")
    return held is not None


def common_arguments(ap):
    g = ap.add_argument_group("robot")
    g.add_argument("--sim", action="store_true", help="the simulation (run/sim.sh): use its clock")
    g.add_argument("--speed-scale", type=float, default=1.0,
                   help="scale all motion speeds (e.g. 0.3 for a first run on the real robot)")
    g.add_argument("--no-home", action="store_true", help="do not return to the home pose after each action")
    g.add_argument("--verbose", action="store_true", help="print every step of the skills, not only the phases")
    g.add_argument("--barrier-file", metavar="FILE",
                   help="the fixed barrier saved by an earlier scan: (barrier.json in its run directory)")
    g.add_argument("--save-frames", action="store_true",
                   help="also save every camera frame (colour, depth, arm joints) in the run directory")
    return ap


def banner(term, title, sim):
    from . import config
    ws = os.path.basename(config.workspace_root())
    term.say(term.bold(f"{ws}: {title}") + f"  ({'simulation' if sim else 'REAL ROBOT'})")
