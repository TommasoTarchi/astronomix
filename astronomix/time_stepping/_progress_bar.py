"""
Host-side progress bar for the time-integration loop.

Renders a single-line, terminal-width-aware progress bar that is driven from
inside the jitted loop via ``jax.debug.callback``. The "iteration" it is fed is
the simulation time, so the bar tracks progress towards ``t_end``. When stdout
is not a terminal (queued or redirected runs) a throttled plain-text log is
written instead.
"""

# general
import math
import shutil
import sys
import time

#: Longest wall-time gap, in seconds, between two lines of the plain-text log,
#: even when the progress percentage has not moved.
_HEARTBEAT_SECONDS = 60.0


class _PlainTextProgressLog:
    """
    The throttling state of the plain-text progress log of one run.

    The progress callback fires once per solver step, so the difference between
    consecutive simulation times IS the current ``dt``: a collapsing ``dt`` (the
    typical symptom of an unstable run) is visible directly in the log, and the
    wall-time heartbeat keeps emitting lines even when the percentage stalls, so
    a stalled run can never look identical to a slow one.

    Attributes:
        last_logged_bucket: The progress bucket (in 0.5 % increments) of the
            last emitted line, or ``None`` before the first line.
        last_logged_wall_time: The wall time of the last emitted line.
        last_simulation_time: The simulation time of the previous callback.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Forget the previous run, so the next callback starts a fresh log."""
        self.last_logged_bucket = None
        self.last_logged_wall_time = None
        self.last_simulation_time = None


#: The plain-text log state of the current run. ``time_integration`` resets it
#: at the start of every run; a run is also detected as new when the simulation
#: time goes backwards (e.g. a jitted function wrapping ``time_integration`` is
#: executed again).
_progress_log = _PlainTextProgressLog()


def _reset_progress_log() -> None:
    """Start a fresh plain-text progress log for a new run."""
    _progress_log.reset()


def _show_progress(
    iteration, total, prefix="", suffix="", decimals=1, fill="█", printEnd="\r"
) -> None:
    """
    Render one frame of the progress bar, sized to the current terminal width.

    When stdout is not a terminal, a carriage-return bar would flood the log
    with full-width frames, so a plain progress line is printed instead: one
    line per 0.5 % of progress, plus a heartbeat line at least every
    ``_HEARTBEAT_SECONDS``, each with the simulation time and the current
    per-step ``dt``. ``decimals``, ``fill`` and ``printEnd`` only apply to the
    terminal bar.

    Args:
        iteration: The current progress value (the simulation time).
        total: The value of ``iteration`` at which the bar is full (``t_end``).
        prefix: Text printed before the bar.
        suffix: Text printed after the percentage.
        decimals: Number of decimal places shown in the percentage.
        fill: Character used for the filled portion of the bar.
        printEnd: Line terminator; ``"\\r"`` keeps overwriting the same line.
    """
    # On a blow-up the simulation time goes non-finite, and ``int(NaN)`` would
    # raise and abort the whole run. Clamp to ``total`` so the bar finishes
    # cleanly instead of crashing; the diagnostics elsewhere report the NaN.
    try:
        time_is_finite = math.isfinite(float(iteration))
    except (TypeError, ValueError):
        time_is_finite = False
    if not time_is_finite:
        iteration = total

    if not sys.stdout.isatty():
        current_time = float(iteration)
        previous_time = _progress_log.last_simulation_time
        if previous_time is not None and current_time < previous_time:
            # The clock went backwards: this callback belongs to a new run.
            _progress_log.reset()
            previous_time = None
        # The step size is inferred from consecutive callbacks; it is unknown
        # for the first callback of a run and meaningless for a clamped NaN time.
        if previous_time is None or not time_is_finite:
            dt = None
        else:
            dt = current_time - previous_time
        _progress_log.last_simulation_time = current_time

        percent = 100.0 * current_time / float(total)
        half_percent_bucket = int(percent * 2)
        wall_time = time.monotonic()
        heartbeat_due = (
            _progress_log.last_logged_wall_time is None
            or wall_time - _progress_log.last_logged_wall_time >= _HEARTBEAT_SECONDS
        )
        if half_percent_bucket == _progress_log.last_logged_bucket and not heartbeat_due:
            return
        _progress_log.last_logged_bucket = half_percent_bucket
        _progress_log.last_logged_wall_time = wall_time
        dt_note = "" if dt is None else f"  dt = {dt:.3e}"
        print(
            f"{prefix}progress {percent:5.1f}%  "
            f"t = {current_time:.6g} / {float(total):.6g}{dt_note} {suffix}".rstrip(),
            flush=True,
        )
        return

    # Recompute the terminal width every frame so the bar keeps filling the
    # line correctly even if the terminal is resized mid-run.
    terminal_width = shutil.get_terminal_size((80, 20)).columns

    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))

    # Size the bar so the whole line fits the terminal: subtract the fixed
    # decorations (prefix, suffix, percentage, separators) from the width, and
    # never shrink below a readable minimum.
    fixed_part = f"{prefix} | | {percent}% {suffix}"
    fixed_length = len(fixed_part)
    bar_length = max(10, terminal_width - fixed_length)

    filled_length = int(bar_length * iteration // total)
    bar = fill * filled_length + "-" * (bar_length - filled_length)

    progress_line = f"{prefix} |{bar}| {percent}% {suffix}"

    # Pad the line out to the full terminal width so a shorter line never leaves
    # leftover characters from the previous, longer frame.
    padded_line = progress_line.ljust(terminal_width)

    print(f"\r{padded_line}", end=printEnd, flush=True)

    # Drop to a fresh line once the bar is full so subsequent output is clean.
    if iteration == total:
        print()
