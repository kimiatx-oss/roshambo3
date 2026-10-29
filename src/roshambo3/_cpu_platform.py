"""CPU OS services and scoped scheduling policy, outside numerical code."""
from contextlib import contextmanager
from functools import wraps
import os
import sys


def available_cpu_count():
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return os.cpu_count() or 1


def validate_qos(policy):
    if policy not in ("inherit", "background", "user_initiated"):
        raise ValueError("cpu_qos must be inherit, background, or user_initiated")
    if policy != "inherit" and sys.platform != "darwin":
        raise ValueError("explicit cpu_qos is only supported on macOS")


@contextmanager
def qos_scope(policy="inherit"):
    validate_qos(policy)
    if policy == "inherit":
        yield
        return
    from .backend.cpu import extension
    native = extension()
    previous = native._qos_snapshot()
    if previous[0] == 0:
        # pthread_set_qos_class_self_np cannot restore UNSPECIFIED. Refuse
        # before changing this thread rather than leaking an explicit policy.
        raise RuntimeError("cannot scope cpu_qos on a thread with unspecified QoS")
    native._set_qos(policy)
    try:
        yield
    finally:
        native._restore_qos(*previous)


def scoped_qos(function):
    """Scope the submitting thread; native pools propagate its policy."""
    @wraps(function)
    def wrapped(*args, **kwargs):
        owner = kwargs.get("executor")
        if owner is None and args:
            owner = args[0]
        policy = kwargs.get("cpu_qos", getattr(owner, "cpu_qos", "inherit"))
        with qos_scope(policy):
            return function(*args, **kwargs)
    return wrapped
