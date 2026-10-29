"""Slow scatter work units, for tests that interrupt a build mid-flight.

Importable rather than defined in a test's ``python -c`` script: under the
``spawn`` start method (macOS's default) a worker unpickles each submitted
callable by module and name, and a function in ``__main__`` of a ``-c`` script
cannot be found. The delay is read in the worker, from
``ROSHAMBO3_TEST_SCATTER_DELAY`` (seconds).
"""
import os
import time

from roshambo3.prep import build

_unit = build._scatter_unit
_next_batch = build._scatter_next_batch


def _pause():
    time.sleep(float(os.environ.get("ROSHAMBO3_TEST_SCATTER_DELAY", "0")))


def slow_scatter_unit(job):
    _pause()
    return _unit(job)


def slow_scatter_next_batch(tmp_dir):
    _pause()
    return _next_batch(tmp_dir)


def install():
    """Route the builder's submissions through the slow wrappers. The parent
    submits these by reference; each worker imports this module and calls the
    original function after the pause."""
    build._scatter_unit = slow_scatter_unit
    build._scatter_next_batch = slow_scatter_next_batch
