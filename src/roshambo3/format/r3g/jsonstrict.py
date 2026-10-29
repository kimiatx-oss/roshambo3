"""Strict and canonical JSON for format 3 (``doc/FORMAT.md`` sections 10.5.1, 16.1).

``loads`` is the only JSON parser the format-3 package uses on corpus bytes:

* the size is checked before anything else (limits 1 and 2);
* nesting depth (limit 3) and string-literal length (limit 4) are measured on
  the raw bytes by a vectorized, chunked pre-scan, so a hostile depth never
  reaches the recursive parser and the scan costs O(n) NumPy work, not one
  Python step per byte;
* duplicate keys are refused (an ``object_pairs_hook``), as are ``NaN``,
  ``Infinity`` and ``-Infinity`` (``parse_constant``) and a float literal that
  overflows to infinity (``1e999``, ``parse_float``);
* the top level must be an object.

Every refusal is a :class:`CorpusValidationError` with a section-16.5 code.
``dumps`` is canonical JSON (sorted keys, no whitespace, UTF-8, no NaN).

Limit 4 is counted as the spec states it: the raw bytes of the string literal
between its quotes, **escapes included** (``"\\n"`` is two bytes). Both validators count the same raw bytes.
"""

from __future__ import annotations

import json
import math
from typing import Any, Optional

import numpy as np

from .errors import CorpusValidationError

#: Section 14.1 limits 1-4.
MAX_BYTES = 16 * 1024 * 1024
MAX_DEPTH = 32
MAX_STRING = 64 * 1024

#: The pre-scan works on chunks of this many bytes so its scratch arrays stay
#: a few MiB whatever the document size (limit 34).
_CHUNK = 1 << 20

_QUOTE, _BACKSLASH = 0x22, 0x5C
_OPEN = (0x7B, 0x5B)    # { [
_CLOSE = (0x7D, 0x5D)   # } ]


def dumps(obj: Any) -> bytes:
    """Canonical JSON (section 10.5.1). Raises ``ValueError`` on NaN/Infinity."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def scan_depth_and_strings(data: bytes) -> tuple[int, int]:
    """``(max bracket depth, longest string literal in raw bytes)`` of a JSON
    text, brackets inside strings ignored, escapes counted as their raw bytes.

    Linear and chunked: the state carried across chunks is the trailing run of
    backslashes, whether the chunk ends inside a string (and where that string
    opened), and the current depth.
    """
    buf = np.frombuffer(memoryview(data).cast("B"), dtype=np.uint8)
    n = buf.size
    bs_run = 0          # backslashes ending the previous chunk
    in_str = False
    open_pos = -1       # absolute position of the open quote when in_str
    depth = containers = 0
    max_depth = 0
    max_str = 0
    for base in range(0, n, _CHUNK):
        a = buf[base:base + _CHUNK]
        m = a.size
        idx = np.arange(m, dtype=np.int64)
        is_bs = a == _BACKSLASH
        # number of consecutive backslashes immediately before each position
        last_non_bs = np.maximum.accumulate(np.where(is_bs, -1, idx))
        prev = np.empty(m, dtype=np.int64)
        prev[0] = -1
        prev[1:] = last_non_bs[:-1]
        n_bs_before = np.where(prev >= 0, idx - 1 - prev, bs_run + idx)
        # a quote is real (opens or closes a string) unless an odd run of
        # backslashes escapes it; outside a string a backslash is a syntax
        # error the parser reports later
        qpos = np.flatnonzero((a == _QUOTE) & (n_bs_before % 2 == 0))
        # brackets outside strings: the quote count before each bracket
        # decides whether it is inside a string
        bpos = np.flatnonzero((a == _OPEN[0]) | (a == _OPEN[1])
                              | (a == _CLOSE[0]) | (a == _CLOSE[1]))
        if bpos.size:
            before = np.searchsorted(qpos, bpos)
            inside = ((before + int(in_str)) % 2) == 1
            ch = a[bpos[~inside]]
            containers += int(np.count_nonzero((ch == _OPEN[0]) | (ch == _OPEN[1])))
            if containers > (1 << 18):
                raise CorpusValidationError("json.containers", "JSON container count exceeds 262144")
            delta = np.where((ch == _OPEN[0]) | (ch == _OPEN[1]), 1, -1)
            if delta.size:
                run = depth + np.cumsum(delta)
                max_depth = max(max_depth, int(run.max()))
                depth = int(run[-1])
        if qpos.size:
            absq = qpos + base
            if in_str:
                max_str = max(max_str, int(absq[0] - open_pos - 1))
                absq = absq[1:]
            opens, closes = absq[0::2], absq[1::2]
            if closes.size:
                max_str = max(max_str, int((closes - opens[:closes.size] - 1).max()))
            in_str = opens.size > closes.size
            if in_str:
                open_pos = int(opens[-1])
        if in_str:
            # an unterminated string still counts toward the limit as it grows
            max_str = max(max_str, base + m - open_pos - 1)
        bs_run = (bs_run + m) if last_non_bs[-1] < 0 else int(m - 1 - last_non_bs[-1])
    return max_depth, max_str


def loads(data: bytes, *, max_bytes: int = MAX_BYTES, max_depth: int = MAX_DEPTH,
          max_str: int = MAX_STRING, what: str = "JSON document",
          path: Optional[str] = None, tier: str = "open") -> dict:
    """Parse one strict JSON object (section 16.1). Raises
    :class:`CorpusValidationError` naming ``what``, ``path`` and the limit."""
    def refuse(code: str, msg: str) -> CorpusValidationError:
        return CorpusValidationError(code, f"{what}: {msg}", tier=tier, path=path)

    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError("jsonstrict.loads takes bytes")
    if len(data) > max_bytes:
        raise refuse("json.size", f"{len(data)} bytes, limit {max_bytes}")
    try:
        depth, longest = scan_depth_and_strings(data)
    except CorpusValidationError as e:
        raise refuse(e.code, str(e)) from None
    if depth > max_depth:
        raise refuse("json.depth", f"nests {depth} deep, limit {max_depth}")
    if longest > max_str:
        raise refuse("json.string_length", f"holds a string literal of {longest} bytes, "
                     f"limit {max_str}")

    def pairs(items):
        d = {}
        for k, v in items:
            if k in d:
                raise refuse("json.duplicate_key", f"duplicate key {k!r}")
            d[k] = v
        return d

    def constant(name):
        raise refuse("json.nan", f"{name} is not allowed")

    def parse_float(text):
        v = float(text)
        if not math.isfinite(v):
            raise refuse("json.nan", f"number {text[:40]!r} overflows to infinity")
        return v

    try:
        text = bytes(data).decode("utf-8")
    except UnicodeDecodeError as e:
        raise refuse("json.syntax", f"not UTF-8 ({e.reason} at byte {e.start})") from None
    try:
        obj = json.loads(text, object_pairs_hook=pairs, parse_constant=constant,
                         parse_float=parse_float)
    except CorpusValidationError:
        raise
    except (ValueError, RecursionError) as e:
        raise refuse("json.syntax", f"not valid JSON: {str(e)[:200]}") from None
    if not isinstance(obj, dict):
        raise refuse("json.syntax", "the top level must be a JSON object")
    check_unicode_scalars(obj, what=what, path=path, tier=tier)
    return obj


def check_unicode_scalars(obj, *, what, path=None, tier="open"):
    """JSON escapes must decode to Unicode scalar values, including in keys."""
    stack = [obj]
    while stack:
        value = stack.pop()
        if isinstance(value, str):
            try:
                value.encode("utf-8")
            except UnicodeEncodeError:
                raise CorpusValidationError("json.syntax", f"{what}: lone surrogate",
                                            path=path, tier=tier) from None
        elif isinstance(value, dict):
            stack.extend(value.keys())
            stack.extend(value.values())
        elif isinstance(value, list):
            stack.extend(value)
