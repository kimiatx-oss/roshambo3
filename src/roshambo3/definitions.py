"""Custom color definitions: shipped defaults by hash, everything else behind an
explicit, bounded opt-in (``doc/FORMAT.md`` sections 10.3 and 15).

A format-3 corpus stores its feature-definition text (an RDKit ``.fdef`` or a
SMARTS-list object) and queries are featurized with it. Compiling definition
text from a downloaded file can hang a reader, so:

* the text of RDKit's own ``BaseFeatures.fdef`` is shipped (``data/definitions``)
  and recognized **by hash**; it needs nothing;
* any other definition needs an explicit opt-in naming its exact SHA-256
  (:func:`trust`, ``roshambo3 definitions trust``, ``--trust-definition``);
* **every** definition passes the RDKit-free pre-pass
  (``reference.check_definition_text``) before RDKit sees it, and every build of
  an RDKit factory from corpus text happens in a time- and memory-limited
  subprocess (:func:`perceive`); trust removes the refusal, never the bounds.

This module imports neither RDKit nor ``roshambo3.pharmacophore``: ``show``,
``trust`` and the gate are RDKit-free. RDKit is imported only in the sandbox's
child process.

Public API: :func:`check_definition` (the gate), :func:`perceive`,
:func:`trust` / :func:`untrust` / :func:`list_trusted`, :func:`show`,
:class:`UntrustedDefinitionError`, :class:`DefinitionPerceptionError`.
"""

from __future__ import annotations

import datetime
import difflib
import os
import re
import tempfile
import warnings
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

from .format.r3g import reference
from .format.r3g.errors import CorpusValidationError

__all__ = ["UntrustedDefinitionError", "DefinitionPerceptionError", "TrustFileWarning",
           "DefinitionVerdict", "Perception", "check_definition", "check_corpus_color",
           "perceive", "trust",
           "untrust", "list_trusted", "trust_file_path", "show", "diff_against_shipped",
           "definition_hash_of", "shipped_definitions", "is_shipped", "escape_control",
           "MIN_PREFIX", "TrustFileError", "shipped_text", "declared_weights",
           "unequal_weights"]

#: The shortest hash prefix accepted anywhere (hex characters).
MIN_PREFIX = 16
_HEX = re.compile(r"^[0-9a-f]+$")
_FULL = re.compile(r"^[0-9a-f]{64}$")
_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "definitions")


class UntrustedDefinitionError(ValueError):
    """The corpus's color definition is custom and not trusted.

    ``str(e)`` is the documented refusal text. Attributes carry its fields:
    ``corpus`` (path or None), ``hash`` (full SHA-256), ``families``,
    ``differs`` (one line against the nearest shipped default), ``size_bytes``,
    ``n_patterns``, ``within_limits`` and ``limit`` (None while within limits)."""

    def __init__(self, *, corpus, hash, families, differs, size_bytes, n_patterns,
                 within_limits=True, limit=None):
        self.corpus, self.hash, self.families = corpus, hash, list(families)
        self.differs, self.size_bytes, self.n_patterns = differs, size_bytes, n_patterns
        self.within_limits, self.limit = within_limits, limit
        super().__init__(_refusal_text(self))


class DefinitionPerceptionError(ValueError):
    """Perception in the limited subprocess failed. ``limit`` names the bound
    that was hit (``"cpu"``, ``"wall"``, ``"memory"``) or is None (the child
    crashed or RDKit rejected the text)."""

    def __init__(self, message, *, limit=None, seconds=0.0, peak_rss_bytes=None):
        super().__init__(message)
        self.limit, self.seconds, self.peak_rss_bytes = limit, seconds, peak_rss_bytes


class TrustFileError(ValueError):
    """The trust file or its directory cannot be used (unreadable, not a regular
    file, not owned by this user, or writable by group or others). Trust is a
    security decision, so an insecure store is refused, not read."""


class TrustFileWarning(UserWarning):
    """A line of the trust file is malformed; it was skipped, never trusted."""


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def escape_control(text: str, keep: str = "\n\t") -> str:
    """Escape everything not printable (RDKit echoes raw bytes; a definition can
    carry an ESC). ``keep`` characters pass through."""
    out = []
    for c in text:
        if c in keep or (c.isprintable()):
            out.append(c)
        elif ord(c) < 0x100:
            out.append(f"\\x{ord(c):02x}")
        elif ord(c) < 0x10000:
            out.append(f"\\u{ord(c):04x}")
        else:
            out.append(f"\\U{ord(c):08x}")
    return "".join(out)


def definition_hash_of(color: dict) -> str:
    """SHA-256 of the corpus color's definition (FORMAT.md section 10.3)."""
    return reference.definition_hash(color["definition"])


def _definition_text(color: dict) -> str:
    """The text ``show`` prints: the fdef, or the canonical JSON of a SMARTS list."""
    if color["generator"] == "rdkit_fdef":
        return reference.normalize_definition(color["definition"])
    return reference.canonical_json(color["definition"]).decode("utf-8")


# ---------------------------------------------------------------------------
# Shipped defaults and the allow-list
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Shipped:
    version: str
    hash: str
    text: str
    date_added: str


_shipped_cache: Optional[list] = None


def shipped_definitions() -> list:
    """The shipped ``BaseFeatures.fdef`` texts, oldest RDKit first, each read from
    package data and listed only if its text really hashes to its allow-list
    entry (a tampered file is dropped, not trusted)."""
    global _shipped_cache
    if _shipped_cache is None:
        out = []
        try:
            lines = open(os.path.join(_DATA, "allowlist.tsv"), encoding="utf-8").read().splitlines()
        except OSError:
            lines = []
        for line in lines:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 3 or not _FULL.match(parts[0]):
                continue
            h, ver, date = parts[0], parts[1], parts[2]
            try:
                with open(os.path.join(_DATA, ver, "BaseFeatures.fdef"), "rb") as f:
                    text = f.read().decode("utf-8")
            except (OSError, UnicodeError):
                continue
            if reference.definition_hash(text) == h:
                out.append(Shipped(ver, h, text, date))
        out.sort(key=lambda s: _version_key(s.version))
        _shipped_cache = out
    return list(_shipped_cache)


def _version_key(v: str):
    return tuple(int(x) if x.isdigit() else 0 for x in re.split(r"[.]", v))


def is_shipped(h: str) -> bool:
    return any(s.hash == h for s in shipped_definitions())


def shipped_text(h: str) -> Optional[str]:
    """The packaged text whose definition hash is ``h`` (verified when loaded), or None."""
    for s in shipped_definitions():
        if s.hash == h:
            return s.text
    return None


# ---------------------------------------------------------------------------
# The trust store
# ---------------------------------------------------------------------------

def trust_file_path() -> str:
    """``$XDG_CONFIG_HOME/roshambo3/trusted-definitions``, default
    ``~/.config/roshambo3/trusted-definitions``."""
    base = os.environ.get("XDG_CONFIG_HOME") or os.path.join(os.path.expanduser("~"), ".config")
    return os.path.join(base, "roshambo3", "trusted-definitions")


def _check_trust_path(path: str) -> None:
    """Refuse a trust file (or its directory) that is not a regular file/directory
    owned by this user and free of group/world write permission."""
    import stat
    for p, want_dir in ((os.path.dirname(path), True), (path, False)):
        try:
            st = os.stat(p)
        except FileNotFoundError:
            continue
        except OSError as e:
            raise TrustFileError(f"cannot use trust store {p}: {e.strerror or e}") from None
        if want_dir != stat.S_ISDIR(st.st_mode) or (not want_dir and not stat.S_ISREG(st.st_mode)):
            raise TrustFileError(f"trust store {p} is not a regular "
                                 f"{'directory' if want_dir else 'file'}")
        if hasattr(os, "geteuid") and st.st_uid != os.geteuid():
            raise TrustFileError(f"trust store {p} is not owned by the current user; refusing "
                                 f"to read trust decisions from it")
        if st.st_mode & 0o022:
            raise TrustFileError(f"trust store {p} is writable by group or others "
                                 f"(mode {stat.S_IMODE(st.st_mode):o}); chmod go-w it, "
                                 f"or remove it and re-trust")


def _read_trust_lines(path: str) -> list:
    _check_trust_path(path)
    try:
        with open(path, encoding="utf-8", errors="replace", newline="") as f:
            return f.read().split("\n")
    except FileNotFoundError:
        return []
    except OSError as e:
        raise TrustFileError(f"cannot read trust file {path}: {e.strerror or e}") from None


def _parse_trust_line(line: str):
    """``(hash, date, note)``, or None for blank/comment, or raises ValueError."""
    s = line.rstrip("\r")
    if not s.strip() or s.lstrip().startswith("#"):
        return None
    parts = s.split("\t", 2)
    if len(parts) < 2:
        raise ValueError("expected <sha256> TAB <date> TAB <note>")
    h, date = parts[0], parts[1]
    note = parts[2] if len(parts) > 2 else ""
    if not _FULL.match(h):
        raise ValueError("the first field is not a 64-character lowercase hex sha256")
    try:
        datetime.datetime.fromisoformat(date)
    except ValueError:
        raise ValueError("the second field is not an ISO-8601 date") from None
    return h, date, note


def list_trusted(path: Optional[str] = None, *, report=True) -> dict:
    """``{sha256: (date, note)}`` from the trust file. A malformed line is
    reported (a :class:`TrustFileWarning` naming its line number, when ``report``)
    and skipped; it never trusts anything."""
    path = path or trust_file_path()
    entries: dict = {}
    for n, line in enumerate(_read_trust_lines(path), 1):
        try:
            p = _parse_trust_line(line)
        except ValueError as e:
            if report:
                warnings.warn(f"{path}:{n}: malformed trust entry skipped ({e})",
                              TrustFileWarning, stacklevel=2)
            continue
        if p is not None:
            entries.setdefault(p[0], (p[1], p[2]))
    return entries


def _atomic_write(path: str, text: str) -> None:
    d = os.path.dirname(path)
    try:
        os.makedirs(d, mode=0o700, exist_ok=True)
        _check_trust_path(path)
        fd, tmp = tempfile.mkstemp(prefix=".trusted-definitions.", dir=d)
    except OSError as e:
        raise TrustFileError(f"cannot write trust file {path}: {e.strerror or e}") from None
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
    except BaseException as e:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        if isinstance(e, OSError):
            raise TrustFileError(f"cannot write trust file {path}: {e.strerror or e}") from None
        raise


def _clean_hash(h: str, *, what: str = "hash") -> str:
    h = str(h).strip().lower()
    if not _HEX.match(h):
        raise ValueError(f"{what} {h!r} is not hexadecimal")
    return h


def trust(hash: str, *, note: str = "", path: Optional[str] = None) -> bool:
    """Trust the definition with this **full** SHA-256 from now on. Returns False
    if it was already trusted (the entry is left as it was). Trust names one exact
    text; there is no way to trust "any custom definition"."""
    h = _clean_hash(hash)
    if not _FULL.match(h):
        raise ValueError("trusting needs the full 64-hex sha256 (or a prefix of at least "
                         f"{MIN_PREFIX} hex characters together with the corpus, "
                         "`roshambo3 definitions trust <prefix> --corpus <corpus>`)")
    path = path or trust_file_path()
    if h in list_trusted(path, report=False):
        return False
    note = " ".join(str(note).replace("\t", " ").splitlines())
    lines = _read_trust_lines(path)
    if lines and lines[-1] == "":
        lines.pop()
    date = datetime.date.today().isoformat()
    lines.append(f"{h}\t{date}\t{note}")
    _atomic_write(path, "\n".join(lines) + "\n")
    return True


def untrust(hash: str, *, path: Optional[str] = None) -> str:
    """Remove a trust entry named by its full hash or an unambiguous prefix of at
    least :data:`MIN_PREFIX` characters of an entry in the file; returns the full
    hash removed. Raises ValueError if there is none or it is ambiguous."""
    h = _clean_hash(hash)
    path = path or trust_file_path()
    if len(h) < MIN_PREFIX:
        raise ValueError(f"a hash prefix needs at least {MIN_PREFIX} hex characters")
    have = [e for e in list_trusted(path, report=False) if e.startswith(h)]
    if not have:
        raise ValueError(f"no trusted definition starts with {h}")
    if len(have) > 1:
        raise ValueError(f"{h} is ambiguous: it starts {len(have)} trusted definitions")
    full = have[0]
    keep = []
    for line in _read_trust_lines(path):
        try:
            p = _parse_trust_line(line)
        except ValueError:
            p = None
        if p is not None and p[0] == full:
            continue
        keep.append(line)
    while keep and keep[-1] == "":
        keep.pop()
    _atomic_write(path, "\n".join(keep) + ("\n" if keep else ""))
    return full


def resolve_prefix(prefix: str, color: Optional[dict]) -> str:
    """Resolve a full hash or a prefix (>= :data:`MIN_PREFIX` hex) against one
    corpus's definition hash; ValueError when it does not name it, or is too short."""
    p = _clean_hash(prefix)
    if _FULL.match(p):
        return p
    if len(p) < MIN_PREFIX:
        raise ValueError(f"a hash prefix needs at least {MIN_PREFIX} hex characters "
                         f"(got {len(p)})")
    if color is None:
        raise ValueError("a hash prefix needs the corpus it is a prefix of (--corpus)")
    h = definition_hash_of(color)
    if not h.startswith(p):
        raise ValueError(f"{p} is not a prefix of this corpus's definition hash "
                         f"({h[:MIN_PREFIX]}...)")
    return h


# ---------------------------------------------------------------------------
# Comparing a definition with the shipped defaults (RDKit-free)
# ---------------------------------------------------------------------------

def _parse_fdef_patterns(text: str) -> dict:
    """``{family: [expanded pattern, ...]}`` of the ``DefineFeature`` blocks of an
    fdef that already passed the pre-pass (so every expansion is bounded)."""
    pos: dict = {}
    neg: dict = {}

    def expand(p):
        out, i = [], 0
        while i < len(p):
            if p[i] == "{":
                j = p.find("}", i + 1)
                name = p[i + 1:j]
                out.append(reference.expand_macro(pos.get(name, []), neg.get(name, [])))
                i = j + 1
            else:
                out.append(p[i])
                i += 1
        return "".join(out)

    fams: dict = {}
    pending = None
    for line in text.split("\n"):
        parts = line.split()
        if not parts or parts[0].startswith("#"):
            continue
        kw = parts[0].lower()
        if kw == "atomtype":
            name = parts[1]
            isneg = name.startswith("!")
            (neg if isneg else pos).setdefault(name[1:] if isneg else name, []).append(expand(parts[2]))
        elif kw == "definefeature":
            pending = expand(parts[2])
        elif kw == "family" and pending is not None:
            fams.setdefault(parts[1], []).append(pending)
            pending = None
    return fams


def declared_weights(text: str) -> list:
    """``[(feature name, families, weights text or None), ...]``, one per
    ``DefineFeature`` block of an fdef, in file order: what each block
    declares on its ``Weights`` line (``None`` when it has none). Keywords are
    case-insensitive whole tokens, as the pre-pass and RDKit read them; the
    weights text is the rest of the line with whitespace removed (RDKit splits
    it on commas)."""
    blocks: list = []
    current = None
    for line in reference.normalize_definition(text).split("\n"):
        parts = line.split()
        if not parts or parts[0].startswith("#"):
            continue
        kw = parts[0].lower()
        if kw == "definefeature":
            current = [parts[1] if len(parts) > 1 else "", [], None]
        elif current is not None and kw == "family" and len(parts) > 1:
            current[1].append(parts[1])
        elif current is not None and kw == "weights":
            current[2] = "".join(parts[1:])
        elif current is not None and kw == "endfeature":
            blocks.append((current[0], tuple(current[1]), current[2]))
            current = None
    return blocks


def unequal_weights(color: dict) -> list:
    """The ``DefineFeature`` blocks of a selected family whose weights format
    3 cannot represent, as ``[(family, feature name, weights text or None)]``
    (empty when the definition is representable).

    A feature's position is the **equal-weight** mean of its atoms
    (``positioning = "centroid"``, FORMAT.md sections 8.3 and 10.2); no weights
    are stored, so a definition whose blocks for a family in ``color.types``
    declare anything but all-ones weights is refused at build.
    A block with **no** ``Weights`` line is refused too (``None`` in the
    result): RDKit cannot place such a feature at query
    time. Families that are not selected are not checked (``BaseFeatures``'
    ``ZnBinder`` weights). A ``smarts_list`` definition has no weights.
    """
    if color.get("generator") != "rdkit_fdef":
        return []
    selected = set(color.get("types") or ())
    out = []
    for name, families, weights in declared_weights(color["definition"]):
        hit = [f for f in families if f in selected]
        if not hit:
            continue
        if weights is None:
            out.extend((family, name, None) for family in hit)
            continue
        try:
            values = [float(w) for w in weights.split(",")]
        except ValueError:
            values = None
        if values is None or not values or any(v != 1.0 for v in values):
            out.extend((family, name, weights) for family in hit)
    return out


def _patterns_of(color: dict) -> dict:
    if color["generator"] == "rdkit_fdef":
        return _parse_fdef_patterns(reference.normalize_definition(color["definition"]))
    return {f: list(p) for f, p in color["definition"].items()}


def _nearest_shipped(color: dict) -> Optional[Shipped]:
    ships = shipped_definitions()
    if not ships:
        return None
    if color["generator"] != "rdkit_fdef":
        return ships[-1]
    mine = reference.normalize_definition(color["definition"]).splitlines()
    best = None
    for s in ships:                                   # oldest first; ties go to the newest
        theirs = s.text.splitlines()
        changed = sum(1 for d in difflib.ndiff(theirs, mine) if d[:1] in "+-")
        if best is None or changed <= best[0]:
            best = (changed, s)
    return best[1]


def _differs_line(color: dict, ship: Optional[Shipped]) -> str:
    """One line: families added/removed, then changed patterns per family."""
    if ship is None:
        return "no shipped default to compare with"
    mine = _patterns_of(color)
    theirs = _parse_fdef_patterns(ship.text)
    added = sorted(set(mine) - set(theirs))
    removed = sorted(set(theirs) - set(mine))
    bits = []
    if added:
        bits.append("adds " + ", ".join(f'family "{escape_control(f)}"' for f in added))
    if removed:
        bits.append("removes " + ", ".join(f'family "{escape_control(f)}"' for f in removed))
    changes = []
    for fam in sorted(set(mine) & set(theirs)):
        a, b = list(theirs[fam]), list(mine[fam])
        only_a = [x for x in a if x not in b]
        only_b = [x for x in b if x not in a]
        n = max(len(only_a), len(only_b))
        if n:
            changes.append(f"{escape_control(fam)} ({n} pattern{'s' if n != 1 else ''})")
    if changes:
        bits.append("changes " + ", ".join(changes))
    if not bits:
        bits.append(f"same families and patterns as RDKit {ship.version}'s BaseFeatures; "
                    f"differs only in other text")
    return "; ".join(bits)


def diff_against_shipped(color: dict) -> tuple:
    """``(shipped version or None, unified-diff lines)`` against the nearest
    shipped default (fewest changed lines; ties to the newest RDKit)."""
    ship = _nearest_shipped(color)
    if ship is None:
        return None, []
    mine = _definition_text(color).splitlines(keepends=True)
    theirs = ship.text.splitlines(keepends=True)
    d = difflib.unified_diff(theirs, mine, fromfile=f"BaseFeatures.fdef (RDKit {ship.version})",
                             tofile="corpus definition")
    return ship.version, list(d)


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DefinitionVerdict:
    """What :func:`check_definition` decided. ``status`` is ``"shipped"`` or
    ``"trusted"``; ``in_process`` says the definition is allow-listed, so a caller
    that wants the fast path may build a factory in its own process (after the
    pre-pass this gate already ran). Trusted custom definitions must go through
    :func:`perceive`."""
    hash: str
    status: str
    in_process: bool
    summary: dict = field(default_factory=dict)


def _size_bytes(color: dict) -> int:
    return len(_definition_text(color).encode("utf-8"))


def _n_patterns(summary: dict) -> int:
    return int(summary.get("definefeature_lines", summary.get("patterns", 0)))


def _refusal_text(e: UntrustedDefinitionError) -> str:
    corpus = e.corpus or "<corpus>"
    h = e.hash
    h12 = h[:MIN_PREFIX]
    size = f"{e.size_bytes / 1024:.1f} KB, {e.n_patterns} patterns  " \
           f"({'within limits' if e.within_limits else 'exceeds ' + str(e.limit)})"
    return (
        f"error: {corpus} uses a custom color definition that is not one of RDKit's defaults.\n"
        f"  definition  sha256 {h12}…{h[-4:]}\n"
        f"  families    {', '.join(escape_control(f) for f in e.families)}\n"
        f"  differs     {e.differs}\n"
        f"  size        {size}\n"
        f"\n"
        f"Custom definitions are compiled and run by RDKit on your queries. Vet it before\n"
        f"trusting it:\n"
        f"  roshambo3 definitions show {corpus}            # full text\n"
        f"  roshambo3 definitions trust {h12} --corpus {corpus}   # trust it from now on\n"
        f"or for this run only:\n"
        f"  roshambo3 screen ... --trust-definition {h12}")


def check_definition(color: Optional[dict], *, trusted: Iterable[str] = (),
                     corpus_path: Optional[str] = None,
                     trust_file: Optional[str] = None) -> Optional[DefinitionVerdict]:
    """The gate: may this corpus's color definition be used to featurize queries?

    ``color`` is the corpus's section-10.2 object (``corpus.json["color"]``); a
    shape-only corpus has none (``color=None``) and needs no definition and no
    trust (returns None; the trust file is never read). ``trusted`` are hashes
    from ``--trust-definition``: full hashes, or prefixes of at least 16 hex
    characters, which are matched against this corpus's definition hash (a
    shorter or non-hex entry is refused with ValueError). The per-user trust file
    is consulted as well.

    Order: (1) the RDKit-free pre-pass on every definition, shipped, trusted or
    not -- a failure raises :class:`~roshambo3.format.r3g.errors.CorpusValidationError`
    naming the limit or rule, even if the hash is trusted; (2) shipped
    ``BaseFeatures`` by hash, or a trusted hash, returns a verdict; (3) anything
    else raises :class:`UntrustedDefinitionError`.

    Passing means "may be perceived", not "cheap": every perception must still go
    through :func:`perceive`'s limited subprocess (trust never lifts the bounds).
    """
    if color is None:
        return None
    entries = list(trusted)
    for t in entries:                                  # validate before anything is decided
        p = _clean_hash(t, what="trusted hash")
        if len(p) < MIN_PREFIX:
            raise ValueError(f"--trust-definition needs at least {MIN_PREFIX} hex characters "
                             f"(got {len(p)})")
    summary = reference.check_definition_text(color["definition"], color["generator"],
                                              color.get("types"))
    h = definition_hash_of(color)
    if is_shipped(h) and color["generator"] == "rdkit_fdef":
        return DefinitionVerdict(h, "shipped", True, summary)
    for t in entries:
        if h.startswith(_clean_hash(t)):
            return DefinitionVerdict(h, "trusted", False, summary)
    if h in list_trusted(trust_file):
        return DefinitionVerdict(h, "trusted", False, summary)
    raise UntrustedDefinitionError(
        corpus=corpus_path, hash=h, families=summary["families"] if isinstance(
            summary["families"], list) else sorted(color["definition"]),
        differs=_differs_line(color, _nearest_shipped(color)), size_bytes=_size_bytes(color),
        n_patterns=_n_patterns(summary))


def check_corpus_color(corpus, generator, trusted_definitions: Iterable[str] = (), *,
                       query_features_enum: Optional[dict] = None):
    """Check a color query against the corpus descriptor and definition trust.

    Format 3 carries the complete semantic descriptor, so its query must come
    with the generator that produced its features. We compare those descriptors
    before asking :func:`check_definition` to authorize the corpus definition.
    A type-number map alone cannot establish that the query and library mean
    the same thing.

    Format 2 has no definition descriptor. Until its removal, preserve its
    historical enum check so the format-3 migration does not silently change
    legacy screens. ``query_features_enum`` is only used for that legacy path.
    Shape-only callers should not call this function.
    """
    if not getattr(corpus, "has_color", False):
        raise ValueError("color screening requires a corpus that stores color features")

    # The reader's Corpus exposes its complete descriptor. Do not fall back to
    # the old features_enum comparison if a format-3 call lacks a generator.
    descriptor = getattr(corpus, "descriptor", None)
    if isinstance(descriptor, dict) and "format" in descriptor:
        if generator is None:
            raise ValueError(
                "format-3 color screening requires the color generator that produced the "
                "query features; a features enum alone cannot verify the descriptor")
        describe = getattr(generator, "descriptor_color", None)
        if not callable(describe):
            raise TypeError("color_generator must provide descriptor_color() for format-3 screening")
        from .format.r3g import descriptor as color_descriptor

        color = descriptor.get("color")
        color_descriptor.compare(color, describe())
        return check_definition(
            color, trusted=trusted_definitions,
            corpus_path=getattr(corpus, "root", None))

    enum = (getattr(generator, "FEATURES_ENUM", None)
            if generator is not None else query_features_enum)
    if not enum:
        raise ValueError(
            "legacy color screening requires a color generator or an explicit "
            "query_features_enum")
    corpus.require_color_features_enum(enum)
    return None


# ---------------------------------------------------------------------------
# show (RDKit-free)
# ---------------------------------------------------------------------------

def _corpus_color(corpus_path: str) -> Optional[dict]:
    from .format.r3g.validate import read_descriptor
    return read_descriptor(corpus_path).data.get("color")


def show(corpus_path: str, *, diff: bool = False) -> str:
    """The corpus's stored definition as printable text (control characters
    escaped), or with ``diff=True`` a unified diff against the nearest shipped
    default. Never imports RDKit and never runs the definition."""
    color = _corpus_color(corpus_path)
    if color is None:
        raise ValueError(f"{corpus_path} is shape-only: it stores no color definition")
    if not diff:
        return escape_control(_definition_text(color))
    ver, lines = diff_against_shipped(color)
    if ver is None:
        raise ValueError("no shipped default definition to compare with")
    if not lines:
        return f"identical to RDKit {ver}'s BaseFeatures.fdef\n"
    return escape_control("".join(l if l.endswith("\n") else l + "\n" for l in lines))


# ---------------------------------------------------------------------------
# Bounded perception
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Perception:
    """One molecule's result. ``features`` are ``(type id, atom indices)`` with
    ids ``1..n_types`` and indices into the molecule **with hydrogens added**
    (``n_atoms`` of them; the first ``n_heavy`` are the heavy atoms)."""
    parsed: bool
    n_heavy: int
    n_atoms: int
    features: Optional[list]


_factory_cache: dict = {}


def _shipped_factory(h: str):
    """RDKit factory built from the **packaged** text of a verified-shipped
    definition (never from the corpus's own copy)."""
    f = _factory_cache.get(h)
    if f is None:
        from rdkit.Chem import ChemicalFeatures
        f = _factory_cache[h] = ChemicalFeatures.BuildFeatureFactoryFromString(shipped_text(h))
    return f


def _perceive_in_process(color: dict, items: list, kind: str) -> list:
    """Same contract and caps as the sandbox child, for verified-shipped fdefs only."""
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    h = definition_hash_of(color)
    factory = _shipped_factory(h)
    tid = {t: i + 1 for i, t in enumerate(color["types"])}
    params = Chem.SmilesParserParams()
    params.allowCXSMILES = False
    L = reference.LIMITS
    out = []
    for text in items:
        mol = (Chem.MolFromMolBlock(text, sanitize=True, removeHs=False) if kind == "molblock"
               else Chem.MolFromSmiles(text, params))
        if mol is None:
            out.append(Perception(False, 0, 0, None))
            continue
        n = (sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() != 1) if kind == "molblock"
             else mol.GetNumAtoms())
        mh = Chem.AddHs(mol)
        feats, per = [], {}
        for f in factory.GetFeaturesForMol(mh):
            fam = f.GetFamily()
            if fam in tid:
                atoms = tuple(int(a) for a in f.GetAtomIds())
                per[tid[fam]] = per.get(tid[fam], 0) + 1
                if (len(atoms) > L["atoms_per_feature"] or per[tid[fam]] > L["features_per_type"]
                        or len(feats) >= L["features_per_molecule"]):
                    raise DefinitionPerceptionError(
                        "perception stopped: a molecule exceeds the feature limits "
                        "(255 features, 16 atoms per feature)", limit="features")
                feats.append((tid[fam], atoms))
        out.append(Perception(True, n, mh.GetNumAtoms(), feats))
    return out


def perceive(color: dict, smiles: Optional[Sequence[str]] = None, *,
             molblocks: Optional[Sequence[str]] = None, limits=None,
             timeout: Optional[float] = None, check: bool = True,
             trusted: Iterable[str] = (), corpus_path: Optional[str] = None,
             in_process: Optional[bool] = None) -> list:
    """Build the definition's factory and perceive each molecule, the one place
    corpus definition text becomes an RDKit factory for query featurization.

    **Input** is exactly one of ``smiles`` (parsed with CXSMILES off; atom indices
    follow the SMILES order, which is what the ``deep`` tier checks) or
    ``molblocks`` (MDL molfile text, for query featurization): a molblock is read
    with ``removeHs=False`` so **its atom order and explicit hydrogens are
    preserved**, and the feature table comes back in that order.

    **Output** is one :class:`Perception` per input: ``features`` are
    ``(type id, atom indices)`` -- ids ``1..n_types``, indices into the input's
    atoms, ``Chem.AddHs`` having been applied first (explicit
    hydrogens keep their indices, any further implicit ones are appended after
    them, so every index is below ``n_atoms``). **No positions are returned**:
    the caller computes them from its own coordinates (the equal-weight centroid
    of a feature's atoms, FORMAT.md section 8.3); a caller that wants every
    hydrogen's coordinates must supply them explicitly in the molblock.
    ``n_heavy`` is the SMILES' atom count, or a molblock's non-hydrogen count.
    A molecule that does not parse yields ``Perception(False, 0, 0, None)``.

    **Where it runs.** A verified-shipped definition (hash on the allow-list, the
    factory built from the packaged file) runs in this process. Anything else
    runs in the limited subprocess (``RLIMIT_AS`` 2 GiB, ``RLIMIT_CPU`` and wall
    budgets that scale with the batch, 250 molecules per child; pass
    ``limits=roshambo3.format.r3g._sandbox.Limits(...)`` to change them): no CUDA
    context, no GPU lock, only validated integers return, and the pre-pass runs
    inside ``run_child``. ``in_process=False`` forces the subprocess,
    ``in_process=True`` demands the shipped fast path (ValueError otherwise).
    With ``check=True`` the gate :func:`check_definition` runs first (``trusted``,
    ``corpus_path`` as there); pass ``check=False`` only when the caller already
    ran it. Raises :class:`DefinitionPerceptionError` (``.limit`` is "cpu",
    "wall", "memory", "features" or "reply" when a bound was hit); the child is
    always reaped.
    """
    from .format.r3g import _sandbox

    if (smiles is None) == (molblocks is None):
        raise ValueError("give exactly one of smiles= and molblocks=")
    kind = "smiles" if smiles is not None else "molblock"
    items = list(smiles if smiles is not None else molblocks)
    if check:
        check_definition(color, trusted=trusted, corpus_path=corpus_path)
    h = definition_hash_of(color)
    shipped = color["generator"] == "rdkit_fdef" and is_shipped(h)
    if in_process and not shipped:
        raise ValueError("in_process=True is only for a shipped RDKit definition")
    if shipped and in_process is not False:
        reference.check_definition_text(color["definition"], color["generator"], color.get("types"))
        return _perceive_in_process(color, items, kind)
    out: list = []
    for i in range(0, len(items), _sandbox.BATCH):
        chunk = items[i:i + _sandbox.BATCH]
        req = {"perceive_any": True, "generator": color["generator"], "input": kind,
               "definition": color["definition"], "types": list(color["types"]),
               "molecules": [[s, True] for s in chunk]}
        try:
            res = _sandbox.run_child(req, timeout=timeout, limits=limits)
        except _sandbox.SandboxFailure as e:
            what = f" ({e.limit} limit)" if e.limit else ""
            raise DefinitionPerceptionError(
                f"color definition {h[:12]}: perception in the limited "
                f"subprocess failed{what}: {e}", limit=e.limit, seconds=e.seconds,
                peak_rss_bytes=e.peak_rss_bytes) from None
        for ok, n, feats, nh in res:
            out.append(Perception(bool(ok), n, nh,
                                  None if feats is None else [(t, tuple(a)) for t, a in feats]))
    return out
