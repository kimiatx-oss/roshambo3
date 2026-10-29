"""The convention linter must actually catch things.

A linter that passes vacuously is worse than none: it reports OK while the rule
it claims to enforce is being broken. These tests feed it known-bad input.
"""

from __future__ import annotations

import importlib.util
import pathlib
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "lint_conventions.py"


@pytest.fixture(scope="module")
def lc():
    spec = importlib.util.spec_from_file_location("lint_conventions", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _scan(lc, tmp_path, name, text):
    """Run the linter's per-file logic over one synthetic file.

    Note the ALLOW check is against the RAW line: the marker lives in a comment,
    and `code` has had its comments blanked. Doing it the other way ignores every
    marker, and this helper made exactly that mistake at first -- and passed,
    because its fixture put the identifier inside a string literal, which is
    blanked regardless.
    """
    p = tmp_path / name
    p.write_text(text)
    code = lc.code_only_python(p) if p.suffix == ".py" else lc.code_only_c(p)
    raw = text.splitlines()
    ids = [m.group(0) for lineno, line in code
           if lc.ALLOW not in raw[lineno - 1]
           for m in lc.TICKET_ID.finditer(line)]
    words = [m.group(1) for line in text.splitlines()
             if lc.ALLOW not in line
             for m in lc.BRITISH_RE.finditer(line)]
    return ids, words


class TestTicketIds:
    def test_catches_them_in_python_code(self, lc, tmp_path):
        ids, _ = _scan(lc, tmp_path, "a.py",
                       "T36GridGeom = 1\ndef t44_topk_create(): ...\nx_t27 = 2\n")
        assert set(ids) == {"T36GridGeom", "t44_topk_create", "x_t27"}

    def test_catches_them_in_cuda_code(self, lc, tmp_path):
        ids, _ = _scan(lc, tmp_path, "a.cu",
                       "struct T36GridGeom { int x; };\n__global__ void t44_k_compact(){}\n")
        assert set(ids) == {"T36GridGeom", "t44_k_compact"}

    def test_catches_them_inside_macro_names(self, lc, tmp_path):
        ids, _ = _scan(lc, tmp_path, "a.cpp",
                       "#ifndef R3_T238_POLICY\n#define R3_T238_POLICY 0\n#endif\n")
        assert ids == ["R3_T238_POLICY", "R3_T238_POLICY"]

    def test_exempts_comments_and_strings(self, lc, tmp_path):
        """Provenance belongs in prose, so prose must not trip the check."""
        ids, _ = _scan(lc, tmp_path, "a.py",
                       '# See T36 / results/T36 for the measurement\n'
                       'name = "T44TopK"\n'
                       '"""Ported from T20."""\n')
        assert ids == []

    def test_exempts_cuda_comments_and_strings(self, lc, tmp_path):
        ids, _ = _scan(lc, tmp_path, "a.cu",
                       '// See T38 for the residency measurement\n'
                       '/* T27 staging */\n'
                       'const char *s = "T44";\n')
        assert ids == []

    def test_a_trailing_comment_does_not_hide_the_code_before_it(self, lc, tmp_path):
        """The bug this guards: blanking a whole line because a comment appears
        on it erases the code in front of the comment."""
        ids, _ = _scan(lc, tmp_path, "a.py", "T36GridGeom = 1  # a trailing comment\n")
        assert ids == ["T36GridGeom"]

    def test_fstring_literal_text_is_exempt(self, lc, tmp_path):
        """PEP 701 (3.12+) tokenizes f-strings as FSTRING_MIDDLE rather than
        STRING. Without handling that, prose inside an f-string trips the check
        on 3.12 but not on 3.11 -- the same file passing on one interpreter and
        failing on the other."""
        ids, _ = _scan(lc, tmp_path, "a.py", 'msg = f"measured by T24 on sm_120"\n')
        assert ids == []

    @pytest.mark.skipif(sys.version_info < (3, 12),
                        reason="on 3.11 the whole f-string is one STRING token, "
                               "so expressions inside it are blanked too")
    def test_fstring_expressions_are_still_scanned_on_312(self, lc, tmp_path):
        ids, _ = _scan(lc, tmp_path, "a.py", 'msg = f"{t44_topk_create()}"\n')
        assert ids == ["t44_topk_create"]

    def test_does_not_fire_on_ordinary_type_names(self, lc, tmp_path):
        ids, _ = _scan(lc, tmp_path, "a.cu",
                       "uint32_t a; int32_t b; float32 c; tex3D<float4> t;\n")
        assert ids == []

    def test_the_allow_marker_exempts_a_line(self, lc, tmp_path):
        """The identifier must be in CODE, not a string literal -- a string is
        blanked anyway, so a fixture using one proves nothing about the marker."""
        marked = f"t44_topk_create()  # {lc.ALLOW}\n"
        assert _scan(lc, tmp_path, "a.py", marked)[0] == []
        # ...and without the marker the very same line is caught
        assert _scan(lc, tmp_path, "b.py", "t44_topk_create()\n")[0] == ["t44_topk_create"]


class TestSpelling:
    def test_catches_british_spellings_including_in_comments(self, lc, tmp_path):
        _, w = _scan(lc, tmp_path, "a.py",  # lint-conventions: allow
                     "colour = 1\ndef optimise(): ...\n# quantise the input\n")  # lint-conventions: allow
        assert {x.lower() for x in w} == {"colour", "optimise", "quantise"}  # lint-conventions: allow

    def test_is_case_insensitive(self, lc, tmp_path):
        _, w = _scan(lc, tmp_path, "a.py", "# Colour and Behaviour\n")  # lint-conventions: allow
        assert len(w) == 2

    def test_leaves_american_spellings_alone(self, lc, tmp_path):
        _, w = _scan(lc, tmp_path, "a.py",
                     "color = 1\ndef optimize(): ...\n# quantize, normalize, analyze\n")
        assert w == []


def test_the_repository_itself_passes():
    r = subprocess.run([sys.executable, str(SCRIPT)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
