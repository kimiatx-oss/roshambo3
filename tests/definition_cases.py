"""Adversarial color definitions, shared by ``tests/test_definitions.py`` and
the definition-parsing measurement."""

from __future__ import annotations

import json
import os

CONFORMANCE_COLOR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "data", "format3", "color")
IMATINIB = "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1"


def conformance_color() -> dict:
    with open(os.path.join(CONFORMANCE_COLOR, "corpus.json"), encoding="utf-8") as f:
        return json.load(f)["color"]


def with_definition(text, types=("Hydrophobe",), generator="rdkit_fdef") -> dict:
    c = dict(conformance_color())
    c.update(generator=generator, definition=text, types=list(types))
    return c


def one_feature(pattern: str, family: str = "Hydrophobe", n_atoms: int = 1) -> str:
    w = ",".join(["1.0"] * n_atoms)          # RDKit wants one weight per pattern atom
    return f"DefineFeature F {pattern}\n  Family {family}\n  Weights {w}\nEndFeature\n"


def macro_bomb(n: int = 18) -> str:
    lines = ["AtomType A0 [#6]"] + [f"AtomType A{i} [{{A{i-1}}},{{A{i-1}}}]"
                                    for i in range(1, n + 1)]
    lines += [f"DefineFeature F [{{A{n}}}]", "  Family Hydrophobe", "  Weights 1.0", "EndFeature"]
    return "\n".join(lines) + "\n"


def star_chain(n: int = 18) -> str:
    return one_feature("*" * n, n_atoms=n)


def nested_smarts(depth: int = 100_000) -> str:
    return one_feature("[$(" * depth + "C" + ")]" * depth)


def any_bond_chain(n: int = 16) -> str:
    """Within the pre-pass limits (<= 16 top-level atoms), still seconds per molecule."""
    return one_feature("~".join(["*"] * n), n_atoms=n)


CASES = {
    "macro_bomb_n18": (macro_bomb, "limit.definition_depth"),
    "star_chain_18": (star_chain, "limit.definition_atoms"),
    "nested_smarts_500kb": (nested_smarts, "limit.definition_size"),
}

