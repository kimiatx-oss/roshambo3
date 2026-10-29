"""`cpu.score_pairs` passes the combo scorer's positional arguments in order.

`optimize_scalar_combo` takes 44 positional arguments, many of them booleans,
so two swapped switches would type-check, run and shift no score this suite
can see. This reads the Python caller and the pybind declaration and requires
every argument that names a parameter to sit in that parameter's slot.
Generalized from a composed-CPU test that pinned specific slots.
"""
import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CPU_SRC = ROOT / "src/roshambo3/backend/_cpu_src"


def _declared_names():
    native = (CPU_SRC / "color_scalar.cpp").read_text()
    declaration = native.split('module.def("optimize_scalar_combo",', 1)[1]
    declaration = declaration.split(");", 1)[0]
    return re.findall(r'py::arg\("([^"]+)"\)', declaration)


def _python_arguments():
    tree = ast.parse((ROOT / "src/roshambo3/backend/cpu.py").read_text())
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute)
             and node.func.attr == "optimize_scalar_combo"]
    assert len(calls) == 1, "expected one native combo call in cpu.py"
    return [ast.unparse(arg) for arg in calls[0].args]


def test_every_named_argument_sits_in_its_own_slot():
    names = _declared_names()
    args = _python_arguments()
    assert len(names) == len(args), (len(names), len(args))
    misplaced = []
    for slot, (name, expr) in enumerate(zip(names, args)):
        mentioned = set(re.findall(r"[A-Za-z_]\w*", expr)) & set(names)
        if mentioned and name not in mentioned:
            misplaced.append((slot, name, expr))
    assert not misplaced, f"argument expressions in another slot's place: {misplaced}"


def test_the_check_would_see_a_swap():
    """The instrument, checked: swapping two named switches is reported."""
    names = _declared_names()
    i, j = names.index("use_color_invariant_plan"), names.index(
        "reuse_exact_final_scratch")
    args = _python_arguments()
    args[i], args[j] = args[j], args[i]
    misplaced = [slot for slot, (name, expr) in enumerate(zip(names, args))
                 if (set(re.findall(r"[A-Za-z_]\w*", expr)) & set(names))
                 and name not in re.findall(r"[A-Za-z_]\w*", expr)]
    assert misplaced == sorted((i, j))
