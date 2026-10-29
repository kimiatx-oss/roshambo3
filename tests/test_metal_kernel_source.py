"""The color-field entry points metal.py generates from kernels.metal.

`_color_kernel_source` derives one `optimize_color_fields_m{mask}` kernel per
plane subset by substituting into the text of `optimize_diagonal_factored`. A
substitution that stops matching leaves a kernel that compiles but binds its
buffers at the wrong indices -- which the reformat of kernels.metal once did.
These checks need no Metal device, so they run on every host.
"""
import re

from roshambo3.backend import metal


def _generated(planes=(1, 2)):
    source = metal._SOURCE.read_text()
    return source, metal._color_kernel_source(source, planes)


def test_every_plane_subset_gets_a_field_kernel_with_its_bindings():
    source, full = _generated()
    added = full[len(source):]
    for mask in (2, 4, 6):
        start = added.index(f"kernel void optimize_color_fields_m{mask}(")
        end = added.find("kernel void optimize_color_fields_m", start + 1)
        body = added[start:end if end >= 0 else None]
        signature = re.sub(r"\s+", "", body.split("{", 1)[0])
        assert ("devicefloat*out,deviceconstGrid*color_grids,"
                "deviceconstfloat4*color_field,deviceconstuint*color_slots,"
                "constantParams&p,") in signature
        call = re.sub(r"\s+", "", body.split("{", 1)[1])
        assert f"optimize_body<true,true,{mask}>(" in call
        assert "out,color_grids,color_field,color_slots,p," in call
    assert "optimize_color_fields_m0(" not in added


def test_a_substitution_that_no_longer_matches_is_an_error():
    source = metal._SOURCE.read_text()
    broken = source.replace("optimize_body<", "optimize_body_renamed<")
    try:
        metal._color_kernel_source(broken, (1,))
    except RuntimeError as exc:
        assert "optimize_body<true,true>" in str(exc)
    else:
        raise AssertionError("a non-matching substitution was ignored")
