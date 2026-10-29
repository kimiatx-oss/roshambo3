// pybind11 layer of the format-3 coordinate codec (r3g_codec.hpp); the
// Python-facing interface is roshambo3/format/r3g/native.pyi.
#pragma once

#include <pybind11/pybind11.h>

namespace r3 {
namespace r3g {

void bind_r3g_codec_api(pybind11::module_& module);

}  // namespace r3g
}  // namespace r3
