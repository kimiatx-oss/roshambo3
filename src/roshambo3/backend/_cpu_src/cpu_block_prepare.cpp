// Decoded format-3 block -> the FP32 coordinate array the native CPU scorer reads.
//
// In the worker-owned scan (Engine.
// run_worker_owned_cpu) each worker decodes its own compressed library block
// and calls this once per block, before scoring. The int16 -> FP32 conversion
// is a required layout copy, not transport overhead, so it is done once here, with the GIL
// released, rather than per pose in the kernels.
//
// Input layout (the decoded block of doc/FORMAT.md section 4.3, as produced by
// the format-3 reader): `shape_xyz` is [rows, 3, L] -- component-
// major, atoms contiguous -- and `counts[row]` is [n_real, n_type1, ...].
// Color comes in one of two layouts:
//   merged: features sit in the shape block right after each row's real atoms
//           (pads to max(n_real + n_color), not to two worst cases);
//   split:  a separate [rows, 3, L_color] block, features from column 0.
// Output is [rows, output_width, 3], atom-major xyz, with each row's color
// features placed immediately after its real atoms in both cases -- the
// count-based view (library_color_offset < 0) the kernels already read.
#include "cpu_block_prepare.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace py = pybind11;

namespace r3 {
namespace cpu {
namespace {

// `coord_scale` is the block's wire quantization step; the result is
// physical coordinates. Output width must equal the physical width (shape-only
// or merged) or shape + color width (split). Zero-filled first, so slots past
// a row's real atoms and features are exact zeros.
py::array_t<float, py::array::c_style> prepare_cpu_block_coords(
    py::array_t<std::int16_t, py::array::c_style> shape_xyz,
    py::array_t<std::uint8_t, py::array::c_style> counts,
    std::size_t output_width, float coord_scale,
    const py::object& split_color_xyz, bool merged_color) {
    if (shape_xyz.ndim() != 3 || shape_xyz.shape(1) != 3 ||
        counts.ndim() != 2 || shape_xyz.shape(0) != counts.shape(0) ||
        output_width == 0 || !std::isfinite(coord_scale) ||
        !(coord_scale > 0.0f)) {
        throw std::invalid_argument(
            "CPU block prep expects int16 [rows,3,L], matching uint8 counts, "
            "a positive output width and finite coordinate scale");
    }
    const std::size_t rows = static_cast<std::size_t>(shape_xyz.shape(0));
    const std::size_t shape_width =
        static_cast<std::size_t>(shape_xyz.shape(2));
    if (rows == 0 || shape_width == 0 || counts.shape(1) < 1) {
        throw std::invalid_argument(
            "CPU block prep does not accept empty blocks");
    }

    py::array_t<std::int16_t, py::array::c_style> color_xyz;
    std::size_t color_width = 0;
    if (merged_color && !split_color_xyz.is_none()) {
        throw std::invalid_argument(
            "merged CPU block cannot also have a split color block");
    }
    if (merged_color) {
        color_width = shape_width;
        if (output_width != shape_width) {
            throw std::invalid_argument(
                "merged CPU block output width must equal physical width");
        }
    } else if (!split_color_xyz.is_none()) {
        color_xyz = split_color_xyz
                        .cast<py::array_t<std::int16_t, py::array::c_style>>();
        if (color_xyz.ndim() != 3 || color_xyz.shape(0) != shape_xyz.shape(0) ||
            color_xyz.shape(1) != 3 || color_xyz.shape(2) == 0) {
            throw std::invalid_argument(
                "split color block must be int16 [same_rows,3,L_color]");
        }
        color_width = static_cast<std::size_t>(color_xyz.shape(2));
        if (shape_width >
                std::numeric_limits<std::size_t>::max() - color_width ||
            output_width != shape_width + color_width) {
            throw std::invalid_argument(
                "split CPU block output width must equal shape plus color width");
        }
    } else if (output_width != shape_width) {
        throw std::invalid_argument(
            "non-split CPU block output width must equal its physical width");
    }

    // Overflow guards on rows * width * 3 before allocation.
    if (output_width >
            static_cast<std::size_t>(std::numeric_limits<py::ssize_t>::max()) ||
        rows > std::numeric_limits<std::size_t>::max() / output_width ||
        rows * output_width > std::numeric_limits<std::size_t>::max() / 3) {
        throw std::length_error("CPU block FP32 output dimensions overflow");
    }
    const std::size_t output_values = rows * output_width * 3;
    if (output_values >
        static_cast<std::size_t>(std::numeric_limits<py::ssize_t>::max())) {
        throw std::length_error("CPU block FP32 output is too large for NumPy");
    }

    py::array_t<float, py::array::c_style> output(
        {static_cast<py::ssize_t>(rows), static_cast<py::ssize_t>(output_width),
         py::ssize_t(3)});
    const std::int16_t* shape = shape_xyz.data();
    const std::int16_t* color =
        merged_color ? shape : (color_width ? color_xyz.data() : nullptr);
    const std::uint8_t* count_data = counts.data();
    const std::size_t n_features = static_cast<std::size_t>(counts.shape(1));
    const float scale = coord_scale;
    float* out = output.mutable_data();
    {
        py::gil_scoped_release nogil;
        std::fill_n(out, output_values, 0.0f);
        for (std::size_t row = 0; row < rows; ++row) {
            const std::uint8_t* count_row = count_data + row * n_features;
            const std::size_t n_real = count_row[0];
            if (n_real > shape_width) {
                throw std::invalid_argument(
                    "n_real exceeds physical shape block width");
            }
            // Shape-only block: every slot, padding included, is converted, so
            // the result matches converting the whole array in numpy.
            if (!color_width) {
                for (std::size_t atom = 0; atom < shape_width; ++atom) {
                    const std::size_t destination =
                        (row * output_width + atom) * 3;
                    for (std::size_t component = 0; component < 3;
                         ++component) {
                        const std::int16_t wire =
                            shape[(row * 3 + component) * shape_width + atom];
                        // int16 values convert exactly to FP32; the following
                        // FP32 multiplication is the same two-step rounding as
                        // numpy's astype(float32) * float32(WIRE_QUANT_STEP).
                        out[destination + component] =
                            static_cast<float>(wire) * scale;
                    }
                }
                continue;
            }

            // Color block: copy n_real atoms, then n_color features packed
            // behind them; the remainder of the row stays zero.
            std::size_t n_color = 0;
            for (std::size_t feature = 1; feature < n_features; ++feature)
                n_color += count_row[feature];
            if (n_color > color_width || n_real > output_width ||
                n_color > output_width - n_real) {
                throw std::invalid_argument(
                    "color counts exceed their physical block widths");
            }
            for (std::size_t atom = 0; atom < n_real; ++atom) {
                const std::size_t destination = (row * output_width + atom) * 3;
                for (std::size_t component = 0; component < 3; ++component) {
                    const std::int16_t wire =
                        shape[(row * 3 + component) * shape_width + atom];
                    out[destination + component] =
                        static_cast<float>(wire) * scale;
                }
            }
            for (std::size_t atom = 0; atom < n_color; ++atom) {
                const std::size_t destination =
                    (row * output_width + n_real + atom) * 3;
                for (std::size_t component = 0; component < 3; ++component) {
                    // Merged features follow the real atoms in the source too;
                    // split features start at column 0 of their own block.
                    const std::size_t source_atom =
                        merged_color ? n_real + atom : atom;
                    const std::int16_t wire =
                        color[(row * 3 + component) * color_width +
                              source_atom];
                    out[destination + component] =
                        static_cast<float>(wire) * scale;
                }
            }
        }
    }
    return output;
}

}  // namespace

void bind_cpu_block_prepare_api(py::module_& module) {
    module.def(
        "prepare_cpu_block_coords", &prepare_cpu_block_coords,
        py::arg("shape_xyz_i16"), py::arg("counts"), py::arg("output_width"),
        py::arg("coord_scale"), py::arg("split_color_xyz_i16") = py::none(),
        py::arg("merged_color") = false,
        "Convert one worker-owned int16 physical block to contiguous FP32 "
        "coordinates; split or merged color is placed immediately after each "
        "row's real atoms for the native count-based CPU view. The conversion "
        "and FP32 wire-scale multiply occur once.");
}

}
}  // namespace r3::cpu
