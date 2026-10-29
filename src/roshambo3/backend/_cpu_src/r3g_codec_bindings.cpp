// pybind11 layer of the format-3 coordinate codec. Thin by design: it turns
// buffers into (pointer, length) pairs while holding the GIL, releases the GIL
// around the core call (the cpu_block_prepare.cpp pattern), and converts a
// non-ok Status into roshambo3.format.r3g.native.CodecError (a ValueError)
// whose message starts with the failed check's code. Outputs are written in
// place into caller-provided buffers; the Python wrapper (native.py)
// allocates the small per-row ones a caller did not pass.
#include "r3g_codec_bindings.hpp"

#include "r3g_codec.hpp"

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <chrono>
#include <cmath>
#include <cstring>
#include <string>
#include <tuple>
#include <vector>

namespace py = pybind11;

namespace r3 {
namespace r3g {
namespace {

[[noreturn]] void raise_codec(Status s, const std::string& detail = std::string()) {
    std::string msg = std::string(status_code(s)) + ": " + status_message(s);
    if (!detail.empty()) msg += " (" + detail + ")";
    py::object cls = py::module_::import("roshambo3.format.r3g.native").attr("CodecError");
    py::object exc = cls(msg, status_code(s), static_cast<int>(s));
    PyErr_SetObject(cls.ptr(), exc.ptr());
    throw py::error_already_set();
}

bool c_contiguous(const py::buffer_info& b) {
    py::ssize_t expect = b.itemsize;
    for (py::ssize_t d = b.ndim - 1; d >= 0; --d) {
        if (b.shape[d] != 1 && b.strides[d] != expect) return false;
        expect *= b.shape[d];
    }
    return true;
}

// A read-only input: C-contiguous bytes of any item type.
struct In {
    py::buffer_info info;
    const std::uint8_t* ptr = nullptr;
    std::size_t bytes = 0;
};
In input(const py::object& obj, const char* name) {
    In r;
    r.info = py::reinterpret_borrow<py::buffer>(obj).request(false);
    if (!c_contiguous(r.info))
        raise_codec(Status::kBadArgument, std::string(name) + " must be C-contiguous");
    r.ptr = static_cast<const std::uint8_t*>(r.info.ptr);
    r.bytes = static_cast<std::size_t>(r.info.size) * static_cast<std::size_t>(r.info.itemsize);
    return r;
}
// A typed input: item size must match (no alignment required).
In typed_input(const py::object& obj, std::size_t itemsize, const char* name) {
    In r = input(obj, name);
    if (static_cast<std::size_t>(r.info.itemsize) != itemsize)
        raise_codec(Status::kBadArgument, std::string(name) + " has the wrong item size");
    return r;
}

// A writeable output with an exact numpy dtype, C-contiguous and aligned.
template <class T>
struct Out {
    py::buffer_info info;
    T* ptr = nullptr;
    std::size_t n = 0;
};
template <class T>
Out<T> output(const py::object& obj, const char* name) {
    Out<T> r;
    if (obj.is_none()) return r;
    auto arr = py::reinterpret_borrow<py::array>(obj);
    if (!py::isinstance<py::array_t<T>>(obj) || !arr.dtype().is(py::dtype::of<T>()))
        raise_codec(Status::kBadArgument, std::string(name) + " has the wrong dtype");
    r.info = arr.request(true);
    if (!c_contiguous(r.info))
        raise_codec(Status::kBadArgument, std::string(name) + " must be C-contiguous");
    if (reinterpret_cast<std::uintptr_t>(r.info.ptr) % alignof(T) != 0)
        raise_codec(Status::kBadArgument, std::string(name) + " is not aligned");
    r.ptr = static_cast<T*>(r.info.ptr);
    r.n = static_cast<std::size_t>(r.info.size);
    return r;
}

struct Block {
    In block, runs;
    In f_off, f_types, f_aoff, f_atoms;
    BlockInput in{};
};

void fill_block(Block& b, const py::object& block, const py::object& runs, std::size_t rows,
                std::size_t L_heavy, std::size_t L, std::size_t n_types,
                const py::object& features) {
    b.block = input(block, "block");
    b.runs = input(runs, "runs");
    if (b.runs.bytes % kRunRecordSize != 0)
        raise_codec(Status::kBadArgument, "the run slice is not a whole number of 16-byte records");
    b.in.in = b.block.ptr;
    b.in.in_len = b.block.bytes;
    b.in.runs = RunSlice{b.runs.ptr, b.runs.bytes / kRunRecordSize};
    b.in.rows = rows;
    b.in.L_heavy = L_heavy;
    b.in.L = L;
    b.in.n_types = n_types;
    b.in.features = FeatureSlice{};
    if (!features.is_none()) {
        auto t = features.cast<py::tuple>();
        if (t.size() != 4)
            raise_codec(Status::kBadArgument, "features is (offsets, types, atom_offsets, atoms)");
        b.f_off = typed_input(t[0], 4, "features.offsets");
        b.f_types = typed_input(t[1], 1, "features.types");
        b.f_aoff = typed_input(t[2], 4, "features.atom_offsets");
        b.f_atoms = typed_input(t[3], 1, "features.atoms");
        FeatureSlice& f = b.in.features;
        f.offsets = reinterpret_cast<const std::uint32_t*>(b.f_off.ptr);
        f.offsets_len = b.f_off.bytes / 4;
        f.types = b.f_types.ptr;
        f.types_len = b.f_types.bytes;
        f.atom_offsets = reinterpret_cast<const std::uint32_t*>(b.f_aoff.ptr);
        f.atom_offsets_len = b.f_aoff.bytes / 4;
        f.atoms = b.f_atoms.ptr;
        f.atoms_len = b.f_atoms.bytes;
        // An empty numpy array may report a null data pointer; the core reads
        // nothing from a zero-length array, so give it a valid one.
        static const std::uint8_t kEmpty[4] = {0, 0, 0, 0};
        if (!f.types) f.types = kEmpty;
        if (!f.atoms) f.atoms = kEmpty;
        if (!f.atom_offsets) f.atom_offsets = reinterpret_cast<const std::uint32_t*>(kEmpty);
        if (!f.offsets) f.offsets = reinterpret_cast<const std::uint32_t*>(kEmpty);
    }
}

struct SelfOverlapArgs {
    In vaa, cs;
    Out<float> vaa_out;
    Out<double> cs_out;
    bool active = false;
    SelfOverlapInput so{};
};

void fill_self_overlaps(SelfOverlapArgs& a, const py::object& vaa, const py::object& cs,
                        std::size_t first_row, const py::object& vaa_out,
                        const py::object& cs_out) {
    if (vaa.is_none()) return;
    a.active = true;
    a.vaa = typed_input(vaa, 4, "vaa");
    a.so.vaa = reinterpret_cast<const float*>(a.vaa.ptr);
    a.so.vaa_len = a.vaa.bytes / 4;
    if (!cs.is_none()) {
        a.cs = typed_input(cs, 8, "color_self");
        a.so.color_self = reinterpret_cast<const double*>(a.cs.ptr);
        a.so.color_self_len = a.cs.bytes / 8;
    }
    a.so.first_row = first_row;
    a.vaa_out = output<float>(vaa_out, "vaa_out");
    a.cs_out = output<double>(cs_out, "color_self_out");
    if (!a.vaa_out.ptr) raise_codec(Status::kBadArgument, "vaa_out is required with vaa");
}

// Reject aliases while the GIL is held, before any output is written.
void disjoint(const Block& b, std::initializer_list<const py::buffer_info*> outputs,
              const SelfOverlapArgs* so = nullptr) {
    std::vector<const py::buffer_info*> inputs = {&b.block.info, &b.runs.info,
        &b.f_off.info, &b.f_types.info, &b.f_aoff.info, &b.f_atoms.info};
    if (so) {
        inputs.push_back(&so->vaa.info);
        inputs.push_back(&so->cs.info);
    }
    std::vector<const py::buffer_info*> outs(outputs);
    if (so) {
        outs.push_back(&so->vaa_out.info);
        outs.push_back(&so->cs_out.info);
    }
    auto overlaps = [](const py::buffer_info* a, const py::buffer_info* c) {
        if (!a->ptr || !c->ptr || !a->size || !c->size) return false;
        const auto ap = reinterpret_cast<std::uintptr_t>(a->ptr);
        const auto cp = reinterpret_cast<std::uintptr_t>(c->ptr);
        const auto an = static_cast<std::size_t>(a->size) * a->itemsize;
        const auto cn = static_cast<std::size_t>(c->size) * c->itemsize;
        return ap <= cp ? cp - ap < an : ap - cp < cn;
    };
    for (std::size_t i = 0; i < outs.size(); ++i) {
        for (const auto* in : inputs)
            if (overlaps(outs[i], in)) raise_codec(Status::kBadArgument, "output overlaps input");
        for (std::size_t j = 0; j < i; ++j)
            if (overlaps(outs[i], outs[j])) raise_codec(Status::kBadArgument, "outputs overlap");
    }
}

using Timing3 = std::tuple<double, double, double>;

// Runs a writer, then the self-overlap binding; raises on error.
template <class F>
std::tuple<Timing3, bool> run_writer(F&& writer, std::size_t rows, SelfOverlapArgs& so) {
    BlockTimings tm;
    Status s;
    bool cs_present = false;
    {
        py::gil_scoped_release nogil;
        s = writer(&tm);
        if (s == Status::kOk && so.active) {
            const auto t0 = std::chrono::steady_clock::now();
            s = bind_self_overlaps(so.so, rows, so.vaa_out.ptr, so.vaa_out.n, so.cs_out.ptr,
                                   so.cs_out.n, &cs_present);
            tm.self_overlap =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        }
    }
    if (s != Status::kOk) raise_codec(s);
    return {Timing3{tm.decode, tm.color, tm.self_overlap}, cs_present};
}

Isa parse_isa(const std::string& name) {
    if (name == "auto") return Isa::kAuto;
    if (name == "scalar") return Isa::kScalar;
    if (name == "avx2") return Isa::kAvx2;
    if (name == "neon") return Isa::kNeon;
    raise_codec(Status::kIsaUnavailable, "unknown ISA '" + name + "'");
}

}  // namespace

void bind_r3g_codec_api(py::module_& m) {
    m.def("r3g_block_length", [](std::size_t total, unsigned wx, unsigned wy, unsigned wz) {
        std::size_t n = 0;
        Status s = block_length(total, wx, wy, wz, &n);
        if (s != Status::kOk) raise_codec(s);
        return n;
    });
    m.def("r3g_gpu_layout_elements", [](std::size_t rows, std::size_t L) {
        std::size_t nr = 0, e = 0;
        Status s = gpu_layout_elements(rows, L, &nr, &e);
        if (s != Status::kOk) raise_codec(s);
        return std::make_tuple(nr, e);
    });
    m.def("r3g_decode_heavy", [](const py::object& block, const py::object& runs,
                                 std::size_t rows, std::size_t L_heavy, const py::object& out) {
        Block b;
        fill_block(b, block, runs, rows, L_heavy, 0, 0, py::none());
        Out<std::int16_t> o = output<std::int16_t>(out, "out");
        disjoint(b, {&o.info});
        Status s;
        {
            py::gil_scoped_release nogil;
            s = decode_heavy(b.in, o.ptr, o.n);
        }
        if (s != Status::kOk) raise_codec(s);
    });
    m.def("r3g_decode_merged", [](const py::object& block, const py::object& runs,
                                  std::size_t rows, std::size_t L_heavy, std::size_t L,
                                  std::size_t n_types, const py::object& features,
                                  const py::object& out, const py::object& counts) {
        Block b;
        fill_block(b, block, runs, rows, L_heavy, L, n_types, features);
        Out<std::int16_t> o = output<std::int16_t>(out, "out");
        Out<std::uint8_t> c = output<std::uint8_t>(counts, "counts");
        disjoint(b, {&o.info, &c.info});
        BlockTimings tm;
        Status s;
        {
            py::gil_scoped_release nogil;
            s = decode_merged(b.in, o.ptr, o.n, c.ptr, c.n, &tm);
        }
        if (s != Status::kOk) raise_codec(s);
        return Timing3{tm.decode, tm.color, tm.self_overlap};
    });
    m.def("r3g_write_gpu", [](const py::object& block, const py::object& runs, std::size_t rows,
                              std::size_t L_heavy, std::size_t L, std::size_t n_types,
                              const py::object& features, int tile_mode, const py::object& out,
                              const py::object& counts, const py::object& words,
                              const py::object& vaa, const py::object& cs, std::size_t first_row,
                              const py::object& vaa_out, const py::object& cs_out) {
        Block b;
        fill_block(b, block, runs, rows, L_heavy, L, n_types, features);
        Out<std::int16_t> o = output<std::int16_t>(out, "out");
        Out<std::uint8_t> c = output<std::uint8_t>(counts, "counts_out");
        Out<std::uint64_t> w = output<std::uint64_t>(words, "count_words_out");
        SelfOverlapArgs so;
        fill_self_overlaps(so, vaa, cs, first_row, vaa_out, cs_out);
        disjoint(b, {&o.info, &c.info, &w.info}, &so);
        return run_writer([&](BlockTimings* tm) {
            return write_gpu_block(b.in, tile_mode, o.ptr, o.n, c.ptr, c.n, w.ptr, w.n, tm);
        }, rows, so);
    });
    m.def("r3g_write_gpu_launch", [](const py::sequence& inputs, std::size_t rows,
                                      std::size_t L_heavy, std::size_t L,
                                      std::size_t n_types, int tile_mode,
                                      const py::object& out, const py::object& counts,
                                      const py::object& mol_ids) {
        const std::size_t nb = inputs.size();
        // Each nonempty block has at least one row; bound native metadata
        // allocation by the caller's launch row count before allocating it.
        if (!nb || nb > rows) raise_codec(Status::kBadArgument, "invalid launch block count");
        Out<std::int16_t> o = output<std::int16_t>(out, "out");
        Out<std::uint8_t> c = output<std::uint8_t>(counts, "counts_out");
        Out<std::int64_t> ids = output<std::int64_t>(mol_ids, "mol_ids_out");
        std::vector<Block> owned(nb);
        std::vector<BlockInput> blocks(nb);
        for (std::size_t bi = 0; bi < nb; ++bi) {
            auto t = py::cast<py::tuple>(inputs[bi]);
            if (t.size() != 4) raise_codec(Status::kBadArgument, "launch block is (bytes, runs, rows, features)");
            fill_block(owned[bi], t[0], t[1], t[2].cast<std::size_t>(),
                       L_heavy, L, n_types, t[3]);
            disjoint(owned[bi], {&o.info, &c.info, &ids.info});
            blocks[bi] = owned[bi].in;
        }
        Status s;
        {
            py::gil_scoped_release nogil;
            s = write_gpu_launch(blocks.data(), nb, rows, tile_mode,
                                 o.ptr, o.n, c.ptr, c.n, ids.ptr, ids.n);
        }
        if (s != Status::kOk) raise_codec(s);
    });
    m.def("r3g_write_cpu", [](const py::object& block, const py::object& runs, std::size_t rows,
                              std::size_t L_heavy, std::size_t L, std::size_t n_types,
                              const py::object& features, float coord_scale,
                              const py::object& out, const py::object& counts,
                              const py::object& words, const py::object& vaa,
                              const py::object& cs, std::size_t first_row,
                              const py::object& vaa_out, const py::object& cs_out) {
        Block b;
        fill_block(b, block, runs, rows, L_heavy, L, n_types, features);
        Out<float> o = output<float>(out, "out");
        Out<std::uint8_t> c = output<std::uint8_t>(counts, "counts_out");
        Out<std::uint64_t> w = output<std::uint64_t>(words, "count_words_out");
        SelfOverlapArgs so;
        fill_self_overlaps(so, vaa, cs, first_row, vaa_out, cs_out);
        disjoint(b, {&o.info, &c.info, &w.info}, &so);
        return run_writer([&](BlockTimings* tm) {
            return write_cpu_block(b.in, coord_scale, o.ptr, o.n, c.ptr, c.n, w.ptr, w.n, tm);
        }, rows, so);
    });
    m.def("r3g_encode", [](const py::object& coords, std::size_t rows, std::size_t L_heavy,
                           const py::object& runs) {
        In c = typed_input(coords, 2, "coords");
        In r = input(runs, "runs");
        if (r.bytes % kRunRecordSize != 0)
            raise_codec(Status::kBadArgument, "the run slice is not a whole number of 16-byte records");
        const RunSlice rs{r.ptr, r.bytes / kRunRecordSize};
        const auto* cp = reinterpret_cast<const std::int16_t*>(c.ptr);
        std::size_t len = 0;
        Status s;
        {
            py::gil_scoped_release nogil;
            s = encode_block(cp, c.bytes / 2, rs, rows, L_heavy, nullptr, 0, &len);
        }
        if (s != Status::kOk && s != Status::kOutputTooSmall) raise_codec(s);
        std::vector<std::uint8_t> buf(len);
        {
            py::gil_scoped_release nogil;
            s = encode_block(cp, c.bytes / 2, rs, rows, L_heavy, buf.data(), buf.size(), &len);
        }
        if (s != Status::kOk) raise_codec(s);
        return py::bytes(reinterpret_cast<const char*>(buf.data()), len);
    });
    m.def("r3g_read_into", [](int fd, std::uint64_t offset, std::size_t length,
                              const py::object& out) {
        Out<std::uint8_t> o = output<std::uint8_t>(out, "out");
        if (o.n < length) raise_codec(Status::kOutputTooSmall);
        Status s;
        {
            py::gil_scoped_release nogil;
            s = read_exact(fd, offset, o.ptr, length);
        }
        if (s != Status::kOk) raise_codec(s);
    });
    m.def("r3g_set_builder_mode", [](const std::string& name) {
        BuilderMode mode;
        if (name == "auto") mode = BuilderMode::kAuto;
        else if (name == "baseline") mode = BuilderMode::kBaseline;
        else if (name == "fused") mode = BuilderMode::kFused;
        else raise_codec(Status::kBadArgument, "unknown builder mode '" + name + "'");
        const Status s = set_builder_mode(mode);
        if (s != Status::kOk) raise_codec(s);
    });
    m.def("r3g_builder_info", [](const std::string& consumer, float step) {
        if (consumer != "cpu" && consumer != "gpu")
            raise_codec(Status::kBadArgument, "builder consumer must be cpu or gpu");
        if (!std::isfinite(step) || !(step > 0.0f))
            raise_codec(Status::kBadArgument, "builder scale must be finite and positive");
        const BuilderMode requested = builder_mode();
        const bool fused = use_fused_builder(consumer == "gpu");
        const Isa isa = active_isa();
        const char* scale = consumer == "gpu" ? "integer" : "generic_multiply";
        if (fused && consumer == "cpu" && isa == Isa::kNeon) {
            if (step == 0x1p-8f) scale = "fixed_2^-8";
            else if (step == 0x1p-10f) scale = "fixed_2^-10";
        }
        py::dict info;
        info["requested_mode"] = builder_mode_name(requested);
        info["isa"] = isa_name(isa);
        info["implementation"] = fused ? "fused" : "baseline";
        info["scale_specialization"] = scale;
        info["tile_lanes"] = kTileRows;
        info["consumer"] = consumer;
        info["coord_scale"] = step;
        info["alias_fallback"] = "baseline";
        return info;
    });
    m.def("r3g_isa", []() { return std::string(isa_name(active_isa())); });
    m.def("r3g_compiled_isas", []() {
        std::vector<std::string> v;
        for (Isa i : {Isa::kScalar, Isa::kAvx2, Isa::kNeon})
            if (isa_compiled(i)) v.emplace_back(isa_name(i));
        return v;
    });
    m.def("r3g_force_isa", [](const std::string& name) {
        Status s = force_isa(parse_isa(name));
        if (s != Status::kOk) raise_codec(s, "'" + name + "' is not compiled into this build");
    });
}

}  // namespace r3g
}  // namespace r3
