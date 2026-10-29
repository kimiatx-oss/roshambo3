// Test-only scalar-anchor instrumentation. Production headers remain unchanged.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cstring>
#include "../src/roshambo3/backend/_cpu_src/simd.hpp"
namespace r3::simd {
static thread_local uint32_t control_seed = 0;
static thread_local float control_amplitude = 0.f;
inline VecF32 ControlledExp2Negative(VecF32 x) {
    float y = std::exp2(x.v);
    uint32_t bits; std::memcpy(&bits, &x.v, sizeof(bits));
    uint32_t h = bits ^ control_seed;
    h ^= h >> 16; h *= 0x7feb352du; h ^= h >> 15;
    return VecF32(y * (1.f + ((h & 1u) ? control_amplitude : -control_amplitude)));
}
}
#define Exp2Negative ControlledExp2Negative
#include "../src/roshambo3/backend/_cpu_src/score.hpp"
#undef Exp2Negative
namespace py = pybind11;
using Array = py::array_t<float, py::array::c_style | py::array::forcecast>;
PYBIND11_MODULE(_numerics_optimizer_control, m) {
    m.def("score", [](Array aa, Array bb, int seed, float amplitude,
                      int steps, float lr_q, float lr_t) {
        if (aa.ndim()!=2 || bb.ndim()!=2 || aa.shape(1)!=3 || bb.shape(1)!=3)
            throw std::invalid_argument("expected [n,3] clouds");
        r3::simd::control_seed=uint32_t(seed); r3::simd::control_amplitude=amplitude;
        using namespace r3::cpu;
        std::vector<float> a(aa.size()), b(bb.size());
        for (size_t i=0;i<a.size();++i) a[i]=COORD_SCALE*aa.data()[i];
        std::copy(bb.data(),bb.data()+bb.size(),b.begin());
        Conformer ac{a.data(),int(aa.shape(0))}, bc{b.data(),int(bb.shape(0))};
        // scalar CPU self library preprocessing uses reduced coordinates.
        std::vector<float> bs(b.size());
        for (size_t i=0;i<b.size();++i) bs[i]=COORD_SCALE*b[i];
        Conformer bsc{bs.data(),bc.n_atoms};
        auto r=optimize_pair(ac,bc,self_overlap(ac,nullptr),self_overlap(bsc,nullptr),
                             1,steps,lr_q,COORD_SCALE*lr_t,nullptr);
        py::array_t<float> out(20); auto p=out.mutable_data();
        std::fill(p,p+20,0.f); p[0]=p[1]=r.tanimoto_shape;
        std::copy(r.pose_q,r.pose_q+4,p+9); std::copy(r.pose_t,r.pose_t+3,p+13); p[16]=1.f;
        return py::make_tuple(out,r.best_start);
    });
}
