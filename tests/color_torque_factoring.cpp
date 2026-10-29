#include "color_torque_factoring.hpp"
#include <cmath>
#include <cstdio>
#include <array>
#include <cstdlib>
using namespace r3::cpu;

int main() {
#if defined(__AVX2__)
    ColorConformerBatch8 batch{};
    batch.query_xyz.resize(2*3*8);
    batch.query_mask.assign(2*8,-1);
    batch.library_mask.assign(8,-1);
    float posed[24], translation[3]={.13f,-.17f,.07f};
    const float target[3]={.4f,-.2f,.3f};
    for (int lane=0; lane<8; ++lane) {
        for (int k=0; k<3; ++k) {
            posed[k*8+lane]=target[k];
            batch.query_xyz[k*8+lane]=float(k+1)*.1f;
            batch.query_xyz[(3+k)*8+lane]=float(k+1)*-.16f;
        }
    }
    alignas(32) int32_t mask[8];
    for (int lane=0;lane<8;++lane) mask[lane]=lane<7?-1:0;
    auto evaluate=[&](const float* coordinates) {
        __m256 value=_mm256_setzero_ps(), descent[6];
        for (auto& component:descent) component=_mm256_setzero_ps();
        accumulate_canonical_factored_torque(batch,1,coordinates,mask,2,1,
            .8f,-.6f,-1.2f,_mm256_set1_ps(translation[0]),
            _mm256_set1_ps(translation[1]),_mm256_set1_ps(translation[2]),
            value,descent, [](__m256 input) {
                alignas(32) float values[8];
                _mm256_store_ps(values,input);
                for(auto& item:values) item=std::exp(item);
                return _mm256_load_ps(values);
            });
        std::array<float,7> output;
        alignas(32) float lanes[8];
        _mm256_store_ps(lanes,value);output[0]=lanes[0];
        if (lanes[7]!=0.f) std::abort();
        for (int k=0;k<6;++k) {
            _mm256_store_ps(lanes,descent[k]);output[k+1]=lanes[0];
            if (lanes[7]!=0.f) std::abort();
        }
        return output;
    };
    const auto actual=evaluate(posed);
    constexpr float epsilon=2.e-3f;
    for(int component=0;component<6;++component) {
        float plus[24],minus[24];
        for(int k=0;k<3;++k) {
            float delta=component>=3 && k==component-3?epsilon:0.f;
            if(component<3) {
                // Left infinitesimal quaternion: db=2*epsilon*(axis cross r).
                const int a=(component+1)%3,b=(component+2)%3;
                if(k==a) delta=-2*epsilon*(target[b]-translation[b]);
                if(k==b) delta= 2*epsilon*(target[a]-translation[a]);
            }
            for(int lane=0;lane<8;++lane) {
                plus[k*8+lane]=posed[k*8+lane]+delta;
                minus[k*8+lane]=posed[k*8+lane]-delta;
            }
        }
        const float expected=-(evaluate(plus)[0]-evaluate(minus)[0])/(2*epsilon);
        if(std::abs(actual[component+1]-expected)>3.e-4f) {
            std::printf("FAIL component%d actual%g expected%g\n",component,
                actual[component+1],expected);return 1;
        }
    }
#endif
    std::puts("PASS T215 force/torque reduced-frame finite differences and tail");
}
