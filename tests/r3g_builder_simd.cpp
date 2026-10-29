// Standalone direct builder checks; no scalar builder or extension is linked.
#include "r3g_build.hpp"
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace d = r3::r3g::detail;
namespace {
#if defined(__aarch64__) || defined(_M_ARM64)
constexpr auto heavy = d::build_heavy_tile_neon;
constexpr auto feature = d::build_feature_tile_neon;
constexpr auto gpu = d::emit_gpu_tile_neon;
constexpr auto cpu = d::emit_cpu_tile_neon;
constexpr const char* isa = "neon";
#else
constexpr auto heavy = d::build_heavy_tile_avx2;
constexpr auto feature = d::build_feature_tile_avx2;
constexpr auto gpu = d::emit_gpu_tile_avx2;
constexpr auto cpu = d::emit_cpu_tile_avx2;
constexpr const char* isa = "avx2";
#endif
constexpr std::int16_t guard = 23117;
std::size_t at(std::size_t slot, std::size_t axis, std::size_t lane) {
    return (slot * 3 + axis) * 32 + lane;
}
void require(bool ok, const std::string& why) {
    if (!ok) throw std::runtime_error(why);
}
std::int16_t sample(std::size_t i) {
    return static_cast<std::int16_t>(static_cast<int>((i * 7919 + 12345) & 65535) - 32768);
}
std::int16_t mean_reference(std::int64_t sum, unsigned k) {
    // Independent signed quotient/remainder construction, not N/D or a magic multiplier.
    auto q = sum / k;
    const auto r = sum % k;
    if (2 * (r < 0 ? -r : r) >= k) q += sum < 0 ? -1 : 1;
    return static_cast<std::int16_t>(q);
}
float poison() {
    const std::uint32_t bits = 0x7fc12345;
    float result;
    std::memcpy(&result, &bits, sizeof result);
    return result;
}
void same_floats(const std::vector<float>& a, const std::vector<float>& b,
                 const std::string& why) {
    require(a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0, why);
}
d::CpuTileEmitter selected(float step) {
#if defined(__aarch64__) || defined(_M_ARM64)
    const auto emitter = d::select_cpu_tile_emitter_neon(step);
    require((step == .001f) == (emitter == cpu), "NEON emitter selector");
    return emitter;
#else
    (void)step;
    return cpu;
#endif
}
void test_heavy_segments() {
    for (const std::size_t n : {1u,2u,7u,8u,9u,15u,16u,17u,31u,32u,33u,63u,64u,65u,255u,256u}) {
        for (std::size_t begin = 0; begin < 32; ++begin) {
            for (std::size_t lanes = 1; lanes <= 32 - begin; ++lanes) {
                const std::size_t pos = 3 + begin % 5, stride = pos + lanes * n, L = n + 3;
                std::vector<std::int16_t> axes(3 * stride + 1);
                for (std::size_t i = 0; i < 3 * stride; ++i) axes[i + 1] = sample(i);
                std::vector<std::int16_t> actual(L * 96 + 2, guard), expected = actual;
                for (std::size_t s = 0; s < n; ++s)
                    for (std::size_t a = 0; a < 3; ++a)
                        for (std::size_t r = 0; r < lanes; ++r)
                            expected[1 + at(s,a,begin+r)] = axes[1 + a*stride+pos+r*n+s];
                heavy(axes.data()+1,stride,pos,n,begin,lanes,actual.data()+1);
                require(actual == expected, "heavy segment n=" + std::to_string(n));
            }
        }
    }
}
void test_feature_segments() {
    constexpr std::size_t n = 17, slot = n + 2, L = slot + 3;
    for (unsigned k = 1; k <= 16; ++k) {
        std::vector<std::uint8_t> atoms(k+1);
        for (unsigned j = 0; j < k; ++j) atoms[j+1] = static_cast<std::uint8_t>((j*7+3)%n);
        for (std::size_t begin = 0; begin < 32; ++begin) {
            for (std::size_t lanes = 1; lanes <= 32-begin; ++lanes) {
                std::vector<std::int16_t> actual(L*96+2,guard);
                for (std::size_t s=0;s<n;++s)
                    for (std::size_t a=0;a<3;++a)
                        for (std::size_t r=0;r<32;++r) actual[1+at(s,a,r)]=sample(at(s,a,r));
                auto expected=actual;
                for (std::size_t a=0;a<3;++a) for (std::size_t r=begin;r<begin+lanes;++r) {
                    std::int64_t sum=0;
                    for(unsigned j=0;j<k;++j) sum+=actual[1+at(atoms[j+1],a,r)];
                    expected[1+at(slot,a,r)]=mean_reference(sum,k);
                }
                feature(actual.data()+1,slot,atoms.data()+1,k,begin,lanes);
                require(actual==expected,"feature segment k="+std::to_string(k));
            }
        }
    }
}
void test_all_attainable_sums() {
    std::uint64_t checked=0;
    std::array<std::uint8_t,16> atoms{};
    for(unsigned j=0;j<16;++j) atoms[j]=static_cast<std::uint8_t>(j);
    for(unsigned k=1;k<=16;++k) {
        std::vector<std::int16_t> tile((k+1)*96+2,guard);
        const auto low=-32768*static_cast<std::int32_t>(k), high=32767*static_cast<std::int32_t>(k);
        for(std::int32_t base=low;base<=high;base+=32) {
            const auto lanes=static_cast<std::size_t>(std::min<std::int32_t>(32,high-base+1));
            std::fill(tile.begin()+1+k*96,tile.begin()+1+(k+1)*96,guard);
            for(std::size_t r=0;r<lanes;++r) {
                const auto sum=base+static_cast<std::int32_t>(r), q=sum/static_cast<int>(k), rem=sum%static_cast<int>(k);
                for(unsigned j=0;j<k;++j) {
                    const auto v=static_cast<std::int16_t>(q+(j<static_cast<unsigned>(rem<0?-rem:rem)?(sum<0?-1:1):0));
                    for(std::size_t a=0;a<3;++a) tile[1+at(j,a,r)]=v;
                }
            }
            feature(tile.data()+1,k,atoms.data(),k,0,lanes);
            for(std::size_t a=0;a<3;++a) for(std::size_t r=0;r<32;++r) {
                const auto expected=r<lanes?mean_reference(base+static_cast<std::int32_t>(r),k):guard;
                if (tile[1+at(k,a,r)]!=expected)
                    require(false,"attainable sum k="+std::to_string(k)+" sum="+std::to_string(base+static_cast<int>(r)));
            }
            require(tile.front()==guard && tile.back()==guard,"feature buffer guards");
            checked+=lanes;
        }
    }
    require(checked==8912776,"attainable-sum census");
    std::cout << "attainable sums: " << checked << " (all arities 1..16)\n";
}
void test_layouts() {
    for(const std::size_t L:{1u,2u,7u,8u,9u,15u,16u,17u,31u,32u,33u}) {
        std::vector<std::int16_t> tile(L*96+2,guard);
        for(std::size_t i=0;i<L*96;++i) tile[i+1]=sample(i);
        for(std::size_t s=L>3?L-3:L;s<L;++s)
            std::fill(tile.begin()+1+s*96,tile.begin()+1+(s+1)*96,0);
        for(const int mode:{-1,1}) {
            std::vector<std::int16_t> actual(L*96+2,guard),expected=actual;
            for(std::size_t s=0;s<L;++s)for(std::size_t a=0;a<3;++a)for(std::size_t r=0;r<32;++r)
                expected[1+(mode<0?at(s,a,r):(s*32+r)*3+a)]=tile[1+at(s,a,r)];
            gpu(tile.data()+1,L,mode,actual.data()+1);
            require(actual==expected,"GPU layout");
        }
        auto alias=tile;
        gpu(alias.data()+1,L,-1,alias.data()+1);
        require(alias==tile,"GPU planar alias no-op");
        for(const float step:{0x1p-8f,0x1p-10f,.001f}) for(std::size_t lanes=1;lanes<=32;++lanes) {
            std::vector<float> actual(lanes*L*3+2,poison()),expected=actual;
            for(std::size_t r=0;r<lanes;++r)for(std::size_t s=0;s<L;++s)for(std::size_t a=0;a<3;++a)
                expected[1+(r*L+s)*3+a]=static_cast<float>(tile[1+at(s,a,r)])*step;
            cpu(tile.data()+1,L,lanes,step,actual.data()+1);
            same_floats(actual,expected,"CPU layout/tail/padding bits");
            std::fill(actual.begin(),actual.end(),poison());
            selected(step)(tile.data()+1,L,lanes,step,actual.data()+1);
            same_floats(actual,expected,"selected CPU emitter bits");
        }
    }
}
void test_all_conversions() {
    constexpr std::size_t L=8, count=L*96;
    std::vector<std::int16_t> tile(count+2,guard);
    for(const float step:{0x1p-8f,0x1p-10f,.001f}) {
        const auto emitter=selected(step);
        for(std::size_t base=0;base<65536;base+=count) {
            std::vector<float> actual(count+2,poison()),expected=actual;
            for(std::size_t r=0;r<32;++r)for(std::size_t s=0;s<L;++s)for(std::size_t a=0;a<3;++a) {
                const auto i=(r*L+s)*3+a;
                const auto v=static_cast<std::int16_t>(base+i<65536?static_cast<int>(base+i)-32768:0);
                tile[1+at(s,a,r)]=v;expected[1+i]=static_cast<float>(v)*step;
            }
            cpu(tile.data()+1,L,32,step,actual.data()+1);
            same_floats(actual,expected,"exhaustive generic conversion bits");
            std::fill(actual.begin(),actual.end(),poison());
            emitter(tile.data()+1,L,32,step,actual.data()+1);
            same_floats(actual,expected,"exhaustive selected conversion bits");
        }
    }
    std::cout << "conversion: all 65536 int16 values at 2^-8, 2^-10, .001f\n";
}
}
int main() {
    try {
        test_heavy_segments();test_feature_segments();test_all_attainable_sums();test_layouts();test_all_conversions();
        std::cout << isa << " builder primitives: PASS\n";
    } catch(const std::exception& error) { std::cerr << "FAIL: " << error.what() << '\n';return 1; }
}
