// Bounded coordinate-builder transposes, exact integer means and layout emitters.
#include "r3g_build.hpp"
#if !defined(__AVX2__)
#error "r3g_build_avx2.cpp requires AVX2 (-march=x86-64-v3)"
#endif
#include <immintrin.h>
#include <array>

namespace r3 { namespace r3g { namespace detail {
namespace {
inline void transpose8(__m128i (&r)[8]) noexcept {
    const auto t0=_mm_unpacklo_epi16(r[0],r[1]), t1=_mm_unpackhi_epi16(r[0],r[1]);
    const auto t2=_mm_unpacklo_epi16(r[2],r[3]), t3=_mm_unpackhi_epi16(r[2],r[3]);
    const auto t4=_mm_unpacklo_epi16(r[4],r[5]), t5=_mm_unpackhi_epi16(r[4],r[5]);
    const auto t6=_mm_unpacklo_epi16(r[6],r[7]), t7=_mm_unpackhi_epi16(r[6],r[7]);
    const auto u0=_mm_unpacklo_epi32(t0,t2), u1=_mm_unpackhi_epi32(t0,t2);
    const auto u2=_mm_unpacklo_epi32(t1,t3), u3=_mm_unpackhi_epi32(t1,t3);
    const auto u4=_mm_unpacklo_epi32(t4,t6), u5=_mm_unpackhi_epi32(t4,t6);
    const auto u6=_mm_unpacklo_epi32(t5,t7), u7=_mm_unpackhi_epi32(t5,t7);
    r[0]=_mm_unpacklo_epi64(u0,u4);r[1]=_mm_unpackhi_epi64(u0,u4);
    r[2]=_mm_unpacklo_epi64(u1,u5);r[3]=_mm_unpackhi_epi64(u1,u5);
    r[4]=_mm_unpacklo_epi64(u2,u6);r[5]=_mm_unpackhi_epi64(u2,u6);
    r[6]=_mm_unpacklo_epi64(u3,u7);r[7]=_mm_unpackhi_epi64(u3,u7);
}

template<unsigned K>
inline __m256i mean8(__m256i sum) noexcept {
    constexpr unsigned D=2*K;
    const auto n=_mm256_add_epi32(_mm256_slli_epi32(_mm256_abs_epi32(sum),1),_mm256_set1_epi32(K));
    __m256i q;
    if constexpr ((D&(D-1))==0) {
        constexpr unsigned shift=[] { unsigned n=0;for(unsigned v=D;v>1;v>>=1)++n;return n;}();
        q=_mm256_srli_epi32(n,shift);
    } else {
        constexpr auto magic=static_cast<std::uint32_t>((std::uint64_t{1}<<32)/D);
        const auto m=_mm256_set1_epi32(static_cast<int>(magic));
        const auto even=_mm256_mul_epu32(n,m);
        const auto odd=_mm256_mul_epu32(_mm256_srli_epi64(n,32),m);
        q=_mm256_blend_epi32(_mm256_srli_epi64(even,32),odd,0xaa);
        // floor(2^32/D) underestimates by at most one for these bounded N.
        const auto rem=_mm256_sub_epi32(n,_mm256_mullo_epi32(q,_mm256_set1_epi32(D)));
        q=_mm256_sub_epi32(q,_mm256_cmpgt_epi32(rem,_mm256_set1_epi32(D-1)));
    }
    return _mm256_sign_epi32(q,sum);
}

template<unsigned K>
void feature_tile(std::int16_t* tile,std::size_t slot,const std::uint8_t* atoms,
                  std::size_t begin,std::size_t lanes) noexcept {
    for(std::size_t a=0;a<3;++a) {
        auto* dst=tile+(slot*3+a)*32+begin;
        std::size_t r=0;
        for(;r+8<=lanes;r+=8) {
            if constexpr(K==1) {
                const auto* src=tile+(static_cast<std::size_t>(atoms[0])*3+a)*32+begin+r;
                _mm_storeu_si128(reinterpret_cast<__m128i*>(dst+r),_mm_loadu_si128(reinterpret_cast<const __m128i*>(src)));
            } else {
                auto sum=_mm256_setzero_si256();
                for(unsigned j=0;j<K;++j) {
                    const auto* src=tile+(static_cast<std::size_t>(atoms[j])*3+a)*32+begin+r;
                    sum=_mm256_add_epi32(sum,_mm256_cvtepi16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(src))));
                }
                const auto q=mean8<K>(sum);
                const auto packed=_mm_packs_epi32(_mm256_castsi256_si128(q),_mm256_extracti128_si256(q,1));
                _mm_storeu_si128(reinterpret_cast<__m128i*>(dst+r),packed);
            }
        }
        for(;r<lanes;++r) {
            if constexpr(K==1) {
                dst[r]=tile[(static_cast<std::size_t>(atoms[0])*3+a)*32+begin+r];
            } else {
                std::int32_t sum=0;
                for(unsigned j=0;j<K;++j)sum+=tile[(static_cast<std::size_t>(atoms[j])*3+a)*32+begin+r];
                dst[r]=builder_mean(sum,K);
            }
        }
    }
}

template<unsigned Part,unsigned Axis>
inline __m256i xyz_mask() noexcept {
    alignas(16) static constexpr auto bytes=[] {
        std::array<std::uint8_t,16> mask{};
        for(unsigned b=0;b<16;++b) {
            const unsigned word=Part*8+b/2;
            mask[b]=word%3==Axis?static_cast<std::uint8_t>(2*(word/3)+b%2):0x80;
        }
        return mask;
    }();
    return _mm256_broadcastsi128_si256(_mm_load_si128(reinterpret_cast<const __m128i*>(bytes.data())));
}
template<unsigned Part>
inline __m256i xyz16(__m256i x,__m256i y,__m256i z) noexcept {
    return _mm256_or_si256(_mm256_or_si256(_mm256_shuffle_epi8(x,xyz_mask<Part,0>()),
        _mm256_shuffle_epi8(y,xyz_mask<Part,1>())),_mm256_shuffle_epi8(z,xyz_mask<Part,2>()));
}
inline void store_xyz4(float* out,__m128 x,__m128 y,__m128 z) noexcept {
    const auto xylo=_mm_unpacklo_ps(x,y), xyhi=_mm_unpackhi_ps(x,y);
    const auto zxlo=_mm_unpacklo_ps(z,x), zxhi=_mm_unpackhi_ps(z,x);
    const auto yzlo=_mm_unpacklo_ps(y,z), yzhi=_mm_unpackhi_ps(y,z);
    _mm_storeu_ps(out,_mm_shuffle_ps(xylo,zxlo,_MM_SHUFFLE(3,0,1,0)));
    _mm_storeu_ps(out+4,_mm_shuffle_ps(yzlo,xyhi,_MM_SHUFFLE(1,0,3,2)));
    _mm_storeu_ps(out+8,_mm_shuffle_ps(zxhi,yzhi,_MM_SHUFFLE(3,2,3,0)));
}
inline void store_xyz8(float* out,__m256 x,__m256 y,__m256 z) noexcept {
    store_xyz4(out,_mm256_castps256_ps128(x),_mm256_castps256_ps128(y),_mm256_castps256_ps128(z));
    store_xyz4(out+12,_mm256_extractf128_ps(x,1),_mm256_extractf128_ps(y,1),_mm256_extractf128_ps(z,1));
}
} // namespace

void build_heavy_tile_avx2(const std::int16_t* axes,std::size_t axis_stride,
    std::size_t pos,std::size_t n_real,std::size_t begin,std::size_t lanes,std::int16_t* tile) noexcept {
    for(std::size_t a=0;a<3;++a) {
        const auto* src=axes+a*axis_stride+pos;
        std::size_t r=0;
        for(;r+8<=lanes;r+=8) {
            std::size_t s=0;
            for(;s+8<=n_real;s+=8) {
                __m128i rows[8];
                for(unsigned j=0;j<8;++j)rows[j]=_mm_loadu_si128(reinterpret_cast<const __m128i*>(src+(r+j)*n_real+s));
                transpose8(rows);
                for(unsigned j=0;j<8;++j)_mm_storeu_si128(reinterpret_cast<__m128i*>(tile+((s+j)*3+a)*32+begin+r),rows[j]);
            }
            for(;s<n_real;++s)for(unsigned j=0;j<8;++j)tile[(s*3+a)*32+begin+r+j]=src[(r+j)*n_real+s];
        }
        for(;r<lanes;++r)for(std::size_t s=0;s<n_real;++s)tile[(s*3+a)*32+begin+r]=src[r*n_real+s];
    }
}
void build_feature_tile_avx2(std::int16_t* tile,std::size_t slot,const std::uint8_t* atoms,
    unsigned arity,std::size_t begin,std::size_t lanes) noexcept {
#define R3_FEATURE_CASE(K) case K:feature_tile<K>(tile,slot,atoms,begin,lanes);return
    switch(arity) {
        R3_FEATURE_CASE(1);R3_FEATURE_CASE(2);R3_FEATURE_CASE(3);R3_FEATURE_CASE(4);
        R3_FEATURE_CASE(5);R3_FEATURE_CASE(6);R3_FEATURE_CASE(7);R3_FEATURE_CASE(8);
        R3_FEATURE_CASE(9);R3_FEATURE_CASE(10);R3_FEATURE_CASE(11);R3_FEATURE_CASE(12);
        R3_FEATURE_CASE(13);R3_FEATURE_CASE(14);R3_FEATURE_CASE(15);R3_FEATURE_CASE(16);
    }
#undef R3_FEATURE_CASE
}
void emit_gpu_tile_avx2(const std::int16_t* tile,std::size_t L,int mode,std::int16_t* out) noexcept {
    if(mode<0) {
        if(out==tile)return;
        for(std::size_t i=0;i<L*96;i+=16)_mm256_storeu_si256(reinterpret_cast<__m256i*>(out+i),_mm256_loadu_si256(reinterpret_cast<const __m256i*>(tile+i)));
        return;
    }
    for(std::size_t s=0;s<L;++s)for(std::size_t r=0;r<32;r+=16) {
        const auto x=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(tile+(s*3)*32+r));
        const auto y=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(tile+(s*3+1)*32+r));
        const auto z=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(tile+(s*3+2)*32+r));
        const auto a=xyz16<0>(x,y,z),b=xyz16<1>(x,y,z),c=xyz16<2>(x,y,z);
        auto* dst=out+(s*32+r)*3;
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst),_mm256_permute2x128_si256(a,b,0x20));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst+16),_mm256_permute2x128_si256(c,a,0x30));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst+32),_mm256_permute2x128_si256(b,c,0x31));
    }
}
void emit_cpu_tile_avx2(const std::int16_t* tile,std::size_t L,std::size_t lanes,float step,float* out) noexcept {
    const auto scale=_mm256_set1_ps(step);
    std::size_t r=0;
    for(;r+8<=lanes;r+=8) {
        std::size_t s=0;
        for(;s+8<=L;s+=8) {
            __m128i x[8],y[8],z[8];
            for(unsigned j=0;j<8;++j) {
                x[j]=_mm_loadu_si128(reinterpret_cast<const __m128i*>(tile+((s+j)*3)*32+r));
                y[j]=_mm_loadu_si128(reinterpret_cast<const __m128i*>(tile+((s+j)*3+1)*32+r));
                z[j]=_mm_loadu_si128(reinterpret_cast<const __m128i*>(tile+((s+j)*3+2)*32+r));
            }
            transpose8(x);transpose8(y);transpose8(z);
            for(unsigned j=0;j<8;++j) {
                const auto xf=_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(x[j])),scale);
                const auto yf=_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(y[j])),scale);
                const auto zf=_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(z[j])),scale);
                store_xyz8(out+((r+j)*L+s)*3,xf,yf,zf);
            }
        }
        for(;s<L;++s)for(unsigned j=0;j<8;++j)for(std::size_t a=0;a<3;++a)
            out[((r+j)*L+s)*3+a]=static_cast<float>(tile[(s*3+a)*32+r+j])*step;
    }
    // Inactive final-tile lanes may be indeterminate: never load them.
    for(;r<lanes;++r)for(std::size_t s=0;s<L;++s)for(std::size_t a=0;a<3;++a)
        out[(r*L+s)*3+a]=static_cast<float>(tile[(s*3+a)*32+r])*step;
}
}}} // namespace r3::r3g::detail
