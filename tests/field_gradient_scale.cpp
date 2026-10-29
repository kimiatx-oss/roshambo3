// Independent value finite differences of every field gradient arm.
// Build/run: g++ -O2 -march=x86-64-v3 -std=c++17 -I src/roshambo3/backend/_cpu_src \
//   tests/field_gradient_scale.cpp -o /tmp/field-gradient && /tmp/field-gradient
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "score.hpp"
#include "field_avx2.hpp"

using namespace r3::cpu;

static void rotation(int axis, double angle, const float m[9], float out[9]) {
    double r[9] = {0};
    for (int k = 0; k < 3; ++k) r[3*k+k] = std::cos(angle);
    r[3*axis+axis] = 1;
    r[3*((axis+1)%3)+(axis+2)%3] = -std::sin(angle);
    r[3*((axis+2)%3)+(axis+1)%3] = std::sin(angle);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            double v = 0;
            for (int k = 0; k < 3; ++k) v += r[3*i+k] * m[3*k+j];
            out[3*i+j] = float(v);
        }
}

// The value oracle uses double and a direct pair sum. It shares neither the
// field's stored gradient nor the analytic derivative's arithmetic.
static double overlap(const std::vector<float>& a, const std::vector<float>& b,
                      const float m[9], const float t[3]) {
    double s = 0;
    for (size_t j = 0; j < b.size()/3; ++j)
        for (size_t i = 0; i < a.size()/3; ++i) {
            double d2 = 0;
            for (int k = 0; k < 3; ++k) {
                const double p = double(m[3*k])*b[3*j] +
                    double(m[3*k+1])*b[3*j+1] +
                    double(m[3*k+2])*b[3*j+2] + t[k];
                const double d = double(a[3*i+k]) - p;
                d2 += d*d;
            }
            s += std::exp2(-d2);
        }
    return s;
}

static void finite_difference(const std::vector<float>& a,
                              const std::vector<float>& b, const float m[9],
                              const float t[3], float gnorm, double out[7]) {
    constexpr double step = 0.001;
    out[0] = 0;
    for (int k = 0; k < 3; ++k) {
        float plus[9], minus[9];
        rotation(k, step, m, plus);
        rotation(k, -step, m, minus);
        out[k+1] = -2*gnorm*(overlap(a,b,plus,t)-overlap(a,b,minus,t))/(2*step);
        float tp[3] = {t[0],t[1],t[2]}, tm[3] = {t[0],t[1],t[2]};
        tp[k] += float(step); tm[k] -= float(step);
        out[k+4] = -gnorm*(overlap(a,b,m,tp)-overlap(a,b,m,tm)) /
            double(tp[k]-tm[k]);
    }
}

struct Result { float g[7] = {}; float score = 0; };

static Result run(const char* arm, int setting, const FieldNaive32& nf,
                  const FieldNaive16& hf, const FieldCoeff32& cf,
                  const FieldEdge32& ef, const FieldAffine32& af,
                  const Conformer& b, const ConformerSoA& soa,
                  const float m[9], const float t[3], float gn) {
    Result r;
#define ARGS m,t,gn,r.g,&r.score,nullptr
    if (!std::strcmp(arm,"reg")) {
        if (setting == 0) field_gradient_avx2_reg<false,true,0,false,false>(nf,soa,ARGS);
        if (setting == 1) field_gradient_avx2_reg<false,true,0,false,true>(nf,soa,ARGS);
        if (setting == 2) field_gradient_avx2_reg<false,true,0,true,false>(nf,soa,ARGS);
        if (setting == 3) field_gradient_avx2_reg<false,true,0,true,true>(nf,soa,ARGS);
    } else if (!std::strcmp(arm,"fp16")) {
        if (setting == 0) field_gradient_avx2_reg<false,true,0,false,false>(hf,soa,ARGS);
        if (setting == 3) field_gradient_avx2_reg<false,true,0,true,true>(hf,soa,ARGS);
    } else if (!std::strcmp(arm,"pairaos")) {
        if (setting == 0) field_gradient_avx2_pairaos<true,false,false>(nf,soa,ARGS);
        if (setting == 3) field_gradient_avx2_pairaos<true,true,true>(nf,soa,ARGS);
    } else if (!std::strcmp(arm,"affine")) {
        if (setting == 0) field_gradient_avx2_affine<true,false,false>(af,soa,ARGS);
        if (setting == 3) field_gradient_avx2_affine<true,true,true>(af,soa,ARGS);
    } else if (!std::strcmp(arm,"coeff")) {
        if (setting == 0) field_gradient_avx2_coeff<true,false,false,false>(cf,soa,ARGS);
        if (setting == 3) field_gradient_avx2_coeff<true,false,true,true>(cf,soa,ARGS);
    } else if (!std::strcmp(arm,"edge")) {
        if (setting == 0) field_gradient_avx2_edgefield<true,false,false>(ef,soa,ARGS);
        if (setting == 3) field_gradient_avx2_edgefield<true,true,true>(ef,soa,ARGS);
    } else if (!std::strcmp(arm,"scalar")) {
        field_gradient(nf,b,ARGS);
    } else if (!std::strcmp(arm,"unroll")) {
        field_gradient_avx2_unrolled<2>(nf,b,ARGS);
    }
#undef ARGS
    return r;
}

static double relerr(const float g[7], const double fd[7], int first) {
    double e=0, n=0;
    for (int k=first; k<first+3; ++k) {
        e += (g[k]-fd[k])*(g[k]-fd[k]); n += fd[k]*fd[k];
    }
    return std::sqrt(e/n);
}

int main(int argc, char** argv) {
    // The Python gate checks this reduced-frame value directly against
    // reference.overlap.pair_overlap / PAIR_PREFACTOR.
    if (argc == 2 && !std::strcmp(argv[1], "--oracle-smoke")) {
        const std::vector<float> a = {
            0.2f*COORD_SCALE, -0.1f*COORD_SCALE, 0.7f*COORD_SCALE,
            -0.5f*COORD_SCALE, 0.4f*COORD_SCALE, 0.3f*COORD_SCALE};
        const std::vector<float> b = {
            0.6f*COORD_SCALE, 0.2f*COORD_SCALE, -0.3f*COORD_SCALE};
        const float m[9] = {1,0,0,0,1,0,0,0,1};
        const float t[3] = {0.1f*COORD_SCALE, -0.2f*COORD_SCALE,
                            0.05f*COORD_SCALE};
        std::printf("%.17g\n", overlap(a,b,m,t));
        return 0;
    }
    const bool perp_control = argc == 2 && !std::strcmp(argv[1], "--perp-control");
    std::mt19937 rng(20260907);
    std::normal_distribution<float> normal(0.f,1.6f);
    std::vector<float> a(42), b(36);
    for (auto& x:a) x=normal(rng);
    for (auto& x:b) x=normal(rng);
    Conformer A{a.data(),14}, B{b.data(),12};
    ConformerSoA soa; soa.build(B);
    const float gn=1.f/(self_overlap(A,nullptr)+self_overlap(B,nullptr));
    const char* arms[] = {"reg","fp16","pairaos","affine","coeff","edge","scalar","unroll"};
    bool ok=true;
    for (int pose=0; pose<4; ++pose) {
        float identity[9]={1,0,0,0,1,0,0,0,1}, m[9];
        rotation(pose%3,0.35+0.21*pose,identity,m);
        const float t[3]={0.31f+0.12f*pose,-0.22f+0.07f*pose,0.17f-0.08f*pose};
        double fd[7]; finite_difference(a,b,m,t,gn,fd);
        Result analytic;
        get_gradient(A,B,m,t,gn,analytic.g,&analytic.score,nullptr);
        if (relerr(analytic.g,fd,1)>0.002 || relerr(analytic.g,fd,4)>0.002) {
            std::printf("FAIL analytic finite difference pose=%d torque=%g translation=%g\n",
                pose,relerr(analytic.g,fd,1),relerr(analytic.g,fd,4));
            ok=false;
        }
        double prev_t[8][4], prev_x[8][4];
        for (int ai=0;ai<8;++ai) for (int setting=0;setting<4;++setting) {
            prev_t[ai][setting]=1e9;
            prev_x[ai][setting]=1e9;
        }
        for (float h:{0.8f,0.4f,0.2f}) {
            float lo[3]={1e9f,1e9f,1e9f},hi[3]={-1e9f,-1e9f,-1e9f};
            for (size_t i=0;i<a.size()/3;++i)
                for (int k=0;k<3;++k) { lo[k]=std::min(lo[k],a[3*i+k]); hi[k]=std::max(hi[k],a[3*i+k]); }
            for (size_t j=0;j<b.size()/3;++j)
                for (int k=0;k<3;++k) {
                    float p=m[3*k]*b[3*j]+m[3*k+1]*b[3*j+1]+m[3*k+2]*b[3*j+2]+t[k];
                    lo[k]=std::min(lo[k],p); hi[k]=std::max(hi[k],p);
                }
            FieldGeometry g; g.hs=h; g.inv_hs=1/h;
            g.ox=lo[0]-8;g.oy=lo[1]-8;g.oz=lo[2]-8;
            g.nx=int((hi[0]+8-g.ox)/h)+2;
            g.ny=int((hi[1]+8-g.oy)/h)+2;
            g.nz=int((hi[2]+8-g.oz)/h)+2;
            FieldNaive32 nf; build_field(a.data(),nullptr,14,g,nf);
            FieldNaive16 hf; quantize_field(nf,hf);
            FieldCoeff32 cf;make_coeff_field(nf,cf);
            FieldEdge32 ef;make_edge_field(nf,ef);
            FieldAffine32 af;make_affine(nf,af);
            Result reg=run("reg",3,nf,hf,cf,ef,af,B,soa,m,t,gn);
            double rt=relerr(reg.g,fd,1),rx=relerr(reg.g,fd,4);
            for (int ai=0;ai<8;++ai) for (int setting=0;setting<4;++setting) {
                const char* arm=arms[ai];
                if ((setting==1||setting==2) && std::strcmp(arm,"reg")) continue;
                if (std::strcmp(arm,"reg") && setting!=0&&setting!=3) continue;
                if ((!std::strcmp(arm,"unroll") || !std::strcmp(arm,"scalar")) && setting!=0) continue;
                Result r=run(arm,setting,nf,hf,cf,ef,af,B,soa,m,t,gn);
                if (perp_control && pose==0 && h<0.3f &&
                    !std::strcmp(arm,"pairaos") && setting==3) {
                    // A torque error perpendicular to reg passes the old
                    // projection-only scale check exactly. Inject at the
                    // finest spacing so the new full-vector gate must fail.
                    const double xy=std::hypot(reg.g[1],reg.g[2]);
                    const double magnitude=0.25*std::sqrt(
                        double(reg.g[1])*reg.g[1]+double(reg.g[2])*reg.g[2]+
                        double(reg.g[3])*reg.g[3]);
                    r.g[1]+=float(magnitude*reg.g[2]/xy);
                    r.g[2]-=float(magnitude*reg.g[1]/xy);
                }
                double scale_num=0,scale_den=0, xdiff=0, xnorm=0;
                for(int k=1;k<=3;++k){scale_num+=double(r.g[k])*reg.g[k];scale_den+=double(reg.g[k])*reg.g[k];}
                for(int k=4;k<=6;++k){xdiff+=double(r.g[k]-reg.g[k])*(r.g[k]-reg.g[k]);xnorm+=double(reg.g[k])*reg.g[k];}
                const double scale=scale_num/scale_den, xr=std::sqrt(xdiff/xnorm);
                // The fp16 field quantizes each stored node; the other arms
                // sample the same fp32 field and share a 1e-4 scale bar.
                const double bar=std::strcmp(arm,"fp16") ? 1e-4 : 0.001;
                if (!std::isfinite(scale)||std::abs(scale-1)>bar||xr>bar) {
                    std::printf("FAIL pose=%d h=%g arm=%s setting=%d torque_scale=%.7f translation_relerr=%.6g\n",
                        pose,h,arm,setting,scale,xr);
                    ok=false;
                }
                const double tr=relerr(r.g,fd,1), xt=relerr(r.g,fd,4);
                if (!std::isfinite(tr)||!std::isfinite(xt)||
                    tr>=prev_t[ai][setting]||xt>=prev_x[ai][setting]||
                    (h<0.3f && (tr>0.025||xt>0.025))) {
                    std::printf("FAIL convergence pose=%d h=%g arm=%s setting=%d torque_fd_relerr=%.6g prior=%.6g translation_fd_relerr=%.6g prior=%.6g\n",
                        pose,h,arm,setting,tr,prev_t[ai][setting],xt,prev_x[ai][setting]);
                    ok=false;
                }
                prev_t[ai][setting]=tr;
                prev_x[ai][setting]=xt;
            }
            std::printf("pose=%d h=%.1f reg torque_fd_relerr=%.6g translation_fd_relerr=%.6g\n",pose,h,rt,rx);
        }
    }
    std::puts(ok?"PASS T147 gradient scale":"FAIL T147 gradient scale");
    return ok?0:1;
}
