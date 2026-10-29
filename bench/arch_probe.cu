////////////////////////////////////////////////////////////////////////////////
/// Does the engine actually run on this GPU, and does it agree with another one?
///
/// A standalone harness with no Python dependency, so it can be built and run on
/// a machine that has a GPU and a CUDA toolkit and nothing else. It exists for
/// two questions the source alone cannot answer:
///
///   1. **Does it run here?** Compiling for an architecture says nothing about
///      the runtime requirements -- the shared-memory opt-in and the shared/L1
///      carveout are the ones that bite, and their limits vary by a factor of
///      three and a half across the architectures this code compiles for.
///
///   2. **Does it agree with another architecture?** Scores are known NOT to be
///      bitwise portable across GPUs; what is portable is the ranked output.
///      This prints both the scores and the ranking so the two claims can be
///      checked separately rather than conflated.
///
/// The input is generated deterministically in-process from a fixed seed, so two
/// machines produce byte-identical input without transferring a file -- which is
/// the whole point, since a difference in the input would be indistinguishable
/// from a difference in the arithmetic.
///
///     nvcc -std=c++17 -arch=sm_70 -o arch_probe arch_probe.cu kernels.cu topk.cu
///     ./arch_probe > volta.txt
///
/// Then diff two hosts' output.
////////////////////////////////////////////////////////////////////////////////

#include <cuda_runtime.h>
#include <channel_descriptor.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>

#include "kernels.cuh"

#define CK(call) do { cudaError_t _e = (call); if (_e != cudaSuccess) { \
    std::printf("CUDA ERROR %s at %s:%d\n", cudaGetErrorString(_e), __FILE__, __LINE__); \
    std::exit(2);} } while (0)

// kernels.cuh defines a __constant__ `D`; this host-side stride needs its
// own name, and `DV` is the score-row width.
static const int STRIDE = 4;
static const int DV = 20;

/// splitmix64, so the input is identical on every host and compiler. A
/// std::mt19937 would also be reproducible, but a hand-rolled integer generator
/// removes any question about library versions.
static unsigned long long s_state = 0x9E3779B97F4A7C15ull;
static double next_uniform(){
    s_state += 0x9E3779B97F4A7C15ull;
    unsigned long long z = s_state;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    z = z ^ (z >> 31);
    return (double)(z >> 11) / 9007199254740992.0;   // [0,1)
}

/// A crude molecule: n atoms scattered in a box, centered. Not chemistry -- the
/// question here is whether the kernels run and agree, and a synthetic set is
/// fine for that. It would NOT be fine for an accuracy claim.
static void make_mol(float * out, int n, int L, double spread){
    double cx = 0, cy = 0, cz = 0;
    std::vector<double> x(n), y(n), z(n);
    for (int i = 0; i < n; ++i){
        x[i] = (next_uniform() - 0.5) * spread;
        y[i] = (next_uniform() - 0.5) * spread;
        z[i] = (next_uniform() - 0.5) * spread;
        cx += x[i]; cy += y[i]; cz += z[i];
    }
    cx /= n; cy /= n; cz /= n;
    std::memset(out, 0, (size_t)L * STRIDE * sizeof(float));
    for (int i = 0; i < n; ++i){
        out[i*STRIDE+0] = (float)(x[i] - cx);
        out[i*STRIDE+1] = (float)(y[i] - cy);
        out[i*STRIDE+2] = (float)(z[i] - cz);
        out[i*STRIDE+3] = 1.0f;
    }
}

/// The transposed 32-lane tile layout the kernels read. Must match
/// roshambo3.layout.tile32 and ROSHAMBO3_BTILE.
static void tile32(const float * in, float * out, int n_rows, int L){
    const int B = ROSHAMBO3_BTILE;
    int n_tiles = n_rows / B;
    for (int t = 0; t < n_tiles; ++t)
        for (int lane = 0; lane < B; ++lane)
            for (int a = 0; a < L; ++a)
                for (int c = 0; c < STRIDE; ++c)
                    out[(((size_t)t*L + a)*B + lane)*STRIDE + c] =
                        in[(((size_t)t*B + lane)*L + a)*STRIDE + c];
}

int main(int argc, char ** argv){
    const int device_id = (argc > 1) ? std::atoi(argv[1]) : 0;
    CK(cudaSetDevice(device_id));

    cudaDeviceProp prop{};
    CK(cudaGetDeviceProperties(&prop, device_id));
    int optin = 0;
    CK(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device_id));
    int rt = 0, drv = 0;
    cudaRuntimeGetVersion(&rt); cudaDriverGetVersion(&drv);
    std::printf("# device            %s\n", prop.name);
    std::printf("# compute           %d.%d\n", prop.major, prop.minor);
    std::printf("# SMs               %d\n", prop.multiProcessorCount);
    std::printf("# smem/block optin  %d bytes (%.0f KB)\n", optin, optin/1024.0);
    std::printf("# cuda runtime      %d\n", rt);
    std::printf("# cuda driver       %d\n", drv);

    // ---- the fixed problem -------------------------------------------------
    const int NQ = 4;
    const int N_REAL = 24;
    const int L = 24;                       // pad_L(24) with L_ALIGN 8
    const int NT = 64;                      // whole tiles
    const int N_ROWS = NQ * NT;

    std::vector<float> hq((size_t)NQ*L*STRIDE, 0.0f), hl((size_t)NT*L*STRIDE, 0.0f);
    for (int q = 0; q < NQ; ++q) make_mol(&hq[(size_t)q*L*STRIDE], N_REAL, L, 8.0);
    for (int j = 0; j < NT; ++j) make_mol(&hl[(size_t)j*L*STRIDE], N_REAL, L, 8.0);
    // library molecule 0 is a copy of query 0: its self-overlap must score ~1.
    std::memcpy(&hl[0], &hq[0], (size_t)L*STRIDE*sizeof(float));

    std::vector<float> hl_tiled((size_t)NT*L*STRIDE, 0.0f);
    tile32(hl.data(), hl_tiled.data(), NT, L);

    std::vector<int> hqn(NQ, N_REAL), hln(NT, N_REAL);
    std::vector<int> hqt((size_t)NQ*L, 0), hlt((size_t)NT*L, 0);
    std::vector<int> h_outl(N_ROWS), h_outq(N_ROWS);
    for (int q = 0; q < NQ; ++q)
        for (int j = 0; j < NT; ++j){ h_outq[q*NT+j] = q; h_outl[q*NT+j] = j; }
    std::vector<float> rmat(1, 1.0f), pmat(1, 0.0f);

    // ---- device -------------------------------------------------------------
    float *dq, *dl, *drm, *dpm, *dva, *dvb, *dsc, *dq_scaled;
    int *dqn, *dln, *dqt, *dlt, *doutl;
    CK(cudaMalloc(&dq,  hq.size()*sizeof(float)));
    CK(cudaMalloc(&dq_scaled, hq.size()*sizeof(float)));
    CK(cudaMalloc(&dl,  hl_tiled.size()*sizeof(float)));
    CK(cudaMalloc(&dqn, NQ*sizeof(int)));   CK(cudaMalloc(&dln, NT*sizeof(int)));
    CK(cudaMalloc(&dqt, hqt.size()*sizeof(int)));
    CK(cudaMalloc(&dlt, hlt.size()*sizeof(int)));
    CK(cudaMalloc(&doutl, N_ROWS*sizeof(int)));
    CK(cudaMalloc(&drm, sizeof(float)));    CK(cudaMalloc(&dpm, sizeof(float)));
    CK(cudaMalloc(&dva, NQ*2*sizeof(float)));
    CK(cudaMalloc(&dvb, NT*2*sizeof(float)));
    CK(cudaMalloc(&dsc, (size_t)N_ROWS*DV*sizeof(float)));
    CK(cudaMemcpy(dq, hq.data(), hq.size()*sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dl, hl_tiled.data(), hl_tiled.size()*sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dqn, hqn.data(), NQ*sizeof(int), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dln, hln.data(), NT*sizeof(int), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dqt, hqt.data(), hqt.size()*sizeof(int), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dlt, hlt.data(), hlt.size()*sizeof(int), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(doutl, h_outl.data(), N_ROWS*sizeof(int), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(drm, rmat.data(), sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dpm, pmat.data(), sizeof(float), cudaMemcpyHostToDevice));

    // self-overlaps, exactly as interface.cpp does them
    compute_self_overlaps_gpu(dl, dlt, dln, L, NT, drm, dpm, 1, false,
                              ROSHAMBO3_COORD_SCALE, dvb, device_id, 1);
    prescale_coords_gpu(dq, dq_scaled, L, NQ, device_id);
    compute_self_overlaps_gpu(dq_scaled, dqt, dqn, L, NQ, drm, dpm, 1, false,
                              1.0f, dva, device_id);

    const float LR_Q = 0.1f, LR_T = 0.4f;
    const int STEPS = 25, START_MODE = 1;

    // ---- work items --------------------------------------------------------
    auto build_wi = [&](long chunk, std::vector<int>& q, std::vector<int>& s,
                        std::vector<int>& c){
        long i = 0;
        while (i < N_ROWS){
            int qq = h_outq[i]; long j = i;
            while (j < N_ROWS && h_outq[j] == qq) ++j;
            for (long k = i; k < j; k += chunk){
                q.push_back(qq); s.push_back((int)k);
                c.push_back((int)std::min(chunk, j - k));
            }
            i = j;
        }
    };
    auto upload_wi = [&](const std::vector<int>& q, const std::vector<int>& s,
                         const std::vector<int>& c, int** dqw, int** dsw, int** dcw){
        CK(cudaMalloc(dqw, q.size()*sizeof(int)));
        CK(cudaMalloc(dsw, s.size()*sizeof(int)));
        CK(cudaMalloc(dcw, c.size()*sizeof(int)));
        CK(cudaMemcpy(*dqw, q.data(), q.size()*sizeof(int), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(*dsw, s.data(), s.size()*sizeof(int), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(*dcw, c.data(), c.size()*sizeof(int), cudaMemcpyHostToDevice));
    };

    std::vector<float> out((size_t)N_ROWS*DV);

    // ---- the field path ----------------------------------------------------
    {
        std::vector<int> wq, ws, wc; build_wi(NTHREADS/4, wq, ws, wc);
        int *dwq,*dws,*dwc; upload_wi(wq, ws, wc, &dwq, &dws, &dwc);

        std::vector<QueryField> hf(NQ);
        std::vector<cudaArray_t> arrs(NQ, nullptr);
        std::vector<cudaTextureObject_t> texs(NQ, 0);
        size_t max_nodes = 0;
        std::vector<FieldGeometry> geoms(NQ);
        for (int q = 0; q < NQ; ++q){
            geoms[q] = plan_field_lattice(&hq[(size_t)q*L*STRIDE], N_REAL, L, 0.8f, 8.0f);
            size_t nn = (size_t)(geoms[q].nx+1)*(geoms[q].ny+1)*(geoms[q].nz+1);
            max_nodes = std::max(max_nodes, nn);
        }
        void * dnodes = nullptr; CK(cudaMalloc(&dnodes, max_nodes*8));
        cudaChannelFormatDesc ch = cudaCreateChannelDescHalf4();
        for (int q = 0; q < NQ; ++q){
            FieldGeometry G = geoms[q];
            int NX=G.nx+1, NY=G.ny+1, NZ=G.nz+1;
            build_query_field(dq + (size_t)q*L*STRIDE, N_REAL, L, G, dnodes, device_id);
            cudaArray_t arr=nullptr;
            CK(cudaMalloc3DArray(&arr, &ch, make_cudaExtent(NX,NY,NZ), 0));
            cudaMemcpy3DParms p; std::memset(&p,0,sizeof(p));
            p.srcPtr = make_cudaPitchedPtr(dnodes, (size_t)NX*8, NX, NY);
            p.dstArray = arr; p.extent = make_cudaExtent(NX,NY,NZ);
            p.kind = cudaMemcpyDeviceToDevice;
            CK(cudaMemcpy3D(&p));
            cudaResourceDesc rd; std::memset(&rd,0,sizeof(rd));
            rd.resType = cudaResourceTypeArray; rd.res.array.array = arr;
            cudaTextureDesc td; std::memset(&td,0,sizeof(td));
            td.addressMode[0]=td.addressMode[1]=td.addressMode[2]=cudaAddressModeBorder;
            td.filterMode = cudaFilterModeLinear;
            td.readMode = cudaReadModeElementType;
            td.normalizedCoords = 0;
            cudaTextureObject_t tex=0;
            CK(cudaCreateTextureObject(&tex,&rd,&td,NULL));
            arrs[q]=arr; texs[q]=tex;
            QueryField f; f.tex=(unsigned long long)tex;
            f.ox=G.ox; f.oy=G.oy; f.oz=G.oz; f.inv_hs=1.0f/G.hs;
            f.nx=G.nx; f.ny=G.ny; f.nz=G.nz; hf[q]=f;
        }
        QueryField * dfields=nullptr;
        CK(cudaMalloc(&dfields, NQ*sizeof(QueryField)));
        CK(cudaMemcpy(dfields, hf.data(), NQ*sizeof(QueryField), cudaMemcpyHostToDevice));

        optimize_pairs_field(dq, dqn, L, dl, dln, L, doutl,
                             dwq, dws, dwc, (int)wq.size(), dfields,
                             dsc, LR_Q, LR_T, STEPS, START_MODE, dva, dvb,
                             /*upd*/2, /*carveout*/50, device_id, /*tier*/0, /*minb*/4,
                             // ADAGRAD_EXP is the default. This probe is a
                             // cross-HOST diff, so it stays pinned to the rule its
                             // earlier outputs were taken at -- and to the same rule
                             // as the analytic arm below, so the two are comparable.
                             /*opt_mode*/UPDATE_ADAGRAD_QADD);
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(out.data(), dsc, out.size()*sizeof(float), cudaMemcpyDeviceToHost));
        std::printf("# path field\n");
        for (int r = 0; r < N_ROWS; ++r)
            std::printf("field %3d %3d %.9e\n", h_outq[r], h_outl[r], out[(size_t)r*DV+1]);
        for (int q = 0; q < NQ; ++q){
            if (texs[q]) cudaDestroyTextureObject(texs[q]);
            if (arrs[q]) cudaFreeArray(arrs[q]);
        }
        cudaFree(dnodes); cudaFree(dfields);
        cudaFree(dwq); cudaFree(dws); cudaFree(dwc);
    }

    // ---- the analytic path -------------------------------------------------
    {
        const int stage_w = (optin >= 256*L*6) ? 256 : ((optin >= 128*L*6) ? 128 : 0);
        std::printf("# analytic stage_width %d (needs %d bytes, have %d)\n",
                    stage_w, stage_w*L*6, optin);
        std::vector<int> wq, ws, wc;
        build_wi(stage_w > 0 ? stage_w : NTHREADS, wq, ws, wc);
        int *dwq,*dws,*dwc; upload_wi(wq, ws, wc, &dwq, &dws, &dwc);
        optimize_pairs_analytic(dq, dqt, dqn, L, dl, dlt, dln, L, doutl,
                                dwq, dws, dwc, (int)wq.size(),
                                drm, dpm, 1, dsc, false, LR_Q, LR_T, STEPS, 0.0f,
                                START_MODE, device_id, false, dva, dvb,
                                /*tile_mode*/1, /*qvec*/true, stage_w,
                                /*stage_i16*/true, /*xhoist*/true, /*cut2*/0.0f,
                                /*now*/true, /*stage_p3*/true, /*opt_mode*/0);
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(out.data(), dsc, out.size()*sizeof(float), cudaMemcpyDeviceToHost));
        std::printf("# path analytic\n");
        for (int r = 0; r < N_ROWS; ++r)
            std::printf("analytic %3d %3d %.9e\n", h_outq[r], h_outl[r], out[(size_t)r*DV+1]);
        cudaFree(dwq); cudaFree(dws); cudaFree(dwc);
    }

    std::printf("# done\n");
    return 0;
}
