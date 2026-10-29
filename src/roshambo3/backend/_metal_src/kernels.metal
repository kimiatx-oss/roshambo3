// The experimental Apple Silicon Metal backend: every GPU kernel it runs.
//
// This file is compiled at runtime, not at build time. backend/metal.py
// prepends three #defines (below), appends one generated entry point per
// subset of color-field planes (`_color_kernel_source`, which slices this
// text between the diagonal-factored optimizer and the rescore kernel), and
// hands the result to runtime.mm, which builds it with MTLMathModeSafe and
// MTLMathFloatingPointFunctionsPrecise. Under those options the unqualified
// exp2/sin/cos/sqrt below are the precise functions; the only fast function
// is the one `search_exp2` opts into. See doc/METAL.md.
//
// Pipeline, per screen call (metal.py drives all of it):
//   query_self, library_self   self-overlaps S_AA and S_BB (exact, precise)
//   build_field, pack_field    the query's shape field, optionally as texture
//   build_color_field          per-plane color fields (dense planes only)
//   optimize*                  per-pair pose search, then exact final scoring
//   rescore                    exact scoring at caller-supplied poses
//   check_scores               nonfinite guard over the output block
//   count_self_overlap_bound   self-overlap bound scan over the output block (opt-in)
//   join_rows .. gather        device top-K (stable merge sort of indices)
//
// Invariants:
//   * Everything is in the CUDA backend's reduced frame (kernels.cuh, "The
//     reduced-overlap frame"): coordinates carry SCALE, so a shape pair term
//     is exactly exp2(-|d|^2), and physical volumes are recovered only when
//     reported, by PREF/CPREF.
//   * Four consecutive threads cooperate on one (query, library) pair and
//     combine their partial sums with the fixed XOR-2 then XOR-1 shuffle in
//     reduce4, so the threadgroup size never changes the arithmetic.
//   * FP32 throughout. Bitwise agreement with CPU/CUDA is NOT promised
//     (doc/METAL.md); acceptance is ranking agreement.
//
// Compile-time policy, set by metal.py from the context options:
//   R3_METAL_FAST_SEARCH_EXP2      fast::exp2 for optimizer color Gaussians
//                                  (default 1); 0 is the precise control
//   R3_METAL_TEXTURE_FIELDS        shape field read through a hardware-
//                                  filtered 3D texture rather than the FP32
//                                  manual trilinear reference
//   R3_METAL_COLOR_TEXTURE_FIELDS  the same choice for color-plane fields
#include <metal_stdlib>
using namespace metal;
#ifndef R3_METAL_FAST_SEARCH_EXP2
#define R3_METAL_FAST_SEARCH_EXP2 1
#endif
#ifndef R3_METAL_COLOR_TEXTURE_FIELDS
#define R3_METAL_COLOR_TEXTURE_FIELDS 0
#endif
#ifndef R3_METAL_TEXTURE_FIELDS
#define R3_METAL_TEXTURE_FIELDS 0
#endif
// The exponential used for color Gaussians INSIDE the optimizer loop only.
// Field construction, self-overlaps and the reported final scores always use
// the precise function (doc/METAL.md, "Numerical contract"). Fast search
// showed no end-to-end advantage (combo precise/fast wall ratio 0.991); fast
// stayed the shipping policy. A CPU-style
// raw-bits exponential was tried here and removed: no speedup, and it failed
// the top-100 gate.
float search_exp2(float x) {
#if R3_METAL_FAST_SEARCH_EXP2
    return metal::fast::exp2(x);
#else
    return metal::precise::exp2(x);
#endif
}

// The reduced frame's constants. Each literal rounds to the same float32 as
// its CUDA counterpart in _cuda_src/kernels.cuh; numerics.py holds the
// authoritative definitions.
//   SCALE  = ROSHAMBO3_COORD_SCALE, sqrt(alpha * log2 e): scaling coordinates
//            by it turns exp(-alpha d^2) into exp2(-|SCALE d|^2).
//   PREF   = ROSHAMBO3_SHAPE_PREFACTOR, 8 (pi/2A)^1.5: reduced shape overlap
//            to volume.
//   CPREF  = ROSHAMBO3_COLOR_PREFACTOR, (pi/2)^1.5: the same for color at the
//            canonical width r = 1.
// The prefactors cancel out of every Tanimoto, so they are applied only to the
// raw volumes written by write_score.
constant float SCALE = 0.77687365f;
constant float PREF = 20.5795288f;
constant float CPREF = 1.96870124f;

// Launch constants, packed by metal.py `_params` as 21 32-bit words and bound
// with setBytes. Field meanings:
//   nq, qwidth     query rows and padded atom slots per query row
//   nlib, width    library conformers and padded slots per conformer
//   nf             feature types, including type 0 (shape); <= 8
//   record         1: library is int16 at the wire-step; 0: float32 xyzw
//   planar         1: planar tile (tile_mode -1); 0: interleaved (tile_mode 1)
//   color          score color at all (final and self terms)
//   color_offset   first color slot in every conformer, or 0xffffffff (-1):
//                  each conformer's run starts at its own shape count
//   pairs          pairs in THIS dispatch (one pair = four threads)
//   steps          optimizer steps per start
//   starts         1, 4 or 10: the start mode's start count
//   combo          color in the optimizer gradient (IN_GRADIENT placement)
//   unused0        despite the name: 1 when the active interaction matrix is
//                  diagonal (metal.py `params[13]`). Selects the stable
//                  per-type query ranges stored after n[nq] (see exact).
//   texture_query_begin/count  query rows served by this dispatch; rows
//                  outside are skipped and texture slots are relative to begin
//   mixing         combo weight of the color Tanimoto
//   lrq, lrt       Adagrad learning rates (rotation, translation in A)
//   padding        field_pad_a; carried for the host, read by no kernel
//   pair_begin     global index of this dispatch's first pair (split
//                  dispatch); output rows stay global
struct Params {
    uint nq, qwidth, nlib, width, nf, record, planar, color;
    uint color_offset, pairs, steps, starts, combo, unused0,
        texture_query_begin, texture_query_count;
    float mixing, lrq, lrt, padding;
    uint pair_begin;
};

// One field lattice, in the reduced frame. origin_h.xyz is node (0,0,0) and
// origin_h.w the node spacing (SCALE * spacing in A); dims_offset.xyz is the
// node count per axis and dims_offset.w the index of the first node in the
// shared field buffer. Nodes are x-fastest. Built on the host by metal.py
// (`query`, `_color_grid_record`), which bounds every count to 32 bits.
struct Grid {
    float4 origin_h;
    uint4 dims_offset;
};

// Sum over the four lanes of one pair. A pair owns threads 4k..4k+3 (row =
// tid / 4, lane = tid % 4), each lane striding the library atoms by 4. Step
// one adds lanes l and l^2, step two lanes l and l^1, so every lane ends with
// the bits of (x0 + x2) + (x1 + x3): FP32 addition is commutative, so all
// four lanes hold the identical sum and step the pose identically. The order
// is fixed in the source rather than left to a reduction primitive, which is
// why the threadgroup size does not select different arithmetic
// (doc/METAL.md). runtime.mm requires threadgroups to be whole SIMD groups,
// and the SIMD width is a multiple of 4, so an aligned quad never straddles
// one. Cf. the CUDA grid kernel's four-lane
// butterfly (kernels.cu).
float reduce4(float x) {
    x += simd_shuffle_xor(x, 2);
    x += simd_shuffle_xor(x, 1);
    return x;
}

float3 reduce4(float3 x) {
    return float3(reduce4(x.x), reduce4(x.y), reduce4(x.z));
}

// Quaternions are float4(w, x, y, z), scalar first. Hamilton product a * b.
float4 mulq(float4 a, float4 b) {
    return float4(a.x * b.x - dot(a.yzw, b.yzw),
                  a.x * b.yzw + b.x * a.yzw + cross(a.yzw, b.yzw));
}

// R(q) v, with q normalized here and `scale` folded into each matrix entry
// before it multiplies v. Callers pass SCALE and a PHYSICAL library
// coordinate, so the reduced frame's scale lives in the pose matrix, the
// convention kernels.cuh documents for the CUDA int16 staging path.
float3 rotate(float4 q, float3 v, float scale = 1.f) {
    q = q / sqrt(dot(q, q));
    // Row expressions preserve the declared shared FP32 pose construction.
    float w = q.x, x = q.y, y = q.z, z = q.w;
    return float3(((1 - 2 * (y * y + z * z)) * scale) * v.x +
                      (2 * (x * y - w * z) * scale) * v.y +
                      (2 * (x * z + w * y) * scale) * v.z,
                  (2 * (x * y + w * z) * scale) * v.x +
                      ((1 - 2 * (x * x + z * z)) * scale) * v.y +
                      (2 * (y * z - w * x) * scale) * v.z,
                  (2 * (x * z - w * y) * scale) * v.x +
                      (2 * (y * z + w * x) * scale) * v.y +
                      ((1 - 2 * (x * x + y * y)) * scale) * v.z);
}

// Start rotation i of the start mode, the same table as CUDA's
// `start_quaternion` (kernels.cu): 0 is the identity; 1..3 are pi about x, y,
// z; 4..6 are +pi/2 and 7..9 -pi/2 about x, y, z. Start mode 1 (4 starts) is
// the first four entries of mode 2 (10 starts), so one formula serves every
// mode and `starts` is not read; the loop bound p.starts selects the mode.
float4 startq(uint starts, uint i) {
    if (!i) return float4(1, 0, 0, 0);
    uint axis = (i - 1) % 3;
    // Half-angle: q = (cos(theta/2), sin(theta/2) * axis).
    float a = (i <= 3 ? M_PI_F : i <= 6 ? M_PI_2_F : -M_PI_2_F) * .5f;
    float4 q = float4(cos(a), 0, 0, 0);
    q[axis + 1] = sin(a);
    return q;
}

// The exponential map: the unit quaternion (cos|d|, sin|d| d/|d|), i.e. a
// rotation by 2|d| about d. This is the rotation update of ADAGRAD_EXP.
float4 expq(float3 d) {
    float a2 = dot(d, d), c, s;
    if (a2 > 1e-12f) {
        float a = sqrt(a2);
        c = cos(a);
        s = sin(a) / a;
    } else {
        // Taylor terms near zero, avoiding sin(a)/a at a ~ 0.
        c = 1 - .5f * a2;
        s = 1 - a2 / 6.f;
    }
    return float4(c, s * d);
}

// Library atom `slot` of conformer `row`, in PHYSICAL Angstrom (the reduced
// scale is applied by the caller). The library is tiled in blocks of 32
// conformers (engine/source.py `tile_mode`):
//   interleaved (tile_mode  1): [tile][slot][lane][component]
//   planar      (tile_mode -1): [tile][slot][component][lane]
// with lane = row % 32. A record has 3 components when int16 (the corpus's
// wire format, the wire-step steps) and 4 when float32 (xyz + weight; the
// weight is not read). metal.py rejects untiled libraries.
float3 lib(device const uchar* bytes, uint row, uint slot, constant Params& p) {
    uint comp = p.record ? 3 : 4;
    size_t index = (size_t(row / 32) * p.width + slot) * 32 * comp;
    if (p.planar)
        index += row % 32;
    else
        index += (row % 32) * comp;
    // Distance between x, y and z of one atom: a whole lane row when planar.
    uint stride = p.planar ? 32 : 1;
    if (p.record) {
        // Dequantize with its own multiply; SCALE is a second, separate
        // rounding in the pose matrix. Folding the wire step and SCALE into one
        // constant changes bits (kernels.cuh, "The two constants called
        // COORD_SCALE").
        device const short* b = (device const short*)bytes;
        return float3(float(b[index]), float(b[index + stride]),
                      float(b[index + 2 * stride])) *
               R3_WIRE_QUANT_STEP;
    }
    device const float* b = (device const float*)bytes;
    return float3(b[index], b[index + stride], b[index + 2 * stride]);
}

// The count vector is uint8 [nlib][nf]: entry 0 is the conformer's shape atom
// count, entry t its count of color features of type t. The color run holds
// the features grouped by type in ascending type order, which color_type
// relies on to recover a feature's type from its position in the run.
uint ncolor(device const uchar* counts, uint row, constant Params& p) {
    uint n = 0;
    for (uint t = 1; t < p.nf; t++)
        n += counts[row * p.nf + t];
    return n;
}

uint color_type(device const uchar* counts, uint row, uint slot,
                constant Params& p) {
    uint end = 0;
    for (uint t = 1; t < p.nf; t++) {
        end += counts[row * p.nf + t];
        if (slot < end) return t;
    }
    return 0;
}

// First color slot of conformer `row`: its own shape count for the merged
// record (color_offset -1), else the launch-wide offset.
uint color_begin(device const uchar* counts, uint row, constant Params& p) {
    return p.color_offset == 0xffffffffu ? counts[row * p.nf] : p.color_offset;
}

// One color pair term, amplitude * exp2(-decay |delta|^2), with delta in the
// reduced frame. `r` and `h` are NOT the raw interaction matrices: metal.py
// prepares them once per query context as decay = r / (2 SCALE^2 ln2), i.e.
// r * INV_TWO_ALPHA, and amplitude = h^2 r^-1.5, the per-pair residual
// kernels.cuh folds into the accumulated term so CPREF stays (pi/2)^1.5.
// A pair with no positive width or zero height is inactive and contributes 0.
// `search` selects the optimizer's exponential; everything else is precise.
float color_value(float3 delta, uint ta, uint tb, device const float* r,
                  device const float* h, constant Params& p,
                  bool search = false) {
    float decay = r[ta * p.nf + tb], amplitude = h[ta * p.nf + tb];
    if (!(decay > 0) || amplitude == 0) return 0;
    float exponent = -decay * dot(delta, delta);
    return amplitude *
           (search ? search_exp2(exponent) : metal::precise::exp2(exponent));
}

// V_AB / (V_AA + V_BB - V_AB), with sum = V_AA + V_BB; 0 if degenerate.
float tanimoto(float v, float sum) {
    float den = sum - v;
    return den > 0 ? v / den : 0;
}

// Query self-overlaps (S_AA shape, S_AA color) as float2, one thread per
// query row. Row layout [qwidth]: shape atoms [0, n[row]) with type 0, then
// color features with type > 0 (or 0 for padding). Recomputed on the device
// rather than read from stored sidecars; part of an observed ranking
// mismatch traced to the CPU reference's stored self-overlaps.
kernel void query_self(device const float4* q, device const int* types,
                       device const uint* n, device const float* r,
                       device const float* h, device float2* out,
                       constant Params& p,
                       uint row [[thread_position_in_grid]]) {
    if (row >= p.nq) return;
    float2 s = 0;
    for (uint j = 0; j < n[row]; j++)
        for (uint i = 0; i < n[row]; i++) {
            float3 d =
                (q[row * p.qwidth + i].xyz - q[row * p.qwidth + j].xyz) * SCALE;
            s.x += exp2(-dot(d, d));
        }
    if (p.color)
        for (uint j = n[row]; j < p.qwidth; j++)
            if (types[row * p.qwidth + j] > 0)
                for (uint i = n[row]; i < p.qwidth; i++)
                    if (types[row * p.qwidth + i] > 0)
                        s.y += color_value((q[row * p.qwidth + i].xyz -
                                            q[row * p.qwidth + j].xyz) *
                                               SCALE,
                                           types[row * p.qwidth + i],
                                           types[row * p.qwidth + j], r, h, p);
    out[row] = s;
}

// Library self-overlaps (S_BB shape, S_BB color), four lanes per conformer
// with the outer atom loop strided by lane and combined by reduce4. Run once
// per screen call, shared by every optimizer dispatch.
kernel void library_self(device const uchar* b, device const uchar* counts,
                         device const float* r, device const float* h,
                         device float2* out, constant Params& p,
                         uint tid [[thread_position_in_grid]]) {
    uint row = tid / 4, lane = tid % 4;
    if (row >= p.nlib) return;
    uint n = counts[row * p.nf];
    float2 s = 0;
    for (uint j = lane; j < n; j += 4)
        for (uint i = 0; i < n; i++) {
            float3 d = (lib(b, row, i, p) - lib(b, row, j, p)) * SCALE;
            s.x += exp2(-dot(d, d));
        }
    if (p.color) {
        uint nc = ncolor(counts, row, p), base = color_begin(counts, row, p);
        for (uint j = lane; j < nc; j += 4)
            for (uint i = 0; i < nc; i++)
                s.y += color_value(
                    (lib(b, row, base + i, p) - lib(b, row, base + j, p)) *
                        SCALE,
                    color_type(counts, row, i, p),
                    color_type(counts, row, j, p), r, h, p);
    }
    s = float2(reduce4(s.x), reduce4(s.y));
    if (!lane) out[row] = s;
}

// The query's shape field: one thread per lattice node, over the nodes of all
// query rows laid end to end. Each node stores the PREFILTERED pair
//   (F - c lap F, dF/du - c d(lap F)/du),   c = h^2 / 12,
// where F(u) = sum_i exp2(-|u - a_i|^2) in the reduced frame and h is the
// reduced node spacing. These are the formulas of CUDA's `field_fill_kernel`
// (kernels.cu), where it was measured that the prefilter makes the signed
// interpolation bias fourth order (1.98 -> 3.98) while RMS and gradient
// errors stay second and first order; it costs nothing at query prep, but the
// exact final rescore, not the prefilter, is what keeps reported scores
// exact. Gradients are d/du in the reduced frame. Precise exp2 throughout.
kernel void build_field(device const float4* q, device const uint* n,
                        device const Grid* grids, device float4* field,
                        constant uint& nq,
                        uint tid [[thread_position_in_grid]]) {
    // Find the query row whose node range holds tid (a linear scan).
    uint row = 0;
    for (; row < nq; row++) {
        Grid g = grids[row];
        uint size = g.dims_offset.x * g.dims_offset.y * g.dims_offset.z;
        if (tid >= g.dims_offset.w && tid < g.dims_offset.w + size) break;
    }
    if (row == nq) return;
    Grid g = grids[row];
    uint k = tid - g.dims_offset.w;
    uint x = k % g.dims_offset.x, y = (k / g.dims_offset.x) % g.dims_offset.y,
         z = k / (g.dims_offset.x * g.dims_offset.y);
    float3 pos = g.origin_h.xyz + g.origin_h.w * float3(x, y, z);
    float4 f = 0, lap = 0;
    // exp2(-d2) = exp(-b d2) with b = ln2; derivatives carry powers of b.
    float b = M_LN2_F, b2 = b * b, b3 = b2 * b,
          c = g.origin_h.w * g.origin_h.w / 12.f;
    // q storage has its width in the extra n[nq] word.
    for (uint i = 0; i < n[row]; i++) {
        float3 d = pos - q[row * n[nq] + i].xyz * SCALE;
        float d2 = dot(d, d), v = metal::precise::exp2(-d2);
        f.x += v;
        f.yzw += (-2 * b * v) * d;
        lap.x += v * (4 * b2 * d2 - 6 * b);
        lap.yzw += v * (20 * b2 - 8 * b3 * d2) * d;
    }
    field[tid] = f - c * lap;
}

// A color field is defined only inside its full interpolation cells. Outside,
// use the analytic interaction rather than truncating the Gaussian tails.
bool inside_field(float3 pos, Grid g) {
    if (any(g.dims_offset.xyz < 2)) return false;
    float3 v = (pos - g.origin_h.xyz) / g.origin_h.w;
    return all(isfinite(v)) && all(v >= 0) &&
           all(v < float3(g.dims_offset.xyz - 1));
}

// One color plane's field for one query row: plane tb = config.y of row
// qr = config.x (config.z = qwidth, config.w = nf), one thread per node. The
// same prefiltered value/gradient construction as build_field, for the
// plane's own Gaussian, amplitude * exp2(-decay d^2): beta = ln2 * decay
// takes the place of ln2 (cf. UNIT_FAC = false in CUDA's field_fill_kernel).
// Only diagonal planes are built, so the self pair's (decay, amplitude) is
// the only one needed. A nonfinite node raises the shared error flag and is
// stored as 0; metal.py then fails the call. Selection (>= 12 sources,
// planes 1/2 by default) and the per-plane lattice are host-side.
kernel void build_color_field(device const float4* q, device const int* types,
                              device const uint* n, device const float* r,
                              device const float* h, device const Grid* grids,
                              device float4* field, device atomic_uint* error,
                              constant uint4& config,
                              uint tid [[thread_position_in_grid]]) {
    uint qr = config.x, tb = config.y, qwidth = config.z, nf = config.w;
    Grid g = grids[qr * nf + tb];
    if (tid >= g.dims_offset.x * g.dims_offset.y * g.dims_offset.z) return;
    uint3 index =
        uint3(tid % g.dims_offset.x, (tid / g.dims_offset.x) % g.dims_offset.y,
              tid / (g.dims_offset.x * g.dims_offset.y));
    float3 pos = g.origin_h.xyz + g.origin_h.w * float3(index);
    float decay = r[tb * nf + tb], amplitude = h[tb * nf + tb];
    float beta = M_LN2_F * decay, b2 = beta * beta, b3 = b2 * beta,
          c = g.origin_h.w * g.origin_h.w / 12.f;
    float4 f = 0, lap = 0;
    for (uint i = n[qr]; i < qwidth; i++)
        if (types[qr * qwidth + i] == int(tb)) {
            float3 d = pos - q[qr * qwidth + i].xyz * SCALE;
            float d2 = dot(d, d),
                  v = amplitude * metal::precise::exp2(-decay * d2);
            f.x += v;
            f.yzw += (-2 * beta * v) * d;
            lap.x += v * (4 * b2 * d2 - 6 * beta);
            lap.yzw += v * (20 * b2 - 8 * b3 * d2) * d;
        }
    float4 value = f - c * lap;
    if (!all(isfinite(value))) {
        atomic_store_explicit(error, 1u, memory_order_relaxed);
        value = 0;
    }
    field[g.dims_offset.w + tid] = value;
}

// The manual FP32 sampler: eight-corner trilinear interpolation of the
// stored (value, gradient) nodes at reduced-frame position pos. It is the
// FP32 reference (`field_sampler="manual"`) the texture paths are compared
// against. It returns 0 outside the complete interior cells
// [0, dims - 1) on each axis; unlike the texture path there is no half-cell
// skirt at the edge.
float4 sample_field(float3 pos, Grid g, device const float4* field) {
    float3 v = (pos - g.origin_h.xyz) / g.origin_h.w;
    // Finiteness of physical inputs does not prove representable grid indices.
    // Check in floating point before conversion; out-of-range float-to-int is
    // undefined and an overflowing lo+1 could otherwise bypass the integer guard.
    if (!all(isfinite(v)) || any(v < 0) ||
        any(v >= float3(g.dims_offset.xyz - 1)))
        return 0;
    int3 lo = int3(floor(v));
    if (any(lo < 0) || any(lo + 1 >= int3(g.dims_offset.xyz))) return 0;
    float3 f = v - float3(lo);
    float4 out = 0;
    for (uint z = 0; z < 2; z++)
        for (uint y = 0; y < 2; y++)
            for (uint x = 0; x < 2; x++) {
                uint index = g.dims_offset.w + (lo.x + x) +
                             g.dims_offset.x *
                                 ((lo.y + y) + g.dims_offset.y * (lo.z + z));
                out += field[index] * (x ? f.x : 1 - f.x) *
                       (y ? f.y : 1 - f.y) * (z ? f.z : 1 - f.z);
            }
    return out;
}

// Test hooks: sample grid config.x at config.y positions, so the samplers can
// be checked against each other and against the analytic field.
kernel void probe_field(device const float4* positions,
                        device const Grid* grids, device const float4* field,
                        device float4* out, constant uint2& config,
                        uint tid [[thread_position_in_grid]]) {
    if (tid < config.y)
        out[tid] = sample_field(positions[tid].xyz, grids[config.x], field);
}

// Hardware linear filtering uses texel centers at index+0.5. Zero-border
// addressing acts per texel, including the half-cell skirt of the volume.
//
// The native sampler (texture16 = RGBA16Float, as CUDA ships; texture32 =
// RGBA32Float, Apple GPU family 9+). Node i sits at lattice coordinate i, so
// the normalized coordinate is (v + 0.5) / dims, the same +0.5 CUDA's
// `sample_field` applies. Within one cell of the outermost nodes the result
// blends edge texels with the zero border; the early return covers the
// region where both taps on some axis lie in the border, and keeps nonfinite
// coordinates away from the sampler. Hardware interpolation rounds its
// fractional weights, so this is a search-arithmetic change even with FP32
// texels and needed its own ranking validation (doc/METAL.md).
float4 sample_texture(float3 pos, Grid g, texture3d<float> field) {
    float3 v = (pos - g.origin_h.xyz) / g.origin_h.w;
    if (!all(isfinite(v)) || any(v <= -1.f) ||
        any(v >= float3(g.dims_offset.xyz)))
        return 0;
    constexpr sampler linear_zero(coord::normalized, address::clamp_to_zero,
                                  filter::linear);
    return field.sample(linear_zero, (v + .5f) / float3(g.dims_offset.xyz),
                        level(0));
}

// Copy grid config.x's FP32 nodes into its private 3D texture, on the device.
// config.y = 1 for a half texture, where a value beyond 65504 (the largest
// finite half) would not be representable: such a node, or any nonfinite
// one, raises the error flag and is written as 0, and metal.py fails the
// call explicitly rather than search on a clipped field.
kernel void pack_field(device const Grid* grids, device const float4* field,
                       device atomic_uint* error, constant uint2& config,
                       texture3d<float, access::write> target [[texture(0)]],
                       uint tid [[thread_position_in_grid]]) {
    Grid g = grids[config.x];
    uint3 dims = g.dims_offset.xyz;
    if (tid >= dims.x * dims.y * dims.z) return;
    uint3 index =
        uint3(tid % dims.x, (tid / dims.x) % dims.y, tid / (dims.x * dims.y));
    float4 value = field[g.dims_offset.w + tid];
    if (!all(isfinite(value)) || (config.y && any(abs(value) > 65504.f))) {
        atomic_store_explicit(error, 1u, memory_order_relaxed);
        value = 0;
    }
    target.write(value, index);
}

kernel void probe_texture(device const float4* positions,
                          device const Grid* grids, device float4* out,
                          constant uint2& config,
                          texture3d<float> field [[texture(0)]],
                          uint tid [[thread_position_in_grid]]) {
    if (tid < config.y)
        out[tid] = sample_texture(positions[tid].xyz, grids[config.x], field);
}

// The exact analytic cross-overlap (S_AB shape, S_AB color) of pair
// (qr, br) at pose (pose, t), t in the reduced frame; this lane's share,
// reduced over the quad before return. Precise exponentials only: this is
// what every reported score comes from, whatever sampler or search
// exponential the optimizer used (the exact-rescore cascade in CUDA).
//
// The `n` buffer is [n_atoms per row (nq)][qwidth][ranges (nq, nf, 2)]. When
// the active matrix is diagonal (p.unused0), metal.py has stably sorted each
// row's color features by type and stored each type's [begin, end), so a
// library feature of type tb visits only matching query features. Otherwise
// the loop visits every query color slot.
float2 exact(uint qr, uint br, float4 pose, float3 t, uint lane,
             device const float4* q, device const int* types,
             device const uint* n, device const uchar* b,
             device const uchar* counts, device const float* r,
             device const float* h, constant Params& p) {
    float2 v = 0;
    for (uint j = lane; j < counts[br * p.nf]; j += 4) {
        float3 pos = rotate(pose, lib(b, br, j, p), SCALE) + t;
        for (uint i = 0; i < n[qr]; i++) {
            float3 d = q[qr * p.qwidth + i].xyz * SCALE - pos;
            v.x += exp2(-dot(d, d));
        }
    }
    if (p.color) {
        uint nc = ncolor(counts, br, p), base = color_begin(counts, br, p);
        for (uint j = lane; j < nc; j += 4) {
            float3 pos = rotate(pose, lib(b, br, base + j, p), SCALE) + t;
            uint tb = color_type(counts, br, j, p);
            uint begin =
                p.unused0 ? n[p.nq + 1 + qr * p.nf * 2 + tb * 2] : n[qr];
            uint end =
                p.unused0 ? n[p.nq + 1 + qr * p.nf * 2 + tb * 2 + 1] : p.qwidth;
            for (uint i = begin; i < end; i++) {
                int ta = types[qr * p.qwidth + i];
                if (ta <= 0) continue;
                v.y += color_value(q[qr * p.qwidth + i].xyz * SCALE - pos, ta,
                                   tb, r, h, p);
            }
        }
    }
    return float2(reduce4(v.x), reduce4(v.y));
}

// Write one 20-column score row (column names in roshambo3/scores.py):
// Tanimotos 0-2, volumes 3-8 (reduced overlaps times PREF/CPREF), pose
// quaternion 9-12, translation 13-15 in physical A (t / SCALE), and the start
// quaternion 16-19 as the identity because the start rotation is already
// folded into the pose (see optimize_body).
void write_score(device float* out, uint row, float4 pose, float3 t, float2 v,
                 float2 aa, float2 bb, constant Params& p) {
    device float* o = out + row * 20;
    float ts = tanimoto(v.x, aa.x + bb.x),
          tc = p.color ? tanimoto(v.y, aa.y + bb.y) : 0;
    o[0] = p.color ? (1 - p.mixing) * ts + p.mixing * tc : ts;
    o[1] = ts;
    o[2] = tc;
    o[3] = v.x * PREF;
    o[4] = v.y * CPREF;
    o[5] = aa.x * PREF;
    o[6] = bb.x * PREF;
    o[7] = aa.y * CPREF;
    o[8] = bb.y * CPREF;
    for (uint k = 0; k < 4; k++)
        o[9 + k] = pose[k];
    for (uint k = 0; k < 3; k++)
        o[13 + k] = t[k] / SCALE;
    o[16] = 1;
    o[17] = o[18] = o[19] = 0;
}

// The pose search for one (query, library) pair, four lanes per pair, then
// the exact score at the best pose. FIELD pose search with ADAGRAD_EXP.
//
// For each start: Adagrad on a rotation (exponential-map update, expq) and a
// translation t (reduced frame), `steps` updates, then one more evaluation
// that ranks the start. Shape reads the query's prefiltered field (texture or
// manual); color, when p.combo, is analytic per library feature, or read from
// a color-plane field for the planes in FIELD_MASK. The best start's pose
// goes to `exact` for the reported scores, so no interpolated or
// fast-exponential value is ever reported.
//
// The start rotation is applied by conjugation, as in CUDA's grid kernel
// (kernels.cu, "THE GRID PATH"): rather than rotating the query by R, the
// search keeps the one field and starts from eff = conj(q_R), i.e. R^T, with
// updates composed on the left of `pose`. The converged `eff` already
// includes the start, which is why write_score reports the identity start
// quaternion. CUDA's signed-permutation argument, that Adagrad's
// per-component cache commutes with this conjugation, covers these starts.
//
// Gradients are carried as the NEGATED gradient of each overlap divided by
// its S_AA + S_BB (CUDA's normalization, kernels.cuh), so t -= lr * g ascends.
// Torque is cross(2 * arm, g) with arm the rotated, untranslated library
// atom, as in CUDA.
//
// Template parameters are compile-time dispatch; metal.py picks the kernel:
//   DIAGONAL       the active interaction matrix is diagonal: hoist the
//                  library color count/base and the per-type (decay,
//                  amplitude), and drop the per-pair query type load.
//                  Visitation and every sum order are unchanged; outputs
//                  were found bitwise identical to the control.
//   FACTOR_TORQUE  sum each library feature's force over query features,
//                  then take ONE cross product with its arm. This reassociates
//                  the rotational FP32 sum, so trajectories can change; it
//                  passed a full-panel ranking gate and is the default.
//   FIELD_MASK     bit tb set: color plane tb is read from its field
//                  (positions inside complete cells only; elsewhere the
//                  analytic loop below). 0 = analytic color. Entry points
//                  for each mask are generated by metal.py.
template <bool DIAGONAL, bool FACTOR_TORQUE, uint FIELD_MASK = 0>
void optimize_body(device const float4* q, device const int* types,
                   device const uint* n, device const uchar* b,
                   device const uchar* counts, device const uint* pq,
                   device const uint* pb, device const float* r,
                   device const float* h, device const float2* qs,
                   device const float2* bs, device const Grid* grids,
                   device const float4* field, device float* out,
                   device const Grid* color_grids,
                   device const float4* color_field,
                   device const uint* color_slots, constant Params& p,
#if R3_METAL_TEXTURE_FIELDS || R3_METAL_COLOR_TEXTURE_FIELDS
                   thread const array<texture3d<float>, 128>& textures,
#endif
                   uint tid) {
    uint row = tid / 4, lane = tid % 4;
    if (row >= p.pairs) return;
    row += p.pair_begin;
    uint qr = pq[row], br = pb[row];
    // A dispatch binds the textures of its query rows only. For pair orders
    // that are not query-major, every pair is dispatched and pairs of other
    // rows are skipped here (metal.py `_texture_batches`).
    if (p.texture_query_count &&
        (qr < p.texture_query_begin ||
         qr >= p.texture_query_begin + p.texture_query_count))
        return;
    float2 sum = qs[qr] + bs[br];
    float best = -INFINITY;
    float4 bestq = float4(1, 0, 0, 0);
    float3 bestt = 0;
    uint color_count = p.combo ? ncolor(counts, br, p) : 0;
    uint color_base = p.combo ? color_begin(counts, br, p) : 0;
    for (uint start = 0; start < p.starts; start++) {
        float4 qstart = startq(p.starts, start);
        qstart.yzw = -qstart.yzw;  // conjugate: the inverse start rotation
        float4 pose = float4(1, 0, 0, 0);
        // cr, ct: Adagrad's per-component sums of squared gradients.
        float3 t = 0, cr = 0, ct = 0;
        // steps updates; iteration `steps` only evaluates and ranks the start.
        for (uint step = 0; step <= p.steps; step++) {
            float4 eff = mulq(pose, qstart);
            float3 gr = 0, gt = 0, cgr = 0, cgt = 0;
            float2 ov = 0;
            for (uint j = lane; j < counts[br * p.nf]; j += 4) {
                float3 arm = rotate(eff, lib(b, br, j, p), SCALE);
                float4 f;
#if R3_METAL_TEXTURE_FIELDS
                // Shape textures occupy slots [0, texture_query_count).
                f = sample_texture(arm + t, grids[qr],
                                   textures[qr - p.texture_query_begin]);
#else
                f = sample_field(arm + t, grids[qr], field);
#endif
                ov.x += f.x;
                float3 g = sum.x > 0 ? -f.yzw / sum.x : float3(0);
                gt += g;
                gr += cross(2 * arm, g);
            }
            if (p.combo) {
                // DIAGONAL is a compile-time dispatch. Preserve library/query
                // visitation, Gaussian expressions, and gradient sum order.
                uint nc = DIAGONAL ? color_count : ncolor(counts, br, p);
                uint base = DIAGONAL ? color_base : color_begin(counts, br, p);
                for (uint j = lane; j < nc; j += 4) {
                    float3 arm = rotate(eff, lib(b, br, base + j, p), SCALE);
                    uint tb = color_type(counts, br, j, p);
                    if constexpr (FIELD_MASK != 0) {
                        if (FIELD_MASK & (1u << tb)) {
                            Grid cg = color_grids[qr * p.nf + tb];
                            if (inside_field(arm + t, cg)) {
                                float4 cf;
#if R3_METAL_COLOR_TEXTURE_FIELDS
                                // Compact binding: only selected planes
                                // are bound. color_slots holds each plane's
                                // global texture index; the type-0 entry of a
                                // row holds that row's first index, so this is
                                // the slot relative to the dispatch, after
                                // the shape textures when those are bound.
                                uint slot =
                                    color_slots[qr * p.nf + tb] -
                                    color_slots[p.texture_query_begin * p.nf];
#if R3_METAL_TEXTURE_FIELDS
                                slot += p.texture_query_count;
#endif
                                cf =
                                    sample_texture(arm + t, cg, textures[slot]);
#else
                                cf = sample_field(arm + t, cg, color_field);
#endif
                                ov.y += cf.x;
                                float3 g =
                                    sum.y > 0 ? -cf.yzw / sum.y : float3(0);
                                cgt += g;
                                cgr += cross(2 * arm, g);
                                continue;
                            }
                        }
                    }
                    // Diagonal matrix: only query features of type tb.
                    uint begin = p.unused0
                                     ? n[p.nq + 1 + qr * p.nf * 2 + tb * 2]
                                     : n[qr];
                    uint end = p.unused0
                                   ? n[p.nq + 1 + qr * p.nf * 2 + tb * 2 + 1]
                                   : p.qwidth;
                    float decay = 0, amplitude = 0;
                    if constexpr (DIAGONAL) {
                        decay = r[tb * p.nf + tb];
                        amplitude = h[tb * p.nf + tb];
                        if (!(decay > 0) || amplitude == 0) continue;
                    }
                    float3 feature_force = 0;
                    for (uint i = begin; i < end; i++) {
                        float v;
                        float gradient_decay;
                        float3 d = q[qr * p.qwidth + i].xyz * SCALE - (arm + t);
                        if constexpr (DIAGONAL) {
                            float exponent = -decay * dot(d, d);
                            v = amplitude * search_exp2(exponent);
                            gradient_decay = decay;
                        } else {
                            int ta = types[qr * p.qwidth + i];
                            if (ta <= 0) continue;
                            v = color_value(d, ta, tb, r, h, p, true);
                            gradient_decay = r[ta * p.nf + tb];
                        }
                        ov.y += v;
                        // -dv/dpos / S, since d/dx exp2(-k x^2) is
                        // -2 ln2 k x exp2(-k x^2).
                        float3 g =
                            sum.y > 0
                                ? (-2 * M_LN2_F * gradient_decay * v / sum.y) *
                                      d
                                : float3(0);
                        // Keep translation summation in its original order.
                        cgt += g;
                        if constexpr (FACTOR_TORQUE)
                            feature_force += g;
                        else
                            cgr += cross(2 * arm, g);
                    }
                    // arm is invariant across query features. This reassociates
                    // rotational FP32 sums, so it is a separately selectable arm.
                    if constexpr (FACTOR_TORQUE)
                        cgr += cross(2 * arm, feature_force);
                }
            }
            ov = float2(reduce4(ov.x), reduce4(ov.y));
            gr = reduce4(gr);
            gt = reduce4(gt);
            if (step == p.steps) {
                // Without combo the sum is fixed per pair, so the raw shape
                // overlap orders starts the same way its Tanimoto would.
                float key = p.combo ? (1 - p.mixing) * tanimoto(ov.x, sum.x) +
                                          p.mixing * tanimoto(ov.y, sum.y)
                                    : ov.x;
                if (key > best) {
                    best = key;
                    bestq = eff;
                    bestt = t;
                }
                break;
            }
            if (p.combo) {
                cgr = reduce4(cgr);
                cgt = reduce4(cgt);
                // Turn the S-normalized overlap gradients into the gradient of
                // the mixed Tanimoto: dT/dV = S / (S - V)^2 for T = V/(S - V).
                float ds = sum.x - ov.x, dc = sum.y - ov.y;
                float ws =
                    ds > 0 ? (1 - p.mixing) * sum.x * sum.x / (ds * ds) : 0;
                float wc = dc > 0 ? p.mixing * sum.y * sum.y / (dc * dc) : 0;
                gr = ws * gr + wc * cgr;
                gt = ws * gt + wc * cgt;
            }
            // Adagrad. t is in the reduced frame, so the translation step is
            // scaled by SCALE (CUDA's host scales lr_t the same way).
            cr += gr * gr;
            ct += gt * gt;
            float3 d = -p.lrq * gr / (sqrt(cr) + 1e-8f);
            t -= p.lrt * SCALE * gt / (sqrt(ct) + 1e-8f);
            pose = mulq(expq(d), pose);
        }
    }
    float2 v =
        exact(qr, br, bestq, bestt, lane, q, types, n, b, counts, r, h, p);
    if (!lane) write_score(out, row, bestq, bestt, v, qs[qr], bs[br], p);
}

// The four optimizer entry points, one per (DIAGONAL, FACTOR_TORQUE). All
// share one binding layout: buffers 0-13 as listed, Params at index 14 (the
// runtime binds the constants after the last buffer), shape/color textures
// from texture index 0. The color-field arguments of optimize_body receive
// placeholders here (FIELD_MASK = 0 never reads them). The diagonal-factored
// entry point is also the template metal.py copies to generate the
// color-field variants, so its text is load-bearing; see that script before
// editing it.
kernel void optimize(device const float4* q, device const int* types,
                     device const uint* n, device const uchar* b,
                     device const uchar* counts, device const uint* pq,
                     device const uint* pb, device const float* r,
                     device const float* h, device const float2* qs,
                     device const float2* bs, device const Grid* grids,
                     device const float4* field, device float* out,
                     constant Params& p,
#if R3_METAL_TEXTURE_FIELDS || R3_METAL_COLOR_TEXTURE_FIELDS
                     array<texture3d<float>, 128> textures [[texture(0)]],
#endif
                     uint tid [[thread_position_in_grid]]) {
    optimize_body<false, false>(q, types, n, b, counts, pq, pb, r, h, qs, bs,
                                grids, field, out, grids, field, n, p,
#if R3_METAL_TEXTURE_FIELDS || R3_METAL_COLOR_TEXTURE_FIELDS
                                textures,
#endif
                                tid);
}

kernel void optimize_diagonal(
    device const float4* q, device const int* types, device const uint* n,
    device const uchar* b, device const uchar* counts, device const uint* pq,
    device const uint* pb, device const float* r, device const float* h,
    device const float2* qs, device const float2* bs, device const Grid* grids,
    device const float4* field, device float* out, constant Params& p,
#if R3_METAL_TEXTURE_FIELDS || R3_METAL_COLOR_TEXTURE_FIELDS
    array<texture3d<float>, 128> textures [[texture(0)]],
#endif
    uint tid [[thread_position_in_grid]]) {
    optimize_body<true, false>(q, types, n, b, counts, pq, pb, r, h, qs, bs,
                               grids, field, out, grids, field, n, p,
#if R3_METAL_TEXTURE_FIELDS || R3_METAL_COLOR_TEXTURE_FIELDS
                               textures,
#endif
                               tid);
}

kernel void optimize_factored(
    device const float4* q, device const int* types, device const uint* n,
    device const uchar* b, device const uchar* counts, device const uint* pq,
    device const uint* pb, device const float* r, device const float* h,
    device const float2* qs, device const float2* bs, device const Grid* grids,
    device const float4* field, device float* out, constant Params& p,
#if R3_METAL_TEXTURE_FIELDS || R3_METAL_COLOR_TEXTURE_FIELDS
    array<texture3d<float>, 128> textures [[texture(0)]],
#endif
    uint tid [[thread_position_in_grid]]) {
    optimize_body<false, true>(q, types, n, b, counts, pq, pb, r, h, qs, bs,
                               grids, field, out, grids, field, n, p,
#if R3_METAL_TEXTURE_FIELDS || R3_METAL_COLOR_TEXTURE_FIELDS
                               textures,
#endif
                               tid);
}

kernel void optimize_diagonal_factored(
    device const float4* q, device const int* types, device const uint* n,
    device const uchar* b, device const uchar* counts, device const uint* pq,
    device const uint* pb, device const float* r, device const float* h,
    device const float2* qs, device const float2* bs, device const Grid* grids,
    device const float4* field, device float* out, constant Params& p,
#if R3_METAL_TEXTURE_FIELDS || R3_METAL_COLOR_TEXTURE_FIELDS
    array<texture3d<float>, 128> textures [[texture(0)]],
#endif
    uint tid [[thread_position_in_grid]]) {
    optimize_body<true, true>(q, types, n, b, counts, pq, pb, r, h, qs, bs,
                              grids, field, out, grids, field, n, p,
#if R3_METAL_TEXTURE_FIELDS || R3_METAL_COLOR_TEXTURE_FIELDS
                              textures,
#endif
                              tid);
}

kernel void rescore(device const float4* q, device const int* types,
                    device const uint* n, device const uchar* b,
                    device const uchar* counts, device const uint* pq,
                    device const uint* pb, device const float* r,
                    device const float* h, device const float2* qs,
                    device const float2* bs, device const float* poses,
                    device float* out, constant Params& p,
                    uint tid [[thread_position_in_grid]]) {
    // Fixed-pose replay: the exact score of each pair at a caller-supplied
    // pose [w, x, y, z, tx, ty, tz] (translation in physical A), four lanes
    // per pair. No search and no fields. Dispatched when the caller passes
    // `poses` to metal.py `score_pairs`, e.g. to replay reported poses.
    uint row = tid / 4, lane = tid % 4;
    if (row >= p.pairs) return;
    device const float* z = poses + row * 7;
    float4 pose = float4(z[0], z[1], z[2], z[3]);
    float3 t = float3(z[4], z[5], z[6]) * SCALE;
    float2 v =
        exact(pq[row], pb[row], pose, t, lane, q, types, n, b, counts, r, h, p);
    if (!lane) write_score(out, row, pose, t, v, qs[pq[row]], bs[pb[row]], p);
}

// A DENSE launch's pair list, written on the device: row q * nlib + t
// is query q against library row t. metal.py dispatches this ahead of the
// optimizer for a launch given no pair list, so nothing is built on the host;
// the optimizer and rescore kernels read pq/pb exactly as they read an
// uploaded list, which leaves their code untouched. p[0] = nq, p[1] = nlib.
kernel void dense_pairs(device uint* pq, device uint* pb, constant uint* p,
                        uint tid [[thread_position_in_grid]]) {
    if (tid >= p[0] * p[1]) return;
    uint q = tid / p[1];
    pq[tid] = q;
    pb[tid] = tid - q * p[1];
}

// Stable merge sorting of indices: each element binary-searches the opposite
// run for its output position. IDs are 64-bit values, never float encodings.
//
// Device top-K, driven by metal.py `MetalTopK.ingest` once per query per
// library block. join_rows appends the block's fresh rows to the retained
// top-K rows; log2(n) merge_indices passes sort an index permutation; gather
// copies the first K rows into fresh retained buffers. Molecule level adds a
// first sort keyed on molecule id, then deduplicate keeps each molecule's
// best row and sets the others' rank score to -inf, and a second sort by score
// puts them last (finalize drops nonfinite rows). doc/METAL.md: this replaces
// CUDA's radix sort and is checked against host sorting.
//
// Sort is the per-pass constant block: n rows, run width, rank column, and
// whether to key on molecule id first.
struct Sort {
    uint n, width, rank, molecule;
};

// Strict ordering of rows a and b: in the molecule pass, ascending molecule
// id first; then descending score, then ascending 64-bit
// GID (the repo's one tie-break, numerics.rank_order), then row index. The
// final index comparison makes this a total order with no equal pairs, so the
// merge below is stable without needing a tie rule of its own.
bool before(uint a, uint b, device const float* scores,
            device const ulong* gids, device const ulong* mids,
            constant Sort& p) {
    if (p.molecule && mids[a] != mids[b]) return mids[a] < mids[b];
    float sa = scores[a * 20 + p.rank], sb = scores[b * 20 + p.rank];
    if (sa != sb) return sa > sb;
    if (gids[a] != gids[b]) return gids[a] < gids[b];
    return a < b;
}

kernel void init_indices(device uint* indices, constant uint& n,
                         uint tid [[thread_position_in_grid]]) {
    if (tid < n) indices[tid] = tid;
}

kernel void merge_indices(device const uint* input, device uint* output,
                          device const float* scores, device const ulong* gids,
                          device const ulong* mids, constant Sort& p,
                          uint tid [[thread_position_in_grid]]) {
    // One thread per element. Runs of p.width are merged in pairs: an element
    // of the left run [base, mid) counts the right-run elements before it, and
    // vice versa; its output position is its own rank plus that count.
    if (tid >= p.n) return;
    uint base = (tid / (2 * p.width)) * (2 * p.width),
         mid = min(base + p.width, p.n), end = min(base + 2 * p.width, p.n);
    bool left = tid < mid;
    uint lo = left ? mid : base, hi = left ? end : mid, begin = lo,
         a = input[tid];
    while (lo < hi) {
        uint m = (lo + hi) / 2;
        if (before(input[m], a, scores, gids, mids, p))
            lo = m + 1;
        else
            hi = m;
    }
    uint own = tid - (left ? base : mid);
    output[base + own + lo - begin] = a;
}

// After the molecule-keyed sort, each molecule's rows are contiguous with its
// best row first; every later row of the same molecule is demoted to -inf.
kernel void deduplicate(device const uint* indices, device float* scores,
                        device const ulong* mids, constant Sort& p,
                        uint tid [[thread_position_in_grid]]) {
    if (tid && tid < p.n && mids[indices[tid]] == mids[indices[tid - 1]])
        scores[indices[tid] * 20 + p.rank] = -INFINITY;
}

// Copy the first sizes.y rows of the sorted order (the retained top-K).
kernel void gather(device const uint* indices, device const float* scores,
                   device const ulong* gids, device const ulong* mids,
                   device float* selected, device ulong* outids,
                   device ulong* outmids, constant uint2& sizes,
                   uint tid [[thread_position_in_grid]]) {
    if (tid >= sizes.y) return;
    uint i = indices[tid];
    for (uint k = 0; k < 20; k++)
        selected[tid * 20 + k] = scores[i * 20 + k];
    outids[tid] = gids[i];
    outmids[tid] = mids[i];
}

// Concatenate retained rows [0, p[1]) and this query's fresh rows.
// p = [n_fresh, n_old, first fresh output row (qi * nlib), GID base low,
// GID base high]; a fresh row's GID is the 64-bit base plus its library row,
// and its molecule id comes from the block's sidecar.
kernel void join_rows(device const float* fresh, device const ulong* mols,
                      device const float* old, device const ulong* oldids,
                      device const ulong* oldmids, device float* joined,
                      device ulong* gids, device ulong* mids, constant uint* p,
                      uint tid [[thread_position_in_grid]]) {
    if (tid >= p[0] + p[1]) return;
    if (tid < p[1]) {
        for (uint k = 0; k < 20; k++)
            joined[tid * 20 + k] = old[tid * 20 + k];
        gids[tid] = oldids[tid];
        mids[tid] = oldmids[tid];
    } else {
        // The block is complete and query-major (MetalTopK.ingest checks a
        // pair list when one is given), so row p[2] + j is library row j and
        // no pair list is bound.
        uint row = p[2] + tid - p[1], br = tid - p[1];
        for (uint k = 0; k < 20; k++)
            joined[tid * 20 + k] = fresh[row * 20 + k];
        gids[tid] = (ulong(p[4]) << 32) + p[3] + br;
        mids[tid] = mols[br];
    }
}

// Count the rows of the output block past the self-overlap bound,
// 2 V_AB > (V_AA + V_BB)(1 + margin), over every row before any top-K ingest,
// so a row the denominator clamp reports as 0 is counted too. The mirror of
// CUDA's `self_overlap_bound_scan` (topk.cu), with one difference: MSL has no
// double, so the comparison is in float32 where CUDA and the CPU widen to
// double. At a 1e-3 or 1e-5 relative margin the two predicates can disagree
// only on a row within float32 rounding (~2e-7 relative) of the margin itself.
// p.x is the row count, p.y the margin's float32 bits: metal.py passes the
// recompute route's margin from roshambo3.scores (no copy of it here). out[0] counts flagged
// rows, out[1] receives the lowest flagged row (metal.py initializes it to
// 0xFFFFFFFF).
kernel void count_self_overlap_bound(device const float* scores,
                                     device atomic_uint* out,
                                     constant uint2& p,
                                     uint row [[thread_position_in_grid]]) {
    if (row >= p.x) return;
    device const float* s = scores + row * 20;
    float selves = s[5] + s[6];
    if (2.0f * s[3] > selves * (1.0f + as_type<float>(p.y))) {
        atomic_fetch_add_explicit(&out[0], 1u, memory_order_relaxed);
        atomic_fetch_min_explicit(&out[1], row, memory_order_relaxed);
    }
}

// Flag any nonfinite value in the output block; metal.py raises on it.
kernel void check_scores(device const float* scores, device atomic_uint* error,
                         constant uint& size,
                         uint tid [[thread_position_in_grid]]) {
    if (tid < size && !isfinite(scores[tid]))
        atomic_store_explicit(error, 1, memory_order_relaxed);
}
