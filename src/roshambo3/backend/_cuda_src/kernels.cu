////////////////////////////////////////////////////////////////////////////////
/// ROSHAMBO3 device kernels.
///
/// Two scoring kernels ship, not eleven:
///
///   optimize_pairs_field     the production path. A precomputed query field on
///                            a lattice drives the pose search; the converged
///                            pose is then rescored with the exact analytic pair
///                            sum in the SAME launch. Tier 2 and Tier 3 are
///                            fused by hand here -- there is no stage graph and
///                            no framework doing it.
///
///   optimize_pairs_analytic  the exact analytic optimizer. This is the Tier-3
///                            specification, the path that carries color, and
///                            what ground truth is generated with.
///
/// ROSHAMBO2 additionally shipped a dense kernel and threadwise, warpwise and
/// blockwise-warp variants. Those were measurement arms, not scoring paths.
/// They are not ported.
///
/// **Read doc/NUMERICS_POLICY.md before changing anything in this file.** A
/// reassociated sum here moves the reported score by as much as --use_fast_math
/// does, and the kernels work in a reduced overlap frame documented in
/// kernels.cuh rather than in volume units.
///
/// Identifiers here are named for what they do.
////////////////////////////////////////////////////////////////////////////////

// MIT License
// 
// Copyright (c) 2026 Kimia Therapeutics
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.


#include <stdio.h>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>
#include <type_traits>
#include <cuda_fp16.h>
#include "kernels.cuh"



////////////////////////////////////////////////////////////////////////////////
/// Math helper functions
////////////////////////////////////////////////////////////////////////////////


// `stride` is the distance in floats between vec's components. It is 1 for a
// packed 3-vector and 32 for the planar library tile, where one library's x, y
// and z are 32 floats apart because the 32 lanes' x components sit together.
// The values read and the order they are combined in are unchanged, so this is
// a layout parameter and not an arithmetic one.
__device__ void matvec3x3x3(float mat[3][3], const float * vec, float * result,
                            int stride = 1){

    result[0] = 0.0;
    result[1] = 0.0;
    result[2] = 0.0;

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            result[i] += mat[i][j] * vec[j*stride];
        }
    }
}

__device__ void cross_product(const float a[3], const float b[3], float c[3]){
    c[0] = a[1] * b[2] - a[2] * b[1];
    c[1] = a[2] * b[0] - a[0] * b[2];
    c[2] = a[0] * b[1] - a[1] * b[0];
}


__host__ __device__ void quaternion_to_rotation_matrix(const float * q, float M[3][3]){

    // temp variables to make more readable
    float w = q[0];
    float x = q[1];
    float y = q[2];
    float z = q[3];

    // Compute rotation matrix elements
    M[0][0] = 1 - 2*y*y - 2*z*z;
    M[0][1] = 2*x*y - 2*w*z;
    M[0][2] = 2*x*z + 2*w*y;
    M[1][0] = 2*x*y + 2*w*z;
    M[1][1] = 1 - 2*x*x - 2*z*z;
    M[1][2] = 2*y*z - 2*w*x;
    M[2][0] = 2*x*z - 2*w*y;
    M[2][1] = 2*y*z + 2*w*x;
    M[2][2] = 1 - 2*x*x - 2*y*y;
}

// Scale a 3x3 matrix in place. Used to fold the reduced-frame coordinate scale
// (see kernels.cuh) into a rotation that is about to be applied, so the
// scale never appears as its own pass over the coordinates.
__device__ void scale_matrix3x3(float M[3][3], float s){
    #pragma unroll
    for (int i = 0; i < 3; ++i){
        #pragma unroll
        for (int j = 0; j < 3; ++j){
            M[i][j] *= s;
        }
    }
}

__device__ void transform_inplace(float mat[3][3], float * mol, int Nmol){
    // only transforms the first 3 values in the array! The 4th value is not a 
    // coordinate so we do not transform it
    const int D=4;

    for(int i=0;i<Nmol;i++){

        float temp[3] = {mol[i*D], mol[i*D+1], mol[i*D+2]};
        matvec3x3x3(mat, &mol[i*D], temp);
        mol[i*D]   = temp[0];
        mol[i*D+1] = temp[1];
        mol[i*D+2] = temp[2];
    }
}

__device__ void translate_inplace(float t[3], float * mol, int Nmol){
    // only transforms the first 3 values in the array! The 4th value is not a 
    // coordinate so we do not transform it
    const int D=4;

    for(int i=0;i<Nmol;i++){

        mol[i*D]   += t[0];
        mol[i*D+1] += t[1];
        mol[i*D+2] += t[2];

    }
}

__host__ __device__ void axis_angle_to_quat(const float * axis_angle, float * q){

    // axis must be normalized. We assume it is
    
    float hangle = axis_angle[3]*0.5;
    q[0] = cos(hangle);
    q[1] = axis_angle[0]*sin(hangle);
    q[2] = axis_angle[1]*sin(hangle);
    q[3] = axis_angle[2]*sin(hangle);

}




// The start-mode rotation as a quaternion. Returns false for "no rotation",
// which is method 0 and nothing else -- method 1's index 0 is an axis-angle
// with angle 0, i.e. the identity applied as a matrix, and it stays that way
// because rotating by the identity and not rotating at all are the same VALUES
// but not necessarily the same FFMAs, which a bitwise comparison would see.
//
// Separate from `start_mode_transform` so the two query record types share
// one copy of the table.
__device__ inline bool start_mode_quat(int method, int index, float * q){
    q[0]=1.0f; q[1]=0.0f; q[2]=0.0f; q[3]=0.0f;
    if (method == 1){
        const float transforms_1[16] = {
            1,0,0, 0,
            1,0,0,PI,
            0,1,0,PI,
            0,0,1,PI
        };
        axis_angle_to_quat(&transforms_1[index*4], q);
        return true;
    }
    if (method == 2){
        const float transforms_2[40] = {
            1,0,0,0,
            1,0,0,PI,
            0,1,0,PI,
            0,0,1,PI,
            1,0,0,0.5f*PI,
            0,1,0,0.5f*PI,
            0,0,1,0.5f*PI,
            1,0,0,-0.5f*PI,
            0,1,0,-0.5f*PI,
            0,0,1,-0.5f*PI
        };
        axis_angle_to_quat(&transforms_2[index*4], q);
        return true;
    }
    return false;
}


// Rotate the query into one start-mode orientation AND put it into the reduced
// frame (see kernels.cuh).
//
// **The float record** is written pre-multiplied by ROSHAMBO3_COORD_SCALE,
// which is what makes the pair term in volume_single/get_gradient exactly
// exp2f(-|delta|^2). The scale is folded into the start-mode rotation matrix,
// so it costs nothing beyond the nine multiplies that build the matrix once per
// query.
//
// **The int16 record** is written in the FORMAT's units -- raw Angstrom
// quantized at the wire step -- and the reduced frame's scale is applied on read by
// `r3_query_coord`, after the format's dequantization. The two multiplies stay
// two multiplies for the reason `kernels.cuh` gives at length: folding
// ROSHAMBO3_STAGE_INV_Q_XYZ into ROSHAMBO3_COORD_SCALE is one rounding where
// there were two, and it differs on 31.9% of the 65,536 int16 values.
//
// The start that does not rotate is therefore EXACT on a panel whose
// coordinates are already multiples of the wire-step -- `r3_quantize` is the identity
// on them (`kernels.cuh`'s exhaustive precondition) -- and a start that does
// rotate re-quantizes at the wire-step, which is the same step and the same declared
// budget the library record carries.
template<typename EltA>
__device__ void start_mode_transform(const float * mol, EltA * transformed_mol,
                                     float * transform, int Nmol, int method, int index){

    // set transform to default values
    transform[0]=1;
    transform[1]=0;
    transform[2]=0;
    transform[3]=0;
    transform[4]=0;
    transform[5]=0;
    transform[6]=0;

    float q[4];
    const bool rotate = start_mode_quat(method, index, q);

    float M[3][3];
    if (rotate){
        quaternion_to_rotation_matrix(q, M);
        // The float record carries the reduced-frame scale; the int16 one does
        // not, because its read applies it after the dequantization.
        if constexpr (std::is_same<EltA, float>::value)
            scale_matrix3x3(M, ROSHAMBO3_COORD_SCALE);
        transform[0] = q[0];
        transform[1] = q[1];
        transform[2] = q[2];
        transform[3] = q[3];
    }

    for(int i=0; i<Nmol; i++){
        if constexpr (std::is_same<EltA, float>::value){
            if (rotate){
                matvec3x3x3(M, &mol[D*i], &transformed_mol[D*i]);
            } else {
                transformed_mol[D*i]   = ROSHAMBO3_COORD_SCALE*mol[D*i];
                transformed_mol[D*i+1] = ROSHAMBO3_COORD_SCALE*mol[D*i+1];
                transformed_mol[D*i+2] = ROSHAMBO3_COORD_SCALE*mol[D*i+2];
            }
            // The weight in slot 3 is not a coordinate and is not scaled.
            transformed_mol[D*i+3] = mol[D*i+3];
        } else {
            const int QD = ROSHAMBO3_QUERY_D16;
            if (rotate){
                float r[3];
                matvec3x3x3(M, &mol[D*i], r);
                transformed_mol[QD*i]   = r3_quantize(&r[0]);
                transformed_mol[QD*i+1] = r3_quantize(&r[1]);
                transformed_mol[QD*i+2] = r3_quantize(&r[2]);
            } else {
                transformed_mol[QD*i]   = r3_quantize(&mol[D*i]);
                transformed_mol[QD*i+1] = r3_quantize(&mol[D*i+1]);
                transformed_mol[QD*i+2] = r3_quantize(&mol[D*i+2]);
            }
        }
    }
}


///////////////////////////////////////////////////////////////////////////////
/// Volume functions
///////////////////////////////////////////////////////////////////////////////


// Reduced shape and color overlap of molA and molB.
//
// Returns S, not V: v[0] is the reduced shape overlap (multiply by
// ROSHAMBO3_SHAPE_PREFACTOR for a volume) and v[1] the reduced color overlap
// (multiply by ROSHAMBO3_COLOR_PREFACTOR). See kernels.cuh for why.
//
// `cscale` is applied to every coordinate as it is read, putting both molecules
// in the reduced frame. Pass 1.0f for coordinates that are already there (the
// query, which start_mode_transform scaled into shared memory) and
// ROSHAMBO3_COORD_SCALE for coordinates read straight out of global memory (the
// library). Scaling each coordinate before the subtraction -- rather than scaling
// the difference -- is deliberate: it is the identical operation the query
// already went through, so V_AA and V_BB are computed by the same arithmetic.
//
// exp2f, not exp/__expf: one transcendental, used identically in this function
// and in get_gradient. Two different exp implementations for the same physical
// quantity is a defect: it once made V_AA/V_BB (via exp) inconsistent with
// V_AB (via __expf).
//
// TILEA/TILEB select the memory layout of each operand and change nothing
// about the arithmetic: 0 is the packed per-molecule layout ([mol][atom][4]),
// 32 is the transposed 32-lane tile ([tile][atom][lane][4]). Only the stride
// between two consecutive atoms of one molecule differs, so the same values are
// read in the same order and summed in the same order. See BTILE in
// kernels.cuh.
// Build the query's per-type color segment table.
//
// qcs[t]/qce[t] bound the run of query color slots carrying feature type t;
// both are 0 for a type the query does not carry, so an empty segment is one
// integer compare in the pair loop. Called once per BLOCK -- one query serves
// every library conformer the block scores, at every start mode and every
// optimizer step -- so the pass over NmolA slots below is amortized over
// tcount * N_starts * (nsteps + 1) gradient calls.
//
// Derived from the type plane rather than from a stored count sidecar. That is
// not a shortcut: the stored counts are a per-CONFORMER record and this is a
// per-QUERY table, the query plane is built by engine/query.py (features
// immediately after the real atoms) rather than by the corpus builder (features
// at the shape plane's padded width), and the same-type loop needs the table
// only on the query side. So the structure that won needs no count plumbed, no
// format change and no rebuild.
// `TypeA` is the query type plane's element type -- `int` for the global
// panel the field kernel scans, `unsigned char` for the analytic kernel's shared
// copy. A type id is 0..N_features and N_features is at most MAX_FEATURES, so a
// byte holds every value the plane can carry; the int32 plane was four bytes per
// query atom of shared memory storing a number below eight.
template<typename TypeA = int>
__device__ inline void build_query_color_segments(const TypeA * molA_type, int n_real, int NmolA,
                                                  int N_features, int * qcs, int * qce){
    for (int t = 0; t < N_features; ++t){ qcs[t] = 0; qce[t] = 0; }
    for (int i = n_real; i < NmolA; ++i){
        int t = (int)molA_type[i];
        // 0 is "not a feature"; a type outside the matrix cannot be scored and
        // query.py refuses one at the boundary, so this is belt and braces.
        if (t <= 0 || t >= N_features) continue;
        if (qce[t] == 0) qcs[t] = i;
        qce[t] = i + 1;
    }
}


// ONE query coordinate, in the reduced frame, from either query record.
//
// The float arm is the four-component record and reads it as it always did. The
// int16 arm applies the FORMAT's inverse quantization and then the reduced
// frame's scale, **as two separate multiplies**:
//
//     ROSHAMBO3_COORD_SCALE * (ROSHAMBO3_STAGE_INV_Q_XYZ * (float)v)
//
// `r3_fetch` writes the inner one and the outer one is written here, so the
// parentheses are structural rather than typographical. Folding the two
// constants into one is ONE rounding where there were two: measured over all
// 65,536 int16 values, the folded form differs on 20,930 of them (31.9%) by up
// to 2 ULPs, and
// `doc/NUMERICS_POLICY.md` 4c records the same trap costing 17.7% of pairs
// their bits on the library side.
template<typename EltA>
__device__ __forceinline__ float r3_query_coord(const EltA * molA, int i, int c){
    if constexpr (std::is_same<EltA, float>::value)
        return molA[i*ROSHAMBO3_D + c];
    else
        return ROSHAMBO3_COORD_SCALE * r3_fetch(molA + i*ROSHAMBO3_QUERY_D16 + c);
}


// CLOOP: the color pair loop's structure, COLOR_LOOP_* in kernels.cuh.
// It is bounded because the loop below once ran to the padded
// width on both operands, so its cost grew as (L - n_real)^2 over pure padding.
//
// The bound was first read off the type plane, because the two planes do not put
// their color slots in the same place: engine/query.py packs a query's features
// immediately after its real atoms, while format/dataset.py::combine_shape_color
// puts a library conformer's at the shape plane's PADDED width. The code keeps
// that fact without the plane -- each operand carries its own `cblo`, and the type
// of a slot comes from the count vector (`r3_slot_type`), which returns exactly
// what the plane held at that slot including both padding regions' 0. Requires
// the stored runs to be contiguous and ascending in type, which
// pharmacophore.py::generate_color_atoms asserts as a postcondition and the
// corpus builder asserts per molecule.
// EltA/EltB are the two operands' record element types -- `float` for the
// four-component record, `short` for the int16 wire record. They are SEPARATE
// parameters because the library and the query are not the same problem: the
// library is where every byte is, and the query is one record per launch that
// the pose transforms on every optimizer step. The library moves to int16; the query
// side is a parameter rather than a rewrite, and is instantiated only at
// `float`.
//
// There is ONE copy of the arithmetic below and two loads. A second copy of the
// pair loop for int16 is exactly the drift doc/NUMERICS_POLICY.md exists to
// prevent, and it would break the invariant that V_AA, V_BB and V_AB come
// from the same code path.
template<int TILEA=0, int TILEB=0, int CLOOP=COLOR_LOOP_DEFAULT, bool SHAPE_W=true,
         typename EltA=float, typename EltB=float>
__device__ void volume_single(const EltA * molA, r3_counts_t cntA, int cbloA, int molA_num_atoms, int NmolA,
                              const EltB * molB, r3_counts_t cntB, int cbloB, int molB_num_atoms, int NmolB,
                              const float * rmat, const float * pmat, int N_features, bool optim_color,
                              float cscale, float * v){

    float vs = 0.0;
    float vc = 0.0;

    // Layout parameters, all compile-time. |TILE| is the tile width; a negative
    // TILE selects the planar variant, in which a tile stores all 32 lanes' x
    // together, then all 32 y, and so on. SA/SB are floats between consecutive
    // atoms of one molecule, CA/CB floats between its components, TA/TB ints
    // between atom types. All fold to the shipped D / 1 / 1 when TILE is 0.
    const int WA = (TILEA < 0) ? -TILEA : TILEA;
    const int WB = (TILEB < 0) ? -TILEB : TILEB;
    // Each operand's record width, not one shared `D`, which is the float32
    // record's width and nothing else.
    const int DA = r3_record_width<EltA>::value;
    const int DB = r3_record_width<EltB>::value;
    const int SA = (TILEA != 0) ? WA*DA : DA;
    const int SB = (TILEB != 0) ? WB*DB : DB;
    const int CA = (TILEA < 0) ? WA : 1;
    const int CB = (TILEB < 0) ? WB : 1;
    // There is no type stride any more. The type plane is gone and a
    // conformer's count vector is one scalar, already in a register.

    // normal atoms first then color
    for(int j=0; j<molB_num_atoms; j++){
        for(int i=0; i<molA_num_atoms; i++){

            // `r3_fetch` is the identity on a float record and
            // `ROSHAMBO3_STAGE_INV_Q_XYZ*(float)v` on an int16 one, so the
            // expression below reads, for int16,
            //   cscale * (ROSHAMBO3_STAGE_INV_Q_XYZ * (float)v)
            // -- TWO roundings, the same two the host's `float32(v)*WIRE_QUANT_STEP`
            // followed by this `cscale*` performed. Folding them is the
            // NUMERICS_POLICY 4c defect.
            float dx = cscale*r3_fetch(molA+i*SA)      - cscale*r3_fetch(molB+j*SB);
            float dy = cscale*r3_fetch(molA+i*SA+CA)   - cscale*r3_fetch(molB+j*SB+CB);
            float dz = cscale*r3_fetch(molA+i*SA+2*CA) - cscale*r3_fetch(molB+j*SB+2*CB);

            // `r3_fetch` on a float record is `*p`, so for the float arm the
            // three lines above are `cscale*molA[i*SA] - cscale*molB[j*SB]`
            // exactly as they were -- verified, not assumed: `V_AA` and `V_BB`
            // come from here and are bit-identical to main on every gate arm.
            float d2 = dx*dx + dy*dy + dz*dz;

            // `now`, and the ONE place in this file where the
            // 4th coordinate component is still read. `SHAPE_W = false` drops
            // it, which is provable only where the loop is bounded by the REAL
            // atom counts -- every weight it can then reach is exactly 1.0f, so
            // the multiply is by one. `self_overlap_kernel` passes real counts
            // for both operands and instantiates it false.
            //
            // `volume_test_kernel` does NOT: it passes `NmolA` for both the
            // real and the padded count, so its shape loop covers the padding
            // and the 0/1 mask is what makes those slots contribute nothing.
            // That call keeps SHAPE_W true. Deleting the read there would be
            // silent -- the padded slots would contribute exp2f(-d2) each --
            // and it is the PyTorch-oracle wrapper behind `ext.test_overlap`.
            float kij;
            if (SHAPE_W){
                // The float record's 4th component, or -- for the int16
                // record, which does not carry one -- the same mask
                // reconstructed from the count word. See `r3_slot_weight`.
                float wa = r3_weight<EltA>::get(molA, i*SA+3*CA, cntA, cbloA, i);
                float wb = r3_weight<EltB>::get(molB, j*SB+3*CB, cntB, cbloB, j);
                // the whole point of the reduced frame: no multiply on d2 here
                kij = exp2f(-d2)*wa*wb;
            } else {
                kij = exp2f(-d2);
            }

            vs += kij;
        }
    }

    // Everything past molX_num_atoms is color features followed by padding.
    // With color scoring off there are no color features and rmat/pmat are a
    // 1x1 placeholder, so this loop would index them out of bounds and would
    // evaluate the interaction with a == 0, i.e. 0/0 -> NaN. Skip it.
    if (optim_color){
        // A loop over TYPES with run bounds, in place of a loop over
        // SLOTS that recovered each slot's type. It visits the same slots in
        // the same ascending order and accumulates the same terms in the same
        // order -- so it is bit-identical -- but every factor that depends only
        // on the pair of TYPES is hoisted per RUN instead of per slot (the mean
        // run is ~3 slots), and no slot outside a real run is ever
        // visited, so no weight outside one is ever read. That is what retires
        // `wa`/`wb` here: every slot this reaches is a real feature, whose
        // weight `combine_shape_color` and `query.py::from_arrays` write as
        // exactly 1.0f. `COLOR_LOOP_MASKED_PADDED` was the only structure that
        // needed a padding-reachable loop, and it is gone (BOUNDED reproduces
        // it bitwise on the canonical matrix).
        //
        // The surviving structures still share ONE inner body, because
        // a second copy of this arithmetic is exactly the drift
        // doc/NUMERICS_POLICY.md exists to prevent.
        int b_lo = cbloB;
        int a_run = cbloA;      // the query's run start for THIS type
        for (int tb = 1; tb < N_features; ++tb){
            const int nbt = r3_count(cntB, tb);
            const int nat_same = r3_count(cntA, tb);
            for (int j = b_lo; j < b_lo + nbt; ++j){
                // SAME_TYPE evaluates only ta == tb, so it walks one query run;
                // BOUNDED walks them all, in the same ascending slot order the
                // slot loop produced.
                const int ta0 = (CLOOP >= COLOR_LOOP_SAME_TYPE) ? tb : 1;
                const int ta1 = (CLOOP >= COLOR_LOOP_SAME_TYPE) ? tb + 1
                                                                : N_features;
                int a_lo = (CLOOP >= COLOR_LOOP_SAME_TYPE) ? a_run : cbloA;
                for (int ta = ta0; ta < ta1; ++ta){
                    const int nat = r3_count(cntA, ta);
                    if (nat){
                        const float a = rmat[ta*N_features+tb];
                        // A zero interaction width means the feature pair does
                        // not interact, not that the Gaussian has zero width:
                        // its contribution is zero and evaluating it would
                        // divide by zero.
                        if (a != 0.0f){
                            const float p = pmat[ta*N_features+tb];
                            // (pi/2a)^1.5 = (pi/2)^1.5 * a^-1.5: the shared part
                            // is the extracted COLOR_PREFACTOR, the per-pair
                            // residual a^-1.5 (and p^2) stays inside the term,
                            // so a non-diagonal interaction matrix is handled
                            // exactly rather than approximately. Once per run
                            // not once per slot.
                            const float inv_a = 1.0f/a;
                            const float cf    = p*p*inv_a*__fsqrt_rn(inv_a);
                            const float fac   = a*ROSHAMBO3_INV_TWO_ALPHA;
                            for (int i = a_lo; i < a_lo + nat; ++i){
                                float dx = cscale*r3_fetch(molA+i*SA)      - cscale*r3_fetch(molB+j*SB);
                                float dy = cscale*r3_fetch(molA+i*SA+CA)   - cscale*r3_fetch(molB+j*SB+CB);
                                float dz = cscale*r3_fetch(molA+i*SA+2*CA) - cscale*r3_fetch(molB+j*SB+2*CB);

                                float d2 = dx*dx + dy*dy + dz*dz;

                                float kij = exp2f(-fac*d2)*cf;

                                vc += kij;
                            }
                        }
                    }
                    a_lo += nat;
                }
            }
            b_lo += nbt;
            a_run += nat_same;
        }
    }

    v[0] = vs;
    v[1] = vc;
}




////////////////////////////////////////////////////////////////////////////////
/// gradient functions
////////////////////////////////////////////////////////////////////////////////



// Compute reduced shape and color overlap gradients of molA and (molB
// transformed by q+t) w.r.t. q and t.
//
// molA must already be in the reduced frame (start_mode_transform put it there);
// molB is read raw from global memory and enters the frame for free, because
// ROSHAMBO3_COORD_SCALE is folded into the rotation matrix that has to be built
// anyway. `t` is therefore in scaled Angstrom, matching molA.
//
// v[0]/v[1] are reduced overlaps, as in volume_single. The gradients come back
// already divided by S_AA + S_BB (via gnorm_shape / gnorm_color, folded into the
// per-term coefficient at no cost), so what accumulates is O(1) rather than O(V)
// and the caller does not have to walk the seven components afterwards. Pass
// gnorm = 1.0f to get the unnormalised reduced gradient.
//
// TILEB: 0 = the packed per-molecule library layout, 32 = the transposed
// 32-lane tile. It changes the stride between two consecutive library atoms and
// nothing else -- same values, same order, same reduction, so it is a data
// layout change and not a numerics change. molA is always the shared-memory
// query at stride D.
// XHOIST (measurement arm): apply
//
//     sum_i temp_i * (C_j x D_ij)  ==  C_j x (sum_i temp_i * D_ij)
//
// to the shape loop, so the cross product moves from per-atom-pair to
// per-library-atom -- and the inner sum it needs IS the translation gradient, so
// one 3-vector accumulator serves both. It removes ~9 of the 36 SASS instructions per
// atom-pair-gradient, which is what makes it a causal test of the issue-bound
// component of the kernel's co-limited diagnosis.
//
// It is NOT bit-preserving and cannot be: it reassociates BOTH the rotational
// and the translational gradient sums, which doc/NUMERICS_POLICY.md section 2
// puts on a par with --use_fast_math. It is built as a measurement arm, not as
// a shipped path.
//
// Applied to the shape loop only. The color loop is left alone deliberately:
// no number behind it is taken with color on, and a second untested copy
// of the same rearrangement is the drift the numerics policy exists to prevent.
// QVEC: read the shared-memory query atom as ONE 16-byte load
// (`float4`) instead of the three-or-four 32-bit `LDS` the compiler emits from
// `molA[i*D+c]`. The four floats are the same four floats, read in the same
// order and combined by the same arithmetic, so this is bit-identical -- it
// changes *how wide* the load is, not what is loaded. It exists because the
// shape gradient loop's SASS contains **zero** `LDG` and **three** `LDS` per
// atom-pair-gradient, i.e. ~94% of this
// kernel's L1TEX requests are shared-memory query reads, not library reads.
// Requires `molA` 16-byte aligned; the callers declare the dynamic shared
// block as `float4` for that reason.
//
// CUT (DECLARED numerics change): skip an atom pair whose reduced squared
// distance exceeds `cut2`. exp2f(-d2) < 1e-3 at d2 = 9.966 (4.06 Angstrom) and
// < 1e-4 at d2 = 13.288 (4.69 Angstrom). It drops terms from the sum, which
// doc/NUMERICS_POLICY.md's rule ("a change to where a value lives is free; a
// change to when it is added is not") puts firmly on the declared side.
// STAGE16 (DECLARED numerics change): the library atoms come from an
// int16-quantized per-thread shared-memory copy instead of from float32 global
// memory. STAGE16 is the block width; the layout is planar in shorts,
// sB16[(atom*4+component)*STAGE16 + threadIdx.x]. Quantization is the wire-step on
// the coordinates (measured: p99 |dT| 7.4e-5, two orders inside
// a ~2e-3 budget) and 1/4096 on the weight column, which represents this
// corpus's {0,1} weights exactly. It halves the staged footprint against
// float32, which is the only way the staged buffer fits a 256-thread block.
// NOW: drop `*wa*wb` from the shape term. The shape loops are
// bounded by molA_num_atoms / molB_num_atoms -- the REAL atom counts, never the
// padded L -- so every weight the shape loop can reach is exactly 1.0f:
// `prepare` writes 1.0f for a real atom and 0.0f only for padding, and
// start_mode_transform copies slot 3 through unscaled (it is not a coordinate).
// x*1.0f is exact in IEEE-754, so this is value-preserving by construction --
// but it changes which expressions ptxas may contract into an FFMA, so it is
// verified bitwise rather than assumed.
//
// It also makes the weight column dead in the shape path, which is what lets
// the staged record drop to three planes.
//
// SPLANES: planes in the int16 staged record, 4 (x,y,z,w) or 3
// (x,y,z). Three planes is 6 bytes per atom instead of 8, which is what keeps
// two blocks per SM resident at L = 32 instead of one -- a measured occupancy
// cliff. Requires NOW, since there is then no weight to read.
// CLOOP: the color pair loop's structure -- COLOR_LOOP_* in kernels.cuh,
// and the same four structures volume_single carries. See the note over
// volume_single for how the loop bound is derived.
//
// `qcs`/`qce` are the query's per-type color segment table, built once per BLOCK
// (not per pair and not per gradient call) by build_query_color_segments and
// left in shared memory: qcs[t]/qce[t] bound the run of query slots carrying
// feature type t. They are what makes the same-type form need nothing from the
// library side but the type it is already reading -- no per-conformer count
// argument, no format change, no rebuild.
// `EltB` is the LIBRARY record's element type. molA is the shared-memory
// query and stays float32 -- it is one record per launch, the pose transforms
// it every step, and it is what the query field cache precomputes. One copy of
// the arithmetic, two loads: `r3_fetch` and one `if constexpr` at the pose
// transform are the only places the record type is visible.
template<int TILEB=0, bool XHOIST=false, bool QVEC=false, bool CUT=false, int STAGE16=0,
         bool NOW=false, int SPLANES=4, int CLOOP=COLOR_LOOP_DEFAULT,
         typename EltB=float, typename EltA=float, typename TypeA=int>
__device__ void get_gradient(const EltA * molA, const TypeA * molA_type, int molA_num_atoms,int NmolA,
                             const EltB * molB, r3_counts_t cntB, int cbloB, int molB_num_atoms,int NmolB,
                             const float * rmat, const float * pmat, int N_features, bool optim_color,
                             float gnorm_shape, float gnorm_color,
                             float * gradient, float * gradient_color, const float * q, const float * t, float * v,
                             float cut2 = 0.0f, const short * molB16 = nullptr,
                             const int * qcs = nullptr, const int * qce = nullptr){


    // An int16 query record has no weight plane, exactly as the
    // three-plane staged library record has none, so NOW is not optional there
    // -- there is nothing to read. It is forced rather than asserted because
    // the NOW = false cells are a MEASUREMENT arm that still exists for the
    // float record and is still instantiated across the dispatch grid; a
    // static_assert here would refuse to compile the grid rather than refuse
    // the arm. `optimize_pairs_analytic` refuses `now = false` by name at the
    // launcher, so no caller can reach a cell where this substitution happens
    // silently.
    constexpr bool NOW_EFF = NOW || !std::is_same<EltA, float>::value;
    static_assert(std::is_same<EltA, float>::value || !QVEC,
                  "QVEC is a 16-byte float4 load of the four-component query "
                  "record; the int16 record is three 2-byte components");

    // Where the diagonal of the interaction matrices lives. The same-type
    // loops read `rmat[t*N+t]` and NOTHING else -- measured: 42 of 42
    // off-diagonal `p` cells identically zero and all 49 `a` cells equal to 1 --
    // so at CLOOP >= SAME_TYPE the caller stages the diagonal alone, N floats
    // rather than N*N, and the diagonal is then at stride 1. Below SAME_TYPE the
    // full matrix is staged and the diagonal is at stride N+1. One expression,
    // two strides, and the launcher sizes the block from the same predicate.
    constexpr bool RPDIAG = (CLOOP >= COLOR_LOOP_SAME_TYPE);
    const int rp_diag = RPDIAG ? 1 : (N_features + 1);

    float vs = 0.0;
    float vc = 0.0;

    const int WB = (TILEB < 0) ? -TILEB : TILEB;
    // The LIBRARY record's own width -- 4 for the float32 record, 3 for
    // the int16 one -- where `D` used to stand for both operands.
    const int DB = r3_record_width<EltB>::value;
    const int SB = (TILEB != 0) ? WB*DB : DB; // elements between library atoms
    const int CB = (TILEB < 0) ? WB : 1;      // elements between their components
    // No type stride. The library's types are a count vector, one scalar
    // per conformer, already in a register.

    // -2*ln2*gnorm: the whole per-term gradient coefficient for the shape
    // channel, constant over the pair, so the normalization is free.
    const float tfac_shape = ROSHAMBO3_MINUS_2LN2*gnorm_shape;
    const float tfac_color = ROSHAMBO3_MINUS_2LN2*gnorm_color;

    float M[3][3] = {{0.0,0.0,0.0},
                     {0.0,0.0,0.0},
                     {0.0,0.0,0.0}};

    quaternion_to_rotation_matrix(q, M);
    // fold the reduced-frame coordinate scale into the pose rotation
    scale_matrix3x3(M, ROSHAMBO3_COORD_SCALE);

    #ifdef DEBUG
        printf("loop limits1: %d %d\n",molA_num_atoms, molB_num_atoms);
    #endif

    // loop over molB in outer loop to minimize global memory access
    for( int j=0; j<molB_num_atoms; j++){

        // store values in registers
        float molBTr[3];

        float wb = 1.0f;
        if (STAGE16){
            // De-quantize this library atom out of the staged int16 copy.
            // Once per library atom per gradient call -- i.e. outside the
            // atom-pair loop, so it costs nothing per atom pair.
            const int SB16 = SPLANES*STAGE16, CB16 = STAGE16;
            float raw[3] = {ROSHAMBO3_STAGE_INV_Q_XYZ*(float)molB16[j*SB16],
                            ROSHAMBO3_STAGE_INV_Q_XYZ*(float)molB16[j*SB16+CB16],
                            ROSHAMBO3_STAGE_INV_Q_XYZ*(float)molB16[j*SB16+2*CB16]};
            matvec3x3x3(M, raw, molBTr, 1);
            if (!NOW) wb = ROSHAMBO3_STAGE_INV_Q_W*(float)molB16[j*SB16+3*CB16];
        } else {
            // molB needs to be transformed by q
            //
            // `if constexpr` at the call site rather than an inline
            // wrapper around `matvec3x3x3`, so that the FLOAT arm's statement
            // is character-identical to what shipped. A `__forceinline__`
            // wrapper is semantically the same and is not the same instruction
            // stream -- an inlined-versus-called `adagrad_step` was measured
            // moving the last bit for exactly this reason, and a wrapper here
            // moved it again.
            //
            // The int16 arm dequantizes into three registers and multiplies at
            // stride 1, which is what the `STAGE16` branch has always done
            // with the staged copy: the same arithmetic on the same values.
            if constexpr (std::is_same<EltB, float>::value){
                matvec3x3x3(M, &molB[j*SB], molBTr, CB);
                // The float record's weight column. Every slot this loop can
                // reach is inside `molB_num_atoms`, so it is exactly 1.0f
                // (the `now` note above); it is read rather than assumed
                // because that is what shipped.
                if (!NOW) wb = molB[j*SB+3*CB];
            } else {
                float raw[3] = {r3_fetch(molB + j*SB),
                                r3_fetch(molB + j*SB + CB),
                                r3_fetch(molB + j*SB + 2*CB)};
                matvec3x3x3(M, raw, molBTr, 1);
                // The int16 record carries no weight column; this is the same
                // mask reconstructed from the count word (`r3_slot_weight`),
                // which on this count-bounded loop is exactly 1.0f.
                if (!NOW) wb = r3_slot_weight(cntB, cbloB, j);
            }
        }

        // XHOIST only: this library atom's share of the translation gradient,
        // which is also the vector the per-library-atom cross product acts on.
        float sj[3] = {0.0f, 0.0f, 0.0f};

        // loop over molA in inner loop. Note it is in shared memory
        for(int i=0; i <molA_num_atoms; i++){

            // QVEC: one 16-byte shared load instead of three or four 4-byte
            // ones. Same four values, same order, same arithmetic below.
            float ax, ay, az, wa = 1.0f;
            if constexpr (QVEC && std::is_same<EltA, float>::value){
                const float4 a4 = *reinterpret_cast<const float4 *>(&molA[i*D]);
                ax = a4.x; ay = a4.y; az = a4.z; if (!NOW_EFF) wa = a4.w;
            } else {
                ax = r3_query_coord(molA, i, 0);
                ay = r3_query_coord(molA, i, 1);
                az = r3_query_coord(molA, i, 2);
                if constexpr (std::is_same<EltA, float>::value)
                    if (!NOW_EFF) wa = molA[i*D+3];
            }

            // molA needs to be translated by -t
            float molATx = ax - t[0];
            float molATy = ay - t[1];
            float molATz = az - t[2];

            float dx = molATx - molBTr[0];
            float dy = molATy - molBTr[1];
            float dz = molATz - molBTr[2];

            float d2 = dx*dx + dy*dy + dz*dz;

            // CUT: a declared numerics change -- it removes terms from the
            // sum rather than moving them.
            if (CUT && d2 > cut2) continue;

            // The innermost operation of the whole program: one ex2.approx and
            // two multiplies. There is no exponent coefficient to apply because
            // it lives in the coordinate scale (see kernels.cuh).
            // NOW: the shape loop can only reach real atoms, whose weights are
            // exactly 1.0f, so this multiply by one is dropped rather than
            // performed. See the NOW note over this function.
            float kij = NOW_EFF ? exp2f(-d2) : exp2f(-d2)*wa*wb;

            float temp = tfac_shape*kij;

            if (XHOIST){
                // no cross product in here at all
                sj[0] += temp*dx;
                sj[1] += temp*dy;
                sj[2] += temp*dz;
            } else {
                // cross product method to get gradient

                float x = molBTr[0];
                float y = molBTr[1];
                float z = molBTr[2];

                float C[3] = {2*x,2*y,2*z};
                float mv[3] = {0.0,0.0,0.0};
                float delta[3] = {dx, dy, dz};

                cross_product(C, delta, mv);

                // dq[0] is always zero
                // dq[1-3]:
                gradient[1] += temp*mv[0];
                gradient[2] += temp*mv[1];
                gradient[3] += temp*mv[2];

                // d[x,y,z]:
                gradient[4] += temp*delta[0];
                gradient[5] += temp*delta[1];
                gradient[6] += temp*delta[2];
            }

            vs += kij;
        }

        if (XHOIST){
            float C[3] = {2*molBTr[0], 2*molBTr[1], 2*molBTr[2]};
            float mv[3] = {0.0,0.0,0.0};
            cross_product(C, sj, mv);
            gradient[1] += mv[0];
            gradient[2] += mv[1];
            gradient[3] += mv[2];
            gradient[4] += sj[0];
            gradient[5] += sj[1];
            gradient[6] += sj[2];
        }
    }

    #ifdef DEBUG
    printf("loop limits2: %d:%d %d:%d\n",molA_num_atoms, NmolA, molB_num_atoms, NmolB);
    #endif
    // See volume_single(): with color scoring off the range below holds only
    // padding, rmat/pmat are a 1x1 placeholder, and a == 0 makes every term NaN.
    if (!optim_color){
        v[0] = vs;
        v[1] = 0.0f;
        return;
    }

    // The structures share one inner body on purpose: a second copy
    // of this arithmetic is exactly the drift doc/NUMERICS_POLICY.md exists to
    // prevent, and the arms have to differ only in their control flow if a
    // difference between them is to be attributable.
    //
    // A loop over TYPES with run bounds off the library's count vector,
    // in place of a loop over slots that recovered each slot's type and
    // rescanned for a zero sentinel. Same slots, same ascending order, same
    // terms -- and the per-type constants are hoisted per RUN rather than per
    // slot. Nothing outside a real run is visited, so `wa`/`wb` are gone: every
    // slot this reaches is a real feature at weight exactly 1.0f.
    //
    // The QUERY side is untouched. It still reads its own type plane (BOUNDED)
    // or the per-block segment table `qcs`/`qce` (the same-type forms), because
    // the query plane is built by engine/query.py, not from the library's
    // count vector.
    int b_lo = cbloB;
    for (int tb = 1; tb < N_features; ++tb){
        const int nbt = r3_count(cntB, tb);
        if (nbt){
            // Same-type: the query's segment carrying THIS type, and every
            // factor that depends on the type alone. Two type loads, two matrix
            // loads, a reciprocal, a __fsqrt_rn and three multiplies per PAIR
            // became the same per library slot; they are now once per RUN.
            int i_lo = molA_num_atoms, i_hi = NmolA;
            float cf_seg = 0.0f, fac_seg = 0.0f;
            bool run_empty = false;
            if (CLOOP >= COLOR_LOOP_SAME_TYPE){
                i_lo = qcs[tb]; i_hi = qce[tb];
                // The query carries no feature of this type, so the whole
                // segment is empty -- and now the whole RUN is skipped rather
                // than each of its slots in turn.
                if (i_lo == i_hi) run_empty = true;
                else {
                    float a_seg = rmat[tb*rp_diag];
                    if (a_seg == 0.0f) run_empty = true;
                    else {
                        float p_seg = pmat[tb*rp_diag];
                        float inv_a_seg = 1.0f/a_seg;
                        cf_seg  = p_seg*p_seg*inv_a_seg*__fsqrt_rn(inv_a_seg);
                        fac_seg = a_seg*ROSHAMBO3_INV_TWO_ALPHA;
                    }
                }
            }
            if (!run_empty){
                for (int j = b_lo; j < b_lo + nbt; ++j){
                    // store values in registers
                    float molBTr[3];

                    // molB needs to be transformed by q. See the shape loop
                    // above for why this is `if constexpr` at the call site
                    // rather than a wrapper.
                    if constexpr (std::is_same<EltB, float>::value)
                        matvec3x3x3(M, &molB[j*SB], molBTr, CB);
                    else {
                        float raw[3] = {r3_fetch(molB + j*SB),
                                        r3_fetch(molB + j*SB + CB),
                                        r3_fetch(molB + j*SB + 2*CB)};
                        matvec3x3x3(M, raw, molBTr, 1);
                    }

                    // loop over molA in inner loop. Note it is in shared memory
                    bool seen_a = false;
                    for(int i=i_lo; i<i_hi; i++){

                        float cf, fac;
                        if (CLOOP >= COLOR_LOOP_SAME_TYPE){
                            cf = cf_seg; fac = fac_seg;
                        } else {
                            int ta = (int)molA_type[i];

                            // The query side's own bound, unchanged: a zero
                            // type before any feature is the query plane's
                            // padding, the first zero after one ends its run.
                            if (ta == 0){ if (seen_a) break; else continue; }
                            seen_a = true;

                            float p = pmat[ta*N_features+tb];
                            float a = rmat[ta*N_features+tb];

                            // A zero interaction width means this feature pair
                            // does not interact; its contribution is zero and
                            // 1/(2a) would be infinite.
                            if (a == 0.0f) continue;

                            // Reduced color term: (pi/2a)^1.5 = (pi/2)^1.5 *
                            // a^-1.5, and the shared (pi/2)^1.5 is
                            // ROSHAMBO3_COLOR_PREFACTOR, applied once at
                            // reporting time. `fac` converts the interaction
                            // width into the shape frame's exp2 coefficient.
                            float inv_a = 1.0f/a;
                            cf  = p*p*inv_a*__fsqrt_rn(inv_a);
                            fac = a*ROSHAMBO3_INV_TWO_ALPHA;
                        }

                        // molA needs to be translated by -t
                        float molATx = r3_query_coord(molA, i, 0) - t[0];
                        float molATy = r3_query_coord(molA, i, 1) - t[1];
                        float molATz = r3_query_coord(molA, i, 2) - t[2];

                        float dx = molATx - molBTr[0];
                        float dy = molATy - molBTr[1];
                        float dz = molATz - molBTr[2];

                        float d2 = dx*dx + dy*dy + dz*dz;

                        float kij = exp2f(-fac*d2)*cf;

                        float x = molBTr[0];
                        float y = molBTr[1];
                        float z = molBTr[2];

                        float C[3] = {2*x,2*y,2*z};
                        float mv[3] = {0.0,0.0,0.0};
                        float delta[3] = {dx, dy, dz};

                        cross_product(C, delta, mv);

                        float temp = tfac_color*fac*kij;

                        // dq[0] is always zero
                        // dq[1-3]:
                        gradient_color[1] += temp*mv[0];
                        gradient_color[2] += temp*mv[1];
                        gradient_color[3] += temp*mv[2];

                        // d[x,y,z]:
                        gradient_color[4] += temp*delta[0];
                        gradient_color[5] += temp*delta[1];
                        gradient_color[6] += temp*delta[2];

                        vc += kij;
                    }
                }
            }
        }
        b_lo += nbt;
    }

    v[0] = vs;
    v[1] = vc; 

}


////////////////////////////////////////////////////////////////////////////////
/// Optimization functions
////////////////////////////////////////////////////////////////////////////////

// Adagrad optimization step
__device__ void adagrad_step(float *q, float *t,  float * g, 
                  float * cache, float lr_q, float lr_t) {

    
    for(int i=0; i<7; ++i){
        cache[i] =  cache[i] + g[i]*g[i];
    }

    q[0] -= lr_q*g[0]/(sqrt(cache[0])+EPSILON);
    q[1] -= lr_q*g[1]/(sqrt(cache[1])+EPSILON);
    q[2] -= lr_q*g[2]/(sqrt(cache[2])+EPSILON);
    q[3] -= lr_q*g[3]/(sqrt(cache[3])+EPSILON);

    t[0] -= lr_t*g[4]/(sqrt(cache[4])+EPSILON);
    t[1] -= lr_t*g[5]/(sqrt(cache[5])+EPSILON);
    t[2] -= lr_t*g[6]/(sqrt(cache[6])+EPSILON);
}

////////////////////////////////////////////////////////////////////////////////
/// The optimizer, reparameterized
///
/// WHAT WAS WRONG. `get_gradient` computes the rotational gradient with the
/// cross-product identity: for a rotation increment applied to the *already
/// rotated* library coordinate `Mb`, d(-|delta|^2)/d(omega) = 2 (Mb x delta).
/// That quantity is a TORQUE -- dV/d(omega), the derivative with respect to a
/// space-frame rotation vector. It is not dV/dq. The shipped `adagrad_step`
/// nevertheless adds it to `q[1..3]` and renormalizes, which is
///
///     q <- normalize(q + (0, d))
///
/// and that is the intended rotation only at q = (1,0,0,0). In general
///
///     dR = q' (x) q^{-1}  ==>  rotation vector  2 (w d - d x v)
///
/// for q = (w, v): the applied rotation is a POSE-DEPENDENT linear map of the
/// step, shrunk by ~cos(theta/2) along v and rotated by the -d x v term. The
/// further the optimizer travels from the start orientation, the smaller and
/// more skewed its rotational steps become -- while Adagrad's `cache` keeps
/// accumulating the *nominal* gradient, so the effective learning rate decays
/// twice. This is also the origin of the `g[0] === 0` oddity (there is no
/// fourth degree of freedom to have a gradient for), of the renormalization
/// step (the update leaves the manifold), and of a fixed bug class
/// (a renormalization applied after the fact is a correction, and a wrong
/// correction is silently a different optimizer).
///
/// THE FIX, at zero arithmetic cost per atom pair: compose on the manifold.
///
///     q <- exp(d) (x) q,   exp(d) = (cos|d|, sin|d| d/|d|)
///
/// Left multiplication because the torque is in the space frame (it was taken
/// against the rotated coordinate). The half-angle convention is chosen so
/// that the step means exactly what it meant before at q = identity, where
/// normalize(1, d) = (cos atan|d|, sin atan|d| dhat) and exp(d) agree to
/// O(|d|^3): `lr_q` therefore keeps its calibration and every step-count
/// number stays comparable with the shipped kernel's.
///
/// `q` is a unit quaternion by construction afterwards -- a product of unit
/// quaternions -- so the renormalization is not merely cheap, it is redundant.
/// The permanent ||q|| assertion is satisfied without a correction. The
/// residual drift is fp32 rounding of the composition, measured
/// from the returned pose rather than asserted.
///
/// MODES. `OPT` selects the update; the gradient, the field and the reduction
/// are untouched in every one of them, so each is a single-factor change.
///
///   0  ADAGRAD_QADD  the shipped kernel, bit-for-bit (the control)
///   1  ADAGRAD_EXP   the same Adagrad step, applied through the exp map
///   2  ADAM_EXP      Adam(0.9, 0.999) + exp map. Adagrad's step decays as
///                    lr/sqrt(k), so its TOTAL reach is ~2 lr sqrt(nsteps) --
///                    2.0 A of translation at the shipped lr_t = 0.1, nsteps
///                    = 100, which is the documented reach limit, and only
///                    1.0 A at 25 steps. Adam's step magnitude is ~lr
///                    independent of k, so the reach is lr*nsteps: 10 A at
///                    100 steps and 2.5 A at 25. This is the reach fix.
///   3  QN2_EXP       block-diagonal two-point quasi-Newton (Barzilai-Borwein
///                    on the rotation block and on the translation block
///                    separately, which is what keeps the units honest) + exp
///                    map. 12 floats of state, ~40 flops/step.
///   4  BFGS_EXP      damped BFGS on the full 6-vector + exp map. 21 floats of
///                    inverse Hessian + 12 of history, ~150 flops/step. The
///                    plan's ask; the register cost is a measured result.
///
/// Modes 3 and 4 take their first step from Adagrad (whose magnitude is
/// well-scaled without knowing anything about the curvature), then switch.
/// Both clip to a trust region expressed in units of the learning rate, so no
/// new tuning constant is introduced: |d| <= TRUST_RADIUS*lr_q and |dt| <=
/// TRUST_RADIUS*lr_t. Both fall back to the Adagrad step whenever the measured
/// curvature is not positive, which is the damping.
////////////////////////////////////////////////////////////////////////////////

// The wire values UPDATE_* live in kernels.cuh: the host side range-checks
// `opt_mode` against UPDATE_RULE_COUNT before it ever reaches a launch.

// The same five values as a SCOPED enum, and that is what the device-side
// template parameter `OPT` is typed as -- not `int`.
//
// This is not decoration. OPT and PROBE were each added as a template parameter
// in the same slot of `optimize_one` (`int OPT` and `bool PROBE`), so on integration
// every existing call site was one silent bool<->int conversion away from
// selecting a different optimizer while compiling perfectly cleanly. A scoped
// enum has no implicit conversion in either direction: `optimize_one<...,PROBE>`
// with PROBE in the OPT slot is now a compile error, and so is an `OPT` in the
// PROBE slot. The parameter order below is therefore checkable by the compiler
// rather than by eye.
enum class UpdateRule : int {
    AdagradQAdd = UPDATE_ADAGRAD_QADD,   // the shipped kernel, bit-for-bit
    AdagradExp  = UPDATE_ADAGRAD_EXP,
    AdamExp     = UPDATE_ADAM_EXP,
    QN2Exp      = UPDATE_QN2_EXP,
    BfgsExp     = UPDATE_BFGS_EXP,
};

// Trust-region radius for the quasi-Newton modes, in units of one learning
// rate. 5 lets a curvature step be five times a first Adagrad step and no more.
#define TRUST_RADIUS 5.0f

// q <- exp(d) (x) q, with |d| the HALF-angle (see the block comment).
// Unit by construction: a product of two unit quaternions.
__device__ __forceinline__ void quat_left_exp(float * q, const float d[3]){

    const float a2 = d[0]*d[0] + d[1]*d[1] + d[2]*d[2];

    float c, s;                       // cos|d|,  sin|d|/|d|
    if (a2 > 1e-12f){
        const float a = sqrtf(a2);
        c = cosf(a);
        s = sinf(a)/a;
    } else {
        // The series, so a zero gradient is a no-op rather than a 0/0.
        c = 1.0f - 0.5f*a2;
        s = 1.0f - a2*(1.0f/6.0f);
    }

    const float e0 = c, e1 = s*d[0], e2 = s*d[1], e3 = s*d[2];
    const float q0 = q[0], q1 = q[1], q2 = q[2], q3 = q[3];

    q[0] = e0*q0 - e1*q1 - e2*q2 - e3*q3;
    q[1] = e0*q1 + e1*q0 + e2*q3 - e3*q2;
    q[2] = e0*q2 - e1*q3 + e2*q0 + e3*q1;
    q[3] = e0*q3 + e1*q2 - e2*q1 + e3*q0;
}

// Clip a 3-vector to a maximum norm, in place. Returns nothing; a non-finite
// input is replaced by zero, which is what makes the quasi-Newton modes safe
// against a singular curvature estimate without a second gradient evaluation.
__device__ __forceinline__ void clip_step(float * v, float rmax){
    const float n2 = v[0]*v[0] + v[1]*v[1] + v[2]*v[2];
    if (!(n2 == n2) || n2 > 1e30f){ v[0] = v[1] = v[2] = 0.0f; return; }
    if (n2 > rmax*rmax){
        const float f = rmax*rsqrtf(n2);
        v[0] *= f; v[1] *= f; v[2] *= f;
    }
}

// How many floats of auxiliary state each mode needs beyond `cache[7]`.
//   ADAM  : m[6] + the two running bias-correction powers
//   QN2   : gprev[6] + s[6]
//   BFGS  : gprev[6] + s[6] + the 21 upper-triangle entries of H
template<UpdateRule OPT> struct update_state { enum { N = 1 }; };
template<> struct update_state<UpdateRule::AdamExp> { enum { N = 8  }; };
template<> struct update_state<UpdateRule::QN2Exp>  { enum { N = 12 }; };
template<> struct update_state<UpdateRule::BfgsExp> { enum { N = 33 }; };

// One optimizer step. `g` is the 7-component gradient the shipped kernel
// builds (g[0] is unused by every mode but 0); `k` is the step index.
template<UpdateRule OPT>
__device__ __forceinline__ void apply_update(float * q, float * t, float * g,
                                         float * cache, float * aux, int k,
                                         float lr_q, float lr_t){

    // ---- mode 0: the shipped kernel, unchanged --------------------------
    if (OPT == UpdateRule::AdagradQAdd){
        adagrad_step(q, t, g, cache, lr_q, lr_t);
        const float magq = sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]);
        q[0] /= magq; q[1] /= magq; q[2] /= magq; q[3] /= magq;
        return;
    }

    float d[3] = {0.0f, 0.0f, 0.0f}, dt[3] = {0.0f, 0.0f, 0.0f};

    // ---- the Adagrad step, in the tangent space ------------------------
    // Used directly by mode 1, and as the first step / the damping fallback
    // by modes 3 and 4. Adam owns `cache` itself, so it is skipped there.
    if (OPT != UpdateRule::AdamExp){
        #pragma unroll
        for (int i = 0; i < 3; ++i){
            cache[i+1] += g[i+1]*g[i+1];
            cache[i+4] += g[i+4]*g[i+4];
            d[i]  = -lr_q*g[i+1]/(sqrtf(cache[i+1])+EPSILON);
            dt[i] = -lr_t*g[i+4]/(sqrtf(cache[i+4])+EPSILON);
        }
    }

    if (OPT == UpdateRule::AdamExp){
        // cache[1..6] hold v (the second moment); aux[0..5] hold m.
        const float b1 = 0.9f, b2 = 0.999f;
        if (k == 0){ aux[6] = 1.0f; aux[7] = 1.0f;
                     #pragma unroll
                     for (int i = 0; i < 6; ++i){ aux[i] = 0.0f; cache[i+1] = 0.0f; } }
        aux[6] *= b1; aux[7] *= b2;
        const float c1 = 1.0f/(1.0f - aux[6]);
        const float c2 = 1.0f/(1.0f - aux[7]);
        #pragma unroll
        for (int i = 0; i < 6; ++i){
            const float gi = g[i+1];
            aux[i]     = b1*aux[i]     + (1.0f-b1)*gi;
            cache[i+1] = b2*cache[i+1] + (1.0f-b2)*gi*gi;
            const float st = (aux[i]*c1)/(sqrtf(cache[i+1]*c2)+EPSILON);
            if (i < 3) d[i]    = -lr_q*st;
            else       dt[i-3] = -lr_t*st;
        }
    }
    else if (OPT == UpdateRule::QN2Exp || OPT == UpdateRule::BfgsExp){
        float * gp = aux;          // gradient at the previous point
        float * sp = aux + 6;      // the step that was taken to get here
        if (k > 0){
            float y[6];
            #pragma unroll
            for (int i = 0; i < 6; ++i) y[i] = g[i+1] - gp[i];

            if (OPT == UpdateRule::QN2Exp){
                // Barzilai-Borwein, one scalar per block. Rotation and
                // translation are in different units and a single scalar
                // would silently pick one of them.
                #pragma unroll
                for (int b = 0; b < 2; ++b){
                    float sy = 0.0f, yy = 0.0f;
                    #pragma unroll
                    for (int i = 3*b; i < 3*b+3; ++i){ sy += sp[i]*y[i]; yy += y[i]*y[i]; }
                    // sy > 0 is the curvature condition; f = -V is locally
                    // convex only where it holds, and where it does not the
                    // Adagrad step already sitting in d/dt is used instead.
                    if (sy > 0.0f && yy > 0.0f){
                        const float alpha = sy/yy;
                        float * o = (b == 0) ? d : dt;
                        #pragma unroll
                        for (int i = 0; i < 3; ++i) o[i] = -alpha*g[3*b+i+1];
                    }
                }
            } else {
                // Damped BFGS on the inverse Hessian, upper triangle in
                // aux[12..32]. H is initialized block-diagonally from the
                // first (s, y) pair so the two blocks' units are separated
                // exactly as in QN2, then updated by the standard
                // rank-two formula.
                float * H = aux + 12;
                #define BFGS_H_OFFSET(i,j) H[((i) <= (j)) ? ((i)*6 - ((i)*((i)+1))/2 + (j)) \
                                                  : ((j)*6 - ((j)*((j)+1))/2 + (i))]
                float sy = 0.0f;
                #pragma unroll
                for (int i = 0; i < 6; ++i) sy += sp[i]*y[i];

                if (k == 1){
                    #pragma unroll
                    for (int i = 0; i < 21; ++i) H[i] = 0.0f;
                    #pragma unroll
                    for (int b = 0; b < 2; ++b){
                        float bsy = 0.0f, byy = 0.0f;
                        #pragma unroll
                        for (int i = 3*b; i < 3*b+3; ++i){ bsy += sp[i]*y[i]; byy += y[i]*y[i]; }
                        const float g0 = (bsy > 0.0f && byy > 0.0f) ? bsy/byy : 0.0f;
                        #pragma unroll
                        for (int i = 3*b; i < 3*b+3; ++i) BFGS_H_OFFSET(i,i) = g0;
                    }
                }
                if (sy > 0.0f){
                    // Hy and yHy
                    float Hy[6]; float yHy = 0.0f;
                    #pragma unroll
                    for (int i = 0; i < 6; ++i){
                        float acc = 0.0f;
                        #pragma unroll
                        for (int j = 0; j < 6; ++j) acc += BFGS_H_OFFSET(i,j)*y[j];
                        Hy[i] = acc; yHy += acc*y[i];
                    }
                    const float rho = 1.0f/sy;
                    const float c   = rho*(1.0f + rho*yHy);
                    #pragma unroll
                    for (int i = 0; i < 6; ++i)
                        #pragma unroll
                        for (int j = i; j < 6; ++j)
                            BFGS_H_OFFSET(i,j) += c*sp[i]*sp[j] - rho*(Hy[i]*sp[j] + sp[i]*Hy[j]);
                }
                // p = -H g, but only once H has been seeded; before that the
                // Adagrad step in d/dt stands.
                if (k >= 1){
                    float p[6];
                    bool ok = true;
                    #pragma unroll
                    for (int i = 0; i < 6; ++i){
                        float acc = 0.0f;
                        #pragma unroll
                        for (int j = 0; j < 6; ++j) acc += BFGS_H_OFFSET(i,j)*g[j+1];
                        p[i] = -acc;
                        if (!(acc == acc)) ok = false;
                    }
                    // A descent direction, or fall back. p.g < 0 for f = -V.
                    float pg = 0.0f;
                    #pragma unroll
                    for (int i = 0; i < 6; ++i) pg += p[i]*g[i+1];
                    if (ok && pg < 0.0f){
                        #pragma unroll
                        for (int i = 0; i < 3; ++i){ d[i] = p[i]; dt[i] = p[i+3]; }
                    }
                }
                #undef BFGS_H_OFFSET
            }
            clip_step(d,  TRUST_RADIUS*lr_q);
            clip_step(dt, TRUST_RADIUS*lr_t);
        }
        // Record this point's gradient and the step about to be taken.
        #pragma unroll
        for (int i = 0; i < 3; ++i){
            gp[i]   = g[i+1];   gp[i+3] = g[i+4];
            sp[i]   = d[i];     sp[i+3] = dt[i];
        }
    }

    quat_left_exp(q, d);
    t[0] += dt[0]; t[1] += dt[1]; t[2] += dt[2];
}

// Per-channel gradient weights used to mix the shape and color gradients.
//
// The reported score is a blend of two Tanimotos, so the gradient of that
// score weights each channel by dT/dV = (V_AA + V_BB) / (V_AA + V_BB - V)^2,
// evaluated at the current pose. Cauchy-Schwarz gives V <= sqrt(V_AA * V_BB)
// < V_AA + V_BB for any rigid pose, so the denominator is strictly positive;
// it is still guarded, because a molecule with no color features at all has
// every color volume equal to zero.
//
// These are the *residual* weights: get_gradient has already divided each
// channel by its own S_AA + S_BB (that is the "normalize early" part of the
// scale-factor extraction), so what is left to apply is dT/dS * S_sum, i.e.
// (S_sum/den)^2 -- and, for the legacy 1/(V_AA+V_BB) weighting, exactly 1.
// Both are dimensionless, which is why extracting the volume prefactor does not
// disturb the relative weighting of the two channels: dT/dV scales as 1/V and
// the gradient scales as V.
__device__ void combo_gradient_weights(const float * va, const float * vb, const float * v,
                                       bool legacy, float * w_shape, float * w_color){

    if (legacy){
        *w_shape = 1.0f;
        *w_color = 1.0f;
        return;
    }

    float sum       = va[0] + vb[0];
    float sum_color = va[1] + vb[1];

    float den = sum - v[0];
    float r   = (den > 0.0f) ? sum/den : 0.0f;
    *w_shape  = r*r;

    float den_color = sum_color - v[1];
    float rc = (den_color > 0.0f) ? sum_color/den_color : 0.0f;
    *w_color = rc*rc;
}

// Optimize the overlap of a single (molA, molB) pair for ONE start mode and
// keep the result if it beats what is already in scores_register.
//
// molA must already be the start-mode-rotated query coordinates and qr the
// corresponding start-mode quaternion (both produced by start_mode_transform).
//
// This is THE per-overlap body: both shipping kernels call it, so there is
// exactly one copy of the
// optimizer to fix, and the three kernels cannot drift apart. It does not touch
// global memory; the caller writes scores_register out once at the end.
//
// `first_start` must be true on the first start mode of a pair and false
// afterwards: the first start mode always writes, so that every row reports a
// real pose. Otherwise a pair whose every start mode scored <= 0 (or NaN, which
// compares false) would keep the register initialiser and be returned as an
// all-zero row, including a zero quaternion that is not a rotation.
//
// `va`/`vb`: the query's and library's own reduced self-overlaps
// [S_shape, S_color], precomputed once at prep time by compute_self_overlap_one_gpu
// / compute_self_overlaps_gpu rather than recomputed by every (pair, start_mode)
// evaluation. Both are pose-invariant (V_AA does not depend on which start-mode
// rotation is currently in molA; V_BB does not depend on the library ever being
// rotated at all), so recomputing them here was pure waste. The precompute uses
// the identical volume_single() code path with cscale=ROSHAMBO3_COORD_SCALE on
// raw, unrotated coordinates -- exactly what start_mode_transform's first start
// (n=0, always the identity rotation) already put in shared memory, so this
// changes nothing for n=0 and, for n>0, replaces a value that used to be
// recomputed on rotated coordinates with the canonical n=0 one.
// NOW: drop `*wa*wb` from the shape term. The shape loops are
// bounded by molA_num_atoms / molB_num_atoms -- the REAL atom counts, never the
// padded L -- so every weight the shape loop can reach is exactly 1.0f:
// `prepare` writes 1.0f for a real atom and 0.0f only for padding, and
// start_mode_transform copies slot 3 through unscaled (it is not a coordinate).
// x*1.0f is exact in IEEE-754, so this is value-preserving by construction --
// but it changes which expressions ptxas may contract into an FFMA, so it is
// verified bitwise rather than assumed.
//
// It also makes the weight column dead in the shape path, which is what lets
// the staged record drop to three planes.
//
// SPLANES: planes in the int16 staged record, 4 (x,y,z,w) or 3
// (x,y,z). Three planes is 6 bytes per atom instead of 8, which is what keeps
// two blocks per SM resident at L = 32 instead of one -- a measured occupancy
// cliff. Requires NOW, since there is then no weight to read.
// OPT: which optimizer update is applied. AdagradQAdd is the shipped
// Adagrad step on q[1..3] plus a renormalization, bit-for-bit; the other four
// apply the step through the exponential map. See the optimizer block over
// apply_update().
//
// PROBE: write this start's shape Tanimoto at a fixed set of *prefix*
// step counts, and at convergence, into `probe_out`. This is the diagnostic tap
// for the start-mode pruning question -- "run all starts for a short prefix,
// rank by V, refine the best" -- and it costs nothing: `vs` as filled by
// get_gradient at the top of iteration k is the overlap at the pose after
// exactly k Adagrad updates, so V at prefix p is already in a register at
// iteration p and needs no extra evaluation. PROBE is a template parameter so
// the shipped instantiation carries no trace of it, and a check compares the
// PROBE build's 20 score columns against the production build's bitwise.
// (ROSHAMBO3_PROBE_NPREFIX / _NSLOT are defined in kernels.cuh.)
//
// ---------------------------------------------------------------------------
// THE PARAMETER ORDER, AND WHY IT IS THIS ONE
//
// OPT and PROBE were each appended to this list, in the same slot. The
// integration orders the union by ROLE, and every kernel that forwards to
// optimize_one repeats the same relative order:
//
//   1. data layout and field arithmetic   TILEB, XHOIST, QVEC, CUT, STAGE16,
//                                         NOW, SPLANES
//   2. the optimizer update rule          OPT
//   3. diagnostic taps, never shipped     PROBE
//
// The rule: a switch a production configuration may set comes before one only
// a measurement harness sets. PROBE is last because no shipping call site
// should ever have to name it; OPT is before it because a shipping call site
// does.
//
//   4. the color loop's structure          CLOOP
//
// CLOOP is appended after PROBE rather than filed with the layout switches it
// resembles, and the rule above is why: a shipping call site names a color
// PLACEMENT (a runtime argument) but never a color loop structure. The four
// structures were measured and the winner pinned as this parameter's default, so
// the only code that names a non-default value is a measurement harness -- which is
// exactly PROBE's standing, and it sorts to the same end of the list.
//
// The order is enforced by the compiler rather than by this comment. OPT is a
// SCOPED enum (UpdateRule) and PROBE is a bool, and neither converts implicitly to
// the other, so the two cannot be transposed without a diagnostic. That is
// deliberate: transposing them compiles cleanly and computes something else,
// which is the one failure this integration existed to prevent.
// ---------------------------------------------------------------------------
template<int TILEB=0, bool XHOIST=false, bool QVEC=false, bool CUT=false, int STAGE16=0,
         bool NOW=false, int SPLANES=4,
         UpdateRule OPT=UpdateRule::AdagradQAdd, bool PROBE=false,
         int CLOOP=COLOR_LOOP_DEFAULT, typename EltB=float,
         typename EltA=float, typename TypeA=int, bool RECORD=false>
__device__ void optimize_one(const EltA * molA, const TypeA * molA_type, int molA_num_atoms_i, int NmolA,
                             const float * qr,
                             const EltB * molB, r3_counts_t cntB, int cbloB, int molB_num_atom, int NmolB,
                             const float * rmat, const float * pmat, int N_features,
                             bool optim_color, float lr_q, float lr_t, int nsteps, float mixing_param,
                             bool legacy_combo_gradient, bool first_start,
                             const float * va, const float * vb,
                             float * scores_register, float cut2 = 0.0f,
                             const short * molB16 = nullptr,
                             float * probe_out = nullptr,
                             const int * qcs = nullptr, const int * qce = nullptr,
                             int color_placement = COLOR_PLACEMENT_IN_GRADIENT,
                             float * start_objective = nullptr,
                             int * winning_start = nullptr, int start_index = 0){

    float q[4] = {1.0,0.0,0.0,0.0}; // initial q
    float t[3] = {0.0,0.0,0.0}; // initial t

    // pair overlap volumes [shape, color]
    float vs[2] = {0.0,0.0};

    float cache[7] = {0.0,0.0,0.0,0.0,0.0,0.0,0.0}; //for optimizer

    // Per-mode optimizer state. One float for the two modes that need
    // none, so the array never has size zero and mode 0 costs nothing.
    float aux[update_state<OPT>::N];
    #pragma unroll
    for(int i=0;i<update_state<OPT>::N;++i) aux[i]=0.0f;

    // Gradient normalization, applied inside the accumulation by get_gradient
    // rather than as a pass over the seven components afterwards, so what is
    // summed is O(1) rather than O(V) whatever the molecule size. Guarded
    // because a molecule with no color features has zero color volume.
    float sum_s   = va[0]+vb[0];
    float sum_c   = va[1]+vb[1];
    float gnorm_s = (sum_s > 0.0f) ? 1.0f/sum_s : 0.0f;
    float gnorm_c = (sum_c > 0.0f) ? 1.0f/sum_c : 0.0f;

    // The placement of the color evaluation, as a RUNTIME choice.
    //
    // COLOR_PLACEMENT_AT_POSE is mode 2 and what doc/ARCHITECTURE.md
    // describes: the pose is found on shape and the color term is evaluated once
    // per start, at the converged pose. COLOR_PLACEMENT_IN_GRADIENT is mode 3,
    // combo-optimal, and is what shipped -- N_starts*(steps+1) = 104 color
    // evaluations per pair at the shipped defaults where the documented
    // objective needs 4.
    //
    // It is one boolean on one call site rather than a second kernel, so mode 3
    // is reachable by an argument and stays reachable: combo is not foreclosed.
    const bool color_in_loop = optim_color
                            && (color_placement == COLOR_PLACEMENT_IN_GRADIENT);

    for(int k=0; k < nsteps; ++k){

        float g[7]   = {0.0,0.0,0.0,0.0,0.0,0.0,0.0};
        float g_c[7] = {0.0,0.0,0.0,0.0,0.0,0.0,0.0};
        float g_combo[7] = {0.0,0.0,0.0,0.0,0.0,0.0,0.0};

        get_gradient<TILEB,XHOIST,QVEC,CUT,STAGE16,NOW,SPLANES,CLOOP,EltB,EltA,TypeA>(molA, molA_type, molA_num_atoms_i, NmolA,
                    molB, cntB, cbloB, molB_num_atom,    NmolB,
                    rmat, pmat, N_features, color_in_loop,
                    gnorm_s, gnorm_c,
                    g, g_c, q, t, vs, cut2, molB16, qcs, qce);

        // PROBE tap. `vs[0]` is the reduced shape overlap at the CURRENT pose,
        // i.e. after exactly k updates, so this is V at prefix k for free.
        if (PROBE){
            const int pref[ROSHAMBO3_PROBE_NPREFIX] = {0,1,2,3,5,10};
            #pragma unroll
            for(int j=0;j<ROSHAMBO3_PROBE_NPREFIX;++j){
                if (k == pref[j]) probe_out[j] = vs[0] / (va[0]+vb[0]-vs[0]);
            }
        }

        // 3. update

        // normalize and combine

        // With the color term at the converged pose the pose search is the
        // shape-only one, so it takes the shape-only branch below and not a
        // combo branch with a zero color gradient. Those are not the same
        // optimizer: `a_shape` is (1-mixing)*w_shape, w_shape moves with the
        // pose, and Adagrad is invariant only to a CONSTANT rescaling of the
        // gradient. Taking this branch is what makes mode 2's pose -- and so its
        // shape Tanimoto -- bit-identical to a shape-only screen's.
        if (color_in_loop){

            // Weight each channel by its own dT/dV before mixing, so
            // that g_combo is the gradient of the score that is
            // actually reported. Scaling both channels by 1/(V_AA+V_BB)
            // instead is not dT/dV; for a single channel that would be
            // harmless, because Adagrad is invariant to a constant
            // rescaling of the gradient, but here it sets the
            // *relative* weight of two differently-scaled gradients, so
            // the converged pose was not a stationary point of the
            // reported score.
            float w_shape, w_color;
            combo_gradient_weights(va, vb, vs, legacy_combo_gradient, &w_shape, &w_color);

            float a_shape = (1-mixing_param)*w_shape;
            float a_color = mixing_param*w_color;

            #pragma unroll
            for(int i=0;i<7;++i){
                g_combo[i] = a_shape*g[i] + a_color*g_c[i];
            }

            // OPT = 0 is written out here, in its original exact textual
            // shape, rather than routed through apply_update().
            // The two are the same arithmetic -- but wrapping it in one more
            // inlined call changed what ptxas contracts into an FFMA, and the
            // measured effect of that on a 100-step Adagrad trajectory is the
            // size of a recompilation: only 1.3% of rows bit-identical and
            // max |dT| 7.5e-3, i.e. the recompilation amplification curve. A
            // control that is one recompilation away from the thing it is a
            // control for is not a control, so it is restored and CHECKED
            // bitwise, which is the same reason NOW was verified
            // bitwise.
            if (OPT == UpdateRule::AdagradQAdd) adagrad_step(q,t,g_combo, cache, lr_q, lr_t);
            else                         apply_update<OPT>(q,t,g_combo, cache, aux, k, lr_q, lr_t);
        }
        else{
            // g is already divided by S_AA + S_BB by get_gradient; there is
            // nothing left to scale.
            if (OPT == UpdateRule::AdagradQAdd) adagrad_step(q,t,g, cache, lr_q, lr_t);
            else                         apply_update<OPT>(q,t,g, cache, aux, k, lr_q, lr_t);
        }

        if (OPT == UpdateRule::AdagradQAdd){
            //normalize q
            float magq = sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]);

            q[0] = q[0]/magq;
            q[1] = q[1]/magq;
            q[2] = q[2]/magq;
            q[3] = q[3]/magq;
        }
    }
    // final step to get latest volumes
    // get_gradient computes the volumes, we dont care about the gradients it returns
    // `optim_color`, not `color_in_loop`. This is the evaluation at the
    // converged pose, which is the one the reported color Tanimoto comes from
    // and the only one mode 2 performs.
    float temp[7]   = {0.0,0.0,0.0,0.0,0.0,0.0,0.0};
    get_gradient<TILEB,XHOIST,QVEC,CUT,STAGE16,NOW,SPLANES,CLOOP,EltB,EltA,TypeA>(molA, molA_type, molA_num_atoms_i, NmolA,
                    molB, cntB, cbloB, molB_num_atom, NmolB,
                    rmat, pmat, N_features, optim_color,
                    gnorm_s, gnorm_c,
                    temp,temp, q, t, vs, cut2, molB16, qcs, qce);


    // Compute Tanimoto scores. A zero denominator means both channels have no
    // represented volume. Use zero by convention: an empty representation is
    // not evidence of a perfect match.

    // shape
    const float den_s = va[0] + vb[0] - vs[0];
    const float ts = (den_s > 0.0f) ? vs[0] / den_s : 0.0f;

    // color
    float tc = 0.0;

    if(optim_color){
        const float den_c = va[1] + vb[1] - vs[1];
        tc = (den_c > 0.0f) ? vs[1] / den_c : 0.0f;
    }

    // we use the mixing param to weight the tanimotos
    // this is the objective function we have optimized
    float total = (ts *(1-mixing_param) + tc*mixing_param);

    // PROBE: the converged value of THIS start, whether or not it wins.
    if (PROBE) probe_out[ROSHAMBO3_PROBE_NPREFIX] = ts;

    // Which start wins, and why the criterion follows the placement.
    //
    // In mode 3 (IN_GRADIENT) the objective IS the combination, every start
    // travelled under it, and the best start is the best combination. Unchanged.
    //
    // In mode 2 (AT_POSE) the pose is shape-optimal by construction, and the
    // choice AMONG the starts is the last place color could still influence it.
    // Selecting on `total` here would cost nothing in this kernel and would
    // score marginally better -- and it would break the property that makes
    // mode 2 worth having: that its pose is EXACTLY a shape-only screen's, so
    // the color term is a rescore of a pose that was already found and stored.
    // A survivor-only second pass
    // receives one converged pose per surviving pair, not four; selecting on the
    // combination would make the fused kernel and that second pass produce
    // different scores, and then the cheap deployment would not be the thing
    // that was validated. So mode 2 selects on shape, mode 3 on the
    // combination, and each is the criterion its own objective names.
    //
    // Checked, not asserted: mode 2's
    // shape Tanimoto is bit-identical to a shape-only screen's over the panel.
    // Slot 0 holds the winning `total`, slot 1 the winning `ts`, so the
    // incumbent to beat is read from whichever slot the criterion is.
    const float sel  = color_in_loop ? total : ts;
    const float best = color_in_loop ? scores_register[0] : scores_register[1];
    if constexpr (RECORD) {
        if (start_objective) *start_objective = sel;
    }
    if (first_start || sel > best){
        if constexpr (RECORD) {
            if (winning_start) *winning_start = start_index;
        }
        // store the scores, volumes, and transformation.
        //
        // This is the ONLY place the reduced overlaps become volumes and the
        // only place the translation leaves the reduced frame: everything above
        // this line is scale-free, which is the whole point (see
        // kernels.cuh). The Tanimotos above needed no conversion at all,
        // because the prefactor cancels out of the ratio identically.
        scores_register[0] = total; // combination tanimoto of shape and color
        scores_register[1] = ts; // shape tanimoto
        scores_register[2] = tc; // color tanimoto
        scores_register[3] = ROSHAMBO3_SHAPE_PREFACTOR*vs[0]; // volume shape
        scores_register[4] = ROSHAMBO3_COLOR_PREFACTOR*vs[1]; // volumes color
        scores_register[5] = ROSHAMBO3_SHAPE_PREFACTOR*va[0]; // self a
        scores_register[6] = ROSHAMBO3_SHAPE_PREFACTOR*vb[0]; // self b
        scores_register[7] = ROSHAMBO3_COLOR_PREFACTOR*va[1]; // self a color
        scores_register[8] = ROSHAMBO3_COLOR_PREFACTOR*vb[1]; // self b color
        scores_register[9]  = q[0]; // fit q
        scores_register[10] = q[1];
        scores_register[11] = q[2];
        scores_register[12] = q[3];
        scores_register[13] = ROSHAMBO3_INV_COORD_SCALE*t[0]; // fit t, in Angstrom
        scores_register[14] = ROSHAMBO3_INV_COORD_SCALE*t[1];
        scores_register[15] = ROSHAMBO3_INV_COORD_SCALE*t[2];
        scores_register[16] = qr[0]; // start_mode q
        scores_register[17] = qr[1];
        scores_register[18] = qr[2];
        scores_register[19] = qr[3];
    }
}

////////////////////////////////////////////////////////////////////////////////
/// Self-overlap precompute
///
/// V_AA (per query) and V_BB (per library conformer) are pose-invariant and
/// query-invariant respectively, but optimize_one() used to recompute both from
/// scratch on every (pair, start_mode) evaluation. This section computes each
/// molecule's own reduced self-overlap once -- at "prep time", immediately
/// after its coordinates land on the device, before any pair or start-mode loop
/// -- via the identical volume_single() code path the rest of the kernel uses.
///
/// Both take a `cscale` argument, exactly like volume_single itself, because
/// the library and the query are not interchangeable here:
///
/// - The library (V_BB) is read raw from global memory and scaled by
///   ROSHAMBO3_COORD_SCALE *inside* volume_single, every time -- that is
///   exactly what optimize_one() already did for every (pair, start_mode)
///   evaluation (molB is never rotated), so calling volume_single the same way
///   here, once, is bit-identical to the old per-pair computation by
///   construction. Confirmed empirically: zero difference
///   on V_BB across every start_mode/nsteps combination tested.
///
/// - The query (V_AA) is different: start_mode_transform's first start (n=0,
///   always the identity rotation) pre-scales the query into shared memory
///   with cscale folded into an *identity* rotation matrix, then optimize_one
///   called volume_single with cscale=1.0f on that already-scaled buffer. That
///   is NOT the same floating-point operation as calling volume_single with
///   cscale=ROSHAMBO3_COORD_SCALE directly on the raw query: `cscale*a -
///   cscale*b` computed fresh (a candidate FMA fusion under this project's
///   pinned -fmad=true) is not guaranteed bit-identical to computing
///   `round(cscale*a)` once, storing it, and later subtracting two such
///   pre-rounded values with cscale=1.0f (trivially exact, so immune to FMA
///   fusion either way). Measured effect of getting this wrong: ~1e-7
///   relative on V_AA -- inside
///   a ~1e-7 acceptance bar, but avoidable, so this precompute
///   reproduces the exact two-step computation instead: prescale_coords_gpu()
///   first (the identical per-atom multiply start_mode_transform performs),
///   then this kernel with cscale=1.0f on the result. See
///   doc/NUMERICS_POLICY.md for the measured agreement.
////////////////////////////////////////////////////////////////////////////////

// Prescale every real coordinate (not the weight column) of n_mols molecules
// by ROSHAMBO3_COORD_SCALE, treating the input as one flat array of
// n_mols*Nmol atoms -- serves both the dense path's single query (n_mols=1)
// and the sparse path's batched queries (n_mols=n_q) with one kernel. This is
// the exact per-atom operation start_mode_transform's scaling loop performs
// (see kernels.cu's start_mode_transform), so that a self-overlap
// computed on its output with cscale=1.0f is bit-identical to what start mode
// n=0 already put in shared memory.
__global__ void prescale_coords_kernel(const float * mols, float * out, long n_atoms_total, bool query_wire_grid){

    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i >= n_atoms_total) return;

    // Analytic cross terms read an int16 query, so its self
    // term must reconstruct that same wire coordinate before scaling.
    for (int c=0; c<3; ++c) {
        const float raw = query_wire_grid
            ? ROSHAMBO3_STAGE_INV_Q_XYZ * float(r3_quantize(mols+i*D+c))
            : mols[i*D+c];
        out[i*D+c] = ROSHAMBO3_COORD_SCALE*raw;
    }
    out[i*D+3] = mols[i*D+3];
}

void prescale_coords_gpu(const float * mols, float * out, int Nmol, long n_mols, int device_id, bool query_wire_grid){
    if (n_mols == 0) return;
    long n_atoms_total = n_mols*(long)Nmol;
    dim3 blockDim(NTHREADS);
    dim3 gridDim((n_atoms_total + blockDim.x - 1)/blockDim.x);
    cudaSetDevice(device_id);
    prescale_coords_kernel<<<gridDim,blockDim>>>(mols, out, n_atoms_total, query_wire_grid);
    cudaSetDevice(device_id); CUDA_CHECK_ERROR(cudaGetLastError());
    cudaSetDevice(device_id); CUDA_CHECK_ERROR(cudaDeviceSynchronize());
}

// Batched precompute: one thread per molecule, each with its own atom count
// (used for the Nv library molecules, and for the n_q queries in the sparse interface).
// `Elt` is the record's element type -- `short` for the library's int16
// wire record, `float` for a query prescaled by `prescale_coords_gpu`. Both
// operands are the SAME molecule here, so there is one parameter and not two.
template<int TILE=0, int CLOOP=COLOR_LOOP_DEFAULT, typename Elt=float>
__global__ void self_overlap_kernel(const Elt * mols, const r3_counts_t * mol_counts,
                                    int Nmol, long n_mols, int color_offset,
                                    const float * rmat, const float * pmat, int N_features, bool optim_color,
                                    float cscale, float * out /* [n_mols, 2]: S_shape, S_color */){

    long idx = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (idx >= n_mols) return;

    float v[2] = {0.0f, 0.0f};
    // Byte 0 of the count word IS the real atom count -- what
    // `mol_num_atoms[idx]` used to be, from the one record rather than from a
    // second array that had to agree with it.
    const r3_counts_t cnt = mol_counts[idx];
    const int n_atoms = r3_n_real(cnt);
    // Where this molecule's color run begins. A library's is the group's padded
    // shape width, uniform over the launch; a query's is its own real atom
    // count, which the caller names by passing a negative offset. The two
    // planes genuinely differ (see the note over volume_single).
    const int cblo = (color_offset >= 0) ? color_offset : n_atoms;
    // With TILE > 0 the array is [tile][atom][lane][4]; molecule idx lives
    // in lane idx%TILE of tile idx/TILE, with TILE*D floats between its atoms.
    // `mol_counts` is NOT tiled -- it is one scalar per molecule, indexed by
    // idx however the coordinates are laid out.
    const int W = (TILE < 0) ? -TILE : TILE;
    // The record's own width, not `D`.
    const int DE = r3_record_width<Elt>::value;
    const Elt * mol        = (TILE > 0)
        ? &mols[((idx/W)*(long)Nmol*W + (idx%W))*DE]
        : ((TILE < 0) ? &mols[(idx/W)*(long)Nmol*W*DE + (idx%W)]
                      : &mols[idx*(long)Nmol*DE]);

    // SHAPE_W = false: both operands are this molecule at its REAL atom count,
    // so the shape loop cannot reach a padding slot and every weight it would
    // read is exactly 1.0f (`now`, verified bitwise).
    volume_single<TILE,TILE,CLOOP,false,Elt,Elt>(mol, cnt, cblo, n_atoms, Nmol,
                  mol, cnt, cblo, n_atoms, Nmol,
                  rmat, pmat, N_features, optim_color, cscale, v);

    out[idx*2]   = v[0];
    out[idx*2+1] = v[1];
}

// Host launcher for the batched precompute (n_mols molecules, each with its own
// atom count read from a device array). mols/mol_types/mol_num_atoms and out
// must already be device pointers. Pass cscale=ROSHAMBO3_COORD_SCALE with raw
// coordinates for a library (matches the old per-pair V_BB computation exactly);
// for a query, prescale_coords_gpu() first and pass cscale=1.0f (see this
// section's header comment for why the two are not interchangeable).
template<typename Elt>
void compute_self_overlaps_gpu(const Elt * mols, const r3_counts_t * mol_counts,
                               int Nmol, long n_mols, int color_offset,
                               const float * rmat, const float * pmat, int N_features, bool optim_color,
                               float cscale, float * out, int device_id, int tile_mode,
                               int color_loop){

    if (n_mols == 0) return;

    dim3 blockDim(NTHREADS);
    dim3 gridDim((n_mols + blockDim.x - 1)/blockDim.x);

    if (color_loop <= COLOR_LOOP_MASKED_PADDED || color_loop >= COLOR_LOOP_COUNT)
        throw std::runtime_error("roshambo3.cuda: color_loop must be "
                                 + std::to_string(COLOR_LOOP_BOUNDED) + ".."
                                 + std::to_string(COLOR_LOOP_COUNT-1)
                                 + ", got " + std::to_string(color_loop)
                                 + " (0, MASKED_PADDED, is retired)");

    cudaSetDevice(device_id);
    // The same structure the pair loop uses, so V_AA and V_BB are produced
    // by the same code path as V_AB -- which is the invariant
    // and the reason this precompute is bit-identical to the per-pair one it
    // replaced. All four structures are value-preserving on the canonical
    // matrix, so this is consistency rather than necessity; it is checked
    // bitwise all the same.
    #define ROSHAMBO3_SELFOVL_CL(TM, CL)                                                \
        self_overlap_kernel<TM,CL,Elt><<<gridDim,blockDim>>>(mols, mol_counts,                \
                                              Nmol, n_mols, color_offset,                    \
                                              rmat, pmat, N_features, optim_color, cscale, out)
    #define ROSHAMBO3_SELFOVL(TM)                                                       \
        do {                                                                            \
            switch (color_loop){                                                        \
              case COLOR_LOOP_BOUNDED:       ROSHAMBO3_SELFOVL_CL(TM, COLOR_LOOP_BOUNDED);       break; \
              case COLOR_LOOP_SAME_TYPE:     ROSHAMBO3_SELFOVL_CL(TM, COLOR_LOOP_SAME_TYPE);     break; \
              case COLOR_LOOP_SAME_TYPE_NOW: ROSHAMBO3_SELFOVL_CL(TM, COLOR_LOOP_SAME_TYPE_NOW); break; \
              default: throw std::runtime_error(                                        \
                  "roshambo3.cuda: color_loop " + std::to_string(color_loop)            \
                  + " is not a structure");                                             \
            }                                                                           \
        } while(0)
    if (tile_mode > 0)       ROSHAMBO3_SELFOVL(ROSHAMBO3_BTILE);
    else if (tile_mode < 0)  ROSHAMBO3_SELFOVL(-ROSHAMBO3_BTILE);
    else                     ROSHAMBO3_SELFOVL(0);
    #undef ROSHAMBO3_SELFOVL
    #undef ROSHAMBO3_SELFOVL_CL
    cudaSetDevice(device_id); CUDA_CHECK_ERROR(cudaGetLastError());
    cudaSetDevice(device_id); CUDA_CHECK_ERROR(cudaDeviceSynchronize());
}

// `compute_self_overlap_one_gpu` and its `self_overlap_one_kernel` were
// here, for "the dense (one-query-per-launch) path". Nothing has called either
// since that path was retired -- verified by grep over src/, bench/ and tests/
// before deleting -- and every caller now goes through the batched form above.

////////////////////////////////////////////////////////////////////////////////
/// Attacking L1TEX *request* volume
///
/// The kernel was measured L1TEX-bound, which and closed the sector question
/// (a planar tile at the 3.99-sectors/request floor is 1.5% slower). What was
/// left was request volume. This kernel is the plain blockwise arm with
/// three switches over it and nothing else changed, so each is a single-factor
/// causal ablation on the production kernel:
///
///   QVEC   read the shared query atom as one 16-byte float4 rather than the
///          three 32-bit LDS the compiler emits. **Bit-identical.**
///   STAGE  stage this thread's library molecule into shared memory ONCE per
///          pair and let all `nsteps` gradient evaluations read it from there,
///          instead of re-reading it from global on every step. STAGE is the
///          block width, and the staged layout is planar --
///          `sB[(atom*4+component)*STAGE + threadIdx.x]` -- so a warp reading
///          one component of one atom touches 32 consecutive floats, i.e. 32
///          distinct banks: conflict-free, one wavefront. **Bit-identical**:
///          the same float32 values are read in the same order, only from a
///          different memory space (doc/NUMERICS_POLICY.md's "a change to
///          where a value lives is free").
///   CUT    drop atom pairs beyond a squared-distance cutoff. **A declared
///          numerics change**; it removes terms from the sum.
///
/// Why QVEC is here at all, given STAGE was the target: the shape
/// gradient loop's sm_120 SASS contains **zero LDG and three LDS per
/// atom-pair-gradient** (the library is
/// already hoisted into the per-library-atom loop by the compiler). Per warp
/// that is 3/32 = 0.094 shared requests per apg against 4/(32*n_A) = 0.006
/// global ones, so ~94% of this kernel's L1TEX requests are *query* reads.
/// The arithmetic was checked against measured wavefront counters.
////////////////////////////////////////////////////////////////////////////////
// The analytic block's shared memory, term for term and in the order the
// kernel lays it out. ONE expression decides both what the launcher asks the
// driver for and where the kernel's pointers land, because the two disagreeing
// by a byte is a buffer overrun on one side and a refused launch on the other.
//
// The record it sizes is the TRIMMED one:
//
//   * the query is three int16 planes, not four float32 ones. The weight plane
//     was always exactly 1.0f wherever a loop could read it, so it is dropped
//     rather than stored (`ROSHAMBO3_QUERY_D16`, and the NOW note over
//     `get_gradient`).
//   * the query's type ids are `unsigned char`, not `int`. A type id is 0..7.
//   * the interaction matrices are staged DIAGONAL ONLY when the color loop
//     reads only the diagonal, which is the shipped structure. That
//     is 2*N floats rather than 2*N*N -- 336 of the 392 bytes at N = 7.
//
// 920 B -> 304 B at a 22-atom query and N = 7, of which 6 B is the tail
// alignment.
//
// `qcs`/`qce` stay int32: they are slot INDICES into a query plane up to 256
// wide, not type ids, and 2*N of them is 56 B.
__host__ __device__ __forceinline__ size_t analytic_query_block_bytes(
        int NmolA, int N_features, bool rp_diag, size_t stage_bytes){
    size_t o = 8*sizeof(float);                                  // qr
    o += 2*(size_t)(rp_diag ? N_features : N_features*N_features)*sizeof(float);
    o += 2*(size_t)N_features*sizeof(int);                       // qcs, qce
    o += (size_t)NmolA*ROSHAMBO3_QUERY_D16*sizeof(short);        // the query
    o += (size_t)NmolA*sizeof(unsigned char);                    // its type ids
    o = (o + 15) & ~(size_t)15;    // the staged library starts 16-byte aligned
    return o + stage_bytes;
}


// `EltB` is the LIBRARY record's element type -- `short` for the int16
// wire record, `float` for the four-component one.
template<int TILEB=0, bool QVEC=false, int STAGE=0, bool CUT=false,
         bool XHOIST=false, bool STAGE_I16=false, bool NOW=false, int SPLANES=4,
         UpdateRule OPT=UpdateRule::AdagradQAdd,
         int CLOOP=COLOR_LOOP_DEFAULT, typename EltB=float, bool RECORD=false>
__global__ void optimize_pairs_analytic_kernel(
        const float * molAs_global, const int * molA_types_global, const int * molA_num_atoms_global, int NmolA,
        const EltB * molBs,         const r3_counts_t * molB_counts,                                  int NmolB,
        int library_color_offset,
        const int * out_library_row,
        const int * wi_qrow, const int * wi_tstart, const int * wi_tcount,
        const float * rmat_global, const float * pmat_global, int N_features,
        float * scores, bool optim_color, float lr_q, float lr_t, int nsteps, float mixing_param, int start_mode,
        bool legacy_combo_gradient,
        const float * va_all /* [n_q,2] */, const float * vb_all /* [n_t,2] */,
        float cut2, int color_placement,
        int * winning_starts, float * start_objectives){

    const int DV = 20;

    int wi     = blockIdx.x;
    int qrow   = wi_qrow[wi];
    int tstart = wi_tstart[wi];
    int tcount = wi_tcount[wi];

    // float4, not float: QVEC's 16-byte query load needs the base 16-byte
    // aligned. The dynamic shared-memory base is, and declaring it this way
    // says so to the compiler rather than relying on it.
    extern __shared__ float4 shared_cell_v4[];
    float * shared    = (float *)shared_cell_v4;
    // The trimmed query record. The order is the fixed-size terms first
    // and the staged library last, which is what lets the launcher size the
    // staging buffer from `stage_width` alone; `analytic_query_block_bytes`
    // above is the same walk in bytes and the two must not drift.
    //
    // RPDIAG is derived from CLOOP rather than passed, so it cannot disagree
    // with what `get_gradient` reads -- and the launcher derives the same
    // predicate from the same `color_loop`.
    constexpr bool RPDIAG = (CLOOP >= COLOR_LOOP_SAME_TYPE);
    const int RP      = RPDIAG ? N_features : N_features*N_features;
    float * qr        = shared;                                   // 8 floats
    float * rmat      = (float *)&shared[8];
    float * pmat      = (float *)&shared[8+RP];
    // The query's per-type color segment table, 2*N_features ints (56 B for
    // the shipped six-type generator).
    int   * qcs       = (int *)&shared[8+2*RP];
    int   * qce       = (int *)&shared[8+2*RP+N_features];
    short * molA      = (short *)&shared[8+2*RP+2*N_features];
    unsigned char * molA_type =
        (unsigned char *)(molA + (size_t)NmolA*ROSHAMBO3_QUERY_D16);
    char  * smem_base = (char *)shared;
    size_t  stage_off = (size_t)((char *)(molA_type + NmolA) - smem_base);
    stage_off = (stage_off + 15) & ~(size_t)15;
    float * sB        = (float *)(smem_base + stage_off);

    const float * molA_global      = &molAs_global[(size_t)qrow*NmolA*D];
    const int   * molA_type_global = &molA_types_global[(size_t)qrow*NmolA];
    int molA_num_atoms_i           = molA_num_atoms_global[qrow];

    size_t tidx = threadIdx.x;

    for(int l=tidx; l<NmolA; l+=blockDim.x){
        molA_type[l] = (unsigned char)molA_type_global[l];
    }
    // The diagonal alone where the color loop reads the diagonal alone.
    // Measured: 42 of 42 off-diagonal `p` cells identically zero and all 49
    // `a` cells equal to 1 on the canonical generator, so the staged copy of
    // the full matrix was 336 of its 392 bytes storing what nothing reads.
    if (RPDIAG){
        for(int l=tidx; l<N_features; l+=blockDim.x){
            rmat[l] = rmat_global[l*N_features+l];
            pmat[l] = pmat_global[l*N_features+l];
        }
    } else {
        for(int l=tidx; l<(N_features*N_features); l+=blockDim.x){
            rmat[l] = rmat_global[l];
            pmat[l] = pmat_global[l];
        }
    }
    __syncthreads();

    // One pass over the query's type plane, once per block, after it is in
    // shared memory. Serialized on thread 0 on purpose -- NmolA is ~136 slots at
    // the widest bucket and this runs once against tcount*N_starts*(nsteps+1)
    // gradient calls, so a parallel reduction would cost more in barriers than
    // the scan costs in total.
    if (CLOOP >= COLOR_LOOP_SAME_TYPE){
        if (tidx == 0)
            build_query_color_segments(molA_type, molA_num_atoms_i, NmolA,
                                       N_features, qcs, qce);
        __syncthreads();
    }

    int N_starts = start_mode_n[start_mode];

    float scores_register[20];
    for(int j=0;j<20;++j) scores_register[j]=0.0f;

    int  row    = tstart + threadIdx.x;
    bool active = (threadIdx.x < tcount);

    const EltB * molB       = nullptr;
    // One 64-bit word per conformer, read straight off `trow`. The
    // coordinate record is tiled and this is not, deliberately -- the 32 lanes
    // of a warp hold 32 consecutive rows, so this is already 256 contiguous
    // bytes in one load, and tiling a per-conformer scalar would scatter it.
    r3_counts_t cntB        = 0;
    int cbloB               = 0;
    int molB_num_atom       = 0;
    const float * vb_t      = nullptr;
    const int WB = (TILEB < 0) ? -TILEB : TILEB;
    // The LIBRARY record's own width -- 4 floats, or 3 shorts.
    const int DB = r3_record_width<EltB>::value;
    if (active){
        int trow = out_library_row[row];
        if (TILEB > 0){          // interleaved tile: [tile][atom][lane][DB]
            molB      = &molBs[((size_t)(trow/WB)*NmolB*WB + (trow%WB))*DB];
        } else if (TILEB < 0){   // planar tile:      [tile][atom][DB][lane]
            molB      = &molBs[(size_t)(trow/WB)*NmolB*WB*DB + (trow%WB)];
        } else {
            molB      = &molBs[(size_t)trow*NmolB*DB];
        }
        cntB          = molB_counts[trow];
        molB_num_atom = r3_n_real(cntB);
        // A negative offset means "immediately after this conformer's own real
        // atoms", which is how engine/query.py lays a QUERY out and how the
        // small hand-built fixtures lay a library out. The corpus path passes
        // the group's padded shape width, because that is where
        // format/dataset.py::combine_shape_color puts a library conformer's
        // features.
        cbloB         = (library_color_offset >= 0) ? library_color_offset
                                                    : molB_num_atom;
        vb_t          = &vb_all[(size_t)trow*2];
    }
    const float * va_q = &va_all[(size_t)qrow*2];

    // ---- STAGE: one copy of this thread's library into shared memory --------
    // Done here, outside the start-mode loop and outside the optimizer-step
    // loop, because the library's *source* coordinates never change; only the
    // pose does. Everything past this point reads it from shared.
    // With the library staged as float32 the staged buffer IS the operand
    // the gradient reads, so the effective element type is float there and the
    // wire record's type everywhere else.
    using EltB_EFF = typename std::conditional<(STAGE != 0 && !STAGE_I16),
                                               float, EltB>::type;
    const EltB_EFF * molB_grad = nullptr;
    if constexpr (!(STAGE != 0 && !STAGE_I16)) molB_grad = (const EltB_EFF *)molB;
    const short * molB16    = nullptr;
    if (STAGE){
        const int SBG = (TILEB != 0) ? WB*DB : DB;   // global strides
        const int CBG = (TILEB < 0) ? WB : 1;
        short * sB16 = (short *)sB;
        for(int j=0; j<NmolB; ++j){
            float b0 = 0.0f, b1 = 0.0f, b2 = 0.0f, b3 = 0.0f;
            short q0 = 0, q1 = 0, q2 = 0;
            if (active){
                b0 = r3_fetch(molB+j*SBG);
                b1 = r3_fetch(molB+j*SBG+CBG);
                b2 = r3_fetch(molB+j*SBG+2*CBG);
                // For the int16 record the staged short IS the record's
                // short -- staging is a re-quantization at the same wire step
                // step, and the exhaustive precondition
                // (kernels.cuh) makes that the identity on
                // every representable value. For the float32 record this is the
                // quantization that shipped, character-identical.
                q0 = r3_quantize(molB+j*SBG);
                q1 = r3_quantize(molB+j*SBG+CBG);
                q2 = r3_quantize(molB+j*SBG+2*CBG);
                // The weight column, or its reconstruction from the count word.
                if (!(STAGE_I16 && SPLANES == 3))
                    b3 = r3_weight<EltB>::get(molB, j*SBG+3*CBG, cntB, cbloB, j);
            }
            if (STAGE_I16){
                sB16[((size_t)j*SPLANES+0)*STAGE + tidx] = q0;
                sB16[((size_t)j*SPLANES+1)*STAGE + tidx] = q1;
                sB16[((size_t)j*SPLANES+2)*STAGE + tidx] = q2;
                // SPLANES == 3 drops the weight plane entirely: with NOW the
                // shape loop never reads a weight, and 6 B/atom is what keeps
                // two blocks per SM resident at L = 32.
                if (SPLANES == 4)
                    sB16[((size_t)j*SPLANES+3)*STAGE + tidx] = (short)__float2int_rn(ROSHAMBO3_STAGE_Q_W*b3);
            } else {
                sB[((size_t)j*4+0)*STAGE + tidx] = b0;
                sB[((size_t)j*4+1)*STAGE + tidx] = b1;
                sB[((size_t)j*4+2)*STAGE + tidx] = b2;
                sB[((size_t)j*4+3)*STAGE + tidx] = b3;
            }
        }
        __syncthreads();
        if (STAGE_I16) molB16    = &sB16[tidx];
        else if constexpr (std::is_same<EltB_EFF, float>::value)
                       molB_grad = &sB[tidx];
    }
    // The float32 staged buffer IS a planar tile of width STAGE, so the
    // existing TILEB < 0 stride arithmetic reads it with no new code:
    // SB = STAGE*4, CB = STAGE. The int16 one is addressed by STAGE16 inside
    // get_gradient instead, so TILEB_EFF stays at the global layout there
    // (nothing else reads molB once staging is on).
    constexpr int TILEB_EFF  = (STAGE != 0 && !STAGE_I16) ? -STAGE : TILEB;
    constexpr int STAGE16_EFF = STAGE_I16 ? STAGE : 0;

    for(int n=0; n<N_starts; n++){
        // only one thread rotates the shared query; all threads share it
        if (tidx==0){
            start_mode_transform(molA_global, molA, qr, NmolA, start_mode, n);
        }
        __syncthreads();

        if (active){
            // Layout switches, then OPT. This kernel carries no PROBE
            // tap (only the plain blockwise path is instrumented), so PROBE
            // is named false rather than left to the default.
            //
            // QVEC is named FALSE here, not forwarded. QVEC is the
            // 16-byte `float4` load of the four-component query record, and the
            // trimmed record is three 2-byte components -- `short3` has
            // `alignof 2` and `short4` needs the 8-byte record this deletes,
            // exactly as `kernels.cuh` records for the library side. So the
            // shape loop reads 6 bytes in three LDS.U16 where it read 16 in one
            // LDS.128: fewer bytes and more instructions, and which of those
            // wins is a measurement rather than an argument. The kernel keeps
            // its QVEC template argument so the dispatch surface is unchanged;
            // it selects the float4 path only for the float record, which this
            // kernel no longer builds.
            optimize_one<TILEB_EFF,XHOIST,/*QVEC=*/false,CUT,STAGE16_EFF,NOW,SPLANES,
                         OPT,false,CLOOP,EltB_EFF,short,unsigned char,RECORD>(
                         molA, molA_type, molA_num_atoms_i, NmolA,
                         qr,
                         molB_grad, cntB, cbloB, molB_num_atom, NmolB,
                         rmat, pmat, N_features,
                         optim_color, lr_q, lr_t, nsteps, mixing_param,
                         legacy_combo_gradient, /*first_start=*/(n == 0),
                         va_q, vb_t,
                         scores_register, cut2, molB16, /*probe_out=*/nullptr,
                         qcs, qce, color_placement,
                         start_objectives ? &start_objectives[(size_t)row*N_starts+n] : nullptr,
                         winning_starts ? &winning_starts[row] : nullptr, n);
        }
        __syncthreads();
    }

    if (active){
        for(int j=0;j<DV;++j) scores[(size_t)row*DV+j] = scores_register[j];
    }
}

// Analytic launcher. Same work items, same data, same library layout as the
// variant launcher above -- the transposed 32-lane tile, always, because
// tile+xhoist is the shipped configuration and that is what these arms have
// to beat. XHOIST is therefore always on here too; an arm with every switch
// off is exactly `blockwise_tiled_xhoist`, and exists as the null control.
//
//   qvec      : 16-byte shared query load. Bit-identical.
//   stage_w   : 0 = off; otherwise the block width, and the library molecule is
//               copied into a per-thread planar shared buffer once per pair
//               instead of being re-read from global on every optimizer step.
//               The caller must chunk work items to the same value.
//   stage_i16 : quantize the staged library to int16 at the wire-step.
//   stage_p3  : store only x,y,z in the staged record (6 B/atom, not 8).
//               Requires `now`, and it is what keeps two blocks per SM
//               resident at L = 32 where the 4-plane record holds only one.
//   now       : drop `*wa*wb` from the shape term. Value-preserving: the shape
//               loops are bounded by the real atom counts, where every weight
//               is exactly 1.0f.
//   cut2      : > 0 enables the distance cutoff. A declared numerics change.
//
// The instantiation list below is explicit rather than a cross product: a
// cross product over six switches is ~100 copies of a 3000-instruction kernel
// in five architectures, and only the dozen combinations these arms measure
// are worth compiling.


/// Where the staged rung sits in the analytic dispatch key.
#define ANALYTIC_STAGE_KEY_SHIFT 5

/// The staged rung's index in the analytic dispatch key, or -1 for a width with
/// no compiled cell.
///
/// `stage_width` is a template parameter AND the block dimension, so it cannot
/// be a free runtime integer: every rung is a compiled instantiation. What is
/// chosen at runtime is which rung, and that choice is made against the
/// device's own `cudaDevAttrMaxSharedMemoryPerBlockOptin`
/// (`engine.config.stage_width_for`), not against a constant.
///
/// The rungs are multiples of `ROSHAMBO3_BTILE` because the launcher chunks
/// work items to `chunk = stage_width`: a width off the tile boundary starts
/// every work item mid-tile and every warp then reads across two tiles.
///
/// Widths ascend so that the index reads as a position on the ladder. Adding a
/// rung means adding an entry here, a `case` in the dispatch switch, and an
/// entry in `engine.config.STAGE_WIDTHS` -- all three, or the ladder offers a
/// width the switch refuses.
static inline int analytic_stage_dispatch_index(int stage_width){
    switch (stage_width){
        case 0:   return 0;
        case 32:  return 1;
        case 64:  return 2;
        case 96:  return 3;
        case 128: return 4;
        case 160: return 5;
        case 192: return 6;
        case 224: return 7;
        case 256: return 8;
        default:  return -1;
    }
}

template<typename EltB>
void optimize_pairs_analytic(
        const float * molAs, const int * molA_types, const int * molA_num_atoms, int NmolA,
        const EltB * molBs, const r3_counts_t * molB_counts, int NmolB,
        int library_color_offset,
        const int * out_library_row,
        const int * wi_qrow, const int * wi_tstart, const int * wi_tcount, int n_work_items,
        const float * rmat, const float * pmat, int N_features,
        float * scores, bool optim_color, float lr_q, float lr_t, int nsteps, float mixing_param,
        int start_mode, int device_id, bool legacy_combo_gradient,
        const float * va_all, const float * vb_all,
        int tile_mode, bool qvec, int stage_w, bool stage_i16, bool xhoist, float cut2,
        bool now, bool stage_p3, int opt_mode, int color_loop, int color_placement,
        int * winning_starts, float * start_objectives){

    if (n_work_items == 0) return;
    (void)xhoist;                    // always the hoist
    // `tile_mode` is no longer discarded. Its SIGN is `TILEB`'s -- > 0 the
    // interleaved tile [tile][atom][lane][component], < 0 the planar one
    // [tile][atom][component][lane] -- and it reaches the kernel as a template
    // argument, so a layout with no instantiated cell is refused by name rather
    // than silently run at the other one. `tile_mode == 0`, the untiled
    // per-molecule layout, is unreachable here: `interface.cpp` has required a
    // whole number of tiles.
    if (tile_mode == 0)
        throw std::runtime_error(
            "roshambo3.cuda: the analytic kernel reads a tiled library; "
            "tile_mode must be > 0 (interleaved) or < 0 (planar)");
    const bool planar_ = (tile_mode < 0);

    lr_t *= ROSHAMBO3_COORD_SCALE;

    // The QUERY record's width. The two operands' strides are split; the query
    // is still four floats and this sizes its shared-memory copy.
    const int D = 4;
    const int nthreads = (stage_w > 0) ? stage_w : NTHREADS;
    const int planes   = (stage_i16 && stage_p3) ? 3 : 4;
    dim3 blockDim(nthreads);
    dim3 gridDim(n_work_items);
    const size_t stage_bytes = (size_t)stage_w*NmolB*planes*(stage_i16 ? sizeof(short) : sizeof(float));
    // The trimmed query record. `analytic_query_block_bytes` is the
    // kernel's own layout walked in bytes -- one expression, so the launcher's
    // request and the kernel's pointers cannot drift by a byte.
    const bool rp_diag = (color_loop >= COLOR_LOOP_SAME_TYPE);
    size_t sharedMemSize = analytic_query_block_bytes(NmolA, N_features,
                                                      rp_diag, stage_bytes);

    if (stage_p3 && !(stage_i16 && now)){
        throw std::runtime_error("roshambo3.cuda: a 3-plane staged record needs "
                                 "int16 staging and `now` (there is no weight to read)");
    }
    // The query record is three int16 planes and has no weight plane, so
    // the shape loop's `wa` has to be the 1.0f it always was. That is exactly
    // NOW, and it is refused by name here rather than read out of a plane that
    // is not allocated.
    if (!now){
        throw std::runtime_error(
            "roshambo3.cuda: the analytic kernel's shared query record is three "
            "int16 planes (x, y, z) and carries no weight, so `now` is "
            "required. Every loop that could read the weight is bounded by the "
            "real atom counts, where it is exactly 1.0f.");
    }

    cudaSetDevice(device_id);

    int max_shared = 0;
    CUDA_CHECK_ERROR(cudaDeviceGetAttribute(&max_shared,
                     cudaDevAttrMaxSharedMemoryPerBlockOptin, device_id));
    if (sharedMemSize > (size_t)max_shared){
        throw std::runtime_error(
            "roshambo3.cuda: staging " + std::to_string(NmolB) +
            " library atoms for " + std::to_string(stage_w) + " threads needs " +
            std::to_string(sharedMemSize/1024) + " KB of shared memory, but device " +
            std::to_string(device_id) + " allows at most " +
            std::to_string(max_shared/1024) + " KB. Use a smaller stage width, int16, "
            "or the 3-plane record.");
    }

    // The optimizer mode is dispatched inside the launch macro so that
    // every existing (qvec, stage, cut, ...) cell keeps its exact meaning and
    // only ONE more template argument varies. The instantiation list below is
    // already explicit rather than a cross product (see the note above it);
    // OPT multiplies it, so only the shipped staged cell is compiled for OPT > 0.
    // `TB` is the library tile's template value -- ROSHAMBO3_BTILE for the
    // interleaved layout, -ROSHAMBO3_BTILE for the planar one. Every cell
    // forwards the interleaved value through ROSHAMBO3_ANALYTIC_LAUNCH_OPT_CL
    // below; no planar cell is instantiated.
    #define ROSHAMBO3_ANALYTIC_LAUNCH_TB(TB, QV, ST, CU, I16, NW, SP, OPT, CL)            \
        do {                                                                             \
            auto launch_metadata_ = [&](auto tag_) {                                     \
            constexpr bool RECORD = decltype(tag_)::value;                              \
            const void * afn_ = (const void *)                                           \
                optimize_pairs_analytic_kernel<TB, QV, ST, CU, true, I16, NW, SP, OPT, CL, EltB, RECORD>; \
            CUDA_CHECK_ERROR(cudaFuncSetAttribute(afn_,                                  \
                cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sharedMemSize));       \
            /* RECORDED, not set. This launcher has never set a carveout, so      */   \
            /* the driver infers the minimum partition that fits ONE block -- which */   \
            /* is why the staged analytic path sits at one block per SM at every    */   \
            /* demand, and why trimming its query record buys occupancy only once   */   \
            /* someone sets one. The explicit-carveout rule applies here and is not */   \
            /* obeyed here; changing that moves the shipped analytic path's         */   \
            /* performance and is not to be done silently.                          */   \
            int anblk_ = 0;                                                              \
            CUDA_CHECK_ERROR(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&anblk_,       \
                afn_, nthreads, sharedMemSize));                                         \
            r3_record_launch(anblk_, (unsigned long long)sharedMemSize, -1, nthreads);   \
            optimize_pairs_analytic_kernel<TB, QV, ST, CU, true, I16, NW, SP, OPT, CL, EltB, RECORD>  \
                <<<gridDim,blockDim,sharedMemSize>>>(                                    \
                molAs, molA_types, molA_num_atoms, NmolA,                                \
                molBs, molB_counts, NmolB, library_color_offset,                          \
                out_library_row, wi_qrow, wi_tstart, wi_tcount,                           \
                rmat, pmat, N_features,                                                  \
                scores, optim_color, lr_q, lr_t, nsteps, mixing_param, start_mode,       \
                legacy_combo_gradient, va_all, vb_all, cut2, color_placement,             \
                winning_starts, start_objectives);                                     \
            };                                                                           \
            if (winning_starts || start_objectives) launch_metadata_(std::true_type{}); \
            else launch_metadata_(std::false_type{});                                   \
        } while(0)
    #define ROSHAMBO3_ANALYTIC_LAUNCH_OPT_CL(QV, ST, CU, I16, NW, SP, OPT, CL)            \
        ROSHAMBO3_ANALYTIC_LAUNCH_TB(ROSHAMBO3_BTILE, QV, ST, CU, I16, NW, SP, OPT, CL)
    // Every cell but the color cell carries the chosen structure and only
    // that one. A shape-only launch never enters the color loop, so this costs
    // no instantiation and no branch; a color launch cannot reach these cells.
    #define ROSHAMBO3_ANALYTIC_LAUNCH_OPT(QV, ST, CU, I16, NW, SP, OPT)                       \
        ROSHAMBO3_ANALYTIC_LAUNCH_OPT_CL(QV, ST, CU, I16, NW, SP, OPT, COLOR_LOOP_DEFAULT)
    // The color loop structures, on the one cell a color launch
    // reaches. `color_loop` is validated at the top of this function, so the
    // default arm below is unreachable and says so rather than silently
    // launching something else.
    #define ROSHAMBO3_ANALYTIC_LAUNCH_CLOOP(QV, ST, CU, I16, NW, SP, OPT)                     \
        do {                                                                             \
            switch (color_loop){                                                         \
              case COLOR_LOOP_BOUNDED:       ROSHAMBO3_ANALYTIC_LAUNCH_OPT_CL(QV, ST, CU, I16, NW, SP, OPT, COLOR_LOOP_BOUNDED);       break; \
              case COLOR_LOOP_SAME_TYPE:     ROSHAMBO3_ANALYTIC_LAUNCH_OPT_CL(QV, ST, CU, I16, NW, SP, OPT, COLOR_LOOP_SAME_TYPE);     break; \
              case COLOR_LOOP_SAME_TYPE_NOW: ROSHAMBO3_ANALYTIC_LAUNCH_OPT_CL(QV, ST, CU, I16, NW, SP, OPT, COLOR_LOOP_SAME_TYPE_NOW); break; \
              default: throw std::runtime_error("roshambo3.cuda: color_loop "             \
                  + std::to_string(color_loop) + " is not a structure");                  \
            }                                                                            \
        } while(0)
    // The shipped cell (qvec + now + int16 + 3-plane + 256-wide staging) is the
    // only one instantiated for every optimizer mode; every other cell keeps
    // OPT = 0 and rejects a non-zero opt_mode rather than silently ignoring it.
    #define ROSHAMBO3_ANALYTIC_LAUNCH(QV, ST, CU, I16, NW, SP)                                \
        do {                                                                             \
            if (opt_mode != UPDATE_ADAGRAD_QADD)                                            \
                throw std::runtime_error("roshambo3.cuda: opt_mode " \
                    + std::to_string(opt_mode) + " is not compiled for this switch "     \
                    "combination. Mode 1 (ADAGRAD_EXP, the default) is available on the " \
                    "shipped analytic cell (qvec+now+i16+p3, stage 128/256) and on the " \
                    "unstaged qvec+now cell the color path uses; modes 2..4 on the "      \
                    "shipped cell only");                                                \
            ROSHAMBO3_ANALYTIC_LAUNCH_OPT(QV, ST, CU, I16, NW, SP, UpdateRule::AdagradQAdd);      \
        } while(0)
    // The two SHIPPING rules on a cell that a DEFAULT configuration reaches
    // but that is not the shipped shape cell. The color path is why it exists:
    // library staging is shape-only, so a color launch resolves stage_width = 0
    // and lands on the unstaged qvec+now cell -- and its default update
    // rule is 1, so leaving that cell at mode 0 only would make every analytic
    // color launch fail at its own defaults. `stage_width_for` also returns 0 for
    // buckets too wide to stage, so the shape path reaches it too.
    // Modes 2..4 stay shipped-cell-only: they are measurement arms, and each one
    // multiplies an instantiation table that is already explicit rather than a
    // cross product.
    #define ROSHAMBO3_ANALYTIC_LAUNCH_SHIPPING_RULES(QV, ST, CU, I16, NW, SP)                 \
        do {                                                                             \
            switch (opt_mode){                                                           \
              /* Which structures each update rule carries, and why it is
                 not the same set.
                 - The ADDITIVE rule carries all four, because the comparison
                   between structures is pinned to it for the same reason
                   bench/xcheck_roshambo2.py pins it: a comparison must not
                   straddle two optimizers.
                 - Every rule carries the chosen structure AND `BOUNDED`.
                   BOUNDED is not a measurement arm: it is the only structure
                   that evaluates a cross-type pair, so it is what a
                   NON-CANONICAL interaction matrix resolves to, and a caller
                   who supplies their own matrix reaches it at the default rule.
                   Leaving it on the additive rule alone would leave that caller
                   with no correct structure at all. */ \
              case UPDATE_ADAGRAD_QADD: ROSHAMBO3_ANALYTIC_LAUNCH_CLOOP(QV, ST, CU, I16, NW, SP, UpdateRule::AdagradQAdd); break; \
              case UPDATE_ADAGRAD_EXP:                                                   \
                switch (color_loop){                                                     \
                  case COLOR_LOOP_BOUNDED:                                               \
                    ROSHAMBO3_ANALYTIC_LAUNCH_OPT_CL(QV, ST, CU, I16, NW, SP, UpdateRule::AdagradExp, COLOR_LOOP_BOUNDED); break; \
                  default:                                                               \
                    ROSHAMBO3_ANALYTIC_LAUNCH_OPT(QV, ST, CU, I16, NW, SP, UpdateRule::AdagradExp); break; \
                }                                                                        \
                break;                                                                   \
              default:                                                                   \
                throw std::runtime_error("roshambo3.cuda: opt_mode "                        \
                    + std::to_string(opt_mode) + " (Adam, QN2 and BFGS) is compiled "     \
                    "only for the shipped analytic cell (qvec+now+i16+p3, stage 128/256); " \
                    "this cell carries modes 0 and 1");                                  \
            }                                                                            \
        } while(0)
    #define ROSHAMBO3_ANALYTIC_LAUNCH_SHIPPED(ST)                                             \
        do {                                                                             \
            switch (opt_mode){                                                           \
              case UPDATE_ADAGRAD_EXP: ROSHAMBO3_ANALYTIC_LAUNCH_OPT(true, ST, false, true, true, 3, UpdateRule::AdagradExp);  break; \
              case UPDATE_ADAM_EXP:    ROSHAMBO3_ANALYTIC_LAUNCH_OPT(true, ST, false, true, true, 3, UpdateRule::AdamExp);     break; \
              case UPDATE_QN2_EXP:     ROSHAMBO3_ANALYTIC_LAUNCH_OPT(true, ST, false, true, true, 3, UpdateRule::QN2Exp);      break; \
              case UPDATE_BFGS_EXP:    ROSHAMBO3_ANALYTIC_LAUNCH_OPT(true, ST, false, true, true, 3, UpdateRule::BfgsExp);     break; \
              case UPDATE_ADAGRAD_QADD:ROSHAMBO3_ANALYTIC_LAUNCH_OPT(true, ST, false, true, true, 3, UpdateRule::AdagradQAdd); break; \
              default:                                                                       \
                throw std::runtime_error("roshambo3.cuda: opt_mode must be 0..4, got " \
                                         + std::to_string(opt_mode));                        \
            }                                                                            \
        } while(0)

    const bool cut = (cut2 > 0.0f);
    // The staged rung enters the key as an INDEX, not as one bit per width.
    // Two bits held exactly two widths, and the ladder they encoded -- 256 and
    // 128 -- is two factors of two apart, so a fit boundary between them cost
    // the whole factor. On Volta at L = 64 the boundary is 253 and the launch
    // ran at 128; at L = 128 and 136 nothing fit and the launch stopped staging
    // altogether. The rungs are `engine.config.STAGE_WIDTHS` and the arithmetic
    // was measured. Bits 0-4 are unchanged, so the
    // unstaged keys -- 5, and 29, which tests name -- still mean what they did.
    const int  stage_idx = analytic_stage_dispatch_index(stage_w);
    if (stage_idx < 0)
        throw std::runtime_error(
            "roshambo3.cuda: stage_width " + std::to_string(stage_w) +
            " has no instantiated cell. The ladder is every multiple of 32 up "
            "to 256, plus 0; a width off it is refused rather than rounded, because a "
            "launch that silently ran at another width would be a measurement "
            "of something the caller did not ask for.");
    const int  key = (qvec ? 1 : 0) | (cut ? 2 : 0) | (now ? 4 : 0)
                   | (stage_i16 ? 8 : 0) | (stage_p3 ? 16 : 0)
                   | (stage_idx << ANALYTIC_STAGE_KEY_SHIFT);
    // Validated here rather than at each launch site, and refused rather
    // than clamped -- a caller who named a structure has a model of the kernel,
    // and silently substituting another one is what makes a measurement mean
    // something else.
    // COLOR_LOOP_MASKED_PADDED (0) is retired. It was the only structure with
    // a padding-reachable loop, BOUNDED reproduces it
    // bitwise, and it is refused here rather than renumbered away, so a stale 0
    // from an archived script fails loudly instead of selecting a different
    // structure.
    if (color_loop <= COLOR_LOOP_MASKED_PADDED || color_loop >= COLOR_LOOP_COUNT)
        throw std::runtime_error("roshambo3.cuda: color_loop must be "
                                 + std::to_string(COLOR_LOOP_BOUNDED) + ".."
                                 + std::to_string(COLOR_LOOP_COUNT - 1)
                                 + ", got " + std::to_string(color_loop)
                                 + " (0, MASKED_PADDED, is retired)");
    if (color_placement < 0 || color_placement >= COLOR_PLACEMENT_COUNT)
        throw std::runtime_error("roshambo3.cuda: color_placement must be 0 "
                                 "(IN_GRADIENT) or 1 (AT_POSE), got "
                                 + std::to_string(color_placement));
    if ((key != 5 || opt_mode != UPDATE_ADAGRAD_QADD)
        && color_loop != COLOR_LOOP_DEFAULT
        && !(key == 5 && opt_mode == UPDATE_ADAGRAD_EXP
             && color_loop == COLOR_LOOP_BOUNDED))
        throw std::runtime_error(
            "roshambo3.cuda: the color loop structures are instantiated only on "
            "the unstaged qvec+now cell (key 5) at the ADDITIVE update rule "
            "(0, ADAGRAD_QADD). That is the only cell a color launch reaches -- "
            "library staging is shape-only, so stage_width resolves to 0 for "
            "color -- and the additive rule is the one the structure comparison "
            "is pinned to, so that it cannot straddle two optimizers. This "
            "launch composed key " + std::to_string(key) + " at update rule "
            + std::to_string(opt_mode) + "; everything else carries color_loop "
            + std::to_string(COLOR_LOOP_DEFAULT) + " and nothing else");
    // No planar cell is instantiated, so a planar request is a refusal by name
    // rather than a silent run at the interleaved layout -- an inert layout and
    // a slow one look identical from outside.
    if (planar_){
        throw std::runtime_error(
            "roshambo3.cuda: library_tile_mode < 0 (the planar library tile) "
            "is not supported; use the interleaved tile (library_tile_mode > 0).");
    } else {
    switch (key){
        // --- unstaged ---
        case 0:                 ROSHAMBO3_ANALYTIC_LAUNCH(false, 0, false, false, false, 4); break;
        case 1:                 ROSHAMBO3_ANALYTIC_LAUNCH(true,  0, false, false, false, 4); break;
        case 3:                 ROSHAMBO3_ANALYTIC_LAUNCH(true,  0, true,  false, false, 4); break;
        case 4:                 ROSHAMBO3_ANALYTIC_LAUNCH(false, 0, false, false, true,  4); break;
        // Unstaged qvec+now is what a COLOR launch resolves to (staging is
        // shape-only), and what a bucket too wide to stage resolves to, so it
        // carries both shipping update rules rather than mode 0 alone.
        case 5:                 ROSHAMBO3_ANALYTIC_LAUNCH_SHIPPING_RULES(true, 0, false, false, true, 4); break;
        case 7:                 ROSHAMBO3_ANALYTIC_LAUNCH(true,  0, true,  false, true,  4); break;
        // --- staged, fp32, 4 planes (round 1's occupancy-cliff arm) ---
        case 0|(4<<ANALYTIC_STAGE_KEY_SHIFT):  ROSHAMBO3_ANALYTIC_LAUNCH(false, 128, false, false, false, 4); break;
        // --- staged, int16, 4 planes (round 1) ---
        case 8|(4<<ANALYTIC_STAGE_KEY_SHIFT):  ROSHAMBO3_ANALYTIC_LAUNCH(false, 128, false, true, false, 4); break;
        case 8|(8<<ANALYTIC_STAGE_KEY_SHIFT):  ROSHAMBO3_ANALYTIC_LAUNCH(false, 256, false, true, false, 4); break;
        case 1|8|(4<<ANALYTIC_STAGE_KEY_SHIFT):  ROSHAMBO3_ANALYTIC_LAUNCH(true,  128, false, true, false, 4); break;
        case 1|8|(8<<ANALYTIC_STAGE_KEY_SHIFT):  ROSHAMBO3_ANALYTIC_LAUNCH(true,  256, false, true, false, 4); break;
        // --- staged, int16, 3 planes (round 2: 6 B/atom) ---
        case 4|8|16|(4<<ANALYTIC_STAGE_KEY_SHIFT):  ROSHAMBO3_ANALYTIC_LAUNCH(false, 128, false, true, true, 3); break;
        case 4|8|16|(8<<ANALYTIC_STAGE_KEY_SHIFT):  ROSHAMBO3_ANALYTIC_LAUNCH(false, 256, false, true, true, 3); break;
        // The ladder, one case per rung. `engine.config.STAGE_WIDTHS`
        // and this list are the same ladder written twice; a rung in one and
        // not the other is a refused launch, which is the loud direction.
        case 1|4|8|16|(1<<ANALYTIC_STAGE_KEY_SHIFT):  ROSHAMBO3_ANALYTIC_LAUNCH_SHIPPED(32);  break;
        case 1|4|8|16|(2<<ANALYTIC_STAGE_KEY_SHIFT):  ROSHAMBO3_ANALYTIC_LAUNCH_SHIPPED(64);  break;
        case 1|4|8|16|(3<<ANALYTIC_STAGE_KEY_SHIFT):  ROSHAMBO3_ANALYTIC_LAUNCH_SHIPPED(96);  break;
        case 1|4|8|16|(4<<ANALYTIC_STAGE_KEY_SHIFT):  ROSHAMBO3_ANALYTIC_LAUNCH_SHIPPED(128); break;
        case 1|4|8|16|(5<<ANALYTIC_STAGE_KEY_SHIFT):  ROSHAMBO3_ANALYTIC_LAUNCH_SHIPPED(160); break;
        case 1|4|8|16|(6<<ANALYTIC_STAGE_KEY_SHIFT):  ROSHAMBO3_ANALYTIC_LAUNCH_SHIPPED(192); break;
        case 1|4|8|16|(7<<ANALYTIC_STAGE_KEY_SHIFT):  ROSHAMBO3_ANALYTIC_LAUNCH_SHIPPED(224); break;
        case 1|4|8|16|(8<<ANALYTIC_STAGE_KEY_SHIFT):  ROSHAMBO3_ANALYTIC_LAUNCH_SHIPPED(256); break;
        default:
            throw std::runtime_error("roshambo3.cuda: this switch combination is "
                                     "not instantiated (key " + std::to_string(key) + ")");
    }
    }
    #undef ROSHAMBO3_ANALYTIC_LAUNCH
    #undef ROSHAMBO3_ANALYTIC_LAUNCH_SHIPPING_RULES
    #undef ROSHAMBO3_ANALYTIC_LAUNCH_CLOOP
    #undef ROSHAMBO3_ANALYTIC_LAUNCH_OPT
    #undef ROSHAMBO3_ANALYTIC_LAUNCH_OPT_CL
    #undef ROSHAMBO3_ANALYTIC_LAUNCH_TB
    cudaSetDevice(device_id); CUDA_CHECK_ERROR(cudaGetLastError());
    cudaSetDevice(device_id); CUDA_CHECK_ERROR(cudaDeviceSynchronize());
}


////////////////////////////////////////////////////////////////////////////////
/// Wrappers for the testing framework
///
/// These expose the device volume and gradient routines directly, so that they
/// can be compared against the PyTorch autograd reference the same way the C++
/// backend's test_overlap/test_gradient already are. Without them there is no
/// test of the CUDA kernels at this level, only end-to-end comparisons that
/// cannot distinguish a defect the two backends share.
////////////////////////////////////////////////////////////////////////////////

__global__ void volume_test_kernel(const float * molAs, int NmolA, long nA,
                                   const float * molBs, int NmolB, long nB,
                                   const int * dummy_type, const float * dummy_mat,
                                   float * V){

    long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nA*nB) return;

    long i = idx / nB;
    long j = idx % nB;

    float v[2] = {0.0f, 0.0f};
    // Raw (unscaled) coordinates in, physical volume out: the test oracle is a
    // PyTorch reference in Angstrom^3, so the wrapper -- not the kernel -- is
    // where the reduced frame is entered and left.
    //
    // **This call passes `NmolA` for BOTH the real atom count and the padded
    // width**, so unlike every other caller its shape loop runs over the
    // padding and the 4th coordinate component is LOAD-BEARING here: the
    // oracle wrapper hands it a block whose padding weight is 0, and that zero
    // is what makes the padded slots contribute nothing. It is why
    // `volume_single`'s shape loop keeps its `wa*wb` (see the SHAPE_W note
    // there). The count word is 0 and the color offset `NmolA`, which cannot
    // matter: `optim_color` is false.
    volume_single(&molAs[i*NmolA*D], (r3_counts_t)0, NmolA, NmolA, NmolA,
                  &molBs[j*NmolB*D], (r3_counts_t)0, NmolB, NmolB, NmolB,
                  dummy_mat, dummy_mat, 1, false, ROSHAMBO3_COORD_SCALE, v);
    V[idx] = ROSHAMBO3_SHAPE_PREFACTOR*v[0];
}

__global__ void gradient_test_kernel(const float * molA, int NmolA,
                                     const float * molB, int NmolB,
                                     const int * dummy_type, const float * dummy_mat,
                                     const float * q, const float * t,
                                     float * gradient, float * vs){

    float g_c[7] = {0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f};
    // gnorm = 1: the oracle wants the raw gradient, not the S_AA+S_BB-normalized
    // one the optimizer uses. molA and t arrive already in the reduced frame
    // (test_gradient_gpu put them there); the caller converts the result back.
    get_gradient(molA, dummy_type, NmolA, NmolA,
                 molB, (r3_counts_t)0, NmolB, NmolB, NmolB,
                 dummy_mat, dummy_mat, 1, false,
                 1.0f, 0.0f,
                 gradient, g_c, q, t, vs);
}

// Shape overlap volumes for every (A, B) pair. molAs/molBs are [n, Nmol, 4].
void test_overlap_gpu(const float * molAs, int NmolA, long nA,
                      const float * molBs, int NmolB, long nB, float * V){

    int max_atoms = (NmolA > NmolB) ? NmolA : NmolB;

    float * d_molAs; float * d_molBs; float * d_V;
    int * d_type; float * d_mat;
    CUDA_CHECK_ERROR(cudaMalloc(&d_molAs, nA*NmolA*D_HOST*sizeof(float)));
    CUDA_CHECK_ERROR(cudaMalloc(&d_molBs, nB*NmolB*D_HOST*sizeof(float)));
    CUDA_CHECK_ERROR(cudaMalloc(&d_V, nA*nB*sizeof(float)));
    CUDA_CHECK_ERROR(cudaMalloc(&d_type, max_atoms*sizeof(int)));
    CUDA_CHECK_ERROR(cudaMalloc(&d_mat, sizeof(float)));
    CUDA_CHECK_ERROR(cudaMemset(d_type, 0, max_atoms*sizeof(int)));
    CUDA_CHECK_ERROR(cudaMemset(d_mat, 0, sizeof(float)));
    CUDA_CHECK_ERROR(cudaMemcpy(d_molAs, molAs, nA*NmolA*D_HOST*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemcpy(d_molBs, molBs, nB*NmolB*D_HOST*sizeof(float), cudaMemcpyHostToDevice));

    long n = nA*nB;
    volume_test_kernel<<<(n + NTHREADS - 1)/NTHREADS, NTHREADS>>>(
        d_molAs, NmolA, nA, d_molBs, NmolB, nB, d_type, d_mat, d_V);
    CUDA_CHECK_ERROR(cudaGetLastError());
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());

    CUDA_CHECK_ERROR(cudaMemcpy(V, d_V, n*sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(d_molAs); cudaFree(d_molBs); cudaFree(d_V);
    cudaFree(d_type); cudaFree(d_mat);
}

// Shape overlap gradient of a single pair w.r.t. the quaternion and translation.
void test_gradient_gpu(const float * molA, int NmolA,
                       const float * molB, int NmolB,
                       const float * q, const float * t, float * gradient){

    int max_atoms = (NmolA > NmolB) ? NmolA : NmolB;

    float * d_molA; float * d_molB; float * d_q; float * d_t;
    float * d_grad; float * d_vs; int * d_type; float * d_mat;
    CUDA_CHECK_ERROR(cudaMalloc(&d_molA, NmolA*D_HOST*sizeof(float)));
    CUDA_CHECK_ERROR(cudaMalloc(&d_molB, NmolB*D_HOST*sizeof(float)));
    CUDA_CHECK_ERROR(cudaMalloc(&d_q, 4*sizeof(float)));
    CUDA_CHECK_ERROR(cudaMalloc(&d_t, 3*sizeof(float)));
    CUDA_CHECK_ERROR(cudaMalloc(&d_grad, 7*sizeof(float)));
    CUDA_CHECK_ERROR(cudaMalloc(&d_vs, 2*sizeof(float)));
    CUDA_CHECK_ERROR(cudaMalloc(&d_type, max_atoms*sizeof(int)));
    CUDA_CHECK_ERROR(cudaMalloc(&d_mat, sizeof(float)));
    CUDA_CHECK_ERROR(cudaMemset(d_grad, 0, 7*sizeof(float)));
    CUDA_CHECK_ERROR(cudaMemset(d_vs, 0, 2*sizeof(float)));
    CUDA_CHECK_ERROR(cudaMemset(d_type, 0, max_atoms*sizeof(int)));
    CUDA_CHECK_ERROR(cudaMemset(d_mat, 0, sizeof(float)));
    // The kernel expects the query already in the reduced frame -- in
    // production start_mode_transform puts it there -- and a translation in the
    // same scaled units. The library is not scaled here: get_gradient folds the
    // scale into the pose rotation it builds from q.
    std::vector<float> molA_scaled(molA, molA + (size_t)NmolA*D_HOST);
    for (int i = 0; i < NmolA; ++i){
        for (int k = 0; k < 3; ++k) molA_scaled[(size_t)i*D_HOST + k] *= ROSHAMBO3_COORD_SCALE;
    }
    float t_scaled[3] = {ROSHAMBO3_COORD_SCALE*t[0],
                         ROSHAMBO3_COORD_SCALE*t[1],
                         ROSHAMBO3_COORD_SCALE*t[2]};

    CUDA_CHECK_ERROR(cudaMemcpy(d_molA, molA_scaled.data(), NmolA*D_HOST*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemcpy(d_molB, molB, NmolB*D_HOST*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemcpy(d_q, q, 4*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemcpy(d_t, t_scaled, 3*sizeof(float), cudaMemcpyHostToDevice));

    gradient_test_kernel<<<1,1>>>(d_molA, NmolA, d_molB, NmolB, d_type, d_mat,
                                  d_q, d_t, d_grad, d_vs);
    CUDA_CHECK_ERROR(cudaGetLastError());
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());

    CUDA_CHECK_ERROR(cudaMemcpy(gradient, d_grad, 7*sizeof(float), cudaMemcpyDeviceToHost));

    // Back to Angstrom^3 per radian / per Angstrom. With V = P*S and t = t'/s,
    // dV/dq = P * dS/dq but dV/dt = P*s * dS/dt', so the translation components
    // pick up one extra factor of the coordinate scale, not its reciprocal.
    for (int k = 0; k < 4; ++k) gradient[k] *= ROSHAMBO3_SHAPE_PREFACTOR;
    for (int k = 4; k < 7; ++k) gradient[k] *= ROSHAMBO3_SHAPE_PREFACTOR*ROSHAMBO3_COORD_SCALE;

    cudaFree(d_molA); cudaFree(d_molB); cudaFree(d_q); cudaFree(d_t);
    cudaFree(d_grad); cudaFree(d_vs); cudaFree(d_type); cudaFree(d_mat);
}


////////////////////////////////////////////////////////////////////////////////
/// THE GRID PATH, IN THE PRODUCTION KERNEL
///
/// Everything above this line evaluates the query's shape field by summing over
/// the query's atoms, once per (library atom, optimizer step). This section
/// precomputes that field onto a lattice, once per query, and reads it back with
/// one hardware-filtered texture fetch per (library atom, step).
///
/// WHAT IS AND IS NOT DIFFERENT FROM `optimize_pairs_analytic_kernel`
///
/// Identical, deliberately, so that this is a single-factor change:
///   * the gradient's algebra -- `tfac = -2 ln2 * gnorm`, the cross product
///     against `C = 2*(M b)` (the ROTATED, UNTRANSLATED library coordinate),
///     `gnorm = 1/(S_AA + S_BB)` from the self-overlap precompute;
///   * the Adagrad update, including `sqrt(cache) + EPSILON` and the explicit
///     renormalization of q that `UpdateRule::AdagradQAdd` performs;
///   * int16 staging of the library at the wire-step on the RAW coordinate, with the
///     coordinate scale folded into the pose matrix exactly as `get_gradient`
///     folds it;
///   * the atom loop bounded by the real atom count, never by L.
///
/// Different, and this is the whole change:
///   * `vs` and `dvs/dp` come from `sample_field` rather than from a sum over
///     `molA_num_atoms` pair terms. That is the only arithmetic that moves.
///   * four lanes cooperate on one pair (the atom loop is strided by 4 and the
///     seven gradient components are reduced with a butterfly). This grain was
///     needed for memory-level parallelism; production's analytic kernel is one
///     thread per pair and needs none.
///   * a start mode is an initial POSE quaternion against the one field, not a
///     rotation of the query: rotating the query by R and optimising p is the
///     same problem as keeping the field and optimising p' = R^T p from
///     p'_0 = R^T. The pose this kernel converges to is therefore already the
///     composite the host reconstructs, so `scores[16..19]` is reported as the
///     IDENTITY quaternion and `scores[9..15]` carries the whole transform.
///     (See roshambo3.results' pose reconstruction: r <- Ms^T (M r + t).)
///   * the reported score is an EXACT analytic rescore at the winning pose
///     (a cascade), so no interpolated value is ever reported.
////////////////////////////////////////////////////////////////////////////////

// ln2 and its powers, in the same spelling the fp64 host reference uses. `MINUS_2LN2` above is the fp32 constant the analytic path
// carries; the field build needs ln2 itself for the Laplacian.
#define ROSHAMBO3_LN2 0.6931471805599453f

// ---------------------------------------------------------------------------
// The fetch. `tex4hnb`: one tex3D<float4> on a half4 node texture, NO software
// bounds test -- the texture object is created with cudaAddressModeBorder, so
// the hardware returns exactly 0 outside the box, which is the field's true
// asymptote. Node values sit at integer lattice positions and a texel center in
// unnormalised coordinates is at index+0.5, hence the +0.5f.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void sample_field(const QueryField g,
                                                float x, float y, float z,
                                                float &val, float &gx, float &gy, float &gz){
    float fx = (x - g.ox) * g.inv_hs;
    float fy = (y - g.oy) * g.inv_hs;
    float fz = (z - g.oz) * g.inv_hs;
    float4 t = tex3D<float4>((cudaTextureObject_t)g.tex, fx + 0.5f, fy + 0.5f, fz + 0.5f);
    val = t.x; gx = t.y; gy = t.z; gz = t.w;
}

// Exercises the production fetch itself, independently of final rescoring.
__global__ void numerics_sample_field_kernel(QueryField field,
                                             const float* points, int count,
                                             float* samples) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    sample_field(field, points[3*i], points[3*i+1], points[3*i+2],
                 samples[4*i], samples[4*i+1], samples[4*i+2], samples[4*i+3]);
}
void numerics_sample_field_gpu(QueryField field, const float* points,
                               int count, float* samples) {
    if (!count) return;
    numerics_sample_field_kernel<<<(count+127)/128,128>>>(field, points, count, samples);
    CUDA_CHECK_ERROR(cudaGetLastError());
}

// The exact two-phase accumulator arm needs the overlap before it
// can form dT/dV, but deliberately does not consume the gradient components on
// that pass.  Keep the value-only spelling separate so ptxas can drop the three
// unused component moves even though the texture fetch itself remains float4.
__device__ __forceinline__ float sample_field_value(const QueryField g,
                                                     float x, float y, float z){
    const float fx = (x - g.ox) * g.inv_hs;
    const float fy = (y - g.oy) * g.inv_hs;
    const float fz = (z - g.oz) * g.inv_hs;
    return tex3D<float4>((cudaTextureObject_t)g.tex,
                        fx + 0.5f, fy + 0.5f, fz + 0.5f).x;
}

// ---------------------------------------------------------------------------
// The field build. One thread per node.
//
//   F(u)      = sum_i w_i exp2(-|u - a_i|^2)                      value
//   dF/du     = -2 ln2 sum_i w_i exp2(-.) (u - a_i)               analytic gradient
//   lap F     = sum_i w_i (4 ln2^2 |u-a_i|^2 - 6 ln2) exp2(-.)
//   d(lap)/du = sum_i w_i exp2(-.) (20 ln2^2 - 8 ln2^3 |u-a_i|^2) (u - a_i)
//
// and what is STORED is the prefiltered pair (F - c lap F, dF/du - c d(lap)/du)
// with c = hs^2/12. Measured, the prefilter does and does not buy this: it
// makes the signed bias fourth order (measured 1.98 -> 3.98) and leaves the RMS
// and the gradient second and first order respectively. It is free at query-prep
// time, so it is kept, but it is not the enabler -- the exact rescore is.
//
// Gradients are stored as d/du in the REDUCED frame, so the kernel rescales
// nothing after the fetch.
//
// The build runs in fp32 where the host reference ran in fp64. That is
// two orders inside the fp16 texel it is about to be rounded into, and
// was measured rather than assumed.
// ---------------------------------------------------------------------------
// UNIT_FAC selects the SHAPE build, in which the Gaussian coefficient is
// exactly 1 and every expression below is character-identical to what shipped.
// UNIT_FAC = false is the color build: one extra multiply inside `exp2f` and
// `b = ln2 * fac` instead of `ln2`. Both share this body deliberately -- a
// second copy of the Laplacian prefilter is exactly the drift
// doc/NUMERICS_POLICY.md exists to prevent, and the shape channel's bit-identity
// against a shape-only screen is what would catch a copy that drifted.
template<bool UNIT_FAC>
__global__ void field_fill_kernel(const float * __restrict__ molA, int i0, int i1, int NmolA,
                                 FieldGeometry G, float fac, __half * __restrict__ out){
    long idx = blockIdx.x * (long)blockDim.x + threadIdx.x;
    const long NX = G.nx + 1, NY = G.ny + 1, NZ = G.nz + 1;
    const long nn = NX*NY*NZ;
    if (idx >= nn) return;

    const int ix = (int)(idx % NX);
    const int iy = (int)((idx / NX) % NY);
    const int iz = (int)(idx / (NX*NY));
    const float x = G.ox + ix*G.hs;
    const float y = G.oy + iy*G.hs;
    const float z = G.oz + iz*G.hs;

    // The Gaussian is exp2(-fac d^2) = exp(-(ln2 fac) d^2), so every derivative
    // below takes ln2*fac wherever the shape build takes ln2.
    const float b  = UNIT_FAC ? ROSHAMBO3_LN2 : (ROSHAMBO3_LN2*fac);
    const float b2 = b*b, b3 = b2*b;

    float F = 0.0f, Lp = 0.0f;
    float g0 = 0.0f, g1 = 0.0f, g2 = 0.0f;
    float l0 = 0.0f, l1 = 0.0f, l2 = 0.0f;
    for (int i = i0; i < i1; ++i){
        // The query arrives raw (Angstrom); the lattice is in the reduced frame.
        const float ax = ROSHAMBO3_COORD_SCALE*molA[i*D+0];
        const float ay = ROSHAMBO3_COORD_SCALE*molA[i*D+1];
        const float az = ROSHAMBO3_COORD_SCALE*molA[i*D+2];
        const float w  = molA[i*D+3];
        const float dx = x - ax, dy = y - ay, dz = z - az;
        const float d2 = dx*dx + dy*dy + dz*dz;
        const float e  = UNIT_FAC ? exp2f(-d2)*w : exp2f(-fac*d2)*w;
        F  += e;
        Lp += e*(4.0f*b2*d2 - 6.0f*b);
        const float cg = -2.0f*b*e;
        const float cl = e*(20.0f*b2 - 8.0f*b3*d2);
        g0 += cg*dx; g1 += cg*dy; g2 += cg*dz;
        l0 += cl*dx; l1 += cl*dy; l2 += cl*dz;
    }
    const float c = G.hs*G.hs/12.0f;
    out[idx*4+0] = __float2half(F  - c*Lp);
    out[idx*4+1] = __float2half(g0 - c*l0);
    out[idx*4+2] = __float2half(g1 - c*l1);
    out[idx*4+3] = __float2half(g2 - c*l2);
}

FieldGeometry plan_field_lattice_range(const float * molA_host, int i0, int i1,
                                       int NmolA, float h_angstrom, float pad_a){
    (void)NmolA;
    const double s   = (double)ROSHAMBO3_COORD_SCALE;
    const double hs  = (double)h_angstrom * s;
    const double pad = (double)pad_a      * s;
    double lo[3] = { 1e30, 1e30, 1e30};
    double hi[3] = {-1e30,-1e30,-1e30};
    for (int i = i0; i < i1; ++i){
        for (int k = 0; k < 3; ++k){
            double p = s*(double)molA_host[i*D_HOST+k];
            if (p < lo[k]) lo[k] = p;
            if (p > hi[k]) hi[k] = p;
        }
    }
    if (i1 <= i0){ lo[0]=lo[1]=lo[2]=0.0; hi[0]=hi[1]=hi[2]=0.0; }
    FieldGeometry G;
    G.hs = (float)hs;
    G.ox = (float)(lo[0]-pad); G.oy = (float)(lo[1]-pad); G.oz = (float)(lo[2]-pad);
    G.nx = (int)ceil((hi[0]+pad - (lo[0]-pad))/hs);
    G.ny = (int)ceil((hi[1]+pad - (lo[1]-pad))/hs);
    G.nz = (int)ceil((hi[2]+pad - (lo[2]-pad))/hs);
    if (G.nx < 1) G.nx = 1;
    if (G.ny < 1) G.ny = 1;
    if (G.nz < 1) G.nz = 1;
    return G;
}

FieldGeometry plan_field_lattice(const float * molA_host, int n_real, int NmolA,
                                 float h_angstrom, float pad_a){
    return plan_field_lattice_range(molA_host, 0, n_real, NmolA, h_angstrom, pad_a);
}

void build_query_field(const float * molA_dev, int n_real, int NmolA,
                         FieldGeometry G, void * out_nodes_dev, int device_id){
    cudaSetDevice(device_id);
    const long nn = (long)(G.nx+1)*(G.ny+1)*(G.nz+1);
    const int  nt = 256;
    const long nb = (nn + nt - 1)/nt;
    field_fill_kernel<true><<<(int)nb, nt>>>(molA_dev, 0, n_real, NmolA, G, 1.0f,
                                             (__half*)out_nodes_dev);
    CUDA_CHECK_ERROR(cudaPeekAtLastError());
}

void build_query_color_field(const float * molA_dev, int i0, int i1, int NmolA,
                         FieldGeometry G, float fac, void * out_nodes_dev,
                         int device_id){
    cudaSetDevice(device_id);
    const long nn = (long)(G.nx+1)*(G.ny+1)*(G.nz+1);
    const int  nt = 256;
    const long nb = (nn + nt - 1)/nt;
    field_fill_kernel<false><<<(int)nb, nt>>>(molA_dev, i0, i1, NmolA, G, fac,
                                              (__half*)out_nodes_dev);
    CUDA_CHECK_ERROR(cudaPeekAtLastError());
}

// ---------------------------------------------------------------------------
// The staged-library footprint, shared with the host launcher so the two cannot
// disagree. Planar over the CONFORMER GROUP index and padded by two groups, so
// that the four `j` one group touches do not collide in the same banks.
// ---------------------------------------------------------------------------
// The second argument is LANES PER LIBRARY ROW, which is WA only while one
// lane group owns a whole row. Once `S` groups share a row (start-per-lane) a
// block covers `block / (WA*S)` rows rather than `block / WA`, one staged record
// serves all `S` of them, and the staged footprint falls by `S`. Passing WA here
// under S > 1 would over-allocate by exactly S and index the wrong slot.
__host__ __device__ __forceinline__ int stage_pad(int block, int lanes_per_row){
    return block/lanes_per_row + 2;
}
__host__ __device__ __forceinline__ size_t stage_bytes(int L, int block, int lanes_per_row){
    return (size_t)L * 3 * stage_pad(block, lanes_per_row) * sizeof(short);
}
__host__ __device__ __forceinline__ size_t query_bytes(int NmolA){
    return ((size_t)NmolA*sizeof(float4) + 15) & ~(size_t)15;
}

// THE START MODES, AND WHY THEY ARE NOT INITIAL POSES
//
// The obvious construction is to make a start an initial pose quaternion:
// rotating the query by R and optimizing Q is the same PROBLEM as keeping the
// field and optimizing Q' = R^T Q from Q'_0 = R^T, and that is what the
// earlier probes did. It is a different OPTIMIZER, and under `AdagradQAdd` a much worse
// one. That rule is `q <- normalize(q + (0, d))`, which the optimizer block comment
// says "is the intended rotation only at q = (1,0,0,0)"; worse, `d` is the
// torque and has no fourth component, so `adagrad_step` below applies
// `q[0] -= lr_q*g[0]/...` with `g[0]` IDENTICALLY ZERO (the gradient loop writes
// only gradient[1..6] -- see "dq[0] is always zero" in get_gradient). q[0]
// therefore never moves, the trailing renormalization cannot rescue a zero, and
// a start launched at q = (0,1,0,0) is confined to the w = 0 set -- the
// 180-degree rotations -- for its whole trajectory. Three of the four start
// modes are. Measured, on 4,000 conformers x 3 queries at h = 0.3, where the
// field is otherwise exact: median pose RMSD 2.586 A and top-100 set agreement
// 0.443, against 0.006 A and 1.000 for the same kernel at start_mode 0.
//
// CORRECTION. EVERYTHING ABOVE IS TRUE OF `opt_mode = 0` AND
// FALSE OF `opt_mode = 1`, WHICH IS THE DEFAULT ON BOTH KERNELS.
// `quat_left_exp` applies q <- exp(d) (x) q, whose scalar part is
// `e0*q0 - e1*q1 - e2*q2 - e3*q3`; at q = (0,1,0,0) that is
// `-e1 = -(sin|d|/|d|) d[0]`, nonzero on the first step with any x-component of
// torque. More generally left multiplication by a unit quaternion is a
// transitive action on the unit sphere, so the parameterization has no
// singularity anywhere and no pose is unreachable. The trap does not exist
// under the shipped update rule.
//
// The conjugation below is still right, for three reasons -- and only the third
// is about the optimizer at all, which is why the correction above does not
// touch it:
//
//   1. ONE FIELD PER QUERY. On the FIELD path the query's field is built once.
//      If each start rotated the query instead, each start would need its own
//      field. Load-bearing, and independent of the update rule.
//   2. ONE STAGED COORDINATE SET FOR ALL STARTS. `R^T b` folds into the pose
//      matrix, so the atom loop reads the same coordinates every start.
//   3. EXACTNESS of the correspondence, which needs the three facts listed
//      below -- of which Adagrad's per-component cache commuting only for a
//      signed permutation is the one that constrains what a start may be.
//
// And it leaves a real question open rather than closed: the exponential
// map and the start set may be SUBSTITUTES -- four pi-flip starts exist partly to keep every trajectory near
// identity, which is exactly the regime where the additive rule's shrinkage
// does not bite. Measured: 97.9% of gold top-1e5 hits are found by
// exactly one start mode, which is equally consistent with distinct basins and
// with cover for a parameterization that cannot travel, and nobody has
// distinguished those two readings.
//
// So the start rotation is put where production puts it -- outside the pose --
// by conjugating the whole problem instead:
//
//     |R a - (Q b + t)|  =  |a - (Q~ b~ + t~)|,
//     b~ = R^T b,   Q~ = R^T Q R,   t~ = R^T t
//
// with Q~ starting at the identity exactly as production's Q does. `R^T b` is
// folded into the pose matrix (M_eff = Q~ R^T, one 4x4 quaternion product per
// step) so the atom loop reads the same staged coordinate every start and pays
// nothing.
//
// The correspondence is EXACT, not approximate, and that is worth stating
// because it is what licenses calling this a single-factor change:
//   * gradients: delta = R delta', C = 2 Q b = R (2 Q~ b~), and cross(Rx, Ry) =
//     R cross(x, y) for a proper rotation, so g_prod = R g_grid componentwise
//     for both the torque and the translation;
//   * the update: (0, R^T d) = q_R^-1 (x) (0, d) (x) q_R, so `q + (0,d)` and the
//     renormalization commute with the conjugation, and q[0] is invariant under
//     it (the conjugation maps (w, v) -> (w, R^T v));
//   * Adagrad's per-component cache: this is the one step that needs the start
//     rotations to be what they are. R^T is a SIGNED PERMUTATION for every start
//     in start_mode 0, 1 and 2 (they are all axis-aligned multiples of 90
//     degrees), so cache[k] is permuted along with g[k] and the componentwise
//     update commutes. For a general R it would NOT, and a new start mode with
//     an off-axis rotation must re-check this rather than inherit it.
//
// The converged pose in the query's own frame is then P = Q~ R^T, which is
// already the composite the host reconstructs, so `scores[16..19]` is reported
// as the identity quaternion (see the section header).

// The start rotation q_R itself, from the same axis/angle tables
// start_mode_transform uses, so the two cannot drift apart.
__host__ __device__ __forceinline__ void start_quaternion(int method, int index, float * q){
    // Written as selects rather than as an indexed table: `index` is a loop
    // variable, so a table would land in local memory and be re-read inside the
    // per-conformer start loop.
    int   axis = 0;      // 0 = x, 1 = y, 2 = z
    float ang  = 0.f;
    if (method == 1){
        axis = (index == 0) ? 0 : (index - 1);
        ang  = (index == 0) ? 0.f : PI;
    } else if (method == 2){
        axis = (index == 0) ? 0 : ((index - 1) % 3);
        ang  = (index == 0) ? 0.f
             : (index <= 3) ? PI
             : (index <= 6) ? 0.5f*PI : -0.5f*PI;
    }
    float aa[4];
    aa[0] = (axis == 0) ? 1.f : 0.f;
    aa[1] = (axis == 1) ? 1.f : 0.f;
    aa[2] = (axis == 2) ? 1.f : 0.f;
    aa[3] = ang;
    axis_angle_to_quat(aa, q);
}

// ---------------------------------------------------------------------------
// THE PRECONDITION THE CONJUGATION RESTS ON, CHECKED INSTEAD OF COMMENTED
//
// The third of the three proofs above -- that Adagrad's per-component cache
// commutes with the conjugation -- holds only because R^T is a SIGNED
// PERMUTATION for every start rotation. A signed permutation sends g[k] to
// +-g[perm(k)], so g[k]*g[k] is permuted without a sign and `cache` is permuted
// along with it; the componentwise update then commutes and the grid trajectory
// is the production trajectory relabeled. For a rotation that mixes components,
// cache[k] accumulates a DIFFERENT quantity in the two frames, the two
// trajectories diverge from the second step onward, and the kernel returns a
// converged pose for the wrong optimization problem -- with no error, no NaN
// and a plausible score. That is the failure this check exists to convert into
// a loud one, for inertia-eigenvalue starts, for anything a future start
// set tries, and for the next reader who assumes the comment is the guarantee.
//
// The predicate is on the rotation MATRIX rather than on the axis and angle it
// was built from, because signed-permutation is the actual condition and it is
// strictly weaker than "an axis-aligned multiple of 90 degrees": the octahedral
// group has 24 elements and every one of them qualifies. Testing the
// construction instead of the consequence would reject start sets that are in
// fact exact.
//
// The tolerance is not a numerics tolerance in doc/NUMERICS_POLICY.md's sense --
// nothing here enters a score, and no arithmetic in the scoring path is changed
// by it. It exists because `start_quaternion` builds each start from cos/sin of
// a float32 multiple of a float32 PI, so a 180-degree turn's scalar part is
// -4.37e-8 rather than 0 and the matrix entries miss {0, +-1} by a few ULPs.
// Measured over all 15 starts of modes 0, 1 and 2,
// the worst miss is 8.742e-08
// (mode 0 is exactly 0), while the smallest miss among the off-axis rotations a
// new start mode would plausibly introduce is 0.0175 (a 1-degree turn) and a
// 45-degree turn misses by 0.2929. 1e-4 sits 1144x above the shipped noise and
// 2929x below a 45-degree turn.
// ---------------------------------------------------------------------------
#define ROSHAMBO3_SIGNED_PERM_TOL 1e-4f

__host__ __device__ __forceinline__ bool matrix_is_signed_permutation(const float M[3][3]){
    // Every entry within tolerance of {-1, 0, +1}, and exactly one entry of
    // magnitude one in every row and every column. The second half cannot fail
    // independently of the first for a proper rotation, but the predicate is
    // three comparisons and a caller may hand it something that is not one.
    int rowhit[3] = {0, 0, 0}, colhit[3] = {0, 0, 0};
    for (int i = 0; i < 3; ++i){
        for (int j = 0; j < 3; ++j){
            const float a = fabsf(M[i][j]);
            if (a <= ROSHAMBO3_SIGNED_PERM_TOL) continue;
            if (fabsf(a - 1.0f) > ROSHAMBO3_SIGNED_PERM_TOL) return false;
            rowhit[i]++; colhit[j]++;
        }
    }
    for (int k = 0; k < 3; ++k)
        if (rowhit[k] != 1 || colhit[k] != 1) return false;
    return true;
}

// The same predicate on a quaternion, which is the form every start is in.
__host__ __device__ __forceinline__ bool quaternion_is_signed_permutation(const float * q){
    float M[3][3];
    quaternion_to_rotation_matrix(q, M);
    return matrix_is_signed_permutation(M);
}

// Host side, once per launch, over the whole start set the mode names. Called
// from optimize_pairs_field before anything is launched and exported to Python
// so a test exercises this exact body rather than a copy of it.
//
// `start_mode_n` lives in __constant__ memory and is not host-addressable, so
// the counts appear here a second time. They are the same three literals and
// this comment is the only thing keeping them together; the exported check is
// range-checked against ROSHAMBO3_START_MODE_COUNT so a fourth mode added
// without touching this function fails here rather than reading past the
// __constant__ array.
void require_signed_permutation_starts(int start_mode){
    if (start_mode < 0 || start_mode >= ROSHAMBO3_START_MODE_COUNT)
        throw std::runtime_error("roshambo3.cuda: start_mode must be 0.."
            + std::to_string(ROSHAMBO3_START_MODE_COUNT - 1)
            + ", got " + std::to_string(start_mode));
    const int n = (start_mode == 0) ? 1 : (start_mode == 1) ? 4 : 10;
    for (int i = 0; i < n; ++i){
        float q[4];
        start_quaternion(start_mode, i, q);
        if (!quaternion_is_signed_permutation(q))
            throw std::runtime_error(
                "roshambo3.cuda: start_mode " + std::to_string(start_mode)
                + " start " + std::to_string(i) + " is not a signed permutation. "
                "The grid kernel conjugates the start rotation out of the pose "
                "(see the block comment over start_quaternion in kernels.cu), and "
                "that reparameterization is exact only for a signed permutation: "
                "Adagrad's per-component cache does not commute with a rotation "
                "that mixes components, so this start would return a converged "
                "pose for a different optimization problem, silently. Either use "
                "an axis-aligned start set or replace the update rule with one "
                "that composes on the manifold.");
    }
}

int signed_permutation_start_count(int start_mode){
    require_signed_permutation_starts(start_mode);
    return (start_mode == 0) ? 1 : (start_mode == 1) ? 4 : 10;
}

// The predicate on one caller-supplied rotation, out of line so a test can
// reach it. This is the same body the launch guard above runs; there is no
// second copy to keep in step.
bool rotation_is_signed_permutation(const float * quat){
    return quaternion_is_signed_permutation(quat);
}

// c = a (x) b, Hamilton product. Used once per step to compose the pose with the
// (constant) start rotation's conjugate.
__device__ __forceinline__ void quat_mul(const float * a, const float * b, float * c){
    c[0] = a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3];
    c[1] = a[0]*b[1] + a[1]*b[0] + a[2]*b[3] - a[3]*b[2];
    c[2] = a[0]*b[2] - a[1]*b[3] + a[2]*b[0] + a[3]*b[1];
    c[3] = a[0]*b[3] + a[1]*b[2] - a[2]*b[1] + a[3]*b[0];
}

// Butterfly all-reduce over the WA cooperating lanes.
template<int WA>
__device__ __forceinline__ void lane_group_reduce(float * v, int n, unsigned mask){
    #pragma unroll
    for (int off = 1; off < WA; off <<= 1)
        for (int k = 0; k < n; ++k)
            v[k] += __shfl_xor_sync(mask, v[k], off, WA);
}

// The reduce-scatter, at any WA that the operation is DEFINED
// for. The width-4 hardwiring below was the last remaining floor in this
// kernel, and generalizing it is what lets `WA` be
// swept at all.
//
// THE ALGORITHM, AND WHY IT LANDS ON THE SAME OWNERSHIP MAP. `NACC` values are
// reduced across `WA` lanes so that afterwards lane `ai` holds the reduced sums
// of components `ai, ai + WA, ai + 2*WA, ...` -- which is exactly the
// `c = ai + t*WA` map the Adagrad bodies already use, so nothing downstream
// changes.
//
// Round k (k = 0 .. log2(WA)-1) halves the live set. Live slots are paired
// adjacently; the lane keeps the member of each pair selected by BIT k of `ai`
// and sends the other to its xor-partner at offset `1 << k`:
//
//     keep = v[2j + b_k],  send = v[2j + 1 - b_k],  v'[j] = keep + shfl_xor(send)
//
// By induction, after k rounds slot t holds component `t*2^k + (ai & (2^k -1))`,
// so after log2(WA) rounds slot t holds component `t*WA + ai`. At WA == 4 that
// is the two-round sequence the hand-written macro performed, expression for
// expression -- checked bitwise rather than asserted, because ptxas may
// schedule the same expressions differently.
//
// WHERE IT IS NOT DEFINED. The operation scatters NACC values onto WA lanes, so
// it needs `WA <= NACC`. With NACC = 8 that is WA in {1, 2, 4, 8}; at WA = 16 or
// 32 there are more lanes than values and there is nothing to give the surplus
// lanes. That is a property of the operation, not a limitation of this
// implementation, and UPD = 2's all-reduce covers those widths.
template<int WA, int NACC>
__device__ __forceinline__ void reduce_scatter_lanes(const float * acc, float * out,
                                                     unsigned mask, int ai){
    static_assert(WA >= 1 && (WA & (WA - 1)) == 0, "WA must be a power of two");
    static_assert(NACC >= WA, "reduce-scatter needs at least one value per lane");
    float v[NACC];
    #pragma unroll
    for (int i = 0; i < NACC; ++i) v[i] = acc[i];
    int live = NACC;
    #pragma unroll
    for (int off = 1; off < WA; off <<= 1){
        const bool hi = ((ai / off) & 1) != 0;
        #pragma unroll
        for (int j = 0; j < NACC/2; ++j){
            if (2*j + 1 < live){
                const float keep = hi ? v[2*j + 1] : v[2*j];
                const float send = hi ? v[2*j]     : v[2*j + 1];
                v[j] = keep + __shfl_xor_sync(mask, send, off, WA);
            }
        }
        live /= 2;
    }
    #pragma unroll
    for (int t = 0; t < NACC/WA; ++t) out[t] = v[t];
}

// The reduce-scatter, WA == 4 only. `__shfl_xor` is a SWAP, so in each round a
// lane puts the value its PARTNER will keep into the shuffled register and keeps
// its own: the two lanes trade different value indices in one instruction, which
// is different data rather than predication. Summation order is preserved
// exactly -- every component is still ((a0+a1)+(a2+a3)) -- so this is
// bit-identical to the all-reduce by construction, not by tolerance.
#define REDUCE_SCATTER4(ACC, MASK, AI, S0, S1)                              \
    float S0, S1;                                                               \
    REDUCE_SCATTER4_BODY(ACC, MASK, AI, S0, S1)
// The same body, writing into a caller-declared array, so the
// WA == 4 path and the generalized path can feed one variable name.
#define REDUCE_SCATTER4_INTO(ACC, MASK, AI, OUT)                            \
    REDUCE_SCATTER4_BODY(ACC, MASK, AI, (OUT)[0], (OUT)[1])
#define REDUCE_SCATTER4_BODY(ACC, MASK, AI, D0, D1)                         \
    {   const int o1 = ((AI) & 1), o2 = ((AI) & 2);                             \
        float k0 = o1 ? ACC[1] : ACC[0], s0 = o1 ? ACC[0] : ACC[1];             \
        float k1 = o1 ? ACC[3] : ACC[2], s1 = o1 ? ACC[2] : ACC[3];             \
        float k2 = o1 ? ACC[5] : ACC[4], s2 = o1 ? ACC[4] : ACC[5];             \
        float k3 = o1 ? ACC[7] : ACC[6], s3 = o1 ? ACC[6] : ACC[7];             \
        float e0 = k0 + __shfl_xor_sync(MASK, s0, 1, 4);                        \
        float e1 = k1 + __shfl_xor_sync(MASK, s1, 1, 4);                        \
        float e2 = k2 + __shfl_xor_sync(MASK, s2, 1, 4);                        \
        float e3 = k3 + __shfl_xor_sync(MASK, s3, 1, 4);                        \
        float m0 = o2 ? e1 : e0, u0 = o2 ? e0 : e1;                             \
        float m1 = o2 ? e3 : e2, u1 = o2 ? e2 : e3;                             \
        D0 = m0 + __shfl_xor_sync(MASK, u0, 2, 4);                              \
        D1 = m1 + __shfl_xor_sync(MASK, u1, 2, 4);                              \
    }

// ---------------------------------------------------------------------------
// The exponential-map update with the seven pose components DISTRIBUTED over WA
// cooperating lanes -- UPD 2's and UPD 3's arrangement, where lane `ai` owns
// components `ai, ai + WA, ...` and keeps their Adagrad accumulators.
//
// WHY THIS FUNCTION HAS TO EXIST AT ALL. `UPD` is update *lanes*, not update
// *rule*, and the two do not compose for free. Six of the seven components
// update componentwise and stay on their owning lane, exactly as the additive
// rule's do. The rotation does not: the exp map's step is a three-vector `d` in
// the tangent space and the pose update is
//
//     q <- exp(d) (x) q
//
// a QUATERNION PRODUCT that mixes all four components of `q` with all three of
// `d`. There is no arrangement of lanes under which lane `c` owning component
// `c` computes its own share of that product -- so the additive rule's
// "component c on lane c % WA" cannot simply be reused, and reusing it anyway
// would compute something else.
//
// WHAT MAKES IT WORK ANYWAY is a property UPD = 2 already has: every lane holds
// the WHOLE pose, because `par` is reassembled with `__shfl_sync` at the end of
// every step. So `d` is gathered onto every lane with three shuffles -- the same
// count and the same idiom as the rotation half of the additive rule's own
// reassembly -- and `quat_left_exp` then runs redundantly, on identical
// operands, in every lane. The lanes therefore agree on `par[0..3]` by
// construction and only the translation has to be shuffled back, so the exp map
// costs SIX shuffles per step against the additive rule's seven.
//
// Redundant is not wasteful here, and that is a measured finding read from the
// other direction: a warp issues one instruction stream, so four lanes doing the
// same operation on the same data cost what one lane does. What UPD = 2 buys is
// fewer component *bodies* (two instead of seven), and the exp map keeps that.
//
// The tangent step and `quat_left_exp` below are the ones `apply_update` uses --
// the same function, and character-identical expressions. A known trap is two
// copies of one formula agreeing because they share a typo; there is one copy of
// the quaternion update in this file.
//
// WHAT THIS DOES *NOT* PRESERVE, MEASURED RATHER THAN ASSUMED. `UpdateLanes`
// used to promise that all three arrangements are bit-identical. Under
// AdagradQAdd they still are, and they still are between UPD 2 and UPD 3 under
// either rule -- those two share this function and differ only in the
// reduce-scatter, which preserves summation order exactly. But under the exp map
// **UPD 0 is NOT bit-identical to UPD 2/3**, and it cannot be made so here.
//
// The cause was isolated, not guessed: marking `quat_left_exp` `__noinline__`
// makes all three arms agree bit for bit at 1, 2, 3 and 100 steps, and reverting
// it restores the split. So the difference is ptxas contracting the sixteen
// products of the quaternion composition into FFMAs differently in two different
// register-pressure contexts -- `apply_update` at UPD 0 with cache[7], this
// function at UPD 2 with cache[2] -- which is the failure mode the analytic
// kernel's dispatch already records for `adagrad_step`.
//
// Envelope, measured over the seeded panel,
// every start mode, 25 and 100 steps: mean |dT| 2.6e-5 to 8.4e-5, p99 3.0e-4 to
// 1.5e-3, max 2.3e-3 to 1.8e-2, top-1 set and order unchanged in every cell.
// That is the recompilation amplification curve for a one-FFMA change, which
// `doc/NUMERICS_POLICY.md` section 2 already predicts from the other direction:
// splitting one pair's work across cooperating lanes and recombining it forfeits
// bit-identity NECESSARILY. The additive rule escaped that for the update only
// because its seven components are independent; a quaternion product's are not.
//
// It is left this way ON PURPOSE. Pinning the contraction -- `__fmaf_rn` through
// `quat_left_exp`, or `__noinline__` -- would restore bit-identity across the
// arms, and it would also change the ANALYTIC path's mode 1..4 arithmetic, which
// is the Tier-3 specification. That is a declared numerics change under section 4
// and needs its procedure run on the real panel; it is not a comment-sized
// decision. The shipped arrangement is UPD 2 and it is reproducible; UPD 0 and
// UPD 3 are measurement arms, and must be compared on recall rather than on
// bits under this rule.
// ---------------------------------------------------------------------------
template<int WA, int NBODY, UpdateRule OPT>
__device__ __forceinline__ void apply_update_lanes(
        float * par, float * pst, float * cache, const float * gl,
        int ai, unsigned mask, float lr_q, float lr_t){

    static_assert(OPT == UpdateRule::AdagradExp,
                  "the distributed update is instantiated for AdagradExp only");
    static_assert(WA >= 1 && (WA & (WA - 1)) == 0, "WA must be a power of two");
    static_assert(NBODY * WA >= 7, "every pose component must have an owner");

    // The Adagrad step, in the tangent space, on the owning lane. Component 0
    // has no tangent direction to step along -- that is the whole point of the
    // reparameterisation -- so it is not accumulated and not stepped.
    float drot[NBODY];
    #pragma unroll
    for (int t = 0; t < NBODY; ++t){
        const int c  = ai + t*WA;
        const float gr = gl[t];
        drot[t] = 0.0f;
        if (c >= 1 && c < 4){
            cache[t] += gr*gr;
            drot[t] = -lr_q*gr/(sqrtf(cache[t]) + EPSILON);
        } else if (c >= 4 && c < 7){
            cache[t] += gr*gr;
            pst[t] += -lr_t*gr/(sqrtf(cache[t]) + EPSILON);
        }
    }

    // Gather d, then compose on the manifold in every lane. `q` is a unit
    // quaternion by construction afterwards, which is why the renormalization
    // the additive rule needs is not merely cheap here but absent.
    //
    // THE INDEX MAP, WRITTEN DOWN BECAUSE IT IS CORRECT ONCE AND THEN QUIETLY
    // WRONG AFTER AN EDIT. The parameterization above is `c = ai + t*WA`, which
    // inverts to: pose component `k` is owned by lane `k % WA` in body `k / WA`.
    // For the three components the quaternion product needs:
    //
    //   WA  | comp 1      | comp 2      | comp 3      | shuffles
    //   ----+-------------+-------------+-------------+---------
    //   1   | (lane 0, b1)| (lane 0, b2)| (lane 0, b3)| 0  -- every one is local
    //   2   | (lane 1, b0)| (lane 0, b1)| (lane 1, b1)| 3
    //   >=4 | (lane 1, b0)| (lane 2, b0)| (lane 3, b0)| 3
    //
    // `k / WA` and `k % WA` are compile-time constants in this unrolled loop, so
    // each gather is ONE `__shfl_sync` of ONE named register at every WA >= 2 --
    // never a general transpose. WA = 2 differs from WA >= 4 only in WHICH body
    // register two of the three shuffles read (body 1 rather than body 0), which
    // is a compile-time index, not a runtime one.
    //
    // WA = 1 is the degenerate case and is specialized rather than expressed as
    // a width-1 shuffle: at WA = 1 there is a single lane, NBODY = 7, and every
    // component is already in this lane's own `drot`/`pst`.
    float d[3];
    if constexpr (WA == 1){
        #pragma unroll
        for (int k = 1; k < 4; ++k) d[k-1] = drot[k];
    } else {
        #pragma unroll
        for (int k = 1; k < 4; ++k) d[k-1] = __shfl_sync(mask, drot[k/WA], k % WA, WA);
    }
    quat_left_exp(par, d);

    // Only the translation was updated in place on its owning lane; par[0..3]
    // needs no shuffle because every lane just computed the same product.
    if constexpr (WA == 1){
        #pragma unroll
        for (int k = 4; k < 7; ++k) par[k] = pst[k];
    } else {
        #pragma unroll
        for (int k = 4; k < 7; ++k) par[k] = __shfl_sync(mask, pst[k/WA], k % WA, WA);
    }
}

// ---------------------------------------------------------------------------
// Color on the field path.
//
// The field is the query's SHAPE field and there is no color field here. What
// the optimizer needs from color is dV_color/dpose, which is an analytic pair
// sum over a small set -- the same-type form is 149.2 pairs per screened pair
// against the shape loop's hundreds -- so the hybrid evaluates the shape
// gradient off the texture and the color gradient analytically in the same
// loop.
//
// The structure is COLOR_LOOP_SAME_TYPE_NOW and only that one: it measured
// as the winner and it is the only structure that cannot reach a padding slot,
// which is what lets the weight multiply go. It is exact only for the canonical
// diagonal interaction matrix, so the host refuses any other one by name rather
// than silently computing a different sum.
//
// `cfseg`/`facseg` are the per-type constants -- p^2 a^-1.5 and a/(2 alpha) --
// hoisted out of the pair loop and out of the STEP loop, into a per-block table
// beside `qcs`/`qce`. The analytic path recomputes them per library color slot
// per gradient call (kernels.cu, get_gradient's same-type branch); on the combo
// objective that is once per slot per step, which is where the hoist pays.
__host__ __device__ __forceinline__ size_t color_table_bytes(int N_features){
    return ((size_t)N_features*(2*sizeof(int) + 2*sizeof(float)) + 15) & ~(size_t)15;
}

// The per-type field table, one QueryField per feature type, read once per
// block. `tex == 0` means "this plane is analytic for this query" -- the host
// decides it per (query, plane) at grid-build time, so the choice is uniform
// over the block by construction and there is no ballot and no per-pair
// branch.
// The staged library-COLOR record. Planar over the row index and
// padded by two rows, exactly as the shape stage is and for the same bank
// reason.
//
// THREE int16 planes per slot: x, y and z. It was four, the fourth carrying the
// feature TYPE, because the type had to travel with the coordinates or the loop
// would still touch global once per slot per step and the tier would save
// nothing. That reason expired when the per-slot type plane did: a slot's type
// is `r3_slot_type(cntB, u, N_features)`, a walk over the per-conformer count
// vector's run bounds indexed by the slot's offset `u` within the conformer's
// color run -- and every reader of the staged record already holds `u`. A
// fourth plane would store what the loop derives. Three planes rather than four
// is 25% more slots per staged byte.
//
// The count is named once because two expressions have to agree on it -- the
// host's `sharedMemSize`, through `cstage_bytes`, and the device's plane stride
// -- and a staged record whose two halves disagree reads past its buffer.
#define ROSHAMBO3_COLOR_STAGE_PLANES 3
__host__ __device__ __forceinline__ int cstage_pad(int block, int lanes_per_row){
    return block/lanes_per_row + 2;
}
__host__ __device__ __forceinline__ size_t cstage_bytes(int cstage, int block,
                                                        int lanes_per_row){
    return (size_t)cstage * ROSHAMBO3_COLOR_STAGE_PLANES
         * cstage_pad(block, lanes_per_row) * sizeof(short);
}

__host__ __device__ __forceinline__ size_t color_grid_bytes(int N_features){
    return ((size_t)N_features*sizeof(QueryField) + 15) & ~(size_t)15;
}

// One library color slot's contribution against the query's segment of the SAME
// feature type, at the pose (M, tr). M must already carry ROSHAMBO3_COORD_SCALE,
// exactly as the shape loop's does.
//
// The arithmetic is get_gradient's color body at CLOOP == COLOR_LOOP_SAME_TYPE_NOW,
// term for term: `wa` and `wb` are exactly 1.0f there and are dropped rather
// than multiplied, `delta` is (a_i - t) - R b, and the torque is taken against
// R b -- the rotated, untranslated library coordinate -- which is the same
// quantity the shape loop's cross product acts on.
//
// GRAD = false is the converged-pose evaluation: the overlap only, no torque.
// `CLOAD` is the COLOR channel's load hint, and it is its own axis
// because the channels have different reuse distances and compete with the
// query texture. Shape re-reads its short row on every step; color reuses a
// slot across its query segment and then again on the next step. `__ldcs` is
// therefore not presumed to win for either channel: its arm asks whether
// evicting one channel early protects the other channel enough to compensate.
// The streaming half of `CTIER`, on the same numbering as the shape tier:
//   1  an ordinary load (what shipped, and the default)
//   2  __ldcg -- caches in L2, never allocates an L1 line
//   4  __ldcs -- allocates L1 but evict-first
// Tiers 0 (SHARED) and 3 (REGISTER) do not reach here; they read the staged
// record or their own registers.
// Templated on the library record's element type. The hint is a property
// of the LOAD; the dequantization the int16 arm adds is a property of the
// RECORD, and `r3_fetch*` keeps the two separate.
template<typename Elt>
__device__ __forceinline__ float color_load(const Elt * p, int ctier){
    return (ctier == 2) ? r3_fetch_cg(p) : (ctier == 4) ? r3_fetch_cs(p) : r3_fetch(p);
}
// There is no type plane to hint a load for. A slot's type comes from the
// conformer's count word, which is already in a register -- so `CLOAD`/`CTIER`
// now govern the color COORDINATES and nothing else, and the type read that
// used to sit beside each of them is gone from the memory system entirely.

#include "color_optimizer_exp.cuh"

template<bool GRAD, bool SCORE, bool SEARCH = true>
__device__ __forceinline__ void color_slot_accumulate(
        const float4 * qa, const float * qa_global, int query_color_tier,
        const int * qcs, const int * qce,
        const float * cfseg, const float * facseg, int tb,
        float b0, float b1, float b2,
        const float M[3][3], const float * tr, float tfac_color,
        float gradient_weight, bool color_ballot,
        float * gacc, float * score_acc){

    const int i_lo = qcs[tb], i_hi = qce[tb];
    const bool query_has_type = i_lo != i_hi;
    if (color_ballot){
        const unsigned live = __activemask();
        if (!__any_sync(live, query_has_type)) return;
    }
    // The query carries no feature of this type: one integer compare, because
    // the segment table is per block rather than per pair.
    if (!query_has_type) return;

    const float cf  = cfseg[tb];
    const float fac = facseg[tb];

    const float rx = M[0][0]*b0 + M[0][1]*b1 + M[0][2]*b2;
    const float ry = M[1][0]*b0 + M[1][1]*b1 + M[1][2]*b2;
    const float rz = M[2][0]*b0 + M[2][1]*b1 + M[2][2]*b2;

    // Every factor that is constant over the query segment, out of the segment.
    // `cf` scales both channels linearly, so it rides on the two accumulators
    // rather than on each of the segment's terms.
    const float tcoef = gradient_weight*tfac_color*fac*cf;
    // The XHOIST identity: this is both the translation gradient and the vector the
    // ONE cross product acts on.
    float sj[3] = {0.0f, 0.0f, 0.0f};
    float vc = 0.0f;

    for (int i = i_lo; i < i_hi; ++i){
        const float4 a = (query_color_tier == 0) ? qa[i] : make_float4(
            ROSHAMBO3_COORD_SCALE*color_load(qa_global + (size_t)i*4, query_color_tier),
            ROSHAMBO3_COORD_SCALE*color_load(qa_global + (size_t)i*4 + 1, query_color_tier),
            ROSHAMBO3_COORD_SCALE*color_load(qa_global + (size_t)i*4 + 2, query_color_tier),
            color_load(qa_global + (size_t)i*4 + 3, query_color_tier));
        const float dx = (a.x - tr[0]) - rx;
        const float dy = (a.y - tr[1]) - ry;
        const float dz = (a.z - tr[2]) - rz;
        const float d2 = dx*dx + dy*dy + dz*dz;
        const float kij = SEARCH ? r3_color_optimizer_exp(fac*d2) : exp2f(-fac*d2);
        if (GRAD){
            const float temp = tcoef*kij;
            sj[0] += temp*dx;
            sj[1] += temp*dy;
            sj[2] += temp*dz;
        }
        if constexpr (SCORE) vc += kij;
    }

    if (GRAD){
        const float C[3] = {2.f*rx, 2.f*ry, 2.f*rz};
        gacc[1] += C[1]*sj[2] - C[2]*sj[1];
        gacc[2] += C[2]*sj[0] - C[0]*sj[2];
        gacc[3] += C[0]*sj[1] - C[1]*sj[0];
        gacc[4] += sj[0];
        gacc[5] += sj[1];
        gacc[6] += sj[2];
    }
    if constexpr (SCORE) score_acc[7] += cf*vc;
}

// The two channel passes, behind names, so that the interleave arm
// changes the ORDER they run in and nothing else. The bodies are unchanged --
// `ROSHAMBO3_FIELD_SHAPE_PASS` is character-identical to the loop that shipped,
// which is what keeps COLOR = 0 and COLOR = 1 codegen where it was.
#define ROSHAMBO3_FIELD_SHAPE_PASS(GOUT, SOUT, GW, GRAD, SCORE)               \
            COORD_ATOM_LOOP(                                                  \
                const float rx = M[0][0]*bx0 + M[0][1]*by0 + M[0][2]*bz0;     \
                const float ry = M[1][0]*bx0 + M[1][1]*by0 + M[1][2]*bz0;     \
                const float rz = M[2][0]*bx0 + M[2][1]*by0 + M[2][2]*bz0;     \
                if constexpr (GRAD){                                          \
                    float vs_, fx_, fy_, fz_;                                 \
                    sample_field(g, rx + par[4], ry + par[5], rz + par[6],   \
                                 vs_, fx_, fy_, fz_);                          \
                    const float gxv = (GW)*gfac*fx_;                           \
                    const float gyv = (GW)*gfac*fy_;                           \
                    const float gzv = (GW)*gfac*fz_;                           \
                    const float C[3] = {2.f*rx, 2.f*ry, 2.f*rz};              \
                    (GOUT)[1] += C[1]*gzv - C[2]*gyv;                          \
                    (GOUT)[2] += C[2]*gxv - C[0]*gzv;                          \
                    (GOUT)[3] += C[0]*gyv - C[1]*gxv;                          \
                    (GOUT)[4] += gxv; (GOUT)[5] += gyv; (GOUT)[6] += gzv;     \
                    if constexpr (SCORE) (SOUT)[7] += vs_;                    \
                } else if constexpr (SCORE){                                  \
                    (SOUT)[7] += sample_field_value(                           \
                        g, rx + par[4], ry + par[5], rz + par[6]);             \
                }                                                             \
            )

// One library color slot per lane, strided exactly as the shape atoms are, so
// a lane's color work sits beside its shape work in one instruction stream.
// Read from global rather than from the staged record: staging stops at the
// real atom count and the color slots live past it. `CLOAD` is the hint.
// One library color slot's body, shared by every CTIER so the arithmetic
// cannot drift between them: `tb`, `b0`, `b1`, `b2` are already in registers
// however they got there.
#define ROSHAMBO3_FIELD_COLOR_SLOT(GOUT, SOUT, GW, GRAD, SCORE)               \
                if (tb > 0 && tb < N_features){                               \
                    const QueryField gc = cfield[tb];                         \
                    if (gc.tex != 0ull){                                      \
                        const float rx = M[0][0]*b0 + M[0][1]*b1 + M[0][2]*b2; \
                        const float ry = M[1][0]*b0 + M[1][1]*b1 + M[1][2]*b2; \
                        const float rz = M[2][0]*b0 + M[2][1]*b1 + M[2][2]*b2; \
                        const float cfv = cfseg[tb];                          \
                        if constexpr (GRAD){                                  \
                            float vcv_, fx_, fy_, fz_;                        \
                            sample_field(gc, rx + par[4], ry + par[5],       \
                                         rz + par[6], vcv_, fx_, fy_, fz_);   \
                            const float w = (GW)*gfac_color*cfv;              \
                            const float gx = w*fx_, gy = w*fy_, gz = w*fz_;  \
                            const float C[3] = {2.f*rx, 2.f*ry, 2.f*rz};      \
                            (GOUT)[1] += C[1]*gz - C[2]*gy;                   \
                            (GOUT)[2] += C[2]*gx - C[0]*gz;                   \
                            (GOUT)[3] += C[0]*gy - C[1]*gx;                   \
                            (GOUT)[4] += gx; (GOUT)[5] += gy; (GOUT)[6] += gz; \
                            if constexpr (SCORE) (SOUT)[7] += cfv*vcv_;       \
                        } else if constexpr (SCORE){                          \
                            (SOUT)[7] += cfv*sample_field_value(               \
                                gc, rx + par[4], ry + par[5], rz + par[6]);   \
                        }                                                     \
                    } else {                                                  \
                        color_slot_accumulate<GRAD, SCORE>(                   \
                            qa, molA_global, QCTIER, qcs, qce,                \
                            cfseg, facseg, tb, b0, b1, b2,                    \
                            M, par + 4, tfac_color, (GW), BALLOT,              \
                            (GOUT), (SOUT));                                  \
                    }                                                         \
                }

// Fetching one slot at CTIER 0/1/2/4. `U` is the slot's index within this
// conformer's color run, `J` the absolute slot. TIER 0 reads the int16 staged
// record for U < CSTAGE and falls back to global for the tail -- the fixed
// count with a spill path that the measured distribution asks for (mean 18.88, max 63).
//
// The TYPE comes from the count vector on BOTH sides, and so is hoisted out of
// the branch: `r3_slot_type` reads one 64-bit word that is already in L1 for
// this conformer, which is why the staged record does not have to carry the
// type and why the branch is over coordinates only.
#define ROSHAMBO3_COLOR_FETCH(U, J, TB, B0, B1, B2)                           \
                TB = r3_slot_type(cntB, (J) - cb_lo, N_features);             \
                if (CTIER == 0 && CSTAGE > 0 && (U) < CSTAGE){                \
                    const int cs_ =                                           \
                        ((U)*ROSHAMBO3_COLOR_STAGE_PLANES)*CPAD + rgrp;       \
                    B0 = ROSHAMBO3_STAGE_INV_Q_XYZ*(float)cB16[cs_];          \
                    B1 = ROSHAMBO3_STAGE_INV_Q_XYZ*(float)cB16[cs_ + CPAD];   \
                    B2 = ROSHAMBO3_STAGE_INV_Q_XYZ*(float)cB16[cs_ + 2*CPAD]; \
                } else {                                                      \
                    B0 = color_load(molB + (size_t)(J)*SBG, CTIER);           \
                    B1 = color_load(molB + (size_t)(J)*SBG + CBG, CTIER);     \
                    B2 = color_load(molB + (size_t)(J)*SBG + 2*CBG, CTIER);   \
                }

// One library color slot per lane, strided exactly as the shape atoms are, so
// a lane's color work sits beside its shape work in one instruction stream.
#define ROSHAMBO3_FIELD_COLOR_PASS(GOUT, SOUT, GW, GRAD, SCORE)               \
            if (CTIER == 3 && CREG > 0){                                      \
                _Pragma("unroll")                                             \
                for (int u_ = 0; u_ < (CREG > 0 ? CREG : 1); ++u_){           \
                    const int j = cb_lo + ai + u_*WA;                         \
                    if (j < cb_hi){                                           \
                        const int   tb = (int)crt[u_];                        \
                        const float b0 = crx[u_], b1 = cry[u_], b2 = crz[u_]; \
                        ROSHAMBO3_FIELD_COLOR_SLOT(GOUT, SOUT, GW, GRAD, SCORE) \
                    }                                                         \
                }                                                             \
                for (int j = cb_lo + ai + CREG*WA; j < cb_hi; j += WA){       \
                    int tb; float b0, b1, b2;                                 \
                    ROSHAMBO3_COLOR_FETCH(CSTAGE, j, tb, b0, b1, b2)          \
                    ROSHAMBO3_FIELD_COLOR_SLOT(GOUT, SOUT, GW, GRAD, SCORE)   \
                }                                                             \
            } else {                                                          \
                for (int j = cb_lo + ai; j < cb_hi; j += WA){                 \
                    const int u_ = j - cb_lo;                                 \
                    int tb; float b0, b1, b2;                                 \
                    ROSHAMBO3_COLOR_FETCH(u_, j, tb, b0, b1, b2)              \
                    ROSHAMBO3_FIELD_COLOR_SLOT(GOUT, SOUT, GW, GRAD, SCORE)   \
                }                                                             \
            }

// ---------------------------------------------------------------------------
// The kernel.
//
//   WA     cooperating lanes per (query, library) pair, over the atom loop.
//   UPD    who runs the seven-component Adagrad update:
//            0  every lane runs all seven -- the shipped grain's arrangement,
//               and the control UPD = 2 is measured against IN
//               PRODUCTION rather than in the probe.
//            2  component c is owned by lane c % WA, so each lane runs
//               ceil(7/WA) = 2 component bodies and the pose is reassembled with
//               seven __shfl_sync. Lanes do the SAME operation on DIFFERENT
//               data, which is what lets a warp issue fewer instructions --
//               predicating three lanes off (UPD = 1) measured a
//               2.2% LOSS, because a warp issues one stream.
//            3  UPD = 2 with the butterfly all-reduce replaced by the
//               reduce-scatter.
//   TILEB  the library layout, exactly as `get_gradient` reads it.
//   STAGEW the block width, and therefore the staged buffer's group stride.
//   OPT    the update RULE, not the update lanes -- a distinction this kernel
//          did not have to make until the exp map became the default.
//            AdagradQAdd  the additive step on q[1..3] plus a renormalization.
//                         Character-identical to what this kernel has always
//                         done, deliberately: `bench/xcheck_roshambo2.py` proves
//                         the port moved no bits against R3f and that gate pins
//                         this rule, so its arithmetic is not touched, not even
//                         to route it through `apply_update` (the analytic
//                         kernel makes the same choice for the same reason --
//                         see the note at its dispatch: an inlined-vs-called
//                         `adagrad_step` changed what ptxas contracts into an
//                         FFMA, and on a 100-step trajectory that is the size of
//                         a recompilation).
//            AdagradExp   THE DEFAULT. The same Adagrad step composed on the
//                         manifold, q <- exp(d) (x) q. `apply_update<OPT>` runs
//                         it at UPD = 0 and `apply_update_lanes` at UPD 2/3;
//                         both call the one `quat_left_exp`.
//          Modes 2..4 (Adam, QN2, BFGS) are NOT instantiated here -- see the
//          static_assert in the kernel body for why, and the host launcher for
//          the error a caller gets.
//
// One work item = one block = STAGEW/WA output rows.
// ---------------------------------------------------------------------------
// The ONE place a storage tier changes anything.  The arithmetic that
// consumes bx0/by0/bz0 is character-identical in every tier, which is why the
// tiers are bit-identical rather than merely equivalent: a short -> float round
// trip is exact and the quantization expression below is the staging
// prologue's, verbatim.
//
// The float4 form is used when the library tile is atom-major with contiguous
// components (TILEB > 0, D == 4, which is the shipped ROSHAMBO3_BTILE = 32
// layout): the three coordinates are then 12 of 16 consecutive bytes at a
// 16-byte-aligned address, so one 128-bit load replaces three 32-bit ones.
#define COORD_FETCH(J, BX, BY, BZ)                                            \
    if (TIER == 0){                                                             \
        BX = ROSHAMBO3_STAGE_INV_Q_XYZ*(float)sB16[(((J)*3)+0)*SPAD + rgrp];       \
        BY = ROSHAMBO3_STAGE_INV_Q_XYZ*(float)sB16[(((J)*3)+1)*SPAD + rgrp];       \
        BZ = ROSHAMBO3_STAGE_INV_Q_XYZ*(float)sB16[(((J)*3)+2)*SPAD + rgrp];       \
    } else if constexpr (std::is_same<EltB, float>::value){                     \
        float b0_, b1_, b2_;                                                    \
        if (TILEB > 0 && ROSHAMBO3_D == 4){                                     \
            const float4 * p4_ = (const float4 *)(molB + (size_t)(J)*SBG);       \
            const float4 v4_ = (TIER == 2) ? __ldcg(p4_)                         \
                             : (TIER == 4) ? __ldcs(p4_) : *p4_;                 \
            b0_ = v4_.x; b1_ = v4_.y; b2_ = v4_.z;                               \
        } else if (TIER == 2){                                                   \
            b0_ = __ldcg(molB + (size_t)(J)*SBG);                                \
            b1_ = __ldcg(molB + (size_t)(J)*SBG + CBG);                          \
            b2_ = __ldcg(molB + (size_t)(J)*SBG + 2*CBG);                        \
        } else if (TIER == 4){                                                   \
            b0_ = __ldcs(molB + (size_t)(J)*SBG);                                \
            b1_ = __ldcs(molB + (size_t)(J)*SBG + CBG);                          \
            b2_ = __ldcs(molB + (size_t)(J)*SBG + 2*CBG);                        \
        } else {                                                                 \
            b0_ = molB[(size_t)(J)*SBG];                                         \
            b1_ = molB[(size_t)(J)*SBG + CBG];                                   \
            b2_ = molB[(size_t)(J)*SBG + 2*CBG];                                 \
        }                                                                        \
        BX = ROSHAMBO3_STAGE_INV_Q_XYZ*(float)(short)__float2int_rn(ROSHAMBO3_STAGE_Q_XYZ*b0_); \
        BY = ROSHAMBO3_STAGE_INV_Q_XYZ*(float)(short)__float2int_rn(ROSHAMBO3_STAGE_Q_XYZ*b1_); \
        BZ = ROSHAMBO3_STAGE_INV_Q_XYZ*(float)(short)__float2int_rn(ROSHAMBO3_STAGE_Q_XYZ*b2_); \
    } else {                                                                     \
        /* The streaming tiers on the int16 record. The float32 arm above  \
           round-trips every coordinate through int16 at the wire-step, which is what \
           makes the tiers bit-identical to each other; the record now ARRIVES  \
           at that step, so the round trip is the identity (exhaustively, over  \
           all 65,536 values) and is not                                        \
           performed. There is no int16 float4 analogue: short3 has alignof 2   \
           and short4 needs the 8-byte record the int16 format deletes, so this is three \
           2-byte loads carrying the same per-tier hint. Whether a PLANAR int16 \
           tile reads better was measured separately. */                        \
        BX = (TIER == 2) ? r3_fetch_cg(molB + (size_t)(J)*SBG)                   \
           : (TIER == 4) ? r3_fetch_cs(molB + (size_t)(J)*SBG)                   \
                         : r3_fetch(molB + (size_t)(J)*SBG);                     \
        BY = (TIER == 2) ? r3_fetch_cg(molB + (size_t)(J)*SBG + CBG)             \
           : (TIER == 4) ? r3_fetch_cs(molB + (size_t)(J)*SBG + CBG)             \
                         : r3_fetch(molB + (size_t)(J)*SBG + CBG);               \
        BZ = (TIER == 2) ? r3_fetch_cg(molB + (size_t)(J)*SBG + 2*CBG)           \
           : (TIER == 4) ? r3_fetch_cs(molB + (size_t)(J)*SBG + 2*CBG)           \
                         : r3_fetch(molB + (size_t)(J)*SBG + 2*CBG);             \
    }

// TIER 3, the register-resident tier.  APL = ceil(L / WA) atoms per lane
// held in registers for the whole conformer, so the coordinates leave the
// L1/SMEM array entirely after one load.  That requires the array to be
// COMPILE-TIME indexed and FULLY unrolled -- otherwise it lands in local
// memory, which is backed by L1 and L2 and inverts the point -- so APL is a
// template parameter and the runtime bound `j < NB` becomes a predicate on each
// unrolled slot.  MINB is the other half: registers and blocks/SM are one knob
// (256 threads x 4 blocks x 64 regs is exactly the 64K file), so the register
// budget cannot be raised without lowering occupancy.
//   MINB = 4 / 3 / 2 / 1  ->  ptxas budget 64 / 85 / 128 / 255 registers.
// The probe measured the packed-int16 variant as neither smaller in registers
// nor faster than this one -- ptxas hoists the loop-invariant dequantisation
// either way -- so only the dequantised-float form is carried here.
// An ordinary load here, deliberately. This runs ONCE per
// conformer and the value then lives in a register for the whole pose search,
// so there is no reuse left in the cache line for a hint to protect or evict --
// which is why the `__ldcs`/`__ldcg` axis is a TIER 1/2/4 question and not a
// TIER 3 one.
#define COORD_REG_PROLOGUE                                                    \
    float rfx[(TIER == 3 && APL > 0) ? APL : 1];                                \
    float rfy[(TIER == 3 && APL > 0) ? APL : 1];                                \
    float rfz[(TIER == 3 && APL > 0) ? APL : 1];                                \
    if (TIER == 3 && APL > 0){                                                  \
        _Pragma("unroll")                                                       \
        for (int t = 0; t < (APL > 0 ? APL : 1); ++t){                          \
            const int j = ai + t*WA;                                            \
            const bool ok = (j < NB);                                           \
            /* `r3_quantize` is the shipped quantization on the float32     \
               record and the identity on the int16 one. */                      \
            const short s0_ = ok ? r3_quantize(molB + (size_t)j*SBG)         : (short)0; \
            const short s1_ = ok ? r3_quantize(molB + (size_t)j*SBG + CBG)   : (short)0; \
            const short s2_ = ok ? r3_quantize(molB + (size_t)j*SBG + 2*CBG) : (short)0; \
            rfx[t] = ROSHAMBO3_STAGE_INV_Q_XYZ*(float)s0_;                      \
            rfy[t] = ROSHAMBO3_STAGE_INV_Q_XYZ*(float)s1_;                      \
            rfz[t] = ROSHAMBO3_STAGE_INV_Q_XYZ*(float)s2_;                      \
        }                                                                       \
    }
// The atom loop.  TIER 3 runs APL predicated slots; every other tier runs the
// shipped dynamic trip count.  Both visit j = ai, ai+WA, ai+2WA, ... in that
// order, so every accumulator sees the same summation order.
#define COORD_ATOM_LOOP(...)                                                  \
    if (TIER == 3 && APL > 0){                                                  \
        _Pragma("unroll")                                                       \
        for (int t = 0; t < (APL > 0 ? APL : 1); ++t){                          \
            const int j = ai + t*WA;                                            \
            if (j < NB){                                                        \
                const float bx0 = rfx[t], by0 = rfy[t], bz0 = rfz[t];            \
                __VA_ARGS__                                                     \
            }                                                                   \
        }                                                                       \
    } else {                                                                    \
        for (int j = ai; j < NB; j += WA){                                      \
            float bx0, by0, bz0;                                                \
            COORD_FETCH(j, bx0, by0, bz0)                                     \
            __VA_ARGS__                                                         \
        }                                                                       \
    }

// TIER selects WHERE the library coordinates live between optimizer steps.
//   TIER 0  staged (shipped): quantized once per conformer into the SMEM
//           3-plane record, re-read from shared on every step.
//   TIER 1  streamed: re-read from global on every step through an ordinary
//           load, quantized inline with the staging prologue's expression.
//   TIER 2  streamed through `__ldcg` -> LDG.E.128.STRONG.GPU, which caches in
//           L2 and does NOT allocate a line in L1.  This is the whole point:
//           Measured: streaming with an ordinary load puts the same
//           (atoms x molecules) array back into L1 96.9x re-read, so it buys
//           nothing; a load that never allocates L1 leaves L1 to the field.
//   TIER 4  streamed through `__ldcs` -> LDG.E.128.STRONG.SYS with the
//           evict-first policy: the line IS allocated in L1 but is marked as
//           the first candidate for eviction. That is a different bet from
//           TIER 2's. Shape does make one linear pass per step, but the row is
//           short and is re-read on every step, so `__ldcg` may still win by
//           retaining it in L2. The actual argument for `__ldcs` here is cache
//           partitioning: cheaply evict one channel to protect the query field
//           or library-color working set. It was added because no
//           `__ldcs`, `__ldca` or `__ldcv` existed anywhere in these sources --
//           the hint axis had exactly one non-default point on it.
// TIER 1 and 2 request ZERO staged shared memory, which is what forces
// blocks/SM down to 3 and 2 at the large buckets.
// All three are bit-identical: the quantization expression, the per-lane atom
// order and the accumulation order are character-identical in every tier.
// OPT goes LAST, after every int. It is a scoped enum, so it cannot be
// transposed with any of the int parameters without a diagnostic -- the same
// reason the analytic kernel types its OPT that way (see the parameter-order note over
// `optimize_one`). Appending it also leaves every existing instantiation's
// meaning exactly where it was.
// `S` -- start-per-lane -- goes LAST, after OPT, for the reason OPT itself
// went last (see the note just above): every existing instantiation keeps its
// exact meaning and only one more argument varies. `S` groups of WA lanes share
// ONE library row and split the N_starts between them; S = 1 is what this kernel
// has always done and is character-identical to it at every line below.
// `COLOR` goes LAST, after `S`, for the reason `OPT` and `S` each went
// last before it: every existing instantiation keeps its exact meaning and only
// one more argument varies.
//
//   COLOR 0  no color planes at all. Mode 1, and EVERY LINE BELOW IS
//            CHARACTER-IDENTICAL to what shipped -- which is what keeps the
//            measured residency and lane numbers valid for it.
//   COLOR 1  Mode 2: the pose search is the shape-only one, unchanged,
//            and the color term is evaluated ONCE at the winning pose. The step
//            loop is textually the COLOR 0 loop, which is what makes the shape
//            channel bit-identical to a shape-only screen rather than merely
//            close to it (a checked acceptance criterion).
//   COLOR 2  Mode 3, combo-optimal: both gradients every step, mixed by
//            `combo_gradient_weights`, and the start is selected on the
//            combination Tanimoto.
// `CLOAD` and `ILV` go LAST, after `COLOR`, for the reason every
// parameter before them went last -- every existing instantiation keeps its
// exact meaning and only one more argument varies.
// `EltB` is the LIBRARY record's element type -- `short` for the int16
// wire record, `float` for the four-component one. It is LAST and defaulted, so
// the dispatch tables below name it nowhere and every existing cell keeps its
// exact meaning.
template<int WA, int UPD, int TILEB, int STAGEW, int TIER = 0, int APL = 0,
         int MINB = 4, UpdateRule OPT = UpdateRule::AdagradExp, int S = 1,
         int COLOR = 0, int CTIER = 1, int ILV = 0, int CSTAGE = 0,
         int QCTIER = 0, bool BALLOT = false, typename EltB = float,
         bool METADATA = false>
__global__ __launch_bounds__(COMBO_NTHREADS, MINB) void optimize_pairs_field_kernel(
        const float * molAs_global, const int * molA_num_atoms_global, int NmolA,
        const EltB * molBs,         const r3_counts_t * molB_counts,   int NmolB,
        int library_color_offset,
        const int * out_library_row,
        const int * wi_qrow, const int * wi_tstart, const int * wi_tcount,
        const QueryField * __restrict__ grids,
        float * scores, float lr_q, float lr_t, int nsteps, int start_mode,
        const float * va_all, const float * vb_all,
        const int * molA_types_global,
        const float * rmat_global, const float * pmat_global, int N_features,
        float mixing_param, bool legacy_combo_gradient,
        const QueryField * __restrict__ color_grids, int query_stage_width,
        int shape_stage_width, int * winning_starts = nullptr,
        float * start_objectives = nullptr){

    // The field kernel carries the two SHIPPING update rules and no more.
    // Adam, QN2 and BFGS each need per-pose auxiliary state -- 8, 12 and 33
    // floats on top of `cache` -- and this kernel's register budget is pinned at
    // exactly 64/thread by __launch_bounds__(COMBO_NTHREADS, 4). They are also
    // analytic-path measurement arms rather than shipping rules
    // (`experimental.SHIPPING_UPDATE_RULE`), so instantiating them here would
    // multiply a 39-cell template surface to buy nothing anyone launches. The
    // host launcher refuses them by name; this is the compile-time half.
    static_assert(OPT == UpdateRule::AdagradQAdd || OPT == UpdateRule::AdagradExp,
                  "the field kernel is instantiated for update rules 0 "
                  "(AdagradQAdd) and 1 (AdagradExp) only");

    const int DV = 20;
    // How many Adagrad component bodies THIS lane runs per step.
    constexpr int NBODY = (UPD == 0) ? 7 : ((7 + WA - 1)/WA);

    const int wi     = blockIdx.x;
    const int qrow   = wi_qrow[wi];
    const int tstart = wi_tstart[wi];
    const int tcount = wi_tcount[wi];

    extern __shared__ char field_smem[];
    float4 * qa = (float4 *)field_smem;
    // The per-block color table, between the query and the staged record.
    // At COLOR == 0 `color_table_bytes` is not added by the host and every
    // pointer below is the one that shipped.
    int   * qcs    = (int   *)(field_smem + query_bytes(query_stage_width));
    int   * qce    = qcs + N_features;
    float * cfseg  = (float *)(qce + N_features);
    float * facseg = cfseg + N_features;
    QueryField * cfield = (QueryField *)(field_smem + query_bytes(query_stage_width)
                                         + ((COLOR > 0) ? color_table_bytes(N_features) : 0));

    const int molA_num_atoms_i = molA_num_atoms_global[qrow];
    const float * molA_global  = &molAs_global[(size_t)qrow*NmolA*D];

    // The query, in the reduced frame, unrotated. This is start_mode_transform's
    // n = 0 output and nothing else: on the grid path the start modes live in
    // the initial pose, not in the query.
    for (int i = threadIdx.x; i < query_stage_width; i += blockDim.x){
        qa[i] = make_float4(ROSHAMBO3_COORD_SCALE*molA_global[i*D+0],
                            ROSHAMBO3_COORD_SCALE*molA_global[i*D+1],
                            ROSHAMBO3_COORD_SCALE*molA_global[i*D+2],
                            molA_global[i*D+3]);
    }
    // The query's per-type color segment table and the per-type constants,
    // built ONCE PER BLOCK -- one query serves every library conformer the block
    // scores, at every start and every step. Serialized on thread 0 for the same
    // reason the analytic kernel serializes its own scan: NmolA is ~136 slots at
    // the widest bucket, against tcount * N_starts * (nsteps + 1) gradient
    // calls, so a parallel form would cost more in barriers than the scan costs
    // in total.
    // The per-type field table. One 40-byte struct per feature type, read
    // once per block and then resident -- the alternative is a global read per
    // library color slot per step, for a value that is constant over the query.
    if (COLOR == 2 && color_grids != nullptr){
        for (int ft = threadIdx.x; ft < N_features; ft += blockDim.x)
            cfield[ft] = color_grids[(size_t)qrow*N_features + ft];
    } else if (COLOR > 0){
        for (int ft = threadIdx.x; ft < N_features; ft += blockDim.x)
            cfield[ft].tex = 0ull;
    }
    if (COLOR > 0 && threadIdx.x == 0){
        build_query_color_segments(&molA_types_global[(size_t)qrow*NmolA],
                                   molA_num_atoms_i, NmolA, N_features, qcs, qce);
        for (int ft = 0; ft < N_features; ++ft){
            const float a = rmat_global[ft*N_features+ft];
            float cf = 0.0f, fac = 0.0f;
            if (a != 0.0f){
                const float inv_a = 1.0f/a;
                const float p     = pmat_global[ft*N_features+ft];
                cf  = p*p*inv_a*__fsqrt_rn(inv_a);
                fac = a*ROSHAMBO3_INV_TWO_ALPHA;
            } else {
                // A zero interaction width means this feature type does not
                // interact with itself: its contribution is zero and 1/(2a)
                // would be infinite. Emptying the segment is the query-side
                // skip, so the pair loop never reaches it.
                qcs[ft] = 0; qce[ft] = 0;
            }
            cfseg[ft] = cf; facseg[ft] = fac;
        }
    }
    __syncthreads();

    // ---- the lane assignment --------------------------------------------
    // Two orthogonal splits, and conflating them is a mistake
    // an earlier measurement had to be corrected for:
    //
    //   WA  splits ONE row's ATOM loop across cooperating lanes.
    //   S   makes S lane groups SHARE a row and split the N_starts between
    //       them -- so a block covers COMBO_NTHREADS/(WA*S) DISTINCT rows, not
    //       COMBO_NTHREADS/WA. `wi_tcount` is built to that same width in
    //       interface.cpp; a mismatch shows up as missing or duplicated
    //       library rows, never as a crash.
    //
    // LPR = WA*S lanes own one row. Bounded to a warp so the cross-group best
    // reduction below is a shuffle rather than shared memory plus a barrier.
    constexpr int LPR = WA * S;
    static_assert(WA >= 1 && (WA & (WA - 1)) == 0, "WA must be a power of two");
    static_assert(S  >= 1 && (S  & (S  - 1)) == 0, "S must be a power of two");
    static_assert(LPR <= 32,
                  "S*WA must fit in a warp: the per-row best is reduced with "
                  "__shfl_xor_sync, and a cross-warp reduction needs shared "
                  "memory and a barrier this kernel does not have");
    // UPD 3 is the reduce-scatter and every shuffle in REDUCE_SCATTER4 is
    // hardwired to width 4. Instantiating it at any other WA compiles and
    // computes something else -- which is what an earlier probe
    // did at WA = 8/16/32; its register readings stand as register readings, but
    // that arm was never a correct kernel. Refused at compile time now, so the
    // next reader gets a diagnostic instead of a wrong answer. Use UPD 2, which
    // generalizes to every WA.
    // The reduce-scatter is generalized (see reduce_scatter_lanes),
    // so what remains is the bound the OPERATION has rather than the one the
    // implementation had: it scatters 8 accumulators onto WA lanes and needs at
    // least one value per lane. At WA = 16 or 32 there are more lanes than
    // values and there is nothing to give the surplus ones. UPD 2's all-reduce
    // covers those widths.
    static_assert(UPD != 3 || WA <= 8,
                  "UPD 3's reduce-scatter distributes 8 accumulators over WA "
                  "lanes and needs WA <= 8; use UPD 2 at WA 16 or 32");
    // REDUCE_SCATTER4 fuses the reduction with the scatter, so no lane ever
    // holds the whole shape and color overlaps -- and those are exactly what
    // `combo_gradient_weights` needs before the two gradients can be mixed. The
    // combination objective therefore reduces first and mixes second; UPD 2 is
    // the arrangement that does that and generalizes to every WA.
    static_assert(COLOR != 2 || UPD != 3,
                  "the combination objective needs both channels' overlaps "
                  "reduced before the gradients are mixed, which UPD 3's "
                  "reduce-scatter does not produce; use UPD 0 or UPD 2");

    const int gidx = threadIdx.x / WA;      // lane group in this block
    const int ai   = threadIdx.x % WA;      // atom-dim lane, CONSECUTIVE lanes
    const int rgrp = (S == 1) ? gidx : (gidx / S);   // which row in this block
    const int sgrp = (S == 1) ? 0    : (gidx % S);   // which start subset
    if (rgrp >= tcount) return;             // whole ROW groups leave together

    // 1u << 32 is undefined and ptxas warns about it ("shift count is too
    // large"); at WA == 32 or LPR == 32 the group IS the warp.
    const unsigned mask  = (WA  == 32) ? 0xffffffffu
        : (((1u << WA)  - 1u) << (((threadIdx.x & 31) / WA)  * WA));
    const unsigned rmask = (LPR == 32) ? 0xffffffffu
        : (((1u << LPR) - 1u) << (((threadIdx.x & 31) / LPR) * LPR));
    // One staged record per ROW, shared by the S groups that own it: this is
    // where TIER 0's per-unit-of-work SMEM falls by S.
    const int SPAD = STAGEW/LPR + 2;
    short * sB16 = (short *)(field_smem + query_bytes(query_stage_width)
                             + ((COLOR > 0) ? (color_table_bytes(N_features)
                                               + color_grid_bytes(N_features)) : 0));
    // The staged library-COLOR record sits after the staged shape
    // one, so a shape-only launch's shared layout is byte-for-byte what it was.
    const int CPAD = STAGEW/LPR + 2;
    // `CREG` is the register tier's per-lane tile: the CSTAGE slots a row owns,
    // split over the WA lanes that own it.
    constexpr int CREG = (CSTAGE > 0) ? (CSTAGE + WA - 1)/WA : 0;
    // Sized by the SHAPE plane's width, not the combined record width.
    // The staging loop writes only j < NB and NB <= L_shape, while `NmolB` is
    // L_shape + L_color -- so allocating by NmolB left 42-60% of the shape
    // stage (57% conformer-weighted on a 10M color corpus) allocated and never
    // touched, and cost the staged tier the occupancy it is limited by. This
    // offset and the host's `sharedMemSize` must use the SAME width or the
    // color record starts past the end of the allocation.
    short * cB16 = (short *)((char *)sB16
                             + ((TIER == 0)
                                ? stage_bytes(shape_stage_width, STAGEW, LPR) : 0));

    const int row  = tstart + rgrp;
    const int trow = out_library_row[row];

    const int WB = (TILEB < 0) ? -TILEB : TILEB;
    // The LIBRARY record's own width -- 4 floats, or 3 shorts.
    const int DB = r3_record_width<EltB>::value;
    const EltB * molB;
    if (TILEB > 0)       molB = &molBs[((size_t)(trow/WB)*NmolB*WB + (trow%WB))*DB];
    else if (TILEB < 0)  molB = &molBs[(size_t)(trow/WB)*NmolB*WB*DB + (trow%WB)];
    else                 molB = &molBs[(size_t)trow*NmolB*DB];
    // The per-conformer count vector, one 64-bit word, indexed by `trow`
    // and NOT tiled. The old defect -- a type plane left in per-conformer order
    // while the coordinates are tiled, which hands the color loop another
    // molecule's types and reads back as exactly +0.0 -- is not expressible
    // here: there is no plane to mis-stride, and a warp's 32 lanes hold 32
    // consecutive rows, so this load is already the 256 contiguous bytes the
    // tile transform exists to produce.
    const r3_counts_t cntB = molB_counts[trow];
    const int   NB  = r3_n_real(cntB);
    const float vb0 = vb_all[(size_t)trow*2];
    const float va0 = va_all[(size_t)qrow*2];
    // The color channel's self-overlaps, which the output row already
    // reported (o[7]/o[8]) and the combination now also optimizes against.
    const float vb1 = (COLOR > 0) ? vb_all[(size_t)trow*2+1] : 0.0f;
    const float va1 = (COLOR > 0) ? va_all[(size_t)qrow*2+1] : 0.0f;

    const float sum_s   = va0 + vb0;
    const float gnorm_s = (sum_s > 0.0f) ? 1.0f/sum_s : 0.0f;
    // `get_gradient` accumulates tfac_shape * kij * delta over the query atoms,
    // with tfac_shape = -2 ln2 * gnorm and delta = a_i - p. That sum is exactly
    // -gnorm * dF/dp, because
    //     dF/dp = -2 ln2 sum_i kij (p - a_i) = 2 ln2 sum_i kij (a_i - p),
    // i.e. the -2 ln2 is ALREADY inside the field's stored gradient and must not
    // be applied a second time. Applying it twice both scales by 2 ln2 and FLIPS
    // THE SIGN, turning the ascent into a descent that walks every pose out of
    // the box. The first build of this kernel did exactly that, and
    // a smoke test caught it: grid Tanimoto 0.013 against
    // the analytic path's 0.389, median pose RMSD 5.72 A, top-100 set agreement
    // 0.06. A bit-identity test could not have caught it any better than a
    // recall test did, but a test that only compared MEAN Tanimoto over the
    // whole population would have.
    const float gfac    = -gnorm_s;
    // The color channel's whole per-term gradient coefficient, constant
    // over the pair, exactly as `get_gradient` forms it. Guarded because a
    // molecule with no color features at all has zero color self-overlap.
    const float sum_c   = va1 + vb1;
    const float gnorm_c = (sum_c > 0.0f) ? 1.0f/sum_c : 0.0f;
    const float tfac_color = ROSHAMBO3_MINUS_2LN2*gnorm_c;
    // The field analogue, exactly as `gfac` is the shape one. A color
    // field stores dF/dp with its own -2 ln2 fac already inside, so what is left
    // to apply is -gnorm_c and the plane's scalar height `cf`. Applying the
    // -2 ln2 twice both scales by 2 ln2 and FLIPS THE SIGN -- the defect
    // caught on the shape path.
    const float gfac_color = -gnorm_c;

    const QueryField g = grids[qrow];

    const int SBG = (TILEB != 0) ? WB*DB : DB;
    const int CBG = (TILEB < 0) ? WB : 1;

    // ---- the library's color slot range ------------------------------------
    //
    // Hoisted out of `get_gradient`, which re-derived it on EVERY call
    // by rescanning the type plane for a zero sentinel -- 104 rescans per pair
    // at the shipped defaults, for a range that cannot have moved. The count
    // vector removes the scan itself: the range was always a per-CONFORMER fact expanded into
    // a per-SLOT plane on the host and scanned back here, and now it travels as
    // the two bytes it is.
    int cb_lo = 0, cb_hi = 0;
    if (COLOR > 0){
        // Two array reads and a scan become two byte extracts from a word
        // this thread already holds. `library_color_offset` is where a library
        // conformer's color run begins -- the group's padded shape width, which
        // is uniform over a launch because every launch is single-bucket -- and
        // byte 7 is the run's length.
        cb_lo = (library_color_offset >= 0) ? library_color_offset : NB;
        cb_hi = cb_lo + r3_n_color(cntB);
    }

    // ---- staging, on the RAW coordinate at the wire-step, three planes -------------
    // TIER 0 only.  The streaming tiers do this per step, in COORD_FETCH.
    if (TIER == 0){
        // All LPR lanes that own this row stage it, once, together -- not S
        // redundant copies. At S = 1 this is `j = ai; j += WA` and the slot is
        // `gidx`, character-identical to what shipped.
        for (int j = sgrp*WA + ai; j < NB; j += LPR){
            // The int16 record's shorts go straight in. `r3_quantize` is
            // the shipped quantization on a float32 record and the identity on
            // an int16 one, which the exhaustive round-trip precondition
            // (kernels.cuh) is what licenses.
            sB16[((j*3)+0)*SPAD + rgrp] = r3_quantize(molB + j*SBG);
            sB16[((j*3)+1)*SPAD + rgrp] = r3_quantize(molB + j*SBG+CBG);
            sB16[((j*3)+2)*SPAD + rgrp] = r3_quantize(molB + j*SBG+2*CBG);
        }
        __syncwarp(rmask);
    }
    // TIER 3 fills its APL register slots here, once per conformer, from
    // the same `molB` the staging prologue reads.
    COORD_REG_PROLOGUE

    // ---- the library COLOR record, once per conformer ---------------------
    // Coordinates only, three planes: the reader derives a slot's TYPE from the
    // count vector at the same index it uses to address this record, so staging
    // the type would put a derived quantity in shared memory. It used to be a
    // fourth plane, when the type was a per-slot library plane and deriving it
    // meant touching global once per slot per step; that plane was replaced with
    // the per-conformer count vector and the fourth plane became redundant. See
    // ROSHAMBO3_COLOR_STAGE_PLANES.
    if (COLOR == 2 && CTIER == 0 && CSTAGE > 0){
        for (int u = sgrp*WA + ai; u < CSTAGE; u += LPR){
            const int j = cb_lo + u;
            const bool ok = (j < cb_hi);
            const int  cs = (u*ROSHAMBO3_COLOR_STAGE_PLANES)*CPAD + rgrp;
            cB16[cs]          = ok ? r3_quantize(molB + (size_t)j*SBG)         : (short)0;
            cB16[cs + CPAD]   = ok ? r3_quantize(molB + (size_t)j*SBG+CBG)     : (short)0;
            cB16[cs + 2*CPAD] = ok ? r3_quantize(molB + (size_t)j*SBG+2*CBG)   : (short)0;
        }
        __syncwarp(rmask);
    }
    // The register tier's tile, filled once and held for the whole pose search.
    // Quantized through the same int16 expression the staged tier uses, so the
    // two tiers are the same arithmetic in two places rather than two.
    float crx[(COLOR == 2 && CTIER == 3 && CREG > 0) ? CREG : 1];
    float cry[(COLOR == 2 && CTIER == 3 && CREG > 0) ? CREG : 1];
    float crz[(COLOR == 2 && CTIER == 3 && CREG > 0) ? CREG : 1];
    short crt[(COLOR == 2 && CTIER == 3 && CREG > 0) ? CREG : 1];
    if (COLOR == 2 && CTIER == 3 && CREG > 0){
        #pragma unroll
        for (int u = 0; u < ((CREG > 0) ? CREG : 1); ++u){
            const int j = cb_lo + ai + u*WA;
            const bool ok = (j < cb_hi);
            const short s0 = ok ? r3_quantize(molB + (size_t)j*SBG)         : (short)0;
            const short s1 = ok ? r3_quantize(molB + (size_t)j*SBG + CBG)   : (short)0;
            const short s2 = ok ? r3_quantize(molB + (size_t)j*SBG + 2*CBG) : (short)0;
            crx[u] = ROSHAMBO3_STAGE_INV_Q_XYZ*(float)s0;
            cry[u] = ROSHAMBO3_STAGE_INV_Q_XYZ*(float)s1;
            crz[u] = ROSHAMBO3_STAGE_INV_Q_XYZ*(float)s2;
            crt[u] = ok ? (short)r3_slot_type(cntB, j - cb_lo, N_features) : (short)0;
        }
    }

    const int N_starts = start_mode_n[start_mode];

    float bscore = -3.0e38f;
    float bpar[7] = {1.f,0.f,0.f,0.f,0.f,0.f,0.f};
    // Which start `bscore` came from. At S = 1 this is bookkeeping only;
    // at S > 1 it is what makes the cross-group argmax reproduce the single
    // group's tie-break instead of breaking ties by lane. See the reduction
    // after the loop.
    int bn = 0x7fffffff;

    // Group `sgrp` of the S sharing this row takes starts sgrp, sgrp+S,
    // ... . At S = 1 that is `n = 0; n < N_starts; ++n`, unchanged. Each start's
    // trajectory is independent of every other, is computed by the same WA lanes
    // in the same order whatever S is, and therefore lands on the same bits.
    #pragma unroll 1
    for (int n = sgrp; n < N_starts; n += S){
        // The pose starts at the IDENTITY for every start mode, exactly as
        // production's does; the start rotation lives in `qRc` (see the block
        // comment over start_quaternion).
        float par[7] = {1.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        float qRc[4];
        { float qR[4]; start_quaternion(start_mode, n, qR);
          qRc[0] = qR[0]; qRc[1] = -qR[1]; qRc[2] = -qR[2]; qRc[3] = -qR[3]; }
        float cache[NBODY];
        float pst[NBODY];
        // `apply_update<OPT>`'s auxiliary state. One float for both rules this
        // kernel instantiates; the array exists so the UPD = 0 path can call
        // the shared update rather than keep a second copy of it.
        float aux[update_state<OPT>::N];
        #pragma unroll
        for (int t = 0; t < NBODY; ++t) cache[t] = 0.0f;
        #pragma unroll
        for (int t = 0; t < NBODY; ++t){
            float v = 0.f;
            if (UPD != 0){
                // The `< 7` guard is compile-time decidable, but the SUBSCRIPT
                // is written out either way, and at WA > 8 (NBODY = 1) it names
                // par[8..WA-1]. Clamping keeps the index in range in source; for
                // every reachable k it is the identity, so codegen at the shipped
                // WA = 4 is unchanged.
                #pragma unroll
                for (int k = 0; k < WA; ++k)
                    if (t*WA + k < 7 && ai == k) v = par[(t*WA + k < 7) ? (t*WA + k) : 0];
            }
            pst[t] = v;
        }

        // ILV encodes two independent experimental factors without widening
        // the public launch signature: bit 0 is the warp-staggered schedule;
        // ILV/2 is 0 dual accumulators, 1 exact two-phase, 2 stale prefactors.
        constexpr int ACCUM_MODE = ILV/2;
        constexpr bool STAGGERED = (ILV & 1) != 0;
        static_assert(ILV >= 0 && ILV <= 5, "interleave strategy must be 0..5");
        float stale_a_shape = 0.0f, stale_a_color = 0.0f;

        // `step == nsteps` is the final field evaluation: the converged pose's
        // own grid score, which is what the argmax across starts compares. It
        // is 1/(nsteps+1) of the work and every arm pays it.
        for (int step = 0; step <= nsteps; ++step){
            // P = Q~ R^T: the pose in the query's own (unrotated) frame, which
            // is both what the field must be sampled in and what gets reported.
            float qP[4]; quat_mul(par, qRc, qP);
            float M[3][3];
            quaternion_to_rotation_matrix(qP, M);
            scale_matrix3x3(M, ROSHAMBO3_COORD_SCALE);

            float acc[8]  = {0,0,0,0,0,0,0,0};
            float cacc[8] = {0,0,0,0,0,0,0,0};
            float sov[8]  = {0,0,0,0,0,0,0,0};
            float cov[8]  = {0,0,0,0,0,0,0,0};
            float ov[2]   = {0.0f, 0.0f};

            // Round 3 measures three genuinely different register/arithmetic
            // trades.  The old arm merely reversed two passes while retaining
            // both gradient vectors; that remains ACCUM_MODE 0's control.
            if constexpr (COLOR == 2 && ACCUM_MODE == 0){
                if (STAGGERED && (((threadIdx.x >> 5) & 1) != 0)){
                    ROSHAMBO3_FIELD_COLOR_PASS(cacc, cacc, 1.0f, true, true)
                    ROSHAMBO3_FIELD_SHAPE_PASS(acc, acc, 1.0f, true, true)
                } else {
                    ROSHAMBO3_FIELD_SHAPE_PASS(acc, acc, 1.0f, true, true)
                    ROSHAMBO3_FIELD_COLOR_PASS(cacc, cacc, 1.0f, true, true)
                }
                ov[0] = acc[7]; ov[1] = cacc[7];
                lane_group_reduce<WA>(ov, 2, mask);
            } else if constexpr (COLOR == 2 && ACCUM_MODE == 1){
                // Exact two-phase arm: obtain the two dynamic dT/dV factors,
                // then let every weighted plane add into ONE gradient vector.
                if (STAGGERED && (((threadIdx.x >> 5) & 1) != 0)){
                    ROSHAMBO3_FIELD_COLOR_PASS(cov, cov, 1.0f, false, true)
                    ROSHAMBO3_FIELD_SHAPE_PASS(sov, sov, 1.0f, false, true)
                } else {
                    ROSHAMBO3_FIELD_SHAPE_PASS(sov, sov, 1.0f, false, true)
                    ROSHAMBO3_FIELD_COLOR_PASS(cov, cov, 1.0f, false, true)
                }
                ov[0] = sov[7]; ov[1] = cov[7];
                lane_group_reduce<WA>(ov, 2, mask);
            } else if constexpr (COLOR == 2){
                // Stale arm: seed once at the start pose, then use the prior
                // pose's factors while collecting the current overlaps for the
                // next step. Final scoring always uses current `ov` below.
                if (step == 0){
                    ROSHAMBO3_FIELD_SHAPE_PASS(sov, sov, 1.0f, false, true)
                    ROSHAMBO3_FIELD_COLOR_PASS(cov, cov, 1.0f, false, true)
                    float seed_ov[2] = {sov[7], cov[7]};
                    lane_group_reduce<WA>(seed_ov, 2, mask);
                    const float va2_[2] = {va0, va1};
                    const float vb2_[2] = {vb0, vb1};
                    float ws_, wc_;
                    combo_gradient_weights(va2_, vb2_, seed_ov,
                                           legacy_combo_gradient, &ws_, &wc_);
                    stale_a_shape = (1.0f - mixing_param)*ws_;
                    stale_a_color = mixing_param*wc_;
                    #pragma unroll
                    for (int k = 0; k < 8; ++k){ sov[k] = 0.0f; cov[k] = 0.0f; }
                }
                if (STAGGERED && (((threadIdx.x >> 5) & 1) != 0)){
                    ROSHAMBO3_FIELD_COLOR_PASS(acc, cov, stale_a_color, true, true)
                    ROSHAMBO3_FIELD_SHAPE_PASS(acc, sov, stale_a_shape, true, true)
                } else {
                    ROSHAMBO3_FIELD_SHAPE_PASS(acc, sov, stale_a_shape, true, true)
                    ROSHAMBO3_FIELD_COLOR_PASS(acc, cov, stale_a_color, true, true)
                }
                ov[0] = sov[7]; ov[1] = cov[7];
                lane_group_reduce<WA>(ov, 2, mask);
            } else {
                ROSHAMBO3_FIELD_SHAPE_PASS(acc, acc, 1.0f, true, true)
            }

            if (step == nsteps){
                // Score this start and stop: no update on the final pass.
                if constexpr (COLOR == 2){
                    // The analytic path's rule, on the field path: mode 3's objective IS the
                    // combination, every start travelled under it, and the best
                    // start is the best combination. Mode 2 (COLOR == 1) selects
                    // on shape, below, and each is the criterion its own
                    // objective names.
                    const float den_s = sum_s - ov[0];
                    const float den_c = (va1 + vb1) - ov[1];
                    const float ts_f  = (den_s > 0.0f) ? ov[0]/den_s : 0.0f;
                    const float tc_f  = (den_c > 0.0f) ? ov[1]/den_c : 0.0f;
                    const float sel   = (1.0f - mixing_param)*ts_f + mixing_param*tc_f;
                    if constexpr (METADATA) {
                        if (ai == 0 && start_objectives)
                            start_objectives[(size_t)row*N_starts+n] = sel;
                    }
                    if (sel > bscore){
                        bscore = sel;
                        bn     = n;
                        #pragma unroll
                        for (int k = 0; k < 4; ++k) bpar[k] = qP[k];
                        #pragma unroll
                        for (int k = 4; k < 7; ++k) bpar[k] = par[k];
                    }
                    break;
                }
                float sv[1] = {acc[7]};
                lane_group_reduce<WA>(sv, 1, mask);
                if constexpr (METADATA) {
                    if (ai == 0 && start_objectives)
                        start_objectives[(size_t)row*N_starts+n] = sv[0];
                }
                if (sv[0] > bscore){
                    bscore = sv[0];
                    bn     = n;
                    #pragma unroll
                    for (int k = 0; k < 4; ++k) bpar[k] = qP[k];
                    #pragma unroll
                    for (int k = 4; k < 7; ++k) bpar[k] = par[k];
                }
                break;
            }

            // ---- mix the two channels, once, before the update --------------
            // `combo_gradient_weights` weights each channel by its own dT/dV at
            // the current pose, which is what makes the mixed vector the
            // gradient of the score that is actually REPORTED.
            // Both overlaps have to be whole before the weights exist, so the
            // reduction happens here and the UPD branches below skip theirs.
            // That is also why UPD 3 is refused at COLOR 2: its reduce-scatter
            // never gives any lane a whole overlap.
            constexpr bool PREREDUCED = (COLOR == 2);
            if constexpr (COLOR == 2){
                const float va2[2] = {va0, va1};
                const float vb2[2] = {vb0, vb1};
                float w_shape, w_color;
                combo_gradient_weights(va2, vb2, ov, legacy_combo_gradient,
                                       &w_shape, &w_color);
                const float a_shape = (1.0f - mixing_param)*w_shape;
                const float a_color = mixing_param*w_color;
                if constexpr (ACCUM_MODE == 0){
                    #pragma unroll
                    for (int k = 0; k < 7; ++k)
                        acc[k] = a_shape*acc[k] + a_color*cacc[k];
                } else if constexpr (ACCUM_MODE == 1){
                    // The score-only pass above made the current factors. The
                    // gradient pass intentionally shares one accumulator and
                    // may reassociate shape/color additions by warp.
                    if (STAGGERED && (((threadIdx.x >> 5) & 1) != 0)){
                        ROSHAMBO3_FIELD_COLOR_PASS(acc, cov, a_color, true, false)
                        ROSHAMBO3_FIELD_SHAPE_PASS(acc, sov, a_shape, true, false)
                    } else {
                        ROSHAMBO3_FIELD_SHAPE_PASS(acc, sov, a_shape, true, false)
                        ROSHAMBO3_FIELD_COLOR_PASS(acc, cov, a_color, true, false)
                    }
                } else {
                    // The pass already used the preceding pose's factors. Save
                    // current factors for the next pose after that work.
                    stale_a_shape = a_shape;
                    stale_a_color = a_color;
                }
                lane_group_reduce<WA>(acc, 7, mask);
            }

            if (UPD == 3){
                // The width-4 macro is kept for WA == 4, where
                // `bench/xcheck_roshambo2.py` and the measured numbers are pinned to
                // its exact expressions; every other width goes through the
                // generalized form, which lands on the same ownership map.
                float rs_[(8/WA > 0) ? (8/WA) : 1];
                if constexpr (WA == 4){
                    REDUCE_SCATTER4_INTO(acc, mask, ai, rs_)
                } else if constexpr (WA <= 8){
                    reduce_scatter_lanes<WA, 8>(acc, rs_, mask, ai);
                } else {
                    // Unreachable: the static_assert above refuses UPD 3 at
                    // WA > 8, where there are more lanes than accumulators.
                    // Named rather than left to instantiate a template whose
                    // own precondition it would violate -- `if (UPD == 3)` is a
                    // runtime if on a template constant, so both arms are
                    // INSTANTIATED even in the kernels that never take this one.
                    rs_[0] = 0.0f;
                }
                const float S0 = rs_[0];
                const float S1 = (8/WA > 1) ? rs_[1] : 0.0f;
                if constexpr (OPT == UpdateRule::AdagradQAdd){
                    #pragma unroll
                    for (int t = 0; t < NBODY; ++t){
                        const int c = ai + t*WA;
                        const float gr = t ? S1 : S0;
                        cache[t] += gr*gr;
                        const float lr = (c < 4) ? lr_q : lr_t;
                        const float dp = lr*gr/(sqrt(cache[t]) + EPSILON);
                        if (c < 7) pst[t] -= dp;
                    }
                    #pragma unroll
                    for (int k = 0; k < 7; ++k) par[k] = __shfl_sync(mask, pst[k/WA], k % WA, WA);
                } else {
                    // Lane 3's second body is component 7, which the
                    // reduce-scatter fills with the SCORE accumulator (acc[7]).
                    // The additive branch above leaves that in `cache[1]` and
                    // relies on `pst[1]` never being read back; zeroing it here
                    // is what keeps UPD 2 and UPD 3 bit-identical under the exp
                    // map by construction rather than by an unread slot.
                    float gl[NBODY];
                    #pragma unroll
                    for (int t = 0; t < NBODY; ++t)
                        gl[t] = (ai + t*WA < 7) ? (t ? S1 : S0) : 0.0f;
                    apply_update_lanes<WA, NBODY, OPT>(par, pst, cache, gl,
                                                       ai, mask, lr_q, lr_t);
                }
            } else if (UPD == 2){
                if constexpr (!PREREDUCED) lane_group_reduce<WA>(acc, 7, mask);
                if constexpr (OPT == UpdateRule::AdagradQAdd){
                    #pragma unroll
                    for (int t = 0; t < NBODY; ++t){
                        const int c = ai + t*WA;
                        float gr = 0.f;
                        #pragma unroll
                        for (int k = 0; k < WA; ++k)
                            if (t*WA + k < 7 && ai == k) gr = acc[(t*WA + k < 7) ? (t*WA + k) : 0];
                        cache[t] += gr*gr;
                        const float lr = (c < 4) ? lr_q : lr_t;
                        const float dp = lr*gr/(sqrt(cache[t]) + EPSILON);
                        if (c < 7) pst[t] -= dp;
                    }
                    #pragma unroll
                    for (int k = 0; k < 7; ++k) par[k] = __shfl_sync(mask, pst[k/WA], k % WA, WA);
                } else {
                    float gl[NBODY];
                    #pragma unroll
                    for (int t = 0; t < NBODY; ++t){
                        float gr = 0.f;
                        #pragma unroll
                        for (int k = 0; k < WA; ++k)
                            if (t*WA + k < 7 && ai == k) gr = acc[(t*WA + k < 7) ? (t*WA + k) : 0];
                        gl[t] = gr;
                    }
                    apply_update_lanes<WA, NBODY, OPT>(par, pst, cache, gl,
                                                       ai, mask, lr_q, lr_t);
                }
            } else {
                if constexpr (!PREREDUCED) lane_group_reduce<WA>(acc, 7, mask);
                if constexpr (OPT == UpdateRule::AdagradQAdd){
                    #pragma unroll
                    for (int k = 0; k < 7; ++k){
                        cache[k] += acc[k]*acc[k];
                        const float lr = (k < 4) ? lr_q : lr_t;
                        par[k] -= lr*acc[k]/(sqrt(cache[k]) + EPSILON);
                    }
                } else {
                    // UPD = 0 holds all seven gradient components and all seven
                    // Adagrad accumulators in every lane, which IS
                    // `apply_update`'s interface: par[0..3] is q, par[4..6] is t,
                    // acc[0..6] is g. So the shared update runs unmodified here
                    // and there is no second copy of it on this path.
                    apply_update<OPT>(par, par + 4, acc, cache, aux, step, lr_q, lr_t);
                }
            }

            // UpdateRule::AdagradQAdd's renormalization, which the production kernel
            // performs on every step and the earlier probes did not (they
            // folded it into quat_to_mat's rsqrtf instead). It is not cosmetic:
            // without it |q| drifts upward and the effective rotational step
            // decays, which is a different optimizer.
            //
            // It does not run under the exponential map, and that is the
            // point rather than an optimization. `q` there is a product of unit
            // quaternions, so it is unit BY CONSTRUCTION and a correction applied
            // to it would be a correction to nothing -- the bug class is a
            // renormalization that is subtly the wrong one, and the way to not
            // have it is to not need it. Asserted, not assumed:
            // tests/test_exp_map_field_cuda.py measures max ||q|| - 1 from the
            // returned pose against a 1e-5 gate.
            if (OPT == UpdateRule::AdagradQAdd){
                const float magq = sqrt(par[0]*par[0] + par[1]*par[1] + par[2]*par[2] + par[3]*par[3]);
                #pragma unroll
                for (int k = 0; k < 4; ++k) par[k] = par[k]/magq;
                // Keeping `pst` in step with `par` on the rotation
                // components. `c = ai + t*WA`, so at the shipped WA == 4 this is
                // exactly body 0 on every lane -- the loop below collapses to
                // the single `pst[0] = pst[0]/magq` that shipped, character for
                // character, and is bit-identical there. It is written as a loop
                // because at WA != 4 components 0..3 do NOT all sit in body 0
                // (WA = 2 puts 2 and 3 in body 1; WA = 1 puts each in its own),
                // and the old form would then renormalize the wrong slot.
                if (UPD != 0){
                    #pragma unroll
                    for (int t = 0; t < NBODY; ++t)
                        if (ai + t*WA < 4) pst[t] = pst[t]/magq;
                }
            }
        }
    }

    // ---- reduce the per-start best across the S groups sharing this row ------
    // The S groups each saw a DISJOINT subset of the starts, so the winner is
    // not known to any one of them. The comparison is
    //
    //     (score DESCENDING, start index ASCENDING)
    //
    // which is exactly the total order one group realizes implicitly by keeping
    // only a STRICT improvement (`sv[0] > bscore`) while walking n upward: the
    // survivor is the SMALLEST n attaining the maximum score. Reducing on score
    // alone would leave float32 ties -- which are common, cf. numerics.rank_order
    // and 50,737 in one measured run -- to be broken by whichever lane the butterfly happened
    // to keep, i.e. by S. Carrying `bn` makes the selection identical for every
    // S, which is what lets the whole axis be bit-identical rather than a
    // declared numerics change. A group that drew no start at all (S >
    // N_starts) keeps bn = INT_MAX and loses every comparison.
    //
    // Once per ROW, not once per step: log2(S) rounds of 9 shuffles.
    if constexpr (S > 1){
        #pragma unroll
        for (int off = WA; off < LPR; off <<= 1){
            const float os = __shfl_xor_sync(rmask, bscore, off, LPR);
            const int   on = __shfl_xor_sync(rmask, bn,     off, LPR);
            float op[7];
            #pragma unroll
            for (int k = 0; k < 7; ++k) op[k] = __shfl_xor_sync(rmask, bpar[k], off, LPR);
            if (os > bscore || (os == bscore && on < bn)){
                bscore = os; bn = on;
                #pragma unroll
                for (int k = 0; k < 7; ++k) bpar[k] = op[k];
            }
        }
    }

    if constexpr (METADATA) {
        if (ai == 0 && sgrp == 0 && winning_starts) winning_starts[row] = bn;
    }

    // ---- the cascade: the EXACT analytic overlap, once, at the winner --------
    // Distributed over the WA lanes on the same stride as the optimizer's atom
    // loop -- over WA, NOT over LPR, and that is deliberate. This is the number
    // that is REPORTED, so its summation order is Tier-3 arithmetic rather than
    // Tier-2 pose search, and widening it to LPR lanes would reassociate the sum
    // and make every S > 1 a declared numerics change for no benefit: the S
    // groups sharing this row now hold identical `bpar`, run identical trip
    // counts over identical `j`, and are lanes of ONE warp, so the warp issues
    // the redundant copies in the same instruction stream it would have issued
    // for one (a measured finding, read from the other direction). Redundant here is
    // free; reassociated would not have been.
    // The probe runs this serially on lane 0; at a 22-atom query that is
    // ~22 NB lane-iterations against the optimizer's ~26 NB, i.e. it would very
    // nearly double the kernel. That deviation from the probe is deliberate and
    // it is one of the reasons the probe's ratio does not transfer intact.
    // With this experimental arm disabled, `bscore` remains the final field
    // candidate key.  It is converted to a field Tanimoto below solely so a
    // dense capture can rank survivors; it is never labeled or returned as an
    // exact score.  The pose and winning-start paths above are unchanged.
    float vsum[1] = {0.0f};
    {
        float M[3][3];
        quaternion_to_rotation_matrix(bpar, M);
        scale_matrix3x3(M, ROSHAMBO3_COORD_SCALE);
        const int SBG = (TILEB != 0) ? WB*DB : DB;
        const int CBG = (TILEB < 0) ? WB : 1;
        for (int j = ai; j < NB; j += WA){
            // From global, not from the int16 stage: this is the number that is
            // reported, so it is computed on the same coordinates the analytic
            // kernel would have used. The int16 record does not change that -- it
            // dequantizes to exactly the float32 the host used to send
            // (`r3_fetch`), and the analytic kernel reads the same record.
            const float b0 = r3_fetch(molB+j*SBG), b1 = r3_fetch(molB+j*SBG+CBG),
                        b2 = r3_fetch(molB+j*SBG+2*CBG);
            const float bx = M[0][0]*b0 + M[0][1]*b1 + M[0][2]*b2 + bpar[4];
            const float by = M[1][0]*b0 + M[1][1]*b1 + M[1][2]*b2 + bpar[5];
            const float bz = M[2][0]*b0 + M[2][1]*b1 + M[2][2]*b2 + bpar[6];
            float s = 0.0f;
            for (int i = 0; i < molA_num_atoms_i; ++i){
                const float4 a = qa[i];
                const float dx = a.x - bx, dy = a.y - by, dz = a.z - bz;
                s += exp2f(-(dx*dx + dy*dy + dz*dz));
            }
            vsum[0] += s;
        }
        lane_group_reduce<WA>(vsum, 1, mask);
    }

    // ---- the color overlap at the winning pose ------------------------------
    // This is the number that is REPORTED, and at COLOR == 1 it is the ONLY
    // color evaluation the pair makes -- mode 2, the rescore of a pose
    // that the shape search already found. Distributed over the same WA lanes
    // and the same stride as the exact shape rescore above it.
    float vcsum = 0.0f;
    if constexpr (COLOR > 0 && ROSHAMBO3_FINAL_EXACT_COLOR){
        float M[3][3];
        quaternion_to_rotation_matrix(bpar, M);
        scale_matrix3x3(M, ROSHAMBO3_COORD_SCALE);
        float cacc[8];
        #pragma unroll
        for (int k = 0; k < 8; ++k) cacc[k] = 0.0f;
        // A plain global read, deliberately: this runs ONCE per pair, at the
        // winning pose, and it is the number that is REPORTED -- so it is
        // computed on the coordinates the analytic kernel would have used,
        // never on a quantized staged copy, whatever CTIER the pose search ran
        // under. The same choice the exact SHAPE rescore above it makes.
        for (int j = cb_lo + ai; j < cb_hi; j += WA){
            const int tb = r3_slot_type(cntB, j - cb_lo, N_features);
            if (tb <= 0 || tb >= N_features) continue;
            color_slot_accumulate<false, true, false>(
                qa, molA_global, QCTIER, qcs, qce, cfseg, facseg, tb,
                r3_fetch(molB+(size_t)j*SBG), r3_fetch(molB+(size_t)j*SBG+CBG),
                r3_fetch(molB+(size_t)j*SBG+2*CBG),
                M, bpar + 4, 0.0f, 1.0f, BALLOT, cacc, cacc);
        }
        float cs[1] = {cacc[7]};
        lane_group_reduce<WA>(cs, 1, mask);
        vcsum = cs[0];
    }

    // One writer per ROW, not one per lane group -- at S > 1 the S groups
    // hold the same `bpar` and the same `vsum`, so `ai == 0` alone would have S
    // threads writing identical bytes to one output row.
    if (ai == 0 && sgrp == 0){
        const float vs = ROSHAMBO3_FINAL_EXACT_SHAPE ? vsum[0] : bscore;
        const float den_s = va0 + vb0 - vs;
        const float ts = (den_s > 0.0f) ? vs/den_s : 0.0f;
        // At COLOR == 0 the three lines below are the three that shipped:
        // `total` folds to `ts`, and the color Tanimoto and volume to +0.0f.
        float tc = 0.0f, total = ts;
        if constexpr (COLOR > 0 && ROSHAMBO3_FINAL_EXACT_COLOR){
            const float den_c = va1 + vb1 - vcsum;
            tc    = (den_c > 0.0f) ? vcsum/den_c : 0.0f;
            total = (1.0f - mixing_param)*ts + mixing_param*tc;
        }
        float * o = &scores[(size_t)row*DV];
        o[0]  = total;   // the combination Tanimoto; == ts with no color plane
        o[1]  = ts;
        o[2]  = tc;
        o[3]  = ROSHAMBO3_SHAPE_PREFACTOR*vs;
        o[4]  = ROSHAMBO3_COLOR_PREFACTOR*vcsum;
        o[5]  = ROSHAMBO3_SHAPE_PREFACTOR*va0;
        o[6]  = ROSHAMBO3_SHAPE_PREFACTOR*vb0;
        o[7]  = ROSHAMBO3_COLOR_PREFACTOR*va_all[(size_t)qrow*2+1];
        o[8]  = ROSHAMBO3_COLOR_PREFACTOR*vb_all[(size_t)trow*2+1];
        o[9]  = bpar[0]; o[10] = bpar[1]; o[11] = bpar[2]; o[12] = bpar[3];
        o[13] = ROSHAMBO3_INV_COORD_SCALE*bpar[4];
        o[14] = ROSHAMBO3_INV_COORD_SCALE*bpar[5];
        o[15] = ROSHAMBO3_INV_COORD_SCALE*bpar[6];
        // The start-mode rotation is already inside bpar (see the header note),
        // so the reconstruction's second rotation is the identity.
        o[16] = 1.0f; o[17] = 0.0f; o[18] = 0.0f; o[19] = 0.0f;
    }
}

template<typename EltB>
void optimize_pairs_field(
        const float * molAs, const int * molA_num_atoms, int NmolA,
        const EltB * molBs, const r3_counts_t * molB_counts, int NmolB,
        int library_color_offset,
        const int * out_library_row,
        const int * wi_qrow, const int * wi_tstart, const int * wi_tcount, int n_work_items,
        const QueryField * grids_dev,
        float * scores, float lr_q, float lr_t, int nsteps, int start_mode,
        const float * va_all, const float * vb_all,
        int upd, int carveout, int device_id, int tier, int minb, int opt_mode,
        int start_groups_per_row,
        // Everything the color planes need, and nothing a shape-only
        // launch has to name. `color_mode` is 0 (no color), 1 (mode 2,
        // color at the converged pose) or 2 (mode 3, color in the gradient).
        const int * molA_types,
        const float * rmat, const float * pmat, int N_features,
        float mixing_param, bool legacy_combo_gradient, int color_mode,
        const QueryField * color_grids,
        int lanes_per_group, int color_residency, int color_stage,
        int interleave, int stream_min_blocks_per_sm,
        int query_color_residency, int query_shape_width,
        int color_ballot, int library_shape_width,
        int tile_mode, int * winning_starts, float * start_objectives){

    // The start set is conjugated out of the pose on this path, and that
    // reparameterization is exact only for a signed permutation. Checked here,
    // once per launch, before anything else -- an off-axis start mode returns a
    // wrong pose with no other symptom.
    require_signed_permutation_starts(start_mode);
    if ((winning_starts || start_objectives) && !field_start_metadata_supported(
            tile_mode, lanes_per_group, tier, upd, opt_mode, query_color_residency,
            color_ballot, interleave, color_residency, color_stage,
            start_groups_per_row, stream_min_blocks_per_sm, color_mode))
        throw std::invalid_argument(
            "CUDA FIELD start metadata requires a supported default dispatch cell");

    if (n_work_items == 0) return;

    // The reduced frame, exactly as the analytic launcher enters it.
    lr_t *= ROSHAMBO3_COORD_SCALE;

    // WA is a runtime choice now. `S = 2` won round 1 by halving
    // the shared memory per unit of work, and WA is the other lever on that
    // same quantity, so leaving it compile-time was the gap it was.
    const int WA = lanes_per_group;
    if (WA < 1 || (WA & (WA - 1)) != 0 || WA > 32)
        throw std::runtime_error(
            "roshambo3.cuda: lanes_per_group must be a power of two in 1..32, got "
            + std::to_string(WA));
    // 0 means "the pinned 4 that shipped"; see the note on the declaration.
    const int MINB_S = stream_min_blocks_per_sm ? stream_min_blocks_per_sm : 4;
    if (MINB_S < 1 || MINB_S > 4)
        throw std::runtime_error(
            "roshambo3.cuda: stream_min_blocks_per_sm must be 0 (the shipped 4) "
            "or 1..4");
    const int ctier       = color_residency;
    const int qctier      = query_color_residency;
    // The library tile's layout, as a boolean rather than as a sign,
    // because that is the shape the dispatch guard wants. `tile_mode == 0` --
    // the untiled per-molecule layout -- is not reachable from here: both pair
    // kernels require a tile and `interface.cpp` checks the
    // row count is a whole number of tiles before it gets this far.
    if (tile_mode == 0)
        throw std::runtime_error(
            "roshambo3.cuda: the field kernel reads a tiled library; "
            "tile_mode must be > 0 (interleaved) or < 0 (planar)");
    const bool planar_    = (tile_mode < 0);
    const bool ballot     = color_ballot != 0;
    const int color_stage_ = color_stage;
    const int ILV          = interleave;
    if (ILV < 0 || ILV > 5)
        throw std::runtime_error(
            "roshambo3.cuda: interleave must be 0..5 (dual sequential/staggered, "
            "two-phase sequential/staggered, stale sequential/staggered)");
    if (ctier < 0 || ctier > 4)
        throw std::runtime_error("roshambo3.cuda: color_residency must be 0 "
                                 "(SHARED), 1 (GLOBAL), 2 (GLOBAL_LDCG), "
                                 "3 (REGISTER) or 4 (GLOBAL_LDCS)");
    if ((ctier == 0 || ctier == 3) && color_stage_ <= 0)
        throw std::runtime_error(
            "roshambo3.cuda: color_residency 0 (SHARED) and 3 (REGISTER) are "
            "sized by color_stage, the fixed number of color slots per row "
            "they hold; it must be > 0");
    if ((ctier != 1 || color_stage_ != 0 || ILV) && color_mode != 2)
        throw std::runtime_error(
            "roshambo3.cuda: color_residency, color_stage and interleave "
            "describe the color pass, which only runs at "
            "color_placement=IN_GRADIENT");
    if (qctier != 0 && qctier != 1 && qctier != 2 && qctier != 4)
        throw std::runtime_error(
            "roshambo3.cuda: query_color_residency must be 0 (SHARED), "
            "1 (GLOBAL), 2 (GLOBAL_LDCG), or 4 (GLOBAL_LDCS); REGISTER would "
            "duplicate the complete query color tile in every lane and is "
            "resource-priced/refused rather than silently substituted");
    if (qctier != 0 && color_mode != 2)
        throw std::runtime_error(
            "roshambo3.cuda: non-shared query_color_residency applies only to "
            "color_placement=IN_GRADIENT");
    if (color_ballot != 0 && color_ballot != 1)
        throw std::runtime_error("roshambo3.cuda: color_ballot must be 0 or 1");
    if (ballot && color_mode != 2)
        throw std::runtime_error(
            "roshambo3.cuda: color_ballot applies only to "
            "color_placement=IN_GRADIENT");
    if (query_shape_width <= 0 || query_shape_width > NmolA)
        throw std::runtime_error(
            "roshambo3.cuda: query_shape_width must be in 1..NmolA");
    const int QWIDTH = (qctier == 0) ? NmolA : query_shape_width;
    // 0 means "caller did not narrow it": fall back to the combined width,
    // which is what shipped and is always safe.
    const int SWIDTH = (library_shape_width > 0) ? library_shape_width : (int)NmolB;
    const int nthreads = COMBO_NTHREADS;
    dim3 blockDim(nthreads);
    dim3 gridDim(n_work_items);

    // ---- start-per-lane, and the four things the host has to get right -------
    // `S` groups of WA lanes share one library row and split the starts. The
    // host's share of that is: (a) validate it, (b) size the staged buffer for
    // COMBO_NTHREADS/(WA*S) rows rather than COMBO_NTHREADS/WA, (c) dispatch,
    // and (d) -- in interface.cpp, not here -- build `wi_tcount` to the same
    // width. (d) is the one that fails silently: a `chunk` that disagrees with
    // this kernel's row count drops or duplicates library rows without an error.
    const int S = start_groups_per_row;
    if (S < 1 || (S & (S - 1)) != 0 || S > 8)
        throw std::runtime_error(
            "roshambo3.cuda: start_groups_per_row must be a power of two in 1..8, got "
            + std::to_string(S));
    if (S * WA > 32)
        throw std::runtime_error(
            "roshambo3.cuda: start_groups_per_row * lanes_per_group must be <= 32 "
            "so the per-row best stays an in-warp reduction");
    if (S > signed_permutation_start_count(start_mode))
        throw std::runtime_error(
            "roshambo3.cuda: start_groups_per_row=" + std::to_string(S) +
            " exceeds the " + std::to_string(signed_permutation_start_count(start_mode)) +
            " starts of start_mode=" + std::to_string(start_mode) +
            "; the surplus groups would own a row and no work");
    // The REGISTER/S refusal is lifted for the explicit measurement
    // grid. S groups still duplicate APL coordinates, which is a reason to
    // expect a loss rather than a reason to skip the production-kernel timing.
    // The lift is mode-3-only: the old shape and AT_POSE surfaces remain
    // deliberately refused, so this experiment cannot silently enlarge or
    // change their established dispatch contract.
    if (tier == 3 && S > 1 && color_mode != 2)
        throw std::runtime_error(
            "roshambo3.cuda: CoordResidency.REGISTER with "
            "start_groups_per_row > 1 is instantiated only for the explicit "
            "IN_GRADIENT combination measurement grid; shape-only and AT_POSE "
            "retain this named refusal because each S group duplicates the "
            "per-lane register coordinate tile");
    // Missing padded-width/WA cells fall through to the named dispatch refusal.
    // The streaming tiers request NO staged shared memory at all.  That is
    // the entire shared commitment -- 12.7 KB at L = 32 and 22.2 KB at L = 56 --
    // and it is what holds blocks/SM at 3 and 2 on the large buckets.
    if (tier < 0 || tier > 4)
        throw std::runtime_error("roshambo3.cuda: tier must be 0..4");
    // ---- what the color surface is instantiated for -------------------------
    // The shape surface is untouched and keeps every cell it had. The color one
    // is deliberately narrower, because each cell is a whole copy of a ~3000
    // instruction kernel:
    // tiers 0-2, UPD 0 and 2, S in {1,2,4}. Every exclusion is refused by name
    // with its reason rather than silently substituted.
    if (color_mode < 0 || color_mode > 2)
        throw std::runtime_error("roshambo3.cuda: color_mode must be 0 (none), "
                                 "1 (AT_POSE) or 2 (IN_GRADIENT)");
    if (color_mode > 0){
        if (upd == 3)
            throw std::runtime_error(
                "roshambo3.cuda: update_lanes=3 (REDUCE_SCATTER) is not "
                "instantiated with color: its reduce-scatter fuses the reduction "
                "with the scatter, so no lane holds the whole overlap of either "
                "channel, and the combination objective needs both before the "
                "gradients can be mixed. Use update_lanes 0 or 2.");
        if (S > 4)
            throw std::runtime_error(
                "roshambo3.cuda: start_groups_per_row > 4 is not instantiated "
                "with color; production runs 4 starts, where S in {1,2,4} all "
                "divide evenly.");
        if (N_features < 1)
            throw std::runtime_error("roshambo3.cuda: a color launch needs an "
                                     "interaction matrix");
    }
    if (minb < 1 || minb > 4)
        throw std::runtime_error("roshambo3.cuda: grid_minb must be 1..4");
    // The per-block color table sits between the query and the staged
    // record, and is not requested at all by a shape-only launch -- which is
    // what keeps every measured COLOR = 0 shared-memory number valid.
    const size_t sharedMemSize = query_bytes(QWIDTH)
        + ((color_mode > 0) ? (color_table_bytes(N_features)
                               + color_grid_bytes(N_features)) : 0)
        + ((tier == 0) ? stage_bytes(SWIDTH, nthreads, WA*S) : 0)
        + ((color_mode == 2 && ctier == 0 && color_stage > 0)
           ? cstage_bytes(color_stage, nthreads, WA*S) : 0);

    cudaSetDevice(device_id);
    int max_shared = 0;
    CUDA_CHECK_ERROR(cudaDeviceGetAttribute(&max_shared,
                     cudaDevAttrMaxSharedMemoryPerBlockOptin, device_id));
    if (sharedMemSize > (size_t)max_shared){
        throw std::runtime_error(
            "roshambo3.cuda: staging " + std::to_string(NmolB) +
            " library atoms needs " + std::to_string(sharedMemSize/1024) +
            " KB of shared memory, but device " + std::to_string(device_id) +
            " allows at most " + std::to_string(max_shared/1024) + " KB.");
    }
    if (carveout < 0 || carveout > 100)
        throw std::runtime_error("roshambo3.cuda: carveout must be 0..100");

    // cudaFuncSetAttribute is STICKY, so "do not set it" is not a control
    // and an unset carveout reads whatever the previous kernel left behind. It
    // is set here on every launch, unconditionally, and the effective value is
    // read back so a harness can record what was actually granted.
    // OPT is threaded through every macro rather than dispatched at the
    // bottom, so that each (upd, tier, apl, minb) cell keeps its exact meaning
    // and only ONE more template argument varies -- the same arrangement the
    // analytic launcher uses, and for the same reason.
    // One master macro with every template argument, and every
    // chain below forwards to it. The classic chains pass the defaults, so
    // their instantiations are the ones that already existed.
    // `TB` is the library tile's template value -- ROSHAMBO3_BTILE for the
    // interleaved layout, -ROSHAMBO3_BTILE for the planar one. Every chain
    // below forwards the interleaved value through R3FL*; no planar cell is
    // instantiated.
    //
    // The guard is the same shape as the (QCT, BAL) one directly below it: a
    // cell whose layout is not the requested one BREAKS rather than launching,
    // so `launched_` stays false and the ladder's terminal throw fires. That is
    // what keeps a planar request a refusal by name instead of a silent run at
    // the interleaved layout -- an inert layout and a slow one look identical
    // from outside.
    #define R3FL13TM(TB, WAV, UPD, TIER, APL, MINB, OPT, SG, CL, CT, ILVV, CST, QCT, BAL, META)     \
        do {                                                                                 \
            if (planar_ != ((TB) < 0)) break;                                                \
            if (qctier != (QCT) || ballot != (BAL)) break;                                   \
            const void * fn_ = (const void*)                                                 \
                optimize_pairs_field_kernel<WAV, UPD, TB, COMBO_NTHREADS,                    \
                              TIER, APL, MINB, OPT, SG, CL, CT, ILVV, CST, QCT, BAL, EltB, META>;   \
            CUDA_CHECK_ERROR(cudaFuncSetAttribute(fn_,                                       \
                cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sharedMemSize));           \
            CUDA_CHECK_ERROR(cudaFuncSetAttribute(fn_,                                       \
                cudaFuncAttributePreferredSharedMemoryCarveout, carveout));                  \
            int nblk = 0;                                                                    \
            CUDA_CHECK_ERROR(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nblk,            \
                fn_, nthreads, sharedMemSize));                                              \
            if (nblk == 0) throw std::runtime_error(                                         \
                "roshambo3.cuda: the configuration cannot launch (0 blocks/SM)");            \
            r3_record_launch(nblk, (unsigned long long)sharedMemSize,                     \
                             carveout, nthreads);                                        \
            optimize_pairs_field_kernel<WAV, UPD, TB, COMBO_NTHREADS,                        \
                          TIER, APL, MINB, OPT, SG, CL, CT, ILVV, CST, QCT, BAL, EltB, META>        \
                <<<gridDim, blockDim, sharedMemSize>>>(                                      \
                molAs, molA_num_atoms, NmolA,                                                \
                molBs, molB_counts, NmolB, library_color_offset,                             \
                out_library_row, wi_qrow, wi_tstart, wi_tcount,                              \
                grids_dev, scores, lr_q, lr_t, nsteps, start_mode, va_all, vb_all,           \
                molA_types, rmat, pmat, N_features,                                          \
                mixing_param, legacy_combo_gradient, color_grids, QWIDTH, SWIDTH, \
                winning_starts, start_objectives);          \
            launched_ = true;                                                                \
        } while(0)
    #define R3FL13T(TB, WAV, UPD, TIER, APL, MINB, OPT, SG, CL, CT, ILVV, CST, QCT, BAL) \
        R3FL13TM(TB, WAV, UPD, TIER, APL, MINB, OPT, SG, CL, CT, ILVV, CST, QCT, BAL, false)
    // The interleaved tile.
    #define R3FL13(WAV, UPD, TIER, APL, MINB, OPT, SG, CL, CT, ILVV, CST, QCT, BAL)           \
        R3FL13T(ROSHAMBO3_BTILE, WAV, UPD, TIER, APL, MINB, OPT, SG, CL, CT, ILVV, CST, QCT, BAL)
    #define R3FL(WAV, UPD, TIER, APL, MINB, OPT, SG, CL, CT, ILVV, CST)                       \
        R3FL13(WAV, UPD, TIER, APL, MINB, OPT, SG, CL, CT, ILVV, CST, 0, false)

    bool launched_ = false;
    const bool defaults_ = (WA == 4 && MINB_S == 4 && ctier == 1 && qctier == 0
                            && color_stage_ == 0 && ILV == 0);

    // Native search diagnostics use separate compile-time specializations.
    // The ordinary dispatch never instantiates metadata instructions. Limit
    // diagnostic surface to existing default cells rather than duplicating the
    // tuning grid; unsupported requests must fail instead of changing settings.
    if (winning_starts || start_objectives) {
        const bool common = !planar_ && WA == 4 && tier == 0 && upd == 2
            && opt_mode == UPDATE_ADAGRAD_EXP && qctier == 0 && !ballot && ILV == 0;
        const bool global_color = ctier == 1 && color_stage_ == 0;
        if (common && global_color && MINB_S == 4 && S == 1 && color_mode == 0)
            R3FL13TM(ROSHAMBO3_BTILE, 4, 2, 0, 0, 4, UpdateRule::AdagradExp, 1, 0, 1, 0, 0, 0, false, true);
        if (common && global_color && MINB_S == 4 && S == 1 && color_mode == 1)
            R3FL13TM(ROSHAMBO3_BTILE, 4, 2, 0, 0, 4, UpdateRule::AdagradExp, 1, 1, 1, 0, 0, 0, false, true);
        if (common && global_color && MINB_S == 2 && S == 1 && color_mode == 2)
            R3FL13TM(ROSHAMBO3_BTILE, 4, 2, 0, 0, 2, UpdateRule::AdagradExp, 1, 2, 1, 0, 0, 0, false, true);
        if (common && global_color && MINB_S == 2 && S == 4 && color_mode == 2)
            R3FL13TM(ROSHAMBO3_BTILE, 4, 2, 0, 0, 2, UpdateRule::AdagradExp, 4, 2, 1, 0, 0, 0, false, true);
        if (common && ctier == 0 && color_stage_ == 48 && MINB_S == 2 && S == 1 && color_mode == 2)
            R3FL13TM(ROSHAMBO3_BTILE, 4, 2, 0, 0, 2, UpdateRule::AdagradExp, 1, 2, 0, 0, 48, 0, false, true);
        if (common && ctier == 0 && color_stage_ == 48 && MINB_S == 2 && S == 4 && color_mode == 2)
            R3FL13TM(ROSHAMBO3_BTILE, 4, 2, 0, 0, 2, UpdateRule::AdagradExp, 4, 2, 0, 0, 48, 0, false, true);
        if (!launched_) throw std::invalid_argument(
            "CUDA FIELD start metadata requires a supported default dispatch cell");
        return;
    }

    // ---- the dispatch ladder ------------------------------------------------
    // The cells a shipped mode reaches, one explicit line each. Every chain is
    // first-match because its conditions are mutually exclusive. Anything not
    // instantiated falls through to the throw at the end, which names every
    // setting of the request.

    // The classic shape chain: shape tiers 0,1,2,4 x S x UPD x rule.
    if (defaults_ && color_mode == 0 && tier != 3){
        // ---- the cells a shipped mode reaches ----
        if (upd == 2 && tier == 0 && S == 1 && opt_mode == 0) R3FL(4, 2, 0, 0, 4, UpdateRule::AdagradQAdd, 1, 0, 1, 0, 0);
        if (upd == 2 && tier == 0 && S == 1 && opt_mode == 1) R3FL(4, 2, 0, 0, 4, UpdateRule::AdagradExp, 1, 0, 1, 0, 0);
    }


    // The classic color chain: shape tiers 0,1,2,4 x S x UPD x rule.
    if (defaults_ && color_mode > 0 && tier != 3){
        // ---- the cells a shipped mode reaches ----
        if (upd == 2 && tier == 0 && S == 1 && opt_mode == 0 && color_mode == 1) R3FL(4, 2, 0, 0, 4, UpdateRule::AdagradQAdd, 1, 1, 1, 0, 0);
        if (upd == 2 && tier == 0 && S == 1 && opt_mode == 1 && color_mode == 1) R3FL(4, 2, 0, 0, 4, UpdateRule::AdagradExp, 1, 1, 1, 0, 0);
    }

    // ---- the shipped COMBINATION cells (mode 3, at the locked point) --------
    // `coord_residency = SHARED`, `color_residency = SHARED`, `color_stage = 48`
    // and `smem_carveout = 25` are the locked operating point;
    // `lanes_per_group = 4`, `stream_min_blocks_per_sm = 2` and
    // `query_color_residency = SHARED` are the schedule's, unchanged by the lock. What
    // is NOT fixed by either is the pair (`start_groups_per_row`, update rule):
    // `score_pairs` resolves S from the START MODE -- min(4, starts), so 1 at
    // start_mode 0 and 4 at start_mode 1 and 2 -- and both ADAGRAD_QADD and
    // ADAGRAD_EXP reach the field kernel. So the shipped combination is FOUR
    // cells, not one. They are here rather than in the extended ladder below
    // because that ladder pins `opt_mode == UPDATE_ADAGRAD_EXP`. CSTAGE 48 is
    // the smallest staged width statistically tied with the best at every
    // bucket measured.
    if (!launched_ && color_mode == 2 && tier == 0 && WA == 4 && MINB_S == 2
        && ctier == 0 && ILV == 0 && color_stage_ == 48 && qctier == 0
        && upd == 2){
        if (S == 1 && opt_mode == 0) R3FL(4, 2, 0, 0, 2, UpdateRule::AdagradQAdd, 1, 2, 0, 0, 48);
        if (S == 1 && opt_mode == 1) R3FL(4, 2, 0, 0, 2, UpdateRule::AdagradExp, 1, 2, 0, 0, 48);
        if (S == 4 && opt_mode == 0) R3FL(4, 2, 0, 0, 2, UpdateRule::AdagradQAdd, 4, 2, 0, 0, 48);
        if (S == 4 && opt_mode == 1) R3FL(4, 2, 0, 0, 2, UpdateRule::AdagradExp, 4, 2, 0, 0, 48);
    }

    // The EXTENDED ladder: one knob at a time off the shipped schedule. Its
    // conditions are full conjunctions over the same axes and pairwise
    // mutually exclusive, which is what lets it be sorted by group.
    if (!launched_ && !defaults_ && tier != 3
        && upd == 2 && opt_mode == UPDATE_ADAGRAD_EXP){
        // The `stream_min_blocks_per_sm` axis is NOT a tuning knob: the schedule ships
        // MINB_S = 2 for the combined field-and-color path. These two are the
        // cells `backend.score_pairs` reaches at ITS OWN defaults, which
        // are the pre-lock `color_residency = GLOBAL, color_stage = 0`
        // rather than ScreenConfig's, so both surfaces' mode-3 cells are
        // compiled. Guarding them failed four tests on the first pruned build.
        // ---- the cells a shipped mode reaches ----
        if (color_mode == 2 && tier == 0 && S == 1 && WA == 4 && MINB_S == 2 && ctier == 1 && ILV == 0 && color_stage_ == 0) R3FL(4, 2, 0, 0, 2, UpdateRule::AdagradExp, 1, 2, 1, 0, 0);
        if (color_mode == 2 && tier == 0 && S == 4 && WA == 4 && MINB_S == 2 && ctier == 1 && ILV == 0 && color_stage_ == 0) R3FL(4, 2, 0, 0, 2, UpdateRule::AdagradExp, 4, 2, 1, 0, 0);
    }

    if (!launched_){
        throw std::runtime_error(
            "roshambo3.cuda: this field-kernel configuration is not instantiated "
            "(lanes_per_group=" + std::to_string(WA) +
            ", start_groups_per_row=" + std::to_string(S) +
            ", coord_residency=" + std::to_string(tier) +
            ", color_residency=" + std::to_string(ctier) +
            ", color_stage=" + std::to_string(color_stage_) +
            ", min_blocks_per_sm=" + std::to_string(minb) +
            ", stream_min_blocks_per_sm=" + std::to_string(MINB_S) +
            ", update_lanes=" + std::to_string(upd) +
            ", update_rule=" + std::to_string(opt_mode) +
            ", color_mode=" + std::to_string(color_mode) +
            ", interleave=" + std::to_string(ILV) +
            ", query_color_residency=" + std::to_string(qctier) +
            ", color_ballot=" + std::to_string((int)ballot) +
            // Without this a planar request that has no cell is refused
            // with a message naming every knob EXCEPT the one that differs,
            // which reads as the interleaved cell being missing too.
            ", library_tile_mode=" + std::to_string(tile_mode) +
            "). This build instantiates only the cells a shipped configuration "
            "selects; return the setting that differs from the shipped "
            "schedule to its default. No planar library-tile cell "
            "(library_tile_mode < 0) is instantiated.");
    }
    #undef R3FL
    #undef R3FL13
    #undef R3FL13T
    #undef R3FL13TM


    #undef ROSHAMBO3_FIELD_LAUNCH_S
    #undef ROSHAMBO3_FIELD_LAUNCH_RA
    #undef ROSHAMBO3_FIELD_LAUNCH_R
    #undef ROSHAMBO3_FIELD_LAUNCH_T
    #undef ROSHAMBO3_FIELD_LAUNCH_U
    cudaSetDevice(device_id); CUDA_CHECK_ERROR(cudaGetLastError());
    cudaSetDevice(device_id); CUDA_CHECK_ERROR(cudaDeviceSynchronize());
}


////////////////////////////////////////////////////////////////////////////////
/// The element types each host entry point is compiled for.
///
/// `short` is the int16 wire record -- the corpus path, and where every byte
/// is. `float` is the four-component record, kept because a caller whose
/// coordinates are NOT on the corpus's wire-step grid cannot be handed the int16
/// one without that being a numerics change rather than an encoding: the
/// committed golden fixture (`bench/make_golden.py`, a seeded random walk), the
/// ROSHAMBO2 port gate and the hand-built test fixtures are all off-grid. So
/// the element type is a parameter of the SAME code rather than two code paths,
/// which is the whole point of the templating.
///
/// Written out rather than left implicit so that a signature drift between
/// `kernels.cuh` and this file is a compile error here, at the one place that
/// names both.
////////////////////////////////////////////////////////////////////////////////
template void compute_self_overlaps_gpu<float>(
        const float *, const r3_counts_t *, int, long, int,
        const float *, const float *, int, bool,
        float, float *, int, int, int);
template void compute_self_overlaps_gpu<short>(
        const short *, const r3_counts_t *, int, long, int,
        const float *, const float *, int, bool,
        float, float *, int, int, int);

template void optimize_pairs_analytic<float>(
        const float *, const int *, const int *, int,
        const float *, const r3_counts_t *, int,
        int,
        const int *,
        const int *, const int *, const int *, int,
        const float *, const float *, int,
        float *, bool, float, float, int, float,
        int, int, bool,
        const float *, const float *,
        int, bool, int, bool, bool, float,
        bool, bool, int, int, int, int *, float *);
template void optimize_pairs_analytic<short>(
        const float *, const int *, const int *, int,
        const short *, const r3_counts_t *, int,
        int,
        const int *,
        const int *, const int *, const int *, int,
        const float *, const float *, int,
        float *, bool, float, float, int, float,
        int, int, bool,
        const float *, const float *,
        int, bool, int, bool, bool, float,
        bool, bool, int, int, int, int *, float *);

template void optimize_pairs_field<float>(
        const float *, const int *, int,
        const float *, const r3_counts_t *, int,
        int,
        const int *,
        const int *, const int *, const int *, int,
        const QueryField *,
        float *, float, float, int, int,
        const float *, const float *,
        int, int, int, int, int, int,
        int,
        const int *,
        const float *, const float *, int,
        float, bool, int,
        const QueryField *,
        int, int, int,
        int, int,
        int, int,
        int, int, int, int *, float *);
template void optimize_pairs_field<short>(
        const float *, const int *, int,
        const short *, const r3_counts_t *, int,
        int,
        const int *,
        const int *, const int *, const int *, int,
        const QueryField *,
        float *, float, float, int, int,
        const float *, const float *,
        int, int, int, int, int, int,
        int,
        const int *,
        const float *, const float *, int,
        float, bool, int,
        const QueryField *,
        int, int, int,
        int, int,
        int, int,
        int, int, int, int *, float *);
