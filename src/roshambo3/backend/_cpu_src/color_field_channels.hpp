// Four-channel CPU color fields with GPU-matched stored-gradient semantics:
// the channel-SIMD implementation.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace r3 {
namespace cpu {

enum ColorFieldBackendCode : int {
    COLOR_FIELD_ANALYTIC = 0,
    COLOR_FIELD_NODE_FP32 = 1,
    COLOR_FIELD_NODE_FP16 = 2,
    // fp32 polynomial coefficients packed from the half-rounded AoS4
    // nodes; storage precision and source-node quantization are distinct.
    COLOR_FIELD_BRICK_FP32 = 3,
};

struct ColorFieldSource {
    float xyz[3]{};
    int type = 0;
};

struct ColorFieldSample {
    double value = 0.0;
    double spatial_gradient[3]{};
    bool used_field = false;
    int backend_code = COLOR_FIELD_ANALYTIC;
};

struct ColorFieldPreparedSample {
    size_t base[3]{};
    float fraction[3]{};
};

// Worker-owned scratch, reused across poses and groups. Output channels retain
// their reduced-frame gradient units until the caller's compatibility boundary.
struct ColorFieldSampleWorkspace {
    std::vector<float> points;
    std::vector<ColorFieldPreparedSample> prepared;
    std::vector<std::array<float, 4>> channels;
};

// One query-scoped collection of per-library-type fields. backend_codes is
// indexed by target type and contains COLOR_FIELD_* above; type zero is
// ignored. Matrix arrays are row-major [query_type, target_type], with
// coefficient = (pi/2)^1.5 * p^2 * r^-1.5 and decay = -r/2. The bundle copies
// sources and matrix values, so its lifetime is independent of the caller's
// query-view buffers.
class ColorFieldBundle {
public:
    ColorFieldBundle(const std::vector<ColorFieldSource>& sources,
                     int n_features, const double* coefficient,
                     const double* decay, const int* backend_codes,
                     double spacing_angstrom, double padding_angstrom);

    ColorFieldSample sample_target_type(int target_type,
                                        const float* xyz_angstrom) const;
    // Fast-path eligibility check. Returns false for an inactive plane or a
    // point outside its fixed query grid without evaluating the analytic
    // fallback; the conformer kernel can then route that whole plane through
    // its AVX2 analytic loop.
    bool can_sample_target_type(int target_type,
                                const float* xyz_angstrom) const;
    // Atomic group eligibility: false means the caller must evaluate the whole
    // original group analytically. No partial output may be consumed on false.
    bool sample_group(int target_type, const float* points, size_t count,
                      size_t point_stride, size_t coordinate_stride,
                      ColorFieldSampleWorkspace& workspace,
                      bool coordinates_reduced = false) const;

    size_t field_bytes() const;
    size_t field_bytes(int target_type) const;

    double build_seconds() const {
        return build_seconds_;
    }

    double build_seconds(int target_type) const;
    double pack_seconds(int target_type) const;
    std::vector<size_t> field_bytes_by_target_type() const;
    std::vector<double> build_seconds_by_target_type() const;
    std::vector<double> pack_seconds_by_target_type() const;
    int backend_code(int target_type) const;
    std::vector<int> backend_codes() const;
    bool field_active(int target_type) const;
    std::vector<int> field_active_by_target_type() const;
    std::array<int, 3> grid_cells(int target_type) const;
    std::array<double, 3> grid_origin_angstrom(int target_type) const;
    double grid_spacing_angstrom(int target_type) const;
    std::vector<int> target_types() const;

    bool avx2_fma_supported() const {
        return avx2_fma_supported_;
    }

    bool f16c_supported() const {
        return f16c_supported_;
    }

    const char* half_decode_path() const;

    uint64_t generation() const {
        return generation_;
    }

    static uint64_t construction_count();

    double preparation_seconds() const {
        return preparation_seconds_;
    }

    size_t peak_payload_bytes() const {
        return peak_payload_bytes_;
    }

    static int coefficient_layout_version();
    std::vector<int> pack_counts() const;
    std::vector<int> inactive_reasons() const;
    std::array<size_t, 3> payload_alignment(int target_type) const;
    const char* sampling_path(int target_type) const;
    static const char* node32_source_quantization();

private:
    struct Geometry {
        float origin[3]{};
        float h = 1.0f;
        float inv_h = 1.0f;
        int cells[3]{1, 1, 1};
    };

    struct Plane {
        Geometry geometry;
        int backend = COLOR_FIELD_ANALYTIC;
        bool active = false;
        std::vector<float> node_fp32;
        std::vector<uint16_t> node_fp16;
        std::vector<float> brick_fp32;
        double build_s = 0.0;
        double pack_s = 0.0;
        int pack_count = 0;
        int inactive_reason = 0;
        size_t bytes() const;
    };

    struct ValueGradient {
        double value = 0.0;
        double grad[3]{};
    };

    size_t cell_index(int query_type, int target_type) const;
    bool in_domain(const Geometry& geometry, const float* xyz) const;
    ValueGradient exact_at(const float* xyz, int target_type) const;
    ValueGradient interpolate(const Plane& plane, const float* xyz) const;
    void sample_prepared_avx2(const Plane&, ColorFieldSampleWorkspace&) const;
    void sample_prepared_f16c(const Plane&, ColorFieldSampleWorkspace&) const;

    int n_features_ = 0;
    std::vector<ColorFieldSource> sources_;
    std::vector<double> coefficient_;
    std::vector<double> decay_;
    std::vector<Plane> planes_;
    double build_seconds_ = 0.0;
    bool avx2_fma_supported_ = false;
    bool f16c_supported_ = false;
    uint64_t generation_ = 0;
    double preparation_seconds_ = 0.0;
    size_t peak_payload_bytes_ = 0;
};

}
}  // namespace r3::cpu
