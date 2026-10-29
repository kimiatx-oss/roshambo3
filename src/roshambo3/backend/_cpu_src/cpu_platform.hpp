// The CPU backend's OS-services layer: worker affinity, ARM capability
// probes, and macOS thread QoS.
//
// See doc/APPLE_SILICON.md. Before the Apple
// Silicon port these services were Linux-only and lived inside the code that
// used them: `sched_{get,set}affinity` in the combo worker pool and
// `getauxval(AT_HWCAP)` FP16/FHM probes in field_neon.cpp. Darwin has neither
// `cpu_set_t` nor HWCAP, so the port gathered these services here, behind one
// platform-neutral interface, rather than scattering `#ifdef __APPLE__` through
// the kernels.
//
// The isolation is also a numerics boundary. The port's goal was a macOS build
// with no change to kernels, numerics or defaults (Linux score/pose output was
// checked bit-identical over 16,498,560 evaluations), so no OS
// probe or scheduling call may sit in an atom, texel or optimizer loop. The
// existing NEON FPCR read at scoring entry and the rounding barriers are not
// here and were not moved.
//
// This header deliberately includes no Python headers (pybind11 is only
// forward-declared), so native code can use the layer without pulling in the
// binding machinery; the bindings live in cpu_platform_bindings.cpp.
#pragma once
#include <string>
#include <vector>

namespace pybind11 {
class module_;
}

// OS services live here, never in numerical kernels. All probes and worker
// policy changes run at setup boundaries, not inside an atom/texel loop.
namespace r3::cpu::platform {
// Affinity: real on Linux; on macOS `affinity_supported()` is false,
// `allowed_cpus()` is empty and `bind_current_thread()` returns false. Apple
// affinity tags are cache-sharing hints, not core masks (doc/APPLE_SILICON.md),
// and this layer does not use them. Strict `require_worker_affinity=True` is
// refused on macOS.
bool affinity_supported() noexcept;
std::vector<int> allowed_cpus();
bool bind_current_thread(int cpu) noexcept;
// Runtime capability probes gating the separately compiled FP16 and FP16FML
// NEON translation units. A failed query reports false, which disables the
// optional instructions rather than failing the scan.
bool fp16_supported() noexcept;
bool fp16_fml_supported() noexcept;
// macOS pthread QoS. A scheduling hint (background favors efficiency,
// user_initiated favors responsiveness), NOT P-core/E-core pinning; read-back
// establishes the requested class, not where a thread ran. Explicit policies
// are refused off macOS.
std::string current_qos();
void set_qos(const std::string& policy);

// The caller's QoS, captured so a scoped policy can be undone on success and
// failure. `qos` is the raw `qos_class_t`; 0 is QOS_CLASS_UNSPECIFIED, which
// pthread cannot restore, so the Python scope (_cpu_platform.qos_scope)
// refuses to mutate such a thread.
struct QosSnapshot {
    unsigned int qos;
    int priority;
};

QosSnapshot qos_snapshot();
void restore_qos(QosSnapshot snapshot);
void validate_qos(const std::string& policy);
void bind_api(pybind11::module_& module);
}
