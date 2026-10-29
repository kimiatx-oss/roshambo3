// Python bindings for the platform layer, kept out of cpu_platform.cpp so the
// native layer stays free of Python headers. The underscore-prefixed entries
// are private hooks for roshambo3._cpu_platform.qos_scope, which snapshots,
// sets and restores the submitting thread's QoS.
#include "cpu_platform.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace r3::cpu::platform {
void bind_api(pybind11::module_& module) {
    module.def("cpu_platform_info", [] {
        pybind11::dict out;
        out["affinity_supported"] = affinity_supported();
        out["allowed_cpus"] = allowed_cpus();
        out["fp16"] = fp16_supported();
        // Same probe as "fp16"; FEAT_FP16 is the arithmetic extension. One
        // capability answers both questions -- stricter than conversion-only
        // storage needs, but never wrong; see field_neon.cpp.
        out["fp16_arithmetic"] = fp16_supported();
        out["fp16_fml"] = fp16_fml_supported();
        out["qos"] = current_qos();
        return out;
    });
    module.def("_current_qos", &current_qos);
    module.def("_set_qos", &set_qos);
    module.def("_qos_snapshot", [] {
        const auto snapshot = qos_snapshot();
        return pybind11::make_tuple(snapshot.qos, snapshot.priority);
    });
    module.def("_restore_qos", [](unsigned int qos, int priority) {
        restore_qos({qos, priority});
    });
}
}
