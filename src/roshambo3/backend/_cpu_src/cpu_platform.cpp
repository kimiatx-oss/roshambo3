// OS-specific implementations of the platform layer; see cpu_platform.hpp for
// why these calls are kept out of the numerical kernels.
#include "cpu_platform.hpp"
#include <stdexcept>

#if defined(__linux__)
#include <sched.h>
#if defined(__aarch64__)
#include <sys/auxv.h>
#include <asm/hwcap.h>
#endif
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#include <pthread.h>
#include <pthread/qos.h>
#endif

namespace r3::cpu::platform {
bool affinity_supported() noexcept {
#if defined(__linux__)
    return true;
#else
    return false;
#endif
}

// The CPUs this process may run on (the inherited affinity mask), in
// ascending id order. Empty where affinity is unsupported.
std::vector<int> allowed_cpus() {
    std::vector<int> cpus;
#if defined(__linux__)
    cpu_set_t mask;
    CPU_ZERO(&mask);
    if (sched_getaffinity(0, sizeof(mask), &mask) == 0)
        for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu)
            if (CPU_ISSET(cpu, &mask)) cpus.push_back(cpu);
#endif
    return cpus;
}

// Pin the calling thread to one CPU. Returns false, never throws, on an out of
// range id, a refused syscall, or a platform without affinity.
bool bind_current_thread(int cpu) noexcept {
#if defined(__linux__)
    if (cpu < 0 || cpu >= CPU_SETSIZE) return false;
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(cpu, &mask);
    return sched_setaffinity(0, sizeof(mask), &mask) == 0;
#else
    (void)cpu;
    return false;
#endif
}

namespace {
#if defined(__APPLE__)
// One `hw.optional.*` sysctl flag. A missing key or a failed query reads as
// "absent", which disables the optional instructions.
bool feature(const char* name) noexcept {
    int value = 0;
    size_t size = sizeof(value);
    return sysctlbyname(name, &value, &size, nullptr, 0) == 0 && value == 1;
}

// Policy name -> qos_class_t. Wider than the public `cpu_qos` set
// (validate_qos), so every name current_qos() reports maps back.
qos_class_t qos_value(const std::string& policy) {
    if (policy == "background") return QOS_CLASS_BACKGROUND;
    if (policy == "user_initiated") return QOS_CLASS_USER_INITIATED;
    if (policy == "default") return QOS_CLASS_DEFAULT;
    if (policy == "utility") return QOS_CLASS_UTILITY;
    if (policy == "user_interactive") return QOS_CLASS_USER_INTERACTIVE;
    if (policy == "unspecified") return QOS_CLASS_UNSPECIFIED;
    throw std::invalid_argument("unknown QoS policy");
}
#endif
}

// FP16 vector arithmetic (FEAT_FP16, Linux HWCAP_ASIMDHP). Probed once and
// cached: callers reach this at setup, never per texel. macOS accepts either
// the `hw.optional.arm.FEAT_FP16` or the `hw.optional.neon_fp16` key.
bool fp16_supported() noexcept {
    static const bool supported = [] {
#if defined(__linux__) && defined(__aarch64__) && defined(HWCAP_ASIMDHP)
        return (getauxval(AT_HWCAP) & HWCAP_ASIMDHP) != 0;
#elif defined(__APPLE__) && defined(__aarch64__)
        return feature("hw.optional.arm.FEAT_FP16") ||
               feature("hw.optional.neon_fp16");
#else
        return false;
#endif
    }();
    return supported;
}

// FP16 widening multiply-accumulate FMLAL/FMLSL (FEAT_FHM, Linux
// HWCAP_ASIMDFHM), used by field_neon_fp16fml.cpp. Cached like the FP16 probe.
bool fp16_fml_supported() noexcept {
    static const bool supported = [] {
#if defined(__linux__) && defined(__aarch64__) && defined(HWCAP_ASIMDFHM)
        return (getauxval(AT_HWCAP) & HWCAP_ASIMDFHM) != 0;
#elif defined(__APPLE__) && defined(__aarch64__)
        return feature("hw.optional.arm.FEAT_FHM");
#else
        return false;
#endif
    }();
    return supported;
}

// The public `cpu_qos` contract, mirrored by roshambo3._cpu_platform: only
// inherit / background / user_initiated, and only `inherit` off macOS.
void validate_qos(const std::string& policy) {
    if (policy != "inherit" && policy != "background" &&
        policy != "user_initiated")
        throw std::invalid_argument(
            "cpu_qos must be inherit, background, or user_initiated");
#if !defined(__APPLE__)
    if (policy != "inherit")
        throw std::invalid_argument(
            "explicit cpu_qos is only supported on macOS");
#endif
}

std::string current_qos() {
#if defined(__APPLE__)
    qos_class_t qos;
    int priority;
    if (pthread_get_qos_class_np(pthread_self(), &qos, &priority) != 0)
        throw std::runtime_error("could not read thread QoS");
    switch (qos) {
    case QOS_CLASS_BACKGROUND:
        return "background";
    case QOS_CLASS_USER_INITIATED:
        return "user_initiated";
    case QOS_CLASS_DEFAULT:
        return "default";
    case QOS_CLASS_UTILITY:
        return "utility";
    case QOS_CLASS_USER_INTERACTIVE:
        return "user_interactive";
    default:
        return "unspecified";
    }
#else
    return "unsupported";
#endif
}

// Apply a QoS class to the calling thread; `inherit` is a no-op on every
// platform so the default path makes no OS call.
void set_qos(const std::string& policy) {
    if (policy == "inherit") return;
#if defined(__APPLE__)
    if (pthread_set_qos_class_self_np(qos_value(policy), 0) != 0)
        throw std::runtime_error("could not set thread QoS to " + policy);
#else
    throw std::invalid_argument("explicit cpu_qos is only supported on macOS");
#endif
}

QosSnapshot qos_snapshot() {
#if defined(__APPLE__)
    qos_class_t qos;
    int priority;
    if (pthread_get_qos_class_np(pthread_self(), &qos, &priority) != 0)
        throw std::runtime_error("could not snapshot thread QoS");
    return {static_cast<unsigned int>(qos), priority};
#else
    return {0, 0};
#endif
}

// Reinstate a snapshot, including its relative priority. Callers must not
// pass an UNSPECIFIED snapshot (see QosSnapshot).
void restore_qos(QosSnapshot snapshot) {
#if defined(__APPLE__)
    if (pthread_set_qos_class_self_np(static_cast<qos_class_t>(snapshot.qos),
                                      snapshot.priority) != 0)
        throw std::runtime_error("could not restore submitting thread QoS");
#else
    (void)snapshot;
#endif
}
}
