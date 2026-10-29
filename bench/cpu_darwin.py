"""macOS benchmark telemetry. Unknown measurements remain explicit."""
import ctypes
import ctypes.util
import subprocess


def sysctl(name, *, string=False):
    libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
    size = ctypes.c_size_t()
    key = name.encode()
    if libc.sysctlbyname(key, None, ctypes.byref(size), None, 0):
        return None
    buf = ctypes.create_string_buffer(size.value)
    if libc.sysctlbyname(key, buf, ctypes.byref(size), None, 0):
        return None
    if string:
        return buf.value.decode()
    return int.from_bytes(buf.raw[:size.value], "little")


def identity():
    return {
        "model": sysctl("machdep.cpu.brand_string", string=True) or "unknown",
        "physical_cores": sysctl("hw.physicalcpu") or 0,
        "logical_cpus": sysctl("hw.logicalcpu") or 0,
        "performance_cores": sysctl("hw.perflevel0.physicalcpu"),
        "efficiency_cores": sysctl("hw.perflevel1.physicalcpu"),
        "l1d_bytes": sysctl("hw.l1dcachesize") or 0,
        "l2_bytes": sysctl("hw.l2cachesize") or 0,
        "isa_flags": [flag for flag, key in (
            ("asimd", "hw.optional.neon"),
            ("asimdhp", "hw.optional.arm.FEAT_FP16"),
            ("asimdfhm", "hw.optional.arm.FEAT_FHM")) if sysctl(key) == 1],
    }


def power_state():
    def read(command):
        try:
            out = subprocess.run(command, capture_output=True, text=True,
                                 timeout=10)
            return out.stdout.strip() if out.returncode == 0 else None
        except (OSError, subprocess.TimeoutExpired):
            return None
    return {"power_settings": read(["pmset", "-g", "custom"]),
            "thermal_state": read(["pmset", "-g", "therm"]),
            "core_residency": None,
            "core_residency_reason": "requires a separate Instruments scheduling trace"}
