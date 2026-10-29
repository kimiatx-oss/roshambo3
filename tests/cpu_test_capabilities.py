"""Native test dispatch choices; unsupported experimental arms stay explicit."""
import platform

IS_ARM = platform.machine().lower() in ("aarch64", "arm64")
RETAINED_COLOR_GRAIN = 16 if IS_ARM else 11
