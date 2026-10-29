"""The clock/power sampler moved from a background thread to a
subprocess, and every ``nvidia-smi`` call in this module is now per-device.

The thread sampler collected one sample on a real two-minute benchmark. This suite
does not need a GPU to show the replacement subprocess sampler collects a
real distribution: a fake ``nvidia-smi`` script on ``PATH`` stands in for
the real one, deterministically, in every test below. No test in this file
needs the substring ``cuda`` -- none of it touches real hardware.
"""

from __future__ import annotations

import os
import stat
import subprocess
import sys
import time

import pytest

from bench import clocks
from bench.gpu_lock import DeviceIdentity

DEV0 = DeviceIdentity(visible_index=0, physical_index=0, uuid="GPU-fake0")
DEV1 = DeviceIdentity(visible_index=0, physical_index=1, uuid="GPU-fake1")

_FAKE_SMI = """#!/usr/bin/env python3
import sys, time

args = sys.argv[1:]
# -pm / -lgc / -lmc / -rgc / -rmc: just report success.
if any(a in ("-pm", "-lgc", "-lmc", "-rgc", "-rmc") for a in args):
    print("ok")
    raise SystemExit(0)

idx = args[args.index("-i") + 1] if "-i" in args else "?"
query = next((a for a in args if a.startswith("--query-gpu=")), "")

if query == "--query-gpu=clocks.max.sm":
    print("1912")
    raise SystemExit(0)
if query == "--query-gpu=clocks.max.mem":
    print("850")
    raise SystemExit(0)

# The 1 Hz sampling row: clocks.sm, clocks.mem, power.draw, temperature.gpu,
# utilization.gpu, clocks_throttle_reasons.active,
# clocks_event_reasons.sw_power_cap -- 7 fields, matching bench.clocks._FIELDS.
t = time.time()
sm = 1400 + int(t) % 5
print(f"{sm},850,150.5,55,30,0x0000000000000000,Not Active")
"""


@pytest.fixture()
def fake_nvidia_smi(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    script = bin_dir / "nvidia-smi"
    script.write_text(_FAKE_SMI)
    script.chmod(script.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    return script


class TestDeviceScopedClockCalls:
    """Every clock-affecting/-reading call takes -i <physical_index>."""

    def _capture(self, monkeypatch):
        calls = []
        real_run = subprocess.run

        def spy(args, **kw):
            calls.append(args)
            return real_run(args, **kw)

        monkeypatch.setattr(clocks.subprocess, "run", spy)
        return calls

    def test_enable_persistence_mode_is_scoped(self, fake_nvidia_smi, monkeypatch):
        calls = self._capture(monkeypatch)
        assert clocks.enable_persistence_mode(DEV1) is True
        assert any("-i" in c and c[c.index("-i") + 1] == "1" for c in calls)

    def test_lock_clocks_is_scoped(self, fake_nvidia_smi, monkeypatch):
        calls = self._capture(monkeypatch)
        clocks.lock_clocks(gpu_clock_mhz=1400, mem_clock_mhz=850, device=DEV1)
        lgc_calls = [c for c in calls if "-lgc" in c]
        assert lgc_calls and lgc_calls[0][lgc_calls[0].index("-i") + 1] == "1"

    def test_max_gpu_clock_mhz_is_scoped(self, fake_nvidia_smi, monkeypatch):
        calls = self._capture(monkeypatch)
        assert clocks.max_gpu_clock_mhz(DEV1) == 1912
        assert any(c[c.index("-i") + 1] == "1" for c in calls if "-i" in c)

    def test_default_device_is_used_when_none_given(self, fake_nvidia_smi, monkeypatch):
        """Single-GPU-host backward compatibility: every older call site
        passed no device at all."""
        monkeypatch.setattr("bench.gpu_lock._list_physical_gpus",
                            lambda: [(0, "GPU-onlyone")])
        assert clocks.max_gpu_clock_mhz() == 1912


class TestClockPowerSamplerSubprocess:
    """The sampler is a real OS subprocess, not a background thread."""

    def test_collects_multiple_samples_over_a_real_interval(self, fake_nvidia_smi):
        with clocks.ClockPowerSampler(DEV0, interval_s=0.2) as sampler:
            time.sleep(1.2)
        summary = sampler.summary()
        # The sample count is reported explicitly.
        assert summary["n_samples"] >= 3
        assert summary["device"]["uuid"] == "GPU-fake0"
        assert 1400 <= summary["achieved_clock_mhz"]["median"] <= 1410

    def test_pins_the_poller_to_the_requested_device(self, tmp_path, monkeypatch):
        """The poller must be invoked with -i <physical_index>, matching
        this device, not device 0 regardless of which card is sampled."""
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        log = tmp_path / "seen_index.txt"
        script = bin_dir / "nvidia-smi"
        script.write_text(
            "#!/usr/bin/env python3\n"
            "import sys\n"
            f"open({str(log)!r}, 'a').write(sys.argv[sys.argv.index('-i')+1] + chr(10))\n"
            "if any(a.startswith('--query-gpu=clocks.sm') for a in sys.argv):\n"
            "    print('1400,850,150.0,55,10,0x0,Not Active')\n"
        )
        script.chmod(script.stat().st_mode | stat.S_IEXEC)
        monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")

        with clocks.ClockPowerSampler(DEV1, interval_s=0.2):
            time.sleep(0.7)

        seen = log.read_text().splitlines() if log.exists() else []
        assert seen, "the fake nvidia-smi was never invoked"
        assert all(idx == "1" for idx in seen), seen

    def test_zero_samples_is_reported_not_hidden(self):
        """A sampler that never got a chance to sample must say n_samples=0
        explicitly, not return an empty/absent summary a caller could read
        as 'not measured' rather than 'measured, saw nothing'."""
        sampler = clocks.ClockPowerSampler(DEV0, interval_s=1.0)
        assert sampler.summary() == {"n_samples": 0, "device": DEV0.as_dict()}


class TestClockSampleParsing:
    """_load()'s parser, exercised directly against a synthetic poller
    output file -- the format bench.gpu_poller writes."""

    def test_parses_a_synthetic_poller_file(self, tmp_path):
        sampler = clocks.ClockPowerSampler(DEV0, interval_s=1.0)
        sampler._t0 = 1000.0
        p = tmp_path / "poll.csv"
        p.write_text(
            "1000.0,1400,850,150.0,55,10,0x0000000000000000,Not Active\n"
            "1001.0,1410,850,151.0,56,12,0x0000000000000000,Active\n"
        )
        sampler._load(str(p))
        assert len(sampler.samples) == 2
        assert sampler.samples[0].t_rel_s == 0.0
        assert sampler.samples[1].t_rel_s == 1.0
        assert sampler.samples[1].sw_power_cap_active is True
        assert sampler.samples[0].sw_power_cap_active is False

    def test_skips_malformed_lines_without_raising(self, tmp_path):
        sampler = clocks.ClockPowerSampler(DEV0, interval_s=1.0)
        sampler._t0 = 0.0
        p = tmp_path / "poll.csv"
        p.write_text("garbage,not,enough,fields\n"
                     "1.0,1400,850,150.0,55,10,0x0,Not Active\n")
        sampler._load(str(p))
        assert len(sampler.samples) == 1

    def test_energy_and_throttle_fractions(self, tmp_path):
        sampler = clocks.ClockPowerSampler(DEV0, interval_s=1.0)
        sampler._t0 = 0.0
        p = tmp_path / "poll.csv"
        p.write_text(
            "0.0,1400,850,100.0,50,10,0x0000000000000000,Not Active\n"
            "1.0,1400,850,200.0,50,10,0x0000000000000004,Active\n"
        )
        sampler._load(str(p))
        assert sampler.energy_joules() == pytest.approx(150.0)
        assert sampler.max_throttle_fraction() == 0.5
        assert sampler.sw_power_cap_fraction() == 0.5
