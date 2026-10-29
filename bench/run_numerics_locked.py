"""Run numerical checks under physical GPU locks, before context creation.

Pass --lock-receipt NEW_JSON and optionally --devices 0,1, followed by ordinary
run_numerics_checks arguments. Device ordinals use CUDA_VISIBLE_DEVICES.
"""
import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path


def device_indices(value):
    try:
        indices = tuple(int(part) for part in value.split(','))
    except ValueError as error:
        raise argparse.ArgumentTypeError('devices must be comma-separated integers') from error
    if not indices or any(index < 0 for index in indices) or len(set(indices)) != len(indices):
        raise argparse.ArgumentTypeError('devices must be distinct nonnegative ordinals')
    if 0 not in indices:
        raise argparse.ArgumentTypeError('devices must include ordinal 0 used by the numerical family')
    return indices


def run_locked(indices, run, *, allowed_pids=(), allocation_reason=None):
    """Acquire every requested physical lock and preflight before importing CUDA."""
    allowed_pids = {str(int(pid)) for pid in allowed_pids}
    if allowed_pids and (not allocation_reason or not allocation_reason.strip()):
        raise ValueError('an explicit reason is required for allowed existing allocations')
    if any(int(pid) <= 0 for pid in allowed_pids):
        raise ValueError('allowed existing PIDs must be positive')
    indices = device_indices(','.join(map(str, indices)))
    from bench.gpu_lock import GpuLock, resolve_device, query_gpu_state
    devices = [resolve_device(index) for index in indices]
    if len({device.uuid for device in devices}) != len(devices):
        raise ValueError('requested ordinals resolve to duplicate physical GPUs')
    with ExitStack() as stack:
        locks = {device.uuid: stack.enter_context(GpuLock('numerics-qualification', device))
                 for device in sorted(devices, key=lambda device: device.uuid)}
        states = [query_gpu_state(device) for device in devices]
        for device, state in zip(devices, states):
            unapproved = [app for app in state.compute_apps if str(app.pid) not in allowed_pids]
            if unapproved or state.utilization_gpu_pct > 0:
                raise RuntimeError('GPU busy before numerical family: ' + device.uuid + ' ' + str(state))
        # Explicit opt-in prevents default single-device checks from accessing an
        # unlocked second GPU merely because the host happens to expose it.
        variable = 'ROSHAMBO3_NUMERICS_DEVICE_IDS'
        previous = os.environ.get(variable)
        os.environ[variable] = ','.join(map(str, indices))
        try:
            code = run()
        finally:
            if previous is None:
                os.environ.pop(variable, None)
            else:
                os.environ[variable] = previous
        return code, [dict(visible_index=device.visible_index, device_uuid=device.uuid,
                           lock_held=locks[device.uuid].acquired, preflight=state.as_dict(),
                           allowed_existing_pids=sorted(allowed_pids),
                           allocation_exception_reason=allocation_reason,
                           correctness_only=bool(allowed_pids))
                      for device, state in zip(devices, states)]


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument('--lock-receipt', type=Path, required=True)
    parser.add_argument('--devices', type=device_indices, default=(0,))
    parser.add_argument('--allow-existing-pid', action='append', type=int, default=[])
    parser.add_argument('--existing-allocation-reason')
    args, remaining = parser.parse_known_args()
    if args.lock_receipt.exists():
        parser.error('lock receipt must be fresh')
    from bench.numerics_report import source_stamp
    from bench.run_numerics_checks import main as run
    code, devices = run_locked(args.devices, lambda: run(remaining),
        allowed_pids=args.allow_existing_pid, allocation_reason=args.existing_allocation_reason)
    args.lock_receipt.parent.mkdir(parents=True, exist_ok=True)
    args.lock_receipt.write_text(json.dumps(dict(
        lock_held=all(device['lock_held'] for device in devices),
        device_uuid=devices[0]['device_uuid'], preflight=devices[0]['preflight'],
        devices=devices, exitcode=int(code), source=source_stamp(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()), indent=2) + '\n')
    return code


if __name__ == '__main__':
    raise SystemExit(main())
