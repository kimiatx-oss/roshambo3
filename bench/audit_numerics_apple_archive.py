"""Read-only inventory of historical Apple archive provenance, never qualification."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    digest = hashlib.sha256()
    with args.archive.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    with tarfile.open(args.archive, 'r:gz') as archive:
        members = archive.getmembers()
        envs = []
        for member in members:
            if not member.name.endswith('/env.json') or not member.isfile():
                continue
            raw = archive.extractfile(member).read()
            value = json.loads(raw)
            install = value.get('install', {})
            # Record only scientific provenance; omit machine personal identifiers.
            profiler = json.loads(value.get('system_profiler') or '{}')
            hardware = profiler.get('SPHardwareDataType', [{}])[0]
            envs.append(dict(member=member.name, member_sha256=hashlib.sha256(raw).hexdigest(),
                checkout_head=install.get('checkout_head'), checkout_status=install.get('checkout_status'),
                editable=install.get('editable'), platform=value.get('platform'),
                chip=hardware.get('chip_type'), tool_stamps=value.get('tool_stamps')))
        numerics_members = [m.name for m in members if any(
            token in m.name.lower() for token in ('t270', 'numerics', 'fault', 'sampler'))]
    args.output.write_text(json.dumps(dict(archive=str(args.archive.resolve()),
        archive_sha256=digest.hexdigest(), members=len(members), environments=envs,
        potential_numerics_members=numerics_members, certified=False,
        conclusion='Historical Apple benchmark evidence only. No numerics test execution or fault qualification inferred.'),
        indent=2) + '\n')


if __name__ == '__main__':
    main()
