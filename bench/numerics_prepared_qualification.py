"""Report source-bound transfer to prepared geometry; nonzero if unqualified.

This command is deliberately separate from execution checks. It does not claim
an independent search of the prepared geometry and never labels a failed
conservative transfer as an observed optimizer loss.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--native-build', type=Path)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    sys.meta_path[:] = [finder for finder in sys.meta_path
                       if 'ScikitBuild' not in type(finder).__name__]
    sys.path[:0] = [str(root / 'src'), str(root)]
    import roshambo3
    if Path(roshambo3.__file__).resolve() != root / 'src/roshambo3/__init__.py':
        raise RuntimeError('roshambo3 was imported from another checkout')
    if args.native_build:
        from bench.run_numerics_checks import select_native_build
        select_native_build(roshambo3, args.native_build)
        from bench.numerics_backends import verified_build
        verified_build('cpu')
    from bench.numerics_end_to_end import prepare_panel, qualify_scored_representation
    from roshambo3.numerics import WIRE_QUANT_STEP
    with tempfile.TemporaryDirectory(prefix='prepared-qualification-') as directory:
        panel = prepare_panel(Path(directory) / 'corpus')
        cases = []
        for mode in ('shape', 'combo'):
            for qi, name in enumerate(panel.fixture_names):
                case = qualify_scored_representation(panel, qi, mode, None)
                cases.append(dict(name=name, mode=mode, **case))
    result = dict(protocol='source-to-prepared-basin-transfer-v1',
                  wire_step=float(WIRE_QUANT_STEP), cases=cases,
                  independent_prepared_geometry_search='not_run',
                  production_qualified=all(c['production_qualified'] for c in cases))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print('prepared fixture qualification:', result['production_qualified'], args.output)
    return int(not result['production_qualified'])


if __name__ == '__main__':
    raise SystemExit(main())
