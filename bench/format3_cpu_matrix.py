"""Untimed builder-produced format-3 block/worker identity matrix.

Run from a clean committed checkout; place synthetic corpora outside the repo.
Existing corpora are opened read-only and never rebuilt or overwritten. Timing
campaigns are deliberately separate so this harness cannot consume a quiet
measurement window by accident.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import psutil

from roshambo3.backend import cpu
from roshambo3.engine import Engine, R3gSource, ScreenConfig, from_arrays
from roshambo3.engine.config import ColorPlacement
from roshambo3.format.catalog import Catalog
from roshambo3.format.r3g.descriptor import generator_from_descriptor
from roshambo3.format.r3g.reader import open_corpus, verify
from roshambo3.pharmacophore import PharmacophoreGenerator
from roshambo3.metrics import canonical_ids
from roshambo3.prep import prepare


def synthetic_molecules(n_molecules=33, conformers=128):
    """A small real RDKit input: 4,224 rows exercise all requested block sizes."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    template = Chem.AddHs(Chem.MolFromSmiles('CO'))
    if AllChem.EmbedMolecule(template, randomSeed=19) != 0:
        raise RuntimeError('could not embed synthetic methanol')
    base = template.GetConformer().GetPositions()
    rng = np.random.default_rng(295)
    molecules = []
    for index in range(n_molecules):
        molecule = Chem.Mol(template)
        molecule.RemoveAllConformers()
        molecule.SetProp('_Name', f'methanol-{index}')
        for _ in range(conformers):
            xyz = base + rng.normal(0, .08, base.shape)
            conf = Chem.Conformer(molecule.GetNumAtoms())
            for atom, point in enumerate(xyz):
                conf.SetAtomPosition(atom, point.tolist())
            molecule.AddConformer(conf, assignId=True)
        molecules.append(molecule)
    return molecules


def build_synthetic(root, block_sizes=(1024, 2048, 4096)):
    root = Path(root)
    paths = {rows: root / f'block-{rows}' for rows in block_sizes}
    if any(path.exists() for path in paths.values()):
        raise ValueError('synthetic output already exists; select a new root or use --corpus')
    dirty = subprocess.check_output(['git', 'status', '--porcelain'], text=True)
    if dirty.strip():
        raise RuntimeError('build synthetic evidence from a clean committed checkout')
    molecules = synthetic_molecules()
    generator = PharmacophoreGenerator()
    for rows, path in paths.items():
        prepare(molecules, str(path), workers=2, block_rows=rows,
                color_generator=generator)
        build = json.loads((path / 'build.json').read_text())
        descriptor = json.loads((path / 'corpus.json').read_text())
        commit = descriptor['prep']['commit']
        if commit is None or build.get('commit') != commit:
            raise RuntimeError('synthetic builder prep.commit is null or disagrees with build.json')
        print(f'built block_rows={rows} path={path}', flush=True)
    return paths


def _query(source):
    d = min(source.cpu_block_descriptors(), key=lambda d: d.gid_base)
    decoded = source.decode_cpu_block(d)
    coords, counts = decoded.result.coords[0], decoded.result.counts[0]
    nr = int(counts[0])
    kwargs = {}
    if source.color:
        types = np.repeat(np.arange(1, counts.size, dtype=np.int32), counts[1:])
        kwargs.update(color=[(coords[nr:nr+types.size], types)],
                      color_generator=source.color_generator)
    return from_arrays([coords[:nr]], ['matrix-query'], **kwargs)


def run_matrix(paths, *, workers=None, k=15):
    worker_counts = sorted(set(workers or (1, 8, psutil.cpu_count(logical=False) or 1)))
    report = dict(kind='untimed_correctness', physical_cores=psutil.cpu_count(logical=False),
                  logical_cores=psutil.cpu_count(logical=True), cells=[])
    expected = {}
    for block_rows, path in sorted(paths.items()):
        path = str(path)
        verify(path)
        corpus = open_corpus(path)
        catalog = Catalog(corpus)
        build_file = Path(path) / 'build.json'
        build = json.loads(build_file.read_text()) if build_file.exists() else None
        for color in (False, True):
            mode = 'combo' if color else 'shape'
            generator = generator_from_descriptor(corpus.color) if color else None
            source = R3gSource(corpus, tile=False, color=color, color_generator=generator)
            query = _query(source)
            config = ScreenConfig(color=color, mixing=.5 if color else 0., steps=25,
                                  color_placement=ColorPlacement.IN_GRADIENT,
                                  field_spacing_a=.5)
            # The shipping CPU worker policy: retained fields for both modes,
            # and canonical grain-16 optimizer color for combo.
            policy = dict(shape_backend='pipelined_field', shape_grain='library_atom',
                shape_width=8, color_grain=16 if color else 0,
                factor_canonical_torque=color,
                optimizer_exp_mode='rawbits_unclamped' if color else 'baseline',
                canonical_color_kernel=color, final_start_policy='winner')
            try:
                for count in worker_counts:
                    engine = Engine(backend='cpu')
                    try:
                        stats, rows = engine.run_worker_owned_cpu(source, query, config,
                            workers=count, k=k, level='molecule', self_overlap_policy='prepared', **policy)
                    finally:
                        engine.close()
                    scores, gids, mids = rows[0]
                    refs = catalog.locate(gids)
                    identity = np.asarray([(ref.mol_id, ref.conf_idx) for ref in refs], np.int64)
                    canonical = canonical_ids(identity[:, 0], identity[:, 1])
                    if mode not in expected:
                        expected[mode] = (scores.copy(), identity.copy(), canonical.copy())
                    # Raw pairs are checked first; independently dense keys alone
                    # would mask a substituted identity with the same relative order.
                    np.testing.assert_array_equal(identity, expected[mode][1])
                    np.testing.assert_array_equal(canonical, expected[mode][2])
                    np.testing.assert_array_equal(scores.view(np.uint32),
                                                  expected[mode][0].view(np.uint32))
                    scan = stats.cpu_scan
                    assert scan['stored_vaa_verified_blocks'] == stats.n_blocks
                    assert scan['library_shape_self_evaluations'] == 0
                    assert scan['library_color_self_evaluations'] == 0
                    assert scan['color_self_sidecar_blocks'] == (stats.n_blocks if color else 0)
                    cell = dict(block_rows=int(block_rows), workers=count, mode=mode,
                        path=path, rows=stats.n_conformers, blocks=stats.n_blocks, policy=policy,
                        identity=identity.tolist(), canonical_ids=canonical.tolist(), scores_sha256=hashlib.sha256(scores.tobytes()).hexdigest(),
                        identity_equal=True, score_columns_bit_equal=True,
                        stored_vaa_verified_blocks=scan['stored_vaa_verified_blocks'],
                        color_self_section_blocks=scan['color_self_sidecar_blocks'],
                        library_shape_self_evaluations=scan['library_shape_self_evaluations'],
                        library_color_self_evaluations=scan['library_color_self_evaluations'],
                        prep_commit=corpus.descriptor['prep']['commit'],
                        build_commit=build.get('commit') if build else None)
                    report['cells'].append(cell)
                    print(f'PASS block_rows={block_rows} workers={count} mode={mode}', flush=True)
            finally:
                source.close()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--synthetic-root', type=Path)
    source.add_argument('--corpus', action='append', metavar='ROWS=PATH')
    parser.add_argument('--workers', default='1,8,physical')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    physical = psutil.cpu_count(logical=False) or 1
    workers = [physical if s == 'physical' else int(s) for s in args.workers.split(',')]
    if any(n < 1 for n in workers):
        parser.error('worker counts must be positive')
    paths = (build_synthetic(args.synthetic_root) if args.synthetic_root else
             {int(s.split('=', 1)[0]): Path(s.split('=', 1)[1]) for s in args.corpus})
    report = run_matrix(paths, workers=workers)
    report['source_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    report['cpu_build'] = cpu.build_stamp()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=str) + '\n')


if __name__ == '__main__':
    main()
