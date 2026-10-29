"""Format-3 producer: native physical-block decoding for CPU and GPU."""
from __future__ import annotations

import threading
from typing import Optional, Sequence

import numpy as np

from ..format.r3g import native
from .buffers import PinnedPool
from .source import (ConformerBlock, Prefetcher, _Job, _check_groups, _padded_rows,
                     DEFAULT_CONFORMERS_PER_LAUNCH, DEFAULT_PREFETCH_DEPTH)


class R3gSource:
    """Stream a scanned format-3 reader.Corpus in its canonical file order.

    Coordinates, counts and per-row molecule IDs are produced by the native
    launch writer. Cached metadata is per molecule, never per conformer; the
    launch owns its expanded arrays until its consumer releases the block.
    """
    is_format3 = True

    def __init__(self, corpus, *, groups: Optional[Sequence[int]] = None,
                 pool: Optional[PinnedPool] = None, workers: int = 8,
                 prefetch_depth: int = DEFAULT_PREFETCH_DEPTH,
                 conformers_per_launch: int = DEFAULT_CONFORMERS_PER_LAUNCH,
                 tile: bool = True, planar: bool = False, color: bool = False,
                 color_generator=None,
                 trust_definitions=()):
        self.corpus = corpus
        self.tile, self.planar = bool(tile), bool(planar)
        if self.planar and not self.tile:
            raise ValueError("planar=True requires tile=True")
        self.workers, self.prefetch_depth = int(workers), int(prefetch_depth)
        self.conformers_per_launch = int(conformers_per_launch)
        if self.conformers_per_launch <= 0:
            raise ValueError("conformers_per_launch must be positive")
        self.color = bool(color)
        self.color_generator = color_generator
        self.trust_definitions = tuple(trust_definitions)
        self.trusted_definitions = self.trust_definitions
        if self.color:
            if self.color_generator is None:
                from ..pharmacophore import RDKitPharmacophoreGenerator
                self.color_generator = RDKitPharmacophoreGenerator()
            from ..definitions import check_corpus_color
            check_corpus_color(corpus, self.color_generator,
                               trusted_definitions=self.trust_definitions)
        self.group_indices = (list(range(len(corpus.group_table))) if groups is None
                              else _check_groups(groups, len(corpus.group_table)))
        self._pool, self._owns_pool = pool, pool is None
        self._lock = threading.Lock()
        self._sidecars: dict[int, dict] = {}
        self._cpu_descriptors = None
        self._cpu_metadata_owned_bytes = 0
        self.prefetcher = None
        self.jobs = self._make_jobs()

    def _make_jobs(self):
        offsets = np.concatenate(([0], np.cumsum(
            [int(g["n_conf"]) for g in self.corpus.group_table], dtype=np.int64)))
        directory = self.corpus.block_directory("file")
        # Plain lists, not structured-array records: indexing a numpy record
        # field per block cost ~0.17 s per screen over ~125k blocks.
        # The packing below is unchanged.
        group_col = directory["group"]
        rows_col = directory["rows"].tolist()
        first_col = directory["first_row"].tolist()
        block_col = directory["block"].tolist()
        jobs = []
        for gi in self.group_indices:
            current = None
            start, end = (int(x) for x in np.searchsorted(group_col, [gi, gi + 1]))
            for i in range(start, end):
                rows, row0, bi = rows_col[i], first_col[i], block_col[i]
                if current is not None and current.n_conf + rows > self.conformers_per_launch:
                    jobs.append(current)
                    current = None
                if current is None:
                    current = _Job(gi, bi, 0, row0, int(offsets[gi]) + row0, 0)
                # Whole molecule-aligned blocks may exceed the launch target.
                # A short interior tile is filled by the next block natively.
                current.n_blocks += 1
                current.n_conf += rows
            if current is not None:
                jobs.append(current)
        return jobs

    @property
    def n_conformers(self):
        return sum(int(self.corpus.group_table[gi]["n_conf"]) for gi in self.group_indices)

    @property
    def n_launches(self):
        return len(self.jobs)

    @property
    def max_block_conformers(self):
        return max((j.n_conf for j in self.jobs), default=0)

    @property
    def max_block_elements(self):
        return max((_padded_rows(j.n_conf, True) *
                    int(self.corpus.group_table[j.group_index]["L"]) * 3
                    for j in self.jobs), default=0)

    def bucket_widths(self):
        return sorted({int(self.corpus.group_table[gi]["L"]) for gi in self.group_indices})

    def _group_sidecars(self, gi):
        with self._lock:
            if gi not in self._sidecars:
                with self.corpus.group(gi) as group:
                    # Section reads are bounded pread copies. Closing the group
                    # drops its memo, while these immutable arrays own bytes.
                    names = ("runs",)
                    if self.color:
                        names += ("feature_offsets", "feature_types",
                                  "feature_atom_offsets", "feature_atoms")
                    self._sidecars[gi] = {name: group.section(name) for name in names}
            return self._sidecars[gi]

    def run_tables(self) -> dict:
        """Group index -> that group's run table, for every group this source
        has read. A screen's hit lookup (``format.Catalog``) reuses them rather
        than reopening every group."""
        with self._lock:
            return {gi: s["runs"] for gi, s in self._sidecars.items()}

    def _decode(self, job):
        entry = self.corpus.group_table[job.group_index]
        L, L_heavy = int(entry["L"]), int(entry["L_heavy"])
        nr = _padded_rows(job.n_conf, True)
        buf = (self._pool.acquire() if self._pool is not None
               else np.empty(nr * L * 3, dtype=np.int16))
        try:
            sections = self._group_sidecars(job.group_index)
            nt = self.corpus.n_types if self.color else 0
            with self.corpus.group(job.group_index) as group:
                inputs = []
                for bi in range(job.block0, job.block0 + job.n_blocks):
                    rec = group.blocks[bi]
                    m0, m1 = int(rec["m0"]), int(rec["m1"])
                    features = (native.FeatureSlice(
                        sections["feature_offsets"][m0:m1 + 1],
                        sections["feature_types"], sections["feature_atom_offsets"],
                        sections["feature_atoms"]) if nt else None)
                    inputs.append((group.block_bytes(bi), sections["runs"][m0:m1],
                                   int(rec["rows"]), features))
                coords, counts, mol_id = native.write_gpu_launch(
                    inputs, job.n_conf, L_heavy, L, nt, buf,
                    tile_mode=-1 if self.planar else 1)
            if not self.color:
                counts = np.ascontiguousarray(counts[:, :1])
            return ConformerBlock(
                coords=coords, counts=counts, n_conformers=job.n_conf, L=L,
                L_shape=int(entry["L_field_width"]), bucket=int(entry["bucket"]),
                gid_base=job.gid_base, buffer=buf if self._pool is not None else None,
                mol_id=mol_id, tile_mode=-1 if self.planar else 1,
                color_offset=-1, tail_molecule_open=False)
        except BaseException:
            if self._pool is not None:
                self._pool.release(buf)
            raise

    def _decode_cpu(self, job):
        # Both CPU consumers use the same bounded, authenticated native decode.
        # Jobs retain file order even though worker descriptors are expensive-first.
        blocks = [self.decode_cpu_block(self._cpu_descriptor_index[job.group_index, bi])
                  for bi in range(job.block0, job.block0 + job.n_blocks)]
        entry = self.corpus.group_table[job.group_index]
        def joined(values):
            return values[0] if len(values) == 1 else np.concatenate(values)
        return ConformerBlock(
            coords=joined([b.result.coords for b in blocks]),
            counts=joined([b.result.counts for b in blocks]),
            n_conformers=job.n_conf, L=int(entry["L"]),
            L_shape=int(entry["L_field_width"]), bucket=int(entry["bucket"]),
            gid_base=job.gid_base, buffer=None,
            mol_id=joined([b.molecule_ids for b in blocks]), tile_mode=0,
            color_offset=-1, tail_molecule_open=False)

    def blocks(self):
        if not self.tile:
            # Authenticate descriptor sections before concurrent claims begin.
            self.cpu_block_descriptors()
            self.prefetcher = Prefetcher(self.jobs, self._decode_cpu, self.workers,
                                         self.prefetch_depth, on_discard=self._release)
            yield from self.prefetcher
            return
        if self._pool is None and self._owns_pool:
            self._pool = PinnedPool(max(2, self.prefetch_depth + 1), self.max_block_elements)
        self.prefetcher = Prefetcher(self.jobs, self._decode, self.workers,
                                     self.prefetch_depth, on_discard=self._release)
        yield from self.prefetcher

    def _release(self, block):
        if self._pool is not None and block is not None and block.buffer is not None:
            self._pool.release(block.buffer)

    @property
    def producer_wait_s(self):
        return 0.0 if self.prefetcher is None else self.prefetcher.wait_s

    def cpu_block_descriptors(self):
        from .r3g_cpu import build_cpu_block_descriptors
        return build_cpu_block_descriptors(self)

    def decode_cpu_block(self, descriptor):
        from .r3g_cpu import decode_cpu_block
        return decode_cpu_block(self, descriptor)

    def close(self):
        self._sidecars.clear()
        for group in getattr(self, "_cpu_groups", {}).values():
            group.close()
        if hasattr(self, "_cpu_groups"):
            self._cpu_groups.clear()
