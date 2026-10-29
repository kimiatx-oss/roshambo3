These four small factual molecular records come from the public LIT-PCBA
crystal ligand template archive. `manifest.json` identifies the upstream URL,
archive and member checksums, PDB identifiers, exact extraction command,
feature and hydrogen policy, and the source's copyright notice. The landing
page did not identify an explicit data redistribution license; this limitation
is recorded rather than interpreted as a license grant.

`protocol.json` was committed before the multistart searches. Selection uses
archive order and distinct canonical isomeric SMILES: first three qualifying
ligands, then the first qualifying ligand adding the missing family. Source
MOL2 records that RDKit cannot sanitize are excluded. The selected records
cover all six shipped color families, repeated same-type features, atom and
multi-atom centroid features, and asymmetric nonplanar geometry. We retain
explicit source hydrogens for feature typing. Shape uses heavy atoms. Neither
geometry nor protonation is regenerated. SDF conversion preserves parsed
source coordinates exactly. RDKit supplies live chemical feature assignments
and matched atom IDs. Protocol version3 uses deterministic `math.fsum` means
of source atoms for the default uniform-weight feature positions and heavy-atom
center; live RDKit positions must agree within a derived fp64 rounding bound.
Exact canonical content hashes remain mandatory across architectures.

`bench.numerics_fixtures.load_panel()` verifies the SDF hashes and returns
centered float64 coordinates and live features. `objective_function` uses
existing reference overlap formulas. `qualify_fixture` uses independent
SciPy BFGS with 128 seeded Haar rotations and independent translations, for
shape and combo separately. Self alignment has intended score 1 by the
Cauchy-Schwarz bound; no expected kernel scores or winning poses are fixtures.
The independently amended (protocol version 2) 0.011732050807568877 A
atom-index RMSD neighborhood and 0.001 minimum score gap are design
requirements. Uncertainty is a separate derived quantity, not a fitted gap.
Every terminal pose must be stationary; a SciPy precision-loss status alone
does not imply nonconvergence if independent central differences certify it.

Finite search is empirical evidence of basin separation, not a proof that all
basins were discovered. Symmetric and nearly symmetric nonplanar controls
must fail the same criterion. The default qualifier reports fp64 search
qualification only. Portable `qualification.json` and eight compressed endpoint
artifacts bind source SDF bytes, live geometry/chemistry/matrices, feature
factory/version, protocol settings, and endpoint bytes. Consumers recompute
stationarity, completeness, pose exclusion, and gap from endpoint poses. Production qualification requires an explicit end-to-end
allowance for the actual path's coordinate/feature representation and
arithmetic. Absence of that allowance remains visible as unqualified.

Reproduce inputs from a separately downloaded public archive (do not copy the
archive into the repository):

    PYTHONPATH=src:. python -m bench.numerics_fixtures --extract /path/to/full_data.tgz

Run independent qualification and write an evidence receipt:

    PYTHONPATH=src:. python -m bench.numerics_fixtures --workers 4 --portable --output /tmp/alignment-qualification.json

The source archive SHA256 is checked before extraction, then each member hash.
Feature policy records the RDKit version and BaseFeatures.fdef checksum.
RDKit versions can change chemical typing; coverage tests make that visible.

Budget: all files in this directory combined <=1 MiB, each file <=256 KiB.
Bound compressed endpoint artifacts and their qualification index are portable
scientific evidence allowed within this budget. Expected production scores or
production winning poses are not fixtures.
