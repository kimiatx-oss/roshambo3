"""Query preparation and atom-order-preserving viewer payloads."""
from pathlib import Path
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Draw

from roshambo3.prep.conformer import prep_conformer, strip_hydrogens


def molblock_at(mol, xyz, *, preserve_atoms=False):
    # Format-3 hit SMILES and pose coordinates share the corpus atom order.
    # Preserve that order while assigning stereochemistry from the pose.
    heavy = Chem.Mol(mol) if preserve_atoms else strip_hydrogens(mol)
    if heavy.GetNumAtoms() != len(xyz):
        raise ValueError("Structure atom order/count is incompatible with the stored pose")
    heavy.RemoveAllConformers()
    conf = Chem.Conformer(len(xyz))
    conf.Set3D(True)
    for i, point in enumerate(xyz):
        conf.SetAtomPosition(i, tuple(float(x) for x in point))
    heavy.AddConformer(conf)
    if preserve_atoms:
        heavy = strip_hydrogens(heavy)
        Chem.AssignStereochemistryFrom3D(heavy, replaceExistingTags=True)
    return Chem.MolToMolBlock(heavy)


def payload(mol):
    heavy = strip_hydrogens(mol)
    xyz = prep_conformer(heavy.GetConformer().GetPositions())
    return molblock_at(heavy, xyz)


def thumbnail(mol):
    copy = Chem.Mol(mol)
    copy.RemoveAllConformers()
    return Draw.MolsToGridImage([copy], molsPerRow=1, subImgSize=(160, 100), useSVG=True)


def prepare_query(directory, request, max_conformers):
    directory = Path(directory)
    warnings = []
    if request.get("smiles"):
        mol = Chem.MolFromSmiles(request["smiles"])
        if mol is None:
            raise ValueError("Cannot parse this SMILES")
        mols = [mol]
    else:
        path = directory / ("input" + request["suffix"])
        if request["suffix"] == ".mol":
            mols = [Chem.MolFromMolFile(str(path), removeHs=False)]
        else:
            mols = list(Chem.SDMolSupplier(str(path), removeHs=False))
        if not mols or any(m is None for m in mols):
            raise ValueError("The uploaded file contains an unreadable molecule record")
    keys = {Chem.MolToSmiles(strip_hydrogens(m)) for m in mols}
    if len(keys) != 1:
        raise ValueError("Upload one molecule, optionally with several conformers; mixed molecules are not supported")
    if any(strip_hydrogens(m).GetNumAtoms() == 0 for m in mols):
        raise ValueError("The query must contain heavy atoms")
    count = int(request.get("conformers", 10))
    seed = 0xC0FFEE
    if request.get("generate") or request.get("smiles"):
        mol = Chem.AddHs(Chem.Mol(mols[0]))
        mol.RemoveAllConformers()
        params = AllChem.ETKDGv3()
        params.randomSeed = seed
        params.numThreads = 2
        params.pruneRmsThresh = 0.5
        ids = list(AllChem.EmbedMultipleConfs(mol, count, params))
        if not ids:
            raise ValueError("Conformer generation failed; supply a 3D SDF instead")
        if request.get("minimize"):
            if not AllChem.MMFFHasAllMoleculeParams(mol):
                raise ValueError("MMFF94s parameters are unavailable for this molecule; turn off minimization")
            status = AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=2, mmffVariant="MMFF94s")
            if any(s[0] for s in status):
                warnings.append("Some conformers did not converge during minimization")
        mols = []
        for cid in ids:
            copy = Chem.Mol(mol)
            conf = Chem.Conformer(mol.GetConformer(cid))
            copy.RemoveAllConformers()
            copy.AddConformer(conf, assignId=True)
            mols.append(copy)
        if len(mols) < count:
            warnings.append(f"Generated {len(mols)} distinct conformers of {count} requested")
    if len(mols) > max_conformers:
        raise ValueError(f"At most {max_conformers} query conformers are allowed")
    for i, mol in enumerate(mols):
        if mol.GetNumConformers() == 0 or not mol.GetConformer().Is3D():
            raise ValueError("The query has no 3D coordinates; enable conformer generation")
        if not np.isfinite(mol.GetConformer().GetPositions()).all():
            raise ValueError("The query contains non-finite coordinates")
        mol.SetProp("_Name", f"query_{i + 1}")
    with Chem.SDWriter(str(directory / "query.sdf")) as writer:
        for mol in mols:
            writer.write(mol)
    from rdkit import rdBase
    return {"conformers": len(mols), "warnings": warnings, "seed": seed,
            "rdkit_version": rdBase.rdkitVersion, "method": "ETKDGv3" if request.get("generate") or request.get("smiles") else "uploaded",
            "minimize": bool(request.get("minimize")), "molblock": payload(mols[0]),
            "thumbnail": thumbnail(mols[0])}
