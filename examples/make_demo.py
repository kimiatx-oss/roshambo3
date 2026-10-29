"""Generate a tiny, public synthetic library without external corpora."""
import argparse
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    molecules = []
    for name, smiles in [("ethanol", "CCO"), ("propanol", "CCCO"), ("benzene", "c1ccccc1")]:
        molecule = Chem.AddHs(Chem.MolFromSmiles(smiles))
        if AllChem.EmbedMolecule(molecule, randomSeed=0xC0FFEE) != 0:
            raise RuntimeError(f"could not embed {name}")
        molecule.SetProp("_Name", name)
        molecules.append(molecule)
    for name, values in [("library.sdf", molecules), ("query.sdf", molecules[:1])]:
        with Chem.SDWriter(str(args.out / name)) as writer:
            for molecule in values:
                writer.write(molecule)

if __name__ == "__main__":
    main()
