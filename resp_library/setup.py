import os
import pandas

from rdkit import Chem
from rdkit.Chem import AllChem


def main():

    df = pandas.read_csv("molecules.csv")

    for resp_type in ["RESP1", "RESP2"]:
        dirname = resp_type
        if not os.path.exists(dirname):
            os.mkdir(dirname)
        for index, row in df.iterrows():
            dirname = f"{resp_type}/" + row["iupac_name"]
            if not os.path.exists(dirname):
                os.mkdir(dirname)
            with open(dirname + "/resp.yaml", "w") as f:
                f.write(build_yaml(row, resp_type))

            # Save an initial structure to define atom symmetry
            smiles = row["smiles"]
            rdmol = Chem.MolFromSmiles(smiles)
            rdmol = Chem.AddHs(rdmol)
            cid = AllChem.EmbedMolecule(rdmol, randomSeed=1)
            Chem.rdmolfiles.MolToPDBFile(rdmol, dirname + "/template.pdb")


def build_yaml(row, resp_type):
    yaml_template = f"""name:   {row["iupac_name"]}
smiles: {row["smiles"]}

conformer_generation:
  n_conformers: 5
  rms_threshold: 0.5
  energy_threshold: 10  # kJ/mol
  random_seed: 1

resp:
  type: {resp_type}
  point_density: 3.0
  vdw_scale_factors:
    - 1.6
    - 1.8
    - 2.0
  charge_constraints:
  equality_constraints:
"""
    return yaml_template


if __name__ == "__main__":
    main()

