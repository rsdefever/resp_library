import psi4
import requests
from rdkit import Chem

from .exceptions import SMILESConversionError

HARTREE_TO_KJMOL = 2625.4996382852165050  # from psi4
BOHR_TO_ANGSTROM = 0.52917721067  # from psi4


def canonicalize_smiles(smiles):
    """Convert smiles to the canonical form

    Parameters
    ----------
    smiles: str
        the smiles string to canonicalize

    Returns
    -------
    canonical_smiles
        The canonical form of the smiles string
    """
    rdmol = Chem.MolFromSmiles(smiles)
    if rdmol is None:
        raise SMILESConversionError(f"Invalid SMILES string: '{smiles}'")
    return Chem.MolToSmiles(rdmol)


def smiles_to_iupac(smiles):
    """Convert smiles to the IUPAC name


    All spaces are replaced with underscores.

    Parameters
    ----------
    smiles: str
        the smiles string to canonicalize

    Returns
    -------
    iupac_name: str
        the iupac name
    """
    url = f"https://cactus.nci.nih.gov/chemical/structure/{smiles}/iupac_name"
    response = requests.get(url)
    try:
        response.raise_for_status()
    except requests.HTTPError:
        raise SMILESConversionError(
            f"Unable to convert smiles to IUPAC name. Please check the "
            f"SMILES string: {smiles} and the status of "
            f"https://cactus.nci.nih.gov/chemical/structure"
        )
    iupac_name = response.text
    return iupac_name.replace(" ", "_")


def yaml_template(smiles, iupac_name, resp_type):
    """Generate a template YAML file"""

    template = f"""name:   {iupac_name}
smiles: {smiles}

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
    return template


def build_p4mol(elements, coords, net_charge):
    """Create a psi4 molecule from a list of elements and coordinates"""

    mol_string = f"{net_charge} 1\n"
    for element, (x, y, z) in zip(elements, coords):
        mol_string += f"{element}   {x}   {y}   {z}\n"

    mol = psi4.geometry(mol_string)
    mol.update_geometry()

    return mol
