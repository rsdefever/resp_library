import psi4
import resp
import ele
import numpy as np
import yaml
import os
import shutil

from rdkit import Chem
from rdkit.Chem import AllChem
from pathlib import Path
from iteround import saferound

from rdkit.Geometry.rdGeometry import Point3D
from scipy.spatial.transform import Rotation

from .utils import canonicalize_smiles
from .utils import smiles_to_iupac
from .utils import yaml_template
from .utils import build_p4mol
from .utils import BOHR_TO_ANGSTROM, HARTREE_TO_KJMOL
from .logger import Logger

from .exceptions import RESPLibraryError

LIB_PATH = Path.joinpath(Path(__file__).parent, "lib/")


def retrieve_charges(smiles):
    """Retrieve the charges for the molecule with the provided smiles string

    Parameters
    ----------
    smiles: str
        the smiles string of the desired molecule

    Returns
    -------
    charges: list
        the charges for the molecule

    Raises
    ------
    ChargesNotFoundError
        if no charges are found
    """


def prepare_charge_calculation(smiles):
    """Setup the directories/templates for the charge calculation for a molecule
    defined by smiles

    Parameters
    ----------
    smiles: str
        the smiles string of the desired molecule

    Returns
    -------


    """
    smiles = canonicalize_smiles(smiles)
    iupac_name = smiles_to_iupac(smiles)
    mol_path = Path.joinpath(LIB_PATH, iupac_name)
    if mol_path.is_dir():
        raise RESPLibraryError(
            f"The directory for the molecule: '{smiles}' with name "
            f"{iupac_name} already exists."
        )

    mol_path.mkdir()
    rdmol = Chem.MolFromSmiles(smiles)
    rdmol = Chem.AddHs(rdmol)
    cid = AllChem.EmbedMolecule(rdmol, randomSeed=1)
    if cid == -1:
        raise RESPLibraryError(
            "RDKIT is unable to generate conformers for this molecule. "
            "Currently you will not be able to use this library to compute "
            "the partial charges. Please raise an issue on our GitHub page."
        )
    template_path = Path.joinpath(mol_path, "template.pdb")
    template_path.write_text(Chem.rdmolfiles.MolToPDBBlock(rdmol))

    for resp_type in ["RESP1", "RESP2"]:
        resp_path = Path.joinpath(mol_path, resp_type)
        resp_path.mkdir()
        yaml_path = Path.joinpath(resp_path, "resp.yaml")
        with yaml_path.open("w") as f:
            f.write(yaml_template(smiles, iupac_name, resp_type))


def calculate_charges(smiles, resp_type):
    """Run the charge calculation for the molecule defined by smiles

    Parameters
    ----------
    smiles: str
        the smiles string of the desired molecule
    resp_type: str
        the type of RESP to perform (RESP1 or RESP2)

    Returns
    -------

    Raises
    ------
    InvalidMoleculeError
        if the smiles string is invalid
    """
    smiles = canonicalize_smiles(smiles)
    iupac_name = smiles_to_iupac(smiles)
    mol_path = Path.joinpath(LIB_PATH, iupac_name)
    if not mol_path.is_dir():
        raise RESPLibraryError(
            f"The directory for the molecule: '{smiles}' with name "
            f"{iupac_name} does not exist. Please run "
            "resp_library.prepare_charge_calculation() first."
        )

    # Initialize RESP stuff
    resp_type = resp_type.upper()
    resp_path = Path.joinpath(mol_path, resp_type)
    inp, log = _initialize_resp(resp_path)

    # Molecule definition
    assert inp['smiles'] == smiles
    assert inp['resp']['type'].upper() == resp_type
    # Parse the YAML file
    n_conformers = inp['conformer_generation']['n_conformers']
    rms_threshold = inp['conformer_generation']['rms_threshold']
    energy_threshold = inp['conformer_generation']['energy_threshold']  # kJ/mol
    conformer_seed = inp['conformer_generation']['random_seed']
    charge_constraints = inp['resp']['charge_constraints']
    equality_constraints = inp['resp']['equality_constraints']
    point_density = inp['resp']['point_density']
    vdw_scale_factors = inp['resp']['vdw_scale_factors']

    if resp_type == "RESP1":
        esp_method = "hf"
        esp_basis_set = "6-31g*"
    elif resp_type == "RESP2":
        esp_method = "pw6b95"
        esp_basis_set = "aug-cc-pV(D+d)Z"
    else:
        raise RESPLibraryError(
            "Invalid RESP type. Only 'RESP1' and 'RESP2' are supported."
        )

    # Create the molecule, add H's
    rdmol = Chem.MolFromSmiles(smiles)
    rdmol = Chem.AddHs(rdmol)

    # Get the net charge
    net_charge = Chem.rdmolops.GetFormalCharge(rdmol)
    log.log(f"The net charge is {net_charge}.")

    # Get the elements
    elements = [a.GetSymbol() for a in rdmol.GetAtoms()]
    vdw_radii = {elem_sym: ele.element_from_symbol(elem_sym).radius_bondi for elem_sym in elements}

    # Generate conformers
    cids = AllChem.EmbedMultipleConfs(
        rdmol,
        numConfs=500,
        pruneRmsThresh=rms_threshold,
        randomSeed=conformer_seed
    )
    AllChem.AlignMolConformers(rdmol)
    if len(cids) < n_conformers:
        raise ValueError(
            "Not enough conformers found. Please reduce the "
            "'rms_threshold' or the 'n_conformers'. For molecules "
            "with < 5 atoms it may be difficult to generate more "
            "than 2 conformers."
        )

    # Select n_conformers at random
    np.random.seed(conformer_seed)
    conformer_ids = np.random.choice([i for i in range(len(cids))], size=n_conformers, replace=False)
    remove = [i for i in range(len(cids)) if i not in conformer_ids]
    for idx in remove:
        rdmol.RemoveConformer(idx)
    # Renumber conformers
    for idx, c in enumerate(rdmol.GetConformers()):
        c.SetId(idx)

    optimized_p4molecules = []
    optimized_energies = []

    # For each conformer, geometry optimize with psi4
    for conformer in rdmol.GetConformers():
        p4mol = build_p4mol(elements, conformer.GetPositions(), net_charge)
        p4mol, energy = _geometry_optimize(p4mol, resp_type)
        # Save optimized structure and energy
        optimized_p4molecules.append(p4mol)
        optimized_energies.append(energy)

        # Extract optimized coordinates; update coordinates RDKIT molecule
        coords = p4mol.geometry().to_array() * BOHR_TO_ANGSTROM
        for i in range(rdmol.GetNumAtoms()):
            x, y, z = coords[i]
            conformer.SetAtomPosition(i, Point3D(x, y, z))

    # Check energies and remove high energy conformers
    _check_relative_conformer_energies(
        rdmol, optimized_p4molecules, optimized_energies, energy_threshold, log
    )

    # Align conformers for easier visual comparison
    _save_aligned_conformers(rdmol, log)

    # Save the conformers used for resp
    Path("structures/optimized_geometries.pdb").write_text(
        Chem.rdmolfiles.MolToPDBBlock(rdmol)
    )
    log.log("Wrote the final optimized gemoetries to 'optimized_geometries.pdb'.\n\n")

    # Finally we do multi-conformer RESP
    pcm = False
    charges = _perform_resp(
        optimized_p4molecules,
        charge_constraints,
        equality_constraints,
        esp_method,
        esp_basis_set,
        pcm,
        point_density,
        vdw_radii,
        vdw_scale_factors,
        log,
    )
    _write_results(
        elements, charges, equality_constraints, "vacuum", log
    )

    if resp_type == "RESP2":
        pcm = True
        charges = _perform_resp(
            optimized_p4molecules,
            charge_constraints,
            equality_constraints,
            esp_method,
            esp_basis_set,
            pcm,
            point_density,
            vdw_radii,
            vdw_scale_factors,
            log,
        )
        _write_results(
            elements, charges, equality_constraints, "pcm", log
        )

    log.close()


def _geometry_optimize(molecule, resp_type):

    # HF/6-31g*
    if resp_type == "RESP1":
        psi4.set_options({"basis": "6-31g*", "geom_maxiter": 500})
        psi4.optimize("hf", molecule=molecule)
        energy = psi4.energy("hf", molecule=molecule)

    # HF/6-31g*, HF/cc-pV(D+d)Z, PW6B95/cc-pV(D+d)Z
    elif resp_type == "RESP2":
        psi4.set_options({"basis": "6-31g*", "geom_maxiter": 500})
        psi4.optimize("hf", molecule=molecule)
        psi4.set_options({"basis": "cc-pV(D+d)Z", "geom_maxiter": 500})
        psi4.optimize("hf", molecule=molecule)
        psi4.set_options({
            "basis": "cc-pV(D+d)Z",
            "geom_maxiter": 200,
            "maxiter": 200,
            "dft_spherical_points": 590,
            "dft_radial_points": 99,
            "dft_pruning_scheme": "robust",
        }
        )
        psi4.optimize("pw6b95", molecule=molecule)
        energy = psi4.energy("pw6b95", molecule=molecule)

    psi4.core.clean()

    return molecule, energy


def _perform_resp(
        molecules,
        charge_constraints,
        equality_constraints,
        method,
        basis_set,
        pcm,
        point_density,
        vdw_radii,
        vdw_scale_factors,
        log,
):
    """Perform RESP for a molecule with n_conformers"""

    resp_options = {
        "VDW_RADII": vdw_radii,
        "VDW_SCALE_FACTORS": vdw_scale_factors,
        "VDW_POINT_DENSITY": point_density,
        "RESP_A": 0.0005,
        "RESP_B": 0.1,
        "METHOD_ESP": method,
        "BASIS_ESP": basis_set,
    }

    if pcm:
        pcm_string = """
        Units = Angstrom
        Medium {
        SolverType = CPCM
        Solvent = Water
        }

        Cavity {
        Type = GePol
        }
        """
        psi4.set_options(
            {
                "pcm": True,
                "pcm_input": pcm_string
            }
        )

    # Call for first stage fit
    charges1 = resp.resp(molecules, resp_options)

    # Save the file
    if pcm:
        log.log("-" * 100)
        log.log("PCM RESP\n")
    else:
        log.log("-" * 100)
        log.log("VACUUM RESP\n")

    log.log("Log from stage 1:\n")
    log.log("-" * 100 + "\n")
    with open("results.out") as f:
        log.log(f.read())
    log.log("-" * 100 + "\n\n")
    os.remove("results.out")

    # Convert to numpy array
    charges1 = np.array(charges1[1])

    # Change the value of the RESP parameter A
    resp_options['RESP_A'] = 0.001

    # Add constraints for second stage fit
    if charge_constraints is not None:
        charge_constraint_list = []
        for constraint_atoms in charge_constraints:
            # Convert to numpy array for indexing...
            constraint_atoms = np.array(constraint_atoms)
            charge_constraint_list.append([np.sum(charges1[constraint_atoms-1]), constraint_atoms])
        resp_options['constraint_charge'] = charge_constraint_list

    if equality_constraints is not None:
        resp_options['constraint_group'] = equality_constraints

    # This is the grid naming...no other choice
    resp_options["GRID"] = [
        f"{i+1}_default_grid.dat" for i in range(len(molecules))
    ]
    resp_options["ESP"] = [
        f"{i+1}_default_grid_esp.dat" for i in range(len(molecules))
    ]

    # Call for second stage fit
    charges2 = resp.resp(molecules, resp_options)

    if pcm:
        fn_suffix = "pcm"
    else:
        fn_suffix = "vacuum"
    for idx, filen in enumerate(resp_options["GRID"]):
        shutil.move(filen, f"esp_grids/{idx}_{fn_suffix}_grid.dat")
    for filen in resp_options["ESP"]:
        shutil.move(filen, f"esp_grids/{idx}_{fn_suffix}_esp.dat")

    # Save the file
    if pcm:
        log.log("-" * 100)
        log.log("PCM RESP\n")
    else:
        log.log("-" * 100)
        log.log("VACUUM RESP\n")

    log.log("Log from stage:\n")
    log.log("-" * 100 + "\n")
    with open("results.out") as f:
        log.log(f.read())
    log.log("-" * 100 + "\n\n")
    os.remove("results.out")

    return np.array(charges2[1])


def _save_aligned_conformers(rdmol, log):

    elements = [a.GetSymbol() for a in rdmol.GetAtoms()]
    ref_coords = rdmol.GetConformer(0).GetPositions()

    xyz_block = ""
    for conformer in rdmol.GetConformers():
        # Get the coordinates of the conformer
        coords = conformer.GetPositions()
        # Translate atom 0 to the same position
        translate = coords[0] - ref_coords[0]
        coords = coords - translate
        # Compute two vectors -- atom 1 - atom 0, atom2 - atom0
        vec1_ref = ref_coords[1] - ref_coords[0]
        vec1_ref = vec1_ref / np.linalg.norm(vec1_ref)
        vec1 = coords[1] - coords[0]
        vec1 = vec1 / np.linalg.norm(vec1)
        vec2_ref = ref_coords[2] - ref_coords[0]
        vec2_ref = vec2_ref / np.linalg.norm(vec2_ref)
        vec2 = coords[2] - coords[0]
        vec2 = vec2 / np.linalg.norm(vec2)
        # Rotate the molecule to best align these two vectors
        a = np.array([vec1_ref, vec2_ref], dtype=np.float64)
        b = np.array([vec1, vec2], dtype=np.float64)
        R, rmsd = Rotation.align_vectors(a, b)
        coords = R.apply(coords)

        xyz_block += f"{len(elements)}\n\n"
        for element, coord in zip(elements, coords):
            x, y, z = coord
            xyz_block += f"{element:5s}{x:12.6f}{y:12.6f}{z:12.6f}\n"

    fpath = Path("structures/aligned_conformers.xyz")
    with fpath.open("w") as f:
        f.write(xyz_block)
    log.log("Wrote all aligned to 'structures/aligned_conformers.xyz'.")


def _check_relative_conformer_energies(
        rdmol, molecules, energies, energy_threshold, log
):
    energies = np.array(energies) * HARTREE_TO_KJMOL
    min_energy = energies.min()
    relative_energies = energies - min_energy
    remove_molecules = np.where(relative_energies > energy_threshold)[0]
    if len(remove_molecules > 0):
        log.log(
            f"Removing conformers {remove_molecules} because their "
            f"energy > {energy_threshold} kJ/mol above the lowest "
            f"energy conformer. Saving coordinates to "
            f"structures/unstable_conformer_xx.xyz."
        )
        for mol_idx in remove_molecules:
            mol = molecules.pop(mol_idx)
            fpath = Path("structures/unstable_conformer_{mol_idx}.xyz")
            mol.save_xyz_file(
                str(fpath), 1
            )
            rdmol.RemoveConformer(int(mol_idx))

    # Renumber conformers
    for idx, c in enumerate(rdmol.GetConformers()):
        c.SetId(idx)


def _initialize_resp(resp_path):
    # Change to resp dir
    os.chdir(resp_path)
    # Setup the logging
    psi4.set_output_file("psi4.log")
    log = Logger(Path("resp.log"))

    # Read the input
    input_path = Path("resp.yaml")
    with input_path.open() as f:
        log.log(f"\n\nLoaded YAML file: resp.yaml")
        log.log("-" * 100)
        log.log(f.read())
        log.log("-" * 100 + "\n\n")
        f.seek(0)
        inp = yaml.load(f, Loader=yaml.FullLoader)

    # TODO: Add YAML validation (https://docs.python-cerberus.org/en/stable/)

    # Make the directories
    Path("structures").mkdir()
    Path("esp_grids").mkdir()
    Path("results").mkdir()

    return inp, log


def _write_results(
        elements, charges, equality_constraints, charge_type, log
):

    fname = "results/charges_" + charge_type + "_full.out"
    result_path = Path(fname)
    with result_path.open("w") as f:
        for element, charge in zip(elements, charges):
            f.write(f"{element:5s}{charge:12.8f}\n")
    log.log(f"\n\nFull precision charges have been saved to '{fname}'.")

    # Round charges to 3 decimals
    # This may violate symmetry; need to edit by hand
    charges = saferound(charges, 3)
    fname = "results/charges_" + charge_type + "_rounded.out"
    result_path = Path(fname)
    with result_path.open("w") as f:
        for element, charge in zip(elements, charges):
            f.write(f"{element:5s}{charge:12.3f}\n")
    log.log(
        f"Rounded charges (3 decimals) have been saved to '{fname}'."
    )

    warn_log = False
    for constraint in equality_constraints:
        for atomi in constraint:
            chargei = charges[atomi-1]
            for atomj in constraint:
                chargej = charges[atomj-1]
                if chargei != chargej:
                    warn_log = True

    if warn_log:
        log.log(
            f"Rounded charges in '{fname}' violate the specified "
            "equality constraints and need to be corrected by hand."
        )
