import pytest
import chemdash
import chemdash.rngs

import ase
import os
import subprocess

os.environ["GULP_LIB"] = ""

#===========================================================================================================================================================
#===========================================================================================================================================================
#Fixtures

@pytest.fixture
def rng(scope = "session"):
    """
    This fixture returns a randomly-seeded random number generator. 

    Parameters
    ----------
    None

    Returns
    -------
    rng : NR_ran
        A randomly-seeded random number generator.

    ---------------------------------------------------------------------------
    Paul Sharp 07/07/2017
    """

    random_seed_command = "od -vAn -N4 -tu4 < /dev/urandom"
    random_seed = int(subprocess.check_output(random_seed_command, shell=True))

    return chemdash.rngs.NR_Ran(random_seed)


#===========================================================================================================================================================
@pytest.fixture
def STOX_structure(scope="session"):
    """
    This fixture returns an ASE atoms object containing a formula unit of SrTiO_{3} and five vacancies ("X"). 

    Parameters
    ----------
    atoms : string
        The chemical symbols and number of atoms of that species for each element in this structure.
    charges : float
        The charge of each atom in the structure.

    Returns
    -------
    structure : ASE atoms
        An ASE atoms object containing SrTiO_{3} and a vacancy.

    ---------------------------------------------------------------------------
    Paul Sharp 07/07/2017
    """

    return ase.Atoms(symbols="SrTiO3X5", cell = [2.0, 2.0, 2.0], charges=[2.0, 4.0, -2.0, -2.0, -2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                     scaled_positions=([0.75, 0.75, 0.25], [0.75, 0.25, 0.25], [0.5, 0.5, 0.5], [0.5, 0.0, 0.0], [0.0, 0.0, 0.5],
                                       [0.0, 0.0, 0.0], [0.25, 0.25, 0.25], [0.75, 0.75, 0.75], [0.0, 0.5, 0.0], [0.25, 0.25, 0.75]),
                     pbc=[True, True, True])


#===========================================================================================================================================================
@pytest.fixture
def STO_atoms(scope = "session"):
    """
    This fixture returns an ASE atoms object containing a formula unit of SrTiO_{3}. 

    Parameters
    ----------
    atoms : string
        The chemical symbols and number of atoms of that species for each element in this structure.
    charges : float
        The charge of each atom in the structure.

    Returns
    -------
    structure : ASE atoms
        An ASE atoms object containing SrTiO_{3}.

    ---------------------------------------------------------------------------
    Paul Sharp 11/09/2017
    """

    return ase.Atoms(symbols = "SrTiO3", cell = [2.0, 2.0, 2.0], charges = [2.0, 4.0, -2.0, -2.0, -2.0],
                     scaled_positions = ([0.75, 0.75, 0.25], [0.75, 0.25, 0.25], [0.5, 0.5, 0.5], [0.5, 0.0, 0.0], [0.0, 0.0, 0.5]),
                     pbc=[True, True, True])
