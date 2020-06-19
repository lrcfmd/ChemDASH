import pytest
import mock

import chemdash.bonding

import ase
import numpy as np

#===========================================================================================================================================================
#===========================================================================================================================================================
# Track tests

# UNIT TESTS
# test_check_R0_values
# test_check_R0_values_false


#INTEGRATION TESTS
# test_bond_valence_sum_for_atoms
# test_bond_valence_sum_for_sites


#===========================================================================================================================================================
#===========================================================================================================================================================
#Fixtures

@pytest.fixture
def STOX_structure(scope = "module"):
    """
    This fixture returns an ASE atoms object containing a formula unit of SrTiO3 and five vacancies ("X"). 

    Parameters
    ----------
    atoms : string
        The chemical symbols and number of atoms of that species for each element in this structure.
    charges : float
        The charge of each atom in the structure.

    Returns
    -------
    structure : ASE atoms
        An ASE atoms object containing SrTiO_{3} and five vacancies.

    ---------------------------------------------------------------------------
    Paul Sharp 15/01/2020
    """

    return ase.Atoms(symbols = "SrTiO3X5", cell = [3.955, 3.955, 3.955], charges = [2.0, 4.0, -2.0, -2.0, -2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                     scaled_positions = ([0.366, 0.214, 0.513], [0.866, 0.714, 0.013], [0.866, 0.714, 0.513], [0.366, 0.714, 0.013], [0.866, 0.214, 0.013],
                                         [0.0, 0.0, 0.0], [0.25, 0.25, 0.25], [0.75, 0.75, 0.75], [0.0, 0.5, 0.0], [0.25, 0.25, 0.75]),
                     pbc=[True, True, True])

#===========================================================================================================================================================
@pytest.fixture
def STO_atoms(scope = "module"):
    """
    This fixture returns an ASE atoms object containing a formula unit of SrTiO3. 

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
    Paul Sharp 15/01/2020
    """

    return ase.Atoms(symbols = "SrTiO3", cell = [3.955, 3.955, 3.955],
                     charges = [2.0, 4.0, -2.0, -2.0, -2.0],
                     scaled_positions = ([0.366, 0.214, 0.513], [0.866, 0.714, 0.013], [0.866, 0.714, 0.513], [0.366, 0.714, 0.013], [0.866, 0.214, 0.013]),
                     pbc=[True, True, True])


#===========================================================================================================================================================
@pytest.fixture
def STS_atoms(scope = "module"):
    """
    This fixture returns an ASE atoms object containing a formula unit of the non-existent SrTiS3. 

    Parameters
    ----------
    atoms : string
        The chemical symbols and number of atoms of that species for each element in this structure.
    charges : float
        The charge of each atom in the structure.

    Returns
    -------
    structure : ASE atoms
        An ASE atoms object containing SrTiS_{3}.

    ---------------------------------------------------------------------------
    Paul Sharp 24/07/2019
    """

    return ase.Atoms(symbols = "SrTiS3", cell = [3.955, 3.955, 3.955],
                     charges = [2.0, 4.0, -2.0, -2.0, -2.0],
                     scaled_positions = ([0.866, 0.714, 0.513], [0.366, 0.714, 0.013], [0.866, 0.214, 0.013], [0.366, 0.214, 0.513], [0.866, 0.714, 0.013]),
                     pbc=[True, True, True])


#===========================================================================================================================================================
#===========================================================================================================================================================
#Unit Tests

@pytest.mark.parametrize("expected_output", [
    ([]),
])

def test_check_R0_values(STOX_structure, expected_output):
    """
    GIVEN a crystal structure

    WHEN we check the R0 values for all cation-anion bonds

    THEN we return a list containing all bonds where an r0 value does not exist

    Parameters
    ----------
    None

    ---------------------------------------------------------------------------
    Paul Sharp 23/07/2019
    """
    
    assert chemdash.bonding.check_R0_values(STOX_structure) == expected_output

#===========================================================================================================================================================
@pytest.mark.parametrize("expected_output", [
    (["Sr2--S-2", "Ti4--S-2"]),
])

def test_check_R0_values_false(STS_atoms, expected_output):
    """
    GIVEN a crystal structure containing bonds that do not have an R0 value

    WHEN we check the R0 values for all cation-anion bonds

    THEN we return a list containing all bonds where an R0 value does not exist

    Parameters
    ----------
    None

    ---------------------------------------------------------------------------
    Paul Sharp 16/20/2020
    """
    
    assert sorted(chemdash.bonding.check_R0_values(STS_atoms)) == expected_output

#===========================================================================================================================================================
@pytest.mark.parametrize("bvs_atom, bonded_atoms, distances, expected_output", [
    (0, [2, 3, 4], np.array([2.79660732, 2.79660732, 1.9775    ]), 1.7814122272482644),
    (1, [2, 3, 4], np.array([2.79660732, 2.79660732, 1.9775    ]), 0.7854389990261801),
    (2, [0, 1], np.array([2.79660732, 2.79660732]), 0.23020148547836117),
])

def test_compute_bond_valence_sum(STO_atoms, bvs_atom, bonded_atoms, distances, expected_output):
    """
    GIVEN a crystal structure, a chosen atom for the BVS and distances to the atoms in may bond with 

    WHEN we calculate the bond valence sum for the chosen atom

    THEN we return a value for the bond valence sum for that atom

    Parameters
    ----------
    None

    ---------------------------------------------------------------------------
    Paul Sharp 25/07/2019
    """
    
    assert chemdash.bonding.compute_bond_valence_sum(STO_atoms, bvs_atom, bonded_atoms, distances) == expected_output
    
#===========================================================================================================================================================
@pytest.mark.parametrize("bvs_atom, bonded_atoms, distances, expected_output", [
    (0, [2, 3, 4], np.array([2.79660732, 2.79660732, 1.9775    ]), 0.0),
])

def test_compute_bond_valence_sum_false(STS_atoms, bvs_atom, bonded_atoms, distances, expected_output):
    """
    GIVEN a crystal structure containing bonds that do not have an R0 value

    WHEN we try to calculate the bond valence sum

    THEN we return the value zero

    Parameters
    ----------
    None

    ---------------------------------------------------------------------------
    Paul Sharp 25/07/2019
    """
    
    assert chemdash.bonding.compute_bond_valence_sum(STS_atoms, bvs_atom, bonded_atoms, distances) == expected_output

    
#===========================================================================================================================================================
#===========================================================================================================================================================
#Integration tests

@pytest.mark.parametrize("expected_output", [
    ([1.9326638785056407, 3.8887216952727934, 1.9404618579261488, 1.940461857926147, 1.9404618579261468]),
])

def test_bond_valence_sum_for_atoms(STO_atoms, expected_output):
    """
    GIVEN a crystal structure

    WHEN we calculate the bond valence sum for all atoms

    THEN we return a list of bond valence sum values, one for each atom

    Parameters
    ----------
    None

    ---------------------------------------------------------------------------
    Paul Sharp 15/01/2020
    """

    bvs = chemdash.bonding.bond_valence_sum_for_atoms(STO_atoms)
    
    assert len(bvs) == len(expected_output)
    assert bvs == pytest.approx(expected_output)

#===========================================================================================================================================================
@pytest.mark.parametrize("expected_output", [
    ([{'Ti/4.0': 0.8521270702920632, 'O/-2.0': 0.10314386494351863, 'Sr/2.0': 1.9326638785056407},
      {'Ti/4.0': 3.8887216952727934, 'O/-2.0': 0.23393508904426485, 'Sr/2.0': 8.81980190048298},
      {'Ti/4.0': 0.5680847135280425, 'O/-2.0': 1.9404618579261488, 'Sr/2.0': 1.2884425856704294},
      {'Ti/4.0': 0.5680847135280424, 'O/-2.0': 1.940461857926147, 'Sr/2.0': 1.2884425856704256},
      {'Ti/4.0': 0.5680847135280426, 'O/-2.0': 1.9404618579261468, 'Sr/2.0': 1.288442585670426},
      {'Ti/4.0': 10.66311114695696, 'O/-2.0': 5.329762028168363, 'Sr/2.0': 24.18443265644778},
      {'Ti/4.0': 2.1387271642484307, 'O/-2.0': 14.235439335872746, 'Sr/2.0': 4.8507328078487335},
      {'Ti/4.0': 9.556073462893677, 'O/-2.0': 6.747648105830908, 'Sr/2.0': 21.673619634863634},
      {'Ti/4.0': 6.947949584676197, 'O/-2.0': 9.608291664150249, 'Sr/2.0': 15.758273220189475},
      {'Ti/4.0': 1.9284235157949423, 'O/-2.0': 18.120841237644175, 'Sr/2.0': 4.373754339432401}]
    ),
])

def test_bond_valence_sum_for_sites(STOX_structure, expected_output):
    """
    GIVEN a crystal structure

    WHEN we calculate the bond valence sum for each atom type of each site

    THEN we return a list of dictionaries of bond valence sum values for each atom type on each site

    Parameters
    ----------
    None

    ---------------------------------------------------------------------------
    Paul Sharp 15/01/2020
    """

    bvs = chemdash.bonding.bond_valence_sum_for_sites(STOX_structure)
    
    assert len(bvs) == len(expected_output)
    assert all([x == pytest.approx(y) for x, y in zip(bvs, expected_output)])
