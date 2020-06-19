import pytest
import chemdash.swap
import chemdash.rngs

from builtins import range

import ase
import numpy as np
import subprocess

#===========================================================================================================================================================
#===========================================================================================================================================================
#Track tests

#UNIT TESTS
#determine_maximum_swaps
#generate_linearly_decreasing_weightings
#generate_selection_pool
#generate_swap_list -- UNFINISHED


#INTEGRATION TESTS
#choose_number_of_atoms_to_swap
#swap_atoms



#===========================================================================================================================================================
#===========================================================================================================================================================
#Fixtures


@pytest.fixture
def STOX_structure(scope = "module"):
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
        An ASE atoms object containing SrTiO_{3} and five vacancies.

    ---------------------------------------------------------------------------
    Paul Sharp 07/07/2017
    """

    return ase.Atoms(symbols = "SrTiO3X5", charges = [2.0, 4.0, -2.0, -2.0, -2.0, 0.0, 0.0, 0.0, 0.0, 0.0], cell = [1.0, 1.0, 1.0],
                     scaled_positions = ([0.75, 0.75, 0.25], [0.75, 0.25, 0.25], [0.5, 0.5, 0.5], [0.5, 0.0, 0.0], [0.0, 0.0, 0.5], [0.0, 0.0, 0.0],
                                         [0.25, 0.25, 0.25], [0.75, 0.75, 0.75], [0.0, 0.5, 0.0], [0.25, 0.25, 0.75]),
                     pbc=[True, True, True])


#===========================================================================================================================================================
@pytest.fixture
def STO_atoms(scope = "module"):
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

    return ase.Atoms(symbols = "SrTiO3", charges = [2.0, 4.0, -2.0, -2.0, -2.0], cell = [1.0, 1.0, 1.0],
                     scaled_positions = ([0.75, 0.75, 0.25], [0.75, 0.25, 0.25], [0.5, 0.5, 0.5], [0.5, 0.0, 0.0], [0.0, 0.0, 0.5]),
                     pbc=[True, True, True])


#===========================================================================================================================================================
#===========================================================================================================================================================
#Unit tests


@pytest.mark.parametrize("selection, reduced_structure, expected_output", [
    ("atoms", STO_atoms, (["Sr", "Ti", "O", "O", "O"], 4)),
])

def test_determine_maximum_swaps_1(STOX_structure, selection, reduced_structure, expected_output, monkeypatch, STO_atoms):
    """
    GIVEN a structure and group of atoms to be involved in a swap

    WHEN we find out the maximum number of atoms that can be swapped

    THEN we get the elements available to be swapped and the maximum number to swap non-trivially


    Parameters
    ----------
    STOX_structure : ase atoms
        A structure containing a single formula unit of Strontium Titanate and some vacancies.
    selection : string
        The chosen swap group.
    reduced_structure : ase atoms
        The pytest fixture corresponding to the structure containing only atoms in the chosen swap group.
    expected_output : (string, int)
        The list of elements to swap and maximum number of swaps.
    STO_atoms : ase atoms
        A structure containing a single formula unit of Strontium Titanate.

    ---------------------------------------------------------------------------
    Paul Sharp 27/06/2019
    """

    monkeypatch.setattr(chemdash.swap, 'reduce_structure', lambda x, y : STO_atoms)
    assert chemdash.swap.determine_maximum_swaps(STOX_structure, selection) == expected_output


#===========================================================================================================================================================
@pytest.mark.parametrize("selection, reduced_structure, expected_output", [
    ("atoms-vacancies", STOX_structure, (["A", "A", "A", "A", "A", "X", "X", "X", "X", "X"], 10)),
])

def test_determine_maximum_swaps_2(STOX_structure, selection, reduced_structure, expected_output, monkeypatch, STO_atoms):
    """
    GIVEN a structure and group of atoms to be involved in a swap

    WHEN we find out the maximum number of atoms that can be swapped

    THEN we get the elements available to be swapped and the maximum number to swap non-trivially


    Parameters
    ----------
    STOX_structure : ase atoms
        A structure containing a single formula unit of Strontium Titanate and some vacancies.
    selection : string
        The chosen swap group.
    reduced_structure : ase atoms
        The pytest fixture corresponding to the structure containing only atoms in the chosen swap group.
    expected_output : (string, int)
        The list of elements to swap and maximum number of swaps.
    STO_atoms : ase atoms
        A structure containing a single formula unit of Strontium Titanate.

    ---------------------------------------------------------------------------
    Paul Sharp 27/06/2019
    """

    monkeypatch.setattr(chemdash.swap, 'reduce_structure', lambda x, y : STOX_structure)
    assert chemdash.swap.determine_maximum_swaps(STOX_structure, selection) == expected_output
    

#===========================================================================================================================================================
@pytest.mark.parametrize("max_atoms, expected_output", [
    (1, [1.0]),
    (2, [2.0, 1.0]),
    (4, [4.0, 3.0, 2.0, 1.0]),
])

def test_generate_linearly_decreasing_weightings(max_atoms, expected_output):
    """
    GIVEN the maximum number of atoms that can be swapped

    WHEN generating probabilities for choosing a particular number of atoms

    THEN we get a list of probabilities for choosing each number from 2 atoms up to the maximum

    Parameters
    ----------
    max_atoms : int
        The maximum number of atoms that can be swapped.
    expected_output : float
        Probabilities for choosing each number of atoms from 2 up to the maximum.

    ---------------------------------------------------------------------------
    Paul Sharp 11/09/2017
    """

    probabilities = chemdash.swap.generate_linearly_decreasing_weightings(max_atoms)

    assert len(probabilities) == max_atoms
    assert probabilities == expected_output


#===========================================================================================================================================================
@pytest.mark.parametrize("max_atoms", [
    (0),
    (-5),
])

def test_generate_linearly_decreasing_weightings_exceptions(max_atoms):
    """
    GIVEN an invalid maximum number of atoms that can be swapped

    WHEN generating probabilities for choosing a particular number of atoms

    THEN we raise an assertion error


    Parameters
    ----------
    max_atoms : int
        The maximum number of atoms that can be swapped.

    ---------------------------------------------------------------------------
    Paul Sharp 06/07/2017
    """
    
    with pytest.raises(AssertionError):
        chemdash.swap.generate_linearly_decreasing_weightings(max_atoms)


#===========================================================================================================================================================
@pytest.mark.parametrize("max_atoms, expected_output", [
    (1, [1.0]),
    (2, [2.0, 1.0]),
    (4, [8.0, 4.0, 2.0, 1.0]),
])

def test_generate_geometrically_decreasing_weightings(max_atoms, expected_output):
    """
    GIVEN the maximum number of atoms that can be swapped

    WHEN generating probabilities for choosing a particular number of atoms

    THEN we get a list of probabilities for choosing each number from 2 atoms up to the maximum

    Parameters
    ----------
    max_atoms : int
        The maximum number of atoms that can be swapped.
    expected_output : float
        Probabilities for choosing each number of atoms from 2 up to the maximum.

    ---------------------------------------------------------------------------
    Paul Sharp 09/08/2018
    """

    probabilities = chemdash.swap.generate_geometrically_decreasing_weightings(max_atoms)

    assert len(probabilities) == max_atoms
    assert probabilities == expected_output


#===========================================================================================================================================================
@pytest.mark.parametrize("max_atoms", [
    (0),
    (-5),
])

def test_generate_geometrically_decreasing_weightings_exceptions(max_atoms):
    """
    GIVEN an invalid maximum number of atoms that can be swapped

    WHEN generating probabilities for choosing a particular number of atoms

    THEN we raise an assertion error


    Parameters
    ----------
    max_atoms : int
        The maximum number of atoms that can be swapped.

    ---------------------------------------------------------------------------
    Paul Sharp 09/08/2018
    """
    
    with pytest.raises(AssertionError):
        chemdash.swap.generate_geometrically_decreasing_weightings(max_atoms)


#===========================================================================================================================================================
@pytest.mark.parametrize("max_atoms, pair_weighting, expected_output", [
    (1, 1, [1.0]),
    (2, 1, [1.0, 1.0]),
    (4, 1, [6.0, 3.0, 2.0, 1.0]),
    (1, 3, [1.0]),
    (2, 3, [3.0, 1.0]),
    (4, 3, [18.0, 3.0, 2.0, 1.0]),
])

def test_pair_pinned_weightings(max_atoms, pair_weighting, expected_output):
    """
    GIVEN the maximum number of atoms that can be swapped

    WHEN generating probabilities for choosing a particular number of atoms

    THEN we get a list of probabilities for choosing each number from 2 atoms up to the maximum

    Parameters
    ----------
    max_atoms : int
        The maximum number of atoms that can be swapped.
    expected_output : float
        Probabilities for choosing each number of atoms from 2 up to the maximum.

    ---------------------------------------------------------------------------
    Paul Sharp 09/08/2018
    """

    probabilities = chemdash.swap.pair_pinned_weightings(max_atoms, pair_weighting)

    assert len(probabilities) == max_atoms
    assert probabilities == expected_output


#===========================================================================================================================================================
@pytest.mark.parametrize("max_atoms, pair_weighting", [
    (0, 1),
    (-5, 1),
])

def test_pair_pinned_weightings_exceptions(max_atoms, pair_weighting):
    """
    GIVEN an invalid maximum number of atoms that can be swapped

    WHEN generating probabilities for choosing a particular number of atoms

    THEN we raise an assertion error


    Parameters
    ----------
    max_atoms : int
        The maximum number of atoms that can be swapped.

    ---------------------------------------------------------------------------
    Paul Sharp 08/09/2018
    """
    
    with pytest.raises(AssertionError):
        chemdash.swap.pair_pinned_weightings(max_atoms, pair_weighting)


#===========================================================================================================================================================
@pytest.mark.parametrize("max_atoms, expected_output", [
    (1, [1.0]),
    (2, [1.0, 1.0]),
    (4, [1.0, 1.0, 1.0, 1.0]),
])

def test_generate_uniform_weightings(max_atoms, expected_output):
    """
    GIVEN the maximum number of atoms that can be swapped

    WHEN generating probabilities for choosing a particular number of atoms

    THEN we get a list of probabilities for choosing each number from 2 atoms up to the maximum

    Parameters
    ----------
    max_atoms : int
        The maximum number of atoms that can be swapped.
    expected_output : float
        Probabilities for choosing each number of atoms from 2 up to the maximum.

    ---------------------------------------------------------------------------
    Paul Sharp 27/06/2019
    """

    probabilities = chemdash.swap.generate_uniform_weightings(max_atoms)

    assert len(probabilities) == max_atoms
    assert probabilities == expected_output


#===========================================================================================================================================================
@pytest.mark.parametrize("max_atoms", [
    (0),
    (-5),
])

def test_generate_uniform_weightings_exceptions(max_atoms):
    """
    GIVEN an invalid maximum number of atoms that can be swapped

    WHEN generating probabilities for choosing a particular number of atoms

    THEN we raise an assertion error


    Parameters
    ----------
    max_atoms : int
        The maximum number of atoms that can be swapped.

    ---------------------------------------------------------------------------
    Paul Sharp 27/06/2019
    """
    
    with pytest.raises(AssertionError):
        chemdash.swap.generate_linearly_decreasing_weightings(max_atoms)

        
#===========================================================================================================================================================
@pytest.mark.parametrize("elements_list, num_swaps, expected_output", [
    (["Sr", "Ti", "O", "O", "O", "X", "X", "X", "X", "X"], 2, (["O", "Sr", "Ti", "X"], 2)),
    (["Sr", "Ti", "O", "O", "O", "X", "X", "X", "X", "X"], 5, (["O", "O", "Sr", "Ti", "X", "X"], 5)),
    (["Sr", "Ti", "O", "O", "O", "X", "X", "X", "X", "X"], 10, (["O", "O", "O","Sr", "Ti", "X", "X", "X", "X", "X"], 10)),
    (["Sr", "Sr", "Ti", "Ti"], 3, (["Sr", "Ti"], 2)),
    (["O", "O", "O",], 2, (["O"], 1)),
])

def test_generate_selection_pool(elements_list, num_swaps, expected_output):
    """
    GIVEN a list of elements and number of atoms we intend to swap

    WHEN generating the list of atoms we will choose from

    THEN we get a valid selection pool and the number of atoms we will swap -- choosing this many atoms from the selection pool WILL result in a non-trivial swap.
 

    Parameters
    ----------
    elements_list : string
        The list of elements from the structure in the chosen swap group.
    num_swaps : int
        The number of atoms we have chosen to swap.
    expected_output : (string, int)
        The selection pool and the number of atoms we will swap

    ---------------------------------------------------------------------------
    Paul Sharp 25/06/2019
    """

    selection_pool, num_swaps = chemdash.swap.generate_selection_pool(elements_list, num_swaps)

    assert len(selection_pool) >= num_swaps
    assert (sorted(selection_pool), num_swaps) == expected_output


#===========================================================================================================================================================
@pytest.mark.parametrize("elements_list, num_swaps", [
    (["Sr", "Ti", "O", "O", "O", "X", "X", "X", "X", "X"], 0),
    (["Sr", "Ti", "O", "O", "O", "X", "X", "X", "X", "X"], 1),
    (["Sr", "Ti", "O", "O", "O", "X", "X", "X", "X", "X"], -5),
])

def test_generate_selection_pool_exceptions(elements_list, num_swaps):
    """
    GIVEN a list of elements and an invalid number of atoms we intend to swap

    WHEN generating the list of atoms we will choose from

    THEN we raise an assertion error


    Parameters
    ----------
    elements_list : string
        The list of elements from the structure in the chosen swap group.
    num_swaps : int
        The number of atoms we have chosen to swap.

    ---------------------------------------------------------------------------
    Paul Sharp 12/07/2017
    """

    with pytest.raises(AssertionError):
        selection_pool, num_swaps = chemdash.swap.generate_selection_pool(elements_list, num_swaps)


#===========================================================================================================================================================
@pytest.mark.parametrize("selection_pool, num_swaps, expected_output", [
    (["a", "b", "c", "d", "e"], 2, ["a", "b"]),
    (["a", "b", "c", "d", "e"], 5, ["a", "b", "c", "d", "e"]),
    (["a", "b", "c", "d", "e"], 0, []),
    (["a", "b", "c", "d", "e"], -1, []),
])

def test_generate_swap_list(selection_pool, num_swaps, rng, expected_output, monkeypatch):
    """
    GIVEN a selection pool and a number of atoms we intend to swap

    WHEN generating the list of elements we will swap

    THEN we get a list of the elements to swap


    Parameters
    ----------
    selection_pool : string
        The list of elements to choose from.
    num_swaps : int
        The number of atoms we have chosen to swap.
    rng : NR_Ran
        Random Number Generator.
    expected_output : string
        The list of elements we will swap.

    ---------------------------------------------------------------------------
    Paul Sharp 25/06/2019
    """

    #NEED TO CONSIDER HOW TO PATCH THE RNG

    monkeypatch.setattr(rng, 'int_range', lambda u_lim : 0)
    assert chemdash.swap.generate_swap_list(selection_pool, num_swaps, rng) == expected_output































#===========================================================================================================================================================
@pytest.mark.parametrize("old_order, force_vacancy_swaps", [
    (np.asarray(["Sr", "Sr", "Ti", "Ti", "O", "O", "X", "X"]), False),
    (np.asarray(["Sr", "Sr", "Ti", "Ti", "O", "O", "X", "X"]), True),
    (np.asarray(["Sr", "Sr", "Ti", "Ti", "O", "O", "X", "X", "X", "X", "X", "X"]), False),
    (np.asarray(["Sr", "Sr", "Ti", "Ti", "O", "O", "X", "X", "X", "X", "X", "X"]), True),
])

def test_permute_atoms(old_order, force_vacancy_swaps, rng):
    """
    The "permute_atoms()" routine should change the reorder a list such that each element is different.


    Parameters
    ----------
    None

    Returns
    -------
    None  

    ---------------------------------------------------------------------------
    Paul Sharp 06/07/2017
    """

    new_order = chemdash.swap.permute_atoms(old_order, rng, force_vacancy_swaps)

    if not force_vacancy_swaps:
        old_order[old_order=="X"]="Z"

    assert (new_order != old_order).all()


#===========================================================================================================================================================
@pytest.mark.parametrize("current_energy, new_energy, temperature, expected_output", [
    (2.0, 1.0, 0.0, True),
    (1.0, 2.0, 0.0, False),
    (-2.0, -1.0, 0.0, False),
    (-1.0, -2.0, 0.0, True),
    (-1.0, "****", 0.0, False),
])

def test_accept_swap(current_energy, new_energy, temperature, rng, expected_output):
    """
    The "accept_swap()" routine should accept all swaps where new_energy < current_energy, and acceptance of higher energy swaps depends on the temperature.


    Parameters
    ----------
    None

    Returns
    -------
    None  

    ---------------------------------------------------------------------------
    Paul Sharp 06/07/2017
    """

    assert chemdash.swap.accept_swap(current_energy, new_energy, temperature, rng) == expected_output

#LOTS OF TEST CASES NEEDED HERE - HOW TO DO FINITE T? -- NEEDED FOR COVERAGE


#===========================================================================================================================================================
@pytest.mark.parametrize("atom_filter, expected_output", [
    ("cations", "SrTi"),
    ("anions", "O3"),
    ("atoms", "O3SrTi"),
    ("vacancies", "X5"),
    ("all", "O3SrTiX5"),
    ("atoms-vacancies", "O3SrTiX5"),
    ("Sr-X", "SrX5"),
    ("O-Ti", "O3Ti"),
])


def test_reduce_structure(STOX_structure, atom_filter, expected_output):
    """
    The "reduce_structure()" routine should remove all atoms that do not fit the specified atom filter.


    Parameters
    ----------
    None

    Returns
    -------
    None  

    ---------------------------------------------------------------------------
    Paul Sharp 06/07/2017
    """

    structure = chemdash.swap.reduce_structure(STOX_structure, atom_filter)

    assert structure.get_chemical_formula() == expected_output


#Need to test for a problem when input is not atoms object


#===========================================================================================================================================================
@pytest.mark.parametrize("basins, new_energy, expected_output", [
    ({}, 1.0, {1.0: 1}),
    ({1.0: 2, 2.0: 1}, 3.0, {1.0: 2, 2.0: 1, 3.0: 1}),
    ({1.0: 2, 2.0: 1}, 1.0, {1.0: 3, 2.0: 1}),
])


def test_update_basins(basins, new_energy, expected_output):
    """
    The "update_basins()" routine should increment the visit count for visited basins and add new basins to the list.


    Parameters
    ----------
    None

    Returns
    -------
    None  

    ---------------------------------------------------------------------------
    Paul Sharp 07/07/2017
    """

    assert chemdash.swap.update_basins(basins, new_energy) == expected_output


#===========================================================================================================================================================
@pytest.mark.parametrize("swap_groups, expected_output", [
    ([["cations"], ["anions"], ["atoms"], ["all"]], [["cations"], ["atoms"], ["all"]]),
    ([["cations"], ["atoms"], ["all"]], [["cations"], ["atoms"], ["all"]]),
    ([["anions"]], []),
])


def test_initialise_default_swap_groups(STOX_structure, swap_groups, expected_output):
    """
    GIVEN a structure and a set of swap groups

    WHEN we call "initialise_default_swap_groups()"

    THEN we return a list of swap groups that will allow for non-trivial swaps in the structure.

    Parameters
    ----------
    None

    Returns
    -------
    None  

    ---------------------------------------------------------------------------
    Paul Sharp 25/10/2017
    """

    assert chemdash.swap.initialise_default_swap_groups(STOX_structure, swap_groups) == expected_output

#Need to test for a problem when input is not atoms object


#===========================================================================================================================================================
@pytest.mark.parametrize("swap_groups, expected_output", [
    ([["cations"], ["anions"], ["atoms"], ["all"]], [["cations", 1.0], ["anions", 1.0], ["atoms", 1.0], ["all", 1.0]]),
    ([["Sr-X"], ["Ti-X"], ["O-X"]], [["Sr-X", 1.0], ["Ti-X", 1.0], ["O-X", 1.0]]),
    ([], []),
])


def test_initialise_default_swap_weightings(STOX_structure, swap_groups, expected_output):
    """
    GIVEN a set of swap groups

    WHEN we call "initialise_default_swap_weightings()"

    THEN we return a list of weightings assigning each swap group equal probability.

    Parameters
    ----------
    None

    Returns
    -------
    None  

    ---------------------------------------------------------------------------
    Paul Sharp 25/10/2017
    """

    assert chemdash.swap.initialise_default_swap_weightings(swap_groups) == expected_output


#===========================================================================================================================================================
@pytest.mark.parametrize("swap_groups, expected_output", [
    ([["cations"], ["atoms"], ["all"]], ([["cations"], ["atoms"], ["all"]], [])),
    ([["cations"], ["atoms"], ["all"], ["atoms-vacancies"]], ([["cations"], ["atoms"], ["all"], ["atoms-vacancies"]], [])),
    ([["cations"], ["anions"], ["atoms"], ["all"]], ([["cations"], ["atoms"], ["all"]], ['"anions" has been specified as a swap group, but there are insufficient different species to enable non-trivial swaps to be made -- there are 1 species of anions.'])),
    ([], ([], ['There are no valid swap groups.'])),
])


def test_verify_swap_groups(STOX_structure, swap_groups, expected_output):
    """
    The "verify_swap_groups()" routine should report an error for any swap groups that do not enable non-trivial swaps for a particular structure.


    Parameters
    ----------
    None

    Returns
    -------
    None  

    ---------------------------------------------------------------------------
    Paul Sharp 25/06/2019
    """

    assert chemdash.swap.verify_swap_groups(STOX_structure, swap_groups) == expected_output


#Need to test for a problem when input is not atoms object


#===========================================================================================================================================================
@pytest.mark.parametrize("swap_groups, expected_output", [
    ([["cations"], ["atoms"], ["all"]], []),
    ([["cations"], ["Sr-X"]], []),
    ([["cations"], ["Fe-Sr-X"]], ['"Fe-Sr-X" has been specified as a swap group, but there are no Fe atoms in the structure.']),
])


def test_check_elements_in_custom_swap_groups(STOX_structure, swap_groups, expected_output):
    """
    The "check_elements_in_custom_swap_groups()" routine should check that the elements given in custom swap groups exist in the structure.


    Parameters
    ----------
    None

    Returns
    -------
    None  

    ---------------------------------------------------------------------------
    Paul Sharp 07/07/2017
    """

    assert chemdash.swap.check_elements_in_custom_swap_groups(swap_groups, STOX_structure.get_chemical_symbols()) == expected_output




#===========================================================================================================================================================
#===========================================================================================================================================================
#Integration Tests


#===========================================================================================================================================================
@pytest.mark.parametrize("max_swaps, weightings", [
    (2, "arithmetic"),
    (5, "arithmetic"),
    (100, "arithmetic"),
    (2, "geometric"),
    (5, "geometric"),
    (100, "geometric"),
    (2, "pinned_pair"),
    (5, "pinned_pair"),
    (100, "pinned_pair"),
    (2, "uniform"),
    (5, "uniform"),
    (100, "uniform"),
])


def test_choose_number_of_atoms_to_swap(max_swaps, weightings, rng):
    """
    The "choose_number_of_atoms_to_swap()" routine should choose a number of atoms to swap between 2 and the maximum possible.


    Parameters
    ----------
    None

    Returns
    -------
    None  

    ---------------------------------------------------------------------------
    Paul Sharp 09/08/2018
    """

    assert 2 <= chemdash.swap.choose_number_of_atoms_to_swap(max_swaps, weightings, rng, 1) <= max_swaps



#===========================================================================================================================================================
@pytest.mark.parametrize("max_swaps", [
    (2),
    (5),
    (100),
])


def test_choose_number_of_atoms_to_swap2(max_swaps, rng, monkeypatch):
    """
    The "choose_number_of_atoms_to_swap()" routine should choose a number of atoms to swap between 2 and the maximum possible.


    Parameters
    ----------
    None

    Returns
    -------
    None  

    ---------------------------------------------------------------------------
    Paul Sharp 08/09/2017
    """

    def mockreturn(elements):
        return [0] * (elements - 1) + [1]

    monkeypatch.setattr(chemdash.swap, 'generate_linearly_decreasing_weightings', mockreturn)
    assert chemdash.swap.choose_number_of_atoms_to_swap(max_swaps, "arithmetic", rng, "") == max_swaps


#===========================================================================================================================================================
@pytest.mark.parametrize("swap_list, ranking_dict, directed_num_atoms, predefined_vacancies, vacancy_exclusion_radius, force_vacancy_swaps", [
    (["Sr", "Ti"], {}, 0, True, 1.0, False),
    (["Sr", "Ti", "O", "O", "X"], {}, 0, True, 1.0, True),
    (["Sr", "Ti", "O", "O", "O", "X", "X", "X", "X", "X"], {}, 0, True, 1.0, True),
])

def test_swap_atoms(STOX_structure, rng, swap_list, ranking_dict, directed_num_atoms, predefined_vacancies, vacancy_exclusion_radius, force_vacancy_swaps):
    """
    The "swap_atoms()" routine should change the positions of "num_swaps" atoms from the selection pool.


    Parameters
    ----------
    None

    Returns
    -------
    None  

    ---------------------------------------------------------------------------
    Paul Sharp 25/06/2019
    """

    swapped_structure, swap_text = chemdash.swap.swap_atoms(STOX_structure.copy(), swap_list, ranking_dict, directed_num_atoms,
                                                            predefined_vacancies, vacancy_exclusion_radius, rng, force_vacancy_swaps)

    #Determine the atoms that were swapped (positions not equal in structures)
    old_elements = STOX_structure.get_chemical_symbols()

    old_positions = STOX_structure.get_scaled_positions()
    new_positions = swapped_structure.get_scaled_positions()

    swapped_atoms = []
    swapped_indices = []
    swapped_positions = []
    num_swapped_atoms = 0

    for i in range(len(STOX_structure)):

        if (new_positions[i] != old_positions[i]).any():

            swapped_atoms.append(old_elements[i])
            swapped_indices.append(i)
            swapped_positions.append(old_positions[i])
            num_swapped_atoms += 1


    assert len(swap_text) == len(swap_list)

    #Remove entries where vacancies swap with one another
    reduced_swap_text = [entry for entry in swap_text if entry[0] != entry[3]]

    if force_vacancy_swaps:
        assert num_swapped_atoms == len(swap_list)
    else:
        assert num_swapped_atoms <= len(swap_list)

    assert sorted([entry[0] for entry in reduced_swap_text]) == sorted(swapped_atoms)
    assert sorted([entry[1].tolist() for entry in reduced_swap_text]) == sorted([swap_pos.tolist() for swap_pos in swapped_positions])
    assert sorted([entry[2] for entry in reduced_swap_text]) == sorted(swapped_indices)
    assert sorted([entry[3] for entry in reduced_swap_text]) == sorted(swapped_atoms)
