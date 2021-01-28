"""
|=============================================================================|
|                                  S W A P                                    |
|=============================================================================|
|                                                                             |
| This module contains routines that swap the positions of atoms in           |
| structures, whether these swaps should be accepted, and track our           |
| exploration of the potential energy surface.                                |
|                                                                             |
| We can swap multiple atoms, and our routines determine the group of atoms   |
| we involve in swapping (cations, anions, all atoms, all atoms and           |
| vacancies), and the number of atoms to swap. We then choose the atoms in    |
| order to ensure that all atoms can be swapped non-trivially, and make sure  |
| that this is the case during the swap.                                      |
|                                                                             |
| Contains                                                                    |
| --------                                                                    |
|     choose_swap_group                                                       |
|     choose_number_of_atoms_to_swap                                          |
|     generate_linearly_decreasing_weightings                                 |
|     generate_geometrically_decreasing_weightings                            |
|     pair_pinned_weightings                                                  |
|     swap_atoms                                                              |
|     generate_selection_pool                                                 |
|     generate_swap_list                                                      |
|     select_atoms_at_random                                                  |
|     select_atoms_directed                                                   |
|     find_vacancies_near_a_selected_vacancy                                  |
|     sort_swap_list                                                          |
|     get_swap_positions                                                      |
|     permute_atoms                                                           |
|     reorder_positions                                                       |
|     accept_swap                                                             |
|     reduce_structure                                                        |
|     update_basins                                                           |
|     check_previous_structures                                               |
|     initialise_default_swap_groups_and_weightings                           |
|     verify_swap_groups                                                      |
|     check_elements_in_custom_swap_groups                                    |
|     update_atom_rankings                                                    |
|     rank_bvs                                                                |
|     rank_site_potential                                                     |
|     rank_bvs_plus                                                           |
|     find_desired_atoms                                                      |
|     rank_atoms                                                              |
|                                                                             |
|-----------------------------------------------------------------------------|
| Paul Sharp 27/03/2020                                                       |
|=============================================================================|
"""

from builtins import range
from builtins import zip

import collections
import numpy as np


# =============================================================================
# =============================================================================
def determine_maximum_swaps(structure, selection):
    """
    Determine the maximum number of atoms that can be swapped given a structure
    and the group of atoms involved in the swap.

    Parameters
    ----------
    structure : ase atoms
        The current structure being considered.
    selection : string
        The chosen swap group.

    Returns
    -------
    elements_list : list of strings
        Symbols of atoms/vacancies we wish to be considered for swapping.
    max_swaps : integer
        Maximum number of atoms/vacancies we can swap non-trivially.

    ---------------------------------------------------------------------------
    Paul Sharp 27/03/2020
    """

    # Find atoms available for swapping
    reduced_structure = reduce_structure(structure.copy(), selection)
    elements_list = reduced_structure.get_chemical_symbols()

    # For the atoms-vacancies swap group, replace atomic chemical symbols with
    # generic atom symbol "A"
    if selection == "atoms-vacancies":
         elements_list = ["A" if element != "X" else element for element in elements_list]

    # Determine maximum number of swaps -- except clause required for when
    # elements_list is of zero length.
    try:
        modal_element = max(elements_list, key=elements_list.count)
    except ValueError:
        print('Failure in gulp_calc.determine_maximum_swaps() when attempting to swap {0}'.format(selection))
        print(structure)
        print(reduced_structure)
        print(elements_list)
        max_swaps = 0
    else:
        remaining_elements = [element for element in elements_list if element != modal_element]
        max_swaps = min(2 * len(remaining_elements), len(elements_list))
    
    return elements_list, max_swaps


# =============================================================================
def choose_number_of_atoms_to_swap(max_swaps, weightings_method, rng, pinned_weighting):
    """
    Determine the number of atoms to be included in this swap.

    Parameters
    ----------
    max_swaps : integer
        Maximum number of atoms/vacancies we can swap non-trivially.
    weightings_method : string
        Sequence used to construct the weightings to be used
    rng : NR_ran
        Random number generator - algorithm from Numerical Recipes 2007.
    pinned_weighting : integer
        The weighting for swapping two atoms compared to any other number,
        i.e, 3 for 75%, 2 for 66.666% and 1 for 50%.

    Returns
    -------
    num_swaps : integer
        Number of atoms/vacancies we wish to swap.

    ---------------------------------------------------------------------------
    Paul Sharp 27/03/2020
    """

    if weightings_method == "arithmetic":
        number_weightings = generate_linearly_decreasing_weightings(max_swaps - 1)
    elif weightings_method == "geometric":
        number_weightings = generate_geometrically_decreasing_weightings(max_swaps - 1)
    elif weightings_method == "uniform":
        number_weightings = generate_uniform_weightings(max_swaps - 1)
    elif weightings_method == "pinned_pair":
        number_weightings = pair_pinned_weightings(max_swaps - 1, pinned_weighting)
    else:
        sys.exit("ERROR in swap.choose_number_of_atoms_to_swap() -- {0} is not a valid choice for constructing the number weightings.".format(weightings_method))

    num_swaps = rng.weighted_choice(number_weightings) + 2

    return num_swaps


# =============================================================================
def generate_linearly_decreasing_weightings(num_elements):
    """
    Generate a set of probabilities for a sorted list of elements such that for
    all elements being considered, each is less likely to be chosen than that
    which preceeds it. The probabilities will cover the full set of elements,
    and the probabilities decrease linearly in an arithmetic sequence.

    Parameters
    ----------
    num_elements : integer
        The number of elements in the list we are considering.

    Returns
    -------
    probabilities : float
        List of probabilities for each element in the list we are considering.

    ---------------------------------------------------------------------------
    Paul Sharp 27/03/2020
    """

    assert num_elements > 0, 'ERROR in swap.generate_linearly_decreasing_weightings() -- number of elements is {0:d}, but should be greater than 0.'.format(num_elements)

    # Probabilities are (num_elements-i); i=0, num_elements-1
    probabilities = []

    for i in range(num_elements, 0, -1):
        probabilities.append(float(i))

    return probabilities


# =============================================================================
def generate_geometrically_decreasing_weightings(num_elements):
    """
    Generate a set of probabilities for a sorted list of elements such that for
    all elements being considered, each is less likely to be chosen than that
    which preceeds it. The probabilities will cover the full set of elements,
    and the probabilities decrease linearly.

    Parameters
    ----------
    num_elements : integer
        The number of elements in the list we are considering.

    Returns
    -------
    probabilities : float
        List of probabilities for each element in the list we are considering.

    ---------------------------------------------------------------------------
    Paul Sharp 27/03/2020
    """

    assert num_elements > 0, 'ERROR in swap.generate_geometrically_decreasing_weightings() -- number of elements is {0:d}, but should be greater than 0.'.format(num_elements)

    # Probabilities are 2**(num_elements-i); i=0, num_elements-1
    probabilities = []

    for i in range(num_elements - 1, -1, -1):
        probabilities.append(2**float(i))

    return probabilities


# =============================================================================
def generate_uniform_weightings(num_elements):
    """
    Generate a set of probabilities for a sorted list of elements such that for
    all elements being considered, each is equally likely to be chosen.

    Parameters
    ----------
    num_elements : integer
        The number of elements in the list we are considering.

    Returns
    -------
    probabilities : float
        List of probabilities for each element in the list we are considering.

    ---------------------------------------------------------------------------
    Paul Sharp 27/03/2020
    """

    assert num_elements > 0, 'ERROR in swap.generate_uniform_weightings() -- number of elements is {0:d}, but should be greater than 0.'.format(num_elements)

    probabilities = [1.0] * num_elements

    return probabilities


# =============================================================================
def pair_pinned_weightings(num_elements, pinned_weighting):
    """
    Generate a set of probabilities for a sorted list of elements such that for
    all elements being considered, each is less likely to be chosen than that
    which preceeds it. The probabilities will cover the full set of elements,
    and the probabilities decrease linearly in an arithmetic sequence.

    HOWEVER, per CC's wishes, the probability of choosing two atoms is pinned
    to the value given by the pinned_weighting input parameter.

    "When all he needed to say was Our Game." -- John Le Carre

    Parameters
    ----------
    num_elements : integer
        The number of elements in the list we are considering.
    pinned_weighting : integer
        The weighting for swapping two atoms compared to any other number,
        i.e, 3 for 75% (3:1), 2 for 66.666% (2:1) and 1 for 50% (1:1).

    Returns
    -------
    probabilities : float
        List of probabilities for each element in the list we are considering.

    ---------------------------------------------------------------------------
    Paul Sharp 27/03/2020
    """

    assert num_elements > 0, 'ERROR in swap.pair_pinned_weightings() -- number of elements is {0:d}, but should be greater than 0.'.format(num_elements)

    # Probabilities are (num_elements-i); i=0, num_elements-1
    probabilities = []

    for i in range(num_elements - 1, 0, -1):
        probabilities.append(float(i))

    sum_prob = sum(probabilities)

    if len(probabilities) > 0:
        probabilities.insert(0, sum_prob * pinned_weighting)
    else:
        probabilities = [1.0]

    return probabilities


# =============================================================================
def swap_atoms(structure, swap_list, ranking_dict, directed_num_atoms,
               predefined_vacancies, vacancy_exclusion_radius, rng, force_vacancy_swaps):
    """
    Swap the positions of multiple atoms/vacancies in the structure.

    We select the atoms in the structure either randomly or using the directed
    swapping method and make a random permutation of their positions such that
    all atoms are swapped non-trivially.

    Parameters
    ----------
    structure : ase atoms
        The current structure being considered.
    swap_list : list of strings
        List of the species of the atoms involved in the swap.
    ranking_dict : dict
        Ranking of the atoms for each species in the structure.
    directed_num_atoms : int
        Number of extra atoms available from the top of the ranking list to choose between.
    predefined_vacancies : boolean
        True if we have started from a predefined structure, with vacancy sites specified.
    vacancy_exclusion_radius : float
        The maximum distance allowed between an atom and a vacancy.
    rng : NR_ran
        Random number generator - algorithm from Numerical Recipes 2007.
    force_vacancy_swaps : boolean
        If True, we insist that vacancies must swap non-trivially.

    Returns
    -------
    structure : ase atoms
        The structure after swapping the positions of the atoms.
    swap_text : [string, float, string]
        The old atom, position and new atom, used for reporting full details of the swap.

    ---------------------------------------------------------------------------
    Paul Sharp 11/04/2019
    """

    # The list of atoms we will swap must be sorted to ensure that atoms of the
    # same species are alongside one another.
    # This is important when we look through the new arrangement for each
    # occurance of a particular element later.
    swap_list.sort()

    new_order = permute_atoms(swap_list, rng, force_vacancy_swaps)

    if all(species in ranking_dict for species in swap_list):

        swap_indices, swap_list, new_order = select_atoms_directed(swap_list,
                                                                   new_order,
                                                                   structure.copy(),
                                                                   ranking_dict,
                                                                   directed_num_atoms,
                                                                   predefined_vacancies,
                                                                   vacancy_exclusion_radius,
                                                                   rng)

    else:

        swap_indices, swap_list, new_order = select_atoms_at_random(swap_list,
                                                                    new_order,
                                                                    structure.copy(),
                                                                    predefined_vacancies,
                                                                    vacancy_exclusion_radius,
                                                                    rng)

    # Ensure that the swap list and swap indices are sorted according
    # to the species of atoms.
    # NOTE -- this routine is mainly necessary for the "atoms-vacancies"
    # swap group, where the species of atoms is only determined once the
    # indices have been selected.
    swap_list, swap_indices = sort_swap_list(swap_list, swap_indices)

    # Find the positions of the atoms we will swap
    swap_positions = get_swap_positions(structure, swap_indices)

    # Write swap text before applying swap
    swap_text = []
    scaled_pos = structure.get_scaled_positions()

    for i in range(0, len(swap_list)):
        swap_text.append([swap_list[i], scaled_pos[swap_indices[i]],
                          swap_indices[i], new_order[i]])

    # Re-order the positions array according to how we have permuted the atoms.
    swapped_elements = np.unique(swap_list)
    swap_positions = reorder_positions(swap_positions, np.asarray(new_order),
                                       swapped_elements)

    # Set new positions in structure
    for i in range(0, len(swap_list)):
        structure[swap_indices[i]].position = swap_positions[i]

    return structure, swap_text


# =============================================================================
def generate_selection_pool(elements_list, num_swaps):
    """
    Create the list of atoms from which we will choose which ones to swap.
    We ensure that a non-trivial swap is possible, reducing the number of atoms
    to swap if necessary.

    Parameters
    ----------
    elements_list : list of strings
        Symbols of atoms/vacancies we wish to be considered for swapping.
    num_swaps : integer
        Number of atoms/vacancies we wish to swap.

    Returns
    -------
    selection_pool : string
        The list of atoms from which we will choose which to swap.
    num_swaps : integer
        Number of atoms/vacancies we will swap -- such that a non-trival swap is possible.

    ---------------------------------------------------------------------------
    Paul Sharp 27/03/2020
    """

    assert num_swaps > 1, 'ERROR in swap.generate_selection_pool() -- num_swaps should be greater than 1, but is: {0:d}'.format(num_swaps)

    element_counter = collections.Counter(elements_list)
    unique_elements = list(set(elements_list))

    while num_swaps > 1:

        selection_pool = []

        for current_element in unique_elements:

            # The number of atoms of each element cannot exceed half of the total number of atoms to swap
            num_entries = min(int(0.5 * num_swaps), element_counter[current_element])
            selection_pool.extend(num_entries * [current_element])

        if len(selection_pool) >= num_swaps:

            break

        else:

            # Reject selection pool if it does not contain enough atoms -- this means a non-trivial swap is not possible.
            # Reduce number of atoms to swap by one and try again.
            num_swaps -= 1

    return selection_pool, num_swaps


# =============================================================================
def generate_swap_list(selection_pool, num_swaps, rng):
    """
    Choose the set of atoms that will be swapped from the selection pool.

    Parameters
    ----------
    selection_pool : string
        The list of atoms from which we will choose which to swap.
    num_swaps : integer
        Number of atoms/vacancies we will swap.
    rng : NR_ran
        Random number generator - algorithm from Numerical Recipes 2007.

    Returns
    -------
    swap_list : string
        List of atoms that we will swap.

    ---------------------------------------------------------------------------
    Paul Sharp 27/03/2020
    """

    assert num_swaps <= len(selection_pool), 'ERROR in swap.generate_swap_list() -- num_swaps should not be greater than the length of the selection_pool, but we are trying to swap {0:d} atoms from: {1}'.format(num_swaps, selection_pool)

    swap_list = []
    len_pool = len(selection_pool)

    for i in range(0, num_swaps):

        random_index = rng.int_range(u_lim=len_pool - i)
        swap_list.append(selection_pool.pop(random_index))

    return swap_list


# =============================================================================
def select_atoms_at_random(swap_list, new_order, structure, predefined_vacancies,
                           vacancy_exclusion_radius, rng):
    """
    Select the atoms that we will swap in the structure at random.

    Parameters
    ----------
    swap_list : list of strings
        List of the species of the atoms involved in the swap.
    new_order : string
        Rearranged list of atoms that we will swap.
    structure : ase atoms
        The full structure of interest.
    predefined_vacancies : boolean
        True if we have started from a predefined structure, with vacancy sites specified.
    vacancy_exclusion_radius : float
        The maximum distance allowed between an atom and a vacancy.
    rng : NR_ran
        Random number generator - algorithm from Numerical Recipes 2007

    Returns
    -------
    swap_indices : integer
        The indices of the atoms we will swap in the atoms object.
    swap_list : list of strings
        List of the species of the atoms involved in the swap, with generic atoms
        updated with actual species.
    new_order : string
        Rearranged list of atoms that we will swap, includung the species of atoms.

    ---------------------------------------------------------------------------
    Paul Sharp 27/03/2020
    """

    all_species = structure.get_chemical_symbols()
    num_swaps = len(swap_list)
    
    swap_indices = [""] * num_swaps

    species_dict = {}
    vacancies_to_swap = swap_list.count("X")

    atom_indices = [i for i, species in enumerate(swap_list) if species != "X"]
    vacancy_indices = [i for i, species in enumerate(swap_list) if species == "X"]

    species_list = list(set(all_species))
    
    # Set up species dictionary that records the indices of all atoms of a particular species
    species_dict["A"] = [atom.index for atom in structure if atom.symbol != "X"]
    
    for species in species_list:
        
        species_dict[species] = [atom.index for atom in structure if atom.symbol == species]

    # Choose atoms at random
    for swap_list_index in atom_indices:

        atom_to_swap = swap_list[swap_list_index]
        remaining_atoms_of_this_species = len(species_dict[atom_to_swap])
        atom_index = species_dict[atom_to_swap].pop(rng.int_range(0, remaining_atoms_of_this_species))

        # Record the index of this atom in the Atoms object
        swap_indices[swap_list_index] = atom_index

        # If we have "A"s in the swap list and new order list,
        # replace them with the actual species of the atoms chosen.
        if "A" in swap_list:
            replace_index = swap_list.index("A")
            swap_list[replace_index] = all_species[atom_index]
            
        if "A" in new_order:
            replace_index = new_order.index("A")
            new_order[replace_index] = all_species[atom_index]

    # Choose vacancies at random
    for swap_list_index in vacancy_indices:

        remaining_vacancies = len(species_dict["X"])     
        vacancy_index = species_dict["X"].pop(rng.int_range(0, remaining_vacancies))
        vacancies_to_swap -= 1

        # If we choose a vacancy, remove all vacancies that are too close to the selected vacancy
        if not predefined_vacancies:
               
            vacancies_to_remove = find_vacancies_near_a_selected_vacancy(structure,
                                                                         vacancy_index,
                                                                         species_dict["X"],
                                                                         vacancy_exclusion_radius)

            # Only remove vacancies if there will be enough remaining to fulfil the swap list
            if len(vacancies_to_remove) <= len(species_dict["X"]) - vacancies_to_swap:
                    
                for vacancy in vacancies_to_remove:
                    
                    species_dict["X"].remove(vacancy)
                                                
            else:

                print("WARNING -- we have tried to remove {0:d} vacancies, but there are only {1:d} vacancies on the grid (excluding those we need to swap)".format(len(vacancies_to_remove), len(species_dict["X"]) - vacancies_to_swap))
            
        # Record the index of this vacancy in the Atoms object
        swap_indices[swap_list_index] = vacancy_index

    return swap_indices, swap_list, new_order


# =============================================================================
def select_atoms_directed(swap_list, new_order, structure, ranking_dict,
                          directed_num_atoms, predefined_vacancies,
                          vacancy_exclusion_radius, rng):
    """
    Select the atoms that we will swap in the structure.

    Parameters
    ----------
    swap_list : list of strings
        List of the species of the atoms involved in the swap.
    new_order : string
        Rearranged list of atoms that we will swap.
    structure : ase atoms
        The full structure of interest.
    ranking_dict : dict
        Ranking of the atoms for each species in the structure.
    directed_num_atoms : int
        Number of extra atoms available from the top of the ranking list to choose from.
    predefined_vacancies : boolean
        True if we have started from a predefined structure, with vacancy sites specified.
    vacancy_exclusion_radius : float
        The maximum distance allowed between an atom and a vacancy.
    rng : NR_ran
        Random number generator - algorithm from Numerical Recipes 2007

    Returns
    -------
    swap_indices : integer
        The indices of the atoms we will swap in the atoms object.
    swap_list : list of strings
        List of the species of the atoms involved in the swap, with generic atoms
        updated with actual species.
    new_order : string
        Rearranged list of atoms that we will swap, includung the species of atoms.

    ---------------------------------------------------------------------------
    Paul Sharp 27/03/2020
    """
    
    all_species = structure.get_chemical_symbols()
    num_swaps = len(swap_list)

    swap_indices = [""] * num_swaps

    num_atoms_dict = {}
    vacancies_to_swap = swap_list.count("X")

    atom_indices = [i for i, species in enumerate(swap_list) if species != "X"]
    vacancy_indices = [i for i, species in enumerate(swap_list) if species == "X"]

    species_list = list(set(all_species))
    
    try:
        species_list.remove("X")
    except ValueError:
        pass
        
    # Set up how many atoms we can choose between from the top of the rankings list for each species.
    # This is determined by the number of that species in the swap list and the number of extra atoms
    # allowed by the minima hopping algorithm.

    # Determine the number of atoms in both the swap list and structure
    num_to_swap = len([y for y in swap_list if y != "X"])
    num_in_structure = len(all_species) - all_species.count("X")

    num_atoms_dict["A"] = min(num_to_swap + directed_num_atoms, num_in_structure)

    for species in species_list:

        # Determine the number of atoms of this species in both the swap list and structure
        num_to_swap = len([x for x in swap_list if x == species])
        num_in_structure = all_species.count(species)

        num_atoms_dict[species] = min(num_to_swap + directed_num_atoms, num_in_structure)

    # Choose atoms from the rankings, using the minima hopping method
    for swap_list_index in atom_indices:

        atom_to_swap = swap_list[swap_list_index]
        atom_index = ranking_dict[atom_to_swap].pop(rng.int_range(-num_atoms_dict[atom_to_swap], 0))
        num_atoms_dict[atom_to_swap] -= 1
 
        # Record the index of this atom in the Atoms object
        swap_indices[swap_list_index] = atom_index
        
        # If we have "A"s in the swap list and new order list,
        # replace them with the actual species of the atoms chosen.
        if "A" in swap_list:
            replace_index = swap_list.index("A")
            swap_list[replace_index] = all_species[atom_index]

        if "A" in new_order:
            replace_index = new_order.index("A")
            new_order[replace_index] = all_species[atom_index]

    replace_list = [old + "--" + new for old, new in zip(swap_list, new_order)]
    
    # Choose vacancies from the rankings, taking the top vacancy on the appropriate list each time.
    # NOTE that we do not use the minima hopping method here because of complications when
    # removing nearby vacancies.
    for swap_list_index in vacancy_indices:

        swap_string = replace_list[swap_list_index]
        vacancy_index = ranking_dict["X"][swap_string][-1]
        vacancies_to_swap -= 1

        # We need to remove this vacancy from consideration in all vacancy lists.
        for vacancy_key in ranking_dict["X"]:
            ranking_dict["X"][vacancy_key].remove(vacancy_index)

        # Now remove all vacancies that are too close to the selected vacancy
        if not predefined_vacancies:

            vacancies_to_remove = find_vacancies_near_a_selected_vacancy(structure, vacancy_index,
                                                                         ranking_dict["X"][swap_string],
                                                                         vacancy_exclusion_radius)

            # Only remove vacancies if there will be enough remaining to fulfil the swap list
            if len(vacancies_to_remove) <= len(ranking_dict["X"][swap_string]) - vacancies_to_swap:

                for vacancy_key in ranking_dict["X"]:
                    for vacancy in vacancies_to_remove:

                        ranking_dict["X"][vacancy_key].remove(vacancy)

            else:

                print("WARNING -- we have tried to remove {0:d} vacancies, but there are only {1:d} vacancies on the grid (excluding those we need to swap)".format(len(vacancies_to_remove), len(ranking_dict["X"][swap_string]) - vacancies_to_swap))
        
        # Record the index of this vacancy in the Atoms object
        swap_indices[swap_list_index] = vacancy_index

    return swap_indices, swap_list, new_order


# =============================================================================
def find_vacancies_near_a_selected_vacancy(structure, selected_vacancy_index,
                                           vacancy_indices, vacancy_exclusion_radius):
    """
    Remove vacancies that are too close to a vacancy chosen to be occupied by
    an atom in a structure.

    Parameters
    ----------
    structure : ase atoms
        The full structure within which we are swapping atoms.
    selected_vacancy_index : int
        The index of the vacancy that will be occupied by an atom.
    vacancy_indices : int
        The indices of the other vacancies in the structure.
    vacancy_exclusion_radius : float
        The maximum distance allowed between an atom and a vacancy.

    Returns
    -------
    vacancies_to_remove : int
        A list of the indices of vacancies too close to the selected vacancy.

    ---------------------------------------------------------------------------
    Paul Sharp 30/01/2019
    """

    vacancies_to_remove = []

    try:
        vacancy_indices.remove(selected_vacancy_index)
    except ValueError:
        pass

    if len(vacancy_indices) > 0:
        atom_vacancy_distances = structure.get_distances(selected_vacancy_index,
                                                         vacancy_indices, mic=True)

        # Record all vacancies that are too close to the selected vacancy
        for index, vac in enumerate(vacancy_indices):

            if atom_vacancy_distances[index] < vacancy_exclusion_radius:

                vacancies_to_remove.append(vac)

    return vacancies_to_remove


# =============================================================================
def sort_swap_list(swap_list, swap_indices):
    """
    Sort the swap list, ensuring that the list of atom indices are sorted
    according to the same key.

    Parameters
    ----------
    swap_list : list of strings
        List of the species of the atoms involved in the swap.
    swap_indices : integer
        The indices of the atoms we will swap in the atoms object.

    Returns
    -------
    swap_list : list of strings
        Sorted list of the species of the atoms involved in the swap.
    swap_indices : integer
        The indices of the atoms we will swap in the atoms object,
        sorted according to their species.

    ---------------------------------------------------------------------------
    Paul Sharp 09/12/2019
    """

    combined_list = list(zip(swap_list, swap_indices))
    combined_list.sort(key=lambda x: x[0])

    swap_list = [x[0] for x in combined_list]
    swap_indices = [x[1] for x in combined_list]

    return swap_list, swap_indices


# =============================================================================
def get_swap_positions(structure, swap_indices):
    """
    Find the positions of the atoms we are going to swap.

    Parameters
    ----------
    structure : ASE atoms
        The structure containing the atoms we are swapping.
    swap_indices : integer
        The indices of the atoms we will swap in the atoms object.

    Returns
    -------
    swap_positions : float
        The positions of the atoms we will swap in the atoms object.

    ---------------------------------------------------------------------------
    Paul Sharp 11/04/2019
    """

    num_swaps = len(swap_indices)
    swap_positions = [""] * num_swaps
    
    all_positions = structure.get_positions()
    
    for i in range(num_swaps):
        swap_positions[i] = all_positions[swap_indices[i]]

    return np.asarray(swap_positions)


# =============================================================================
def permute_atoms(old_order, rng, force_vacancy_swaps):
    """
    Make a random permutation of the list of atoms we are swapping, to give the
    new order of the atoms.

    We permute the atoms on an element-by-element basis, in order to ensure
    that all atoms are swapped non-trivially.

    Parameters
    ----------
    old_order : string
        List of atoms that we will swap.
    rng : NR_ran
        Random number generator - algorithm from Numerical Recipes 2007.
    force_vacancy_swaps : boolean
        If True, we insist that vacancies must swap non-trivially.

    Returns
    -------
    new_order : string
        Rearranged list of atoms that we will swap.

    ---------------------------------------------------------------------------
    Paul Sharp 06/02/2019
    """

    num_atoms = len(old_order)
    new_order = [""] * num_atoms

    # Need to order elements based on number -- we should tackle the elements that occur most first
    element_counter = collections.Counter(old_order)
    element_order = [element for elements, count in element_counter.most_common() for element in [elements]]

    for element in element_order:

        # Find available sites in new order
        available_sites = [index for index in range(len(new_order)) if new_order[index] == "" and old_order[index] != element]

        num_element = element_counter[element]

        if len(available_sites) < num_element or (element == "X" and not force_vacancy_swaps):

            available_sites = [index for index in range(len(new_order)) if new_order[index] == ""]

        # Populate atoms
        for i in range(0, num_element):

            index = available_sites.pop(rng.int_range(0, len(available_sites)))

            new_order[index] = element

    # Deal with any stray atoms
    if not force_vacancy_swaps:
        old_order = ["Z" if element == "X" else element for element in old_order]

    if any([new_order[i] == old_order[i] for i in range(num_atoms)]):

        unswapped_atoms = [i for i in range(num_atoms) if new_order[i] == old_order[i]]

        for atom in unswapped_atoms:

            sites_to_swap_to = [index for index in range(num_atoms) if new_order[index] != new_order[atom] and old_order[index] != new_order[atom]]

            swap_atom = sites_to_swap_to[rng.int_range(0, len(sites_to_swap_to))]

            new_order[atom], new_order[swap_atom] = new_order[swap_atom], new_order[atom]

    old_order = ["X" if element == "Z" else element for element in old_order]

    return new_order


# =============================================================================
def reorder_positions(swap_positions, new_order, swapped_elements):
    """
    Change the order of the array of the positions of the atoms we are swapping
    to reflect the new order of the atoms.

    Parameters
    ----------
    swap_positions : float
        The positions of the atoms we will swap -- ordered according to the current order of atoms.
    new_order : string
        The rearranged list of the chemical symbols of the atoms we are swapping.
    swapped_elements : string
        The atomic species of the atoms we are swapping.

    Returns
    -------
    swap_positions : float
        The positions of the atoms we will swap -- ordered according to the new order of atoms.

    ---------------------------------------------------------------------------
    Paul Sharp 22/05/2017
    """

    positions_index_array = []
    for element in swapped_elements:
        positions_index_array.extend(np.where(new_order == element)[0])
        
    positions_index_array = np.asarray(positions_index_array)
    swap_positions = swap_positions[positions_index_array]

    return swap_positions


# =============================================================================
def accept_swap(current_energy, new_energy, temperature, rng):
    """
    Determines whether or not to accept a proposed new structure using the
    energies of the relaxed structures.

    The Metropolis algorithm is used, so we accept all moves that reduce the
    energy, and accept moves that raise the energy with probability exp(-dE/T).

    The use of the relaxed energies means that we are using a basin-hopping
    procedure.

    Parameters
    ----------
    current_energy, new_energy : float
        The relaxed energies of the current structure being considered,
        and the proposed new structure.
    temperature : float
        The Monte Carlo temperature.
    rng : NR_ran
        Random number generator - algorithm from Numerical Recipes 2007.

    Returns
    -------
    accept : boolean
        Logical to determine whether the swap is accepted or not.

    ---------------------------------------------------------------------------
    Paul Sharp 09/12/2019
    """

    # Evaluate difference in energy - if energy of new structure is not
    # representable as a float (e.g. due to underflow), automatically reject it
    try:
        delta_E = new_energy - current_energy
    except TypeError:
        return False

    # Evaluate the Boltzman factor for the current structures
    # If temperature is zero, set factor to +/-1 depending on delta_E
    try:
        boltzman = np.exp(-delta_E / temperature)
    except ZeroDivisionError:
        boltzman = np.sign(-delta_E)
    except RuntimeWarning:
        boltzman = np.sign(-delta_E)
        print("Runtime Warning")

    acceptance_threshold = rng.real()
    accept = acceptance_threshold < boltzman

    return accept


# =============================================================================
def reduce_structure(structure, atom_filter):
    """
    Removes all atoms of the structure that do not correspond to the chosen
    filter.

    Parameters
    ----------
    structure : ase_atoms
        The structure being considered.
    atom_filter : string
        The filter used to determine which atoms to preserve.

    Returns
    -------
    structure : ase_atoms
        The reduced structure containing only the desired atoms.

    ---------------------------------------------------------------------------
    Paul Sharp 06/02/2019
    """

    if atom_filter == "cations":

        del structure[[atom.index for atom in structure if atom.symbol == "X"]]
        del structure[[atom.index for atom in structure if atom.charge <= 0.0]]

    elif atom_filter == "anions":

        del structure[[atom.index for atom in structure if atom.symbol == "X"]]
        del structure[[atom.index for atom in structure if atom.charge >= 0.0]]

    elif atom_filter == "atoms":

        del structure[[atom.index for atom in structure if atom.symbol == "X"]]

    elif atom_filter == "vacancies":

        del structure[[atom.index for atom in structure if atom.symbol != "X"]]

    elif atom_filter == "all" or atom_filter == "atoms-vacancies":

        pass

    else:

        # This is for custom filters that specify particular elements
        atom_list = atom_filter.split("-")
        del structure[[atom.index for atom in structure if atom.symbol not in atom_list]]

    return structure


# =============================================================================
def update_basins(basins, new_energy, energy_dp=5):
    """
    Manages a dictionary of explored basins, by iterating the number of repeat
    visits of basins, and adding the energies of newly-explored basins.

    This routine is called after a structure has been accepted in the code.

    Parameters
    ----------
    basins : dict
        The dictionary of basins explored, and how many times each basin has
        been visited.
    new_energy : float
        The energy of the basin being considered.
    energy_dp : int, optional
        The number of decimal places of tolerance of the agreement between
        energies. Default is 5 (from studies of STO).

    Returns
    -------
    basins : dict
        The updated dictionary of basins explored, and how many times each
        basin has been visited.

    ---------------------------------------------------------------------------
    Paul Sharp 04/08/2017
    """

    # If basin has been visited, increment the visit number, else the add new basin
    energy_key = round(new_energy, energy_dp)

    if energy_key in basins:

        basins[energy_key] += 1

    else:

        basins[energy_key] = 1

    return basins


# =============================================================================
def check_previous_structures(new_structure, positions, atomic_numbers,
                              pos_tol=1.0E-12):
    """
    Check whether or not a structure has been considered earlier in the code.
    Repeat structures will be rejected by the code without being optimised
    again. Structures are identified by the number of atoms of each species,
    and the positions of each of these atoms.

    Part of this routine is based on the routine "check_identical_atoms()" in
    SimDope, written by Matthew Dyer.

    Parameters
    ----------
    new_structure : ase_atoms
        The structure currently being considered.
    positions : ndarray
        A list of the absolute atomic positions of all structures considered so far,
        sorted by atomic position.
    atomic_numbers : ndarray
        A list of the atomic numbers for all atoms in all of the structures
        considered so far, sorted by atomic position.
    pos_tol : float, optional
        Tolerance in the difference between atomic positions. Default is 1.0e-12.

    Returns
    -------
    visited_structure : boolean
        Flag to determine whether or not this structure has been considered previously.
        If True, we do not optimise the structure.
    positions : ndarray
        A list of the absolute atomic positions of all structures considered so far,
        sorted by atomic position and updated with new structure if necessary.
    atomic_numbers : ndarray
        A list of the atomic numbers for all atoms in all of the structures considered so far,
        sorted by atomic position and updated with new structure if necessary.

    ---------------------------------------------------------------------------
    Paul Sharp 29/08/2018
    """

    # Remove vacancies from structure
    del new_structure[[atom.index for atom in new_structure if atom.symbol == "X"]]

    natoms = len(new_structure)
    visited_structure = False

    # Get atomic numbers and positions of new structure considered so far,
    # and sort according to position. We do this now so each structure considered
    # need only be sorted once, removing a significant bottleneck.

    # We use absolute positions, to avoid the theoretical possibility of atoms in
    # the same scaled positions but different cells.

    # Get indices to sort them according to z, y and x values
    sort_indices = np.lexsort(new_structure.get_positions().T)

    new_positions = new_structure.get_positions()[sort_indices]
    new_atomic_numbers = new_structure.get_atomic_numbers()[sort_indices]

    # Check against previous structures
    for i in range(0, len(positions)):

        # Check for correct number of atoms
        if len(positions[i]) != natoms:
            continue

        # Check species and then positions
        if (new_atomic_numbers == atomic_numbers[i]).all():
            # Calculate difference in sorted positions
            difference = np.abs(new_positions - positions[i])
            visited_structure = (difference < pos_tol).all()

        if visited_structure:
            break

    else:

        # If no matching structure is found, this is a new structure, so we add it to our lists
        atomic_numbers.append(new_atomic_numbers)
        positions.append(new_positions)

    return visited_structure, positions, atomic_numbers


# =============================================================================
def initialise_default_swap_groups(structure, swap_groups):
    """
    Checks which of the possible swap groups are valid for the structure we are
    considering, and returns all valid groups.

    Parameters
    ----------
    structure : ase_atoms
        The initial structure for this run of the code.
    swap_groups : string
        The default set of swap groups -- all possible swaps

    Returns
    -------
    swap_groups : string
        List of swap groups that will allow for non-trivial swaps of atoms in
        the structure.
    swap_weightings : float
        Set of weightings that gives every valid swap group equal probability.

    ---------------------------------------------------------------------------
    Paul Sharp 06/11/2017
    """

    # Check for invalid groups, and remove them
    for group in swap_groups[:]:

        reduced_structure = reduce_structure(structure.copy(), group[0])
        elements_list = list(set(reduced_structure.get_chemical_symbols()))

        if len(elements_list) < 2:
            swap_groups.remove(group)

    return swap_groups


# =============================================================================
def initialise_default_swap_weightings(swap_groups):
    """
    Takes the set of swap groups we are considering and assigns each an equal
    weighting.

    Parameters
    ----------
    swap_groups : string
        The set of swap groups

    Returns
    -------
    swap_groups : float
        The set of swap groups, each with a weighting that gives every swap
        group equal probability.

    ---------------------------------------------------------------------------
    Paul Sharp 07/11/2017
    """

    default_weighting = 1.0

    for group in swap_groups:

        group.append(default_weighting)

    return swap_groups


# =============================================================================
def verify_swap_groups(structure, swap_groups):
    """
    Checks that the swap groups specified for this run of the code will allow
    for non-trival swaps of the specific group of atoms. If not, a list of
    errors is returned and execution terminated.

    Parameters
    ----------
    structure : ase_atoms
        The initial structure for this run of the code.
    swap_groups : string
        The swap groups specified for this run of the code.

    Returns
    -------
    swap_group_errors : string
        List of swap groups that will not allow for non-trivial swaps of atoms
        in the structure.

    ---------------------------------------------------------------------------
    Paul Sharp 27/03/2020
    """

    valid_groups = []
    swap_group_errors = []
    swap_group_names = [group[0] for group in swap_groups]

    for i, group in enumerate(swap_group_names):

        reduced_structure = reduce_structure(structure.copy(), group)
        elements_list = reduced_structure.get_chemical_symbols()

        # For the "atoms-vacancies" swap group, change atom symbols to a generic atom "A".
        if group == "atoms-vacancies":
            elements_list = ["A" if element != "X" else element for element in elements_list]

        elements_list = list(set(elements_list))
        num_elements = len(elements_list)

        if num_elements >= 2:
            valid_groups.append(swap_groups[i])
        else:
            swap_group_errors.append('"{0}" has been specified as a swap group, but there are insufficient different species to enable non-trivial swaps to be made -- there are {1:d} species of {2}.'.format(group, num_elements, group))

    if len(valid_groups) == 0:
        swap_group_errors.append('There are no valid swap groups.')
        
    return valid_groups, swap_group_errors


# =============================================================================
def check_elements_in_custom_swap_groups(swap_groups, all_atoms):
    """
    Checks that the swap groups specified for this run of the code will allow
    for non-trival swaps of the specific group of atoms. If not, a list of
    errors is returned and execution terminated.

    Parameters
    ----------
    swap_groups : string
        The swap groups specified for this run of the code.
    all_atoms : string
        A list of chemical symbols of all atoms in the structure.

    Returns
    -------
    swap_group_errors : string
        List of swap groups that contain atoms that are not in the structure.

    ---------------------------------------------------------------------------
    Paul Sharp 27/03/20120
    """

    swap_group_errors = []
    main_swap_groups = ["cations", "anions", "atoms", "all", "atoms-vacancies"]

    swap_group_names = [group[0] for group in swap_groups]

    for group in swap_group_names:

        if group in main_swap_groups:
            continue

        # Now considering only custom swap groups
        swap_group_elements = group.split("-")

        if not set(swap_group_elements).issubset(all_atoms):
            swap_group_errors.append('"{0}" has been specified as a swap group, but there are no {1} atoms in the structure.'.format(group, ', '.join([str(x) for x in [y for y in swap_group_elements if y not in all_atoms]])))

    return swap_group_errors


# =============================================================================
def update_atom_rankings(structure, ranking_measure):
    """
    Produce a new set of swap rankings for a newly accepted structure according
    to the desired measure (e.g., Bond Valence Sum, site potential).

    Parameters
    ----------
    structure : ChemDASH Structure
        The structure, including BVS and site potential values, that we are
        considering.
    ranking_measure : string
        The list of values used to rank the atoms.

    Returns
    -------
    structure : ChemDASH Structure
        The structure, including rankinhgs for atoms according to their BVS
        and/or site potential values.

    ---------------------------------------------------------------------------
    Paul Sharp 10/04/2019
    """

    unique_elements = list(set(structure.atoms.get_chemical_symbols()))
    try:
        unique_elements.remove("X")
    except ValueError:
        pass

    if ranking_measure == "bvs":

        structure = rank_bvs(structure, unique_elements)

    elif ranking_measure == "bvs+":

        structure = rank_bvs_plus(structure, unique_elements)
                        
    elif ranking_measure == "site_pot":

        structure = rank_site_potential(structure, unique_elements)

    elif ranking_measure == "random":

        structure.ranked_atoms = {}

    return structure


# =============================================================================
def rank_bvs(structure, unique_elements):
    """
    Rank the atoms in a structure according to the values of the Bond Valence Sum

    Parameters
    ----------
    structure : ChemDASH Structure
        The structure in which we are ranking atoms according to their BVS values.
    unique_elements : string
        The atomic species (and vacancies) in the structure.

    Returns
    -------
    structure : ChemDASH Structure
        The structure with atoms ranked according to their BVS values.

    ---------------------------------------------------------------------------
    Paul Sharp 07/08/2019
    """

    structure.ranked_atoms = {}
    charges = structure.atoms.get_initial_charges()

    # We treat atoms and vacancies separately, because atoms has a BVS defined,
    # and we rank vacancies depending on the atom that occupies them.
    # Hence, vacancies have a dictionary of values, rather than a single value

    # Atoms -- all atoms, then each element
    atom_indices = [atom.index for atom in structure.atoms if atom.symbol != "X"]
    atom_symbols = [atom.symbol for atom in structure.atoms if atom.symbol != "X"]
    bvs_rank = [abs(abs(charges[index]) - structure.bvs_atoms[index]) for index in atom_indices]
    structure.ranked_atoms["A"] = rank_indices(bvs_rank, atom_indices)

    for element in unique_elements:

        element_indices = [atom.index for atom in structure.atoms if atom.symbol == element]
        bvs_rank = [abs(abs(charges[index]) - structure.bvs_atoms[index]) for index in element_indices]
        structure.ranked_atoms[element] = rank_indices(bvs_rank, element_indices)

    # Vacancies
    vacancy_indices = [atom.index for atom in structure.atoms if atom.symbol == "X"]

    if len(vacancy_indices) > 0:

        structure.ranked_atoms["X"] = {}

        for atom_key in structure.bvs_sites[vacancy_indices[0]]:

            if all([atom_key in structure.bvs_sites[index] for index in vacancy_indices]):

                charge = float(atom_key.split('/')[1])
                # The factor of -1 is here because we want the best vacancies for the appropriate element.
                bvs_rank = [(-1.0 * abs(abs(charge) - structure.bvs_sites[index][atom_key])) for index in vacancy_indices]
                structure.ranked_atoms["X"]["X--" + atom_key.split('/')[0]] = rank_indices(bvs_rank, vacancy_indices)                 

    return structure


# =============================================================================
def rank_site_potential(structure, unique_elements):
    """
    Rank the atoms in a structure according to the values of the site potential

    Parameters
    ----------
    structure : ChemDASH Structure
        The structure in which we are ranking atoms according to their site
        potential values.
    unique_elements : string
        The atomic species (and vacancies) in the structure.

    Returns
    -------
    structure : ChemDASH Structure
        The structure with atoms ranked according to their site potential values.

    ---------------------------------------------------------------------------
    Paul Sharp 11/06/2019
    """

    structure.ranked_atoms = {}
    charges = structure.atoms.get_initial_charges()
    
    # We treat atoms and vacancies separately, because we rank vacancies
    # depending on the atom that occupies them.

    # Atoms -- all atoms, then each element
    atom_indices = [atom.index for atom in structure.atoms if atom.symbol != "X"]

    # We divide the potentials by the atomic charge so that we always
    # minimise the ranking for cations and anions, and can compare
    # different species on the same basis.
    pot_rank = [structure.potentials[index] / charges[index] for index in atom_indices]    
    structure.ranked_atoms["A"] = rank_indices(pot_rank, atom_indices)

    for element in unique_elements:

        element_indices = [atom.index for atom in structure.atoms if atom.symbol == element]

        # We divide the potentials by the atomic charge so that we always
        # minimise the ranking for cations and anions.
        pot_rank = [structure.potentials[index] / charges[index] for index in element_indices]

        structure.ranked_atoms[element] = rank_indices(pot_rank, element_indices)

    # Vacancies
    vacancy_indices = [atom.index for atom in structure.atoms if atom.symbol == "X"]

    if len(vacancy_indices) > 0:

        structure.ranked_atoms["X"] = {}

        #Construct keys
        reduced_structure = reduce_structure(structure.atoms.copy(), "atoms")
        atom_symbols = list(set(reduced_structure.get_chemical_symbols()))

        for atom in atom_symbols:

            # We use the modal charge for atoms of that element in the structure
            charge_list = [reduced_structure[i].charge for i in range(len(reduced_structure)) if reduced_structure[i].symbol == atom]
            charge = max(charge_list, key=charge_list.count)

            # The factor of -1 is here because we want the best vacancies for the appropriate element.
            pot_rank = [-1.0 * structure.potentials[index] / charge for index in vacancy_indices]
            structure.ranked_atoms["X"]["X--" + atom] = rank_indices(pot_rank, vacancy_indices)

    return structure


# =============================================================================
def rank_bvs_plus(structure, unique_elements):
    """
    Rank the atoms in a structure according to the values of the Bond Valence Sum,
    with site potentials used to determine cation/anion sites.

    Parameters
    ----------
    structure : ChemDASH Structure
        The structure in which we are ranking atoms according to their BVS values,
        with site potentials used to determine cation/anion sites.
    unique_elements : string
        The atomic species (and vacancies) in the structure.

    Returns
    -------
    structure : ChemDASH Structure
        The structure with atoms ranked according to their BVS values,
        with site potentials used to determine cation/anion sites.

    ---------------------------------------------------------------------------
    Paul Sharp 07/11/2019
    """

    structure.ranked_atoms = {}
    charges = structure.atoms.get_initial_charges()
    
    # We treat atoms and vacancies separately, because atoms has a BVS defined,
    # and we rank vacancies depending on the atom that occupies them, considering
    # cation and anion sites (according to the site potential) separately.
    # Hence, vacancies have a dictionary of values, rather than a single value

    # Atoms -- all atoms, then each element
    atom_indices = [atom.index for atom in structure.atoms if atom.symbol != "X"]
    bvs_rank = [abs(abs(charges[index]) - structure.bvs_atoms[index]) for index in atom_indices]
    structure.ranked_atoms["A"] = rank_indices(bvs_rank, atom_indices)

    for element in unique_elements:

        element_indices = [atom.index for atom in structure.atoms if atom.symbol == element]
        bvs_rank = [abs(abs(charges[index]) - structure.bvs_atoms[index]) for index in element_indices]
        structure.ranked_atoms[element] = rank_indices(bvs_rank, element_indices)

    # Vacancies
    vacancy_indices = [atom.index for atom in structure.atoms if atom.symbol == "X"]

    # Cation vacancies are surrounded by negatively charged anions, so have a negative site potential
    vacancy_cation_indices = [index for index in vacancy_indices if structure.potentials[index] < 0.0]
    vacancy_anion_indices  = [index for index in vacancy_indices if structure.potentials[index] > 0.0]
    
    if len(vacancy_indices) > 0:

        structure.ranked_atoms["X"] = {}
        
        for atom_key in structure.bvs_sites[vacancy_indices[0]]:

            if all([atom_key in structure.bvs_sites[index] for index in vacancy_indices]):

                charge = float(atom_key.split('/')[1])

                if charge > 0.0:

                    # The factor of -1 is here because we want the best vacancies for the appropriate element.
                    # For cations, we start with anion sites, then consider cation sites, since we draw from
                    # the end of the list
                    bvs_rank = [(-1.0 * abs(abs(charge) - structure.bvs_sites[index][atom_key])) for index in vacancy_anion_indices]
                    structure.ranked_atoms["X"]["X--" + atom_key.split('/')[0]] = rank_indices(bvs_rank, vacancy_anion_indices)
                    
                    bvs_rank = [(-1.0 * abs(abs(charge) - structure.bvs_sites[index][atom_key])) for index in vacancy_cation_indices]
                    structure.ranked_atoms["X"]["X--" + atom_key.split('/')[0]] += rank_indices(bvs_rank, vacancy_cation_indices)

                elif charge < 0.0:

                    # The factor of -1 is here because we want the best vacancies for the appropriate element.
                    # For anions, we start with cation sites, then consider anion sites, since we draw from
                    # the end of the list
                    bvs_rank = [(-1.0 * abs(abs(charge) - structure.bvs_sites[index][atom_key])) for index in vacancy_cation_indices]
                    structure.ranked_atoms["X"]["X--" + atom_key.split('/')[0]] = rank_indices(bvs_rank, vacancy_cation_indices)
                    
                    bvs_rank = [(-1.0 * abs(abs(charge) - structure.bvs_sites[index][atom_key])) for index in vacancy_anion_indices]
                    structure.ranked_atoms["X"]["X--" + atom_key.split('/')[0]] += rank_indices(bvs_rank, vacancy_anion_indices)

    return structure


# =============================================================================
def find_desired_atoms(bvs_sites, atom_indices):
    """
    Produce a new set of swap rankings for a newly accepted structure according
    to the desired measure (e.g., Bond Valence Sum, site potential).

    Parameters
    ----------
    bvs_sites : dict
        The values of the Bond Valence Sum for each site for each atom type.
    atom_indices : int
        The indices of the atoms of interest in the bvs_sites list.

    Returns
    -------
    desired_atoms : dict
        Dictionary of each atom site, with a list of their most to least desired
        atoms.

    ---------------------------------------------------------------------------
    Paul Sharp 28/01/2019
    """

    desired_atoms = {}

    for atom in atom_indices:

        bvs_values = list(bvs_sites[atom].items())

        # Get lists of species and charges from the first and second part of each key
        species = [bvs_values[i][0].split('/')[0] for i in range(len(bvs_values))]
        charges = [float(bvs_values[i][0].split('/')[1]) for i in range(len(bvs_values))]

        # Find how far from the ideal each value is, and rank according to this.
        bvs_rank = [(abs(abs(charges[i]) - bvs_values[i][1])) for i in range(len(bvs_values))]
        desired_atoms[atom] = rank_indices(bvs_rank, species)
                
    return desired_atoms


# =============================================================================
def rank_indices(rankings, indices):
    """
    Order the atomic indices of a structure according to a set of values for
    each atom in the structure, e.g., Bond Valence Sum, site potential.

    We apply our ordering to each atomic species separately. The sorting in
    this routine is done with a Schwarzian Transform based approach.
    (Also known as a DSU - decorate-sort-undecorate approach).

    Parameters
    ----------
    rankings : float
        The list of values for each atom we wish to sort by.
    indices : int
        The positions of these atoms in the structure we are considering.

    Returns
    -------
    sorted_indices : list
        List of indices for the this atomic species, sorted by the values
        in the rankings.

    ---------------------------------------------------------------------------
    Paul Sharp 27/03/2020
    """

    assert len(rankings) == len(indices), 'ERROR in swap.rank_indices() -- there are {0:d} ranking values, and {1:d} indices. They should be the same'.format(len(rankings), len(indices))

    sorted_indices = []
    if len(rankings) > 0:
        sorted_indices = list(list(zip(*sorted(list(zip(rankings, indices)))))[1])

    return sorted_indices
