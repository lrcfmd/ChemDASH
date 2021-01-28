"""
|=============================================================================|
|                            I N I T I A L I S E                              |
|=============================================================================|
|                                                                             |
| This module contains routines that set the initial layout of atoms on a     |
| grid.                                                                       |
|                                                                             |
| Contains                                                                    |
| --------                                                                    |
|     read_atoms_file                                                         |
|     check_charge_balance                                                    |
|     set_up_grids                                                            |
|     populate_grids_with_atoms                                               |
|     scale_cell                                                              |
|     determine_atom_positions                                                |
|     populate_points_with_vacancies                                          |
|     create_vacancy_grid                                                     |
|     get_distances_to_atoms                                                  |
|     initialise_from_cif                                                     |
|     generate_random_stacking_sequence                                       |
|     initialise_close_packed_grid_points                                     |
|                                                                             |
|-----------------------------------------------------------------------------|
| Paul Sharp 27/03/2020                                                       |
|=============================================================================|
"""

from builtins import range

import numpy as np
import ase
import ase.io
import sys


# =============================================================================
# =============================================================================
def read_atoms_file(atoms_file):
    """
    Read species, number and charge of ions from the ".atoms" file.

    Parameters
    ----------
    atoms_file : string
        Name of the ".atoms" file for this ChemDASH run.

    Returns
    -------
    atoms_data : list
        List of atomic species, number of atoms and oxidation state for each atomic species.

    ---------------------------------------------------------------------------
    Paul Sharp 06/08/2018
    """

    atoms_data = []
    with open(atoms_file, mode="r") as atoms:

        atoms_data = [line.rstrip() for line in atoms]

    return atoms_data

# =============================================================================
def check_charge_balance(atoms_data):
    """
    Read species, number and charge of ions from the ".atoms" file.

    Parameters
    ----------
    atoms_data : list
        List of atomic species, number of atoms and oxidation state for each atomic species.

    Returns
    -------
    charge_balance : int
        Value of (number of atoms * charge) for all atoms -- should be zero.

    ---------------------------------------------------------------------------
    Paul Sharp 11/10/2019
    """

    charge_balance = 0
    
    for species in atoms_data:

        num = int(species.split()[1])
        charge = int(species.split()[2])
                
        charge_balance += num * charge
 
    return charge_balance


# =============================================================================
def set_up_grids(grid_type, grid_points, stacking_sequence, lattice):
    """
    Initialise the grids on which we may place atoms, establishing cation and anion points.

    Parameters
    ----------
    grid_type : string
        The chosen arrangment of grid points for cations and anions.
    grid_points : integer
        List of the number of grid points along each dimension to form an (a, b, c) grid.
    stacking_sequence : str
        For close packed grids, the list of anion layers that form a close packed stacking sequence.
    lattice : str
        For close packed grids, states whether we are using the oblique or centred rectangular 2D lattice.

    Returns
    -------
    initial_struct : ase atoms
        The initial atoms object with the cell set to accomodate the desired number of grid points.
    anion grid, cation grid : float
        The list of points on which we may place the appropriate ionic species.

    ---------------------------------------------------------------------------
    Paul Sharp 27/03/2020
    """

    if grid_type == "orthorhombic":

        # Need to multiply up because we only define the anion grid on input
        grid_points[:] = [x * 2 for x in grid_points]

        initial_struct = ase.Atoms(cell=[float(grid_points[0]),
                                         float(grid_points[1]),
                                         float(grid_points[2])],
                                   pbc=[True, True, True])

        anion_grid = [[float(i) / float(grid_points[0]),
                       float(j) / float(grid_points[1]),
                       float(k) / float(grid_points[2])]
                      for i in range(0, grid_points[0], 2)
                      for j in range(0, grid_points[1], 2)
                      for k in range(0, grid_points[2], 2)]

        cation_grid = [[float(i) / float(grid_points[0]),
                        float(j) / float(grid_points[1]),
                        float(k) / float(grid_points[2])]
                       for i in range(1, grid_points[0], 2)
                       for j in range(1, grid_points[1], 2)
                       for k in range(1, grid_points[2], 2)]

    elif grid_type == "rocksalt":

        # Need to multiply up because we only define the anion grid on input
        grid_points[:] = [x * 2 for x in grid_points]

        initial_struct = ase.Atoms(cell=[float(grid_points[0]),
                                         float(grid_points[1]),
                                         float(grid_points[2])],
                                   pbc=[True, True, True])

        # Need to establish cation and anion points using cartesian coordinates,
        # and then convert to fractional coordinates
        full_grid = [[float(i), float(j), float(k)]
                     for i in range(0, grid_points[0])
                     for j in range(0, grid_points[1])
                     for k in range(0, grid_points[2])]

        anion_points = [x for x in full_grid if sum(x) % 2 == 0]
        cation_points = [x for x in full_grid if sum(x) % 2 != 0]

        anion_grid = [[point[0] / float(grid_points[0]),
                       point[1] / float(grid_points[1]),
                       point[2] / float(grid_points[2])] for point in anion_points]
        cation_grid = [[point[0] / float(grid_points[0]),
                        point[1] / float(grid_points[1]),
                        point[2] / float(grid_points[2])] for point in cation_points]

    elif grid_type == "close_packed":

        if lattice == "oblique":

            initial_struct = ase.Atoms(cell=[(float(grid_points[0]), 0.0, 0.0),
                                             (-0.5*float(grid_points[1]), 0.5*np.sqrt(3.0)*float(grid_points[1]), 0.0),
                                             (0.0, 0.0, 2.0*float(grid_points[2]))],
                                       pbc=[True, True, True])

        elif lattice == "centred_rectangular":

            initial_struct = ase.Atoms(cell=[(float(grid_points[0]), 0.0, 0.0),
                                             (0.0, float(grid_points[1]), 0.0),
                                             (0.0, 0.0, 2.0*float(grid_points[2]))],
                                       pbc=[True, True, True])

        else:

            sys.exit('ERROR in initialise.set_up_grids(), {0} is not a valid 2D lattice for a close packed grid.'.format(lattice))

        anion_grid, cation_grid = initialise_close_packed_grid_points(stacking_sequence, grid_points, lattice)

    else:

        sys.exit('ERROR in initialise.set_up_grids(), {0} is not a valid grid type.'.format(grid_type))

    return initial_struct, anion_grid, cation_grid


# =============================================================================
def populate_grids_with_atoms(initial_struct, anion_grid, cation_grid,
                              atom_data, rng):
    """
    Set up the initial structure with cations and anions randomly distributed
    on two grids. Cations and anions are placed on different grid points,
    according to the grid type.

    Parameters
    ----------
    initial_struct : ase atoms
        The initial structure with the unit cell set.
    anion grid, cation grid : float
        The list of points on which we may place the appropriate ionic species.
    atoms_data : string
        List of atomic species, number of atoms and oxidation state for each atomic species.
    rng : NR_ran
        Random number generator - algorithm from Numerical Recipes 2007

    Returns
    -------
    initial_struct : ase atoms
        The initial structure with the cations and anions randomly distributed on their grids, and unoccupied points taken by "X" atoms.

    ---------------------------------------------------------------------------
    Paul Sharp 27/03/2020
    """

    # Build up the initial structure on a species by species basis
    for species_data in atom_data:

        # Extract species, number and charge of atoms
        species, num_atoms, ionic_charge = species_data.split()
        num_atoms = int(num_atoms)
        ionic_charge = float(ionic_charge)

        if num_atoms > 0:

            if ionic_charge > 0.0:
                try:
                    cation_grid, pos = determine_atom_positions(cation_grid, num_atoms, rng)
                except (AssertionError, IndexError):
                    sys.exit('ERROR in initialise.populate_grids_with_atoms() -- a {0} atom could not be placed on the grid because all points are occupied.'.format(species))

            else:
                try:
                    anion_grid, pos = determine_atom_positions(anion_grid, num_atoms, rng)
                except (AssertionError, IndexError):
                    sys.exit('ERROR in initialise.populate_grids_with_atoms() -- a {0} atom could not be placed on the grid because all points are occupied.'.format(species))

            initial_struct.extend(ase.Atoms(species + str(num_atoms),
                                            cell=initial_struct.get_cell(),
                                            scaled_positions=pos,
                                            charges=[ionic_charge] * num_atoms,
                                            pbc=[True, True, True]))

    return initial_struct, anion_grid, cation_grid


# =============================================================================
def scale_cell(struct, cell_spacing):
    """
    Set the unit cell dimensions according to the spacing between grid points.

    Parameters
    ----------
    struct : ase atoms
        The structure with the unit cell set.
    cell_spacing : float
        The desired spacing between grid points.

    Returns
    -------
    struct : ase atoms
        The structure with the unit cell dimensions scaled according to the
        spacing between grid points.

    ---------------------------------------------------------------------------
    Paul Sharp 25/10/2017
    """

    scaled_cell = []

    for i in range(0, 3):
        row = struct.get_cell()[i]
        scaled_cell.append([row[0] * cell_spacing[i],
                            row[1] * cell_spacing[i],
                            row[2] * cell_spacing[i]])

    struct.set_cell(scaled_cell, scale_atoms=True)

    return struct


# =============================================================================
def determine_atom_positions(grid, num_atoms, rng):
    """
    Place all of the atoms of one species onto a grid, removing occupied points
    as we go.

    Parameters
    ----------
    grid : float
        The list of possible points for atoms of this species.
    num_atoms : integer
        The number of atoms of this species.
    rng : NR_ran
        Random number generator - algorithm from Numerical Recipes 2007

    Returns
    -------
    initial_struct : ase atoms
        The initial structure updated with atoms of this species
    grid : float
        The updated list of points, with occupied points removed.
    positions : float
        The list of points for this set of atoms to occupy.

    ---------------------------------------------------------------------------
    Paul Sharp 27/10/2017
    """

    assert len(grid) >= num_atoms, 'ERROR in "initialise.determine_atom_positions()" -- we are trying to place more atoms than the number of points available.'

    # Set up atomic positions of the ions of this species
    positions = []
    charge_list = []

    # Random position for each atom, ensuring two atoms cannot occupy the same point
    for atoms in range(0, num_atoms):

        point = grid.pop(rng.int_range(0, len(grid)))
        positions.append(point)

    return grid, positions


# =============================================================================
def populate_points_with_vacancies(struct, vacancy_points):
    """
    Add a list of points to a structure as vacancies.

    Parameters
    ----------
    struct : ase atoms
        The structure to which we will add vacancies.
    vacancy_points : float
        The list of points to be recorded as vacancies.

    Returns
    -------
    struct : ase atoms
        The structure with the points set as vacancies.

    ---------------------------------------------------------------------------
    Paul Sharp 25/10/2017
    """

    num_vacancies = len(vacancy_points)

    if num_vacancies > 0:
        struct.extend(ase.Atoms("X" + str(num_vacancies), cell=struct.get_cell(),
                                scaled_positions=vacancy_points,
                                charges=[0] * num_vacancies,
                                pbc=[True, True, True]))

    return struct


# =============================================================================
def create_vacancy_grid(structure, vacancy_separation, exclusion_radius):
    """
    Create a grid of vacancy points for the current structure.

    Parameters
    ----------
    structure : ase atoms
        The structure for which we are constructing a vacancy grid
    vacancy_separation : float
        The minimum distance between any two vacancies which defines the size
        of the vacancy grid.
    exclusion_radius : float
        The minimum allowed distance between an atom and a vacancy in the structure.

    Returns
    -------
    vacancy_grid : float
        The points on the vacancy grid in scaled cell coordinates.

    ---------------------------------------------------------------------------
    Paul Sharp 29/04/2019
    """

    cell_vectors = structure.get_cell()
    
    # Determine the length of each axis of the cartesian grid
    vertices = [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 1.0, 1.0],
                [1.0, 0.0, 0.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0],
                [1.0, 1.0, 1.0]]

    a = []
    b = []
    c = []
    
    for point in vertices:

        cart_vertex = np.dot(np.array(point), cell_vectors)

        a.append(cart_vertex[0])
        b.append(cart_vertex[1])
        c.append(cart_vertex[2])

    x0 = min(a)
    y0 = min(b)
    z0 = min(c)
        
    x = max(a) - x0
    y = max(b) - y0
    z = max(c) - z0
        
    # Determine the number of grid points on each axis
    grid_points = [int(x / vacancy_separation),
                   int(y / vacancy_separation),
                   int(z / vacancy_separation)]

    cart_vacancy_grid_separation = [x / grid_points[0],
                                    y / grid_points[1],
                                    z / grid_points[2]]

    origin_offset = [0.5 * separation for separation in cart_vacancy_grid_separation]
    
    cart_vacancy_grid = [[(float(i) * cart_vacancy_grid_separation[0]) + origin_offset[0] + x0,
                          (float(j) * cart_vacancy_grid_separation[1]) + origin_offset[1] + y0,
                          (float(k) * cart_vacancy_grid_separation[2]) + origin_offset[2] + z0]
                         for i in range(0, grid_points[0])
                         for j in range(0, grid_points[1])
                         for k in range(0, grid_points[2])]
    
    # Convert coordinates from cartesian to cell coordinates, and reject points that do not lie
    # within the unit cell
    vacancy_grid = []
    
    for i in range(0, len(cart_vacancy_grid)):
        
        point = np.linalg.solve(cell_vectors.T, np.array(cart_vacancy_grid[i]).T).T

        if not(any([coord < 0.0 for coord in point]) or any([coord > 1.0 for coord in point])):

            # Add the point if it does not lie within the exclusion region of any of the atoms
            if min(get_distances_to_atoms(structure.copy(), point)) > exclusion_radius:
                
                vacancy_grid.append(point)
     
    return vacancy_grid


# =============================================================================
def get_distances_to_atoms(structure, vacancy_position):
    """
    Find the distances between a proposed vacancy and the atoms in a structure.

    Parameters
    ----------
    structure : ase atoms
        The structure containing the atoms we want to find the distances to.
    vacancy_position : float
        The position of the proposed vacancy.

    Returns
    -------
    distances : float
        A list of positions from the proposed vacancy to each atom in the
        structure.

    ---------------------------------------------------------------------------
    Paul Sharp 09/12/2019
    """

    num_atoms = len(structure)
    
    # Add the vacancy to the structure
    structure += ase.Atoms("X", cell=structure.get_cell(),
                           scaled_positions=[vacancy_position],
                           charges=[0], pbc=[True, True, True])

    distances = structure.get_distances(-1, list(range(num_atoms)), mic=True)

    return distances

    
# =============================================================================
def initialise_from_cif(cif_file, atoms_file_data):
    """
    Initialise an atoms object with a specific structure contained in a cif file.

    ASE cannot read in charge data, so we read it ourselves from either the
    cif file or the ".atoms" file.

    Parameters
    ----------
    cif_file : string
        The ".cif" file containing the initial structure.
    atoms_file_data : str, int, float
        The species, number of atoms, and charge for all atoms in the structure
        from the cif file.

    Returns
    -------
    initial_struct : ase atoms
        The initial structure from the cif file, including charges for each atom.

    ---------------------------------------------------------------------------
    Paul Sharp 16/01/2020
    """

    loop_marker = "loop_"
    symbol_marker = "_atom_type_symbol"
    oxidation_number_marker = "_atom_type_oxidation_number"

    names = []
    charges = []

    # Read cif file into atoms object
    initial_struct = ase.io.read(cif_file)

    with open(cif_file, mode="r") as cif:
        cif_file_data = cif.readlines()

    # Read charges from either cif file or atoms file
    read_from_cif = False
    for line in cif_file_data:
        if oxidation_number_marker in line:
            read_from_cif = True
            break

    if read_from_cif:

        # Remove blank lines and newline characters
        cif_file_data[:] = [x.strip() for x in cif_file_data if not x.startswith("\n")]

        # Find loops in cif file, and the location of oxidation numbers
        loop_indices = [i for i, x in enumerate(cif_file_data) if x == loop_marker]
        oxidation_index = cif_file_data.index(oxidation_number_marker)

        # Find loop with oxidation numbers
        for index in loop_indices:
            if index > oxidation_index:
                end_index = index
                break
        else:
            end_index = len(cif_file_data)

        try:
            start_index = loop_indices[loop_indices.index(end_index) - 1]
        except ValueError:
            start_index = loop_indices[-1]

        oxidation_loop = cif_file_data[start_index:end_index]

        symbol_pos = oxidation_loop.index(symbol_marker) - 1
        oxidation_pos = oxidation_loop.index(oxidation_number_marker) - 1

        # Underscores are not allowed in atom symbol code -- hence separate header from data
        oxidation_loop_data = [x for x in oxidation_loop if "_" not in x]

        elements_list = initial_struct.get_chemical_symbols()
        delete_chars = "0123456789+-"
        
        for line in oxidation_loop_data:

            # Check if python 3 method is available
            try:
                trans_table = line.split()[symbol_pos].maketrans("", "", delete_chars)
            # Python 2.7 solution
            except AttributeError:
                element = line.split()[symbol_pos].translate(None, delete_chars)
            # Python 3 solution
            else:
                element = line.split()[symbol_pos].translate(trans_table)

            names.extend([element] * elements_list.count(element))
            charges.extend([float(line.split()[oxidation_pos])] * elements_list.count(element))

    # If charge data is not included in the cif file, read it from the atoms file
    else:

        for entry in atoms_file_data:
            names.extend([entry.split()[0]] * int(entry.split()[1]))
            charges.extend([float(entry.split()[2])] * int(entry.split()[1]))

    # Set charges from list if list matches atoms object
    if names != initial_struct.get_chemical_symbols():
        sys.exit("ERROR in initialise.initialise_from_cif() -- the list of atoms with charges does not match up with the atoms in the cif file.")

    initial_struct.set_initial_charges(charges)

    return initial_struct


# =============================================================================
def generate_random_stacking_sequence(num_layers, rng):
    """
    For close packed grids, this routine generates a random sequence of A, B
    and C stacking layers subject to the condition that no two adjacent layers
    are the same.

    Parameters
    ----------
    num_layers : int
        The number of anion layers in the stacking sequence.
    rng : NR_ran
        Random number generator - algorithm from Numerical Recipes 2007

    Returns
    -------
    stacking_sequence : char
        List of anion layers that form a close packed stacking sequence.

    ---------------------------------------------------------------------------
    Paul Sharp 26/10/2017
    """

    stacking_layers = ["A", "B", "C"]
    stacking_sequence = []

    remaining_layers = stacking_layers

    if num_layers >= 1:

        # Add new layers, ensuring that the next layer will be chosen from the two other layers
        for layer in range(0, num_layers - 1):

            new_layer = remaining_layers[rng.int_range(0, len(remaining_layers))]
            stacking_sequence.append(new_layer)

            remaining_layers = [layer for layer in stacking_layers if layer != new_layer]

        # For the final layer, must ensure that it is different from both previous layer and first layer (due to periodic bounary conditions)
        try:
            remaining_layers = [layer for layer in stacking_layers if layer != stacking_sequence[0] and layer != stacking_sequence[-1]]
        except IndexError:
            remaining_layers = stacking_layers

        final_layer = remaining_layers[rng.int_range(0, len(remaining_layers))]
        stacking_sequence.append(final_layer)

    return stacking_sequence


# =============================================================================
def initialise_close_packed_grid_points(stacking_sequence, grid_points, lattice):
    """
    For close packed grids, this routine generates the lists of points for
    cations and anions according to the stacking sequence.

    Parameters
    ----------
    stacking_sequence : str
        List of anion layers that form a close packed stacking sequence.
    grid_points : integer
        List of the number of grid points along each dimension to form an (a, b, c) grid.
    lattice : str
        States whether we are using the oblique or centred rectangular 2D lattice.

    Returns
    -------
    anion grid, cation grid : float
        The list of points on which we may place the appropriate ionic species.

    ---------------------------------------------------------------------------
    Paul Sharp 27/03/2020
    """

    # Check that stacking sequence contains only valid values
    assert set(stacking_sequence).issubset(["A", "B", "C"]), 'ERROR in "initialise.initialise_close_packed_grid_points()" -- stacking sequence contains an entry that is not one of the allowed layers: "A", "B" or "C".'

    # Determines what to add to a and b coordinates to form the requisite layer
    if lattice == "oblique":
        layer_offset = {"A": [0.5, 0.5], "B": [5.0/6.0, 1.0/6.0], "C": [1.0/6.0, 5.0/6.0]}
    elif lattice == "centred_rectangular":
        layer_offset = {"A": [0.5, 0.5], "B": [0.5, 5.0/6.0], "C": [0.5, 1.0/6.0]}
    else:
        sys.exit('ERROR in "initialise.initialise_close_packed_grid_points()" -- {0} is not a valid 2D lattice.'.format(lattice))

    anion_grid = []
    cation_grid = []
    z = 0.0

    for i, current_layer in enumerate(stacking_sequence):

        # Return first layer at the end (for peridic boundary conditions)
        next_layer = stacking_sequence[(i + 1) % len(stacking_sequence)]

        octahedral_layer = [layer for layer in ["A", "B", "C"]
                            if layer != current_layer and layer != next_layer][0]

        # Use layer_offset dictionary to determine where the current anion layer is placed
        anion_grid.extend([[(float(i) + layer_offset[current_layer][0]) / float(grid_points[0]),
                            (float(j) + layer_offset[current_layer][1]) / float(grid_points[1]),
                            0.5 * z / float(grid_points[2])]
                           for i in range(0, grid_points[0])
                           for j in range(0, grid_points[1])])

        # For the cation sites, we place all three layers between the current layer and the next. Two of these layers (the two that are the same as
        # the anion layers) are the tetrahedral sites, with the remaining layer representing the octahedral sites.
        # The octahedral layer should be placed halfway between the anion layers, with the tetrahedral layers placed in a 3:1 ratio away from
        # the like anion layer.
        cation_grid.extend([[(float(i) + layer_offset[next_layer][0]) / float(grid_points[0]),
                             (float(j) + layer_offset[next_layer][1]) / float(grid_points[1]),
                             (0.5 * (z + 1) - 0.25) / float(grid_points[2])]
                            for i in range(0, grid_points[0])
                            for j in range(0, grid_points[1])])

        cation_grid.extend([[(float(i) + layer_offset[octahedral_layer][0]) / float(grid_points[0]),
                             (float(j) + layer_offset[octahedral_layer][1]) / float(grid_points[1]),
                             0.5 * (z + 1) / float(grid_points[2])]
                            for i in range(0, grid_points[0])
                            for j in range(0, grid_points[1])])

        cation_grid.extend([[(float(i) + layer_offset[current_layer][0]) / float(grid_points[0]),
                             (float(j) + layer_offset[current_layer][1]) / float(grid_points[1]),
                             (0.5 * (z + 1) + 0.25) / float(grid_points[2])]
                            for i in range(0, grid_points[0])
                            for j in range(0, grid_points[1])])

        # Add the additional points in the unit cell if we are using the centred rectangular lattice
        if lattice == "centred_rectangular":

            anion_grid.extend([[(0.5 * float(i) + layer_offset[current_layer][0]) / float(grid_points[0]),
                                (0.5 * float(j) + layer_offset[current_layer][1]) / float(grid_points[1]),
                                0.5 * z / float(grid_points[2])]
                               for i in range(1, 2 * grid_points[0], 2)
                               for j in range(1, 2 * grid_points[1], 2)])

            cation_grid.extend([[(0.5 * float(i) + layer_offset[next_layer][0]) / float(grid_points[0]),
                                 (0.5 * float(j) + layer_offset[next_layer][1]) / float(grid_points[1]),
                                 (0.5 * (z + 1) - 0.25)/float(grid_points[2])]
                                for i in range(1, 2 * grid_points[0], 2)
                                for j in range(1, 2 * grid_points[1], 2)])

            cation_grid.extend([[(0.5 * float(i) + layer_offset[octahedral_layer][0]) / float(grid_points[0]),
                                 (0.5 * float(j) + layer_offset[octahedral_layer][1]) / float(grid_points[1]),
                                 0.5 * (z + 1) / float(grid_points[2])]
                                for i in range(1, 2 * grid_points[0], 2)
                                for j in range(1, 2 * grid_points[1], 2)])

            cation_grid.extend([[(0.5 * float(i) + layer_offset[current_layer][0]) / float(grid_points[0]),
                                 (0.5 * float(j) + layer_offset[current_layer][1]) / float(grid_points[1]),
                                 (0.5 * (z + 1) + 0.25) / float(grid_points[2])]
                                for i in range(1, 2 * grid_points[0], 2)
                                for j in range(1, 2 * grid_points[1], 2)])

        z += 2.0

    return anion_grid, cation_grid
