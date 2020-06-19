import pytest
import mock

import chemdash.initialise

import ase
import numpy as np

#===========================================================================================================================================================
#===========================================================================================================================================================
# Track tests

# UNIT TESTS
# test_read_atoms_file
# test_set_up_grids (tests 1 & 2)
# test_set_up_grids_exceptions
# test_populate_grids_with_atoms
# test_populate_grids_with_atoms_exception_1
# test_populate_grids_with_atoms_exception_2
# test_scale_cell
# test_determine_atom_positions
# test_determine_atom_positions_exception
# test_populate_points_with_vacancies
# test_create_vacancy_grid
# test_get_distances_to_atoms
# test_initialise_from_cif
# test_initialise_from_cif_exception
# test_generate_random_stacking_sequence
# test_initialise_close_packed_grid_points
# test_initialise_close_packed_grid_points_exception_1
# test_initialise_close_packed_grid_points_exception_2

#INTEGRATION TESTS
#test_set_up_grids (tests 3 & 4)


#===========================================================================================================================================================
#===========================================================================================================================================================
#Unit Tests

@pytest.mark.parametrize("test_atoms_file, expected_output", [
    (
'''O 3 -2
Sr 1 +2
Ti 1 +4''',
["O 3 -2", "Sr 1 +2", "Ti 1 +4"]),
])

def test_read_atoms_file(test_atoms_file, expected_output, monkeypatch):
    """
    GIVEN an atoms file

    WHEN we read in the atoms file

    THEN we return a list containing the species, number of atoms and charge for
         each species of atom

    Parameters
    ----------
    test_atoms_file: str
        A mock atoms file.

    ---------------------------------------------------------------------------
    Paul Sharp 07/08/2018
    """

    # We need to ensure we use "mock_open" to return our test data instead of reading a file
    # We also need to ensure that the return value of this Mock produces an iterable,
    # so we can loop over the lines of the file
    iterable_mock_file = mock.mock_open(read_data = test_atoms_file)
    iterable_mock_file.return_value.__iter__ = lambda x : iter(x.readline, '')

    # Note the lack of lambda -- we want the mock itself, not its return value
    # We pass a blank string for the file argument for this reason
    monkeypatch.setattr('builtins.open', iterable_mock_file)
    
    assert chemdash.initialise.read_atoms_file("") == expected_output


#===========================================================================================================================================================
@pytest.mark.parametrize("grid_type, grid_points, stacking_sequence, lattice, expected_output", [
    ("orthorhombic", [2, 2, 2], "", "",
     (ase.Atoms(cell = [4.0, 4.0, 4.0], pbc = [True, True, True]),
      [[0.0, 0.0, 0.0], [0.0, 0.0, 0.5], [0.0, 0.5, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.0], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0], [0.5, 0.5, 0.5]],
      [[0.25, 0.25, 0.25], [0.25, 0.25, 0.75], [0.25, 0.75, 0.25], [0.25, 0.75, 0.75], [0.75, 0.25, 0.25], [0.75, 0.25, 0.75], [0.75, 0.75, 0.25], [0.75, 0.75, 0.75]])),
    ("rocksalt", [1, 1, 1], "", "",
     (ase.Atoms(cell = [2.0, 2.0, 2.0], pbc = [True, True, True]), 
      [[0.0, 0.0, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]],
      [[0.0, 0.0, 0.5], [0.0, 0.5, 0.0], [0.5, 0.0, 0.0], [0.5, 0.5, 0.5]])),
    ("close_packed", [2, 2, 2], ["A", "B"], "oblique",
     (ase.Atoms(cell = [[2.0, 0.0, 0.0], [-1.0, 1.7320508075688772, 0.0], [0.0, 0.0, 4.0]], pbc=[True, True, True]),
     [[0.25, 0.25, 0.0], [0.25, 0.75, 0.0], [0.75, 0.25, 0.0], [0.75, 0.75, 0.0], [0.4166666666666667, 0.08333333333333333, 0.5], [0.4166666666666667, 0.5833333333333334, 0.5],
      [0.9166666666666667, 0.08333333333333333, 0.5], [0.9166666666666667, 0.5833333333333334, 0.5]],
     [[0.4166666666666667, 0.08333333333333333, 0.125], [0.4166666666666667, 0.5833333333333334, 0.125], [0.9166666666666667, 0.08333333333333333, 0.125],
      [0.9166666666666667, 0.5833333333333334, 0.125], [0.08333333333333333, 0.4166666666666667, 0.25], [0.08333333333333333, 0.9166666666666667, 0.25],
      [0.5833333333333334, 0.4166666666666667, 0.25], [0.5833333333333334, 0.9166666666666667, 0.25], [0.25, 0.25, 0.375], [0.25, 0.75, 0.375], [0.75, 0.25, 0.375],
      [0.75, 0.75, 0.375], [0.25, 0.25, 0.625], [0.25, 0.75, 0.625], [0.75, 0.25, 0.625], [0.75, 0.75, 0.625], [0.08333333333333333, 0.4166666666666667, 0.75],
      [0.08333333333333333, 0.9166666666666667, 0.75], [0.5833333333333334, 0.4166666666666667, 0.75], [0.5833333333333334, 0.9166666666666667, 0.75],
      [0.4166666666666667, 0.08333333333333333, 0.875], [0.4166666666666667, 0.5833333333333334, 0.875], [0.9166666666666667, 0.08333333333333333, 0.875],
      [0.9166666666666667, 0.5833333333333334, 0.875]])),
    ("close_packed", [1, 1, 3], ["A", "B", "C"], "centred_rectangular",
     (ase.Atoms(cell=[1.0, 1.0, 6.0], pbc=[True, True, True]),
      [[0.5, 0.5, 0.0], [1.0, 1.0, 0.0], [0.5, 0.8333333333333334, 0.3333333333333333], [1.0, 1.3333333333333335, 0.3333333333333333],
       [0.5, 0.16666666666666666, 0.6666666666666666], [1.0, 0.6666666666666666, 0.6666666666666666]],
      [[0.5, 0.8333333333333334, 0.08333333333333333], [0.5, 0.16666666666666666, 0.16666666666666666], [0.5, 0.5, 0.25], [1.0, 1.3333333333333335, 0.08333333333333333],
       [1.0, 0.6666666666666666, 0.16666666666666666], [1.0, 1.0, 0.25], [0.5, 0.16666666666666666, 0.4166666666666667], [0.5, 0.5, 0.5],
       [0.5, 0.8333333333333334, 0.5833333333333334], [1.0, 0.6666666666666666, 0.4166666666666667], [1.0, 1.0, 0.5], [1.0, 1.3333333333333335, 0.5833333333333334],
       [0.5, 0.5, 0.75], [0.5, 0.8333333333333334, 0.8333333333333334], [0.5, 0.16666666666666666, 0.9166666666666666], [1.0, 1.0, 0.75],
       [1.0, 1.3333333333333335, 0.8333333333333334], [1.0, 0.6666666666666666, 0.9166666666666666]])),
])

def test_set_up_grids(grid_type, grid_points, stacking_sequence, lattice, expected_output):
    """
    GIVEN a grid type, data, and set of grid points

    WHEN we initialise the atoms object and cation and anion grids according to the grid type

    THEN we get an atoms object with the unit cell defined and two grids for anion and cation points

    Parameters
    ----------
    grid_type : string
        The chosen arrangment of grid points for cations and anions.
    grid_points : integer
        List of the number of grid points along each dimension to form an (a, b, c) grid.
    stacking_sequence : str
        For close packed grids, the list of anion layers that form a close packed stacking sequence.
    lattice : str
        The lattice used for anion layers in close packed grids.

    ---------------------------------------------------------------------------
    Paul Sharp 26/10/2017
    """

    assert chemdash.initialise.set_up_grids(grid_type, grid_points, stacking_sequence, lattice) == expected_output


#===========================================================================================================================================================
@pytest.mark.parametrize("grid_type, grid_points, stacking_sequence, lattice", [
    ("foobar", [2, 2, 2], "", ""),
    ("close_packed", [2, 2, 2], ["A", "B"], "foobar"),
])

def test_set_up_grids_exception(grid_type, grid_points, stacking_sequence, lattice):
    """
    GIVEN an invalid grid type or cell setting

    WHEN we initialise the atoms object and cation and anion grids according to the grid type

    THEN we raise an exception

    Parameters
    ----------
    grid_type : string
        The chosen arrangment of grid points for cations and anions.
    grid_points : integer
        List of the number of grid points along each dimension to form an (a, b, c) grid.
    stacking_sequence : str
        For close packed grids, the list of anion layers that form a close packed stacking sequence.
    lattice : str
        The lattice used for anion layers in close packed grids.

    ---------------------------------------------------------------------------
    Paul Sharp 26/10/2017
    """

    with pytest.raises(SystemExit):
        chemdash.initialise.set_up_grids(grid_type, grid_points, stacking_sequence, lattice)


#===========================================================================================================================================================
@pytest.mark.parametrize("initial_struct, anion_grid, cation_grid, atom_data, expected_output", [
    (ase.Atoms(cell = [2.0, 2.0, 2.0], pbc=[True, True, True]),
     [[0.0, 0.0, 0.0], [0.0, 0.0, 0.5], [0.0, 0.5, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.0], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0], [0.5, 0.5, 0.5]],
     [[0.25, 0.25, 0.25], [0.25, 0.25, 0.75], [0.25, 0.75, 0.25], [0.25, 0.75, 0.75], [0.75, 0.25, 0.25], [0.75, 0.25, 0.75], [0.75, 0.75, 0.25], [0.75, 0.75, 0.75]],
     ["Sr 1 +2.0", "Ti 1 +4.0", "O 3 -2.0"],
     (ase.Atoms(symbols = "SrTiO3", cell = [2.0, 2.0, 2.0], charges = [2.0, 4.0, -2.0, -2.0, -2.0],
                    scaled_positions = ([0.75, 0.75, 0.25], [0.75, 0.25, 0.25], [0.5, 0.5, 0.5], [0.5, 0.0, 0.0], [0.0, 0.0, 0.5]),
                    pbc=[True, True, True]),
      [[0.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]],
      [[0.25, 0.25, 0.25], [0.25, 0.25, 0.75], [0.25, 0.75, 0.25], [0.25, 0.75, 0.75], [0.75, 0.25, 0.75], [0.75, 0.75, 0.75]])),
])

# ATOMS OBJECT IN OUTPUT IS FIXTURE "STO_ATOMS" -- NEED TO FIGURE OUT HOW TO INCLUDE THIS.
def test_populate_grids_with_atoms(initial_struct, anion_grid, cation_grid, atom_data, rng, STO_atoms, expected_output, monkeypatch):
    """
    GIVEN an atoms object with the unit cell defined, cation and anion grids, and the species, number, and charge of the desired atoms

    WHEN we populate those atoms onto the grids

    THEN we get an atoms object with the atoms placed at the desired positions, along with the remaining cation and anion points

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

    ---------------------------------------------------------------------------
    Paul Sharp 27/06/2019
    """

    # We patch the calls to "initialise.determine_atom_positions()" in order to choose the points we want.
    mock_ftn_call = mock.MagicMock(side_effect = [([[0.25, 0.25, 0.25], [0.25, 0.25, 0.75], [0.25, 0.75, 0.25], [0.25, 0.75, 0.75],
                                                    [0.75, 0.25, 0.25], [0.75, 0.25, 0.75], [0.75, 0.75, 0.75]], [[0.75, 0.75, 0.25]]),
                                                  ([[0.25, 0.25, 0.25], [0.25, 0.25, 0.75], [0.25, 0.75, 0.25], [0.25, 0.75, 0.75],
                                                    [0.75, 0.25, 0.75], [0.75, 0.75, 0.75]], [[0.75, 0.25, 0.25]]),
                                                  ([[0.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]],
                                                   [[0.5, 0.5, 0.5], [0.5, 0.0, 0.0], [0.0, 0.0, 0.5]])])

    monkeypatch.setattr(chemdash.initialise, 'determine_atom_positions', lambda x1, x2, x3 : mock_ftn_call())
    updated_struct, updated_anion_grid, updated_cation_grid = chemdash.initialise.populate_grids_with_atoms(initial_struct, anion_grid, cation_grid, atom_data, rng)
    
    # Class definition is that two ase Atoms objects are the same if they have the same atoms, unit cell,
    # positions and boundary conditions. Therefore we need to test charges separately.
    assert (updated_struct, updated_anion_grid, updated_cation_grid) == expected_output
    assert all([x == y for x, y in zip(updated_struct.get_initial_charges(), expected_output[0].get_initial_charges())])


#===========================================================================================================================================================
@pytest.mark.parametrize("initial_struct, anion_grid, cation_grid, atom_data", [
    (ase.Atoms(cell = [2.0, 2.0, 2.0], pbc=[True, True, True]),
     [[0.0, 0.0, 0.0], [0.0, 0.0, 0.5], [0.0, 0.5, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.0], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0], [0.5, 0.5, 0.5]],
     [[0.25, 0.25, 0.25], [0.25, 0.25, 0.75], [0.25, 0.75, 0.25], [0.25, 0.75, 0.75], [0.75, 0.25, 0.25], [0.75, 0.25, 0.75], [0.75, 0.75, 0.25], [0.75, 0.75, 0.75]],
     ["Sr 1 +2.0", "Ti 1 +4.0", "O 10 -2.0"]),
])

# ATOMS OBJECT IN OUTPUT IS FIXTURE "STO_ATOMS" -- NEED TO FIGURE OUT HOW TO INCLUDE THIS.
def test_populate_grids_with_atoms_exception_1(initial_struct, anion_grid, cation_grid, atom_data, rng, monkeypatch):
    """
    GIVEN an atoms object with the unit cell defined, cation and anion grids, and the species, number, and charge of the desired atoms,
    but too many anions

    WHEN we populate those atoms onto the grids

    THEN we cause a system exit

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

    ---------------------------------------------------------------------------
    Paul Sharp 27/10/2017
    """

    # We patch the calls to "initialise.determine_atom_positions()" in order to choose the points we want, and raise the exception correctly.
    mock_exception = mock.MagicMock(side_effect = [([[0.25, 0.25, 0.25], [0.25, 0.25, 0.75], [0.25, 0.75, 0.25], [0.25, 0.75, 0.75],
                                                    [0.75, 0.25, 0.25], [0.75, 0.25, 0.75], [0.75, 0.75, 0.75]], [[0.75, 0.75, 0.25]]),
                                                  ([[0.25, 0.25, 0.25], [0.25, 0.25, 0.75], [0.25, 0.75, 0.25], [0.25, 0.75, 0.75],
                                                    [0.75, 0.25, 0.75], [0.75, 0.75, 0.75]], [[0.75, 0.25, 0.25]]), AssertionError])

    monkeypatch.setattr(chemdash.initialise, 'determine_atom_positions', lambda x1, x2, x3 : mock_exception())

    with pytest.raises(SystemExit):
        chemdash.initialise.populate_grids_with_atoms(initial_struct, anion_grid, cation_grid, atom_data, rng)


#===========================================================================================================================================================
@pytest.mark.parametrize("initial_struct, anion_grid, cation_grid, atom_data", [
    (ase.Atoms(cell = [2.0, 2.0, 2.0], pbc=[True, True, True]),
     [[0.0, 0.0, 0.0], [0.0, 0.0, 0.5], [0.0, 0.5, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.0], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0], [0.5, 0.5, 0.5]],
     [[0.25, 0.25, 0.25], [0.25, 0.25, 0.75], [0.25, 0.75, 0.25], [0.25, 0.75, 0.75], [0.75, 0.25, 0.25], [0.75, 0.25, 0.75], [0.75, 0.75, 0.25], [0.75, 0.75, 0.75]],
     ["Sr 10 +2.0", "Ti 1 +4.0", "O 3 -2.0"]),
])

# ATOMS OBJECT IN OUTPUT IS FIXTURE "STO_ATOMS" -- NEED TO FIGURE OUT HOW TO INCLUDE THIS.
def test_populate_grids_with_atoms_exception_2(initial_struct, anion_grid, cation_grid, atom_data, rng, monkeypatch):
    """
    GIVEN an atoms object with the unit cell defined, cation and anion grids, and the species, number, and charge of the desired atoms
    but too many cations

    WHEN we populate those atoms onto the grids

    THEN we cause a system exit

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

    ---------------------------------------------------------------------------
    Paul Sharp 27/10/2017
    """

    # We patch the calls to "initialise.determine_atom_positions()" in order to choose the points we want, and raise the exception correctly.
    mock_exception = mock.MagicMock(side_effect = (AssertionError))

    monkeypatch.setattr(chemdash.initialise, 'determine_atom_positions', lambda x1, x2, x3 : mock_exception())

    with pytest.raises(SystemExit):
        chemdash.initialise.populate_grids_with_atoms(initial_struct, anion_grid, cation_grid, atom_data, rng)


#===========================================================================================================================================================
@pytest.mark.parametrize("cell_spacing, expected_output", [
    ([0.5, 0.5, 0.5], ase.Atoms(symbols = "SrTiO3", cell = [1.0, 1.0, 1.0],
                                charges = [2.0, 4.0, -2.0, -2.0, -2.0],
                                scaled_positions = ([0.75, 0.75, 0.25], [0.75, 0.25, 0.25], [0.5, 0.5, 0.5], [0.5, 0.0, 0.0], [0.0, 0.0, 0.5]),
                                pbc=[True, True, True])),
    ([0.1, 0.2, 0.3], ase.Atoms(symbols = "SrTiO3", cell = [0.2, 0.4, 0.6], charges = [2.0, 4.0, -2.0, -2.0, -2.0],
                                scaled_positions = ([0.75, 0.75, 0.25], [0.75, 0.25, 0.25], [0.5, 0.5, 0.5], [0.5, 0.0, 0.0], [0.0, 0.0, 0.5]),
                                pbc=[True, True, True])),
])

def test_scale_cell(STO_atoms, cell_spacing, expected_output):
    """
    GIVEN an atoms object and desired cell spacing

    WHEN we scale the unit cell

    THEN we return the atoms object with the unit cell scaled according to the desired spacing between grid points

    Parameters
    ----------
    struct : ase atoms
        The structure with the unit cell set.
    cell_spacing : float
        The desired spacing between grid points.

    ---------------------------------------------------------------------------
    Paul Sharp 24/07/2019
    """

    #print chemdash.initialise.scale_cell(STO_atoms, [1.0,1.0,1.0]).get_scaled_positions()
    #print expected_output.get_scaled_positions()
    
    assert chemdash.initialise.scale_cell(STO_atoms, cell_spacing) == expected_output


#===========================================================================================================================================================
@pytest.mark.parametrize("grid, num_atoms, expected_output", [
    ([[0.0, 0.0, 0.0], [0.0, 0.0, 0.5], [0.0, 0.5, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.0], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0], [0.5, 0.5, 0.5]], -2,
     ([[0.0, 0.0, 0.0], [0.0, 0.0, 0.5], [0.0, 0.5, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.0], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0], [0.5, 0.5, 0.5]], [])),
    ([[0.0, 0.0, 0.0], [0.0, 0.0, 0.5], [0.0, 0.5, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.0], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0], [0.5, 0.5, 0.5]], 0,
     ([[0.0, 0.0, 0.0], [0.0, 0.0, 0.5], [0.0, 0.5, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.0], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0], [0.5, 0.5, 0.5]], [])),
    ([[0.0, 0.0, 0.0], [0.0, 0.0, 0.5], [0.0, 0.5, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.0], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0], [0.5, 0.5, 0.5]], 1,
     ([[0.0, 0.0, 0.5], [0.0, 0.5, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.0], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0], [0.5, 0.5, 0.5]], [[0.0, 0.0, 0.0]])),
    ([[0.0, 0.0, 0.0], [0.0, 0.0, 0.5], [0.0, 0.5, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.0], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0], [0.5, 0.5, 0.5]], 4,
     ([[0.0, 0.0, 0.5], [0.5, 0.0, 0.0], [0.5, 0.5, 0.0], [0.5, 0.5, 0.5]], [[0.0, 0.0, 0.0], [0.0, 0.5, 0.5], [0.0, 0.5, 0.0], [0.5, 0.0, 0.5]])),
    ([[0.0, 0.0, 0.0], [0.0, 0.0, 0.5], [0.0, 0.5, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.0], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0], [0.5, 0.5, 0.5]], 8,
     ([], [[0.0, 0.0, 0.0], [0.0, 0.5, 0.5], [0.0, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.0, 0.5], [0.5, 0.5, 0.0], [0.5, 0.0, 0.0], [0.5, 0.5, 0.5]])),
])

def test_determine_atom_positions(grid, num_atoms, rng, expected_output, monkeypatch):
    """
    GIVEN a grid of available points and a number of atoms to populate

    WHEN we choose the positions for our atoms

    THEN we return the remaining grid points and a list of the positions chosen for these atoms

    Parameters
    ----------
    grid : float
        The list of possible points for atoms of this species.
    num_atoms : integer
        The number of atoms of this species.
    rng : NR_ran
        Random number generator - algorithm from Numerical Recipes 2007

    ---------------------------------------------------------------------------
    Paul Sharp 27/10/2017
    """

    # We do not want the test to rely on random input, therefore for the duration of the test we patch the rng as a
    # hard coded sequence of numbers. These numbers came from rng.int_range(u_lim=2) with seed=42
    mock_rng = mock.MagicMock(side_effect = [0, 2, 1, 2, 0, 1, 0, 0, 0])
    monkeypatch.setattr(rng, 'int_range', lambda l_lim, u_lim : mock_rng())

    assert chemdash.initialise.determine_atom_positions(grid, num_atoms, rng) == expected_output


#===========================================================================================================================================================
@pytest.mark.parametrize("grid, num_atoms", [
    ([[0.0, 0.0, 0.0], [0.0, 0.0, 0.5], [0.0, 0.5, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.0], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0], [0.5, 0.5, 0.5]], 9),
])

def test_determine_atom_positions_exception(grid, num_atoms, rng):
    """
    GIVEN a grid of available points and a number of atoms to populate that exceeds the number of available points

    WHEN we choose the positions for our atoms

    THEN we raise an assertion error

    Parameters
    ----------
    grid : float
        The list of possible points for atoms of this species.
    num_atoms : integer
        The number of atoms of this species.
    rng : NR_ran
        Random number generator - algorithm from Numerical Recipes 2007

    ---------------------------------------------------------------------------
    Paul Sharp 27/10/2017
    """

    with pytest.raises(AssertionError):
        chemdash.initialise.determine_atom_positions(grid, num_atoms, rng)


#===========================================================================================================================================================
@pytest.mark.parametrize("vacancy_points", [
    ([[0.0, 0.0, 0.0], [0.25, 0.25, 0.25], [0.75, 0.75, 0.75], [0.0, 0.5, 0.0], [0.25, 0.25, 0.75]]),
])

def test_populate_points_with_vacancies(STO_atoms, vacancy_points, STOX_structure):
    """
    GIVEN an atoms object and a list of points

    WHEN we add those points to the structure as vacancies

    THEN we return the atoms object with the list of points included as vacancies

    Parameters
    ----------
    STO_atoms : ase atoms
        The structure to which we will add vacancies.
    vacancy_points : float
        The list of points to be recorded as vacancies.

    ---------------------------------------------------------------------------
    Paul Sharp 26/10/2017
    """

    assert chemdash.initialise.populate_points_with_vacancies(STO_atoms, vacancy_points) == STOX_structure

    
#===========================================================================================================================================================
@pytest.mark.parametrize("vacancy_separation, exclusion_radius, expected_output", [
    (1.0, 1.0, [np.array([0.16666667, 0.16666667, 0.16666667]), np.array([0.16666667, 0.16666667, 0.83333333]), np.array([0.16666667, 0.5       , 0.16666667]),
                np.array([0.16666667, 0.5       , 0.5       ]), np.array([0.16666667, 0.5       , 0.83333333]), np.array([0.16666667, 0.83333333, 0.16666667]),
                np.array([0.16666667, 0.83333333, 0.5       ]), np.array([0.16666667, 0.83333333, 0.83333333]), np.array([0.5       , 0.16666667, 0.16666667]),
                np.array([0.5       , 0.16666667, 0.83333333]), np.array([0.5       , 0.5       , 0.16666667]), np.array([0.5, 0.5, 0.5]),
                np.array([0.5       , 0.5       , 0.83333333]), np.array([0.5       , 0.83333333, 0.5       ]), np.array([0.5       , 0.83333333, 0.83333333]),
                np.array([0.83333333, 0.16666667, 0.5       ]), np.array([0.83333333, 0.5       , 0.16666667]), np.array([0.83333333, 0.5       , 0.83333333])]),
])

def test_create_vacancy_grid(STO_atoms, vacancy_separation, exclusion_radius, expected_output):
    """
    GIVEN an atoms object, vacancy spacing and exclusion radius

    WHEN we create a vacancy grid

    THEN we return the a grid of points separated by the vacancy spacing, with ponits within the exclusion
         radius of an atom removed

    Parameters
    ----------
    STO_atoms : ase atoms
        The structure on which we will construct a vacancy grid.
    vacancy_separation : float
        The separation between vacancy points.
    exclusion radius : float
        Vacancies within this distance of an atom are removed from the grid.

    ---------------------------------------------------------------------------
    Paul Sharp 24/07/2019
    """

    assert all([x == pytest.approx(y) for x, y in zip(chemdash.initialise.create_vacancy_grid(STO_atoms, vacancy_separation, exclusion_radius), expected_output)])

#===========================================================================================================================================================
@pytest.mark.parametrize("vacancy_position, expected_output", [
    ([0.125, 0.125, 0.125], np.array([1.08972474, 0.8291562 , 1.29903811, 0.8291562 , 0.8291562])),
])

def test_get_distances_to_atoms(STO_atoms, vacancy_position, expected_output):
    """
    GIVEN an atoms object and a vacancy point

    WHEN we check the distances to atoms

    THEN we return a list of distacnes between the vacancy and each atom

    Parameters
    ----------
    STO_atoms : ase atoms
        The structure on which we will construct a vacancy grid.
    vacancy_position : float
        The position of the vacancy we are considering.

    ---------------------------------------------------------------------------
    Paul Sharp 24/07/2019
    """

    assert all([x == pytest.approx(y) for x, y in zip(chemdash.initialise.get_distances_to_atoms(STO_atoms, vacancy_position), expected_output)])


#===========================================================================================================================================================
@pytest.mark.parametrize("test_cif, atoms_data", [
    ('''data_image0
    _cell_length_a       2
    _cell_length_b       2
    _cell_length_c       2
    _cell_angle_alpha    90
    _cell_angle_beta     90
    _cell_angle_gamma    90

    _symmetry_space_group_name_H-M    "P 1"
    _symmetry_int_tables_number       1

    loop_
    _symmetry_equiv_pos_as_xyz
    'x, y, z'

    loop_
    _atom_site_label
    _atom_site_occupancy
    _atom_site_fract_x
    _atom_site_fract_y
    _atom_site_fract_z
    _atom_site_thermal_displace_type
    _atom_site_B_iso_or_equiv
    _atom_site_type_oxidation_number
    _atom_site_type_symbol
    Sr1      1.0000 0.75000  0.75000  0.25000  Biso   1.000  2.00000  Sr
    Ti1      1.0000 0.75000  0.25000  0.25000  Biso   1.000  4.00000  Ti
    O1       1.0000 0.50000  0.50000  0.50000  Biso   1.000  -2.00000  O
    O2       1.0000 0.50000  0.00000  0.00000  Biso   1.000  -2.00000  O
    O3       1.0000 0.00000  0.00000  0.50000  Biso   1.000  -2.00000  O''',
     ["Sr 1 +2.0", "Ti 1 +4.0", "O 3 -2.0"]),
    ('''data_image0
    _cell_length_a       2
    _cell_length_b       2
    _cell_length_c       2
    _cell_angle_alpha    90
    _cell_angle_beta     90
    _cell_angle_gamma    90

    _symmetry_space_group_name_H-M    "P 1"
    _symmetry_int_tables_number       1

    loop_
    _atom_site_label
    _atom_site_occupancy
    _atom_site_fract_x
    _atom_site_fract_y
    _atom_site_fract_z
    _atom_site_thermal_displace_type
    _atom_site_B_iso_or_equiv
    _atom_site_type_oxidation_number
    _atom_site_type_symbol
    Sr1      1.0000 0.75000  0.75000  0.25000  Biso   1.000  2.00000  Sr
    Ti1      1.0000 0.75000  0.25000  0.25000  Biso   1.000  4.00000  Ti
    O1       1.0000 0.50000  0.50000  0.50000  Biso   1.000  -2.00000  O
    O2       1.0000 0.50000  0.00000  0.00000  Biso   1.000  -2.00000  O
    O3       1.0000 0.00000  0.00000  0.50000  Biso   1.000  -2.00000  O

    loop_
    _symmetry_equiv_pos_as_xyz
    'x, y, z' ''',
     ["Sr 1 +2.0", "Ti 1 +4.0", "O 3 -2.0"]),
    ('''data_image0
    _cell_length_a       2
    _cell_length_b       2
    _cell_length_c       2
    _cell_angle_alpha    90
    _cell_angle_beta     90
    _cell_angle_gamma    90

    _symmetry_space_group_name_H-M    "P 1"
    _symmetry_int_tables_number       1

    loop_
    _symmetry_equiv_pos_as_xyz
    'x, y, z'

    loop_
    _atom_site_label
    _atom_site_occupancy
    _atom_site_fract_x
    _atom_site_fract_y
    _atom_site_fract_z
    _atom_site_thermal_displace_type
    _atom_site_B_iso_or_equiv
    _atom_site_type_symbol
    Sr1      1.0000 0.75000  0.75000  0.25000  Biso   1.000  Sr
    Ti1      1.0000 0.75000  0.25000  0.25000  Biso   1.000  Ti
    O1       1.0000 0.50000  0.50000  0.50000  Biso   1.000  O
    O2       1.0000 0.50000  0.00000  0.00000  Biso   1.000  O
    O3       1.0000 0.00000  0.00000  0.50000  Biso   1.000  O''',
     ["Sr 1 +2.0", "Ti 1 +4.0", "O 3 -2.0"]),
])

def test_initialise_from_cif(test_cif, atoms_data, STO_atoms, monkeypatch):
    """
    GIVEN a cif file and data from an atoms file

    WHEN we initialise a structure from the cif file

    THEN we return an atoms object with the data from the cif file and initial charges also set

    Parameters
    ----------
    cif_file : string
        The ".cif" file containing the initial structure.
    atoms_file_data : str, int, float
        The species, number of atoms, and charge for all atoms in the structure from the cif file.

    ---------------------------------------------------------------------------
    Paul Sharp 27/06/2019
    """

    # Need two patches here, first need to set ase.io.read to read the atoms object we want
    monkeypatch.setattr(ase.io, 'read', lambda x : 
                        ase.Atoms(symbols = "SrTiO3", cell = [2.0, 2.0, 2.0],
                                  scaled_positions = ([0.75, 0.75, 0.25], [0.75, 0.25, 0.25], [0.5, 0.5, 0.5], [0.5, 0.0, 0.0], [0.0, 0.0, 0.5]),
                                  pbc=[True, True, True]))

    # Then we need to ensure we use "mock_open" to return our test data instead of reading a file
    # Note the lack of lambda -- we want the mock itself, not its return value
    # We pass a blank string for the file argument for this reason
    monkeypatch.setattr('builtins.open', mock.mock_open(read_data = test_cif))
    
    structure = chemdash.initialise.initialise_from_cif("", atoms_data)
    
    # Class definition is that two ase Atomchemdash.initialise.initialise_from_cif("", atoms_data)s objects are the same if they have the same atoms, unit cell,
    # positions and boundary conditions. Therefore we need to test charges separately.
    assert structure == STO_atoms
    assert all([x == y for x, y in zip(structure.get_initial_charges(), STO_atoms.get_initial_charges())])


#===========================================================================================================================================================
@pytest.mark.parametrize("test_cif, atoms_data", [
    ('''data_image0
    _cell_length_a       2
    _cell_length_b       2
    _cell_length_c       2
    _cell_angle_alpha    90
    _cell_angle_beta     90
    _cell_angle_gamma    90

    _symmetry_space_group_name_H-M    "P 1"
    _symmetry_int_tables_number       1

    loop_
    _symmetry_equiv_pos_as_xyz
    'x, y, z'

    loop_
    _atom_site_label
    _atom_site_occupancy
    _atom_site_fract_x
    _atom_site_fract_y
    _atom_site_fract_z
    _atom_site_thermal_displace_type
    _atom_site_B_iso_or_equiv
    _atom_site_type_symbol
    Sr1      1.0000 0.75000  0.75000  0.25000  Biso   1.000  Sr
    Ti1      1.0000 0.75000  0.25000  0.25000  Biso   1.000  Ti
    O1       1.0000 0.50000  0.50000  0.50000  Biso   1.000  O
    O2       1.0000 0.50000  0.00000  0.00000  Biso   1.000  O
    O3       1.0000 0.00000  0.00000  0.50000  Biso   1.000  O''',
     ["Sr 1 +2.0", "Ti 1 +4.0", "O 1 -2.0"]),
])

def test_initialise_from_cif_exception(test_cif, atoms_data, STO_atoms, monkeypatch):
    """
    GIVEN a cif file and data from an atoms file which do not match

    WHEN we initialise a structure from the cif file

    THEN we raise an exception

    Parameters
    ----------
    cif_file : string
        The ".cif" file containing the initial structure.
    atoms_file_data : str, int, float
        The species, number of atoms, and charge for all atoms in the structure from the cif file.

    ---------------------------------------------------------------------------
    Paul Sharp 27/10/2017
    """

    # Need two patches here, first need to set ase.io.read to read the atoms object we want
    monkeypatch.setattr(ase.io, 'read', lambda x : 
                        ase.Atoms(symbols = "SrTiO3", cell = [2.0, 2.0, 2.0],
                                  scaled_positions = ([0.75, 0.75, 0.25], [0.75, 0.25, 0.25],
                                                      [0.5, 0.5, 0.5], [0.5, 0.0, 0.0], [0.0, 0.0, 0.5]),
                                  pbc=[True, True, True]))

    # Then we need to ensure we use "mock_open" to return our test data instead of reading a file
    # Note th lack of lambda -- we want the mock itself, not its return value
    # We pass a blank string for the file argument for this reason
    monkeypatch.setattr('builtins.open', mock.mock_open(read_data = test_cif))

    with pytest.raises(SystemExit):
        chemdash.initialise.initialise_from_cif("", atoms_data)


#===========================================================================================================================================================
@pytest.mark.parametrize("num_layers, expected_output", [
    (-5, []),
    (0, []),
    (1, ["A"]),
    (2, ["A", "B"]),
    (5, ["A", "B", "A", "C", "B"]),
])

def test_generate_random_stacking_sequence(num_layers, rng, expected_output, monkeypatch):
    """
    GIVEN a number of stacking layers

    WHEN we generate a random stacking sequence

    THEN we return a stacking sequence of the desired length with no repeated layers.

    Parameters
    ----------
    num_layers : int
        The number of anion layers in the stacking sequence.
    rng : NR_ran
        Random number generator - algorithm from Numerical Recipes 2007

    ---------------------------------------------------------------------------
    Paul Sharp 26/10/2017
    """

    # We do not want the test to rely on random input, therefore for the duration of the test we patch the rng as a
    # hard coded sequence of numbers. These numbers came from rng.int_range(u_lim=2) with seed=42
    mock_rng = mock.MagicMock(side_effect = [0, 0, 0, 1, 0])
    monkeypatch.setattr(rng, 'int_range', lambda l_lim, u_lim : mock_rng())

    assert chemdash.initialise.generate_random_stacking_sequence(num_layers, rng) == expected_output

#===========================================================================================================================================================
@pytest.mark.parametrize("stacking_sequence, grid_points, cell_setting, expected_output", [
    (["A", "B"], [2, 2, 2], "oblique", 
     ([[0.25, 0.25, 0.0], [0.25, 0.75, 0.0], [0.75, 0.25, 0.0], [0.75, 0.75, 0.0], [0.4166666666666667, 0.08333333333333333, 0.5],
       [0.4166666666666667, 0.5833333333333334, 0.5], [0.9166666666666667, 0.08333333333333333, 0.5], [0.9166666666666667, 0.5833333333333334, 0.5]],
      [[0.4166666666666667, 0.08333333333333333, 0.125], [0.4166666666666667, 0.5833333333333334, 0.125], [0.9166666666666667, 0.08333333333333333, 0.125],
       [0.9166666666666667, 0.5833333333333334, 0.125], [0.08333333333333333, 0.4166666666666667, 0.25], [0.08333333333333333, 0.9166666666666667, 0.25],
       [0.5833333333333334, 0.4166666666666667, 0.25], [0.5833333333333334, 0.9166666666666667, 0.25], [0.25, 0.25, 0.375], [0.25, 0.75, 0.375],
       [0.75, 0.25, 0.375], [0.75, 0.75, 0.375], [0.25, 0.25, 0.625], [0.25, 0.75, 0.625], [0.75, 0.25, 0.625], [0.75, 0.75, 0.625],
       [0.08333333333333333, 0.4166666666666667, 0.75], [0.08333333333333333, 0.9166666666666667, 0.75], [0.5833333333333334, 0.4166666666666667, 0.75],
       [0.5833333333333334, 0.9166666666666667, 0.75], [0.4166666666666667, 0.08333333333333333, 0.875], [0.4166666666666667, 0.5833333333333334, 0.875],
       [0.9166666666666667, 0.08333333333333333, 0.875], [0.9166666666666667, 0.5833333333333334, 0.875]])),
    (["A", "B", "C"], [1, 1, 3], "centred_rectangular",
     ([[0.5, 0.5, 0.0], [1.0, 1.0, 0.0], [0.5, 0.8333333333333334, 0.3333333333333333], [1.0, 1.3333333333333335, 0.3333333333333333],
       [0.5, 0.16666666666666666, 0.6666666666666666], [1.0, 0.6666666666666666, 0.6666666666666666]],
      [[0.5, 0.8333333333333334, 0.08333333333333333], [0.5, 0.16666666666666666, 0.16666666666666666], [0.5, 0.5, 0.25], [1.0, 1.3333333333333335, 0.08333333333333333],
       [1.0, 0.6666666666666666, 0.16666666666666666], [1.0, 1.0, 0.25], [0.5, 0.16666666666666666, 0.4166666666666667], [0.5, 0.5, 0.5],
       [0.5, 0.8333333333333334, 0.5833333333333334], [1.0, 0.6666666666666666, 0.4166666666666667], [1.0, 1.0, 0.5], [1.0, 1.3333333333333335, 0.5833333333333334],
       [0.5, 0.5, 0.75], [0.5, 0.8333333333333334, 0.8333333333333334], [0.5, 0.16666666666666666, 0.9166666666666666], [1.0, 1.0, 0.75],
       [1.0, 1.3333333333333335, 0.8333333333333334], [1.0, 0.6666666666666666, 0.9166666666666666]])),
])

def test_initialise_close_packed_grid_points(stacking_sequence, grid_points, cell_setting, expected_output):
    """
    GIVEN a close packed stacking sequence, cell setting and a set of grid points

    WHEN we initialise cation and anion grids according to the stacking sequence

    THEN we get two grids for anion and cation points

    Parameters
    ----------
    stacking_sequence : str
        List of anion layers that form a close packed stacking sequence.
    grid_points : integer
        List of the number of grid points along each dimension to form an (a, b, c) grid.
    cell_setting : str
        States whether we are using the primitive or orthorhombic cell settings.

    ---------------------------------------------------------------------------
    Paul Sharp 26/10/2017
    """

    assert chemdash.initialise.initialise_close_packed_grid_points(stacking_sequence, grid_points, cell_setting) == expected_output

#===========================================================================================================================================================
@pytest.mark.parametrize("stacking_sequence, grid_points, lattice", [
    (["A", "b"], [2, 2, 2], "oblique"),
    (["A", "D"], [2, 2, 2], "oblique"),
])

def test_initialise_close_packed_grid_points_exception_1(stacking_sequence, grid_points, lattice):
    """
    GIVEN an invalid close packed stacking sequence

    WHEN we initialise cation and anion grids according to the stacking sequence

    THEN we raise an assertion exception

    Parameters
    ----------
    stacking_sequence : str
        List of anion layers that form a close packed stacking sequence.
    grid_points : integer
        List of the number of grid points along each dimension to form an (a, b, c) grid.
    lattice : str
        The lattice used for anion layers in close packed grids.

    ---------------------------------------------------------------------------
    Paul Sharp 26/10/2017
    """

    with pytest.raises(AssertionError):
        chemdash.initialise.initialise_close_packed_grid_points(stacking_sequence, grid_points, lattice)

#===========================================================================================================================================================
@pytest.mark.parametrize("stacking_sequence, grid_points, lattice", [
    (["A", "B"], [2, 2, 2], "foobar"),
])

def test_initialise_close_packed_grid_points_exception_2(stacking_sequence, grid_points, lattice):
    """
    GIVEN an invalid close packed cell setting

    WHEN we initialise cation and anion grids according to the stacking sequence

    THEN we raise an exception

    Parameters
    ----------
    stacking_sequence : str
        List of anion layers that form a close packed stacking sequence.
    grid_points : integer
        List of the number of grid points along each dimension to form an (a, b, c) grid.
    lattice : str
        The lattice used for anion layers in close packed grids.

    ---------------------------------------------------------------------------
    Paul Sharp 27/10/2017
    """

    with pytest.raises(SystemExit):
        chemdash.initialise.initialise_close_packed_grid_points(stacking_sequence, grid_points, lattice)
