import pytest
import chemdash.symmetry

import ase

#===========================================================================================================================================================
#===========================================================================================================================================================
#Tests

# Tests are: 5 atom perovskite stays same, 5 atom non-perovskite goes to perovskite, 10 atom ilmenite stays same (rather then going to 30 atoms)
@pytest.mark.parametrize("structure, expected_output", [
    (ase.Atoms(symbols='O3SrTi', cell=[[3.95198873, 0.00000000, 0.00000000], [9.55906259e-04, 3.95375364, 0.00000000], [6.30880988e-04, 8.66206579e-04, 3.95960629]],
               scaled_positions=[[0.43799, 0.03559, 0.56411], [0.93733, 0.03698, 0.06473], [0.43782, 0.53618, 0.06366], [0.93562, 0.53759, 0.56504], [0.43718, 0.0366, 0.06438]],
               charges = [-2.0, -2.0, -2.0, 2.0, 4.0], pbc=[True, True, True]),
     ase.Atoms(symbols='O3SrTi', cell=[[3.95199, 0.00000, 0.00000], [0.00096, 3.95375, 0.00000], [0.00063, 0.00087, 3.95961]],
               scaled_positions=[[0.43799, 0.03559, 0.56411], [0.93733, 0.03698, 0.06473], [0.43782, 0.53618, 0.06366], [0.93562, 0.53759, 0.56504], [0.43718, 0.0366, 0.06438]],
               charges = [-2.0, -2.0, -2.0, 2.0, 4.0], pbc=[True, True, True])),
    (ase.Atoms(symbols='O3SrTi', cell=[[5.59498, 0.0, 0.0], [-2.79358065929833, 4.84291712446066, 0.0], [-0.008048261527840725, -6.460226438557249, 2.283346036511398]],
               scaled_positions=[[0.5976, 0.72372, 0.16171], [0.09965, 0.22705, 0.16438], [0.09836, 0.7242, 0.16202], [0.60197, 0.23139, 0.16701], [0.59942, 0.22662, 0.6638]],
               charges = [-2.0, -2.0, -2.0, 2.0, 4.0], pbc=[True, True, True]),
     ase.Atoms(symbols='O3SrTi', cell=[[3.95199, 0.00000, 0.00000], [0.00096, 3.95375, 0.00000], [0.00063, 0.00087, 3.95961]],
               scaled_positions=[[0.43799, 0.03559, 0.56411], [0.93733, 0.03698, 0.06473], [0.43782, 0.53618, 0.06366], [0.93562, 0.53759, 0.56504], [0.43718, 0.0366, 0.06438]],
               charges = [-2.0, -2.0, -2.0, 2.0, 4.0], pbc=[True, True, True])),
    (ase.Atoms("O6Sr2Ti2", cell=[[4.693916, 2.391504, 1.260338], [1.530938, 7.504244, 3.338840], [0.156070, 2.825644, 5.697025]],
               scaled_positions=([0.392947, 0.774012, 0.636917], [0.192144, 0.363774, 0.047155], [0.536893, 0.496920, 0.292168], [0.803185, 0.162970, 0.247959],
                                 [0.147934, 0.296117, 0.492971], [0.947131, 0.885878, 0.903210], [0.937930, 0.696000, 0.502172], [0.402148, 0.963891, 0.037954],
                                 [0.349283, 0.490323, 0.090820], [0.990795, 0.169567, 0.449307]),
               charges = [-2.0, -2.0, -2.0, -2.0, -2.0, -2.0, 2.0, 2.0, 4.0, 4.0], pbc=[True, True, True]),
     ase.Atoms("O6Sr2Ti2", cell=[[4.69392, 2.39150, 1.26034], [1.53094, 7.50424, 3.33884], [0.15607, 2.82564, 5.69702]],
               scaled_positions=[[0.39295, 0.77401, 0.63692], [0.19214, 0.36377, 0.04715], [0.53689, 0.49692, 0.29217], [0.80318, 0.16297, 0.24796], [0.14793, 0.29612, 0.49297],
                                 [0.94713, 0.88588, 0.90321], [0.93793, 0.69600, 0.50217], [0.40215, 0.96389, 0.03795], [0.34928, 0.49032, 0.09082], [0.99080, 0.16957, 0.44931]],
               charges = [-2.0, -2.0, -2.0, -2.0, -2.0, -2.0, 2.0, 2.0, 4.0, 4.0], pbc=[True, True, True]))
])

def test_symmetrise_atoms(structure, expected_output):
    """
    GIVEN an atoms object

    WHEN we use spglib to transform the unit cell to its standard setting

    THEN we return a structure with the same number of atoms and the unit cell transformed to its standard setting

    Parameters
    ----------
    structure : ase atoms
        The structure we are looking to transform to the standard setting.

    Returns
    -------
    None  

    ---------------------------------------------------------------------------
    Paul Sharp 16/01/2020
    """

    # We cannot simply run the routine and compare the answer, because spglib works to a large number of decimal places that
    # can easily be affected by machine precision etc. Therefore, we round off the atomic positions and cell parameters to 5 d.p.
    # and then compare to what we expect.
    symm_structure = chemdash.symmetry.symmetrise_atoms(structure)

    rounded_cell = [[round(vec[0], 5), round(vec[1], 5), round(vec[2], 5)] for vec in symm_structure.get_cell()]
    rounded_positions = [[round(pos[0], 5), round(pos[1], 5), round(pos[2], 5)] for pos in symm_structure.get_scaled_positions()]

    symm_structure.set_cell(rounded_cell)
    symm_structure.set_scaled_positions(rounded_positions) 


    # Class definition is that two ase Atoms objects are the same if they have the same atoms, unit cell,
    # positions and boundary conditions. Therefore we need to test charges separately.
    assert symm_structure.get_chemical_symbols() == expected_output.get_chemical_symbols()

    symm_structure_flat_cell = [x for vec in symm_structure.get_cell() for x in vec]
    expected_output_flat_cell = [x for vec in expected_output.get_cell() for x in vec] 
    assert all([x == pytest.approx(y) for x, y in zip(symm_structure_flat_cell, expected_output_flat_cell)])
    
    symm_structure_flat_pos = [x for pos in symm_structure.get_scaled_positions() for x in pos]
    expected_output_flat_pos = [x for pos in expected_output.get_scaled_positions() for x in pos]
    assert all([x == pytest.approx(y) for x, y in zip(symm_structure_flat_pos, expected_output_flat_pos)])

    assert list(symm_structure.get_pbc()) == list(expected_output.get_pbc())
   
    assert all([x == y for x, y in zip(symm_structure.get_initial_charges(), expected_output.get_initial_charges())])
