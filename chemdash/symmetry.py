"""
|=============================================================================|
|                                S Y M M E T R Y                              |
|=============================================================================|
|                                                                             |
| This module contains a routine that finds the symmetries in an Atoms object |
| using spglib.                                                               |
|                                                                             |
| Contains                                                                    |
| --------                                                                    |
|     symmetrise_atoms                                                        |
|                                                                             |
|-----------------------------------------------------------------------------|
| Paul Sharp 20/01/2020                                                       |
|=============================================================================|
"""

from builtins import range

import ase

try:
    import spglib
except ImportError:
    from pyspglib import spglib


# =============================================================================
# =============================================================================
def symmetrise_atoms(atoms):
    """
    Use spglib in order to find a standardised unit cell from an atoms object
    and return an atoms object with the standardised unit cell.

    Parameters
    ----------
    atoms : ase atoms
        The atoms object containing the structure for which we wish to find symmetries.

    Returns
    -------
    symmetrised_atoms : ase atoms
        The atoms object after finding symmetries.

    ---------------------------------------------------------------------------
    Paul Sharp 20/01/2020
    """

    # Get inputs for spglib from ASE atoms object
    spglib_cell = (atoms.get_cell(),
                   atoms.get_scaled_positions(),
                   atoms.get_atomic_numbers())
    
    lattice, scaled_pos, atomic_numbers = spglib.standardize_cell(spglib_cell)

    # The structure returned from spglib may have a different number of atoms
    # from the original structure. In this case we cannot insert the charges
    # correctly so we continue with the original structure.
    try:
        symmetrised_atoms = ase.Atoms(cell=lattice,
                                      scaled_positions=scaled_pos,
                                      numbers=atomic_numbers,
                                      charges=atoms.get_initial_charges(),
                                      pbc=[True, True, True])
    except ValueError:
        symmetrised_atoms = atoms

    return symmetrised_atoms
