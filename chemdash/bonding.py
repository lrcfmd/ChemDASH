"""
|=============================================================================|
|                                 B O N D I N G                               |
|=============================================================================|
|                                                                             |
| This module contains routines that analyse the bonding environments of each |
| atom in a structure.                                                        |
|                                                                             |
| References: I. D. Brown & D. Altermatt, Acta Cryst., B41, 244-247, (1985).  |
|                                                                             |
| Contains                                                                    |
| --------                                                                    |
|     bond_valence_sum_for_atoms                                              |
|     bond_valence_sum_for_sites                                              |
|     compute_bond_valence_sum                                                |
|     check_R0_values                                                         |
|                                                                             |
|-----------------------------------------------------------------------------|
| Paul Sharp 09/12/2019                                                       |
|=============================================================================|
"""

from builtins import range
from builtins import zip

import ase.neighborlist
import math
import numpy as np


# Initialise dictionary containing R0 values for every bond type
R0_values = {"Ag1--O-2": 1.842,
             "Ag1--S-2": 2.119,
             "Al3--Cl-1": 2.032,
             "Al3--F-1": 1.545,
             "Al3--O-2": 1.651,
             "As3--O-2": 1.789,
             "As3--S-2": 2.272,
             "As5--F-1": 1.620,
             "As5--O-2": 1.767,
             "B3--F-1": 1.281,
             "B3--O-2": 1.371,
             "Ba2--F-1": 2.188,
             "Ba2--O-2": 2.285,
             "Ba2--S-2": 2.769,
             "Be2--F-1": 1.281,
             "Be2--O-2": 1.381,
             "Bi3--O-2": 2.094,
             "Bi3--S2": 2.570,
             "C4--N-3": 1.442,
             "C4--O-2": 1.390,
             "Ca2--F-1": 1.842,
             "Ca2--O-2": 1.967,
             "Cd2--Cl-1": 2.212,
             "Cd2--O-2": 1.904,
             "Cd2--S-2": 2.304,
             "Cl7--O-2": 1.632,
             "Co2--Cl-1": 2.033,
             "Co2--O-2": 1.692,
             "Co3--C2": 1.634,
             "Cr3--F-1": 1.657,
             "Cr3--O-2": 1.724,
             "Cr6--O-2": 1.794,
             "Cs1--Cl-1": 2.791,
             "Cs1--O-2": 2.417,
             "Cu1--I-1": 2.108,
             "Cu1--S-2": 1.898,
             "Cu2--F-1": 1.594,
             "Cu2--O-2": 1.679,
             "Cu2--S-2": 2.054,
             "D1--O-2": 0.927,
             "Dy3--O-2": 2.001,
             "Er3--F-1": 1.904,
             "Er3--O-2": 1.988,
             "Eu2--S-2": 2.584,
             "Eu3--O-2": 2.074,
             "Fe2--O-2": 1.734,
             "Fe3--C2": 1.689,
             "Fe3--F-1": 1.679,
             "Fe3--O-2": 1.759,
             "Fe3--S-2": 2.149,
             "Ga3--O-2": 1.730,
             "Ga3--S-2": 2.163,
             "Ge4--O-2": 1.748,
             "Ge4--S-2": 2.217,
             "H1--N-3": 0.885,
             "H1--O-2": 0.882,
             "Hg2--O-2": 1.972,
             "Hg2--S-2": 2.308,
             "Ho3--O-2": 2.025,
             "I5--O-2": 2.003,
             "In3--F-1": 1.792,
             "In3--O-2": 1.902,
             "In3--S-2": 2.370,
             "K1--Cl-1": 2.519,
             "K1--F-1": 1.992,
             "K1--O-2": 2.132,
             "La3--O-2": 2.172,
             "La3--S-2": 2.643,
             "Li1--F-1": 1.360,
             "Li1--O-2": 1.466,
             "Mg2--F-1": 1.578,
             "Mg2--O-2": 1.693,
             "Mn2--Cl-1": 2.133,
             "Mn2--F-1": 1.698,
             "Mn2--O-2": 1.790,
             "Mn3--O-2": 1.760,
             "Mn4--O-2": 1.753,
             "Mo6--O-2": 1.907,
             "N3--O-2": 1.361,
             "N5--O-2": 1.432,
             "Na1--F-1": 1.677,
             "Na1--O-2": 1.803,
             "Na1--S-2": 2.300,
             "Nb5--O-2": 1.911,
             "Nd3--O-2": 2.105,
             "Ni2--F-1": 1.596,
             "Ni2--O-2": 1.654,
             "P5--N-3": 1.704,
             "P5--O-2": 1.617,
             "P5--S-2": 2.110,
             "Pb2--O-2": 2.112,
             "Pb2--S-2": 2.541,
             "Pb4--O-2": 2.042,
             "Pr3--O-2": 2.138,
             "Pt2--C2": 1.760,
             "Pt4--O-2": 1.879,
             "Rb1--Cl-1": 2.652,
             "Rb1--O-2": 2.263,
             "S2--N-2": 1.597,
             "S2--N-3": 1.682,
             "S4--N-3": 1.762,
             "S4--O-2": 1.644,
             "S6--O-2": 1.624,
             "Sb3--F-1": 1.883,
             "Sb3--O-2": 1.973,
             "Sb3--S-2": 2.474,
             "Sb5--F-1": 1.797,
             "Sb5--O-2": 1.942,
             "Sc3--O-2": 1.849,
             "Sc3--S-2": 2.321,
             "Se4--O-2": 1.811,
             "Se6--O-2": 1.788,
             "Si4--C-4": 1.883,
             "Si4--N-3": 1.724,
             "Si4--O-2": 1.624,
             "Si4--S-2": 2.126,
             "Sn2--F-1": 1.925,
             "Sn4--Cl-1": 2.276,
             "Sn4--F-1": 1.843,
             "Sn4--O-2": 1.905,
             "Sn4--S-2": 2.399,
             "Sr2--O-2": 2.118,
             "Ta5--O-2": 1.920,
             "Tb3--O-2": 2.032,
             "Te4--O-2": 1.977,
             "Te6--O-2": 1.917,
             "Th4--F-1": 2.068,
             "Ti4--O-2": 1.815,
             "Tl1--I-1": 2.822,
             "Tl1--S-2": 2.545,
             "U4--F-1": 2.038,
             "U6--O-2": 2.075,
             "V3--O-2": 1.743,
             "V4--O-2": 1.784,
             "V5--O-2": 1.803,
             "W6--O-2": 1.917,
             "Y3--O-2": 2.019,
             "Yb3--O-2": 1.965,
             "Zn2--Cl-1": 2.027,
             "Zn2--O-2": 1.704,
             "Zr4--F-1": 1.846,
             "Zr4--O-2": 1.928,
             #
             #
             "Al3--S-2": 2.13,
             "Li1--S-2": 1.94,
}


# =============================================================================
# =============================================================================
def bond_valence_sum_for_atoms(structure, neighbour_dist=5.0, B=0.37):
    """
    Compute the Bond Valence Sum for all atoms in a structure.

    The Bond Valence Sum is given by V_i=sum_j(S_ij), with the bond valence
    given by S_ij=exp[(R_0-R_ij)/B].

    The values of the parameters R_0 and B are taken from:
    I. D. Brown & D. Altermatt, Acta Cryst., B41, 244-247, (1985).

    Parameters
    ----------
    structure : ase atoms
        The atoms object containing the structure for which we wish to compute
        the bond valence sum.
    neighbour_dist : float
        Distance between which two atoms are defined as neighbours.
        Default is 5.0 A0, corresponding to two atoms within 10.0 A0.
    B : float
        Parameter in the expression for the experimental bond valence.
        Default is 0.37, taken from the reference.

    Returns
    -------
    bond_valence_sums : float
        The value of the Bond Valence Sum for each atom.

    ---------------------------------------------------------------------------
    Paul Sharp 30/07/2019
    """

    atom_indices = [atom.index for atom in structure if atom.symbol != "X"]

    # Initialise list in order to ensure BVS is ordered in the same way as the atoms.
    bond_valence_sums = [None] * len(structure)

    # Treat cations and anions separately due to bipartite bonding
    for i in atom_indices:

        neighbour_indices, distances = get_distances_to_neighbours(structure, i, neighbour_dist)
        bond_valence_sums[i] = compute_bond_valence_sum(structure, i, neighbour_indices, distances, B)

    return bond_valence_sums


# =============================================================================
def bond_valence_sum_for_sites(structure, neighbour_dist=5.0, B=0.37):
    """
    Compute the Bond Valence Sum for all vacancy sites in the
    structure, trying each atom type in the site.

    The Bond Valence Sum is given by V_i=sum_j(S_ij), with the bond valence
    given by S_ij=exp[(R_0-R_ij)/B].

    The values of the parameters R_0 and B are taken from:
    I. D. Brown & D. Altermatt, Acta Cryst., B41, 244-247, (1985).

    Parameters
    ----------
    structure : ase atoms
        The atoms object containing the structure for which we wish to compute
        the bond valence sum.
    neighbour_dist : float
        Distance between which two atoms are defined as neighbours.
        Default is 5.0 A0, corresponding to two atoms within 10.0 A0.
    B : float
        Parameter in the expression for the experimental bond valence.
        Default is 0.37, taken from the reference.

    Returns
    -------
    bond_valence_sums : float
        The value of the Bond Valence Sum for each site with each atom type.

    ---------------------------------------------------------------------------
    Paul Sharp 30/07/2019
    """

    # Initialise list in order to ensure BVS is ordered in the same way as the atoms.
    bond_valence_sums = [None] * len(structure)

    # Construct dictionary keys
    reduced_structure = structure.copy()
    del reduced_structure[[atom.index for atom in reduced_structure if atom.symbol == "X"]]
    atom_symbols = list(set(reduced_structure.get_chemical_symbols()))

    # Determine charges -- we use the modal charge for atoms of that element in the structure
    charges = {}
    
    for atom in atom_symbols:
        charge_list = [reduced_structure[i].charge for i in range(len(reduced_structure)) if reduced_structure[i].symbol == atom]
        charges[atom] = max(charge_list, key=charge_list.count)

    # Calculate BVS for each site
    for i in range(0, len(structure)):

        bond_valence_sum = {}
        neighbour_indices, distances = get_distances_to_neighbours(structure, i, neighbour_dist)

        atom_symbol = structure[i].symbol
        atom_charge = structure[i].charge
        
        for atom in atom_symbols:

            structure[i].symbol = atom      
            structure[i].charge = charges[atom]

            atom_key = atom + "/" + str(charges[atom])
            bond_valence_sum[atom_key] = compute_bond_valence_sum(structure, i, neighbour_indices, distances, B)
            
        bond_valence_sums[i] = bond_valence_sum

        structure[i].symbol = atom_symbol
        structure[i].charge = atom_charge
       
    return bond_valence_sums


# =============================================================================
def compute_bond_valence_sum(structure, bvs_atom, bonded_atoms, distances, B=0.37):
    """
    Compute the Bond Valence Sum for an atom in a structure.

    The Bond Valence Sum is given by V_i=sum_j(S_ij), with the bond valence
    given by S_ij=exp[(R_0-R_ij)/B].

    The values of the parameters R_0 and B are taken from:
    I. D. Brown & D. Altermatt, Acta Cryst., B41, 244-247, (1985).

    Parameters
    ----------
    structure : ase atoms
        The atoms object containing the atoms for which we wish to compute the bond valence.
    bvs_atom_index : int
        The index of the atom for which we wish to calculate the Bond Valence Sum.
    bonded_atoms : int
        The indices of the atoms in the structure which the BVS atom may be bonded to,
        which we need to calculate the Bond Valence Sum.
    distance : float
        The distance between the BVS atom and all of the atoms it may be bonded to in the structure.
    B : float
        Parameter in the expression for the experimental bond valence.
        Default is 0.37, taken from the reference.

    Returns
    -------
    bond_valence : float
        The value of the Bond Valence Sum for the atom of interest.

    ---------------------------------------------------------------------------
    Paul Sharp 27/02/2019
    """

    bond_valence_sum = 0.0
    bvs_atom_string = structure[bvs_atom].symbol + str(int(structure[bvs_atom].charge))
    
    for index, bond_atom in enumerate(bonded_atoms):

        # Determine bond
        if structure[bvs_atom].charge > structure[bond_atom].charge:
            bond_string = bvs_atom_string + "--" + structure[bond_atom].symbol + str(int(structure[bond_atom].charge))
        else:
            bond_string = structure[bond_atom].symbol + str(int(structure[bond_atom].charge)) + "--" + bvs_atom_string

        # Find value of R0 parameter and calculate bond valence
        # Key errors will occur for neighbours that are like ions.
        try:
            R0 = R0_values[bond_string]
        except KeyError:
            pass
        else:
            bond_valence_sum += math.exp((R0 - distances[index]) / B)

    return bond_valence_sum


# =============================================================================
def get_distances_to_neighbours(structure, bvs_atom, neighbour_dist=5.0):
    """
    Find the set of distances between the atom we are calculating the bond
    valence sum for and all of its neighbours.

    By using an ase neighbour list, we account for atoms appearing more than
    once due to the main atom being a neighbour of multiple periodic images of
    the same atom.

    Parameters
    ----------
    structure : ase atoms
        The atoms object containing the structure for which we wish to compute
        the bond valence sum.
    bvs_atom : int
        The index of the atom for which we wish to calculate the Bond Valence Sum.
    neighbour_dist : float
        Distance between which two atoms are defined as neighbours.
        Default is 5.0 A0, corresponding to two atoms within 10.0 A0.

    Returns
    -------
    neighbour_indices : int
        The indices of each neighbouring atom -- if a periodic image is also
        a neighbour then its index will appear more than once.
    distances : float
        The distances from to the appropriate periodic image of each neighbour.

    ---------------------------------------------------------------------------
    Paul Sharp 30/07/2019
    """

    cell = structure.get_cell()
    symbols = structure.get_chemical_symbols()
    
    # Construct neighbour list
    nl = ase.neighborlist.NeighborList([neighbour_dist] * len(structure), self_interaction=False, bothways=True)
    nl.update(structure)
    full_indices, full_offsets = nl.get_neighbors(bvs_atom)

    # Remove vacancies from list of neighbours
    neighbour_indices, offsets = list(zip(*[(i, o) for i, o in zip(full_indices, full_offsets) if symbols[i] != "X"]))
    
    # Find distances -- the offset indicates which periodic image the neighbouring atom is in.
    distances = []
    pos = structure.positions[bvs_atom]

    for i, offset in zip(neighbour_indices, offsets):

        neighbour_pos = structure.positions[i] + np.dot(offset, cell)
        distances.append(np.sqrt((pos[0]-neighbour_pos[0])**2 +
                                 (pos[1]-neighbour_pos[1])**2 +
                                 (pos[2]-neighbour_pos[2])**2))

    return neighbour_indices, distances


# =============================================================================
def check_R0_values(structure):
    """
    Check that an R0 value exists for every cation-anion bond in the structure.

    The values of the parameters R_0 and B are taken from I. D. Brown & D. Altermatt, Acta Cryst., B41, 244-247, (1985).

    Parameters
    ----------
    structure : ase atoms
        The atoms object containing the structure for which we intend to calculate the Bond Valence Sum

    Returns
    -------
    missing_bonds : str
        List of the bonds in the structures for which there is no R0 value.

    ---------------------------------------------------------------------------
    Paul Sharp 19/10/2017
    """

    missing_bonds = []
    del structure[[atom.index for atom in structure if atom.symbol == "X"]]

    cation_symbols = [atom.symbol + str(int(atom.charge)) for atom in structure if atom.charge > 0.0]
    anion_symbols = [atom.symbol + str(int(atom.charge)) for atom in structure if atom.charge < 0.0]

    cation_strings = list(set(cation_symbols))
    anion_strings = list(set(anion_symbols))

    for cation_string in cation_strings:
       
        for anion_string in anion_strings:

            bond_string = cation_string + "--" + anion_string

            if bond_string not in R0_values:

                missing_bonds.append(bond_string)

    return missing_bonds
