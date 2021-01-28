"""
|=============================================================================|
|                               C h e m D A S H                               |
|=============================================================================|
|                                                                             |
| This module contains the main code for performing structure prediction with |
| ChemDASH.                                                                   |
|                                                                             |
| This code deals with initialisation of the structures, swapping atoms in    |
| the structures, and evaluating the relaxed structures with the basin        |
| hopping method.                                                             |
|                                                                             |
| Contains                                                                    |
| --------                                                                    |
|     ChemDASH                                                                |
|     Structure                                                               |
|     Counts                                                                  |
|     write_output_file_header                                                |
|     optimise_structure                                                      |
|     update_potentials                                                       |
|     generate_new_structure                                                  |
|     strip_vacancies                                                         |
|     output_list                                                             |
|     report_rejected_structure                                               |
|     report_statistics                                                       |
|     read_restart_file                                                       |
|     write_restart_file                                                      |
|     search_local_neighbourhood                                              |
|                                                                             |
|-----------------------------------------------------------------------------|
| Paul Sharp 25/03/2020                                                       |
|=============================================================================|
"""

from builtins import range

import collections
import copy
import os
import subprocess
import sys
import time

import ase.io

import numpy as np

try:
    import chemdash.bonding as bonding
except ModuleNotFoundError:
    import bonding
    import gulp_calc
    import initialise 
    import inputs
    import neighbourhood 
    import rngs 
    import swap  
    import symmetry
    import vasp_calc
else:
    import chemdash.gulp_calc as gulp_calc
    import chemdash.initialise as initialise
    import chemdash.inputs as inputs
    import chemdash.neighbourhood as neighbourhood
    import chemdash.rngs as rngs
    import chemdash.swap as swap 
    import chemdash.symmetry as symmetry
    import chemdash.vasp_calc as vasp_calc


# =============================================================================
# =============================================================================
def ChemDASH(calc_name):
    """
    ChemDASH predicts the structure of new materials by placing atomic species
    at random positions on a grid and exploring the potential energy surface
    using a basin hopping approach.

    The structure is manipulated by swapping the positions of atoms.
    The swaps involve either: cations, anions, cations and anions, or cations,
    anions and vacancies, along with any custom group of atoms. We can swap any
    number of atoms per structure manipulation.

    The code makes extensive use of the Atomic Simulation Environment (ase),
    allowing us to perform structure relaxation using GULP and VASP.

    Parameters
    ----------
    calc_name : string
        The seed name for this calculation, which forms the name of the
        .input and .atoms files.

    Returns
    -------
    None

    ---------------------------------------------------------------------------
    Paul Sharp 25/03/2020
    """

    start_time = time.time()

    # Check that a ".input" file exists, stop execution if not
    input_file = calc_name + ".input"
    if (not os.path.isfile(input_file)):
        sys.exit('ERROR - there is no ".input" file')

    # Set parameters from input file
    params, errors = inputs.parse_input(input_file, calc_name)

    # Convert, and ensure all input file parameters are of the correct type
    params, errors = inputs.convert_values(params, errors)
    params, errors = inputs.handle_dependencies(params, errors)

    error_log = calc_name + ".error"

    if len(errors) > 0:
        inputs.report_input_file_errors(errors, input_file, error_log)
        sys.exit("Terminating execution - see error file {0}".format(error_log))

    with open(params["output_file"]["value"], mode="a") as output:

        write_output_file_header(output, params)

        # Override options if necessary
        if params["restart"]["value"] and (not os.path.isfile(params["restart_file"]["value"])):
            output.write('OVERRIDING OPTION -- "restart" is specified as True, but the restart file "{0}" file does not exist. Proceeding with restart = False.\n'.format(params["restart_file"]["value"]))
            output.write("\n")
            params["restart"]["value"] = False

        # Set GULP command or VASP script
        if params["calculator"]["value"] == "gulp":
            gulp_calc.set_gulp_command(params["gulp_executable"]["value"], params["calculator_cores"]["value"], params["calculator_time_limit"]["value"], "")

        elif params["calculator"]["value"] == "vasp":
            vasp_script = "run_vasp.py"
            vasp_calc.set_vasp_script(vasp_script, params["vasp_executable"]["value"], params["calculator_cores"]["value"], params["vasp_pp_dir"]["value"])

        else:
            sys.exit('ERROR - The calculator "{0}" is not currently supported in ChemDASH.'.format(params["calculator"]["value"]))

        # Set keywords and options for each stage of GULP/VASP calculations
        additional_inputs = {"gulp_keywords": [],
                             "gulp_options": [],
                             "gulp_max_gnorms": [],
                             "vasp_settings": []
                         }

        for i in range(1, params["num_calc_stages"]["value"] + 1):

            # GULP inputs
            additional_inputs["gulp_keywords"].append(params["gulp_calc_" + str(i) + "_keywords"]["value"])
            additional_inputs["gulp_options"].append(params["gulp_calc_" + str(i) + "_options"]["value"])
            additional_inputs["gulp_max_gnorms"].append(params["gulp_calc_" + str(i) + "_max_gnorm"]["value"])

            # VASP inputs
            additional_inputs["vasp_settings"].append(params["vasp_calc_" + str(i) + "_settings"]["value"])

        # Initialise and seed random number generator
        if not params["random_seed"]["specified"]:
            params["random_seed"]["value"] = rngs.generate_random_seed(params["seed_bits"]["value"])

        output.write("The seed for the random number generator is {0:d}\n".format(params["random_seed"]["value"]))

        rng = rngs.NR_Ran(params["random_seed"]["value"])
        rng.warm_up(params["rng_warm_up"]["value"])

        # =====================================================================
        # Initialise atoms -- either a new random setup or from a restart file

        # PS review -- can we package into classes here????????
        if params["restart"]["value"]:

            best_structure, current_structure, atomic_numbers_list,  positions_list, basins, outcomes, structure_count, structure_index = read_restart_file(params["restart_file"]["value"])

        else:

            # Check that a ".atoms" file exists, stop if not
            if not os.path.isfile(params["atoms_file"]["value"]):
                sys.exit('ERROR - the ".atoms" file "{0}" does not exist.'.format(params["atoms_file"]["value"]))

            atoms_data = initialise.read_atoms_file(params["atoms_file"]["value"])

            # Check that the structure is charge balanced, stop if not
            charge_balance = initialise.check_charge_balance(atoms_data)

            if charge_balance != 0:
                sys.exit('ERROR - the structure is not charge balanced. The overall charge is: {0}.'.format(charge_balance))

            if params["initial_structure_file"]["specified"]:

                output.write('The initial structure will be read from the file: "{0}".\n'.format(params["initial_structure_file"]["value"]))

                initial_atoms = initialise.initialise_from_cif(params["initial_structure_file"]["value"], atoms_data)

            else:

                output.write("The atoms will be initialised on a {0} grid.\n".format((params["grid_type"]["value"].replace("_", " "))))

                if params["grid_type"]["value"] == "close_packed":

                    # Generate a sequence of close packed anion layers if one has not been specified
                    if not params["cp_stacking_sequence"]["specified"]:
                    
                        params["cp_stacking_sequence"]["value"] = initialise.generate_random_stacking_sequence(params["grid_points"]["value"][2], rng)

                    # This allows us to define the anion sub-lattice of the close-packed
                    # grid in the input file, whilst fitting in with existing routines.
                    params["cell_spacing"]["value"][2] *= 0.5
                    output.write("The close packed stacking sequence is: {0}.\n".format("".join(params["cp_stacking_sequence"]["value"])))

                elif any(grid_type in params["grid_type"]["value"] for grid_type in ["orthorhombic", "rocksalt"]):

                    # Adjust the cell spacing to define according to full grid rather than anion grid
                    params["cell_spacing"]["value"][:] = [0.5 * x for x in params["cell_spacing"]["value"]]
                   

                # Set up atoms object and grid, and list of unoccupied points (all possible points)
                initial_cell, anion_grid, cation_grid = initialise.set_up_grids(params["grid_type"]["value"], params["grid_points"]["value"],
                                                                                params["cp_stacking_sequence"]["value"], params["cp_2d_lattice"]["value"])
                scaled_cell = initialise.scale_cell(initial_cell, params["cell_spacing"]["value"])
                initial_atoms, anion_grid, cation_grid = initialise.populate_grids_with_atoms(scaled_cell, anion_grid, cation_grid, atoms_data, rng)

                # Use unused points as vacancies if we are not using vacancy grids
                if not params["vacancy_grid"]["value"]:
                    initial_atoms = initialise.populate_points_with_vacancies(initial_atoms.copy(), anion_grid + cation_grid)

            # Set up "Structure" object for the structure currently under consideration
            current_structure = Structure(initial_atoms, 0)

            atomic_numbers_list = []
            positions_list = []

            basins = {}
            outcomes = {}

            structure_count = Counts()

            current_structure.write_cif("current.cif")

        output.write("The set of atoms used in this simulation is {0}.\n".format(strip_vacancies(current_structure.atoms.copy()).get_chemical_formula()))
        output.write("\n")

        # Set up ASE trajectory files -- we track all structures, as well as only accepted structures (relaxed and unrelaxed)
        if params["output_trajectory"]["value"]:

            all_traj = ase.io.trajectory.Trajectory(filename="all.traj", mode="w")
            all_relaxed_traj = ase.io.Trajectory(filename="all_relaxed.traj", mode="w")
            accepted_traj = ase.io.Trajectory(filename="accepted.traj", mode="w")
            accepted_relaxed_traj = ase.io.Trajectory(filename="accepted_relaxed.traj", mode="w")

        output.write("The cell parameters are:\n")
        output.write("\n")
        output.write(" a = \t{c[0]:.8f}\talpha = \t{c[3]:.8f}\n b = \t{c[1]:.8f}\tbeta  = \t{c[4]:.8f}\n c = \t{c[2]:.8f}\tgamma = \t{c[5]:.8f}\n".format(c=current_structure.atoms.get_cell_lengths_and_angles()))
        output.write("\n")

        # =====================================================================
        # Determine the swap groups are valid

        # Set default swap groups and weightings if none were given in the input
        if not params["swap_groups"]["specified"]:
            params["swap_groups"]["value"] = swap.initialise_default_swap_groups(current_structure.atoms, params["swap_groups"]["value"])

        swap_lengths = [len(group) for group in params["swap_groups"]["value"]]
        if all(length == 1 for length in swap_lengths):
            params["swap_groups"]["value"] = swap.initialise_default_swap_weightings(params["swap_groups"]["value"])

        # Check swap groups against the initial structure
        unique_elements = list(set(current_structure.atoms.get_chemical_symbols()))
        test_atoms = current_structure.atoms.copy()

        # Include vacancies in check if we are using vacancy grids
        if params["vacancy_grid"]["value"]:
            unique_elements += "X"
            test_atoms.extend(ase.Atoms("X", cell=test_atoms.get_cell(),
                                        charges=[0], pbc=[True, True, True]))

        custom_swap_group_errors = swap.check_elements_in_custom_swap_groups(params["swap_groups"]["value"], unique_elements)
        valid_swap_groups, verifying_swap_group_errors = swap.verify_swap_groups(test_atoms, params["swap_groups"]["value"])

        swap_group_errors = custom_swap_group_errors + verifying_swap_group_errors

        if len(swap_group_errors) > 0:
            inputs.report_input_file_errors(swap_group_errors, input_file, error_log)
            sys.exit("Terminating execution - see error file {0}".format(error_log))

        swap_group_names = [group[0] for group in params["swap_groups"]["value"]]
        no_valid_swap_groups = False

        output.write('We will consider the swap groups: {0}.\n'.format(', '.join([str(x) for x in swap_group_names])))
        output.write("\n")

        # =====================================================================
        # Determine whether or not the Bond Valence Sum and site potential need
        # to be, and can be, calculated.

        calc_bvs = False
        calc_pot = False

        if params["atom_rankings"]["value"] == "bvs" or params["atom_rankings"]["value"] == "bvs+":

            calc_bvs = True
            missing_bonds = bonding.check_R0_values(current_structure.atoms.copy())

            if len(missing_bonds) > 0:

                output.write("The Bond Valence Sum cannot be calculated for this structure because we do not have R0 values for the following bonds: {0}.\n".format("".join(missing_bonds)))

                calc_bvs = False

                output.write("OVERRIDING OPTION -- The atom rankings were specified to be determined by the Bond Valence Sum, but the Bond Valence Sum cannot be correctly calculated. Proceeding with atom rankings determined at random.")

                params["atom_rankings"]["value"] = "random"
                
        if params["atom_rankings"]["value"] == "site_pot" or params["atom_rankings"]["value"] == "bvs+":
            
            calc_pot = True

        # =====================================================================
        # Optimise first structure

        with open(params["energy_step_file"]["value"], mode="a") as energy_step, \
             open(params["energy_file"]["value"], mode="a") as energy_file, \
             open(params["bvs_file"]["value"], mode="a") as bvs_file, \
             open(params["potentials_file"]["value"], mode="a") as potentials_file, \
             open(params["potential_derivs_file"]["value"], mode="a") as derivs_file:

            open_files = [output, energy_step, energy_file, bvs_file, potentials_file, derivs_file]

            if not params["restart"]["value"]:

                output.write("Initial structure\n")

                visited, positions_list, atomic_numbers_list = swap.check_previous_structures(current_structure.atoms.copy(), positions_list, atomic_numbers_list)

                ##################

                if params["search_local_neighbourhood"]["value"]:
                    
                    current_structure = search_local_neighbourhood(current_structure, output, params)
                    
                ##################
                
                # Relax initial structure
                # current_structure.atoms = (current_structure.atoms.copy()).rattle(params["rattle_stdev"]["value"])
                current_structure, result, outcomes, relax_time = optimise_structure(current_structure, params, additional_inputs, 0, outcomes)

                output.write("This calculation is: {0}, {1} calculation time: {2:f}s\n".format(result, params["calculator"]["value"].upper(), relax_time))

                # Abort if first structure is not converged, or if GULP/VASP failed or has timed out.
                if params["converge_first_structure"]["value"] and result != "converged":

                    if result == params["calculator"]["value"] + " failure":
                        output.write("ERROR -- {0} has failed to perform an optimisation of the initial structure, aborting calculation.\n".format(params["calculator"]["value"].upper()))
                    if result == "timed out":
                        output.write("ERROR -- Optimisation of initial structure has timed out, aborting calculation.\n")
                    if result == "unconverged":
                        output.write("ERROR -- Optimisation of initial structure not achieved, aborting calculation.\n")

                    output.write("Time taken: {0:f}s\n".format(time.time() - start_time))
                    sys.exit("Terminating Execution")

                # Find symmetries in final structure

                # There is a bug here in spglib where something has a NoneType
                # The bug cannot be diagnosed, hence the try-except statement
                try:
                    current_structure.atoms = symmetry.symmetrise_atoms(current_structure.atoms)
                except TypeError:
                    pass

                if params["output_trajectory"]["value"]:

                    all_traj.write(atoms=strip_vacancies(initial_atoms.copy()))
                    all_relaxed_traj.write(atoms=strip_vacancies(current_structure.atoms.copy()))

                current_structure, basins = accept_structure(current_structure, params, output, energy_step,
                                                             result, basins, calc_bvs, calc_pot, energy_file,
                                                             bvs_file, potentials_file, derivs_file, initial_atoms)             

                # Set the current structure as the best structure found so far
                best_structure = Structure(current_structure.atoms.copy(), 0,
                                           current_structure.energy,
                                           current_structure.volume,
                                           current_structure.ranked_atoms,
                                           current_structure.bvs_atoms,
                                           current_structure.bvs_sites,
                                           current_structure.potentials,
                                           current_structure.derivs)

                # Accept structure if it converges, set dummy values if not
                if result == "converged":
                    
                    structure_count.zero_conv = 1
                    structure_count.converged += 1
                    
                    strip_vacancies(current_structure.atoms.copy()).write("structure_0.cif", format="cif")
                    output.write("The energy of structure {0:d} is {1:.8f} eV/atom\n".format(current_structure.index, current_structure.energy))

                    if params["output_trajectory"]["value"]:

                        accepted_traj.write(atoms=strip_vacancies(initial_atoms.copy()))
                        accepted_relaxed_traj.write(atoms=strip_vacancies(current_structure.atoms.copy()))

                else:

                    structure_count = report_rejected_structure(output, result, params["calculator"]["value"], structure_count)
                        
                # Output current and best structure
                current_structure.write_cif("current.cif")
                best_structure.write_cif("best.cif")

                output.write("\n")
                structure_index = 1

            # =================================================================
            # Basin Hopping loop

            output.write("Swapping atoms\n")
            total_structures = params["max_structures"]["value"]
            default_directed_num_atoms = params["directed_num_atoms"]["value"]

            if params["num_structures"]["specified"]:
                total_structures = min(structure_index + params["num_structures"]["value"], params["max_structures"]["value"])

            for i in range(structure_index, total_structures):

                output.write("\n")
                output.write("Structure {0:d}\n".format(i))

                # Write restart file
                write_restart_file(best_structure, current_structure, atomic_numbers_list, positions_list, basins, outcomes, structure_count, i)

                # Update all output files
                for f in open_files:
                    f.flush()

                # =============================================================
                # Generate the next structure by swapping atoms and vacancies

                # Check which swap groups remain valid for this structure
                valid_swap_groups, verifying_swap_group_errors = swap.verify_swap_groups(current_structure.atoms.copy(), params["swap_groups"]["value"])

                # End the simulation if there are no valid swap groups
                if len(valid_swap_groups) == 0:
                    no_valid_swap_groups = True
                    break
                
                new_atoms = generate_new_structure(current_structure.atoms.copy(), params, output,
                                                   valid_swap_groups, current_structure.ranked_atoms, rng)
                new_structure = Structure(new_atoms.copy(), i)

                # Check whether this proposed structure has been previously considered, if so try a new swap
                visited, positions_list, atomic_numbers_list = swap.check_previous_structures(new_structure.atoms.copy(), positions_list, atomic_numbers_list)

                # Minima hopping -- larger moves for repeated basins, smaller moves for new basins
                if visited:

                    output.write("This structure has been considered previously, and will therefore be rejected.\n")
                    structure_count.repeated += 1
                    params["pair_weighting"]["value"] /= params["pair_weighting_scale_factor"]["value"]
                    params["directed_num_atoms"]["value"] += params["directed_num_atoms_increment"]["value"]
                    continue

                else:

                    params["pair_weighting"]["value"] *= params["pair_weighting_scale_factor"]["value"]
                
                # =============================================================
                # Optimise the structure

                ##################

                if params["search_local_neighbourhood"]["value"]:

                    new_structure = search_local_neighbourhood(new_structure, output, params)
                    
                ##################
                
                new_structure, result, outcomes, relax_time = optimise_structure(new_structure, params, additional_inputs, i, outcomes)

                output.write("This calculation is: {0}, {1} calculation time: {2:f}s\n".format(result, params["calculator"]["value"].upper(), relax_time))

                if result != "converged":

                    structure_count = report_rejected_structure(output, result, params["calculator"]["value"], structure_count)
                    continue

                structure_count.converged += 1

                output.write("The energy of structure {0:d} is {1:.8f} eV/atom. The difference in energy is {2:f} eV/atom.\n".format(i, new_structure.energy, new_structure.energy - current_structure.energy))
                output.write("\n")

                # Find symmetries in final structure and output symmetrised unit cell as a ".cif" file

                # There is a bug here in spglib where something has a NoneType
                # The bug has not be diagnosed, hence the try-except statement
                try:
                    new_structure.atoms = symmetry.symmetrise_atoms(new_structure.atoms.copy())
                except TypeError:
                    pass
                
                strip_vacancies(new_structure.atoms.copy()).write("structure_" + str(i) + ".cif", format="cif")

                if params["output_trajectory"]["value"]:

                    all_traj.write(atoms=strip_vacancies(new_atoms.copy()))
                    all_relaxed_traj.write(atoms=strip_vacancies(new_structure.atoms.copy()))

                # =============================================================
                # Determine whether or not the swap should be accepted

                if structure_count.converged == 0 or swap.accept_swap(current_structure.energy, new_structure.energy, params["temp"]["value"], rng):

                    output.write("The swap is accepted.\n")

                    params["directed_num_atoms"]["value"] = default_directed_num_atoms

                    if structure_count.converged > 0:
                        energy_step.write("{0:d} {1:.8f} {2:.8f}\n".format(i, current_structure.energy, current_structure.volume))

                    structure_count.accepted += 1
                    params["temp"]["value"] /= params["temp_scale_factor"]["value"]

                    current_structure = Structure(new_structure.atoms.copy(), i,
                                                  new_structure.energy,
                                                  new_structure.volume,
                                                  new_structure.ranked_atoms,
                                                  new_structure.bvs_atoms,
                                                  new_structure.bvs_sites,
                                                  new_structure.potentials,
                                                  new_structure.derivs)

                    current_structure, basins = accept_structure(current_structure, params, output,
                                                                 energy_step, result, basins,
                                                                 calc_bvs, calc_pot, energy_file,
                                                                 bvs_file, potentials_file, derivs_file,
                                                                 new_atoms)
                    current_structure.write_cif("current.cif")

                    if params["output_trajectory"]["value"]:

                        accepted_traj.write(atoms=strip_vacancies(new_atoms.copy()))
                        accepted_relaxed_traj.write(atoms=strip_vacancies(new_structure.atoms.copy()))
                    
                    # Keep track of best structure
                    if current_structure.energy < best_structure.energy:

                        best_structure = Structure(new_structure.atoms.copy(), i,
                                                   current_structure.energy,
                                                   current_structure.volume,
                                                   current_structure.ranked_atoms,
                                                   current_structure.bvs_atoms,
                                                   current_structure.bvs_sites,
                                                   current_structure.potentials,
                                                   current_structure.derivs)

                        best_structure.write_cif("best.cif")

                else:

                    output.write("The swap is rejected.\n")
                    params["temp"]["value"] *= params["temp_scale_factor"]["value"]

            # When loop is finished, write final part of outputs
            output_list(potentials_file, "{0:d}".format(total_structures - 1), current_structure.potentials)
            output_list(derivs_file, "{0:d}".format(total_structures - 1), current_structure.derivs)
            output.write("\n")

        # =====================================================================
        # Finish up

        write_restart_file(best_structure, current_structure, atomic_numbers_list,
                           positions_list, basins, outcomes, structure_count, total_structures)

        structures_considered = total_structures
        
        if no_valid_swap_groups:
            output.write("The supplied swap groups are no longer valid for this structure.\n")
            structures_considered = i
        elif total_structures == params["max_structures"]["value"]:
            output.write("Swapping complete.\n")
        else:
            output.write("Requested number of structures considered.\n")

        report_statistics(output, basins, outcomes, structure_count, structures_considered, params["calculator"]["value"])

        output.write("The best structure is structure {0:d}, with energy {1:.8f} eV/atom and volume {2:.8f} A0^3, and has been written to best.cif\n".format(best_structure.index, best_structure.energy, best_structure.volume))

        output.write("\n")
        output.write("Time taken: {0:f}s\n".format(time.time() - start_time))

    return None


# =============================================================================
# =============================================================================
class Structure(object):
    """
    Stores data referring to particular structures.

    ---------------------------------------------------------------------------
    Paul Sharp 10/04/2019
    """

    # =========================================================================
    def __init__(self, atoms, index, energy=0.0, volume=0.0, ranked_atoms={},
                 bvs_atoms=[], bvs_sites=[], potentials=[], derivs=[]):
        """
        Initialise the Structure data.

        Parameters
        ----------
        atoms : ase atoms
            The atoms object containing the structure.
        index : int
            The number of basin hopping moves to get to this structure.
        energy : float
            The energy of the structure.
        volume : float
            The volume of the structure.
        ranked_atoms : int
            Lists of integers ranking the atoms according to BVS/site potential
            for each atomic species in the structure.
        bvs_atoms : float
            The value of the Bond Valence Sum for each atom in the structure.
        bvs_sites : float
            The value of the Bond Valence Sum for each sites in the structure, with every type of 
            atom present in each site.
        potentials : float
            Site potential for each atom in the current structure.
        derivs : float
            Resolved derivatives of the site potentials for each atom in the current structure.

        Returns
        -------
        None

        -----------------------------------------------------------------------
        Paul Sharp 10/04/2019
        """

        self.atoms = atoms
        self.index = index
        self.energy = energy
        self.volume = volume
        self.ranked_atoms = ranked_atoms
        self.bvs_atoms = bvs_atoms
        self.bvs_sites = bvs_sites
        self.potentials = potentials
        self.derivs = derivs

    # =========================================================================
    def write_cif(self, filename):
        """
        Write the structure to a cif file.

        Parameters
        ----------
        filename : str
            The name of the cif file.

        Returns
        -------
        None

        -----------------------------------------------------------------------
        Paul Sharp 03/08/2017
        """

        self.atoms.write(filename, format="cif")


# =============================================================================
# =============================================================================
class Counts(object):
    """
    Keeps track of the number of structures that achieve certain outcomes.

    ---------------------------------------------------------------------------
    Paul Sharp 09/08/2017
    """

    # =========================================================================
    def __init__(self, accepted = 0, converged = 0, unconverged = 0, repeated = 0, timed_out = 0, zero_conv = 0):
        """
        Initialise the counts.

        Parameters
        ----------
        accepted : int
            Number of accepted basin hopping moves.
        accepted : int
            Number of converged structures.
        failed : int
            Number of failed structures.
        repeated : int
            Number of moves resulting in repeated structures.
        timed_out : int
            Number of structures that time out in GULP.
        zero_conv : int
            1 if the first structure converged, 0 otherwise.

        Returns
        -------
        None

        -----------------------------------------------------------------------
        Paul Sharp 09/08/2017
        """

        self.accepted = accepted
        self.converged = converged
        self.unconverged = unconverged
        self.repeated = repeated
        self.timed_out = timed_out
        self.zero_conv = zero_conv


# =============================================================================
# =============================================================================
def write_output_file_header(output, params):
    """
    Write the header for the start of the run in the ChemDASH output file

    Parameters
    ----------
    output : file
        The open file object for the ChemDASH output file.
    params : dict
        Dictionary containing each ChemDASH parameter with its value.

    Returns
    -------
    None

    ---------------------------------------------------------------------------
    Paul Sharp 26/03/2020
    """

    line_chars = 80
    title_chars = int(0.5 * (line_chars - 10))

    output.write("\n")
    output.write("#" * line_chars + "\n")
    output.write("#" + " " * title_chars + "ChemDASH" + " " * title_chars + "#\n")
    output.write("#" * line_chars + "\n")
    output.write("\n")

    output.write("Summary of Inputs\n")
    output.write("\n")
    output.write("-" * line_chars + "\n")

    sorted_keywords = sorted(params.keys())
    for keyword in sorted_keywords:

        if params[keyword]["specified"]:
            value = params[keyword]["value"]
 
            # Check for lists, output them as comma separated values
            if not isinstance(value, str) and isinstance(value, collections.Sequence):

                # Check for lists of lists, output them as comma separated values
                if value: # Checks list is non-empty
                    
                    if not isinstance(value[0], str) and isinstance(value[0], collections.Sequence):
                        output.write("{0:30} = {1}\n".format(keyword, ', '.join(sorted([str(', '.join([str(y) for y in x])) for x in value]))))
                    else:
                        output.write("{0:30} = {1}\n".format(keyword, ', '.join(sorted([str(x) for x in value]))))

            # Write dictionaries as comma separated "key: value" pairs
            elif isinstance(value, dict):
                output.write("{0:30} = {1}\n".format(keyword, ', '.join(sorted(['{0}: {1}'.format(key, val) for (key, val) in value.items()]))))

            else:
                output.write("{0:30} = {1}\n".format(keyword, value))

    output.write("-" * line_chars + "\n")
    output.write("\n")

    return None


# =============================================================================
def optimise_structure(structure, params, additional_inputs, structure_index, outcomes):
    """
    This routine optimises the input structure using the chosen calculator.

    Parameters
    ----------
    structure : ChemDASH structure
        The ChemDASH structure class containing ASE atoms object and properties.
    params : dict
        Dictionary containing each ChemDASH parameter with its value.
    additional_inputs : dict
        Dictionary containing calculator inputs for each individual stage of the optimisation.
    structure_index : int
        The index for the structure being considered - used to label the calculator output files.
    outcomes : dict
        Dictionary of the different GULP outcomes and the number of times they occured.

    Returns
    -------
    structure : ChemDASH structure
        The ChemDASH structure with atomic positions and unit cell parameters optimised.
    result : string
        The result of the calculation,
        either "converged", "unconverged", "[calculator] failure", or "timed out"
    outcomes : dict
        Updated dictionary of the different GULP outcomes and the number of times they occured.
    time : float
        Time taken for the optimisation.

    ---------------------------------------------------------------------------
    Paul Sharp 15/06/2020
    """

    if params["calculator"]["value"] == "gulp":

        gulp_files = ["structure_" + str(structure_index) + "_" + suffix for suffix in params["gulp_files"]["value"]]
        structure, result, outcome, time, = gulp_calc.multi_stage_gulp_calc(structure, params["num_calc_stages"]["value"], gulp_files, params["gulp_keywords"]["value"], additional_inputs["gulp_keywords"], params["gulp_options"]["value"], additional_inputs["gulp_options"], additional_inputs["gulp_max_gnorms"], params["gulp_shells"]["value"], params["gulp_library"]["value"])

        outcomes = gulp_calc.update_outcomes(outcome, outcomes)

    elif params["calculator"]["value"] == "vasp":

        vasp_file = "structure_" + str(structure_index) + ".vasp"
        structure, result, time = vasp_calc.multi_stage_vasp_calc(structure, params["num_calc_stages"]["value"], vasp_file, params["vasp_settings"]["value"], additional_inputs["vasp_settings"], params["vasp_max_convergence_calcs"]["value"], params["save_outcar"]["value"])

    return structure, result, outcomes, time


# =============================================================================
def accept_structure(structure, params, output, energy_step, result, basins,
                     calc_bvs, calc_pot, energy_file, bvs_file, potentials_file,
                     derivs_file, unrelaxed_atoms):
    """
    Update ChemDASH records when a new structure is accepted.

    Parameters
    ----------
    structure : ChemDASH structure
        The ChemDASH structure class containing ASE atoms object and properties.
    params : dict
        Dictionary containing each ChemDASH parameter with its value.
    output : file
        The open file object for the ChemDASH output file.
    energy_step : file
        The open file object for the ChemDASH energy step file.
    result : string
        The result of the GULP/VASP calculation,
        either "converged", "unconverged", "[calculator] failure", or "timed out"
    basins : dict
        Value of energy for each basin visited, and the number of times each
        basin was visited.
    calc_bvs, calc_pot : boolean
        True if BVS/site potential values need to be calculated.
    energy_file : file
        The open file object for the ChemDASH energy file.
    bvs_file, potentials_file, derivs_file : string
        Files recording BVS/site potential values.
    unrelaxed_atoms : ASE atoms
        ASE atoms object for the structure before relaxation.

    Returns
    -------
    structure : ChemDASH structure
        The ChemDASH structure with atomic positions and unit cell parameters optimised.
    basins : dict
        Value of energy for each basin visited, and the number of times each
        basin was visited.

    ---------------------------------------------------------------------------
    Paul Sharp 26/03/2020
    """
    
    # Set up vacancy grids if we are using optimised geometries.
    if params["update_atoms"]["value"] and params["vacancy_grid"]["value"]:

        # Set up vacancy grid and update atoms in current structure if we are using this optimised geometry
        vacancy_grid = initialise.create_vacancy_grid(strip_vacancies(structure.atoms.copy()), params["vacancy_separation"]["value"], params["vacancy_exclusion_radius"]["value"])
        structure.atoms = initialise.populate_points_with_vacancies(strip_vacancies(structure.atoms.copy()), vacancy_grid)

    # Structures will usually have converged, this covers the case where
    # the initial structure does not converge but the run continues.
    if result == "converged":

        energy_file.write("{0:d} {1:.8f} {2:.8f}\n".format(structure.index, structure.energy, structure.volume))
        energy_step.write("{0:d} {1:.8f} {2:.8f}\n".format(structure.index, structure.energy, structure.volume))

        basins = swap.update_basins(basins, structure.energy)

        if calc_bvs:

            structure.bvs_atoms = bonding.bond_valence_sum_for_atoms(structure.atoms.copy())
            structure.bvs_sites = bonding.bond_valence_sum_for_sites(structure.atoms.copy())
            output_list(bvs_file, structure.index, structure.bvs_sites)

            atom_indices = [atom.index for atom in structure.atoms if atom.symbol != "X"]
            desired_atoms = swap.find_desired_atoms(structure.bvs_sites, atom_indices)

        # Obtain values for the site potential of all atoms and vacancies in the relaxed structure if necessary.
        if calc_pot:

            pot_structure = Structure(structure.atoms, structure.index)        
            structure.potentials, structure.derivs, _, _, _ = update_potentials(pot_structure, params["gulp_library"]["value"])

        output_list(potentials_file, structure.index, structure.potentials)
        output_list(derivs_file, structure.index, structure.derivs)

        # Set swap rankings for each atom
        structure = swap.update_atom_rankings(structure, params["atom_rankings"]["value"])

    # If we use not using optimised geometries, reset the geometry to that of the unrelaxed structure and set up vacancy grid.
    if not params["update_atoms"]["value"]:

        structure.atoms = unrelaxed_atoms.copy()

        if params["vacancy_grid"]["value"]:

            # Set up vacancy grid and in current structure if we are using the unoptimised geometry
            vacancy_grid = initialise.create_vacancy_grid(strip_vacancies(structure.atoms.copy()), params["vacancy_separation"]["value"], params["vacancy_exclusion_radius"]["value"])
            structure.atoms = initialise.populate_points_with_vacancies(strip_vacancies(structure.atoms.copy()), vacancy_grid)

    return structure, basins
    

# =============================================================================
def update_potentials(structure, gulp_lib):
    """
    Re-calculate the potentials for the ChemDASH structure.

    Parameters
    ----------
    structure : ChemDASH structure
        The ChemDASH structure class containing ASE atoms object and properties.
    gulp_lib : string
        Name of the file containing the GULP forcefield.

    Returns
    -------
    structure : ChemDASH structure
        The ChemDASH structure class with updated potentials.
    result : string
        The result of the calculation,
        either "converged", "unconverged", "GULP failure", or "timed out"
    outcomes : dict
        Updated dictionary of the different GULP outcomes and the number of times they occured.
    time : float
        Time taken for the site potential calculation.

    ---------------------------------------------------------------------------
    Paul Sharp 09/04/2019
    """

    structure, result, outcome, time = gulp_calc.multi_stage_gulp_calc(structure,
                                                                       1, ["pot"],
                                                                       "sing pot",
                                                                       [""], [""],
                                                                       [[""]],
                                                                       [""], [],
                                                                       gulp_lib,
                                                                       remove_vacancies=False)

    return structure.potentials, structure.derivs, result, outcome, time


# =============================================================================
def generate_new_structure(atoms, params, output, valid_swap_groups,
                           sorted_atomic_indices, rng):
    """
    Generate a new structure by swapping atoms.

    Parameters
    ----------
    atoms : ASE atoms
        The ASE atoms object for the current structure.
    params : dict
        Dictionary containing each ChemDASH parameter with its value.
    output : file
        The open file object for the ChemDASH output file.
    valid_swap_groups : list
        List of all valid swap groups with their weightings.
    sorted_atomic_indices : dict
        Dictionary of indices for the each atomic species, sorted by the values
        in the chosen ranking list.
    rng : NR_ran
        Random number generator - algorithm from Numerical Recipes 2007.


    Returns
    -------
    new_atoms : ASE atoms
        The ASE atoms object for the new structure.

    ---------------------------------------------------------------------------
    Paul Sharp 26/03/2020
    """

    swap_weightings = [group[1] for group in valid_swap_groups]
    atom_group = valid_swap_groups[rng.weighted_choice(swap_weightings)][0]
    elements_list, max_swaps = swap.determine_maximum_swaps(atoms.copy(), atom_group)

    num_swaps = swap.choose_number_of_atoms_to_swap(max_swaps, params["number_weightings"]["value"],
                                                    rng, params["pair_weighting"]["value"])

    output.write("We will attempt to perform a swap of {0:d} atoms from the group: {1}\n".format(num_swaps, atom_group))

    selection_pool, num_swaps = swap.generate_selection_pool(elements_list, num_swaps)
    output.write("A valid selection pool has been generated to swap {0:d} atoms.\n".format(num_swaps))

    if params["atom_rankings"]["value"] != "random":
        output.write("We will consider an extra {0:d} atoms at the top of the ranking list for each species.\n".format(params["directed_num_atoms"]["value"]))

    output.write("The value of kT is {0:f} eV/atom".format(params["temp"]["value"]))

    swap_list = swap.generate_swap_list(selection_pool, num_swaps, rng)
    
    new_atoms, swap_text = swap.swap_atoms(atoms.copy(), swap_list, copy.deepcopy(sorted_atomic_indices),
                                           params["directed_num_atoms"]["value"], params["initial_structure_file"]["specified"],
                                           params["vacancy_exclusion_radius"]["value"], rng, params["force_vacancy_swaps"]["value"])
        
    if params["verbosity"]["value"] == "verbose":
        output.write("\n")
        for entry in swap_text:
            output.write("The {e[0]} atom at {e[1]} (atom {e[2]:d}) has been replaced by a {e[3]} atom\n".format(e=entry))

    output.write("\n")

    return new_atoms


# =============================================================================
def strip_vacancies(structure):
    """
    This code removes vacancies, represented by "X" atoms, from an ase structure.

    Parameters
    ----------
    structure : ase atoms
        A structure that includes vacancies represented as an "X" atom.

    Returns
    -------
    structure : ase atoms
        The structure with vacancies removed.

    ---------------------------------------------------------------------------
    Paul Sharp 02/05/2017
    """

    del structure[[atom.index for atom in structure if atom.symbol == "X"]]

    return structure


# =============================================================================
def output_list(file_object, header, list_of_floats):
    """
    This routine outputs a list of floats to a file, keeping the values in the
    floating point format if possible.

    Parameters
    ----------
    file_object : file
        An open file object
    header : string
        A string that preceeds the list of floating point values.
    list_of_floats : float
        The list of floats we wish to output.

    Returns
    -------
    None

    ---------------------------------------------------------------------------
    Paul Sharp 05/06/2020
    """

    file_object.write(str(header))

    for element in list_of_floats:
        try:
            file_object.write(" {0:.8f}".format(element))
        except ValueError:
            file_object.write(" {0}".format(element))

    file_object.write("\n")

    return None


# =============================================================================
def report_rejected_structure(output, result, calculator, structure_count):
    """
    Write the reason for a rejection of a structure to the output file.

    Parameters
    ----------
    output : file
        The open file object for the ChemDASH output file.
    result : string
        The result of the structural optimisation.
    calculator : string
        The name of the materials modelling code used for the optimisation.
    structure_count : Counts
        Number of accepted, converged, unconverged/failed, and repeated
        structures and timed out calculations considered so far.

    Returns
    -------
    structure_count : Counts
        Updated number of accepted, converged, unconverged/failed, and
        repeated structures and timed out calculations considered so far.

    ---------------------------------------------------------------------------
    Paul Sharp 26/03/2020
    """

    if result == calculator + " failure":
        output.write("{0} has failed to perform the calculation for this structure, so it will be rejected.\n".format(calculator.upper()))

    if result == "timed out":
        output.write("The optimisation of this structure has timed out, so it will be rejected.\n")
        structure_count.timed_out += 1

    if result == "unconverged":
        output.write("The optimisation of this structure has not converged, so it will be rejected.\n")
        structure_count.unconverged += 1

    return structure_count


# =============================================================================
def report_statistics(output, basins, outcomes, structure_count,
                      total_structures, calculator):
    """
    This routine outputs the visited basins, GULP outcomes and output file.

    Parameters
    ----------
    output : file
        An open file object
    basins : dict
        Value of energy for each basin visited, and the number of times each
        basin was visited.
    outcomes : dict
        Outcome of each GULP calculation.
    structure_count : Counts
        Number of accepted, converged, unconverged/failed, and repeated
        structures and timed out GULP calculations considered so far.
    total_structures : int
        Number of structures considered so far.
    calculator : str
        The materials modelling code used for structure relaxation.

    Returns
    -------
    None

    ---------------------------------------------------------------------------
    Paul Sharp 26/03/2020
    """

    line_chars = 80
    basin_chars = 20

    # Report record of visited basins
    output.write("\n")
    output.write("In total, {0:d} basins were visited\n".format(len(basins)))
    output.write("\n")

    output.write("Basins:\n")
    output.write("-" * basin_chars + "\n")
    for energy in basins:
        output.write("| {0:.5f} | {1:4d} |\n".format(energy, basins[energy]))

    output.write("-" * basin_chars + "\n")
    output.write("\n")

    # Report convergence rate
    unique_structures = total_structures - structure_count.repeated
    try:
        convergence_rate = 100.0 * float(structure_count.converged) / float(unique_structures)
    except ZeroDivisionError:
        convergence_rate = 0.0

    output.write("{0:d} structures out of {1:d} converged, so the convergence rate is {2:f}%\n".format(structure_count.converged, unique_structures, convergence_rate))
    output.write("\n")

    # Report results of optimisations
    output.write("Optimisation Results:\n")
    output.write("-" * line_chars + "\n")
    output.write("| {0:4d} | Converged\n".format(structure_count.converged))
    output.write("| {0:4d} | Unconverged\n".format(structure_count.unconverged))
    output.write("| {0:4d} | Timed Out\n".format(structure_count.timed_out))
    output.write("-" * line_chars + "\n")
    output.write("\n")

    # Report GULP outcomes
    if calculator == "gulp":

        output.write("GULP Outcomes:\n")
        output.write("-" * line_chars + "\n")

        for gulp_outcome in outcomes:
            output.write("| {0:4d} | {1}\n".format(outcomes[gulp_outcome], gulp_outcome))

        output.write("-" * line_chars + "\n")
        output.write("\n")

    # Report swap acceptance rate
    try:
        swap_acceptance = 100.0 * float(structure_count.accepted) / float(structure_count.converged - structure_count.zero_conv)
    except ZeroDivisionError:
        swap_acceptance = 0.0

    output.write("Of the converged structures, {0:d} out of {1:d} swaps were accepted, so the swap acceptance rate was {2:f}%\n".format(structure_count.accepted, structure_count.converged-structure_count.zero_conv, swap_acceptance))
    output.write("\n")

    return None


# =============================================================================
def read_restart_file(restart_file):
    """
    This routine reads a ".npz" file with all information needed to restart a
    calculation.

    Parameters
    ----------
    restart_file : str
       Name of the ".npz" archive.

    Returns
    -------
    best_structure, current_structure : Structure
        ASE atoms object and associated data for the best and current structures.
    atomic_numbers_list : int
        Atomic numbers of each atom for all unique structures considered so far.
    positions_list : float
        Positions of each atom for all unique structures considered so far.
    basins : dict
        Value of energy for each basin visited, and the number of times each basin was visited.
    outcomes : dict
        Outcome of each GULP calculation.
    structure_count : Counts
        Number of accepted, converged, unconverged/failed, and repeated
        structures and timed out GULP calculations considered so far.
    structure_index : int
        Number of structures considered so far.

    ---------------------------------------------------------------------------
    Paul Sharp 10/04/2019
    Chris Collins 07/12/2020
    """

    with np.load(restart_file,allow_pickle=True) as restart_data:

        best_structure = restart_data["best_structure"][()]
        current_structure = restart_data["current_structure"][()]

        atomic_numbers_list = restart_data["atomic_numbers_list"].tolist()
        positions_list = restart_data["positions_list"].tolist()

        # Final index needed to return a dictionary as opposed to an array
        basins = restart_data["basins"][()]
        outcomes = restart_data["outcomes"][()]

        structure_count = restart_data["structure_count"][()]
        structure_index = int(restart_data["structure_index"])

    return best_structure, current_structure, atomic_numbers_list, positions_list, basins, outcomes, structure_count, structure_index


# =============================================================================
def search_local_neighbourhood(structure, output, params):
    """
    LCN algorithm.

    Parameters
    ----------
    structure : ChemDASH Structure
    output : file
    params : dict
        

    Returns
    -------
    structure : ChemDASH Structure
        

    ---------------------------------------------------------------------------
    Paul Sharp 26/03/2020
    """

    lcn_structure = Structure(structure.atoms, structure.index)
    
    lcn_structure, _, _, _ = gulp_calc.multi_stage_gulp_calc(lcn_structure, 1, ["lcn"], "sing", [""], [""], [[""]], [""], [], params["gulp_library"]["value"])
    lcn_initial_energy = lcn_structure.energy
    output.write("\n")
    output.write("LCN {0:d}: Initial energy = {1:.8f} eV/atom\n".format(structure.index, lcn_structure.energy))
    lcn_structure, lcn_final_energy, lcn_time, atom_loops, sp_calcs = neighbourhood.local_combinatorial_neighbourhood(lcn_structure,
                                                                                                                      params["neighbourhood_atom_distance_limit"]["value"],
                                                                                                                      params["num_neighbourhood_points"]["value"],
                                                                                                                      params["gulp_library"]["value"])
    
    output.write("LCN {0:d}: Final energy = {1:.8f} eV/atom\n".format(structure.index, lcn_final_energy))
    output.write("LCN {0:d}: Change in energy = {1:.8f} eV/atom\n".format(structure.index, lcn_final_energy - lcn_initial_energy))
    output.write("LCN {0:d}: Time = {1:.8f}s\n".format(structure.index, lcn_time))
    output.write("LCN {0:d}: Number of atom loops = {1:d}\n".format(structure.index, atom_loops))
    output.write("LCN {0:d}: Number of single-point energy calculations = {1:d}\n".format(structure.index, sp_calcs))
    output.write("LCN {0:d} Data: {1:.8f} {2:.8f} {3:d} {4:d}\n".format(structure.index, lcn_final_energy - lcn_initial_energy, lcn_time, atom_loops, sp_calcs))
    output.write("\n")
    
    structure.atoms = lcn_structure.atoms
    
    return structure


# =============================================================================
def write_restart_file(best_structure, current_structure, atomic_numbers_list,
                       positions_list, basins, outcomes, structure_count,
                       structure_index):
    """
    This routine writes the file with all information needed to restart a calculation.

    Parameters
    ----------
    best_structure, current_structure : Structure
        ASE atoms object and associated data for the best and current structures.
    atomic_numbers_list : int
        Atomic numbers of each atom for all unique structures considered so far.
    positions_list : float
        Positions of each atom for all unique structures considered so far.
    basins : dict
        Value of energy for each basin visited, and the number of times each
        basin was visited.
    outcomes : dict
        Outcome of each GULP calculation.
    structure_count : Counts
        Number of accepted, converged, unconverged/failed, and repeated
        structures and timed out GULP calculations considered so far.
    structure_index : int
        Number of structures considered so far.

    Returns
    -------
    None

    ---------------------------------------------------------------------------
    Paul Sharp 10/04/2019
    """

    np.savez("restart",
             best_structure=best_structure,
             current_structure=current_structure,
             atomic_numbers_list=atomic_numbers_list,
             positions_list=positions_list,
             basins=basins,
             outcomes=outcomes,
             structure_count=structure_count,
             structure_index=structure_index)

    return None




############ NEED TO INCLUDE TRAJECTORIES IN RESTART FILE #############

############ ALSO OVERRIDE OUTPUT_TRAJECTORY WHEN VERSION OF ASE IS INSUFFICIENT #################
