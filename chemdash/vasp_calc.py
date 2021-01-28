"""
|=============================================================================|
|                             V A S P   C A L C                               |
|=============================================================================|
|                                                                             |
| This module contains routines that run custom VASP calculations, and check  |
| the status of VASP calculations.                                            |
|                                                                             |
| Contains                                                                    |
| --------                                                                    |
|     multi_stage_vasp_calc                                                   |
|     run_vasp                                                                |
|     set_vasp_script                                                         |
|     check_float                                                             |
|     converged_in_one_scf_cycle                                              |
|     populate_points_with_vacancies                                          |
|                                                                             |
|-----------------------------------------------------------------------------|
| Paul Sharp 27/03/2020                                                       |
|=============================================================================|
"""

from builtins import range

from ase.calculators.vasp import Vasp
import ase
import os
import shutil
import time


# =============================================================================
# =============================================================================
def multi_stage_vasp_calc(structure, num_calcs, vasp_file, main_settings,
                          additional_settings, max_convergence_calcs,
                          save_outcar):
    """
    Run a VASP calculation that consists of many stages, which can involve different settings.
    The final calculation is run until it converges in one SCF cycle.

    Parameters
    ----------
    structure : ChemDASH Structure
        The structure class containing the ase atoms object to be used in the VASP calculation.
    num_calcs : integer
        The number of VASP calculation stages.
    vasp_file : str
        Output file concatenating all OSZICAR files for this calculation.
    main_settings : dict
        List of VASP settings for all stages of the calculation.
    additional_settings : dict
        List of extra VASP settings for individual stages of the calculation.
    max_convergence_calcs : int
        The maximum number of VASP calculations run for the final stage -- after this the calculation is considered unconverged.
    save_outcar : bool
        If true, retains final OUTCAR file as "OUTCAR_[structure_index]".

    Returns
    -------
    atoms : ase atoms
        The final structure after the calculation.
    result : string
        The result of the VASP calculation,
        either "converged", "unconverged", "vasp failure", or "timed out"
    energy : float
        Energy of the structure in eV.
    calc_time : float
        Time taken for this VASP calculation.

    ---------------------------------------------------------------------------
    Paul Sharp 27/03/2020
    """

    assert len(additional_settings) >= num_calcs, 'ERROR in vasp_calc.multi_stage_vasp_calc() -- number of VASP calculations is set to {0:d}, but only {1:d} sets of additional settings are provided.'.format(num_calcs, len(additional_settings))

    continue_to_final_stage = True
    vasp_start_time = time.time()

    # Strip out X atoms from the structure - they are only there to mark vacancies and have no POTCAR file.
    vacancy_positions = determine_vacancy_positions(structure.atoms.copy())
    del structure.atoms[[atom.index for atom in structure.atoms if atom.symbol == "X"]]

    with open(vasp_file, mode="w") as vasp_out:

        for i in range(0, num_calcs - 1):
            
            vasp_settings = main_settings.copy()
            vasp_settings.update(additional_settings[i])

            structure.atoms, structure.energy, result = run_vasp(structure.atoms, vasp_settings)

            if os.path.isfile("OSZICAR"):
                with open("OSZICAR", mode="r") as oszicar_file:
                    shutil.copyfileobj(oszicar_file, vasp_out)

            # Continue only if the calculation was successfully performed
            # without timing out, or if the energy has blown up such that
            # it is not representable as a float
            if result == "vasp failure":

                continue_to_final_stage = False
                break

            if not check_float(structure.energy):

                result = "unconverged"
                continue_to_final_stage = False
                break

            if converged_in_one_scf_cycle("OUTCAR"):

                result = "converged"
                continue_to_final_stage = False
                break

            # Copy CONTCAR to POSCAR for next stage of calculation -- use "copy2()" because it copies metadata and permissions
            if os.path.isfile("CONTCAR"):
                shutil.copy2("CONTCAR", "POSCAR")

        if continue_to_final_stage:

            # With calculation completed up to the penultimate stage, run final stage of calculation until convergence within a single SCF cycle is achieved
            vasp_settings = main_settings.copy()

            # Add additional settings, but allow for zero calc stages.
            try:
                vasp_settings.update(additional_settings[-1])
            except IndexError:
                pass

            for i in range(0, max_convergence_calcs):

                structure.atoms, structure.energy, result = run_vasp(structure.atoms, vasp_settings)

                if os.path.isfile("OSZICAR"):
                    with open("OSZICAR", mode="r") as oszicar_file:
                        shutil.copyfileobj(oszicar_file, vasp_out)

                # If energy is not representable as a float, calculation is unconverged.
                if not check_float(structure.energy):

                    result = "unconverged"

                if converged_in_one_scf_cycle("OUTCAR"):

                    result = "converged"

                # Exit loop if the result of the calculation is determined -- whatever it may be
                if result != "":
                    break

                # Copy CONTCAR to POSCAR for next stage of calculation -- use "copy2()" because it copies metadata and permissions
                if os.path.isfile("CONTCAR"):
                    shutil.copy2("CONTCAR", "POSCAR")

            # If we exit the loop normally having run out of convergence calculations, the overall calculation is unconverged
            else:

                result = "unconverged"

    # Convert energy to units of eV/atom
    try:
        structure.energy /= float(len(structure.atoms))
    except (TypeError, ValueError):
        pass

    # Remove files ready for a new calculation
    files_to_remove = ["CONTCAR", "CHGCAR", "WAVECAR", "XDATCAR"]

    for rm_file in files_to_remove:

        try:
            os.remove(rm_file)
        except OSError:
            pass

    # Retain final OUTCAR file if desired
    if save_outcar:
        os.rename("OUTCAR", "OUTCAR_" + str(structure.index))
        
    # Replace the vacancies in the atoms object
    structure.atoms = populate_points_with_vacancies(structure.atoms, vacancy_positions)
    structure.volume = structure.atoms.get_volume()
    
    calc_time = time.time() - vasp_start_time
    
    return structure, result, calc_time


# =============================================================================
def run_vasp(structure, vasp_settings):
    """
    Run a single VASP calculation using an ASE calculator.
    This routine will make use of WAVECAR and CONTCAR files if they are available.

    Parameters
    ----------
    structure : ASE atoms
        The structure used in the VASP calculation
    vasp_settings : dict
        The set of VASP options to apply with their values.

    Returns
    -------
    structure : ase atoms
        The structure after performing this calculation.
    energy : float
        Energy of the structure in eV.
    result : string
        The result of the VASP calculation,
        either "converged", "unconverged", "vasp failure", or "timed out"

    ---------------------------------------------------------------------------
    Paul Sharp 25/09/2017
    """

    # Set files for calculation
    energy = 0.0
    result = ""

    # Use ASE calculator -- the use of **kwargs in the function call allows us to set the desired arguments using a dictionary
    structure.set_calculator(Vasp(**vasp_settings))

    # Run calculation, and consider any exception to be a VASP failure
    try:
        energy = structure.get_potential_energy()

    except:
        result = "vasp failure"

    return structure, energy, result


# =============================================================================
def set_vasp_script(vasp_script, executable, num_cores, vasp_pseudopotentials):
    """
    Writes the script that is used to run Vasp from within ASE.

    Parameters
    ----------
    vasp_script : string
        Filename of the script to be written.
    executable : string
        The filepath to the Vasp executable to be used.
    num_cores : integer
        The number of parallel cores to be used for Vasp calculations.
    vasp_pseudopotentials : string
        Filepath to the folder containing Vasp pseudopotential files.

    Returns
    -------
    None

    ---------------------------------------------------------------------------
    Paul Sharp 03/08/2018
    """

    with open(vasp_script, mode="w") as script:

        if num_cores > 1:
            script.write("import os\nexitcode = os.system('mpirun -np " + str(num_cores) + " " + executable + "')")

        else:
            script.write("import os\nexitcode = os.system('" + executable + "')")

    os.environ["VASP_SCRIPT"] = "./" + vasp_script
    os.environ["VASP_PP_PATH"] = vasp_pseudopotentials

    return None


# =============================================================================
def check_float(value):
    """
    Determines whether or not a value is representable as a floating point.

    Parameters
    ----------
    value : str
        The value of interest

    Returns
    -------
    representable : boolean
        True if value is representable as a float.

    ---------------------------------------------------------------------------
    Paul Sharp 07/09/2017
    """

    representable = True

    try:
        float(value)
    except ValueError:
        representable = False

    return representable


# =============================================================================
def converged_in_one_scf_cycle(outcar_file):
    """
    Determines whether a VASP calculation converged in a single SCF cycle.

    Parameters
    ----------
    outcar_file : str
        OUTCAR file for this vasp calculation.

    Returns
    -------
    converged : boolean
        True if the calculation has converged, and did so within a single SCF cycle.

    ---------------------------------------------------------------------------
    Paul Sharp 27/10/2017
    """

    aborting_ionic_loop_marker = "aborting loop"
    convergence_marker = "reached required accuracy"

    num_convergence_strings = 0
    num_ionic_loops = 0

    with open(outcar_file, mode="r") as outcar:

        for line in outcar:

            num_convergence_strings += line.count(convergence_marker)
            num_ionic_loops += line.count(aborting_ionic_loop_marker)

    converged = num_convergence_strings > 0 and num_ionic_loops == 1

    return converged


# =============================================================================
def determine_vacancy_positions(structure):
    """
    Find the scaled positions of vacancies in a structure

    Parameters
    ----------
    structure : ase atoms
        The structure in which we wish to determine the vacancy positions.

    Returns
    -------
    vacancy_positions : float
        The list of vacancy positions.

    ---------------------------------------------------------------------------
    Paul Sharp 27/07/2018
    """

    del structure[[atom.index for atom in structure if atom.symbol != "X"]]
    vacancy_positions = structure.get_scaled_positions()

    return vacancy_positions


# =============================================================================
def populate_points_with_vacancies(structure, vacancy_positions):
    """
    Add a list of points to a structure as vacancies.

    Parameters
    ----------
    structure : ase atoms
        The structure to which we will add vacancies.
    vacancy_positions : float
        The list of points to be recorded as vacancies.

    Returns
    -------
    structure : ase atoms
        The structure with the points set as vacancies.

    ---------------------------------------------------------------------------
    Paul Sharp 08/08/2018
    """

    num_vacancies = len(vacancy_positions)

    if num_vacancies > 0:
        structure.extend(ase.Atoms("X" + str(num_vacancies),
                                   cell=structure.get_cell(),
                                   scaled_positions=vacancy_positions,
                                   charges=[0] * num_vacancies,
                                   pbc=[True, True, True]))

    return structure
