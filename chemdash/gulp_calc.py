"""
|=============================================================================|
|                             G U L P   C A L C                               |
|=============================================================================|
|                                                                             |
| This module contains routines that run custom GULP calculations, and check  |
| the status of GULP calculations.                                            |
|                                                                             |
| Contains                                                                    |
| --------                                                                    |
|     multi_stage_gulp_calc                                                   |
|     run_gulp                                                                |
|     set_gulp_command                                                        |
|     update_atoms_from_restart_file                                          |
|     create_input_file_from_restart_file                                     |
|     execute_gulp_command_or_script                                          |
|     read_energy                                                             |
|     read_gnorm                                                              |
|     check_convergence                                                       |
|     check_float                                                             |
|     check_timed_out                                                         |
|     read_outcome                                                            |
|     update_outcomes                                                         |
|     read_potentials                                                         |
|     strip_vacancies                                                         |
|     determine_vacancy_positions                                             |
|     populate_points_with_vacancies                                          |
|                                                                             |
|-----------------------------------------------------------------------------|
| Paul Sharp 30/11/2020                                                       |
| Chris Collins 26/11/2020                                                    |
|=============================================================================|
"""

from builtins import range
from builtins import str

try:
    from ase.calculators.gulp import GULP
except ImportError:
    from ase.calculators.gulp import Gulp as GULP

import ase
import math
import numpy as np
import os
import platform
import re
import subprocess
import time


# =============================================================================
# =============================================================================
def multi_stage_gulp_calc(structure, num_calcs, gulp_files, main_keywords,
                          additional_keywords, main_options,
                          additional_options, max_gnorms, gulp_shells=[],
                          gulp_library="", gulp_conditions=None,
                          remove_vacancies=True):
    """
    Run a GULP calculation that consists of many stages, which can involve
    different keywords and options.

    Parameters
    ----------
    structure : ChemDASH structure
        The structure class containing ASE atoms object and properties.
    num_calcs : integer
        The number of GULP calculations to run
    gulp_files : string
        Name used for input and output files for each GULP calculation
    main_keywords : list
        List of GULP keywords. Default is "sing" for a single point energy
        calculation.
    additional_keywords : list
        List of extra GULP keywords for each calculation. Default is None
    main_options : list
        List of GULP options. Default is None
    additional_options : list
        List of extra GULP options for each calculation. Default is None
    max_gnorms : list
        List of maximum values of the final gnorm for each calculation.
    gulp_shells : dictionary, optional
        A dictionary with atom labels as keys and charges as values for each
        atomic species to have a shell attached.
    gulp_library : string, optional
        The library file containing the forcefield to be used in the calculation.
    gulp_conditions : ASE Conditions, optional
        The conditions used to determine the label given to particular atoms if
        the forcefield has different atom types.

    Returns
    -------
    atoms : ase atoms
        The structure after the split calculation.
    result : string
        The result of the GULP calculation,
        either "converged", "unconverged", "gulp failure", or "timed out"
    energy : float
        Energy of the structure in eV/atom.
    outcome : string
        GULP outcome of the final stage of this calculation.
    calc_time : float
        Time taken for this GULP calculation.
    potentials : float
        Site potentials for each atom.
    derivs : float
        Magnitude of the derivatives of the potential for each atom.

    ---------------------------------------------------------------------------
    Paul Sharp 27/03/2020
    """
    
    assert len(gulp_files) >= num_calcs, 'ERROR in gulp_calc.multi_stage_gulp_calc() -- number of GULP calculations is set to {0:d}, but only {1:d} filenames are provided.'.format(num_calcs, len(gulp_files))
    assert len(additional_keywords) >= num_calcs, 'ERROR in gulp_calc.multi_stage_gulp_calc() -- number of GULP calculations is set to {0:d}, but only {1:d} sets of additional keywords are provided.'.format(num_calcs, len(additional_keywords))
    assert len(additional_options) >= num_calcs, 'ERROR in gulp_calc.multi_stage_gulp_calc() -- number of GULP calculations is set to {0:d}, but only {1:d} sets of additional options are provided.'.format(num_calcs, len(additional_options))
   
    gulp_start_time = time.time()
    
    failed_outcome = "Too many failed attempts to optimise "
    calc_files = {"input": "",
                  "output": "",
                  "restart": "",
    }

    structure.potentials = []
    structure.derivs = []

    # Strip out vacancies from the structure
    if remove_vacancies:
        vacancy_positions = determine_vacancy_positions(structure.atoms.copy())
        del structure.atoms[[atom.index for atom in structure.atoms if atom.symbol == "X"]]

    for i in range(0, num_calcs):
        
        composite_keys = (main_keywords + " " + additional_keywords[i]).split()
        gulp_keywords = " ".join(sorted(set(composite_keys), key=composite_keys.index))

        composite_options = main_options + additional_options[i]
        composite_options.append("dump " + gulp_files[i] + ".res")
        gulp_options = list(sorted(set(composite_options), key=composite_options.index))

        structure.atoms, structure.energy, gnorm, result, calc_files = run_gulp(structure.atoms,
                                                                                gulp_files[i],
                                                                                calc_files["restart"],
                                                                                gulp_keywords,
                                                                                gulp_options,
                                                                                gulp_shells,
                                                                                gulp_library,
                                                                                gulp_conditions)

        # Continue only if the calculation was successfully performed without timing out
        if result == "timed out" or result == "gulp failure":

            outcome = result
            if os.path.isfile(calc_files["output"]):
                outcome = read_outcome(calc_files["output"])

            if outcome == "":
                outcome = "Bash timeout"

            break

        outcome = read_outcome(calc_files["output"])

        # We abandon the calculation if the outcome "Too many failed attempts to optimise" occurs,
        # the energy and gradient norm have blown up such that they are not representable as floats,
        # or the max gnorm for this calculation, if provided, is exceeded.

        # if (outcome == failed_outcome) or (not check_float(structureenergy)) or (not check_float(gnorm)):
        if (not check_float(structure.energy)) or (not check_float(gnorm)):
            result = "unconverged"
            break

        if isinstance(max_gnorms[i], float):
            if gnorm > max_gnorms[i]:
                result = "unconverged"
                break

        # Update atoms object before continuing
        # This is necessary because of a bug in ASE, where it only updates atomic positions if they are given in cartesian coordinates
        try:
            structure.atoms, structure.energy = update_atoms_from_restart_file(structure.atoms, structure.energy, calc_files["restart"])
        except ValueError:
            result = "unconverged"
            break
        
    else:
        
        if os.path.isfile(calc_files["output"]):
            
            if any(pot_keyword in gulp_keywords for pot_keyword in ["pot", "potential"]):
                structure.potentials, structure.derivs = read_potentials(calc_files["output"])

            if check_convergence(calc_files["output"], gulp_keywords):
                result = "converged"
            else:
                result = "unconverged"

        else:
            outcome = ""
            result = "unconverged"

    # Replace the vacancies in the atoms object
    if remove_vacancies:
        structure.atoms = populate_points_with_vacancies(structure.atoms, vacancy_positions)
        
    structure.volume = structure.atoms.get_volume()
    calc_time = time.time() - gulp_start_time
    
    return structure, result, outcome, calc_time


# =============================================================================
def run_gulp(structure, gulp_file, previous_restart_file="",
             gulp_keywords=["conp gradients"], gulp_options=[], gulp_shells=[],
             gulp_library="", gulp_conditions=None):
    """
    Run GULP in order to perform any stage of a split calculation.
    The calculation is run either by using an ASE calculator or as a system
    command depending on whether or not we have shells present -- if so, the
    system approach is required in order to preserve the positions of atomic
    shells.

    Parts of this routine are based on the "restart_gulp" routine,
    written by Chris Collins, and the section that runs GULP as a system command
    is taken from ASE-GULP.

    Parameters
    ----------
    structure : ASE atoms
        The structure used in the GULP calculation
    gulp_file : string
        Used to construct input and output files for GULP
    previous_restart_file : string, optional
        Restart file for previous GULP calculation -- used to construct new input file.
    gulp_keywords : list, optional
        List of GULP keywords. Default is "sing" for a single point energy calculation.
    gulp_options : list, optional
        List of GULP options. Default is None.
    gulp_shells : dictionary, optional
        List of each atomic species to have a shell attached.
    gulp_library : string, optional
        The library file containing the forcefield to be used in the calculation.
    gulp_conditions : ASE Conditions, optional
        The conditions used to determine the label given to particular atoms if
        the forcefield has different atom types.

    Returns
    -------
    structure : ase atoms
        The structure after performing this calculation.
    energy : float
        Energy of the structure in eV/atom.
    result : string
        The result of the GULP calculation,
        either "converged", "unconverged", "gulp failure", or "timed out"
    calc_files : dict
        The input, output and restart files for this GULP calculation.

    ---------------------------------------------------------------------------
    Paul Sharp 02/08/2018
    """

    # Set files for calculation
    calc_files = {"input": gulp_file + ".gin",
                  "output": gulp_file + ".got",
                  "restart":  gulp_file + ".res",
    }
        
    energy = ""
    gnorm = ""
    result = ""
    timeout_outcome = "CPU limit has been exceeded - restart optimisation "

    # Determine how to do the second calculation depending on whether or not we have shells present
    if (gulp_shells == []) or (structure.get_calculator() is None):

        # Use ASE calculator -- set keywords and options for this calculation
        calc = (GULP(label=gulp_file, keywords=gulp_keywords,
                     options=gulp_options, shel=gulp_shells,
                     library=gulp_library, conditions=gulp_conditions))

        structure.set_calculator(calc)

        # Run calculation
        try:
            energy = structure.get_potential_energy()

        except:
            pass

        else:
            gnorm = structure.calc.get_Gnorm()

    else:

        # Run as system command -- need to construct the GULP input file from the ".res" file of the previous calculation
        if os.path.isfile(previous_restart_file):

            create_input_file_from_restart_file(calc_files["input"],
                                                previous_restart_file,
                                                structure.get_cell(),
                                                gulp_keywords, gulp_options,
                                                gulp_library)

            execute_gulp_command_or_script(gulp_file, calc_files["output"])

            # Read energy from output file
            if os.path.isfile(calc_files["output"]):

                energy = read_energy(calc_files["output"])
                gnorm = read_gnorm(calc_files["output"])

        else:

            raise RuntimeError('ERROR in "gulp_calc.run_gulp()" -- shells are used in this GULP calculation, but a ".res" file was not generated in the previous stage of the calculation.\n'
                               'This means that the positions of the shells cannot be tracked accurately. Please specify "dump <restart>.res" in the "gulp_options".')

    if not os.path.isfile(calc_files["output"]):
        result = "gulp failure"

    # The calculation can timeout either within GULP or via the bash timeout command
    elif read_outcome(calc_files["output"]) == timeout_outcome or check_timed_out(calc_files["output"]):
        result = "timed out"

    # Convert energy to units of eV/atom
    try:
        energy /= float(len(strip_vacancies(structure.copy())))
    except (TypeError, ValueError):
        pass

    return structure, energy, gnorm, result, calc_files


# =============================================================================
def set_gulp_command(executable, num_cores, time_limit, gulp_library):
    """
    Set the environment variables required to use the ASE Gulp calculator.
    The environment variables in question are: $GULP_LIB and $ASE_GULP_COMMAND.

    Parameters
    ----------
    executable : string
        The filepath to the gulp executable to be used.
    num_cores : integer
        The number of parallel cores to be used for GULP calculations.
    time_limit : integer
        The time limit in seconds to be used in the bash "timeout" command.
    gulp_library : string
        Filepath to the folder containing GULP potential files.
        (Not generally used).

    Returns
    -------
    None

    ---------------------------------------------------------------------------
    Paul Sharp 03/08/2018
    Chris Collins 26/11/2020
    """

    if platform.system() == 'Windows':
        os.environ["ASE_GULP_COMMAND"] = str(executable + " < PREFIX.gin > PREFIX.got")

    if platform.system() == 'Linux':
        
        os.environ["GULP_LIB"] = ""
        
        if num_cores > 1:
            #os.environ["ASE_GULP_COMMAND"] = "timeout " + str(time_limit) + " mpirun -np " + str(num_cores) + " " + executable + " < PREFIX.gin > PREFIX.got"
            os.environ["ASE_GULP_COMMAND"] = "timeout --kill-after=5 " + str(time_limit) + " mpirun -np " + str(num_cores) + " " + executable + " < PREFIX.gin > PREFIX.got"
        else:
            #os.environ["ASE_GULP_COMMAND"] = "timeout " + str(time_limit) + " " + executable + " < PREFIX.gin > PREFIX.got"
            os.environ["ASE_GULP_COMMAND"] = "timeout --kill-after=5 " + str(time_limit) + " " + executable + " < PREFIX.gin > PREFIX.got"
    

    return None


# =============================================================================
def update_atoms_from_restart_file(structure, energy, gulp_res_file):
    """
    Read unit cell, atomic positions and energy from a GULP ".res" file, where
    they are quoted to greater precision than the output file (and hence the
    ASE atoms object if available).

    For the GULP calculator in ASE 3.14-, the unit cell and atomic positions
    (if in fractional coordinates) are not recorded after a calculation and so
    must be obtained from a restart file (or the output).

    Parameters
    ----------
    structure : ASE atoms
        The structure used in the GULP calculation.
    energy : float
        Energy of the structure in eV from the calculation.
    gulp_res_file : string
        The restart file written from this GULP calculation.

    Returns
    -------
    structure : ASE atoms
        The structure used in the GULP calculation with unit cell and atomic
        positions taken from the restart file.
    energy : float
        A high-precision value of the final energy of the calculated structure,
        converted to units of eV/atom.

    ---------------------------------------------------------------------------
    Paul Sharp 09/12/2019
    """

    if os.path.isfile(gulp_res_file):

        res_file = read_restart_file(gulp_res_file)

        for i, line in enumerate(res_file):

            # This matches only if the line consists of "cell" and nothing else
            # Hence, we avoid a match on the "cellonly" keyword
            if re.search("^cell$", line):

                cell_line = res_file[i+1].split()
                structure.set_cell(np.array([float(cell_line[0]), float(cell_line[1]),
                                             float(cell_line[2]), float(cell_line[3]),
                                             float(cell_line[4]), float(cell_line[5])]))

            # This matches only if the line consists of "vectors" and nothing
            # else. Hence, we avoid a match on the "eigenvectors" keyword
            if re.search("^vectors$", line):

                cell_vectors = []
                for j in range(1, 4):

                    vector_line = res_file[i+j].split()
                    cell_vectors.append([float(vector_line[0]),
                                         float(vector_line[1]),
                                         float(vector_line[2])])

                cell_vectors = np.array(cell_vectors)
                structure.set_cell(cell_vectors)

            # This may match on either the coordinate type as desired, or the
            # "cartesian" keyword. If it matches on the keyword it will break
            # the loop immediately as neither "core" nor "shell" should be on
            # the next line.
            if "cartesian" in line or "fractional" in line:

                coordinate_type = line.split()[0]
                pos = []

                for j in range(i+1, len(res_file)):

                    if "core" in res_file[j]:

                        atom_pos = []
                        atom_line = res_file[j].split()

                        for line_index in range(2, 5):

                            coord = atom_line[line_index]
                            if "/" in coord:
                                atom_pos.append(float(coord.split("/")[0]) / float(coord.split("/")[1]))
                            else:
                                atom_pos.append(float(coord))

                        pos.append(atom_pos)

                    elif "shell" in res_file[j]:
                        continue

                    else:
                        break

                pos = np.array(pos)

            if "totalenergy" in line:

                try:
                    energy = float(line.split()[1]) / float(len(strip_vacancies(structure.copy())))
                except ValueError:
                    energy = line.split()[1]

        # Replace positions of atoms in structure, depending on coordinate type
        try:
            coordinate_type
        except NameError:
            print('WARNING in "gulp_calc.update_atoms_from_restart_file()" -- no valid coordinate type found in ".res" file')
        else:
            if coordinate_type == "fractional":
                structure.set_scaled_positions(pos)
            elif coordinate_type == "cartesian":
                structure.set_positions(pos)
            else:
                print('WARNING in "gulp_calc.update_atoms_from_restart_file()" -- no valid coordinate type found in ".res" file')

    return structure, energy


# =============================================================================
def create_input_file_from_restart_file(gulp_input_file, gulp_restart_file,
                                        unit_cell, gulp_keywords, gulp_options,
                                        gulp_library):
    """
    Use the restart file of a previous GULP calculation to create an input file
    for a new calculation.

    Parameters
    ----------
    gulp_input_file, gulp_restart_file : string
        The new input file for this GULP calculation and the restart file from
        the previous GULP calculation
    unit_cell : float
        The unit cell of the structure under consideration.
    gulp_keywords : list
        List of GULP keywords for the new GULP calculation.
    gulp_options : list
        List of GULP options for the new GULP calculation.
    gulp_library : string
        The library file containing the forcefield used in this calculation.

    Returns
    -------
    None

    ---------------------------------------------------------------------------
    Paul Sharp 10/08/2018
    """

    # Set the markers in gulp output files above the atomic positions
    start_markers = ["fractional", "cartesian"]

    res_file = read_restart_file(gulp_restart_file)

    with open(gulp_input_file, mode="w") as new_input_file:

        # Keywords
        new_input_file.write(gulp_keywords + "\n")
        new_input_file.write("\n")

        # Write vectors from ase in order to preserve the orientation of the unit cell -- can be lost in conversion from cell parameters
        new_input_file.write("vectors" + "\n")
        for vector in unit_cell:
            new_input_file.write("    " + str(vector[0]) + "    " + str(vector[1]) + "    " + str(vector[2]) + "\n")

        # Find structure in restart file and write to new input file
        for i, line in enumerate(res_file):

            # This may match on either the coordinate type as desired, or the
            # "cartesian" keyword. If it matches on the keyword it will break
            # the loop immediately as neither "core" nor "shell" should be on
            # the next line.
            if "cartesian" in line or "fractional" in line:

                if "core" in res_file[i+1] or "shel" in res_file[i+1]:

                    new_input_file.write(line + "\n")

                    for j in range(i+1, len(res_file)):

                        if "core" in res_file[j] or "shel" in res_file[j]:

                            new_input_file.write(res_file[j] + "\n")

                        else:

                            break

        new_input_file.write("\n")
        new_input_file.write("library " + gulp_library + "\n")

        for option in gulp_options:
            new_input_file.write(option + "\n")

    return None


# =============================================================================
def read_restart_file(restart_file):
    """
    Read in a GULP restart file.

    Parameters
    ----------
    restart_file : string
        The restart file from the previous GULP calculation

    Returns
    -------
    res_file_data : list
        A list of each line in the restart file, with newline characters removed.

    ---------------------------------------------------------------------------
    Paul Sharp 10/08/2018
    """

    with open(restart_file, mode="r") as f:
        res_file_data = [string.strip() for string in f.readlines()]

    return res_file_data


# =============================================================================
def execute_gulp_command_or_script(gulp_job_name, gulp_output):
    """
    If a GULP calculation is to be run manually, execute the appropriate
    command or script.

    Parameters
    ----------
    gulp_job_name : str
        The base name of the input file used as the command line argument to GULP.
    gulp_output : str
        Name of the GULP output file.

    Returns
    -------
    None

    ---------------------------------------------------------------------------
    Paul Sharp 27/03/2020
    """

    # Perform the GULP calculation
    if 'ASE_GULP_COMMAND' in os.environ:

        gulp_command = os.environ.get('ASE_GULP_COMMAND').replace('PREFIX', gulp_job_name)

        # PS -- Need to consider how to resolve issues with both os.system and shell=True.
        # This is a consequence of the complexity of the GULP_COMMAND used to enable timeouts
        with open(gulp_output, mode="w") as output_file:
            exitcode = subprocess.call('{0}'.format(gulp_command),
                                       stdout=output_file,
                                       stderr=subprocess.STDOUT,
                                       shell=True)
        #exitcode = os.system('{0}'.format(gulp_command))

    elif 'GULP_SCRIPT' in os.environ:

        gulp_script = os.environ['GULP_SCRIPT']
        locals = {}
        exec(compile(open(gulp_script).read(), gulp_script, 'exec'), {}, locals)
        exitcode = locals['exitcode']

    else:
        raise RuntimeError('Please set either GULP_COMMAND or GULP_SCRIPT environment variable')

    if exitcode != 0:
        raise RuntimeError('Gulp exited with exit code: {0}.'.format(exitcode))

    return None


# =============================================================================
def read_energy(gulp_output):
    """
    If a GULP calculation is has been run manually, read the energy from the
    output file.

    Parameters
    ----------
    gulp_output : str
        Name of the GULP output file.

    Returns
    -------
    energy : float
        Energy of the structure in eV.

    ---------------------------------------------------------------------------
    Paul Sharp 15/01/2018
    """

    with open(gulp_output, mode="r") as output_file:

        energy_marker = "Total lattice energy"
        energy_unit = "eV"

        # Check full file in order to get final energy
        for line in output_file:
            if energy_marker in line and energy_unit in line:
                try:
                    energy = float(line.split()[4])
                except ValueError:
                    energy = line.split()[4]

    return energy


# =============================================================================
def read_gnorm(gulp_output):
    """
    If a GULP calculation is has been run manually, read the gnorm from the
    output file.

    Parameters
    ----------
    gulp_output : str
        Name of the GULP output file.

    Returns
    -------
    gnorm : float
        Final gnorm of the structure.

    ---------------------------------------------------------------------------
    Paul Sharp 18/01/2018
    """

    with open(gulp_output, mode="r") as output_file:

        gnorm_marker = "Final Gnorm"

        # Check full file in order to get final gnorm
        for line in output_file:
            if gnorm_marker in line:
                try:
                    gnorm = float(line.split()[3])
                except ValueError:
                    gnorm = line.split()[3]

    return gnorm


# =============================================================================
def check_convergence(gulp_out_file, gulp_keywords):
    """
    Examines an output file of a GULP calculation to determine whether or not a
    structure optimisation was successful.

    Parameters
    ----------
    gulp_out_file : string
        The output file of a GULP calculation
    gulp_keywords : string
        The keywords used for this GULP calculation

    Returns
    -------
    converged : boolean
        Determines whether the GULP calculation has successfully optimised the
        structure

    ---------------------------------------------------------------------------
    Paul Sharp 22/09/2016
    """

    optimisation_keywords = ["opti", "optimise", "grad", "gradient", "fit"]

    if any(keyword in gulp_keywords for keyword in optimisation_keywords):
        optimisation_marker = "**** Optimisation achieved ****"
    else:
        optimisation_marker = "Components of energy"

    converged = False

    with open(gulp_out_file, mode="r") as out_file:

        for line in out_file:
            if optimisation_marker in line:
                converged = True
                break

    return converged


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
def check_timed_out(gulp_out_file):
    """
    Determines hether a GULP calculation has timed out via the bash "timeout"
    command.

    Parameters
    ----------
    gulp_out_file : string
        The output file from a GULP calculation.

    Returns
    -------
    timed_out : boolean
        True if calculation timed out.

    ---------------------------------------------------------------------------
    Paul Sharp 30/11/2020
    Chris Collins 26/11/2020
    """

    finished_marker = "Job Finished"
    terminated_marker = "Program terminated"

    if platform.system() == 'Linux':
    	 final_lines = subprocess.check_output(["tail", "-2", gulp_out_file]).decode()

    elif platform.system() == 'Windows':
    	 text = open(gulp_out_file).readlines()
    	 final_lines = ''.join(text[-3:])
    	 
    else:
        
        final_lines = ""

    timed_out = True
    if (finished_marker in final_lines) or (terminated_marker in final_lines):
        timed_out = False

    return timed_out


# =============================================================================
def read_outcome(gulp_out_file):
    """
    Reads the outcome of a GULP calculation from the output file.

    Parameters
    ----------
    gulp_out_file : string
        The output file of a GULP calculation

    Returns
    -------
    outcome : string
        The outcome ofthis GULP calculation.

    ---------------------------------------------------------------------------
    Paul Sharp 17/08/2017
    """

    outcome_marker = " **** "
    outcome = ""

    with open(gulp_out_file, mode="r") as out_file:

        for line in out_file:
            if outcome_marker in line:
                outcome = outcome + line.split(outcome_marker)[1][:-5]

    return outcome.strip()


# =============================================================================
def update_outcomes(new_outcome, outcomes):
    """
    Updates the outcome dictionary with the outcome of the latest GULP
    calculation.

    Parameters
    ----------
    new_outcome : string
        The outcome of the latest GULP calculation.
    outcomes : dict
        Dictionary of the different outcomes and the number of times they
        occured.

    Returns
    -------
    outcomes : dict
        Updated dictionary of the different outcomes and the number of times
        they occured.

    ---------------------------------------------------------------------------
    Paul Sharp 17/08/2017
    """

    # If outcome has already occurred, increment the number of occurances,
    # else add the new outcome
    if new_outcome in outcomes:

        outcomes[new_outcome] += 1

    else:

        outcomes[new_outcome] = 1

    return outcomes


# =============================================================================
def read_potentials(gulp_out_file):
    """
    Reads the site potentials for each atoms from the output file of a GULP
    calculation.

    Parameters
    ----------
    gulp_out_file : string
        The output file of a GULP calculation.

    Returns
    -------
    potentials : float
        Site potentials for each atom.
    derivs : float
        Magnitude of the derivatives of the potential for each atom.

    ---------------------------------------------------------------------------
    Paul Sharp 16/01/2020
    """

    potentials = []
    derivs = []
    num_cores_shells = 0
    atom = 0

    with open(gulp_out_file, mode="r") as output_file:

        for line in output_file:
            if 'Total number atoms/shells' in line:
                num_cores_shells = int(line.split()[4])
                break

        for line in output_file:

            if 'Electrostatic potential at atomic positions :' in line:
                potentials = []
                derivs = []

                for line in output_file:

                    temp = line.split()

                    # Only include cores
                    if len(temp) > 1 and temp[2] == 'c':
                        try:
                            potentials += [float(temp[3])]
                        except(TypeError, ValueError):
                            potentials += ["--"]
                        try:
                            derivs += [math.sqrt(float(temp[4])**2 +
                                                 float(temp[5])**2 +
                                                 float(temp[6])**2)]
                        except(TypeError, ValueError):
                            derivs += ["--"]

                        atom += 1

                    # Make sure to count shells
                    elif len(temp) > 1 and temp[2] == 's':
                        
                        atom += 1
                        
                    if atom == num_cores_shells:
                        break

                break

    return potentials, derivs


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
    Paul Sharp 27/02/2019
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
    Paul Sharp 27/02/2019
    """

    num_vacancies = len(vacancy_positions)

    if num_vacancies > 0:
        structure.extend(ase.Atoms("X" + str(num_vacancies),
                                   cell=structure.get_cell(),
                                   scaled_positions=vacancy_positions,
                                   charges=[0] * num_vacancies,
                                   pbc=[True, True, True]))

    return structure

