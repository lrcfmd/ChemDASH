"""
|=============================================================================|
|                                 I N P U T S                                 |
|=============================================================================|
|                                                                             |
| This module contains a routine that reads in the input file and processes   |
| its arguments.                                                              |
|                                                                             |
| Contains                                                                    |
| --------                                                                    |
|     initialise_default_params                                               |
|     write_defaults_to_file                                                  |
|     parse_input                                                             |
|     report_input_file_error                                                 |
|     convert_values                                                          |
|     handle_dependencies                                                     |
|     check_positive_float                                                    |
|     check_list_positive_float                                               |
|     check_positive_int                                                      |
|     check_list_positive_int                                                 |
|     check_even_int                                                          |
|     check_list_even_int                                                     |
|     check_int_on_list                                                       |
|     check_unsigned_n_bit_int                                                |
|     check_str_on_list                                                       |
|     check_str_list_on_list                                                  |
|     check_boolean                                                           |
|     check_duplicates_in_list                                                |
|     check_swap_groups_valid                                                 |
|     convert_to_three_element_list                                           |
|     convert_to_dict_with_yaml                                               |
|                                                                             |
|-----------------------------------------------------------------------------|
| Paul Sharp 28/01/2021                                                       |
|=============================================================================|
"""

from builtins import range
from builtins import map

import ase
import os
import yaml

ase_chemical_symbols = ase.data.chemical_symbols


# =============================================================================
# =============================================================================
def initialise_default_params(calc_name):
    """
    This routine sets up the params dictionary with all of its keywords and
    their default values.

    Parameters
    ----------
    calc_name : string
        Name of this job -- default value for name of atoms file.

    Returns
    -------
    params : dict
        Dictionary containing each parameter with its default value.

    ---------------------------------------------------------------------------
    Paul Sharp 28/01/2021
    """
    
    # Set up initial parameter dictionary, with all default values.
    params = {
        "atoms_file": {"value": calc_name + ".atoms",
                       "specified": False,
                       "type": "string",
                       "description": 'File in which the species, number and oxidation state of the atoms used in this calculation are specified.',
                   },
         "atom_labels": {"value": "",
                       "specified": False,
                       "type": "list of strings",
                       "description": 'Label applied for the atoms on each line of the atoms file -- used to distinguish atoms in forcefields.',
                   },  
        "atom_rankings": {"value": "random",
                          "specified": False,
                          "supported_values": ["random", "bvs"],
                          "type": "string on list",
                          "description": 'The metric used to rank atoms for swapping. Supported values are: "random" (default), "bvs", "bvs+", "site_pot". Note that site potential and bvs+ directed swapping are only supported for gulp.',
                      },
        "bvs_file": {"value": calc_name + "_bvs.dat",
                     "specified": False,
                     "type": "string",
                     "description": 'Raw Bond Valence Sum file for this calculation. Records the bond valence sum for the atoms in each structure.',
                 },
        "calculator": {"value": "gulp",
                       "specified": False,
                       "supported_values": ["gulp", "vasp"],
                       "type": "string on list",
                       "description": 'The materials modelling code used for calculations. Default: gulp',
                   },
        "calculator_cores": {"value": 1,
                             "specified": False,
                             "type": "integer",
                             "description": 'The number of parallel cores used for the calculator. Default: 1.',
                         },
        "calculator_time_limit": {"value": 0,
                                  "specified": False,
                                  "type": "integer",
                                  "description": 'Used in the bash "timeout" command, GULP calculations will automatically terminate after this amount of time has expired.',
                              },
        "cell_spacing": {"value": "2.0",
                         "specified": False,
                         "type": "float",
                         "description": 'The spacing between two ANION grid points. Default: 2.0 A0',
                     },   
        "converge_first_structure": {"value": "True",
                                     "specified": False,
                                     "type": "boolean",
                                     "description": 'If True, abort the run if the initial structure is not converged. Default: True',
                                 },
        "cp_2d_lattice": {"value": "oblique",
                            "specified": False,
                            "supported_values": ["oblique", "centred_rectangular"],
                            "type": "string on list",
                            "description": 'Lattice type for anion layers in close packed grids. Supported values are: "oblique" (default) and "centred_rectangular"',
                        },
        "cp_stacking_sequence": {"value": "",
                                 "specified": False,
                                 "supported_values": ["A", "B", "C"],
                                 "type": "string list on list",
                                 "description": 'Anion layer stacking sequence for close packed grids.',
                             },      
        "directed_num_atoms": {"value": 0,
                               "specified": False,
                               "type": "integer",
                               "description": 'For directed swapping, the number of extra atoms available to choose between from the top of the list for each species. Default: 0',
                           },
        "directed_num_atoms_increment": {"value": 0,
                                         "specified": False,
                                         "type": "integer",
                                         "description": 'For directed swapping, the amount by which to increase (decrease) the number of extra values available to choose between from the top of the list for each species when a structure is (not) repeated. Default: 0',
                                     },
        "energy_file": {"value": calc_name + "_energy.dat",
                        "specified": False,
                        "type": "string",
                        "description": 'Energy file for this calculation. Records the structure number, energies and volumes of accepted structures.',
                         },
        "energy_step_file": {"value": calc_name + "_energy_step.dat",
                             "specified": False,
                             "type": "string",
                             "description": 'Energy step file for this calculation. Records the structure number, energies and volumes of accepted structures for plotting.',
                         },
        "force_vacancy_swaps": {"value": "True",
                                "specified": False,
                                "type": "boolean",
                                "description": 'If True, vacancies cannot swap with each other, they must be replaced by atoms. Default: True.',
                            },
        "grid_points": {"value": "2",
                        "specified": False,
                        "type": "integer",
                        "description": 'The number of points on each dimension of the ANION grid, to form an a x b x c grid for anions (cation points defined by grid type). Default: 2x2x2',
                    },
        "grid_type": {"value": "orthorhombic",
                      "specified": False,
                      "supported_values": ["orthorhombic", "rocksalt", "close_packed"],
                      "type": "string on list",
                      "description": 'Initial layout of cation and anion grids. Supported values are "orthorhombic" (default), rocksalt", close_packed". Default: "orthorhombic".',
                  },
        "gulp_executable": {"value": "gulp",
                            "specified": False,
                            "type": "string",
                            "description": 'The filepath of the GULP executable to be used. Default: "./gulp".',
                        },
        "gulp_files": {"value": "a,b,c,d,e",
                       "specified": False,
                       "type": "list of strings",
                       "description": 'Strings appended to each of the GULP files used to distinguish each calculation.',
                   },
        "gulp_keywords": {"value": "opti, pot",
                          "specified": False,
                          "type": "list to string",
                          "description": 'Comma-separated list of keywords for all GULP calculations. Default: "opti, pot"',
                      },
        "gulp_calc_1_keywords": {"value": "",
                                 "specified": False,
                                 "type": "list to string",
                                 "description": 'Comma-separated list of keywords for first GULP calculation. Default: None',
                             },
        "gulp_calc_2_keywords": {"value": "",
                                 "specified": False,
                                 "type": "list to string",
                                 "description": 'Comma-separated list of keywords for second GULP calculation. Default: None',
                             },
        "gulp_calc_3_keywords": {"value": "",
                                 "specified": False,
                                 "type": "list to string",
                                 "description": 'Comma-separated list of keywords for third GULP calculation. Default: None',
                             },
        "gulp_calc_4_keywords": {"value": "",
                                 "specified": False,
                                 "type": "list to string",
                                 "description": 'Comma-separated list of keywords for fourth GULP calculation. Default: None',
                             },
        "gulp_calc_5_keywords": {"value": "",
                                 "specified": False,
                                 "type": "list to string",
                                 "description": 'Comma-separated list of keywords for fifth GULP calculation. Default: None',
                             },
        "gulp_calc_1_max_gnorm": {"value": "",
                                  "specified": False,
                                  "type": "float specified",
                                  "description": 'If specified, terminate a GULP calculation if the final gnorm exceeds this value after the first stage.',
                              },
        "gulp_calc_2_max_gnorm": {"value": "",
                                  "specified": False,
                                  "type": "float specified",
                                  "description": 'If specified, terminate a GULP calculation if the final gnorm exceeds this value after the second stage.',
                              },
        "gulp_calc_3_max_gnorm": {"value": "",
                                  "specified": False,
                                  "type": "float specified",
                                  "description": 'If specified, terminate a GULP calculation if the final gnorm exceeds this value after the third stage.',
                              },
        "gulp_calc_4_max_gnorm": {"value": "",
                                  "specified": False,
                                  "type": "float specified",
                                  "description": 'If specified, terminate a GULP calculation if the final gnorm exceeds this value after the fourth stage.',
                              },
        "gulp_calc_5_max_gnorm": {"value": "",
                                  "specified": False,
                                  "type": "float specified",
                                  "description": 'If specified, terminate a GULP calculation if the final gnorm exceeds this value after the fifth stage.',
                              },
        "gulp_library": {"value": "",
                         "specified": False,
                         "type": "string",
                         "description": 'Library file containing the forcefield to be used in GULP calculations. NOTE -- this takes precedence over a library specified in "gulp_options".',
                     },
        "gulp_options": {"value": [],
                         "specified": False,
                         "type": "list",
                         "description": 'Options for all GULP calculations. Default: None',
                     },
        "gulp_calc_1_options": {"value": [],
                                "specified": False,
                                "type": "list",
                                "description": 'Options for first GULP calculation. Default: None',
                            },
        "gulp_calc_2_options": {"value": [],
                                "specified": False,
                                "type": "list",
                                "description": 'Options for second GULP calculation. Default: None',
                            },
        "gulp_calc_3_options": {"value": [],
                                "specified": False,
                                "type": "list",
                                "description": 'Options for third GULP calculation. Default: None',
                            },
        "gulp_calc_4_options": {"value": [],
                                "specified": False,
                                "type": "list",
                                "description": 'Options for fourth GULP calculation. Default: None',
                            },
        "gulp_calc_5_options": {"value": [],
                                "specified": False,
                                "type": "list",
                                "description": 'Options for fifth GULP calculation. Default: None',
                            },
        "gulp_shells": {"value": [],
                        "specified": False,
                        "type": "list",
                        "description": 'List of atoms to have a shell attached.',
                    },
        "initial_structure_file": {"value": "",
                                   "specified": False,
                                   "type": "string",
                                   "description": 'If specified, read in the initial structure from this cif file.',
                               },
        "max_structures": {"value": 10,
                           "specified": False,
                           "type": "integer",
                           "description": 'This run of the code will terminate after this number of structures have been considered in this and all previous runs.',
                       },
        "neighbourhood_atom_distance_limit": {"value": 1.0,
                                              "specified": False,
                                              "type": "float",
                                              "description": 'The minimum distance allowed between atoms in the local combinatorial neighbourhood method. Default: 1.0',
        },        
        "num_calc_stages": {"value": 1,
                            "specified": False,
                            "type": "integer",
                            "description": 'Number of GULP/VASP calculations to be run for each structure. Default: 1.',
                        },
        "num_neighbourhood_points": {"value": 1,
                                     "specified": False,
                                     "type": "integer",
                                     "description": 'The number of points used along each axis in the local combinatorial neighbourhood method. Default: 1',
        },
        "num_structures": {"value": 10,
                           "specified": False,
                           "type": "integer",
                           "description": 'The number of structures we will consider in this run of the code.',
                       },
        "number_weightings": {"value": "arithmetic",
                              "specified": False,
                              "supported_values": ["arithmetic", "geometric", "uniform", "pinned_pair"],
                              "type": "string on list",
                              "description": 'The method used to construct the weightings used to choose the number of atoms to swap. Supported values are "arithmetic" (default), "geometric", "uniform", and "pinned_pair".',
                      },
        "output_file": {"value": calc_name + ".chemdash",
                        "specified": False,
                        "type": "string",
                        "description": 'Output file for this calculation. Records the swaps for each structure, energies and acceptances.',
                    },
        "output_trajectory": {"value": "True",
                              "specified": False,
                              "type": "boolean",
                              "description": 'If true, write ASE trajectory files. Default: True',
        },
        "pair_weighting": {"value": 1.0,
                           "specified": False,
                           "type": "float",
                           "description": 'The initial proportional probability of swapping 2 atoms compared to any other number when using the "pinned_pair" option for "number_weightings". Default: 1.0',
                      },
        "pair_weighting_scale_factor": {"value": 1.0,
                                        "specified": False,
                                        "type": "float",
                                        "description": 'The factor by which we increase the proportional probability of swapping 2 atoms compared to any other number when we explore new basins (we decrease by the inverse factor for repeated basins) when using the "pinned_pair" option for "number_weightings". Default: 1.0',
                      },
        "potentials_file": {"value": calc_name + "_potentials.dat",
                            "specified": False,
                            "type": "string",
                            "description": 'Potentials file for this calculation. Records the site potentials for each structure.',
                        },
        "potential_derivs_file": {"value": calc_name + "_derivs.dat",
                                  "specified": False,
                                  "type": "string",
                                  "description": 'Potential derivs file for this calculation. Records the resolved derivatives of the site potentials for each structure.',
                              },
        "random_seed": {"value": 42,
                        "specified": False,
                        "type": "integer",
                        "description": 'The value used to seed the random number generator. Alternatively, the code can generate one itself, which is the default behaviour.',
                    },
        "restart": {"value": "False",
                    "specified": False,
                    "type": "boolean",
                    "description": 'If True, use data in a numpy archive (specified by restart_file keyword) to continue a previous run. Default: False',
                },
        "restart_file": {"value": "restart.npz",
                         "specified": False,
                         "type": "string",
                         "description": 'Name of the numpy archive from which to read data in order to continue a previous run.  Default: "restart.npz"',
                     },
        "rng_warm_up": {"value": 0,
                        "specified": False,
                        "type": "integer",
                        "description": 'Number of values from the RNG to generate and discard after seeding the generator. Default: 0.',
                    },
        "save_outcar": {"value": "False",
                        "specified": False,
                        "type": "boolean",
                        "description": 'If True, retain the final OUTCAR file from each structure optimised with VASP as "OUTCAR_[structure_index]". Default: False.',
                               },
        "search_local_neighbourhood": {"value": "False",
                                       "specified": False,
                                       "type": "boolean",
                                       "description": 'If True, uses the local combinatorial neighbourhood method to try and lower the energy of structures prior to relaxation. Default: False',
        },
        "seed_bits": {"value": 64,
                      "specified": False,
                      "supported_values": [32, 64],
                      "type": "integer on list",
                      "description": 'The number of bits used in the seed of the random number generator, The allowed values are 32 and 64. Default: 64',
                  },
        "swap_groups": {"value": "cations,anions,atoms,all",
                        "specified": False,
                        "supported_values": ["cations", "anions", "atoms", "all", "atoms-vacancies"],
                        "type": "list of lists",
                                             "description": 'The groups of atoms that can be involved in swaps. The default groups are: "cations", "anions", "atoms", and "all" (atoms and vacancies). The input can include these, the additional swap group "atoms-vacancies" (always swap atoms with vacancies), or any custom swap group in the format [Chemical Symbol]-[Chemical Symbol]-[Chemical Symbol]. . . e.g., "Sr-X". A weighting can also be specified for each group as follows: "cations:1, atoms:2, all:3".',
                    },
        "temp": {"value": 0.0,
                 "specified": False,
                 "type": "float",
                 "description": 'The Monte-Carlo temperature (strictly, the value of kT in eV). Determines whether swaps to basins of higher energy are accepted. Default: 0.0',
             },
        "temp_scale_factor": {"value": 1.0,
                              "specified": False,
                              "type": "float",
                              "description": 'The factor by which we increase the temperature after rejected structures (we decrease by the inverse factor for accepted structures). Default: 1.0',
             },
        "update_atoms": {"value": "True",
                         "specified": False,
                         "type": "boolean",
                         "description": 'If true, swap atoms based on relaxed structures, rather than initial structures. Default: True.',
                     },
        "vacancy_grid": {"value": "True",
                         "specified": False,
                         "type": "boolean",
                         "description": 'If true, apply vacancy grids to each structure in which we will swap atoms. Default: True.'
                           },
        "vacancy_exclusion_radius": {"value": "2.0",
                               "specified": False,
                               "type": "float",
                               "description": 'The minimum allowable distance between an atom and a vacancy on the vacancy grid. Default: 2.0 A0',
                           },
        "vacancy_separation": {"value": "1.0",
                               "specified": False,
                               "type": "float",
                               "description": 'The nearest neighbour distance between two vacancies on the vacancy grid. Default: 1.0 A0',
                           },
        "vasp_executable": {"value": "vasp",
                            "specified": False,
                            "type": "string",
                            "description": 'The filepath of the vasp executable to be used. Default: "./vasp"',
                        },
        "vasp_kpoints": {"value": "1",
                         "specified": False,
                         "type": "integer",
                         "description": 'Number of k points to use in VASP calculations. Default: 1',
                     },
        "vasp_max_convergence_calcs": {"value": 10,
                                       "specified": False,
                                       "type": "integer",
                                       "description": 'Maximum number of VASP calculations performed in the final stage for convergence -- we abandon the calculation after this. Default: 10.',
                                   },
        "vasp_pp_dir": {"value": ".",
                        "specified": False,
                        "type": "string",
                        "description": 'Path to directory containing VASP pseudopotential files. Default: ".".',
                    },
        "vasp_pp_setups": {"value": "",
                           "specified": False,
                           "type": "dictionary",
                           "description": 'Pseudopotential file extensions for each element.',
                       },
        "vasp_settings": {"value": "",
                          "specified": False,
                          "type": "dictionary",
                          "description": 'Settings for all VASP calculations. Default: None.',
                      },
        "vasp_calc_1_settings": {"value": "",
                                 "specified": False,
                                 "type": "dictionary",
                                 "description": 'Settings for the first stage of the VASP calculation. Default: None.',
                             },
        "vasp_calc_2_settings": {"value": "",
                                 "specified": False,
                                 "type": "dictionary",
                                 "description": 'Settings for the second stage of the VASP calculation. Default: None.',
                             },
        "vasp_calc_3_settings": {"value": "",
                                 "specified": False,
                                 "type": "dictionary",
                                 "description": 'Settings for the third stage of the VASP calculation. Default: None.',
                             },
        "vasp_calc_4_settings": {"value": "",
                                 "specified": False,
                                 "type": "dictionary",
                                 "description": 'Settings for the fourth stage of the VASP calculation. Default: None.',
                             },
        "vasp_calc_5_settings": {"value": "",
                                 "specified": False,
                                 "type": "dictionary",
                                 "description": 'Settings for the fifth stage of the VASP calculation. Default: None.',
                             },
        "verbosity": {"value": "verbose",
                      "specified": False,
                      "supported_values": ["terse", "verbose"],
                      "type": "string on list",
                      "description": 'Controls the level of detail in the output. Valid options are: "verbose", "terse". Default: "verbose"',
                  },
    }

    return params


# =============================================================================
def write_defaults_to_file(input_file):
    """
    This routine writes the default values for all code options to an input
    file.

    Parameters
    ----------
    input_file : string
        Name of input file we will write the default values to.

    Returns
    -------
    None

    ---------------------------------------------------------------------------
    Paul Sharp 27/03/2020
    """

    calc_name = input_file.split(".")[0]
    params = initialise_default_params(calc_name)
    sorted_items = sorted(list(params.items()))

    # Write to file
    with open(input_file, mode="w") as new_input_file:

        for item in sorted_items:
            new_input_file.write('{0}={1}\n'.format(item[0], item[1]["value"]))

    return None


# =============================================================================
def parse_input(input_file, calc_name):
    """
    This routine runs through the input file for this run of the code, checks
    all the keywords are valid, and applies all values.

    Parameters
    ----------
    input_file : string
        Name of input file for this run of the code.
    calc_name : string
        Name of this job -- default value for name of atoms file.

    Returns
    -------
    params : dict
        Dictionary containing each parameter with its value.
    errors : string
        List of error messages for invalid options.

    ---------------------------------------------------------------------------
    Paul Sharp 27/03/2020
    """

    # Set up initial parameter dictionary, with all default values.
    params = initialise_default_params(calc_name)
    errors = []

    # Read in input file
    with open(input_file, mode="r") as in_file:
        file_contents = in_file.readlines()

    # Remove entries that start with a comment character -- either # or ! -- or newline
    file_contents[:] = [x for x in file_contents if not x.startswith(("#", "!", "\n"))]

    # Remove spaces from each entry and newlines -- exclude GULP options because they will contain spaces
    for i in range(0, len(file_contents)):
        if ("gulp" and "options") not in file_contents[i]:
            file_contents[i] = ''.join(file_contents[i].split())

    # Check whether the keywords are valid
    invalid_keywords = []
    for entry in file_contents[:]:
        keyword = entry.split("=")[0].strip()

        if keyword in params:
            params[keyword]["specified"] = True
        else:
            invalid_keywords.append(keyword)
            file_contents.remove(entry)

    if len(invalid_keywords) > 0:
        errors.append("The keyword(s) {0} are invalid.".format(", ".join(invalid_keywords)))

    # Apply the new values to the keywords
    for entry in file_contents:

        keyword = entry.split("=")[0].strip()
        value = entry.split("=")[1].strip()

        params[keyword]["value"] = value

    return params, errors


# =============================================================================
def report_input_file_errors(errors, input_file, error_file):
    """
    This routine prints the error list for an input file in the correct format.

    Parameters
    ----------
    errors : string
        List of error messages for invalid options.
    input_file : string
        Name of input file that was parsed.
    error_file : string
        Name of the file in which we report the errors.

    Returns
    -------
    None

    ---------------------------------------------------------------------------
    Paul Sharp 27/03/2020
    """

    with open(error_file, mode="a") as error_log:

        error_log.write('{0:d} errors were found in the input file: "{1}"\n'.format(len(errors), input_file))

        for error in errors:
            error_log.write('\tERROR in input file "{0}" -- {1}\n'.format(input_file, error))

    return None


# =============================================================================
def convert_values(params, errors):
    """
    This routine runs through the dictionary of input parameters, converts them
    to the correct type, and ensures they are valid.

    Parameters
    ----------
    params : dict
        Dictionary containing each parameter with its value.
    errors : string
        List of error messages for invalid options.

    Returns
    -------
    params : dict
        Dictionary containing each parameter with its value.
    errors : string
        Updated list of error messages for invalid options.

    ---------------------------------------------------------------------------
    Paul Sharp 27/03/2020
    """

    if params["calculator"]["value"] == "gulp":
        params["atom_rankings"]["supported_values"].extend(["bvs+", "site_pot"])

    # Convert to a list with three elements, for a, b, c directions.
    # We expand the list if fewer than three elements are specified.
    # Must do this first as we then need to convert values to integers or floats.
    three_element_list_keys = ["cell_spacing", "grid_points", "vasp_kpoints"]

    for key in three_element_list_keys:
        params[key]["value"], errors = convert_to_three_element_list(key, params[key]["value"], errors)

    # Now take each input option and perform an operation on its value, according to its type.
    # Let GULP/VASP deal with full verification of its keywords and options
    for key, option in params.items():

        if option["type"] == "boolean":
            option["value"], errors = check_boolean(key, option["value"], errors)

        elif option["type"] == "dictionary":
            option["value"], errors = convert_to_dict_with_yaml(key, option["value"], errors)

        elif option["type"] == "integer":
            option["value"], errors = check_positive_int(key, option["value"], errors)

        elif option["type"] == "integer on list":
            option["value"], errors, _ = check_int_on_list(key, option["value"], errors, option["supported_values"])

        elif option["type"] == "float":
            option["value"], errors = check_positive_float(key, option["value"], errors)

        elif option["type"] == "float specified":
            if option["specified"]:
                option["value"], errors = check_positive_float(key, option["value"], errors)

        elif option["type"] == "list":
            try:
                option["value"] = [entry.strip() for entry in option["value"].split(",")]
            except AttributeError:
                pass

        elif option["type"] == "list of lists":
            # This splits the string into a list of lists where the inner list is delimited
            # by a ":" and the outer list is delimited by a ","
            option["value"] = [entry.strip().split(":") for entry in option["value"].split(",")]

        elif option["type"] == "list to string":
            option["value"] = option["value"].replace(",", " ")

        elif option["type"] == "string":
            option["value"] = str(option["value"])

        elif option["type"] == "list of strings":
            option["value"] = str(option["value"])
            option["value"] = option["value"].split(",")

        elif option["type"] == "string on list":
            option["value"], errors, _ = check_str_on_list(key, option["value"], errors, option["supported_values"])

        elif option["type"] == "string list on list":
            option["value"], errors, _ = check_str_list_on_list(key, list(option["value"].upper()),
                                                                errors, option["supported_values"])
        else:
            errors.append("The keyword {0} has the type (1}, which is not a valid type.".format(key, option["type"]))

    return params, errors


# =============================================================================
def handle_dependencies(params, errors):
    """
    This routine runs through the dictionary of input parameters,
    cross-checking options that depend on others.

    Parameters
    ----------
    params : dict
        Dictionary containing each parameter with its value.
    errors : string
        List of error messages for invalid options.

    Returns
    -------
    params : dict
        Dictionary containing each parameter with its value.
    errors : string
        Updated list of error messages for invalid options.

    ---------------------------------------------------------------------------
    Paul Sharp 27/03/2020
    """

    # Extra checks for swap groups
    swap_groups = [group[0] for group in params["swap_groups"]["value"]]
    swap_groups, errors, swap_groups_valid = check_duplicates_in_list("swap_groups", swap_groups, errors)

    params["swap_groups"]["value"], errors, swap_groups_valid = check_swap_groups_valid(params["swap_groups"]["value"],
                                                                                        errors,
                                                                                        params["swap_groups"]["supported_values"])

    # Structures for this run cannot be more than the total number of structures -- REVIEW THESE PARAMETERS
    if params["num_structures"]["value"] > params["max_structures"]["value"]:
        errors.append('"num_structures" should not be larger than "max_structures". "num_structures" is {0:d} and "max_structures" is {1:d}.'.format(params["num_structures"]["value"], params["max_structures"]["value"]))

    # Random seed must be the correct number of bits.
    params["seed_bits"]["value"], errors, seed_bits_valid = check_int_on_list("seed_bits", params["seed_bits"]["value"], errors, params["seed_bits"]["supported_values"])
    if seed_bits_valid:
        params["random_seed"]["value"], errors = check_unsigned_n_bit_int("random_seed", params["random_seed"]["value"], errors, params["seed_bits"]["value"])
    else:
        errors.append('"random_seed" has not been verified because it must be either a 32- or 64-bit integer, but an invalid number of bits was entered.')

    # For close packed grids, the number of layers must be a multiple of the stacking sequence as we cannot have repeated layers
    if params["cp_stacking_sequence"]["specified"]:

        for i in range(0, len(params["cp_stacking_sequence"]["value"])):
            if params["cp_stacking_sequence"]["value"][i - 1] == params["cp_stacking_sequence"]["value"][i]:
                errors.append("The supplied close packed stacking sequence is {0}, which features repeated layers.".format("".join(params["cp_stacking_sequence"]["value"])))
                break

        if (int(params["grid_points"]["value"][2]) % len(params["cp_stacking_sequence"]["value"]) != 0):
            errors.append("The supplied close packed stacking sequence is {0}, but this is incommensurate with the {1:d} anion layers requested.".format("".join(params["cp_stacking_sequence"]["value"]), params["grid_points"]["value"][2]))
        else:
            # If stacking sequence is ok, multiply up to the number of layers
            params["cp_stacking_sequence"]["value"] *= params["grid_points"]["value"][2] // len(params["cp_stacking_sequence"]["value"])

    # GULP files need to be the length of the number of calculations
    params["gulp_files"]["value"] = params["gulp_files"]["value"][:params["num_calc_stages"]["value"]]

    # If we are not restarting, we need a ".atoms" file -- check the file given is valid and exists
    if not params["restart"]["value"]:
        if (params["atoms_file"]["value"].split(".")[1] != "atoms"):
            errors.append('the ".atoms" file is specified as "{0}", but that file does not have the ".atoms" extension.'.format(params["atoms_file"]["value"]))
        if (not os.path.isfile(params["atoms_file"]["value"])):
            errors.append('the ".atoms" file is specified as "{0}", but that file does not exist.'.format(params["atoms_file"]["value"]))

    # If we are restarting, we need a ".npz" file -- check the file given is valid and exists
    else:
        if (params["restart_file"]["value"].split(".")[1] != "npz"):
            errors.append('the restart ".npz" file is specified as "{0}", but that file does not have the ".npz" extension.'.format(params["restart_file"]["value"]))
        if (not os.path.isfile(params["restart_file"]["value"])):
            errors.append('the restart ".npz" file is specified as "{0}", but that file does not exist.'.format(params["restart_file"]["value"]))

    # If we are initialising from a ".cif" file -- check the file given is valid and exists
    if params["initial_structure_file"]["specified"]:
        if (params["initial_structure_file"]["value"].split(".")[1] != "cif"):
            errors.append('the ".cif" file for the initial structure is specified as "{0}", but that file does not have the ".cif" extension.'.format(params["initial_structure_file"]["value"]))
        if (not os.path.isfile(params["initial_structure_file"]["value"])):
            errors.append('the ".cif" file for the initial structure is specified as "{0}", but that file does not exist.'.format(params["initial_structure_file"]["value"]))

    # If specifying weightings for each swap group, we must specify a single weighting for all of the groups.
    swap_lengths = [len(group) for group in params["swap_groups"]["value"]]
    first_length = swap_lengths[0]

    if not all(length == first_length for length in swap_lengths):
        errors.append('An unequal number of weightings have been specified for the "swap_groups", "swap_groups" should be specified alone, or ALL should be given a weighting.')

    elif not (all(length == 1 for length in swap_lengths) or all(length == 2 for length in swap_lengths)):
        errors.append('Too many weightings have been specified for the "swap_groups".')

    # This means we have swap groups and weightings -- we need to convert the weights to floats
    if all(length == 2 for length in swap_lengths):

        for i in range(0, len(params["swap_groups"]["value"])):

            # This if statement allows us to deal with either fractions or decimals
            if "/" in params["swap_groups"]["value"][i][1]:
                try:
                    params["swap_groups"]["value"][i][1] = float(params["swap_groups"]["value"][i][1].split("/")[0]) / float(params["swap_groups"]["value"][i][1].split("/")[1])
                except ValueError:
                    errors.append('The weighting of the swap group {x[0]} is {x[1]}, but should be a positive number.'.format(x=params["swap_groups"]["value"][i]))
                else:
                    if params["swap_groups"]["value"][i][1] < 0.0:
                        errors.append('The weighting of the swap group {x[0]} is {x[1]}, but should be a positive number.'.format(x=params["swap_groups"]["value"][i]))
            else:
                try:
                    params["swap_groups"]["value"][i][1] = float(params["swap_groups"]["value"][i][1])
                except ValueError:
                    errors.append('The weighting of the swap group {x[0]} is {x[1]}, but it should be a positive number.'.format(params["swap_groups"]["value"][i]))
                else:
                    if params["swap_groups"]["value"][i][1] < 0.0:
                        errors.append('The weighting of the swap group {x[0]} is {x[1]}, but should be a positive number.'.format(params["swap_groups"]["value"][i]))

    # We need to merge the kpoints and pseudopotential setups values into the main vasp settings
    params["vasp_settings"]["value"].update({"kpts": params["vasp_kpoints"]["value"], "setups": params["vasp_pp_setups"]["value"]})

    # If we are not updating atom geometries AND using vacancy grids, we cannot use directed swapping.
    if (not params["update_atoms"]["value"]) and (params["vacancy_grid"]["value"]) and (params["atom_rankings"]["value"] != "random"):
        errors.append('Directed swapping is not supported for the use of vacancy grids when atom geometries are not updated. Please set either "update_atoms" to "True" and/or "vacancy_grid" to "False" in order to use directed swapping.')        
  
    return params, errors


# =============================================================================
def check_positive_float(keyword, value, errors):
    """
    This routine takes a single value or a list of values and ensures that all
    are positive floats, updating the error list if not.

    Parameters
    ----------
    keyword : str
        Dictionary keyword for the input option we are considering.
    value : str
        The value or list of values inputted for this option.
    errors : string
        List of error messages for invalid options.

    Returns
    -------
    value : float/str
        The value or list of values converted to a positive float,
        or kept as a string if conversion is not possible.
    errors : string
        Updated list of error messages for invalid options.

    ---------------------------------------------------------------------------
    Paul Sharp 27/03/2020
    """

    try:
        value = float(value)
    except ValueError:
        errors.append('"{0}" is {1}, but should be a positive number.'.format(keyword, value))
    # A TypeError means that a list was inputted
    except TypeError:
        value, errors = check_list_positive_float(keyword, value, errors)
    else:
        if value < 0.0:
            errors.append('"{0}" is {1}, but should be a positive number.'.format(keyword, value))

    return value, errors


# =============================================================================
def check_list_positive_float(keyword, value, errors):
    """
    This routine takes a list of values and ensures that they are all positive
    floats, updating the error list if not.

    Parameters
    ----------
    keyword : str
        Dictionary keyword for the input option we are considering.
    value : str
        The list of values inputted for this option.
    errors : string
        List of error messages for invalid options.

    Returns
    -------
    value : float/str
        The list of values converted to a positive floats,
        or kept as a string if conversion is not possible.
    errors : string
        Updated list of error messages for invalid options.

    ---------------------------------------------------------------------------
    Paul Sharp 27/03/2020
    """

    try:
        value = list(map(float, value))
    except ValueError:
        errors.append('"{0}" is {1}, but all values should be positive floats.'.format(keyword, value))
    else:
        if any(i < 0.0 for i in value):
            errors.append('"{0}" is {1}, but all values should be positive floats.'.format(keyword, value))

    return value, errors


# =============================================================================
def check_positive_int(keyword, value, errors):
    """
    This routine takes a value or list of values and ensures that it is a
    positive integer, updating the error list if not.

    Parameters
    ----------
    keyword : str
        Dictionary keyword for the input option we are considering.
    value : str
        The value or list of values inputted for this option.
    errors : string
        List of error messages for invalid options.

    Returns
    -------
    value : int/str
        The value or list of values converted to a positive int,
        or kept as a string if conversion is not possible.
    errors : string
        Updated list of error messages for invalid options.

    ---------------------------------------------------------------------------
    Paul Sharp 27/03/2020
    """

    try:
        value = int(value)
    except ValueError:
        errors.append('"{0}" is {1}, but should be a positive integer.'.format(keyword, value))
    # A TypeError means that a list was inputted
    except TypeError:
        value, errors = check_list_positive_int(keyword, value, errors)
    else:
        if value < 0:
            errors.append('"{0}" is {1}, but should be a positive integer.'.format(keyword, value))

    return value, errors


# =============================================================================
def check_list_positive_int(keyword, value, errors):
    """
    This routine takes a list of values and ensures that they are all positive
    integers, updating the error list if not.

    Parameters
    ----------
    keyword : str
        Dictionary keyword for the input option we are considering.
    value : str
        The list of values inputted for this option.
    errors : string
        List of error messages for invalid options.

    Returns
    -------
    value : int/str
        The list of values converted to a positive ints,
        or kept as a string if conversion is not possible.
    errors : string
        Updated list of error messages for invalid options.

    ---------------------------------------------------------------------------
    Paul Sharp 27/03/2020
    """

    try:
        value = list(map(int, value))
    except ValueError:
        errors.append('"{0}" is {1}, but all values should be positive integers.'.format(keyword, value))
    else:
        if any(i < 0 for i in value):
            errors.append('"{0}" is {1}, but all values should be positive integers.'.format(keyword, value))

    return value, errors


# =============================================================================
def check_even_int(keyword, value, errors):
    """
    This routine takes a value or list of values and ensures that it is an even
    integer, updating the error list if not.

    Parameters
    ----------
    keyword : str
        Dictionary keyword for the input option we are considering.
    value : str
        The value or list of values inputted for this option.
    errors : string
        List of error messages for invalid options.

    Returns
    -------
    value : int/str
        The value or list of values converted to an even int,
        or kept as a string if conversion is not possible.
    errors : string
        Updated list of error messages for invalid options.

    ---------------------------------------------------------------------------
    Paul Sharp 27/03/2020
    """

    try:
        value = int(value)
    except ValueError:
        errors.append('"{0}" is {1}, but should be an even integer.'.format(keyword, value))
    # A TypeError means that a list was inputted
    except TypeError:
        value, errors = check_list_even_int(keyword, value, errors)
    else:
        if value % 2 != 0:
            errors.append('"{0}" is {1}, but should be an even integer.'.format(keyword, value))

    return value, errors


# =============================================================================
def check_list_even_int(keyword, value, errors):
    """
    This routine takes a list of values and ensures that they are all even
    integers, updating the error list if not.

    Parameters
    ----------
    keyword : str
        Dictionary keyword for the input option we are considering.
    value : str
        The list of values inputted for this option.
    errors : string
        List of error messages for invalid options.

    Returns
    -------
    value : int/str
        The list of values converted to even ints,
        or kept as a string if conversion is not possible.
    errors : string
        Updated list of error messages for invalid options.

    ---------------------------------------------------------------------------
    Paul Sharp 27/03/2020
    """

    try:
        value = list(map(int, value))
    except ValueError:
        errors.append('"{0}" is {1}, but all values should be even integers.'.format(keyword, value))
    else:
        if any(i % 2 != 0 for i in value):
            errors.append('"{0}" is {1}, but all values should be even integers.'.format(keyword, value))

    return value, errors


# =============================================================================
def check_int_on_list(keyword, value, errors, allowed_ints):
    """
    This routine takes a value and ensures that it is an integer that is
    included in the specified list, updating the error list if not.

    Parameters
    ----------
    keyword : str
        Dictionary keyword for the input option we are considering.
    value : str
        The value inputted for this option.
    errors : string
        List of error messages for invalid options.
    allowed_ints : int
        List of allowed integers for value.

    Returns
    -------
    value : int/str
        The value converted to a positive int,
        or kept as a string if conversion is not possible.
    errors : string
        Updated list of error messages for invalid options.
    valid : logical
        True if value is in allowed_ints.

    ---------------------------------------------------------------------------
    Paul Sharp 27/03/2020
    """

    valid = True
    try:
        value = int(value)
    except ValueError:
        errors.append('"{0}" is {1}, but should be one of the supported values: {2}'.format(keyword, value, ', '.join([str(x) for x in allowed_ints])))
        valid = False
    else:
        if value not in allowed_ints:
            errors.append('"{0}" is {1}, but should be one of the supported values: {2}'.format(keyword, value, ', '.join([str(x) for x in allowed_ints])))
            valid = False

    return value, errors, valid


# =============================================================================
def check_unsigned_n_bit_int(keyword, value, errors, bits):
    """
    This routine takes a value and ensures that it is a positive integer of the
    specified number of bits or fewer, updating the error list if not.

    Parameters
    ----------
    keyword : str
        Dictionary keyword for the input option we are considering.
    value : str
        The value inputted for this option.
    errors : string
        List of error messages for invalid options.
    bits : int
        Limit of the number of bits for this integer.

    Returns
    -------
    value : int/str
        The value converted to a positive int,
        or kept as a string if conversion is not possible.
    errors : string
        Updated list of error messages for invalid options.

    ---------------------------------------------------------------------------
    Paul Sharp 27/03/2020
    """

    try:
        value = int(value)
    except ValueError:
        errors.append('"{0}" is {1}, but should be an unsigned {2:d}-bit integer.'.format(keyword, value, bits))
    else:
        if (value < 0) or (value > (2 ** int(bits)) - 1):
            errors.append('"{0}" is {1}, but should be an unsigned {2:d}-bit integer.'.format(keyword, value, bits))

    return value, errors


# =============================================================================
def check_str_on_list(keyword, value, errors, allowed_strings):
    """
    This routine takes a string and ensures that it is included in the
    specified list, updating the error list if not.

    Parameters
    ----------
    keyword : str
        Dictionary keyword for the input option we are considering.
    value : str
        The values inputted for this option.
    errors : string
        List of error messages for invalid options.
    allowed_strings : str
        List of allowed strings for value.

    Returns
    -------
    value : str
        The string of interest.
    errors : string
        Updated list of error messages for invalid options.
    valid : logical
        True if value is in allowed_strings.

    ---------------------------------------------------------------------------
    Paul Sharp 27/03/2020
    """

    valid = True
    if value not in allowed_strings:
        errors.append('"{0}" is {1}, but should only be one of the supported values: {2}'.format(keyword, value, ', '.join([str(x) for x in allowed_strings])))
        valid = False

    return value, errors, valid


# =============================================================================
def check_str_list_on_list(keyword, value, errors, allowed_strings):
    """
    This routine takes a list of values and ensures that each is a string that
    is included in the specified list, updating the error list if not.

    Parameters
    ----------
    keyword : str
        Dictionary keyword for the input option we are considering.
    value : str
        The values inputted for this option.
    errors : string
        List of error messages for invalid options.
    allowed_strings : str
        List of allowed strings for value.

    Returns
    -------
    value : str
        The string of interest.
    errors : string
        Updated list of error messages for invalid options.
    valid : logical
        True if value is in allowed_strings.

    ---------------------------------------------------------------------------
    Paul Sharp 27/03/2020
    """

    valid = True
    for item in value:
        if item not in allowed_strings:
            errors.append('"{0}" contains {1}, but should only contain the supported values: {2}'.format(keyword, item, ', '.join([str(x) for x in allowed_strings])))
            valid = False

    return value, errors, valid


# =============================================================================
def check_boolean(keyword, value, errors):
    """
    This routine takes a value and ensures that it corresponds to an allowed
    set of values for True or False, updating the error list if not.

    Parameters
    ----------
    keyword : str
        Dictionary keyword for the input option we are considering.
    value : str
        The value inputted for this option.
    errors : string
        List of error messages for invalid options.

    Returns
    -------
    value : bool/str
        The value converted to a boolean,
        or kept as a string if conversion is not possible.
    errors : string
        Updated list of error messages for invalid options.

    ---------------------------------------------------------------------------
    Paul Sharp 27/03/2020
    """

    true_synonyms = ["true", "t", "yes", "on"]
    false_synonyms = ["false", "f", "no", "off"]

    if value.lower() in true_synonyms:
        value = True
    elif value.lower() in false_synonyms:
        value = False
    else:
        errors.append('"{0}" is {1}, but should be "True" or "False".'.format(keyword, value))

    return value, errors


# =============================================================================
def check_duplicates_in_list(keyword, value, errors):
    """
    This routine takes a list of values and ensures that each is a unique
    string, updating the error list if not.

    Parameters
    ----------
    keyword : str
        Dictionary keyword for the input option we are considering.
    value : str
        The values inputted for this option.
    errors : string
        List of error messages for invalid options.

    Returns
    -------
    value : str
        The string of interest.
    errors : string
        Updated list of error messages for invalid options.
    valid : logical
        True if all strings in the list are unique.

    ---------------------------------------------------------------------------
    Paul Sharp 27/03/2020
    """

    valid = True
    uniques = []
    for item in value:
        if item not in uniques:
            uniques.append(item)
        else:
            errors.append('"{0}" contains {1} more than once, but each item in this list should only appear once.'.format(keyword, item))
            valid = False

    return value, errors, valid


# =============================================================================
def check_swap_groups_valid(swap_groups, errors, main_swap_groups):
    """
    This routine takes the swap groups and ensures that each is a string that
    is included in the specified list, or a valid custom swap group, updating
    the error list if not.

    Parameters
    ----------
    swap_groups : str
        The values inputted as swap groups.
    errors : string
        List of error messages for invalid options.
    main_swap_groups : str
        List of allowed main swap groups.

    Returns
    -------
    swap_groups : str
        The string of swap groups.
    errors : string
        Updated list of error messages for invalid options.
    valid : logical
        True if all swap groups are valid.

    ---------------------------------------------------------------------------
    Paul Sharp 27/03/2020
    """

    valid = True
    swap_group_names = [group[0] for group in swap_groups]

    for group in swap_group_names:

        if group in main_swap_groups:
            continue

        # If not a main swap group, then check custom swap groups
        # NB format should be [Chemical Symbol]-[Chemical Symbol]-[Chemical Symbol] . . .
        group_elements = group.split("-")

        # Custom swap group is invalid if there is only one element, elements are repeated, or any of the chemical symbols are not valid.
        if (len(group_elements) < 2 or sorted(group_elements) != sorted(list(set(group_elements))) or not set(group_elements).issubset(ase_chemical_symbols)):

            errors.append('"swap_groups" contains "{0}", which is not a valid swap group. The main swap groups are "{1}", and custom swap groups should be in the format "[Chemical Symbol]-[Chemical Symbol]-[Chemical Symbol] . . .", with each chemical symbol appearing only once.'.format(group, '", "'.join([str(x) for x in main_swap_groups])))
            valid = False

    return swap_groups, errors, valid


# =============================================================================
def convert_to_three_element_list(keyword, value, errors):
    """
    This routine takes a string consisting of a particular number of values and
    converts it to a list of exactly three values.

    Parameters
    ----------
    value : str
        String that we want to convert to a three-element list.
    errors : str
        List of error messages for invalid options.

    Returns
    -------
    value : list
        A list of three elements, containing the specified values.
    errors : string
        Updated list of error messages for invalid options.

    ---------------------------------------------------------------------------
    Paul Sharp 27/03/2020
    """

    value = value.split(",")

    if len(value) > 3:
        errors.append('"{0}" should be a list of up to three values, but {1:d} values have been supplied.'.format(keyword, len(value)))

    else:
        # This operation expands the list of values in the a, b, c directions up to three values
        # This is done by expanding the first value in the list the appropriate number of times
        # If one value is specified we will get [a, a, a] -- cubic
        # If two values are specified we will get [a, a, c] -- tetragonal
        # If three values are specified we will get [a, b, c] -- orthorhombic
        value[0:1] *= (4 - len(value))

    return value, errors


# =============================================================================
def convert_to_dict_with_yaml(keyword, value, errors):
    """
    This routine takes a string and converts it to valid YAML syntax, then uses
    YAML to convert it to a dictionary, with variable types automatically
    detected.

    Parameters
    ----------
    value : str
        String that we want to convert to a dictionary.
    errors : str
        List of error messages for invalid options.

    Returns
    -------
    yaml_dict : dict
        A dictionary of the input, with types automatically detected.
    errors : string
        Updated list of error messages for invalid options.

    ---------------------------------------------------------------------------
    Paul Sharp 27/03/2020
    """

    if value is not None:

        # Convert input to syntactically valid YAML
        yaml_input = "{" + value.replace(":", ": ") + "}"

        yaml_input = yaml_input.replace("none", "null")
        yaml_input = yaml_input.replace("None", "null")
        yaml_input = yaml_input.replace("NONE", "null")

        try:
            yaml_dict = yaml.safe_load(yaml_input)
        except yaml.scanner.ScannerError:
            yaml_dict = {}
            errors.append('"{0}" is "{1}". YAML failed to convert this to a valid dictionary. The contents should be syntactically valid Python.'.format(keyword, value))

    else:
        yaml_dict = None

    return yaml_dict, errors
