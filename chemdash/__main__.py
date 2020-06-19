"""
|=============================================================================|
|                             C H E M   D A S H                               |
|=============================================================================|
|                                                                             |
| Welcome to the ChemDASH structure prediction code.                          |
|                                                                             |
| ChemDASH predicts the structure of new materials by placing atoms on a grid |
| and exploring the potential energy surface using a basin hopping approach.  |
|                                                                             |
| The structure is manipulated by swapping the positions of a number of atoms |
| in the structure with other atoms or vacant points on the original grid.    |
|                                                                             |
| The code makes extensive use of the Atomic Simulation library (ase),        |
| allowing us to perform structure relaxation using GULP or VASP.             |
|                                                                             |
| In order to use the code, type:                                             |
|                                                                             |
|         python chemdash <job name>                                          |
|                                                                             |
| where <job name> is the prefix of the files <job name>.input and (by        |
| default) <job name>.atoms. The input file contains options for this code    |
| and GULP/VASP, and the atoms file contains the symbol, number and oxidation |
| state of all atoms to be used.                                              |
|                                                                             |
|                                                                             |
| Options                                                                     |
| --------                                                                    |
|     -h, --help                                                              |
|         Print this menu and exit.                                           |
|     -i, --input                                                             |
|         Print all options for the ".input" file with a description of each  |
|         option.                                                             |
|     -p, --parse                                                             |
|         Parse input file(s), report any errors and exit.                    |
|     -s, --symm, --symmetry                                                  |
|         Look for higher symmetry groups in a supplied cif file.             |
|     -w, --write                                                             |
|         Write an input file that includes all keywords with their default   |
|         values and exit.                                                    |
|                                                                             |
|-----------------------------------------------------------------------------|
| Paul Sharp 27/03/2020                                                       |
|=============================================================================|
"""

from builtins import range

import argparse
import ase.io
import os
import sys

import inputs
import master_code
import symmetry


# =============================================================================
# =============================================================================
def main(args=None):
    """
    Parse the command line arguments given when the package is run, then call the main ChemDASH code.

    Parameters
    ----------
    args : string
        The command line arguments provided when the package is run.

    Returns
    -------
    None

    ---------------------------------------------------------------------------
    Paul Sharp 27/03/2020
    """

    if args is None:
        args = sys.argv[1:]

    parser = argparse.ArgumentParser(description='ChemDASH Structure Prediction Code',
                                     epilog='This code is developed by Paul Sharp at the University of Liverpool. Please direct enquiries to Paul.Sharp@liverpool.ac.uk',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('seed', nargs='?', default=None,
                        help='The seed name for this calculation, which forms the name of the ".input" and ".atoms" files.')
    parser.add_argument('-i', '--input', action="store_true", default=False,
                        help='Print all options for the ".input" file with a description of each option.')
    parser.add_argument('-p', '--parse', nargs='+', metavar="<input file>",
                        help='Parse the given input file, report any errors and exit.')  # include: type="File --see docs"
    parser.add_argument('-s', '--symm', '--symmetry', metavar="<cif file>",
                        help='Use spglib to look for higher symmetry in the supplied cif file, and write to a new file "<cif file>_symm.cif".')  # include: type="File --see docs"
    # parser.add_argument('-t', '--test', action='store_true', help='Run the test suite.')  # include: type="File --see docs"
    parser.add_argument('-w', '--write', nargs='?', const="defaults.input",
                        metavar="<input file>",
                        help='Write an input file that includes all keywords with their default values to the given file and exit.')  # include: type="File --see docs"
    parser.add_argument('-v', '--version', action='version',
                        version='%(prog)s 1.1')

    opts = parser.parse_args(args)

    if opts.input:

        params = inputs.initialise_default_params("")
        sorted_items = sorted(list(params.items()))
        print()

        for item in sorted_items:
            print("{0:35} {1}".format(item[0], item[1]["description"]))
        print()

    if opts.parse is not None:

        for input_file in opts.parse:

            # Check file exists
            if not os.path.isfile(input_file):
                print('WARNING - the file "{0}" does not exist'.format(input_file))
                continue

            calc_name = input_file.split(".")[0]
            params, errors = inputs.parse_input(input_file, calc_name)

            # Convert, and ensure all input file parameters are of the correct type
            params, errors = inputs.convert_values(params, errors)
            params, errors = inputs.handle_dependencies(params, errors)

            if len(errors) > 0:
                print('{0:d} errors were found in the input file: "{1}"\n'.format(len(errors), input_file))
                print('Please refer to the error log "{0}".\n'.format(calc_name + ".error"))
                inputs.report_input_file_errors(errors, input_file, calc_name + ".error")
            else:
                print('The file "{0}" is a valid input file.'.format(input_file))
                print()
                print("\tPlease note that keywords, options and settings for GULP/VASP will be verified in GULP/VASP calculations.")
                print("\tAlso, when the initial structure is composed, the validity of any supplied swap groups will be verified at that point.")
                print()

    if opts.symm is not None:

        print(opts.symm)

        if not os.path.isfile(opts.symm):

            print('ERROR -- the given file "{0}" does not exist'.format(opts.symm))

        else:

            structure = ase.io.read(opts.symm)
            symmetrised_structure = symmetry.symmetrise_atoms(structure)
            symmetrised_structure_file = opts.symm.split(".")[0] + "_symm.cif"
            symmetrised_structure.write(symmetrised_structure_file, format="cif")
            print()
            print('The structure generated by spglib has been written to "{0}".'.format(symmetrised_structure_file))
            print()

    if opts.write is not None:

        # Check file exists, call write routine if not
        if not os.path.isfile(opts.write):
            inputs.write_defaults_to_file(opts.write)
            print()
            print('Default values written to input file "{0}"'.format(opts.write))
            print()

        else:
            print()
            print('ERROR - the file "{0}" already exists'.format(opts.write))
            print()

    # if opts.test:

        # pytest.main()

    if opts.seed is not None:
        calc_seed = opts.seed.split(".")[0]
        master_code.ChemDASH(calc_seed)
    # else:
    #     print('Please supply the calculation seed from your ".input" file -- USAGE: "python prime <seed>" for input file "<seed>.input".')

    return None

# =============================================================================
# =============================================================================

if __name__ == "__main__":
    main()
