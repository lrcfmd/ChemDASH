# ChemDASH
Chemically Directed Atom Swap Hopping -- Crystal structure prediction by swapping atoms in unfavourable chemical environments

Introduction
============

ChemDASH is a crystal structure prediction code written by Paul Sharp and developed at the
University of Liverpool. ChemDASH is written in python 3.5+, and depends
on the atomic simulation environment (ASE), spglib, and their subsequent dependencies. ChemDASH implements
the basin hopping method to explore the potential energy surface, with
atom swaps used to generate new structures. Atoms can be swapped at
random, or we can use the method of *directed swapping* to rank each
atom according to its chemical environment, with atoms in the least
favourable environments prioritised for swapping. Structures in ChemDASH
can be initialised by populating cation and anion sites on
initialisation grids, or from a CIF file. Structural optimisation can be
done using either the GULP or VASP packages.

Usage
=====

To run a ChemDASH calculation, two input files are required: a “.atoms”
file and a “.input” file. By default they must both have the same
basename. With valid files and a copy of ChemDASH in the current working
directory, ChemDASH is run by typing:

      python chemdash <basename>

where <span>$\left\langle \text{basename} \right\rangle$</span> is the
basename of both the “.atoms” and “.input” files. The output of the
calculation is written to the file
“<span>$\left\langle \text{basename} \right\rangle$</span>.chemdash”. If
there are errors in either the “.atoms” or the “.input” files, then the
calculation is stopped, with errors listed in the file
“<span>$\left\langle \text{basename} \right\rangle$</span>.error”. To
restart a ChemDASH run, with a restart file present, run with
“restart=True” in the input file.\
\
ChemDASH does not have to be installed in the working directory, in this
case, run ChemDASH with:

      python <filepath_to_ChemDASH_directory> <basename>

for example:

      python /home/software/ChemDASH/chemdash <basename>

Options
-------

There are a number of flags that enable ChemDASH options, these are
listed by typing:

      python chemdash -h

These options are:

      -h, --help            show this help message and exit

      -i, --input           Print all options for the ".input" file with a
                            description of each option. (default: False)

      -p <input file> [<input file> ...], --parse <input file> [<input file> ...] 
                            Parse the given input file, report any errors and
                            exit. (default: None)

      -s <cif file>, --symm <cif file>, --symmetry <cif_file>
                            Use spglib to look for higher symmetry in the supplied
                            cif file, and write to a new file "<cif_file>_symm.cif".
                            (default: None)

      -w [<input file>], --write [<input file>]
                            Write an input file that includes all keywords with
                            their default values to the given file and exit.
                            (default: None)

      -v, --version         show ChemDASH version number and exit

Python Libraries
----------------

ChemDASH requires python version 3.5+, and the following python
libraries:

      ase (Atomic Simulation Environment)
      numpy
      argparse
      collections
      copy
      math
      os
      re
      shutil
      subprocess
      sys
      time
      yaml

Atoms File
----------

The atoms file contains a list of all of the atoms to be used in the
simulation. On each line, we have the atomic symbol for a particular
element, the number of atoms of that element, and the ionic charge
(oxidation state) of these atoms. For example, the atom file for a
single formula unit (i.e., five atoms) of Strontium Titanate
(SrTiO<span>~3~</span>) reads:

      O  3 -2
      Sr 1 +2
      Ti 1 +4

An atoms file is required even when the initial structure is to be read
from a CIF file. In that case, the order of atoms listed must match the
order they are listed in the CIF, and vacancies can be specified using
the chemical symbol “X”, i.e,

      X 5 0

Input file
----------

The input file lists the values of all of the options for a ChemDASH
calculation in the format:

      <option>=<value>

where a “\#” is a comment character. A minimal working example of an
input file is given below:

      # General inputs
      #
      grid_type=orthorhombic
      temp=0.025
      grid_points=2,2,3
      cell_spacing=1.0
      atom_rankings=random
      vacancy_separation=1.0
      vacancy_exclusion_radius=2.0
      max_structures=10
      #
      # GULP inputs
      #
      calculator=gulp
      gulp_executable=gulp
      calculator_time_limit=300
      num_calc_stages=2
      gulp_files=conj, bfgs
      gulp_library=ff.lib
      #
      # GULP Keywords and Options
      #
      gulp_keywords=opti, c6, pot, conp
      gulp_calc_1_keywords=conj
      gulp_calc_2_keywords=lbfgs
      #
      gulp_options=time 5 minutes
      gulp_calc_1_options=stepmx 0.1
      gulp_calc_2_options=stepmx 0.5, lbfgs_order 5000, maxcyc 1000
      #
      #

The “temp” option states the value of kT in eV for the Monte–Carlo
temperature that determines whether or not we hop to higher–energy
basins during the run. The total number of structures explored in the
run is given by “max\_structures”. The other options listed in this
example are explained in the following sections, and a full list of
input options, with default and supported values, is given in
the "Full List of Input Options" section.

Test Suite
----------

ChemDASH has a test suite written in pytest, contained in the directory
“tests”. If pytest is installed, the test suite can be run by typing:

     py.test tests

If any tests fail, please contact the developers.

Initialisation
==============

Initialisation Grids
--------------------

There are three possible initialisation grids in ChemDASH:
“orthorhombic”, “rocksalt”, and “close\_packed”. These are specified in
the “grid\_type” input option. There are two more input options that
need to be considered. Firstly “grid\_points” is used to specify the
number of grid points on the **ANION** sublattice. This can be input as
a single number for an $a\times a \times a$ grid, two comma–separated
numbers to give an $a\times a \times c$ grid, or three comma–separated
numbers to give an $a\times b \times c$ grid. Secondly, “cell\_spacing”
is used to specify the distance between anion points on the
initialisation grid, this is specified in the same format as the anion
grid points.

Initialise from CIF
-------------------

When initialising from a CIF file, the file should be specified in the
input file with the option “initial\_structure\_file”. A “.atoms” file
is still required, with the atoms listed in the same order in both the
“.atoms” file and the CIF file. In addition to setting “grid\_points”
and “cell\_spacing”, for close–packed initialisation grids we can set
the stacking sequence with “cp\_stacking\_sequence” using a string
consisting of “A”, “B”, and “C” provided the number of layers is equal
to the final value in “grid\_points”. We can also choose from an
“oblique” or “centred\_rectangular” lattice using “cp\_2d\_lattice”.

Vacancies
---------

ChemDASH gives the option of using a vacancy grid by setting the option
“vacancy\_grid” to True. A vacancy grid is a cubic grid of points placed
onto the structure, with points that lie within a certain distance of an
atom removed. The spacing of the vacancies is set with
“vacancy\_separation”, and the exclusion radius around each atom within
which the points on the vacancy grid are removed is set using
“vacancy\_exclusion\_radius”. If a vacancy grid is not used, then the
leftover points from the initialisation grid are used as vacancies.

Optimisation
============

Structural optimisation in ChemDASH is handled by either GULP or VASP.
The desired software is set by the input option “calculator”, with
“calculator\_cores” used to set how many cores are desired for parallel
calculations. The option “update\_atoms” (default=True) is used to
decide whether to swap atoms in optimised geometries (if True), or
revert to the original, unoptimised geometry for the swap.

In ChemDASH, it is possible to run structural optimisations in a number
of stages, with a different set of optimisation settings for each stage.
For example, different stages of the calculation can be used to switch
between conjugate gradient and BFGS algorithms, or to switch to higher
precision parameters as the calculation progresses. The number of stages
in the calculation is set with “num\_calc\_stages”, and ChemDASH
provides the options to set GULP/VASP options for each stage of the
calculation (see below).

GULP
----

The filepath of the GULP executable should be given as
“gulp\_executable” in the input file. The keywords to be applied to
**ALL** stages of the gulp calculation are listed in the ChemDASH input
file as “gulp\_keywords”, whilst keywords to apply to a particular stage
of the calculation are given as
“gulp\_calc\_<span>$\left\langle \text{number} \right\rangle$</span>\_keywords”
(e.g., “gulp\_calc\_1\_keywords”). Similarly, for GULP options we use
“gulp\_options” for all stages and
“gulp\_calc\_<span>$\left\langle \text{number} \right\rangle$</span>\_options”
for a particular stage in the ChemDASH input file. Both keywords and
options are given as comma–separated lists. When optimising using GULP,
it is possible to terminate the calculation if the gnorm is above a
certain value after a particular stage by giving a value for
“gulp\_calc\_<span>$\left\langle \text{number} \right\rangle$</span>\_max\_gnorm”.

For each GULP calculation, the GULP output files are saved as
“structure\_<span>$\left\langle \text{number} \right\rangle$</span>\_<span>$\left\langle \text{stage} \right\rangle$</span>.<span>$\left\langle \text{gin|got|res} \right\rangle$</span>”.
The strings for each stage are given as a comma–separated list in the
ChemDASH input option “gulp\_files”. GULP uses force fields to optimise
structures, the file containing the forcefield for the calculation is
found from the option “gulp\_library”. If any elements in this
forcefield use a shell, these elements need to be listed in the
“gulp\_shells” input option. GULP optimisation are at risk of running
for an extremely long time., even with the gulp option “timeout”
enabled. Therefore, there is a ChemDASH input option
“calculator\_time\_limit” that can be used to terminate GULP
calculations after the given number of seconds.

VASP
----

The filepath of the VASP executable should be given as
“vasp\_executable” in the input file. The settings to be applied to
**ALL** stages of the VASP calculation are listed in the ChemDASH input
file as “vasp\_settings”, whilst settings to apply to a particular stage
of the calculation are given as
“vasp\_calc\_<span>$\left\langle \text{number} \right\rangle$</span>\_settings”
(e.g., “vasp\_calc\_1\_settings”). The settings required for this input
into ChemDASH are the contents of a VASP INCAR file. The format for VASP
settings is that of a python dictionary, which consists of a
comma–separated list of
“<span>$\left\langle \text{key} \right\rangle$</span>:<span>$\left\langle \text{value} \right\rangle$</span>”
pairs. For example,

     vasp_settings=xc:PBE, prec:Normal, encut:600

The VASP k–points are provided to ChemDASH using the option
“vasp\_kpoints”, where one, two or three numbers can be provided to
define a $k_{1}\times k_{2}\times k_{3}$ grid. For the pseudopotential,
the option “vasp\_pp\_dir” requires the filepath of the POTCAR
directory, and any elements that do not use the standard POTCAR file
should be listed with their extension (i.e., characters after the
chemical symbol), for example:

     vasp_settings=Li:_sv, Mg:_pv

Vasp optimisations are run until they successfully converge in a single
self-consistent field loop, or they hit the limit provided by the
“vasp\_max\_convergence\_calcs” option.

Swapping Atoms
==============

The method of ranking atoms for directed swapping is controlled by the
“atom\_rankings” input option. For random swapping this should be set as
“random”, otherwise set it to “bvs”, “site\_pot” or “bvs+” for thye
respective methods of directed swapping. Note that the “site\_pot” and
“bvs+” directed swapping is only supported for GULP, i.e.,
“calculator=gulp”.

When swapping atoms in ChemDASH, the first choice made is the *swap
group*, which is the set of atoms available for swapping. The possible
groups are:

-   cations – non–trivially swap a set of cations,

-   anions – non–trivially swap a set of anions,

-   atoms – non–trivially swap any atoms, but not vacancies,

-   all – non–trivially swap any atoms and vacancies,

-   atoms–vacancies – choose a set of atoms and swap each one with
    a vacancy.

where the first four groups are the default set of swap groups in
ChemDASH. Note that the “all" group differs from the “atoms–vacancies”
group in that the “all" group consists of atom–atom swaps and/or
atom–vacancy swaps, whereas the “atoms–vacancies” group is restricted to
atom–vacancy swaps. In addition, custom swap groups can be specified
that enable swaps to be restricted to atoms of particular species, for
example, “Sr–O” would restrict swaps to Sr and O atoms, with vacancies
are denoted as “X”. Custom swap groups can be constructed from any
combination of elements, provided there are at least two elements
present in the swap group and all of the elements are present in the
structure. The choice of swap groups can be weighted by specifying the
weight for each group in dictionary format. If weights are used, then a
weight **must** be specified for each swap group. If no weights are
specified, then all swap groups are equally likely to be chosen. An
example of the “swap\_groups” option is:

     swap_groups=cations:1, atoms:1, all:1, Sr-X:2

In this example, ChemDASH can choose between the cations, atoms, all,
and Sr-X groups for each swap, with the Sr-X group being twice as likely
to be chosen as the others.

Full List of Input Options
==========================

| ChemDASH Input file option        | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                      |                        |
|-----------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------|
| atom_rankings                     |The metric used to rank atoms for swapping. Supported values are: "random" (default), "bvs", "bvs+", "site_pot". Note that site potential and bvs+ directed swapping are only supported for gulp.                                                                                                                                                                                                                                                                |                        |
| atoms_file                        | File in which the species, number and oxidation state of the atoms used in this calculation are specified.                                                                                                                                                                                                                                                                                                                                                       |                        |
| bvs_file                          | Raw Bond Valence Sum file for this calculation. Records the bond valence sum for the atoms in each structure.                                                                                                                                                                                                                                                                                                                                                    |                        |
| calculator                        | The materials modelling code used for calculations. Default: gulp                                                                                                                                                                                                                                                                                                                                                                                                |                        |
| calculator_cores                  | The number of parallel cores used for the calculator. Default: 1.                                                                                                                                                                                                                                                                                                                                                                                                |                        |
| calculator_time_limit             | Used in the bash "timeout" command, GULP calculations will automatically terminate after this amount of time has expired.                                                                                                                                                                                                                                                                                                                                        |                        |
| cell_spacing                      | The spacing between two ANION grid points. Default: 2.0 A0                                                                                                                                                                                                                                                                                                                                                                                                       |                        |
| converge_first_structure          | If True, abort the run if the initial structure is not converged. Default: True                                                                                                                                                                                                                                                                                                                                                                                  |                        |
| cp_2d_lattice                     | Lattice type for anion layers in close packed grids. Supported values are: "oblique" (default) and "centred_rectangular"                                                                                                                                                                                                                                                                                                                                         |                        |
| cp_stacking_sequence              | Anion layer stacking sequence for close packed grids.                                                                                                                                                                                                                                                                                                                                                                                                            |                        |
| directed_num_atoms                | For directed swapping, the number of extra atoms available to choose between from the top of the list for each species. Default: 0                                                                                                                                                                                                                                                                                                                               |                        |
| directed_num_atoms_increment      | For directed swapping, the amount by which to increase (decrease) the number of extra values available to choose between from the top of the list for each species when a structure is (not) repeated. Default: 0                                                                                                                                                                                                                                                |                        |
| energy_file                       | Energy file for this calculation. Records the structure number, energies and volumes of accepted structures.                                                                                                                                                                                                                                                                                                                                                     |                        |
| energy_step_file                  | Energy step file for this calculation. Records the structure number, energies and volumes of accepted structures for plotting.                                                                                                                                                                                                                                                                                                                                   |                        |
| force_vacancy_swaps               | If True, vacancies cannot swap with each other, they must be replaced by atoms. Default: True.                                                                                                                                                                                                                                                                                                                                                                   |                        |
| grid_points                       | The number of points on each dimension of the ANION grid, to form an a x b x c grid for anions (cation points defined by grid type). Default: 2x2x2                                                                                                                                                                                                                                                                                                              |                        |
| grid_type                         | Initial layout of cation and anion grids. Supported values are "orthorhombic" (default), rocksalt", close_packed". Default: "orthorhombic".                                                                                                                                                                                                                                                                                                                      |                        |
| gulp_calc_1_keywords              | Comma-separated list of keywords for first GULP calculation. Default: None                                                                                                                                                                                                                                                                                                                                                                                       |                        |
| gulp_calc_1_max_gnorm             | If specified, terminate a GULP calculation if the final gnorm exceeds this value after the first stage.                                                                                                                                                                                                                                                                                                                                                          |                        |
| gulp_calc_1_options               | Options for first GULP calculation. Default: None                                                                                                                                                                                                                                                                                                                                                                                                                |                        |
| gulp_calc_2_keywords              | Comma-separated list of keywords for second GULP calculation. Default: None                                                                                                                                                                                                                                                                                                                                                                                      |                        |
| gulp_calc_2_max_gnorm             | If specified, terminate a GULP calculation if the final gnorm exceeds this value after the second stage.                                                                                                                                                                                                                                                                                                                                                         |                        |
| gulp_calc_2_options               | Options for second GULP calculation. Default: None                                                                                                                                                                                                                                                                                                                                                                                                               |                        |
| gulp_calc_3_keywords              | Comma-separated list of keywords for third GULP calculation. Default: None                                                                                                                                                                                                                                                                                                                                                                                       |                        |
| gulp_calc_3_max_gnorm             | If specified, terminate a GULP calculation if the final gnorm exceeds this value after the third stage.                                                                                                                                                                                                                                                                                                                                                          |                        |
| gulp_calc_3_options               | Options for third GULP calculation. Default: None                                                                                                                                                                                                                                                                                                                                                                                                                |                        |
| gulp_calc_4_keywords              | Comma-separated list of keywords for fourth GULP calculation. Default: None                                                                                                                                                                                                                                                                                                                                                                                      |                        |
| gulp_calc_4_max_gnorm             | If specified, terminate a GULP calculation if the final gnorm exceeds this value after the fourth stage.                                                                                                                                                                                                                                                                                                                                                         |                        |
| gulp_calc_4_options               | Options for fourth GULP calculation. Default: None                                                                                                                                                                                                                                                                                                                                                                                                               |                        |
| gulp_calc_5_keywords              | Comma-separated list of keywords for fifth GULP calculation. Default: None                                                                                                                                                                                                                                                                                                                                                                                       |                        |
| gulp_calc_5_max_gnorm             | If specified, terminate a GULP calculation if the final gnorm exceeds this value after the fifth stage.                                                                                                                                                                                                                                                                                                                                                          |                        |
| gulp_calc_5_options               | Options for fifth GULP calculation. Default: None                                                                                                                                                                                                                                                                                                                                                                                                                |                        |
| gulp_executable                   | The filepath of the GULP executable to be used. Default: "./gulp".                                                                                                                                                                                                                                                                                                                                                                                               |                        |
| gulp_files                        | Strings appended to each of the GULP files used to distinguish each calculation.                                                                                                                                                                                                                                                                                                                                                                                 |                        |
| gulp_keywords                     | Comma-separated list of keywords for all GULP calculations. Default: "opti, pot"                                                                                                                                                                                                                                                                                                                                                                                 |                        |
| gulp_library                      | Library file containing the forcefield to be used in GULP calculations. NOTE -- this takes precedence over a library specified in "gulp_options".                                                                                                                                                                                                                                                                                                                |                        |
| gulp_options                      | Options for all GULP calculations. Default: None                                                                                                                                                                                                                                                                                                                                                                                                                 |                        |
| gulp_shells                       | List of atoms to have a shell attached.                                                                                                                                                                                                                                                                                                                                                                                                                          |                        |
| initial_structure_file            | If specified, read in the initial structure from this cif file.                                                                                                                                                                                                                                                                                                                                                                                                  |                        |
| max_structures                    | This run of the code will terminate after this number of structures have been considered in this and all previous runs.                                                                                                                                                                                                                                                                                                                                          |                        |
| neighbourhood_atom_distance_limit | The minimum distance allowed between atoms in the local combinatorial neighbourhood method. Default: 1.0                                                                                                                                                                                                                                                                                                                                                         |                        |
| num_calc_stages                   | Number of GULP/VASP calculations to be run for each structure. Default: 1.                                                                                                                                                                                                                                                                                                                                                                                       |                        |
| num_neighbourhood_points          | The number of points used along each axis in the local combinatorial neighbourhood method. Default: 1                                                                                                                                                                                                                                                                                                                                                            |                        |
| num_structures                    | The number of structures we will consider in this run of the code.                                                                                                                                                                                                                                                                                                                                                                                               |                        |
| number_weightings                 | The method used to construct the weightings used to choose the number of atoms to swap. Supported values are "arithmetic" (default), "geometric", "uniform", and "pinned_pair".                                                                                                                                                                                                                                                                                  |                        |
| output_file                       | Output file for this calculation. Records the swaps for each structure, energies and acceptances.                                                                                                                                                                                                                                                                                                                                                                |                        |
| output_trajectory                 | If true, write ASE trajectory files. Default: True                                                                                                                                                                                                                                                                                                                                                                                                               |                        |
| pair_weighting                    | The initial proportional probability of swapping 2 atoms compared to any other number when using the "pinned_pair" option for "number_weightings". Default: 1.0                                                                                                                                                                                                                                                                                                  |                        |
| pair_weighting_scale_factor       | The factor by which we increase the proportional probability of swapping 2 atoms compared to any other number when we explore new basins (we decrease by the inverse factor for repeated basins) when using the "pinned_pair" option for "number_weightings". Default: 1.0                                                                                                                                                                                       |                        |
| potential_derivs_file             | Potential derivs file for this calculation. Records the resolved derivatives of the site potentials for each structure.                                                                                                                                                                                                                                                                                                                                          |                        |
| potentials_file                   | Potentials file for this calculation. Records the site potentials for each structure.                                                                                                                                                                                                                                                                                                                                                                            |                        |
| random_seed                       | The value used to seed the random number generator. Alternatively, the code can generate one itself, which is the default behaviour.                                                                                                                                                                                                                                                                                                                             |                        |
| restart                           | If True, use data in a numpy archive (specified by restart_file keyword) to continue a previous run. Default: False                                                                                                                                                                                                                                                                                                                                              |                        |
| restart_file                      | Name of the numpy archive from which to read data in order to continue a previous run. Default: "restart.npz" |
| rng_warm_up                       | Number of values from the RNG to generate and discard after seeding the generator. Default: 0.                                                                                                                                                                                                                                                                                                                                                                   |                        |
| save_outcar                       | If True, retain the final OUTCAR file from each structure optimised with VASP as "OUTCAR_[structure_index]". Default: False.                                                                                                                                                                                                                                                                                                                                     |                        |
| search_local_neighbourhood        | If True, uses the local combinatorial neighbourhood method to try and lower the energy of structures prior to relaxation. Default: False                                                                                                                                                                                                                                                                                                                         |                        |
| seed_bits                         | The number of bits used in the seed of the random number generator, The allowed values are 32 and 64. Default: 64                                                                                                                                                                                                                                                                                                                                                |                        |
| swap_groups                       | The groups of atoms that can be involved in swaps. The default groups are: "cations", "anions", "atoms", and "all" (atoms and vacancies). The input can include these, the additional swap group "atoms-vacancies" (always swap atoms with vacancies), or any custom swap group in the format [Chemical Symbol]-[Chemical Symbol]-[Chemical Symbol]. . . e.g., "Sr-X". A weighting can also be specified for each group as follows: "cations:1, atoms:2, all:3". |                        |
| temp                              | The Monte-Carlo temperature (strictly, the value of kT in eV). Determines whether swaps to basins of higher energy are accepted. Default: 0.0                                                                                                                                                                                                                                                                                                                    |                        |
| temp_scale_factor                 | The factor by which we increase the temperature after rejected structures (we decrease by the inverse factor for accepted structures). Default: 1.0                                                                                                                                                                                                                                                                                                              |                        |
| update_atoms                      | If true, swap atoms based on relaxed structures, rather than initial structures. Default: True.                                                                                                                                                                                                                                                                                                                                                                  |                        |
| vacancy_exclusion_radius          | The minimum allowable distance between an atom and a vacancy on the vacancy grid. Default: 2.0 A0                                                                                                                                                                                                                                                                                                                                                                |                        |
| vacancy_grid                      | If true, apply vacancy grids to each structure in which we will swap atoms. Default: True.                                                                                                                                                                                                                                                                                                                                                                       |                        |
| vacancy_separation                | The nearest neighbour distance between two vacancies on the vacancy grid. Default: 1.0 A0                                                                                                                                                                                                                                                                                                                                                                        |                        |
| vasp_calc_1_settings              | Settings for the first stage of the VASP calculation. Default: None.                                                                                                                                                                                                                                                                                                                                                                                             |                        |
| vasp_calc_2_settings              | Settings for the second stage of the VASP calculation. Default: None.                                                                                                                                                                                                                                                                                                                                                                                            |                        |
| vasp_calc_3_settings              | Settings for the third stage of the VASP calculation. Default: None.                                                                                                                                                                                                                                                                                                                                                                                             |                        |
| vasp_calc_4_settings              | Settings for the fourth stage of the VASP calculation. Default: None.                                                                                                                                                                                                                                                                                                                                                                                            |                        |
| vasp_calc_5_settings              | Settings for the fifth stage of the VASP calculation. Default: None.                                                                                                                                                                                                                                                                                                                                                                                             |                        |
| vasp_executable                   | The filepath of the vasp executable to be used. Default: "./vasp"                                                                                                                                                                                                                                                                                                                                                                                                |                        |
| vasp_kpoints                      | Number of k points to use in VASP calculations. Default: 1                                                                                                                                                                                                                                                                                                                                                                                                       |                        |
| vasp_max_convergence_calcs        | Maximum number of VASP calculations performed in the final stage for convergence -- we abandon the calculation after this. Default: 10.                                                                                                                                                                                                                                                                                                                          |                        |
| vasp_pp_dir                       | Path to directory containing VASP pseudopotential files. Default: ".".                                                                                                                                                                                                                                                                                                                                                                                           |                        |
| vasp_pp_setups                    | Pseudopotential file extensions for each element.                                                                                                                                                                                                                                                                                                                                                                                                                |                        |
| vasp_settings                     | Settings for all VASP calculations. Default: None.                                                                                                                                                                                                                                                                                                                                                                                                               |                        |
| verbosity                         | Controls the level of detail in the output. Valid options are: "verbose", "terse". Default: "verbose"                                                                                                                                                                                                                                                                                                                                                            |                        |

Acknowledgements
================

We acknowledge funding from the EPSRC Programme Grant: EP/N004884/1 "Integration of Computation and Experiment for Accelerated Materials Discovery"
