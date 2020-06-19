import pytest
import mock

import chemdash.gulp_calc

import ase
from ase.calculators.gulp import GULP
import numpy as np
import os
import subprocess

#===========================================================================================================================================================
#===========================================================================================================================================================
#Track tests

#UNIT TESTS

#test_run_gulp

#test_update_atoms_object
#test_read_energy
#test_check_float


#===========================================================================================================================================================
#===========================================================================================================================================================
#Tests

@pytest.mark.parametrize("gulp_keywords, gulp_options, gulp_shells, gulp_library, gulp_conditions, expected_output", [
    ('opti lbfgs conp c6', ["library STO.lib", "time 10 minutes"], [], "STO.lib", None, 
     (ase.Atoms(symbols = "SrTiO3", cell = [2.0, 2.0, 2.0], charges = [2.0, 4.0, -2.0, -2.0, -2.0],
                scaled_positions = ([0.75, 0.75, 0.25], [0.75, 0.25, 0.25], [0.5, 0.5, 0.5], [0.5, 0.0, 0.0], [0.0, 0.0, 0.5]),
                pbc=[True, True, True]),
      -31.71670925, "", ["gulp.gin", "gulp.got", "gulp.res"])),
])

#Need to find a way to determine the calculator matches what is expected
@pytest.mark.xfail
def test_run_gulp(STO_atoms, gulp_keywords, gulp_options, gulp_shells, gulp_library, gulp_conditions, expected_output, monkeypatch):
    """
    GIVEN a structure and a set of gulp options

    WHEN we run a gulp calculation

    THEN we return the updated structure with it's energy and information about the result of the calculation.

    Parameters
    ----------
    STO_atoms : ASE atoms
        An ASE atoms object containing SrTiO_{3} and five vacancies.
    gulp_keywords : list, optional
        List of GULP keywords. Default is "sing" for a single point energy calculation.
    gulp_options : list, optional
        List of GULP options. Default is None.
    gulp_shells : list, optional
        List of each atomic species to have a shell attached.
    gulp_library : string, optional
        The library file containing the forcefield to be used in the calculation.
    gulp_setups : list, optional
        A list of symbols for each atom if they are not wanted to be the same as in the original atoms object

    ---------------------------------------------------------------------------
    Paul Sharp 16/01/2018
    """

    monkeypatch.setattr(STO_atoms, 'get_potential_energy', lambda : -158.58354625)
    monkeypatch.setattr(os.path, 'isfile', lambda x : True)

    monkeypatch.setattr(chemdash.gulp_calc, 'read_outcome', lambda x : "Optimisation Achieved")
    monkeypatch.setattr(chemdash.gulp_calc, 'check_timed_out', lambda x : False)

    output = chemdash.gulp_calc.run_gulp(STO_atoms, "gulp", "", gulp_keywords, gulp_options, gulp_shells, gulp_library)
 
    assert output == expected_output
    assert STO_atoms.get_calculator() == Gulp(filename = "gulp", keywords = gulp_keywords, options = gulp_options,
                                              shells = gulp_shells, library = gulp_library, conditions = gulp_conditions)


#===========================================================================================================================================================
@pytest.mark.parametrize("gulp_keywords, gulp_options, gulp_shells, gulp_library, gulp_conditions, expected_output", [
    ('opti lbfgs conp c6', ["library STO.lib", "time 10 minutes"], ["Sr", "Ti", "O"], "STO.lib", None, 
     (ase.Atoms(symbols = "SrTiO3", cell = [2.0, 2.0, 2.0], charges = [2.0, 4.0, -2.0, -2.0, -2.0],
                scaled_positions = ([0.75, 0.75, 0.25], [0.75, 0.25, 0.25], [0.5, 0.5, 0.5], [0.5, 0.0, 0.0], [0.0, 0.0, 0.5]),
                pbc=[True, True, True]),
      -31.71670925, "", ["gulp.gin", "gulp.got", "gulp.res"])),
])

def test_run_gulp_exception(STO_atoms, gulp_keywords, gulp_options, gulp_shells, gulp_library, gulp_conditions, expected_output, monkeypatch):
    """
    GIVEN a structure and a set of gulp options, including shells

    WHEN we run a gulp calculation, but do not have a restart file

    THEN we raise an exception.

    Parameters
    ----------
    STO_atoms : ASE atoms
        An ASE atoms object containing SrTiO_{3} and five vacancies.
    gulp_keywords : list, optional
        List of GULP keywords.
    gulp_options : list, optional
        List of GULP options.
    gulp_shells : list, optional
        List of each atomic species to have a shell attached.
    gulp_library : string, optional
        The library file containing the forcefield to be used in the calculation.
    gulp_setups : list, optional
        A list of symbols for each atom if they are not wanted to be the same as in the original atoms object

    ---------------------------------------------------------------------------
    Paul Sharp 18/01/2018
    """

    # Make sure we follow path where the restart file does not exist
    monkeypatch.setattr(os.path, 'isfile', lambda x : False)

    STO_atoms.set_calculator(GULP())

    with pytest.raises(RuntimeError):
        chemdash.gulp_calc.run_gulp(STO_atoms, "gulp", "", gulp_keywords, gulp_options, gulp_shells, gulp_library, gulp_conditions)


#===========================================================================================================================================================
@pytest.mark.parametrize("test_gulp_output, expected_output", [
    ('''Total lattice energy = -1.0 eV
    Total lattice energy = -96.485 kJ/(mole unit cells)
    Total lattice energy = -2.0 eV
    Total lattice energy = -192.969 kJ/(mole unit cells)''',
     -2.0),
])

def test_read_energy(test_gulp_output, expected_output, monkeypatch):
    """
    GIVEN a gulp output file

    WHEN we read the energy

    THEN we get the energy

    Parameters
    ----------
    test_gulp_output : str
        A string used as a mock of the gulp output file

    ---------------------------------------------------------------------------
    Paul Sharp 08/11/2017
    """

    # We need to ensure we use "mock_open" to return our test data instead of reading a file
    # We also need to ensure that the return value of this Mock produces an iterable,
    # so we can loop over the lines of the file
    iterable_mock_file = mock.mock_open(read_data = test_gulp_output)
    iterable_mock_file.return_value.__iter__ = lambda x: iter(x.readline, '')

    # Note the lack of lambda -- we want the mock itself, not its return value
    # We pass a blank string for the file argument for this reason
    monkeypatch.setattr('builtins.open', iterable_mock_file)

    assert chemdash.gulp_calc.read_energy("") == expected_output


#===========================================================================================================================================================
@pytest.mark.parametrize("energy, test_gulp_restart, expected_output", [
    (-158.5,
     '''
# 
# Keywords:
# 
conp opti c6  
# 
# Options:
# 
cell
2.000000   2.000000   2.000000  90.000000  90.000000  90.000000
fractional  1   
Sr    core 0.4325591 0.4215044 0.3911638 2.00000000 1.00000 0.00000             
Ti    core 0.9325605 0.9215058 0.8911618 4.00000000 1.00000 0.00000             
O     core 0.9325659 0.4215040 0.8911569 -2.0000000 1.00000 0.00000             
O     core 0.4325622 0.9215053 0.8911646 -2.0000000 1.00000 0.00000             
O     core 0.9325597 0.9215046 0.3911620 -2.0000000 1.00000 0.00000             
totalenergy          -158.5835462160 eV
species   3
Sr     core    2.000000                  
Ti     core    4.000000                  
O      core   -2.000000                  
buck     
O     core Sr    core  1952.39000     0.336850  19.22000      0.00 12.00
buck     
O     core Ti    core  4590.72790     0.261000  0.000000      0.00 12.00
buck     
O     core O     core  1388.77000     0.362620  175.0000      0.00 12.00
time        600.0
stepmx opt     0.500000
maxcyc opt    10000
maxcyc fit    10000
dump structure_937.res
     ''',
     (ase.Atoms(symbols = "SrTiO3", charges = [2.0, 4.0, -2.0, -2.0, -2.0],
                cell = [2.0, 2.0, 2.0],
                scaled_positions = ([0.4325591, 0.4215044, 0.3911638],
                                    [0.9325605, 0.9215058, 0.8911618],
                                    [0.9325659, 0.4215040, 0.8911569],
                                    [0.4325622, 0.9215053, 0.8911646],
                                    [0.9325597, 0.9215046, 0.3911620]),
                pbc=[True, True, True]),
      -31.7167092432)),
    ("******",
     '''
# 
# Keywords:
# 
conp opti c6  
# 
# Options:
# 
cell
2.000000   2.000000   2.000000  90.000000  90.000000  90.000000
fractional  1   
Sr    core 3/4       3/4       1/4       2.00000000 1.00000 0.00000             
Ti    core 3/4       1/4       1/4       4.00000000 1.00000 0.00000             
O     core 1/2       1/2       1/2       -2.0000000 1.00000 0.00000             
O     core 1/2       0.0000000 0.0000000 -2.0000000 1.00000 0.00000             
O     core 0.0000000 0.0000000 1/2       -2.0000000 1.00000 0.00000             
species   3
Sr     core    2.000000                  
Ti     core    4.000000                  
O      core   -2.000000                  
buck     
O     core Sr    core  1952.39000     0.336850  19.22000      0.00 12.00
buck     
O     core Ti    core  4590.72790     0.261000  0.000000      0.00 12.00
buck     
O     core O     core  1388.77000     0.362620  175.0000      0.00 12.00
time        600.0
stepmx opt     0.500000
maxcyc opt    10000
maxcyc fit    10000
dump structure_0.res
     ''',
     (ase.Atoms(symbols = "SrTiO3", charges = [2.0, 4.0, -2.0, -2.0, -2.0],
                cell = [2.0, 2.0, 2.0],
                scaled_positions = ([0.75, 0.75, 0.25], [0.75, 0.25, 0.25],
                                    [0.5, 0.5, 0.5], [0.5, 0.0, 0.0],
                                    [0.0, 0.0, 0.5]),
                pbc=[True, True, True]),
      "******")),
])

def test_update_atoms_from_restart_file(STO_atoms, energy, test_gulp_restart, expected_output, monkeypatch):
    """
    GIVEN an atoms object, energy and a gulp restart file

    WHEN we read the atomic positions and energy

    THEN we get the atoms object with refined atomic positions and energy

    Parameters
    ----------
    test_gulp_output : str
        A string used as a mock of the gulp output file

    ---------------------------------------------------------------------------
    Paul Sharp 16/01/2018
    """

    # We need to ensure we use "mock_open" to return our test data instead of reading a file
    # We also need to ensure that the return value of this Mock produces an iterable,
    # so we can loop over the lines of the file
    iterable_mock_file = mock.mock_open(read_data = test_gulp_restart)
    iterable_mock_file.return_value.__iter__ = lambda x : iter(x.readline, '')

    # Note the lack of lambda -- we want the mock itself, not its return value
    # We pass a blank string for the file argument for this reason
    monkeypatch.setattr('builtins.open', iterable_mock_file)

    monkeypatch.setattr(os.path, 'isfile', lambda x : True)

    assert chemdash.gulp_calc.update_atoms_from_restart_file(STO_atoms, energy, "") == expected_output


#===========================================================================================================================================================
@pytest.mark.parametrize("test_res_file, unit_cell, gulp_keywords, gulp_options, gulp_library, expected_output", [
    (['#',
      '# Keywords:',
      '#',
      'conp opti c6',
      '#',
      '# Options:',
      '#',
      'cell',
      '2.000000   2.000000   2.000000  90.000000  90.000000  90.000000',
      'fractional  1',
      'Sr    core 0.4325591 0.4215044 0.3911638 2.00000000 1.00000 0.00000',
      'Ti    core 0.9325605 0.9215058 0.8911618 4.00000000 1.00000 0.00000',     
      'O     core 0.9325659 0.4215040 0.8911569 -2.0000000 1.00000 0.00000',     
      'O     core 0.4325622 0.9215053 0.8911646 -2.0000000 1.00000 0.00000',     
      'O     core 0.9325597 0.9215046 0.3911620 -2.0000000 1.00000 0.00000',     
      'totalenergy          -158.5835462160 eV',
      'species   3',
      'Sr     core    2.000000',
      'Ti     core    4.000000',          
      'O      core   -2.000000',          
      'buck',
      'O     core Sr    core  1952.39000     0.336850  19.22000      0.00 12.00',
      'buck',
      'O     core Ti    core  4590.72790     0.261000  0.000000      0.00 12.00',
      'buck',
      'O     core O     core  1388.77000     0.362620  175.0000      0.00 12.00',
      'time        600.0',
      'stepmx opt     0.500000',
      'maxcyc opt    10000',
      'maxcyc fit    10000',
      'dump structure_937.res'],
     [[2.000000, 0.000000, 0.000000],
      [0.000000, 2.000000, 0.000000],
      [0.000000, 0.000000, 2.000000]],
     'conp opti c6 lbfgs',
     ['stepmx 0.1', 'time 2 minutes', 'lbfgs_order 5000', 'maxcyc 10000'],
     'STO.lib',
     [mock.call("conp opti c6 lbfgs\n"),
      mock.call("\n"),
      mock.call("vectors\n"),
      mock.call("    2.0    0.0    0.0\n"),
      mock.call("    0.0    2.0    0.0\n"),
      mock.call("    0.0    0.0    2.0\n"),
      mock.call("fractional  1\n"),
      mock.call("Sr    core 0.4325591 0.4215044 0.3911638 2.00000000 1.00000 0.00000\n"),
      mock.call("Ti    core 0.9325605 0.9215058 0.8911618 4.00000000 1.00000 0.00000\n"),
      mock.call("O     core 0.9325659 0.4215040 0.8911569 -2.0000000 1.00000 0.00000\n"),
      mock.call("O     core 0.4325622 0.9215053 0.8911646 -2.0000000 1.00000 0.00000\n"),
      mock.call("O     core 0.9325597 0.9215046 0.3911620 -2.0000000 1.00000 0.00000\n"),
      mock.call("\n"),
      mock.call("library STO.lib\n"),
      mock.call("stepmx 0.1\n"),
      mock.call("time 2 minutes\n"),
      mock.call("lbfgs_order 5000\n"),
      mock.call("maxcyc 10000\n"),])
])

def test_create_input_file_from_restart_file(test_res_file, unit_cell, gulp_keywords, gulp_options, gulp_library, expected_output, monkeypatch):
    """
    GIVEN a gulp restart file and settings for a new calculation

    WHEN we write a new input file

    THEN we get a GULP input file for a new calculation

    Parameters
    ----------
    test_res_file : str
        The read data for the mock restart file.

    ---------------------------------------------------------------------------
    Paul Sharp 10/08/2018
    """

    monkeypatch.setattr(chemdash.gulp_calc, 'read_restart_file', lambda x : test_res_file)

    # We need to ensure we use "mock_open" to avoid writing to a file
    # Note the lack of lambda -- we want the mock itself, not its return value
    # We pass a blank string for the file argument for this reason
    mock_file = mock.mock_open()
    monkeypatch.setattr('builtins.open', mock_file)

    chemdash.gulp_calc.create_input_file_from_restart_file("", "", unit_cell, gulp_keywords, gulp_options, gulp_library)

    # We check the calls to the write function as we do not return what we write
    assert mock_file().write.call_args_list == expected_output


#===========================================================================================================================================================
@pytest.mark.parametrize("test_gulp_output, expected_output", [
    ('''
      Cycle:     51 Energy:      -156.016148  Gnorm:      0.006229  CPU:    0.097
    
      **** Optimisation achieved ****


      Final energy =    -156.01626275 eV
      Final Gnorm  =       0.00148753

      Components of energy :

    --------------------------------------------------------------------------------
    Interatomic potentials     =          27.89201680 eV
    Monopole - monopole (real) =         -59.75345130 eV
    Monopole - monopole (recip)=        -121.10105893 eV
    Monopole - monopole (total)=        -180.85451024 eV
    Dispersion (real+recip)    =          -3.05376931 eV
    --------------------------------------------------------------------------------
    Total lattice energy       =        -156.01626275 eV
    --------------------------------------------------------------------------------
    Total lattice energy       =          -15053.1844 kJ/(mole unit cells)
    --------------------------------------------------------------------------------
    ''', -156.01626275),
    ('''
      Cycle:    132 Energy:*****************  Gnorm:**************  CPU:    0.136


      **** Too many failed attempts to optimise ****

      Final energy = **************** eV
      Final Gnorm  = ****************

      Components of energy : 

    --------------------------------------------------------------------------------
    Interatomic potentials     =        1944.74640591 eV
    Monopole - monopole (real) =      -43566.54368149 eV
    Monopole - monopole (recip)=         -36.62339022 eV
    Monopole - monopole (total)=      -43603.16707171 eV
    Dispersion (real+recip)    = ******************** eV
    --------------------------------------------------------------------------------
    Total lattice energy       = ******************** eV
    --------------------------------------------------------------------------------
    Total lattice energy       = ******************** kJ/(mole unit cells)
    --------------------------------------------------------------------------------
    ''', "********************"),
])

def test_read_energy(test_gulp_output, expected_output, monkeypatch):
    """
    GIVEN a gulp output file

    WHEN we read in the final energy

    THEN we get a value of the final energy

    Parameters
    ----------
    test_gulp_output : str
        The read data for the mock output file.

    ---------------------------------------------------------------------------
    Paul Sharp 10/08/2018
    """

    # We need to ensure we use "mock_open" to return our test data instead of reading a file
    # We also need to ensure that the return value of this Mock produces an iterable,
    # so we can loop over the lines of the file
    iterable_mock_file = mock.mock_open(read_data = test_gulp_output)
    iterable_mock_file.return_value.__iter__ = lambda x: iter(x.readline, '')

    # Note the lack of lambda -- we want the mock itself, not its return value
    # We pass a blank string for the file argument for this reason
    monkeypatch.setattr('builtins.open', iterable_mock_file)

    assert chemdash.gulp_calc.read_energy("") == expected_output


#===========================================================================================================================================================
@pytest.mark.parametrize("test_gulp_output, expected_output", [
    ('''
      Cycle:     51 Energy:      -156.016148  Gnorm:      0.006229  CPU:    0.097
    
      **** Optimisation achieved ****


      Final energy =    -156.01626275 eV
      Final Gnorm  =       0.00148753

      Components of energy :

    --------------------------------------------------------------------------------
    Interatomic potentials     =          27.89201680 eV
    Monopole - monopole (real) =         -59.75345130 eV
    Monopole - monopole (recip)=        -121.10105893 eV
    Monopole - monopole (total)=        -180.85451024 eV
    Dispersion (real+recip)    =          -3.05376931 eV
    --------------------------------------------------------------------------------
    Total lattice energy       =        -156.01626275 eV
    --------------------------------------------------------------------------------
    Total lattice energy       =          -15053.1844 kJ/(mole unit cells)
    --------------------------------------------------------------------------------
    ''', 0.00148753),
    ('''
      Cycle:    132 Energy:*****************  Gnorm:**************  CPU:    0.136


      **** Too many failed attempts to optimise ****

      Final energy = **************** eV
      Final Gnorm  = ****************

      Components of energy : 

    --------------------------------------------------------------------------------
    Interatomic potentials     =        1944.74640591 eV
    Monopole - monopole (real) =      -43566.54368149 eV
    Monopole - monopole (recip)=         -36.62339022 eV
    Monopole - monopole (total)=      -43603.16707171 eV
    Dispersion (real+recip)    = ******************** eV
    --------------------------------------------------------------------------------
    Total lattice energy       = ******************** eV
    --------------------------------------------------------------------------------
    Total lattice energy       = ******************** kJ/(mole unit cells)
    --------------------------------------------------------------------------------
    ''', "****************"),
])

def test_read_gnorm(test_gulp_output, expected_output, monkeypatch):
    """
    GIVEN a gulp output file

    WHEN we read in the final gnorm

    THEN we get a value of the final gnorm

    Parameters
    ----------
    test_gulp_output : str
        The read data for the mock output file.

    ---------------------------------------------------------------------------
    Paul Sharp 10/08/2018
    """

    # We need to ensure we use "mock_open" to return our test data instead of reading a file
    # We also need to ensure that the return value of this Mock produces an iterable,
    # so we can loop over the lines of the file
    iterable_mock_file = mock.mock_open(read_data = test_gulp_output)
    iterable_mock_file.return_value.__iter__ = lambda x: iter(x.readline, '')

    # Note the lack of lambda -- we want the mock itself, not its return value
    # We pass a blank string for the file argument for this reason
    monkeypatch.setattr('builtins.open', iterable_mock_file)

    assert chemdash.gulp_calc.read_gnorm("") == expected_output


#===========================================================================================================================================================
@pytest.mark.parametrize("test_gulp_output, gulp_keywords, expected_output", [
    ('''**** Optimisation achieved ****''', 'opti', True),
    ('''**** Too many failed attempts to optimise ****''', 'opti', False),
    ('''**** Optimisation achieved ****''', '', False),
    ('''Components of energy''', '', True),
    ('''Components of energy''', 'opti', False),
    ('''**** Too many failed attempts to optimise ****''', 'opti', False),
])

def test_check_convergence(test_gulp_output, gulp_keywords, expected_output, monkeypatch):
    """
    GIVEN a gulp output file and set of keywords

    WHEN we read check if the calculation has converged

    THEN we get a logical set according to whether or not the calculation converged.

    Parameters
    ----------
    test_gulp_output : str
        A string used as a mock of the gulp output file.
    gulp_keywords : str
        The set of keywords used in the gulp calculation.

    ---------------------------------------------------------------------------
    Paul Sharp 16/01/2018
    """

    # We need to ensure we use "mock_open" to return our test data instead of reading a file
    # We also need to ensure that the return value of this Mock produces an iterable,
    # so we can loop over the lines of the file
    iterable_mock_file = mock.mock_open(read_data = test_gulp_output)
    iterable_mock_file.return_value.__iter__ = lambda x : iter(x.readline, '')

    # Note the lack of lambda -- we want the mock itself, not its return value
    # We pass a blank string for the file argument for this reason
    monkeypatch.setattr('builtins.open', iterable_mock_file)

    assert chemdash.gulp_calc.check_convergence("", gulp_keywords) == expected_output

#===========================================================================================================================================================
@pytest.mark.parametrize("value, expected_output", [
    (1.0, True),
    (0.0, True),
    (-1.0, True),
    ("1.0", True),
    ("0.0", True),
    ("-1.0", True),
    ("***", False),
])


def test_check_float(value, expected_output):
    """
    GIVEN an input string

    WHEN we check whether or not it can be represented as a float

    THEN we get a logical set depending on whether or not the string can be represented as a float.

    Parameters
    ----------
    value : string
        The string that we wish to check whether or not it can be represented as a report.

    ---------------------------------------------------------------------------
    Paul Sharp 08/09/2017
    """

    assert chemdash.gulp_calc.check_float(value) == expected_output


#===========================================================================================================================================================
@pytest.mark.parametrize("test_final_lines,  expected_output", [
    (b'''  Job Finished at 16:46.01  8th November   2017                               
    
    ''', False),
    (b'''  Program terminated at 16:46.01  8th November   2017                               
    
    ''', False),
    (b'''**** Too many failed attempts to optimise ****                  
    
    ''', True),
])

def test_check_timed_out(test_final_lines, expected_output, monkeypatch):
    """
    GIVEN a gulp output file 

    WHEN we check if the calculation timed out

    THEN we get a logical set according to whether or not the calculation timed out

    Parameters
    ----------
    test_final_lines : byte str
        A string used as a mock of the gulp output file.

    ---------------------------------------------------------------------------
    Paul Sharp 17/01/2020
    """

    monkeypatch.setattr(subprocess, 'check_output', lambda x : test_final_lines)

    assert chemdash.gulp_calc.check_timed_out("") == expected_output


#===========================================================================================================================================================
@pytest.mark.parametrize("test_gulp_output,  expected_output", [
    (''' **** Optimisation Achieved ****''', 'Optimisation Achieved'),
    (''' **** Too many failed attempts to optimise ****''', 'Too many failed attempts to optimise'),
    (''' **** Conditions for a minimum have not been satisfied. However ****
     **** no lower point can be found - treat results with caution  ****
     **** unless gradient norm is small (less than 0.1)             ****''',
     'Conditions for a minimum have not been satisfied. However no lower point can be found - treat results with caution  unless gradient norm is small (less than 0.1)'),
])

def test_read_outcome(test_gulp_output, expected_output, monkeypatch):
    """
    GIVEN a gulp output file 

    WHEN we read the outcome

    THEN we get the outcome of the gulp calculation

    Parameters
    ----------
    test_gulp_output : str
        A string used as a mock of the gulp output file.

    ---------------------------------------------------------------------------
    Paul Sharp 08/11/2017
    """

    # We need to ensure we use "mock_open" to return our test data instead of reading a file
    # We also need to ensure that the return value of this Mock produces an iterable,
    # so we can loop over the lines of the file
    iterable_mock_file = mock.mock_open(read_data = test_gulp_output)
    iterable_mock_file.return_value.__iter__ = lambda x : iter(x.readline, '')

    # Note the lack of lambda -- we want the mock itself, not its return value
    # We pass a blank string for the file argument for this reason
    monkeypatch.setattr('builtins.open', iterable_mock_file)

    assert chemdash.gulp_calc.read_outcome("") == expected_output


#===========================================================================================================================================================
@pytest.mark.parametrize("new_outcome, outcome_dict, expected_output", [
    ('Optimisation Achieved', {}, {'Optimisation Achieved': 1}),
    ('Timed out in GULP', {'Optimisation Achieved': 2, 'Too many failed attempts to optimise': 1},
     {'Optimisation Achieved': 2, 'Too many failed attempts to optimise': 1, 'Timed out in GULP': 1}),
    ('Optimisation Achieved', {'Optimisation Achieved': 2, 'Too many failed attempts to optimise': 1},
     {'Optimisation Achieved': 3, 'Too many failed attempts to optimise': 1}),
])

def test_update_outcomes(new_outcome, outcome_dict, expected_output):
    """
    GIVEN an outcomes dictionary

    WHEN we update the dictionary with the latest outcome

    THEN we get an updated dictionary

    Parameters
    ----------
    outcome_dict : dict
        Dictionary of each outcome with the number of times they have occurred in this run.

    ---------------------------------------------------------------------------
    Paul Sharp 08/11/2017
    """

    assert chemdash.gulp_calc.update_outcomes(new_outcome, outcome_dict) == expected_output


#===========================================================================================================================================================
@pytest.mark.parametrize("test_gulp_output,  expected_output", [
    ('''
    Total number atoms/shells =       5

    Electrostatic potential at atomic positions :

    -------------------------------------------------------------------------------
    Site  Atomic      Potential                Derivatives (V/Angs)
    No.   Label          (V)                 x           y           z
    -------------------------------------------------------------------------------
        1 O     c     23.504379           0.000538    0.000469    0.000788
        2 O     c     23.504364           0.001245    0.001332    0.000086
        3 O     c     23.504395           0.000425    0.000414    0.000345
        4 Sr    c    -19.613508           0.000765    0.000968    0.000518
        5 Ti    c    -45.063322           0.000721    0.000623    0.000351
    -------------------------------------------------------------------------------
    ''',
     ([23.504379, 23.504364, 23.504395, -19.613508, -45.063322],
      [0.001063178724392094, 0.001825279430662604, 0.00068632790996724, 0.0013381229390455871, 0.0010154659029233822])),
    ('''
    Total number atoms/shells =       6

    Electrostatic potential at atomic positions :

    -------------------------------------------------------------------------------
    Site  Atomic      Potential                Derivatives (V/Angs)
    No.   Label          (V)                 x           y           z
    -------------------------------------------------------------------------------
        1 Zn    c    -23.610229          -0.000906   -0.001806************
        2 Zn    c    -23.610229          -0.000899   -0.001803************
        3 O     c     23.610236           0.000003    0.000001    0.027683
        4 O     c     23.610236          -0.000003   -0.000001    0.027683
        5 Zn    s    -23.610236           0.000019    0.000043************
        6 Zn    s    -23.610236           0.000025    0.000045************
    -------------------------------------------------------------------------------
    ''',
     ([-23.610229, -23.610229, 23.610236, 23.610236],
      ['--', '--', 0.02768300018061626, 0.02768300018061626])),
    ('''
    Total number atoms/shells =       5

    Electrostatic potential at atomic positions :

    -------------------------------------------------------------------------------
    Site  Atomic      Potential                Derivatives (V/Angs)
    No.   Label          (V)                 x           y           z
    -------------------------------------------------------------------------------
        1 O     c     ************        ********************************
        2 O     c     ************        ********************************
        3 O     c     ************        ********************************
        4 Sr    c     ************        ********************************
        5 Ti    c     ************        ********************************
    -------------------------------------------------------------------------------
    ''',
     (['--', '--', '--', '--', '--'], ['--', '--', '--', '--', '--'])),
])

def test_read_potentials(test_gulp_output, expected_output, monkeypatch):
    """
    GIVEN a gulp output file 

    WHEN we read the potentials

    THEN we get list of the potentials and their derivatives for each atom

    Parameters
    ----------
    test_gulp_output : str
        A string used as a mock of the gulp output file.

    ---------------------------------------------------------------------------
    Paul Sharp 10/11/2017
    """

    # We need to ensure we use "mock_open" to return our test data instead of reading a file
    # We also need to ensure that the return value of this Mock produces an iterable,
    # so we can loop over the lines of the file, and a next to access later lines
    iterable_mock_file = mock.mock_open(read_data = test_gulp_output)
    iterable_mock_file.return_value.__iter__ = lambda x : iter(x.readline, '')

    # This will work with next() in python 3!
    #iterable_mock_file.return_value.__next__ = lambda x : next(iter(x.readline, ''))

    # Note the lack of lambda -- we want the mock itself, not its return value
    # We pass a blank string for the file argument for this reason
    monkeypatch.setattr('builtins.open', iterable_mock_file)

    assert chemdash.gulp_calc.read_potentials("") == expected_output

    
#===========================================================================================================================================================
@pytest.mark.parametrize("expected_output", [
    ([0.0, 0.0, 0.0], [0.25, 0.25, 0.25], [0.75, 0.75, 0.75], [0.0, 0.5, 0.0], [0.25, 0.25, 0.75]),
])

def test_determine_vacancy_positions(STOX_structure, expected_output):
    """
    GIVEN an atoms object 

    WHEN we look for the positions of vacancies

    THEN we return a list of vacancy positions

    Parameters
    ----------
    None

    ---------------------------------------------------------------------------
    Paul Sharp 27/06/2019
    """

    assert all([np.allclose(x, y) for x, y in zip(chemdash.gulp_calc.determine_vacancy_positions(STOX_structure), expected_output)])

    
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
    struct : ase atoms
        The structure to which we will add vacancies.
    vacancy_points : float
        The list of points to be recorded as vacancies.

    ---------------------------------------------------------------------------
    Paul Sharp 26/10/2017
    """

    assert chemdash.gulp_calc.populate_points_with_vacancies(STO_atoms, vacancy_points) == STOX_structure
