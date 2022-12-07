import pytest
import mock

import chemdash.master_code
import chemdash.vasp_calc

import ase
import numpy as np
import os
import subprocess
import shutil
import time


#===========================================================================================================================================================
#===========================================================================================================================================================
#Tests

@pytest.mark.parametrize("structure, num_calcs, main_settings, additional_settings, max_convergence_calcs, save_outcar, expected_output", [
     (chemdash.master_code.Structure(index = 0, volume = 7.999999999999998, labels = [], atoms=
                ase.Atoms(symbols = "SrTiO3", cell = [2.0, 2.0, 2.0],
                charges = [2.0, 4.0, -2.0, -2.0, -2.0],
                scaled_positions = ([0.75, 0.75, 0.25], [0.75, 0.25, 0.25], [0.5, 0.5, 0.5], [0.5, 0.0, 0.0], [0.0, 0.0, 0.5]),
                pbc=[True, True, True])),
      2, {}, [{}, {}], 1, False,
      (chemdash.master_code.Structure(index = 0, volume = 7.999999999999998, labels = [], atoms=
                 ase.Atoms(symbols = "SrTiO3", cell = [2.0, 2.0, 2.0],
                 charges = [2.0, 4.0, -2.0, -2.0, -2.0],
                 scaled_positions = ([0.75, 0.75, 0.25], [0.75, 0.25, 0.25], [0.5, 0.5, 0.5], [0.5, 0.0, 0.0], [0.0, 0.0, 0.5]),
                 pbc=True)), "unconverged", 0.0))
])

def test_multi_stage_vasp_calc(structure, num_calcs, main_settings, additional_settings, max_convergence_calcs,  expected_output, save_outcar, monkeypatch):
    """
    GIVEN a structure and set of vasp settings

    WHEN we run vasp on that structure with those settings

    THEN we return the optimised structure with the energy and result of the calculation

    Parameters
    ----------
    structure : ASE atoms
        The structure used in the VASP calculation
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

    ---------------------------------------------------------------------------
    Paul Sharp 07/12/2022
    """

    # Need to patch all file operations -- we don't want to perform any during tests
    monkeypatch.setattr(os, 'remove', lambda x : None)
    monkeypatch.setattr(shutil, 'copy2', lambda x, y : None)
    monkeypatch.setattr(shutil, 'copyfileobj', lambda x, y : None)
    monkeypatch.setattr('builtins.open', mock.mock_open())

    # Patch the "run_vasp()" routine in order to control the output
    mock_run = mock.MagicMock(side_effect =
                              [(ase.Atoms(symbols = "SrTiO3",
                                          cell = [2.0, 2.0, 2.0], charges = [2.0, 4.0, -2.0, -2.0, -2.0],
                                          scaled_positions = ([0.75, 0.75, 0.25], [0.75, 0.25, 0.25], [0.5, 0.5, 0.5], [0.5, 0.0, 0.0], [0.0, 0.0, 0.5]),
                                          pbc=[True, True, True]), 0.0, ""),
                               (ase.Atoms(symbols = "SrTiO3",
                                          cell = [2.0, 2.0, 2.0], charges = [2.0, 4.0, -2.0, -2.0, -2.0],
                                          scaled_positions = ([0.75, 0.75, 0.25], [0.75, 0.25, 0.25], [0.5, 0.5, 0.5], [0.5, 0.0, 0.0], [0.0, 0.0, 0.5]),
                                          pbc=[True, True, True]), 0.0, "")])

    monkeypatch.setattr(chemdash.vasp_calc, 'run_vasp', lambda x1, x2 : mock_run())

    # Patch the vasp_time -- not predictable during test
    monkeypatch.setattr(time, 'time', lambda : 0.0)

    # These are different instances of the same class, so we should compare the contents rather than compare them directly
    output = chemdash.vasp_calc.multi_stage_vasp_calc(structure, num_calcs, "", main_settings, additional_settings, max_convergence_calcs, save_outcar)
    assert output[0].__dict__ == expected_output[0].__dict__
    assert output[1:] == expected_output[1:]


#===========================================================================================================================================================
@pytest.mark.parametrize("structure, vasp_settings, expected_output", [
    (ase.Atoms(symbols = "SrTiO3", cell = [2.0, 2.0, 2.0],
               scaled_positions = ([0.75, 0.75, 0.25], [0.75, 0.25, 0.25], [0.5, 0.5, 0.5], [0.5, 0.0, 0.0], [0.0, 0.0, 0.5]),
               calculator=None,
               pbc=[True, True, True]), {'ibrion':1},
     (ase.Atoms(symbols = "SrTiO3", cell = [2.0, 2.0, 2.0],
                scaled_positions = ([0.75, 0.75, 0.25], [0.75, 0.25, 0.25], [0.5, 0.5, 0.5], [0.5, 0.0, 0.0], [0.0, 0.0, 0.5]),
                calculator=ase.calculators.vasp.Vasp(),
                pbc=[True, True, True]),
     1.0, ""))
])

def test_run_vasp(structure, vasp_settings, expected_output, monkeypatch):
    """
    GIVEN a structure and set of vasp settings

    WHEN we run vasp on that structure with those settings

    THEN we return the optimised structure with the energy and result of the calculation

    Parameters
    ----------
    structure : ASE atoms
        The structure used in the VASP calculation
    vasp_settings : dict
        The set of VASP options to apply with their values.

    ---------------------------------------------------------------------------
    Paul Sharp 27/06/2019
    """

    monkeypatch.setattr(structure, 'get_potential_energy', lambda: 1.0)

    final_structure, energy, result = chemdash.vasp_calc.run_vasp(structure, vasp_settings)

    # Class definition is that two ase Atoms objects are the same if they have the same atoms, unit cell,
    # positions and boundary conditions.
    # Therefore we need a separate test to ensure the calculator is attached.
    assert final_structure.calc.name is "Vasp"
    assert (final_structure, energy, result) == expected_output

    
#===========================================================================================================================================================
@pytest.mark.parametrize("structure, vasp_settings, expected_output", [
    (ase.Atoms(symbols = "SrTiO3", cell = [2.0, 2.0, 2.0],
               scaled_positions = ([0.75, 0.75, 0.25], [0.75, 0.25, 0.25], [0.5, 0.5, 0.5], [0.5, 0.0, 0.0], [0.0, 0.0, 0.5]),
               calculator=None,
               pbc=[True, True, True]), {'ibrion':1},
     (ase.Atoms(symbols = "SrTiO3", cell = [2.0, 2.0, 2.0],
                scaled_positions = ([0.75, 0.75, 0.25], [0.75, 0.25, 0.25], [0.5, 0.5, 0.5], [0.5, 0.0, 0.0], [0.0, 0.0, 0.5]),
                calculator=ase.calculators.vasp.Vasp(),
                pbc=[True, True, True]),
     0.0, "vasp failure"))
])

def test_run_vasp_exception(structure, vasp_settings, expected_output, monkeypatch):
    """
    GIVEN a structure and set of vasp settings

    WHEN vasp fails to run on that structure with those settings

    THEN we return the original structure with the "vasp failure" result.

    Parameters
    ----------
    structure : ASE atoms
        The structure used in the VASP calculation
    vasp_settings : dict
        The set of VASP options to apply with their values.

    ---------------------------------------------------------------------------
    Paul Sharp 27/06/2019
    """
    
    # We patch the calls to "structure.get_potential_energy()" in order to raise the exception.
    mock_exception = mock.MagicMock(side_effect = (AssertionError))

    monkeypatch.setattr(structure, 'get_potential_energy', lambda : mock_exception())

    assert chemdash.vasp_calc.run_vasp(structure, vasp_settings) == expected_output


#===========================================================================================================================================================
@pytest.mark.parametrize("executable, num_cores, vasp_pseudopotentials, expected_output", [
    ("/home/vasp", 2, "/home/bin/", [mock.call("import os\nexitcode = os.system('mpirun -np 2 /home/vasp')")]),
    ("/home/vasp", 1, "/home/bin/", [mock.call("import os\nexitcode = os.system('/home/vasp')")]),
])

def test_set_vasp_script(executable, num_cores, vasp_pseudopotentials, expected_output, monkeypatch):
    """
    GIVEN a structure and set of vasp settings

    WHEN vasp fails to run on that structure with those settings

    THEN we return the original structure with the "vasp failure" result.

    Parameters
    ----------
    executable : str
        The filepath for the vasp executable
    num_cores : int
        The number of cores over which vasp will be run
    vasp_pseudopotentials : str
        The filepath for the vasp pseudopotentials library

    ---------------------------------------------------------------------------
    Paul Sharp 27/06/2019
    """

    vasp_script = "run_vasp.sh"
    
    # We need to ensure we use "mock_open" to avoid writing to a file
    # Note the lack of lambda -- we want the mock itself, not its return value
    mock_file = mock.mock_open()
    monkeypatch.setattr('builtins.open', mock_file)

    chemdash.vasp_calc.set_vasp_script(vasp_script, executable, num_cores, vasp_pseudopotentials)

    # We check the calls to the write function as we do not return what we write
    assert mock_file().write.call_args_list == expected_output
    
    assert os.environ["VASP_SCRIPT"] == "./" + vasp_script
    assert os.environ["VASP_PP_PATH"] == vasp_pseudopotentials
    
    
#===========================================================================================================================================================
@pytest.mark.parametrize("value, expected_output", [
    (1.0, True),
    (0.0, True),
    (-1.0, True),
    ("1.0", True),
    ("0.0", True),
    ("-1.0", True),
    ("str", False),
    ("***", False),
])


def test_check_float(value, expected_output):
    """
    GIVEN an input value.

    WHEN we try to convert it to a float

    THEN we return a logical set depending on whether or not the value can be represented as a floating point number.


    Parameters
    ----------
    value : str
        The value of interest

    ---------------------------------------------------------------------------
    Paul Sharp 25/10/2017
    """

    assert chemdash.vasp_calc.check_float(value) == expected_output


#===========================================================================================================================================================
@pytest.mark.parametrize("test_outcar, expected_output", [
    ('''------------------------ aborting loop because EDIFF is reached ----------------------------------------
    reached required accuracy - stopping structural energy minimisation''', True),
    ('''reached required accuracy - stopping structural energy minimisation
    ------------------------ aborting loop because EDIFF is reached ----------------------------------------
    reached required accuracy - stopping structural energy minimisation''', True),
    ('''------------------------ aborting loop because EDIFF is reached ----------------------------------------
    ------------------------ aborting loop because EDIFF is reached ----------------------------------------
    reached required accuracy - stopping structural energy minimisation''', False),
    ('''------------------------ aborting loop because EDIFF is reached ----------------------------------------
    ------------------------ aborting loop because EDIFF is reached ----------------------------------------''', False),
    ('''reached required accuracy - stopping structural energy minimisation''', False),
    ('''------------------------ aborting loop because EDIFF is reached ----------------------------------------''', False),
])

def test_converged_in_one_scf_cycle(test_outcar, expected_output, monkeypatch):
    """
    GIVEN an OUTCAR file

    WHEN we check whether the calculation converged.

    THEN we return a logical set depending on whether or not the calculation converged in one SCF cycle


    Parameters
    ----------
    test_outcar_file : str
        OUTCAR file for this vasp calculation.

    ---------------------------------------------------------------------------
    Paul Sharp 10/11/2017
    """

    # We need to ensure we use "mock_open" to return our test data instead of reading a file
    # We also need to ensure that the return value of this Mock produces an iterable,
    # so we can loop over the lines of the file
    iterable_mock_file = mock.mock_open(read_data = test_outcar)
    iterable_mock_file.return_value.__iter__ = lambda x : iter(x.readline, '')

    # Note the lack of lambda -- we want the mock itself, not its return value
    # We pass a blank string for the file argument for this reason
    monkeypatch.setattr('builtins.open', iterable_mock_file)

    assert chemdash.vasp_calc.converged_in_one_scf_cycle("") == expected_output


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

    assert all([np.allclose(x, y) for x, y in zip(chemdash.vasp_calc.determine_vacancy_positions(STOX_structure), expected_output)])


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

    assert chemdash.vasp_calc.populate_points_with_vacancies(STO_atoms, vacancy_points) == STOX_structure
