import pytest
import mock

import chemdash.master_code

import ase

#===========================================================================================================================================================
#===========================================================================================================================================================
#Track tests

#UNIT TESTS


#INTEGRATION TESTS



#===========================================================================================================================================================
#===========================================================================================================================================================
#Fixtures

@pytest.fixture
def structure_count(scope="module"):
    """
    This fixture returns a Counts object.

    Parameters
    ----------
    None

    Returns
    -------
    structure_count : Counts
        An object that keeps track of the number of structures that achieve certain outcomes.

    ---------------------------------------------------------------------------
    Paul Sharp 09/08/2018
    """

    return chemdash.master_code.Counts()

#===========================================================================================================================================================
@pytest.fixture
def test_params(scope="module"):
    """
    This fixture returns a smaller version of the "params" dictionary used in the code for input options.

    Returns
    -------
    params : dict
        A dictionary containg a sample of the input options.

    ---------------------------------------------------------------------------
    Paul Sharp 25/06/2019
    """

    return {"calculator": {"value": "gulp", "specified": True},        
            "restart": {"value": "False", "specified": True},
            "verbosity": {"value": "verbose", "specified": False},
            "temp": {"value": 0.0, "specified": True},
            "grid_points": {"value": [8, 8, 8], "specified": True},
            "gulp_keywords": {"value": "opti pot", "specified": True},
            "gulp_options": {"value": ["time 5 minutes", "stepmx 0.5"], "specified": True},
            "vasp_settings":{"value": {"xc": "PBE", "prec": "Normal", "encut": 600},
                             "specified": True}
        }


#===========================================================================================================================================================
#===========================================================================================================================================================
#Unit Tests

@pytest.mark.parametrize("expected_output", [
    ([
mock.call('\n'),
mock.call('################################################################################\n'),
mock.call('#                                   ChemDASH                                   #\n'),
mock.call('################################################################################\n'),
mock.call('\n'),
mock.call('Summary of Inputs\n'),
mock.call('\n'),
mock.call('--------------------------------------------------------------------------------\n'),
mock.call('calculator                     = gulp\n'),
mock.call('grid_points                    = 8, 8, 8\n'),
mock.call('gulp_keywords                  = opti pot\n'),
mock.call('gulp_options                   = stepmx 0.5, time 5 minutes\n'),
mock.call('restart                        = False\n'),
mock.call('temp                           = 0.0\n'),
mock.call('vasp_settings                  = encut: 600, prec: Normal, xc: PBE\n'),
mock.call('--------------------------------------------------------------------------------\n'),
mock.call('\n')])
])

def test_write_output_file_header(test_params, expected_output, monkeypatch):
    """
    GIVEN a ChemDASH file and set of input parameters

    WHEN we write the output file header

    THEN we get an output file with the ChemDASH header

    Parameters
    ----------
    None

    Returns
    -------
    None  

    ---------------------------------------------------------------------------
    Paul Sharp 20/01/2020
    """

    # We need to ensure we use "mock_open" to avoid writing to a file
    # Note the lack of lambda -- we want the mock itself, not its return value
    # We pass a blank string for the file argument for this reason
    mock_file = mock.mock_open()
    monkeypatch.setattr('builtins.open', mock_file)

    file_handle = mock_file()
    chemdash.master_code.write_output_file_header(file_handle, test_params)

    # We check the calls to the write function as we do not return what we write
    assert file_handle.write.call_args_list == expected_output


#===========================================================================================================================================================
@pytest.mark.parametrize("test_input, expected_output", [
    ("OX", "O"),
    ("X", ""),
    ("NO", "NO"),
])

def test_strip_vacancies(test_input, expected_output):
    """
    The "strip_vacancies()" routine should remove vacancies -- represented as "X" atoms -- from an ase
    atoms object.


    Parameters
    ----------
    None

    Returns
    -------
    None  

    ---------------------------------------------------------------------------
    Paul Sharp 06/07/2017
    """

    struct = ase.Atoms(test_input)

    struct = chemdash.master_code.strip_vacancies(struct)

    assert struct.get_chemical_formula() == expected_output


#===========================================================================================================================================================
@pytest.mark.parametrize("test_list, expected_output", [
    ([1.0, 2.0, 3.0], [mock.call('0'), mock.call(' 1.00000000'), mock.call(' 2.00000000'), mock.call(' 3.00000000'), mock.call('\n')]),
    (['a', 'b', 'c'], [mock.call('0'), mock.call(' a'), mock.call(' b'), mock.call(' c'), mock.call('\n')]),
])

def test_output_list(test_list, expected_output, monkeypatch):
    """
    GIVEN a list of floats

    WHEN we write the list to a file

    THEN we get a file containig each of the values in the list 

    Parameters
    ----------
    None

    Returns
    -------
    None  

    ---------------------------------------------------------------------------
    Paul Sharp 09/08/2018
    """

    # We need to ensure we use "mock_open" to avoid writing to a file
    # Note the lack of lambda -- we want the mock itself, not its return value
    # We pass a blank string for the file argument for this reason
    mock_file = mock.mock_open()
    monkeypatch.setattr('builtins.open', mock_file)

    file_handle = mock_file()
    chemdash.master_code.output_list(file_handle, "0", test_list)

    # We check the calls to the write function as we do not return what we write
    assert file_handle.write.call_args_list == expected_output


#===========================================================================================================================================================
@pytest.mark.parametrize("result, expected_output, expected_text", [
    ("gulp failure",
     chemdash.master_code.Counts(),
     [mock.call('GULP has failed to perform the calculation for this structure, so it will be rejected.\n')]),
    ("timed out", 
     chemdash.master_code.Counts(timed_out = 1),
     [mock.call('The optimisation of this structure has timed out, so it will be rejected.\n')]),
    ("unconverged", 
     chemdash.master_code.Counts(unconverged = 1),
     [mock.call('The optimisation of this structure has not converged, so it will be rejected.\n')]),
])

def test_report_rejected_structures(result, expected_output, expected_text, structure_count, monkeypatch):
    """
    GIVEN a list of floats

    WHEN we write the list to a file

    THEN we get a file containig each of the values in the list 

    Parameters
    ----------
    None

    Returns
    -------
    None  

    ---------------------------------------------------------------------------
    Paul Sharp 09/08/2018
    """

    # We need to ensure we use "mock_open" to avoid writing to a file
    # Note the lack of lambda -- we want the mock itself, not its return value
    # We pass a blank string for the file argument for this reason
    mock_file = mock.mock_open()
    monkeypatch.setattr('builtins.open', mock_file)

    file_handle = mock_file()
    updated_structure_count = chemdash.master_code.report_rejected_structure(file_handle, result, "gulp", structure_count)

    # We check the calls to the write function as we do not return what we write
    assert file_handle.write.call_args_list == expected_text

    #assert updated_structure_count == expected_output
