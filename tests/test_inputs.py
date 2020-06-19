import pytest
import mock

import chemdash.inputs

import yaml

#===========================================================================================================================================================
#===========================================================================================================================================================
#Fixtures

@pytest.fixture
def test_params(scope="module"):
    """
    This fixture returns a smaller version of the "params" dictionary used in the code for input options.

    Returns
    -------
    params : dict
        A dictionary containg a sample of the input options.

    ---------------------------------------------------------------------------
    Paul Sharp 20/07/2017
    """

    return {"calculator": {"value": "gulp", "specified": False},        
            "restart": {"value": "False", "specified": False},
            "verbosity": {"value": "verbose", "specified": False},
            "temp": {"value": 0.0, "specified": False},
            "grid_points": {"value": "8", "specified": False},
            "gulp_keywords": {"value": "opti, pot", "specified": False},
            "gulp_options": {"value": "", "specified": False},
        }


#===========================================================================================================================================================
#===========================================================================================================================================================
#Tests

@pytest.mark.parametrize("expected_output", [
    ([mock.call('calculator=gulp\n'),
      mock.call('grid_points=8\n'),
      mock.call('gulp_keywords=opti, pot\n'),
      mock.call('gulp_options=\n'),
      mock.call('restart=False\n'),
      mock.call('temp=0.0\n'),
      mock.call('verbosity=verbose\n')])
])

def test_write_defaults_to_file(test_params, expected_output, monkeypatch):
    """
    GIVEN a file

    WHEN we obtain the default ChemDASH parameters and write them to file

    THEN we get a file containing the default values of the ChemDASH input options

    Parameters
    ----------
    None

    Returns
    -------
    None  

    ---------------------------------------------------------------------------
    Paul Sharp 09/08/2018
    """
    
    monkeypatch.setattr(chemdash.inputs, 'initialise_default_params', lambda x : test_params)

    # We need to ensure we use "mock_open" to avoid writing to a file
    # Note the lack of lambda -- we want the mock itself, not its return value
    # We pass a blank string for the file argument for this reason
    mock_file = mock.mock_open()
    monkeypatch.setattr('builtins.open', mock_file)

    chemdash.inputs.write_defaults_to_file("")

    # We check the calls to the write function as we do not return what we write
    assert mock_file().write.call_args_list == expected_output


#===========================================================================================================================================================
@pytest.mark.parametrize("test_input, expected_output", [
    ('''temp = 0.01
    grid_points = 6
    calculator = vasp
    gulp_options = time 10 minutes, stepmax 0.5''',
     ({"temp": {"value": "0.01", "specified": True},
       "grid_points": {"value": "6", "specified": True},
       "calculator": {"value": "vasp", "specified": True},
       "restart": {"value": "False", "specified": False},
       "verbosity": {"value": "verbose", "specified": False},
       "gulp_keywords": {"value": "opti, pot", "specified": False},
       "gulp_options": {"value": "time 10 minutes, stepmax 0.5", "specified": True},
       },
      [])),
    ('''temp:0.01
    grid_points = 6
    calculator = vasp
    foo=bar''',
     ({"temp": {"value": 0.0, "specified": False},
       "grid_points": {"value": "6", "specified": True},
       "calculator": {"value": "vasp", "specified": True},
       "restart": {"value": "False", "specified": False},
       "verbosity": {"value": "verbose", "specified": False},
       "gulp_keywords": {"value": "opti, pot", "specified": False},
       "gulp_options": {"value": "", "specified": False},
       },
      ['The keyword(s) temp:0.01, foo are invalid.'])),
])


def test_parse_input(test_params, test_input, expected_output, monkeypatch):
    """
    GIVEN an input file

    WHEN we update the params dictionary with the desired options

    THEN we return the dictionary and a list of errors

    Parameters
    ----------
    test_params: dict
        A subset of the full list of input parameters.
    test_input: str
        A mock input file.

    ---------------------------------------------------------------------------
    Paul Sharp 20/07/2017
    """

    monkeypatch.setattr(chemdash.inputs, 'initialise_default_params', lambda x : test_params)

    # We need to ensure we use "mock_open" to return our test data instead of reading a file
    # Note the lack of lambda -- we want the mock itself, not its return value
    # We pass a blank string for the file argument for this reason
    monkeypatch.setattr('builtins.open', mock.mock_open(read_data = test_input))

    assert chemdash.inputs.parse_input("", "") == expected_output


#===========================================================================================================================================================
@pytest.mark.parametrize("test_errors, expected_output", [
    (['The keyword(s) temp:0.01, foo are invalid.',
      '"grid_points" is 5, but should be an even integer.',
      '"calculator" is castep, but should only be one of the supported values: gulp, vasp'],
     [mock.call('3 errors were found in the input file: ""\n'),
      mock.call('\tERROR in input file "" -- The keyword(s) temp:0.01, foo are invalid.\n'),
      mock.call('\tERROR in input file "" -- "grid_points" is 5, but should be an even integer.\n'),
      mock.call('\tERROR in input file "" -- "calculator" is castep, but should only be one of the supported values: gulp, vasp\n')])
])

def test_report_input_file_errors(test_errors, expected_output, monkeypatch):
    """
    GIVEN a list of input file errors

    WHEN we write them to file

    THEN we get a file containing the full list of errors in the input file

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

    chemdash.inputs.report_input_file_errors(test_errors, "", "")

    # We check the calls to the write function as we do not return what we write
    assert mock_file().write.call_args_list == expected_output


#===========================================================================================================================================================
@pytest.mark.parametrize("keyword, value, errors, list_output, expected_output", [
    ("temp", "0", [], (), (0.0, [])),
    ("temp", "1", [], (), (1.0, [])),
    ("temp", "4", [], (), (4.0, [])),
    ("temp", "4.0", [], (), (4.0, [])),
    ("temp", "3.5", [], (), (3.5, [])),
    ("temp", "-4", [], (), (-4.0, ['"temp" is -4.0, but should be a positive number.'])),
    ("temp", "foobar", [], (), ("foobar", ['"temp" is foobar, but should be a positive number.'])),
    ("temp", "['4', '4', '4']", [], (), ("['4', '4', '4']", ['"temp" is [\'4\', \'4\', \'4\'], but should be a positive number.'])),
    ("temp", ['4', '4', '4'], [], ([4.0, 4.0, 4.0], []), ([4.0, 4.0, 4.0], [])),
])

def test_check_positive_float(keyword, value, errors, list_output, expected_output, monkeypatch):
    """
    GIVEN an input keyword and value.

    WHEN when we convert the value to a float.

    THEN we return the value converted to a float, or an error message.

    Parameters
    ----------
    None

    Returns
    -------
    None  

    ---------------------------------------------------------------------------
    Paul Sharp 09/08/2018
    """
    
    monkeypatch.setattr(chemdash.inputs, 'check_list_positive_float', lambda x, y, z : list_output)

    assert chemdash.inputs.check_positive_float(keyword, value, errors) == expected_output


#===========================================================================================================================================================
@pytest.mark.parametrize("keyword, value, errors, expected_output", [
    ("temp", "foobar", [], ("foobar", ['"temp" is foobar, but all values should be positive floats.'])),
    ("temp", "['4', '4', '4']", [], ("['4', '4', '4']", ['"temp" is [\'4\', \'4\', \'4\'], but all values should be positive floats.'])),
    ("temp", ['1', '1', '1'], [], ([1.0, 1.0, 1.0], [])),
    ("temp", ['4', '4', '4'], [], ([4.0, 4.0, 4.0], [])),
    ("temp", ['3.5', '3.5', '3.5'], [], ([3.5, 3.5, 3.5], [])),
    ("temp", ['4', '-4', '4'], [], ([4.0, -4.0, 4.0], ['"temp" is [4.0, -4.0, 4.0], but all values should be positive floats.'])),
    ("temp", ['1', '-2', '3'], [], ([1.0, -2.0, 3.0], ['"temp" is [1.0, -2.0, 3.0], but all values should be positive floats.'])),
])

def test_check_list_positive_float(keyword, value, errors, expected_output, monkeypatch):
    """
    GIVEN an input keyword and list of values.

    WHEN when we convert the values to floats.

    THEN we return the values converted to a list of floats, or an error message.

    Parameters
    ----------
    None

    Returns
    -------
    None  

    ---------------------------------------------------------------------------
    Paul Sharp 09/08/2018
    """
    
    assert chemdash.inputs.check_list_positive_float(keyword, value, errors) == expected_output


#===========================================================================================================================================================
@pytest.mark.parametrize("keyword, value, errors, list_output, expected_output", [
    ("grid_points", "0", [], (), (0, [])),
    ("grid_points", "1", [], (), (1, [])),
    ("grid_points", "4", [], (), (4, [])),
    ("grid_points", "4.0", [], (), ("4.0", ['"grid_points" is 4.0, but should be a positive integer.'])),
    ("grid_points", "3.5", [], (), ("3.5", ['"grid_points" is 3.5, but should be a positive integer.'])),
    ("grid_points", "-4", [], (), (-4, ['"grid_points" is -4, but should be a positive integer.'])),
    ("grid_points", "foobar", [], (), ("foobar", ['"grid_points" is foobar, but should be a positive integer.'])),
    ("grid_points", "['4', '4', '4']", [], (), ("['4', '4', '4']", ['"grid_points" is [\'4\', \'4\', \'4\'], but should be a positive integer.'])),
    ("grid_points", ['4', '4', '4'], [], ([4, 4, 4], []), ([4, 4, 4], [])),
])


def test_check_positive_int(keyword, value, errors, list_output, expected_output, monkeypatch):
    """
    GIVEN an input keyword and value.

    WHEN when we convert the value to a integer.

    THEN we return the value converted to a integer, or an error message.

    Parameters
    ----------
    None

    Returns
    -------
    None  

    ---------------------------------------------------------------------------
    Paul Sharp 09/08/2018
    """

    monkeypatch.setattr(chemdash.inputs, 'check_list_positive_int', lambda x, y, z : list_output)

    assert chemdash.inputs.check_positive_int(keyword, value, errors) == expected_output


#===========================================================================================================================================================
@pytest.mark.parametrize("keyword, value, errors, expected_output", [
    ("grid_points", "foobar", [], ("foobar", ['"grid_points" is foobar, but all values should be positive integers.'])),
    ("grid_points", "['4', '4', '4']", [], ("['4', '4', '4']", ['"grid_points" is [\'4\', \'4\', \'4\'], but all values should be positive integers.'])),
    ("grid_points", ['1', '1', '1'], [], ([1, 1, 1], [])),
    ("grid_points", ['4', '4', '4'], [], ([4, 4, 4], [])),
    ("grid_points", ['3.5', '3.5', '3.5'], [], (["3.5", "3.5", "3.5"], ['"grid_points" is [\'3.5\', \'3.5\', \'3.5\'], but all values should be positive integers.'])),
    ("grid_points", ['4', '-4', '4'], [], ([4, -4, 4], ['"grid_points" is [4, -4, 4], but all values should be positive integers.'])),
    ("grid_points", ['1', '-2', '3'], [], ([1, -2, 3], ['"grid_points" is [1, -2, 3], but all values should be positive integers.'])),
])

def test_check_list_positive_int(keyword, value, errors, expected_output, monkeypatch):
    """
    GIVEN an input keyword and list of values.

    WHEN when we convert the values to integers.

    THEN we return the values converted to a list of integers, or an error message.

    Parameters
    ----------
    None

    Returns
    -------
    None  

    ---------------------------------------------------------------------------
    Paul Sharp 09/08/2018
    """
    
    assert chemdash.inputs.check_list_positive_int(keyword, value, errors) == expected_output


#===========================================================================================================================================================
@pytest.mark.parametrize("keyword, value, errors, list_output, expected_output", [
    ("grid_points", "0", [], (), (0, [])),
    ("grid_points", "1", [], (), (1, ['"grid_points" is 1, but should be an even integer.'])),
    ("grid_points", "4", [], (), (4, [])),
    ("grid_points", "4.0", [], (), ("4.0", ['"grid_points" is 4.0, but should be an even integer.'])),
    ("grid_points", "3.5", [], (), ("3.5", ['"grid_points" is 3.5, but should be an even integer.'])),
    ("grid_points", "-4", [], (), (-4, [])),
    ("grid_points", "foobar", [], (), ("foobar", ['"grid_points" is foobar, but should be an even integer.'])),
    ("grid_points", "['4', '4', '4']", [], (), ("['4', '4', '4']", ['"grid_points" is [\'4\', \'4\', \'4\'], but should be an even integer.'])),
    ("grid_points", ['4', '4', '4'], [], ([4, 4, 4], []), ([4, 4, 4], [])),
])


def test_check_even_int(keyword, value, errors, list_output, expected_output, monkeypatch):
    """
    GIVEN an input keyword and value.

    WHEN when we convert the value to an even integer.

    THEN we return the value converted to an even integer, or an error message.

    Parameters
    ----------
    None

    Returns
    -------
    None  

    ---------------------------------------------------------------------------
    Paul Sharp 09/08/2018
    """

    monkeypatch.setattr(chemdash.inputs, 'check_list_even_int', lambda x, y, z : list_output)

    assert chemdash.inputs.check_even_int(keyword, value, errors) == expected_output


#===========================================================================================================================================================
@pytest.mark.parametrize("keyword, value, errors, expected_output", [
    ("grid_points", "foobar", [], ("foobar", ['"grid_points" is foobar, but all values should be even integers.'])),
    ("grid_points", "['4', '4', '4']", [], ("['4', '4', '4']", ['"grid_points" is [\'4\', \'4\', \'4\'], but all values should be even integers.'])),
    ("grid_points", ['1', '1', '1'], [], ([1, 1, 1], ['"grid_points" is [1, 1, 1], but all values should be even integers.'])),
    ("grid_points", ['4', '4', '4'], [], ([4, 4, 4], [])),
    ("grid_points", ['3.5', '3.5', '3.5'], [], (["3.5", "3.5", "3.5"], ['"grid_points" is [\'3.5\', \'3.5\', \'3.5\'], but all values should be even integers.'])),
    ("grid_points", ['4', '-4', '4'], [], ([4, -4, 4], [])),
    ("grid_points", ['1', '-2', '3'], [], ([1, -2, 3], ['"grid_points" is [1, -2, 3], but all values should be even integers.'])),
])

def test_check_list_even_int(keyword, value, errors, expected_output, monkeypatch):
    """
    GIVEN an input keyword and list of values.

    WHEN when we convert the values to even integers.

    THEN we return the values converted to a list of even integers, or an error message.

    Parameters
    ----------
    None

    Returns
    -------
    None  

    ---------------------------------------------------------------------------
    Paul Sharp 09/08/2018
    """
    
    assert chemdash.inputs.check_list_even_int(keyword, value, errors) == expected_output


#===========================================================================================================================================================
@pytest.mark.parametrize("keyword, value, errors, allowed_ints, expected_output", [
    ("seed_bits", "64", [], [32, 64], (64, [], True)),
    ("seed_bits", "4", [], [32, 64], (4, ['"seed_bits" is 4, but should be one of the supported values: 32, 64'], False)),
    ("seed_bits", "64.0", [], [32, 64], ("64.0", ['"seed_bits" is 64.0, but should be one of the supported values: 32, 64'], False)),
    ("seed_bits", "3.5", [], [32, 64], ("3.5", ['"seed_bits" is 3.5, but should be one of the supported values: 32, 64'], False)),
    ("seed_bits", "foobar", [], [32, 64], ("foobar", ['"seed_bits" is foobar, but should be one of the supported values: 32, 64'], False)),
])


def test_check_int_on_list(keyword, value, errors, allowed_ints, expected_output):
    """
    The "check_int_on_list()" routine should convert the input string to an int, and check that the int is included in the list of supported values, adding
    an error to the error list if either of these are not the case.


    Parameters
    ----------
    None

    Returns
    -------
    None  

    ---------------------------------------------------------------------------
    Paul Sharp 13/07/2017
    """

    assert chemdash.inputs.check_int_on_list(keyword, value, errors, allowed_ints) == expected_output


#===========================================================================================================================================================
@pytest.mark.parametrize("keyword, value, errors, bits, expected_output", [
    ("random_seed", "0", [], 32, (0, [])),
    ("random_seed", "-12", [], 32, (-12, ['"random_seed" is -12, but should be an unsigned 32-bit integer.'])),
    ("random_seed", "4294967295", [], 32, (4294967295, [])),
    ("random_seed", "4294967296", [], 32, (4294967296, ['"random_seed" is 4294967296, but should be an unsigned 32-bit integer.'])),
    ("random_seed", "4294967296", [], 64, (4294967296, [])),
    ("random_seed", "foobar", [], 64, ("foobar", ['"random_seed" is foobar, but should be an unsigned 64-bit integer.'])),
])


def test_check_unsigned_n_bit_int(keyword, value, errors, bits, expected_output):
    """
    The "check_unsigned_n_bit_int()" routine should convert the input string to an int, and check that the int is an unsigned int with the specified number
    of bits, adding an error to the error list if either of these are not the case.


    Parameters
    ----------
    None

    Returns
    -------
    None  

    ---------------------------------------------------------------------------
    Paul Sharp 13/07/2017
    """

    assert chemdash.inputs.check_unsigned_n_bit_int(keyword, value, errors, bits) == expected_output


#===========================================================================================================================================================
@pytest.mark.parametrize("keyword, value, errors, allowed_strings, expected_output", [
    ("verbosity", "terse", [], ["terse", "verbose"], ("terse", [], True)),
    ("verbosity", "foobar", [], ["terse", "verbose"], ("foobar", ['"verbosity" is foobar, but should only be one of the supported values: terse, verbose'], False)),
])


def test_check_str_on_list(keyword, value, errors, allowed_strings, expected_output):
    """
    The "check_str_on_list()" routine should check that the input string is included in the list of supported values, adding an error to the error list if not.


    Parameters
    ----------
    None

    Returns
    -------
    None  

    ---------------------------------------------------------------------------
    Paul Sharp 13/07/2017
    """

    assert chemdash.inputs.check_str_on_list(keyword, value, errors, allowed_strings) == expected_output


#===========================================================================================================================================================
@pytest.mark.parametrize("keyword, value, errors, allowed_strings, expected_output", [
    ("old_swap_groups", ["cations", "atoms", "all"], [], ["cations", "anions", "atoms", "all"], (["cations", "atoms", "all"], [], True)),
    ("old_swap_groups", ["cations", "atoms", "foobar"], [], ["cations", "anions", "atoms", "all"], (["cations", "atoms", "foobar"], ['"old_swap_groups" contains foobar, but should only contain the supported values: cations, anions, atoms, all'], False)),
    ("old_swap_groups", ["foo", "bar"], [], ["cations", "anions", "atoms", "all"], (["foo", "bar"], ['"old_swap_groups" contains foo, but should only contain the supported values: cations, anions, atoms, all', '"old_swap_groups" contains bar, but should only contain the supported values: cations, anions, atoms, all'], False)),
])


def test_check_str_list_on_list(keyword, value, errors, allowed_strings, expected_output):
    """
    The "check_str_list_on_list()" routine should check that each string in the input list of strings is included in the list of supported values, adding an
    error to the error list if not.


    Parameters
    ----------
    None

    Returns
    -------
    None  

    ---------------------------------------------------------------------------
    Paul Sharp 13/07/2017
    """

    assert chemdash.inputs.check_str_list_on_list(keyword, value, errors, allowed_strings) == expected_output


#===========================================================================================================================================================
@pytest.mark.parametrize("keyword, value, errors, expected_output", [
    ("restart", "True", [], (True, [])),
    ("restart", "False", [], (False, [])),
    ("restart", "TRUE", [], (True, [])),
    ("restart", "FALSE", [], (False, [])),
    ("restart", "true", [], (True, [])),
    ("restart", "false", [], (False, [])),
    ("restart", "t", [], (True, [])),
    ("restart", "f", [], (False, [])),
    ("restart", "yes", [], (True, [])),
    ("restart", "no", [], (False, [])),
    ("restart", "on", [], (True, [])),
    ("restart", "off", [], (False, [])),
    ("restart", "y", [], ("y", ['"restart" is y, but should be "True" or "False".'])),
    ("restart", "n", [], ("n", ['"restart" is n, but should be "True" or "False".'])),
])


def test_check_boolean(keyword, value, errors, expected_output):
    """
    The "check_boolean()" routine should check that the input string can be represented as a synonym of True or False, adding an error to the error list if
    this is not possible.


    Parameters
    ----------
    None

    Returns
    -------
    None  

    ---------------------------------------------------------------------------
    Paul Sharp 13/07/2017
    """

    assert chemdash.inputs.check_boolean(keyword, value, errors) == expected_output


#===========================================================================================================================================================
@pytest.mark.parametrize("keyword, value, errors, expected_output", [
    ("swap_groups", ["cations", "anions", "atoms", "all"], [], (["cations", "anions", "atoms", "all"], [], True)),
    ("swap_groups", ["cations", "atoms", "atoms", "all"], [], (["cations", "atoms", "atoms", "all"], ['"swap_groups" contains atoms more than once, but each item in this list should only appear once.'], False)),
    ("swap_groups", ["cations", "cations", "all", "all"], [], (["cations", "cations", "all", "all"], ['"swap_groups" contains cations more than once, but each item in this list should only appear once.', '"swap_groups" contains all more than once, but each item in this list should only appear once.'], False)),
])


def test_check_duplicates_in_list(keyword, value, errors, expected_output):
    """
    The "check_duplicates_in_list()" routine should check that each string in an input list of strings is unique, adding an error to the error list if not.


    Parameters
    ----------
    None

    Returns
    -------
    None  

    ---------------------------------------------------------------------------
    Paul Sharp 13/07/2017
    """

    assert chemdash.inputs.check_duplicates_in_list(keyword, value, errors) == expected_output


#===========================================================================================================================================================
@pytest.mark.parametrize("swap_groups, errors, main_swap_groups, expected_output", [
    ([["cations"], ["anions"], ["atoms"], ["all"]], [], ["cations", "anions", "atoms", "all"], ([["cations"], ["anions"], ["atoms"], ["all"]], [], True)),
    ([["Sr-X"], ["O-Ti"]], [], ["cations", "anions", "atoms", "all"], ([["Sr-X"], ["O-Ti"]], [], True)),
    ([["Sr:X"], ["O:Ti"]], [], ["cations", "anions", "atoms", "all"], ([["Sr:X"], ["O:Ti"]], ['"swap_groups" contains "Sr:X", which is not a valid swap group. The main swap groups are "cations", "anions", "atoms", "all", and custom swap groups should be in the format "[Chemical Symbol]-[Chemical Symbol]-[Chemical Symbol] . . .", with each chemical symbol appearing only once.', '"swap_groups" contains "O:Ti", which is not a valid swap group. The main swap groups are "cations", "anions", "atoms", "all", and custom swap groups should be in the format "[Chemical Symbol]-[Chemical Symbol]-[Chemical Symbol] . . .", with each chemical symbol appearing only once.'], False)),
    ([["Sr-"], ["X-Sr-X"], ["O:T"]], [], ["cations", "anions", "atoms", "all"], ([["Sr-"], ["X-Sr-X"], ["O:T"]], ['"swap_groups" contains "Sr-", which is not a valid swap group. The main swap groups are "cations", "anions", "atoms", "all", and custom swap groups should be in the format "[Chemical Symbol]-[Chemical Symbol]-[Chemical Symbol] . . .", with each chemical symbol appearing only once.', '"swap_groups" contains "X-Sr-X", which is not a valid swap group. The main swap groups are "cations", "anions", "atoms", "all", and custom swap groups should be in the format "[Chemical Symbol]-[Chemical Symbol]-[Chemical Symbol] . . .", with each chemical symbol appearing only once.', '"swap_groups" contains "O:T", which is not a valid swap group. The main swap groups are "cations", "anions", "atoms", "all", and custom swap groups should be in the format "[Chemical Symbol]-[Chemical Symbol]-[Chemical Symbol] . . .", with each chemical symbol appearing only once.'], False)),
])


def test_check_swap_groups_valid(swap_groups, errors, main_swap_groups, expected_output):
    """
    The "check_swap_groups_valid()" routine should check that each of the swap groups is one of the main swap groups or follows the correct format for a
    custom swap group, adding an error to the error list if not.


    Parameters
    ----------
    None

    Returns
    -------
    None  

    ---------------------------------------------------------------------------
    Paul Sharp 13/07/2017
    """

    assert chemdash.inputs.check_swap_groups_valid(swap_groups, errors, main_swap_groups) == expected_output


#===========================================================================================================================================================
@pytest.mark.parametrize("test_list, expected_output", [
    ("2", (["2", "2", "2"], [])),
    ("2,4", (["2", "2", "4"], [])),
    ("2,4,6", (["2", "4", "6"], [])),
    ("", (["", "", ""], [])),
    ("2,4,6,8", (["2", "4", "6", "8"], ['"grid_points" should be a list of up to three values, but 4 values have been supplied.'])),
])


def test_convert_to_three_element_list(test_list, expected_output):
    """
    GIVEN a list of three or fewer values

    WHEN we convert it to a three element list

    THEN we get a list of three elements, made by extending the occurances of the first entry

    Parameters
    ----------
    test_list
        The values to extend to a three element list.

    ---------------------------------------------------------------------------
    Paul Sharp 20/07/2017
    """

    assert chemdash.inputs.convert_to_three_element_list("grid_points", test_list, []) == expected_output


#===========================================================================================================================================================
@pytest.mark.parametrize("test_dict, expected_output", [
    ('xc:PBE, prec:Normal, encut: 600, ediff:1e-4, ediffg:-0.02',
     ({'ediff': '1e-4','ediffg': -0.02,'encut': 600,'prec':'Normal','xc':'PBE'}, [])),
    ('xc:PBE, prec:Normal, encut: 600, ediff:1e-4, ediffg:-0.02, restart:None',
     ({'ediff': '1e-4','ediffg': -0.02,'encut': 600,'prec':'Normal','restart': None,'xc':'PBE'}, [])),
    ('xc:PBE, prec:Normal, encut: 600, ediff:1e-4, ediffg:-0.02, restart:none',
     ({'ediff': '1e-4','ediffg': -0.02,'encut': 600,'prec':'Normal','restart': None,'xc':'PBE'}, [])),
    ('xc:PBE, prec:Normal, encut: 600, ediff:1e-4, ediffg:-0.02, restart:NONE',
     ({'ediff': '1e-4','ediffg': -0.02,'encut': 600,'prec':'Normal','restart': None,'xc':'PBE'}, [])),
    (None, (None, []))
])


def test_convert_to_dict_with_yaml(test_dict, expected_output):
    """
    GIVEN a list of inputs with values

    WHEN we convert it to a dictionary using YAML

    THEN we get a list of three elements, made by extending the occurances of the first entry

    Parameters
    ----------
    test_dict
        The inputs and values to convert to a dictionary.

    ---------------------------------------------------------------------------
    Paul Sharp 09/08/2018
    """

    assert chemdash.inputs.convert_to_dict_with_yaml("", test_dict, []) == expected_output

    
#===========================================================================================================================================================
@pytest.mark.parametrize("test_dict, expected_output", [
    ('I AM ERROR',
     ({}, ['"" is "I AM ERROR". YAML failed to convert this to a valid dictionary. The contents should be syntactically valid Python.'])),
    (None, (None, []))
])


def test_convert_to_dict_with_yaml_exception(test_dict, expected_output, monkeypatch):
    """
    GIVEN a list of inputs with values

    WHEN we convert it to a dictionary using YAML

    THEN we get a list of three elements, made by extending the occurances of the first entry

    Parameters
    ----------
    test_dict
        The inputs and values to convert to a dictionary.

    ---------------------------------------------------------------------------
    Paul Sharp 27/06/2019
    """

    # We patch the calls to "yaml.safe_load()" in order to raise the exception.
    mock_exception = mock.MagicMock(side_effect = yaml.scanner.ScannerError)

    monkeypatch.setattr(yaml, 'safe_load', lambda x : mock_exception())

    assert chemdash.inputs.convert_to_dict_with_yaml("", test_dict, []) == expected_output
