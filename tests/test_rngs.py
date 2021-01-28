import pytest
import chemdash.initialise

from builtins import int
from builtins import range

import numpy as np

#===========================================================================================================================================================
#===========================================================================================================================================================
#Tests


def test_NR_Ran_int_64_bit(rng):
    """
    The "int_64_bit()" routine should generate a random 64-bit integer.


    Parameters
    ----------
    None

    Returns
    -------
    None  

    ---------------------------------------------------------------------------
    Paul Sharp 16/01/2020
    """

    ran_vals = []
    for i in range(10000):

        ran_vals.append(rng.int_64_bit())

    assert min(ran_vals) >= 0.0
    assert max(ran_vals) <= 2**64

    assert all(type(item) is int for item in ran_vals)


#===========================================================================================================================================================
def test_NR_Ran_int_32_bit(rng):
    """
    The "int_32_bit()" routine should generate a random 32-bit integer.


    Parameters
    ----------
    None

    Returns
    -------
    None  

    ---------------------------------------------------------------------------
    Paul Sharp 14/07/2017
    """

    ran_vals = []
    for i in range(10000):

        ran_vals.append(rng.int_32_bit())

    assert min(ran_vals) >= 0.0
    assert max(ran_vals) <= 2**32

    assert all(type(item) is np.uint32 for item in ran_vals)


#===========================================================================================================================================================
def test_real(rng):
    """
    The "real()" routine should generate a random floating point between 0.0 and 1.0.


    Parameters
    ----------
    None

    Returns
    -------
    None  

    ---------------------------------------------------------------------------
    Paul Sharp 14/07/2017
    """

    ran_vals = []
    for i in range(10000):

        ran_vals.append(rng.real())

    assert min(ran_vals) >= 0.0
    assert max(ran_vals) <= 1.0

    assert all(type(item) is float for item in ran_vals)


#===========================================================================================================================================================
@pytest.mark.parametrize("l_lim, u_lim", [
    (0.0, 1.0),
    (5.0, 10.0),
    (-5.0, 0.0)
])


def test_real_range(rng, l_lim, u_lim):
    """
    The "real_range()" routine should generate a random floating point in the range [l_lim, u_lim].


    Parameters
    ----------
    None

    Returns
    -------
    None  

    ---------------------------------------------------------------------------
    Paul Sharp 14/07/2017
    """

    ran_vals = []
    for i in range(10000):

        ran_vals.append(rng.real_range(l_lim, u_lim))

    assert min(ran_vals) >= l_lim
    assert max(ran_vals) <= u_lim

    assert all(type(item) is float for item in ran_vals)


#===========================================================================================================================================================
@pytest.mark.parametrize("l_lim, u_lim", [
    (0, 1),
    (5, 10),
    (-5, 0)
])


def test_real_range(rng, l_lim, u_lim):
    """
    The "int_range()" routine should generate a random integer in the range [l_lim, u_lim].


    Parameters
    ----------
    None

    Returns
    -------
    None  

    ---------------------------------------------------------------------------
    Paul Sharp 14/07/2017
    """

    ran_vals = []
    for i in range(10000):

        ran_vals.append(rng.real_range(l_lim, u_lim))

    assert min(ran_vals) >= l_lim
    assert max(ran_vals) <= u_lim

    assert all(type(item) is float for item in ran_vals)


#===========================================================================================================================================================
def test_point_3D(rng):
    """
    The "point_3D()" routine should generate a random cartesian coordinate for a point in 3D space with each coordinate between 0 and 1.


    Parameters
    ----------
    None

    Returns
    -------
    None  

    ---------------------------------------------------------------------------
    Paul Sharp 16/01/2020
    """

    ran_vals = []
    for i in range(10000):

        ran_vals.append(rng.point_3D())

    assert [len(coordinate)==3 for coordinate in ran_vals]

    assert [0.0 <= point <= 1.0 for coordinate in ran_vals for point in coordinate]

    coordinates = [point for coordinate in ran_vals for point in coordinate]
    assert all(type(point) is float for point in coordinates)
