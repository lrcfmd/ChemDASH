"""
|=============================================================================|
|               R A N D O M   N U M B E R   G E N E R A T O R S               |
|=============================================================================|
|                                                                             |
| This module contains random number generators (RNG) for use in scientific   |
| programming. RNGs are implemented as classes in order to preserve their     |
| internal states.                                                            |
|                                                                             |
| The main code in this module tests the RNG by determining the mean and      |
| variance of random numbers, and plotting histograms and scatter plots in    |
| order to examine the distribution and (lack of) correlation of the random   |
| numbers.                                                                    |
|                                                                             |
| Contains                                                                    |
| --------                                                                    |
|     NR_Ran                                                                  |
|     generate_random_seed                                                    |
|                                                                             |
|-----------------------------------------------------------------------------|
| Paul Sharp 27/03/2020                                                       |
|=============================================================================|
"""

from builtins import range

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:
    pass

import subprocess


# =============================================================================
class NR_Ran(object):
    """
    Generates random numbers using the "Ran" algorithm from the 2007 edition of
    "Numerical Recipes" (where the algorithm is written in C++).

    This algorithm combines Marsaglia 64-bit XORshift, multiply with carry
    (base 2^32) and Linear Congruential (modulo 2^64) methods of generating
    random numbers. Specifically, a generator is composed from the MLCG and a
    64-bit xorshift, added to another xorshift generator, with this output XORed
    with a multiply with carry generator -- i.e. the generator is combined from
    three constituent generators.

    This RNG is shown to have no systematic failures, and relatively few failures
    in the BigCrush suite of tests in the TestU01 framework. The period of the
    generator is ~ 3.158*10^57.

    NOTE: The C++ algorithm as written in Numerical Recipes relies on the way C++
          treats overflowing integers, that is, to take the value modulo 2^N, where
          N is the number of bits. Python, on the other hand, promotes such integers
          to long type which allows for arbitrary precision arithmetic. Therefore,
          in order to ensure that the random numbers generated do not exceed 2^64,
          a division modulo 2^64 is performed wherever there is a risk of overflow
          in the algorithm.

    References: W. H. Press et al, "Numerical Recipes: The Art of Scientific Programming",
                Cambridge University Press, Cambridge, (2007).
                S. Vigna, "An experimental exploration of Marsaglia's xorshift generators, scrambled",
                ACM Transactions in Mathematical Software, 42, 4, (2014).
                http://xoroshiro.di.unimi.it/ -- accessed 20/07/2016

    ---------------------------------------------------------------------------
    Paul Sharp 24/05/2017
    """

    # =========================================================================
    def __init__(self, seed):
        """
        Initialise the RNG with an integer seed. The RNG is then run in order to
        determine an initial value for each of the three constituent generators.

        Parameters
        ----------
        seed : integer
            The first value used to start off the RNG.

        Returns
        -------
        None

        -----------------------------------------------------------------------
        Paul Sharp 27/03/2020
        """

        # Initialise class variables and RNG parameters
        self.seed = seed
        self.maximum = 2**64
        self.v = 4101842887655102017
        self.w = 1

        # Seed cannot be equal to value of v above
        assert (seed != self.v), 'ERROR in rng.NR_Ran.__init__ -- seed must not equal value of intrinsic parameter, v={0:d}'.format(v)

        # Initialise the constituent RNGS
        self.u = self.seed ^ self.v
        self.int_64_bit()
        self.v = self.u
        self.int_64_bit()
        self.w = self.v
        self.int_64_bit()

    # =========================================================================
    def int_64_bit(self):
        """
        Use the generator to get a 64-bit integer random number.

        Parameters
        ----------
        None

        Returns
        -------
        : integer
            A pseudorandom 64-bit integer.

        -----------------------------------------------------------------------
        Paul Sharp 09/12/2019
        """

        # First generator composed from Linear Congruential Generator (LCG) and 64-bit XORshift generator
        # Linear Congruential generator
        self.u = self.u * 2862933555777941757 + 7046029254386353087
        self.u = self.u % self.maximum

        # Feed LCG output into 64-bit XORshift generator
        x = self.u ^ (self.u << 21)
        x = x % self.maximum
        x ^= x >> 35
        x = x % self.maximum
        x ^= x << 4
        x = x % self.maximum

        # Second generator: 64-bit XORshift generator
        self.v ^= self.v >> 17
        self.v = self.v % self.maximum
        self.v ^= self.v << 31
        self.v = self.v % self.maximum
        self.v ^= self.v >> 8
        self.v = self.v % self.maximum

        # Third generator: Multiply with carry
        self.w = 4294957665 * (self.w & (2**32 - 1)) + ((self.w >> 32) % self.maximum)
        self.w = self.w % self.maximum

        # Combine the output of the three generators
        ran_num = (x + self.v) ^ self.w
        return ran_num % self.maximum

    # =========================================================================
    def int_32_bit(self):
        """
        Use the generator to get a 32-bit integer random number.

        Parameters
        ----------
        None

        Returns
        -------
        : integer
            A pseudorandom 32-bit integer.

        -----------------------------------------------------------------------
        Paul Sharp 21/07/2016
        """

        return np.uint32(self.int_64_bit())

    # =========================================================================
    def real(self):
        """
        Use the generator to get a real random number in the range [0, 1].

        Parameters
        ----------
        None

        Returns
        -------
        : integer
            A pseudorandom real number in the range [0, 1]

        -----------------------------------------------------------------------
        Paul Sharp 21/07/2016
        """

        return float(5.421010862427522170037264004E-20 * self.int_64_bit())

    # =========================================================================
    def real_range(self, l_lim=0.0, u_lim=1.0):
        """
        Use the generator to get a real random number in the range [l_lim, u_lim].

        Parameters
        ----------
        l_lim, u_lim : float, optional
            The minimum and maximum values desired for the real random number

        Returns
        -------
        : float
            A pseudorandom real number in the range [l_lim, u_lim]

        -----------------------------------------------------------------------
        Paul Sharp 24/11/2016
        """

        return (self.real() * (u_lim - l_lim)) + l_lim

    # =========================================================================
    def int_range(self, l_lim=0, u_lim=2):
        """
        Use the generator to get a random integer in the range [l_lim, u_lim-1].

        Parameters
        ----------
        l_lim, u_lim : int, optional
            The minimum and maximum values desired for the random integer.
            Note that the int() function rounds down, so the maximum value is one
            less than the input "u_lim" (like the python intrinsic "range()").

        Returns
        -------
        : int
            A pseudorandom integer in the range [l_lim, u_lim-1]

        -----------------------------------------------------------------------
        Paul Sharp 24/11/2016
        """

        return int(self.real() * (u_lim - l_lim)) + l_lim

    # =========================================================================
    def point_3D(self):
        """
        Use the generator to get a real point in 3D space - three random coordinates in the range [0, 1].

        Parameters
        ----------
        None

        Returns
        -------
        : [float, float, float]
            A point in 3D space in (x, y, z) format with each coordinate in the range [0, 1]

        -----------------------------------------------------------------------
        Paul Sharp 25/07/2016
        """

        return [self.real(), self.real(), self.real()]

    # =========================================================================
    def weighted_choice(self, weights):
        """
        Choose an element of a list according to a set of weightings. This
        algorithm works by considering the range [0, sum(weights)] as divisions
        with the size corresponding to each weighting. A random number is then
        chosen in that range and the division it lands in yields the chosen
        element -- i.e., the roulette wheel method.

        This routine is based on a routine taken from:
        http://eli.thegreenplace.net/2010/01/22/weighted-random-generation-in-python, accessed on 24/05/2017.

        Parameters
        ----------
        weights : float
           The list of weights corresponding to each element in the list of interest.

        Returns
        -------
        index : int
            The index of the chosen element in the list of interest.

        -----------------------------------------------------------------------
        Paul Sharp 24/05/2017
        """

        ran_num = self.real_range(u_lim=sum(weights))
        index = 0  # Default in case of problems

        # Subtract the weights from the random number until we hit the correct region.
        for i, weight in enumerate(weights):
            ran_num -= weight
            if ran_num < 0:
                index = i
                break

        return index

    # =========================================================================
    def warm_up(self, num_values):
        """
        Generate and discard some random values in order to "warm up" the generator.

        Parameters
        ----------
        num_values : integer
            The number of random values cycled through.

        Returns
        -------
        None

        -----------------------------------------------------------------------
        Paul Sharp 13/09/2016
        """

        for i in range(num_values):
            x = self.int_64_bit()

        return None


# =============================================================================
# =============================================================================
def generate_random_seed(seed_bits):
    """
    Generate an integer seed of the apropriate number of bytes to use in a
    random number generator.

    This is done by using /dev/urandom, which is a special file on unix-based
    systems that accumulates noise from system hardware timings. Reading from
    this file generates pseudorandom values.

    Parameters
    ----------
    seed_bits : integer
        The number of bits available to store the random seed. Ideally, this
        should be equal to the bit-value of the RNG to be used, e.g., 64 for
        a 64-bit RNG.

    Returns
    -------
    random_seed : integer
        The value that will be used to start off the RNG.

    -----------------------------------------------------------------------
    Paul Sharp 09/12/2018
    """

    seed_bytes = str(seed_bits // 8)

    # This command will take the appropriate number of bytes of memory and generate a single, N-bit decimal integer
    random_seed_command = "od -vAn -N" + seed_bytes + " -tu" + seed_bytes + " < /dev/urandom"
    random_seed = int(subprocess.check_output(random_seed_command, shell=True))

    return random_seed

# =============================================================================
# =============================================================================
# Test suite for RNGs
if __name__ == '__main__':

    # Set up RNG with seed
    random_seed_command = "od -vAn -N8 -tu8 < /dev/urandom"
    random_seed = int(subprocess.check_output(random_seed_command, shell=True))
    rng = NR_Ran(random_seed)

    # Set number of random pairs (pairs needed for the correlation test - there are twice as many random values)
    N = 10000

    # Set up arrays for random numbers
    ran_nums = np.zeros((N, 2))

    # Generate the random numbers
    for i in range(0, N):
        ran_nums[i, 0] = rng.real()
        ran_nums[i, 1] = rng.real()

    # Mean, Variance and errors
    var = np.var(ran_nums)

    print('The mean of the random numbers is {0:f} +/- {1:f}'.format(np.mean(ran_nums), np.sqrt(var / float(2 * N))))
    print('The variance of the random numbers is {0:f} +/- {1:f}'.format(var, var * np.sqrt(2.0 / (float(2 * N) - 1.0))))

    # Histogram for distribution
    plt.hist(ran_nums.flatten(), bins=100, range=(0.0, 1.0))
    plt.title("Histogram of Uniformally Distributed Random Numbers")
    plt.xlabel("Random Number")
    plt.ylabel("Frequency")
    plt.show()

    # Plot of pairs for correlation
    plt.plot(ran_nums[:, 0], ran_nums[:, 1], "o")
    plt.title("Scatter plot of paired random numbers to test for correlation")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
