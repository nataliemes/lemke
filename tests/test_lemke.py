"""
    This file contains tests for LCP solver based on Lemke's algorithm.


    There are 3 types of test cases included:
            
    (1) KNOWN_SUCCESS_CASES 
        LCPs with known solutions that were either computed by hand, or provided in the literature.

    (2) FAILURE_CASES
        LCPs where Lemke's algorithm cannot find a standard solution and it reaches a secondary ray.
        The termination behavior was computed by hand.

    (3) POS_DEF_CASES
        LCPs with positive definite M matrix.
        Positive definiteness of M guarantees a unique solution, which Lemke's algorithm must find.
        However, these LCPs are larger and harder to solve by hand, so we don't have pre-computed solutions.

        
    The correctness of solutions is checked as follows:

    - For FAILURE_CASES, the tests check that the program correctly calls exit(1).
    - For POS_DEF_CASES, the tests verify that the output satisfies the LCP conditions:
        z0 = 0,
        z >= 0, w >= 0,
        z[i] * w[i] = 0 for all i,
        w = Mz + q.
    - For KNOWN_SUCCESS_CASES, the tests check both that the output satisfies the LCP conditions
      and that it matches the pre-calculated solution.
    
"""


import dataclasses
import typing
import pytest

from lemke.lemke import lcp, tableau
from fractions import Fraction as F



def _lcp_from_data(M: list[list[F]], q: list[F], d: list[F]) -> lcp:

    """ Creates an LCP instance with given M, q, d. """

    # This function will not be needed if lcp class constructor can accept M, q, d
    # instead of the current implementation where it either
    # creates 0-filled instaces, or reads values directly from a file.

    n = len(q)

    if len(d) != n:
        raise ValueError("q and d must have the same length")

    if len(M) != n:
        raise ValueError("M must have as many rows as q")

    if any(len(row) != n for row in M):
        raise ValueError("M must be a square matrix")

    lcp_instance = lcp(n)

    lcp_instance.M = [[F(x) for x in row] for row in M]
    lcp_instance.q = [F(x) for x in q]
    lcp_instance.d = [F(x) for x in d]

    return lcp_instance



def _lemke_solver(lcp_instance: lcp) -> tableau:

    """ Runs the Lemke solver & returns the final tableau."""

    # This function will not be needed if the solver can 
    # directly return the final tableau or the solution.

    tabl = tableau(lcp_instance)
    tabl.runlemke(verbose=False, z0=False, silent=True)
    return tabl



@dataclasses.dataclass
class LCPTestCase:
    """ Defines data for one LCP test case for the Lemke solver. """

    factory: typing.Callable[[], lcp]            # produces an LCP instance
    solver: typing.Callable[[lcp], tableau]      # runs Lemke solver & returns final tableau
    expected: typing.Optional[list] = None       # expected solution vector (if the solution exists & we know it)
    tol: F = F(0)                                # tolerance for checking solution (0 by default)



# ---   TRIVIAL CASES   --------------------------------------------------------------------
# Cases where q >= 0.
# Lemke's algorithm should find the trivial solution: Z = 0, W = q, Z0 = 0.
# These tests currently fail, as the algorithm does not check if q >= 0 at the beginning.
TRIVIAL_CASES = [

    pytest.param(
        LCPTestCase(
            factory=lambda: _lcp_from_data(
                M=[
                    [2, -1], 
                    [-1, 1]
                ],
                q=[3, 1],
                d=[1, 1],
            ),
            solver=_lemke_solver,
            expected=[F(0), F(0), F(0), F(3), F(1)]
        ),
        id="trivial_2x2"
    ),

    pytest.param(
        LCPTestCase(
            factory=lambda: _lcp_from_data(
                M=[
                    [1/2, 2, 3/4], 
                    [-1, -11, 12/5],
                    [12, 34.3, -7]
                ],
                q=[0, 1.2, F(13,5)],
                d=[1, 1, 1],
            ),
            solver=_lemke_solver,
            expected=[F(0), F(0), F(0), F(0), F(0), F(1.2), F(13, 5)]
        ),
        id="trivial_3x3"
    ),

    pytest.param(
        LCPTestCase(
            factory=lambda: _lcp_from_data(
                M=[
                    [11.3, -4, F(2,7), 0], 
                    [0, F(-10,9), 12, 3],
                    [-17, 22.1, 0, -2],
                    [1, 2, 3, 4]
                ],
                q=[5, 0, 12.3, F(3,7)],
                d=[1, 1, 1, 1],
            ),
            solver=_lemke_solver,
            expected=[F(0), F(0), F(0), F(0), F(0), F(5), F(0), F(12.3), F(3, 7)]
        ),
        id="trivial_4x4"
    ),
  
]



# ---   IDENTITY MATRIX CASES   -------------------------------------------------------------
# Cases where M is an identity matrix.
# The solution can be easily verified by formula: Z[i] = max(0, -q[i]), W[i] = max(0, q[i]).
IDENTITY_MATRIX_CASES = [

    pytest.param(
        LCPTestCase(
            factory=lambda: _lcp_from_data(
                M=[
                    [1, 0], 
                    [0, 1]
                ],
                q=[-2, -1],
                d=[1, 1],
            ),
            solver=_lemke_solver,
            expected=[F(0), F(2), F(1), F(0), F(0)]
        ),
        id="identity_2x2"
    ),

    
    pytest.param(
        LCPTestCase(
            factory=lambda: _lcp_from_data(
                M=[
                    [1, 0, 0], 
                    [0, 1, 0],
                    [0, 0, 1]
                ],
                q=[F(1,3), 2, -10],
                d=[1, 1, 1],
            ),
            solver=_lemke_solver,
            expected=[F(0), F(0), F(0), F(10), F(1, 3), F(2), F(0)]
        ),
        id="identity_3x3"
    ),


    pytest.param(
        LCPTestCase(
            factory=lambda: _lcp_from_data(
                M=[
                    [1, 0, 0, 0], 
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]
                ],
                q=[-5.1, F(2,7), 10, -8],
                d=[1, 1, 1, 1],
            ),
            solver=_lemke_solver,
            expected=[F(0), F(5.1), F(0), F(0), F(8), F(0), F(2,7), F(10), F(0)]
        ),
        id="identity_4x4"
    ),
]



# ---   NON-DEGENERATE CASES   -----------------------------------------
# Cases where non-degenerate solution exists and is found.
NON_DEGENERATE_CASES = [
    
    pytest.param(
        LCPTestCase(
            factory=lambda: _lcp_from_data(
                M=[
                    [2, -1], 
                    [-1, 1]
                ],
                q=[-3, 1],
                d=[1, 1],
            ),
            solver=_lemke_solver,
            expected=[F(0), F(2), F(1), F(0), F(0)]
        ),
        id="non_degenerate_2x2"
    ),


    # example 4.3.3 (Cottle, Pang, Stone)
    pytest.param(
        LCPTestCase(
            factory=lambda: _lcp_from_data(
                M=[
                    [0, -1, 2], 
                    [2, 0, -2],
                    [-1, 1, 0]
                ],
                q=[-3, 6, -1],
                d=[1, 1, 1],
            ),
            solver=_lemke_solver,
            expected=[F(0), F(0), F(1), F(3), F(2), F(0), F(0)]
        ),
        id="non_degenerate_3x3"
    ),


    pytest.param(
        LCPTestCase(
            factory=lambda: _lcp_from_data(
                M=[
                    [0, 1, 4, -1], 
                    [2, 4, -1, 0],
                    [-1, -1, 2, -1],
                    [-2, -4, -1, 1]
                ],
                q=[2, 5, 0, -1],
                d=[1, 1, 1, 1],
            ),
            solver=_lemke_solver,
            expected=[F(0), F(0), F(0), F(1), F(2), F(4), F(4), F(0), F(0)]
        ),
        id="non_degenerate_integer_4x4"
    ),

    # example 4.4.17 (Cottle, Pang, Stone)
    pytest.param(
        LCPTestCase(
            factory=lambda: _lcp_from_data(
                M=[
                    [-1, 0, 2, -1], 
                    [0, 1, 1, 4],
                    [-2, -1, 0, 0],
                    [1, -4, 0, 0]
                ],
                q=[1/2, -1/2, 6, 6],
                d=[1, 1, 1, 1],
            ),
            solver=_lemke_solver,
            expected=[F(0), F(0), F(1, 2), F(0), F(0), F(1, 2), F(0), F(11, 2), F(4)]
        ),
        id="non_degenerate_fractional_4x4"
    ),

]



# ---   DEGENERATE CASES   ----------------------------------------------------------------
# Cases for which multiple solutions exist, but Lemke's algorithm finds only one and.
# the found (z, w) solution is degenerate - it has fewer than n positive components.
# Here, multiple candidates tie in a ratio test and can be chosen as the leaving variable.
# The lexicographic minimum ratio test is used for tie-breaking.
DEGENERATE_CASES = [

    # infinite solutions z1 = t, z2 = t+1, w = 0, t >= 0
    pytest.param(
        LCPTestCase(
            factory=lambda: _lcp_from_data(
                M=[
                    [1, -1], 
                    [-1, 1]
                ],
                q=[1, -1],
                d=[1, 1],
            ),
            solver=_lemke_solver,
            expected=[F(0), F(0), F(1), F(0), F(0)]
        ),
        id="degenerate_2x2_1"
    ),

    # page 141 in Cottle, Pang, Stone (1992)
    # infinite solutions: z1 + z2 = 1, w = 0, z >= 0
    pytest.param(
        LCPTestCase(
            factory=lambda: _lcp_from_data(
                M=[
                    [1, 1], 
                    [1, 1]
                ],
                q=[-1, -1],
                d=[1, 1],
            ),
            solver=_lemke_solver,
            expected=[F(0), F(0), F(1), F(0), F(0)]
        ),
        id="degenerate_2x2_2"
    ),

    # also a trivial case, since q = 0
    # infinite solutions: w = 0, z1 = z2, z >= 0
    pytest.param(
        LCPTestCase(
            factory=lambda: _lcp_from_data(
                M=[
                    [2, -2], 
                    [-1, 1]
                ],
                q=[0, 0],
                d=[1, 1],
            ),
            solver=_lemke_solver,
            expected=[F(0), F(0), F(0), F(0), F(0)]
        ),
        id="degenerate_2x2_3"
    ),

]




SUCCESS_CASES = []
SUCCESS_CASES += TRIVIAL_CASES
SUCCESS_CASES += IDENTITY_MATRIX_CASES
SUCCESS_CASES += DEGENERATE_CASES
SUCCESS_CASES += NON_DEGENERATE_CASES



@pytest.mark.parametrize("test_case", SUCCESS_CASES)
def test_lemke_by_expected_results(test_case: LCPTestCase, subtests):
    """
    Test the Lemke solver on LCPs with a known solution
    by comparing the solution size and values to the expected results.
    """

    lcp_instance = test_case.factory()
    final_tableau = test_case.solver(lcp_instance)

    sol = final_tableau.solution
    n = lcp_instance.n

    # print(sol)

    assert len(sol) == 2*n + 1

    for i, val in enumerate(sol):

        label = f"z{i}" if i <= n else f"w{i-n}"
        expected_val = test_case.expected[i]

        with subtests.test(f"{label} value"):
            assert abs(val - expected_val) <= test_case.tol





# ---   FAILURE_CASES   ---------------------------------------------------
# These matrices are selected to create LCPs for which Lemke fails 
# to find a solution and terminates on a secondary ray.
# For instance, if M = 0 and q < 0, the LCP has no solution.
FAILURE_CASES = [

    pytest.param(
        LCPTestCase(
            factory=lambda: _lcp_from_data(
                M=[
                    [0, 0], 
                    [0, 0]
                ],
                q=[-2/5, -1/2],
                d=[1, 1],
            ),
            solver=_lemke_solver
        ),
        id="failure_case_2x2_1"
    ),


    pytest.param(
        LCPTestCase(
            factory=lambda: _lcp_from_data(
                M=[
                    [0, -1], 
                    [-1, 0]
                ],
                q=[-1, -1],
                d=[1, 1],
            ),
            solver=_lemke_solver
        ),
        id="failure_case_2x2_2"
    ),

    pytest.param(
        LCPTestCase(
            factory=lambda: _lcp_from_data(
                M=[
                    [0, -1], 
                    [5, 1]
                ],
                q=[-7, -1],
                d=[1, 1],
            ),
            solver=_lemke_solver
        ),
        id="failure_case_2x2_3"
    ),

    pytest.param(
        LCPTestCase(
            factory=lambda: _lcp_from_data(
                M=[
                    [7, 1], 
                    [8, -2]
                ],
                q=[10, -1],
                d=[1, 1],
            ),
            solver=_lemke_solver
        ),
        id="failure_case_2x2_4"
    ),

    pytest.param(
        LCPTestCase(
            factory=lambda: _lcp_from_data(
                M=[
                    [1/2, -1/3], 
                    [-1, 0]
                ],
                q=[-1, -1/2],
                d=[1, 1],
            ),
            solver=_lemke_solver
        ),
        id="failure_case_2x2_5"
    ),

    pytest.param(
        LCPTestCase(
            factory=lambda: _lcp_from_data(
                M=[
                    [0, -1, 0], 
                    [-2, 0, 0],
                    [0, 1, 0]
                ],
                q=[-1, -1, -1],
                d=[1, 1, 1],
            ),
            solver=_lemke_solver
        ),
        id="failure_case_3x3"
    ),

    pytest.param(
        LCPTestCase(
            factory=lambda: _lcp_from_data(
                M=[
                    [17, -10, 0, 4], 
                    [-1, 9, 22, 0],
                    [0, -1, -1, -2],
                    [-13, 9, 41, 5]
                ],
                q=[8, 17, 0, -9],
                d=[1, 1, 1, 1],
            ),
            solver=_lemke_solver
        ),
        id="failure_case_4x4"
    ),
]



@pytest.mark.parametrize("test_case", FAILURE_CASES)
def test_lemke_failure(test_case: LCPTestCase):
    """
    Test the Lemke solver on LCPs that terminate on a secondary ray 
    by verifying that it raises SystemExit with code 1.
    """

    # Instead of calling exit(1), the code should be raising an error,
    # since currently, even if the program fails for some other reason,
    # these tests will still pass.

    lcp_instance = test_case.factory()
    
    with pytest.raises(SystemExit) as exc_info:
        test_case.solver(lcp_instance)
    
    assert exc_info.value.code == 1




# ---   POSITIVE DEFINITE CASES   ---------------------------------------------------
# Cases where M is a positive definite matrix.
# There exists a unique solution, which Lemke's algorithm is able to find.
POS_DEF_CASES = [
    pytest.param(
        LCPTestCase(
            factory=lambda: _lcp_from_data(
                M=[
                    [6, 2, 1, 0, 0],
                    [2, 5, 1, 0, 0],
                    [1, 1, 4, 1, 0],
                    [0, 0, 1, 3, 1],
                    [0, 0, 0, 1, 2]
                ],
                q=[-4, 1, -7, 10, -23],
                d=[1, 1, 1, 1, 1],
            ),
            solver=_lemke_solver,
        ),
        id="pos_def_5x5"
    ),


    pytest.param(
        LCPTestCase(
            factory=lambda: _lcp_from_data(
                M=[
                    [9/2, -1.2, 0.75, 0, -1/2, 0.3],
                    [-1.2, 5, -0.75, 1.2, 0, -0.4],
                    [3/4, -0.75, 6.0, -1.0, 0.5, 0],
                    [0, 1.2, -1.0, 4.0, -3/4, 0.6],
                    [-0.5, 0, 1/2, -0.75, 3.5, -1.2],
                    [0.3, -0.4, 0, 0.6, -1.2, 4.0]
                ],
                q=[2, -12, 3/2, 8.2, -1, 0],
                d=[1, 1, 1, 1, 1, 1],
            ),
            solver=_lemke_solver,
        ),
        id="pos_def_6x6"
    ),


    pytest.param(
        LCPTestCase(
            factory=lambda: _lcp_from_data(
                M=[
                    [13, -1.3, F(1,3), 0, -2, 0.75, -1],
                    [-1.3, 12, -3/4, 1.2, 0, -2, 0.5],
                    [F(1,3), -3/4, 14, -1, 2, 0, -1.2],
                    [0, 1.2, -1, 11, F(-2,3), 0.6, 1],
                    [-2, 0, 2, F(-2,3), 15, -1.5, 0.8],
                    [0.75, -2, 0, 0.6, -1.5, 12, -1],
                    [-1, 0.5, -1.2, 1, 0.8, -1, 13]
                ],
                q=[7, F(1,12), -2, -10, -42.1, 5, 2],
                d=[1, 1, 1, 1, 1, 1, 1],
            ),
            solver=_lemke_solver,
        ),
        id="pos_def_7x7"
    ),

    pytest.param(
        LCPTestCase(
            factory=lambda: _lcp_from_data(
                M=[
                    [12, -1.3, 1/2, 0, -2, 0.75, -1, 2],
                    [-1.3, 14, -3/4, 1.2, 0, -2, 0.5, -1],
                    [1/2, -3/4, 13, -1, 2, 0, -1.2, 0.8],
                    [0, 1.2, -1, 15, F(-2,3), 0.6, 1, -0.5],
                    [-2, 0, 2, F(-2,3), 16, -1.5, 0.8, F(1,3)],
                    [0.75, -2, 0, 0.6, -1.5, 14, -1, 1.2],
                    [-1, 0.5, -1.2, 1, 0.8, -1, 13, -2/5],
                    [2, -1, 0.8, -0.5, F(1,3), 1.2, -2/5, 15]
                ],
                q=[0, 1, F(-4,3), 32, -19.3, 5, -2, 3],
                d=[1, 1, 1, 1, 1, 1, 1, 1],
            ),
            solver=_lemke_solver,
        ),
        id="pos_def_8x8"
    ),

    pytest.param(
        LCPTestCase(
            factory=lambda: _lcp_from_data(
                M=[
                    [14, -1.2, F(1,3), 0, -2, 0.75, -1, 2, -3/4],
                    [-1.2, 15, -3/4, 1.2, 0, -2, 0.5, -1, 1.3],
                    [F(1,3), -3/4, 16, -1, 2, 0, -1.2, 0.8, F(-2,3)],
                    [0, 1.2, -1, 13, F(-2,3), 0.6, 1, -0.5, 2],
                    [-2, 0, 2, F(-2,3), 17, -1.5, 0.8, F(1,3), -1],
                    [0.75, -2, 0, 0.6, -1.5, 14, -1, 1.2, -0.4],
                    [-1, 0.5, -1.2, 1, 0.8, -1, 15, -2/5, 0.9],
                    [2, -1, 0.8, -0.5, F(1,3), 1.2, -2/5, 16, -1.1],
                    [-3/4, 1.3, F(-2,3), 2, -1, -0.4, 0.9, -1.1, 14]
                ],
                q=[1, 2, F(-23,12), 4.61, -17, 2, 0, -9/2, -4],
                d=[1, 1, 1, 1, 1, 1, 1, 1, 1],
            ),
            solver=_lemke_solver,
        ),
        id="pos_def_9x9"
    ),

    pytest.param(
        LCPTestCase(
            factory=lambda: _lcp_from_data(
                M=[
                    [15, -1.2, F(1,3), 0, -2, 0.75, -1, 2, -3/4, 1.5],
                    [-1.2, 16, -3/4, 1.2, 0, -2, 0.5, -1, 1.3, -2],
                    [F(1,3), -3/4, 17, -1, 2, 0, -1.2, 0.8, F(-2,3), 0.6],
                    [0, 1.2, -1, 14, F(-2,3), 0.6, 1, -0.5, 2, -1],
                    [-2, 0, 2, F(-2,3), 18, -1.5, 0.8, F(1,3), -1, 0.9],
                    [0.75, -2, 0, 0.6, -1.5, 15, -1, 1.2, -0.4, 1/2],
                    [-1, 0.5, -1.2, 1, 0.8, -1, 16, -2/5, 0.9, -1.3],
                    [2, -1, 0.8, -0.5, F(1,3), 1.2, -2/5, 17, -1.1, 2],
                    [-3/4, 1.3, F(-2,3), 2, -1, -0.4, 0.9, -1.1, 15, F(-2,3)],
                    [1.5, -2, 0.6, -1, 0.9, 1/2, -1.3, 2, F(-2,3), 18]
                ],
                q=[-4, 12, 0, 3/5, 2.6, -71, 2, -3, 0, 1],
                d=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            ),
            solver=_lemke_solver,
        ),
        id="pos_def_10x10"
    ),
]



@pytest.mark.parametrize("test_case", SUCCESS_CASES + POS_DEF_CASES)
def test_lemke_by_lcp_conditions(test_case: LCPTestCase, subtests):
    """
    Test the Lemke solver output by verifying LCP conditions:

    - Artificial variable z0 = 0
    - Nonnegativity: z >= 0, w >= 0
    - Complementarity: z[i] * w[i] = 0 (for each i)
    - Equation is satisfied: w = M * z + q
    """
    
    lcp_instance = test_case.factory()
    final_tableau = test_case.solver(lcp_instance)
    sol = final_tableau.solution


    n = lcp_instance.n
    z0 = sol[0]
    z = sol[1:n+1]
    w = sol[n+1:]

    # print("z:", z)
    # print("w:", w)

    with subtests.test("z0 = 0"):
        assert z0 == 0


    for i, val in enumerate(z):
        with subtests.test(f"z{i} nonnegativity"):
            assert val >= 0
    for i, val in enumerate(w):
        with subtests.test(f"w{i+1} nonnegativity"):
            assert val >= 0

    
    for i in range(n):
        with subtests.test(f"z{i+1} * w{i+1} = 0"):
            assert abs(z[i] * w[i]) <= test_case.tol

   
    for i in range(n):
        expected_w = sum(lcp_instance.M[i][j] * z[j] for j in range(n)) + lcp_instance.q[i]
        with subtests.test(f"w{i+1} = M * z + q"):
            assert abs(w[i] - expected_w) <= test_case.tol


