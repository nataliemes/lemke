"""
    This file contains tests for a bimatrix game solver that
    converts the game to a Linear Complementarity Problem and then 
    solves it using either Lemke-Howson algorithm, or a tracing procedure.

    Each test case is verified in 2 ways:

    (1) Checking that the output of LH algorithm matches pre-computed equilibria for LH.
    (2) Checking that the output of both LH and tracing methods
        is a subset of the equilibria computed by pygambit.nash.enummixed_solve().
"""


import dataclasses
import typing
import pytest
import random
import pygambit

from lemke.bimatrix import bimatrix, payoffmatrix, uniform
from lemke import randomstart
from fractions import Fraction as F



def _bimatrix_from_data(A: list[F], B: list[F]) -> bimatrix:

    """ Creates a bimatrix instance with given A and B. """

    # This function will not be needed if bimatrix class constructor 
    # can accept A, B directly instead of reading values from a file.

    def to_frac_matrix(M):
        return [[F(int(x)) for x in row] for row in M]
    
    G = bimatrix.__new__(bimatrix)
    G.A = payoffmatrix(to_frac_matrix(A))
    G.B = payoffmatrix(to_frac_matrix(B))
    return G


def _LH_solver(G: bimatrix) -> list[list[F]]:

    """ Run LH algorithm & return the list of equilibria found. """

    lh_eqs_dict = G.LH("1-"+str(G.A.numrows + G.A.numcolumns))
    return [list(eq_key) for eq_key in lh_eqs_dict.keys()]


def _tracing_solver(G: bimatrix, trace: int = 10, seed: int = -1, accuracy: int = 1000) -> list[list[F]]:
    
    """ Run tracing & return the list of equilibria found. """
    
    if trace < 0:
        return []

    m = G.A.numrows
    n = G.A.numcolumns
    trset = {}  # dict to count how often each equilibrium is found

    if trace == 0:
        xprior = uniform(m)
        yprior = uniform(n)
        eq = G.runtrace(xprior, yprior)
        trset[eq] = 1
        trace = 1
    else:
        for k in range(trace):
            if seed >= 0:
                random.seed(10*trace*seed + k)
            x = randomstart.randInSimplex(m)
            xprior = randomstart.roundArray(x, accuracy)
            y = randomstart.randInSimplex(n)
            yprior = randomstart.roundArray(y, accuracy)
            eq = G.runtrace(xprior, yprior)
            if eq in trset:
                trset[eq] += 1
            else:
                trset[eq] = 1

    return [list(eq) for eq in trset.keys()]



@dataclasses.dataclass
class GameTestCase:
    factory: typing.Callable[[], bimatrix]                                   # returns bimatrix game (A, B)
    solver: typing.Optional[typing.Callable[[], list[list]]] = _LH_solver    # runs game solver & returns equilibria
    expected_LH: typing.Optional[list] = None                                # expected_LH solution for LH solver
    tol: F = F(0)                                                            # tolerance for checking solution (0 by default)



# ---   SINGLE STRATEGY GAMES   --------------------------------------------------------------
# Simple games where at least one player has a single strategy.
SINGLE_STRATEGY_GAMES = [
    pytest.param(
        GameTestCase(
            factory=lambda: _bimatrix_from_data(
                A=[
                    [3]
                ],
                B=[
                    [5]
                ]
            ),
            expected_LH=[[F(1), F(1)]]
        ),
        id="single_strategy_game_1x1"
    ),

    pytest.param(
        GameTestCase(
            factory=lambda: _bimatrix_from_data(
                A=[
                    [1, 2]
                ],
                B=[
                    [3, 4]
                ]
            ),
            expected_LH=[[F(1), F(0), F(1)]]
        ),
        id="single_strategy_game_1x2"
    ),

     pytest.param(
        GameTestCase(
            factory=lambda: _bimatrix_from_data(
                A=[
                    [1],
                    [2]
                ],
                B=[
                    [3],
                    [4]
                ]
            ),
            expected_LH=[[F(0), F(1), F(1)]]
        ),
        id="single_strategy_game_2x1"
    ),
]



# ---   DOMINANT STRATEGY GAMES   --------------------------------------------------------------------
# Games where at least one player has a strategy that strictly dominates all their other strategies.
DOMINANT_STRATEGY_GAMES = [

    pytest.param(
        GameTestCase(
            factory=lambda: _bimatrix_from_data(
                A=[
                    [3, 0],
                    [5, 1]
                ],
                B=[
                    [3, 5],
                    [0, 1]
                ]
            ),
            expected_LH=[[F(0), F(1), F(0), F(1)]]
        ),
        id="dominant_1_prisoners_dilemma"
    ),

    pytest.param(
        GameTestCase(
            factory=lambda: _bimatrix_from_data(
                A=[
                    [3, 1],
                    [4, 2]
                ],
                B=[
                    [2, 3],
                    [1, 4]
                ]
            ),
            expected_LH=[[F(0), F(1), F(0), F(1)]]
        ),
        id="dominant_2"
    ),

    pytest.param(
        GameTestCase(
            factory=lambda: _bimatrix_from_data(
                A=[
                    [1, 0, 0],
                    [2, 1, 0],
                    [3, 2, 1]
                ],
                B=[
                    [1, 2, 3],
                    [0, 1, 2],
                    [0, 0, 1]
                ]
            ),
            expected_LH=[[F(0), F(0), F(1), F(0), F(0), F(1)]]
        ),
        id="dominant_3"
    ),

]



# ---   ZERO SUM GAMES   ---------------------------------------------
# Games where advantage of one player means a loss for the other.
# These games usually do not have pure strategy equilibria.
ZERO_SUM_GAMES = [

    pytest.param(
        GameTestCase(
            factory=lambda: _bimatrix_from_data(
                A=[
                    [1, 0],
                    [0, 1]
                ],

                B=[
                    [-1, 0],
                    [0, -1]
                ]
            ),
            expected_LH=[[F(1, 2), F(1, 2), F(1, 2), F(1, 2)]]
        ),
        id="zero_sum_game_1"
    ),

    pytest.param(
        GameTestCase(
            factory=lambda: _bimatrix_from_data(
                A=[
                    [1,-1],
                    [-1,1]
                ],
                B=[
                    [-1,1],
                    [1,-1]
                ]
            ),
            expected_LH=[[F(1, 2), F(1, 2), F(1, 2), F(1, 2)]]
        ),
        id="zero_sum_game_2_matching_pennies"
    ),

    
    pytest.param(
        GameTestCase(
            factory=lambda: _bimatrix_from_data(
                A=[
                    [0, 1, -1],
                    [-1, 0, 1],
                    [1, -1, 0]
                ],
                B=[
                    [0, -1, 1],
                    [1, 0, -1],
                    [-1, 1, 0]
                ]
            ),
            expected_LH=[[F(1, 3), F(1, 3), F(1, 3), F(1, 3), F(1, 3), F(1, 3)]]
        ),
        id="zero_sum_game_3_rock_paper_scissors"
    ),


    pytest.param(
        GameTestCase(
            factory=lambda: _bimatrix_from_data(
                A=[
                    [2, -1, 0],
                    [-2, 1, 0]
                ],
                B=[
                    [-2, 1, 0],
                    [2, -1, 0]
                ]
            ),
            expected_LH=[
                [F(1, 2), F(1, 2), F(0), F(0), F(1)],
                [F(1, 2), F(1, 2), F(1, 3), F(2, 3), F(0)]
            ]
        ),
        id="zero_sum_game_4"
    ),

]



# ---   COORDINATION GAMES   ------------------------------------------------------------------
# Games where players get a higher payoff when they select the same or compatible strategies.
# These games usually have multiple Nash equilibria with pure strategies.
COORDINATION_GAMES = [

    pytest.param(
        GameTestCase(
            factory=lambda: _bimatrix_from_data(
                A=[
                    [8, 0],
                    [0, 8]
                ],
                B=[
                    [8, 0],
                    [0, 8]
                ]
            ),
            expected_LH=[
                [F(1), F(0), F(1), F(0)],
                [F(0), F(1), F(0), F(1)]
            ]
        ),
        id="coordination_game_1_pure"
    ),

    
    pytest.param(
        GameTestCase(
            factory=lambda: _bimatrix_from_data(
                A=[
                    [3, 0],
                    [0, 2]
                ],
                B=[
                    [2, 0],
                    [0, 3]
                ]
            ),
            expected_LH=[
                [F(1), F(0), F(1), F(0)],
                [F(0), F(1), F(0), F(1)]
            ]
        ),
        id="coordination_game_2_battle_of_the_sexes"
    ),

    pytest.param(
        GameTestCase(
            factory=lambda: _bimatrix_from_data(
                A=[
                    [8, 0],
                    [0, 5]
                ],
                B=[
                    [8, 0],
                    [0, 5]
                ]
            ),
            expected_LH=[
                [F(1), F(0), F(1), F(0)],
                [F(0), F(1), F(0), F(1)]
            ]
        ),
        id="coordination_game_3_assurance"
    ),

]



# ---   DEGENERATE GAMES   ----------------------------------------------------------------------
# Games where there exists a mixed strategy with more pure best responses than its support size.
DEGENERATE_GAMES = [

    pytest.param(
        GameTestCase(
            factory=lambda: _bimatrix_from_data(
                A=[
                    [1, 1],
                    [1, 1]
                ],
                B=[
                    [2, 2],
                    [2, 2]
                ]
            ),
            expected_LH=[
                [F(1), F(0), F(0), F(1)],
                [F(0), F(1), F(0), F(1)],
                [F(0), F(1), F(1), F(0)]
            ]
        ),
        id="degenarate_game_1"
    ),

    pytest.param(
        GameTestCase(
            factory=lambda: _bimatrix_from_data(
                A=[
                    [1, 1, 1],
                    [2, 2, 2]
                ],
                B=[
                    [3, 3, 3],
                    [4, 4, 4]
                ]
            ),
            expected_LH=[
                [F(0), F(1), F(0), F(0), F(1)],
                [F(0), F(1), F(1), F(0), F(0)],
                [F(0), F(1), F(0), F(1), F(0)],
            ]
        ),
        id="degenarate_game_2"
    ),

]



# ---   GENERAL GAMES   -----------------------------------------------
# Games for extra testing that do not fall into the above categories.
GENERAL_GAMES = [

    pytest.param(
        GameTestCase(
            factory=lambda: _bimatrix_from_data(
                A=[
                    [2, 3],
                    [2, 1.5],
                    [0, 1.5],
                    [0, 0]
                ],
                B=[
                    [1, 0],
                    [0.5, 1],
                    [2.5, 1],
                    [2, 2]
                ]
            ),
            expected_LH=[
                [F(1), F(0), F(0), F(0), F(1), F(0)],
                [F(1, 2), F(1, 2), F(0), F(0), F(1), F(0)]
            ]
        ),
        id="general_game_1"
    ),

    pytest.param(
        GameTestCase(
            factory=lambda: _bimatrix_from_data(
                A=[
                    [3, 1],
                    [0, 2],
                    [2, 0]
                ],
                B=[
                    [2, 0],
                    [1, 3],
                    [0, 2]
                ]
            ),
            expected_LH=[
                [F(1), F(0), F(0), F(1), F(0)],
                [F(0), F(1), F(0), F(0), F(1)]
            ]
        ),
        id="general_game_2"
    ),

]



GAMES = []
GAMES += SINGLE_STRATEGY_GAMES
GAMES += DOMINANT_STRATEGY_GAMES
GAMES += ZERO_SUM_GAMES
GAMES += COORDINATION_GAMES
GAMES += DEGENERATE_GAMES
GAMES += GENERAL_GAMES


@pytest.mark.parametrize("test_case", GAMES)
def test_bimatrix_LH_by_expected_results(test_case: GameTestCase, subtests):

    """ Tests LH algorithm by comparing the output to the pre-computed solution. """

    G = test_case.factory()
    lh_eqs = test_case.solver(G)  # list of equilibria from LH solver

    
    assert len(lh_eqs) == len(test_case.expected_LH)

    for eq_idx, (eq, exp) in enumerate(zip(lh_eqs, test_case.expected_LH)):
        for comp_idx, (a, b) in enumerate(zip(eq, exp)):
            with subtests.test(f"Eq. {eq_idx}, comp. {comp_idx}"):
                assert abs(a - b) <= test_case.tol, \
                    f"Equilibrium {eq_idx}, component {comp_idx}: actual {a}, expected {b} (tol={test_case.tol})"




SOLVERS = [
    _LH_solver,
    _tracing_solver
]


@pytest.mark.parametrize("test_case", GAMES)
@pytest.mark.parametrize("solver", SOLVERS)
def test_bimatrix_with_pygambit(test_case: GameTestCase, solver, subtests):

    """ Tests both LH and tracing methods against pygambit solutions. """
    
    G = test_case.factory()
    lh_eqs = solver(G)

    # building the pygambit game
    A = [[float(x) for x in row] for row in G.A.matrix]
    B = [[float(x) for x in row] for row in G.B.matrix]
    g = pygambit.Game.new_table([len(A), len(A[0])])
    p1, p2 = g.players
    for i, row in enumerate(A):
        for j, val in enumerate(row):
            g[i, j][p1] = F(val)
            g[i, j][p2] = F(B[i][j])


    # getting the pygambit results
    # enummixed_solve() is used to get all equilibria
    pyg_eqs = []
    for eq in pygambit.nash.enummixed_solve(g, rational=True).equilibria:
        flat_eq = [s[1] for s in eq[p1]] + [s[1] for s in eq[p2]]
        pyg_eqs.append([F(x) for x in flat_eq])  # convert to Fraction (otherwise it's Rational)
    

    # print("PYGAMBIT EQUILIBRIA:  ", pyg_eqs)
    # print("OUTPUTTED EQUILIBRIA: ", lh_eqs)


    # each equilibrium LH found should appear in equilibria that pygambit found
    for idx, eq in enumerate(lh_eqs):
        with subtests.test(f"Equilibrium {idx} pygambit check"):
            matched = any(
                all(abs(a - b) <= test_case.tol for a, b in zip(eq, ne))
                for ne in pyg_eqs
            )
            assert matched, f"Equilibrium not in pygambit set: {eq}"


