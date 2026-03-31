import os
import pytest
import pygambit

from fractions import Fraction as Fr
from tests.efg_to_sequence_form import solve_via_sequence_form


TEST_FILES_DIR = "tests/games"
FILES = [os.path.join(TEST_FILES_DIR, f) for f in os.listdir(TEST_FILES_DIR) if os.path.isfile(os.path.join(TEST_FILES_DIR, f))]


@pytest.mark.parametrize("file_path", FILES)
def test_efg_max_regret(file_path, subtests):
    
    game = pygambit.read_efg(file_path)

    x_probs, y_probs = solve_via_sequence_form(game)

    profile = game.mixed_behavior_profile([x_probs, y_probs], rational=True)
    max_r = profile.max_regret()
    
    with subtests.test("max regret = 0"):
        assert max_r == Fr(0)




# test by cross-checking with lcp_solve() from gambit
# currently fails, for example, on general_sum_perfect_info.efg
# since efg --> lcp conversion is not done exactly like in lcp_solve()


# @pytest.mark.parametrize("file_path", FILES)
# def test_efg_by_pygambit(file_path, subtests):
    
#     game = pygambit.read_efg(file_path)
#     x_probs, y_probs = solve_via_sequence_form(game)
#     pygambit_eq = pygambit.nash.lcp_solve(game).equilibria[0]


#     for i, player in enumerate(game.players):
        
#         my_probs = x_probs if i == 0 else y_probs
#         gambit_probs = pygambit_eq[player]

#         for i, infoset in enumerate(player.infosets):
#             my_dist = my_probs[i]
#             gambit_dist = gambit_probs[infoset]

#             for i, action in enumerate(infoset.actions):
#                 actual = my_dist[i]
#                 expected = gambit_dist[action]
#                 with subtests.test(f"pygambit cross check: player={player.label}, infoset={infoset.number}, action={action.label}, actual={actual}, expected={expected}"):
#                     assert abs(actual - expected) == 0

