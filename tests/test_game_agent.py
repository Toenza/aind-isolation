"""This file is provided as a starting template for writing your own unit
tests to run and debug your minimax and alphabeta agents locally.  The test
cases used by the project assistant are not public.
"""

import unittest

import isolation
import game_agent
import timeit

from game_agent import AlphaBetaPlayer
from game_agent import MinimaxPlayer
from sample_players import GreedyPlayer

from importlib import reload


class IsolationTest(unittest.TestCase):
    """Unit tests for isolation agents"""

    def setUp(self):
        reload(game_agent)
        self.player1 = AlphaBetaPlayer()
        self.player2 = GreedyPlayer()
        self.game = isolation.Board(self.player1, self.player2)

    def test_example(self):
        self.game.apply_move((2, 3))
        self.game.apply_move((0, 5))
        # time_millis = lambda: 1000 * timeit.default_timer()
        # print(self.game.to_string())
        # game_copy = self.game.copy()
        # move_start = time_millis()
        # time_left = lambda: 200 - (time_millis() - move_start)
        # curr_move = self.game.active_player.get_move(game_copy, time_left)

        # self.game.apply_move(curr_move)
        self.game.play()


if __name__ == '__main__':
    unittest.main()
