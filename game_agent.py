"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random

from math import sqrt


class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


def distance_to_opponent(game):
    player1pos = game.get_player_location(game.active_player)
    player2pos = game.get_player_location(game.inactive_player)
    return sqrt((player1pos[0] - player2pos[0]) ** 2 + (player1pos[1] - player2pos[1]) ** 2)


def centrality(game, m):
    player = game.active_player
    new_game = game.forecast_move(m)
    return distance_to_center(new_game, player)


def distance_to_center(game, player):
    w, h = game.width / 2., game.height / 2.
    y, x = game.get_player_location(player)
    return float((h - y) ** 2 + (w - x) ** 2)


def proximity_to_borders(game, player):
    # w, h = game.width / 2., game.height / 2.
    # y, x = game.get_player_location(player)
    # return float((h - y) ** 2 + (w - x) ** 2)
    return 0


def in_corner(game, player):
    pos = game.get_player_location(player)
    return (pos[0] == 0 or pos[0] == game.height) and (pos[1] == 0 or pos[1] == game.width)


def common_moves(game):
    moves1, moves2 = game.get_legal_moves(game.active_player), game.get_legal_moves(game.inactive_player)
    num = len(moves1 and moves2)
    return num


def filling(game):
    return game.move_count / (game.width * game.height)


def emptiness(game):
    return 1 / filling(game)


def is_opening(game):
    return filling(game) < 0.4


def is_middle_game(game):
    return not (is_opening(game) or is_endgame(game))


def is_endgame(game):
    return filling(game) > 0.6


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """

    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    val = 0

    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    mobility = float(own_moves - opp_moves)

    val += mobility

    val += sum(centrality(game, m) for m in game.get_legal_moves(player))
    val += common_moves(game)
    # val -= distance_to_center(game, player)
    val += emptiness(game) * (1 / distance_to_center(game, player))
    val -= emptiness(game) * common_moves(game)

    val -= in_corner(game, player) * 2

    return val


def custom_score_2(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    val = 0
    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    mobility = float(own_moves - opp_moves)

    val += mobility

    val += sum(centrality(game, m) for m in game.get_legal_moves(player))

    return val


def custom_score_3(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    val = 0
    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    mobility = float(own_moves - opp_moves)

    val += mobility

    val += sum(centrality(game, m) for m in game.get_legal_moves(player))
    val += emptiness(game) * (1 / distance_to_center(game, player))

    return val


class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            best_move = self.minimax(game, self.search_depth)

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.

        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        return self.eval_minimax(game, depth)[0]

    def eval_minimax(self, game, depth):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        if depth == 0:
            return game.get_player_location(self), self.score(game, self)

        best_score, func, best_move = None, None, (-1, -1)
        if game.active_player == self:
            func, best_score = max, float("-inf")
        else:
            func, best_score = min, float("inf")

        for move in game.get_legal_moves():
            next_ply = game.forecast_move(move)
            score = self.eval_minimax(next_ply, depth - 1)[1]
            if func(best_score, score) == score:
                best_move = move
                best_score = score

        return best_move, best_score


class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """

        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)
        if len(game.get_legal_moves()) > 0:
            best_move = random.choice(game.get_legal_moves())
        else:
            return best_move

        curr_depth = 1

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            while True:
                self.time_left = time_left
                if self.time_left() < self.TIMER_THRESHOLD:
                    raise SearchTimeout()
                best_move = self.alphabeta(game, curr_depth)
                curr_depth += 1

        except SearchTimeout:
            pass
            # if len(game.get_legal_moves()) > 0:
            #     best_move = random.choice(game.get_legal_moves())
            # else:
            #     best_move = (-1, -1)

        # Return the best move from the last completed search iteration
        return best_move

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        return self.eval_alphabeta(game, depth, alpha, beta)[0]

    def eval_alphabeta(self, game, depth, alpha, beta):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        if depth == 0:
            return (-1, -1), self.score(game, self)

        best_score, func, best_move, is_alpha = None, None, (-1, -1), True

        if game.active_player == self:
            best_score = float("-inf")
            func = max
            is_alpha = True
        else:
            best_score = float("inf")
            func = min
            is_alpha = False

        for move in game.get_legal_moves():
            next_ply = game.forecast_move(move)
            score = self.eval_alphabeta(next_ply, depth - 1, alpha, beta)[1]
            if score == func(best_score, score):
                best_score = score
                best_move = move
            if is_alpha:
                if best_score >= beta:
                    return best_move, best_score
                else:
                    alpha = max(best_score, alpha)
            else:
                if best_score <= alpha:
                    return best_move, best_score
                else:
                    beta = min(best_score, beta)

        return best_move, best_score
