"""
Tic Tac Toe Player from Project 1: Tic-Tac-Toe (CS50AI)
Python Version: 3.10
"""

import math
from copy import deepcopy
from typing import List, Tuple, Union, Literal

X = "X"
O = "O"
EMPTY = None

# Declare types
BoardSign = Union[X, O, EMPTY]
PlayerSign = Union[X, O]

Board = List[List[BoardSign]]
EmptyBoard = List[List[EMPTY]]

BoardAction = Tuple[int, int]
BoardActionSet = {BoardAction}
Score = Literal[-1, 0, 1]


def initial_state() -> EmptyBoard:
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board: Board) -> PlayerSign:
    """
    Returns player who has the next turn on a board.
    """
    if terminal(board):
        return None  # Game is already over

    # Count EMPTY cells in a list in a pythonic way
    total: int = sum([row.count(EMPTY) for row in board])

    # If amount of EMPTY cells is even, it's X's turn, otherwise it's O's turn
    return O if total % 2 == 0 else X


def actions(board: Board) -> BoardActionSet:
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    actions_set: BoardActionSet = set()

    for i in range(3):
        for j in range(3):
            if board[i][j] == EMPTY:
                actions_set.add((i, j))

    return actions_set or None


def result(board: Board, action: BoardAction) -> Board:
    """
    Returns the board that results from making move (i, j) on the board.
    """
    i, j = action
    if i not in range(3) or j not in range(3):
        raise Exception("Invalid move!")

    copy_board: Board = deepcopy(board)

    if copy_board[i][j] is not EMPTY:
        raise Exception(f"Invalid action! ({i}, {j})")

    copy_board[i][j] = player(copy_board)
    return copy_board


def winner(board: Board) -> PlayerSign:
    """
    Returns the winner of the game, if there is one.
    """
    for i in range(3):
        row_matched = board[i][0] == board[i][1] == board[i][2]
        column_matched = board[0][i] == board[1][i] == board[2][i]

        if row_matched:
            row_element = board[i][0]
            if row_element is not None:
                return row_element  # Row match

        elif column_matched:
            column_element = board[0][i]
            if column_element is not None:
                return column_element  # Column match

    left_to_right_diagonal_matched = board[0][0] == board[1][1] == board[2][2]
    right_to_left_diagonal_matched = board[0][2] == board[1][1] == board[2][0]

    # Check diagonal matches
    if left_to_right_diagonal_matched or right_to_left_diagonal_matched:
        center = board[1][1]
        if center is not EMPTY:
            return center

    return None


def terminal(board: Board) -> bool:
    """
    Returns True if game is over, False otherwise.
    """
    # If there is a winner or there are no more actions, game is over
    return not actions(board) or winner(board) is not None


def utility(board: Board) -> Score:
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    game_winner: PlayerSign = winner(board)

    # If game is not over, return 0. Otherwise, return 1 if X won, -1 if O won
    return 1 if game_winner == X else -1 if game_winner == O else 0


def min_player(board: Board, alpha: Score) -> Score:
    if terminal(board):
        return utility(board)

    # Get all possible actions
    possible_actions: BoardActionSet = actions(board)

    # Minimize the score
    min_point = math.inf
    for action in possible_actions:
        min_point = min(min_point, max_player(result(board, action), min_point))
        if min_point <= alpha:
            break  # Prune
        alpha = min(alpha, min_point)  # Update alpha

    return min_point


def max_player(board: Board, beta: Score) -> Score:
    if terminal(board):
        return utility(board)

    # Get all possible actions
    possible_actions: BoardActionSet = actions(board)

    # Maximize the score
    max_point = -math.inf
    for action in possible_actions:
        max_point = max(max_point, min_player(result(board, action), max_point))
        if max_point >= beta:
            break  # Prune
        beta = max(beta, max_point)  # Update beta

    return max_point


def minimax(board: Board) -> Union[BoardAction, None]:
    """
    Returns the optimal action for the current player on the board.
    """
    if terminal(board):
        return None

    # Get all possible actions
    possible_actions: BoardActionSet = actions(board)

    # Get the current player
    ai_player: PlayerSign = player(board)

    # AI is X
    if ai_player == X:
        # Optimal move is the one with the highest score
        optimal_decision: (Score, BoardAction) = (-math.inf, None)

        for action in possible_actions:
            action_score = min_player(result(board, action), optimal_decision[0])
            optimal_decision = max(optimal_decision, (action_score, action), key=lambda item: item[0])

        return optimal_decision[1]

    # AI is O
    optimal_decision = (math.inf, None)

    for action in possible_actions:
        action_score: Score = max_player(result(board, action), optimal_decision[0])
        optimal_decision = min(optimal_decision, (action_score, action), key=lambda item: item[0])

    return optimal_decision[1]
