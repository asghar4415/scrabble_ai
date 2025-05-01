# ai_player.py
from typing import Dict, Optional, List
from copy import deepcopy
from .board import Board
from .game_utils import LETTER_SCORES
import random


def get_ai_move(board: Board, depth: int = 2) -> Optional[Dict]:
    """Improved AI using Minimax with Alpha-Beta pruning"""
    best_move = None
    alpha = -float('inf')
    beta = float('inf')

    possible_moves = generate_possible_moves(board)
    if not possible_moves:
        return None

    for move in possible_moves:
        board_copy = deepcopy(board)
        board_copy.place_word(move["word"], move["row"], move["col"],
                              move["direction"], is_ai=True)

        score = minimax(board_copy, depth-1, alpha, beta, False)

        if score > alpha:
            alpha = score
            best_move = move

    return best_move


def minimax(board: Board, depth: int, alpha: float, beta: float,
            maximizing_player: bool) -> float:
    if depth == 0 or board.game_over:
        return evaluate_position(board)

    possible_moves = generate_possible_moves(board)

    if maximizing_player:
        max_eval = -float('inf')
        for move in possible_moves:
            board_copy = deepcopy(board)
            board_copy.place_word(move["word"], move["row"], move["col"],
                                  move["direction"], is_ai=True)
            evaluation = minimax(board_copy, depth-1, alpha, beta, False)
            max_eval = max(max_eval, evaluation)
            alpha = max(alpha, evaluation)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = float('inf')
        for move in possible_moves:
            board_copy = deepcopy(board)
            board_copy.place_word(move["word"], move["row"], move["col"],
                                  move["direction"], is_ai=False)
            evaluation = minimax(board_copy, depth-1, alpha, beta, True)
            min_eval = min(min_eval, evaluation)
            beta = min(beta, evaluation)
            if beta <= alpha:
                break
        return min_eval


def generate_possible_moves(board: Board) -> List[Dict]:
    """Generate all possible valid moves"""
    moves = []
    relevant_words = get_relevant_words(board)

    for word in relevant_words:
        for row in range(board.size):
            for col in range(board.size):
                for direction in ["horizontal", "vertical"]:
                    if board.validate_move(word, row, col, direction, "ai"):
                        moves.append({
                            "word": word,
                            "row": row,
                            "col": col,
                            "direction": direction
                        })
    return moves


def get_relevant_words(board: Board) -> List[str]:
    """Get words that can be formed with current rack"""
    # In a real implementation, this would use the AI's rack letters
    # and a dictionary to find possible words
    # For now, return a small subset for testing
    return ["WORD", "TEST", "SCRABBLE", "AI", "PLAY"]


def evaluate_position(board: Board) -> int:
    """Evaluate board position with multiple heuristics"""
    score_diff = board.scores["ai"] - board.scores["human"]

    # Add bonuses for board control
    center_control = calculate_center_control(board)
    premium_control = calculate_premium_control(board)

    # Bonus for long words
    word_length_bonus = sum(len(word)
                            for word in get_relevant_words(board)) / 10

    return score_diff + center_control * 5 + premium_control * 3 + word_length_bonus


def calculate_center_control(board: Board) -> int:
    """Calculate control of center squares"""
    center_squares = [(7, 7), (7, 8), (8, 7), (8, 8)]
    control = 0
    for r, c in center_squares:
        if board.board[r][c] is not None:
            control += 1 if board.board[r][c] == "ai" else -1
    return control


def calculate_premium_control(board: Board) -> int:
    """Calculate control of premium squares"""
    premium_control = 0
    for (r, c), square_type in board.premium_squares.items():
        if board.board[r][c] is not None:
            value = 2 if square_type in ['DW', 'TW'] else 1
            premium_control += value if board.board[r][c] == "ai" else -value
    return premium_control
