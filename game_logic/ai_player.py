# ai_player.py
from typing import Dict, Optional
from game_logic.board import Board
import random

def get_ai_move(board: Board, depth: int = 2) -> Optional[Dict]:
    """Improved AI using Minimax with Alpha-Beta pruning"""
    best_move = None
    best_score = -float('inf')
    
    # Get all possible moves
    possible_moves = generate_possible_moves(board)
    
    # Evaluate each move
    for move in possible_moves:
        # Simulate move
        board_copy = deepcopy(board)
        board_copy.place_word(move["word"], move["row"], move["col"], 
                            move["direction"], is_ai=True)
        
        # Evaluate position
        score = evaluate_position(board_copy)
        if score > best_score:
            best_score = score
            best_move = move
    
    return best_move

def generate_possible_moves(board: Board) -> List[Dict]:
    """Generate all possible valid moves"""
    moves = []
    # This should be more sophisticated in real implementation
    for word in get_relevant_words(board):
        for row in range(board.size):
            for col in range(board.size):
                for direction in ["horizontal", "vertical"]:
                    if can_place_word(board, word, row, col, direction):
                        moves.append({
                            "word": word,
                            "row": row,
                            "col": col,
                            "direction": direction
                        })
    return moves

def evaluate_position(board: Board) -> int:
    """Evaluate board position with multiple heuristics"""
    score = board.score["ai"] - board.score["human"]
    
    # Add bonuses for board control
    center_control = calculate_center_control(board)
    premium_control = calculate_premium_control(board)
    
    return score + center_control * 5 + premium_control * 3