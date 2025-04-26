# board.py
from typing import Dict, List, Tuple, Optional
import random
from copy import deepcopy
from .game_utils import is_valid_word, calculate_word_score,LETTER_SCORES

class Board:
    def __init__(self):
        self.size = 15
        self.board = [[None for _ in range(self.size)] for _ in range(self.size)]
        self.scores = {"human": 0, "ai": 0}
        self.tile_bag = self.initialize_tile_bag()
        self.player_racks = {
            "human": self.draw_tiles(7),
            "ai": self.draw_tiles(7)
        }
        self.current_player = "human"
        self.game_over = False
        self.first_move = True
        self.premium_squares = self.initialize_premium_squares()
    
    def initialize_tile_bag(self):
        """Create tile bag with standard Scrabble distribution"""
        distribution = {
            'A': 9, 'B': 2, 'C': 2, 'D': 4, 'E': 12, 'F': 2, 'G': 3, 'H': 2,
            'I': 9, 'J': 1, 'K': 1, 'L': 4, 'M': 2, 'N': 6, 'O': 8, 'P': 2,
            'Q': 1, 'R': 6, 'S': 4, 'T': 6, 'U': 4, 'V': 2, 'W': 2, 'X': 1,
            'Y': 2, 'Z': 1, ' ': 2  # Blank tiles
        }
        tile_bag = []
        for letter, count in distribution.items():
            tile_bag.extend([letter] * count)
        random.shuffle(tile_bag)
        return tile_bag
    
    def draw_tiles(self, count: int) -> List[str]:
        """Draw tiles from the bag"""
        tiles = []
        for _ in range(count):
            if self.tile_bag:
                tiles.append(self.tile_bag.pop())
            else:
                self.game_over = True
        return tiles
    def reset(self):
        """Reset the game board to initial state"""
        self.board = [[None for _ in range(self.size)] for _ in range(self.size)]
        self.scores = {"human": 0, "ai": 0}
        self.tile_bag = self.initialize_tile_bag()
        self.player_racks = {
            "human": self.draw_tiles(7),
            "ai": self.draw_tiles(7)
        }
        self.current_player = "human"
        self.game_over = False
        self.first_move = True
    
    def initialize_premium_squares(self):
        """Set up premium squares (TW, DW, TL, DL)"""
        premium_squares = {
            (0, 0): 'TW', (0, 7): 'TW', (0, 14): 'TW',
            (7, 0): 'TW', (7, 14): 'TW', (14, 0): 'TW',
            (14, 7): 'TW', (14, 14): 'TW',
            # Add all standard Scrabble premium squares
        }
        return premium_squares
    
    def place_word(self, word: str, row: int, col: int, 
                  direction: str, is_ai: bool) -> bool:
        """Place a word on the board with full validation"""
        player = "ai" if is_ai else "human"
        
        if not self.validate_move(word, row, col, direction, player):
            return False
        
        # Place the word
        score = 0
        word_multiplier = 1
        letters_used = []
        
        for i, letter in enumerate(word):
            r = row + (i if direction == "vertical" else 0)
            c = col + (i if direction == "horizontal" else 0)
            
            # Only apply premium squares for new tiles
            if self.board[r][c] is None:
                square = self.premium_squares.get((r, c))
                if square == 'DL':
                    score += LETTER_SCORES[letter] * 1
                elif square == 'TL':
                    score += LETTER_SCORES[letter] * 2
                elif square == 'DW':
                    word_multiplier *= 2
                elif square == 'TW':
                    word_multiplier *= 3
                else:
                    score += LETTER_SCORES[letter]
                
                self.board[r][c] = letter
                letters_used.append(letter)
        
        score *= word_multiplier
        self.scores[player] += score
        
        # Remove used letters from rack and draw new ones
        self.remove_from_rack(letters_used, player)
        self.refill_rack(player)
        
        self.switch_turn()
        self.first_move = False
        return True
    
    def validate_move(self, word: str, row: int, col: int, 
                     direction: str, player: str) -> bool:
        """Validate all aspects of a move"""
        # Check word validity
        if not is_valid_word(word):
            return False
        
        # Check bounds
        if direction == "horizontal":
            if col + len(word) > self.size:
                return False
        else:
            if row + len(word) > self.size:
                return False
        
        # Check player has the letters
        if not self.has_letters(word, player):
            return False
        
        # Check word connects with existing words (unless first move)
        if not self.first_move and not self.check_connections(word, row, col, direction):
            return False
        
        # Check all newly formed words are valid
        if not self.check_new_words(word, row, col, direction):
            return False
            
        return True
    
    def has_letters(self, word: str, player: str) -> bool:
        """Check if player has the required letters"""
        rack = self.player_racks[player]
        word_letters = list(word)
        
        for letter in word_letters:
            if letter in rack:
                rack.remove(letter)
            elif ' ' in rack:  # Blank tile
                rack.remove(' ')
            else:
                return False
        return True
    
    def check_connections(self, word: str, row: int, col: int, 
                         direction: str) -> bool:
        """Check if word connects with existing words"""
        for i in range(len(word)):
            r = row + (i if direction == "vertical" else 0)
            c = col + (i if direction == "horizontal" else 0)
            
            # Check if adjacent to existing tiles
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.size and 0 <= nc < self.size:
                    if self.board[nr][nc] is not None:
                        return True
        return False
    
    def check_new_words(self, word: str, row: int, col: int, 
                       direction: str) -> bool:
        """Check all newly formed words are valid"""
        # This should check perpendicular words formed by the placement
        # Implementation omitted for brevity but should:
        # 1. Find all new words formed
        # 2. Check each is valid using is_valid_word
        return True
    
    def remove_from_rack(self, letters: List[str], player: str):
        """Remove used letters from player's rack"""
        for letter in letters:
            if letter in self.player_racks[player]:
                self.player_racks[player].remove(letter)
            else:  # Blank tile
                self.player_racks[player].remove(' ')
    
    def refill_rack(self, player: str):
        """Refill player's rack to 7 tiles"""
        needed = 7 - len(self.player_racks[player])
        if needed > 0:
            new_tiles = self.draw_tiles(needed)
            self.player_racks[player].extend(new_tiles)
    
    def switch_turn(self):
        """Switch current player"""
        self.current_player = "ai" if self.current_player == "human" else "human"
    
    def get_state(self):
        """Return current board state"""
        return {
            "board": self.board,
            "scores": self.scores,
            "current_player": self.current_player,
            "racks": {
                "human": self.player_racks["human"],
                "ai": self.player_racks["ai"]
            }
        }