# board.py
from typing import Dict, List, Tuple
import random

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
        return tiles
    
    def initialize_premium_squares(self):
        """Set up premium squares (TW, DW, TL, DL)"""
        premium_squares = {}
        # Add standard Scrabble premium squares here
        return premium_squares
    
    def place_word(self, word: str, row: int, col: int, 
                  direction: str, is_ai: bool) -> bool:
        """Enhanced word placement with full validation"""
        # Add comprehensive validation:
        # 1. Word connects with existing words (except first move)
        # 2. All newly formed words are valid
        # 3. Player has required tiles
        # 4. Proper scoring with premium squares
        
        # After successful placement:
        self.switch_turn()
        self.refill_rack(is_ai)
        return True