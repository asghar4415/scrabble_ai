from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional, Tuple
import random
from copy import deepcopy
import enchant

app = FastAPI()

try:
    english_dict = enchant.Dict("en_US")
except enchant.errors.DictNotFoundError:
    english_dict = enchant.DictWithPWL("en_US", "custom_words.txt")


# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# --------------------------
# Game Constants and Utilities
# --------------------------

LETTER_SCORES = {
    'A': 1, 'B': 3, 'C': 3, 'D': 2, 'E': 1,
    'F': 4, 'G': 2, 'H': 4, 'I': 1, 'J': 8,
    'K': 5, 'L': 1, 'M': 3, 'N': 1, 'O': 1,
    'P': 3, 'Q': 10, 'R': 1, 'S': 1, 'T': 1,
    'U': 1, 'V': 4, 'W': 4, 'X': 8, 'Y': 4,
    'Z': 10, ' ': 0  # Blank tile
}

# Simple word dictionary - replace with a proper dictionary in production
# VALID_WORDS = {
#     "DAD", "MOM", "ADD", "CAT", "DOG", "HOUSE", "COMPUTER", "PHONE", "WATER", "FIRE",
#     "EARTH", "AIR", "LOVE", "HATE", "PEACE", "WAR", "BOOK",
#     "MEN", "PAPER", "CHAIR", "TABLE", "DOOR", "WINDOW", "PEN"
#     "QI", "ZA", "AX", "EX", "JO", "OX", "XI", "XU"
# }


def is_valid_word(word: str) -> bool:
    """Check if word exists in dictionary using pyenchant"""
    # Minimum 2 letters for Scrabble rules
    if len(word) < 2:
        return False

    # Check if word contains only letters
    if not word.isalpha():
        return False

    # Standard dictionary check
    return english_dict.check(word.upper())


def calculate_word_score(word: str, letter_multipliers: List[int] = None,
                         word_multiplier: int = 1) -> int:
    """Calculate score for a word with multipliers"""
    if letter_multipliers is None:
        letter_multipliers = [1] * len(word)

    score = 0
    for i, letter in enumerate(word.upper()):
        letter_score = LETTER_SCORES.get(letter, 0)
        score += letter_score * letter_multipliers[i]

    return score * word_multiplier
# --------------------------
# Board and Game Logic
# --------------------------


class Board:
    def __init__(self):
        self.size = 15
        self.board = [[None for _ in range(self.size)]
                      for _ in range(self.size)]
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

    def initialize_premium_squares(self):
        """Set up premium squares (TW, DW, TL, DL)"""
        premium_squares = {
            # Triple Word
            (0, 0): 'TW', (0, 7): 'TW', (0, 14): 'TW',
            (7, 0): 'TW', (7, 14): 'TW', (14, 0): 'TW',
            (14, 7): 'TW', (14, 14): 'TW',
            # Double Word
            (1, 1): 'DW', (2, 2): 'DW', (3, 3): 'DW', (4, 4): 'DW',
            (10, 10): 'DW', (11, 11): 'DW', (12, 12): 'DW', (13, 13): 'DW',
            # Triple Letter
            (1, 5): 'TL', (1, 9): 'TL', (5, 1): 'TL', (5, 5): 'TL',
            (5, 9): 'TL', (5, 13): 'TL', (9, 1): 'TL', (9, 5): 'TL',
            (9, 9): 'TL', (9, 13): 'TL', (13, 5): 'TL', (13, 9): 'TL',
            # Double Letter
            (0, 3): 'DL', (0, 11): 'DL', (2, 6): 'DL', (2, 8): 'DL',
            (3, 0): 'DL', (3, 7): 'DL', (3, 14): 'DL', (6, 2): 'DL',
            (6, 6): 'DL', (6, 8): 'DL', (6, 12): 'DL', (7, 3): 'DL',
            (7, 11): 'DL', (8, 2): 'DL', (8, 6): 'DL', (8, 8): 'DL',
            (8, 12): 'DL', (11, 0): 'DL', (11, 7): 'DL', (11, 14): 'DL',
            (12, 6): 'DL', (12, 8): 'DL', (14, 3): 'DL', (14, 11): 'DL'
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
                    score += LETTER_SCORES[letter] * 2
                elif square == 'TL':
                    score += LETTER_SCORES[letter] * 3
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
        """Enhanced validation"""
        # Check word validity
        if not is_valid_word(word):
            return False

        # Check bounds
        word_len = len(word)
        if direction == "horizontal":
            if col + word_len > self.size:
                return False
        else:
            if row + word_len > self.size:
                return False

        # Check first move goes through center
        if self.first_move:
            center = self.size // 2
            if direction == "horizontal":
                if not (row == center and col <= center < col + word_len):
                    return False
            else:
                if not (col == center and row <= center < row + word_len):
                    return False

    # Create a copy of the rack for validation
        rack_copy = self.player_racks[player].copy()
        if not self._has_letters(word, rack_copy):
            return False

        if not self.first_move and not self.check_connections(word, row, col, direction):
            return False

        return True

    def _has_letters(self, word: str, rack: List[str]) -> bool:
        """Helper method that doesn't modify the rack"""
        for letter in word:
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
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.size and 0 <= nc < self.size:
                    if self.board[nr][nc] is not None:
                        return True
        return False

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
            },
            "game_over": self.game_over
        }

# --------------------------
# AI Player Logic
# --------------------------


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
    rack_letters = board.player_racks["ai"]
    possible_words = []

    # This is simplified - in production you'd use a proper word dictionary
    for word in VALID_WORDS:
        if can_form_word(word, rack_letters):
            possible_words.append(word)

    return possible_words


def can_form_word(word: str, rack: List[str]) -> bool:
    """Check if word can be formed from rack letters"""
    rack_copy = rack.copy()
    for letter in word.upper():
        if letter in rack_copy:
            rack_copy.remove(letter)
        elif ' ' in rack_copy:  # Blank tile
            rack_copy.remove(' ')
        else:
            return False
    return True


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

# --------------------------
# API Models and Endpoints
# --------------------------


class MoveRequest(BaseModel):
    word: str
    row: int
    col: int
    direction: str  # "horizontal" or "vertical"


class GameStateResponse(BaseModel):
    board: List[List[Optional[str]]]
    scores: Dict[str, int]
    current_player: str
    player_rack: List[str]
    ai_rack: List[str]
    game_over: bool
    message: Optional[str] = None


# Global game board instance
game_board = Board()


@app.get("/api/game/start", response_model=GameStateResponse)
def start_game():
    """Initialize a new game"""
    global game_board
    game_board = Board()
    return get_game_state("Game started!")


@app.post("/api/game/move", response_model=GameStateResponse)
def player_move(move: MoveRequest):
    """Process a player move and AI response"""
    if game_board.current_player != "human":
        raise HTTPException(status_code=400, detail="Not your turn")

    # Enhanced error messages
    if not is_valid_word(move.word):
        raise HTTPException(
            status_code=400,
            detail=f"'{move.word}' is not a valid Scrabble word"
        )

    if not game_board.has_letters(move.word, "human"):
        raise HTTPException(
            status_code=400,
            detail="You don't have the required letters for this word"
        )

    if not game_board.validate_move(move.word, move.row, move.col, move.direction, "human"):
        raise HTTPException(
            status_code=400,
            detail="Invalid move position - must connect with existing words"
        )

    if not game_board.place_word(move.word, move.row, move.col, move.direction, is_ai=False):
        raise HTTPException(
            status_code=400, detail="Move failed - unknown reason")

    # AI move
    ai_move = get_ai_move(game_board)
    if ai_move:
        game_board.place_word(ai_move["word"], ai_move["row"], ai_move["col"],
                              ai_move["direction"], is_ai=True)

    return get_game_state(f"AI played {ai_move['word'] if ai_move else 'passed'}")


@app.get("/api/game/state", response_model=GameStateResponse)
def get_current_state():
    """Get current game state"""
    return get_game_state()


def get_game_state(message: str = None):
    """Helper function to format game state response"""
    state = game_board.get_state()
    return {
        "board": state["board"],
        "scores": state["scores"],
        "current_player": state["current_player"],
        "player_rack": state["racks"]["human"],
        "ai_rack": state["racks"]["ai"],
        "game_over": game_board.game_over,
        "message": message
    }

# Run the app with: uvicorn scrabble_api:app --reload
