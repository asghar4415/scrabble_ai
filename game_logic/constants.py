
LETTER_SCORES = {
    'A': 1, 'B': 3, 'C': 3, 'D': 2, 'E': 1, 'F': 4, 'G': 2, 'H': 4, 'I': 1, 'J': 8,
    'K': 5, 'L': 1, 'M': 3, 'N': 1, 'O': 1, 'P': 3, 'Q': 10, 'R': 1, 'S': 1, 'T': 1,
    'U': 1, 'V': 4, 'W': 4, 'X': 8, 'Y': 4, 'Z': 10, ' ': 0,
}

POWER_TILE_DOUBLE_TURN_MARKER = 'D*'
POWER_TILE_TYPES = {
    POWER_TILE_DOUBLE_TURN_MARKER: {"effect": "double_turn", "display": "D"}
}

OBJECTIVE_TYPES = [
    {"id": "score_gt_30", "desc": "Score 30+ points in a single turn", "bonus": 20},
    {"id": "use_q_z_x_j", "desc": "Play a word using Q, Z, X, or J", "bonus": 15},
    {"id": "form_7_letter",
        "desc": "Form a 7-letter word (Bingo already gives 50)", "bonus": 25},
    {"id": "use_corner",
        "desc": "Play a tile on a corner square (0,0/0,14/14,0/14,14)", "bonus": 10},
]

MAX_RACK_PERMUTATION_LENGTH = 5
AI_THINKING_TIME_LIMIT_SECONDS = 7.0
