from pydantic import BaseModel
from typing import List

class MoveRequest(BaseModel):
    word: str
    row: int
    col: int
    direction: str  # "horizontal" or "vertical"

class MoveResponse(BaseModel):
    player_board: list
    ai_word: str
    ai_row: int
    ai_col: int
    ai_direction: str

class AIMove(BaseModel):
    word: str
    row: int
    col: int
    direction: str

class AIvsAIResponse(BaseModel):
    final_board: list
    moves: List[AIMove]
