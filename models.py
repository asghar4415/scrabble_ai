
from pydantic import BaseModel
from typing import Dict, List, Optional, Any


class MoveRequest(BaseModel):
    word: str
    row: int
    col: int
    direction: str


class GameStateResponse(BaseModel):
    board: List[List[Optional[str]]]
    scores: Dict[str, int]
    current_player: str
    player_rack: List[str]
    game_over: bool
    message: Optional[str] = None
    first_move: bool
    tiles_in_bag: int
    human_objective: Optional[Dict[str, Any]] = None
