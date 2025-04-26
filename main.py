from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from game_logic.board import Board
from game_logic.ai_player import get_ai_move
import json



app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Your Vite frontend
        "http://127.0.0.1:5173",  # Alternative localhost
        "http://localhost:8000",  # For testing
    ],
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
    expose_headers=["*"]  # Expose all headers
)

board = Board()

class MoveRequest(BaseModel):
    word: str
    row: int
    col: int
    direction: str  # "horizontal" or "vertical"

class GameStateResponse(BaseModel):
    board: List[List[Optional[str]]]
    scores: dict
    current_player: str
    player_rack: List[str]
    ai_rack: List[str]
    game_over: bool
    message: Optional[str] = None

@app.get("/api/game/start", response_model=GameStateResponse)
def start_game():
    # Create a new board instance instead of trying to reset
    global board
    board = Board()  # This will automatically initialize a fresh game
    return get_game_state("Game started!")

@app.post("/api/game/move", response_model=GameStateResponse)
def player_move(move: MoveRequest):
    if board.current_player != "human":
        raise HTTPException(status_code=400, detail="Not your turn")
    
    if not board.place_word(move.word, move.row, move.col, move.direction, is_ai=False):
        raise HTTPException(status_code=400, detail="Invalid move")
    
    # AI move
    ai_move = get_ai_move(board)
    if ai_move:
        board.place_word(ai_move["word"], ai_move["row"], ai_move["col"], 
                        ai_move["direction"], is_ai=True)
    
    return get_game_state(f"AI played {ai_move['word'] if ai_move else 'passed'}")

@app.get("/api/game/state", response_model=GameStateResponse)
def get_current_state():
    return get_game_state()

def get_game_state(message: str = None):
    state = board.get_state()
    return {
        "board": state["board"],
        "scores": state["scores"],
        "current_player": state["current_player"],
        "player_rack": state["racks"]["human"],
        "ai_rack": state["racks"]["ai"],
        "game_over": board.game_over,
        "message": message
    }