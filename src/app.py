from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import logging
from typing import Literal
from pathlib import Path
from pydantic import BaseModel
import asyncio
from src.player_tracker import YoloPlayerTracker, DFinePlayerTracker
from src.color_assigner import ColorAssigner
from src.utils.custom_types import FrameDetections


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Football Referee Evaluation API")

class TrackRequest(BaseModel):
    video_path: str
    model: Literal["yolo", "dfine"]
    results_path: str | None = None

class ColorAssignRequest(BaseModel):
    video_path: str
    detections: list[FrameDetections]
    results_path: str | None = None

@app.get("/")
async def root():
    return {"message": "Football Referee Evaluation API is running"} 

@app.on_event("startup")
async def initialize_models():
    global yolo_player_tracker
    global d_fine_player_tracker
    yolo_player_tracker = YoloPlayerTracker()
    d_fine_player_tracker = DFinePlayerTracker()

    global color_assigner
    color_assigner = ColorAssigner()

def detections_json(detections: list[FrameDetections]) -> list[dict]:
    return [frame.model_dump() for frame in detections]

def validate_video_path(video_path: str) -> None:
    video_path_obj = Path(video_path)
    if not str(video_path_obj).startswith("data/"):
        raise HTTPException(status_code=400, detail="Video path must start with 'data/'")
    if not video_path_obj.exists():
        raise HTTPException(status_code=400, detail=f"File not found: {video_path}. Make sure the file is inside the data directory.")
    if not video_path_obj.name.lower().endswith(".mp4"):
        raise HTTPException(status_code=400, detail="Only MP4 files are supported")
    
def validate_results_path(results_path: str) -> None:
    results_path_obj = Path(results_path)
    if not str(results_path_obj).startswith("data/"):
        raise HTTPException(status_code=400, detail="Results path must start with 'data/'")


@app.post("/track")
async def track_players(request: TrackRequest) -> JSONResponse:
    logger.info(f"Starting tracking for video: {request.video_path}")

    validate_video_path(request.video_path)
    if request.results_path is not None:
        validate_results_path(request.results_path)

    match request.model:
        case "yolo":
            player_tracker = yolo_player_tracker
        case "dfine":
            player_tracker = d_fine_player_tracker
        case _:
            raise HTTPException(status_code=400, detail="Invalid model")
    
    # Process video with selected model in a separate thread to avoid blocking
    detections = await asyncio.to_thread(
        player_tracker.track_players,
        input_path=request.video_path,
        intermediate_results_folder=request.results_path if request.results_path is not None else None
    )

    return JSONResponse(
        content={
            "detections": detections_json(detections),
        }
    )

@app.post("/assign-colors")
async def assign_colors(request: ColorAssignRequest) -> JSONResponse:
    logger.info(f"Starting color assignment for video: {request.video_path}")

    validate_video_path(request.video_path)
    if request.results_path is not None:
        validate_results_path(request.results_path)

    # Process video with color assigner in a separate thread to avoid blocking
    detections = await asyncio.to_thread(
        color_assigner.process_video,
        input_path=request.video_path,
        detections=request.detections,
        intermediate_results_folder=request.results_path if request.results_path is not None else None
    )

    return JSONResponse(
        content={
            "results": detections_json(detections),
        }
    )
    
    
    