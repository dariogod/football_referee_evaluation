from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import os
import uuid
import logging
from typing import Literal
import shutil
from pathlib import Path

from src.player_tracker import PlayerTracker

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Football Referee Evaluation API")

@app.post("/track")
async def track_players(
    video_path: str,
    model: Literal["yolo", "d-fine"] = "yolo",
    store_predictions: bool = False,
):
    """
    Track players in a football video using YOLO or D-FINE model.
    
    Parameters:
    - video_path: Path to an MP4 video file inside the data directory
    - model: Model to use for tracking (yolo or d-fine)
    - store_predictions: Whether to store predictions in the same folder as the input video
    
    Returns:
    - JSON with tracking results
    """
    logger.info(f"Starting tracking for video: {video_path}")

    # Convert to Path object for easier handling
    video_path = Path(video_path)
    
    # Validate file exists
    if not video_path.exists():
        raise HTTPException(status_code=400, detail=f"File not found: {video_path}. Make sure the file is inside the data directory.")
    
    # Validate file type
    if not video_path.name.lower().endswith(".mp4"):
        raise HTTPException(status_code=400, detail="Only MP4 files are supported")
        
    # Set up results folder based on store_predictions
    results_folder: str | None = None
    if store_predictions:
        # Store in the same folder as the input video
        results_folder = str(Path("data/predictions") / video_path.relative_to(Path("data")).parent)
    
    try:
        # Process video with selected model
        player_tracker = PlayerTracker(underlying_model=model)
        detections = player_tracker.track_players(
            input_path=str(video_path),
            intermediate_results_folder=results_folder
        )
        
        # Convert to JSON-serializable format
        results = [frame.model_dump() for frame in detections]
            
        # Prepare the response
        response_data = {
            "results": results, 
            "output_path": results_folder
        }
        
        return JSONResponse(content=response_data)
    
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Football Referee Evaluation API is running"} 