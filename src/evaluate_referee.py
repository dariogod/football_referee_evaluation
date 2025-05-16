from src.player_tracker import PlayerTracker
from src.utils.frames_to_mp4 import create_video_from_frames
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main(video_path: str, intermediate_results_folder: str):
    player_tracker = PlayerTracker(underlying_model="yolo")
    detections = player_tracker.track_players(video_path, intermediate_results_folder)
    return detections

if __name__ == "__main__":
    dataset_split = "test"
    video_id = "SNGS-130"

    frames_folder = f"data/SoccerNet/SN-GSR-2025/{dataset_split}/{video_id}/img1"
    video_path = f"data/SoccerNet/SN-GSR-2025/{dataset_split}/{video_id}/{video_id}.mp4"
    fps = 25

    try:
        create_video_from_frames(frames_folder, video_path, fps)
    except Exception as e:
        msg = f"Error creating video: {e}"
        logger.error(msg)
        raise Exception(msg)

    main(
        video_path=video_path,
        intermediate_results_folder=f"data/predictions/{dataset_split}/{video_id}"
    )