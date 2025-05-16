import os
import cv2
import glob
from tqdm import tqdm

def create_video_from_frames(input_folder: str, output_file: str, fps: int = 25) -> None:

    frame_files = glob.glob(os.path.join(input_folder, '*.jpg'))
    if not frame_files:
        frame_files = glob.glob(os.path.join(input_folder, '*.png'))
    
    if not frame_files:
        msg = f"No image files found in {input_folder}"
        raise ValueError(msg)
    
    if os.path.exists(output_file):
        os.remove(output_file)
    
    frame_files.sort()
    
    first_frame = cv2.imread(frame_files[0])
    height, width, _ = first_frame.shape
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
    
    for frame_file in tqdm(frame_files, desc=f"Converting frames to video for {input_folder}"):
        frame = cv2.imread(frame_file)
        video_writer.write(frame)
    
    video_writer.release()

    if not os.path.exists(output_file):
        msg = "Something went wrong while creating the video"
        raise Exception(msg)
