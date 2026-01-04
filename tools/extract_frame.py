import os
import cv2
import json
from typing import List
from loguru import logger
import numpy as np
def extract_frame(video_id: str, dataset: str = "FakeSV") -> List[np.ndarray]:
    try:
        logger.info(f"Extracting frames for video_id: {video_id} from dataset: {dataset}")
        
        
        if dataset == "FakeSV":
            video_path = f"./data/FakeSV/videos/{video_id}.mp4"
        elif dataset == "FakeTT":
            video_path = f"./data/FakeTT/videos/{video_id}.mp4"
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")
        
        
        if not os.path.exists(video_path):
            logger.error(f"Video file not found: {video_path}")
            return []
        
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return []
        
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        logger.info(f"Video properties - FPS: {fps}, Total frames: {total_frames}, Duration: {duration:.2f}s")
        
        frames = []
        frame_interval = int(fps)  
        
        for second in range(int(duration) + 1):
            frame_number = second * frame_interval
            
            
            if frame_number >= total_frames:
                break
            
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            
            
            ret, frame = cap.read()
            if ret:
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
                logger.debug(f"Extracted frame at second {second} (frame {frame_number})")
            else:
                logger.warning(f"Failed to read frame at second {second} (frame {frame_number})")
        
        cap.release()
        
        logger.info(f"Successfully extracted {len(frames)} frames from video {video_id}")
        return frames
        
    except Exception as e:
        logger.error(f"Error extracting frames from video {video_id}: {str(e)}")
        return []