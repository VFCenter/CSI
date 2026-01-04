import os
import json
import math
from typing import Optional
from loguru import logger
def get_shots(video_id: str, dataset: str = "FakeSV") -> str:
    try:
        logger.info(f"Getting shot boundaries for video_id: {video_id} from dataset: {dataset}")
        
        
        if dataset == "FakeSV":
            shots_path = "./data/FakeSV/shots.json"
        elif dataset == "FakeTT":
            shots_path = "./data/FakeTT/shots.json"
        else:
            logger.error(f"Unsupported dataset: {dataset}")
            return f"Error: Unsupported dataset {dataset}"
        
        
        if not os.path.exists(shots_path):
            logger.error(f"Shots file not found: {shots_path}")
            return f"Error: Shots file not found for dataset {dataset}"
        
        video_data = None
        with open(shots_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                if data.get('video_id') == video_id:
                    video_data = data
                    break
        
        if not video_data:
            logger.warning(f"Video data not found for video_id: {video_id}")
            return f"Video {video_id} not found in {dataset} dataset"
        
        
        fps = video_data.get('fps', 30.0)
        frame_count = video_data.get('frame_count', 0)
        transnetv2_segs = video_data.get('transnetv2_segs', [])
        
        logger.info(f"Video properties - FPS: {fps}, Frame count: {frame_count}, Segments: {len(transnetv2_segs)}")
        
        
        shot_descriptions = []
        total_duration = math.ceil(frame_count / fps)
        
        for shot_idx, (start_frame, end_frame) in enumerate(transnetv2_segs):
            
            start_second = math.floor(start_frame / fps)
            end_second = math.ceil(end_frame / fps)
            
            
            end_second = min(end_second, total_duration)
            
            
            shot_duration = end_second - start_second
            
            shot_descriptions.append(
                f"Shot {shot_idx + 1}: seconds {start_second}-{end_second} "
                f"({shot_duration}s duration, frames {start_frame}-{end_frame})"
            )
            
            logger.debug(f"Shot {shot_idx + 1}: {start_second}s-{end_second}s ({start_frame}-{end_frame} frames)")
        
        
        result_description = (
            f"Video {video_id} shot analysis: "
            f"Total duration {total_duration} seconds with {len(transnetv2_segs)} shots. "
            f"Shot boundaries: {'; '.join(shot_descriptions)}. "
            f"Each second corresponds to approximately {fps} frames at {fps} FPS."
        )
        
        logger.info(f"Successfully processed shot boundaries for video {video_id}")
        return result_description
        
    except Exception as e:
        logger.error(f"Error processing shot boundaries for video {video_id}: {str(e)}")
        return f"Error processing shot boundaries for video {video_id}: {str(e)}"