import os
import subprocess
from loguru import logger
from typing import Optional
def extract_audio(video_id: str, dataset: str = "FakeSV") -> Optional[str]:
    try:
        logger.info(f"Extracting audio for video_id: {video_id} from dataset: {dataset}")
        
        
        if dataset == "FakeSV":
            video_path = f"./data/FakeSV/videos/{video_id}.mp4"
            audio_dir = f"./data/FakeSV/audio"
        elif dataset == "FakeTT":
            video_path = f"./data/FakeTT/videos/{video_id}.mp4"
            audio_dir = f"./data/FakeTT/audio"
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")
        
        
        if not os.path.exists(video_path):
            logger.error(f"Video file not found: {video_path}")
            return None
        
        
        os.makedirs(audio_dir, exist_ok=True)
        
        
        audio_path = os.path.join(audio_dir, f"{video_id}.mp3")
        
        
        if os.path.exists(audio_path):
            logger.info(f"Audio file already exists: {audio_path}")
            return audio_path
        
        
        command = [
            'ffmpeg',
            '-i', video_path,
            '-vn',  
            '-acodec', 'libmp3lame',  
            '-ar', '44100',  
            '-ac', '2',  
            '-ab', '192k',  
            '-y',  
            audio_path
        ]
        
        
        result = subprocess.run(command, capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info(f"Audio extraction successful: {audio_path}")
            return audio_path
        else:
            logger.error(f"Audio extraction failed: {result.stderr}")
            return None
            
    except Exception as e:
        logger.error(f"Error extracting audio from video {video_id}: {str(e)}")
        return None