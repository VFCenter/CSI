import os
import sys
import shutil
from typing import Dict, Any
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from util import ask_gemini, prepare_video_path, extract_frames_from_video
from tools.casefile_manager import update_multimodal_forensics_results
class VisualAnalyst:
    def __init__(self, temperature=0.0):
        self.temperature = temperature
    
    def analyze(self, video_id: str, input_data: Dict[str, Any]) -> str:
        try:
            dataset = input_data.get('dataset', 'FakeTT')
            prompt = " "
            video_path = prepare_video_path(video_id, dataset)
            
            if os.path.exists(video_path):
                response = ask_gemini(
                    prompt_text=prompt,
                    video_path=video_path,
                    temperature=self.temperature
                )
            else:
                frames_dir = f"temp_frames_{video_id}"
                os.makedirs(frames_dir, exist_ok=True)
                try:
                    frame_paths = extract_frames_from_video(video_path, frames_dir, num_frames=10)
                    if frame_paths:
                        response = ask_gemini(
                            prompt_text=prompt,
                            image_paths=frame_paths,
                            temperature=self.temperature
                        )
                        shutil.rmtree(frames_dir, ignore_errors=True)
                    else:
                        response = ""
                except Exception:
                    response = ask_gemini(
                        prompt_text=prompt,
                        temperature=self.temperature
                    )
            
            update_multimodal_forensics_results(video_id, vision_result=response)
            return response
        except Exception as e:
            error_result = f"Error: {str(e)}"
            update_multimodal_forensics_results(video_id, vision_result=error_result)
            return error_result