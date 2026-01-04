import os
import sys
from typing import Dict, Any
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from util import ask_gemini, prepare_video_path
from tools.extract_audio import extract_audio
from tools.casefile_manager import update_multimodal_forensics_results
class AcousticAnalyst:
    def __init__(self, temperature=0.0):
        self.temperature = temperature
    
    def analyze(self, video_id: str, input_data: Dict[str, Any]) -> str:
        try:
            dataset = input_data.get('dataset', 'FakeTT')
            extract_audio(video_id, dataset)
            prompt = " "
            video_path = prepare_video_path(video_id, dataset)
            
            if os.path.exists(video_path):
                response = ask_gemini(
                    prompt_text=prompt,
                    video_path=video_path,
                    temperature=self.temperature
                )
            else:
                response = ask_gemini(
                    prompt_text=prompt,
                    temperature=self.temperature
                )
            
            update_multimodal_forensics_results(video_id, audio_result=response)
            return response
        except Exception as e:
            error_result = f"Error: {str(e)}"
            update_multimodal_forensics_results(video_id, audio_result=error_result)
            return error_result