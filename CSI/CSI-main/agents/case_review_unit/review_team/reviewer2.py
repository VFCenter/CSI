import os
import sys
from typing import List, Dict
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from util import ask_gemini
class Reviewer2:
    def __init__(self, temperature=1.0):
        self.temperature = temperature
    
    def analyze(self, case_file: str, video_id: str, discussion_history: List[Dict] = None) -> str:
        try:
            prompt = " "
            response = ask_gemini(prompt_text=prompt, temperature=self.temperature)
            return response
        except Exception as e:
            return f"Error: {str(e)}"
    
    def analyze_with_meeting_log(self, case_file: str, video_id: str, meeting_log: str) -> str:
        try:
            prompt = " "
            response = ask_gemini(prompt_text=prompt, temperature=self.temperature)
            return response
        except Exception as e:
            return f"Error: {str(e)}"

