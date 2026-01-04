import os
import json
import time
from datetime import datetime
from typing import Dict, Any, List, Optional
from loguru import logger
class CaseFileManager:
    
    def __init__(self, case_files_dir: str = "case_files"):
        self.case_files_dir = case_files_dir
        self.ensure_directory()
        
    def ensure_directory(self):
        if not os.path.exists(self.case_files_dir):
            os.makedirs(self.case_files_dir)
            logger.info(f"Created case files directory: {self.case_files_dir}")
    
    def get_case_file_path(self, video_id: str) -> str:
        return os.path.join(self.case_files_dir, f"{video_id}_case.json")
    
    def create_case_file(self, video_id: str, dataset: str = "FakeTT") -> Dict[str, Any]:
        case_file = {
            "video_id": video_id,
            "dataset": dataset,
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "multimodal_forensics_unit": {
                "vision_analysis": None,
                "audio_analysis": None, 
                "text_analysis": None,
                "official_reports": None,
                "completed": False,
                "timestamp": None
            },
            "case_review_unit": {
                "review_team_discussion": None,
                "reasoning_results": None,
                "final_verdict": None,
                "analysis_summary": None,
                "completed": False,
                "timestamp": None
            },
            "status": "initialized"
        }
        
        
        self.save_case_file(video_id, case_file)
        logger.info(f"Created case file for video: {video_id}")
        
        return case_file
    
    def load_case_file(self, video_id: str) -> Optional[Dict[str, Any]]:
        case_file_path = self.get_case_file_path(video_id)
        
        if not os.path.exists(case_file_path):
            logger.warning(f"Case file not found: {case_file_path}")
            return None
            
        try:
            with open(case_file_path, 'r', encoding='utf-8') as f:
                case_file = json.load(f)
                logger.info(f"Loaded case file for video: {video_id}")
                return case_file
        except Exception as e:
            logger.error(f"Failed to load case file {case_file_path}: {e}")
            return None
    
    def save_case_file(self, video_id: str, case_file: Dict[str, Any]):
        case_file_path = self.get_case_file_path(video_id)
        case_file["last_updated"] = datetime.now().isoformat()
        
        try:
            with open(case_file_path, 'w', encoding='utf-8') as f:
                json.dump(case_file, f, indent=2, ensure_ascii=False)
                logger.info(f"Saved case file for video: {video_id}")
        except Exception as e:
            logger.error(f"Failed to save case file {case_file_path}: {e}")
            raise
    
    def update_multimodal_forensics_results(self, 
                                 video_id: str, 
                                 vision_result: str = None,
                                 audio_result: str = None, 
                                 text_result: str = None,
                                 official_reports: str = None) -> Dict[str, Any]:
        case_file = self.load_case_file(video_id)
        
        if case_file is None:
            logger.warning(f"Case file not found for {video_id}, creating new one")
            case_file = self.create_case_file(video_id)
        
        
        if vision_result is not None:
            case_file["multimodal_forensics_unit"]["vision_analysis"] = vision_result
        if audio_result is not None:
            case_file["multimodal_forensics_unit"]["audio_analysis"] = audio_result
        if text_result is not None:
            case_file["multimodal_forensics_unit"]["text_analysis"] = text_result
        if official_reports is not None:
            case_file["multimodal_forensics_unit"]["official_reports"] = official_reports
        
        
        acquisition = case_file["multimodal_forensics_unit"]
        if all([acquisition["vision_analysis"], 
                acquisition["audio_analysis"], 
                acquisition["text_analysis"], 
                acquisition["official_reports"]]):
            acquisition["completed"] = True
            acquisition["timestamp"] = datetime.now().isoformat()
            case_file["status"] = "multimodal_forensics_completed"
        
        self.save_case_file(video_id, case_file)
        logger.info(f"Updated Multimodal Forensics Unit results for video: {video_id}")
        return case_file
    
    def get_case_for_review(self, video_id: str) -> Optional[Dict[str, Any]]:
        case_file = self.load_case_file(video_id)
        
        if case_file is None:
            logger.error(f"Case file not found for video: {video_id}")
            return None
        
        if not case_file["multimodal_forensics_unit"]["completed"]:
            logger.warning(f"Multimodal Forensics Unit not completed for video: {video_id}")
            return None
        
        
        deliberation_input = {
            "video_id": video_id,
            "dataset": case_file["dataset"],
            "evidence": {
                "vision_analysis": case_file["multimodal_forensics_unit"]["vision_analysis"],
                "audio_analysis": case_file["multimodal_forensics_unit"]["audio_analysis"], 
                "text_analysis": case_file["multimodal_forensics_unit"]["text_analysis"],
                "official_reports": case_file["multimodal_forensics_unit"]["official_reports"]
            },
            "multimodal_forensics_timestamp": case_file["multimodal_forensics_unit"]["timestamp"]
        }
        
        logger.info(f"Prepared case for Case Review Unit: {video_id}")
        return deliberation_input
    
    def update_case_review_results(self,
                                  video_id: str,
                                  review_team_discussion: str = None,
                                  reasoning_results: List[Dict] = None,
                                  final_verdict: str = None,
                                  analysis_summary: str = None) -> Dict[str, Any]:
        case_file = self.load_case_file(video_id)
        
        if case_file is None:
            logger.error(f"Case file not found for video: {video_id}")
            return None
        
        
        if review_team_discussion is not None:
            case_file["case_review_unit"]["review_team_discussion"] = review_team_discussion
        if reasoning_results is not None:
            case_file["case_review_unit"]["reasoning_results"] = reasoning_results
        if final_verdict is not None:
            case_file["case_review_unit"]["final_verdict"] = final_verdict
        if analysis_summary is not None:
            case_file["case_review_unit"]["analysis_summary"] = analysis_summary
        
        
        deliberation = case_file["case_review_unit"]
        if all([deliberation["review_team_discussion"],
                deliberation["reasoning_results"],
                deliberation["final_verdict"]]):
            deliberation["completed"] = True
            deliberation["timestamp"] = datetime.now().isoformat()
            case_file["status"] = "completed"
        
        self.save_case_file(video_id, case_file)
        logger.info(f"Updated Case Review Unit results for video: {video_id}")
        
        return case_file
    
    def get_all_cases(self) -> List[Dict[str, Any]]:
        cases = []
        
        if not os.path.exists(self.case_files_dir):
            return cases
        
        for filename in os.listdir(self.case_files_dir):
            if filename.endswith("_case.json"):
                video_id = filename.replace("_case.json", "")
                case_file = self.load_case_file(video_id)
                if case_file:
                    cases.append(case_file)
        
        logger.info(f"Retrieved {len(cases)} case files")
        return cases
    
    def get_cases_by_status(self, status: str) -> List[Dict[str, Any]]:
        all_cases = self.get_all_cases()
        filtered_cases = [case for case in all_cases if case.get("status") == status]
        
        logger.info(f"Found {len(filtered_cases)} cases with status: {status}")
        return filtered_cases
def create_case_file(video_id: str, dataset: str = "FakeTT") -> Dict[str, Any]:
    manager = CaseFileManager()
    return manager.create_case_file(video_id, dataset)
def update_multimodal_forensics_results(video_id: str, 
                             vision_result: str = None,
                             audio_result: str = None,
                             text_result: str = None, 
                             official_reports: str = None) -> Dict[str, Any]:
    manager = CaseFileManager()
    return manager.update_multimodal_forensics_results(video_id, vision_result, audio_result, text_result, official_reports)
def get_case_for_review(video_id: str) -> Optional[Dict[str, Any]]:
    manager = CaseFileManager()
    return manager.get_case_for_review(video_id)
def update_case_review_results(video_id: str, 
                              review_team_discussion: str = None,
                              reasoning_results: List[Dict] = None,
                              final_verdict: str = None,
                              analysis_summary: str = None) -> Dict[str, Any]:
    manager = CaseFileManager()
    return manager.update_case_review_results(video_id, review_team_discussion, reasoning_results, final_verdict, analysis_summary)