import os
from loguru import logger
from dotenv import load_dotenv
from csi_MFUpy import execute_acquisition_flow
from csi_CRU import execute_deliberation_flow
from util import set_csi_environment_variables
load_dotenv()
def execute_csi_acquisition_flow(video_id: str, dataset: str = "FakeTT", use_async: bool = False):
    set_csi_environment_variables(dataset, video_id)
    
    return execute_acquisition_flow(video_id, dataset)
def execute_csi_deliberation_flow(case_file: str, video_id: str, dataset: str = "FakeTT", max_rounds: int = 3):
    set_csi_environment_variables(dataset, video_id)
    
    return execute_deliberation_flow(case_file, video_id, dataset, max_rounds)
def execute_csi_multi_agent(video_id: str, dataset: str = "FakeTT", use_async: bool = False, max_rounds: int = 3):
    logger.info(f"Starting complete CSI multi-agent analysis: {video_id}")
    
    try:
        set_csi_environment_variables(dataset, video_id)
        logger.info("Starting Multimodal Forensics Unit...")
        acq_status, acq_results, acq_prompts = execute_csi_acquisition_flow(video_id, dataset, use_async)
        
        if acq_status != 0:
            logger.error("Multimodal Forensics Unit failed")
            return -1, {}, {}
        
        case_file = acq_results.get("C", "")
        if not case_file:
            logger.error("No case file generated from Multimodal Forensics Unit")
            return -1, acq_results, acq_prompts
        
        logger.info("Starting Case Review Unit...")
        delib_results = execute_csi_deliberation_flow(case_file, video_id, dataset, max_rounds)
        
        final_results = {
            **acq_results,
            **delib_results
        }
        
        agent_prompts = {
            "multimodal_forensics_unit": "Multimodal Forensics Unit agents executed with new Gemini API",
            "case_review_unit": "Case Review Unit agents executed with new Gemini API"
        }
        
        logger.info(f"Complete CSI analysis finished: {video_id}")
        return 0, final_results, agent_prompts
            
    except Exception as e:
        logger.error(f"CSI multi-agent execution failed: {e}")
        return -1, {}, {}
def process_multimodal_forensics_only(input_data, use_async: bool = False):
    video_id = input_data.get('video_id', 'unknown')
    dataset = input_data.get('dataset', 'FakeTT')
    
    status, results, prompts = execute_csi_acquisition_flow(video_id, dataset, use_async)
    return results if status == 0 else {}
def process_review_team_only(input_data, use_async: bool = False):
    video_id = input_data.get('video_id', 'unknown')
    dataset = input_data.get('dataset', 'FakeTT')
    
    case_file = f"""
Video ID: {video_id}
Dataset: {dataset}
Original Content: {input_data.get('original_text', '')}
Visual Analysis: {input_data.get('vision_analysis', '')}
Acoustic Analysis: {input_data.get('audio_analysis', '')}
Intelligence Analysis: {input_data.get('text_analysis', '')}
Analysis needed for Review Team review.
"""
    results = execute_csi_deliberation_flow(case_file, video_id, dataset)
    return results