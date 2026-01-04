from agents.multimodal_forensics_unit.visual_analyst import VisualAnalyst
from agents.multimodal_forensics_unit.acoustic_analyst import AcousticAnalyst
from agents.multimodal_forensics_unit.intelligence_analyst import IntelligenceAnalyst
from util import read_json_file, set_csi_environment_variables
import os
import json
import concurrent.futures

def load_csi_data(video_id: str, dataset: str = "FakeTT"):
    try:
        data_file = f"data/{dataset}/{dataset.lower()}.json"
        
        if os.path.exists(data_file):
            data = read_json_file(data_file)
            
            if isinstance(data, dict) and video_id in data:
                video_data = data[video_id]
                video_data['dataset'] = dataset
                video_data['video_id'] = video_id
                return video_data
            
            elif isinstance(data, list):
                for item in data:
                    if item.get('video_id') == video_id:
                        item['dataset'] = dataset
                        return item
        
        if os.path.exists(data_file):
            with open(data_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        video_data = json.loads(line.strip())
                        if video_data.get('video_id') == video_id:
                            video_data['dataset'] = dataset
                            return video_data
                    except json.JSONDecodeError:
                        continue
        
        return {
            'video_id': video_id,
            'dataset': dataset,
            'original_text': f'No data found for video {video_id}',
            'label': 'unknown'
        }
    except Exception as e:
        return {}

def execute_acquisition_flow(video_id: str, dataset: str = "FakeTT"):
    try:
        input_data = load_csi_data(video_id, dataset)
        if not input_data:
            return -1, {}, {}
        
        set_csi_environment_variables(dataset, video_id)
        
        vision_agent = VisualAnalyst(temperature=0.0)
        audio_agent = AcousticAnalyst(temperature=0.0) 
        text_agent = IntelligenceAnalyst(temperature=0.0)
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_vision = executor.submit(vision_agent.analyze, video_id, input_data)
            future_audio = executor.submit(audio_agent.analyze, video_id, input_data)
            future_text = executor.submit(text_agent.analyze, video_id, input_data)
            
            P_vision = future_vision.result()
            P_audio = future_audio.result()
            text_result = future_text.result()
        
        T_c = f"Structured analysis of: {input_data.get('original_text', '')[:100]}..."
        E = text_result
        
        C = f"""
=== CSI Case File ===
Video ID: {video_id}
Dataset: {dataset}
Original Text: {input_data.get('original_text', '')}
=== Visual Analysis (P_vision) ===
{P_vision}
=== Acoustic Analysis (P_audio) ===  
{P_audio}
=== Structured Title (T_c) ===
{T_c}
=== Official Reports (E) ===
{E}
"""
        
        results = {
            "P_vision": P_vision,
            "P_audio": P_audio,
            "T_c": T_c,
            "E": E,
            "C": C
        }
        
        agent_prompts = {}
        
        return 0, results, agent_prompts
        
    except Exception as e:
        return -1, {}, {}