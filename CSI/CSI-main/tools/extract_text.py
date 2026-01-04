import os
import json
from typing import Optional, Dict, Any
from loguru import logger
def extract_text(video_id: str, dataset: str = "FakeSV") -> str:
    try:
        logger.info(f"Extracting text for video_id: {video_id} from dataset: {dataset}")
        
        
        if dataset == "FakeSV":
            data_path = "./data/FakeSV/fake_sv.json"
            text_fields = ["title", "ocr"]
        elif dataset == "FakeTT":
            data_path = "./data/FakeTT/fake_tt.json"
            text_fields = ["description", "ocr"]
        else:
            logger.error(f"Unsupported dataset: {dataset}")
            return f"Error: Unsupported dataset {dataset}"
        
        
        if not os.path.exists(data_path):
            logger.error(f"Dataset file not found: {data_path}")
            return f"Error: Dataset file not found for {dataset}"
        
        
        video_data = None
        with open(data_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    if data.get('video_id') == video_id:
                        video_data = data
                        logger.info(f"Found video data at line {line_num}")
                        break
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse JSON at line {line_num}: {e}")
                    continue
        
        if not video_data:
            logger.warning(f"Video {video_id} not found in {dataset} dataset")
            return f"Video {video_id} not found in {dataset} dataset"
        
        
        extracted_texts = []
        
        for field in text_fields:
            field_content = video_data.get(field)
            if field_content:
                
                cleaned_text = clean_text_content(field_content)
                if cleaned_text.strip():
                    extracted_texts.append(f"{field.upper()}: {cleaned_text}")
                    logger.debug(f"Extracted {field}: {len(cleaned_text)} characters")
            else:
                logger.debug(f"Field '{field}' is empty or missing")
        
        
        if extracted_texts:
            combined_text = "\n\n".join(extracted_texts)
            logger.info(f"Successfully extracted text content: {len(combined_text)} total characters")
            return combined_text
        else:
            logger.warning(f"No text content found for video {video_id}")
            return f"No text content found for video {video_id} in {dataset} dataset"
        
    except Exception as e:
        logger.error(f"Error extracting text for video {video_id}: {str(e)}")
        return f"Error extracting text for video {video_id}: {str(e)}"
def clean_text_content(text: str) -> str:
    try:
        if not isinstance(text, str):
            text = str(text)
        
        
        cleaned = ' '.join(text.split())
        
        
        cleaned = cleaned.replace('\t', ' ')
        cleaned = cleaned.replace('\r', ' ')
        cleaned = cleaned.replace('\n', ' ')
        
        
        while '  ' in cleaned:
            cleaned = cleaned.replace('  ', ' ')
        
        return cleaned.strip()
        
    except Exception as e:
        logger.warning(f"Text cleaning failed: {e}")
        return str(text) if text else ""
def get_video_metadata(video_id: str, dataset: str = "FakeSV") -> Optional[Dict[str, Any]]:
    try:
        logger.info(f"Getting metadata for video_id: {video_id} from dataset: {dataset}")
        
        
        if dataset == "FakeSV":
            data_path = "./data/FakeSV/fake_sv.json"
        elif dataset == "FakeTT":
            data_path = "./data/FakeTT/fake_tt.json"
        else:
            logger.error(f"Unsupported dataset: {dataset}")
            return None
        
        
        if not os.path.exists(data_path):
            logger.error(f"Dataset file not found: {data_path}")
            return None
        
        
        with open(data_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    if data.get('video_id') == video_id:
                        logger.info(f"Found video metadata at line {line_num}")
                        return data
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse JSON at line {line_num}: {e}")
                    continue
        
        logger.warning(f"Video {video_id} not found in {dataset} dataset")
        return None
        
    except Exception as e:
        logger.error(f"Error getting video metadata: {str(e)}")
        return None