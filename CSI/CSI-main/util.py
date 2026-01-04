import os
import time
import json
import random
import glob
import base64
from retry import retry
from filelock import FileLock
from mimetypes import guess_type
import numpy as np
from google import genai
from google.genai import types
from loguru import logger
def local_image_to_data_url(image_path):
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = 'application/octet-stream'
    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')
    return f"data:{mime_type};base64,{base64_encoded_data}"
@retry(tries=3, delay=3)
def ask_gemini(prompt_text: str, video_path: str = "", image_paths: list = None, temperature=0.7) -> str:
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    
    model_name = "gemini-2.0-flash"
    
    generation_config = types.GenerateContentConfig(
        max_output_tokens=3000,
        temperature=temperature,
        seed=42,
        safety_settings=[
            types.SafetySetting(
                category='HARM_CATEGORY_HATE_SPEECH',
                threshold='BLOCK_NONE'
            ),
            types.SafetySetting(
                category='HARM_CATEGORY_HARASSMENT',
                threshold='BLOCK_NONE'
            ),
            types.SafetySetting(
                category='HARM_CATEGORY_SEXUALLY_EXPLICIT',
                threshold='BLOCK_NONE'
            ),
            types.SafetySetting(
                category='HARM_CATEGORY_DANGEROUS_CONTENT',
                threshold='BLOCK_NONE'
            ),
            types.SafetySetting(
                category='HARM_CATEGORY_CIVIC_INTEGRITY',
                threshold='BLOCK_NONE'
            ),
        ]
    )
    
    contents = []
    
    if video_path:
        cache_file_path = "csi_gemini_video_cache.json"
        video_key = os.path.basename(os.path.splitext(video_path)[0])
        video_file_name = None
        
        lock = FileLock(f"{cache_file_path}.lock")
        
        with lock:
            if os.path.exists(cache_file_path):
                try:
                    with open(cache_file_path, 'r') as f:
                        video_cache = json.load(f)
                    if video_key in video_cache:
                        video_file_name = video_cache[video_key]
                    elif len(video_cache) >= 1000:
                        logger.info("Cache limit reached. Clearing video cache...")
                        video_files_to_delete = list(video_cache.values())
                        import concurrent.futures
                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            delete_futures = [
                                executor.submit(client.files.delete, name=file_name)
                                for file_name in video_files_to_delete
                            ]
                            concurrent.futures.wait(delete_futures)
                        
                        video_cache = {}
                        with open(cache_file_path, 'w') as f:
                            json.dump(video_cache, f)
                        logger.info("Cache cleared.")
                except json.JSONDecodeError:
                    video_cache = {}
    else:
                video_cache = {}
        
        if video_file_name:
            try:
                video_file = client.files.get(name=video_file_name)
                logger.info(f'Found cached video: {video_file.name}')
        except Exception as e:
                logger.warning(f'Error retrieving cached video: {e}')
                video_file_name = None
        
        if not video_file_name:
            logger.info(f"Uploading video: {video_path}")
            video_file = client.files.upload(file=video_path)
            
            with lock:
                if os.path.exists(cache_file_path):
                    try:
                        with open(cache_file_path, 'r') as f:
                            video_cache = json.load(f)
                    except json.JSONDecodeError:
                        video_cache = {}
                
                video_cache[video_key] = video_file.name
                with open(cache_file_path, 'w') as f:
                    json.dump(video_cache, f)
                logger.info(f'Added video to cache: {video_key} -> {video_file.name}')
        while video_file.state == "PROCESSING":
            logger.info('Waiting for video to be processed.')
            time.sleep(10)
            video_file = client.files.get(name=video_file.name)
        if video_file.state == "FAILED":
            raise ValueError(f"Video processing failed: {video_file.state}")
        logger.info(f'Video processing complete: {video_file.uri}')
        
        contents.append(video_file)
    
    if image_paths:
        for image_path in image_paths:
            if os.path.exists(image_path):
                image_file = client.files.upload(file=image_path)
                contents.append(image_file)
                logger.info(f"Added image: {image_path}")
    
    contents.append(prompt_text)
    
    response = client.models.generate_content(
        model=model_name,
        contents=contents,
        config=generation_config
    )
    
    logger.info(f"ask_gemini: prompt_tokens={response.usage_metadata.prompt_token_count}, "
                f"completion_tokens={response.usage_metadata.candidates_token_count}, "
                f"total_tokens={response.usage_metadata.total_token_count}")
    
    return response.text

