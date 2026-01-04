import os
import json
import re
import time
from typing import Dict, Any, List, Optional
from loguru import logger
import requests
from urllib.parse import quote_plus
from datetime import datetime
from tools.extract_text import extract_text
def calculate_days_ago(timestamp_ms: int) -> int:
    try:
        video_time = datetime.fromtimestamp(timestamp_ms / 1000)
        current_time = datetime.now()
        days_ago = (current_time - video_time).days
        return max(0, days_ago)
    except Exception as e:
        logger.error(f"Error calculating days ago: {e}")
        return None
def search_google_custom(query: str, num_results: int = 10, video_publish_time: Optional[int] = None) -> List[Dict[str, Any]]:
    try:
        api_key = os.getenv("GOOGLE_SEARCH_API_KEY")
        search_engine_id = os.getenv("GOOGLE_SEARCH_ENGINE_ID")
        
        if not api_key or not search_engine_id:
            logger.warning("Google Search API configuration missing")
            return []
        
        base_url = "https://www.googleapis.com/customsearch/v1"
        params = {
            'key': api_key,
            'cx': search_engine_id,
            'q': query,
            'num': min(num_results, 10),
            'safe': 'active',
            'fields': 'items(title,link,snippet,displayLink,pagemap)'
        }
        
        response = requests.get(base_url, params=params, timeout=30)
        response.raise_for_status()
        
        search_data = response.json()
        results = []
        
        if 'items' in search_data:
            for item in search_data['items']:
                publish_time = extract_publish_time(item)
                
                result = {
                    'title': item.get('title', ''),
                    'url': item.get('link', ''),
                    'content': item.get('snippet', ''),
                    'publisher': item.get('displayLink', ''),
                    'publish_time': publish_time
                }
                results.append(result)
        
        logger.info(f"Google search returned {len(results)} results")
        return results
        
    except Exception as e:
        logger.error(f"Google search failed: {e}")
        return []
def extract_publish_time(item: Dict[str, Any]) -> Optional[int]:
    try:
        pagemap = item.get('pagemap', {})
        sources = [
            pagemap.get('metatags', [{}])[0].get('article:published_time'),
            pagemap.get('metatags', [{}])[0].get('pubdate'),
            pagemap.get('metatags', [{}])[0].get('date'),
            pagemap.get('newsarticle', [{}])[0].get('datepublished'),
            pagemap.get('article', [{}])[0].get('datepublished')
        ]
        
        for date_str in sources:
            if date_str:
                timestamp = parse_date_string(date_str)
                if timestamp:
                    return timestamp
        
        return None
        
    except Exception as e:
        logger.debug(f"Failed to extract publish time: {e}")
        return None
def parse_date_string(date_str: str) -> Optional[int]:
    try:
        formats = [
            '%Y-%m-%dT%H:%M:%S%z',
            '%Y-%m-%dT%H:%M:%SZ',
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%d',
            '%Y/%m/%d',
            '%d/%m/%Y',
            '%m/%d/%Y'
        ]
        
        for fmt in formats:
            try:
                dt = datetime.strptime(date_str.replace('Z', '+0000'), fmt)
                return int(dt.timestamp() * 1000)
            except ValueError:
                continue
        
        return None
        
    except Exception:
        return None
def get_video_publish_time(video_id: str, dataset: str = "FakeSV") -> Optional[int]:
    try:
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
            for line in f:
                data = json.loads(line.strip())
                if data.get('video_id') == video_id:
                    publish_time = data.get('publish_time') or data.get('publish_time_norm')
                    return publish_time
        
        logger.warning(f"Video {video_id} not found in dataset {dataset}")
        return None
        
    except Exception as e:
        logger.error(f"Error getting video publish time: {e}")
        return None
def validate_publish_time(search_results: List[Dict[str, Any]], video_publish_time: int) -> List[Dict[str, Any]]:
    try:
        valid_results = []
        
        for result in search_results:
            result_publish_time = result.get('publish_time')
            
            if result_publish_time is None:
                result['time_validation'] = 'unknown'
                valid_results.append(result)
            elif result_publish_time < video_publish_time:
                result['time_validation'] = 'valid'
                valid_results.append(result)
            else:
                result['time_validation'] = 'invalid'
                logger.info(f"Filtered out result published after video: {result.get('title', '')}")
        
        logger.info(f"Time validation: {len(valid_results)}/{len(search_results)} results are valid")
        return valid_results
        
    except Exception as e:
        logger.error(f"Time validation failed: {e}")
        return search_results
def retrieve_google_search(query: str, video_id: str = "", dataset: str = "FakeSV") -> str:
    try:
        logger.info(f"Starting Google search retrieval for query: {query}")
        video_publish_time = None
        if video_id:
            video_publish_time = get_video_publish_time(video_id, dataset)
            if video_publish_time:
                logger.info(f"Video publish time: {video_publish_time}")
        search_results = search_google_custom(query, num_results=10, video_publish_time=video_publish_time)
        
        if not search_results:
            return json.dumps({
                "query": query,
                "video_id": video_id,
                "dataset": dataset,
                "results": [],
                "message": "No search results found"
            })
        if video_publish_time:
            search_results = validate_publish_time(search_results, video_publish_time)
        unique_results = remove_duplicate_results(search_results)
        formatted_results = []
        for result in unique_results:
            formatted_result = {
                'title': result.get('title', ''),
                'url': result.get('url', ''),
                'content': result.get('content', ''),
                'publisher': result.get('publisher', ''),
                'publish_time': result.get('publish_time'),
                'time_validation': result.get('time_validation', 'unknown')
            }
            formatted_results.append(formatted_result)
        
        response_data = {
            "query": query,
            "video_id": video_id,
            "dataset": dataset,
            "video_publish_time": video_publish_time,
            "total_results": len(formatted_results),
            "results": formatted_results
        }
        
        logger.info(f"Google search retrieval completed: {len(formatted_results)} results")
        return json.dumps(response_data, ensure_ascii=False, indent=2)
        
    except Exception as e:
        logger.error(f"Google search retrieval failed: {e}")
        return json.dumps({
            "query": query,
            "video_id": video_id,
            "dataset": dataset,
            "error": str(e),
            "results": []
        })
def remove_duplicate_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    try:
        seen_urls = set()
        unique_results = []
        
        for result in results:
            url = result.get('url', '')
            if url not in seen_urls:
                seen_urls.add(url)
                unique_results.append(result)
        
        logger.info(f"After deduplication, {len(unique_results)} results remain")
        return unique_results
        
    except Exception as e:
        logger.error(f"Deduplication failed: {e}")
        return results
def retrieve_related_texts(video_id: str, dataset: str = "FakeTT") -> str:
    try:
        extracted_text = extract_text(video_id, dataset)
        if "Error:" in extracted_text:
            return extracted_text
        
        query = extracted_text[:200] if len(extracted_text) > 200 else extracted_text
        search_results = retrieve_google_search(query, video_id, dataset)
        
        return search_results
        
    except Exception as e:
        logger.error(f"Error retrieving related texts: {e}")
        return f"Error retrieving related texts: {str(e)}"
retrieve_google_search.name = "retrieve_google_search"
retrieve_google_search.description = " "