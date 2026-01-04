import os
import time
import argparse
import json
from multiprocessing import Pool
from functools import partial
from loguru import logger
import traceback
from util import (
    save_csi_result, 
    get_unprocessed_videos, 
    set_csi_environment_variables,
    read_json_file
)
from csi_framework import execute_csi_multi_agent
def process_single_video(dataset, max_rounds, video_data):
    video_id, json_data = video_data
    
    try:
        logger.info(f"Processing video_id: {video_id}")
        logger.debug(f"JSON data: {json_data}")
        set_csi_environment_variables(dataset, video_id)
        status, results, agent_prompts = execute_csi_multi_agent(
            video_id=video_id,
            dataset=dataset,
            use_async=False,
            max_rounds=max_rounds
        )
        if status == 0:
            results_file = f"results_{dataset}_csi.json"
            save_csi_result(
                file_path=results_file,
                video_id=video_id,
                agent_prompts=agent_prompts,
                agent_response=results,
                analysis_result=results,
                save_backup=False
            )
            
            logger.info(f"Successfully processed video {video_id}")
            logger.info(f"Analysis generated: {bool(results.get('analysis'))}")
            logger.info(f"Analysis length: {len(results.get('analysis', ''))}")
            
            return True
        else:
            logger.error(f"CSI analysis failed for video {video_id}")
            return False
            
    except Exception as e:
        logger.error(f"Error processing video {video_id}: {e}")
        logger.error(traceback.format_exc())
        time.sleep(1)
        return False
def get_csi_unprocessed_videos(dataset: str, max_items: int = 1000):
    try:
        dataset_file = f"data/{dataset}/{dataset.lower()}.json"
        
        if not os.path.exists(dataset_file):
            logger.error(f"Dataset file not found: {dataset_file}")
            return []
        
        try:
            data = read_json_file(dataset_file)
            
            if isinstance(data, dict):
                unprocessed_videos = []
                count = 0
                for video_id, json_data in data.items():
                    if count >= max_items:
                        break
                    if "analysis_result" not in json_data or json_data.get("analysis_result") is None:
                        unprocessed_videos.append((video_id, json_data))
                        count += 1
                return unprocessed_videos
            
            elif isinstance(data, list):
                unprocessed_videos = []
                count = 0
                for item in data:
                    if count >= max_items:
                        break
                    video_id = item.get('video_id')
                    if video_id and ("analysis_result" not in item or item.get("analysis_result") is None):
                        unprocessed_videos.append((video_id, item))
                        count += 1
                return unprocessed_videos
                
        except Exception as e:
            logger.warning(f"Failed to read as JSON object, trying line-by-line: {e}")
        
        unprocessed_videos = []
        count = 0
        
        with open(dataset_file, 'r', encoding='utf-8') as f:
            for line in f:
                if count >= max_items:
                    break
                try:
                    json_data = json.loads(line.strip())
                    video_id = json_data.get('video_id')
                    if video_id and ("analysis_result" not in json_data or json_data.get("analysis_result") is None):
                        unprocessed_videos.append((video_id, json_data))
                        count += 1
                except json.JSONDecodeError:
                    continue
        
        logger.info(f"Found {len(unprocessed_videos)} unprocessed videos in {dataset}")
        return unprocessed_videos
        
    except Exception as e:
        logger.error(f"Error reading dataset file: {e}")
        return []
def main():
    parser = argparse.ArgumentParser(description="CSI Multi-Agent Video Analysis System")
    parser.add_argument('--dataset', type=str, required=True,
                       help="Dataset name (e.g., FakeTT, FakeSV)")
    parser.add_argument('--num_workers', type=int, default=1,
                       help="Number of worker processes (default: 1)")
    parser.add_argument('--max_items', type=int, default=10,
                       help="Maximum number of videos to process (default: all)")
    parser.add_argument('--max_rounds', type=int, default=3,
                       help="Maximum case review rounds (default: 3)")
    parser.add_argument('--log_level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help="Logging level (default: INFO)")
    
    args = parser.parse_args()
    logger.remove()
    logger.add(
        f"logs/csi_{args.dataset}_{time.strftime('%Y%m%d_%H%M%S')}.log",
        level=args.log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
        rotation="100 MB"
    )
    logger.add(
        lambda msg: print(msg, end=''),
        level=args.log_level,
        format="{time:HH:mm:ss} | {level} | {message}"
    )
    os.environ["DATASET"] = args.dataset
    
    dataset_path = f"data/{args.dataset}"
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset directory not found: {dataset_path}")
        return
    
    logger.info(f"Starting CSI processing for dataset: {args.dataset}")
    logger.info(f"Max workers: {args.num_workers}")
    logger.info(f"Max items: {args.max_items}")
    logger.info(f"Max case review rounds: {args.max_rounds}")
    unprocessed_videos = get_csi_unprocessed_videos(args.dataset, max_items=args.max_items)
    
    if not unprocessed_videos:
        logger.info("No unprocessed videos found")
        return
    
    logger.info(f"Found {len(unprocessed_videos)} unprocessed videos")
    num_workers = min(args.num_workers, len(unprocessed_videos))
    
    logger.info(f"Starting processing with {num_workers} workers")
    
    start_time = time.time()
    
    if num_workers == 1:
        logger.info("Running in single-threaded mode")
        results = []
        for video_data in unprocessed_videos:
            result = process_single_video(args.dataset, args.max_rounds, video_data)
            results.append(result)
    else:
        logger.info(f"Running in multi-process mode with {num_workers} workers")
        
        with Pool(num_workers) as pool:
            process_func = partial(process_single_video, args.dataset, args.max_rounds)
            
            results = pool.map(process_func, unprocessed_videos)
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    successful = sum(1 for r in results if r)
    failed = len(results) - successful
    
    logger.info(f"\n{'='*50}")
    logger.info(f"CSI Processing Complete")
    logger.info(f"{'='*50}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Total videos processed: {len(results)}")
    logger.info(f"Successfully processed: {successful}")
    logger.info(f"Failed to process: {failed}")
    logger.info(f"Success rate: {successful/len(results)*100:.1f}%")
    logger.info(f"Total processing time: {processing_time:.1f} seconds")
    logger.info(f"Average time per video: {processing_time/len(results):.1f} seconds")
    logger.info(f"Results saved to: results_{args.dataset}_csi.json")
if __name__ == "__main__":
    main()