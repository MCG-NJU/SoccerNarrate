import os
import json
import tqdm
import argparse
import multiprocessing
from functools import partial
from typing import Dict, List, Any, Optional
from demo.infer import LiveCCDemoInfer
from utils.multiprocessor import local_mp


def build_video_path(video_root: str, league: str, season: str, match: str, half: str) -> str:
    """Build the video file path"""
    return os.path.join(video_root, league, season, match, f"{half}_720p.mkv")


def load_test_json(test_json_path: str) -> Dict[str, Any]:
    """Load the test.json file"""
    with open(test_json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def flatten_to_halfs(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Flatten nested JSON into a list of halves
    Each half contains: league, season, match, half, segments
    This keeps all segments from the same half on one worker in order
    """
    halfs = []
    for league, seasons in data.items():
        for season, matches in seasons.items():
            for match, halfs_dict in matches.items():
                for half, segments in halfs_dict.items():
                    halfs.append({
                        "league": league,
                        "season": season,
                        "match": match,
                        "half": half,
                        "segments": segments,
                    })
    return halfs


def livecc_worker(
    device_id: int,
    model_name_or_path: str,
    test_json_path: str,
    video_root: str,
    output_json_path: str,
    simple_ctx: bool,
    repetition_penalty: float,
    num_workers: int,
):
    """LiveCC worker function that handles data on the selected GPU."""
    # Load the model
    infer = LiveCCDemoInfer(model_path=model_name_or_path, device=f'cuda:{device_id}')
    
    # Load test data
    test_data = load_test_json(test_json_path)
    
    # Flatten data into a list of halves（group by half so all segments from the same half stay on one worker）
    all_halfs = flatten_to_halfs(test_data)
    
    # Assign halves to this device
    half_idxs = list(range(len(all_halfs)))
    half_idxs_on_device = half_idxs[device_id::num_workers]
    
    # Store results while preserving the original structure
    results = {}
    
    # Show progress with tqdm
    pbar = tqdm.tqdm(
        half_idxs_on_device,
        desc=f"Device {device_id}",
        position=device_id,
        leave=True
    )
    
    total_segments_processed = 0
    
    for half_idx in pbar:
        half_data = all_halfs[half_idx]
        league = half_data["league"]
        season = half_data["season"]
        match = half_data["match"]
        half = half_data["half"]
        segments = half_data["segments"]
        
        # Build the video path
        video_path = build_video_path(video_root, league, season, match, half)
        if not os.path.exists(video_path):
            pbar.set_postfix({"skipped": "video_missing"})
            continue
        
        # Initialize the result list for this half
        if league not in results:
            results[league] = {}
        if season not in results[league]:
            results[league][season] = {}
        if match not in results[league][season]:
            results[league][season][match] = {}
        if half not in results[league][season][match]:
            results[league][season][match][half] = []
        
        # Store generated commentary for this half to build preasr
        preasr_parts = []
        
        # Process all segments in this half in order
        for seg_idx, segment in enumerate(segments):
            # Build preasr by concatenating commentary from previous segments
            preasr = ' '.join(preasr_parts).strip() if preasr_parts else ''
            
            # Build the prompt
            if simple_ctx:
                overall_prompt = preasr.strip() if preasr else "Commentate on this match."
            else:
                commentary_prompt = (
                    "You are an expert video commentator providing real-time, insightful, "
                    "and engaging commentary on visual content.\n"
                )
                overall_prompt = commentary_prompt
                if preasr:
                    overall_prompt += f"Here is previous commentary of the video:\n\n{preasr}\n\n"
                    overall_prompt += "Please continue to comment the video."
                else:
                    overall_prompt += "Please comment on this video."
            
            # Run inference
            try:
                responses = infer.live_cc_once_for_evaluation(
                    query=overall_prompt,
                    video=video_path,
                    video_start=segment["start"],
                    video_end=segment["end"],
                    max_new_tokens=32,
                    repetition_penalty=repetition_penalty,
                )
                
                # Merge responses
                overall_cc = (
                    ' '.join(cc.replace(' ...', '') for _, _, cc in responses if cc)
                    .strip()
                )
                
                # Use a default value when empty
                if not overall_cc:
                    overall_cc = "..."
                else:
                    overall_cc += "..."
                
                # Update preasr_parts for the next segment
                preasr_parts.append(overall_cc)
                
                # Create a result segment while preserving the test.json structure
                result_segment = {
                    "start": segment["start"],
                    "end": segment["end"],
                    "commentary": overall_cc,
                    "class": segment.get("class", "1")  # Preserve the class field
                }
                results[league][season][match][half].append(result_segment)
                
                total_segments_processed += 1
                
            except Exception as e:
                print(f"\n[Device {device_id}] {league}/{season}/{match} {half} segment {seg_idx} failed: {e}", flush=True)
                # Even on failure, save the original segment structure with empty commentary
                result_segment = {
                    "start": segment["start"],
                    "end": segment["end"],
                    "commentary": "",
                    "class": segment.get("class", "1")
                }
                results[league][season][match][half].append(result_segment)
                # Also append an empty string to preasr_parts on failure
                preasr_parts.append("")
                continue
        
        pbar.set_postfix({"processed_segments": total_segments_processed})
    
    pbar.close()
    
    return results


def merge_results(all_results: List[Dict[str, Any]], test_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge all worker results while preserving the original JSON structure
    Different workers process different halves, so the results must be merged
    """
    merged = {}
    
    for result in all_results:
        for league, seasons in result.items():
            if league not in merged:
                merged[league] = {}
            for season, matches in seasons.items():
                if season not in merged[league]:
                    merged[league][season] = {}
                for match, halfs in matches.items():
                    if match not in merged[league][season]:
                        merged[league][season][match] = {}
                    for half, segments in halfs.items():
                        # Assign directly because all segments for each half are processed in order by one worker
                        merged[league][season][match][half] = segments
    
    return merged


def parse_args():
    parser = argparse.ArgumentParser(
        description="Distributed LiveCC generation for SoccerNet ASR test.json"
    )
    parser.add_argument(
        "--model_name_or_path", type=str, required=True,
        help="HuggingFace model path, e.g., chenjoya/LiveCC-7B-Instruct"
    )
    parser.add_argument(
        "--not_instruct_model", action="store_true", dest="not_instruct_model",
        help="Disable instruct model mode"
    )
    parser.add_argument(
        "--num_workers", type=int, default=8,
        help="Number of parallel processes/gpus to use"
    )
    parser.add_argument(
        "--repetition_penalty", type=float, default=1.15,
        help="Repetition penalty for generation. When performing livecc, 1.15 can remove most repetition."
    )
    parser.add_argument(
        "--test_json", type=str,
        default="/root/work/datasets/SoccerNet/asr/annos/test.json",
        help="Path to test.json file"
    )
    parser.add_argument(
        "--video_root", type=str,
        default="/root/work/datasets/SoccerNet/videos",
        help="Root directory of video files"
    )
    parser.add_argument(
        "--output_json", type=str, required=True,
        help="Path to save output JSON file (same structure as test.json)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    multiprocessing.set_start_method('spawn', force=True)
    
    # Upstream implementation note.
    test_data = load_test_json(args.test_json)
    
    # Create the worker function
    worker_fn = partial(
        livecc_worker,
        model_name_or_path=args.model_name_or_path,
        test_json_path=args.test_json,
        video_root=args.video_root,
        output_json_path=args.output_json,
        simple_ctx=args.not_instruct_model,
        repetition_penalty=args.repetition_penalty,
        num_workers=args.num_workers,
    )
    
    # Use multiprocessing
    worker_results = local_mp(
        list(range(args.num_workers)),
        worker_fn,
        desc="livecc_generation",
        num_workers=args.num_workers
    )
    
    # Merge all worker results
    print("Merging results...")
    merged_results = merge_results(worker_results, test_data)
    
    # Save results
    print(f"Saving results to {args.output_json}...")
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, 'w', encoding='utf-8') as f:
        json.dump(merged_results, f, ensure_ascii=False, indent=2)
    
    print("Done!")
