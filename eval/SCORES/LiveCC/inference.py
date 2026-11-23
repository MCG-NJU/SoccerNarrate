import argparse
import gc
import json
import math
import multiprocessing
import os
import random
import shutil
import sys
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from typing import List, Dict, Any, Optional, Deque

from tqdm import tqdm

import torch


def _maybe_append_path(path: str):
    if os.path.isdir(path) and path not in sys.path:
        sys.path.append(path)


# Potential repo roots in different environments
_maybe_append_path("livecc")
_maybe_append_path("livecc")
_maybe_append_path("streaming-vlm")
_maybe_append_path("streaming-vlm")

from demo.infer import LiveCCDemoInfer
from livecc_utils import get_smart_resized_video_reader

SOCCER_COMMENTARY_QUERIES: List[str] = [
    "Commentate on this match.",
    "Describe what's happening in this video.",
    "Provide live commentary for this clip.",
    "What's happening in this football match?",
    "Comment on the action in this video.",
    "Give a commentary of this play.",
    "Describe the events in this clip.",
    "What's going on in this match?",
    "Provide commentary for this moment.",
    "Tell me what's happening here.",
    "Commentate on this action.",
    "Describe the play in this video.",
    "What can you see in this clip?",
    "Give me a live commentary.",
    "What's the action in this video?",
    "Comment on this football moment.",
    "Describe the match events.",
    "What's happening on the pitch?",
    "Provide match commentary.",
    "Commentate on this scene."
]


class CommentaryContextWindow:
    """
    Maintain a moving buffer of generated commentary text for roughly
    `window_seconds` of video. When exceeding the budget, drop the oldest
    half of the tokens to keep prompts bounded.
    """

    def __init__(self, window_seconds: float, tokens_per_second: float):
        self.window_seconds = float(window_seconds)
        self.tokens_per_second = float(tokens_per_second)
        self.entries: Deque[Dict[str, Any]] = deque()
        self.total_tokens: int = 0

        if self.window_seconds > 0 and self.tokens_per_second > 0:
            limit = int(round(self.window_seconds * self.tokens_per_second))
            self.window_token_limit: Optional[int] = max(1, limit)
            self.half_window_tokens: Optional[int] = max(
                1, self.window_token_limit // 2
            )
        else:
            self.window_token_limit = None
            self.half_window_tokens = None

    def get_context_text(self) -> str:
        return " ".join(entry["text"] for entry in self.entries if entry["text"]).strip()

    def add_commentary(self, text: str, duration_seconds: float):
        cleaned_text = (text or "").strip()
        if not cleaned_text:
            return
        tokens = self._duration_to_tokens(duration_seconds)
        self.entries.append({"text": cleaned_text, "tokens": tokens})
        self.total_tokens += tokens
        self._trim_buffer()

    def _duration_to_tokens(self, duration_seconds: float) -> int:
        try:
            duration_val = float(duration_seconds)
        except (TypeError, ValueError):
            duration_val = 0.0
        duration_val = max(0.0, duration_val)
        if self.tokens_per_second <= 0:
            return 0
        estimated = int(round(duration_val * self.tokens_per_second))
        return max(1, estimated)

    def _trim_buffer(self):
        if not self.window_token_limit or not self.half_window_tokens:
            return

        while self.total_tokens > self.window_token_limit and self.entries:
            tokens_to_drop = self.half_window_tokens
            while self.entries and tokens_to_drop > 0:
                removed = self.entries.popleft()
                self.total_tokens -= removed["tokens"]
                tokens_to_drop -= removed["tokens"]
            if self.total_tokens <= 0:
                self.total_tokens = 0
                break


def time_to_seconds(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return 0.0
        if ":" in stripped:
            try:
                parts = [float(part) for part in stripped.split(":")]
            except ValueError:
                return 0.0
            seconds = 0.0
            for part in parts:
                seconds = seconds * 60.0 + part
            return seconds
        try:
            return float(stripped)
        except ValueError:
            return 0.0

    return 0.0


def aggregate_responses_for_segment(
    responses: List[Dict[str, Any]], seg_start: float, seg_end: float
) -> str:
    if seg_end <= seg_start:
        seg_end = seg_start + 1e-3
    overlapped = []
    for resp in responses:
        resp_start = float(resp.get("start_time", 0.0))
        resp_end = float(resp.get("end_time", resp_start))
        if resp_end <= seg_start or resp_start >= seg_end:
            continue
        text = resp.get("response", "").strip()
        if text:
            overlapped.append(text)
    commentary = " ".join(overlapped).strip()
    if commentary and not commentary.endswith("..."):
        commentary += "..."
    return commentary or "..."


def random_prompt() -> str:
    """Choose a prompt at random"""
    return random.choice(SOCCER_COMMENTARY_QUERIES)


def build_video_path(video_root: str, league: str, season: str, match: str, half: str) -> str:
    """Build the video file path"""
    return os.path.join(video_root, league, season, match, f"{half}_720p.mkv")


def split_league_season(league_season: str) -> tuple:
    """Split a league-season string into league and season"""
    # Split from the last underscore, for example "germany_bundesliga_2016-2017" -> ("germany_bundesliga", "2016-2017")
    parts = league_season.rsplit('_', 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return league_season, ""


def flatten_to_halfs(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Flatten nested JSON into a list of halves
    Each half contains: league, season, match, half, segments, league_season_key
    This keeps all segments from the same half on one worker in order
    """
    halfs = []
    for item in data:
        for league_season_key, matches_dict in item.items():
            league, season = split_league_season(league_season_key)
            for match, halfs_dict in matches_dict.items():
                for half, segments in halfs_dict.items():
                    halfs.append({
                        "league": league,
                        "season": season,
                        "match": match,
                        "half": half,
                        "segments": segments,
                        "league_season_key": league_season_key,
                    })
    return halfs


def get_video_duration_seconds(video_path: str, max_pixels: int) -> float:
    video_reader, _, _ = get_smart_resized_video_reader(video_path, max_pixels)
    video_reader.get_frame_timestamp(0)
    pts = torch.from_numpy(video_reader._frame_pts[:, 1])
    return pts[-1].item()


def stream_livecc_half(
    infer: LiveCCDemoInfer,
    video_path: str,
    query: str,
    duration_limit: Optional[float],
    reset_interval: float,
    context_window_seconds: float,
    tokens_per_second: float,
    repetition_penalty: float,
    max_pixels: int,
    streaming_eos_base_threshold: Optional[float],
    streaming_eos_threshold_step: Optional[float],
    do_sample: bool,
) -> List[Dict[str, Any]]:
    total_duration = get_video_duration_seconds(video_path, max_pixels)
    if duration_limit is None or duration_limit <= 0:
        duration = total_duration
    else:
        duration = min(duration_limit, total_duration)

    state: Dict[str, Any] = {
        "video_path": video_path,
        "last_timestamp": -1 / infer.fps,
    }
    context_window = CommentaryContextWindow(context_window_seconds, tokens_per_second)
    responses: List[Dict[str, Any]] = []
    conversation_start_time = 0.0
    conversation_responses: List[str] = []

    frame_interval = infer.frame_time_interval
    max_steps = math.ceil(duration / frame_interval) + 1

    for step in range(max_steps):
        state["video_timestamp"] = min(step * frame_interval, duration)
        current_time = state["video_timestamp"]

        if (
            conversation_responses
            and reset_interval > 0
            and current_time - conversation_start_time >= reset_interval
        ):
            conversation_text = " ".join(conversation_responses).strip()
            if conversation_text:
                context_window.add_commentary(
                    conversation_text, current_time - conversation_start_time
                )
                state["previous_text"] = context_window.get_context_text()
            state["past_key_values"] = None
            state["past_ids"] = None
            state["message"] = None
            conversation_start_time = current_time
            conversation_responses = []

        for (start_timestamp, stop_timestamp), chunk_text, updated_state in infer.live_cc(
            message=query,
            state=state,
            max_pixels=max_pixels,
            default_query=query,
            do_sample=do_sample,
            repetition_penalty=repetition_penalty,
            streaming_eos_base_threshold=streaming_eos_base_threshold,
            streaming_eos_threshold_step=streaming_eos_threshold_step,
        ):
            state = updated_state
            cleaned = chunk_text.rstrip()
            if cleaned.endswith(" ..."):
                cleaned = cleaned[:-4].rstrip()
            if not cleaned:
                continue
            responses.append(
                {
                    "response": cleaned,
                    "start_time": float(start_timestamp),
                    "end_time": float(stop_timestamp),
                }
            )
            conversation_responses.append(cleaned)

        if state.get("video_end", False):
            break

    return responses


def worker_process(device_id: int, halfs: List[Dict], shared_args: Dict):
    """Worker process that handles data on the selected GPU"""
    torch.cuda.set_device(device_id)
    device = f"cuda:{device_id}"

    print(f"Device {device_id}: Loading LiveCC model from {shared_args['model_path']}...")
    infer = LiveCCDemoInfer(model_path=shared_args["model_path"], device=device)
    print(f"Device {device_id}: Model loaded successfully!")

    # Store results while preserving the original structure
    results = []

    pbar = tqdm(halfs, desc=f"Device {device_id}", position=device_id)
    total_segments_processed = 0

    for half_data in pbar:
        league = half_data["league"]
        season = half_data["season"]
        match = half_data["match"]
        half = half_data["half"]
        segments = half_data["segments"]
        league_season_key = half_data["league_season_key"]

        # Build the video path
        video_path = build_video_path(shared_args["video_root"], league, season, match, half)
        if not os.path.exists(video_path):
            pbar.set_postfix({"skipped": "video_missing"})
            continue

        # Initialize the result structure
        result_item = None
        for item in results:
            if league_season_key in item:
                result_item = item
                break
        
        if result_item is None:
            result_item = {league_season_key: {}}
            results.append(result_item)
        
        if match not in result_item[league_season_key]:
            result_item[league_season_key][match] = {}
        
        if half not in result_item[league_season_key][match]:
            result_item[league_season_key][match][half] = []

        base_prompt = random_prompt()
        chunk_responses = stream_livecc_half(
            infer=infer,
            video_path=video_path,
            query=base_prompt,
            duration_limit=shared_args["stream_duration"],
            reset_interval=shared_args["reset_interval"],
            context_window_seconds=shared_args["context_window_seconds"],
            tokens_per_second=shared_args["tokens_per_second"],
            repetition_penalty=shared_args["repetition_penalty"],
            max_pixels=shared_args["max_pixels"],
            streaming_eos_base_threshold=shared_args["streaming_eos_base_threshold"],
            streaming_eos_threshold_step=shared_args["streaming_eos_threshold_step"],
            do_sample=shared_args["do_sample"],
        )

        for segment in segments:
            seg_start = time_to_seconds(segment.get("start", 0.0))
            seg_end = time_to_seconds(segment.get("end", seg_start))
            commentary = aggregate_responses_for_segment(
                chunk_responses, seg_start, seg_end
            )

            result_segment = {
                "start": segment["start"],
                "end": segment["end"],
                "commentary": commentary,
            }
            result_item[league_season_key][match][half].append(result_segment)

            total_segments_processed += 1

        torch.cuda.empty_cache()
        gc.collect()

        pbar.set_postfix({"processed_segments": total_segments_processed})

    pbar.close()

    # Save temporary results
    temp_dir = os.path.join(shared_args["output_dir"], f"temp_device_{device_id}")
    os.makedirs(temp_dir, exist_ok=True)
    with open(os.path.join(temp_dir, "results.json"), 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    return device_id


def merge_results(output_dir: str, original_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge all worker results while preserving the original JSON structure"""
    temp_dirs = [d for d in os.listdir(output_dir) if d.startswith("temp_device_")]
    all_results: List[Dict[str, Any]] = []
    
    for temp_dir in temp_dirs:
        temp_path = os.path.join(output_dir, temp_dir, "results.json")
        if os.path.exists(temp_path):
            with open(temp_path, 'r', encoding='utf-8') as f:
                worker_results = json.load(f)
                all_results.extend(worker_results)
    
    # Merge results with the same league_season_key
    merged_dict = {}
    for item in all_results:
        for league_season_key, matches_dict in item.items():
            if league_season_key not in merged_dict:
                merged_dict[league_season_key] = {}
            for match, halfs_dict in matches_dict.items():
                if match not in merged_dict[league_season_key]:
                    merged_dict[league_season_key][match] = {}
                for half, segments in halfs_dict.items():
                    # Assign directly because all segments for each half are processed in order by one worker
                    merged_dict[league_season_key][match][half] = segments
    
    # Convert back to the original array format
    merged_results = [{k: v} for k, v in merged_dict.items()]
    
    return merged_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LiveCC inference for SoccerNet ASR test.json")
    parser.add_argument("--model_path", type=str, default="livecc/checkpoints/qwen3vl/asr12_lr2e-5_nodes4_gpus8/checkpoint-35", help="Path to LiveCC model")
    parser.add_argument("--test_json", type=str, default="/path/to/SoccerNet/asr/eval/SCORES/test.json", help="Path to test.json file")
    parser.add_argument("--video_root", type=str, default="/path/to/SoccerNet/videos", help="Root directory of video files")
    parser.add_argument("--output_dir", type=str, default="/path/to/SoccerNet/asr/eval/SCORES/LiveCC", help="Output directory (default: same as test.json directory)")
    parser.add_argument("--repetition_penalty", type=float, default=1.15, help="Repetition penalty for generation")
    parser.add_argument("--num_workers", type=int, default=None, help="Number of parallel processes/gpus (default: number of available GPUs)")
    parser.add_argument(
        "--max_pixels",
        type=int,
        default=384 * 28 * 28,
        help="Maximum pixels for video resizing when streaming frames.",
    )
    parser.add_argument(
        "--context_window_seconds",
        type=float,
        default=64.0,
        help="Approximate duration (in seconds) of video commentary retained in the prompt before trimming.",
    )
    parser.add_argument(
        "--tokens_per_second",
        type=float,
        default=2.0,
        help="Estimated number of tokens generated per video second. Used to translate video duration into token budgets.",
    )
    parser.add_argument(
        "--reset_interval",
        type=float,
        default=64.0,
        help="Seconds between LiveCC conversation resets (affects KV cache length).",
    )
    parser.add_argument(
        "--stream_duration",
        type=float,
        default=None,
        help="Optional cap on streaming duration (seconds) processed per half.",
    )
    parser.add_argument(
        "--streaming_eos_base_threshold",
        type=float,
        default=None,
        help="Base threshold for the streaming EOS token (see LiveCC demo).",
    )
    parser.add_argument(
        "--streaming_eos_threshold_step",
        type=float,
        default=None,
        help="Increment for the streaming EOS threshold.",
    )
    parser.add_argument(
        "--do_sample",
        action="store_true",
        default=False,
        help="Enable sampling when generating commentary.",
    )
    args = parser.parse_args()

    # Set the output directory
    if args.output_dir is None:
        # By default, save next to livecc in the directory containing inference.py
        args.output_dir = os.path.dirname(os.path.abspath(__file__))
    
    os.makedirs(args.output_dir, exist_ok=True)

    # Load test data
    print(f"Loading test data from {args.test_json}...")
    with open(args.test_json, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    # Flatten data into a list of halves
    all_halfs = flatten_to_halfs(test_data)
    print(f"Total {len(all_halfs)} halfs to process")

    # Determine the number of GPUs to use
    available_gpu_ids = list(range(torch.cuda.device_count()))
    if args.num_workers is None:
        num_workers = len(available_gpu_ids)
    else:
        num_workers = min(args.num_workers, len(available_gpu_ids))
    
    if num_workers == 0:
        raise RuntimeError("No available GPUs.")
    
    if len(all_halfs) == 0:
        raise RuntimeError("No halfs to process.")

    selected_gpu_ids = available_gpu_ids[:num_workers]
    print(f"Using {num_workers} GPUs to process {len(all_halfs)} halfs...")

    # Assign tasks
    halfs_per_device = {gpu_id: [] for gpu_id in selected_gpu_ids}
    for idx, half_data in enumerate(all_halfs):
        gpu_id = selected_gpu_ids[idx % num_workers]
        halfs_per_device[gpu_id].append(half_data)

    shared_args = {
        "model_path": args.model_path,
        "video_root": args.video_root,
        "repetition_penalty": args.repetition_penalty,
        "output_dir": args.output_dir,
        "context_window_seconds": args.context_window_seconds,
        "tokens_per_second": args.tokens_per_second,
        "reset_interval": args.reset_interval,
        "stream_duration": args.stream_duration,
        "max_pixels": args.max_pixels,
        "streaming_eos_base_threshold": args.streaming_eos_base_threshold,
        "streaming_eos_threshold_step": args.streaming_eos_threshold_step,
        "do_sample": args.do_sample,
    }

    multiprocessing.set_start_method('spawn', force=True)

    # Start multiprocessing
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for gpu_id, halfs in halfs_per_device.items():
            if not halfs:
                continue
            futures.append(executor.submit(
                worker_process,
                gpu_id,
                halfs,
                shared_args,
            ))
        for future in tqdm(futures, desc="Processing"):
            future.result()

    print("Merging results...")
    merged_results = merge_results(args.output_dir, test_data)
    
    # Save final results
    output_json_path = os.path.join(args.output_dir, "results.json")
    print(f"Saving results to {output_json_path}...")
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(merged_results, f, ensure_ascii=False, indent=2)
    
    # Clean up temporary files
    temp_dirs = [d for d in os.listdir(args.output_dir) if d.startswith("temp_device_")]
    for temp_dir in temp_dirs:
        temp_path = os.path.join(args.output_dir, temp_dir)
        shutil.rmtree(temp_path, ignore_errors=True)
    
    print("All done!")
