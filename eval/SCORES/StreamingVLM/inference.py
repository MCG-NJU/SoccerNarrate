import argparse
import gc
import json
import multiprocessing
import os
import random
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from typing import List, Dict, Any

from tqdm import tqdm

import torch


def _maybe_append_path(path: str):
    if os.path.isdir(path) and path not in sys.path:
        sys.path.append(path)


# Add StreamingVLM/LiveCC paths
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_maybe_append_path(os.path.join(_REPO_ROOT, "streaming-vlm"))
_maybe_append_path("streaming-vlm")

from streaming_vlm.inference.inference import streaming_inference, load_model_and_processor
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


def random_prompt() -> str:
    return random.choice(SOCCER_COMMENTARY_QUERIES)


def build_video_path(video_root: str, league: str, season: str, match: str, half: str) -> str:
    return os.path.join(video_root, league, season, match, f"{half}_720p.mkv")


def split_league_season(league_season: str) -> tuple:
    parts = league_season.rsplit("_", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return league_season, ""


def flatten_to_halfs(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    halfs = []
    for item in data:
        for league_season_key, matches_dict in item.items():
            league, season = split_league_season(league_season_key)
            for match, halfs_dict in matches_dict.items():
                for half, segments in halfs_dict.items():
                    halfs.append(
                        {
                            "league": league,
                            "season": season,
                            "match": match,
                            "half": half,
                            "segments": segments,
                            "league_season_key": league_season_key,
                        }
                    )
    return halfs


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


def get_video_duration_seconds(video_path: str, max_pixels: int) -> float:
    video_reader, _, _ = get_smart_resized_video_reader(video_path, max_pixels)
    video_reader.get_frame_timestamp(0)
    pts = torch.from_numpy(video_reader._frame_pts[:, 1])
    return pts[-1].item()


def worker_process(device_id: int, halfs: List[Dict[str, Any]], shared_args: Dict[str, Any]):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    torch.cuda.set_device(0)

    print(
        f"Device {device_id}: Loading StreamingVLM model from {shared_args['model_path']}..."
    )
    model, processor = load_model_and_processor(
        shared_args["model_path"], shared_args["model_base"]
    )
    print(f"Device {device_id}: Model loaded successfully!")

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

        video_path = build_video_path(
            shared_args["video_root"], league, season, match, half
        )
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
        video_duration = get_video_duration_seconds(video_path, shared_args["max_pixels"])
        if shared_args["stream_duration"] is not None and shared_args["stream_duration"] > 0:
            run_duration = min(shared_args["stream_duration"], video_duration)
        else:
            run_duration = video_duration

        responses = streaming_inference(
            model_path=shared_args["model_path"],
            video_path=video_path,
            output_dir=None,
            model_base=shared_args["model_base"],
            model=model,
            processor=processor,
            window_size=shared_args["window_size"],
            chunk_duration=shared_args["chunk_duration"],
            text_round=shared_args["text_round"],
            previous_text=shared_args["previous_text"],
            pos_mode=shared_args["pos_mode"],
            all_text=shared_args["all_text"],
            skip_first_chunk=shared_args["skip_first_chunk"],
            recompute=shared_args["recompute"],
            text_sink=shared_args["text_sink"],
            text_sliding_window=shared_args["text_sliding_window"],
            temperature=shared_args["temperature"],
            duration=run_duration,
            query=base_prompt,
            repetition_penalty=shared_args["repetition_penalty"],
            quiet=shared_args["quiet"],
            emit_json=False,
            time_test=False,
        )

        for segment in segments:
            seg_start = time_to_seconds(segment.get("start", 0.0))
            seg_end = time_to_seconds(segment.get("end", seg_start))
            commentary = aggregate_responses_for_segment(responses, seg_start, seg_end)
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

    temp_dir = os.path.join(shared_args["output_dir"], f"temp_device_{device_id}")
    os.makedirs(temp_dir, exist_ok=True)
    with open(os.path.join(temp_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    return device_id


def merge_results(output_dir: str) -> List[Dict[str, Any]]:
    temp_dirs = [d for d in os.listdir(output_dir) if d.startswith("temp_device_")]
    all_results: List[Dict[str, Any]] = []

    for temp_dir in temp_dirs:
        temp_path = os.path.join(output_dir, temp_dir, "results.json")
        if os.path.exists(temp_path):
            with open(temp_path, "r", encoding="utf-8") as f:
                worker_results = json.load(f)
                all_results.extend(worker_results)

    merged_dict: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for item in all_results:
        for league_season_key, matches_dict in item.items():
            merged_dict.setdefault(league_season_key, {})
            for match, halfs_dict in matches_dict.items():
                merged_dict[league_season_key].setdefault(match, {})
                for half, segments in halfs_dict.items():
                    merged_dict[league_season_key][match][half] = segments

    merged_results = [{k: v} for k, v in merged_dict.items()]
    return merged_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="StreamingVLM inference for SoccerNet ASR test.json")
    parser.add_argument("--model_path", type=str, default="livecc/checkpoints/qwen3vl/asr12_lr2e-5_nodes4_gpus8/checkpoint-35", help="Path to StreamingVLM model")
    parser.add_argument("--model_base", type=str, default="Qwen2_5", choices=["Qwen2_5", "Qwen2"], help="Base architecture name")
    parser.add_argument("--test_json", type=str, default="/path/to/SoccerNet/asr/eval/SCORES/test.json", help="Path to test.json file")
    parser.add_argument("--video_root", type=str, default="/path/to/SoccerNet/videos", help="Root directory of video files")
    parser.add_argument("--output_dir", type=str, default="/path/to/SoccerNet/asr/eval/SCORES/StreamingVLM", help="Output directory")
    parser.add_argument("--num_workers", type=int, default=None, help="Number of parallel processes/gpus (default: number of available GPUs)")

    # StreamingVLM specific args
    parser.add_argument("--window_size", type=int, default=16)
    parser.add_argument("--chunk_duration", type=int, default=1)
    parser.add_argument(
        "--max_pixels",
        type=int,
        default=384 * 28 * 28,
        help="Maximum pixels used when decoding videos to estimate duration.",
    )
    parser.add_argument("--text_round", type=int, default=16)
    parser.add_argument("--previous_text", type=str, default="")
    parser.add_argument("--pos_mode", type=str, default="shrink", choices=["append", "shrink"])
    parser.add_argument("--all_text", action="store_true", default=False)
    parser.add_argument("--skip_first_chunk", type=int, default=0)
    parser.add_argument("--recompute", action="store_true", default=False)
    parser.add_argument("--text_sink", type=int, default=512)
    parser.add_argument("--text_sliding_window", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.05)
    parser.add_argument("--quiet", action="store_true", default=False)
    parser.add_argument("--stream_duration", type=float, default=None, help="Duration in seconds to run streaming inference for each half. If not set, use segment coverage.")

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "StreamingVLM")

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading test data from {args.test_json}...")
    with open(args.test_json, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    all_halfs = flatten_to_halfs(test_data)
    print(f"Total {len(all_halfs)} halfs to process")

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

    halfs_per_device = {gpu_id: [] for gpu_id in selected_gpu_ids}
    for idx, half_data in enumerate(all_halfs):
        gpu_id = selected_gpu_ids[idx % num_workers]
        halfs_per_device[gpu_id].append(half_data)

    shared_args = {
        "model_path": args.model_path,
        "model_base": args.model_base,
        "video_root": args.video_root,
        "window_size": args.window_size,
        "chunk_duration": args.chunk_duration,
        "max_pixels": args.max_pixels,
        "text_round": args.text_round,
        "previous_text": args.previous_text,
        "pos_mode": args.pos_mode,
        "all_text": args.all_text,
        "skip_first_chunk": args.skip_first_chunk,
        "recompute": args.recompute,
        "text_sink": args.text_sink,
        "text_sliding_window": args.text_sliding_window,
        "temperature": args.temperature,
        "repetition_penalty": args.repetition_penalty,
        "quiet": args.quiet,
        "output_dir": args.output_dir,
        "stream_duration": args.stream_duration,
    }

    multiprocessing.set_start_method("spawn", force=True)

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for gpu_id, halfs in halfs_per_device.items():
            if not halfs:
                continue
            futures.append(
                executor.submit(
                    worker_process,
                    gpu_id,
                    halfs,
                    shared_args,
                )
            )
        for future in tqdm(futures, desc="Processing"):
            future.result()

    print("Merging results...")
    merged_results = merge_results(args.output_dir)
    output_json_path = os.path.join(args.output_dir, "results.json")
    print(f"Saving results to {output_json_path}...")
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(merged_results, f, ensure_ascii=False, indent=2)

    temp_dirs = [d for d in os.listdir(args.output_dir) if d.startswith("temp_device_")]
    for temp_dir in temp_dirs:
        shutil.rmtree(os.path.join(args.output_dir, temp_dir), ignore_errors=True)

    print("All done!")
