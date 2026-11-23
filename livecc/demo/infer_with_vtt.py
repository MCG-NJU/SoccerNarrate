#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run LiveCC inference and save a VTT subtitle file
Usage:
    python infer_with_vtt.py \
        --model_path /path/to/model \
        --video_path /path/to/video.mp4 \
        --output_vtt output.vtt \
        --query "Please commentate on this match"
"""
import argparse
import time
from infer import LiveCCDemoInfer
import sys
import os
sys.path.append("streaming-vlm")
from streaming_vlm.utils.vtt_utils import open_vtt, sec2ts
def streaming_inference_with_vtt(
    model_path: str,
    video_path: str,
    output_vtt: str,
    query: str = "Please describe the video.",
    max_pixels: int = 384 * 28 * 28,
    do_sample: bool = True,
    repetition_penalty: float = 1.05,
    streaming_eos_base_threshold: float = None,
    streaming_eos_threshold_step: float = None,
    device: str = None,
    model_family: str = None,
    duration: float = None,
    previous_text: str = "",
    reset_interval: float = 16.0,
    quiet: bool = False,
):
    """
    Run LiveCC inference and save a VTT subtitle file
    
    Arguments:
        model_path: Model path
        video_path: Video file path
        output_vtt: Output VTT file path
        query: User query
        max_pixels: Maximum pixel count
        do_sample: Whether to use sampling
        repetition_penalty: Repetition penalty
        streaming_eos_base_threshold: Base threshold for streaming EOS
        streaming_eos_threshold_step: Step size for the streaming EOS threshold
        device: Device (cuda/cpu/mps)
        model_family: Model family (qwen2/qwen3)
        duration: Video duration to process in seconds; None means the full video
        previous_text: Initial previous text for extra context
        reset_interval: Conversation reset interval in seconds; previous text is accumulated after each reset
        quiet: Whether to run in quiet mode
    """
    
    def printq(*args, **kwargs):
        if not quiet:
            print(*args, **kwargs)
    
    # 1. Initialize the inferencer
    printq(f"Loading model: {model_path}")
    inferencer = LiveCCDemoInfer(
        model_path=model_path,
        device=device,
        model_family=model_family,
    )
    printq("Model loaded")
    
    # 2. Prepare the VTT file
    if os.path.exists(output_vtt):
        os.remove(output_vtt)
    with open_vtt(output_vtt):  # Write the WEBVTT header
        pass
    printq(f"VTT subtitles will be saved to: {output_vtt}")
    
    # 3. Initialize state
    state = {
        'video_path': video_path,
        'last_timestamp': -1 / inferencer.fps,
    }
    initial_previous_text = (previous_text or "").strip()
    if initial_previous_text:
        state['previous_text'] = initial_previous_text
    
    # 4. Get total video duration
    from livecc_utils import get_smart_resized_video_reader
    video_reader, _, _ = get_smart_resized_video_reader(video_path, max_pixels)
    video_reader.get_frame_timestamp(0)
    import torch
    video_pts = torch.from_numpy(video_reader._frame_pts[:, 1])
    total_duration = video_pts[-1].item()
    
    if duration is None or duration > total_duration:
        duration = total_duration
    
    printq(f"Video duration: {total_duration:.2f}s, processing: {duration:.2f}s")
    
    # 5. Streaming inference
    printq("\nStarting inference...")
    printq("=" * 80)
    
    chunk_index = 0
    total_inference_time = 0
    accumulated_responses = []
    import math
    max_steps = math.ceil(duration / inferencer.frame_time_interval) + 1
    
    # Conversation reset variables
    conversation_start_time = 0.0  # Current conversation start timestamp
    conversation_responses = []   # Text generated during the current conversation
    
    for step in range(max_steps):
        state['video_timestamp'] = min(step * inferencer.frame_time_interval, duration)
        chunk_start_time = time.perf_counter()
        
        # Check whether to reset the conversation once per reset_interval seconds
        current_time = state['video_timestamp']
        if current_time - conversation_start_time >= reset_interval:
            # Reset the conversation: clear the KV cache and append current text to previous_text
            if conversation_responses:
                # Append all text from the current conversation to previous_text
                conversation_text = " ".join(conversation_responses)
                if state.get('previous_text', ''):
                    state['previous_text'] = state['previous_text'] + " " + conversation_text
                else:
                    state['previous_text'] = conversation_text
            
            # Reset the KV cache
            state['past_key_values'] = None
            state['past_ids'] = None
            state['message'] = None  # Reset message so the query is added again
            
            # Start a new conversation
            conversation_start_time = current_time
            conversation_responses = []
            if not quiet:
                printq(f"\n[Reset conversation] time: {current_time:.1f}s, previous_text length: {len(state.get('previous_text', ''))}")
        
        for (start_timestamp, stop_timestamp), response, updated_state in inferencer.live_cc(
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
            
            response_clean = response.rstrip()
            if response_clean.endswith(' ...'):
                response_clean = response_clean[:-4].rstrip()
            if not response_clean:
                continue
            
            chunk_time = time.perf_counter() - chunk_start_time
            total_inference_time += chunk_time
            
            hms_start = time.strftime('%H:%M:%S', time.gmtime(int(start_timestamp)))
            hms_end = time.strftime('%H:%M:%S', time.gmtime(int(stop_timestamp)))
            printq(f"[{chunk_index:3d}] {hms_start}-{hms_end} | inference time: {chunk_time:.3f}s")
            printq(f"      \033[1m\033[34m{response_clean}\033[0m")
            
            ts_start = sec2ts(start_timestamp)
            ts_end = sec2ts(stop_timestamp)
            with open_vtt(output_vtt) as vf:
                vf.write(f"{ts_start} --> {ts_end}\n")
                vf.write(f"{response_clean}\n\n")
            accumulated_responses.append(response_clean)
            conversation_responses.append(response_clean)  # Append to the response list for the current conversation
            
            chunk_index += 1
            chunk_start_time = time.perf_counter()
        
        if state.get('video_end', False):
            break
    
    # 7. Summary
    printq("=" * 80)
    printq(f"\nInference complete!")
    printq(f"   Processed {chunk_index} chunks")
    printq(f"   Total inference time: {total_inference_time:.2f}s")
    printq(f"   Average per chunk: {total_inference_time/max(chunk_index, 1):.3f}s")
    printq(f"   Subtitles saved to: {output_vtt}")
    
    return {
        'num_chunks': chunk_index,
        'total_time': total_inference_time,
        'responses': accumulated_responses,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run LiveCC inference and save a VTT subtitle file"
    )
    
    # Upstream implementation note.
    parser.add_argument(
        "--model_path",
        type=str,
        default="livecc/checkpoints/qwen3vl/lr2e-5_16_64_5/checkpoint-125",
        help="Model path"
    )
    parser.add_argument(
        "--video_path",
        type=str,
        default="/path/to/SoccerNet/videos/england_epl/2015-2016/2015-09-26 - 19-30 Newcastle Utd 2 - 2 Chelsea/1_720p.mkv",
        help="Video file path"
    )
    parser.add_argument(
        "--output_vtt",
        type=str,
        default="livecc/demo/test.vtt",
        help="Output VTT subtitle file path"
    )
    
    # Upstream implementation note.
    parser.add_argument(
        "--query",
        type=str,
        default="Please describe the video.",
        help="User query"
    )
    parser.add_argument(
        "--max_pixels",
        type=int,
        default=384 * 28 * 28,
        help="Maximum pixel count"
    )
    parser.add_argument(
        "--do_sample",
        default=True,
        help="Whether to use sampling"
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.05,
        help="Repetition penalty"
    )
    parser.add_argument(
        "--streaming_eos_base_threshold",
        type=float,
        default=None,
        help="Base threshold for streaming EOS"
    )
    parser.add_argument(
        "--streaming_eos_threshold_step",
        type=float,
        default=None,
        help="Step size for the streaming EOS threshold"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (cuda/cpu/mps)"
    )
    parser.add_argument(
        "--model_family",
        type=str,
        default=None,
        choices=["qwen3"],
        help="Model family"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Video duration to process in seconds; omit to process the full video"
    )
    parser.add_argument(
        "--previous_text",
        type=str,
        default="",
        help="Initial previous text"
    )
    parser.add_argument(
        "--reset_interval",
        type=float,
        default=16.0,
        help="Conversation reset interval in seconds; previous text is accumulated after each reset, similar to the per-sample duration in training data"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Quiet mode with less output"
    )
    
    args = parser.parse_args()
    
    streaming_inference_with_vtt(**vars(args))


if __name__ == "__main__":
    main()
