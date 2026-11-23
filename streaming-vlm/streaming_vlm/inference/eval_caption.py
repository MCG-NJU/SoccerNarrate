#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Streaming video description generation
Run example:
    python streaming_inference.py \
        --model_path Qwen/Qwen2-VL-7B-Instruct \
        --video_path demo/sources/howto_fix_laptop_mute_1080p.mp4
"""
import sys
sys.path.append("streaming-vlm")
import warnings
import logging
# Upstream implementation note.
warnings.filterwarnings("ignore", message=".*Unused or unrecognized kwargs.*")
# Filter specific Transformers processor warnings
transformers_logger = logging.getLogger("transformers")
transformers_logger.addFilter(lambda record: "Unused or unrecognized kwargs" not in str(record.getMessage()))
import torch, functools, os, argparse
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2VLForConditionalGeneration, Qwen2_5_VLProcessor, Qwen2VLProcessor, AutoProcessor
from streaming_vlm.inference.streaming_args import StreamingArgs
from streaming_vlm.utils.get_qwen_range import *
from qwen_vl_utils.vision_process import (
    FORCE_QWENVL_VIDEO_READER, VIDEO_TOTAL_PIXELS, FPS_MAX_FRAMES, VIDEO_MIN_PIXELS, VIDEO_MAX_PIXELS, FRAME_FACTOR, IMAGE_FACTOR, FPS,
    smart_nframes, smart_resize
)
import json
import math
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple, Optional
from streaming_vlm.inference.qwen2_5.patch_model import convert_qwen2_5_to_streaming
from streaming_vlm.inference.qwen2.patch_model import convert_qwen2_to_streaming
from contextlib import contextmanager          
from transformers import set_seed
set_seed(42)   
from streaming_vlm.data.lmm_dataset import LMMDataset
from livecc_utils import  get_smart_resized_video_reader
from livecc_utils.video_process_patch import _read_video_decord_plus, _spatial_resize_video
import time
import multiprocessing
from functools import partial
try:
    from utils.multiprocessor import local_mp
except ImportError:
    # Use a simple fallback if multiprocessor is unavailable
    import tqdm
    from concurrent.futures import ProcessPoolExecutor
    def local_mp(datums: list, func: callable, desc: str = None, num_workers=16):
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            if desc is None:
                return list(executor.map(func, datums))
            return list(tqdm.tqdm(executor.map(func, datums), total=len(datums), desc=desc))
# -----------------------------------------------------------------
# Global configuration
# -----------------------------------------------------------------
TOTAL_VIDEO_DURATION = 6000  # Total video duration to process (seconds)
DEFAULT_CHUNK_DURATION = 1       # Duration of each video chunk (seconds)
DEFAULT_WINDOW_SIZE = 16
DEFAULT_TEXT_ROUND = 16
DEFAULT_TEXT_SINK = 512
DEFAULT_TEXT_SLIDING_WINDOW = 512
DEFAULT_TEMPERATURE = 0.9
DEFAULT_REPETITION_PENALTY = 1.05

NFRAMES = FPS * DEFAULT_WINDOW_SIZE
MAX_PIXELS = max(min(VIDEO_MAX_PIXELS, VIDEO_TOTAL_PIXELS / NFRAMES * FRAME_FACTOR), int(VIDEO_MIN_PIXELS * 1.05))
MAX_TOKEN_PER_DURATION = 20

# -----------------------------------------------------------------
# Helper: KV cache pruning
# -----------------------------------------------------------------
def prune_id_and_kv_cache(input_ids, past_key_values, start_index, end_index):
    # Closed interval
    input_ids = torch.cat([input_ids[:,:start_index], input_ids[:,end_index+1:]],dim=1)

    for i, (k_layer, v_layer) in enumerate(past_key_values):
        seq_len = k_layer.shape[2]
        indices_to_keep = list(range(start_index)) + list(range(end_index + 1, seq_len))
        indices_tensor = torch.tensor(indices_to_keep, device=k_layer.device)
        past_key_values.key_cache[i] = torch.index_select(k_layer, 2, indices_tensor)
        past_key_values.value_cache[i] = torch.index_select(v_layer, 2, indices_tensor)

    return input_ids, past_key_values

def contiguous_id_and_kv(input_ids, past_key_values):
    input_ids = input_ids.contiguous()
    for i, (k_layer, v_layer) in enumerate(past_key_values):
        past_key_values.key_cache[i] = k_layer.contiguous()
        past_key_values.value_cache[i] = v_layer.contiguous()
    return input_ids, past_key_values

def load_model_and_processor(model_path, model_base = 'Qwen2_5', device_id: Optional[int] = None):
    """
    Load the model and processor
    device_id: if specified, load on that GPU; otherwise use automatic placement
    """
    if device_id is not None:
        device_map = f"cuda:{device_id}"
    else:
        device_map = "cuda"
    
    if model_base == 'Qwen2_5':
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype="auto", device_map=device_map,
            attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager"
        )
        model = convert_qwen2_5_to_streaming(model)
        processor = AutoProcessor.from_pretrained(model_path, use_fast=False)
    elif model_base == 'Qwen2':
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype="auto", device_map=device_map,
            attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager"
        )
        model = convert_qwen2_to_streaming(model)
        processor = AutoProcessor.from_pretrained(model_path, use_fast=False)
    return model, processor

def process_past_kv(past_key_values, i, text_round, visual_round, full_conversation_history, prev_generated_ids,
                    assistant_start_bias, assistant_end_bias, recent_video_window_clips, recent_pixel_values_videos,
                    text_sink,text_sliding_window):

    if i>=text_round:
        # Move the assistant text generated later into "previous text"
        if full_conversation_history[0]['role'] != 'previous text':
            breakpoint()
        assert full_conversation_history[0]['role'] == 'previous text'
        assert full_conversation_history[-2*text_round]['role'] == 'user'
        assert full_conversation_history[-(2*text_round-1)]['role'] == 'assistant' # The last pattern must be ... User Assistant User Assistant; from the end, find the first assistant
        full_conversation_history[0]['content'] += full_conversation_history[-(2*text_round-1)]['content'][:-4] # Remove the trailing " ..."

        def resort_id_and_kv(input_ids, past_key_values, src_start_idx, src_end_idx, dst_idx):
            # Move from src_start_idx..src_end_idx to after dst_idx
            assert  dst_idx < src_start_idx <= src_end_idx
            input_ids = torch.cat([input_ids[:,:dst_idx+1], input_ids[:,src_start_idx:src_end_idx+1], input_ids[:,dst_idx+1:src_start_idx], input_ids[:,src_end_idx+1:]],dim=1)
            
            for i, (k_layer, v_layer) in enumerate(past_key_values):
                past_key_values.key_cache[i] = torch.cat([k_layer[:,:,:dst_idx+1], k_layer[:,:,src_start_idx:src_end_idx+1], k_layer[:,:,dst_idx+1:src_start_idx], k_layer[:,:,src_end_idx+1:]],dim=2)
                past_key_values.value_cache[i] = torch.cat([v_layer[:,:,:dst_idx+1], v_layer[:,:,src_start_idx:src_end_idx+1], v_layer[:,:,dst_idx+1:src_start_idx], v_layer[:,:,src_end_idx+1:]],dim=2)
            return input_ids, past_key_values

        assistant_text_start_idx, assistant_text_end_idx = get_qwen_range(prev_generated_ids, 'assistant', 0)

        previous_text_start_idx, previous_text_end_idx = get_qwen_range(prev_generated_ids, 'previous text', 0, contain_lf=False)

        if prev_generated_ids[0,assistant_text_end_idx] == TOKEN_IDS["\n"]:
            src_start_idx = assistant_text_start_idx+assistant_start_bias
            src_end_idx = assistant_text_end_idx-assistant_end_bias-1
        else:
            src_start_idx = assistant_text_start_idx+assistant_start_bias
            src_end_idx = assistant_text_end_idx-assistant_end_bias
        if src_start_idx <= src_end_idx:
            prev_generated_ids,past_key_values = resort_id_and_kv(prev_generated_ids, past_key_values, src_start_idx, src_end_idx, previous_text_end_idx-1)

        # Prune: delete user_text and assistant formatting
        for k,item in enumerate(full_conversation_history[-2*text_round]['content']):
            if item['type'] == 'text':
                del full_conversation_history[-2*text_round]['content'][k] # In livecc format, the first text is "time xx-xx"; just delete the first one to avoid removing the first query
                break
        
        del full_conversation_history[-(2*text_round-1)]
        if visual_round > text_round:
            # If more visual rounds are kept than text rounds, delete this text round separately;
            # otherwise, delete the entire user at the end
            user_text_start_idx, user_text_end_idx = get_qwen_range(prev_generated_ids, 'user_text', -text_round, contain_lf=False)
            prev_generated_ids,past_key_values = prune_id_and_kv_cache(prev_generated_ids, past_key_values, user_text_start_idx, user_text_end_idx)
        try:
            assistant_text_start_idx, assistant_text_end_idx = get_qwen_range(prev_generated_ids, 'assistant', -text_round)
        except Exception as e:
            breakpoint()
        prev_generated_ids,past_key_values = prune_id_and_kv_cache(prev_generated_ids, past_key_values, assistant_text_start_idx, assistant_text_end_idx)

    if i >= visual_round :
        recent_video_window_clips.pop(0)
        recent_pixel_values_videos.pop(0)
        if visual_round < text_round:
            # Similarly:
            # If more text rounds are kept than visual rounds, delete this visual round separately;
            # otherwise, delete the entire user at the end
            if visual_round >= text_round:
                full_conversation_history[1]['content'] = [item for item in full_conversation_history[1]['content'] if item['type']!='video']
            else:
                full_conversation_history[-2*visual_round]['content'] = [item for item in full_conversation_history[-2*visual_round]['content'] if item['type']!='video']
            
            video_token_start_index, video_token_end_index = get_qwen_range(prev_generated_ids, 'vision_block', 0)
            prev_generated_ids, past_key_values = prune_id_and_kv_cache(prev_generated_ids, past_key_values, video_token_start_index, video_token_end_index)
    
    if i >= max(visual_round,text_round):
        # If both the vision and text of this round have been deleted, then delete the entire block
        del full_conversation_history[1]
        user_range = get_qwen_range(prev_generated_ids, 'user', 0, strict=False)
        if user_range is not None:
            user_start_idx, user_end_idx = user_range
            prev_generated_ids, past_key_values = prune_id_and_kv_cache(prev_generated_ids, past_key_values, user_start_idx, user_end_idx)

    if i > 0: 
        if text_sink is not None or text_sliding_window is not None:
            previous_text_start_idx, previous_text_end_idx = get_qwen_range(prev_generated_ids, 'previous text', 0)
            cut_start_idx = previous_text_start_idx + text_sink + 4 if text_sink is not None else previous_text_start_idx
            cut_end_idx = previous_text_end_idx - text_sliding_window - 1 if text_sliding_window is not None else previous_text_end_idx
            # 4 and 1 are magic numbers because the structure of "previous text" cannot be broken. (<|im_start|>previous text\n ... <|im_end|>)
            if cut_start_idx <= cut_end_idx:
                prev_generated_ids, past_key_values = prune_id_and_kv_cache(prev_generated_ids, past_key_values, cut_start_idx, cut_end_idx)
        prev_generated_ids, past_key_values = contiguous_id_and_kv(prev_generated_ids, past_key_values)
    
    return past_key_values, prev_generated_ids, recent_video_window_clips, recent_pixel_values_videos

def printq(*args, quiet=False, **kwargs): 
    """Use like print; when quiet=True, suppress output.""" 
    if not quiet: 
        print(*args, **kwargs)

# -----------------------------------------------------------------
# Data loading functions (from qwen3vl.py)
# -----------------------------------------------------------------
def load_goal_json(goal_json_path: str) -> Dict[str, Any]:
    with open(goal_json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_prompt(prompt_path: str) -> str:
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read().strip()


def build_video_path(video_root: str, league: str, season: str, match: str, half: str) -> str:
    return os.path.join(video_root, league, season, match, f"{half}_720p.mkv")


def flatten_samples(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Flatten hierarchical data into a sample list; each sample contains:
    league, season, match, half, start, end, gt
    """
    rows: List[Dict[str, Any]] = []
    for league, seasons in data.items():
        for season, matches in seasons.items():
            for match, halfs in matches.items():
                for half, items in halfs.items():
                    for it in items:
                        rows.append(
                            {
                                "league": league,
                                "season": season,
                                "match": match,
                                "half": str(half),
                                "start": float(it.get("start", 0.0)),
                                "end": float(it.get("end", 0.0)),
                                "gt": (it.get("commentary") or "").strip(),
                            }
                        )
    return rows


# -----------------------------------------------------------------
# Evaluation metrics (from qwen3vl.py)
# -----------------------------------------------------------------
def tokenize(text: str) -> List[str]:
    # Simple tokenizer for mostly English text; character splitting can be used for Chinese
    return [t for t in text.strip().lower().split() if t]


def ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    return [tuple(tokens[i : i + n]) for i in range(0, max(0, len(tokens) - n + 1))]


def bleu_score(references: List[str], candidates: List[str], max_n: int = 4) -> float:
    """
    Simplified BLEU: compute BLEU-4 for each sentence and average the scores.
    Use add-1 smoothing.
    """
    eps = 1e-9
    scores: List[float] = []
    for ref, cand in zip(references, candidates):
        ref_toks = tokenize(ref)
        cand_toks = tokenize(cand)
        if not cand_toks:
            scores.append(0.0)
            continue

        # modified precision with add-1 smoothing
        log_p_ns: List[float] = []
        for n in range(1, max_n + 1):
            ref_counts = Counter(ngrams(ref_toks, n))
            cand_ngrams = ngrams(cand_toks, n)
            cand_counts = Counter(cand_ngrams)
            match = 0
            for ng, cnt in cand_counts.items():
                match += min(cnt, ref_counts.get(ng, 0))
            p_n = (match + 1.0) / (sum(cand_counts.values()) + 1.0)  # add-1
            log_p_ns.append(math.log(max(p_n, eps)))

        # BP
        ref_len = len(ref_toks)
        cand_len = len(cand_toks)
        if cand_len == 0:
            bp = 0.0
        elif cand_len > ref_len:
            bp = 1.0
        else:
            bp = math.exp(1 - float(ref_len) / float(cand_len))

        score = bp * math.exp(sum(log_p_ns) / max_n)
        scores.append(score)
    return float(sum(scores) / max(1, len(scores)))


def meteor_score(references: List[str], candidates: List[str]) -> float:
    """
    Simplified METEOR: compute word-level precision/recall Fmean with a higher recall weight, without stemming, synonym alignment, or fragmentation penalty.
    Fmean = (10 * P * R) / (R + 9 * P + 1e-9)
    """
    scores: List[float] = []
    for ref, cand in zip(references, candidates):
        ref_toks = tokenize(ref)
        cand_toks = tokenize(cand)
        if not cand_toks:
            scores.append(0.0)
            continue
        ref_counts = Counter(ref_toks)
        cand_counts = Counter(cand_toks)
        match = 0
        for w, c in cand_counts.items():
            match += min(c, ref_counts.get(w, 0))
        P = match / max(1, sum(cand_counts.values()))
        R = match / max(1, sum(ref_counts.values()))
        Fmean = (10.0 * P * R) / (R + 9.0 * P + 1e-9)
        scores.append(Fmean)
    return float(sum(scores) / max(1, len(scores)))


def lcs_length(a: List[str], b: List[str]) -> int:
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        ai = a[i - 1]
        for j in range(1, n + 1):
            if ai == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def rouge_l_score(references: List[str], candidates: List[str], beta: float = 1.2) -> float:
    """
    Follow the ROUGE-L definition and use an LCS-based F-score.
    """
    scores: List[float] = []
    for ref, cand in zip(references, candidates):
        ref_toks = tokenize(ref)
        cand_toks = tokenize(cand)
        if not cand_toks or not ref_toks:
            scores.append(0.0)
            continue
        lcs = lcs_length(ref_toks, cand_toks)
        prec = lcs / max(1, len(cand_toks))
        rec = lcs / max(1, len(ref_toks))
        if prec == 0 and rec == 0:
            scores.append(0.0)
            continue
        num = (1 + beta * beta) * prec * rec
        den = rec + beta * beta * prec + 1e-9
        scores.append(num / den)
    return float(sum(scores) / max(1, len(scores)))


def build_idf(refs: List[str], max_n: int = 4) -> Dict[Tuple[str, ...], float]:
    """
    Build n-gram IDF values from the reference corpus.
    """
    N = len(refs)
    df: Counter = Counter()
    for ref in refs:
        toks = tokenize(ref)
        ngram_set = set()
        for n in range(1, max_n + 1):
            ngram_set.update(ngrams(toks, n))
        df.update(ngram_set)
    idf = {}
    for ng, d in df.items():
        idf[ng] = math.log((N + 1) / (d + 1)) + 1.0  # Add-1 smoothing
    return idf


def cider_score(references: List[str], candidates: List[str], max_n: int = 4) -> float:
    """
    Simplified CIDEr: average TF-IDF cosine similarity and multiply by 10 to match the paper scale.
    """
    idf = build_idf(references, max_n=max_n)

    def tfidf_vec(text: str) -> Dict[Tuple[str, ...], float]:
        toks = tokenize(text)
        vec: Dict[Tuple[str, ...], float] = defaultdict(float)
        for n in range(1, max_n + 1):
            counts = Counter(ngrams(toks, n))
            total_ngrams = sum(counts.values())
            if total_ngrams == 0:
                continue
            for ng, cnt in counts.items():
                # TF: term frequency (count / total ngrams for this n)
                # IDF: inverse document frequency
                tf = cnt / total_ngrams
                vec[ng] += tf * idf.get(ng, 0.0)
        return vec

    def cosine(a: Dict[Tuple[str, ...], float], b: Dict[Tuple[str, ...], float]) -> float:
        if not a or not b:
            return 0.0
        common = set(a.keys()) & set(b.keys())
        num = sum(a[k] * b[k] for k in common)
        na = math.sqrt(sum(v * v for v in a.values()))
        nb = math.sqrt(sum(v * v for v in b.values()))
        if na == 0 or nb == 0:
            return 0.0
        return num / (na * nb)

    scores: List[float] = []
    for ref, cand in zip(references, candidates):
        vr = tfidf_vec(ref)
        vc = tfidf_vec(cand)
        scores.append(cosine(vr, vc) * 10.0)
    return float(sum(scores) / max(1, len(scores)))

# -----------------------------------------------------------------
# Multi-GPU worker function
# -----------------------------------------------------------------
def streaming_worker(
    device_id: int,
    model_path: str,
    model_base: str,
    rows: List[Dict],
    video_root: str,
    prompt: str,
    window_size: int,
    chunk_duration: int,
    text_round: int,
    previous_text: str,
    pos_mode: str,
    all_text: bool,
    recompute: bool,
    text_sink: Optional[int],
    text_sliding_window: Optional[int],
    temperature: float,
    repetition_penalty: float,
    num_workers: int,
    print_interval: int,
    start_idx: int,
):
    """StreamingVLM worker function that handles data on the selected GPU"""
    # Set the device
    torch.cuda.set_device(device_id)
    device = f"cuda:{device_id}"
    
    # Load the model（each worker loads on its own GPU）
    model, processor = load_model_and_processor(model_path, model_base, device_id=device_id)
    set_token_id_map(processor.tokenizer)
    
    # Assign samples to this device
    idxs = list(range(len(rows)))
    idxs_on_device = idxs[device_id::num_workers]
    
    references: List[str] = []
    candidates: List[str] = []
    results: List[Dict] = []
    num_skipped = 0
    total_samples = len(rows)
    
    for local_idx, idx in enumerate(idxs_on_device):
        r = rows[idx]
        video_path = build_video_path(video_root, r["league"], r["season"], r["match"], r["half"])
        if not os.path.exists(video_path):
            num_skipped += 1
            continue
        
        # Compute the time range
        start_time = r["start"]
        end_time = r["end"]
        duration = end_time - start_time
        
        if duration <= 0:
            num_skipped += 1
            continue
        
        # For batch evaluation, process the specific time range [start, end)
        actual_duration = min(duration, TOTAL_VIDEO_DURATION)
        
        # Call the inference function
        pred_responses = streaming_inference(
            model_path=model_path,
            video_path=video_path,
            output_dir=None,
            model_base=model_base,
            model=model,
            processor=processor,
            window_size=window_size,
            chunk_duration=chunk_duration,
            text_round=text_round,
            previous_text=previous_text,
            pos_mode=pos_mode,
            all_text=all_text,
            skip_first_chunk=0,
            recompute=recompute,
            text_sink=text_sink,
            text_sliding_window=text_sliding_window,
            temperature=temperature,
            duration=actual_duration,
            query=prompt,
            repetition_penalty=repetition_penalty,
            quiet=True,
            emit_json=False,
            time_test=False,
            video_start_offset=start_time,
        )
        
        # Merge all responses as the prediction
        if pred_responses:
            responses = [resp.get('response', '').strip() for resp in pred_responses if resp.get('response', '').strip()]
            if responses:
                pred_text = " ".join(responses).strip()
                if pred_text and not pred_text[-1] in '.!?':
                    pred_text += "."
            else:
                pred_text = ""
        else:
            pred_text = ""
        
        references.append(r["gt"])
        candidates.append(pred_text)
        
        results.append({
            "league": r["league"],
            "season": r["season"],
            "match": r["match"],
            "half": r["half"],
            "start": r["start"],
            "end": r["end"],
            "gt": r["gt"],
            "pred": pred_text,
            "video_path": video_path,
        })
        
        # Print ground truth and prediction every fixed number of samples, including the first sample
        if print_interval > 0 and ((local_idx + 1) % print_interval == 0 or local_idx == 0):
            print(f"\n{'='*80}")
            print(f"[GPU {device_id}] [sample {start_idx + idx + 1}/{total_samples}] {r['league']}/{r['season']}/{r['match']} {r['half']} [{start_time:.1f}s-{end_time:.1f}s]")
            print(f"{'='*80}")
            print(f"GT:   {r['gt']}")
            print(f"Pred: {pred_text}")
            print(f"{'='*80}\n")
    
    return {
        "references": references,
        "candidates": candidates,
        "results": results,
        "num_skipped": num_skipped,
    }

# -----------------------------------------------------------------
# Main logic
# -----------------------------------------------------------------
def streaming_inference(model_path="", 
                        video_path="", output_dir=None, 
                        model_base = 'Qwen2_5',
                        model = None,
                        processor = None,
                        window_size = DEFAULT_WINDOW_SIZE, 
                        chunk_duration = DEFAULT_CHUNK_DURATION, 
                        text_round = DEFAULT_TEXT_ROUND, 
                        previous_text = "", 
                        test_data_json = None, 
                        test_data_idx = None, 
                        pos_mode = "shrink", 
                        all_text = False, # All PEs are 1D to adapt to livecc's buggy training code
                        skip_first_chunk = 0,
                        recompute = False,
                        gt_json = None,
                        gt_idx = None,
                        text_sink = None,
                        text_sliding_window = None,
                        temperature = DEFAULT_TEMPERATURE,
                        duration = TOTAL_VIDEO_DURATION,
                        query = "Commentate on this match",
                        repetition_penalty=DEFAULT_REPETITION_PENALTY,
                        quiet=False,
                        emit_json=False,
                        time_test = False,
                        video_start_offset = 0.0,  # Video read start offset in seconds
                        ): 
    def _sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    NUM_WINDOW_TO_KEEP = window_size // chunk_duration
    assert NUM_WINDOW_TO_KEEP * chunk_duration == window_size, \
        "window_size must be divisible by chunk_duration"

    # Load model
    device = model.device if model is not None else "cuda" if torch.cuda.is_available() else "cpu"
    streaming_args = StreamingArgs(pos_mode=pos_mode, all_text=all_text)

    if model is None or processor is None:
        model, processor = load_model_and_processor(model_path, model_base)
    else: 
        if model_base == 'Qwen2_5':
            model = convert_qwen2_5_to_streaming(model)
        elif model_base == 'Qwen2':
            model = convert_qwen2_to_streaming(model)
    
    assistant_start_bias = len(processor(text="<|im_start|>assistant\n")['input_ids'][0])
    assistant_end_bias = len(processor(text=" ...<|im_end|>")['input_ids'][0])

    # Load GT
    use_gt = False
    if gt_json is not None:
        use_gt = True
        with open(gt_json, 'r') as f:
            for i,line in enumerate(f):
                if i == gt_idx:
                    gt_dict = json.loads(line)
                    break
    
                # f[gt_idx] 
    # Load from dataset #########################################################
    from_dataset = False
    if test_data_json is not None:
        assert test_data_idx is not None
        dataset = LMMDataset(
            annotation_paths=[
               test_data_json
            ], 
            processor=processor, # Pass in the initialized processor
            with_context=False, # Do not use context
            return_conversation=True,
        )[test_data_idx]
        from_dataset = True

    if not from_dataset:
        print("os.getenv('DATASET_PATH', ", os.getenv('DATASET_PATH', ""), ")")
        print("os.path.join(os.getenv('DATASET_PATH', ""), video_path)", os.path.join(os.getenv('DATASET_PATH', ""), video_path))

        # Try different path combinations to find the video
        video_reader = None
        if os.path.exists(video_path):
            video_reader, resized_height, resized_width = get_smart_resized_video_reader(video_path, MAX_PIXELS)
        elif os.path.exists(os.path.join(os.getenv('DATASET_PATH', ""), video_path)):
            video_path = os.path.join(os.environ['DATASET_PATH'], video_path)
            video_reader, resized_height, resized_width = get_smart_resized_video_reader(video_path, MAX_PIXELS)
        elif os.path.exists(os.path.join(os.getenv('EVAL_DATASET_PATH', ""), video_path)):
            video_path = os.path.join(os.environ['EVAL_DATASET_PATH'], video_path)
            video_reader, resized_height, resized_width = get_smart_resized_video_reader(video_path, MAX_PIXELS)
        elif os.getenv('UPLOAD_DATASET_PATH') and os.path.exists(os.path.join(os.getenv('UPLOAD_DATASET_PATH', ""), video_path)):
            # Handle uploaded videos with UPLOAD_DATASET_PATH
            video_path = os.path.join(os.environ['UPLOAD_DATASET_PATH'], video_path)
            video_reader, resized_height, resized_width = get_smart_resized_video_reader(video_path, MAX_PIXELS)
        else:
            # If none of the above paths work, raise an error with debug info
            raise FileNotFoundError(
                f"Video file not found. Tried the following paths:\n"
                f"  1. {video_path} (exists: {os.path.exists(video_path)})\n"
                f"  2. {os.path.join(os.getenv('DATASET_PATH', ''), video_path)} (exists: {os.path.exists(os.path.join(os.getenv('DATASET_PATH', ''), video_path))})\n"
                f"  3. {os.path.join(os.getenv('EVAL_DATASET_PATH', ''), video_path)} (exists: {os.path.exists(os.path.join(os.getenv('EVAL_DATASET_PATH', ''), video_path))})\n"
                f"  4. {os.path.join(os.getenv('UPLOAD_DATASET_PATH', ''), video_path) if os.getenv('UPLOAD_DATASET_PATH') else 'UPLOAD_DATASET_PATH not set'}"
            )

        video_reader.get_frame_timestamp(0) # array([0.0, 0.01668333], dtype=float32)
        # video_pts = torch.from_numpy(video_reader._frame_pts[:, 1]) # End timestamp for each frame
        printq(f"resized_height: {resized_height}, resized_width: {resized_width}", quiet=quiet)
    ##############################################################################

    past_key_values = None
    full_conversation_history = []
    prev_generated_ids = None  # Position embeddings must be recomputed every round, so we must keep this
    recent_video_window_clips = [] # Qwen: video before passing through the vision tower
    recent_pixel_values_videos = [] # 
    num_chunks = int((duration + chunk_duration - 1) // chunk_duration)

    responses = []
    ground_truths = []
    printq(f"num_chunks: {num_chunks}", quiet=quiet)
    if time_test:
        time_results = []
    for i in range(num_chunks):
        _sync()
        loop_start = time.perf_counter()
        section_time = {k: 0.0 for k in ['PKV', 'CHECK', 'VIDEO', 'INPUT', 'GEN', 'POST']}
    
        start_time = (i + skip_first_chunk) * chunk_duration + video_start_offset 

        ########################## Handle past_key_values ###################################
        _sync(); _t = time.perf_counter()

        past_key_values, prev_generated_ids, recent_video_window_clips, recent_pixel_values_videos = process_past_kv(past_key_values, i, 
                                                                                         text_round=text_round, visual_round=window_size, 
                                                                                         full_conversation_history=full_conversation_history, 
                                                                                         prev_generated_ids=prev_generated_ids, 
                                                                                         assistant_start_bias=assistant_start_bias, 
                                                                                         assistant_end_bias=assistant_end_bias, 
                                                                                         recent_video_window_clips=recent_video_window_clips,
                                                                                         recent_pixel_values_videos=recent_pixel_values_videos,
                                                                                         text_sink=text_sink,
                                                                                         text_sliding_window=text_sliding_window)
        _sync();section_time['PKV'] += (time.perf_counter() - _t)

        _sync(); _t = time.perf_counter()

        try:
            if from_dataset:
                current_video_chunk = dataset['conversation'][2*i+1]['content'][1]['video'] 
                video_start = dataset['start_timestamp']  # Start time of this segment itself
                start_time = video_start + start_time
                ground_truths.append({'ground_truth':dataset['conversation'][2*i+2]['content'][0]['text'] , 'start_time':start_time, 'end_time':start_time+chunk_duration})
            else:
                    # Compute actual time relative to the video start
                    actual_video_start = start_time
                    ele = {'type': 'video', 'video': video_path, 'video_start': actual_video_start, 'video_end': (actual_video_start+chunk_duration)}
                    current_video_chunk, sample_fps, clip_pts = _read_video_decord_plus(ele,  return_pts=True, strict_fps=True, only_get_last_frame=int(chunk_duration*FPS), vr=video_reader)
                    current_video_chunk = _spatial_resize_video(current_video_chunk)
        except Exception as e:
            print(f"Error in streaming_inference: {e}")
            break
        recent_video_window_clips.append(current_video_chunk)
        _sync();section_time['VIDEO'] += (time.perf_counter() - _t)

        ########################## Build input ##############################################
        _sync(); _t = time.perf_counter()
        if i == 0:
            prompt = f'Time={start_time:.1f}-{start_time+chunk_duration:.1f}s'
            if from_dataset: # Load from livecc dataset
                previous_text = dataset['conversation'][0]['content']
                user_content = [
                    {"type": "text", "text": prompt},
                    dataset['conversation'][1]['content'][1], # video
                    {"type": "text", "text": query},
                ]
            else: # Load from local file
                if video_start_offset > 0:
                    user_content = [
                        {"type": "text", "text": prompt},
                        {"type": "video", "video": video_path, "start": video_start_offset, "duration": chunk_duration},
                        {"type": "text", "text": query},
                    ]
                else:
                    user_content = [
                        {"type": "text", "text": prompt},
                        {"type": "video", "video": video_path },
                        {"type": "text", "text": query},
                    ]
            full_conversation_history = [
                {"role": "previous text", "content": previous_text},
                {"role": "user", "content": user_content}
                ]
            text = processor.apply_chat_template(full_conversation_history, tokenize=False, add_generation_prompt=True)
            
        else:
            prompt = f'Time={start_time:.1f}-{start_time+chunk_duration:.1f}s'

            if from_dataset:
                user_content = [
                    {"type": "text", "text": prompt},
                    dataset['conversation'][2*i+1]['content'][1], # video
                ]
            else:
                user_content = [
                    {"type": "text", "text": prompt},
                    {"type": "video", "video": video_path, "start": start_time,
                    "duration": chunk_duration},
                ]
            full_conversation_history.append({"role": "user", "content": user_content})
            text = processor.apply_chat_template([{"role": "user", "content": user_content}], tokenize=False, add_generation_prompt=True)
            text = '\n' + text[SYSTEM_PROMPT_OFFSET:]

        inputs = processor(
            text=[text],
            videos=recent_video_window_clips[-1],
            padding=True,
            return_tensors="pt",
        ).to(device)
     
        if prev_generated_ids is not None:
            # Because the model's last generated token is <|im_end|>, a new round will add an extra "\n".
            # We must keep that "\n", hence the -1.
            # Special case: (when only one round of text is kept) if the last assistant was removed,
            # then the last token of the previous user will be "\n"
            if prev_generated_ids[:,-1].item()!=TOKEN_IDS["\n"]:
                inputs['input_ids'] = torch.cat([prev_generated_ids,inputs['input_ids']],dim=1) 
            else:
                inputs['input_ids'] = torch.cat([prev_generated_ids,inputs['input_ids'][:,1:]],dim=1) 
            inputs['attention_mask'] = torch.ones_like(inputs['input_ids'])

        recent_pixel_values_videos.append(inputs['pixel_values_videos'])
        if streaming_args.pos_mode == "shrink" or recompute:
            streaming_args.input_ids = inputs['input_ids']
            if i==0:
                streaming_args.video_grid_thw = inputs['video_grid_thw']
                streaming_args.second_per_grid_ts = inputs['second_per_grid_ts'] if 'second_per_grid_ts' in inputs else None
            else:
                streaming_args.video_grid_thw = torch.cat([streaming_args.video_grid_thw,inputs['video_grid_thw']],dim=0)
                streaming_args.second_per_grid_ts = torch.cat([streaming_args.second_per_grid_ts,inputs['second_per_grid_ts']],dim=0) if 'second_per_grid_ts' in inputs else None

        current_input_len = inputs['input_ids'].shape[1]

        _sync();section_time['INPUT'] += (time.perf_counter() - _t)
        ########################## Model generation (commentary) ############################
        _sync(); _t = time.perf_counter()
        if recompute:
            if i>0:
                inputs['pixel_values_videos'] = torch.cat(recent_pixel_values_videos,dim=0)
                inputs['video_grid_thw'] = streaming_args.video_grid_thw[-len(recent_pixel_values_videos):]
                inputs['second_per_grid_ts'] = streaming_args.second_per_grid_ts[-len(recent_pixel_values_videos):]
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_TOKEN_PER_DURATION,
                use_cache=True,
                return_dict_in_generate=True,
                do_sample=True,
                repetition_penalty=repetition_penalty,
                streaming_args=streaming_args,
                pad_token_id=151645,
                temperature=temperature,
            )
        else:
            outputs = model.generate(
                **inputs,
                past_key_values=past_key_values,
                max_new_tokens=MAX_TOKEN_PER_DURATION,
                use_cache=True,
                return_dict_in_generate=True,
                do_sample=True,
                repetition_penalty=repetition_penalty,
                streaming_args=streaming_args,
                pad_token_id=151645,
                temperature=temperature,
            )
        _sync();section_time['GEN'] += (time.perf_counter() - _t)

        ########################## Post-process input_ids ##################################
        _sync(); _t = time.perf_counter()
        generated_ids = outputs.sequences
        if generated_ids[0,-1].item() != 151645: # <|im_end|>
            # If the last token is not <|im_end|>, append it to keep the format consistent
            generated_ids = torch.cat([generated_ids, torch.tensor([[151645]], device = device)],dim=1)
        newly_generated_ids = generated_ids[:, current_input_len:]

        response = processor.batch_decode(newly_generated_ids, skip_special_tokens=True)[0]
        responses.append({'response':response[:-4], 'start_time':start_time, 'end_time':start_time+chunk_duration})

        time_key = f'Time={start_time:.1f}-{start_time+chunk_duration:.1f}s'
        past_key_values = outputs.past_key_values
        hms_start = time.strftime('%H:%M:%S', time.gmtime(int(start_time)))
        hms_end = time.strftime('%H:%M:%S', time.gmtime(int(start_time + chunk_duration)))
        printq(f"Time={hms_start}-{hms_end}: \033[1m\033[34m{response}\033[0m", f'past_key_values: {past_key_values.get_seq_length() if past_key_values is not None else 0}', flush=True, quiet=quiet)
        if emit_json:
            try:
                if emit_json:
                    sys.stdout.write(json.dumps({
                        "type": "segment",
                        "start": float(start_time),
                        "end": float(start_time + chunk_duration),
                        "text": response[:-4]
                    }, ensure_ascii=False) + "\n")
                    sys.stdout.flush()
            except Exception as _e:
                pass
        prev_generated_ids = generated_ids.clone()
        if use_gt and gt_dict[time_key]['phrase'] != response:
            printq(f"Decoded text [{response}] is incorrect. Use ground truth [{gt_dict[time_key]['phrase']}] instead", quiet=quiet)
            prune_id_and_kv_cache(prev_generated_ids,past_key_values,current_input_len,past_key_values.get_seq_length()-1)
            response = gt_dict[time_key]['phrase']
            prev_generated_ids = torch.cat([inputs['input_ids'], torch.tensor(processor(text=[gt_dict[time_key]['phrase']+"<|im_end|>\n"])['input_ids'],device = device)],dim=1)
        assistant_turn = {"role": "assistant", "content": response}
        full_conversation_history.append(assistant_turn)
            

        _sync();section_time['POST'] += (time.perf_counter() - _t)

        # ------------------------- Print profiling results -----------------
        _sync()
        loop_total = time.perf_counter() - loop_start
        printq(
            f"[Loop {i}] total={loop_total:.3f}s | "
            f"PKV={section_time['PKV']:.3f}s | "
            f"CHECK={section_time['CHECK']:.3f}s | "
            f"VIDEO={section_time['VIDEO']:.3f}s | "
            f"INPUT={section_time['INPUT']:.3f}s | "
            f"GEN={section_time['GEN']:.3f}s | "
            f"POST={section_time['POST']:.3f}s",
            flush=True,
            quiet=quiet,
        )
        if time_test:
            time_results.append(section_time)
        # ============================================================
    if time_test:
        return time_results
    return responses


if __name__ == "__main__":
    default_video = "Baidu_NBA_EN/NBA.2015.06.09.Warriors.vs.Cavaliers.720p.HDTV.60fps.x264.mp4"

    args = argparse.ArgumentParser()
    args.add_argument("--pos_mode", type=str, default="shrink", choices=["append", "shrink"])
    args.add_argument("--all_text", action="store_true", default=False)
    args.add_argument("--model_path", type=str, default="mit-han-lab/StreamingVLM")
    args.add_argument("--model_base", type=str, choices=["Qwen2_5", "Qwen2", "VILA"], default="Qwen2_5")
    args.add_argument("--video_path", type=str, default=default_video)
    args.add_argument("--window_size", type=int, default=DEFAULT_WINDOW_SIZE)
    args.add_argument("--chunk_duration", type=int, default=DEFAULT_CHUNK_DURATION)
    args.add_argument("--text_round", type=int, default=DEFAULT_TEXT_ROUND)
    args.add_argument("--previous_text", type=str, default="THis is a video with title 'Golden State Warriors’ BEST PLAYS from the 2024-25 NBA Season'. This is Highlight of Warriors. Curry is going to score a 3-pointer")
    args.add_argument("--skip_first_chunk", type=int, default=0)
    args.add_argument("--recompute", action='store_true')
    args.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    args.add_argument("--repetition_penalty", type=float, default=DEFAULT_REPETITION_PENALTY)
    # Both None: no truncation
    # One None and the other not: treat None as 0 (keep nothing), so both are applied
    # Both non-None: apply both
    args.add_argument("--text_sink", type=int, default=DEFAULT_TEXT_SINK)
    args.add_argument("--text_sliding_window", type=int, default=DEFAULT_TEXT_SLIDING_WINDOW)

    args.add_argument("--output_dir", type=str)
    args.add_argument("--emit_json", action="store_true", help="Emit per-second JSON lines to stdout")

    args.add_argument("--test_data_json", type=str, default=None)
    args.add_argument("--test_data_idx", type=int, default=None)

    args.add_argument("--gt_json", type=str, default=None)
    args.add_argument("--gt_idx", type=int, default=0)
    
    # Upstream implementation note.
    args.add_argument("--goal_json", type=str, default="eval/GOAL/caption/goal.json", help="Path to goal.json; enables batch evaluation when provided")
    args.add_argument("--video_root", type=str, default="/path/to/SoccerNet/videos", help="Video root directory")
    args.add_argument("--prompt_file", type=str, default="eval/GOAL/caption/prompt.txt", help="Prompt file path")
    args.add_argument("--max_samples", type=int, default=0, help="Optional: evaluate only the first N samples; 0 means all")
    args.add_argument("--eval_output", type=str, default=None, help="Optional: JSONL path for saved predictions")
    args.add_argument("--print_interval", type=int, default=10, help="Print ground truth and predictions every N samples")
    args.add_argument("--num_workers", type=int, default=8, help="Number of GPUs to use; 0 means single GPU, >0 means multi-GPU parallelism")

    
    args = args.parse_args()

    # Run batch evaluation when goal_json is provided
    if args.goal_json is not None:
        # Batch evaluation mode
        data = load_goal_json(args.goal_json)
        prompt = load_prompt(args.prompt_file) if args.prompt_file else "Commentate on this match"
        rows = flatten_samples(data)
        if args.max_samples and args.max_samples > 0:
            rows = rows[: args.max_samples]
        
        # Determine the number of GPUs to use
        if args.num_workers > 0:
            num_workers = min(args.num_workers, torch.cuda.device_count())
        else:
            num_workers = 1
        
        if num_workers == 0:
            print("No GPU detected; using CPU mode")
            num_workers = 1
            use_multiprocessing = False
        else:
            use_multiprocessing = num_workers > 1
            if use_multiprocessing:
                print(f"Using {num_workers} GPUs for parallel inference")
            else:
                print("Using single-GPU mode")
        
        t0 = time.time()
        total_samples = len(rows)
        
        if use_multiprocessing:
            # Multi-GPU parallel processing
            multiprocessing.set_start_method('spawn', force=True)
            
            # Create the worker function
            worker_fn = partial(
                streaming_worker,
                model_path=args.model_path,
                model_base=args.model_base,
                rows=rows,
                video_root=args.video_root,
                prompt=prompt,
                window_size=args.window_size,
                chunk_duration=args.chunk_duration,
                text_round=args.text_round,
                previous_text=args.previous_text,
                pos_mode=args.pos_mode,
                all_text=args.all_text,
                recompute=args.recompute,
                text_sink=args.text_sink,
                text_sliding_window=args.text_sliding_window,
                temperature=args.temperature,
                repetition_penalty=args.repetition_penalty,
                num_workers=num_workers,
                print_interval=args.print_interval,
                start_idx=0,
            )
            
            # Use multiprocessing
            worker_results = local_mp(
                list(range(num_workers)),
                worker_fn,
                desc="streaming_inference",
                num_workers=num_workers
            )
            
            # Merge all worker results
            all_references: List[str] = []
            all_candidates: List[str] = []
            all_results: List[Dict] = []
            total_skipped = 0
            
            for result in worker_results:
                all_references.extend(result["references"])
                all_candidates.extend(result["candidates"])
                all_results.extend(result["results"])
                total_skipped += result["num_skipped"]
            
            references = all_references
            candidates = all_candidates
            results = all_results
            num_skipped = total_skipped
        else:
            # Single-GPU or CPU mode
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            model, processor = load_model_and_processor(args.model_path, args.model_base)
            
            references: List[str] = []
            candidates: List[str] = []
            results: List[Dict] = []
            num_skipped = 0
            
            for idx, r in enumerate(rows):
                video_path = build_video_path(args.video_root, r["league"], r["season"], r["match"], r["half"])
                if not os.path.exists(video_path):
                    num_skipped += 1
                    print(f"[{idx+1}/{total_samples}] Skipping: video file does not exist {video_path}")
                    continue
                
                # Compute the time range
                start_time = r["start"]
                end_time = r["end"]
                duration = end_time - start_time
                
                if duration <= 0:
                    num_skipped += 1
                    print(f"[{idx+1}/{total_samples}] Skipping: invalid time range {start_time}-{end_time}")
                    continue
                
                # For batch evaluation, process the specific time range [start, end)
                # Upstream implementation note.
                actual_duration = min(duration, TOTAL_VIDEO_DURATION)
                
                print(f"[{idx+1}/{total_samples}] Processing: {r['league']}/{r['season']}/{r['match']} {r['half']} [{start_time:.1f}s-{end_time:.1f}s]")
                
                # Upstream implementation note.
                pred_responses = streaming_inference(
                    model_path=args.model_path,
                    video_path=video_path,
                    output_dir=None,  # Do not save per-video outputs during batch evaluation
                    model_base=args.model_base,
                    model=model,
                    processor=processor,
                    window_size=args.window_size,
                    chunk_duration=args.chunk_duration,
                    text_round=args.text_round,
                    previous_text=args.previous_text,
                    pos_mode=args.pos_mode,
                    all_text=args.all_text,
                    skip_first_chunk=0,  # Start from the beginning of the time range
                    recompute=args.recompute,
                    text_sink=args.text_sink,
                    text_sliding_window=args.text_sliding_window,
                    temperature=args.temperature,
                    duration=actual_duration,
                    query=prompt,
                    repetition_penalty=args.repetition_penalty,
                    quiet=True,
                    emit_json=False,
                    time_test=False,
                    video_start_offset=start_time,  # Start reading the video from the specified time
                )
                
                # Merge all responses as the prediction
                # For the full [start, end) range, merge all chunk responses into one complete commentary
                # Each sample corresponds to one time range and one complete commentary, so multiple chunk responses must be merged
                if pred_responses:
                    # Collect all non-empty responses; streaming_inference has already stripped response[:-4] and removed " ..."
                    responses = [resp.get('response', '').strip() for resp in pred_responses if resp.get('response', '').strip()]
                    if responses:
                        # Merge all responses into one complete commentary
                        # Join chunk responses with spaces to form one complete commentary sentence
                        pred_text = " ".join(responses).strip()
                        # Upstream implementation note.
                        if pred_text and not pred_text[-1] in '.!?':
                            pred_text += "."
                    else:
                        pred_text = ""
                else:
                    pred_text = ""
                
                references.append(r["gt"])
                candidates.append(pred_text)
                
                results.append({
                    "league": r["league"],
                    "season": r["season"],
                    "match": r["match"],
                    "half": r["half"],
                    "start": r["start"],
                    "end": r["end"],
                    "gt": r["gt"],
                    "pred": pred_text,
                    "video_path": video_path,
                })
                
                # Print ground truth and prediction every fixed number of samples, including the first sample
                if args.print_interval > 0 and ((idx + 1) % args.print_interval == 0 or idx == 0):
                    print(f"\n{'='*80}")
                    print(f"[sample {idx + 1}/{total_samples}] {r['league']}/{r['season']}/{r['match']} {r['half']} [{start_time:.1f}s-{end_time:.1f}s]")
                    print(f"{'='*80}")
                    print(f"GT:   {r['gt']}")
                    print(f"Pred: {pred_text}")
                    print(f"{'='*80}\n")
                
                # Show progress
                if (idx + 1) % 10 == 0:
                    print(f"Processed {idx + 1}/{total_samples} samples (skipped {num_skipped})")
        
        # Save results
        if args.eval_output:
            with open(args.eval_output, "w", encoding="utf-8") as f:
                for res in results:
                    f.write(json.dumps(res, ensure_ascii=False) + "\n")
            print(f"Predictions saved to: {args.eval_output}")
        
        # Filter empty predictions for more robust evaluation
        paired = [(ref, cand) for ref, cand in zip(references, candidates) if cand]
        if not paired:
            print("No valid predictions; metrics cannot be computed.")
        else:
            refs, cands = zip(*paired)
            refs = list(refs)
            cands = list(cands)
            
            # Compute metrics
            bleu = bleu_score(refs, cands)
            meteor = meteor_score(refs, cands)
            rouge_l = rouge_l_score(refs, cands)
            cider = cider_score(refs, cands)
            
            print("\n" + "="*50)
            print("Evaluation complete:")
            print(f"- BLEU:   {bleu:.4f}")
            print(f"- METEOR: {meteor:.4f}")
            print(f"- ROUGE-L:{rouge_l:.4f}")
            print(f"- CIDEr:  {cider:.4f}")
            print(f"Elapsed: {time.time() - t0:.2f}s, samples: {len(paired)}, skipped: {num_skipped}")
            print("="*50)
    else:
        # Single-video inference mode using the original logic
        if args.output_dir is None:
            os.makedirs("output", exist_ok=True)
            args.output_dir = f"output/{args.model_path.replace('/','_')}_viswin{args.window_size}_txtwin{args.text_round}_prvsink{args.text_sink}_prvwin{args.text_sliding_window}_tprt{args.temperature}.json"
        streaming_inference(**args.__dict__)
