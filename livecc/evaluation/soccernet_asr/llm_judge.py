import os
import json
import argparse
import torch
from functools import partial
from typing import Dict, List, Tuple, Any
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.multiprocessor import local_mt


def load_qwen3_model(model_path: str, device: str = "cuda:0"):
    """Load the Qwen3 model"""
    print(f"Loading model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    
    print(f"Model loaded on {device}")
    return model, tokenizer


def load_test_json(test_json_path: str) -> Dict[str, Any]:
    """Load the test.json file as ground truth."""
    print(f"Loading ground truth: {test_json_path}")
    with open(test_json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_pred_json(pred_json_path: str) -> Dict[str, Any]:
    """Load the prediction JSON file"""
    print(f"Loading predictions: {pred_json_path}")
    with open(pred_json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def flatten_to_segments(data: Dict[str, Any]) -> Dict[str, str]:
    """
    Flatten nested JSON into a segment dictionary
    key: "{league}|{season}|{match}|{half}|{segment_index}" (string format for JSON serialization)
    value: commentary
    """
    segments = {}
    for league, seasons in data.items():
        for season, matches in seasons.items():
            for match, halfs in matches.items():
                for half, segments_list in halfs.items():
                    for seg_idx, segment in enumerate(segments_list):
                        key = f"{league}|{season}|{match}|{half}|{seg_idx}"
                        segments[key] = segment.get("commentary", "").strip()
    return segments


def judge_ab(model, tokenizer, a_id_with_pred, b_id_with_pred, gt_asr):
    """Use a local Qwen3 model for A-vs-B judging"""
    a_id, a_pred = a_id_with_pred
    b_id, b_pred = b_id_with_pred
    
    ab_prompt = (
        'You are an expert in video commentary. '
        'Your task is to review two commentaries (Commentary A and Commentary B), and select the one that better aligns with the human commentary. '
        'You should consider the criteria:\n'
        '1. Semantic Alignment: The commentary should convey the same meaning, details, and key points as the human commentary.\n'
        'If the above criteria is not enough to judge, then consider:\n'
        '2. Stylistic Consistency: The commentary should maintain a tone, word choice, and structure similar to the human commentary.\n'
        f'\n---Commentary A---\n{a_pred}\n----------\n'
        f'\n---Commentary B---\n{b_pred}\n----------\n'
        f'\n---Human Commentary---\n{gt_asr}\n----------\n'
        '\nYour response should be "Commentary A is better aligned with the human commentary" or "Commentary B is better aligned with the human commentary".\n'
    )
    
    # Build messages
    messages = [
        {"role": "user", "content": ab_prompt}
    ]
    
    # Process messages with the tokenizer
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    # Generate
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0,
            do_sample=False,
        )
    
    # Decode
    response = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    ).strip()
    
    # Judge the result
    if 'Commentary A' in response:
        ab_winner = a_id
    elif 'Commentary B' in response:
        ab_winner = b_id
    else:
        ab_winner = 'tie'
    
    return ab_winner


def judge(item, model, tokenizer, baseline_id, baseline_preds, gt_segments):
    """Judge a single sample"""
    key, model_pred = item
    gt_asr = gt_segments.get(key, "")
    baseline_pred = baseline_preds.get(key, "")
    
    ab_winner = judge_ab(model, tokenizer, ['model', model_pred], [baseline_id, baseline_pred], gt_asr)
    ba_winner = judge_ab(model, tokenizer, [baseline_id, baseline_pred], ['model', model_pred], gt_asr)
    
    # Parse the key into fields for readability
    parts = key.split('|')
    return {
        'league': parts[0],
        'season': parts[1],
        'match': parts[2],
        'half': parts[3],
        'segment_index': int(parts[4]),
        'ab_winner': ab_winner,
        'ba_winner': ba_winner
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="LLM Judge for SoccerNet ASR using local Qwen3 model"
    )
    parser.add_argument(
        '--model_path', type=str, default="Qwen/Qwen3-235B-A22B-Instruct-2507",
        help='Path to Qwen3-235B model'
    )
    parser.add_argument(
        '--baseline_id', type=str, default='Qwen3-VL-8B',
        help='Baseline model ID'
    )
    parser.add_argument(
        '--baseline_json', type=str, default="eval/SoccerNet-Eval/win_rate/pred/qwen3vl",
        help='Path to baseline predictions JSON file (same structure as test.json)'
    )
    parser.add_argument(
        '--gt_json', type=str,
        default='eval/SoccerNet-Eval/win_rate/gt/test_win_rate.json',
        help='Path to ground truth JSON file (test.json)'
    )
    parser.add_argument(
        '--pred_json', type=str, default="eval/SoccerNet-Eval/win_rate/pred/livecc.json",
        help='Path to model predictions JSON file (same structure as test.json)'
    )
    parser.add_argument(
        '--output_dir', type=str,
        default='eval/SoccerNet-Eval/win_rate/judges',
        help='Directory to save judgment results'
    )
    parser.add_argument(
        '--num_workers', type=int, default=16,
        help='Number of parallel workers for judgment'
    )
    parser.add_argument(
        '--device', type=str, default='cuda',
        help='Device to load model on'
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    # Load the model
    model, tokenizer = load_qwen3_model(args.model_path, args.device)
    
    # Load data
    gt_data = load_test_json(args.gt_json)
    baseline_data = load_pred_json(args.baseline_json)
    pred_data = load_pred_json(args.pred_json)
    
    # Flatten into a segment dictionary
    gt_segments = flatten_to_segments(gt_data)
    baseline_segments = flatten_to_segments(baseline_data)
    pred_segments = flatten_to_segments(pred_data)
    
    # Check that all keys exist
    for key in baseline_segments:
        assert key in gt_segments, f'Missing GT for {key}'
        assert key in pred_segments, f'Missing prediction for {key}'
    
    for key in pred_segments:
        assert key in gt_segments, f'Missing GT for {key}'
        assert key in baseline_segments, f'Missing baseline for {key}'
    
    print(f"{len(pred_segments)} samples to judge")
    
    # Create the output directory
    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, f'{args.baseline_id}_model.jsonl')
    
    # Run judging
    model_id = 'model'
    print(f'{model_id} vs. {args.baseline_id}')
    
    winner_results = local_mt(
        pred_segments.items(),
        partial(
            judge,
            model=model,
            tokenizer=tokenizer,
            baseline_id=args.baseline_id,
            baseline_preds=baseline_segments,
            gt_segments=gt_segments
        ),
        desc=f'Call Qwen3 for {model_id} vs. {args.baseline_id}',
        num_workers=args.num_workers
    )
    
    # Save results
    with open(save_path, 'w', encoding='utf-8') as f:
        for winner_result in winner_results:
            f.write(json.dumps(winner_result, ensure_ascii=False) + '\n')
    
    # Compute win rate
    win_count, count = 0, 0
    for winner_result in winner_results:
        if winner_result['ab_winner'] == model_id:
            win_count += 1
        if winner_result['ba_winner'] == model_id:
            win_count += 1
        count += 2
    
    win_rate = win_count / count * 100 if count > 0 else 0
    output = f'Winning Rate for {model_id} vs. {args.baseline_id}: {win_rate:.2f}%'
    print(output)
    
    # Save to the log
    with open(os.path.join(args.output_dir, 'log.txt'), 'a', encoding='utf-8') as f:
        f.write(output + '\n')
    
    print("Done!")
