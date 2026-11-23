import argparse
import json
import os
import re
from datetime import datetime

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

model = None
tokenizer = None


def load_model(model_path):
    global model, tokenizer
    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    print("Model loaded successfully!")
    return model, tokenizer


def generate_response(system_message, prompt, temperature=0.1):
    global model, tokenizer
    if model is None or tokenizer is None:
        raise ValueError("Model not loaded. Please call load_model() first.")
    
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt}
    ]
    
    if hasattr(tokenizer, 'apply_chat_template'):
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    else:
        text = f"{system_message}\n\nUser: {prompt}\n\nAssistant:"
    
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=64,
            temperature=temperature,
            do_sample=True if temperature > 0 else False,
            top_p=0.9 if temperature > 0 else None,
            pad_token_id=tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id
        )
    
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    response = response.strip()
    return response


task_description = "Sports commentary refers to the practice of providing live or post-event verbal descriptions, analyses, and insights about a sports competition by professional commentators. \nIt aims to enhance the viewing experience for audiences by offering detailed information, tactical breakdowns, and highlights of the game."

requirement = "Your task is to grade the model commentary refer to the given ground truth commentary, based on the similarity. \nPlease grade the model commentary now, and ONLY GIVE ME ONE floating-point number between 0-10 and NOTHING ELSE.(0 means extremely bad and 10 means perfect)"


def extract_numbers(input_string):
    pattern = r'\b(?:[0-9]|10)(?:\.[0-9]{1,2})?\b'
    matches = re.findall(pattern, input_string)
    return matches


def string_to_list_of_numbers(input_string):
    trimmed_string = input_string.strip()
    split_string = trimmed_string.split(', ')
    numbers_list = [int(num) for num in split_string]
    return numbers_list


def generate_prompt(gt_commentary, model_commentary):
    system_message = task_description + "\n" + requirement
    prompt = "Here are the gt_commentary and the model_commentary: \n"
    prompt += "gt commentary: " + gt_commentary + "\n"
    prompt += "model commentary: " + model_commentary + "\n"
    prompt += "Please grade the model commentary now, based on the similarity with gt commentary, and PLEASE ONLY GIVE me ONE floating-point number between 0-10 and NOTHING ELSE.(0 means extremely bad and 10 means perfect)"
    return system_message, prompt


def iterate_segments(nested_data):
    """
    Traverse the nested structure of test.json or results.json
    Return (key, commentary) pairs
    """
    if not isinstance(nested_data, list):
        return
    for entry in nested_data:
        if not isinstance(entry, dict):
            continue
        for league_season, matches in entry.items():
            if not isinstance(matches, dict):
                continue
            for match_name, halves in matches.items():
                if not isinstance(halves, dict):
                    continue
                for half_label, segments in halves.items():
                    if not isinstance(segments, list):
                        continue
                    for idx, segment in enumerate(segments, start=1):
                        if not isinstance(segment, dict):
                            continue
                        commentary = segment.get("commentary", "").strip()
                        key = f"{league_season} | {match_name} | half_{half_label} | seg_{idx}"
                        yield key, commentary


def load_commentary_map(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    commentary_map = {}
    for key, commentary in iterate_segments(data):
        commentary_map[key] = commentary
    return commentary_map


def evaluate(test_json: str, results_json: str):
    gt_map = load_commentary_map(test_json)
    pred_map = load_commentary_map(results_json)
    
    scores = []
    zero_cnt = 0
    missing_cnt = 0
    
    for key, gt_commentary in tqdm(gt_map.items(), desc="Evaluating"):
        model_commentary = pred_map.get(key, "").strip()
        
        if not model_commentary:
            score = 0.0
            missing_cnt += 1
        else:
            system_message, prompt = generate_prompt(gt_commentary, model_commentary)
            response = generate_response(system_message, prompt, temperature=0.1)
            numbers = extract_numbers(response)
            score = float(numbers[0]) if numbers else 0.0
        
        if score == 0.0:
            zero_cnt += 1
        scores.append(score)
    
    avg_score = sum(scores) / len(scores) if scores else 0.0
    summary = {
        "samples": len(scores),
        "average_score": avg_score,
        "zero_score_count": zero_cnt,
        "missing_prediction_count": missing_cnt
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def save_scores(scores_dir: str, model_name: str, summary: dict):
    os.makedirs(scores_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(scores_dir, f"{model_name}_scores_{timestamp}.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Scores saved to: {output_path}")


parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="Qwen3-VL")
parser.add_argument("--qwen_model_path", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
parser.add_argument("--test_json", type=str, default="/root/work/datasets/SoccerNet/asr/annos/test.json")
parser.add_argument("--results_path", type=str, required=True, help="Model output file; structure must match test.json")
parser.add_argument("--scores_dir", type=str, default="/root/work/datasets/SoccerNet/asr/eval/SCORES")
args = parser.parse_args()

load_model(args.qwen_model_path)
summary = evaluate(test_json=args.test_json, results_json=args.results_path)
save_scores(args.scores_dir, args.model_name, summary)

