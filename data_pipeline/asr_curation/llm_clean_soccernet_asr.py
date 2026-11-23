#!/usr/bin/env python3
"""Use Qwen3-235B to evaluate and polish ASR transcription segments.

This script processes cleaned ASR annotations with a local LLM:
1. Evaluate each segment's quality and add 'llm_keep' field (true/false)
2. Polish the text and add 'llm_polished_text' field
3. Original files are modified in-place

For swift deployment (http://localhost:8000/v1/chat/completions):
  python llm_clean_soccernet_asr.py \
    --input-root /path/to/SoccerNet/asr/annos_cleaned \
    --api-base http://localhost:8000/v1 \
    --model Qwen/Qwen2.5-72B-Instruct \
    --limit 5

Note: The OpenAI client will automatically append '/chat/completions' to api-base.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: openai package not found. Install with: pip install openai")
    exit(1)


EVALUATION_PROMPT = """You are evaluating soccer match commentary transcription quality. Analyze the current text segment in context and determine if it should be kept.

**Context for reference:**
{context}

**Current segment to evaluate:**
"{text}"

**KEEP if:**
- Contains meaningful match commentary (player names, actions, tactics)
- Contains crowd reactions or atmosphere descriptions
- Contains score updates or match events
- Is grammatically coherent, even with minor errors
- Continues or complements the context naturally

**REMOVE if:**
- Contains only meaningless filler words or noise
- Is completely unintelligible or garbled
- Contains only punctuation or special characters
- Is a repeated fragment already covered in context
- Is an isolated name/word that doesn't fit the context

**Respond with ONLY one word:** KEEP or REMOVE

Your response:"""

POLISH_PROMPT = """You are polishing soccer match commentary transcription. Make minimal edits to improve readability while preserving the original meaning and style.

**Edit guidelines:**
1. Fix obvious spelling errors (especially player/team names if you can identify them)
2. Correct punctuation and capitalization
3. Fix word boundaries (e.g., "goalfor" → "goal for")
4. Remove trailing punctuation fragments
5. DO NOT change the speaking style or add information
6. DO NOT translate or paraphrase
7. Keep abbreviations and colloquial expressions
8. If unsure, keep the original

**Original text:**
"{text}"

**Respond with ONLY the polished text, no explanations.**

Your response:"""


@dataclass
class LLMConfig:
    """Configuration for LLM API calls."""
    api_base: str
    model: str
    temperature: float
    max_tokens: int
    timeout: float
    max_retries: int
    retry_delay: float
    request_delay: float  # Delay between normal requests
    evaluation_rounds: int  # Number of evaluation rounds for each segment


@dataclass
class ProcessingStats:
    """Statistics for LLM processing."""
    total_segments: int = 0
    evaluated_segments: int = 0
    kept_segments: int = 0
    removed_segments: int = 0
    polished_segments: int = 0
    api_errors: int = 0
    
    def print_summary(self) -> None:
        print(f"\n{'='*60}")
        print("Processing Summary:")
        print(f"  Total segments: {self.total_segments}")
        print(f"  Evaluated: {self.evaluated_segments}")
        print(f"  Kept: {self.kept_segments} ({100*self.kept_segments/self.evaluated_segments if self.evaluated_segments > 0 else 0:.1f}%)")
        print(f"  Removed: {self.removed_segments} ({100*self.removed_segments/self.evaluated_segments if self.evaluated_segments > 0 else 0:.1f}%)")
        print(f"  Polished: {self.polished_segments}")
        print(f"  API errors: {self.api_errors}")
        print(f"{'='*60}")


def create_client(config: LLMConfig) -> OpenAI:
    """Create OpenAI-compatible client for local LLM."""
    return OpenAI(
        base_url=config.api_base,
        api_key="dummy",  # Local deployment usually doesn't need real key
        timeout=config.timeout,
    )


def call_llm(
    client: OpenAI,
    config: LLMConfig,
    prompt: str,
    max_retries: Optional[int] = None,
) -> Optional[str]:
    """Call LLM with retry logic."""
    retries = max_retries if max_retries is not None else config.max_retries
    
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=config.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            )
            
            if response.choices and len(response.choices) > 0:
                content = response.choices[0].message.content
                if content:
                    # Add delay between normal requests (not on retries)
                    if attempt == 0 and config.request_delay > 0:
                        time.sleep(config.request_delay)
                    return content.strip()
            
            return None
            
        except Exception as e:
            if attempt < retries - 1:
                print(f"    API error (attempt {attempt+1}/{retries}): {e}")
                time.sleep(config.retry_delay)
            else:
                print(f"    API error (all retries failed): {e}")
                return None
    
    return None


def format_context(prev_segments: List[Dict[str, Any]], next_segments: List[Dict[str, Any]]) -> str:
    """Format context segments for display in prompt."""
    lines = []
    
    if prev_segments:
        lines.append("Previous segments:")
        for i, seg in enumerate(prev_segments, 1):
            text = seg.get("original_text", "")
            lines.append(f"  [{-len(prev_segments)+i}] {text}")
    
    if next_segments:
        lines.append("Next segments:")
        for i, seg in enumerate(next_segments, 1):
            text = seg.get("original_text", "")
            lines.append(f"  [+{i}] {text}")
    
    if not lines:
        return "(No context available - first/last segment)"
    
    return "\n".join(lines)


def evaluate_segment(
    client: OpenAI,
    config: LLMConfig,
    text: str,
    prev_segments: List[Dict[str, Any]],
    next_segments: List[Dict[str, Any]],
) -> Tuple[bool, List[str]]:
    """Evaluate if segment should be kept with multiple rounds.
    
    Returns:
        Tuple of (should_keep, list of responses)
    """
    context = format_context(prev_segments, next_segments)
    prompt = EVALUATION_PROMPT.format(text=text, context=context)
    
    responses = []
    keep_votes = 0
    remove_votes = 0
    
    for round_num in range(config.evaluation_rounds):
        response = call_llm(client, config, prompt)
        
        if response is None:
            # On error, count as KEEP vote
            responses.append("ERROR")
            keep_votes += 1
            continue
        
        responses.append(response)
        response_upper = response.upper().strip()
        
        if "KEEP" in response_upper:
            keep_votes += 1
        elif "REMOVE" in response_upper:
            remove_votes += 1
        else:
            # Unclear response, count as KEEP
            keep_votes += 1
    
    # Majority voting: need 2+ REMOVE votes to actually remove
    should_keep = remove_votes < 2
    
    return should_keep, responses


def polish_segment(client: OpenAI, config: LLMConfig, text: str) -> str:
    """Polish segment text. Returns polished text or original on error."""
    prompt = POLISH_PROMPT.format(text=text)
    response = call_llm(client, config, prompt)
    
    if response is None or len(response) == 0:
        # On error, return original text
        return text
    
    # Basic sanity check: polished text shouldn't be drastically different
    if len(response) > len(text) * 2 or len(response) < len(text) * 0.3:
        print(f"    Suspicious polish (length ratio: {len(response)/len(text):.2f}), keeping original")
        return text
    
    return response


def process_segments(
    segments: List[Dict[str, Any]],
    client: OpenAI,
    config: LLMConfig,
    stats: ProcessingStats,
    context_window: int = 2,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """Process all segments with LLM evaluation and polishing, adding new fields."""
    processed: List[Dict[str, Any]] = []
    
    for i, seg in enumerate(segments):
        stats.total_segments += 1
        text = seg.get("original_text", "")
        
        if verbose and (i + 1) % 10 == 0:
            print(f"  Processing segment {i+1}/{len(segments)}...")
        
        # Get context segments
        prev_segments = segments[max(0, i-context_window):i]
        next_segments = segments[i+1:min(len(segments), i+1+context_window)]
        
        # Step 1: Evaluate (multiple rounds)
        if verbose:
            print(f"    Evaluating ({config.evaluation_rounds} rounds): '{text[:60]}...'")
        
        try:
            keep, eval_responses = evaluate_segment(client, config, text, prev_segments, next_segments)
            stats.evaluated_segments += 1
            
            if verbose:
                keep_count = sum(1 for r in eval_responses if "KEEP" in r.upper())
                remove_count = sum(1 for r in eval_responses if "REMOVE" in r.upper())
                print(f"    Votes: KEEP={keep_count}, REMOVE={remove_count}")
        except Exception as e:
            print(f"    Evaluation error: {e}")
            stats.api_errors += 1
            keep = True  # Default to keeping on error
            eval_responses = []
        
        # Always keep the segment in list, but mark with llm_keep field
        processed_seg = dict(seg)
        processed_seg["llm_keep"] = keep
        
        if keep:
            stats.kept_segments += 1
            
            # Step 2: Polish
            if verbose:
                print(f"    ✅ KEEP, polishing...")
            
            try:
                polished_text = polish_segment(client, config, text)
                stats.polished_segments += 1
                
                if polished_text != text:
                    if verbose:
                        print(f"    Original: {text}")
                        print(f"    Polished: {polished_text}")
                
                processed_seg["llm_polished_text"] = polished_text
            except Exception as e:
                print(f"    Polish error: {e}")
                stats.api_errors += 1
                processed_seg["llm_polished_text"] = text  # Keep original on error
        else:
            stats.removed_segments += 1
            if verbose:
                print(f"    ❌ REMOVE")
            # Don't polish if marked for removal
            processed_seg["llm_polished_text"] = text
        
        processed.append(processed_seg)
    
    return processed


def load_json(path: Path) -> Dict[str, Any]:
    """Load JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, data: Dict[str, Any]) -> None:
    """Save JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8"
    )


def process_file(
    input_path: Path,
    client: OpenAI,
    config: LLMConfig,
    context_window: int,
    verbose: bool,
) -> ProcessingStats:
    """Process a single file in-place."""
    print(f"\nProcessing: {input_path}")
    
    data = load_json(input_path)
    segments = data.get("segments")
    
    if not isinstance(segments, list):
        print(f"  No segments found, skipping")
        return ProcessingStats()
    
    print(f"  Found {len(segments)} segments")
    
    # Process segments
    stats = ProcessingStats()
    processed = process_segments(
        segments, client, config, stats,
        context_window=context_window,
        verbose=verbose
    )
    
    # Update data
    data["segments"] = processed
    
    # Save back to original file
    dump_json(input_path, data)
    
    print(f"  Updated: {input_path}")
    print(f"  Result: {stats.kept_segments} kept, {stats.removed_segments} marked for removal")
    
    return stats


def iter_json_files(root: Path, pattern: str) -> Iterable[Path]:
    """Iterate JSON files matching pattern."""
    return sorted(root.rglob(pattern))


def build_arg_parser() -> argparse.ArgumentParser:
    """Build argument parser."""
    p = argparse.ArgumentParser(
        description="LLM-based cleaning of SoccerNet ASR annotations (in-place modification)"
    )
    
    # I/O paths
    p.add_argument(
        "--input-root",
        type=Path,
        default="/path/to/SoccerNet/asr/annos",
        help="Root folder containing cleaned JSON files (will be modified in-place)",
    )
    p.add_argument(
        "--pattern",
        type=str,
        default="*_clean.json",
        help="Glob pattern for input files (default: *_clean.json)",
    )
    
    # LLM config
    p.add_argument(
        "--api-base",
        type=str,
        default="http://localhost:8000/v1",
        help="Base URL for OpenAI-compatible API (e.g., http://localhost:8000/v1 for swift deployment, client will append /chat/completions)",
    )
    p.add_argument(
        "--model",
        type=str,
        default="Qwen3-235B-A22B-Instruct-2507",
        help="Model name",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Sampling temperature (lower = more deterministic)",
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Max tokens in response",
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=6.0,
        help="API timeout in seconds",
    )
    p.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Max API retry attempts",
    )
    p.add_argument(
        "--retry-delay",
        type=float,
        default=0.5,
        help="Delay between retries in seconds (default: 0.5)",
    )
    p.add_argument(
        "--request-delay",
        type=float,
        default=0.0,
        help="Delay between normal API requests in seconds (default: 0.0)",
    )
    p.add_argument(
        "--evaluation-rounds",
        type=int,
        default=3,
        help="Number of evaluation rounds per segment (default: 3, need 2+ REMOVE to actually remove)",
    )
    
    # Processing options
    p.add_argument(
        "--context-window",
        type=int,
        default=2,
        help="Number of previous/next segments to use as context (default: 2)",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process at most N files (0 = no limit)",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce verbosity (only show file-level progress)",
    )
    
    return p


def main() -> None:
    """Main entry point."""
    args = build_arg_parser().parse_args()
    
    # Setup LLM config
    config = LLMConfig(
        api_base=args.api_base,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay,
        request_delay=args.request_delay,
        evaluation_rounds=args.evaluation_rounds,
    )
    
    # Create client
    print(f"Connecting to LLM at {config.api_base}")
    print(f"Full endpoint: {config.api_base}/chat/completions")
    print(f"Model: {config.model}")
    print(f"\nWARNING: Files will be modified IN-PLACE. Make sure you have backups!")
    client = create_client(config)
    
    # Find files
    input_root: Path = args.input_root
    
    files = list(iter_json_files(input_root, args.pattern))
    if args.limit > 0:
        files = files[:args.limit]
    
    print(f"\nFound {len(files)} files to process")
    
    # Process files
    global_stats = ProcessingStats()
    
    for i, file_path in enumerate(files, 1):
        print(f"\n{'='*60}")
        print(f"File {i}/{len(files)}")
        
        try:
            file_stats = process_file(
                file_path,
                client,
                config,
                context_window=args.context_window,
                verbose=not args.quiet,
            )
            
            # Aggregate stats
            global_stats.total_segments += file_stats.total_segments
            global_stats.evaluated_segments += file_stats.evaluated_segments
            global_stats.kept_segments += file_stats.kept_segments
            global_stats.removed_segments += file_stats.removed_segments
            global_stats.polished_segments += file_stats.polished_segments
            global_stats.api_errors += file_stats.api_errors
            
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
            break
        except Exception as e:
            print(f"ERROR processing {file_path}: {e}")
            continue
    
    # Print summary
    global_stats.print_summary()
    print(f"\nFiles modified in-place: {input_root}")


if __name__ == "__main__":
    main()
