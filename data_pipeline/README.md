# SoccerNarrate-Data Preparation & Construction Pipeline

**SoccerNarrate-Data** is built on top of the official [**SoccerNet**](https://www.soccer-net.org/) full-match broadcast dataset covering **436 full matches (872 halves)**: **391 training matches (782 halves = 690 train + 92 valid)** and **45 held-out evaluation matches (90 test halves)** across 6 major competitions (2014–2017 seasons).

> If you only want to **train or evaluate directly** without re-running the curation pipeline, download our pre-built SFT and DPO annotations from 🤗 **[JiaZihan1/SoccerNarrate-Data](https://huggingface.co/datasets/JiaZihan1/SoccerNarrate-Data)** (`sft/` and `dpo/`) and follow **Section 1 (SoccerNet Video & Label Setup)** below.

---

## 1. Expected Dataset Directory Layout (SoccerNet)

Organize the official SoccerNet 720p videos, SoccerNet-label event annotations, and SoccerNarrate JSONL files under a unified `SoccerNet/` root directory:

```text
/path/to/SoccerNet/
├── videos/                                       # Official SoccerNet 720p full-match videos (NDA required)
│   └── <league>/                                 # e.g., england_epl, europe_uefa-champions-league, ...
│       └── <season>/                             # e.g., 2014-2015, 2015-2016, 2016-2017
│           └── <match>/                          # e.g., 2015-02-21 - 18-00 Chelsea 1 - 1 Burnley
│               ├── 1_720p.mkv
│               └── 2_720p.mkv
├── SoccerNet-label/                              # Official SoccerNet Labels-v2.json event annotations
│   └── raw_annos/
│       └── <league>/<season>/<match>/
│           └── Labels-v2.json
└── asr/
    ├── annos/
    │   ├── train/<league_season>/<match>/{1,2}-224p_processed_clean.json   # 690 train halves
    │   ├── valid/<league_season>/<match>/{1,2}-224p_processed_clean.json   #  92 valid halves
    │   ├── test/<league_season>/<match>/{1,2}-224p_processed_clean.json    #  90 test halves (45 eval matches)
    │   └── 12_16_48/                             # Stage 2 Streaming SFT JSONL (from HuggingFace sft/)
    │       ├── soccernet_train_with_seeks.jsonl  # 52,302 samples + trailing byte-seek index
    │       └── soccernet_test_with_seeks.jsonl   #  6,115 samples + trailing byte-seek index
    └── dpo_annos/                                # Stage 3 MWPA / DPO JSONL (from HuggingFace dpo/)
        └── dpo_cross_event_15s.jsonl             # 13,683 cross-event preference pairs
```

### Downloading Official SoccerNet Videos & Labels

Install the official package (`pip install SoccerNet`) and use `soccernet/download_soccernet.py`:

```bash
# 1. Download Labels-v2.json (no password required)
python data_pipeline/soccernet/download_soccernet.py \
  --local-dir /path/to/SoccerNet/SoccerNet-label/raw_annos \
  --download-labels \
  --splits train valid test

# 2. Download 720p match videos (requires SoccerNet NDA password)
export SOCCERNET_PASSWORD="YOUR_SOCCERNET_NDA_PASSWORD"
python data_pipeline/soccernet/download_soccernet.py \
  --local-dir /path/to/SoccerNet/videos \
  --download-videos \
  --splits train valid test
```

Additional helpers in `soccernet/`:
- `create_split_video_symlinks.py`: Organizes SoccerNet videos into `train/valid/test` symlinked split directories.
- `add_mentioned_happened.py`: Appends `mentioned` and `happened` boolean flags from `event_mentions.jsonl` into `Labels-v2.json`.

---

## 2. Pipeline Directory Overview

```text
data_pipeline/
├── soccernet/        # SoccerNet video/label download, split symlinks, and label enrichment
├── asr_curation/     # Stage 1: Rule-based ASR cleaning, LLM quality filtering, phoneme entity calibration
├── sft_build/        # Stage 2: Build StreamingVLM word-level SFT JSONL (*_with_seeks.jsonl)
└── dpo_build/        # Stage 3: Event entailment mining & counterfactual 15s MWPA/DPO pair construction
```

---

## 3. Step-by-Step Construction Pipeline

### Stage 1: ASR Cleaning, Quality Filtering & Entity Calibration (`asr_curation/`)

Starting from word-level WhisperX-large-v3 ASR transcripts with pyannote speaker diarization (see `livecc/data/production/distributed_whisperx.py` for distributed WhisperX extraction) and match roster metadata:

1. **Rule-based segment merging & fragment pruning** (`clean_soccernet_asr.py`):
   Merges short/adjacent fragments and filters low-density noise segments while preserving match metadata and lineups:
   ```bash
   python data_pipeline/asr_curation/clean_soccernet_asr.py \
     --input-root /path/to/SoccerNet/asr/raw_annos \
     --output-root /path/to/SoccerNet/asr/annos_cleaned
   ```
2. **Context-aware LLM quality gate & text polishing** (`llm_clean_soccernet_asr.py`):
   Calls a locally served OpenAI-compatible `Qwen3-235B` endpoint with multi-round voting (`--evaluation-rounds 3`, `--context-window 2`) to write `llm_keep` and `llm_polished_text`:
   ```bash
   python data_pipeline/asr_curation/llm_clean_soccernet_asr.py \
     --input-root /path/to/SoccerNet/asr/annos_cleaned \
     --api-base http://localhost:8000/v1 \
     --model Qwen/Qwen3-235B-A22B-Instruct-2507
   ```
3. **Phoneme-based entity calibration against match rosters** (`phonetic_roster_linker.py`):
   Extracts person mentions with `Qwen3-235B` and aligns them to match rosters (players, coaches, referees) via phoneme-level similarity, writing `corrected_name_text` and `name_replacements`:
   ```bash
   python data_pipeline/asr_curation/phonetic_roster_linker.py \
     --annos-root /path/to/SoccerNet/asr/annos \
     --api-base http://localhost:8000/v1 \
     --model Qwen/Qwen3-235B-A22B-Instruct-2507
   ```
   *(Also includes `llm_correct_names.py` for iterative LLM name verification and `tag_corrected_name_text.py` for role-tagged entity spans.)*

---

### Stage 2: Streaming SFT Dataset Construction (`sft_build/`)

Convert the curated per-half JSONs (`train` + `valid` merged into train, `test` held out) into StreamingVLM-compatible word-level `text_stream` JSONL files with a trailing byte-offset seek index (`soccernet_train_with_seeks.jsonl` and `soccernet_test_with_seeks.jsonl`, matching the `12_16_48` release):

```bash
python data_pipeline/sft_build/process_soccernet_data_stream.py \
  --data_dir /path/to/SoccerNet/asr/annos \
  --video_base_dir /path/to/SoccerNet/videos \
  --output_dir /path/to/SoccerNet/asr/annos/12_16_48 \
  --split all \
  --min_clip_sec 16 \
  --max_clip_sec 48 \
  --max_silence_sec 10.0 \
  --category_filter_mode 1+2
```

*(Also includes `streamingvlm_process.py` for fixed sliding-window chunking.)*

---

### Stage 3: MWPA / DPO Counterfactual Preference Pair Construction (`dpo_build/`)

1. **Event entailment annotation on human ASR** (`stat_soccernet_asr_event_mentions.py`):
   Aligns curated ASR streams with SoccerNet `Labels-v2.json` visible events and judges whether each event window's human commentary explicitly entails the target event using `Qwen3-235B`, producing `event_mentions.jsonl`:
   ```bash
   export QWEN_API_KEY="your_api_key"
   python data_pipeline/dpo_build/stat_soccernet_asr_event_mentions.py \
     --asr-root /path/to/SoccerNet/asr/annos \
     --label-root /path/to/SoccerNet/SoccerNet-label/raw_annos \
     --api-url http://localhost:8000/v1/chat/completions \
     --model Qwen3-235B-A22B-Instruct-2507 \
     --window-seconds 15 \
     --num-workers 64 \
     --out-jsonl /path/to/SoccerNet/asr/event_mentions/event_mentions.jsonl \
     --out-summary /path/to/SoccerNet/asr/event_mentions/event_mentions_summary.json
   ```
   *(For multi-window sensitivity sweeps across 10s–70s as in Fig. 3, see `qwen3_soccernet_asr_event_mentions_vs_window.py`.)*

2. **Build 15s cross-event counterfactual MWPA preference pairs** (`dpo_cross_event.py`):
   Pairs each positive 15-second event window's human commentary rollout (`chosen_steps`) with a type-mismatched human commentary rollout sampled from another event in the same match (`rejected_steps`), filtering sparse windows (`--min_gt_words_per_sec 1.5`) and excluding the `test` split:
   ```bash
   python data_pipeline/dpo_build/dpo_cross_event.py \
     --event_mentions_jsonl /path/to/SoccerNet/asr/event_mentions/event_mentions.jsonl \
     --asr_root /path/to/SoccerNet/asr/annos \
     --video_root /path/to/SoccerNet/videos \
     --label_root /path/to/SoccerNet/SoccerNet-label/raw_annos \
     --model_path mit-han-lab/StreamingVLM \
     --require_window_seconds 15 \
     --min_gt_words_per_sec 1.5 \
     --out_jsonl /path/to/SoccerNet/asr/dpo_annos/dpo_cross_event_15s.jsonl
   ```

3. **Optional per-label ratio filtering / scaling ablation** (`filter_dpo_by_label_ratio.py`):
   Deterministically subsamples a fixed ratio per event class from `dpo_cross_event_15s.jsonl`.
