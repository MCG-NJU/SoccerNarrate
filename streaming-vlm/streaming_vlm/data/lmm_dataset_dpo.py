import sys
from pathlib import Path
from functools import lru_cache
import math
import re

# Allow running this file directly:
#   cd streaming-vlm && python streaming_vlm/data/lmm_dataset_dpo.py ...
_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.append(str(_PROJECT_ROOT))
if str(_PROJECT_ROOT / "streaming_vlm") not in sys.path:
    sys.path.append(str(_PROJECT_ROOT / "streaming_vlm"))

sys.path.append("streaming-vlm")
sys.path.append("streaming-vlm/streaming_vlm")

from dataclasses import dataclass
import json
import os
import random
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from transformers import AutoProcessor, logging
from qwen_vl_utils.vision_process import process_vision_info, VIDEO_MAX_TOKEN_NUM, SPATIAL_MERGE_SIZE, FRAME_FACTOR
from streaming_vlm.utils.get_qwen_range import get_qwen_range
from streaming_vlm.utils.text_window_utils import compute_middle_truncation

logger = logging.get_logger(__name__)

_TRAILING_ELLIPSIS_RE = re.compile(r"(?:\s*(?:\.\.\.|…)+)\s*$")


def _strip_trailing_ellipsis(text: str) -> str:
    return _TRAILING_ELLIPSIS_RE.sub("", (text or "").strip()).rstrip()


def _ensure_trailing_ellipsis(text: str) -> str:
    core = _strip_trailing_ellipsis(text)
    if core:
        return core + " ..."
    # Keep a leading space to match the SFT streaming format (phrase + ' ...').
    return " ..."


@lru_cache(maxsize=256)
def _get_video_info(video_path: str) -> Tuple[Optional[int], Optional[float], Optional[float]]:
    """Return (total_frames, fps, duration) if available."""
    try:
        import decord

        vr = decord.VideoReader(video_path)
        total_frames = len(vr)
        fps = float(vr.get_avg_fps())
        if total_frames > 0 and fps > 0:
            return total_frames, fps, total_frames / fps
    except Exception:
        pass

    try:
        from torchvision.io import VideoReader

        vr = VideoReader(video_path, "video")
        meta = vr.get_metadata() or {}
        video_meta = meta.get("video", {})
        fps_list = video_meta.get("fps", []) or []
        dur_list = video_meta.get("duration", []) or []
        nframes_list = video_meta.get("num_frames", []) or []
        fps = float(fps_list[0]) if fps_list else None
        duration = float(dur_list[0]) if dur_list else None
        if duration is None and fps and nframes_list:
            duration = float(nframes_list[0]) / fps
        if fps and duration:
            total_frames = int(round(duration * fps))
            return total_frames, fps, duration
    except Exception:
        pass

    return None, None, None


def _is_valid_window(start: float, end: float, total_frames: int, fps: float) -> bool:
    if total_frames <= 0 or fps <= 0:
        return False
    max_duration = total_frames / fps
    start_clamped = max(0.0, min(float(start), max_duration))
    end_clamped = max(0.0, min(float(end), max_duration))
    start_frame = math.ceil(start_clamped * fps)
    end_frame = math.floor(end_clamped * fps)
    end_frame = min(end_frame, total_frames - 1)
    return start_frame < end_frame


@dataclass
class DataArguments:
    train_annotation_paths: list[str] = None
    chunk_duration: float = 1.0
    include_prefix_video: bool = True
    text_sink: int = 0
    text_sliding_window: int = 0
    max_samples: int = 0


@dataclass
class EvalDataArguments:
    eval_annotation_paths: list[str] = None


def _resolve_video_path(video_path: str) -> str:
    if not video_path:
        return video_path
    if os.path.exists(video_path):
        return video_path

    marker = os.sep + "SoccerNet" + os.sep + "videos" + os.sep
    if marker in video_path:
        rel = video_path.split(marker, 1)[1]
        for env_key in ("SOCCERNET_VIDEOS_ROOT", "DATASET_PATH", "EVAL_DATASET_PATH"):
            root = os.environ.get(env_key)
            if not root:
                continue
            candidate = os.path.join(root, rel)
            if os.path.exists(candidate):
                return candidate

    if not os.path.isabs(video_path):
        for env_key in ("SOCCERNET_VIDEOS_ROOT", "DATASET_PATH", "EVAL_DATASET_PATH"):
            root = os.environ.get(env_key)
            if not root:
                continue
            candidate = os.path.join(root, video_path)
            if os.path.exists(candidate):
                return candidate

    return video_path


class LMMDataset(Dataset):
    """DPO dataset for StreamingVLM-style sliding window training."""

    def __init__(
        self,
        *,
        train_annotation_paths: list[str] = None,
        eval_annotation_paths: list[str] = None,
        processor: AutoProcessor,
        tokenizer=None,
        chunk_duration: float = DataArguments.chunk_duration,
        include_prefix_video: bool = DataArguments.include_prefix_video,
        text_sink: int = DataArguments.text_sink,
        text_sliding_window: int = DataArguments.text_sliding_window,
        max_samples: int = DataArguments.max_samples,
        **kwargs,
    ):
        super().__init__()
        self.processor = processor
        self.tokenizer = tokenizer or processor.tokenizer
        self.chunk_duration = float(chunk_duration)
        self.include_prefix_video = include_prefix_video
        self.text_sink = int(text_sink)
        self.text_sliding_window = int(text_sliding_window)
        # qwen_vl_utils video preprocessing controls.
        #
        # NOTE: Latest `qwen_vl_utils` (0.0.14) controls sampling/resize via the *per-video dict* keys:
        #   - frames: `fps` + `min_frames`/`max_frames` (or `nframes`)
        #   - size: `min_pixels`/`max_pixels` and (for video) `total_pixels`
        # It does NOT read legacy `VIDEO_*` env vars. We support both:
        #   - preferred: `QWENVL_VIDEO_{MIN,MAX,TOTAL}_PIXELS`
        #   - legacy:    `VIDEO_{MIN,MAX,TOTAL}_PIXELS`
        self.video_fps = float(os.environ.get("QWENVL_VIDEO_FPS", os.environ.get("QWENVL_FPS", "2.0")))
        self.video_min_frames = int(os.environ.get("QWENVL_VIDEO_MIN_FRAMES", os.environ.get("QWENVL_MIN_FRAMES", "2")))
        self.video_max_frames = int(
            os.environ.get("QWENVL_VIDEO_MAX_FRAMES", os.environ.get("QWENVL_MAX_FRAMES", os.environ.get("FPS_MAX_FRAMES", "0")))
        )
        self.video_min_pixels = int(
            os.environ.get(
                "QWENVL_VIDEO_MIN_PIXELS",
                os.environ.get("QWENVL_MIN_PIXELS", os.environ.get("VIDEO_MIN_PIXELS", "0")),
            )
        )
        self.video_max_pixels = int(
            os.environ.get(
                "QWENVL_VIDEO_MAX_PIXELS",
                os.environ.get("QWENVL_MAX_PIXELS", os.environ.get("VIDEO_MAX_PIXELS", "0")),
            )
        )
        self.video_total_pixels = int(
            os.environ.get(
                "QWENVL_VIDEO_TOTAL_PIXELS",
                os.environ.get("QWENVL_TOTAL_PIXELS", os.environ.get("VIDEO_TOTAL_PIXELS", "0")),
            )
        )
        image_factor = 14 * SPATIAL_MERGE_SIZE
        self._video_frame_max_pixels_limit = int(VIDEO_MAX_TOKEN_NUM * image_factor * image_factor)

        # For long streaming windows, applying the same pixel budget to *every* video clip will
        # make total visual tokens scale linearly with the number of clips (often causing OOM).
        #
        # When `LMM_DPO_VIDEO_TOTAL_PIXELS_MODE=auto|per_sample`, we treat `*_VIDEO_TOTAL_PIXELS`
        # as a per-sample budget and divide it across all clips in the constructed conversation.
        self.video_total_pixels_mode = str(os.environ.get("LMM_DPO_VIDEO_TOTAL_PIXELS_MODE", "auto")).strip().lower()
        if self.video_total_pixels_mode not in {"auto", "per_window", "per_sample"}:
            logger.warning(
                f"Invalid LMM_DPO_VIDEO_TOTAL_PIXELS_MODE={self.video_total_pixels_mode!r}; "
                "fallback to 'auto'."
            )
            self.video_total_pixels_mode = "auto"

        vision_dtype = str(os.environ.get("LMM_DPO_VISION_DTYPE", "")).strip().lower()
        self.vision_dtype: Optional[torch.dtype] = None
        if vision_dtype in {"fp16", "float16"}:
            self.vision_dtype = torch.float16
        elif vision_dtype in {"bf16", "bfloat16"}:
            self.vision_dtype = torch.bfloat16
        elif vision_dtype in {"fp32", "float32"}:
            self.vision_dtype = torch.float32
        elif vision_dtype:
            logger.warning(
                f"Invalid LMM_DPO_VISION_DTYPE={vision_dtype!r}; expected fp16/bf16/fp32. "
                "Keep processor default dtype."
            )

        if eval_annotation_paths is not None:
            self.is_eval_dataset = True
            annotation_paths = eval_annotation_paths
        else:
            self.is_eval_dataset = False
            annotation_paths = train_annotation_paths

        if not annotation_paths:
            raise ValueError("train/eval annotation paths must be provided")

        self.handles: List[Tuple[Path, int]] = []
        for annotation_path in annotation_paths:
            path = Path(annotation_path)
            if not path.is_file():
                raise FileNotFoundError(f"annotation file not found: {path}")
            self._index_file(path)

        if max_samples and max_samples > 0:
            self.handles = self.handles[: int(max_samples)]

        # Track and skip permanently-broken samples (e.g., timestamps beyond video boundary).
        # This avoids noisy warning spam and prevents deep recursion in __getitem__ retry logic.
        self._skipped_indices: set[int] = set()
        self._skipped_reasons: Dict[int, str] = {}
        self._max_getitem_retries = int(os.environ.get("LMM_DATASET_GETITEM_MAX_RETRIES", "50"))

        if "Qwen2VL" in processor.__class__.__name__:
            self.im_start_id, self.assistant_id, self.newline_id, self.im_end_id = (
                processor.tokenizer("<|im_start|>assistant\n<|im_end|>").input_ids
            )
            self.get_range = get_qwen_range
            self.model_base = "Qwen2"
        elif "Qwen2_5_VL" in processor.__class__.__name__:
            self.im_start_id, self.assistant_id, self.newline_id, self.im_end_id = (
                processor.tokenizer("<|im_start|>assistant\n<|im_end|>").input_ids
            )
            self.get_range = get_qwen_range
            self.model_base = "Qwen2.5"
        elif "Qwen3VL" in processor.__class__.__name__:
            self.im_start_id, self.assistant_id, self.newline_id, self.im_end_id = (
                processor.tokenizer("<|im_start|>assistant\n<|im_end|>").input_ids
            )
            self.get_range = get_qwen_range
            self.model_base = "Qwen3"
        else:
            raise NotImplementedError(
                f"Video preprocessing for {processor.__class__.__name__} is not implemented"
            )

    def _index_file(self, path: Path) -> None:
        with path.open("rb") as f:
            offset = f.tell()
            line = f.readline()
            while line:
                if line.strip():
                    self.handles.append((path, offset))
                offset = f.tell()
                line = f.readline()

    def _load_row(self, index: int) -> Dict[str, Any]:
        annotation_path, seek = self.handles[index]
        with annotation_path.open("r", encoding="utf-8") as f:
            f.seek(seek)
            line = f.readline()
        return json.loads(line)

    def _decode_prefix(self, token_ids: List[int]) -> str:
        if not token_ids:
            return ""
        return self.tokenizer.decode(
            token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

    def _build_conversation(
        self,
        row: Dict[str, Any],
        steps: List[List[str]],
    ) -> Tuple[List[Dict[str, Any]], List[bool]]:
        meta = row.get("meta", {}) if isinstance(row.get("meta"), dict) else {}
        video_path = meta.get("video_path") or row.get("video_path") or ""
        video_path = _resolve_video_path(str(video_path))
        if not os.path.exists(video_path):
            raise ValueError(f"Video {video_path} not found")

        prefix_token_ids = row.get("prefix_text_tokens", []) or []
        prefix_text = self._decode_prefix(prefix_token_ids)

        prefix_secs = row.get("prefix_video_secs", []) or []
        if not self.include_prefix_video:
            prefix_secs = []
        chunk_secs = row.get("chunk_video_secs", []) or []

        total_frames, fps, duration = _get_video_info(video_path)
        if total_frames and fps:
            filtered_prefix = []
            for sec in prefix_secs:
                if _is_valid_window(sec, float(sec) + self.chunk_duration, total_frames, fps):
                    filtered_prefix.append(sec)
            prefix_secs = filtered_prefix

            filtered_chunk = []
            filtered_steps = []
            for idx, sec in enumerate(chunk_secs):
                step = steps[idx] if idx < len(steps) else []
                if _is_valid_window(sec, float(sec) + self.chunk_duration, total_frames, fps):
                    filtered_chunk.append(sec)
                    filtered_steps.append(step)
            chunk_secs = filtered_chunk
            steps = filtered_steps
            if not chunk_secs:
                raise ValueError("no valid chunk seconds after clamping to video duration")

        all_secs = list(prefix_secs) + list(chunk_secs)
        assistant_mask = [False] * len(prefix_secs) + [True] * len(chunk_secs)

        num_windows = len(all_secs)
        per_window_total_pixels: int = int(self.video_total_pixels or 0)
        if per_window_total_pixels > 0:
            if self.video_total_pixels_mode == "per_sample" or (
                self.video_total_pixels_mode == "auto" and num_windows > 1
            ):
                per_window_total_pixels = max(1, int(per_window_total_pixels / max(1, num_windows)))

        conversation: List[Dict[str, Any]] = []
        if prefix_text:
            conversation.append({"role": "previous text", "content": prefix_text})

        row_query = ""
        if isinstance(row.get("query"), str) and row.get("query").strip():
            row_query = row.get("query").strip()
        else:
            hyperparams = meta.get("hyperparams", {}) if isinstance(meta.get("hyperparams"), dict) else {}
            if isinstance(hyperparams.get("query"), str) and hyperparams.get("query").strip():
                row_query = hyperparams.get("query").strip()

        added_query = False
        step_idx = 0
        prefix_count = len(prefix_secs)
        for sec_idx, sec in enumerate(all_secs):
            start = float(sec)
            end = start + self.chunk_duration
            if duration is not None and duration > 0:
                start = max(0.0, min(start, duration))
                end = max(0.0, min(end, duration))
                if end <= start:
                    raise ValueError(f"invalid video window after clamp: start={start}, end={end}")
            # Estimate nframes for max_pixels budgeting to avoid qwen_vl_utils warnings:
            # "The given max_pixels[...] exceeds limit[...]".
            dur = max(0.0, float(end) - float(start))
            nframes_est = max(dur * float(self.video_fps), float(self.video_min_frames))
            if self.video_max_frames and self.video_max_frames > 0:
                nframes_est = min(nframes_est, float(self.video_max_frames))
            # qwen_vl_utils: floor_by_factor(nframes, FRAME_FACTOR)
            nframes_est = max(float(FRAME_FACTOR), math.floor(nframes_est / FRAME_FACTOR) * FRAME_FACTOR)

            video_ele: Dict[str, Any] = {
                "type": "video",
                "video": video_path,
                "video_start": start,
                "video_end": end,
                # qwen_vl_utils frame/pixel budget controls
                # (these keys are consumed inside qwen_vl_utils.vision_process.fetch_video/smart_nframes/smart_resize)
                "fps": self.video_fps,
                "min_frames": self.video_min_frames,
            }
            if self.video_max_frames and self.video_max_frames > 0:
                video_ele["max_frames"] = self.video_max_frames
            if self.video_min_pixels and self.video_min_pixels > 0:
                video_ele["min_pixels"] = self.video_min_pixels
            if per_window_total_pixels and per_window_total_pixels > 0:
                video_ele["total_pixels"] = per_window_total_pixels
            if self.video_max_pixels and self.video_max_pixels > 0:
                # Clamp to qwen_vl_utils internal limit to avoid noisy warnings like:
                #   "The given max_pixels[...] exceeds limit[...]"
                max_pixels = min(self.video_max_pixels, self._video_frame_max_pixels_limit)
                if per_window_total_pixels and per_window_total_pixels > 0:
                    # qwen_vl_utils: max_pixels <= total_pixels / nframes * FRAME_FACTOR
                    max_pixels_from_total = int(per_window_total_pixels / max(1.0, nframes_est) * FRAME_FACTOR)
                    max_pixels = min(max_pixels, max_pixels_from_total)
                video_ele["max_pixels"] = max_pixels

            user_content = [
                {"type": "text", "text": f"Time={start:.1f}-{end:.1f}s"},
                video_ele,
            ]
            if (not added_query and sec_idx >= prefix_count and row_query):
                user_content.append({"type": "text", "text": row_query})
                added_query = True

            conversation.append({"role": "user", "content": user_content})

            if sec_idx < prefix_count:
                # Keep the streaming format consistent (each second ends with a pause marker),
                # but prefix seconds are masked out in labels (no loss).
                assistant_text = _ensure_trailing_ellipsis("")
            else:
                words = steps[step_idx] if step_idx < len(steps) else []
                assistant_text = _ensure_trailing_ellipsis(" ".join(words))
                step_idx += 1

            conversation.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": assistant_text}],
                }
            )

        return conversation, assistant_mask

    def _apply_previous_text_truncation(
        self, inputs: Dict[str, torch.Tensor], has_previous_text: bool
    ) -> None:
        if not has_previous_text:
            return
        if self.text_sink == 0 and self.text_sliding_window == 0:
            return
        input_ids = inputs["input_ids"]
        prev_range = self.get_range(input_ids, "previous text", 0, contain_lf=True, strict=False)
        if prev_range is None:
            return
        prev_start, prev_end = prev_range
        trunc = compute_middle_truncation(
            input_ids=input_ids,
            body_start=prev_start,
            body_end=prev_end,
            keep_left=self.text_sink,
            keep_right=self.text_sliding_window,
            tokenizer=self.tokenizer,
            align_right_boundary=True,
        )
        truncate_start = trunc["cut_start"]
        truncate_end = trunc["cut_end"]
        if truncate_start is None or truncate_end is None:
            return
        inputs["input_ids"] = torch.cat(
            [input_ids[:, :truncate_start], input_ids[:, truncate_end + 1 :]], dim=1
        ).contiguous()
        if "attention_mask" in inputs:
            attn = inputs["attention_mask"]
            inputs["attention_mask"] = torch.cat(
                [attn[:, :truncate_start], attn[:, truncate_end + 1 :]], dim=1
            ).contiguous()

    def _build_labels(self, inputs: Dict[str, torch.Tensor], assistant_mask: List[bool]) -> torch.Tensor:
        input_ids = inputs["input_ids"]
        labels = torch.full_like(input_ids, fill_value=-100, dtype=input_ids.dtype)

        im_start_idxs = (input_ids == self.im_start_id).nonzero()
        im_end_idxs = (input_ids == self.im_end_id).nonzero()
        assistant_idx = 0
        for (sample_idx, im_start_idx), (sample_idx, im_end_idx) in zip(im_start_idxs, im_end_idxs):
            if input_ids[sample_idx, im_start_idx + 1] != self.assistant_id:
                continue
            labels[sample_idx, im_start_idx + 3 : im_end_idx + 1] = input_ids[
                sample_idx, im_start_idx + 3 : im_end_idx + 1
            ]
            if assistant_idx < len(assistant_mask) and not assistant_mask[assistant_idx]:
                labels[sample_idx, im_start_idx : im_end_idx + 1] = -100
            assistant_idx += 1

        return labels

    def _build_inputs(
        self,
        conversation: List[Dict[str, Any]],
        assistant_mask: List[bool],
        image_inputs=None,
        video_inputs=None,
    ) -> Tuple[Dict[str, torch.Tensor], Any, Any]:
        if image_inputs is None and video_inputs is None:
            image_inputs, video_inputs = process_vision_info(conversation)

        text = self.processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=False,
            return_tensors="pt",
        )
        inputs = self.processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
        )
        if self.vision_dtype is not None:
            for k in ("pixel_values", "pixel_values_videos"):
                v = inputs.get(k, None)
                if torch.is_tensor(v) and v.dtype.is_floating_point:
                    inputs[k] = v.to(dtype=self.vision_dtype)
        has_previous_text = any(m.get("role") == "previous text" for m in conversation)
        self._apply_previous_text_truncation(inputs, has_previous_text)
        inputs["labels"] = self._build_labels(inputs, assistant_mask)
        return inputs, image_inputs, video_inputs

    def getitem(self, index: int) -> Dict[str, torch.Tensor]:
        row = self._load_row(index)
        chosen_steps = row.get("chosen_steps", []) or []
        rejected_steps = row.get("rejected_steps", []) or []

        chosen_conv, chosen_mask = self._build_conversation(row, chosen_steps)
        chosen_inputs, image_inputs, video_inputs = self._build_inputs(chosen_conv, chosen_mask)

        rejected_conv, rejected_mask = self._build_conversation(row, rejected_steps)
        rejected_inputs, _, _ = self._build_inputs(
            rejected_conv,
            rejected_mask,
            image_inputs=image_inputs,
            video_inputs=video_inputs,
        )

        # Chosen/rejected share identical visual inputs. To avoid duplicating huge tensors in GPU transfer,
        # we keep vision tensors only in the chosen branch and let the trainer reuse them for rejected.
        for shared_key in (
            "pixel_values",
            "pixel_values_videos",
            "image_grid_thw",
            "video_grid_thw",
            "second_per_grid_ts",
        ):
            rejected_inputs.pop(shared_key, None)

        output: Dict[str, torch.Tensor] = {}
        for key, value in chosen_inputs.items():
            output[f"chosen_{key}"] = value
        for key, value in rejected_inputs.items():
            output[f"rejected_{key}"] = value
        return output

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        if not self.handles:
            raise IndexError("empty dataset")

        dataset_len = len(self.handles)
        idx = int(index) % dataset_len
        last_exc: Optional[BaseException] = None
        phase = "Training" if not self.is_eval_dataset else "Eval"

        for attempt in range(self._max_getitem_retries):
            if idx in self._skipped_indices:
                idx = (idx * 13 + 7 + attempt) % dataset_len
                continue
            try:
                return self.getitem(idx)
            except Exception as exc:
                last_exc = exc
                if idx not in self._skipped_indices:
                    self._skipped_indices.add(idx)
                    self._skipped_reasons[idx] = str(exc)
                    logger.warning(f"{phase}: skipping bad sample index={idx}: {exc}")
                # Deterministic resample to avoid cycles like `idx * 13 % N` when N is divisible by 13.
                idx = (idx * 13 + 7 + attempt) % dataset_len

        # Fall back to random sampling (seeded via `transformers.set_seed`) before giving up.
        for _ in range(min(100, dataset_len)):
            idx = random.randrange(dataset_len)
            if idx in self._skipped_indices:
                continue
            try:
                return self.getitem(idx)
            except Exception as exc:
                last_exc = exc
                if idx not in self._skipped_indices:
                    self._skipped_indices.add(idx)
                    self._skipped_reasons[idx] = str(exc)
                    logger.warning(f"{phase}: skipping bad sample index={idx}: {exc}")

        raise RuntimeError(
            f"{phase}: failed to fetch a valid sample after retries "
            f"(dataset_len={dataset_len}, skipped={len(self._skipped_indices)}). "
            f"Last error: {last_exc}"
        ) from last_exc

    def data_collator(self, batched_inputs, **kwargs):
        assert len(batched_inputs) == 1, "batch size must be 1"
        return batched_inputs[0]

    def __len__(self):
        return len(self.handles)


if __name__ == "__main__":
    import argparse
    import json as _json
    import tqdm
    from torch.utils.data import DataLoader
    from transformers import set_seed

    def _str2bool(v: str) -> bool:
        if isinstance(v, bool):
            return v
        value = str(v).strip().lower()
        if value in {"1", "true", "t", "yes", "y"}:
            return True
        if value in {"0", "false", "f", "no", "n"}:
            return False
        raise argparse.ArgumentTypeError(f"invalid bool value: {v}")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        nargs="+",
        default=["/path/to/SoccerNet/asr/dpo_annos/dpo_data.jsonl"],
        help="One or more DPO JSONL annotation files.",
    )
    parser.add_argument(
        "--processor_name_or_path",
        type=str,
        default=os.environ.get(
            "QWENVL_PROCESSOR", "Qwen/Qwen2.5-VL-7B-Instruct"
        ),
    )
    parser.add_argument("--idx", type=int, default=None, help="Inspect a specific sample index.")
    parser.add_argument("--num_samples", type=int, default=4, help="How many samples to iterate.")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1314)
    parser.add_argument("--check_json_only", action="store_true", help="Only validate JSON schema, skip video IO.")
    parser.add_argument(
        "--scan_invalid_windows",
        action="store_true",
        help="Scan all rows and count invalid (start,end) windows near video boundary.",
    )
    parser.add_argument("--max_scan", type=int, default=0, help="Limit scan rows; 0 means scan all.")
    parser.add_argument("--print_bad", type=int, default=5, help="Print first N problematic rows.")

    parser.add_argument("--chunk_duration", type=float, default=1.0)
    parser.add_argument("--include_prefix_video", type=_str2bool, default=True)
    parser.add_argument("--text_sink", type=int, default=512)
    parser.add_argument("--text_sliding_window", type=int, default=512)
    args = parser.parse_args()

    set_seed(args.seed)

    if args.scan_invalid_windows:
        total_rows = 0
        missing_video = 0
        missing_video_info = 0
        rows_with_any_invalid = 0
        rows_all_chunk_invalid = 0
        invalid_prefix_secs_total = 0
        invalid_chunk_secs_total = 0
        bad_examples: List[Dict[str, Any]] = []

        for path in args.data_path:
            with open(path, "r", encoding="utf-8") as f:
                for line_idx, line in enumerate(tqdm.tqdm(f, desc=f"scan {os.path.basename(path)}")):
                    if not line.strip():
                        continue
                    row = _json.loads(line)
                    meta = row.get("meta", {}) if isinstance(row.get("meta"), dict) else {}
                    video_path = meta.get("video_path") or row.get("video_path") or ""
                    video_path = _resolve_video_path(str(video_path))
                    if not video_path or not os.path.exists(video_path):
                        missing_video += 1
                        total_rows += 1
                        if args.max_scan and total_rows >= args.max_scan:
                            break
                        continue

                    total_frames, fps, duration = _get_video_info(video_path)
                    if not total_frames or not fps:
                        missing_video_info += 1
                        total_rows += 1
                        if args.max_scan and total_rows >= args.max_scan:
                            break
                        continue

                    prefix_secs = row.get("prefix_video_secs", []) or []
                    chunk_secs = row.get("chunk_video_secs", []) or []

                    invalid_prefix = [
                        float(s)
                        for s in prefix_secs
                        if not _is_valid_window(float(s), float(s) + args.chunk_duration, total_frames, fps)
                    ]
                    invalid_chunk = [
                        float(s)
                        for s in chunk_secs
                        if not _is_valid_window(float(s), float(s) + args.chunk_duration, total_frames, fps)
                    ]

                    if invalid_prefix or invalid_chunk:
                        rows_with_any_invalid += 1
                        invalid_prefix_secs_total += len(invalid_prefix)
                        invalid_chunk_secs_total += len(invalid_chunk)
                        if chunk_secs and len(invalid_chunk) == len(chunk_secs):
                            rows_all_chunk_invalid += 1

                        if len(bad_examples) < int(args.print_bad):
                            bad_examples.append(
                                {
                                    "file": str(path),
                                    "line": int(line_idx),
                                    "match_id": row.get("match_id"),
                                    "t0": row.get("t0"),
                                    "video_path": video_path,
                                    "duration": duration,
                                    "invalid_prefix_secs": invalid_prefix[:16],
                                    "invalid_chunk_secs": invalid_chunk[:16],
                                    "num_prefix_secs": len(prefix_secs),
                                    "num_chunk_secs": len(chunk_secs),
                                }
                            )

                    total_rows += 1
                    if args.max_scan and total_rows >= args.max_scan:
                        break
            if args.max_scan and total_rows >= args.max_scan:
                break

        print("==== scan_invalid_windows summary ====")
        print(f"rows_scanned: {total_rows}")
        print(f"missing_video_path: {missing_video}")
        print(f"missing_video_info: {missing_video_info}")
        print(f"rows_with_any_invalid_secs: {rows_with_any_invalid}")
        print(f"rows_all_chunk_secs_invalid: {rows_all_chunk_invalid}")
        print(f"invalid_prefix_secs_total: {invalid_prefix_secs_total}")
        print(f"invalid_chunk_secs_total: {invalid_chunk_secs_total}")
        if bad_examples:
            print("---- examples ----")
            for ex in bad_examples:
                print(_json.dumps(ex, ensure_ascii=False))
        raise SystemExit(0)

    if args.check_json_only:
        required = [
            "match_id",
            "t0",
            "prefix_text_tokens",
            "prefix_video_secs",
            "chunk_video_secs",
            "chosen_steps",
            "rejected_steps",
        ]
        for path in args.data_path:
            print(f"[check_json_only] {path}")
            with open(path, "r", encoding="utf-8") as f:
                seen = 0
                for line_idx, line in enumerate(f):
                    if not line.strip():
                        continue
                    row = _json.loads(line)
                    missing = [k for k in required if k not in row]
                    if missing:
                        raise ValueError(f"line {line_idx}: missing keys: {missing}")
                    if not isinstance(row.get("chosen_steps", []), list) or not isinstance(
                        row.get("rejected_steps", []), list
                    ):
                        raise ValueError(f"line {line_idx}: chosen_steps/rejected_steps must be list")
                    if len(row.get("chunk_video_secs", [])) != len(row.get("chosen_steps", [])) or len(
                        row.get("chunk_video_secs", [])
                    ) != len(row.get("rejected_steps", [])):
                        raise ValueError(
                            f"line {line_idx}: len(chunk_video_secs) must equal len(chosen_steps) and len(rejected_steps)"
                        )
                    seen += 1
                    if args.idx is not None and line_idx == args.idx:
                        print(_json.dumps(row, ensure_ascii=False)[:2000])
                        break
                    if args.idx is None and seen >= args.num_samples:
                        break
        raise SystemExit(0)

    processor = AutoProcessor.from_pretrained(
        args.processor_name_or_path,
        padding_side="right",
        trust_remote_code=True,
    )

    dataset = LMMDataset(
        train_annotation_paths=args.data_path,
        processor=processor,
        tokenizer=None,
        chunk_duration=args.chunk_duration,
        include_prefix_video=args.include_prefix_video,
        text_sink=args.text_sink,
        text_sliding_window=args.text_sliding_window,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True if args.idx is None else False,
        num_workers=args.num_workers,
        collate_fn=dataset.data_collator,
    )

    def _print_sample(sample: Dict[str, Any]) -> None:
        for k in sorted(sample.keys()):
            v = sample[k]
            if torch.is_tensor(v):
                print(f"{k}: shape={tuple(v.shape)} dtype={v.dtype}")
            else:
                print(f"{k}: type={type(v)}")

        for branch in ("chosen", "rejected"):
            labels = sample[f"{branch}_labels"][0]
            input_ids = sample[f"{branch}_input_ids"][0]
            sup_mask = labels != -100
            sup_count = int(sup_mask.sum().item())
            print(f"[{branch}] supervised_tokens={sup_count}")
            if sup_count > 0:
                sup_ids = input_ids[sup_mask].tolist()
                decoded = processor.tokenizer.decode(sup_ids, skip_special_tokens=False)
                print(f"[{branch}] decoded_supervised_text (first 400 chars): {decoded[:400]!r}")

    if args.idx is not None:
        sample = dataset.getitem(args.idx)
        _print_sample(sample)
    else:
        for i, batch in tqdm.tqdm(enumerate(dataloader), total=args.num_samples):
            _print_sample(batch)
            if i + 1 >= args.num_samples:
                break
