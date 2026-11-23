import sys
sys.path.append("streaming-vlm")
sys.path.append("streaming-vlm/streaming_vlm")

from dataclasses import dataclass
import functools
import io
import json
import os
from typing import List, Optional, Tuple

import torch
import tqdm
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor
from transformers import AutoProcessor, logging, set_seed

from livecc_utils import _read_video_decord_plus, _spatial_resize_video
from qwen_vl_utils.vision_process import FPS, process_vision_info, smart_nframes
from streaming_vlm.utils.get_qwen_range import get_qwen_range, set_token_id_map
from streaming_vlm.utils.text_window_utils import compute_middle_truncation

VIDEO_TOTAL_PIXELS = int(os.environ.get("VIDEO_TOTAL_PIXELS", 128 * 640 * 360))  # decrease this for efficiency
VIDEO_MIN_PIXELS = int(os.environ.get("VIDEO_MIN_PIXELS", 100 * 28 * 28))  # follow qwen2vl paper
FPS_MAX_FRAMES = int(os.environ.get("FPS_MAX_FRAMES", 128))  # decrease this for efficiency
FRAME_FACTOR = int(os.environ.get("FRAME_FACTOR", 2))  # keep aligned with qwen defaults
VIDEO_MAX_PIXELS = int(os.environ.get("VIDEO_MAX_PIXELS", 1280 * 720))


class mute_stderr_ffmpeg:
    def __enter__(self):
        self._stderr_fd = os.dup(2)
        self._devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(self._devnull, 2)
        return self

    def __exit__(self, *exc):
        os.dup2(self._stderr_fd, 2)
        os.close(self._devnull)
        os.close(self._stderr_fd)


logger = logging.get_logger(__name__)


@dataclass
class DataArguments:
    train_annotation_paths: list[str] = None
    initial_fps_frames: int = int(FPS)
    streaming_fps_frames: int = int(FPS)
    with_context: bool = False
    text_sink: int = 0
    text_sliding_window: int = 0


@dataclass
class EvalDataArguments:
    eval_annotation_paths: list[str] = None


def readlastline(path: str):
    """Efficiently read the last line of a file."""
    with open(path, "rb") as f:
        f.seek(-2, 2)
        while f.read(1) != b"\n":
            f.seek(-2, 1)
        return f.readline()


def bytes_to_pil(image_bytes):
    """Convert bytes to a PIL Image object."""
    image = Image.open(io.BytesIO(image_bytes))
    if image.mode == "P":
        image = image.convert("RGBA")
    return image.convert("RGB")


def get_phrase_before_timestamp(text_stream, timestamp, start_from: int = 0):
    """From a word stream with timestamps, extract the phrase consisting of all words whose end time is before timestamp."""
    if not text_stream or len(text_stream) == 0:
        return "", 0, None
    if start_from < 0:
        start_from = 0
    if start_from >= len(text_stream):
        return "", len(text_stream), None

    phrase = ""
    last_category = None
    actual_i = start_from
    for i, item in enumerate(text_stream[start_from:]):
        if not isinstance(item, (list, tuple)) or len(item) < 3:
            continue
        # Support both old and new formats: [ws, we, word, category] or [ws, we, word]
        if len(item) >= 4:
            ws, we, word, category = item[0], item[1], item[2], item[3]
        else:
            ws, we, word = item[0], item[1], item[2]
            category = None

        if timestamp >= we:
            phrase += " " + str(word).strip()
            if category is not None:
                last_category = category
            actual_i = i + start_from
            if i == len(text_stream[start_from:]) - 1:
                actual_i += 1
                break
        else:
            break
    return phrase.strip(), actual_i, last_category


class LMMDataset(Dataset):
    """PyTorch dataset for Qwen-style multimodal chat models (Qwen3-VL-first)."""

    def __init__(
        self,
        *,
        train_annotation_paths: list[str] = None,
        eval_annotation_paths: list[str] = None,
        processor: AutoProcessor,
        tokenizer=None,
        initial_fps_frames: int = DataArguments.initial_fps_frames,
        streaming_fps_frames: int = DataArguments.streaming_fps_frames,
        with_context: str = DataArguments.with_context,
        return_conversation: bool = False,
        text_sink: int = DataArguments.text_sink,
        text_sliding_window: int = DataArguments.text_sliding_window,
        model_architecture: str | None = None,
        **kwargs,
    ):
        super().__init__()
        self.return_conversation = return_conversation
        self.handles: list[tuple[str, int]] = []

        if eval_annotation_paths is not None:
            self.is_eval_dataset = True
            annotation_paths = eval_annotation_paths
        else:
            self.is_eval_dataset = False
            annotation_paths = train_annotation_paths or []

        for annotation_path in annotation_paths:
            assert annotation_path.endswith(
                ".jsonl"
            ), "Please organize annotation data as JSONL and store the final line as the seek index."
            root, fname = os.path.split(annotation_path)
            stem = fname.replace("_with_seeks", "").rsplit(".jsonl", 1)[0]
            seek_path = os.path.join(root, f"{stem}_seeks.jsonl")

            logger.warning(f"Loading {annotation_path}")
            logger.warning(f"Loading seek index from {seek_path}")

            seeks = json.loads(readlastline(annotation_path))
            self.handles.extend(zip([annotation_path] * len(seeks), seeks))
            logger.warning(f"Successfully loaded {annotation_path}")

        tok = processor.tokenizer
        set_token_id_map(tok)

        self.im_start_id = tok("<|im_start|>", add_special_tokens=False).input_ids
        self.assistant_id = tok("assistant", add_special_tokens=False).input_ids
        self.newline_id = tok("\n", add_special_tokens=False).input_ids
        self.im_end_id = tok("<|im_end|>", add_special_tokens=False).input_ids

        # template_ids = tok("<|im_start|>assistant\n<|im_end|>", add_special_tokens=False).input_ids
        # if len(template_ids) >= 4:
        #     self.im_start_id, self.assistant_id, self.newline_id, self.im_end_id = template_ids[:4]
        # else:
        #     from streaming_vlm.utils.get_qwen_range import TOKEN_IDS

        #     self.im_start_id = TOKEN_IDS["<|im_start|>"]
        #     self.assistant_id = TOKEN_IDS["assistant"]
        #     self.newline_id = TOKEN_IDS["\n"]
        #     self.im_end_id = TOKEN_IDS["<|im_end|>"]

        self.assistant_start_bias = len(tok("<|im_start|>assistant\n", add_special_tokens=False).input_ids) # 3
        self.get_range = functools.partial(get_qwen_range, strict=False)

        self.processor = processor
        self.with_context = with_context
        self.initial_fps_frames = initial_fps_frames
        self.streaming_fps_frames = streaming_fps_frames
        self.text_sink = text_sink
        self.text_sliding_window = text_sliding_window

    def load_conversation(self, index):
        """Load a single conversation by index."""
        annotation_path, seek = self.handles[index]
        with open(annotation_path) as f:
            f.seek(seek)
            line = f.readline()
        return json.loads(line)

    def preprocess_image(self, element: dict):
        """Preprocess image data."""
        if hasattr(self, "remote_loader"):
            return Image.open(self.remote_loader(element["image"]))
        return element["image"]

    def preprocess_video(self, element: dict):
        """Preprocess video data."""
        if "pos" in element:
            positions = [0] + element["pos"]
            nframes = smart_nframes(element, total_frames=len(positions) - 1, video_fps=FPS)
            sampler = torch.linspace(0, len(positions) - 2, nframes).round().long()
            data_bytes = self.remote_loader(element["video"], length_check=True, return_io=False)
            video = torch.stack(
                [
                    pil_to_tensor(bytes_to_pil(data_bytes[positions[i] : positions[i + 1]]))
                    for i in sampler
                ]
            )
            video = _spatial_resize_video(video)
            return video
        return element["video"]

    def preprocess_text(self, element: str):
        """Preprocess text data."""
        return element["text"]

    def _fake_video_metadata(
        self,
        video: torch.Tensor,
        fps: Optional[float] = None,
        start_frame: int = 0,
        total_num_frames: Optional[int] = None,
    ) -> dict:
        """Build minimal VideoMetadata dict compatible with transformers Qwen3VLProcessor."""
        num_frames = int(video.shape[0])
        if total_num_frames is None:
            total_num_frames = num_frames
        frames_indices = list(range(start_frame, start_frame + num_frames))
        return {
            "total_num_frames": int(total_num_frames),
            "fps": float(fps) if fps is not None else None,
            "frames_indices": frames_indices,
        }

    def preprocess_conversation_stream(
        self, conversation: list
    ) -> Tuple[list, List[torch.Tensor], List[dict]]:
        """Simulate converting timestamped conversation into a streaming multi-turn dialogue."""
        user_message, assistant_message = conversation
        user_content, assistant_content = user_message["content"], assistant_message["content"]

        user_video_dict, user_query_dict = user_content
        video_start = user_video_dict["video_start"]
        video_end = user_video_dict["video_end"]

        assert "video" in user_video_dict, "First user content must contain video information"

        assistant_text_stream = assistant_message["content"][0]["text_stream"]
        qa_stream = (
            assistant_message["content"][0].get("qa_stream", [])
            if isinstance(assistant_message["content"][0], dict)
            else []
        )

        with mute_stderr_ffmpeg():
            clip, _, clip_pts = _read_video_decord_plus(user_video_dict, return_pts=True, strict_fps=True)
        clip = _spatial_resize_video(clip)

        start_timestamp, end_timestamp = video_start, video_start + self.initial_fps_frames / FPS

        phrase, next_start_from, category = get_phrase_before_timestamp(
            assistant_text_stream, end_timestamp, start_from=0
        )

        # if category == "1":
        #     phrase = "<play>" + phrase
        # elif category == "2":
        #     phrase = "<analysis>" + phrase

        if len(qa_stream) > 0 and start_timestamp < qa_stream[0][1] and end_timestamp >= qa_stream[0][1]:
            question = qa_stream[0][2]
            answer = qa_stream[0][3]
            qa_stream = qa_stream[1:]
        else:
            question = ""
            answer = ""

        user_content = [
            {"type": "text", "text": f"Time={start_timestamp:.1f}-{end_timestamp:.1f}s{question}"},
            {"type": "video", "video": clip[: self.initial_fps_frames]},
        ]
        assistant_content = [{"type": "text", "text": answer + "\n" + phrase + " ..."}]
        conversation = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]

        frames_list = [clip[: self.initial_fps_frames]]
        metadata_list = [
            self._fake_video_metadata(
                frames_list[0],
                fps=float(FPS),
                start_frame=round(video_start * FPS),
                total_num_frames=len(clip),
            )
        ]

        for i in range(self.initial_fps_frames, len(clip), self.streaming_fps_frames):
            start_timestamp = video_start + i / FPS
            end_timestamp = video_start + (i + self.streaming_fps_frames) / FPS

            phrase, next_start_from, category = get_phrase_before_timestamp(
                assistant_text_stream, end_timestamp, start_from=next_start_from
            )

            if category == "1":
                phrase = "<play>" + phrase
            elif category == "2":
                phrase = "<analysis>" + phrase

            if len(qa_stream) > 0 and start_timestamp < qa_stream[0][1] and end_timestamp >= qa_stream[0][1]:
                question = qa_stream[0][2]
                answer = qa_stream[0][3]
                qa_stream = qa_stream[1:]
            else:
                question = ""
                answer = ""

            frames = clip[i : i + self.streaming_fps_frames]
            user_content = [
                {"type": "text", "text": f"Time={start_timestamp:.1f}-{end_timestamp:.1f}s {question}"},
                {"type": "video", "video": frames},
            ]
            assistant_content = [{"type": "text", "text": answer + "\n" + phrase + " ..."}]

            conversation.extend(
                [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": assistant_content},
                ]
            )
            frames_list.append(frames)
            metadata_list.append(
                self._fake_video_metadata(
                    frames,
                    fps=float(FPS),
                    start_frame=round(video_start * FPS) + i,
                    total_num_frames=len(clip),
                )
            )

        return conversation, frames_list, metadata_list

    def getitem(self, index, return_text: bool = False):
        """Core logic to get and preprocess a single data item."""
        conversation = self.load_conversation(index)

        # Preserve original start timestamp (used by some inference paths).
        orig_start_ts = None
        try:
            if conversation and conversation[0].get("role") == "user":
                first_ele = conversation[0].get("content", [None])[0]
                if isinstance(first_ele, dict):
                    orig_start_ts = first_ele.get("video_start", None)
        except Exception:
            orig_start_ts = None

        special_process_for_stream = False
        previous_text = ""
        precollected_videos: List[torch.Tensor] = []
        precollected_metadata: List[dict] = []
        precollected_images: List = []

        for message in conversation:
            if message.get("role") == "user":
                for element in message.get("content", []):
                    if "previous" in element:
                        previous_text = element.get("previous", "")
                        element["previous"] = ""
                    if hasattr(self, "remote_loader"):
                        element["remote_loader"] = self.remote_loader
                    modal = element.get("type")
                    if modal and hasattr(self, f"preprocess_{modal}"):
                        element[modal] = getattr(self, f"preprocess_{modal}")(element)

                    if modal == "video" and isinstance(element.get("video"), torch.Tensor):
                        precollected_videos.append(element["video"])
                        fps_val = element.get("raw_fps") or element.get("sample_fps") or FPS
                        fps_val = float(fps_val) if fps_val is not None else None
                        precollected_metadata.append(self._fake_video_metadata(element["video"], fps=fps_val))
                    if modal == "image" and isinstance(element.get("image"), Image.Image):
                        precollected_images.append(element["image"])
            else:
                for element in message.get("content", []):
                    if isinstance(element, dict) and "text_stream" in element:
                        special_process_for_stream = True
                        break

        # Normalize missing video paths (only for string paths).
        first_video = conversation[0]["content"][0].get("video") if conversation else None
        if isinstance(first_video, str) and not os.path.exists(first_video):
            dataset_root = os.environ.get("DATASET_PATH")
            if dataset_root:
                alt_path = os.path.join(dataset_root, first_video)
                if os.path.exists(alt_path):
                    conversation[0]["content"][0]["video"] = alt_path
                else:
                    raise ValueError(f"Video {first_video} not found")
            else:
                raise ValueError(f"Video {first_video} not found")

        image_inputs = None
        video_inputs = None
        video_metadata = None

        if special_process_for_stream:
            conversation, video_inputs, video_metadata = self.preprocess_conversation_stream(conversation)
        else:
            if precollected_videos:
                # Guard against mixed formats within a single sample.
                for msg in conversation:
                    if msg.get("role") != "user":
                        continue
                    for ele in msg.get("content", []):
                        if ele.get("type") == "video" and isinstance(ele.get("video"), str):
                            raise ValueError(
                                "Mixed pre-sampled video tensors with video paths in one sample."
                            )
                        if ele.get("type") == "image" and isinstance(ele.get("image"), str):
                            raise ValueError(
                                "Mixed pre-sampled video tensors with unresolved image paths in one sample."
                            )
                video_inputs = precollected_videos
                video_metadata = precollected_metadata or None
                image_inputs = precollected_images or None
            else:
                image_inputs, raw_video_inputs, _ = process_vision_info(
                    conversation, return_video_kwargs=True, return_video_metadata=True
                )
                if raw_video_inputs:
                    video_inputs = [v for (v, _) in raw_video_inputs]
                    video_metadata = [m for (_, m) in raw_video_inputs]

        if previous_text:
            conversation = [{"role": "previous text", "content": previous_text}] + conversation

        if return_text:
            return conversation

        texts = self.processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=False, return_tensors="pt"
        )

        processor_kwargs = {
            "text": texts,
            "images": image_inputs,
            "videos": video_inputs,
            "return_tensors": "pt",
        }
        if video_inputs is not None:
            videos_kwargs = {"do_sample_frames": False}
            if video_metadata is not None:
                videos_kwargs["video_metadata"] = video_metadata
            processor_kwargs["videos_kwargs"] = videos_kwargs
        inputs = self.processor(**processor_kwargs)

        if previous_text and (self.text_sink != 0 or self.text_sliding_window != 0):
            prev_range = self.get_range(inputs.input_ids, "previous text", 0, contain_lf=True)
            if prev_range is not None:
                previous_text_start_idx, previous_text_end_idx = prev_range
                trunc = compute_middle_truncation(
                    input_ids=inputs.input_ids,
                    body_start=previous_text_start_idx,
                    body_end=previous_text_end_idx,
                    keep_left=self.text_sink,
                    keep_right=self.text_sliding_window,
                    tokenizer=getattr(self.processor, "tokenizer", None),
                    align_right_boundary=True,
                )
                truncate_start = trunc["cut_start"]
                truncate_end = trunc["cut_end"]
                if truncate_start is not None and truncate_end is not None:
                    inputs["input_ids"] = torch.cat(
                        [inputs.input_ids[:, :truncate_start], inputs.input_ids[:, truncate_end + 1 :]],
                        dim=1,
                    ).contiguous()
                    if "attention_mask" in inputs:
                        inputs["attention_mask"] = torch.cat(
                            [
                                inputs.attention_mask[:, :truncate_start],
                                inputs.attention_mask[:, truncate_end + 1 :],
                            ],
                            dim=1,
                        ).contiguous()

        input_ids = inputs.input_ids
        labels = torch.full_like(input_ids, fill_value=-100, dtype=input_ids.dtype)
        im_start_idxs = (input_ids == self.im_start_id).nonzero()
        im_end_idxs = (input_ids == self.im_end_id).nonzero()
        for (sample_idx, im_start_idx), (sample_idx2, im_end_idx) in zip(im_start_idxs, im_end_idxs):
            if input_ids[sample_idx, im_start_idx + 1] == self.assistant_id:
                start = im_start_idx + self.assistant_start_bias
                end = im_end_idx + 1
                if start < end:
                    labels[sample_idx, start:end] = input_ids[sample_idx, start:end]

        inputs["labels"] = labels
        if self.return_conversation:
            inputs["conversation"] = conversation
            if orig_start_ts is not None:
                inputs["start_timestamp"] = orig_start_ts

        return inputs

    def __getitem__(self, index):
        """Dataset standard method with retry."""
        try:
            return self.getitem(index)
        except Exception as e:
            logger.warning(
                f"{'Training' if not self.is_eval_dataset else 'Eval'}: bug at video: "
                f"{self.load_conversation(index)[0]['content'][0].get('video')} {e}"
            )
        return self.__getitem__(index * 13 % len(self.handles))

    def data_collator(self, batched_inputs, **kwargs):
        assert len(batched_inputs) == 1, "batch size must be 1"
        return batched_inputs[0]

    def __len__(self):
        return len(self.handles)


def get_ground_truth(dataset, idx, processor):
    video_text = dataset.getitem(idx, return_text=True)
    ground_truths = []
    for round in video_text:
        if round["role"] == "assistant":
            ground_truth = round["content"][0]["text"]
            ground_truths.append({"ground_truth": ground_truth})
    return ground_truths


if __name__ == "__main__":
    import argparse

    set_seed(1314)
    args = argparse.ArgumentParser()
    args.add_argument(
        "--data_path",
        type=List[str],
        default=[
            "/path/to/SoccerNet/asr/annos/12/soccernet_test_with_seeks.jsonl"
        ],
    )
    args.add_argument("--idx", type=int, default=None)
    args.add_argument("--text_sink", type=int, default=512)
    args.add_argument("--text_sliding_window", type=int, default=512)
    args = args.parse_args()

    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen3-VL-8B-Instruct", padding_side="right"
    )

    dataset = LMMDataset(
        train_annotation_paths=args.data_path,
        tokenizer=None,
        processor=processor,
        text_sink=args.text_sink,
        text_sliding_window=args.text_sliding_window,
        with_context=False,
    )

    from torch.utils.data import DataLoader

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=8,
        collate_fn=dataset.data_collator,
    )

    if args.idx is not None:
        _ = dataset[args.idx]
        breakpoint()

    for _, _ in tqdm.tqdm(enumerate(dataloader)):
        import pdb; pdb.set_trace()  # noqa: E702
        pass
