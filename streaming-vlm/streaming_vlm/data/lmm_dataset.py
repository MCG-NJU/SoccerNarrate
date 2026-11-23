import sys
sys.path.append("streaming-vlm")
sys.path.append("streaming-vlm/streaming_vlm")
from dataclasses import dataclass, field
import json, torch, random, tqdm, io, functools,os
from PIL import Image
from torch.utils.data import Dataset
from transformers import logging, AutoProcessor, AutoModel, Qwen2_5_VLForConditionalGeneration, Qwen3VLForConditionalGeneration
from torchvision.transforms.functional import pil_to_tensor
from transformers.feature_extraction_utils import BatchFeature
from collections import defaultdict
from livecc_utils import _read_video_decord_plus, _spatial_resize_video
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import List
from qwen_vl_utils.vision_process import smart_nframes, process_vision_info, FPS, FORCE_QWENVL_VIDEO_READER
VIDEO_TOTAL_PIXELS = int(os.environ.get('VIDEO_TOTAL_PIXELS', 128 * 640 * 360)) # decrease this for efficiency 
VIDEO_MIN_PIXELS = int(os.environ.get('VIDEO_MIN_PIXELS', 100 * 28 * 28)) # follow qwen2vl paper
FPS_MAX_FRAMES = int(os.environ.get('FPS_MAX_FRAMES', 128)) # decrease this for efficiency 
FRAME_FACTOR = int(os.environ.get('FRAME_FACTOR', 2)) # keep aligned with qwen defaults
VIDEO_MAX_PIXELS = int(os.environ.get('VIDEO_MAX_PIXELS', 1280*720))
EVENT_PRE_SEC = float(os.environ.get('EVENT_PRE_SEC', 2.0))   # treat event active before t
EVENT_POST_SEC = float(os.environ.get('EVENT_POST_SEC', 4.0)) # treat event active after t


from streaming_vlm.utils.get_qwen_range import get_qwen_range
from streaming_vlm.utils.text_window_utils import compute_middle_truncation
from transformers import set_seed

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
    text_sink:int = 0
    text_sliding_window:int = 0

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
    if image.mode == 'P':
        image = image.convert('RGBA')
    return image.convert('RGB')


def get_phrase_before_timestamp(text_stream, timestamp, start_from: int = 0):
    """From a word stream with timestamps, extract the phrase consisting of all words whose end time is before the given timestamp."""
    if not text_stream or len(text_stream) == 0:
        return '', 0, None
    if start_from < 0:
        start_from = 0
    if start_from >= len(text_stream):
        return '', len(text_stream), None
    
    phrase = ''
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
            phrase += ' ' + str(word).strip()
            if category is not None:
                last_category = category
            actual_i = i + start_from
            if i == len(text_stream[start_from:]) - 1:
                actual_i += 1
                break
        else:
            break
    return phrase.strip(), actual_i, last_category

def _filter_events_in_window(
    event_stream,
    start_timestamp: float,
    end_timestamp: float,
    *,
    pre_seconds: float = EVENT_PRE_SEC,
    post_seconds: float = EVENT_POST_SEC,
):
    events_in_window = []
    if not event_stream:
        return events_in_window
    for e in event_stream:
        if not isinstance(e, dict):
            continue
        try:
            t = float(e.get('t'))
        except Exception:
            continue
        # An event at time t is treated as active in [t-pre_seconds, t+post_seconds].
        # Include it if this active interval overlaps the current [start_timestamp, end_timestamp].
        if (t + post_seconds) < start_timestamp or (t - pre_seconds) > end_timestamp:
            continue
        label = e.get('label', None)
        if not label:
            continue
        ev = {'t': t, 'label': str(label)}
        team = e.get('team', None)
        if team:
            ev['team'] = str(team)
        visibility = e.get('visibility', None)
        if visibility:
            ev['visibility'] = str(visibility)
        conf = e.get('conf', None)
        if conf is not None:
            try:
                ev['conf'] = float(conf)
            except Exception:
                pass
        events_in_window.append(ev)
    events_in_window.sort(key=lambda x: x.get('t', 0.0))
    return events_in_window


def _format_event_candidates(events_in_window) -> str:
    if not events_in_window:
        return "\n<event_candidates>\nnone\n</event_candidates>"

    lines = []
    for e in events_in_window:
        label = e.get("label", None)
        if not label:
            continue
        conf = e.get("conf", None)
        try:
            conf_str = f"{float(conf):.2f}" if conf is not None else "None"
        except Exception:
            conf_str = "None"
        team = e.get("team", None) or "None"
        visibility = e.get("visibility", None) or "None"
        lines.append(f"label={label}, conf={conf_str}, team={team}, visibility={visibility}")

    body = ";\n".join(lines) if lines else "none"
    return f"\n<event_candidates>\n{body}\n</event_candidates>"

class LMMDataset(Dataset):
    """PyTorch dataset for multimodal large language models."""
    def __init__(
        self, *, train_annotation_paths: list[str] = None, eval_annotation_paths: list[str] = None, processor: AutoProcessor, tokenizer = None,
        initial_fps_frames: int = DataArguments.initial_fps_frames, streaming_fps_frames: int = DataArguments.streaming_fps_frames, 
        with_context: str = DataArguments.with_context, return_conversation: bool = False,text_sink:int = DataArguments.text_sink,
        text_sliding_window:int = DataArguments.text_sliding_window,
        **kwargs
    ):
        super().__init__()
        self.return_conversation = return_conversation
        self.handles = []
        if eval_annotation_paths is not None:
            self.is_eval_dataset = True
            annotation_paths = eval_annotation_paths
        else:
            self.is_eval_dataset = False
            annotation_paths = train_annotation_paths
            
        for annotation_path in annotation_paths:
            assert annotation_path.endswith('.jsonl'), "Please organize annotation data as JSONL (one sample per line) and store the final line as the seek index."
            root, fname = os.path.split(annotation_path)
            stem = fname.replace("_with_seeks", "").rsplit(".jsonl", 1)[0]
            seek_path = os.path.join(root, f"{stem}_seeks.jsonl")

            logger.warning(f"Loading {annotation_path}")
            logger.warning(f"Loading seek index from {seek_path}")

            # with open(annotation_path) as f:
            #     seeks = json.load(f)    
            seeks = json.loads(readlastline(annotation_path)) 
                
            self.handles.extend(zip([annotation_path] * len(seeks), seeks))
            logger.warning(f"Successfully loaded {annotation_path}")

        if 'Qwen2VL' in processor.__class__.__name__:
            self.im_start_id, self.assistant_id, self.newline_id, self.im_end_id = processor.tokenizer('<|im_start|>assistant\n<|im_end|>').input_ids
            self.get_range = get_qwen_range
            self.model_base = 'Qwen2'
        elif 'Qwen2_5_VL' in processor.__class__.__name__:
            self.im_start_id, self.assistant_id, self.newline_id, self.im_end_id = processor.tokenizer('<|im_start|>assistant\n<|im_end|>').input_ids
            self.get_range = get_qwen_range
            self.model_base = 'Qwen2.5'
        elif 'Qwen3VL' in processor.__class__.__name__:
            self.im_start_id, self.assistant_id, self.newline_id, self.im_end_id = processor.tokenizer('<|im_start|>assistant\n<|im_end|>').input_ids
            self.get_range = get_qwen_range
            self.model_base = 'Qwen3'
        else:
            raise NotImplementedError(f"Video preprocessing for {processor.__class__.__name__} is not implemented")

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
        line = json.loads(line)
        return line

    def preprocess_image(self, element: dict):
        """Preprocess image data."""
        if hasattr(self, 'remote_loader'):
            return Image.open(self.remote_loader(element['image']))
        return element['image']
    
    def preprocess_video(self, element: dict):
        """Preprocess video data."""
        if 'pos' in element:
            positions = [0] + element['pos']
            nframes = smart_nframes(element, total_frames=len(positions) - 1, video_fps=FPS)
            sampler = torch.linspace(0, len(positions) - 2, nframes).round().long()
            data_bytes = self.remote_loader(element['video'], length_check=True, return_io=False)
            video = torch.stack([pil_to_tensor(bytes_to_pil(data_bytes[positions[i]:positions[i+1]])) for i in sampler])
            video = _spatial_resize_video(video)
            return video
        return element['video']

    def preprocess_text(self, element: str):
        """Preprocess text data."""
        return element['text']

    def preprocess_conversation_stream(self, conversation: list):
        """Simulate converting timestamped conversation into a streaming multi-turn dialogue."""
        user_message, assistant_message = conversation
        user_content, assistant_content = user_message['content'], assistant_message['content']

        user_video_dict, user_query_dict = user_content
        video_start = user_video_dict['video_start']
        video_end = user_video_dict['video_end']
        event_stream = user_video_dict.get('event_stream', None)
        event_source = user_video_dict.get('event_source', '')
        has_event_stream = ('event_stream' in user_video_dict) or ('event_source' in user_video_dict)
        
        assert 'video' in user_video_dict, 'Please check your data: the first user content must contain video information'

        assistant_text_stream = assistant_message['content'][0]['text_stream']
        qa_stream = assistant_message['content'][0]['qa_stream'] if 'qa_stream' in assistant_message['content'][0] else []
        
        with mute_stderr_ffmpeg():
            clip, _, clip_pts = _read_video_decord_plus(
                user_video_dict, return_pts=True, strict_fps=True
            )
        clip = _spatial_resize_video(clip)

        start_timestamp, end_timestamp = video_start, video_start + self.initial_fps_frames / FPS

        phrase, next_start_from, category = get_phrase_before_timestamp(
            assistant_text_stream,
            end_timestamp,
            start_from=0
        )

        # Add a special-token prefix based on category
        # if category == '1':
        #     phrase = '<play>' + phrase
        # elif category == '2':
        #     phrase = '<analysis>' + phrase

        if len(qa_stream) > 0 and start_timestamp < qa_stream[0][1] and end_timestamp >= qa_stream[0][1]:
            question = qa_stream[0][2]
            answer = qa_stream[0][3]
            qa_stream = qa_stream[1:]
        else:
            question = ''
            answer = ''
        
        event_text = ''
        if has_event_stream:
            events_in_window = _filter_events_in_window(event_stream, start_timestamp, end_timestamp)
            event_text = _format_event_candidates(events_in_window)

        user_content = [
                        {'type': 'text',  'text': f'Time={start_timestamp:.1f}-{end_timestamp:.1f}s' + f'{question}' + event_text},
                        {'type': 'video', 'video': clip[:self.initial_fps_frames]},
                    ]
        assistant_content = [{'type': 'text', 'text': answer + '\n' + phrase + ' ...'}]
        conversation = [
            {
                'role': 'user',
                'content': user_content
            },
            {
                'role': 'assistant',
                'content': assistant_content
            }
        ] 
        frames_list = [clip[:self.initial_fps_frames]]
        
        for i in range(self.initial_fps_frames, len(clip), self.streaming_fps_frames):
            start_timestamp, end_timestamp = video_start + i / FPS, video_start + (i + self.streaming_fps_frames) / FPS
            
            phrase, next_start_from, category = get_phrase_before_timestamp(
                assistant_text_stream,
                end_timestamp,
                start_from=next_start_from
            )
            
            # Add a special-token prefix based on category
            # if category == '1':
            #     phrase = '<play>' + phrase
            # elif category == '2':
            #     phrase = '<analysis>' + phrase
            
            if len(qa_stream) > 0 and start_timestamp < qa_stream[0][1] and end_timestamp >= qa_stream[0][1]:
                question = qa_stream[0][2]
                answer = qa_stream[0][3]
                qa_stream = qa_stream[1:]
            else:
                question = ''
                answer = ''
            
            frames = clip[i : i + self.streaming_fps_frames]

            event_text = ''
            if has_event_stream:
                events_in_window = _filter_events_in_window(event_stream, start_timestamp, end_timestamp)
                event_text = _format_event_candidates(events_in_window)

            user_content = [
                    {'type': 'text',  'text': f'Time={start_timestamp:.1f}-{end_timestamp:.1f}s {question}' + event_text},
                    {'type': 'video', 'video': frames},
                ]
            assistant_content = [{'type': 'text', 'text': answer + '\n' + phrase + ' ...'}]

            conversation.extend([
                {
                    'role': 'user',
                    'content': user_content
                },
                {
                    'role': 'assistant',
                    'content': assistant_content
                }
            ])
            frames_list.append(frames)

        return conversation, frames_list

    def getitem(self, index, return_text=False):
        """Core logic to get and preprocess a single data item."""
        conversation = self.load_conversation(index)

        special_process_for_stream, image_inputs, video_inputs = False, None, None
        previous_text = ''
        for message in conversation:
            if message['role'] == 'user':
                for element in message['content']:
                    if 'previous' in element:
                        previous_text = element['previous']
                        element['previous'] = ''
                    if hasattr(self, 'remote_loader'):
                        element['remote_loader'] = self.remote_loader
                    modal = element['type']
                    element[modal] = getattr(self, f'preprocess_{modal}')(element)
                    if isinstance(element[modal], torch.Tensor):
                        if video_inputs is None:
                            video_inputs = [element[modal]]
                        else:
                            video_inputs.append(element[modal])
            else:
                for element in message['content']:
                    special_process_for_stream = 'text_stream' in element
                    break
        
        if not os.path.exists(conversation[0]['content'][0]['video']):
            if os.path.exists(os.path.join(os.environ['DATASET_PATH'], conversation[0]['content'][0]['video'])):    
                conversation[0]['content'][0]['video'] = os.path.join(os.environ['DATASET_PATH'], conversation[0]['content'][0]['video'])
            else:
                raise ValueError(f"Video {conversation[0]['content'][0]['video']} not found")
        if special_process_for_stream:
            conversation, video_inputs = self.preprocess_conversation_stream(conversation)
            image_inputs = None
        else:
            if not video_inputs and not image_inputs:
                image_inputs, video_inputs = process_vision_info(conversation)

        if previous_text:
            conversation = [{"role": "previous text", "content": previous_text}] + conversation

        if return_text:
            return conversation
        texts = self.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False, return_tensors='pt')
        
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
        )

        if previous_text and (self.text_sink != 0 or self.text_sliding_window != 0):
            previous_text_start_idx, previous_text_end_idx = self.get_range(inputs.input_ids, 'previous text', 0, contain_lf=True)
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
                inputs['input_ids'] = torch.cat([inputs.input_ids[:,:truncate_start],inputs.input_ids[:,truncate_end+1:]],dim=1).contiguous()
                if 'attention_mask' in inputs:
                    inputs['attention_mask'] = torch.cat([inputs.attention_mask[:,:truncate_start],inputs.attention_mask[:,truncate_end+1:]],dim=1).contiguous()

        input_ids = inputs.input_ids
        labels = torch.full_like(input_ids, fill_value=-100, dtype=input_ids.dtype)
        im_start_idxs = (input_ids == self.im_start_id).nonzero()
        im_end_idxs = (input_ids == self.im_end_id).nonzero()
        for (sample_idx, im_start_idx), (sample_idx, im_end_idx) in zip(im_start_idxs, im_end_idxs):
            if input_ids[sample_idx, im_start_idx + 1] == self.assistant_id:
                labels[sample_idx, im_start_idx+3:im_end_idx+1] = input_ids[sample_idx, im_start_idx+3:im_end_idx+1]

        inputs['labels'] = labels
        if self.return_conversation:
            inputs['conversation'] = conversation
            inputs['start_timestamp'] = conversation[0]['content'][0]['video_start']

        return inputs

    def __getitem__(self, index):
        """Dataset standard method with retry."""
        # return self.getitem(index) 
        try: 
            return self.getitem(index) 
        except Exception as e:
            logger.warning(
                f"{'Training' if not self.is_eval_dataset else 'Eval'}: bug at video: "
                f"{self.load_conversation(index)[0]['content'][0]['video']}"
                f"{e}"
            )
        return self.__getitem__( index*13 %len(self.handles))

    def data_collator(self, batched_inputs, **kwargs):
        assert len(batched_inputs) == 1, "batch size must be 1"
        return batched_inputs[0]

    def __len__(self):
        """Return the total number of samples."""
        return len(self.handles)

def get_ground_truth(dataset, idx, processor):
    video_text = dataset.getitem(idx, return_text=True)
    ground_truths = []
    for round in video_text:
        if round['role'] == 'assistant':
            ground_truth = round['content'][0]['text']
            ground_truths.append({'ground_truth':ground_truth})
        else:
            continue
    return ground_truths

if __name__ == "__main__":
    import argparse
    set_seed(1314)
    args = argparse.ArgumentParser()
    args.add_argument('--data_path', type=List[str], default=['/path/to/SoccerNet/asr/annos/12/soccernet_train_with_seeks.jsonl']) 
    args.add_argument('--model_base', type=str, default='Qwen',choices=['Qwen'])
    args.add_argument('--idx', type=int, default=None)
    args.add_argument('--text_sink', type=int, default=512)
    args.add_argument('--text_sliding_window', type=int, default=512)
    args = args.parse_args()

    from transformers import AutoProcessor
    
    processor = AutoProcessor.from_pretrained('Qwen/Qwen2.5-VL-7B-Instruct', padding_side='right') 
    if args.idx is not None:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained('Qwen/Qwen2.5-VL-7B-Instruct',trust_remote_code=True,device_map='auto',attn_implementation='flash_attention_2')
        
    dataset = LMMDataset(
        train_annotation_paths=args.data_path, 
        tokenizer=None,
        processor=processor,
        text_sink=args.text_sink,
        text_sliding_window=args.text_sliding_window,
        with_context=False,
    )
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=8, collate_fn=dataset.data_collator)
    
    if args.idx is not None:
        data = dataset[args.idx]
        breakpoint()

        model.generate(input_ids=data.input_ids.to('cuda'),media=data.media,media_config=defaultdict(dict), generation_config=model.default_generation_config)
        breakpoint()
    else:
        for i, batch in tqdm.tqdm(enumerate(dataloader)):
            pass
