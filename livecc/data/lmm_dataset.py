import sys
sys.path.append("livecc")
from dataclasses import dataclass, field
import json, torch, random, tqdm, io, functools
import numpy as np
import decord
from PIL import Image
from torch.utils.data import Dataset
from transformers import logging, AutoProcessor
from torchvision.transforms.functional import pil_to_tensor
import os

from livecc_utils import _spatial_resize_video
from qwen_vl_utils.vision_process import smart_nframes, process_vision_info, FPS, FORCE_QWENVL_VIDEO_READER
VIDEO_TOTAL_PIXELS = int(os.environ.get('VIDEO_TOTAL_PIXELS', 128 * 640 * 360)) # decrease this for efficiency 
VIDEO_MIN_PIXELS = int(os.environ.get('VIDEO_MIN_PIXELS', 100 * 28 * 28)) # follow qwen2vl paper
FPS_MAX_FRAMES = int(os.environ.get('FPS_MAX_FRAMES', 128)) # decrease this for efficiency 
FRAME_FACTOR = int(os.environ.get('FRAME_FACTOR', 2)) # keep aligned with qwen defaults

logger = logging.get_logger(__name__)

logger.warning(f'{__name__}: {FORCE_QWENVL_VIDEO_READER=}, {FPS_MAX_FRAMES=}, {VIDEO_MIN_PIXELS=}, {VIDEO_TOTAL_PIXELS=}')

@dataclass
class DataArguments:
    annotation_paths: list[str] = field(default_factory=list)
    initial_fps_frames: int = int(FPS) * 3
    streaming_fps_frames: int = int(FPS)
    with_context: bool = False

# --- some utils ---
def readlastline(path: str):
    with open(path, "rb") as f:
        f.seek(-2, 2) # avoid last \n
        while f.read(1) != b"\n":  
            f.seek(-2, 1)
        return f.readline()

def bytes_to_pil(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    if image.mode == 'P':
        image = image.convert('RGBA')
    return image.convert('RGB')

def get_phrase_before_timestamp(text_stream, timestamp, start_from: int = 0):
    """
    Return all words before a timestamp
    
    Logic: include all words whose end time is <= timestamp
    """
    phrase = ''
    last_category = None
    i = start_from - 1  # Initialize to start_from - 1 so -1 is returned if the loop does not run
    if not text_stream or start_from >= len(text_stream):
        return phrase.strip(), start_from, last_category
    for idx, item in enumerate(text_stream[start_from:]):
        # Support both old and new formats: [ws, we, word, category] or [ws, we, word]
        if not item or len(item) < 3:
            continue
        if len(item) >= 4:
            ws, we, word, category = item[0], item[1], item[2], item[3]
        else:
            ws, we, word = item[0], item[1], item[2]
            category = None
        
        # Use the end time: include the word when timestamp >= word end time
        if timestamp >= we:
            phrase += ' ' + word.strip()
            if category is not None:
                last_category = category
        else:
            break
        i = start_from + idx
    return phrase.strip(), i + 1, last_category
# --- some utils ---

class LMMDataset(Dataset):
    def __init__(
        self, *, annotation_paths: list[str], processor: AutoProcessor, 
        initial_fps_frames: int = DataArguments.initial_fps_frames, streaming_fps_frames: int = DataArguments.streaming_fps_frames, 
        with_context: str = DataArguments.with_context, 
        **kwargs
    ):
        super().__init__()
        self.handles = []
        for annotation_path in annotation_paths:
            assert annotation_path.endswith('.jsonl'), f"Please organize the annotations in JSONL format, with each data sample on a separate line, and the last line stores the seek indices"
            logger.warning(f'Load {annotation_path}. Please ensure its last line stores the seek indices...')
            seeks = json.loads(readlastline(annotation_path))
            self.handles.extend(zip([annotation_path] * len(seeks), seeks))
            logger.warning(f'Successfully loaded {annotation_path}')
        if 'Qwen3VL' in processor.__class__.__name__:
            self.im_start_id, self.assistant_id, self.newline_id, self.im_end_id = processor.tokenizer(
                '<|im_start|>assistant\n<|im_end|>').input_ids
        elif 'Qwen2VL' in processor.__class__.__name__:
            self.im_start_id, self.assistant_id, self.newline_id, self.im_end_id = processor.tokenizer(
                '<|im_start|>assistant\n<|im_end|>').input_ids
        else:
            raise NotImplementedError(f"Video preprocess not implemented for {processor.__class__.__name__}")
        self.processor = processor
        self.with_context = with_context
        self.initial_fps_frames = initial_fps_frames
        self.streaming_fps_frames = streaming_fps_frames
    
    def load_conversation(self, index):
        annotation_path, seek = self.handles[index]
        # Open in binary mode to avoid encoding issues
        # Note: file I/O can become a bottleneck with multiprocessing
        with open(annotation_path, 'rb') as f:
            f.seek(seek)
            line = f.readline()
        line = json.loads(line.decode('utf-8'))
        return line

    def preprocess_image(self, element: dict):
        if hasattr(self, 'remote_loader'):
            return Image.open(self.remote_loader(element['image']))
        return element['image']
    
    def preprocess_video(self, element: dict):
        if 'pos' in element: # for sharegpt. implement smart_nframes and smart_resize for pil images video
            if not hasattr(self, 'remote_loader') or self.remote_loader is None:
                raise ValueError(f"remote_loader is required for video preprocessing with 'pos', but it's not set")
            positions = [0] + element['pos']
            nframes = smart_nframes(element, total_frames=len(positions) - 1, video_fps=FPS)
            sampler = torch.linspace(0, len(positions) - 2, nframes).round().long()
            data_bytes = self.remote_loader(element['video'], length_check=True, return_io=False)
            video = torch.stack([pil_to_tensor(bytes_to_pil(data_bytes[positions[i]:positions[i+1]])) for i in sampler])
            video = _spatial_resize_video(video)
            return video
        return element['video']

    def preprocess_text(self, element: str):
        if self.with_context and ('title' in element or 'previous' in element):
            previous = element.get('previous', '')
            if previous:
                title = ''
            else:
                title = element.get('title', '')
            return (element['text'] + f"\n{title}\n{previous}").strip()
        return element['text']

    def preprocess_conversation_stream(self, conversation: list):
        user_message, assistant_message = conversation
        user_content, assistant_content = user_message['content'], assistant_message['content']
        user_video_dict, user_query_dict = user_content
        assert 'video' in user_video_dict, 'please check your data, ensure the video info in the first user content'
        assistant_text_stream = assistant_message['content'][0]['text_stream']
        
        # Ensure remote_loader is passed into video_dict when present
        if hasattr(self, 'remote_loader') and self.remote_loader is not None:
            if 'remote_loader' not in user_video_dict:
                user_video_dict = user_video_dict.copy()
                user_video_dict['remote_loader'] = self.remote_loader
        
        video_reader = self._open_video_reader(user_video_dict)
        sampled_frame_indices, sampled_pts = self._select_strict_fps_indices(video_reader, user_video_dict)
        total_frames = len(sampled_frame_indices)
        if total_frames == 0:
            raise ValueError(f"No frames sampled from video {user_video_dict.get('video')}")
        # load first window on demand
        clip = self._fetch_frames(video_reader, sampled_frame_indices[:min(total_frames, self.initial_fps_frames)])
        if total_frames < self.initial_fps_frames:
            actual_initial_frames = total_frames
        else:
            actual_initial_frames = self.initial_fps_frames

        # make conversation
        # The window start timestamp is 0 relative to the video start
        start_timestamp, end_timestamp = 0, actual_initial_frames / FPS
        
        # First call: accumulate from the window start to the current time
        # Always start from start_from=0 so all text from the window start is accumulated
        # Bounds check: ensure clip_pts has enough elements
        pts_idx = max(min(actual_initial_frames - 1, len(sampled_pts) - 1), 0)
        phrase, next_start_from, category = get_phrase_before_timestamp(assistant_text_stream, sampled_pts[pts_idx], start_from=0)
        
        # Add a special-token prefix based on category
        if category == '1':
            phrase = '<play> ' + phrase
        elif category == '2':
            phrase = '<analysis> ' + phrase
        
        # Ensure phrase is not empty; use a placeholder if needed
        if not phrase.strip():
            phrase = '...'  # Use a placeholder if the initial phrase is empty
        
        conversation = [
            {
            'role': 'user', 'content': [
                    {'type': 'text', 'text': f'Time={start_timestamp:.1f}-{end_timestamp:.1f}s'}, 
                    {'type': 'video', 'video': clip},
                    user_query_dict,
                ]
            },
            {'role': 'assistant', 'content': [{'type': 'text', 'text':  phrase + ' ...'}]} # ' ...' denotes the streaming is not ended
        ]
        frames_list = [clip]
        for i in range(actual_initial_frames, total_frames, self.streaming_fps_frames):
            start_timestamp = i / FPS
            
            # Bounds check: avoid out-of-range indices
            end_frame_idx = min(i + self.streaming_fps_frames, total_frames)
            end_timestamp = end_frame_idx / FPS
            
            # Access clip_pts safely to avoid out-of-bounds errors
            pts_idx = min(end_frame_idx - 1, len(sampled_pts) - 1)
            # Key change: always accumulate from start_from=0 to include all text from the window start to the current time
            # This lets each assistant reply see all earlier text within the current window
            phrase, next_start_from, category = get_phrase_before_timestamp(assistant_text_stream, sampled_pts[pts_idx], start_from=0)
            
            # Add a special-token prefix based on category
            if category == '1':
                phrase = '<play> ' + phrase
            elif category == '2':
                phrase = '<analysis> ' + phrase
            
            # Use a placeholder if phrase is empty
            if not phrase.strip():
                phrase = '...'
            
            frames = self._fetch_frames(video_reader, sampled_frame_indices[i:end_frame_idx])
            conversation.extend([
                {
                    'role': 'user', 'content': [
                        {'type': 'text', 'text': f'Time={start_timestamp:.1f}-{end_timestamp:.1f}s'}, 
                        {'type': 'video', 'video': frames},
                    ]
                },
                {'role': 'assistant', 'content': [{'type': 'text', 'text': phrase + ' ...'}]} # ' ...' denotes the streaming is not ended
            ])
            frames_list.append(frames)
        # remove the last with no phrase
        # Safely remove empty assistant replies to avoid out-of-bounds errors
        while (len(conversation) >= 2 and 
               conversation[-1].get('role') == 'assistant' and
               conversation[-1].get('content') and 
               len(conversation[-1]['content']) > 0 and
               conversation[-1]['content'][0].get('text') == ' ...'):
            conversation = conversation[:-2]
            if frames_list:
                frames_list = frames_list[:-1]
            # Stop if conversation is reduced to only a user message
            if len(conversation) < 2:
                break
        
        # Ensure the final assistant reply ends with a period and remove the ... marker
        if (conversation and 
            conversation[-1].get('role') == 'assistant' and
            conversation[-1].get('content') and 
            len(conversation[-1]['content']) > 0):
            last_text = conversation[-1]['content'][0].get('text', '')
            # Remove the trailing ...
            if last_text.endswith(' ...'):
                last_text = last_text[:-len(' ...')]
            # Ensure the text ends with punctuation if it does not already
            if last_text and not last_text[-1] in ',.!?':
                last_text += '.'
            conversation[-1]['content'][0]['text'] = last_text
        
        del video_reader
        return conversation, frames_list

    def getitem(self, index):
        conversation = self.load_conversation(index)

        special_process_for_stream, image_inputs, video_inputs = False, None, None
        for message in conversation:
            if message['role'] == 'user':
                for element in message['content']:
                    # Ensure remote_loader is passed correctly
                    if hasattr(self, 'remote_loader') and self.remote_loader is not None:
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
        if special_process_for_stream:
            conversation, video_inputs = self.preprocess_conversation_stream(conversation)
            image_inputs = None
        else:
            if not video_inputs and not image_inputs:
                image_inputs, video_inputs = process_vision_info(conversation)
        # Note: apply_chat_template should return a string and should not use return_tensors='pt'
        # This lets the processor handle padding correctly
        texts = self.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
        # Ensure texts is a list even for one sample
        if isinstance(texts, str):
            texts = [texts]
        # Ensure video_inputs is a list because the processor expects one
        if video_inputs is not None and not isinstance(video_inputs, list):
            video_inputs = [video_inputs]
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            padding=True,  # Important: enable padding in distributed training so all ranks have equal sequence lengths
        )
        input_ids = inputs.input_ids
        labels = torch.full_like(input_ids, fill_value=-100, dtype=input_ids.dtype)
        im_start_idxs = (input_ids == self.im_start_id).nonzero()
        im_end_idxs = (input_ids == self.im_end_id).nonzero()
        for (sample_idx, im_start_idx), (sample_idx, im_end_idx) in zip(im_start_idxs, im_end_idxs):
            if input_ids[sample_idx, im_start_idx + 1] == self.assistant_id:
                labels[sample_idx, im_start_idx+3:im_end_idx+1] = input_ids[sample_idx, im_start_idx+3:im_end_idx+1]
        inputs['labels'] = labels
        return inputs

    def __getitem__(self, index):
        # Call getitem directly and remove try/except for better performance
        # Data issues should be fixed during preparation rather than retried during training
        return self.getitem(index)

    def _open_video_reader(self, video_dict: dict):
        video_path = video_dict['video']
        remote_loader = video_dict.get('remote_loader')
        if os.path.exists(video_path):
            return decord.VideoReader(video_path, num_threads=2)
        if remote_loader is not None:
            return decord.VideoReader(remote_loader(video_path), num_threads=2)
        raise ValueError(f"Video {video_path} not found and no remote_loader provided")

    def _select_strict_fps_indices(self, video_reader: decord.VideoReader, video_dict: dict):
        video_reader.get_frame_timestamp(0)
        video_pts = video_reader._frame_pts[:, 1]
        clip_indices = np.arange(len(video_pts))
        video_start = video_dict.get('video_start')
        video_end = video_dict.get('video_end')
        if video_start is not None or video_end is not None:
            start = video_pts[0] if video_start is None else video_start
            end = video_pts[-1] if video_end is None else video_end
            valid_mask = (start <= video_pts) & (video_pts <= end)
            clip_indices = clip_indices[valid_mask]
            video_pts = video_pts[valid_mask]
        if len(video_pts) == 0:
            return [], []
        expected_timestamps = np.arange(video_pts[0], video_pts[-1] + 1e-6, 1 / FPS)
        if len(expected_timestamps) > FPS_MAX_FRAMES:
            expected_timestamps = expected_timestamps[:FPS_MAX_FRAMES]
        expected_mapping = (expected_timestamps[:, None] <= video_pts).argmax(axis=1)
        sampled_indices = clip_indices[expected_mapping]
        sampled_pts = video_pts[expected_mapping]
        while len(sampled_indices) % FRAME_FACTOR != 0:
            sampled_indices = np.append(sampled_indices, sampled_indices[-1])
            sampled_pts = np.append(sampled_pts, sampled_pts[-1])
        return sampled_indices.tolist(), sampled_pts.tolist()

    def _fetch_frames(self, video_reader: decord.VideoReader, index_slice: list[int]):
        if len(index_slice) == 0:
            raise ValueError("index_slice is empty, unable to fetch frames")
        batch = video_reader.get_batch(index_slice).asnumpy()
        frames = torch.from_numpy(batch).permute(0, 3, 1, 2)
        return _spatial_resize_video(frames)

    def data_collator(self, batched_inputs, **kwargs):
        # Return directly for batch_size=1, the common case
        if len(batched_inputs) == 1:
            return batched_inputs[0]
        # For multi-sample batches, padding must be handled manually
        # Batching is more complex because video lengths may differ
        # Use the processor default batching here and fall back to manual handling if unsupported
        if hasattr(self.processor, 'pad'):
            return self.processor.pad(batched_inputs, return_tensors="pt", padding=True)
        # Fallback: manual batching; this simple implementation may need task-specific tuning
        # For video data, keep batch_size=1 and use gradient_accumulation_steps to simulate larger batches
        return batched_inputs[0] if len(batched_inputs) == 1 else batched_inputs

    def __len__(self):
        return len(self.handles)

if __name__ == "__main__":
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    processor = AutoProcessor.from_pretrained('Qwen/Qwen2-VL-7B-Instruct', padding_side='right') 
    # model = Qwen2VLForConditionalGeneration.from_pretrained('Qwen/Qwen2-VL-7B', torch_dtype='auto', attn_implementation='flash_attention_2', device_map='cuda')
    # model.to('cuda')
    
    dataset = LMMDataset(
        annotation_paths=[
            'live_whisperx_526k_with_seeks.jsonl', 
            'llava_video_178k_with_seeks.jsonl', 
            # 'llava_hound_video_with_seeks.jsonl', 
            # 'llava_ov_multi_image_with_seeks.jsonl', 
            # 'llava_ov_single_image_text_mix_with_seeks.jsonl'
        ], 
        processor=processor,
        with_context=False,
    )
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=16, collate_fn=dataset.data_collator)
    
    for batch in tqdm.tqdm(dataloader):
        pass
    # for i in tqdm.tqdm(range(len(dataset))):
    #     conversation = dataset.__getitem__(i)
        # inputs.to('cuda')
        # with torch.inference_mode():
        #     model(**inputs)