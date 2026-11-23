from dataclasses import asdict
import transformers
from transformers import Trainer, AutoProcessor, HfArgumentParser, TrainingArguments, AutoConfig, logging

from models import ModelArguments
from data.lmm_dataset import DataArguments, LMMDataset

import sys
sys.path.append('livecc/livecc-utils/src')

logger = logging.get_logger(__name__)

if __name__ == "__main__":
    # Enable offline mode to avoid blocking network requests
    import os
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
    
    training_args, model_args, data_args = HfArgumentParser((TrainingArguments, ModelArguments, DataArguments)).parse_args_into_dataclasses()
    # Use local_files_only=True to force local files and avoid network requests
    config = AutoConfig.from_pretrained(model_args.pretrained_model_name_or_path, local_files_only=True)
    model = getattr(transformers, config.architectures[0]).from_pretrained(
        model_args.pretrained_model_name_or_path, 
        torch_dtype="auto", attn_implementation='flash_attention_2', local_files_only=True
    )
    for m in model_args.freeze_modules:
        logger.warning(f"Freezing module {m}")
        getattr(model, m).requires_grad_(False)
    if 'Qwen2VL' in model.config.architectures[0]:
        # Note: if using a HuggingFace model ID, download it first and use the local path
        # This assumes 'Qwen/Qwen2-VL-7B-Instruct' is already in the local cache or referenced by a local path
        processor = AutoProcessor.from_pretrained('Qwen/Qwen2-VL-7B-Instruct', padding_side='right', local_files_only=True) # Qwen2vl-base processor has some bugs. otherwise we do not need this
    elif 'Qwen3VL' in model.config.architectures[0]:
        processor = AutoProcessor.from_pretrained('Qwen/Qwen3-VL-8B-Instruct', padding_side='right', local_files_only=True) # Qwen2vl-base processor has some bugs. otherwise we do not need this
    else:
        processor = AutoProcessor.from_pretrained(model_args.pretrained_model_name_or_path, padding_side='right', local_files_only=True)
    
    # Add new special tokens: <play>, <analysis>
    special_tokens = ['<play>', '<analysis>']
    num_added_tokens = processor.tokenizer.add_tokens(special_tokens, special_tokens=True)
    if num_added_tokens > 0:
        logger.warning(f'Added {num_added_tokens} special tokens: {special_tokens}')
        # Resize the model embedding layer
        model.resize_token_embeddings(len(processor.tokenizer))
        # Initialize new token embeddings with the mean of existing token embeddings
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data
        input_embeddings_avg = input_embeddings[:-num_added_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_added_tokens].mean(dim=0, keepdim=True)
        input_embeddings[-num_added_tokens:] = input_embeddings_avg
        output_embeddings[-num_added_tokens:] = output_embeddings_avg
        logger.warning(f'Resized model embeddings to {len(processor.tokenizer)} tokens')
    
    train_dataset = LMMDataset(**asdict(data_args), **asdict(training_args), **asdict(model_args), processor=processor)
    Trainer(
        model=model, args=training_args, 
        data_collator=train_dataset.data_collator, 
        train_dataset=train_dataset, processing_class=processor
    ).train(resume_from_checkpoint=not training_args.overwrite_output_dir)