from types import MethodType
# patch incorrect liger kernel

from transformers.models.qwen2.modeling_qwen2 import Qwen2Model

from streaming_vlm.utils.patch_trainer import compute_loss_logging_labels

from dataclasses import asdict
import transformers
from transformers import Trainer, AutoProcessor, HfArgumentParser, TrainingArguments, AutoConfig, logging

from streaming_vlm.inference.qwen2.pos_emb import get_rope_index as get_qwen2_rope_index
from streaming_vlm.inference.qwen2_5.pos_emb import get_rope_index as get_qwen2_5_rope_index
from streaming_vlm.inference.qwen3.pos_emb import get_rope_index as get_qwen3_rope_index
import os
import torch
from models import ModelArguments
from streaming_vlm.data.lmm_dataset import DataArguments, LMMDataset, EvalDataArguments
from transformers import set_seed

logger = logging.get_logger(__name__)


def _select_get_rope_index(architecture_name: str):
    """Pick the RoPE index implementation that matches the loaded architecture."""
    if "Qwen3" in architecture_name:
        return get_qwen3_rope_index
    if "Qwen2_5" in architecture_name or "Qwen2.5" in architecture_name:
        return get_qwen2_5_rope_index
    return get_qwen2_rope_index

def find_resume_checkpoint(run_name: str, output_dir: str):
    """
    In the parent directory of output_dir, search in reverse order by {run_name}_YYYYmmdd_HHMMSS:
      - Enter the latest run directory
      - Find the first checkpoint containing train_stats.json in reverse order by step
    Return None if not found
    """
    parent = os.path.dirname(os.path.abspath(output_dir))
    if not os.path.isdir(parent):
        return None

    # Directory names can be sorted in reverse lexicographic order (YYYYmmdd_HHMMSS is consistent with lexicographic order)

    run_dirs = sorted(
        [d for d in os.listdir(parent) if d.startswith(run_name)],
        reverse=True
    )
    for d in run_dirs:
        run_path = os.path.join(parent, d)
        print(f"[resume] Checking directory {run_path}")
        if not os.path.isdir(run_path):
            continue

        # Find checkpoint-*, sort by number in descending order
        ckpts = []
        for name in os.listdir(run_path):
            if name.startswith("checkpoint-"):
                try:
                    step = int(name.split("-", 1)[1])
                except:
                    step = -1
                ckpts.append((step, os.path.join(run_path, name)))
        ckpts.sort(key=lambda x: x[0], reverse=True)

        for _, cp in ckpts:
            if os.path.isfile(os.path.join(cp, "trainer_state.json")):
                print(f"[resume] Resuming from {cp}")
                return cp
    print(f"[resume] No checkpoint found")
    return None

if __name__ == "__main__":
    training_args, model_args, data_args, eval_data_args = HfArgumentParser((TrainingArguments, ModelArguments, DataArguments, EvalDataArguments)).parse_args_into_dataclasses()

    # Simple resume strategy: find the latest run with the same RUN_NAME, select the last checkpoint containing train_stats.json
    resume_ckpt = find_resume_checkpoint(training_args.run_name, training_args.output_dir)

    config = AutoConfig.from_pretrained(model_args.pretrained_model_name_or_path, trust_remote_code=True)
    model = getattr(transformers, config.architectures[0]).from_pretrained(
            model_args.pretrained_model_name_or_path, 
            dtype="auto", attn_implementation='flash_attention_2'
        )
    rope_getter = _select_get_rope_index(config.architectures[0])
    model.get_rope_index = MethodType(rope_getter, model)
    # for m in ["visual", "vision_tower"]:
    for m in ["visual"]:
        try:
            getattr(model, m).requires_grad_(False)
            print(f"Freezing module {m}")
        except:
            print(f"Module {m} not found in model")

    processor = AutoProcessor.from_pretrained(
        model_args.pretrained_model_name_or_path,
        padding_side='right',
        trust_remote_code=True,
    )

    train_dataset = LMMDataset(
        **asdict(data_args),
        **asdict(training_args),
        **asdict(model_args),
        processor=processor,
        model_architecture=config.architectures[0],
    )
    eval_dataset = LMMDataset(
        **asdict(data_args),
        **asdict(eval_data_args),
        **asdict(training_args),
        **asdict(model_args),
        processor=processor,
        model_architecture=config.architectures[0],
    )
    # Add after model is built but before Trainer
    if hasattr(model, "llm_model_embed_tokens"):
        print("delattr llm_model_embed_tokens")
        delattr(model, "llm_model_embed_tokens")

    # Make same-name access a property that returns the actual weights (still shared, no extra memory usage)
    setattr(type(model), "llm_model_embed_tokens", property(lambda self: self.llm.model.embed_tokens))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=train_dataset.data_collator,
        processing_class=processor
    )
    trainer.compute_loss = MethodType(compute_loss_logging_labels, trainer)
    # Pass specific path or False depending on whether resuming training
    trainer.train(resume_from_checkpoint=resume_ckpt if resume_ckpt else False)
