from types import MethodType
# patch incorrect liger kernel


from dataclasses import asdict, dataclass
from contextlib import contextmanager
from typing import Optional
import transformers
from transformers import Trainer, AutoProcessor, HfArgumentParser, TrainingArguments, AutoConfig, logging

from streaming_vlm.inference.qwen2.pos_emb import get_rope_index as get_qwen2_rope_index
from streaming_vlm.inference.qwen2_5.pos_emb import get_rope_index as get_qwen2_5_rope_index
from streaming_vlm.inference.qwen3.pos_emb import get_rope_index as get_qwen3_rope_index
import os
import torch
import torch.nn.functional as F
from models import ModelArguments
from streaming_vlm.data.lmm_dataset_dpo import DataArguments, LMMDataset, EvalDataArguments
from transformers import set_seed

logger = logging.get_logger(__name__)

@contextmanager
def _temporary_unset_hf_deepspeed_config():
    """
    When using DeepSpeed ZeRO-3, HF sets a global HfDeepSpeedConfig weakref that makes *all*
    subsequent `from_pretrained` calls load params as ZeRO-3 partitioned 1D shards.

    That's fine for the *training* model (it will be wrapped by DeepSpeed engine), but it breaks
    running an auxiliary `ref_model` forward directly (e.g., embedding weight becomes 1D).
    """
    try:
        from transformers.integrations import deepspeed as hf_deepspeed
    except Exception:
        yield
        return

    weak_ref = getattr(hf_deepspeed, "_hf_deepspeed_config_weak_ref", None)
    cfg_obj = weak_ref() if weak_ref is not None else None
    if cfg_obj is None:
        yield
        return

    hf_deepspeed.unset_hf_deepspeed_config()
    try:
        yield
    finally:
        hf_deepspeed.set_hf_deepspeed_config(cfg_obj)


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


@dataclass
class DPOArguments:
    beta: float = 0.1
    ref_model_name_or_path: Optional[str] = None
    length_norm: bool = False


def _get_batch_logps(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Memory-safe logp sum for `labels != -100` positions.

    NOTE: This does NOT compute a full `log_softmax` tensor (which can OOM for long sequences + large vocab).
    """
    shift_logits = logits[:, :-1, :]  # predicts labels[:, 1:]
    shift_labels = labels[:, 1:]
    mask = shift_labels != -100
    batch_size = logits.shape[0]

    if mask.sum() == 0:
        return (
            torch.zeros(batch_size, device=logits.device, dtype=torch.float32),
            torch.zeros(batch_size, device=logits.device, dtype=torch.long),
        )

    # Select only supervised positions (often tiny compared to seq_len).
    selected_logits = shift_logits[mask]  # (N, vocab)
    selected_labels = shift_labels[mask]  # (N,)
    nll = F.cross_entropy(selected_logits, selected_labels, reduction="none")  # (N,)
    token_logps = -nll  # (N,)

    # Sum per sample.
    sample_ids = mask.nonzero(as_tuple=False)[:, 0]  # (N,)
    logps = torch.zeros(batch_size, device=logits.device, dtype=token_logps.dtype)
    logps.scatter_add_(0, sample_ids, token_logps)
    lengths = torch.zeros(batch_size, device=logits.device, dtype=torch.long)
    lengths.scatter_add_(0, sample_ids, torch.ones_like(sample_ids, dtype=torch.long))
    return logps, lengths


def _fill_shared_vision_inputs(chosen_inputs: dict, rejected_inputs: dict) -> None:
    """Share vision tensors between chosen/rejected to avoid duplicating large video tensors."""
    shared_keys = (
        "pixel_values",
        "pixel_values_videos",
        "image_grid_thw",
        "video_grid_thw",
        "second_per_grid_ts",
    )
    for k in shared_keys:
        if k not in rejected_inputs and k in chosen_inputs:
            rejected_inputs[k] = chosen_inputs[k]


def _forward_logps(
    model,
    model_inputs: dict,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Return (logp_sum, num_supervised_tokens) for this batch.

    Fast path: use `logits_to_keep=<indices>` to only compute logits at supervised positions.
    Fallback: compute full logits then select supervised positions.
    """
    batch_size = labels.shape[0]
    if batch_size != 1:
        outputs = model(**model_inputs, use_cache=False)
        return _get_batch_logps(outputs.logits, labels)

    shift_labels = labels[:, 1:]
    mask = shift_labels != -100
    positions = mask[0].nonzero(as_tuple=False).squeeze(1)  # positions in logits[:, :-1]
    if positions.numel() == 0:
        return (
            torch.zeros(1, device=labels.device, dtype=torch.float32),
            torch.zeros(1, device=labels.device, dtype=torch.long),
        )

    # Prefer CPU indices for broad compatibility across wrappers/backends.
    logits_to_keep = positions.detach().to("cpu")
    try:
        try:
            outputs = model(**model_inputs, use_cache=False, logits_to_keep=logits_to_keep)
        except TypeError:
            outputs = model(**model_inputs, use_cache=False, logits_to_keep=logits_to_keep.tolist())
        logits = outputs.logits
        targets = labels[:, positions + 1]  # (1, K)
        if os.environ.get("DPO_DEBUG_SHAPES"):
            try:
                rank = int(os.environ.get("RANK") or os.environ.get("LOCAL_RANK") or "0")
            except Exception:
                rank = 0
            if rank == 0:
                max_n = int(os.environ.get("DPO_DEBUG_SHAPES_N") or 3)
                debug_cnt = int(getattr(_forward_logps, "_debug_cnt", 0))
                if debug_cnt < max_n:
                    logger.warning(
                        f"[DPO_DEBUG] logits={tuple(logits.shape)} labels={tuple(labels.shape)} "
                        f"keep={int(positions.numel())} targets={tuple(targets.shape)}"
                    )
                    setattr(_forward_logps, "_debug_cnt", debug_cnt + 1)

        # Some model wrappers may ignore `logits_to_keep` and return full logits.
        # Support three cases:
        # 1) logits is already (1, K, V) aligned to `positions`
        # 2) logits is full (1, L, V) aligned to input_ids (or shifted)
        # 3) logits is last-N (1, N, V) (rare)
        if logits.dim() == 3:
            if logits.shape[1] == positions.numel():
                selected_logits = logits
            else:
                # Infer alignment of returned logits.
                if logits.shape[1] == labels.shape[1]:
                    base_logits = logits[:, :-1, :]
                    offset = 0
                elif logits.shape[1] == labels.shape[1] - 1:
                    base_logits = logits
                    offset = 0
                elif logits.shape[1] < labels.shape[1] - 1:
                    # Assume last-N of shift logits.
                    base_logits = logits
                    offset = (labels.shape[1] - 1) - logits.shape[1]
                else:
                    base_logits = logits[:, :-1, :]
                    offset = 0

                rel_pos = positions - offset
                rel_min = int(rel_pos.min().item())
                rel_max = int(rel_pos.max().item())
                if rel_pos.numel() != positions.numel() or rel_min < 0 or rel_max >= base_logits.shape[1]:
                    # Fallback to a full forward (without logits_to_keep) for correctness.
                    outputs_full = model(**model_inputs, use_cache=False)
                    return _get_batch_logps(outputs_full.logits, labels)
                selected_logits = base_logits[:, rel_pos, :]

            nll = F.cross_entropy(
                selected_logits.reshape(-1, selected_logits.size(-1)),
                targets.reshape(-1),
                reduction="sum",
            )
        else:
            # Unexpected logits shape, fallback to full forward for correctness.
            outputs_full = model(**model_inputs, use_cache=False)
            return _get_batch_logps(outputs_full.logits, labels)
        logp_sum = (-nll).view(1)
        length = torch.tensor([positions.numel()], device=labels.device, dtype=torch.long)
        return logp_sum, length
    except TypeError as exc:
        # Some models (e.g. Qwen2-VL) don't expose `logits_to_keep` on the top-level forward.
        # To avoid materializing full (B, L, V) logits (often OOM), compute hidden_states then slice lm_head manually.
        if "logits_to_keep" not in str(exc):
            raise

        base_model = getattr(model, "model", None)
        lm_head = getattr(model, "lm_head", None)
        if base_model is None or lm_head is None:
            outputs = model(**model_inputs, use_cache=False)
            return _get_batch_logps(outputs.logits, labels)

        outputs = base_model(
            **model_inputs,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        hidden_states = outputs[0]
        selected_logits = lm_head(hidden_states[:, logits_to_keep, :])
        nll = F.cross_entropy(
            selected_logits.reshape(-1, selected_logits.size(-1)),
            targets.reshape(-1),
            reduction="sum",
        )
        logp_sum = (-nll).view(1)
        length = torch.tensor([positions.numel()], device=labels.device, dtype=torch.long)
        return logp_sum, length
    except Exception as exc:
        # Some backends accept `logits_to_keep` but error at runtime (dtype/device expectations).
        if "logits_to_keep" in str(exc):
            outputs = model(**model_inputs, use_cache=False)
            return _get_batch_logps(outputs.logits, labels)
        raise


def compute_loss_dpo(self, model, inputs, return_outputs=False, num_items_in_batch=None):
    chosen_inputs = {k[len("chosen_"):]: v for k, v in inputs.items() if k.startswith("chosen_")}
    rejected_inputs = {k[len("rejected_"):]: v for k, v in inputs.items() if k.startswith("rejected_")}

    chosen_labels = chosen_inputs.pop("labels")
    rejected_labels = rejected_inputs.pop("labels")

    _fill_shared_vision_inputs(chosen_inputs, rejected_inputs)

    chosen_logps, chosen_lengths = _forward_logps(model, chosen_inputs, chosen_labels)
    rejected_logps, rejected_lengths = _forward_logps(model, rejected_inputs, rejected_labels)

    ref_model = getattr(self, "ref_model", None)
    if ref_model is None:
        raise ValueError("ref_model is required for DPO training")

    input_device = chosen_inputs["input_ids"].device
    ref_device = next(ref_model.parameters()).device
    if ref_device != input_device:
        ref_model.to(input_device)

    with torch.no_grad():
        ref_chosen_logps, _ = _forward_logps(ref_model, chosen_inputs, chosen_labels)
        ref_rejected_logps, _ = _forward_logps(ref_model, rejected_inputs, rejected_labels)

    if getattr(self, "dpo_length_norm", False):
        chosen_den = chosen_lengths.clamp(min=1)
        rejected_den = rejected_lengths.clamp(min=1)
        chosen_logps = chosen_logps / chosen_den
        rejected_logps = rejected_logps / rejected_den
        ref_chosen_logps = ref_chosen_logps / chosen_den
        ref_rejected_logps = ref_rejected_logps / rejected_den

    beta = float(getattr(self, "dpo_beta", 0.1))
    logits = beta * ((chosen_logps - rejected_logps) - (ref_chosen_logps - ref_rejected_logps))
    loss = (-F.logsigmoid(logits)).mean()
    return (loss, {"dpo_logits": logits}) if return_outputs else loss


if __name__ == "__main__":
    training_args, model_args, data_args, eval_data_args, dpo_args = HfArgumentParser(
        (TrainingArguments, ModelArguments, DataArguments, EvalDataArguments, DPOArguments)
    ).parse_args_into_dataclasses()

    # DPO dataset provides keys like `chosen_input_ids`/`rejected_input_ids`.
    # The default `remove_unused_columns=True` would drop them all (not in model.forward signature),
    # resulting in an empty batch before `compute_loss_dpo` sees the inputs.
    training_args.remove_unused_columns = False

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

    ref_path = dpo_args.ref_model_name_or_path or model_args.pretrained_model_name_or_path
    # NOTE: With ZeRO-3 enabled, `from_pretrained` will otherwise create a partitioned ref_model
    # (embedding weight becomes 1D), and calling `ref_model(...)` directly will crash:
    #   RuntimeError: 'weight' must be 2-D
    with _temporary_unset_hf_deepspeed_config():
        ref_model = getattr(transformers, config.architectures[0]).from_pretrained(
            ref_path,
            dtype="auto",
            attn_implementation='flash_attention_2'
        )
    ref_model.get_rope_index = MethodType(rope_getter, ref_model)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

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
    trainer.ref_model = ref_model
    trainer.dpo_beta = dpo_args.beta
    trainer.dpo_length_norm = dpo_args.length_norm
    trainer.compute_loss = MethodType(compute_loss_dpo, trainer)
    # Pass specific path or False depending on whether resuming training
    trainer.train(resume_from_checkpoint=resume_ckpt if resume_ckpt else False)
