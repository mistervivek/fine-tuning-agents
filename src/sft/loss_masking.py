"""Token-level loss masking for agent Supervised Fine-Tuning (SFT).

Critical Research Engineer insight:
In agent fine-tuning, loss MUST ONLY be calculated on assistant tokens (reasoning,
tool call parameters, final answers). Computing cross-entropy loss on User prompts
or Environment Tool observations causes the model to memorize external API outputs
and hallucinate database results instead of invoking tools.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
import torch


def create_masked_labels_for_conversation(
    tokenizer: Any,
    messages: List[Dict[str, str]],
    max_length: int = 2048,
    ignore_index: int = -100,
) -> Dict[str, torch.Tensor]:
    """Build input_ids, attention_mask, and labels where only assistant tokens have non-masked labels.

    Args:
        tokenizer: Hugging Face PreTrainedTokenizer
        messages: List of dicts with keys 'role' ('system', 'user', 'assistant', 'tool') and 'content'
        max_length: Maximum sequence length
        ignore_index: Target token value ignored by PyTorch CrossEntropyLoss (-100)

    Returns:
        Dict with tensors: 'input_ids', 'attention_mask', 'labels'
    """
    input_ids_list: List[int] = []
    labels_list: List[int] = []

    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        # Format turn header and content depending on chat template
        role_header = f"<|im_start|>{role}\n"
        role_footer = "<|im_end|>\n"
        full_turn = f"{role_header}{content}{role_footer}"

        turn_tokens = tokenizer.encode(full_turn, add_special_tokens=False)

        input_ids_list.extend(turn_tokens)

        # Loss Masking Decision:
        # Only compute loss if role == 'assistant'
        if role == "assistant":
            # Header is prompt syntax, so we can optionally mask header and unmask content + footer
            header_len = len(tokenizer.encode(role_header, add_special_tokens=False))
            # Mask the header tokens
            labels_list.extend([ignore_index] * header_len)
            # Unmask the assistant generated tokens and closing tag
            labels_list.extend(turn_tokens[header_len:])
        else:
            # Mask user prompts, system prompts, and tool observations completely
            labels_list.extend([ignore_index] * len(turn_tokens))

    # Truncate if exceeding max_length
    if len(input_ids_list) > max_length:
        input_ids_list = input_ids_list[:max_length]
        labels_list = labels_list[:max_length]

    attention_mask = [1] * len(input_ids_list)

    return {
        "input_ids": torch.tensor(input_ids_list, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels_list, dtype=torch.long),
    }


def compute_loss_mask_efficiency(labels: torch.Tensor, ignore_index: int = -100) -> float:
    """Calculate percentage of tokens trained on vs masked out."""
    total_tokens = labels.numel()
    if total_tokens == 0:
        return 0.0
    active_tokens = (labels != ignore_index).sum().item()
    return float(active_tokens) / float(total_tokens)
