"""Model loading and prompt formatting.

Deliberately thin: everything downstream operates on raw HF modules through
PyTorch forward hooks, so nothing here depends on TransformerLens or nnsight
(both of which commonly pin transformers<5).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class LoadedModel:
    name: str
    model: torch.nn.Module
    tokenizer: AutoTokenizer
    is_chat: bool
    supports_system: bool = True
    # Distinguishes runs of the same checkpoint at different precisions on disk.
    # Results key on this, not on `name`, so a 4-bit run cannot silently
    # overwrite the bf16 one.
    label: str = ""

    def __post_init__(self):
        if not self.label:
            self.label = self.name

    @property
    def layers(self) -> torch.nn.ModuleList:
        """The decoder layer stack. Covers Llama/Qwen/Mistral-style architectures."""
        inner = getattr(self.model, "model", self.model)
        if hasattr(inner, "layers"):
            return inner.layers
        raise AttributeError(f"cannot locate decoder layers on {type(self.model)}")

    @property
    def n_layers(self) -> int:
        return len(self.layers)

    @property
    def d_model(self) -> int:
        return self.model.config.hidden_size

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def format(self, user_content: str, system: str | None = None) -> str:
        """Render a single-turn prompt, ending where the model must answer.

        Chat models get the chat template with a generation prompt. Base models
        get a plain completion framing, since applying a chat template to a base
        checkpoint measures the template rather than the model.

        Not every chat template accepts a `system` role. Mistral-v0.3's raises.
        Where it does not, the persona is merged into the first user turn so the
        manipulation still reaches the model. `supports_system` is probed once at
        load time and recorded, because it changes what the persona conditions
        mean and therefore belongs in the write-up.
        """
        if self.is_chat:
            if system and self.supports_system:
                messages = [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_content},
                ]
            elif system:
                messages = [{"role": "user", "content": f"{system}\n\n{user_content}"}]
            else:
                messages = [{"role": "user", "content": user_content}]
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        prefix = f"{system}\n\n" if system else ""
        return f"{prefix}{user_content}\nAnswer:"


def _probe_system_support(tokenizer) -> bool:
    """Does this chat template accept a system role at all?"""
    try:
        tokenizer.apply_chat_template(
            [{"role": "system", "content": "x"}, {"role": "user", "content": "y"}],
            tokenize=False,
            add_generation_prompt=True,
        )
        return True
    except Exception:
        return False


def load_model(
    name: str,
    is_chat: bool | None = None,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    quant: str | None = None,
) -> LoadedModel:
    """Load a causal LM for hook-based analysis.

    `is_chat` is inferred from the presence of a chat template when not given.

    `quant` may be "4bit" or "8bit" (bitsandbytes), which is what makes models
    above ~10B reachable on a 24GB card. Quantisation is a confound for this
    project, it changes the very computation being measured, so any quantised
    run should be accompanied by a quantised run of a checkpoint whose full-
    precision behaviour is already known, to show the instrument still works.
    """
    tokenizer = AutoTokenizer.from_pretrained(name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Left padding keeps the final position aligned across a batch, which is
    # what every logprob read and residual capture below relies on.
    tokenizer.padding_side = "left"

    # Checkpoints that ship already quantised (e.g. unsloth/*-bnb-4bit) must be
    # loaded with device_map and no quantization_config; passing one, or calling
    # .to() afterwards, errors.
    from transformers import AutoConfig

    try:
        prequantized = getattr(AutoConfig.from_pretrained(name), "quantization_config", None) is not None
    except Exception:
        prequantized = False

    if prequantized:
        model = AutoModelForCausalLM.from_pretrained(name, device_map={"": device})
        quant = quant or "prequantized"
    elif quant:
        from transformers import BitsAndBytesConfig

        if quant == "4bit":
            cfg = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=dtype,
                bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
        elif quant == "8bit":
            cfg = BitsAndBytesConfig(load_in_8bit=True)
        else:
            raise ValueError(f"unknown quant {quant!r}; use '4bit' or '8bit'")
        # device_map places the model; calling .to() afterwards is an error.
        model = AutoModelForCausalLM.from_pretrained(
            name, quantization_config=cfg, device_map={"": device})
    else:
        model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype).to(device)

    model.eval()
    model.requires_grad_(False)

    if is_chat is None:
        is_chat = getattr(tokenizer, "chat_template", None) is not None

    supports_system = _probe_system_support(tokenizer) if is_chat else False

    return LoadedModel(
        name=name,
        model=model,
        tokenizer=tokenizer,
        is_chat=is_chat,
        supports_system=supports_system,
        label=f"{name}-{quant}" if quant else name,
    )
