# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");

import torch
import torch.nn as nn
from typing import Optional, List
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import AutoModelForImageTextToText, AutoProcessor

from accelerate.logging import get_logger

logger = get_logger(__name__)

IGNORE_INDEX = -100


class _SmolVLM_Interface(nn.Module):
    """
    Lightweight wrapper around SmolVLM for starVLA.

    Supports:
    - SmolVLM2-500M-Video-Instruct
    - SmolVLM2-2.2B-Instruct
    - Other SmolVLM variants
    """

    def __init__(self, config: Optional[dict] = None, **kwargs):
        super().__init__()

        qwenvl_config = config.framework.get("qwenvl", {})
        model_id = qwenvl_config.get("base_vlm", "HuggingFaceTB/SmolVLM2-2.2B-Instruct")

        # Load model
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype="auto",
            trust_remote_code=True,
            output_hidden_states=True,
        )
        # Ensure hidden states are always returned
        self.model.config.output_hidden_states = True
        if hasattr(self.model.config, 'text_config'):
            self.model.config.text_config.output_hidden_states = True

        # Load processor
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.processor.tokenizer.padding_side = "left"

        self.config = config

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_attention_mask: Optional[torch.BoolTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = True,
        return_dict: Optional[bool] = True,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """Forward pass delegating to underlying SmolVLM backbone."""

        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                pixel_attention_mask=pixel_attention_mask,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                **kwargs,
            )
        return outputs

    def generate(self, **kwargs):
        """Generation interface for auto-regressive decoding."""
        with torch.autocast("cuda", dtype=torch.float16):
            generation_output = self.model.generate(**kwargs)
        return generation_output

    def build_qwenvl_inputs(self, images, instructions, solutions=None, states=None, **kwargs):
        """
        Construct and tokenize multimodal inputs for SmolVLM (batched).

        Parameters:
            images: List[List[PIL.Image.Image]] - list of images per sample
            instructions: List[str] - instruction text list
            solutions: List[str] | None - optional solution text (for training)
            states: reserved parameter for backward compatibility

        Returns:
            BatchFeature: contains input_ids, attention_mask, pixel_values, etc.
        """
        assert len(images) == len(instructions), "Images and instructions must have the same length"

        messages_batch = []

        for i, (imgs, instruction) in enumerate(zip(images, instructions)):
            # Build content with images
            content = []
            for img in imgs:
                content.append({"type": "image", "image": img})

            # Build the prompt
            prompt = instruction
            if hasattr(self.config, 'datasets') and hasattr(self.config.datasets, 'vla_data'):
                if "CoT_prompt" in self.config.datasets.vla_data:
                    CoT_prompt = self.config.datasets.vla_data.get("CoT_prompt", "")
                    prompt = CoT_prompt.replace("{instruction}", instruction)

            content.append({"type": "text", "text": prompt})

            # Build message
            msg = [{"role": "user", "content": content}]

            if solutions is not None and i < len(solutions):
                msg.append({"role": "assistant", "content": [{"type": "text", "text": solutions[i]}]})

            messages_batch.append(msg)

        # Apply chat template - pass messages directly to get proper image token handling
        texts = self.processor.apply_chat_template(
            messages_batch,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Process inputs - pass images as list of lists (per sample)
        has_images = any(len(imgs) > 0 for imgs in images)
        if has_images:
            batch_input = self.processor(
                text=texts,
                images=images,
                padding=True,
                return_tensors="pt",
            )
        else:
            batch_input = self.processor(
                text=texts,
                padding=True,
                return_tensors="pt",
            )

        # Set labels for training
        if solutions is not None:
            labels = batch_input['input_ids'].clone()
            labels[labels == self.processor.tokenizer.pad_token_id] = IGNORE_INDEX
            batch_input['labels'] = labels

        return batch_input.to(self.model.device)


if __name__ == "__main__":
    from omegaconf import OmegaConf

    cfg = OmegaConf.create({
        "framework": {
            "qwenvl": {
                "base_vlm": "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
            }
        }
    })

    model = _SmolVLM_Interface(config=cfg)
    print("SmolVLM model loaded successfully!")
