# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");

import torch
import torch.nn as nn
from typing import Optional, List
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import BatchFeature

from accelerate.logging import get_logger

logger = get_logger(__name__)

IGNORE_INDEX = -100


class _PaliGemma_Interface(nn.Module):
    """
    Lightweight wrapper around PaliGemma/PaliGemma2 for starVLA.

    Supports:
    - PaliGemma 1: google/paligemma-3b-pt-224, google/paligemma-3b-mix-224, etc.
    - PaliGemma 2: google/paligemma2-3b-pt-224, google/paligemma2-3b-ft-docci-448, etc.
    """

    def __init__(self, config: Optional[dict] = None, **kwargs):
        super().__init__()

        qwenvl_config = config.framework.get("qwenvl", {})
        model_id = qwenvl_config.get("base_vlm", "google/paligemma-3b-pt-224")

        # Dynamic import of transformers
        import importlib
        tfm = importlib.import_module("transformers")

        # Select the correct class based on model name
        if "paligemma2" in model_id.lower():
            vlm_cls = getattr(tfm, "PaliGemma2ForConditionalGeneration")
        else:
            vlm_cls = getattr(tfm, "PaliGemmaForConditionalGeneration")

        # Load model (PaliGemma uses sdpa, does not support flash_attention_2)
        self.model = vlm_cls.from_pretrained(
            model_id,
            attn_implementation="sdpa",
            torch_dtype="auto",
            trust_remote_code=True,
            output_hidden_states=True,
        )
        # Ensure hidden states are always returned
        self.model.config.output_hidden_states = True
        if hasattr(self.model.config, 'text_config'):
            self.model.config.text_config.output_hidden_states = True

        # Load processor
        PaliGemmaProcessor = getattr(tfm, "PaliGemmaProcessor")
        self.processor = PaliGemmaProcessor.from_pretrained(model_id)
        self.processor.tokenizer.padding_side = "left"

        self.config = config
        self.image_token = "<image>"

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = True,
        return_dict: Optional[bool] = True,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """Forward pass delegating to underlying PaliGemma backbone."""

        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
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
        Construct and tokenize multimodal inputs for PaliGemma (batched).

        Parameters:
            images: List[List[PIL.Image.Image]] - list of images per sample
            instructions: List[str] - instruction text list
            solutions: List[str] | None - optional solution text (for training)
            states: reserved parameter for backward compatibility

        Returns:
            BatchFeature: contains input_ids, attention_mask, pixel_values, etc.
        """
        assert len(images) == len(instructions), "Images and instructions must have the same length"

        batch_texts = []
        batch_images = []

        for i, (imgs, instruction) in enumerate(zip(images, instructions)):
            # Build prompt, each image corresponds to one <image> token
            num_images = len(imgs)
            prompt = "".join([self.image_token] * num_images) + instruction

            # If CoT prompt exists, perform replacement
            if hasattr(self.config, 'datasets') and hasattr(self.config.datasets, 'vla_data'):
                if "CoT_prompt" in self.config.datasets.vla_data:
                    CoT_prompt = self.config.datasets.vla_data.get("CoT_prompt", "")
                    prompt = CoT_prompt.replace("{instruction}", prompt)

            batch_texts.append(prompt)
            batch_images.append(imgs if imgs else None)

        # Call processor
        if solutions is not None:
            # Training mode: use suffix
            batch_input = self.processor(
                text=batch_texts,
                images=batch_images,
                suffix=solutions,
                padding=True,
                return_tensors="pt",
            )
            # Set labels
            if "labels" in batch_input:
                labels = batch_input["labels"].clone()
                labels[labels == self.processor.tokenizer.pad_token_id] = IGNORE_INDEX
                batch_input["labels"] = labels
        else:
            # Inference mode
            batch_input = self.processor(
                text=batch_texts,
                images=batch_images,
                padding=True,
                return_tensors="pt",
            )

        return batch_input.to(self.model.device)


if __name__ == "__main__":
    from omegaconf import OmegaConf

    cfg = OmegaConf.create({
        "framework": {
            "qwenvl": {
                "base_vlm": "google/paligemma-3b-pt-224"
            }
        }
    })

    model = _PaliGemma_Interface(config=cfg)
    print("PaliGemma model loaded successfully!")
