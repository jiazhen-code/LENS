"""LLaVA-1.5 MLLM backbone.

This wraps the frozen ``LlavaForConditionalGeneration`` and exposes the ``encode`` method
that the conditioner consumes. The body of ``encode`` is the original
``MLLM_Conditioner.encode`` moved here unchanged -- only the surrounding object that owns
``model`` / ``processor`` / ``selected_layer_id`` has changed.
"""

from typing import List, Tuple, Union

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor

from ..llava import LlavaForConditionalGeneration
from .base import MLLMBackbone, force_eager_attention, register_backbone


@register_backbone("llava-1.5-7b")
class LlavaBackbone(MLLMBackbone):
    def __init__(self, cfg):
        super().__init__()
        self.model = LlavaForConditionalGeneration.from_pretrained(
            cfg.model_name,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        self.processor = AutoProcessor.from_pretrained(cfg.model_name)

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        self.hidden_size = cfg.hidden_size
        self.num_image_tokens = cfg.num_image_tokens
        self.image_grid_size = int(cfg.num_image_tokens ** 0.5)
        self.selected_layer_id = cfg.selected_layer_id
        self.grounding_init_start_layer = cfg.grounding_init_start_layer

    @property
    def dtype(self):
        return self.model.dtype

    def _text_model(self):
        return self.model.language_model

    def build_fusion_model(self, num_layers):
        """Reproduce the original LLaVA-shaped fusion head EXACTLY (fresh ``LlamaConfig``
        with default eps/rope), so the paper's LLaVA results stay bit-for-bit unchanged --
        rather than deep-copying LLaVA's full config. The mirrored dims (hidden / heads /
        intermediate) still come from LLaVA's own layer.
        """
        from transformers import LlamaConfig
        from transformers.models.llama.modeling_llama import LlamaModel
        src = self.model.language_model.config
        config = LlamaConfig(
            hidden_size=self.hidden_size,
            num_attention_heads=src.num_attention_heads,
            intermediate_size=src.intermediate_size,
            num_hidden_layers=num_layers,
            output_attentions=True,
        )
        config._attn_implementation = "eager"
        fusion = LlamaModel(config)
        del fusion.embed_tokens
        return force_eager_attention(fusion)

    @torch.no_grad
    def encode(
        self,
        image: Union[np.ndarray, List[str], List[Image.Image]],
        text: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        image_inputs = []
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        if isinstance(image, np.ndarray):
            image_inputs = [Image.fromarray(img.astype('uint8'), 'RGB') for img in image]
        elif isinstance(image, list):
            image_inputs = [Image.open(item).convert("RGB") if isinstance(item, str) else item for item in image]

        # prompts = [f"<image>\nUSER: {t}\nASSISTANT:" for t in text]
        prompts = text
        inputs = self.processor(
            text=prompts,
            images=image_inputs,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        # Manually move to the device of the model's first parameter
        inputs = {
            k: (
                v.to(self.model.device).to(self.model.dtype)  # 如果是浮点，转换 device 和 dtype
                if v.is_floating_point()
                else v.to(self.model.device)  # 如果是整数，只转换 device
            )
            for k, v in inputs.items()
            if isinstance(v, torch.Tensor)  # 最好还是加一个检查，确保 v 是张量
        }

        outputs = self.model(**inputs, output_hidden_states=True)
        # Using a later layer might capture higher-level concepts
        layer_id = self.selected_layer_id  # The last hidden state
        multimodal_features = outputs.hidden_states[layer_id]
        last_multimodal_features = outputs.last_hidden_state
        attentions = outputs.attentions

        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        # 1. Isolate the prompt part of the text (everything up to and including "ASSISTANT:")
        #    This assumes your input text is formatted with "ASSISTANT:" as a separator.
        prompts_until_assistant = [t.split('ASSISTANT:')[0] + 'ASSISTANT:' for t in text]

        # 2. Tokenize the prompt part with the images to get the exact length in tokens.
        #    This is crucial because the number of image tokens can vary.
        len_inputs = self.processor(
            text=prompts_until_assistant,
            images=image_inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        len_inputs = {k: v.to(self.model.device) for k, v in len_inputs.items()}

        # 3. Calculate the number of tokens in the prompt and the total number of non-padding tokens.
        prompt_lengths = len_inputs['attention_mask'].sum(dim=1)
        total_lengths = attention_mask.sum(dim=1)

        # 4. The number of answer tokens is the difference.
        answer_lengths = total_lengths - prompt_lengths

        # 5. Create a mask that is True only for the answer tokens.
        #    This works for both left and right padding because the answer is always
        #    the last `answer_lengths` tokens of the non-padded sequence.
        seq_len = input_ids.shape[1]
        indices = torch.arange(seq_len, device=self.model.device).unsqueeze(0).expand(input_ids.shape[0], -1)

        # The answer starts at the index marking the end of all padding and prompt tokens.
        answer_start_indices = seq_len - answer_lengths
        answer_mask = indices >= answer_start_indices.unsqueeze(1)

        # Ensure we don't include padding tokens in the answer.
        final_answer_mask = answer_mask & (attention_mask == 1)

        # 5. Create the `ids` tensor. Start with a copy of input_ids.
        ids = input_ids.clone()

        ids[~final_answer_mask] = -100

        # LLaVA 1.5 has 576 image tokens
        num_image_tokens = 576
        image_token_index = self.processor.tokenizer.convert_tokens_to_ids('<image>')

        # Create a mask for image tokens. This is more robust.
        image_token_mask = (input_ids == image_token_index)

        # Text tokens are everything that is not an image token and not padding
        initial_text_mask = (input_ids != image_token_index) & (attention_mask == 1)

        last_img_indices = image_token_mask.cumsum(dim=1).argmax(dim=1)

        # 4. Create a positional mask that is True for all tokens AFTER the last image token.
        seq_len = input_ids.shape[1]
        indices = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        after_image_mask = indices > last_img_indices.unsqueeze(1)  # Broadcasting comparison

        # 5. The final text_token_mask is the intersection of the initial text mask
        #    and the "after image" positional mask.
        text_token_mask = initial_text_mask & after_image_mask

        # ids = input_ids.clone()
        ids[attention_mask == 0] = -100

        return multimodal_features, attention_mask, image_token_mask, text_token_mask, last_multimodal_features, ids
