import torch
from typing import Optional, Union, List
from transformers import HunYuanVLForConditionalGeneration
from transformers.modeling_outputs import CausalLMOutputWithPast

class MyCustomHunYuanVL(HunYuanVLForConditionalGeneration):
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        
        # 1. Nếu chưa có inputs_embeds, ta tạo nó từ input_ids
        if inputs_embeds is None and input_ids is not None:
            print("Creating input embeddings from input_ids...")
            inputs_embeds = self.model.embed_tokens(input_ids).clone()

        # 2. Xử lý Hình ảnh (Logic được mang từ generate sang)
        if pixel_values is not None and self.vit is not None:
            print("Processing image inputs...")
            # Đảm bảo cùng kiểu dữ liệu với model (thường là bfloat16)
            pixel_values = pixel_values.to(self.dtype)
            image_embeds = self.vit(pixel_values, image_grid_thw)

            # Đưa về cùng device với LLM
            device = inputs_embeds.device
            image_embeds = image_embeds.to(device, non_blocking=True)

            # Tìm vị trí các token hình ảnh để chèn embedding vào
            image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            
            # Chèn embedding ảnh vào đúng vị trí mask
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        # 3. Gọi model LLM cơ sở (Truyền input_ids=None vì đã dùng inputs_embeds)
        outputs = self.model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        # 4. Tính toán Loss (Rất quan trọng cho Trainer)
        loss = None
        if labels is not None:
            print("Calculating loss...")
            loss = self.loss_function(
                logits=logits, 
                labels=labels, 
                vocab_size=self.config.vocab_size, 
                **kwargs
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
