import math

import torch
import yaml
import torch.nn.functional as F
from torch import nn

from lavis.models.blip2_models.blip2 import Blip2Base, disabled_train, LayerNorm
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers import LlamaTokenizer

import loralib as lora

from LoraViT import create_lora_eva_vit_g
from LoraQformer import BertConfig, BertLMHeadModel
from llm_model import AssistantVicuna


PRETRAINED_MODEL_CONFIG_DICT = {
    "vicuna7b": "configs/blip2_instruct_vicuna7b.yaml",
    "vicuna13b": "configs/blip2_instruct_vicuna13b.yaml",
}


class FINER_MLLM(Blip2Base):
    """
    UEM-MLLM model without RAG/ref-token constraints.

    Training objective:
        L_total = L_caption + L_ortho / (2 * sigma_ortho^2) + log(sigma_ortho)

    Difference-Aware Visual Cropping is used only in inference by calling:
        model.generate_with_crop(...)
    """

    def __init__(
        self,
        vit_model="eva_clip_g",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        num_query_token=32,
        llm_model="",
        prompt="",
        max_txt_len=128,
        max_output_txt_len=256,
        qformer_text_input=False,
        apply_lora_for_qformer=True,
        qformer_lora_k=4,
        vit_lora_k=16,
    ):
        super().__init__()

        self.qformer_lora_k = qformer_lora_k
        self.vit_lora_k = vit_lora_k

        # Learnable uncertainty parameter for L_ortho.
        self.log_sigma_ortho = nn.Parameter(torch.tensor(0.0))

        self.tokenizer = self.init_tokenizer()
        self.tokenizer.truncation_side = "left"

        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model,
            img_size,
            drop_path_rate,
            use_grad_checkpoint,
            vit_precision,
        )

        if freeze_vit:
            lora.mark_only_lora_as_trainable(self.visual_encoder)
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train

        if apply_lora_for_qformer:
            self.Qformer, self.query_tokens = self.init_LoraQformer(
                num_query_token,
                self.visual_encoder.num_features,
                lora_k=self.qformer_lora_k,
            )
            lora.mark_only_lora_as_trainable(self.Qformer)
        else:
            self.Qformer, self.query_tokens = self.init_Qformer(
                num_query_token,
                self.visual_encoder.num_features,
            )

        if not qformer_text_input:
            self.Qformer.bert.embeddings.word_embeddings = None
            self.Qformer.bert.embeddings.position_embeddings = None
            for layer in self.Qformer.bert.encoder.layer:
                layer.output = None
                layer.intermediate = None
        else:
            self.Qformer.resize_token_embeddings(len(self.tokenizer))

        self.Qformer.cls = None

        self.llm_tokenizer = LlamaTokenizer.from_pretrained(
            llm_model,
            use_fast=False,
            truncation_side="left",
        )
        self.llm_tokenizer.add_special_tokens({"pad_token": "</s>"})
        self.llm_tokenizer.pad_token = self.llm_tokenizer.unk_token

        llm_config = LlamaConfig.from_pretrained(llm_model)
        self.llm_model = AssistantVicuna(llm_model, llm_config)

        self.llm_model.config.pad_token_id = 0
        self.llm_model.config.bos_token_id = 1
        self.llm_model.config.eos_token_id = 2

        self.llm_proj = nn.Linear(
            self.Qformer.config.hidden_size,
            self.llm_model.config.hidden_size,
        )

        self.max_txt_len = max_txt_len
        self.max_output_txt_len = max_output_txt_len
        self.prompt = prompt
        self.qformer_text_input = qformer_text_input

    def _encode_image_pair(self, bef_image, aft_image, prompt):
        device = bef_image.device
        batch_size = bef_image.size(0)
        query_tokens = self.query_tokens.expand(batch_size, -1, -1)

        with self.maybe_autocast():
            bef_image_embeds = self.ln_vision(self.visual_encoder(bef_image))
            aft_image_embeds = self.ln_vision(self.visual_encoder(aft_image))

            bef_image_atts = torch.ones(
                bef_image_embeds.size()[:-1],
                dtype=torch.long,
                device=device,
            )
            aft_image_atts = torch.ones(
                aft_image_embeds.size()[:-1],
                dtype=torch.long,
                device=device,
            )

            if self.qformer_text_input:
                text_Qformer = self.tokenizer(
                    prompt,
                    padding="longest",
                    truncation=True,
                    max_length=self.max_txt_len,
                    return_tensors="pt",
                ).to(device)

                query_atts = torch.ones(
                    query_tokens.size()[:-1],
                    dtype=torch.long,
                    device=device,
                )
                Qformer_atts = torch.cat(
                    [query_atts, text_Qformer.attention_mask],
                    dim=1,
                )

                bef_query_output = self.Qformer.bert(
                    text_Qformer.input_ids,
                    attention_mask=Qformer_atts,
                    query_embeds=query_tokens,
                    encoder_hidden_states=bef_image_embeds,
                    encoder_attention_mask=bef_image_atts,
                    return_dict=True,
                    output_attentions=True,
                )
                aft_query_output = self.Qformer.bert(
                    text_Qformer.input_ids,
                    attention_mask=Qformer_atts,
                    query_embeds=query_tokens,
                    encoder_hidden_states=aft_image_embeds,
                    encoder_attention_mask=aft_image_atts,
                    return_dict=True,
                    output_attentions=True,
                )
            else:
                bef_query_output = self.Qformer.bert(
                    query_embeds=query_tokens,
                    encoder_hidden_states=bef_image_embeds,
                    encoder_attention_mask=bef_image_atts,
                    return_dict=True,
                    output_attentions=True,
                )
                aft_query_output = self.Qformer.bert(
                    query_embeds=query_tokens,
                    encoder_hidden_states=aft_image_embeds,
                    encoder_attention_mask=aft_image_atts,
                    return_dict=True,
                    output_attentions=True,
                )

            bef_llm = self.llm_proj(
                bef_query_output.last_hidden_state[:, : query_tokens.size(1), :]
            )
            aft_llm = self.llm_proj(
                aft_query_output.last_hidden_state[:, : query_tokens.size(1), :]
            )

            inputs_llm = torch.cat([bef_llm, aft_llm], dim=1)
            atts_llm = torch.ones(
                inputs_llm.size()[:-1],
                dtype=torch.long,
                device=device,
            )

        return inputs_llm, atts_llm, bef_query_output, aft_query_output

    def forward(self, bef_image, aft_image, caption):
        device = bef_image.device
        batch_size = bef_image.size(0)
        prompt = [self.prompt] * batch_size

        inputs_llm, atts_llm, bef_query_output, aft_query_output = self._encode_image_pair(
            bef_image,
            aft_image,
            prompt,
        )

        self.llm_tokenizer.padding_side = "right"
        self.llm_tokenizer.truncation_side = "left"

        text_input_tokens = self.llm_tokenizer(
            prompt,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
        ).to(device)

        self.llm_tokenizer.truncation_side = "right"
        text_output_tokens = self.llm_tokenizer(
            [t + self.llm_tokenizer.eos_token for t in caption],
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_output_txt_len,
        ).to(device)

        llm_tokens, input_part_targets_len = self.concat_text_input_output(
            text_input_tokens.input_ids,
            text_input_tokens.attention_mask,
            text_output_tokens.input_ids,
            text_output_tokens.attention_mask,
        )

        targets = llm_tokens["input_ids"].masked_fill(
            llm_tokens["input_ids"] == self.llm_tokenizer.pad_token_id,
            -100,
        )

        # Do not calculate loss on prompt tokens.
        for i, length in enumerate(input_part_targets_len):
            targets[i][:length] = -100

        # Do not calculate loss on visual query tokens.
        empty_targets = torch.ones(
            atts_llm.size(),
            dtype=torch.long,
            device=device,
        ).fill_(-100)
        targets = torch.cat([empty_targets, targets], dim=1)

        inputs_embeds = self.llm_model.backbone.get_input_embeddings()(
            llm_tokens["input_ids"]
        )
        inputs_embeds = torch.cat([inputs_llm, inputs_embeds], dim=1)
        attention_mask = torch.cat([atts_llm, llm_tokens["attention_mask"]], dim=1)

        with self.maybe_autocast():
            outputs = self.llm_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )

        orthogonal_loss = self.attention_orthogonal_loss(
            bef_query_output.cross_attentions[-2],
            aft_query_output.cross_attentions[-2],
        )

        sigma_ortho = torch.exp(self.log_sigma_ortho)
        ortho_weighted_loss = (
            orthogonal_loss / (2.0 * sigma_ortho.pow(2)) + self.log_sigma_ortho
        )

        loss = outputs.loss + ortho_weighted_loss
        return loss

    @torch.no_grad()
    def generate(
        self,
        bef_image,
        aft_image,
        use_nucleus_sampling=False,
        num_beams=5,
        max_length=50,
        min_length=5,
        top_p=0.9,
        repetition_penalty=1.5,
        length_penalty=1,
        num_captions=1,
        temperature=1,
    ):
        self.llm_tokenizer.padding_side = "left"

        batch_size = bef_image.size(0)
        device = bef_image.device
        prompt = [self.prompt] * batch_size

        inputs_llm, atts_llm, _, _ = self._encode_image_pair(
            bef_image,
            aft_image,
            prompt,
        )

        llm_tokens = self.llm_tokenizer(
            prompt,
            padding="longest",
            return_tensors="pt",
        ).to(device)

        prompt_length = llm_tokens.attention_mask.shape[1]

        with self.maybe_autocast():
            inputs_embeds = self.llm_model.backbone.get_input_embeddings()(
                llm_tokens.input_ids
            )
            inputs_embeds = torch.cat([inputs_llm, inputs_embeds], dim=1)
            attention_mask = torch.cat([atts_llm, llm_tokens.attention_mask], dim=1)

            outputs = self.llm_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                do_sample=use_nucleus_sampling,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_length=max_length,
                min_length=min_length,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
            )

        output_text = self.llm_tokenizer.batch_decode(
            outputs[:, prompt_length:],
            skip_special_tokens=True,
        )
        return [text.strip() for text in output_text]

    def generate_with_crop(
        self,
        bef_image,
        aft_image,
        use_nucleus_sampling=False,
        num_beams=5,
        max_length=50,
        min_length=5,
        top_p=0.9,
        repetition_penalty=1.5,
        length_penalty=1,
        num_captions=1,
        temperature=1,
        num_visual_tokens=16,
        bbox_base_size=224,
    ):
        """
        Inference-only Difference-Aware Visual Cropping.

        This function:
            1. computes gradient-aware saliency from Q-Former cross-attention;
            2. selects a shared crop box for before/after tensors;
            3. resizes the cropped pair back to the model input size;
            4. generates the caption from the cropped pair.

        Training never calls this function.
        """
        captions = []
        was_training = self.training
        self.eval()

        for i in range(bef_image.size(0)):
            bef_i = bef_image[i : i + 1]
            aft_i = aft_image[i : i + 1]

            bbox = self._infer_crop_bbox(
                bef_i,
                aft_i,
                num_visual_tokens=num_visual_tokens,
                bbox_base_size=bbox_base_size,
            )

            bef_crop = self._crop_and_resize_tensor(bef_i, bbox)
            aft_crop = self._crop_and_resize_tensor(aft_i, bbox)

            cap = self.generate(
                bef_crop,
                aft_crop,
                use_nucleus_sampling=use_nucleus_sampling,
                num_beams=num_beams,
                max_length=max_length,
                min_length=min_length,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_captions=num_captions,
                temperature=temperature,
            )
            captions.extend(cap)

        if was_training:
            self.train()

        return captions

    def _infer_crop_bbox(self, bef_image, aft_image, num_visual_tokens=16, bbox_base_size=224):
        """
        Compute a shared bbox for one before/after tensor pair.
        """
        device = bef_image.device
        _, _, image_h, image_w = bef_image.shape
        prompt = [self.prompt]

        # We need gradients only for the saliency computation.
        self.zero_grad(set_to_none=True)

        with torch.enable_grad():
            inputs_llm, atts_llm, bef_query_output, aft_query_output = self._encode_image_pair(
                bef_image,
                aft_image,
                prompt,
            )

            llm_tokens = self.llm_tokenizer(
                prompt,
                padding="longest",
                return_tensors="pt",
            ).to(device)

            with self.maybe_autocast():
                inputs_embeds = self.llm_model.backbone.get_input_embeddings()(
                    llm_tokens.input_ids
                )
                inputs_embeds = torch.cat([inputs_llm, inputs_embeds], dim=1)
                attention_mask = torch.cat([atts_llm, llm_tokens.attention_mask], dim=1)

                outputs = self.llm_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    return_dict=True,
                )

            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
            last_logits = logits[:, -1, :].float()
            pred_token = last_logits.argmax(dim=-1)
            score = F.log_softmax(last_logits, dim=-1).gather(1, pred_token[:, None]).sum()

            bef_att = bef_query_output.cross_attentions[-2]
            aft_att = aft_query_output.cross_attentions[-2]

            bef_grad, aft_grad = torch.autograd.grad(
                score,
                [bef_att, aft_att],
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )

            if bef_grad is None:
                bef_grad = torch.ones_like(bef_att)
            if aft_grad is None:
                aft_grad = torch.ones_like(aft_att)

            bef_map = self._attention_to_saliency_map(
                bef_att,
                bef_grad,
                num_visual_tokens=num_visual_tokens,
            )
            aft_map = self._attention_to_saliency_map(
                aft_att,
                aft_grad,
                num_visual_tokens=num_visual_tokens,
            )

            pair_map = torch.maximum(bef_map, aft_map)
            pair_map = self._normalize_map(pair_map)

        return self._bbox_from_attention_map(
            pair_map.detach(),
            image_h=image_h,
            image_w=image_w,
            bbox_base_size=bbox_base_size,
        )

    def _attention_to_saliency_map(self, att, grad, num_visual_tokens=16):
        """
        Convert Q-Former cross-attention and its gradient into a 2D saliency map.
        att/grad shape: [B, heads, query_tokens, image_tokens]
        """
        # Remove CLS token when visual tokens are 1 + square patches.
        num_source_tokens = att.shape[-1]
        if self._is_square(num_source_tokens - 1):
            start = 1
            patch_count = num_source_tokens - 1
        elif self._is_square(num_source_tokens):
            start = 0
            patch_count = num_source_tokens
        else:
            # Fallback: use visual encoder patch number.
            start = 1
            patch_count = self.visual_encoder.patch_embed.num_patches

        grid = int(round(math.sqrt(patch_count)))
        n_tokens = min(num_visual_tokens, att.shape[-2])

        att = att[0, :, :n_tokens, start : start + patch_count]
        grad = grad[0, :, :n_tokens, start : start + patch_count]

        saliency = (att * F.relu(grad)).mean(dim=(0, 1))  # [patch_count]
        saliency = saliency.reshape(grid, grid)
        return self._normalize_map(saliency)

    @staticmethod
    def _is_square(n):
        if n <= 0:
            return False
        r = int(round(math.sqrt(n)))
        return r * r == n

    @staticmethod
    def _normalize_map(att_map, eps=1e-8):
        att_map = torch.nan_to_num(att_map.float(), nan=0.0, posinf=0.0, neginf=0.0)
        att_map = torch.clamp(att_map, min=0.0)
        min_v = att_map.min()
        max_v = att_map.max()
        if (max_v - min_v) < eps:
            return torch.zeros_like(att_map)
        return (att_map - min_v) / (max_v - min_v + eps)

    @staticmethod
    def _resolution_aware_ratios(short_edge):
        if short_edge <= 256:
            return [0.2, 0.3, 0.4, 0.5, 0.6]
        if short_edge <= 448:
            return [0.4, 0.5, 0.6, 0.8, 1.0]
        if short_edge <= 768:
            return [0.5, 0.6, 0.8, 1.0, 1.2]
        return [0.5, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]

    def _bbox_from_attention_map(self, att_map, image_h, image_w, bbox_base_size=224):
        """
        Sliding-window crop selection on the attention map.
        Returns bbox in tensor coordinates: (x1, y1, x2, y2).
        """
        att = self._normalize_map(att_map)
        map_h, map_w = att.shape
        ratios = self._resolution_aware_ratios(min(image_h, image_w))

        block_w = image_w / float(map_w)
        block_h = image_h / float(map_h)

        best_candidate = None

        for ratio in ratios:
            crop_side = float(bbox_base_size) * float(ratio)
            win_w = max(1, min(int(round(crop_side / block_w)), map_w))
            win_h = max(1, min(int(round(crop_side / block_h)), map_h))

            slide_h = map_h - win_h + 1
            slide_w = map_w - win_w + 1
            if slide_h <= 0 or slide_w <= 0:
                continue

            scores = []
            for y in range(slide_h):
                row = []
                for x in range(slide_w):
                    row.append(att[y : y + win_h, x : x + win_w].sum())
                scores.append(torch.stack(row))
            scores = torch.stack(scores)

            best_idx = torch.argmax(scores)
            best_y = int(best_idx // slide_w)
            best_x = int(best_idx % slide_w)
            best_val = scores[best_y, best_x]

            neighbors = []
            if best_x > 0:
                neighbors.append(scores[best_y, best_x - 1])
            if best_x < slide_w - 1:
                neighbors.append(scores[best_y, best_x + 1])
            if best_y > 0:
                neighbors.append(scores[best_y - 1, best_x])
            if best_y < slide_h - 1:
                neighbors.append(scores[best_y + 1, best_x])

            if neighbors:
                contrast = (best_val - torch.stack(neighbors).mean()) / float(win_w * win_h)
            else:
                contrast = best_val / float(win_w * win_h)

            candidate = (float(contrast.detach().cpu()), best_x, best_y, win_w, win_h, crop_side)
            if best_candidate is None or candidate[0] > best_candidate[0]:
                best_candidate = candidate

        if best_candidate is None:
            return (0, 0, image_w, image_h)

        _, best_x, best_y, win_w, win_h, selected_side = best_candidate

        x_center = int(round(best_x * block_w + block_w * win_w / 2.0))
        y_center = int(round(best_y * block_h + block_h * win_h / 2.0))

        side = int(round(min(selected_side, image_w, image_h)))
        side = max(1, side)
        half = side // 2

        x1 = max(0, min(image_w - side, x_center - half))
        y1 = max(0, min(image_h - side, y_center - half))
        x2 = min(image_w, x1 + side)
        y2 = min(image_h, y1 + side)

        return (int(x1), int(y1), int(x2), int(y2))

    @staticmethod
    def _crop_and_resize_tensor(image_tensor, bbox):
        """
        Crop one image tensor and resize it back to the original resolution.
        image_tensor shape: [1, C, H, W]
        """
        _, _, image_h, image_w = image_tensor.shape
        x1, y1, x2, y2 = bbox

        crop = image_tensor[:, :, y1:y2, x1:x2]
        crop = F.interpolate(
            crop,
            size=(image_h, image_w),
            mode="bilinear",
            align_corners=False,
        )
        return crop

    def concat_text_input_output(self, input_ids, input_atts, output_ids, output_atts):
        input_part_targets_len = []
        llm_tokens = {"input_ids": [], "attention_mask": []}

        for i in range(input_ids.size(0)):
            this_input_ones = input_atts[i].sum()
            input_part_targets_len.append(this_input_ones)

            llm_tokens["input_ids"].append(
                torch.cat(
                    [
                        input_ids[i][:this_input_ones],
                        output_ids[i][1:],
                        input_ids[i][this_input_ones:],
                    ]
                )
            )
            llm_tokens["attention_mask"].append(
                torch.cat(
                    [
                        input_atts[i][:this_input_ones],
                        output_atts[i][1:],
                        input_atts[i][this_input_ones:],
                    ]
                )
            )

        llm_tokens["input_ids"] = torch.stack(llm_tokens["input_ids"])
        llm_tokens["attention_mask"] = torch.stack(llm_tokens["attention_mask"])

        return llm_tokens, input_part_targets_len

    @classmethod
    def load_pretrained_model_from_blip2(
        cls,
        model_type,
        qformer_lora_k=4,
        vit_lora_k=16,
        **kwargs,
    ):
        # kwargs is kept for compatibility with old scripts.
        # Removed arguments such as consist_w and ortho_w are ignored.
        cfg_path = PRETRAINED_MODEL_CONFIG_DICT[model_type]

        with open(cfg_path, "r", encoding="utf-8") as f:
            model_cfg = yaml.load(f, Loader=yaml.FullLoader)

        cfg = model_cfg["model"]

        model = cls(
            vit_model=cfg.get("vit_model", "eva_clip_g"),
            img_size=cfg.get("image_size"),
            drop_path_rate=cfg.get("drop_path_rate", 0),
            use_grad_checkpoint=cfg.get("use_grad_checkpoint", False),
            vit_precision=cfg.get("vit_precision", "fp16"),
            freeze_vit=cfg.get("freeze_vit", True),
            num_query_token=cfg.get("num_query_token"),
            llm_model=cfg.get("llm_model"),
            prompt=cfg.get("prompt", ""),
            max_txt_len=cfg.get("max_txt_len", 128),
            max_output_txt_len=cfg.get("max_output_txt_len", 256),
            qformer_lora_k=qformer_lora_k,
            vit_lora_k=vit_lora_k,
        )

        model.load_checkpoint_from_config(cfg)
        return model

    def attention_orthogonal_loss(self, bef_attention_score, aft_attention_score):
        num_patches = self.visual_encoder.patch_embed.num_patches + 1

        bef_attention_score = bef_attention_score[:, :, :, 1:num_patches]
        aft_attention_score = aft_attention_score[:, :, :, 1:num_patches]

        orthogonal_loss = (
            self.attention_orthogonal_regularization(bef_attention_score)
            + self.attention_orthogonal_regularization(aft_attention_score)
        ) / 2.0

        return orthogonal_loss

    def attention_orthogonal_regularization(self, attn):
        _, _, num_queries, num_tokens = attn.shape

        attn = F.softmax(attn, dim=-1)
        attn = attn.reshape(-1, num_queries, num_tokens)

        cosine_score = torch.matmul(attn, attn.permute(0, 2, 1).contiguous())
        soft_logits = -F.log_softmax(cosine_score / 0.05, dim=-1)

        return soft_logits.diagonal(dim1=1, dim2=2).mean()

    def init_LoraQformer(
        self,
        num_query_token,
        vision_width,
        cross_attention_freq=2,
        apply_lora=True,
        lora_k=4,
    ):
        encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        encoder_config.encoder_width = vision_width
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token
        encoder_config.apply_lora = apply_lora
        encoder_config.lora_r = lora_k
        encoder_config.lora_alpha = 1
        encoder_config.apply_adapter = False

        Qformer = BertLMHeadModel.from_pretrained(
            "bert-base-uncased",
            config=encoder_config,
        )

        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(
            mean=0.0,
            std=encoder_config.initializer_range,
        )

        return Qformer, query_tokens

    def init_vision_encoder(
        self,
        model_name,
        img_size,
        drop_path_rate,
        use_grad_checkpoint,
        precision,
    ):
        visual_encoder = create_lora_eva_vit_g(
            img_size,
            drop_path_rate,
            use_grad_checkpoint,
            precision,
            lora_r=self.vit_lora_k,
        )

        ln_vision = LayerNorm(visual_encoder.num_features)
        self.vit_name = model_name

        return visual_encoder, ln_vision
