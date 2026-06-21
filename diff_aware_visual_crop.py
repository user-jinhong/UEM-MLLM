from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw


ImageLike = Union[str, os.PathLike, Image.Image]
BBox = Tuple[int, int, int, int]
MapCombine = Literal["max", "mean", "sum", "diff", "diff_plus_max"]


@dataclass
class DifferenceAwareCropResult:
    """Return object for paired visual cropping."""

    bbox_before: BBox
    bbox_after: BBox
    crop_before: Image.Image
    crop_after: Image.Image
    att_before: np.ndarray
    att_after: np.ndarray
    att_pair: np.ndarray
    prompt: str

    def metadata(self) -> dict:
        """JSON-serializable metadata without PIL image payloads."""
        data = asdict(self)
        data.pop("crop_before", None)
        data.pop("crop_after", None)
        data["att_before_shape"] = list(self.att_before.shape)
        data["att_after_shape"] = list(self.att_after.shape)
        data["att_pair_shape"] = list(self.att_pair.shape)
        data.pop("att_before", None)
        data.pop("att_after", None)
        data.pop("att_pair", None)
        return data


def load_rgb_image(image: ImageLike) -> Image.Image:
    """Load an image path/PIL image and return RGB PIL image."""
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    return Image.open(image).convert("RGB")


def _get_model_device_dtype(model: torch.nn.Module) -> tuple[torch.device, torch.dtype]:
    """Infer the main device and floating dtype of a model."""
    try:
        p = next(model.parameters())
        return p.device, p.dtype if p.is_floating_point() else torch.float32
    except StopIteration:
        return torch.device("cpu"), torch.float32


def _move_batch_to_device(batch, device: torch.device, dtype: torch.dtype):
    """Move a HF BatchFeature to device; only cast floating tensors."""
    for k, v in list(batch.items()):
        if torch.is_tensor(v):
            if v.is_floating_point():
                batch[k] = v.to(device=device, dtype=dtype)
            else:
                batch[k] = v.to(device=device)
    return batch


def _is_square_token_count(n: int) -> bool:
    if n <= 0:
        return False
    r = int(round(math.sqrt(n)))
    return r * r == n


def _patch_start_and_grid(num_source_tokens: int, remove_cls_token: bool = True) -> tuple[int, int]:
    """
    Decide whether Q-Former cross-attention source tokens include a CLS token.

    For ViT-L/14 at 224, source tokens are often 257 = 1 CLS + 16*16 patches.
    The provided reference code slices q_former_atts[..., 1:], so we follow that
    when num_source_tokens - 1 is a square.
    """
    if remove_cls_token and _is_square_token_count(num_source_tokens - 1):
        patch_count = num_source_tokens - 1
        return 1, int(round(math.sqrt(patch_count)))
    if _is_square_token_count(num_source_tokens):
        return 0, int(round(math.sqrt(num_source_tokens)))
    raise ValueError(
        f"Cannot infer square patch grid from {num_source_tokens} Q-Former source tokens. "
        "Check whether your vision encoder uses a non-square patch layout."
    )


def normalize_attention_map(att_map: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Replace invalid values and min-max normalize an attention map."""
    x = np.asarray(att_map, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = np.maximum(x, 0.0)
    max_v = float(x.max()) if x.size else 0.0
    min_v = float(x.min()) if x.size else 0.0
    if max_v - min_v < eps:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - min_v) / (max_v - min_v + eps)).astype(np.float32)


def caption_to_image_grad_attention_blip(
    image: ImageLike,
    prompt: str,
    model,
    processor,
    *,
    lm_layer: int = 15,
    qformer_layer: int = 2,
    num_visual_tokens: int = 16,
    remove_cls_token: bool = True,
    normalize: bool = True,
) -> np.ndarray:
    """
    Build a gradient-weighted caption-to-image saliency map for one image.

    Implements the paper equations in code form:
        G_ct = A_ct * ReLU(d s / d A_ct)
        G_ti = A_ti * ReLU(d s / d A_ti)
        G_ci = G_ct @ G_ti

    Args:
        image: PIL image or image path.
        prompt: ICC prompt, e.g. "Question: Describe the visual changes between these images. Short answer:".
        model: InstructBLipForConditionalGeneration or compatible model.
        processor: InstructBlipProcessor or compatible processor.
        lm_layer: Decoder layer m used for caption-to-token attention.
        qformer_layer: Q-Former cross-attention layer k used for token-to-image attention.
        num_visual_tokens: N visual/query tokens selected from the connector. The paper uses N=16.
        remove_cls_token: Drop the visual encoder CLS token when present.
        normalize: Whether to min-max normalize the final 2D map.

    Returns:
        np.ndarray of shape [patch_grid, patch_grid], usually 16x16 for 224/14.
    """
    pil_image = load_rgb_image(image)
    device, dtype = _get_model_device_dtype(model)
    if device.type == "cpu" and dtype in (torch.float16, torch.bfloat16):
        # Safer CPU fallback.
        dtype = torch.float32

    model.eval()
    model.zero_grad(set_to_none=True)

    inputs = processor(images=pil_image, text=prompt, return_tensors="pt", padding=True)
    inputs = _move_batch_to_device(inputs, device, dtype)

    # Gradients must be enabled even though the model is in eval mode.
    outputs = model(**inputs, output_attentions=True, return_dict=True)

    if not hasattr(outputs, "qformer_outputs") or not hasattr(outputs, "language_model_outputs"):
        raise RuntimeError("Model outputs do not expose qformer_outputs/language_model_outputs.")

    qformer_atts = outputs.qformer_outputs.cross_attentions
    lm_atts = outputs.language_model_outputs.attentions
    if qformer_atts is None or lm_atts is None:
        raise RuntimeError("Attention tensors are None. Ensure output_attentions=True and attn_implementation='eager' if needed.")

    q_att_layer = qformer_atts[qformer_layer]
    lm_att_layer = lm_atts[lm_layer]

    # Score s = log prob of the first/current predicted token, following the reference code's
    # "take argmax token and backprop its log-probability" behavior.
    last_logits = outputs.logits[:, -1, :].float()
    pred_token = last_logits.argmax(dim=-1)
    score = F.log_softmax(last_logits, dim=-1).gather(1, pred_token[:, None]).sum()

    q_grad, lm_grad = torch.autograd.grad(
        score,
        [q_att_layer, lm_att_layer],
        retain_graph=False,
        create_graph=False,
        allow_unused=False,
    )

    # Q-Former cross attention: [batch, heads, query_tokens, image_tokens]
    source_start, patch_grid = _patch_start_and_grid(q_att_layer.shape[-1], remove_cls_token)
    q_available = q_att_layer.shape[-2]
    lm_available = lm_att_layer.shape[-1]
    n_tokens = min(num_visual_tokens, q_available, lm_available)
    if n_tokens <= 0:
        raise ValueError("No overlapping visual/query tokens found between LM attention and Q-Former attention.")

    token_to_image_att = q_att_layer[0, :, :n_tokens, source_start:]
    token_to_image_grad = q_grad[0, :, :n_tokens, source_start:]

    # LM self/cross attention over projected visual/query prefix tokens:
    # [batch, heads, seq, seq], take last generated position -> first n visual tokens.
    caption_to_token_att = lm_att_layer[0, :, -1, :n_tokens]
    caption_to_token_grad = lm_grad[0, :, -1, :n_tokens]

    g_ti = (token_to_image_att * F.relu(token_to_image_grad)).mean(dim=0)  # [N, patches]
    g_ct = (caption_to_token_att * F.relu(caption_to_token_grad)).mean(dim=0)  # [N]

    saliency = torch.matmul(g_ct.unsqueeze(0), g_ti).squeeze(0)  # [patches]
    saliency = saliency.detach().float().cpu().numpy()

    expected = patch_grid * patch_grid
    if saliency.size != expected:
        raise ValueError(f"Expected {expected} patch values, got {saliency.size}.")

    att_map = saliency.reshape(patch_grid, patch_grid)
    return normalize_attention_map(att_map) if normalize else att_map


def get_resolution_aware_ratios(short_edge: int) -> list[float]:
    """Crop ratios from the paper's resolution-aware strategy."""
    if short_edge <= 256:
        return [0.2, 0.3, 0.4, 0.5, 0.6]
    if short_edge <= 448:
        return [0.4, 0.5, 0.6, 0.8, 1.0]
    if short_edge <= 768:
        return [0.5, 0.6, 0.8, 1.0, 1.2]
    return [0.5, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]


def bbox_from_attention_map_resolution_aware(
    att_map: np.ndarray,
    image_size: tuple[int, int],
    *,
    bbox_base_size: int = 224,
    ratios: Optional[Sequence[float]] = None,
) -> BBox:
    """
    Convert a 2D saliency map into an image-space square bbox.

    For each crop ratio, this slides a window across the attention map, picks the
    max-sum location, then selects the ratio whose max window most differs from
    its immediate neighbors. This follows the provided reference implementation,
    with the paper's resolution-aware ratios.
    """
    att = normalize_attention_map(att_map)
    map_h, map_w = att.shape
    img_w, img_h = image_size
    if map_h <= 0 or map_w <= 0 or img_w <= 0 or img_h <= 0:
        raise ValueError(f"Invalid att_map shape {att.shape} or image_size {image_size}.")

    if ratios is None:
        ratios = get_resolution_aware_ratios(min(img_w, img_h))

    block_w = img_w / float(map_w)
    block_h = img_h / float(map_h)

    candidates: list[tuple[float, tuple[int, int], tuple[int, int], float]] = []

    for ratio in ratios:
        crop_side = float(bbox_base_size) * float(ratio)
        win_w = max(1, min(int(round(crop_side / block_w)), map_w))
        win_h = max(1, min(int(round(crop_side / block_h)), map_h))

        # If the ratio covers the whole attention map and there are smaller ratios,
        # still evaluate it but its neighbor contrast will usually be low.
        slide_h = map_h - win_h + 1
        slide_w = map_w - win_w + 1
        if slide_h <= 0 or slide_w <= 0:
            continue

        sliding = np.zeros((slide_h, slide_w), dtype=np.float32)
        best_val = -np.inf
        best_pos = (0, 0)
        for y in range(slide_h):
            for x in range(slide_w):
                val = float(att[y : y + win_h, x : x + win_w].sum())
                sliding[y, x] = val
                if val > best_val:
                    best_val = val
                    best_pos = (x, y)

        neighbors = []
        x, y = best_pos
        if x > 0:
            neighbors.append(sliding[y, x - 1])
        if x < slide_w - 1:
            neighbors.append(sliding[y, x + 1])
        if y > 0:
            neighbors.append(sliding[y - 1, x])
        if y < slide_h - 1:
            neighbors.append(sliding[y + 1, x])

        if neighbors:
            contrast = (best_val - float(np.mean(neighbors))) / float(win_w * win_h)
        else:
            contrast = best_val / float(win_w * win_h)
        candidates.append((contrast, best_pos, (win_w, win_h), crop_side))

    if not candidates:
        return (0, 0, img_w, img_h)

    _, best_pos, best_win, selected_side = max(candidates, key=lambda x: x[0])
    best_x, best_y = best_pos
    win_w, win_h = best_win

    x_center = int(round(best_x * block_w + block_w * win_w / 2.0))
    y_center = int(round(best_y * block_h + block_h * win_h / 2.0))

    side = int(round(min(selected_side, img_w, img_h)))
    side = max(1, side)
    half = side // 2

    x1 = max(0, min(img_w - side, x_center - half))
    y1 = max(0, min(img_h - side, y_center - half))
    x2 = min(img_w, x1 + side)
    y2 = min(img_h, y1 + side)
    return (int(x1), int(y1), int(x2), int(y2))


def combine_pair_attention(att_before: np.ndarray, att_after: np.ndarray, mode: MapCombine = "max") -> np.ndarray:
    """Combine before/after saliency maps into one pair-level saliency map."""
    a = normalize_attention_map(att_before)
    b = normalize_attention_map(att_after)
    if a.shape != b.shape:
        raise ValueError(f"Attention maps must have same shape, got {a.shape} and {b.shape}.")

    if mode == "max":
        out = np.maximum(a, b)
    elif mode == "mean":
        out = 0.5 * (a + b)
    elif mode == "sum":
        out = a + b
    elif mode == "diff":
        out = np.abs(a - b)
    elif mode == "diff_plus_max":
        out = np.abs(a - b) + np.maximum(a, b)
    else:
        raise ValueError(f"Unknown combine mode: {mode}")
    return normalize_attention_map(out)


def difference_aware_visual_crop_pair(
    before_image: ImageLike,
    after_image: ImageLike,
    prompt: str,
    model,
    processor,
    *,
    lm_layer: int = 15,
    qformer_layer: int = 2,
    num_visual_tokens: int = 16,
    bbox_base_size: int = 224,
    combine: MapCombine = "max",
) -> DifferenceAwareCropResult:
    """
    Compute Difference-Aware Visual Cropping for an ICC image pair.

    If before/after images have the same size, one pair-level bbox is selected and
    applied to both images. If sizes differ, separate bboxes are selected.
    """
    before = load_rgb_image(before_image)
    after = load_rgb_image(after_image)

    att_before = caption_to_image_grad_attention_blip(
        before,
        prompt,
        model,
        processor,
        lm_layer=lm_layer,
        qformer_layer=qformer_layer,
        num_visual_tokens=num_visual_tokens,
    )
    att_after = caption_to_image_grad_attention_blip(
        after,
        prompt,
        model,
        processor,
        lm_layer=lm_layer,
        qformer_layer=qformer_layer,
        num_visual_tokens=num_visual_tokens,
    )

    if before.size == after.size and att_before.shape == att_after.shape:
        att_pair = combine_pair_attention(att_before, att_after, mode=combine)
        bbox = bbox_from_attention_map_resolution_aware(
            att_pair,
            before.size,
            bbox_base_size=bbox_base_size,
        )
        bbox_before = bbox_after = bbox
    else:
        # Fallback for uncommon ICC pairs with different image sizes.
        att_pair = combine_pair_attention(att_before, att_after, mode=combine) if att_before.shape == att_after.shape else att_before
        bbox_before = bbox_from_attention_map_resolution_aware(
            att_before,
            before.size,
            bbox_base_size=bbox_base_size,
        )
        bbox_after = bbox_from_attention_map_resolution_aware(
            att_after,
            after.size,
            bbox_base_size=bbox_base_size,
        )

    crop_before = before.crop(bbox_before)
    crop_after = after.crop(bbox_after)

    return DifferenceAwareCropResult(
        bbox_before=bbox_before,
        bbox_after=bbox_after,
        crop_before=crop_before,
        crop_after=crop_after,
        att_before=att_before,
        att_after=att_after,
        att_pair=att_pair,
        prompt=prompt,
    )


def draw_bbox(image: ImageLike, bbox: BBox, width: int = 4) -> Image.Image:
    """Return a copy of image with bbox drawn in red."""
    img = load_rgb_image(image).copy()
    draw = ImageDraw.Draw(img)
    draw.rectangle(bbox, outline="red", width=width)
    return img


def save_crop_result(result: DifferenceAwareCropResult, before_image: ImageLike, after_image: ImageLike, output_dir: Union[str, os.PathLike]) -> None:
    """Save crops, bbox visualizations, saliency maps, and metadata."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    result.crop_before.save(out / "before_crop.png")
    result.crop_after.save(out / "after_crop.png")
    draw_bbox(before_image, result.bbox_before).save(out / "before_bbox.png")
    draw_bbox(after_image, result.bbox_after).save(out / "after_bbox.png")

    np.save(out / "att_before.npy", result.att_before)
    np.save(out / "att_after.npy", result.att_after)
    np.save(out / "att_pair.npy", result.att_pair)

    with open(out / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(result.metadata(), f, ensure_ascii=False, indent=2)


def build_default_icc_prompt() -> str:
    """Default prompt used by the paper for ICC-style generation."""
    return "Question: Describe the visual changes between these images. Short answer:"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Difference-Aware Visual Cropping for ICC image pairs")
    parser.add_argument("--before", required=True, help="Path to before/original image")
    parser.add_argument("--after", required=True, help="Path to after/edited image")
    parser.add_argument("--model_id", default="Salesforce/instructblip-vicuna-7b", help="HF InstructBLIP model id/path")
    parser.add_argument("--output_dir", default="diff_aware_crop_out", help="Directory to save crops and metadata")
    parser.add_argument("--prompt", default=build_default_icc_prompt(), help="Text prompt for attention computation")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--lm_layer", type=int, default=15)
    parser.add_argument("--qformer_layer", type=int, default=2)
    parser.add_argument("--num_visual_tokens", type=int, default=16)
    parser.add_argument("--bbox_base_size", type=int, default=224)
    parser.add_argument("--combine", choices=["max", "mean", "sum", "diff", "diff_plus_max"], default="max")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    from transformers import InstructBlipForConditionalGeneration, InstructBlipProcessor

    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    model = InstructBlipForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(args.device)
    processor = InstructBlipProcessor.from_pretrained(args.model_id)

    result = difference_aware_visual_crop_pair(
        args.before,
        args.after,
        args.prompt,
        model,
        processor,
        lm_layer=args.lm_layer,
        qformer_layer=args.qformer_layer,
        num_visual_tokens=args.num_visual_tokens,
        bbox_base_size=args.bbox_base_size,
        combine=args.combine,
    )
    save_crop_result(result, args.before, args.after, args.output_dir)
    print(json.dumps(result.metadata(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
