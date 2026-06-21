
import argparse
import concurrent.futures
import json
import os
import random
import threading
from pathlib import Path

import cv2
import clip
import lpips
import numpy as np
import torch
from diffusers import FluxKontextPipeline
from PIL import Image
from qwen_vl_utils import process_vision_info
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


# Global objects initialized in main().
qwen_model = None
qwen_processor = None
pipe = None
loss_fn = None
clip_model = None
clip_preprocess = None
clip_device = None

# Locks are used because several model calls are not thread-safe.
model_lock = threading.Lock()
data_lock = threading.Lock()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate synthetic ICC samples with Qwen2.5-VL and FLUX-Kontext."
    )

    # Data paths.
    parser.add_argument(
        "--caption_json",
        type=str,
        required=True,
        help="Path to a JSON file mapping image names to captions.",
    )
    parser.add_argument(
        "--image_root",
        type=str,
        required=True,
        help="Directory containing the original images.",
    )
    parser.add_argument(
        "--output_jsonl",
        type=str,
        default="outputs/synthetic_samples.jsonl",
        help="Path to save generated metadata in JSONL format.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/generated_images",
        help="Directory to save edited images.",
    )

    # Model paths.
    parser.add_argument(
        "--qwen_model_path",
        type=str,
        required=True,
        help="Path or HuggingFace name of the instruction-tuned Qwen2.5-VL model.",
    )
    parser.add_argument(
        "--flux_model_name",
        type=str,
        default="black-forest-labs/FLUX.1-Kontext-dev",
        help="Path or HuggingFace name of the FLUX-Kontext image editing model.",
    )

    # Generation settings.
    parser.add_argument("--max_images", type=int, default=30000)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--change_ratio", type=float, default=0.8)
    parser.add_argument("--guidance_scale", type=float, default=2.5)
    parser.add_argument("--num_inference_steps", type=int, default=28)
    parser.add_argument("--seed", type=int, default=42)

    # Filtering thresholds.
    parser.add_argument("--change_ssim_max", type=float, default=0.92)
    parser.add_argument("--change_lpips_min", type=float, default=0.08)
    parser.add_argument("--change_clipscore_min", type=float, default=20.0)
    parser.add_argument("--nochange_ssim_min", type=float, default=0.94)
    parser.add_argument("--nochange_lpips_max", type=float, default=0.10)

    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_models(args):
    global qwen_model, qwen_processor, pipe
    global loss_fn, clip_model, clip_preprocess, clip_device

    qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.qwen_model_path,
        torch_dtype="auto",
        device_map="auto",
    )
    qwen_processor = AutoProcessor.from_pretrained(args.qwen_model_path)

    pipe = FluxKontextPipeline.from_pretrained(
        args.flux_model_name,
        torch_dtype=torch.bfloat16,
    ).to("cuda")

    loss_fn = lpips.LPIPS(net="alex").cuda()

    clip_device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=clip_device)
    clip_model.eval()


def calculate_similarity(orig_img, gen_img):
    orig_arr = np.array(orig_img)
    gen_arr = np.array(gen_img)

    gray_orig = cv2.cvtColor(orig_arr, cv2.COLOR_RGB2GRAY)
    gray_gen = cv2.cvtColor(gen_arr, cv2.COLOR_RGB2GRAY)
    ssim_score = ssim(gray_orig, gray_gen, data_range=255)

    def pil_to_tensor(pil_img):
        tensor = torch.tensor(np.array(pil_img)).float() / 255.0
        tensor = tensor.permute(2, 0, 1).unsqueeze(0) * 2 - 1
        return tensor.cuda()

    with model_lock:
        with torch.no_grad():
            lpips_score = loss_fn(pil_to_tensor(orig_img), pil_to_tensor(gen_img)).item()

    return ssim_score, lpips_score


def calculate_clip_score(image, text):
    image_input = clip_preprocess(image).unsqueeze(0).to(clip_device)
    text_input = clip.tokenize([text]).to(clip_device)

    with model_lock:
        with torch.no_grad():
            image_features = clip_model.encode_image(image_input)
            text_features = clip_model.encode_text(text_input)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            similarity = (image_features * text_features).sum().item()

    return 100 * max(similarity, 0)


def generate_instruction(image_path, caption, change_ratio):
    is_change = random.random() < change_ratio
    prefix = "change -> " if is_change else "no change -> "
    caption_with_prefix = prefix + caption

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": caption_with_prefix},
            ],
        }
    ]

    text = qwen_processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, video_inputs = process_vision_info(messages)

    try:
        with model_lock:
            inputs = qwen_processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(qwen_model.device)

            with torch.no_grad():
                output = qwen_model.generate(**inputs, max_new_tokens=128)
                trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, output)]
                text_out = qwen_processor.batch_decode(trimmed, skip_special_tokens=True)

        if text_out and "Edit Instruction:" in text_out[0]:
            lines = text_out[0].split("\n")
            if len(lines) >= 2:
                edit_inst = lines[0].replace("Edit Instruction:", "").strip()
                change_cap = lines[1].replace("Change Caption:", "").strip()
                return is_change, edit_inst, change_cap
    except Exception as exc:
        print(f"Instruction generation failed for {image_path}: {exc}")

    return None, None, None


def process_image_caption(
    image_name,
    caption,
    args,
    global_stats,
    image_edit_counter,
    progress_bar,
):
    image_path = Path(args.image_root) / image_name
    if not image_path.exists():
        return None

    with data_lock:
        if args.max_images is not None and global_stats["saved_samples"] >= args.max_images:
            return None

    image_id = image_path.stem

    try:
        input_image = Image.open(image_path).convert("RGB")
        width, height = input_image.size

        is_change, edit_inst, change_cap = generate_instruction(
            str(image_path),
            caption,
            args.change_ratio,
        )
        if not edit_inst or not change_cap:
            return None

        with model_lock:
            result_img = pipe(
                image=input_image,
                prompt=edit_inst,
                width=width,
                height=height,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
            ).images[0]
        result_img = result_img.resize((width, height), Image.LANCZOS)

        ssim_score, lpips_score = calculate_similarity(input_image, result_img)

        if is_change:
            gen_clip_score = calculate_clip_score(result_img, edit_inst)
            similarity_valid = (
                ssim_score < args.change_ssim_max
                and lpips_score > args.change_lpips_min
            )
            clip_valid = gen_clip_score > args.change_clipscore_min
        else:
            gen_clip_score = calculate_clip_score(result_img, caption)
            similarity_valid = (
                ssim_score > args.nochange_ssim_min
                and lpips_score < args.nochange_lpips_max
            )
            clip_valid = True

        valid = similarity_valid and clip_valid

        with data_lock:
            if is_change:
                key = "valid_changes" if valid else "invalid_changes"
            else:
                key = "valid_no_changes" if valid else "invalid_no_changes"
            global_stats[key] += 1

            if not valid:
                return None

            if args.max_images is not None and global_stats["saved_samples"] >= args.max_images:
                return None

            edit_idx = image_edit_counter.get(image_id, 0)
            out_file = Path(args.output_dir) / f"{image_id}_edit_{edit_idx}.jpg"
            image_edit_counter[image_id] = edit_idx + 1
            global_stats["saved_samples"] += 1

        result_img.save(out_file)

        result_data = {
            "image": str(image_path),
            "edited_image": str(out_file),
            "input_caption": caption,
            "edit_instruction": edit_inst,
            "change_caption": change_cap,
            "is_change": is_change,
            "ssim": ssim_score,
            "lpips": lpips_score,
            "gen_clip_score": gen_clip_score,
            "similarity_valid": similarity_valid,
            "clip_valid": clip_valid,
            "overall_valid": valid,
        }

        progress_bar.update(1)
        return result_data

    except Exception as exc:
        print(f"Error on {image_name}: {exc}")
        return None


def generate_all(args):
    with open(args.caption_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    tasks = []
    for image_name, captions in data.items():
        num_captions = min(random.randint(1, 2), len(captions))
        selected_captions = random.sample(captions, num_captions)
        for caption in selected_captions:
            tasks.append((image_name, caption))

    random.shuffle(tasks)

    global_stats = {
        "valid_changes": 0,
        "invalid_changes": 0,
        "valid_no_changes": 0,
        "invalid_no_changes": 0,
        "saved_samples": 0,
    }
    image_edit_counter = {}
    results = []

    total_tasks = min(len(tasks), args.max_images) if args.max_images else len(tasks)
    progress_bar = tqdm(total=total_tasks, desc="Generating valid samples")

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [
            executor.submit(
                process_image_caption,
                image_name,
                caption,
                args,
                global_stats,
                image_edit_counter,
                progress_bar,
            )
            for image_name, caption in tasks
        ]

        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result:
                with data_lock:
                    results.append(result)
                    if args.max_images is not None and len(results) >= args.max_images:
                        break

    progress_bar.close()

    output_jsonl = Path(args.output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    total_valid = global_stats["valid_changes"] + global_stats["valid_no_changes"]
    total_invalid = global_stats["invalid_changes"] + global_stats["invalid_no_changes"]
    total = total_valid + total_invalid
    avg_clip_score = (
        sum(r["gen_clip_score"] for r in results) / len(results) if results else 0
    )

    print("\nFinal Statistics:")
    print(f"Saved samples: {len(results)}")
    if total > 0:
        print(f"Valid samples: {total_valid} ({total_valid / total * 100:.1f}%)")
        print(f"Invalid samples: {total_invalid} ({total_invalid / total * 100:.1f}%)")
        print(f"Valid changes: {global_stats['valid_changes']}")
        print(f"Invalid changes: {global_stats['invalid_changes']}")
        print(f"Valid no-changes: {global_stats['valid_no_changes']}")
        print(f"Invalid no-changes: {global_stats['invalid_no_changes']}")
    print(f"Average CLIPScore: {avg_clip_score:.2f}")
    print(f"Edited images are saved to: {args.output_dir}")
    print(f"Metadata is saved to: {args.output_jsonl}")


def main():
    args = parse_args()
    set_seed(args.seed)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.output_jsonl).parent.mkdir(parents=True, exist_ok=True)

    load_models(args)
    generate_all(args)


if __name__ == "__main__":
    main()
