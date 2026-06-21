import os
import sys
import argparse
import logging
import warnings
import json
import random
import subprocess
import re

import numpy as np
import torch
from torch.utils.data import dataloader
from torch.utils.data import ConcatDataset
from tqdm import tqdm

import utils
import datasets
import model as diff_model

from lavis.models import load_preprocess
from omegaconf import OmegaConf
from torch.cuda.amp import autocast, GradScaler
from lavis.common.optims import LinearWarmupCosineLRScheduler


warnings.filterwarnings("ignore")
torch.set_num_threads(8)


def parse_args():
    parser = argparse.ArgumentParser(description="Train/evaluate UEM-MLLM without RAG and without LEVIR.")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=["train", "test"], default="train")
    parser.add_argument(
        "--dataset",
        default="clevr",
        help="Evaluation dataset: clevr, spot, IER, or bird. Alias 'clver' is also accepted.",
    )

    # Dataset paths. No absolute paths are hard-coded.
    parser.add_argument("--clevr_path", default="", help="Path to CLEVR-Change data directory.")
    parser.add_argument("--spot_path", default="", help="Path to Spot-the-Diff root directory.")
    parser.add_argument("--ier_path", default="", help="Path to Image-Editing-Request root directory.")
    parser.add_argument("--bird_image_dir", default="", help="Path to Birds-to-Words image directory.")
    parser.add_argument("--bird_data_root", default="", help="Path to Birds-to-Words annotation root.")

    # Synthetic JSONL dataset.
    parser.add_argument(
        "--synthetic_jsonl",
        default="",
        help="Path to generated synthetic JSONL file, e.g., data/synthetic/sampler_train_filtered.jsonl.",
    )
    parser.add_argument(
        "--use_synthetic",
        action="store_true",
        help="Use synthetic JsonlEditDataset during training.",
    )
    parser.add_argument(
        "--train_mixed",
        action="store_true",
        help="Train on mixed datasets: CLEVR, Bird, Spot, IER, and optional synthetic data.",
    )

    # Inference-only crop.
    parser.add_argument(
        "--use_crop",
        action="store_true",
        help="Enable Difference-Aware Visual Cropping only during evaluation/testing.",
    )
    parser.add_argument("--crop_num_visual_tokens", type=int, default=16)
    parser.add_argument("--crop_bbox_base_size", type=int, default=224)

    # Optional bird official evaluation.
    parser.add_argument("--bird_gt_file", default="", help="Ground-truth file for Birds-to-Words official evaluation.")
    parser.add_argument("--bird_eval_root", default="", help="Directory containing eval_models.py for bird evaluation.")
    parser.add_argument("--bird_python", default="", help="Python binary for bird official evaluation. Default: current Python.")

    # Model and output.
    parser.add_argument("--model_type", type=str, default="vicuna7b")
    parser.add_argument("--model_pth", default="", help="Checkpoint path for testing.")
    parser.add_argument("--model_dir", default="exp/finerall", help="Directory to save logs, predictions, and checkpoints.")
    parser.add_argument("--gt_dir", default="./eval_data", help="Directory containing ground-truth caption JSON files.")

    # Training hyperparameters.
    parser.add_argument("--vit_lora_k", type=int, default=16)
    parser.add_argument("--qformer_lora_k", type=int, default=4)
    parser.add_argument("--warmup_steps", type=int, default=4000)
    parser.add_argument("--max_length", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=120)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--peak_lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--eval_frequency", type=int, default=1)
    parser.add_argument("--early_stop_num", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=8)

    # Kept for old commands, but not used.
    parser.add_argument("--consist_w", type=float, default=None, help="Deprecated and ignored.")
    parser.add_argument("--ortho_w", type=float, default=None, help="Deprecated and ignored.")

    parser.add_argument(
        "--prompt",
        type=str,
        default="the difference between the before image and after image is that",
    )
    parser.add_argument(
        "--tmp_dir",
        default="",
        help="Optional TMPDIR. If empty, TMPDIR is not modified.",
    )

    args = parser.parse_args()

    # Fix common typo in old scripts.
    if args.dataset == "clver":
        args.dataset = "clevr"

    valid_datasets = {"clevr", "spot", "IER", "bird"}
    if args.dataset not in valid_datasets:
        raise ValueError(f"Unsupported dataset: {args.dataset}. Please use one of {sorted(valid_datasets)}.")

    return args


args = parse_args()

if args.tmp_dir:
    os.environ["TMPDIR"] = args.tmp_dir


def require_path(path, name):
    if not path:
        raise ValueError(f"--{name} is required.")
    return path


def get_transform():
    cfg = OmegaConf.load("configs/blip2_instruct_vicuna7b.yaml")
    img_preprocess, _ = load_preprocess(cfg.preprocess)
    return img_preprocess["eval"]


def parse_bird_stdout(text):
    metrics = {}

    patterns = {
        "Bleu_1": r"Bleu_1:\s*([0-9.]+)",
        "Bleu_2": r"Bleu_2:\s*([0-9.]+)",
        "Bleu_3": r"Bleu_3:\s*([0-9.]+)",
        "Bleu_4": r"Bleu_4:\s*([0-9.]+)",
        "METEOR": r"METEOR:\s*([0-9.]+)",
        "ROUGE_L": r"ROUGE_L:\s*([0-9.]+)",
        "CIDEr": r"CIDEr:\s*([0-9.]+)",
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            metrics[key] = float(match.group(1))

    if "CIDEr" not in metrics:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        for i, line in enumerate(lines):
            header = re.split(r"\s+", line)
            if "Task" in header and "Model" in header and "CIDEr" in header:
                if i + 1 < len(lines):
                    values = re.split(r"\s+", lines[i + 1].strip())
                    if len(values) == len(header):
                        row = dict(zip(header, values))
                        for key in ["Bleu_4", "METEOR", "ROUGE_L", "CIDEr"]:
                            if key in row:
                                metrics[key] = float(row[key])
                break

    return metrics


def run_bird_eval(pred_jsonl_path):
    if not args.bird_gt_file or not args.bird_eval_root:
        raise ValueError("Bird official evaluation requires --bird_gt_file and --bird_eval_root.")

    python_bin = args.bird_python if args.bird_python else sys.executable

    cmd = [
        python_bin,
        "eval_models.py",
        "--dataset",
        "bird",
        "--testfile",
        os.path.abspath(pred_jsonl_path),
        "--gtfile",
        args.bird_gt_file,
    ]

    result = subprocess.run(
        cmd,
        cwd=args.bird_eval_root,
        capture_output=True,
        text=True,
        check=True,
    )

    logging.info("[bird eval stdout]\n%s", result.stdout)

    metrics = parse_bird_stdout(result.stdout)
    if "CIDEr" not in metrics:
        raise RuntimeError(
            "Failed to parse CIDEr from bird official evaluation output.\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    return metrics, result.stdout


def build_dataset(data_name, split, transform):
    if data_name == "clevr":
        return datasets.CLEVR_Dataset(
            data_path=require_path(args.clevr_path, "clevr_path"),
            transform=transform,
            split=split,
            prompt=args.prompt,
        )

    if data_name == "spot":
        spot_root = require_path(args.spot_path, "spot_path")
        return datasets.SpotDataset(
            image_path=os.path.join(spot_root, "resized_images/resized_images"),
            anno_path=os.path.join(spot_root, "captions"),
            transform=transform,
            split=split,
            prompt=args.prompt,
        )

    if data_name == "IER":
        return datasets.Image_Edit_Request(
            data_path=require_path(args.ier_path, "ier_path"),
            transform=transform,
            split=split,
            prompt=args.prompt,
        )

    if data_name == "bird":
        return datasets.BirdDataset(
            image_dir=require_path(args.bird_image_dir, "bird_image_dir"),
            data_root=require_path(args.bird_data_root, "bird_data_root"),
            transform=transform,
            split=split,
            prompt=args.prompt,
        )

    raise ValueError(f"Unsupported dataset: {data_name}")


def build_synthetic_dataset(split, transform):
    if not args.synthetic_jsonl:
        raise ValueError("--use_synthetic is enabled, but --synthetic_jsonl is empty.")

    return datasets.JsonlEditDataset(
        jsonl_path=args.synthetic_jsonl,
        transform=transform,
        split=split,
        prompt=args.prompt,
    )


def get_dataset(data_name, split):
    transform = get_transform()
    return build_dataset(data_name, split, transform)


def get_dataset_train(data_name, split):
    transform = get_transform()

    if args.train_mixed:
        train_sets = [
            build_dataset("clevr", split, transform),
            build_dataset("bird", split, transform),
            build_dataset("spot", split, transform),
            build_dataset("IER", split, transform),
        ]

        if args.use_synthetic:
            train_sets.append(build_synthetic_dataset(split, transform))

        return ConcatDataset(train_sets)

    train_sets = [build_dataset(data_name, split, transform)]

    if args.use_synthetic:
        train_sets.append(build_synthetic_dataset(split, transform))

    if len(train_sets) == 1:
        return train_sets[0]

    return ConcatDataset(train_sets)


def create_model_and_optimizer(need_optim=True):
    model = diff_model.FINER_MLLM.load_pretrained_model_from_blip2(
        model_type=args.model_type,
        vit_lora_k=args.vit_lora_k,
        qformer_lora_k=args.qformer_lora_k,
    )
    model.cuda()

    if need_optim:
        optimizer = torch.optim.AdamW(
            filter(lambda x: x.requires_grad, model.parameters()),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        return model, optimizer

    return model


def train_one_epoch(model, optimizer, train_loader, scaler, cur_epoch, cur_step, scheduler):
    model.train()
    loss_avg = utils.RunningAverage()

    with tqdm(total=len(train_loader), mininterval=60, disable=False) as t:
        for data in train_loader:
            if cur_step < args.warmup_steps:
                scheduler.step(0, cur_step)
            else:
                skip_epoch = int(args.warmup_steps // len(train_loader))
                scheduler.step(cur_epoch - skip_epoch, cur_step)
            cur_step += 1

            bef_imgs = data["bef_img"].cuda()
            aft_imgs = data["aft_img"].cuda()
            captions = data["caption"]

            optimizer.zero_grad()

            with autocast():
                # Training does NOT use Difference-Aware Visual Cropping.
                loss = model(bef_imgs, aft_imgs, captions)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_avg.update(loss.item())
            t.set_postfix(loss="{:05.3f}".format(loss_avg()))
            t.update()

    return loss_avg(), cur_step


def train_and_evaluate(model, optimizer, trainset, valset):
    train_loader = dataloader.DataLoader(
        trainset,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    current_best_score = float("-inf")
    scaler = GradScaler()
    early_stop = 0
    best_epoch = -1
    best_epoch_score = None
    best_model_path = ""
    cur_step = 0

    scheduler = LinearWarmupCosineLRScheduler(
        optimizer,
        max_epoch=args.num_epochs,
        min_lr=args.min_lr,
        init_lr=args.peak_lr,
        warmup_steps=args.warmup_steps,
        warmup_start_lr=args.lr,
    )

    for epoch in range(args.num_epochs):
        early_stop += 1
        logging.info("Epoch %d/%d", epoch + 1, args.num_epochs)

        epoch_loss, cur_step = train_one_epoch(
            model,
            optimizer,
            train_loader,
            scaler,
            epoch,
            cur_step,
            scheduler,
        )
        logging.info("loss=%05.3f", epoch_loss)

        if (epoch + 1) % args.eval_frequency == 0:
            # If --use_crop is set, validation also uses crop, because validation is inference.
            score = eval_on_single_gpu(model, valset)
            logging.info("Epoch %d", epoch + 1)
            logging.info(score)

            if "CIDEr" not in score:
                raise KeyError(f"CIDEr not found in score: {score}")

            if current_best_score < float(score["CIDEr"]):
                current_best_score = float(score["CIDEr"])
                early_stop = 0
                best_model_path = save_model(model)
                best_epoch = epoch + 1
                best_epoch_score = score

        if early_stop == args.early_stop_num:
            logging.info("early stop at epoch %d.", epoch + 1)
            break

    logging.info("Best Epoch is %s", best_epoch)
    logging.info(best_epoch_score)
    logging.info("model checkpoint saved at %s", best_model_path)


def eval_on_single_gpu(model, valset):
    model.eval()

    loader = dataloader.DataLoader(
        valset,
        batch_size=args.batch_size,
        drop_last=False,
        num_workers=args.num_workers,
    )

    generate_results = []
    bird_jsonl_results = []

    for data in tqdm(loader, mininterval=60, disable=False):
        bef_imgs = data["bef_img"].cuda()
        aft_imgs = data["aft_img"].cuda()
        img_ids = data["img_id"]

        if args.use_crop:
            captions = model.generate_with_crop(
                bef_imgs,
                aft_imgs,
                max_length=args.max_length,
                num_visual_tokens=args.crop_num_visual_tokens,
                bbox_base_size=args.crop_bbox_base_size,
            )
        else:
            with torch.no_grad():
                captions = model.generate(
                    bef_imgs,
                    aft_imgs,
                    max_length=args.max_length,
                )

        for img_id, caption in zip(img_ids, captions):
            generate_results.append(
                {
                    "image_id": img_id,
                    "caption": caption,
                }
            )

            if args.dataset == "bird":
                fixed_id = utils.format_bird_img_id(img_id)
                bird_jsonl_results.append(
                    {
                        "candidates": [caption],
                        "ImgId": fixed_id,
                        "description": "",
                    }
                )

    output_json_path = os.path.join(args.model_dir, "test_total_captions.json")
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(generate_results, f, ensure_ascii=False, indent=4)

    if args.dataset == "bird" and args.bird_gt_file and args.bird_eval_root:
        bird_jsonl_path = os.path.join(args.model_dir, "bird_eval_epoch.jsonl")

        with open(bird_jsonl_path, "w", encoding="utf-8") as f:
            for item in bird_jsonl_results:
                json.dump(item, f, ensure_ascii=False)
                f.write("\n")

        generation_metrics, raw_stdout = run_bird_eval(bird_jsonl_path)

        with open(os.path.join(args.model_dir, "bird_metric_stdout.txt"), "w", encoding="utf-8") as f:
            f.write(raw_stdout)

        with open(os.path.join(args.model_dir, "bird_metric.json"), "w", encoding="utf-8") as f:
            json.dump(generation_metrics, f, ensure_ascii=False, indent=4)
    else:
        gt_file = os.path.join(
            args.gt_dir,
            f"{args.dataset}_test_change_captions_reformat.json",
        )
        with utils.HiddenPrints():
            generation_metrics = utils.generation_score(gt_file, output_json_path)

    return generation_metrics


def test(model, testset):
    generation_metrics = eval_on_single_gpu(model, testset)
    logging.info(generation_metrics)
    return generation_metrics


def save_model(model):
    save_file_path = os.path.join(args.model_dir, f"{args.dataset}_model_params.pth")
    torch.save(model.state_dict(), save_file_path)
    return save_file_path


def setup_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False


if __name__ == "__main__":
    os.makedirs(args.model_dir, exist_ok=True)

    setup_seed(args.seed)
    utils.set_logger(os.path.join(args.model_dir, "train.log"))

    logging.info("save arguments...")
    for key, value in vars(args).items():
        logging.info("\t'%s'=%s", key, value)

    logging.info("Loading datasets and model...")

    if args.mode == "train":
        train_set = get_dataset_train(args.dataset, split="train")
        val_set = get_dataset(args.dataset, split="test")

        model, optimizer = create_model_and_optimizer()
        train_and_evaluate(model, optimizer, train_set, val_set)

    elif args.mode == "test":
        if not args.model_pth:
            raise ValueError("--model_pth is required in test mode.")

        test_set = get_dataset(args.dataset, split="test")
        model = create_model_and_optimizer(need_optim=False)

        state_dict = torch.load(args.model_pth, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)

        logging.info("eval on test set...")
        test(model, test_set)
