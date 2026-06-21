# UEM-MLLM

Official implementation of **Unified Multimodal Learning with Synthetic Supervision and Gradient Perception for Image Change Captioning**.

This repository provides code for synthetic ICC data generation, UEM-MLLM training/evaluation, and inference with Difference-Aware Visual Cropping.

## Environment

The codebase is mainly built with the following libraries:

* Python 3.9 / 3.10
* PyTorch and torchvision
* Hugging Face Transformers
* Salesforce LAVIS
* LoRALib
* Diffusers
* OpenCV, Pillow, NumPy, and scikit-image
* pycocoevalcap and pycocotools

## Preparation

### Models

Please prepare the following models:

* **Qwen2.5-VL**: used to generate edit instructions and change captions.
* **FLUX.1-Kontext-dev**: used to synthesize edited images.
* **InstructBLIP/Vicuna**: used as the backbone of UEM-MLLM.

Example directory:

```text
checkpoints/
├── qwen2_5_vl/
├── flux_kontext/
└── uem_mllm/
```

### Data

For synthetic data generation, prepare image-caption data such as MS-COCO 2014:

```text
data/
└── coco/
    ├── train2014/
    └── image_caption_map.json
```

For training and evaluation, prepare ICC datasets:

```text
data/
├── clevr_change/
├── spot/
├── image_editing_request/
├── bird/
└── synthetic/
```

Ground-truth files for evaluation can be placed under:

```text
eval_data/
```

## Synthetic Data Generation

Run:

```bash
python tools/generate_synthetic_data.py \
  --caption_json data/coco/image_caption_map.json \
  --image_root data/coco/train2014 \
  --qwen_model_path checkpoints/qwen2_5_vl \
  --output_dir outputs/generated_images \
  --output_jsonl data/synthetic/sampler_train_filtered.jsonl \
  --max_images 30000 \
  --num_workers 4
```

The generated JSONL file contains synthetic ICC samples with original images, edited images, edit instructions, and change captions.

## Training

Training does **not** use Difference-Aware Visual Cropping.

```bash
model_dir=exp/finerall

python train.py \
  --batch_size 32 \
  --num_epochs 120 \
  --lr 1e-5 \
  --min_lr 1e-5 \
  --peak_lr 5e-5 \
  --warmup_steps 4000 \
  --vit_lora_k 16 \
  --qformer_lora_k 4 \
  --weight_decay 0.1 \
  --max_length 60 \
  --dataset clevr \
  --mode train \
  --train_mixed \
  --use_synthetic \
  --clevr_path data/clevr_change/data \
  --spot_path data/spot \
  --ier_path data/image_editing_request \
  --bird_image_dir data/bird/bird-to-words \
  --bird_data_root data/bird \
  --synthetic_jsonl data/synthetic/sampler_train_filtered.jsonl \
  --gt_dir eval_data \
  --model_dir ${model_dir}
```

The best checkpoint is saved according to the CIDEr score on the selected validation dataset.

## Testing with Difference-Aware Visual Cropping

Difference-Aware Visual Cropping is enabled only during inference/testing by adding `--use_crop`.

```bash
python train.py \
  --mode test \
  --dataset clevr \
  --model_pth exp/finerall/clevr_model_params.pth \
  --clevr_path data/clevr_change/data \
  --gt_dir eval_data \
  --model_dir exp/finerall \
  --use_crop
```

To evaluate multiple datasets with cropping, run:

```bash
bash run_test_crop.sh
```

## Testing without Cropping

```bash
python train.py \
  --mode test \
  --dataset clevr \
  --model_pth exp/finerall/clevr_model_params.pth \
  --clevr_path data/clevr_change/data \
  --gt_dir eval_data \
  --model_dir exp/finerall
```

## Crop Visualization

The following script is only used to visualize the selected crop region and attention maps for a single image pair. It is not required for normal benchmark testing.

```bash
python diff_aware_visual_crop.py \
  --before before.jpg \
  --after after.jpg \
  --model_id /path/to/your/uem-mllm-checkpoint \
  --output_dir crop_out
```

The output directory contains cropped images, bounding-box visualizations, attention maps, and metadata.

## Acknowledgement

This repository is built upon and inspired by the following excellent works:

* [MLLMs Know Where to Look](https://github.com/saccharomycetes/mllms_know)
* [FINER-MLLM](https://github.com/xianzhangzx/FINER-MLLM)

We sincerely thank the authors for releasing their code and for their contributions to the community.


