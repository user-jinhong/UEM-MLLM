#!/bin/bash

# Test UEM-MLLM with Difference-Aware Visual Cropping.
# Training does not use crop. Testing uses crop by adding --use_crop.

model_dir=exp/finerall
model_pth=exp/finerall/clevr_model_params.pth
gt_dir=eval_data

# CLEVR-Change
python train.py \
  --batch_size 16 \
  --vit_lora_k 16 \
  --qformer_lora_k 4 \
  --max_length 60 \
  --dataset clevr \
  --mode test \
  --model_dir ${model_dir} \
  --clevr_path data/clevr_change/data \
  --gt_dir ${gt_dir} \
  --model_pth ${model_pth} \
  --use_crop


# Spot-the-Diff
python train.py \
  --batch_size 16 \
  --vit_lora_k 16 \
  --qformer_lora_k 4 \
  --max_length 60 \
  --dataset spot \
  --mode test \
  --model_dir ${model_dir} \
  --spot_path data/spot \
  --gt_dir ${gt_dir} \
  --model_pth ${model_pth} \
  --use_crop


# Image-Editing-Request
python train.py \
  --batch_size 16 \
  --vit_lora_k 16 \
  --qformer_lora_k 4 \
  --max_length 60 \
  --dataset IER \
  --mode test \
  --model_dir ${model_dir} \
  --ier_path data/image_editing_request \
  --gt_dir ${gt_dir} \
  --model_pth ${model_pth} \
  --use_crop


# Birds-to-Words
python train.py \
  --batch_size 16 \
  --vit_lora_k 16 \
  --qformer_lora_k 4 \
  --max_length 60 \
  --dataset bird \
  --mode test \
  --model_dir ${model_dir} \
  --bird_image_dir data/bird/bird-to-words \
  --bird_data_root data/bird \
  --gt_dir ${gt_dir} \
  --model_pth ${model_pth} \
  --use_crop