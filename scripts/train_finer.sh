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