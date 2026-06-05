# FLEXRec

This is the official repo of our proposed framework **FLEXRec: Fusion of Layer-wise Exits for Sequential Recommendation**.

It contains the following workflow:

1. preprocess raw data into sequential format
2. pretrain `LLM4Rec`
3. pretrain `LLM4RecWithMultiPredHead`
4. train `FLEXRec`
5. run eval-only scripts for each stage if needed

## Repo layout

- `datasets/code`: preprocessing scripts
- `datasets/sequential`: processed sequential datasets
- `model.py`: `LLM4Rec`, `LLM4RecWithMultiPredHead`, checkpoint helpers
- `MOE_model.py`: `FLEXRec`, `ACRouter`
- `finetune.py`: stage-1 training
- `train_intermediate_heads.py`: stage-2 training
- `train_FLEXRec.py`: stage-3 training
- `LLM4Rec_eval_from_config.py`: eval-only for stage 1
- `LLM4RecWithMultiPredHead_eval_from_config.py`: eval-only for stage 2
- `FLEXRec_eval_from_config.py`: eval-only for stage 3

## Setup

```bash
# Assuming Anaconda is installed
conda create -n flexrec python=3.9 -y
conda activate flexrec

# Install PyTorch first. Choose the right command for your CPU/GPU setup:
# https://pytorch.org/get-started/locally/
pip install torch==2.8.0 torchvision==0.23.0

# Then install the remaining project dependencies
pip install -r requirements.txt
```

Note: this codebase currently supports single-GPU training only. Set the GPU to use via the `CUDA_VISIBLE_DEVICES` environment variable, for example:

```bash
export CUDA_VISIBLE_DEVICES=0
```

## Workflow


### 1. Preprocess raw data (optional — preprocessed datasets included)

Preprocessed sequential datasets are included in `datasets/sequential`. You can skip this step if you want to use the provided datasets (for example: `datasets/sequential/Toys_and_Games/Toys_and_Games.txt`).

If you prefer to regenerate datasets, the preprocessing scripts in `datasets/code` use relative paths, so run them from that directory.

Example for Amazon `Toys_and_Games`:

```bash
cd datasets/code
python -c "import data_process; data_process.root_path='../'; data_process.main('Toys_and_Games', data_type='Amazon')"
python -c "from generate_test import sample_test_data; sample_test_data('Toys_and_Games')"
cd ../..
```

This should generate at least:

- `datasets/sequential/Toys_and_Games/Toys_and_Games.txt`
- `datasets/sequential/Toys_and_Games/Toys_and_Games_sample.txt`
- `datasets/sequential/Toys_and_Games/Toys_and_Games_item2attributes.json`

Notes:

- `data_process.py` expects raw files under `raw_datasets/`
- `generate_test.py` creates the negative candidate file used by candidate-item evaluation

### 2. Stage 1: pretrain LLM4Rec

Example command:

```bash
export HF_TOKEN=your_huggingface_token
export CUDA_VISIBLE_DEVICES=0
python finetune.py \
  --base_model Qwen/Qwen3-0.6B \
  --data_path ./datasets/sequential/Toys_and_Games \
  --cache_dir your/dir/to/.cache \
  --output_dir ./output/ \
  --task_type sequential \
  --batch_size 128 \
  --micro_batch_size 32 \
  --num_epochs 2 \
  --learning_rate 0.0002 
```

(Tip: set `CUDA_VISIBLE_DEVICES` before running to select the single GPU, e.g. `export CUDA_VISIBLE_DEVICES=0`.)

This writes a stage-1 folder like:

- `output/Toys_and_Games/Qwen3-0.6B-sequential-2-epochs/`

Important files:

- `finetune_params.json`
- `finetune_model.pt`

### 3. Stage 2: pretrain LLM4RecWithMultiPredHead

`exit_layer_intervals` is fixed to `1` in this repo.

Example command:

```bash
export HF_TOKEN=your_huggingface_token
export CUDA_VISIBLE_DEVICES=0
python train_intermediate_heads.py \
  --load_model_path ./output/Toys_and_Games/Qwen3-0.6B-sequential-2-epochs \
  --batch_size 128 \
  --micro_batch_size 32 \
  --num_epochs 2 \
  --learning_rate 0.0003 \
  --group_by_length=False \
  --eval_over_candidate_items=False \
  --save_all_user_eval_res=False \
  --prompt_template_name alpaca
```

(Tip: set `CUDA_VISIBLE_DEVICES` before running to select the single GPU, e.g. `export CUDA_VISIBLE_DEVICES=0`.)

This writes:

- `output/Toys_and_Games/Qwen3-0.6B-sequential-2-epochs/LLM4RecWithMultiPredHead_exit_intervals_1/`

Important files:

- `train_intermediate_heads_params.json`
- `trained_intermediate_heads_model.pt`

### 4. Stage 3: train FLEXRec

This stage only loads the stage-2 checkpoint produced with `exit_layer_intervals=1`.

Example command:

```bash
export HF_TOKEN=your_huggingface_token
export CUDA_VISIBLE_DEVICES=0
python train_FLEXRec.py \
  --load_model_path ./output/Toys_and_Games/Qwen3-0.6B-sequential-2-epochs/LLM4RecWithMultiPredHead_exit_intervals_1 \
  --batch_size 32 \
  --num_epochs 2 \
  --learning_rate 0.0003 \
  --val_set_size -1 \
  --group_by_length=False \
  --eval_over_candidate_items=False \
  --save_all_user_eval_res=True \
  --eval_steps 500 \
  --warmup_steps 100 \
  --target_k 3 \
  --tau 10.0 \
  --_lambda 1 \
  --alpha 0.05 \
  --beta 0.001 \
  --gamma 2 \
  --early_stop_patience 10
```

(Tip: set `CUDA_VISIBLE_DEVICES` before running to select the single GPU, e.g. `export CUDA_VISIBLE_DEVICES=0`.)

This writes a stage-3 folder under:

- `.../LLM4RecWithMultiPredHead_exit_intervals_1/FLEXRec/`

Important files:

- `train_FLEXRec_params.json`
- `trained_FLEXRec.pt`
- test result files

## Eval-only scripts

### Evaluate stage 1 checkpoint

```bash
export HF_TOKEN=your_huggingface_token
export CUDA_VISIBLE_DEVICES=0
python LLM4Rec_eval_from_config.py \
  --load_model_path ./output/Toys_and_Games/Qwen3-0.6B-sequential-2-epochs \
  --eval_over_candidate_items=False \
  --save_eval_res=True
```

(Tip: for eval runs that use the GPU, set `CUDA_VISIBLE_DEVICES` before running, e.g. `export CUDA_VISIBLE_DEVICES=0`.)

### Evaluate stage 2 checkpoint

```bash
export HF_TOKEN=your_huggingface_token
export CUDA_VISIBLE_DEVICES=0
python LLM4RecWithMultiPredHead_eval_from_config.py \
  --load_model_path ./output/Toys_and_Games/Qwen3-0.6B-sequential-2-epochs/LLM4RecWithMultiPredHead_exit_intervals_1 \
  --eval_over_candidate_items=True \
  --save_eval_res=True
```

(Tip: for eval runs that use the GPU, set `CUDA_VISIBLE_DEVICES` before running, e.g. `export CUDA_VISIBLE_DEVICES=0`.)

### Evaluate stage 3 checkpoint

```bash
export HF_TOKEN=your_huggingface_token
export CUDA_VISIBLE_DEVICES=0
python FLEXRec_eval_from_config.py \
  --load_model_path ./output/Toys_and_Games/Qwen3-0.6B-sequential-2-epochs/LLM4RecWithMultiPredHead_exit_intervals_1/FLEXRec/<stage3-folder> \
  --eval_over_candidate_items=False \
  --save_eval_res=True
```

(Tip: for eval runs that use the GPU, set `CUDA_VISIBLE_DEVICES` before running, e.g. `export CUDA_VISIBLE_DEVICES=0`.)



## Checkpoints

Stage 1 and stage 2 checkpoints are saved in compact incremental form rather than full base-model form.

- stage 1 saves LoRA plus recommender trainable components
- stage 2 saves stage-1 incremental weights plus intermediate heads
- stage 3 saves only the FLEXRec router parameters



## Baseline Implementation & Hyperparameters

For all traditional sequential recommendation methods and standard LLM-based methods, we utilize their default hyperparameter settings as reported in their respective original implementations. 

To ensure a fair comparison for the routing architectures, we implement the MoE baselines on top of our multi-layer E4SRec architecture. The specific hyperparameters tuned for these routing baselines are detailed below:

### 1. SparseToken
* **Target Experts ($k$):** `3`
* **Load Balancing Loss Weight:** Tuned from `{0.01, 0.05, 0.1, 0.25, 0.5}`

### 2. DynamicMoE
* **Confidence Threshold ($p$):** `0.4` (set to avoid over-concentration on a single prediction layer)
* **Load Balancing Loss Weight:** Tuned from `{0.01, 0.05, 0.1, 0.25, 0.5}`
* **Dynamic Loss Factor:** Tuned from `{1e-5, 1e-4, 1e-3, 1e-2, 1e-1}`

### 3. LD-MoLE
* **Target Experts ($k$):** `3`
* **Load Balancing Loss Factor:** Tuned from `{0.005, 0.01, 0.05, 0.1, 0.15}`
* **Sparsity Loss Factor:** Tuned from `{0.01, 0.05, 0.25, 0.5, 1.0}`