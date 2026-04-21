# Progressive Modality Decoupling (PMD) for Dense Video Captioning

Code for the paper *"Progressive Modality Decoupling for Diagnosing and
Mitigating Visual Over-Reliance in Dense Video Captioning"* (anonymous
submission).

This repository contains the HiCM$^2$-based code used in our PMD
experiments, including the implementations of Visual Modality Attenuation
(VMA) and Semantic Transition Tokens (STT).

---

## Environment

Create the environment from the provided YAML file:

```bash
# Optional but recommended: remove the local prefix line from environment.yml
# before creating the environment.
conda env create -f environment.yml
conda activate pmd
python -m spacy download en_core_web_sm
```

If your local setup differs, please adjust the CUDA / PyTorch versions
accordingly.
g
---

## STT generation

The Phase 3 command requires a precomputed boundary-token JSON file
(`boundary_tokens.json`). This file is generated from the dataset
annotations using the provided builders, which are self-contained and
use spaCy (`en_core_web_sm`) for action / object extraction.

For YouCook2:

```bash
python build_boundary_tokens_yc2.py \
    --json_path   ./data/yc2/train.json \
    --output_path ./data/yc2/boundary_tokens.json
```

For ViTT:

```bash
python build_boundary_tokens_vitt.py \
    --json_path   ./data/vitt/train.json \
    --output_path ./data/vitt/boundary_tokens.json
```

The two builders use different action / object dictionaries, reflecting
the cooking-only nature of YouCook2 and the multi-domain coverage
(cooking, beauty, music, craft, structural tags) required for ViTT. Both
classify each adjacent-event transition into strong / mid-action /
mid-object / weak (paper Sec. III-C); only non-weak boundaries produce
output tokens.

---

## Training

PMD follows a three-stage training schedule.

### Pretrained language model

The codebase expects the T5 weights to be prepared first:

```bash
python down_t5.py
```
This step downloads the language-model weights used by the backbone. It
does not create a Phase-1 baseline checkpoint.

### Phase 1 — Baseline checkpoint

Phase 2 and Phase 3 both assume that a Phase-1 baseline checkpoint is
already available. Depending on your setup, this can be:

- the official released HiCM$^2$ checkpoint, or
- your own baseline checkpoint produced by the standard HiCM$^2$ training
  recipe.

The example commands below assume that the Phase-1 checkpoint is stored at:

```text
./presave/baseline/best_model.pth
```

### Phase 2 — Geometric grounding (SaliGT)

```bash
SAVE_DIR="phase2"
python -m torch.distributed.launch --nproc_per_node 4 --master_port=29111 --use_env dvc_ret.py \
  --use_saliency_reweight --saliency_alpha 10.0 \
  --bank_type yc2 --window_size 10 --sim_match anchor_cos --sampling origin \
  --save_dir=${SAVE_DIR} \
  --load ./presave/baseline/best_model.pth \
  --epochs=20 --lr=1e-6 \
  --combine_datasets youcook --combine_datasets_val youcook \
  --batch_size=2 --batch_size_val=2 --schedule=cosine_with_warmup \
  --ret_option hier_concat --hier_ret_num top-k --soft_k 10 \
  --LLM_ver 70 --hier_use level_4 level_3 level_2 level_1
```

This produces a Phase-2 checkpoint (for example):

```text
./presave/${SAVE_DIR}/best_model.pth
```

### Phase 3 — VMA + STT

```bash
SAVE_DIR="phase3"
python -m torch.distributed.launch --nproc_per_node 4 --master_port=29111 --use_env dvc_ret.py \
  --use_vma --vma_lambda 0.0 \
  --use_boundary_tokens --boundary_tokens_path ./data/yc2/boundary_tokens.json \
  --filter_weak_boundary \
  --bank_type yc2 --window_size 10 --sim_match anchor_cos --sampling origin \
  --save_dir=${SAVE_DIR} \
  --load ./presave/phase2/best_model.pth \
  --epochs=20 --lr=1e-6 \
  --combine_datasets youcook --combine_datasets_val youcook \
  --batch_size=2 --batch_size_val=2 --schedule=cosine_with_warmup \
  --ret_option hier_concat --hier_ret_num top-k --soft_k 10 \
  --LLM_ver 70 --hier_use level_4 level_3 level_2 level_1
```

This produces the final PMD checkpoint (for example):

```text
./presave/${SAVE_DIR}/best_model.pth
```

### Reproducing the lambda sweep (paper Tab. V)

To reproduce the lambda sweep ablation, use the same Phase 3 command above
but vary `--vma_lambda`:

| Setting | Command flag           | Reported CIDEr |
|---------|------------------------|----------------|
| λ=1.0   | `--vma_lambda 1.0`     | 72.63          |
| λ=0.5   | `--vma_lambda 0.5`     | 76.30          |
| λ=0.3   | `--vma_lambda 0.3`     | 77.58          |
| λ=0.0   | `--vma_lambda 0.0`     | 78.34          |


For ViTT, replace the dataset / bank settings and boundary-token path with
the corresponding ViTT configuration.

---

## Evaluation

To evaluate a trained checkpoint:

```bash
python -m torch.distributed.launch --nproc_per_node 4 --master_port=29111 --use_env dvc_ret.py \
  --eval --load ./presave/phase3/best_model.pth \
  --bank_type yc2 --window_size 10 --sim_match anchor_cos --sampling origin \
  --combine_datasets youcook --combine_datasets_val youcook \
  --batch_size_val=2 \
  --ret_option hier_concat --hier_ret_num top-k --soft_k 10 \
  --LLM_ver 70 --hier_use level_4 level_3 level_2 level_1 \
  --save_dir=eval_phase3
```

VMA and STT are training-only mechanisms and are not specified at
evaluation: VMA is gated by `self.training` in the model, and STT
boundary tokens require GT event annotations which are not available at
inference. Evaluation therefore uses the standard HiCM$^2$ inference
pipeline (CLIP visual features + ASR + retrieval).

---

## Notes

- VMA is controlled by `--use_vma` and `--vma_lambda` (paper default: 0.0).
- STT requires a precomputed boundary-token JSON file. See the
  *STT generation* section above for how to produce it from the dataset
  annotations.
- Phase 2 and Phase 3 in our experiments both use 20 epochs and a learning
  rate of `1e-6`.

---

## Acknowledgment

This repository is substantially based on the HiCM2-DVC codebase:
https://github.com/ailab-kyunghee/HiCM2-DVC

We thank the original authors for making their code publicly available.
Our implementation includes modifications and extensions for Progressive Modality Decoupling (PMD), Visual Modality Attenuation (VMA), and Semantic Transition Tokens (STT).

This project retains the original MIT License notice from the upstream repository.