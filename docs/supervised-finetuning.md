# Supervised Fine-Tuning (SFT)

This document explains the Supervised Fine-Tuning pipeline in
[src/finetune/sft/](../src/finetune/sft/). It covers what the pipeline does, the
reasons behind its design, and how to configure, run and evaluate it.

For domain adaptation on raw text, see
[Continued Pre-Training](continued-pretraining.md).

---

## 1. What SFT is for

Supervised Fine-Tuning teaches a model a **task** from input/output pairs. Each
example is shown as a chat turn: the user message holds an instruction plus the
input, and the assistant message holds the expected output. The model learns to
produce the output when given the input.

The task shipped with this repo is **Vietnamese address normalisation**:

| Input (`text`) | Output (`label`) |
| --- | --- |
| `68, Xthuy, Cgiay, HN` | `68, Xuân Thủy, Cầu Giấy, Hà Nội` |
| `123, Ngtroi, Q1, HCM` | `123, Nguyễn Du, Quận 1, Hồ Chí Minh` |

The model must restore diacritics, fix misspellings and expand abbreviations
(`Q.` → `Quận`, `TP.` → `Thành phố`).

SFT can start from the original base model or from a model already adapted with
[CPT](continued-pretraining.md).

---

## 2. Pipeline at a glance

```
data/address_dataset/{train,val,test}.json     [{"text": ..., "label": ...}]
        │
        ▼  Dataloader.preprocess_fn
  prompt     = [{"role": "user",      "content": PROMPT_TEMPLATE.format(text)}]
  completion = [{"role": "assistant", "content": label}]
        │
        ▼  TRL SFTTrainer
  apply chat template → input_ids + completion_mask → labels (-100 on prompt)
        │
        ▼
  train with completion-only cross-entropy
  every epoch: eval_loss + teacher-forced metrics, early stopping, best ckpt kept
        │
        ▼
  save adapter → merge into base (16-bit, CPU)
        │
        ▼  Predictor
  batched greedy generation on the test split
  → exact_match / BLEU / ROUGE-L / lexical similarity → evaluation_results.json
```

---

## 3. Quick start

```bash
# Full pipeline: train -> merge -> generation-based evaluation on test
python src/finetune/sft/train.py

# Evaluate an existing checkpoint (adapter or merged model) without training
python src/finetune/sft/evaluation.py
python src/finetune/sft/evaluation.py training_arguments.output_dir=./finetuning-checkpoints-merged

# Swap model / dataset / hyperparameters (Hydra syntax)
python src/finetune/sft/train.py model=qwen3 dataset=address-dataset
python src/finetune/sft/train.py model=llama3 training_arguments.learning_rate=1e-4

# Fine-tune on top of a CPT model
python src/finetune/sft/train.py model.model_name_or_path=./cpt-checkpoints-merged
```

All relative paths resolve against the **project root**, not the shell's working
directory. GPU selection comes from `GPU_DEVICES` in `.env`.

---

## 4. Dataset

### 4.1 Format

`.json` (a list) or `.jsonl` (one object per line):

```json
[
  {"text": "68, Xthuy, Cgiay, HN", "label": "68, Xuân Thủy, Cầu Giấy, Hà Nội"}
]
```

Column names are set by `dataset.text_col` and `dataset.label_col`.

### 4.2 Dataset config

File: [src/conf/dataset/address-dataset.yaml](../src/conf/dataset/address-dataset.yaml).

| Key | Default | Meaning |
| --- | --- | --- |
| `train_file` | `data/address_dataset/train.json` | 682 examples. Required |
| `validation_file` | `data/address_dataset/val.json` | 151 examples. Required when `do_eval=true` |
| `test_file` | `data/address_dataset/test.json` | 179 examples. Used for the final generation-based evaluation |
| `max_train_samples` / `max_eval_samples` / `max_test_samples` | `null` | Cap a split (for quick experiments) |
| `shuffle` | `true` | Shuffle train with `seed` |
| `preprocessing_num_workers` | `4` | Capped at the size of the smallest split |
| `overwrite_cache` | `true` | Ignore cached `datasets.map` results |

### 4.3 The three splits have separate roles

- **train**: gradient updates.
- **validation**: `eval_loss` each epoch, early stopping and best-checkpoint
  selection.
- **test**: the final reported numbers only.

`do_eval=true` without a `validation_file` is an **error**, not a silent fallback
to the test set. Otherwise early stopping and `load_best_model_at_end` would pick
the checkpoint on the same split the final report scores, which inflates the
result. If you have no validation file, carve one out of train, or set
`do_eval=false`.

---

## 5. Prompt construction

`Dataloader.preprocess_fn` in [train.py](../src/finetune/sft/train.py) turns every
`{text, label}` row into a TRL **conversational prompt-completion** example:

```python
{
  "prompt":     [{"role": "user",      "content": PROMPT_TEMPLATE.format(text=text)}],
  "completion": [{"role": "assistant", "content": label}],
  "chat_template_kwargs": {"enable_thinking": False},   # only if the model YAML sets any
}
```

`PROMPT_TEMPLATE` ([prompts.py](../src/finetune/sft/prompts.py)) is a zero-shot
Vietnamese instruction: a role description, four normalisation rules, and the
input address:

```
### Role:
Bạn là một trợ lý AI chuyên về chuẩn hóa địa chỉ ở Việt Nam. ...
### Instruction:
1. Thêm dấu tiếng Việt đầy đủ và chính xác.
2. Sửa các lỗi chính tả phổ biến.
3. Chuẩn hóa các từ viết tắt (ví dụ: 'Q.' thành 'Quận', 'TP.' thành 'Thành phố').
4. Chỉ phản hồi địa chỉ hoàn thiện, ...

### Hãy chuẩn hóa địa chỉ dưới đây:
### Địa chỉ gốc:
{text}

### Địa chỉ hoàn thiện:
```

`PROMPT_TEMPLATE_FEWSHOT` (the same prompt with three worked examples) is also
defined but is **not used** by training or evaluation. To switch templates, change
the import in **both** `train.py` and `predictor.py`, so that training and
inference render the same prompt.

**`chat_template_kwargs`** come from the model YAML. For Qwen3 and Qwen3.5,
`enable_thinking: false` stops the chat template from inserting an empty
`<think></think>` block. That keeps the target to the address alone and matches
what inference will render.

---

## 6. Tokenization and the completion mask

TRL's `SFTTrainer` tokenizes each example as follows:

1. **Prompt only**:
   `apply_chat_template(prompt, add_generation_prompt=True, **chat_template_kwargs)`.
   The result ends with the assistant header (e.g. `<|im_start|>assistant\n`).
2. **Prompt + completion**:
   `apply_chat_template(prompt + completion, **chat_template_kwargs)`.
   The result ends with the assistant's end-of-turn token (e.g. `<|im_end|>`).
3. **Completion mask**: `0` for the first `len(prompt_ids)` tokens, `1` for the
   rest. TRL warns if the prompt tokens are not an exact prefix of the
   prompt+completion tokens.
4. **Labels**: `labels[t] = input_ids[t]` where the mask is 1, and `-100`
   elsewhere.
5. **Truncation** to `max_length` (`512`), keeping the start
   (`truncation_mode="keep_start"`). Examples whose labels end up entirely `-100`
   (a prompt that fills the whole budget) are **dropped**, because they would
   contribute no loss.

For conversational datasets TRL does **not** append an extra EOS. The chat
template's end-of-turn token is already part of the completion, so the model
learns to emit it, and generation stops on it.

Example with a ChatML template (Qwen):

```
<|im_start|>user\n### Role: ... ### Địa chỉ hoàn thiện:\n<|im_end|>\n<|im_start|>assistant\n   ← labels = -100
68, Xuân Thủy, Cầu Giấy, Hà Nội<|im_end|>\n                                                 ← labels = token ids
```

**Training and inference render the same string.** `Predictor.format_prompt`
renders the prompt with the same template, `add_generation_prompt=True`, and the
same `chat_template_kwargs`. The point where the loss mask starts during training
is therefore exactly where generation starts at inference.

---

## 7. Training objective and loss

### 7.1 Loss function

The loss is next-token **cross-entropy** (negative log-likelihood), computed
**only on the completion tokens**:

$$
\mathcal{L}_{\text{SFT}}(\theta) \;=\; -\frac{1}{\sum_i |C_i|}\sum_{i}\sum_{t \in C_i}
\log p_\theta\!\left(y^{(i)}_{t}\,\middle|\,x^{(i)},\,y^{(i)}_{<t}\right)
$$

where $x^{(i)}$ is the rendered prompt, $y^{(i)}$ the rendered assistant turn
(address + end-of-turn token), and $C_i$ the positions whose label is not `-100`.

- **`completion_only_loss`** is not set in the config. TRL resolves it to `True`
  automatically because the dataset has `prompt` and `completion` columns.
- **`loss_type: "nll"`** is set explicitly: standard cross-entropy over the full
  logits. TRL 1.9's default, `chunked_nll`, computes the same value with less
  memory, but it cannot be used here because it breaks the per-epoch metrics
  (§9.3).
- **Normalisation**: summed over completion tokens and divided by the
  completion-token count of the whole accumulated batch (`num_items_in_batch`), so
  gradient accumulation gives the same result as one large batch.

### 7.2 Why completion-only

The prompt template is a fixed instruction of roughly 180 tokens, while the
answer is about 15–30 tokens. If the loss covered the whole sequence (for example
by feeding one pre-rendered `text` field), most of the gradient would go into
re-learning the fixed template, which carries no information. Masking the prompt
puts the whole training signal on the part the model actually has to produce.

To train on the prompt as well, set `training_arguments.completion_only_loss:
false`. This is rarely useful here.

### 7.3 Packing

`packing: false` for SFT. The examples are short, and with `max_length: 512` and
right padding the padding overhead is small. Packing is also force-disabled
regardless for models with `supports_packing: false` (Qwen3.5), because their
linear-attention layers would leak state across packed samples.

---

## 8. Training configuration

File: [src/conf/sft-conf.yaml](../src/conf/sft-conf.yaml). Defaults compose
`dataset: address-dataset` and `model: qwen3_5` (`Qwen/Qwen3.5-0.8B`, 4-bit QLoRA,
LoRA `r=16`, `alpha=32`).

| Key | Value | Notes |
| --- | --- | --- |
| `num_train_epochs` | `5` | Upper bound. Early stopping usually ends sooner |
| `per_device_train_batch_size` | `4` | |
| `gradient_accumulation_steps` | `8` | Effective batch = 32 examples per GPU |
| `per_device_eval_batch_size` | `4` | |
| `eval_accumulation_steps` | `4` | Move eval predictions to CPU every 4 steps |
| `max_length` | `512` | Training truncation, also the inference prompt budget |
| `packing` | `false` | §7.3 |
| `loss_type` | `nll` | Not TRL's default `chunked_nll` (§9.3) |
| `learning_rate` | `2e-4` | Suited to LoRA |
| `lr_scheduler_type` | `linear` | with `warmup_ratio: 0.05` |
| `weight_decay` | `0.01` | |
| `optim` | `adamw_torch_fused` | |
| `gradient_checkpointing` | `true` | `use_reentrant: false` set automatically |
| `eval_strategy` / `save_strategy` | `epoch` | |
| `save_only_model` / `save_total_limit` | `true` / `2` | |
| `load_best_model_at_end` | `true` | |
| `metric_for_best_model` | `eval_loss` | `greater_is_better: false` |
| `early_stopping_patience` | `3` | Stop after 3 epochs without `eval_loss` improvement. Needs `load_best_model_at_end` |

**Precision is not configurable.** `bf16`/`fp16` are set from the GPU in
[build_training_args](../src/utils/training_args.py): bf16 on Ampere or newer,
fp16 on older GPUs, and `tf32` is honoured only on Ampere or newer.

Model-level settings (QLoRA, LoRA targets, attention implementation,
`chat_template_kwargs`) live in [src/conf/model/](../src/conf/model/). Model
loading, QLoRA and LoRA work the same way as in CPT. See §6 of the
[CPT document](continued-pretraining.md#6-model-loading).

---

## 9. Evaluation during training (per epoch)

### 9.1 `eval_loss`

The validation split's completion-only cross-entropy, computed exactly like the
training loss. It is the signal for **best-checkpoint selection** and **early
stopping**.

### 9.2 Teacher-forced metrics

`compute_metrics_fn` in [metrics.py](../src/finetune/sft/metrics.py) also reports
`eval_exact_match`, `eval_bleu_score`, `eval_rouge_l` and
`eval_lexical_similarity`:

1. `preprocess_logits_for_metrics` takes the **argmax on the GPU** before
   predictions are gathered. Otherwise the Trainer would accumulate
   `[batch, seq, vocab]` logits for the whole eval set (with a ≈248k vocabulary
   that is tens of GB of host RAM).
2. Predictions are **shifted by one** (`preds[:, :-1]` vs `labels[:, 1:]`),
   because the logits at position *t* predict token *t+1*.
3. Only positions with `label != -100` (the completion) are decoded and compared.

These metrics are **teacher-forced**: each token is predicted from the *gold*
prefix, not from the model's own previous outputs. They are therefore optimistic.
Use them as a training signal, not as reported quality. This is why
`metric_for_best_model` is `eval_loss`. The reported numbers come from real
generation (§10).

### 9.3 Why `loss_type` is `nll`, not `chunked_nll`

[sft-conf.yaml](../src/conf/sft-conf.yaml) sets `loss_type: "nll"` on purpose.
With TRL's default `loss_type="chunked_nll"`, the model's forward pass **returns
no logits** whenever labels are present. During evaluation the Trainer would then
pass the scalar counters from that output (`num_correct_tokens`, ...) to
`preprocess_logits_for_metrics` instead of logits, and `compute_metrics` would
fail at the end of the first epoch:

```
File "src/finetune/sft/metrics.py", line 187, in compute_metrics
IndexError: too many indices for array: array is 1-dimensional, but 2 were indexed
```

`nll` gives the same loss value. Only the memory saving of the chunked path is
lost. Do not remove `loss_type` from the config while `compute_metrics` is in use.

CPT keeps the default `chunked_nll`, because it has no `compute_metrics`.

### 9.4 Other logged values

TRL logs `loss`, `grad_norm`, `learning_rate`, `entropy`, `mean_token_accuracy`
(computed over completion tokens) and `num_tokens`. The repo's callbacks add GPU
memory (`allocated_mem`, `GPU_reserved`, `peak_mem`, in GB) and `epoch_time_sec`.

---

## 10. Final evaluation: generation on the test split

After training, the model **generates** an answer for every test input with
[Predictor](../src/finetune/sft/predictor.py), and the outputs are scored against
the labels.

### 10.1 Generation

- The prompt is rendered exactly as in training (§6).
- **Batched**, `evaluation_arguments.batch_size` (default `8`).
- **Left padding** and **left truncation** during generation. Decoder-only batched
  generation needs left padding. Truncation is left-sided because the *tail* of
  the rendered prompt is the generation prompt (`<|im_start|>assistant`), and
  cutting it off would leave a prompt that never asks for an answer. The
  tokenizer's original settings are restored afterwards.
- Prompt budget: `training_arguments.max_length`.
- **Greedy decoding** by default (`do_sample: false`), because normalisation has
  one correct answer and greedy output is reproducible. `temperature`, `top_p` and
  `top_k` apply only when `do_sample: true`. `repetition_penalty` is passed only
  when it is not `1.0`.
- `max_new_tokens: 128`.
- **Stop tokens come from the checkpoint's `generation_config`.** The code never
  overrides `eos_token_id`, because the shipped list is wider than
  `tokenizer.eos_token_id`. Qwen3 stops on `<|im_end|>` *and* `<|endoftext|>`, and
  narrowing the list would let generation run past the end of the turn.
- The KV cache is re-enabled temporarily for generation (training sets
  `use_cache=false`).

### 10.2 Metrics

Defined in [metrics.py](../src/finetune/sft/metrics.py). All comparisons are
case-insensitive.

| Metric | Definition | What it tells you |
| --- | --- | --- |
| `exact_match` | Share of predictions equal to the label after `strip().lower()` | Strict task accuracy: the address is exactly right |
| `bleu_score` | Sentence BLEU (up to 4-grams, smoothing method 1) on **Vietnamese word tokens** from `pyvi.ViTokenizer`, averaged | n-gram overlap. Rewards partially correct addresses |
| `rouge_l` | F1 of the longest common subsequence over the same word tokens, averaged | Order-aware overlap. Robust to a missing or extra component |
| `lexical_similarity` | `difflib.SequenceMatcher` character ratio, averaged | Character-level closeness. Catches diacritic and typo errors |

Word tokenization with `pyvi` joins Vietnamese multi-syllable words
(`Hà_Nội`, `Cầu_Giấy`), so BLEU and ROUGE-L work on words rather than syllables.

A metric that cannot be computed (for example on an empty set) is reported as
**`NaN`, not `0.0`**, because a zero would look the same as a genuinely bad model.

### 10.3 Which model is evaluated

| Situation | Model scored |
| --- | --- |
| `merge.enabled=true` and `merge.evaluate_merged=true` (default) | The **merged** 16-bit model in `<output_dir>-merged` |
| Merge disabled, or `evaluate_merged=false` | The trained in-memory model (best checkpoint, adapter on the quantised base) |
| Standalone `evaluation.py` | `training_arguments.output_dir`, loaded as an adapter if `adapter_config.json` exists, otherwise as a full model |

The merged model is re-evaluated because merging into a dequantised 16-bit base
is not bit-exact with the 4-bit base the adapter was trained against.

### 10.4 Results file

A report is printed and `evaluation_results.json` is written to the project root:

```json
{
  "model_path": ".../finetuning-checkpoints",
  "metrics": {
    "exact_match": 0.5028,
    "bleu_score": 0.8760,
    "rouge_l": 0.9526,
    "lexical_similarity": 0.9622,
    "num_samples": 179
  },
  "detailed_results": [
    {"id": 0, "input": "...", "prediction": "...", "reference": "...", "exact_match": true}
  ]
}
```

`detailed_results` lists every test example, so you can filter on
`"exact_match": false` for error analysis.

---

## 11. Merging the adapter

Same mechanism as CPT ([src/models/merge.py](../src/models/merge.py)). After
training, if `merge.enabled=true` and LoRA is used, the training model is
released, the base is loaded in 16-bit on CPU (`merge.dtype`), the adapter is
merged, and the result plus tokenizer are written to `merge.output_dir`, or
`<output_dir>-merged` when that is null. With full fine-tuning (no `lora` block),
the merge is skipped because `output_dir` already holds a complete model.

Manual merge:

```bash
python merge_model.py --adapter ./finetuning-checkpoints \
    --output ./finetuning-checkpoints-merged --base Qwen/Qwen3.5-0.8B
```

---

## 12. Outputs

| Path | Content |
| --- | --- |
| `finetuning-checkpoints/` | Best LoRA adapter, tokenizer, `checkpoint-*` dirs (max 2) |
| `finetuning-checkpoints-merged/` | Merged standalone model + tokenizer |
| `evaluation_results.json` | Test metrics + per-example predictions |

Set `training_arguments.report_to=mlflow` (with credentials in `.env`) to log to
the MLflow experiment `llm-finetuning`.

---

## 13. Adapting to a new task

1. **Data**: produce `train/val/test` JSON with an input column and an output
   column. Add `src/conf/dataset/<name>.yaml` (copy `address-dataset.yaml`) and set
   `text_col` / `label_col`.
2. **Prompt**: write a new template in [prompts.py](../src/finetune/sft/prompts.py)
   with a `{text}` placeholder, and use it in **both** `train.py` and
   `predictor.py`.
3. **Lengths**: make sure the rendered prompt plus the longest answer fits in
   `max_length`, and the longest answer fits in `evaluation_arguments.max_new_tokens`.
4. **Decoding**: keep greedy decoding for tasks with one right answer. Enable
   sampling only for open-ended generation.
5. **Metrics**: exact match, BLEU and ROUGE-L suit short, deterministic outputs.
   For other tasks, extend `EvaluateMetrics`.
6. Run `python src/finetune/sft/train.py dataset=<name>`.

---

## 14. Tuning and troubleshooting

| Symptom / goal | What to change |
| --- | --- |
| `IndexError` in `metrics.py` at the end of epoch 1 | `loss_type` was removed or set to `chunked_nll`. Restore `loss_type: "nll"` (§9.3) |
| Out of memory | Lower `per_device_train_batch_size` (raise `gradient_accumulation_steps` to keep the effective batch), keep `qlora: true`, lower `per_device_eval_batch_size` |
| High teacher-forced metrics, low test metrics | Expected gap (§9.2). Trust the generation-based numbers |
| Outputs keep going after the address | Check `chat_template_kwargs` and the checkpoint's `generation_config.eos_token_id`. Training and inference must use the same template |
| Output contains `<think>` text | Set `chat_template_kwargs.enable_thinking: false` for Qwen3/Qwen3.5 |
| `eval_loss` stops improving early | Expected. Early stopping keeps the best epoch. Raise data quality or quantity, or LoRA `r` |
| Adapter learns very little | Learning rate too low for LoRA (`2e-5` is a full fine-tuning value), or `r` too small |
| `Mismatch between tokenized prompt and the start of tokenized prompt+completion` warning | The chat template renders the prompt differently when the completion follows. The mask boundary may be off by a few tokens. Check the template |
| `do_eval=true requires dataset.validation_file` | Provide a validation split, or set `do_eval=false` (and disable early stopping) |
| `do_train=false has no meaning here` | Use `evaluation.py` to evaluate without training |

---

## 15. Code map

| File | Role |
| --- | --- |
| [src/finetune/sft/train.py](../src/finetune/sft/train.py) | Entry point: dataset → train → merge → evaluate |
| [src/finetune/sft/prompts.py](../src/finetune/sft/prompts.py) | Prompt templates |
| [src/finetune/sft/metrics.py](../src/finetune/sft/metrics.py) | Metric definitions, teacher-forced `compute_metrics` |
| [src/finetune/sft/predictor.py](../src/finetune/sft/predictor.py) | Batched generation, scoring, results file |
| [src/finetune/sft/evaluation.py](../src/finetune/sft/evaluation.py) | Standalone checkpoint evaluation |
| [src/conf/sft-conf.yaml](../src/conf/sft-conf.yaml) | Task configuration |
| [src/conf/dataset/address-dataset.yaml](../src/conf/dataset/address-dataset.yaml) | Dataset paths and columns |
| [src/models/](../src/models/) | Tokenizer/model loading, QLoRA/LoRA, merge |
| [src/utils/training_args.py](../src/utils/training_args.py) | Precision resolution, packing guard |
| [tests/test_metrics.py](../tests/test_metrics.py) | Metric unit tests |
