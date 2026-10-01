# Continued Pre-Training (CPT)

This document explains the Continued Pre-Training pipeline in
[src/finetune/cpt/](../src/finetune/cpt/). It covers what the pipeline does, the
reasons behind its design, and how to configure, run and evaluate it.

For the instruction-tuning counterpart, see
[Supervised Fine-Tuning](supervised-finetuning.md).

---

## 1. What CPT is for

Continued Pre-Training adapts a pretrained LLM to a **domain** by training it on
raw, unlabeled text from that domain. In this repo the domain is Vietnamese
administrative reports (`data/report_dataset/*.md`).

There are no instructions, prompts or answers. The model reads domain text and
learns to predict it, which teaches it:

- domain vocabulary and terminology (agency names, legal references, report jargon);
- the writing style and document structure of the domain;
- facts and phrasing that recur across the corpus.

CPT is usually a first stage. The domain-adapted model can then be
instruction-tuned with [SFT](supervised-finetuning.md).

---

## 2. Pipeline at a glance

```
data/report_dataset/*.md            (one file = one report)
        │
        ▼  prepare_cpt_data.py   (runs automatically when prepare.auto=true)
  ┌─────────────────────────────────────────────────────────────┐
  │ 1. read documents                                           │
  │ 2. strip administrative boilerplate     (boilerplate.py)    │
  │ 3. shuffle + split train/val by DOCUMENT                    │
  │ 4. structure-aware chunking to ≤ max_length-1 tokens        │
  │                                          (chunking.py)      │
  │ 5. dedupe chunks, drop val chunks that also occur in train  │
  └─────────────────────────────────────────────────────────────┘
        │
        ▼
data/prepared_dataset/{train,val}.json    [{"text": "..."}]
        │
        ▼  cpt_train.py
  load tokenizer + base model (4-bit QLoRA by default) + LoRA adapter
        │
        ▼
  TRL SFTTrainer, language-modelling objective (loss on every token)
  eval_loss every epoch, best checkpoint kept
        │
        ▼
  save adapter → merge into base (16-bit, CPU) → perplexity on val
```

---

## 3. Quick start

```bash
# Full pipeline: chunk -> train -> merge -> perplexity
python src/finetune/cpt/cpt_train.py

# Force re-chunking (e.g. after adding reports or changing max_length)
python src/finetune/cpt/cpt_train.py prepare.force=true

# Use the prepared JSON as-is, never re-chunk
python src/finetune/cpt/cpt_train.py prepare.auto=false

# Keep the administrative frame (header, signature, recipients)
python src/finetune/cpt/cpt_train.py prepare.boilerplate.strip=false

# Switch base model / override hyperparameters (Hydra syntax)
python src/finetune/cpt/cpt_train.py model=qwen3 training_arguments.learning_rate=2e-4

# Perplexity of an existing checkpoint, no training
python src/finetune/cpt/cpt_evaluation.py
python src/finetune/cpt/cpt_evaluation.py training_arguments.output_dir=./cpt-checkpoints-merged
```

All relative paths (`source_dir`, `train_file`, `output_dir`, result files)
resolve against the **project root**, not the shell's working directory. See
`resolve_path` in [src/env_setup.py](../src/env_setup.py).

GPU selection comes from `GPU_DEVICES` in `.env`.

---

## 4. Input data

### 4.1 Raw corpus

| Key | Default | Meaning |
| --- | --- | --- |
| `dataset.source_dir` | `data/report_dataset` | Directory scanned recursively for `**/*.md` |
| `dataset.train_file` | `data/prepared_dataset/train.json` | Where prepared training chunks are written and read |
| `dataset.validation_file` | `data/prepared_dataset/val.json` | Where prepared validation chunks are written and read |

Rules ([read_documents](../src/finetune/cpt/prepare_cpt_data.py)):

- **One `.md` file is one document.** Do not concatenate several reports into
  one file, or the document-level split (§5.2) can no longer keep them apart.
- Empty files are skipped with a warning. If every file is empty, preparation fails.
- Files are read as UTF-8 and stripped of leading and trailing whitespace.

### 4.2 Prepared format

```json
[
  {"text": "chunk of report text ..."},
  {"text": "another chunk ..."}
]
```

You can skip the preparation step (`prepare.auto=false`) and supply your own
files in this format, as `.json` or `.jsonl`. Each `text` should fit in
`training_arguments.max_length - 1` tokens. Anything longer is truncated by the
trainer.

---

## 5. Data preparation

Entry point: `prepare_from_config` in
[prepare_cpt_data.py](../src/finetune/cpt/prepare_cpt_data.py). It is called at
the start of `cpt_train.py`.

### 5.1 When preparation runs

| `prepare.auto` | `prepare.force` | Prepared files exist? | Result |
| --- | --- | --- | --- |
| `false` | any | any | Never prepares. The JSON files are used as-is |
| `true` | `false` | yes | Skipped (logged) |
| `true` | `false` | no | Prepares |
| `true` | `true` | any | Re-prepares, overwriting |

Constraints:

- `prepare.auto=true` requires `dataset.source_dir`.
- `train_file` and `validation_file` must be in the **same directory**.
- The chunk budget comes from `training_arguments.max_length` (falling back to
  `model.model_max_length`). If you change `max_length`, re-chunk with
  `prepare.force=true`, because the existing files were cut for the old budget.

### 5.2 Order of operations

1. **Read** all documents.
2. **Strip boilerplate** (§5.3), if `prepare.boilerplate.strip=true`.
3. **Shuffle documents** with `seed`, then **split at document level** with
   `prepare.val_ratio` (default `0.15`). No chunk of a training report can then
   leak into validation.
   - The split always leaves both sides non-empty. The validation count is
     `min(n-1, max(1, round(n * val_ratio)))`.
   - With **only one document**, the split falls back to chunk level and logs a
     warning. Validation is then much less independent, so add more reports when
     you can.
4. **Chunk** each document (§5.4).
5. **Shuffle** the chunks of each split.
6. **Deduplicate** (§5.5), if `prepare.dedupe=true`.
7. **Write** JSON files and log chunk/token counts and average tokens per chunk.

### 5.3 Boilerplate stripping

Module: [boilerplate.py](../src/finetune/cpt/boilerplate.py).

Official Vietnamese reports carry an administrative frame: the republic header
and motto, the place/date line, the document number, the recipient list, the
signature block and digital-signature metadata. This frame is the same in every
document. If it stays in, the model sees it hundreds of times per epoch and
spends capacity memorising it instead of the domain.

Stripping runs **before** chunking. After chunking, the frame would sit inside
the first and last chunk of each report, mixed with unique content, where
chunk-level deduplication cannot remove it.

Two detectors run together because neither is enough on its own:

**a) Universal patterns (`PATTERNS`)**: these work on a single file. Lines are
compared after removing diacritics and upper-casing, so `Nơi nhận:` and
`NOI NHAN:` both match.

| Reason | Example line removed |
| --- | --- |
| `republic` | `CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM` |
| `motto` | `Độc lập - Tự do - Hạnh phúc` |
| `place_date` | `Hà Nội, ngày 05 tháng 5 năm 2025` |
| `doc_number` | `Số: 316/BC-CP` |
| `recipients` | `Nơi nhận:` (and the bullet lines directly under it → `recipient_entry`) |
| `archive` | `- Lưu: VT, ...` |
| `signature_role` | `TM. CHÍNH PHỦ`, `KT. THỦ TƯỚNG`, `TL. BỘ TRƯỞNG` |
| `signed_marker` | `(Đã ký)` |
| `digital_signature` | `Ký bởi: ...`, `Thời gian ký: ...` |
| `end_marker` | `./.` |
| `page_number` / `page_dashes` | `Trang 3/12`, `- 3 -`, a bare number |

**b) Corpus-learned template lines (`build_boilerplate_index`)**: this needs at
least **5 documents**. It finds short lines (≤ `max_line_chars`, default 120)
that appear in at least `min_document_ratio` (default 50%) of documents. A line
is counted once per document, so a line repeated inside one report does not look
like template text. This catches whatever *this* corpus's template repeats, such
as the issuing body or standing closings. Removals are logged as
`corpus_template`.

**c) Repeated lines in the document head**: a line that occurs twice within
the first 40 lines of a document is dropped as `repeated_in_head`.

**Safety valve:** if stripping would remove more than `max_strip_ratio` (default
25%) of a document's characters, the detectors are probably wrong about that
document, so it is left **untouched**. The number of skipped documents is logged
once, as a single warning.

The log always states what was removed and why, so the first run on a new
corpus can be audited:

```
Learned 9 template line(s) present in >= 500/1000 documents. Examples: [...]
Boilerplate removed by reason: {'corpus_template': 3000, 'recipient_entry': 2000, ...}
Corpus size 16,930,899 -> 16,757,899 chars (1.0% removed)
```

| Config key (`prepare.boilerplate.*`) | Default | Effect |
| --- | --- | --- |
| `strip` | `true` | Turn the whole step on or off |
| `min_document_ratio` | `0.5` | Share of documents a short line must appear in to count as template text |
| `max_line_chars` | `120` | Longer lines are never treated as template text (universal patterns ignore this) |
| `max_strip_ratio` | `0.25` | Per-document cap on removed characters |

### 5.4 Structure-aware chunking

Module: [chunking.py](../src/finetune/cpt/chunking.py).

Fixed-size windows would cut through headings, tables and sentences. The chunker
follows the document's structure instead, in three phases:

**Phase 1: `parse_blocks`.** The document is split into **heading**, **table**
and **paragraph** blocks. A block records the line range it occupies, not its
text, so chunks are later sliced verbatim from the original lines and no
whitespace is reconstructed or corrupted.

Recognised headings (lines ≤ 200 characters, except markdown headings):

| Pattern | Example |
| --- | --- |
| Markdown | `# Title`, `## Section` |
| `phan` | `PHẦN I. ...` |
| `chuong` | `CHƯƠNG II. ...` |
| `muc` | `MỤC 1. ...` |
| `upper_alpha` | `A. ...` |
| `roman` | `I. ...`, `IV) ...` |
| `arabic` | `1. ...` |
| `decimal` | `1.1. ...` |
| `sub_decimal` | `1.1.1. ...` |
| `lower_alpha` | `a) ...` |

Table blocks are runs of consecutive `| ... |` lines. Paragraphs are runs of
non-blank lines that start no other block. Heading levels are normalised to the
patterns each document actually uses.

**Phase 2: `pack_blocks`.** Consecutive blocks are packed into a chunk until the
next block would exceed the budget. A heading at the end of a full chunk is
**carried over** to the next chunk so it stays with the content it introduces. A
chunk made only of headings is merged into the previous chunk when it fits.

**Phase 3: `split_oversized`.** A single block larger than the whole budget (a
long table, a very long paragraph) is split recursively on separators, in order
of preference: `"\n\n"`, `"\n"`, `". "`, `"; "`, `", "`, `" "`. If no separator
exists, it is cut in half. Each piece keeps its separator, so no character is
lost. A heading attached to an oversized block is put back in front of the first
piece.

Guarantees, covered by [tests/test_chunking.py](../tests/test_chunking.py):

- no chunk exceeds the token budget, so the trainer never truncates one;
- every character survives, in order;
- no heading is separated from the content it introduces;
- no chunk consists only of headings.

**Token budget.** Chunks are measured with the *training tokenizer*
(`add_special_tokens=False`). The budget is `max_length - 1`, because TRL appends
an EOS token to every language-modelling sample (`EOS_RESERVE = 1` in
`prepare_cpt_data.py`).

> **Note:** a tokenizer that also prepends a BOS token (e.g. Llama 3,
> `<|begin_of_text|>`) adds one more special token that the budget does not
> reserve. A chunk filled to exactly `max_length - 1` tokens then goes one token
> over and loses its final token (the EOS) to truncation. Qwen tokenizers do not
> add a BOS, so the default setup is unaffected.

### 5.5 Deduplication and leak removal

With `prepare.dedupe=true` (default):

1. **Repeat chunks** are dropped within train and within validation. Identity
   ignores whitespace: two chunks that differ only in blank lines count as the
   same. The first occurrence is kept, so output stays stable.
2. **Validation chunks whose text also occurs in training** are dropped. The
   document-level split keeps chunks of the same report apart, but identical text
   in two *different* reports (shared appendices, standing paragraphs) still gets
   through. Scoring perplexity on text the model trained on would make the number
   optimistic.

If deduplication empties either split (a corpus of near-identical documents),
preparation fails with a message telling you to set `prepare.dedupe=false`.

---

## 6. Model loading

Shared with SFT. See [src/models/](../src/models/).

- **Tokenizer** (`load_tokenizer`): loaded with `model.padding_side` (right). If
  the tokenizer has no pad token, it uses the EOS token.
- **Base model** (`load_base_model`): loaded with the class chosen by
  `model.model_class` (`causal_lm` → `AutoModelForCausalLM`). For Qwen3.5, which
  ships as a VLM checkpoint, `causal_lm` loads only the language model and
  ignores the vision tower and the MTP head. The compute dtype is bf16 on Ampere
  or newer, fp16 on older GPUs, and fp32 on CPU.
- **QLoRA** (`model.qlora: true`, default): base weights are 4-bit NF4 with double
  quantisation. Compute happens in the dtype above.
  `prepare_model_for_kbit_training` is applied before LoRA.
- **LoRA** (`model.lora`): `r=16`, `lora_alpha=32`, `lora_dropout=0.05`, on the
  projection modules listed per architecture. For Qwen3.5 that includes the Gated
  DeltaNet projections `in_proj_qkv`, `in_proj_z` and `out_proj`. Setting
  `target_modules: null` uses PEFT's `"all-linear"`. Removing the `lora` block
  entirely switches to **full fine-tuning**.
- `use_cache=false` during training, which gradient checkpointing requires. It
  is re-enabled temporarily for generation.

---

## 7. Training objective and loss

### 7.1 Objective: causal language modelling over the whole chunk

`CPTFinetuning.train` passes the `{"text": ...}` dataset to TRL's `SFTTrainer`
without a custom loss. Because the dataset has a single text field
(`dataset_text_field: "text"`) and no `prompt`/`completion` columns, TRL treats it
as a **language-modelling dataset**:

1. TRL appends `tokenizer.eos_token` to every text, so the model learns where a
   passage ends.
2. Each text is tokenized as-is, with no chat template.
3. `labels = input_ids`, with **no masking**. Every token contributes to the loss
   (`completion_only_loss` resolves to `False` for this dataset type).

### 7.2 Loss function

The loss is the standard next-token **cross-entropy** (negative log-likelihood),
averaged over all predicted tokens:

$$
\mathcal{L}_{\text{CPT}}(\theta) \;=\; -\frac{1}{N}\sum_{i}\sum_{t=1}^{T_i-1}
\log p_\theta\!\left(x^{(i)}_{t+1}\,\middle|\,x^{(i)}_{\le t}\right)
$$

where $N$ is the number of predicted, non-padding tokens in the (accumulated)
batch. Two implementation details:

- **`loss_type`**: not set in the config, so TRL 1.9 uses its default
  **`chunked_nll`**. It computes the same value as plain `nll`, but skips the
  `lm_head` projection on ignored positions and computes the cross-entropy in
  chunks of 256 tokens, which lowers peak activation memory. This matters with
  large vocabularies (≈248k tokens for Qwen3.5). Alternatives: `nll` and `dft`
  (Dynamic Fine-Tuning). `chunked_nll` does not work if `lm_head` itself is a
  LoRA target, so keep `lm_head` out of `target_modules`.
- **Normalisation across gradient accumulation**: the loss is summed over tokens
  and divided by the token count of the whole accumulated batch
  (`num_items_in_batch`), not averaged per micro-batch. Long and short chunks
  therefore carry weight proportional to their token count.

### 7.3 Packing

`cpt-conf.yaml` sets `packing: false`: each chunk is one training sample. With
`per_device_train_batch_size: 1` there is no padding either, and the chunker
already fills each chunk close to the `max_length - 1` budget, so packing would
save little.

Packing could not be used with the default model anyway. Models that declare
`supports_packing: false`, including **Qwen3.5**, have it **force-disabled** (with
a warning) by `resolve_packing` in
[training_args.py](../src/utils/training_args.py). Their Gated DeltaNet
(linear-attention) layers respect sample boundaries only inside the
`flash-linear-attention` kernels, and the pure-torch fallback would leak the
recurrent state from one packed sample into the next.

For models that support it (`model=qwen3`, `model=llama3`), packing can be
enabled with `training_arguments.packing=true`. With TRL's default `bfd`
(best-fit decreasing) strategy, several chunks are then concatenated into one
`max_length` sequence and trained **padding-free**: `position_ids` restart at each
chunk boundary, so attention does not cross samples (with FlashAttention). This is
worth it mainly when many chunks are much shorter than the budget, or with
`per_device_train_batch_size` > 1.

### 7.4 Metrics logged during training

TRL logs `loss`, `learning_rate`, `grad_norm`, `entropy`, `mean_token_accuracy`
and `num_tokens` every `logging_steps`. The repo's callbacks add GPU memory
(`allocated_mem`, `GPU_reserved`, `peak_mem` in GB) and `epoch_time_sec`.

---

## 8. Training configuration

File: [src/conf/cpt-conf.yaml](../src/conf/cpt-conf.yaml). Defaults compose
`dataset: report-dataset` and `model: qwen3_5`.

| Key | Value | Notes |
| --- | --- | --- |
| `num_train_epochs` | `2` | CPT overfits quickly on small corpora. Watch `eval_loss` |
| `per_device_train_batch_size` | `1` | Long sequences (2048 tokens) |
| `gradient_accumulation_steps` | `32` | Effective batch = 32 sequences per GPU |
| `max_length` | `2048` | Also the chunking budget (minus 1 for EOS) |
| `packing` | `false` | Can be enabled for models with `supports_packing: true` (§7.3) |
| `learning_rate` | `1e-4` | Suited to LoRA. Use ~`2e-5` only for full fine-tuning |
| `lr_scheduler_type` | `cosine` | with `warmup_ratio: 0.03` |
| `weight_decay` | `0.01` | |
| `optim` | `adamw_torch_fused` | |
| `gradient_checkpointing` | `true` | `use_reentrant: false` is set automatically |
| `eval_strategy` / `save_strategy` | `epoch` | |
| `save_only_model` / `save_total_limit` | `true` / `2` | No optimizer state, at most 2 checkpoints |
| `load_best_model_at_end` | `true` | Selected by `eval_loss` (lower is better) |
| `early_stopping_patience` | `0` | Disabled for CPT |
| `seed` / `deterministic` | `42` / `false` | `deterministic=true` can fail with fused/flash kernels |

**Precision is not configurable.** `bf16`/`fp16` are set from the GPU in
[build_training_args](../src/utils/training_args.py) before the config object is
built: bf16 on Ampere or newer, fp16 on older GPUs. `tf32` is honoured only on
Ampere or newer. Setting these in YAML has no effect.

---

## 9. Evaluation

CPT is scored with **perplexity**, the exponential of the average per-token
cross-entropy. Lower is better. A perplexity of *k* means the model is, on
average, as uncertain as choosing uniformly among *k* tokens.

### 9.1 During training

- `eval_loss` on the validation chunks every epoch. Its loss is computed exactly
  like the training loss (§7), including the appended EOS.
- After training, the final `eval_loss` and `perplexity = exp(eval_loss)` are
  logged.

### 9.2 After training: `CPTEvaluator`

[cpt_evaluation.py](../src/finetune/cpt/cpt_evaluation.py) computes a
**token-weighted** perplexity over the validation file:

```
for each validation chunk:
    loss_i   = model(input_ids, labels=input_ids).loss    # mean over predicted tokens
    n_i      = (#tokens - 1)                               # positions actually predicted
total_loss  += loss_i * n_i
perplexity   = exp(total_loss / Σ n_i)
```

Weighting by `n_i` gives a true per-token average, so short chunks do not count
as much as long ones.

Which model is evaluated:

| Situation | Model scored |
| --- | --- |
| `merge.enabled=true` and `merge.evaluate_merged=true` (default) | The **merged** 16-bit model in `<output_dir>-merged` |
| Merge disabled, or `evaluate_merged=false` | The trained in-memory model (adapter on the quantised base) |
| Standalone `cpt_evaluation.py` | `training_arguments.output_dir`, loaded as an adapter if `adapter_config.json` exists, otherwise as a full model |

Results are printed and written to `cpt_evaluation_results.json` in the project
root:

```json
{
  "model_path": ".../cpt-checkpoints-merged",
  "perplexity": {"perplexity": <float>, "avg_loss": <float>, "total_tokens": <int>},
  "generation_results": []
}
```

`CPTEvaluator.full_evaluation(generation_prompts=[...])` can also produce sampled
continuations (`temperature=0.7`, `top_p=0.9`, 256 new tokens) for a quick
qualitative check of the domain style. The CLI entry point passes no prompts, so
this is opt-in from Python.

### 9.3 Reading the numbers

- The standalone perplexity and `exp(eval_loss)` from the trainer are close but
  **not identical**. The evaluator does not append the EOS token, scores one chunk
  at a time, and may score a merged 16-bit model instead of the 4-bit-base
  adapter. Compare a model with itself using one method consistently.
- To measure domain adaptation, compare perplexity **before and after** CPT on
  the same validation file. Point `training_arguments.output_dir` at the base
  model id or a directory and run `cpt_evaluation.py`.
- Perplexity is comparable only between models that share a **tokenizer**.
- Training loss falling while `eval_loss` rises means the model is memorising
  the corpus. Reduce epochs or the learning rate, or add data.

---

## 10. Merging the adapter

When `merge.enabled=true` and LoRA is in use, the adapter is merged into the base
model right after training ([src/models/merge.py](../src/models/merge.py)):

- The training model is released first to free VRAM.
- The base model is loaded in **16-bit on CPU** (`merge.dtype`, default
  `bfloat16`), even when training used 4-bit QLoRA. Merging into dequantised
  weights is the standard approach, but it is not bit-exact with what the adapter
  saw during training, which is why the merged model is evaluated again.
- Output goes to `merge.output_dir`, or `<training_arguments.output_dir>-merged`
  when that is null. The tokenizer is saved alongside.

Manual merge:

```bash
python merge_model.py --adapter ./cpt-checkpoints --output ./cpt-checkpoints-merged \
    --base Qwen/Qwen3.5-0.8B --model-class causal_lm --dtype bfloat16
```

To chain CPT into SFT, point the SFT run's `model.model_name_or_path` at the
merged CPT directory.

---

## 11. Outputs

| Path | Content |
| --- | --- |
| `data/prepared_dataset/train.json`, `val.json` | Prepared chunks |
| `cpt-checkpoints/` | Final (best) LoRA adapter, tokenizer, `checkpoint-*` dirs (max 2) |
| `cpt-checkpoints-merged/` | Merged standalone model + tokenizer |
| `cpt_evaluation_results.json` | Perplexity results |

Set `training_arguments.report_to=mlflow` (with credentials in `.env`) to log the
run to the MLflow experiment `logging.mlflow.experiment_name` (`llm-cpt`).

---

## 12. Tuning and troubleshooting

| Symptom / goal | What to change |
| --- | --- |
| Out of memory | Lower `max_length` (then `prepare.force=true`), keep `qlora: true`, keep gradient checkpointing, reduce LoRA `r` |
| Very slow on Qwen3.5 | Install `flash-linear-attention` and `causal-conv1d`. Without them the Gated DeltaNet layers use a pure-torch fallback |
| `eval_loss` rises after epoch 1 | Fewer epochs, lower learning rate, or set `early_stopping_patience` > 0 |
| Adapter barely changes the model | Learning rate too low for LoRA (`2e-5` is a full fine-tuning value), or `r` too small |
| Model forgets general/instruction ability | Expected side effect of CPT. Mix in general text, lower the learning rate, or follow up with SFT |
| "Deduplication emptied a side" | Corpus of near-identical documents. Set `prepare.dedupe=false` or add more varied reports |
| Many documents "left untouched" by stripping | Check the learned template lines in the log, then adjust `min_document_ratio` / `max_strip_ratio` |
| Changed `max_length` but chunks did not change | Prepared files already existed. Rerun with `prepare.force=true` |
| `do_train=false has no meaning here` | Use `cpt_evaluation.py` to evaluate without training |

---

## 13. Code map

| File | Role |
| --- | --- |
| [src/finetune/cpt/cpt_train.py](../src/finetune/cpt/cpt_train.py) | Entry point: prepare → train → merge → evaluate |
| [src/finetune/cpt/prepare_cpt_data.py](../src/finetune/cpt/prepare_cpt_data.py) | Reading, splitting, dedup, writing prepared JSON |
| [src/finetune/cpt/boilerplate.py](../src/finetune/cpt/boilerplate.py) | Administrative-frame removal |
| [src/finetune/cpt/chunking.py](../src/finetune/cpt/chunking.py) | Structure-aware chunker |
| [src/finetune/cpt/cpt_evaluation.py](../src/finetune/cpt/cpt_evaluation.py) | Perplexity + optional generation samples |
| [src/conf/cpt-conf.yaml](../src/conf/cpt-conf.yaml) | Task configuration |
| [src/conf/dataset/report-dataset.yaml](../src/conf/dataset/report-dataset.yaml) | Corpus and prepared-file paths |
| [src/models/](../src/models/) | Tokenizer/model loading, QLoRA/LoRA, merge |
| [src/utils/training_args.py](../src/utils/training_args.py) | Precision resolution, packing guard |
| [tests/](../tests/) | Unit tests for boilerplate, chunking, preparation |
