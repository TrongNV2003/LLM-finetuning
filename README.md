# LLM Fine-tuning Project

Pipeline for fine-tuning LLMs with TRL + LoRA/QLoRA, supporting **Supervised
Fine-Tuning (SFT)** and **Continued Pre-Training (CPT)**.

## Overview

1. **CPT** — domain adaptation on long, unlabeled text (e.g. reports).
2. **SFT** — task training on instruction/response pairs (e.g. Vietnamese
   address normalisation).

## Installation

```bash
conda create -n finetuning python=3.13
conda activate finetuning
pip install -e .
cp .env.example .env      # set GPU_DEVICES, HF_TOKEN, MLflow creds
```

Optional kernels (compiled against your CUDA, install by hand):

```bash
pip install flash-attn --no-build-isolation          # for attn_implementation=flash_attention_2
pip install flash-linear-attention causal-conv1d     # speeds up Qwen3.5's Gated DeltaNet layers
```

GPU selection comes from `GPU_DEVICES` in `.env` (e.g. `GPU_DEVICES=0,1`)

## Usage

### 1. Continued Pre-Training (CPT)

Put the report `.md` files in `data/report_dataset` — **one file is one report
document** — and run:

```bash
python src/finetune/cpt/cpt_train.py       # chunk -> train -> merge -> perplexity
python src/finetune/cpt/cpt_evaluation.py  # perplexity only, standalone
```

```bash
python src/finetune/cpt/cpt_train.py prepare.force=true   # re-chunk
python src/finetune/cpt/cpt_train.py prepare.auto=false   # use existing JSON as-is
```

Prepared format dataset: `[{"text": "chunk of text"}]`.

#### Preprocessing

**Boilerplate** removes the administrative frame around each report **before** chunking — afterwards the frame sits inside the first and last chunk of every report, where chunk-level dedup cannot reach it (those chunks also hold unique content).

Two detectors, because neither is enough alone:

| Detector | Catches | Needs |
| --- | --- | --- |
| `PATTERNS` | republic header + motto, place/date line, `Số: …/BC-…`, `Nơi nhận:` and its recipient bullets, `TM./KT./TL.` signature roles, `(Đã ký)`, digital-signature metadata, `./.`, page furniture | nothing — works on one file |
| `build_boilerplate_index` | whatever *this* corpus's template repeats — issuing body, session line, standing closings | ≥ 5 documents |


The log always names what was removed and why, so the first run on a new corpus is
auditable:

```
Learned 9 template line(s) present in >= 500/1000 documents. Examples: ['báo cáo', 'quốc hội', 'nơi nhận:', ...]
Boilerplate removed by reason: {'corpus_template': 3000, 'recipient_entry': 2000, 'republic': 1000, ...}
Corpus size 16,930,899 -> 16,757,899 chars (1.0% removed)
```

Turn it off with `prepare.boilerplate.strip=false`, or `--no_strip_boilerplate` on
the CLI.

#### Chunking

**Chunking** is structure-aware, in three phases:
parse the document into heading / table / paragraph blocks, pack consecutive
blocks up to the token budget, and recursively split any single block that is
bigger than the budget on its own.

It recognises Vietnamese report numbering as headings (`PHẦN I.`, `CHƯƠNG II.`,
`MỤC 1.`, `I.`, `A.`, `1.`, `1.1.`, `1.1.1.`, `a)`) as well as markdown `#`, and
guarantees four things:

- no chunk exceeds the budget, so the trainer never truncates one;
- every character survives, in order;
- no heading is separated from the content it introduces;
- no chunk is nothing but headings.

`prepare.dedupe` (on by default) then drops repeat chunks and any validation chunk
whose text also occurs in training.
### 2. Supervised Fine-Tuning (SFT)

```bash
python src/finetune/sft/train.py                           # train -> merge -> generation eval
python src/finetune/sft/evaluation.py                      # eval a checkpoint standalone
```

Dataset format: `[{"text": "incomplete_address", "label": "complete_address"}]`.

### Switching model or dataset

Hydra composes `model` and `dataset` groups, so nothing needs editing to swap
either:

```bash
python src/finetune/sft/train.py model=qwen3 dataset=address-dataset
python src/finetune/sft/train.py model=llama3 training_arguments.learning_rate=1e-4
```

## Configuration

`src/conf/sft-conf.yaml` and `src/conf/cpt-conf.yaml` hold the task settings;
`src/conf/model/*.yaml` hold everything architecture-specific. Shipped models:
[qwen3_5.yaml](src/conf/model/qwen3_5.yaml) (Qwen3.5, VLM + hybrid),
[qwen3.yaml](src/conf/model/qwen3.yaml) (Qwen3 dense),
[llama3.yaml](src/conf/model/llama3.yaml).

Two things are deliberately *not* configurable:

- **`bf16` / `fp16`** are resolved from the GPU in
  [build_training_args](src/utils/training_args.py), before the config object is built.
  `TrainingArguments` raises for `bf16=true` on pre-Ampere hardware and freezes
  the Accelerator's mixed-precision mode at construction, so setting it in YAML
  either crashes or is silently ignored. `tf32` stays opt-in but is downgraded the
  same way.
- **Relative paths** (`train_file`, `output_dir`, result files) resolve against the
  project root, never the shell cwd — see `resolve_path` in
  [src/env_setup.py](src/env_setup.py). Run the scripts from anywhere and the
  outputs land in the same place.

### Adding a model

Copy the closest YAML and set:

| Key | What to consider |
| --- | --- |
| `model_class` | `causal_lm` for text (also the right choice to train only the LM of a VLM checkpoint), `image_text_to_text` to keep the vision tower |
| `attn_implementation` | `sdpa` unless `flash-attn` is installed |
| `supports_packing` | `false` for hybrid/linear-attention/SSM models — packing leaks recurrent state across samples |
| `lora.target_modules` | the projection names of *this* architecture; `null` falls back to PEFT `"all-linear"` |
| `chat_template_kwargs` | e.g. `enable_thinking: false` for Qwen3/Qwen3.5 |
| `qlora` | `true` for 4-bit base weights |

To discover the projection names of a new architecture:

```python
from src.models import find_all_linear_names
print(find_all_linear_names(model))   # inspection helper, not a config value
```

### Hyperparameters worth tuning

- **LoRA rank `r`** — higher = more capacity, more memory. Prefer raising `r`
  over enabling `use_dora`, which is slow under 4-bit and harder to merge.
- **Learning rate** — LoRA wants `1e-4`–`2e-4` for both SFT and CPT. `2e-5` is a
  full fine-tuning value and barely moves an adapter.
- **Epochs** — watch `eval_loss`; `early_stopping_patience` is on by default.

## Design notes

**Loss is computed on the completion only.** The SFT dataloader emits
`prompt`/`completion` columns, which makes TRL build a `completion_mask` and set
`completion_only_loss=True`. Feeding one pre-rendered `text` field instead trains
on the prompt too — with a fixed ~180-token instruction template, most of the
gradient would go into re-learning that template.

**Training and inference render the same string.** TRL tokenizes the prompt half
with `add_generation_prompt=True`, and `Predictor.format_prompt` does the same
with the same `chat_template_kwargs`, so the completion mask starts exactly where
generation will.

**Eval metrics come in two flavours.** The per-epoch `exact_match`/`bleu`/
`rouge_l`/`lexical_similarity` are teacher-forced (each token predicted from the
gold prefix) and therefore optimistic, which is why `metric_for_best_model` is
`eval_loss`. The reported numbers come from real generation via `Predictor`.
`preprocess_logits_for_metrics` argmaxes on the GPU first — otherwise the Trainer
accumulates `[batch, seq, vocab]` logits for the whole eval set (tens of GB of
host RAM with a 248k-token vocab). A metric that cannot be computed reports `NaN`,
not `0.0`: a zero is indistinguishable from a genuinely bad model.

**The validation split must be a real validation split.** `do_eval=true` with no
`validation_file` is an error rather than a silent fallback to test: early stopping
and `load_best_model_at_end` would otherwise select the checkpoint on the very
split the final report scores.

**Stop tokens come from the checkpoint.** Neither loading nor generation overwrites
`generation_config.eos_token_id` — the shipped list is wider than
`tokenizer.eos_token_id` (Qwen3 stops on `<|im_end|>` *and* `<|endoftext|>`,
Llama-3 on `<|eot_id|>` among others), and narrowing it lets generation run past
the token the chat template actually emits. Prompt truncation is left-sided for the
same reason: the tail of a rendered prompt *is* the generation prompt.

**Packing is off for SFT** and force-disabled for models with
`supports_packing: false`. TRL's bfd packing flattens a batch into one sequence
and relies on `cu_seq_lens`; Qwen3.5's Gated DeltaNet layers only honour that
inside the `flash-linear-attention` kernels, and the pure-torch fallback ignores
it.

## Memory optimisation

1. **4-bit quantization** (`qlora: true`) — roughly 75% less weight memory.
2. **LoRA** — only a small fraction of parameters get gradients.
3. **Gradient accumulation** — larger effective batch on a small GPU.
4. **Gradient checkpointing** — enabled by default (`use_reentrant: false`).
