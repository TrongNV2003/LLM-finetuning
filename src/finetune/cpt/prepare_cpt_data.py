"""Turn a directory of report .md files into chunked train/val JSON for CPT.

One .md file is one report document. Documents are chunked with the
structure-aware chunker in `chunking.py`, then split at *document* level so no
chunk of a training report leaks into validation.
"""

import os
import sys
import glob
import json
import random
import logging
from typing import Callable, Dict, List, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from src.env_setup import resolve_path  # noqa: E402  (sets env vars before torch)

from src.finetune.cpt.boilerplate import (
    DEFAULT_MAX_LINE_CHARS,
    DEFAULT_MAX_STRIP_RATIO,
    DEFAULT_MIN_DOCUMENT_RATIO,
    strip_corpus,
)
from src.finetune.cpt.chunking import chunk_document
from src.utils.config import resolve_max_length

logger = logging.getLogger(__name__)

# Reserve room for eos_token during training
EOS_RESERVE = 1


def reserve_eos(max_length: int) -> int:
    return max(1, max_length - EOS_RESERVE)


def tokenizer_counter(tokenizer) -> Callable[[str], int]:
    def count_tokens(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False))

    return count_tokens


def read_documents(source_dir: str) -> List[Dict]:
    """Read every .md under `source_dir`; one file is one report document."""
    md_files = sorted(glob.glob(os.path.join(source_dir, "**/*.md"), recursive=True))
    if not md_files:
        raise FileNotFoundError(f"No .md files found in {source_dir}")

    logger.info(f"Found {len(md_files)} .md file(s) in {source_dir}")

    documents = []
    for filepath in md_files:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read().strip()
        if content:
            documents.append({"filepath": filepath, "content": content})
        else:
            logger.warning(f"Skipping empty file: {filepath}")

    if not documents:
        raise ValueError(f"Every .md file in {source_dir} is empty")
    return documents


def _dedupe_key(text: str) -> str:
    """Whitespace-insensitive identity: two chunks differing only in blank lines
    are the same training signal."""
    return " ".join(text.split())


def deduplicate(chunks: List[Dict]) -> Tuple[List[Dict], int]:
    """Drop repeat chunks, keeping the first occurrence (so output stays stable).

    A corpus of same-template reports repeats boilerplate — the republic header,
    the signature block, shared legal appendices — once per document. Left in, the
    model sees those chunks hundreds of times per epoch and spends capacity
    memorising them instead of the domain.
    """
    seen = set()
    kept = []
    for chunk in chunks:
        key = _dedupe_key(chunk["text"])
        if key in seen:
            continue
        seen.add(key)
        kept.append(chunk)
    return kept, len(chunks) - len(kept)


def drop_seen_in_train(
    val_chunks: List[Dict], train_chunks: List[Dict]
) -> Tuple[List[Dict], int]:
    """Remove validation chunks whose text also occurs in training.

    The document-level split keeps chunks *of the same report* out of validation,
    but identical text living in two different reports slips through it. Scoring
    perplexity on chunks the model literally trained on makes the number
    optimistic for reasons that have nothing to do with domain adaptation.
    """
    train_keys = {_dedupe_key(chunk["text"]) for chunk in train_chunks}
    kept = [chunk for chunk in val_chunks if _dedupe_key(chunk["text"]) not in train_keys]
    return kept, len(val_chunks) - len(kept)


def split_two_ways(items: List, val_ratio: float, label: str) -> Tuple[List, List]:
    """Split `items` into (train, val), guaranteeing both sides are non-empty.

    `max(1, ...)` alone hands the only item to validation and leaves train empty —
    which is how the previous split broke on a one-document corpus. Capping the
    validation count at `len(items) - 1` prevents that at either level.
    """
    if len(items) < 2:
        raise ValueError(
            f"Need at least 2 {label} to build a train/validation split, got {len(items)}"
        )
    val_count = min(len(items) - 1, max(1, round(len(items) * val_ratio)))
    return items[val_count:], items[:val_count]


def prepare_cpt_data(
    source_dir: str,
    output_dir: str,
    count_tokens: Callable[[str], int],
    max_seq_length: int,
    val_ratio: float = 0.15,
    seed: int = 42,
    dedupe: bool = True,
    strip_boilerplate: bool = True,
    boilerplate_min_ratio: float = DEFAULT_MIN_DOCUMENT_RATIO,
    boilerplate_max_line_chars: int = DEFAULT_MAX_LINE_CHARS,
    boilerplate_max_strip_ratio: float = DEFAULT_MAX_STRIP_RATIO,
    train_filename: str = "train.json",
    val_filename: str = "val.json",
) -> Dict[str, str]:
    """Chunk `source_dir` into `output_dir`. Returns the written paths."""
    random.seed(seed)
    source_dir = resolve_path(source_dir)
    output_dir = resolve_path(output_dir)

    budget = reserve_eos(max_seq_length)
    logger.info(f"Chunking to <= {budget} tokens ({max_seq_length} minus EOS)")

    documents = read_documents(source_dir)

    if strip_boilerplate:
        documents = strip_corpus(
            documents,
            min_document_ratio=boilerplate_min_ratio,
            max_line_chars=boilerplate_max_line_chars,
            max_strip_ratio=boilerplate_max_strip_ratio,
        )

    def chunks_of(docs: List[Dict]) -> List[Dict]:
        out = []
        for doc in docs:
            chunks = chunk_document(doc["content"], count_tokens, budget)
            if not chunks:
                logger.warning(f"No chunks produced from {doc['filepath']}")
            out.extend({"text": chunk} for chunk in chunks)
        return out

    random.shuffle(documents)

    if len(documents) >= 2:
        train_docs, val_docs = split_two_ways(documents, val_ratio, "documents")
        logger.info(f"Document-level split: {len(train_docs)} train, {len(val_docs)} val")
        train_chunks = chunks_of(train_docs)
        val_chunks = chunks_of(val_docs)
        if not train_chunks or not val_chunks:
            raise ValueError(
                "The document-level split produced an empty side "
                f"({len(train_chunks)} train / {len(val_chunks)} val chunks) — "
                "check that the source documents are not near-empty."
            )
    else:
        logger.warning(
            "Only 1 document found -> splitting at chunk level. Validation will be far "
            "less independent than a document-level split; add more reports."
        )
        all_chunks = chunks_of(documents)
        random.shuffle(all_chunks)
        train_chunks, val_chunks = split_two_ways(all_chunks, val_ratio, "chunks")

    random.shuffle(train_chunks)
    random.shuffle(val_chunks)

    if dedupe:
        train_chunks, train_repeats = deduplicate(train_chunks)
        val_chunks, val_repeats = deduplicate(val_chunks)
        val_chunks, leaked = drop_seen_in_train(val_chunks, train_chunks)
        if train_repeats or val_repeats:
            logger.info(f"Dropped repeat chunks: {train_repeats} train, {val_repeats} val")
        if leaked:
            logger.info(
                f"Dropped {leaked} validation chunk(s) whose text also occurs in training "
                "(shared boilerplate across reports; a document-level split does not catch it)"
            )
        if not train_chunks or not val_chunks:
            raise ValueError(
                f"Deduplication emptied a side ({len(train_chunks)} train / {len(val_chunks)} "
                "val chunks) — the corpus is probably near-identical documents. Pass "
                "dedupe=false (prepare.dedupe=false) to keep the repeats."
            )

    train_tokens = sum(count_tokens(c["text"]) for c in train_chunks)
    val_tokens = sum(count_tokens(c["text"]) for c in val_chunks)
    logger.info(f"Chunks: {len(train_chunks)} train, {len(val_chunks)} val")
    logger.info(f"Tokens: {train_tokens:,} train, {val_tokens:,} val")
    logger.info(
        f"Avg tokens/chunk: {train_tokens // len(train_chunks)} train, "
        f"{val_tokens // len(val_chunks)} val"
    )

    os.makedirs(output_dir, exist_ok=True)
    paths = {
        "train": os.path.join(output_dir, train_filename),
        "validation": os.path.join(output_dir, val_filename),
    }
    for key, chunks in (("train", train_chunks), ("validation", val_chunks)):
        with open(paths[key], "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved: {paths[key]} ({len(chunks)} samples)")

    return paths


def prepare_from_config(cfg, tokenizer=None) -> None:
    """Chunk `dataset.source_dir` before training, when the config asks for it."""
    prepare_cfg = cfg.get("prepare") or {}
    if not prepare_cfg.get("auto", False):
        return

    train_file = resolve_path(cfg.dataset.train_file)
    val_file = resolve_path(cfg.dataset.validation_file)
    force = bool(prepare_cfg.get("force", False))

    if os.path.exists(train_file) and os.path.exists(val_file) and not force:
        logger.info(
            f"Prepared data already present ({train_file}), skipping the chunking step "
            "(prepare.force=true to re-chunk)"
        )
        return

    source_dir = cfg.dataset.get("source_dir")
    if not source_dir:
        raise ValueError("prepare.auto=true requires dataset.source_dir (the raw .md directory)")

    output_dir = os.path.dirname(train_file)
    if os.path.dirname(val_file) != output_dir:
        raise ValueError(
            "prepare.auto=true needs dataset.train_file and dataset.validation_file in the "
            f"same directory, got {output_dir} and {os.path.dirname(val_file)}"
        )

    if tokenizer is None:
        from src.models import load_tokenizer

        tokenizer = load_tokenizer(cfg)

    boilerplate_cfg = prepare_cfg.get("boilerplate") or {}

    logger.info(f"Preparing CPT data from {source_dir}" + (" (forced)" if force else ""))
    prepare_cpt_data(
        source_dir=source_dir,
        output_dir=output_dir,
        count_tokens=tokenizer_counter(tokenizer),
        max_seq_length=resolve_max_length(cfg),
        val_ratio=float(prepare_cfg.get("val_ratio", 0.15)),
        seed=int(cfg.get("seed", 42)),
        dedupe=bool(prepare_cfg.get("dedupe", True)),
        strip_boilerplate=bool(boilerplate_cfg.get("strip", True)),
        boilerplate_min_ratio=float(
            boilerplate_cfg.get("min_document_ratio", DEFAULT_MIN_DOCUMENT_RATIO)
        ),
        boilerplate_max_line_chars=int(
            boilerplate_cfg.get("max_line_chars", DEFAULT_MAX_LINE_CHARS)
        ),
        boilerplate_max_strip_ratio=float(
            boilerplate_cfg.get("max_strip_ratio", DEFAULT_MAX_STRIP_RATIO)
        ),
        train_filename=os.path.basename(train_file),
        val_filename=os.path.basename(val_file),
    )
