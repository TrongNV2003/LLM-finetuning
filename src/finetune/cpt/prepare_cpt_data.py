import os
import sys
import json
import glob
import random
import argparse

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.append(project_root)

from loguru import logger
from transformers import AutoTokenizer


def read_md_file(filepath: str) -> str:
    """Read a markdown file and return its content."""
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read().strip()


def chunk_text_by_sentences(text: str, tokenizer, max_seq_length: int, overlap_sentences: int = 2):
    """Chunk text into pieces that fit within max_seq_length tokens.
    
    Splits by sentences and groups them into chunks, with optional
    sentence overlap between consecutive chunks for context continuity.
    
    Args:
        text: Input text to chunk.
        tokenizer: HuggingFace tokenizer for counting tokens.
        max_seq_length: Maximum number of tokens per chunk.
        overlap_sentences: Number of sentences to overlap between chunks.
    
    Returns:
        List of text chunks, each fitting within max_seq_length tokens.
    """
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if not sentences:
        return []
    
    chunks = []
    start_idx = 0
    
    while start_idx < len(sentences):
        current_chunk_sentences = []
        current_tokens = 0
        idx = start_idx
        
        while idx < len(sentences):
            sentence = sentences[idx]
            sentence_tokens = len(tokenizer.encode(sentence, add_special_tokens=False))
            
            # Check if adding this sentence would exceed the limit
            if current_tokens + sentence_tokens > max_seq_length and current_chunk_sentences:
                break
            
            current_chunk_sentences.append(sentence)
            current_tokens += sentence_tokens
            idx += 1
        
        if current_chunk_sentences:
            chunk_text = " ".join(current_chunk_sentences)
            chunks.append(chunk_text)
        
        # Move forward, accounting for overlap
        sentences_added = len(current_chunk_sentences)
        if idx >= len(sentences):
            break
        
        # Step forward by (sentences_added - overlap), but at least 1
        step = max(1, sentences_added - overlap_sentences)
        start_idx += step
    
    return chunks


def prepare_cpt_data(
    source_dir: str,
    output_dir: str,
    model_name: str,
    max_seq_length: int = 2048,
    val_ratio: float = 0.15,
    overlap_sentences: int = 2,
    seed: int = 42,
    token: str = None,
):
    """Read .md files, chunk them, split into train/val, and save as JSON.
    
    Args:
        source_dir: Directory containing .md files.
        output_dir: Directory to save train.json and val.json.
        model_name: HuggingFace model name for tokenizer.
        max_seq_length: Maximum tokens per chunk.
        val_ratio: Fraction of documents for validation.
        overlap_sentences: Sentence overlap between chunks.
        seed: Random seed for reproducibility.
        token: HuggingFace token for gated models.
    """
    random.seed(seed)
    
    # Find all .md files
    md_files = sorted(glob.glob(os.path.join(source_dir, "**/*.md"), recursive=True))
    
    if not md_files:
        logger.error(f"No .md files found in {source_dir}")
        return
    
    logger.info(f"Found {len(md_files)} .md files in {source_dir}")
    
    # Load tokenizer for counting tokens
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token, trust_remote_code=True)
    
    # Read all documents
    documents = []
    for filepath in md_files:
        content = read_md_file(filepath)
        if content:
            documents.append({
                "filepath": filepath,
                "content": content,
            })
    
    logger.info(f"Read {len(documents)} non-empty documents")
    
    # Shuffle and split at document level
    random.shuffle(documents)
    val_count = max(1, int(len(documents) * val_ratio))
    val_docs = documents[:val_count]
    train_docs = documents[val_count:]
    
    logger.info(f"Split: {len(train_docs)} train docs, {len(val_docs)} val docs")
    
    # Chunk documents
    train_chunks = []
    val_chunks = []
    
    for doc in train_docs:
        chunks = chunk_text_by_sentences(
            doc["content"], tokenizer, max_seq_length, overlap_sentences
        )
        for chunk in chunks:
            train_chunks.append({"text": chunk})
    
    for doc in val_docs:
        chunks = chunk_text_by_sentences(
            doc["content"], tokenizer, max_seq_length, overlap_sentences
        )
        for chunk in chunks:
            val_chunks.append({"text": chunk})
    
    # Shuffle chunks
    random.shuffle(train_chunks)
    random.shuffle(val_chunks)
    
    logger.info(f"Chunks: {len(train_chunks)} train, {len(val_chunks)} val")
    
    # Calculate token statistics
    train_tokens = sum(len(tokenizer.encode(c["text"], add_special_tokens=False)) for c in train_chunks)
    val_tokens = sum(len(tokenizer.encode(c["text"], add_special_tokens=False)) for c in val_chunks)
    
    logger.info(f"Total tokens: {train_tokens:,} train, {val_tokens:,} val")
    logger.info(f"Avg tokens/chunk: {train_tokens // max(1, len(train_chunks))} train, "
                f"{val_tokens // max(1, len(val_chunks))} val")
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    
    train_path = os.path.join(output_dir, "train.json")
    val_path = os.path.join(output_dir, "val.json")
    
    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_chunks, f, ensure_ascii=False, indent=2)
    
    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(val_chunks, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Saved: {train_path} ({len(train_chunks)} samples)")
    logger.info(f"Saved: {val_path} ({len(val_chunks)} samples)")


def main():
    parser = argparse.ArgumentParser(description="Prepare CPT data from .md files")
    parser.add_argument("--source_dir", type=str, required=True,
                        help="Directory containing .md files")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for JSON files (default: same as source_dir)")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3.5-0.8B",
                        help="Model name for tokenizer")
    parser.add_argument("--max_seq_length", type=int, default=2048,
                        help="Maximum sequence length in tokens")
    parser.add_argument("--val_ratio", type=float, default=0.15,
                        help="Validation split ratio")
    parser.add_argument("--overlap_sentences", type=int, default=2,
                        help="Number of sentences to overlap between chunks")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--token", type=str, default=None,
                        help="HuggingFace token for gated models")
    
    args = parser.parse_args()
    
    output_dir = args.output_dir or args.source_dir
    
    prepare_cpt_data(
        source_dir=args.source_dir,
        output_dir=output_dir,
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        val_ratio=args.val_ratio,
        overlap_sentences=args.overlap_sentences,
        seed=args.seed,
        token=args.token,
    )


if __name__ == "__main__":
    main()
