"""Chunker invariants, checked against the real report corpus.

Runs with stdlib only — the token counter is a stub, because what needs testing is
the chunking algorithm, not the tokenizer.
"""

import os
import re
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.env_setup import PROJECT_ROOT  # noqa: E402
from src.finetune.cpt.chunking import (  # noqa: E402
    chunk_document,
    detect_heading,
    parse_blocks,
    split_oversized,
)

CORPUS_DIR = os.path.join(PROJECT_ROOT, "data", "report_dataset")


def count_tokens(text: str) -> int:
    """Stub tokenizer: roughly one token per whitespace-separated word."""
    return len(text.split())


def no_whitespace(text: str) -> str:
    return re.sub(r"\s", "", text)


def corpus_documents():
    if not os.path.isdir(CORPUS_DIR):
        return []
    paths = sorted(
        os.path.join(root, name)
        for root, _, names in os.walk(CORPUS_DIR)
        for name in names
        if name.endswith(".md")
    )
    return paths


REPORTS = corpus_documents()


# --- heading detection ---------------------------------------------------------


@pytest.mark.parametrize(
    "line,expected",
    [
        ("I. ĐÁNH GIÁ BỔ SUNG KẾT QUẢ NĂM 2024", "roman"),
        ("III. NHIỆM VỤ VÀ GIẢI PHÁP CHỦ YẾU THỜI GIAN TỚI", "roman"),
        ("1. Kết quả đạt được những tháng đầu năm 2025", "arabic"),
        ("a) Về kinh tế", "lower_alpha"),
        ("d) Về quốc phòng, an ninh, đối ngoại", "lower_alpha"),
        ("PHẦN II. Đánh giá chung", "phan"),
        ("Chương I. Quy định chung", "chuong"),
        ("MỤC 2. Nguyên tắc áp dụng", "muc"),
        ("2.1. Chỉ tiêu tăng trưởng", "decimal"),
        ("2.1.3. Chi tiết theo vùng", "sub_decimal"),
        ("B. Kiến nghị", "upper_alpha"),
        ("## Tổng quan", "markdown_2"),
    ],
)
def test_detect_heading_recognises_vietnamese_numbering(line, expected):
    assert detect_heading(line) == expected


@pytest.mark.parametrize(
    "line",
    [
        "Doanh thu thuần đạt 1.245 tỷ đồng, tăng 12,3% so với cùng kỳ.",
        "Chi nhánh TP. Hồ Chí Minh dẫn đầu với 412 tỷ đồng.",
        "Căn cứ Nghị định số 15/2020/NĐ-CP, đơn vị đã hoàn tất quyết toán.",
        "| Chỉ tiêu | Kế hoạch | Thực hiện |",
        "",
        "   ",
    ],
)
def test_detect_heading_rejects_body_text(line):
    assert detect_heading(line) is None


def test_long_numbered_line_is_not_a_heading():
    """A paragraph opening with '1. ' must not be mistaken for a section heading."""
    line = "1. " + "Hỗ trợ người có công với cách mạng bốn nghìn bốn trăm căn nhà. " * 5
    assert len(line) > 200
    assert detect_heading(line) is None


# --- structure parsing --------------------------------------------------------


def test_parse_blocks_separates_headings_tables_and_paragraphs():
    doc = "\n".join(
        [
            "I. TỔNG QUAN",
            "",
            "Doanh thu đạt 1.245 tỷ đồng.",
            "Tăng 12,3% so với cùng kỳ.",
            "",
            "| Chỉ tiêu | KH | TH |",
            "| --- | --- | --- |",
            "| Doanh thu | 1.200 | 1.245 |",
            "",
            "1. Kiến nghị",
            "",
            "Rà soát định mức tồn kho.",
        ]
    )
    blocks = parse_blocks(doc.split("\n"))
    assert [b.kind for b in blocks] == [
        "heading",
        "paragraph",
        "table",
        "heading",
        "paragraph",
    ]
    # Levels are normalised to the depths this document actually uses.
    assert [b.level for b in blocks if b.kind == "heading"] == [1, 2]


def test_table_rows_stay_in_one_block():
    doc = "| a | b |\n| --- | --- |\n| 1 | 2 |\n| 3 | 4 |"
    blocks = parse_blocks(doc.split("\n"))
    assert len(blocks) == 1
    assert blocks[0].kind == "table"
    assert (blocks[0].start, blocks[0].end) == (0, 3)


# --- oversized splitting ------------------------------------------------------


def test_split_oversized_loses_no_characters():
    """A plain str.split('. ') would drop the period at every boundary."""
    text = " ".join(f"Câu số {i} của đoạn văn dài." for i in range(200))
    pieces = split_oversized(text, count_tokens, 50)
    assert len(pieces) > 1
    assert all(count_tokens(p) <= 50 for p in pieces)
    assert no_whitespace("".join(pieces)) == no_whitespace(text)


def test_split_oversized_handles_a_table_with_no_sentence_punctuation():
    """The old sentence chunker emitted this whole table as one over-limit chunk."""
    table = "\n".join(f"| Khoản mục số {i} của đơn vị | {i * 10} | {i * 11} |" for i in range(1, 60))
    pieces = split_oversized(table, count_tokens, 60)
    assert all(count_tokens(p) <= 60 for p in pieces)
    assert no_whitespace("".join(pieces)) == no_whitespace(table)


def test_split_oversized_survives_a_run_with_no_separators():
    text = "x" * 5000
    pieces = split_oversized(text, lambda t: len(t), 100)
    assert all(len(p) <= 100 for p in pieces)
    assert "".join(pieces) == text


# --- end-to-end invariants on the real corpus ---------------------------------


@pytest.mark.skipif(not REPORTS, reason="no .md reports in data/report_dataset")
@pytest.mark.parametrize("path", REPORTS)
@pytest.mark.parametrize("budget", [2047, 1024, 512, 256, 128, 64])
def test_real_report_chunking_invariants(path, budget):
    with open(path, encoding="utf-8") as f:
        doc = f.read().strip()

    chunks = chunk_document(doc, count_tokens, budget)
    assert chunks, f"{path} produced no chunks"

    # 1. Nothing exceeds the budget, so the trainer never truncates a chunk.
    oversized = [c for c in chunks if count_tokens(c) > budget]
    assert not oversized, f"{len(oversized)} chunk(s) over {budget} tokens"

    # 2. Every character of content survives, in order.
    assert no_whitespace("".join(chunks)) == no_whitespace(doc)

    # 3. No chunk is nothing but headings — that would be an empty training sample.
    for chunk in chunks:
        lines = [line for line in chunk.split("\n") if line.strip()]
        assert not all(detect_heading(line) for line in lines), f"heading-only chunk: {chunk[:80]!r}"

    # 4. No heading is severed from its title (what `(?<=[.!?])\s+` used to do:
    #    'II.' on one side, 'VỀ TÌNH HÌNH…' on the other).
    for chunk in chunks:
        for line in chunk.split("\n"):
            assert not re.fullmatch(r"\s*([IVX]+|\d+|[a-z])\s*[.)]\s*", line), (
                f"severed heading marker: {line!r}"
            )


@pytest.mark.skipif(not REPORTS, reason="no .md reports in data/report_dataset")
@pytest.mark.parametrize("path", REPORTS)
def test_real_report_keeps_document_structure(path):
    """The sentence chunker destroyed 286 of 340 newlines on this document."""
    with open(path, encoding="utf-8") as f:
        doc = f.read().strip()

    chunks = chunk_document(doc, count_tokens, 2047)
    kept = sum(c.count("\n") for c in chunks)
    original = doc.count("\n")
    # Only blank lines at chunk boundaries are dropped.
    assert kept >= original * 0.9, f"kept {kept}/{original} newlines"


def test_chunk_document_rejects_a_useless_budget():
    with pytest.raises(ValueError, match="max_tokens"):
        chunk_document("nội dung", count_tokens, 0)


def test_empty_document_yields_no_chunks():
    assert chunk_document("", count_tokens, 100) == []
    assert chunk_document("\n\n   \n", count_tokens, 100) == []
