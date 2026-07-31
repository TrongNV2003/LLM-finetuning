"""The prepare step: split arithmetic, EOS reserve, and the real corpus end to end."""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

pytest.importorskip("transformers")
pytest.importorskip("omegaconf")  # prepare_cpt_data reads the config via src.utils.config

from src.env_setup import PROJECT_ROOT
from src.finetune.cpt.prepare_cpt_data import (
    deduplicate,
    drop_seen_in_train,
    prepare_cpt_data,
    read_documents,
    reserve_eos,
    split_two_ways,
)

CORPUS_DIR = os.path.join(PROJECT_ROOT, "data", "report_dataset")
HAS_CORPUS = os.path.isdir(CORPUS_DIR) and any(
    name.endswith(".md") for name in os.listdir(CORPUS_DIR)
)


def count_tokens(text: str) -> int:
    return len(text.split())


# --- split arithmetic ---------------------------------------------------------


def test_split_never_leaves_train_empty():
    """The bug this replaces: max(1, int(1 * 0.15)) handed the only item to val."""
    train, val = split_two_ways(list(range(2)), 0.15, "items")
    assert len(train) == 1 and len(val) == 1


@pytest.mark.parametrize("size", range(2, 40))
def test_split_both_sides_non_empty_at_every_size(size):
    train, val = split_two_ways(list(range(size)), 0.15, "items")
    assert train and val
    assert len(train) + len(val) == size
    assert not set(train) & set(val)


def test_split_rejects_a_single_item():
    with pytest.raises(ValueError, match="at least 2"):
        split_two_ways([1], 0.15, "documents")


def test_split_ratio_is_respected_when_there_is_room():
    train, val = split_two_ways(list(range(100)), 0.15, "items")
    assert len(val) == 15


def test_split_handles_an_extreme_ratio():
    train, val = split_two_ways(list(range(10)), 1.0, "items")
    assert len(train) == 1 and len(val) == 9


# --- EOS reserve --------------------------------------------------------------


def test_reserve_eos_leaves_room_for_the_token_trl_appends():
    assert reserve_eos(2048) == 2047
    assert reserve_eos(1) == 1  # never returns a useless budget


# --- deduplication ------------------------------------------------------------


def test_deduplicate_keeps_first_occurrence():
    chunks = [{"text": "A"}, {"text": "B"}, {"text": "A"}, {"text": "C"}, {"text": "B"}]
    kept, dropped = deduplicate(chunks)
    assert [c["text"] for c in kept] == ["A", "B", "C"]
    assert dropped == 2


def test_deduplicate_ignores_whitespace_differences():
    chunks = [{"text": "Báo cáo\n\nnội dung"}, {"text": "Báo cáo\nnội dung"}]
    kept, dropped = deduplicate(chunks)
    assert dropped == 1


def test_deduplicate_leaves_distinct_chunks_alone():
    chunks = [{"text": f"Nội dung số {i}"} for i in range(50)]
    kept, dropped = deduplicate(chunks)
    assert dropped == 0
    assert len(kept) == 50


def test_drop_seen_in_train_closes_the_boilerplate_leak():
    """Identical text in two different reports slips past a document-level split."""
    header = "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM\nĐộc lập - Tự do - Hạnh phúc"
    train = [{"text": header}, {"text": "Riêng của báo cáo train"}]
    val = [{"text": header}, {"text": "Riêng của báo cáo val"}]
    kept, dropped = drop_seen_in_train(val, train)
    assert dropped == 1
    assert [c["text"] for c in kept] == ["Riêng của báo cáo val"]


def test_shared_boilerplate_never_reaches_validation(tmp_path):
    """End to end: every report opens with the same header block."""
    header = "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM\nĐộc lập - Tự do - Hạnh phúc\nQUỐC HỘI"
    source = tmp_path / "src"
    source.mkdir()
    for i in range(20):
        body = " ".join(f"Nội dung riêng {i} câu {j}." for j in range(40))
        (source / f"r{i}.md").write_text(f"{header}\n\n1. Phần một\n\n{body}", encoding="utf-8")

    paths = prepare_cpt_data(
        source_dir=str(source),
        output_dir=str(tmp_path / "out"),
        count_tokens=count_tokens,
        max_seq_length=64,  # small, so the header lands in a chunk of its own
        seed=5,
    )
    train = json.loads(open(paths["train"], encoding="utf-8").read())
    val = json.loads(open(paths["validation"], encoding="utf-8").read())

    train_texts = {" ".join(c["text"].split()) for c in train}
    val_texts = {" ".join(c["text"].split()) for c in val}
    assert not train_texts & val_texts
    # And the header is not repeated 20 times on the training side.
    assert sum(1 for t in train_texts if "CỘNG HÒA" in t) <= 1


def test_dedupe_can_be_turned_off(tmp_path):
    header = "Tiêu đề dùng chung của mọi báo cáo trong tập dữ liệu này"
    source = tmp_path / "src"
    source.mkdir()
    for i in range(8):
        (source / f"r{i}.md").write_text(f"{header}\n\nNội dung {i}.", encoding="utf-8")

    kept_paths = prepare_cpt_data(
        source_dir=str(source),
        output_dir=str(tmp_path / "keep"),
        count_tokens=count_tokens,
        max_seq_length=2048,
        seed=1,
        dedupe=False,
    )
    rows = json.loads(open(kept_paths["train"], encoding="utf-8").read())
    assert sum(1 for r in rows if header in r["text"]) > 1


# --- boilerplate stripping, through prepare -----------------------------------


def _write_framed_report(path, index):
    """A report wrapped in the usual Vietnamese administrative frame."""
    body = " ".join(f"Nội dung riêng của báo cáo {index} câu {j}." for j in range(120))
    path.write_text(
        "\n".join(
            [
                "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM",
                "Độc lập - Tự do - Hạnh phúc",
                f"Hà Nội, ngày {1 + index % 28} tháng 5 năm 2025",
                "BỘ TÀI CHÍNH",
                "",
                f"1. Nội dung báo cáo {index}",
                "",
                body,
                "",
                "Nơi nhận:",
                "- Ban Bí thư;",
                "- Lưu: VT.",
            ]
        ),
        encoding="utf-8",
    )


def test_frame_is_gone_from_the_prepared_chunks(tmp_path):
    source = tmp_path / "src"
    source.mkdir()
    for i in range(12):
        _write_framed_report(source / f"r{i}.md", i)

    paths = prepare_cpt_data(
        source_dir=str(source),
        output_dir=str(tmp_path / "out"),
        count_tokens=count_tokens,
        max_seq_length=2048,
        seed=3,
    )
    rows = json.loads(open(paths["train"], encoding="utf-8").read())
    rows += json.loads(open(paths["validation"], encoding="utf-8").read())
    blob = "\n".join(row["text"] for row in rows)

    for frame in ("CỘNG HÒA", "Độc lập - Tự do", "tháng 5 năm 2025", "Nơi nhận", "Lưu: VT"):
        assert frame not in blob, f"{frame!r} survived into the training data"
    assert "BỘ TÀI CHÍNH" not in blob  # learned from the corpus, not a built-in pattern
    assert "Nội dung riêng của báo cáo" in blob


def test_boilerplate_stripping_can_be_turned_off(tmp_path):
    source = tmp_path / "src"
    source.mkdir()
    for i in range(12):
        _write_framed_report(source / f"r{i}.md", i)

    paths = prepare_cpt_data(
        source_dir=str(source),
        output_dir=str(tmp_path / "out"),
        count_tokens=count_tokens,
        max_seq_length=2048,
        seed=3,
        strip_boilerplate=False,
    )
    rows = json.loads(open(paths["train"], encoding="utf-8").read())
    assert any("CỘNG HÒA" in row["text"] for row in rows)


# --- corpus reading -----------------------------------------------------------


def test_read_documents_rejects_a_directory_with_no_markdown(tmp_path):
    with pytest.raises(FileNotFoundError, match="No .md files"):
        read_documents(str(tmp_path))


def test_read_documents_rejects_only_empty_files(tmp_path):
    (tmp_path / "a.md").write_text("   \n\n")
    with pytest.raises(ValueError, match="empty"):
        read_documents(str(tmp_path))


def test_read_documents_treats_one_file_as_one_document(tmp_path):
    (tmp_path / "r1.md").write_text("Báo cáo một.")
    (tmp_path / "r2.md").write_text("Báo cáo hai.")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "r3.md").write_text("Báo cáo ba.")
    docs = read_documents(str(tmp_path))
    assert len(docs) == 3


# --- end to end ---------------------------------------------------------------


def _write_report(path, index, sections=4):
    body = [f"BÁO CÁO SỐ {index}", ""]
    for section in range(1, sections + 1):
        body += [
            f"{section}. Nội dung phần {section}",
            "",
            " ".join(f"Câu số {i} của báo cáo {index} phần {section}." for i in range(30)),
            "",
        ]
    path.write_text("\n".join(body), encoding="utf-8")


def test_multi_document_corpus_splits_at_document_level(tmp_path):
    source = tmp_path / "src"
    source.mkdir()
    for i in range(10):
        _write_report(source / f"report_{i}.md", i)

    out = tmp_path / "out"
    paths = prepare_cpt_data(
        source_dir=str(source),
        output_dir=str(out),
        count_tokens=count_tokens,
        max_seq_length=200,
        val_ratio=0.2,
        seed=7,
    )

    train = json.loads(open(paths["train"], encoding="utf-8").read())
    val = json.loads(open(paths["validation"], encoding="utf-8").read())
    assert train and val
    assert all(set(row) == {"text"} for row in train + val)
    assert all(count_tokens(row["text"]) <= reserve_eos(200) for row in train + val)

    # No report contributes chunks to both sides: that is the point of splitting
    # at document level rather than chunk level.
    def report_ids(rows):
        return {int(row["text"].split("báo cáo ")[1].split()[0]) for row in rows if "báo cáo " in row["text"]}

    assert not report_ids(train) & report_ids(val)


def test_single_document_corpus_falls_back_to_chunk_level(tmp_path):
    source = tmp_path / "src"
    source.mkdir()
    _write_report(source / "only.md", 0, sections=6)

    out = tmp_path / "out"
    paths = prepare_cpt_data(
        source_dir=str(source),
        output_dir=str(out),
        count_tokens=count_tokens,
        max_seq_length=100,
        seed=7,
    )
    train = json.loads(open(paths["train"], encoding="utf-8").read())
    val = json.loads(open(paths["validation"], encoding="utf-8").read())
    assert train and val


def test_single_tiny_document_fails_loudly(tmp_path):
    """One document that yields one chunk cannot be split; say so instead of
    producing an empty train set."""
    source = tmp_path / "src"
    source.mkdir()
    (source / "tiny.md").write_text("Một câu duy nhất.", encoding="utf-8")

    with pytest.raises(ValueError, match="at least 2 chunks"):
        prepare_cpt_data(
            source_dir=str(source),
            output_dir=str(tmp_path / "out"),
            count_tokens=count_tokens,
            max_seq_length=2048,
        )


def test_output_is_deterministic_for_a_given_seed(tmp_path):
    source = tmp_path / "src"
    source.mkdir()
    for i in range(6):
        _write_report(source / f"r{i}.md", i)

    runs = []
    for run in range(2):
        paths = prepare_cpt_data(
            source_dir=str(source),
            output_dir=str(tmp_path / f"out{run}"),
            count_tokens=count_tokens,
            max_seq_length=200,
            seed=123,
        )
        runs.append(open(paths["train"], encoding="utf-8").read())
    assert runs[0] == runs[1]


@pytest.mark.skipif(not HAS_CORPUS, reason="no .md reports in data/report_dataset")
def test_real_corpus_prepares_end_to_end(tmp_path):
    paths = prepare_cpt_data(
        source_dir=CORPUS_DIR,
        output_dir=str(tmp_path),
        count_tokens=count_tokens,
        max_seq_length=2048,
        seed=42,
    )
    train = json.loads(open(paths["train"], encoding="utf-8").read())
    val = json.loads(open(paths["validation"], encoding="utf-8").read())

    assert train and val
    budget = reserve_eos(2048)
    for row in train + val:
        assert count_tokens(row["text"]) <= budget
        assert row["text"].strip()
