"""Boilerplate stripping: universal patterns, corpus learning, and the safety guard."""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.env_setup import PROJECT_ROOT
from src.finetune.cpt.boilerplate import (
    build_boilerplate_index,
    deaccent,
    matches_pattern,
    normalise,
    strip_boilerplate,
    strip_corpus,
)

CORPUS_DIR = os.path.join(PROJECT_ROOT, "data", "report_dataset")
REPORTS = (
    sorted(
        os.path.join(CORPUS_DIR, name)
        for name in os.listdir(CORPUS_DIR)
        if name.endswith(".md")
    )
    if os.path.isdir(CORPUS_DIR)
    else []
)

LONG_PARAGRAPH = " ".join(f"Nội dung thực chất của báo cáo câu số {i}." for i in range(60))


# --- de-accenting -------------------------------------------------------------


def test_deaccent_handles_tone_marks_and_dstroke():
    assert deaccent("Cộng hòa xã hội") == "CONG HOA XA HOI"
    assert deaccent("Độc lập") == "DOC LAP"
    assert deaccent("đường") == "DUONG"


def test_deaccent_is_why_spelling_variants_both_match():
    """'HÒA' carries the tone on the O, 'HOÀ' on the A — both are in real use."""
    assert deaccent("CỘNG HÒA") == deaccent("CỘNG HOÀ")


# --- universal patterns -------------------------------------------------------


@pytest.mark.parametrize(
    "line,expected",
    [
        ("CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM", "republic"),
        ("CỘNG HOÀ XÃ HỘI CHỦ NGHĨA VIỆT NAM", "republic"),
        ("Độc lập - Tự do - Hạnh phúc", "motto"),
        ("Độc lập – Tự do – Hạnh phúc", "motto"),
        ("Hà Nội, ngày 05 tháng 5 năm 2025", "place_date"),
        ("Thành phố Hồ Chí Minh, ngày 1 tháng 12 năm 2024", "place_date"),
        ("Số: 316/BC-CP", "doc_number"),
        ("Số 45/BC-BKHĐT", "doc_number"),
        ("Nơi nhận:", "recipients"),
        ("- Lưu: VT, KTTH.", "archive"),
        ("TM. CHÍNH PHỦ", "signature_role"),
        ("KT. THỦ TƯỚNG CHÍNH PHỦ", "signature_role"),
        ("(Đã ký)", "signed_marker"),
        ("Ký bởi: Nguyễn Văn A", "digital_signature"),
        ("Thời gian ký: 05/05/2025", "digital_signature"),
        ("Signature Not Verified", "digital_signature"),
        ("./.", "end_marker"),
        ("Trang 3/12", "page_number"),
        ("7", "page_number"),
        ("- 7 -", "page_dashes"),
    ],
)
def test_patterns_match_administrative_frame(line, expected):
    assert matches_pattern(line) == expected


@pytest.mark.parametrize(
    "line",
    [
        "I. ĐÁNH GIÁ BỔ SUNG KẾT QUẢ NĂM 2024",
        "1. Kết quả đạt được những tháng đầu năm 2025",
        "a) Về kinh tế",
        "Doanh thu thuần đạt 1.245 tỷ đồng, tăng 12,3% so với cùng kỳ.",
        "Căn cứ Nghị định số 15/2020/NĐ-CP, đơn vị đã hoàn tất quyết toán.",
        "- Thực hiện việc tinh gọn, sắp xếp tổ chức bộ máy, bảo đảm hoạt động thông suốt.",
        "Số lượng doanh nghiệp thành lập mới tăng 9%.",
        "",
    ],
)
def test_patterns_leave_content_alone(line):
    assert matches_pattern(line) is None


# --- corpus learning ---------------------------------------------------------


def test_learns_lines_shared_across_documents():
    header = "SỞ TÀI CHÍNH TỈNH X"
    docs = [f"{header}\n\nNội dung riêng số {i}." for i in range(10)]
    learned = build_boilerplate_index(docs)
    assert normalise(header) in learned
    assert normalise("Nội dung riêng số 3.") not in learned


def test_learning_needs_enough_documents():
    """With 1-4 documents a ratio is meaningless, so nothing is learned."""
    docs = ["Tiêu đề chung\n\nNội dung một.", "Tiêu đề chung\n\nNội dung hai."]
    assert build_boilerplate_index(docs) == frozenset()


def test_learning_ignores_long_lines():
    """A repeated long paragraph is content or a duplicate chunk, not template."""
    docs = [f"{LONG_PARAGRAPH}\n\nRiêng {i}." for i in range(10)]
    learned = build_boilerplate_index(docs)
    assert normalise(LONG_PARAGRAPH) not in learned


def test_learning_counts_documents_not_occurrences():
    """A line repeated 50 times inside one report is not corpus template text."""
    docs = ["\n".join(["Lặp lại nhiều lần"] * 50)] + [f"Riêng {i}." for i in range(9)]
    learned = build_boilerplate_index(docs)
    assert normalise("Lặp lại nhiều lần") not in learned


def test_learning_respects_the_ratio():
    docs = [f"Chung\n\nRiêng {i}." for i in range(5)] + [f"Riêng {i}." for i in range(5)]
    assert normalise("Chung") not in build_boilerplate_index(docs, min_document_ratio=0.6)
    assert normalise("Chung") in build_boilerplate_index(docs, min_document_ratio=0.4)


# --- stripping one document ---------------------------------------------------


def test_strips_header_keeps_body():
    doc = "\n".join(
        [
            "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM",
            "Độc lập - Tự do - Hạnh phúc",
            "Hà Nội, ngày 05 tháng 5 năm 2025",
            "",
            "I. TỔNG QUAN",
            "",
            LONG_PARAGRAPH,
        ]
    )
    clean, reasons = strip_boilerplate(doc)
    assert "CỘNG HÒA" not in clean
    assert "Độc lập" not in clean
    assert "ngày 05 tháng 5" not in clean
    assert "I. TỔNG QUAN" in clean
    assert LONG_PARAGRAPH in clean
    assert set(reasons) == {"republic", "motto", "place_date"}


def test_strips_recipient_block_but_not_body_bullets():
    doc = "\n".join(
        [
            LONG_PARAGRAPH,
            "",
            "- Thực hiện việc tinh gọn, sắp xếp tổ chức bộ máy, bảo đảm hoạt động thông suốt "
            "và tổ chức lại đơn vị hành chính các cấp theo mô hình hai cấp.",
            "",
            "Nơi nhận:",
            "- Ban Bí thư;",
            "- Thủ tướng Chính phủ;",
            "- Lưu: VT, KTTH.",
        ]
    )
    clean, reasons = strip_boilerplate(doc)
    assert "Nơi nhận:" not in clean
    assert "Ban Bí thư" not in clean
    assert "Lưu: VT" not in clean
    assert "Thực hiện việc tinh gọn" in clean  # a long body bullet survives
    # All three bullets under "Nơi nhận:" are recipient entries, including the
    # archive line — inside the block it is consumed before pattern matching.
    assert reasons["recipient_entry"] == 3


def test_archive_line_is_stripped_outside_a_recipient_block():
    doc = "\n".join([LONG_PARAGRAPH, "", "- Lưu: VT, KTTH."])
    clean, reasons = strip_boilerplate(doc)
    assert "Lưu: VT" not in clean
    assert reasons["archive"] == 1


def test_strips_signature_block():
    doc = "\n".join([LONG_PARAGRAPH, "", "TM. CHÍNH PHỦ", "THỦ TƯỚNG", "(Đã ký)", "./."])
    clean, _ = strip_boilerplate(doc)
    assert "TM. CHÍNH PHỦ" not in clean
    assert "(Đã ký)" not in clean
    assert "./." not in clean


def test_drops_a_title_block_the_extractor_emitted_twice():
    title = "ĐÁNH GIÁ BỔ SUNG KẾT QUẢ THỰC HIỆN KẾ HOẠCH PHÁT TRIỂN KINH TẾ - XÃ HỘI NĂM 2024"
    assert len(title) > 60
    doc = "\n".join(["BÁO CÁO", title, "", "BÁO CÁO", title, "", LONG_PARAGRAPH])
    clean, reasons = strip_boilerplate(doc)
    assert clean.count(title) == 1
    assert clean.count("BÁO CÁO") == 1
    assert reasons["repeated_in_head"] == 2


def test_repeated_body_text_far_from_the_head_is_untouched():
    line = "Kính thưa Quốc hội!"
    doc = "\n".join([line] + [LONG_PARAGRAPH] * 12 + [line])
    clean, _ = strip_boilerplate(doc)
    assert clean.count(line) == 2


# --- the safety guard ---------------------------------------------------------


def test_guard_is_measured_in_characters_not_lines():
    """Regression: a line-based ratio made a 1%-of-text frame look like half the
    document, and disabled stripping exactly where it was needed."""
    frame = ["CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM", "Độc lập - Tự do - Hạnh phúc"]
    doc = "\n".join(frame + ["", LONG_PARAGRAPH])

    by_lines = len(frame) / (len(frame) + 1)
    assert by_lines > 0.25, "the frame is a majority of the lines"

    clean, reasons = strip_boilerplate(doc, max_strip_ratio=0.25)
    assert "CỘNG HÒA" not in clean, "characters, not lines, decide"
    assert "skipped_over_ratio" not in reasons


def test_guard_leaves_a_document_untouched_when_stripping_goes_too_far():
    doc = "\n".join(["CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM", "Ngắn."])
    clean, reasons = strip_boilerplate(doc, max_strip_ratio=0.25)
    assert clean == doc
    assert reasons["skipped_over_ratio"] == 1


def test_guard_can_be_relaxed():
    doc = "\n".join(["CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM", "Ngắn."])
    clean, reasons = strip_boilerplate(doc, max_strip_ratio=0.95)
    assert "CỘNG HÒA" not in clean
    assert "skipped_over_ratio" not in reasons


def test_patterns_can_be_turned_off():
    doc = "\n".join(["CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM", "", LONG_PARAGRAPH])
    clean, _ = strip_boilerplate(doc, use_patterns=False)
    assert "CỘNG HÒA" in clean


# --- whole corpus -------------------------------------------------------------


def test_strip_corpus_removes_the_shared_frame():
    frame_top = ["CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM", "Độc lập - Tự do - Hạnh phúc", "BỘ Y TẾ"]
    frame_bottom = ["Nơi nhận:", "- Lưu: VT."]
    documents = [
        {
            "filepath": f"r{i}.md",
            "content": "\n".join(frame_top + ["", f"1. Phần riêng {i}", "", LONG_PARAGRAPH, ""] + frame_bottom),
        }
        for i in range(10)
    ]
    cleaned = strip_corpus(documents)

    assert len(cleaned) == 10
    for doc in cleaned:
        assert "CỘNG HÒA" not in doc["content"]
        assert "BỘ Y TẾ" not in doc["content"]  # learned, not a built-in pattern
        assert "Nơi nhận" not in doc["content"]
        assert LONG_PARAGRAPH in doc["content"]


def test_strip_corpus_raises_if_nothing_survives():
    documents = [
        {"filepath": f"r{i}.md", "content": "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM"}
        for i in range(6)
    ]
    with pytest.raises(ValueError, match="removed every document"):
        strip_corpus(documents, max_strip_ratio=1.0)


# --- the real report ----------------------------------------------------------


@pytest.mark.skipif(not REPORTS, reason="no .md reports in data/report_dataset")
@pytest.mark.parametrize("path", REPORTS)
def test_real_report_frame_is_removed_and_content_kept(path):
    with open(path, encoding="utf-8") as f:
        doc = f.read().strip()

    clean, reasons = strip_boilerplate(doc, label=os.path.basename(path))
    assert "skipped_over_ratio" not in reasons

    # The frame goes.
    assert "CỘNG HÒA XÃ HỘI CHỦ NGHĨA" not in clean
    assert "Độc lập - Tự do - Hạnh phúc" not in clean

    # The substance stays: every section heading survives.
    for heading in ("I. ĐÁNH GIÁ", "II. VỀ TÌNH HÌNH", "III. NHIỆM VỤ", "a) Về kinh tế"):
        assert heading in clean, f"lost heading {heading!r}"

    # And stripping stays small on a real document.
    assert len(clean) > 0.9 * len(doc)
