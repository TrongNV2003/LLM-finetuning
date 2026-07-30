"""Strip administrative boilerplate from report documents before chunking."""

import logging
import re
import unicodedata
from collections import Counter
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Set, Tuple

logger = logging.getLogger(__name__)

DEFAULT_MAX_LINE_CHARS = 120
DEFAULT_MIN_DOCUMENT_RATIO = 0.5
MIN_DOCUMENTS_TO_LEARN = 5
DEFAULT_MAX_STRIP_RATIO = 0.25
HEAD_WINDOW_LINES = 40
PATTERNS: Sequence[Tuple[str, "re.Pattern"]] = (
    # "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM" and the motto line under it.
    ("republic", re.compile(r"^CONG\s+HOA\s+XA\s+HOI\s+CHU\s+NGHIA")),
    ("motto", re.compile(r"^DOC\s+LAP\s*[-–—]\s*TU\s+DO\s*[-–—]\s*HANH\s+PHUC")),
    # "Hà Nội, ngày 05 tháng 5 năm 2025"
    ("place_date", re.compile(r"^.{0,40},\s*NGAY\s+\d{1,2}\s+THANG\s+\d{1,2}\s+NAM\s+\d{4}$")),
    # "Số: 316/BC-CP", "Số 316/BC-CP"
    ("doc_number", re.compile(r"^SO\s*:?\s*\d+\s*/[\w\-./]+$")),
    # Recipient list header, and "Lưu: VT" style archive lines.
    ("recipients", re.compile(r"^NOI\s+NHAN\s*:$")),
    ("archive", re.compile(r"^[-–—+*]?\s*LUU\s*:")),
    # Signature block: "TM. CHÍNH PHỦ", "KT. THỦ TƯỚNG", "TL. BỘ TRƯỞNG", "(Đã ký)".
    ("signature_role", re.compile(r"^(TM|KT|TL|TUQ)\s*\.\s*\S")),
    ("signed_marker", re.compile(r"^\(?\s*DA\s+KY\s*\)?\.?$")),
    # Digital-signature metadata left behind by signing tools (.signed.md files).
    (
        "digital_signature",
        re.compile(r"^(KY\s+BOI|KY\s+NGAY|THOI\s+GIAN\s+KY|SIGNATURE\s+NOT\s+VERIFIED)\s*:?"),
    ),
    # "./." — the Vietnamese official end-of-document marker, alone on a line.
    ("end_marker", re.compile(r"^\.\s*/\s*\.$")),
    # Page furniture: "Trang 3/12", "- 3 -", a bare number.
    ("page_number", re.compile(r"^(TRANG\s+)?\d+\s*(/\s*\d+)?$")),
    ("page_dashes", re.compile(r"^[-–—]\s*\d+\s*[-–—]$")),
)

_DEACCENT_MAP = str.maketrans({"đ": "d", "Đ": "D"})

# Bullet lines directly under "Nơi nhận:" are recipients, not content.
_BULLET = re.compile(r"^[-–—+*]\s*\S")


def normalise(line: str) -> str:
    """Whitespace- and case-insensitive identity for a line."""
    return unicodedata.normalize("NFC", " ".join(line.split())).casefold()


def deaccent(text: str) -> str:
    """Upper-case, diacritic-free form used for pattern matching only."""
    folded = unicodedata.normalize("NFD", text.translate(_DEACCENT_MAP))
    return "".join(c for c in folded if not unicodedata.combining(c)).upper()


def matches_pattern(line: str) -> Optional[str]:
    """Name of the boilerplate pattern this line matches, or None."""
    stripped = " ".join(line.split())
    if not stripped:
        return None
    candidate = deaccent(stripped)
    for name, pattern in PATTERNS:
        if pattern.match(candidate):
            return name
    return None


def build_boilerplate_index(
    contents: Iterable[str],
    min_document_ratio: float = DEFAULT_MIN_DOCUMENT_RATIO,
    max_line_chars: int = DEFAULT_MAX_LINE_CHARS,
) -> FrozenSet[str]:
    """Learn this corpus's template lines: short lines recurring across documents.

    Counted once per document, not per occurrence, so a line repeated many times
    inside one report does not look like template text.
    """
    contents = list(contents)
    if len(contents) < MIN_DOCUMENTS_TO_LEARN:
        logger.info(
            f"Only {len(contents)} document(s): skipping corpus-learned boilerplate "
            f"(needs >= {MIN_DOCUMENTS_TO_LEARN}); the universal patterns still apply"
        )
        return frozenset()

    document_counts: Counter = Counter()
    for content in contents:
        seen = {
            normalise(line)
            for line in content.split("\n")
            if line.strip() and len(line.strip()) <= max_line_chars
        }
        document_counts.update(seen)

    threshold = max(2, int(len(contents) * min_document_ratio))
    learned = frozenset(key for key, count in document_counts.items() if count >= threshold)

    if learned:
        sample = sorted(learned, key=len)[:8]
        logger.info(
            f"Learned {len(learned)} template line(s) present in >= {threshold}/"
            f"{len(contents)} documents. Examples: {sample}"
        )
    else:
        logger.info("No corpus-wide template lines found above the threshold")
    return learned


def _classify(
    lines: Sequence[str],
    learned: FrozenSet[str],
    use_patterns: bool,
    max_line_chars: int,
) -> Tuple[List[bool], Counter]:
    """Mark each line for removal, and count why."""
    drop = [False] * len(lines)
    reasons: Counter = Counter()
    in_recipient_block = False
    head_seen: Set[str] = set()

    for index, line in enumerate(lines):
        stripped = line.strip()

        if not stripped:
            continue

        short = len(stripped) <= max_line_chars

        if in_recipient_block:
            if short and _BULLET.match(stripped):
                drop[index] = True
                reasons["recipient_entry"] += 1
                continue
            in_recipient_block = False

        pattern = matches_pattern(line) if use_patterns else None
        if pattern:
            drop[index] = True
            reasons[pattern] += 1
            if pattern == "recipients":
                in_recipient_block = True
            continue

        if short and normalise(line) in learned:
            drop[index] = True
            reasons["corpus_template"] += 1
            continue

        if index < HEAD_WINDOW_LINES:
            key = normalise(line)
            if key in head_seen:
                drop[index] = True
                reasons["repeated_in_head"] += 1
                continue
            head_seen.add(key)

    return drop, reasons


def _collapse_blank_runs(lines: List[str]) -> List[str]:
    out: List[str] = []
    blanks = 0
    for line in lines:
        if line.strip():
            blanks = 0
            out.append(line)
        else:
            blanks += 1
            if blanks <= 1:
                out.append(line)
    return out


def strip_boilerplate(
    text: str,
    learned: FrozenSet[str] = frozenset(),
    use_patterns: bool = True,
    max_line_chars: int = DEFAULT_MAX_LINE_CHARS,
    max_strip_ratio: float = DEFAULT_MAX_STRIP_RATIO,
    label: str = "document",
) -> Tuple[str, Counter]:
    """Remove boilerplate lines from one document.

    Returns the cleaned text and a count per reason. If the removal would exceed
    `max_strip_ratio` of the document's characters, the document is returned
    untouched: that much matching means the detectors are wrong about this corpus,
    and deleting a quarter of a report is worse than leaving its header in.
    """
    lines = text.split("\n")
    drop, reasons = _classify(lines, learned, use_patterns, max_line_chars)

    if not any(drop):
        return text, reasons

    total_chars = sum(len(line.strip()) for line in lines)
    dropped_chars = sum(len(line.strip()) for line, remove in zip(lines, drop) if remove)
    if total_chars and dropped_chars / total_chars > max_strip_ratio:
        logger.debug(
            f"{label}: would remove {dropped_chars}/{total_chars} chars "
            f"(> {max_strip_ratio:.0%}) — left untouched"
        )
        return text, Counter({"skipped_over_ratio": 1})

    kept = _collapse_blank_runs([line for line, remove in zip(lines, drop) if not remove])
    return "\n".join(kept).strip(), reasons


def strip_corpus(
    documents: List[Dict],
    min_document_ratio: float = DEFAULT_MIN_DOCUMENT_RATIO,
    max_line_chars: int = DEFAULT_MAX_LINE_CHARS,
    max_strip_ratio: float = DEFAULT_MAX_STRIP_RATIO,
    use_patterns: bool = True,
) -> List[Dict]:
    """Strip boilerplate across a corpus of `{"filepath", "content"}` documents.

    Learning runs over the whole corpus first, so per-document stripping knows
    what this template looks like.
    """
    learned = build_boilerplate_index(
        (doc["content"] for doc in documents),
        min_document_ratio=min_document_ratio,
        max_line_chars=max_line_chars,
    )

    totals: Counter = Counter()
    before = sum(len(doc["content"]) for doc in documents)
    cleaned = []
    for doc in documents:
        content, reasons = strip_boilerplate(
            doc["content"],
            learned=learned,
            use_patterns=use_patterns,
            max_line_chars=max_line_chars,
            max_strip_ratio=max_strip_ratio,
            label=doc["filepath"],
        )
        totals.update(reasons)
        if content:
            cleaned.append({**doc, "content": content})
        else:
            logger.warning(f"{doc['filepath']}: nothing left after stripping, dropping it")

    after = sum(len(doc["content"]) for doc in cleaned)
    skipped = totals.pop("skipped_over_ratio", 0)

    if totals:
        logger.info(f"Boilerplate removed by reason: {dict(totals.most_common())}")
        logger.info(
            f"Corpus size {before:,} -> {after:,} chars "
            f"({100 * (before - after) / max(1, before):.1f}% removed)"
        )
    else:
        logger.info("No boilerplate matched")

    # One aggregate line, not one per document: a mis-tuned threshold would
    # otherwise bury the run in a thousand identical warnings.
    if skipped:
        logger.warning(
            f"{skipped}/{len(documents)} document(s) left untouched because stripping would "
            f"have removed over {max_strip_ratio:.0%} of their characters. Re-run with "
            "boilerplate.max_strip_ratio raised, or a higher min_document_ratio, after "
            "checking the learned template lines above."
        )

    if not cleaned:
        raise ValueError("Boilerplate stripping removed every document")
    return cleaned
