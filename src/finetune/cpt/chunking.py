"""Structure-aware chunking for CPT over report documents.

Three phases:

1. `parse_blocks`  — split the document into heading / table / paragraph blocks,
   recording each block's *line span* rather than its text. Chunks are later
   sliced verbatim out of the original lines, so no whitespace is ever
   reconstructed (and therefore never corrupted).
2. `pack_blocks`   — fill chunks with consecutive blocks up to the budget,
   refusing to leave a heading stranded at the end of a chunk without the
   content it introduces.
3. `split_oversized` — a single block bigger than the whole budget (a long table,
   a monster paragraph) is split down separator by separator, so nothing is
   silently truncated by the trainer later.
"""

import re
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

MARKDOWN_HEADING = re.compile(r"^\s*#{1,6}\s+\S")
TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")

HEADING_MAX_CHARS = 200

# Vietnamese report numbering
_RULE_BASED_HEADINGS = (
    ("phan", re.compile(r"^\s*PH[ẦA]N\s+[IVX]+\s*[.)]\s+\S", re.IGNORECASE)),
    ("chuong", re.compile(r"^\s*CH[ƯU][ƠO]NG\s+[IVX]+\s*[.)]\s+\S", re.IGNORECASE)),
    ("muc", re.compile(r"^\s*M[ỤU]C\s+\d+\s*[.)]\s+\S", re.IGNORECASE)),
    ("sub_decimal", re.compile(r"^\s*\d+\.\d+\.\d+\s*[.)]\s+\S")),
    ("decimal", re.compile(r"^\s*\d+\.\d+\s*[.)]\s+\S")),
    ("upper_alpha", re.compile(r"^\s*[A-HJ-UW-Z]\s*[.)]\s+\S")),
    ("roman", re.compile(r"^\s*[IVX]+\s*[.)]\s+\S")),
    ("arabic", re.compile(r"^\s*\d+\s*[.)]\s+\S")),
    ("lower_alpha", re.compile(r"^\s*[a-z]\s*[.)]\s+\S")),
)

_HIERARCHY = (
    "phan",
    "chuong",
    "muc",
    "upper_alpha",
    "roman",
    "arabic",
    "decimal",
    "sub_decimal",
    "lower_alpha",
)

_SEPARATORS = ("\n\n", "\n", ". ", "; ", ", ", " ")

TokenCounter = Callable[[str], int]


@dataclass
class Block:
    """A structural unit of the document, identified by the lines it occupies."""
    kind: str  # "heading" | "table" | "paragraph"
    start: int
    end: int  # inclusive
    pattern: Optional[str] = None
    level: Optional[int] = None


def detect_heading(line: str) -> Optional[str]:
    """Return the heading pattern name for `line`, or None if it is not one."""
    if MARKDOWN_HEADING.match(line):
        return f"markdown_{len(line.strip()) - len(line.strip().lstrip('#'))}"
    if len(line.strip()) > HEADING_MAX_CHARS:
        return None
    for name, pattern in _RULE_BASED_HEADINGS:
        if pattern.match(line):
            return name
    return None


def _normalise_levels(blocks: List[Block]) -> None:
    """Assign levels 1..n to the heading patterns this document actually uses."""
    seen: List[str] = []
    for block in blocks:
        # A heading block always carries the pattern that identified it.
        if block.kind == "heading" and block.pattern and block.pattern not in seen:
            seen.append(block.pattern)

    markdown = sorted(p for p in seen if p.startswith("markdown_"))
    rule_based = [p for p in _HIERARCHY if p in seen]

    levels = {pattern: index for index, pattern in enumerate(markdown, start=1)}
    offset = len(markdown)
    levels.update({pattern: offset + i for i, pattern in enumerate(rule_based, start=1)})

    for block in blocks:
        if block.kind == "heading" and block.pattern:
            block.level = levels.get(block.pattern)


def parse_blocks(lines: Sequence[str]) -> List[Block]:
    """Split lines into structural blocks. Blank lines belong to no block."""
    blocks: List[Block] = []
    index = 0

    while index < len(lines):
        if not lines[index].strip():
            index += 1
            continue

        pattern = detect_heading(lines[index])
        if pattern is not None:
            blocks.append(Block("heading", index, index, pattern=pattern))
            index += 1
            continue

        if TABLE_ROW.match(lines[index]):
            start = index
            while index < len(lines) and TABLE_ROW.match(lines[index]):
                index += 1
            blocks.append(Block("table", start, index - 1))
            continue

        # Paragraph: consecutive non-blank lines that start no other block.
        start = index
        while index < len(lines):
            line = lines[index]
            if not line.strip() or detect_heading(line) or TABLE_ROW.match(line):
                break
            index += 1
        blocks.append(Block("paragraph", start, index - 1))

    _normalise_levels(blocks)
    return blocks


def _slice_after(text: str, separator: str) -> List[str]:
    """Cut `text` after each occurrence of `separator`, losing nothing.

    Each piece keeps its own trailing separator, so `"".join(pieces) == text`.
    A plain `text.split(". ")` would drop the "." at every boundary — real
    characters, not whitespace — which silently deletes content from the corpus.
    """
    pieces: List[str] = []
    start = 0
    while True:
        found = text.find(separator, start)
        if found == -1:
            break
        end = found + len(separator)
        pieces.append(text[start:end])
        start = end
    if start < len(text):
        pieces.append(text[start:])
    return pieces


def split_oversized(text: str, count_tokens: TokenCounter, max_tokens: int) -> List[str]:
    """Break a block that alone exceeds the budget, preserving every character."""
    if count_tokens(text) <= max_tokens:
        return [text]

    for separator in _SEPARATORS:
        parts = _slice_after(text, separator)
        if len(parts) < 2:
            continue

        pieces: List[str] = []
        current = ""
        for part in parts:
            if current and count_tokens(current + part) > max_tokens:
                pieces.append(current)
                current = part
            else:
                current += part
        if current:
            pieces.append(current)

        if len(pieces) < 2:
            continue

        result: List[str] = []
        for piece in pieces:
            if count_tokens(piece) > max_tokens:
                result.extend(split_oversized(piece, count_tokens, max_tokens))
            else:
                result.append(piece.strip())
        return [piece for piece in result if piece]

    # No separator left (one enormous run with no spaces): cut in half and recurse.
    middle = len(text) // 2
    if middle == 0:
        return [text]
    return split_oversized(text[:middle], count_tokens, max_tokens) + split_oversized(
        text[middle:], count_tokens, max_tokens
    )


def _slice(lines: Sequence[str], blocks: List[Block]) -> str:
    """Verbatim text of a run of blocks, straight out of the original lines."""
    return "\n".join(lines[blocks[0].start : blocks[-1].end + 1]).strip()


def _trailing_headings(blocks: List[Block]) -> List[Block]:
    """The run of headings at the end of `blocks`, which must not be orphaned."""
    cut = len(blocks)
    while cut > 0 and blocks[cut - 1].kind == "heading":
        cut -= 1
    return blocks[cut:] if cut < len(blocks) else []


def pack_blocks(
    lines: Sequence[str],
    blocks: List[Block],
    count_tokens: TokenCounter,
    max_tokens: int,
) -> List[str]:
    chunks: List[str] = []
    current: List[Block] = []

    def flush() -> None:
        if not current:
            return
        text = _slice(lines, current)
        if text:
            headings_only = all(block.kind == "heading" for block in current)
            if headings_only and chunks and count_tokens(f"{chunks[-1]}\n\n{text}") <= max_tokens:
                chunks[-1] = f"{chunks[-1]}\n\n{text}"
            else:
                chunks.append(text)
        current.clear()

    def detach_trailing_headings() -> str:
        """Remove the trailing heading run from `current` and return its text."""
        carried = _trailing_headings(current)
        if not carried:
            return ""
        del current[len(current) - len(carried) :]
        return "\n".join(lines[carried[0].start : carried[-1].end + 1]).strip()

    def attach_heading(heading: str, pieces: List[str]) -> List[str]:
        """Put `heading` back in front of the content it introduces."""
        if count_tokens(f"{heading}\n\n{pieces[0]}") <= max_tokens:
            pieces[0] = f"{heading}\n\n{pieces[0]}"
            return pieces
        # Make room by splitting the first piece finer
        room = max_tokens - count_tokens(heading) - 1
        if room < 1:
            return [heading] + pieces
        sub = split_oversized(pieces[0], count_tokens, room)
        return [f"{heading}\n\n{sub[0]}"] + sub[1:] + pieces[1:]

    for block in blocks:
        text = "\n".join(lines[block.start : block.end + 1])
        tokens = count_tokens(text)

        if tokens > max_tokens:
            heading = detach_trailing_headings()
            flush()
            pieces = split_oversized(text.strip(), count_tokens, max_tokens)
            if heading:
                pieces = attach_heading(heading, pieces)
            chunks.extend(pieces)
            continue

        if current and count_tokens(_slice(lines, current + [block])) > max_tokens:
            # Carry any trailing headings over so they stay with their content.
            carried = _trailing_headings(current)
            del current[len(current) - len(carried) :]
            flush()
            current.extend(carried)

        current.append(block)

    flush()
    return chunks


def chunk_document(text: str, count_tokens: TokenCounter, max_tokens: int) -> List[str]:
    """Chunk one document into pieces of at most `max_tokens` tokens.

    `max_tokens` should already exclude the EOS token TRL appends to every
    language-modelling sample — see `reserve_eos` in prepare_cpt_data.
    """
    if max_tokens < 1:
        raise ValueError(f"max_tokens must be >= 1, got {max_tokens}")
    lines = text.split("\n")
    blocks = parse_blocks(lines)
    if not blocks:
        return []
    return pack_blocks(lines, blocks, count_tokens, max_tokens)
