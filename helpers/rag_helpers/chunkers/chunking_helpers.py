import re
from typing import Optional

DEFAULT_CHUNK_MIN = 800
DEFAULT_CHUNK_MAX = 1600
DEFAULT_OVERLAP_RATIO = 0.25


def detect_sections(text: str) -> list[dict[str, int]]:
    sections = []
    lines = text.split("\n")

    for i, line in enumerate(lines):
        line_stripped = line.strip()

        if re.match(r"^[A-Z][A-Z\s]{10,}$", line_stripped):
            sections.append({"type": "heading", "index": i, "text": line_stripped})
        elif re.match(r"^[A-Z][a-z]+.*:?\s*$", line_stripped) and len(line_stripped) < 100:
            sections.append({"type": "heading", "index": i, "text": line_stripped})
        elif re.match(r"^\d+[\.\)]\s+[A-Z]", line_stripped):
            sections.append({"type": "numbered", "index": i, "text": line_stripped})

    return sections


def chunk_by_sections(
    text: str,
    sections: list[dict[str, int]],
    min_chunk: int = DEFAULT_CHUNK_MIN,
    max_chunk: int = DEFAULT_CHUNK_MAX,
) -> list[tuple[int, int, Optional[str]]]:
    chunks = []
    lines = text.split("\n")

    if not sections:
        return []

    for i, section in enumerate(sections):
        start_idx = section["index"]
        end_idx = sections[i + 1]["index"] if i + 1 < len(sections) else len(lines)

        section_lines = lines[start_idx:end_idx]
        section_text = "\n".join(section_lines)
        section_chars = len(section_text)

        heading = section.get("text")

        if section_chars <= max_chunk:
            start_char = text.find(section_text)
            if start_char >= 0:
                end_char = start_char + section_chars
                chunks.append((start_char, end_char, heading))
        else:
            char_start = text.find(section_text)
            if char_start >= 0:
                offset = 0
                while offset < section_chars:
                    chunk_end = min(offset + max_chunk, section_chars)
                    chunk_text = section_text[offset:chunk_end]

                    if chunk_end < section_chars:
                        last_space = chunk_text.rfind("\n")
                        if last_space > max_chunk * 0.7:
                            chunk_text = chunk_text[:last_space]
                            chunk_end = offset + len(chunk_text)

                    chunks.append((char_start + offset, char_start + chunk_end, heading))
                    offset += int(max_chunk * (1 - DEFAULT_OVERLAP_RATIO))

    return chunks


def chunk_by_sliding_window(
    text: str,
    min_chunk: int = DEFAULT_CHUNK_MIN,
    max_chunk: int = DEFAULT_CHUNK_MAX,
    overlap_ratio: float = DEFAULT_OVERLAP_RATIO,
) -> list[tuple[int, int, Optional[str]]]:
    chunks = []
    text_length = len(text)

    if text_length <= max_chunk:
        return [(0, text_length, None)]

    offset = 0
    while offset < text_length:
        chunk_end = min(offset + max_chunk, text_length)
        chunk_text = text[offset:chunk_end]

        if chunk_end < text_length:
            for break_char in ["\n\n", "\n", ". ", " "]:
                last_break = chunk_text.rfind(break_char)
                if last_break > max_chunk * 0.7:
                    chunk_text = chunk_text[: last_break + len(break_char)]
                    chunk_end = offset + len(chunk_text)
                    break

        if len(chunk_text) < min_chunk and offset > 0:
            if chunks:
                prev_start, prev_end, _ = chunks[-1]
                chunks[-1] = (prev_start, chunk_end, None)
            offset = chunk_end
            continue

        chunks.append((offset, chunk_end, None))
        offset += int(max_chunk * (1 - overlap_ratio))

    return chunks
