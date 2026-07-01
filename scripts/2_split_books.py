"""
Split `data/odyssey.txt` into per-book files under `data/books/`.

Each book starts with a heading like "BOOK I", "BOOK XXIV", etc.
The script preserves the heading block and writes each book to:
  data/books/book_i.txt
  data/books/book_xxiv.txt
"""

import re
from pathlib import Path

BOOK_HEADING = re.compile(r"^(BOOK\s+([IVXLCDM]+))\s*$", re.MULTILINE)


def data_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "data"


def load_odyssey_text() -> str:
    source = data_dir() / "odyssey.txt"
    if not source.exists():
        raise FileNotFoundError(f"Missing source file: {source}")
    return source.read_text(encoding="utf-8", errors="replace")


def find_book_ranges(text: str) -> list[tuple[str, int, int]]:
    matches = list(BOOK_HEADING.finditer(text))
    if not matches:
        raise ValueError("No BOOK headings found in odyssey.txt")

    ranges = []
    for idx, match in enumerate(matches):
        start = match.start()
        if start > 0:
            prev_newline = text.rfind("\n", 0, start - 1)
            if prev_newline != -1:
                prev_line = text[prev_newline + 1 : start].strip()
                if prev_line and set(prev_line) <= {"-"}:
                    start = prev_newline + 1

        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        book_label = match.group(2).lower()
        ranges.append((book_label, start, end))
    return ranges


def write_books(text: str, ranges: list[tuple[str, int, int]]) -> None:
    out_dir = data_dir() / "books"
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, (book_label, start, end) in enumerate(ranges):
        target = out_dir / f"book_{book_label} ({i + 1}).txt"
        content = text[start:end].rstrip() + "\n"
        target.write_text(content, encoding="utf-8")
        print(f"Saved {target}")


def main() -> None:
    text = load_odyssey_text()
    ranges = find_book_ranges(text)
    write_books(text, ranges)
    print(f"Extracted {len(ranges)} books to data/books/")


if __name__ == "__main__":
    main()
