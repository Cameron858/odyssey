"""
Fetch Homer's Odyssey text into `data/` and split it into reference and material.

Behavior:
- If `data/odyssey.mb.txt`, `data/odyssey.txt`, and `data/reference.txt` all exist: print a message and exit.
- If `.mb.txt` exists but split files are missing: perform the split.
- If none exist: fetch then split.
"""

import sys
from pathlib import Path

import httpx
from pyprojroot import here

URL = "https://classics.mit.edu/Homer/odyssey.mb.txt"


def data_dir() -> Path:
    d = here("data")
    return Path(d)


def is_downloaded(out_dir: Path) -> bool:
    return (out_dir / "odyssey.mb.txt").exists()


def is_split(out_dir: Path) -> bool:
    return (out_dir / "odyssey.txt").exists() and (out_dir / "reference.txt").exists()


def fetch_file() -> None:
    out_dir = data_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "odyssey.mb.txt"

    print(f"Fetching {URL}...")
    with httpx.Client(timeout=30.0) as client:
        resp = client.get(URL, follow_redirects=True)
        resp.raise_for_status()
        out_file.write_bytes(resp.content)

    print(f"Saved: {out_file}")


def split_file() -> None:
    out_dir = data_dir()
    file_path = out_dir / "odyssey.mb.txt"

    if not file_path.exists():
        raise FileNotFoundError(f"Source file not found: {file_path}")

    # Read using utf-8 with fallback
    text = file_path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines(keepends=True)

    header_lines = lines[:12]
    material_lines = lines[12:]

    (out_dir / "reference.txt").write_text("".join(header_lines), encoding="utf-8")
    (out_dir / "odyssey.txt").write_text("".join(material_lines), encoding="utf-8")

    print(f"Split into: {out_dir / 'reference.txt'} and {out_dir / 'odyssey.txt'}")


def main() -> None:
    out_dir = data_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    mb = out_dir / "odyssey.mb.txt"
    od = out_dir / "odyssey.txt"
    ref = out_dir / "reference.txt"

    if mb.exists() and od.exists() and ref.exists():
        print("All files present:")
        print(f" - {mb}")
        print(f" - {ref}")
        print(f" - {od}")
        return

    if mb.exists():
        print("Found odyssey.mb.txt - performing split...")
        split_file()
        return

    print("No odyssey files found - fetching and splitting...")
    fetch_file()
    split_file()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - surface errors to user
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
