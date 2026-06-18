"""
Fetch Homer's Odyssey text and save to data/odyssey.mb.txt
"""

import httpx
from pyprojroot import here

URL = "https://classics.mit.edu/Homer/odyssey.mb.txt"


def main() -> None:
    out_dir = here("data")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "odyssey.txt"

    print(f"Fetching {URL}...")
    with httpx.Client(timeout=30.0) as client:
        resp = client.get(URL, follow_redirects=True)
        resp.raise_for_status()
        out_file.write_bytes(resp.content)

    print(f"Saved: {out_file}")


if __name__ == "__main__":
    main()
