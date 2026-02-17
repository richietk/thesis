#!/usr/bin/env python3

import re
import requests
from pathlib import Path

BIB_PATH = Path("/home/richard/Documents/Thesis/tex/refs.bib")
PAPERS_DIR = Path("/home/richard/Documents/Thesis/wip_dirs/papers")

ARXIV_ABS_PATTERN = re.compile(r"https?://arxiv\.org/abs/([^\s\}]+)")
DOI_PATTERN = re.compile(r"doi\s*=\s*\{([^\}]+)\}", re.IGNORECASE)
ENTRY_PATTERN = re.compile(r"@[\w]+\{([^,]+),")

def parse_bib_entries(bib_text):
    entries = []
    raw_entries = re.split(r"\n@", bib_text)
    for i, chunk in enumerate(raw_entries):
        if i == 0 and not chunk.strip().startswith("@"):
            continue
        if i > 0:
            chunk = "@" + chunk
        match = ENTRY_PATTERN.search(chunk)
        if match:
            entry_id = match.group(1).strip().lower()
            entries.append((entry_id, chunk))
    return entries

def extract_arxiv_pdf_url(entry_text):
    match = ARXIV_ABS_PATTERN.search(entry_text)
    if not match:
        return None
    arxiv_id = match.group(1)
    return f"https://arxiv.org/pdf/{arxiv_id}.pdf"

def extract_doi(entry_text):
    match = DOI_PATTERN.search(entry_text)
    if match:
        return match.group(1)
    return None

def download_pdf(url, target_path):
    print(f"Downloading {url} -> {target_path}")
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        with open(target_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        return True
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        return False

def main():
    if not BIB_PATH.exists():
        raise FileNotFoundError(f"Bib file not found: {BIB_PATH}")

    PAPERS_DIR.mkdir(parents=True, exist_ok=True)
    bib_text = BIB_PATH.read_text(encoding="utf-8")
    entries = parse_bib_entries(bib_text)

    for entry_id, entry_text in entries:
        pdf_path = PAPERS_DIR / f"{entry_id}.pdf"
        if pdf_path.exists():
            continue

        # Try arXiv first
        pdf_url = extract_arxiv_pdf_url(entry_text)
        if pdf_url:
            if download_pdf(pdf_url, pdf_path):
                continue

        # If no arXiv, try DOI via Sci-Hub
        doi = extract_doi(entry_text)
        if doi:
            sci_hub_url = f"https://sci-hub.ru/{doi}"
            download_pdf(sci_hub_url, pdf_path)

if __name__ == "__main__":
    main()
