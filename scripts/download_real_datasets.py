"""
EcoSort-Edge: Automated Dataset Downloader & Ingestion Script
Downloads official Stanford TrashNet (42.8 MB) and UCI RealWaste (656 MB)
Prepares the raw data for RTX 4050 two-stage transfer learning.
"""

import sys
import os
import zipfile
import requests
from pathlib import Path
from tqdm import tqdm

# Ensure UTF-8 output
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

TRASHNET_URL = "https://github.com/garythung/trashnet/raw/master/data/dataset-resized.zip"
REALWASTE_URL = "https://archive.ics.uci.edu/static/public/908/realwaste.zip"

RAW_DIR = Path("data/raw")

def download_file(url: str, dest_path: Path):
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists() and dest_path.stat().st_size > 1024 * 1024:
        print(f"[*] Archive already exists ({dest_path.stat().st_size / (1024*1024):.1f} MB): {dest_path.name}")
        return True

    print(f"\n[+] Connecting to: {url}")
    response = requests.get(url, stream=True, timeout=30)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024 * 64  # 64 KB chunks

    with open(dest_path, "wb") as f, tqdm(
        total=total_size, unit="B", unit_scale=True, desc=dest_path.name
    ) as bar:
        for chunk in response.iter_content(block_size):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))
    print(f"[+] Download complete: {dest_path.name}")
    return True

def extract_archive(zip_path: Path, extract_to: Path):
    print(f"[+] Unzipping {zip_path.name} to {extract_to}...")
    extract_to.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"[+] Extraction complete for {zip_path.name}")

def main():
    print("=" * 75)
    print(" EcoSort-Edge: Dataset Ingestion (TrashNet + UCI RealWaste)")
    print(" Target Storage: A:\\Projects\\ARM\\data\\raw")
    print("=" * 75)

    # 1. Download Stanford TrashNet (~42.8 MB)
    trashnet_zip = RAW_DIR / "trashnet.zip"
    trashnet_extracted = RAW_DIR / "trashnet"
    try:
        download_file(TRASHNET_URL, trashnet_zip)
        extract_archive(trashnet_zip, trashnet_extracted)
    except Exception as e:
        print(f"[!] TrashNet download error: {e}")

    # 2. Download UCI RealWaste (~656 MB)
    realwaste_zip = RAW_DIR / "realwaste.zip"
    realwaste_extracted = RAW_DIR / "realwaste"
    try:
        download_file(REALWASTE_URL, realwaste_zip)
        extract_archive(realwaste_zip, realwaste_extracted)
    except Exception as e:
        print(f"[!] RealWaste download error: {e}")

    print("\n" + "=" * 75)
    print("[SUCCESS] All raw archives downloaded and extracted.")
    print("Next step: Run `python scripts/01_curate_and_dedup.py` to clean and split.")
    print("=" * 75)

if __name__ == "__main__":
    main()
