"""
Jack-o'-lantern data download: LAION metadata -> filtered candidate list -> images.

Stage 1  python download_data.py metadata --max-shards 200
         Downloads Re-LAION shards one at a time, keeps jack-o'-lantern rows,
         deletes the shard. Resumable: finished shards are recorded in
         work/done_shards.txt and skipped on re-run.

Stage 2  python download_data.py images
         Downloads the candidate images with img2dataset (512px, center crop).

Needs: pip install -r requirements.txt, and `huggingface-cli login` first
(you must have accepted the dataset terms on huggingface.co).
"""
import argparse
import glob
import os
import re
import subprocess
import sys

import pandas as pd

REPO = "laion/relaion2B-en-research-safe"
WORK = "work"
SHARD_TMP = f"{WORK}/shard_tmp"
CAND_DIR = f"{WORK}/candidates"          # one small parquet per processed shard
DONE_LOG = f"{WORK}/done_shards.txt"
MERGED = f"{WORK}/candidates.parquet"
IMAGES_OUT = f"{WORK}/raw_images"

POSITIVE = re.compile(r"jack[\s\-']*o[\s\-']*lantern|carved pumpkin|pumpkin carving", re.I)
NEGATIVE = re.compile(
    r"clip\s*art|vector|cartoon|icon|svg|coloring|stencil|pattern|template|"
    r"costume|cookie|cake|sticker|wallpaper|illustration|drawing", re.I)

MIN_SIDE = 256
MIN_SIMILARITY = 0.30


def filter_df(df: pd.DataFrame) -> pd.DataFrame:
    """Keep rows that look like real jack-o'-lantern photos."""
    cap = df["TEXT"].fillna("")
    m = cap.str.contains(POSITIVE) & ~cap.str.contains(NEGATIVE)
    m &= (df["WIDTH"] >= MIN_SIDE) & (df["HEIGHT"] >= MIN_SIDE)
    m &= df["similarity"] >= MIN_SIMILARITY
    out = df.loc[m, ["URL", "TEXT"]].rename(columns={"URL": "url", "TEXT": "caption"})
    return out


def load_done() -> set:
    if not os.path.exists(DONE_LOG):
        return set()
    with open(DONE_LOG) as f:
        return {line.strip() for line in f if line.strip()}


def stage_metadata(max_shards: int, start: int):
    from huggingface_hub import HfApi, hf_hub_download

    for d in (WORK, SHARD_TMP, CAND_DIR):
        os.makedirs(d, exist_ok=True)

    files = sorted(
        f for f in HfApi().list_repo_files(REPO, repo_type="dataset") if f.endswith(".parquet")
    )
    if not files:
        sys.exit("No parquet files found. Are you logged in and have you accepted the dataset terms?")
    print(f"{len(files)} shards in repo; processing {start}..{start + max_shards}")

    done = load_done()
    total = 0
    for i, f in enumerate(files[start:start + max_shards], start=start):
        if f in done:
            continue
        try:
            path = hf_hub_download(REPO, f, repo_type="dataset", local_dir=SHARD_TMP)
        except Exception as e:  # network blip, rate limit, etc: skip, will retry next run
            print(f"[{i}] download failed ({e}); will retry on next run")
            continue
        try:
            df = pd.read_parquet(path, columns=["URL", "TEXT", "WIDTH", "HEIGHT", "similarity"])
            out = filter_df(df)
            out.to_parquet(f"{CAND_DIR}/shard_{i:05d}.parquet")
            total += len(out)
            print(f"[{i}] {f}: {len(df):,} rows -> {len(out)} kept")
        finally:
            os.remove(path)
        with open(DONE_LOG, "a") as log:
            log.write(f + "\n")

    parts = glob.glob(f"{CAND_DIR}/*.parquet")
    if not parts:
        sys.exit("No candidates yet.")
    merged = pd.concat([pd.read_parquet(p) for p in parts]).drop_duplicates("url")
    merged.to_parquet(MERGED)
    print(f"\nThis run kept {total}. Merged candidates across all shards: {len(merged):,} -> {MERGED}")


def stage_images(processes: int, threads: int):
    if not os.path.exists(MERGED):
        sys.exit(f"{MERGED} not found. Run the metadata stage first.")
    subprocess.run(
        [
            "img2dataset",
            "--url_list", MERGED,
            "--input_format", "parquet",
            "--url_col", "url",
            "--caption_col", "caption",
            "--output_format", "files",
            "--output_folder", IMAGES_OUT,
            "--image_size", "512",
            "--resize_mode", "center_crop",
            "--processes_count", str(processes),
            "--thread_count", str(threads),
            "--timeout", "8",
            "--retries", "1",
        ],
        check=True,
    )
    n = len(glob.glob(f"{IMAGES_OUT}/**/*.jpg", recursive=True))
    print(f"\nDownloaded {n} images into {IMAGES_OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="stage", required=True)

    m = sub.add_parser("metadata")
    m.add_argument("--max-shards", type=int, default=200)
    m.add_argument("--start", type=int, default=0, help="index of first shard (to continue past a previous slice)")

    i = sub.add_parser("images")
    i.add_argument("--processes", type=int, default=8)
    i.add_argument("--threads", type=int, default=64)

    a = ap.parse_args()
    if a.stage == "metadata":
        stage_metadata(a.max_shards, a.start)
    else:
        stage_images(a.processes, a.threads)