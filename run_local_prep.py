#!/usr/bin/env python3
"""
run_local_prep.py — Chạy 3 bước chuẩn bị dữ liệu BGL trên local:
  Step 1: Split (chronological 80/20)
  Step 2: Preprocess (BGL regex normalization)
  Step 3: Train WordPiece tokenizer (vocab=1000)

Output sẽ nằm ở thư mục: kaggle_input/
→ Upload thư mục này lên Kaggle làm Dataset.

Usage:
    python3 run_local_prep.py --bgl_log /path/to/BGL.log

Nếu không truyền --bgl_log, script sẽ tự tải về từ Zenodo.
"""
import argparse
import os
import sys
import json

# ── Resolve paths ───────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
LANOBERT_DIR = os.path.join(SCRIPT_DIR, "LAnoBERT")
sys.path.insert(0, LANOBERT_DIR)

OUTPUT_DIR   = os.path.join(SCRIPT_DIR, "kaggle_input", "bgl-preprocessed")
DATA_DIR     = os.path.join(OUTPUT_DIR, "data", "BGL")
TOK_DIR      = os.path.join(OUTPUT_DIR, "tokenizer")
CONFIG_PATH  = os.path.join(OUTPUT_DIR, "bgl.yaml")


def download_bgl(dest_log: str) -> None:
    """Download BGL.zip from Zenodo and extract BGL.log."""
    import urllib.request, zipfile, shutil

    zip_path = dest_log.replace(".log", ".zip")
    os.makedirs(os.path.dirname(dest_log), exist_ok=True)

    url = "https://zenodo.org/records/8196385/files/BGL.zip"
    print(f"[download] {url}")
    print("[download] Downloading... (748 MB, có thể mất vài phút)")
    try:
        import subprocess
        subprocess.run(
            ["wget", "-c", "--show-progress", "-O", zip_path, url],
            check=True
        )
    except FileNotFoundError:
        urllib.request.urlretrieve(url, zip_path)

    print(f"[download] Extracting {zip_path} ...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(os.path.dirname(dest_log))

    if not os.path.exists(dest_log):
        # tên file trong zip có thể khác
        extracted = [f for f in os.listdir(os.path.dirname(dest_log)) if f.endswith(".log")]
        if extracted:
            shutil.move(os.path.join(os.path.dirname(dest_log), extracted[0]), dest_log)

    print(f"[download] Done → {dest_log}  ({os.path.getsize(dest_log)/1e9:.2f} GB)")


def write_config(raw_log: str) -> str:
    """Ghi config yaml với đường dẫn tuyệt đối cho local."""
    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    cfg_content = f"""# LAnoBERT — BGL config (local preprocessing)
# Exact author hyperparameters — chỉ thay paths thành absolute local paths.
dataset: BGL
run_name: bgl

paths:
  raw_log:      {raw_log}
  train_raw:    {DATA_DIR}/BGL_train_normal.raw
  test_raw:     {DATA_DIR}/BGL_test.raw
  test_label:   {DATA_DIR}/BGL_test_label.log
  train_normal: {DATA_DIR}/BGL_train_normal_parsed.log
  test_log:     {DATA_DIR}/BGL_test_parsed.log
  tokenizer_dir: {TOK_DIR}
  model_dir:    {OUTPUT_DIR}/model        # không dùng ở local
  result_dir:   {OUTPUT_DIR}/results      # không dùng ở local

split:
  method: line
  train_ratio: 0.8
  label_marker: "-"

preprocess:
  regex_profile: bgl
  mask_block_id: true
  mask_ip: true
  mask_number: true
  drop_header_fields: 3

tokenizer:
  vocab_size: 1000
  min_frequency: 2
  lowercase: false

train:
  max_len: 512
  mlm_probability: 0.20
  num_train_epochs: 10
  per_device_train_batch_size: 32
  learning_rate: 1.0e-4
  weight_decay: 0.01
  warmup_ratio: 0.1
  save_steps: 50000
  save_total_limit: 2
  logging_steps: 1000
  seed: 42

inference:
  hf_model: null
  hf_subfolder: null
  score: error_mean
  top_k: 5
  top_ks: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
  batch_size: 16
  max_eval_samples: null
  pretrained_model: null
"""
    with open(CONFIG_PATH, "w") as f:
        f.write(cfg_content)
    print(f"[config] Written → {CONFIG_PATH}")
    return CONFIG_PATH


def step1_split(cfg) -> dict:
    from lanobert.split import run as split_run
    trw = cfg.get_path("paths.train_raw")
    if os.path.exists(trw) and os.path.getsize(trw) > 0:
        print(f"[split] SKIP — already done: {trw}")
        stats_path = os.path.join(DATA_DIR, "split_stats.json")
        if os.path.exists(stats_path):
            return json.load(open(stats_path))
        return {}
    print("\n" + "="*55)
    print("STEP 1: Chronological Split (80/20)")
    print("="*55)
    return split_run(cfg)


def step2_preprocess(cfg) -> None:
    from lanobert.preprocess import _options_from_config, preprocess_file
    opt        = _options_from_config(cfg)
    train_norm = cfg.get_path("paths.train_normal")
    test_log   = cfg.get_path("paths.test_log")

    print("\n" + "="*55)
    print("STEP 2: BGL Regex Preprocessing")
    print("="*55)

    if os.path.exists(train_norm) and os.path.getsize(train_norm) > 0:
        print(f"[preprocess] SKIP train — {train_norm}")
    else:
        n = preprocess_file(cfg.get_path("paths.train_raw"), train_norm, opt)
        print(f"[preprocess] train: {n:,} lines written")

    if os.path.exists(test_log) and os.path.getsize(test_log) > 0:
        print(f"[preprocess] SKIP test  — {test_log}")
    else:
        n = preprocess_file(cfg.get_path("paths.test_raw"), test_log, opt)
        print(f"[preprocess] test : {n:,} lines written")


def step3_tokenizer(cfg) -> str:
    from lanobert.tokenizer import train_tokenizer, vocab_path_for, load_tokenizer
    vocab_file = vocab_path_for(cfg)

    print("\n" + "="*55)
    print("STEP 3: WordPiece Tokenizer (vocab=1000)")
    print("="*55)

    if os.path.exists(vocab_file):
        print(f"[tokenizer] SKIP — already exists: {vocab_file}")
    else:
        vocab_file = train_tokenizer(cfg)

    tok = load_tokenizer(vocab_file, max_len=512)
    print(f"[tokenizer] vocab_size = {tok.vocab_size}")
    return vocab_file


def print_summary(stats: dict, vocab_file: str) -> None:
    print("\n" + "="*55)
    print("  ✅  LOCAL PREP COMPLETE")
    print("="*55)
    if stats:
        print(f"  train_normal : {stats.get('train_normal', '?'):,} lines")
        print(f"  test_total   : {stats.get('test_total',   '?'):,} lines")
        print(f"  test_anomaly : {stats.get('test_anomaly', '?'):,}")
        print(f"  test_normal  : {stats.get('test_normal',  '?'):,}")
    print(f"  tokenizer    : {vocab_file}")
    print(f"\n  📁 Upload thư mục này lên Kaggle Dataset:")
    print(f"     {OUTPUT_DIR}")
    print("\n  Files cần upload:")
    for root, _, files in os.walk(OUTPUT_DIR):
        for fn in sorted(files):
            fp  = os.path.join(root, fn)
            rel = os.path.relpath(fp, OUTPUT_DIR)
            sz  = os.path.getsize(fp)
            print(f"     {rel:50s}  {sz/1e6:8.2f} MB")
    print("="*55)


def main():
    parser = argparse.ArgumentParser(
        description="Chạy local: split + preprocess + tokenizer cho BGL"
    )
    parser.add_argument(
        "--bgl_log",
        default=None,
        help="Đường dẫn tới BGL.log (nếu không có sẽ tự tải từ Zenodo)"
    )
    args = parser.parse_args()

    # ── Xác định đường dẫn BGL.log ──────────────────────────────────────────
    os.makedirs(DATA_DIR, exist_ok=True)

    if args.bgl_log:
        if not os.path.exists(args.bgl_log):
            print(f"[error] File không tồn tại: {args.bgl_log}")
            sys.exit(1)
        raw_log = os.path.abspath(args.bgl_log)
        print(f"[info] Dùng BGL.log từ: {raw_log}")
    else:
        raw_log = os.path.join(DATA_DIR, "BGL.log")
        if os.path.exists(raw_log):
            print(f"[info] BGL.log đã có: {raw_log}")
        else:
            download_bgl(raw_log)

    # ── Write config → load ─────────────────────────────────────────────────
    write_config(raw_log)

    from lanobert.utils import load_config
    cfg = load_config(CONFIG_PATH)

    # ── Run steps ───────────────────────────────────────────────────────────
    stats      = step1_split(cfg)
    step2_preprocess(cfg)
    vocab_file = step3_tokenizer(cfg)

    # ── Summary ─────────────────────────────────────────────────────────────
    print_summary(stats, vocab_file)


if __name__ == "__main__":
    main()
