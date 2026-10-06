"""Generate kaggle_lanobert_bgl.ipynb — uses git clone from GitHub instead of embedding source."""
import json, textwrap

GITHUB_REPO = "https://github.com/rubyhcm/at-LAnoBERT"
REPO_SUBDIR = "LAnoBERT"   # subfolder inside the cloned repo that contains lanobert/ and configs/

def md(src):
    return {"cell_type": "markdown", "id": None, "metadata": {},
            "source": src if isinstance(src, list) else [src]}

def code(src, cid=None):
    src = textwrap.dedent(src).lstrip("\n")
    lines = [l + "\n" for l in src.splitlines()]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    return {"cell_type": "code", "execution_count": None, "id": cid,
            "metadata": {}, "outputs": [], "source": lines}

cells = []

# ── header ─────────────────────────────────────────────────────────────────
cells.append(md(
    "# LAnoBERT — BGL Baseline (Author Config)\n\n"
    f"**Source**: [{GITHUB_REPO}]({GITHUB_REPO})  \n"
    "**Paper**: [LAnoBERT: System Log Anomaly Detection based on BERT Masked Language Model (ASC 2023)](https://doi.org/10.1016/j.asoc.2023.110689)  \n"
    "**HF Hub**: [yukyung/LAnoBERT](https://huggingface.co/yukyung/LAnoBERT)\n\n"
    "Faithful reproduction of the **BGL main config** from the authors:\n"
    "- From-scratch BERT with **log-specific WordPiece vocabulary** (vocab_size=1000)\n"
    "- MLM pretraining on **normal logs only** (train_ratio=0.8, chronological)\n"
    "- batch=32 · lr=1e-4 · 10 epochs · mlm_probability=0.20\n"
    "- **error_mean** scoring (recommended by authors)\n\n"
    "Expected results: **AUROC 1.000 / Best-F1 1.000**"
))

# ── 0. Env ─────────────────────────────────────────────────────────────────
cells.append(md("## 0. Environment Check"))
cells.append(code("""
    import subprocess, sys, os, torch

    r = subprocess.run(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader'],
                       capture_output=True, text=True)
    print('GPU    :', r.stdout.strip() or 'None')
    print(f'PyTorch: {torch.__version__}')
    print(f'CUDA   : {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f'Device : {p.name}  VRAM={p.total_memory/1e9:.1f} GB')
"""))

# ── 1. Install ─────────────────────────────────────────────────────────────
cells.append(md("## 1. Install Dependencies"))
cells.append(code("""
    %%capture
    !pip install -q \\
        'transformers>=4.48' 'tokenizers>=0.20' 'accelerate>=0.26' \\
        'datasets>=2.14'     'scikit-learn>=1.0' 'tqdm>=4.60'       \\
        'PyYAML>=6.0'        'matplotlib>=3.4'   'tensorboard>=2.12'
"""))
cells.append(code("""
    import transformers, tokenizers, accelerate
    print(f'transformers : {transformers.__version__}')
    print(f'tokenizers   : {tokenizers.__version__}')
    print(f'accelerate   : {accelerate.__version__}')
"""))

# ── 2. Clone repo ──────────────────────────────────────────────────────────
cells.append(md(
    "## 2. Clone Source Code\n\n"
    f"Clones `{GITHUB_REPO}` and installs the `lanobert` package in-place."
))
cells.append(code(f"""
    import os, subprocess

    CLONE_DIR = '/kaggle/working/at-LAnoBERT'
    WORK_DIR  = f'{{CLONE_DIR}}/{REPO_SUBDIR}'   # contains lanobert/ and configs/

    if os.path.isdir(CLONE_DIR):
        print('Repo already cloned. Pulling latest...')
        subprocess.run(['git', 'pull'], cwd=CLONE_DIR, check=True)
    else:
        print('Cloning repo...')
        subprocess.run(['git', 'clone', '--depth', '1',
                        '{GITHUB_REPO}', CLONE_DIR], check=True)
        print('Clone done.')

    print('\\nRepo contents:')
    for item in sorted(os.listdir(WORK_DIR)):
        print(' ', item)
"""))
cells.append(code("""
    # Install the lanobert package so imports work from any cell
    result = subprocess.run(['pip', 'install', '-q', '-e', '.'],
                            cwd=WORK_DIR, capture_output=True, text=True)
    if result.returncode != 0:
        print('ERROR:', result.stderr[-2000:])
    else:
        print('lanobert package installed OK')

    # Add to sys.path as fallback
    import sys
    if WORK_DIR not in sys.path:
        sys.path.insert(0, WORK_DIR)
"""))

# ── 3. Download BGL ────────────────────────────────────────────────────────
cells.append(md(
    "## 3. Download BGL Dataset\n\n"
    "Source: [loghub – Zenodo record 8196385](https://zenodo.org/records/8196385)  \n"
    "Compressed size ≈ 748 MB · Uncompressed ≈ 708 MB"
))
cells.append(code("""
    import os

    DATA_DIR = '/kaggle/working/data/BGL'
    os.makedirs(DATA_DIR, exist_ok=True)
    BGL_ZIP = f'{DATA_DIR}/BGL.zip'
    BGL_LOG = f'{DATA_DIR}/BGL.log'

    if os.path.exists(BGL_LOG):
        print(f'BGL.log already present ({os.path.getsize(BGL_LOG)/1e9:.2f} GB)')
    else:
        print('Downloading BGL.zip from Zenodo ...')
        !wget -q --show-progress -c \\
            'https://zenodo.org/records/8196385/files/BGL.zip' -O {BGL_ZIP}
        print('Extracting ...')
        !unzip -q -o {BGL_ZIP} -d {DATA_DIR}
        print(f'Done. BGL.log = {os.path.getsize(BGL_LOG)/1e9:.2f} GB')
"""))
cells.append(code("""
    !wc -l {BGL_LOG}
    print('First 3 raw lines:')
    !head -3 {BGL_LOG}
"""))

# ── 4. Write BGL config with absolute Kaggle paths ─────────────────────────
cells.append(md(
    "## 4. Write BGL Config\n\n"
    "Identical to the author's `configs/bgl.yaml`, but with absolute Kaggle paths.  \n"
    "All hyperparameters are **unchanged** from the original."
))
cells.append(code("""
    import os

    BGL_CFG = \"\"\"# LAnoBERT — BGL (main configuration)
# Exact author config: from-scratch custom-vocab BERT, MLM-only, batch 32 / lr 1e-4.
dataset: BGL
run_name: bgl

paths:
  raw_log:      /kaggle/working/data/BGL/BGL.log
  train_raw:    /kaggle/working/data/BGL/BGL_train_normal.raw
  test_raw:     /kaggle/working/data/BGL/BGL_test.raw
  test_label:   /kaggle/working/data/BGL/BGL_test_label.log
  train_normal: /kaggle/working/data/BGL/BGL_train_normal_parsed.log
  test_log:     /kaggle/working/data/BGL/BGL_test_parsed.log
  tokenizer_dir: /kaggle/working/outputs/BGL/tokenizer
  model_dir:    /kaggle/working/outputs/BGL/model
  result_dir:   /kaggle/working/outputs/BGL/results

split:
  method: line        # BGL is a line-ordered stream
  train_ratio: 0.8    # first 80% chronologically → train region
  label_marker: \"-\"   # first token \"-\" => normal, else anomaly

preprocess:
  regex_profile: bgl  # faithful port of reference preprocess_bgl.py
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
\"\"\"

    CFG_PATH = f'{WORK_DIR}/configs/bgl_kaggle.yaml'
    with open(CFG_PATH, 'w') as f:
        f.write(BGL_CFG)
    print(f'Config written: {CFG_PATH}')
"""))

# ── 5. Full pipeline ────────────────────────────────────────────────────────
cells.append(md(
    "## 5. Run Full Pipeline\n\n"
    "Mirrors `bash scripts/run_pipeline.sh configs/bgl.yaml`  \n"
    "Each step checks if its output already exists and skips if so (safe to re-run)."
))

# Step 1 split
cells.append(md("### Step 1 — Chronological Split (80 / 20)"))
cells.append(code("""
    import os, sys
    sys.path.insert(0, WORK_DIR)

    from lanobert.utils import load_config
    from lanobert.split import run as split_run

    cfg  = load_config(CFG_PATH)
    trw  = cfg.get_path('paths.train_raw')

    if os.path.exists(trw) and os.path.getsize(trw) > 0:
        print(f'Split already done: {trw}')
    else:
        stats = split_run(cfg)
        print('Split stats:', stats)
"""))

# Step 2 preprocess
cells.append(md("### Step 2 — BGL Regex Preprocessing"))
cells.append(code("""
    from lanobert.preprocess import _options_from_config, preprocess_file

    opt        = _options_from_config(cfg)
    train_norm = cfg.get_path('paths.train_normal')
    test_log   = cfg.get_path('paths.test_log')

    if os.path.exists(train_norm) and os.path.getsize(train_norm) > 0:
        print(f'Train preprocess already done: {train_norm}')
    else:
        n = preprocess_file(cfg.get_path('paths.train_raw'), train_norm, opt)
        print(f'[preprocess] train: {n:,} lines written')

    if os.path.exists(test_log) and os.path.getsize(test_log) > 0:
        print(f'Test  preprocess already done: {test_log}')
    else:
        n = preprocess_file(cfg.get_path('paths.test_raw'), test_log, opt)
        print(f'[preprocess] test : {n:,} lines written')
"""))
cells.append(code("""
    # Sanity check — inspect preprocessed samples
    print('=== train (first 5 lines) ===')
    with open(train_norm) as f:
        for i, l in enumerate(f):
            if i >= 5: break
            print(f'  {i}: {l.rstrip()}')

    print('\\n=== test (first 5 lines) ===')
    with open(test_log) as f:
        for i, l in enumerate(f):
            if i >= 5: break
            print(f'  {i}: {l.rstrip()}')
"""))

# Step 3 tokenizer
cells.append(md(
    "### Step 3 — Train WordPiece Tokenizer\n\n"
    "**vocab_size=1000 · min_frequency=2 · lowercase=False**  \n"
    "Log-specific vocabulary — the key factor for BGL performance (see paper ablation)."
))
cells.append(code("""
    from lanobert.tokenizer import train_tokenizer, vocab_path_for, load_tokenizer

    vocab_file = vocab_path_for(cfg)

    if os.path.exists(vocab_file):
        print(f'Tokenizer already exists: {vocab_file}')
    else:
        vocab_file = train_tokenizer(cfg)

    tok = load_tokenizer(vocab_file, max_len=512)
    print(f'Vocab size : {tok.vocab_size}')
    print(f'First 30   : {list(tok.vocab.keys())[:30]}')
"""))

# Step 4 train
cells.append(md(
    "### Step 4 — MLM Pretraining (from scratch)\n\n"
    "**BERT-base architecture · batch=32 · lr=1e-4 · 10 epochs · mlm_prob=0.20**\n\n"
    "> ⏱ Estimated ~3–4 h on a Kaggle T4 GPU (single).  \n"
    "> 💡 Enable GPU: *Runtime → Change runtime type → GPU*"
))
cells.append(code("""
    from lanobert.train import train as lanobert_train

    model_final = os.path.join(cfg.get_path('paths.model_dir'), 'final')

    if os.path.isdir(model_final):
        print(f'Model already trained: {model_final}')
    else:
        lanobert_train(cfg, vocab_file=vocab_file)
        print('Training complete.')
"""))

# Step 5 inference
cells.append(md(
    "### Step 5 — Inference & Evaluation\n\n"
    "Scoring method: **error_mean** — mean cross-entropy over all masked words per line.  \n"
    "Expected: **AUROC ≈ 1.000 · Best-F1 ≈ 1.000**"
))
cells.append(code("""
    from lanobert.inference import run as inference_run

    results = inference_run(cfg)

    import json
    print('\\nFull results JSON:')
    print(json.dumps(results, indent=2))
"""))

# ── 6. Summary ─────────────────────────────────────────────────────────────
cells.append(md("## 6. Results Summary"))
cells.append(code("""
    import json, os
    from IPython.display import Image, display

    rdir   = cfg.get_path('paths.result_dir')
    rjson  = os.path.join(rdir, 'results.json')

    if os.path.exists(rjson):
        r = json.load(open(rjson))
        print('=' * 55)
        print('  LAnoBERT — BGL Baseline (Author Config)')
        print('=' * 55)
        print(f"  Score   : {r.get('score_name', 'error_mean')}")
        print(f"  AUROC   : {r['auroc']:.4f}   (target 1.0000)")
        print(f"  Best-F1 : {r['best_f1']:.4f}   (target 1.0000)")
        print(f"  AP      : {r['ap']:.4f}")
        print(f"  Thresh  : {r['threshold']:.4f}")
        print('=' * 55)

        roc_path = os.path.join(rdir, 'roc.png')
        if os.path.exists(roc_path):
            display(Image(roc_path))
    else:
        print('No results.json found — run the inference step first.')
"""))

# ── 7. Save archive ─────────────────────────────────────────────────────────
cells.append(md("## 7. Save Outputs Archive"))
cells.append(code("""
    import shutil, os

    arc = '/kaggle/working/lanobert_bgl_outputs.tar.gz'
    shutil.make_archive(arc.replace('.tar.gz', ''), 'gztar',
                        root_dir='/kaggle/working', base_dir='outputs')
    print(f'Archive: {arc}  ({os.path.getsize(arc)/1e6:.1f} MB)')
    print('Files inside:')
    import subprocess
    subprocess.run(['tar', '-tzf', arc])
"""))

# ── assemble ────────────────────────────────────────────────────────────────
nb = {
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {"name": "python", "version": "3.10.0"},
        "kaggle": {
            "accelerator": "gpu",
            "dataSources": [],
            "dockerImageVersionId": 30918,
            "isInternetConnected": True,
            "language": "python",
            "sourceType": "notebook"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5,
    "cells": cells
}

for i, cell in enumerate(nb["cells"]):
    cell["id"] = f"cell-{i:03d}"

out = "kaggle_lanobert_bgl.ipynb"
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"Written: {out}  ({len(nb['cells'])} cells)")
