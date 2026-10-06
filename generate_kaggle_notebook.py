"""Generate kaggle_lanobert_bgl.ipynb.

Strategy:
  - Steps 1-3 (split/preprocess/tokenizer) → chạy LOCAL bằng run_local_prep.py
  - Output được upload lên Kaggle làm Dataset input
  - Notebook Kaggle chỉ chạy Step 4 (train GPU) + Step 5 (inference + eval)
"""
import json, textwrap

GITHUB_REPO  = "https://github.com/rubyhcm/at-LAnoBERT"
REPO_SUBDIR  = "LAnoBERT"
KAGGLE_INPUT = "/kaggle/input/bgl-preprocessed/bgl-preprocessed"   # tên dataset trên Kaggle

def md(src):
    return {"cell_type": "markdown", "id": None, "metadata": {},
            "source": src if isinstance(src, list) else [src]}

def code(src):
    src = textwrap.dedent(src).lstrip("\n")
    lines = [l + "\n" for l in src.splitlines()]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    return {"cell_type": "code", "execution_count": None, "id": None,
            "metadata": {}, "outputs": [], "source": lines}

cells = []

# ── Title ───────────────────────────────────────────────────────────────────
cells.append(md(
    "# LAnoBERT — BGL Baseline (Author Config)\n\n"
    f"**Source**: [{GITHUB_REPO}]({GITHUB_REPO})  \n"
    "**Paper**: [LAnoBERT: System Log Anomaly Detection based on BERT Masked Language Model (ASC 2023)](https://doi.org/10.1016/j.asoc.2023.110689)\n\n"
    "## Workflow\n"
    "```\n"
    "LOCAL  → run_local_prep.py  (split + preprocess + tokenizer)  → kaggle_input/\n"
    "KAGGLE → notebook này        (train GPU + inference + eval)\n"
    "```\n\n"
    "Input dataset (`/kaggle/input/bgl-preprocessed/`) chứa:\n"
    "- `data/BGL/BGL_train_normal_parsed.log`  — train corpus (đã normalize)\n"
    "- `data/BGL/BGL_test_parsed.log`           — test corpus  (đã normalize)\n"
    "- `data/BGL/BGL_test_label.log`            — 0/1 labels\n"
    "- `tokenizer/BGL_LogBERT-vocab.txt`        — WordPiece vocab (1000 tokens)\n"
    "- `bgl.yaml`                               — config file\n\n"
    "Expected results: **AUROC 1.000 / Best-F1 1.000**"
))

# ── 0. Env ──────────────────────────────────────────────────────────────────
cells.append(md("## 0. Environment Check"))
cells.append(code("""
    import subprocess, torch

    r = subprocess.run(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader'],
                       capture_output=True, text=True)
    print('GPU    :', r.stdout.strip() or 'None — enable GPU accelerator!')
    print(f'PyTorch: {torch.__version__}')
    print(f'CUDA   : {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f'Device : {p.name}  VRAM={p.total_memory/1e9:.1f} GB')
"""))

# ── 1. Install ───────────────────────────────────────────────────────────────
cells.append(md("## 1. Install Dependencies"))
cells.append(code("""
    %%capture
    !pip install -q \\
        'transformers>=4.48' 'tokenizers>=0.20' 'accelerate>=0.26' \\
        'scikit-learn>=1.0'  'tqdm>=4.60'       'PyYAML>=6.0'       \\
        'matplotlib>=3.4'    'tensorboard>=2.12'
"""))
cells.append(code("""
    import transformers, tokenizers, accelerate
    print(f'transformers : {transformers.__version__}')
    print(f'tokenizers   : {tokenizers.__version__}')
    print(f'accelerate   : {accelerate.__version__}')
"""))

# ── 2. Clone repo ────────────────────────────────────────────────────────────
cells.append(md(
    "## 2. Clone Source Code\n\n"
    f"Clones `{GITHUB_REPO}` và install package `lanobert`."
))
cells.append(code(f"""
    import os, subprocess, sys

    CLONE_DIR = '/kaggle/working/at-LAnoBERT'
    WORK_DIR  = f'{{CLONE_DIR}}/{REPO_SUBDIR}'

    if os.path.isdir(CLONE_DIR):
        print('Repo đã clone. Pulling latest...')
        subprocess.run(['git','pull'], cwd=CLONE_DIR, check=True)
    else:
        subprocess.run(['git','clone','--depth','1',
                        '{GITHUB_REPO}', CLONE_DIR], check=True)

    if WORK_DIR not in sys.path:
        sys.path.insert(0, WORK_DIR)
        print(f'Đã thêm {{WORK_DIR}} vào sys.path')

"""))

# ── 3. Verify input data ─────────────────────────────────────────────────────
cells.append(md(
    "## 3. Verify Input Dataset\n\n"
    f"Kiểm tra dữ liệu được upload từ local (`{KAGGLE_INPUT}/`)."
))
cells.append(code(f"""
    import os

    INPUT_DIR  = '{KAGGLE_INPUT}'
    TRAIN_FILE = f'{{INPUT_DIR}}/data/BGL/BGL_train_normal_parsed.log'
    TEST_FILE  = f'{{INPUT_DIR}}/data/BGL/BGL_test_parsed.log'
    LABEL_FILE = f'{{INPUT_DIR}}/data/BGL/BGL_test_label.log'
    VOCAB_FILE = f'{{INPUT_DIR}}/tokenizer/BGL_LogBERT-vocab.txt'
    SRC_CONFIG = f'{{INPUT_DIR}}/bgl.yaml'

    files = {{
        'Train corpus' : TRAIN_FILE,
        'Test corpus'  : TEST_FILE,
        'Test labels'  : LABEL_FILE,
        'Vocab'        : VOCAB_FILE,
        'Config'       : SRC_CONFIG,
    }}

    all_ok = True
    for name, path in files.items():
        if os.path.exists(path):
            sz = os.path.getsize(path)
            print(f'  ✅ {{name:15s}} {{sz/1e6:8.2f}} MB  {{path}}')
        else:
            print(f'  ❌ {{name:15s}} NOT FOUND: {{path}}')
            all_ok = False

    if all_ok:
        print('\\n✅ All input files present.')
    else:
        print('\\n❌ Missing files! Hãy chạy run_local_prep.py và upload kaggle_input/')
"""))
cells.append(code(f"""
    # Sanity-check: in thử vài dòng train và test
    import subprocess
    print('=== train sample (3 lines) ===')
    !head -3 {{TRAIN_FILE}}
    print('\\n=== test sample (3 lines) ===')
    !head -3 {{TEST_FILE}}

    with open(LABEL_FILE) as f:
        labels = [int(l.strip()) for l in f if l.strip()]
    n_anom = sum(labels)
    print(f'\\nTest labels: {{len(labels):,}} total  |  anomaly={{n_anom:,}}  normal={{len(labels)-n_anom:,}}')
"""))

# ── 4. Write Kaggle config ───────────────────────────────────────────────────
cells.append(md(
    "## 4. Write BGL Config\n\n"
    "Config giữ **nguyên** hyperparameters của tác giả, chỉ trỏ paths đến Kaggle working dirs."
))
cells.append(code(f"""
    import os

    OUTPUT_DIR = '/kaggle/working/outputs/BGL'
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    BGL_CFG = f\"\"\"# LAnoBERT — BGL (Kaggle config: train+inference only)
# Preprocessed data comes from Dataset input.
# Hyperparameters are IDENTICAL to the author's original bgl.yaml.
dataset: BGL
run_name: bgl

paths:
  raw_log:      {KAGGLE_INPUT}/data/BGL/BGL.log        # không dùng (đã preprocess)
  train_raw:    {KAGGLE_INPUT}/data/BGL/BGL_train_normal.raw
  test_raw:     {KAGGLE_INPUT}/data/BGL/BGL_test.raw
  test_label:   {KAGGLE_INPUT}/data/BGL/BGL_test_label.log
  train_normal: {KAGGLE_INPUT}/data/BGL/BGL_train_normal_parsed.log
  test_log:     {KAGGLE_INPUT}/data/BGL/BGL_test_parsed.log
  tokenizer_dir: {KAGGLE_INPUT}/tokenizer
  model_dir:    /kaggle/working/outputs/BGL/model
  result_dir:   /kaggle/working/outputs/BGL/results

split:
  method: line
  train_ratio: 0.8
  label_marker: \"-\"

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
\"\"\"

    CFG_PATH = '/kaggle/working/bgl_kaggle.yaml'
    with open(CFG_PATH, 'w') as f:
        f.write(BGL_CFG)
    print(f'Config written: {{CFG_PATH}}')

    from lanobert.utils import load_config
    cfg = load_config(CFG_PATH)
    print(f'Dataset     : {{cfg.get(\"dataset\")}}')
    print(f'Train file  : {{cfg.get_path(\"paths.train_normal\")}}')
    print(f'Tokenizer   : {{cfg.get_path(\"paths.tokenizer_dir\")}}')
    print(f'Model out   : {{cfg.get_path(\"paths.model_dir\")}}')
"""))

# ── 5. Train ─────────────────────────────────────────────────────────────────
cells.append(md(
    "## 5. MLM Pretraining (from scratch)\n\n"
    "**BERT-base architecture · batch=32 · lr=1e-4 · 10 epochs · mlm_prob=0.20**\n\n"
    "> ⏱ Ước tính ~3–4 h trên T4 GPU.  \n"
    "> 💡 Bật GPU: *Settings (⚙️) → Accelerator → GPU T4 x2 hoặc P100*"
))
cells.append(code("""
    import os
    from lanobert.train import train as lanobert_train

    model_final = os.path.join(cfg.get_path('paths.model_dir'), 'final')

    if os.path.isdir(model_final):
        print(f'✅ Model đã train: {model_final}')
    else:
        print('🚀 Bắt đầu training...')
        lanobert_train(cfg)
        print(f'✅ Training xong → {model_final}')
"""))

# ── 6. Inference ─────────────────────────────────────────────────────────────
cells.append(md(
    "## 6. Inference & Evaluation\n\n"
    "Scoring: **error_mean** (mean cross-entropy trên từng masked word/line)  \n"
    "Expected: **AUROC 1.000 / Best-F1 1.000**"
))
cells.append(code("""
    from lanobert.inference import run as inference_run

    results = inference_run(cfg)

    import json
    print('\\nResults JSON:')
    print(json.dumps(results, indent=2))
"""))

# ── 7. Summary ───────────────────────────────────────────────────────────────
cells.append(md("## 7. Results Summary"))
cells.append(code("""
    import json, os
    from IPython.display import Image, display

    rdir  = cfg.get_path('paths.result_dir')
    rjson = os.path.join(rdir, 'results.json')

    if os.path.exists(rjson):
        r = json.load(open(rjson))
        print('=' * 55)
        print('  LAnoBERT — BGL Baseline Results')
        print('=' * 55)
        print(f"  Score   : {r.get('score_name','error_mean')}")
        print(f"  AUROC   : {r['auroc']:.4f}   (target 1.0000)")
        print(f"  Best-F1 : {r['best_f1']:.4f}   (target 1.0000)")
        print(f"  AP      : {r['ap']:.4f}")
        print(f"  Thresh  : {r['threshold']:.4f}")
        print('=' * 55)
        roc = os.path.join(rdir, 'roc.png')
        if os.path.exists(roc):
            display(Image(roc))
    else:
        print('results.json chưa có — chạy inference trước.')
"""))

# ── 8. Save archive ───────────────────────────────────────────────────────────
cells.append(md("## 8. Save Outputs Archive"))
cells.append(code("""
    import shutil, os

    arc = '/kaggle/working/lanobert_bgl_outputs.tar.gz'
    shutil.make_archive(arc.replace('.tar.gz',''), 'gztar',
                        root_dir='/kaggle/working', base_dir='outputs')
    print(f'Archive: {arc}  ({os.path.getsize(arc)/1e6:.1f} MB)')
"""))

# ── Assemble notebook ─────────────────────────────────────────────────────────
nb = {
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
        "kaggle": {
            "accelerator": "gpu",
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
for i, c in enumerate(nb["cells"]):
    c["id"] = f"cell-{i:03d}"

out = "kaggle_lanobert_bgl.ipynb"
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"Written: {out}  ({len(nb['cells'])} cells)")
