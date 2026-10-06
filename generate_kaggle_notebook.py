"""Generate kaggle_lanobert_bgl.ipynb for running LAnoBERT BGL baseline."""
import json, textwrap

# ── helpers ────────────────────────────────────────────────────────────────
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

# ── cells ──────────────────────────────────────────────────────────────────
cells = []

cells.append(md(
    "# LAnoBERT — BGL Baseline (Author Config)\n\n"
    "**Paper**: [LAnoBERT: System Log Anomaly Detection based on BERT Masked Language Model (ASC 2023)](https://doi.org/10.1016/j.asoc.2023.110689)  \n"
    "**HF Hub**: [yukyung/LAnoBERT](https://huggingface.co/yukyung/LAnoBERT)\n\n"
    "Faithful reproduction of the **BGL main config**:\n"
    "- From-scratch BERT with **log-specific WordPiece vocab** (vocab_size=1000)\n"
    "- MLM pretraining on **normal logs only** (train_ratio=0.8, chronological)\n"
    "- batch=32 · lr=1e-4 · 10 epochs · mlm_probability=0.20\n"
    "- **error_mean** scoring (recommended by authors)\n\n"
    "Expected: **AUROC 1.000 / Best-F1 1.000**"
))

# 0. Environment
cells.append(md("## 0. Environment Check"))
cells.append(code("""
    import subprocess, sys, os, torch

    r = subprocess.run(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader'],
                       capture_output=True, text=True)
    print('GPU:', r.stdout.strip() or 'None')
    print(f'PyTorch : {torch.__version__}')
    print(f'CUDA    : {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'Device  : {torch.cuda.get_device_name(0)}')
        print(f'VRAM    : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')
"""))

# 1. Install
cells.append(md("## 1. Install Dependencies"))
cells.append(code("""
    %%capture
    !pip install -q \\
        'transformers>=4.48' 'tokenizers>=0.20' 'accelerate>=0.26' \\
        'datasets>=2.14' 'scikit-learn>=1.0' 'tqdm>=4.60' \\
        'PyYAML>=6.0' 'matplotlib>=3.4' 'tensorboard>=2.12'
    print('done')
"""))
cells.append(code("""
    import transformers, tokenizers, accelerate
    print(f'transformers : {transformers.__version__}')
    print(f'tokenizers   : {tokenizers.__version__}')
    print(f'accelerate   : {accelerate.__version__}')
"""))

# 2. Download
cells.append(md(
    "## 2. Download BGL Dataset\n\n"
    "Source: [loghub – Zenodo record 8196385](https://zenodo.org/records/8196385)  \n"
    "Compressed size ~748 MB"
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
        !wget -q --show-progress -c \\
            'https://zenodo.org/records/8196385/files/BGL.zip' -O {BGL_ZIP}
        !unzip -q -o {BGL_ZIP} -d {DATA_DIR}
        print(f'Done. size={os.path.getsize(BGL_LOG)/1e9:.2f} GB')
"""))
cells.append(code("""
    !wc -l {BGL_LOG}
    print('First 3 raw lines:')
    !head -3 {BGL_LOG}
"""))

# 3. Write source
cells.append(md("## 3. Write LAnoBERT Source\nAll modules are embedded directly so no git-clone is needed."))

# 3a. Paths / config
cells.append(code("""
    import os

    WORK_DIR = '/kaggle/working/LAnoBERT'
    os.makedirs(f'{WORK_DIR}/configs', exist_ok=True)
    os.makedirs(f'{WORK_DIR}/lanobert', exist_ok=True)

    BGL_CFG = \"\"\"dataset: BGL
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
    with open(f'{WORK_DIR}/configs/bgl.yaml', 'w') as f:
        f.write(BGL_CFG)
    print('bgl.yaml written')
"""))

# utils.py
cells.append(code(r"""
    UTILS = '''
from __future__ import annotations
import os, random
from typing import Any
import numpy as np, yaml

def ensure_dir(path):
    os.makedirs(path, exist_ok=True); return path

def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    try:
        import torch; torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    except ImportError: pass

class _Cfg:
    def __init__(self, data, source_path=None):
        self._data = data; self._source_path = source_path
        self._base = os.path.dirname(os.path.abspath(source_path)) if source_path else os.getcwd()
    def get(self, key, default=None): return self._data.get(key, default)
    def __getitem__(self, k): return self._data[k]
    def __contains__(self, k): return k in self._data
    def _resolve(self, dotted):
        v = self._data
        for p in dotted.split('.'):
            if not isinstance(v, dict) or p not in v: raise KeyError(dotted)
            v = v[p]
        return v
    def get_path(self, dotted):
        try: val = self._resolve(dotted)
        except KeyError: return None
        if val is None: return None
        return val if os.path.isabs(str(val)) else os.path.join(self._base, str(val))

def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    return _Cfg(data, source_path=path)
'''
    with open(f'{WORK_DIR}/lanobert/utils.py', 'w') as f: f.write(UTILS)
    print('utils.py')
"""))

# split.py
cells.append(code(r"""
    SPLIT = '''
from __future__ import annotations
import json, os, re
from collections import OrderedDict
from typing import List, Optional, Tuple
from tqdm import tqdm
from .utils import ensure_dir, load_config

def _write_lines(lines, path):
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        for ln in lines: f.write(ln if ln.endswith("\n") else ln + "\n")

def split_line_stream(raw_path, train_ratio, label_marker):
    with open(raw_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.rstrip("\n") for ln in f]
    cut = int(len(lines) * train_ratio)
    train_r, test_r = lines[:cut], lines[cut:]
    ok = lambda ln: ln[:1] == label_marker
    train_n  = [ln for ln in train_r if ok(ln)]
    test_n   = [ln for ln in test_r  if ok(ln)]
    abn_all  = [ln for ln in lines   if not ok(ln)]
    return train_n, test_n + abn_all, [0]*len(test_n) + [1]*len(abn_all)

def split_blocks(raw_path, label_csv, train_ratio, block_id_regex):
    import csv
    blk_re = re.compile(block_id_regex)
    lmap = {}
    with open(label_csv, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            lmap[row["BlockId"]] = 1 if row["Label"].strip().lower() == "anomaly" else 0
    blocks = OrderedDict()
    with open(raw_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in tqdm(f, desc="group blocks"):
            for blk in set(blk_re.findall(line)):
                blocks.setdefault(blk, []).append(line.strip())
    normals, anomalies = [], []
    for blk, logs in blocks.items():
        (anomalies if lmap.get(blk, 0) else normals).append(" ".join(logs))
    cut = int(len(normals) * train_ratio)
    test_n = normals[cut:]
    return normals[:cut], test_n + anomalies, [0]*len(test_n) + [1]*len(anomalies)

def run(cfg, train_ratio=None):
    scfg = cfg.get("split", {}) or {}
    method = str(scfg.get("method", "line"))
    ratio  = train_ratio if train_ratio is not None else float(scfg.get("train_ratio", 0.8))
    raw = cfg.get_path("paths.raw_log")
    trw = cfg.get_path("paths.train_raw")
    tsw = cfg.get_path("paths.test_raw")
    tlb = cfg.get_path("paths.test_label")
    print(f"[split] {cfg.get('dataset')} method={method} ratio={ratio}")
    if method == "block":
        tr, ts, lb = split_blocks(raw,
            label_csv=cfg.get_path("paths.test_label_csv") or scfg.get("label_csv"),
            train_ratio=ratio,
            block_id_regex=str(scfg.get("block_id_regex", r"blk_-?\d+")))
    else:
        tr, ts, lb = split_line_stream(raw, ratio, str(scfg.get("label_marker", "-")))
    _write_lines(tr, trw); _write_lines(ts, tsw)
    _write_lines([str(x) for x in lb], tlb)
    stats = {"train_normal": len(tr), "test_total": len(ts),
             "test_anomaly": int(sum(lb)), "test_normal": int(len(lb)-sum(lb))}
    with open(os.path.join(os.path.dirname(trw) or ".", "split_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[split] {stats}")
    return stats
'''
    with open(f'{WORK_DIR}/lanobert/split.py', 'w') as f: f.write(SPLIT)
    print('split.py')
"""))

# preprocess.py
cells.append(code(r"""
    PREPROC = r'''
from __future__ import annotations
import os, re, unicodedata
from dataclasses import dataclass
from typing import Iterable
from tqdm import tqdm
from .utils import ensure_dir

_PUNCT_RE = re.compile(r"([.!?])")

# BGL regexes (faithful port of preprocess_bgl.py)
_BGL_IP_RE       = re.compile(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(:\d{1,5})?")
_BGL_DATETIME_RE = re.compile(r"\d{1,4}\-\d{1,2}\-\d{1,2}-\d{1,2}.\d{1,2}.\d{1,2}.\d+")
_BGL_DATE_RE     = re.compile(r"\d{1,4}\.\d{1,2}\.\d{1,2}")
_BGL_PATH_RE     = re.compile(r".\S+(?=.[0-9a-zA-Z])(?=[/]).\S+")
_BGL_SERVER_RE   = re.compile(r"\S+(?=.*[0-9])(?=.*[a-zA-Z])(?=[:]+)\S+")
_BGL_SERVER2_RE  = re.compile(r"\S+(?=.*[0-9])(?=.*[a-zA-Z])(?=[-])\S+")
_BGL_ECID_RE     = re.compile(r"[A-Z0-9]{28}")
_BGL_SERIAL_RE   = re.compile(r"[a-zA-Z0-9]{48}")
_BGL_MEMORY_RE   = re.compile(r"0[xX][0-9a-fA-F]\S+")
_BGL_IAR_RE      = re.compile(r"[0-9a-fA-F]{8}")
_BGL_NUM_RE      = re.compile(r"(\d+)")
_BGL_NONALPHA_RE = re.compile(r"[^a-zA-Z<>]+")

# HDFS regexes
_HDFS_ID_RE       = re.compile(r"blk_.\d+")
_HDFS_IP_RE       = re.compile(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(:\d{1,5})?")
_HDFS_NUM_RE      = re.compile(r"\d*\d")
_HDFS_NONALPHA_RE = re.compile(r"[^a-zA-Z.!?]+")

def _u2a(s): return "".join(c for c in unicodedata.normalize("NFD",s) if unicodedata.category(c)!="Mn")

def _bgl(line, drop=3):
    t = _BGL_IP_RE.sub(" IP ", line)
    t = _BGL_DATETIME_RE.sub(" TIME ", t); t = _BGL_DATE_RE.sub(" TIME ", t)
    t = _BGL_PATH_RE.sub(" PATH ", t)
    t = _BGL_SERVER_RE.sub(" SERVER ", t); t = _BGL_SERVER2_RE.sub(" SERVER ", t)
    t = _BGL_ECID_RE.sub(" ECID ", t); t = _BGL_SERIAL_RE.sub(" SERIAL ", t)
    t = _BGL_MEMORY_RE.sub(" MEMORY ", t); t = _BGL_IAR_RE.sub(" IAR ", t)
    t = _BGL_NUM_RE.sub(" NUM ", t)
    s = _BGL_NONALPHA_RE.sub(" ", _u2a(t.lower().strip()))
    return " ".join(s.split()[drop:])

def _hdfs(line, drop=0):
    s = _u2a(line.lower().strip())
    s = _HDFS_ID_RE.sub("BLK", s); s = _HDFS_IP_RE.sub("IP", s); s = _HDFS_NUM_RE.sub("NUM", s)
    s = _PUNCT_RE.sub(" ", s); s = _HDFS_NONALPHA_RE.sub(" ", s).strip()
    return " ".join(s.split()[drop:]) if drop > 0 else s

@dataclass
class ParseOptions:
    mask_block_id: bool = True; mask_ip: bool = True; mask_number: bool = True
    drop_header_fields: int = 0; lowercase: bool = True; regex_profile: str = "generic"

def normalize_line(line, opt):
    if opt.regex_profile == "bgl":  return _bgl(line, opt.drop_header_fields)
    if opt.regex_profile == "hdfs": return _hdfs(line, opt.drop_header_fields)
    s = _u2a((line.strip().lower() if opt.lowercase else line.strip()))
    if opt.mask_block_id: s = re.sub(r"blk_-?\d+","BLK",s)
    if opt.mask_ip:       s = re.sub(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(:\d{1,5})?","IP",s)
    if opt.mask_number:   s = re.sub(r"\d+","NUM",s)
    s = _PUNCT_RE.sub(" ",s); s = re.sub(r"[^a-zA-Z.!?]+"," ",s).strip()
    return " ".join(s.split()[opt.drop_header_fields:]) if opt.drop_header_fields else s

def _opts(cfg):
    pre = cfg.get("preprocess",{}) or {}
    return ParseOptions(
        mask_block_id=bool(pre.get("mask_block_id",True)),
        mask_ip=bool(pre.get("mask_ip",True)),
        mask_number=bool(pre.get("mask_number",True)),
        drop_header_fields=int(pre.get("drop_header_fields",0)),
        lowercase=bool(pre.get("lowercase",True)),
        regex_profile=str(pre.get("regex_profile","generic")),
    )

def preprocess_file(in_path, out_path, opt):
    ensure_dir(os.path.dirname(out_path) or ".")
    n = 0
    with open(out_path,"w",encoding="utf-8") as out:
        with open(in_path,"r",encoding="utf-8",errors="ignore") as f:
            for line in tqdm(f, desc=f"parse {os.path.basename(in_path)}"):
                norm = normalize_line(line, opt)
                if norm: out.write(norm+"\n"); n += 1
    return n

def _options_from_config(cfg): return _opts(cfg)
'''
    with open(f'{WORK_DIR}/lanobert/preprocess.py', 'w') as f: f.write(PREPROC)
    print('preprocess.py')
"""))

# tokenizer.py
cells.append(code(r"""
    TOK = '''
from __future__ import annotations
import os
from .utils import ensure_dir

def vocab_path_for(cfg):
    return os.path.join(cfg.get_path("paths.tokenizer_dir"), "vocab.txt")

def train_tokenizer(cfg):
    from tokenizers import BertWordPieceTokenizer
    tc = cfg.get("tokenizer",{})
    vocab_size  = int(tc.get("vocab_size", 1000))
    min_freq    = int(tc.get("min_frequency", 2))
    lowercase   = bool(tc.get("lowercase", False))
    train_file  = cfg.get_path("paths.train_normal")
    tok_dir     = cfg.get_path("paths.tokenizer_dir")
    ensure_dir(tok_dir)
    print(f"[tokenizer] vocab_size={vocab_size} min_freq={min_freq} lowercase={lowercase}")
    t = BertWordPieceTokenizer(lowercase=lowercase)
    t.train(files=[train_file], vocab_size=vocab_size, min_frequency=min_freq,
            special_tokens=["[PAD]","[UNK]","[CLS]","[SEP]","[MASK]"])
    t.save_model(tok_dir)
    print(f"[tokenizer] saved -> {tok_dir}")
    return vocab_path_for(cfg)

def load_tokenizer(vocab_file, max_len=512):
    from transformers import BertTokenizerFast
    return BertTokenizerFast(vocab_file=vocab_file, do_lower_case=False, model_max_length=max_len)
'''
    with open(f'{WORK_DIR}/lanobert/tokenizer.py', 'w') as f: f.write(TOK)
    print('tokenizer.py')
"""))

# dataset.py
cells.append(code(r"""
    DS = '''
from torch.utils.data import Dataset

class LogLineDataset(Dataset):
    def __init__(self, tokenizer, file_path, max_len=512):
        self.tokenizer = tokenizer; self.max_len = max_len
        with open(file_path,"r",encoding="utf-8",errors="ignore") as f:
            self.lines = [ln.strip() for ln in f if ln.strip()]
        print(f"[dataset] {len(self.lines):,} lines loaded")
    def __len__(self): return len(self.lines)
    def __getitem__(self, idx):
        return dict(self.tokenizer(self.lines[idx], max_length=self.max_len,
                    truncation=True, padding=False, return_special_tokens_mask=True))
'''
    with open(f'{WORK_DIR}/lanobert/dataset.py', 'w') as f: f.write(DS)
    print('dataset.py')
"""))

# metrics.py
cells.append(code(r"""
    MET = '''
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, f1_score

def best_f1(scores, labels):
    pr, rc, thr = precision_recall_curve(labels, scores)
    f1 = 2*pr*rc/(pr+rc+1e-9); idx = int(np.argmax(f1))
    return float(f1[idx]), float(thr[idx] if idx < len(thr) else thr[-1])

def evaluate(scores, labels, score_name="score"):
    auroc = float(roc_auc_score(labels, scores))
    ap    = float(average_precision_score(labels, scores))
    bf1, thresh = best_f1(scores, labels)
    return {"score_name": score_name, "auroc": auroc, "ap": ap,
            "best_f1": bf1, "threshold": thresh,
            "f1_at_best_thresh": float(f1_score(labels, (scores>=thresh).astype(int), zero_division=0))}
'''
    with open(f'{WORK_DIR}/lanobert/metrics.py', 'w') as f: f.write(MET)
    print('metrics.py')
"""))

# train.py
cells.append(code(r"""
    TRAIN = '''
from __future__ import annotations
import os
from .dataset   import LogLineDataset
from .tokenizer import load_tokenizer, vocab_path_for
from .utils     import ensure_dir, set_seed

def build_model(vocab_size, max_len, attn="sdpa"):
    from transformers import BertConfig, BertForMaskedLM
    cfg = BertConfig(vocab_size=vocab_size, max_position_embeddings=max_len)
    try:    return BertForMaskedLM(config=cfg, attn_implementation=attn)
    except: return BertForMaskedLM(config=cfg)

def train(cfg, vocab_file=None):
    import torch
    from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments
    from torch.utils.data import random_split

    tc = cfg.get("train", {})
    set_seed(int(tc.get("seed", 42)))
    vocab_file = vocab_file or vocab_path_for(cfg)
    max_len    = int(tc.get("max_len", 512))

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32       = True

    tokenizer = load_tokenizer(vocab_file, max_len=max_len)
    print(f"[train] tokenizer vocab={tokenizer.vocab_size}")
    model = build_model(tokenizer.vocab_size, max_len, str(tc.get("attn_implementation","sdpa")))
    print(f"[train] params={model.num_parameters():,}")

    full_ds   = LogLineDataset(tokenizer, cfg.get_path("paths.train_normal"), max_len)
    eval_sz   = max(1, int(len(full_ds) * float(tc.get("eval_ratio", 0.01))))
    tr_sz     = len(full_ds) - eval_sz
    tr_ds, ev_ds = random_split(full_ds, [tr_sz, eval_sz],
                   generator=torch.Generator().manual_seed(int(tc.get("seed",42))))
    print(f"[train] train={tr_sz:,}  eval={eval_sz:,}")

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True,
        mlm_probability=float(tc.get("mlm_probability", 0.15)),
        pad_to_multiple_of=8)

    model_dir  = ensure_dir(cfg.get_path("paths.model_dir"))
    eval_steps = int(tc.get("eval_steps", tc.get("save_steps", 50000)))
    seed       = int(tc.get("seed", 42))

    args = TrainingArguments(
        output_dir=model_dir, overwrite_output_dir=True,
        seed=seed, data_seed=seed,
        num_train_epochs=float(tc.get("num_train_epochs", 10)),
        per_device_train_batch_size=int(tc.get("per_device_train_batch_size", 8)),
        per_device_eval_batch_size=int(tc.get("per_device_eval_batch_size", 64)),
        learning_rate=float(tc.get("learning_rate", 5e-5)),
        weight_decay=float(tc.get("weight_decay", 0.01)),
        warmup_ratio=float(tc.get("warmup_ratio", 0.1)),
        lr_scheduler_type=str(tc.get("lr_scheduler_type", "cosine")),
        adam_beta2=float(tc.get("adam_beta2", 0.98)),
        adam_epsilon=float(tc.get("adam_epsilon", 1e-6)),
        bf16=bool(tc.get("bf16", torch.cuda.is_available())),
        eval_strategy="steps", eval_steps=eval_steps,
        save_strategy="steps", save_steps=eval_steps,
        save_total_limit=int(tc.get("save_total_limit", 2)),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss", greater_is_better=False,
        logging_steps=int(tc.get("logging_steps", 1000)),
        logging_dir=os.path.join(model_dir, "logs"),
        dataloader_num_workers=4,
        report_to=["tensorboard"],
    )
    trainer = Trainer(model=model, args=args, data_collator=collator,
                      train_dataset=tr_ds, eval_dataset=ev_ds)
    print("[train] start")
    trainer.train()
    final = os.path.join(model_dir, "final")
    trainer.save_model(final); tokenizer.save_pretrained(final)
    print(f"[train] saved -> {final}")
    return final
'''
    with open(f'{WORK_DIR}/lanobert/train.py', 'w') as f: f.write(TRAIN)
    print('train.py')
"""))

# inference.py
cells.append(code(r"""
    INF = '''
from __future__ import annotations
import json, os
from typing import List, Optional
import numpy as np, torch
from tqdm import tqdm
from transformers import BertForMaskedLM
from .metrics  import evaluate
from .tokenizer import load_tokenizer, vocab_path_for
from .utils    import ensure_dir

def _load(cfg):
    ic = cfg.get("inference",{})
    tc = cfg.get("train",{})
    max_len = int(tc.get("max_len",512))
    if ic.get("hf_model"):
        from transformers import AutoModelForMaskedLM, AutoTokenizer
        kw = {"subfolder": ic["hf_subfolder"]} if ic.get("hf_subfolder") else {}
        return AutoModelForMaskedLM.from_pretrained(ic["hf_model"],**kw), \
               AutoTokenizer.from_pretrained(ic["hf_model"],**kw)
    if ic.get("pretrained_model"):
        from transformers import AutoModelForMaskedLM, AutoTokenizer
        return AutoModelForMaskedLM.from_pretrained(ic["pretrained_model"]), \
               AutoTokenizer.from_pretrained(ic["pretrained_model"])
    mdir = os.path.join(cfg.get_path("paths.model_dir"),"final")
    vf   = os.path.join(mdir,"vocab.txt")
    if not os.path.exists(vf): vf = vocab_path_for(cfg)
    return BertForMaskedLM.from_pretrained(mdir), load_tokenizer(vf, max_len=max_len)

def score_lines(model, tok, lines, batch_size=16, max_len=512):
    import torch.nn.functional as F
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device).eval()
    mask_id = tok.mask_token_id
    special = {tok.cls_token_id, tok.sep_token_id, tok.pad_token_id}
    scores  = []
    with torch.no_grad():
        for line in tqdm(lines, desc="scoring"):
            enc  = tok(line, max_length=max_len, truncation=True, return_tensors="pt")
            ids  = enc["input_ids"][0]
            attn = enc["attention_mask"][0]
            widx = [i for i,t in enumerate(ids.tolist()) if t not in special]
            if not widx: scores.append(0.0); continue
            ws = []
            for s in range(0, len(widx), batch_size):
                chunk = widx[s:s+batch_size]; B = len(chunk)
                bid   = ids.unsqueeze(0).repeat(B,1).to(device)
                bat   = attn.unsqueeze(0).repeat(B,1).to(device)
                lbl   = torch.full_like(bid,-100)
                for r,p in enumerate(chunk):
                    lbl[r,p] = bid[r,p]; bid[r,p] = mask_id
                out = model(input_ids=bid, attention_mask=bat, labels=lbl)
                for r,p in enumerate(chunk):
                    ws.append(F.cross_entropy(out.logits[r,p].unsqueeze(0),
                                              lbl[r,p].unsqueeze(0)).item())
            scores.append(float(np.mean(ws)))
    return np.array(scores, dtype=np.float32)

def run(cfg):
    ic = cfg.get("inference",{}); tc = cfg.get("train",{})
    max_len    = int(tc.get("max_len",512))
    batch_size = int(ic.get("batch_size",16))
    max_eval   = ic.get("max_eval_samples",None)

    rdir = ensure_dir(cfg.get_path("paths.result_dir"))
    with open(cfg.get_path("paths.test_log"),  "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    with open(cfg.get_path("paths.test_label"), "r", encoding="utf-8") as f:
        labels = [int(ln.strip()) for ln in f if ln.strip()]
    if max_eval: lines = lines[:int(max_eval)]; labels = labels[:int(max_eval)]
    labels = np.array(labels, dtype=int)
    print(f"[inference] {len(lines):,} test examples  anomaly={labels.sum():,}")

    model, tok = _load(cfg)
    scores = score_lines(model, tok, lines, batch_size=batch_size, max_len=max_len)
    np.save(f"{rdir}/scores_error_mean.npy", scores)
    np.save(f"{rdir}/labels.npy", labels)

    res = evaluate(scores, labels, "error_mean")
    print(f"\n{'='*55}")
    print(f"  Dataset : {cfg.get('dataset')}")
    print(f"  AUROC   : {res['auroc']:.4f}  (target 1.0000)")
    print(f"  Best-F1 : {res['best_f1']:.4f}  (target 1.0000)")
    print(f"  AP      : {res['ap']:.4f}")
    print(f"{'='*55}\n")
    with open(f"{rdir}/results.json","w") as f: json.dump(res,f,indent=2)

    try:
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve
        fpr,tpr,_ = roc_curve(labels, scores)
        plt.figure(figsize=(6,5))
        plt.plot(fpr,tpr,lw=2,label=f"error_mean AUROC={res['auroc']:.4f}")
        plt.plot([0,1],[0,1],"k--"); plt.xlabel("FPR"); plt.ylabel("TPR")
        plt.title(f"LAnoBERT — {cfg.get('dataset')} ROC"); plt.legend(); plt.tight_layout()
        plt.savefig(f"{rdir}/roc.png", dpi=150); plt.show()
    except Exception as e:
        print(f"[inference] plot skipped: {e}")
    return res
'''
    with open(f'{WORK_DIR}/lanobert/inference.py', 'w') as f: f.write(INF)
    print('inference.py')
"""))

# __init__ + setup
cells.append(code(r"""
    with open(f'{WORK_DIR}/lanobert/__init__.py', 'w') as f:
        f.write('from .utils import load_config, set_seed\n')
    with open(f'{WORK_DIR}/setup.py', 'w') as f:
        f.write('from setuptools import setup,find_packages\n'
                'setup(name="lanobert",version="1.0",packages=find_packages())\n')

    import subprocess
    r = subprocess.run(['pip','install','-q','-e','.'], cwd=WORK_DIR,
                       capture_output=True, text=True)
    if r.returncode: print('ERR:', r.stderr[-1000:])
    else: print('lanobert package installed OK')
"""))

# 4. Pipeline
cells.append(md(
    "## 4. Run Full Pipeline\n\n"
    "Mirrors `bash scripts/run_pipeline.sh configs/bgl.yaml`"
))

# Step 1 split
cells.append(md("### Step 1 — Chronological Split (80 / 20)"))
cells.append(code("""
    import sys, os
    sys.path.insert(0, WORK_DIR)

    from lanobert.utils import load_config
    from lanobert.split import run as split_run

    cfg = load_config(f'{WORK_DIR}/configs/bgl.yaml')
    trw = cfg.get_path('paths.train_raw')

    if os.path.exists(trw) and os.path.getsize(trw) > 0:
        print(f'Split already done: {trw}')
    else:
        stats = split_run(cfg)
        print('Stats:', stats)
"""))

# Step 2 preprocess
cells.append(md("### Step 2 — BGL Regex Preprocessing"))
cells.append(code("""
    from lanobert.preprocess import _options_from_config, preprocess_file

    opt        = _options_from_config(cfg)
    train_norm = cfg.get_path('paths.train_normal')
    test_log   = cfg.get_path('paths.test_log')

    if not (os.path.exists(train_norm) and os.path.getsize(train_norm) > 0):
        n = preprocess_file(cfg.get_path('paths.train_raw'), train_norm, opt)
        print(f'train: {n:,} lines')
    else:
        print(f'Train preprocess done: {train_norm}')

    if not (os.path.exists(test_log) and os.path.getsize(test_log) > 0):
        n = preprocess_file(cfg.get_path('paths.test_raw'), test_log, opt)
        print(f'test:  {n:,} lines')
    else:
        print(f'Test preprocess done:  {test_log}')
"""))
cells.append(code("""
    print('=== train sample (5 lines) ===')
    with open(train_norm) as f:
        for i,l in enumerate(f):
            if i>=5: break
            print(f' {i}: {l.rstrip()}')

    print('\\n=== test sample (5 lines) ===')
    with open(test_log) as f:
        for i,l in enumerate(f):
            if i>=5: break
            print(f' {i}: {l.rstrip()}')
"""))

# Step 3 tokenizer
cells.append(md(
    "### Step 3 — WordPiece Tokenizer\n"
    "**vocab_size=1000 · min_frequency=2 · lowercase=False**"
))
cells.append(code("""
    from lanobert.tokenizer import train_tokenizer, vocab_path_for, load_tokenizer

    vocab_file = vocab_path_for(cfg)
    if os.path.exists(vocab_file):
        print(f'Tokenizer exists: {vocab_file}')
    else:
        vocab_file = train_tokenizer(cfg)

    tok = load_tokenizer(vocab_file, max_len=512)
    print(f'Vocab size: {tok.vocab_size}')
    print('First 30 tokens:', list(tok.vocab.keys())[:30])
"""))

# Step 4 train
cells.append(md(
    "### Step 4 — MLM Pretraining (from scratch)\n"
    "**BERT-base arch · batch=32 · lr=1e-4 · 10 epochs · mlm_prob=0.20**\n\n"
    "> ⏱ Expected ~3-4 h on a single T4 GPU."
))
cells.append(code("""
    from lanobert.train import train as lanobert_train

    model_final = os.path.join(cfg.get_path('paths.model_dir'), 'final')
    if os.path.isdir(model_final):
        print(f'Already trained: {model_final}')
    else:
        lanobert_train(cfg, vocab_file=vocab_file)
"""))

# Step 5 inference
cells.append(md(
    "### Step 5 — Inference & Evaluation\n"
    "Scoring: **error_mean** (mean cross-entropy per masked word per line)"
))
cells.append(code("""
    from lanobert.inference import run as inference_run

    results = inference_run(cfg)
    import json
    print(json.dumps(results, indent=2))
"""))

# 5. Summary
cells.append(md("## 5. Results Summary"))
cells.append(code("""
    import json, os
    from IPython.display import Image, display

    rdir = cfg.get_path('paths.result_dir')
    rjson = os.path.join(rdir, 'results.json')
    if os.path.exists(rjson):
        r = json.load(open(rjson))
        print('='*55)
        print('  LAnoBERT — BGL Baseline (Author Config)')
        print('='*55)
        print(f"  Score   : {r.get('score_name','error_mean')}")
        print(f"  AUROC   : {r['auroc']:.4f}   (target 1.0000)")
        print(f"  Best-F1 : {r['best_f1']:.4f}   (target 1.0000)")
        print(f"  AP      : {r['ap']:.4f}")
        print(f"  Thresh  : {r['threshold']:.4f}")
        print('='*55)
        roc = os.path.join(rdir,'roc.png')
        if os.path.exists(roc): display(Image(roc))
"""))

# 6. Archive outputs
cells.append(md("## 6. Save Outputs as Archive"))
cells.append(code("""
    import shutil
    arc = '/kaggle/working/lanobert_bgl_outputs.tar.gz'
    shutil.make_archive(arc.replace('.tar.gz',''), 'gztar',
                        root_dir='/kaggle/working', base_dir='outputs')
    print(f'Saved: {arc}  ({os.path.getsize(arc)/1e6:.1f} MB)')
"""))

# ── assemble notebook ───────────────────────────────────────────────────────
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

# assign sequential IDs
for i, cell in enumerate(nb["cells"]):
    cell["id"] = f"cell-{i:03d}"

out_path = "kaggle_lanobert_bgl.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"Written: {out_path}  ({len(nb['cells'])} cells)")
