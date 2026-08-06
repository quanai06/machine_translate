"""Kaggle GPU — Back-Translation Pipeline EN→VI (Transformer).
"""
from __future__ import annotations
import base64 as _b64, io as _io, os as _os, sys as _sys, zipfile as _zf
from pathlib import Path as _Path

try:
    ROOT = _Path(__file__).resolve().parent  # script mode
except NameError:
    ROOT = _Path("/kaggle/working") if _Path("/kaggle/working").exists() else _Path(".")  # notebook mode
_EXTRACT = _Path("/kaggle/working/.modules") if _Path("/kaggle/working").exists() else (ROOT / ".modules_extracted")
if not _EXTRACT.exists():
    _EXTRACT.mkdir(parents=True, exist_ok=True)
    _DATA = "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"
    with _zf.ZipFile(_io.BytesIO(_b64.b64decode(_DATA))) as zf:
        zf.extractall(_EXTRACT)
_sys.path.insert(0, str(_EXTRACT))

if _Path("/kaggle/working").exists():
    import subprocess as _sp
    _sp.run([_sys.executable, "-m", "pip", "install", "-q",
             "sentencepiece>=0.2.0", "sacrebleu>=2.4.0"], check=False)

import argparse
import math
import random
import shutil
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm

try:
    ROOT = Path(__file__).resolve().parent  # script mode
except NameError:
    ROOT = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")  # notebook mode
sys.path.insert(0, str(ROOT))

from common.cli import get_device, set_seed, print_header
from common.data import (EOS_ID, PAD_ID, BOS_ID,
                         TranslationDataset, load_split, collate_batch)
from common.engine import (InverseSqrtSchedule, load_best,
                           run_training, score_split, translate_dataset)
from common.metrics import corpus_bleu, detokenize
from common.tokenizer import load_tokenizers
from translate_transformers.config import TransformerConfig
from translate_transformers.model import build_model

# ── Detect Kaggle vs local ────────────────────────────────────────
_ON_KAGGLE = Path("/kaggle/working").exists()
if _ON_KAGGLE:
    DEFAULT_DATA_DIR = "/kaggle/input/datasets/leehoangquan006/iwslt15-envi-data"
    DEFAULT_TOKENIZER_DIR = "/kaggle/input/datasets/leehoangquan006/iwslt15-envi-data"
    DEFAULT_OUTPUT_DIR = "/kaggle/working/runs"
else:
    DEFAULT_DATA_DIR = str(ROOT / "data" / "iwslt15_en_vi")
    DEFAULT_TOKENIZER_DIR = str(ROOT / "data" / "tokenizer")
    DEFAULT_OUTPUT_DIR = str(ROOT / "runs" / "kaggle")


# ═══════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════

def train_model(
    cfg: TransformerConfig,
    output_dir: Path,
    src_lang: str,
    tgt_lang: str,
    data_dir: str,
    epochs: int,
    resume: bool = True,
) -> None:
    cfg.data_dir = data_dir
    cfg.src_lang = src_lang
    cfg.tgt_lang = tgt_lang
    cfg.output_dir = str(output_dir)
    cfg.epochs = epochs
    cfg.resume = resume

    set_seed(cfg.seed)
    device = get_device()

    tok_src, tok_tgt = load_tokenizers(cfg.tokenizer_dir, src_lang, tgt_lang)
    cfg.src_vocab_size = tok_src.vocab_size
    cfg.tgt_vocab_size = tok_tgt.vocab_size

    print_header(f"TRAIN {src_lang}→{tgt_lang}  |  epochs={epochs}  |  {output_dir}", cfg, device)

    splits = {}
    for name, filter_long in (("train", True), ("dev", False), ("test", False)):
        src, tgt = load_split(data_dir, name, src_lang, tgt_lang)
        splits[name] = TranslationDataset(
            src, tgt, tok_src, tok_tgt, max_len=cfg.max_len, filter_long=filter_long
        )
        print(f"  {name:<5} {len(splits[name]):>7,} câu "
              f"(loại vì quá dài: {splits[name].n_dropped})")

    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  tham số: {n_params:,} ({n_params / 1e6:.1f}M)")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-7, betas=cfg.betas,
        eps=cfg.eps, weight_decay=cfg.weight_decay,
    )
    scheduler = InverseSqrtSchedule(
        optimizer, d_model=cfg.d_model,
        warmup_steps=cfg.warmup_steps, scale=cfg.lr_scale,
    )

    rec = run_training(
        model, cfg, splits["train"], splits["dev"], tok_tgt, device,
        optimizer=optimizer, scheduler=scheduler,
        output_dir=output_dir, run_name=f"{src_lang}2{tgt_lang}",
    )

    print("\nĐánh giá trên test set với checkpoint tốt nhất ...")
    ckpt = load_best(model, output_dir, device)
    print(f"  checkpoint epoch {ckpt['epoch']} (dev BLEU {ckpt['dev_bleu']})")

    for beam in sorted({1, cfg.beam_size}):
        scores, hyps = score_split(
            model, splits["test"], tok_tgt, device,
            beam_size=beam, max_new_tokens=cfg.max_new_tokens,
            length_penalty=cfg.length_penalty,
        )
        key = "greedy" if beam == 1 else f"beam{beam}"
        print(f"  [{key:>7}] BLEU(tok)={scores['bleu_tokenized']:>5.2f}  "
              f"BLEU(detok)={scores['bleu_detok']:>5.2f}  chrF2={scores['chrf2']:>5.2f}")


# ═══════════════════════════════════════════════════════════════════
# Translate file
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def translate_file(
    model, tok_src, tok_tgt,
    input_path: Path, output_path: Path, device,
    beam_size: int = 5, max_new_tokens: int = 120,
    length_penalty: float = 1.0, batch_size: int = 32,
) -> None:
    lines = input_path.read_text(encoding="utf-8").strip().splitlines()
    if not lines:
        print("File input rỗng.")
        return

    encoded = [tok_src.encode(line) + [EOS_ID] for line in lines]
    lengths = [len(e) for e in encoded]
    order = sorted(range(len(encoded)), key=lambda i: lengths[i])
    results: dict[int, str] = {}

    model.eval()
    for start in tqdm(range(0, len(order), batch_size), desc="dịch", unit="batch"):
        chunk_indices = order[start: start + batch_size]
        batch_lines = [encoded[i] for i in chunk_indices]
        max_len = max(len(s) for s in batch_lines)

        src = torch.full((len(batch_lines), max_len), PAD_ID, dtype=torch.long, device=device)
        src_len = torch.tensor([len(s) for s in batch_lines], dtype=torch.long, device=device)
        for i, s in enumerate(batch_lines):
            src[i, :len(s)] = torch.tensor(s, dtype=torch.long)

        hyp = (model.beam_search(src, src_len, beam_size=beam_size,
                                 max_new_tokens=max_new_tokens, length_penalty=length_penalty)
               if beam_size > 1 else
               model.greedy_decode(src, src_len, max_new_tokens=max_new_tokens))

        for row, orig_idx in zip(hyp.tolist(), chunk_indices):
            ids = [t for t in row if t not in (EOS_ID, PAD_ID, BOS_ID)]
            results[orig_idx] = tok_tgt.decode(ids)

    output_path.write_text(
        "\n".join(results[i] for i in range(len(lines))) + "\n", encoding="utf-8")
    print(f"Đã dịch {len(lines):,} câu → {output_path}")


# ═══════════════════════════════════════════════════════════════════
# Stages
# ═══════════════════════════════════════════════════════════════════

def stage_forward(args, base_cfg: TransformerConfig) -> None:
    print("\n" + "=" * 70)
    print("STAGE 1/5: Train EN→VI baseline (forward)")
    print("=" * 70)
    output_dir = Path(args.output_dir) / "forward"
    train_model(base_cfg, output_dir, src_lang="en", tgt_lang="vi",
                data_dir=args.data_dir, epochs=args.epochs, resume=args.resume)


def stage_reverse(args, base_cfg: TransformerConfig) -> None:
    print("\n" + "=" * 70)
    print("STAGE 2/5: Train VI→EN reverse model")
    print("=" * 70)
    output_dir = Path(args.output_dir) / "reverse"
    train_model(base_cfg, output_dir, src_lang="vi", tgt_lang="en",
                data_dir=args.data_dir, epochs=args.epochs_reverse, resume=args.resume)


def stage_backtrans(args, base_cfg: TransformerConfig) -> None:
    print("\n" + "=" * 70)
    print("STAGE 3/5: Back-translation data generation")
    print("=" * 70)

    reverse_ckpt = Path(args.output_dir) / "reverse" / "best.pt"
    if not reverse_ckpt.exists():
        raise SystemExit(f"Thiếu reverse checkpoint: {reverse_ckpt}")

    device = get_device()
    ckpt = torch.load(reverse_ckpt, map_location="cpu", weights_only=False)
    ckpt_cfg = ckpt.get("config", {})
    for k, v in ckpt_cfg.items():
        if hasattr(base_cfg, k) and k not in {"output_dir", "data_dir"}:
            setattr(base_cfg, k, v)

    base_cfg.src_lang, base_cfg.tgt_lang = "vi", "en"
    tok_src, tok_tgt = load_tokenizers(base_cfg.tokenizer_dir, "vi", "en")
    base_cfg.src_vocab_size = tok_src.vocab_size
    base_cfg.tgt_vocab_size = tok_tgt.vocab_size

    model = build_model(base_cfg).to(device)
    model.load_state_dict(ckpt["model"])
    print(f"Loaded reverse model: epoch={ckpt.get('epoch')}, dev_bleu={ckpt.get('dev_bleu')}")
    base_cfg.src_lang, base_cfg.tgt_lang = "en", "vi"

    out_dir = Path(args.output_dir)
    mono_path = out_dir / "mono.vi"
    if not mono_path.exists():
        shutil.copy(Path(args.data_dir) / "train.vi", mono_path)
        n = sum(1 for _ in open(mono_path, encoding="utf-8"))
        print(f"mono data: {mono_path} ({n:,} câu)")

    back_dir = out_dir / "backtrain"
    back_dir.mkdir(parents=True, exist_ok=True)
    synthetic_path = back_dir / "synthetic.en"

    print(f"Back-translating mono.vi → synthetic.en (beam={args.translate_beam})...")
    translate_file(model, tok_src, tok_tgt, mono_path, synthetic_path, device,
                   beam_size=args.translate_beam, batch_size=args.translate_batch)

    shutil.copy(mono_path, back_dir / "mono.vi")

    print("Trộn dữ liệu gốc + back-translated...")
    orig_src_lines = Path(args.data_dir, "train.en").read_text(encoding="utf-8").strip().splitlines()
    orig_tgt_lines = Path(args.data_dir, "train.vi").read_text(encoding="utf-8").strip().splitlines()
    back_src_lines = synthetic_path.read_text(encoding="utf-8").strip().splitlines()
    back_tgt_lines = (back_dir / "mono.vi").read_text(encoding="utf-8").strip().splitlines()

    if args.tag_synthetic:
        back_src_lines = ["<bt> " + s for s in back_src_lines]

    all_src = orig_src_lines + back_src_lines
    all_tgt = orig_tgt_lines + back_tgt_lines
    rng = random.Random(42)
    pairs = list(zip(all_src, all_tgt))
    rng.shuffle(pairs)
    all_src, all_tgt = zip(*pairs)

    combined_dir = out_dir / "combined"
    combined_dir.mkdir(parents=True, exist_ok=True)
    (combined_dir / "combined.en").write_text("\n".join(all_src) + "\n", encoding="utf-8")
    (combined_dir / "combined.vi").write_text("\n".join(all_tgt) + "\n", encoding="utf-8")
    print(f"  Gốc: {len(orig_src_lines):,}  |  Back-BT: {len(back_src_lines):,}  |  "
          f"Tổng: {len(all_src):,} câu")


def stage_final(args, base_cfg: TransformerConfig) -> None:
    print("\n" + "=" * 70)
    print("STAGE 4/5: Train EN→VI final on combined data")
    print("=" * 70)

    combined_dir = Path(args.output_dir) / "combined"
    if not (combined_dir / "combined.en").exists():
        raise SystemExit("Chưa có combined data. Chạy stage backtrans trước.")

    final_data_dir = Path(args.output_dir) / "final_data"
    final_data_dir.mkdir(parents=True, exist_ok=True)
    data_path = Path(args.data_dir)
    for sf in ["tst2012.en", "tst2012.vi", "tst2013.en", "tst2013.vi"]:
        src_f, dst_f = data_path / sf, final_data_dir / sf
        if src_f.exists() and not dst_f.exists():
            shutil.copy(src_f, dst_f)

    shutil.copy(combined_dir / "combined.en", final_data_dir / "train.en")
    shutil.copy(combined_dir / "combined.vi", final_data_dir / "train.vi")

    output_dir = Path(args.output_dir) / "final"
    train_model(base_cfg, output_dir, src_lang="en", tgt_lang="vi",
                data_dir=str(final_data_dir), epochs=args.epochs, resume=args.resume)


def stage_submit(args, base_cfg: TransformerConfig) -> None:
    print("\n" + "=" * 70)
    print("STAGE 5/5: Generate results.csv")
    print("=" * 70)

    final_ckpt = Path(args.output_dir) / "final" / "best.pt"
    if not final_ckpt.exists():
        raise SystemExit(f"Thiếu final checkpoint: {final_ckpt}")

    device = get_device()
    ckpt = torch.load(final_ckpt, map_location="cpu", weights_only=False)
    ckpt_cfg = ckpt.get("config", {})
    for k, v in ckpt_cfg.items():
        if hasattr(base_cfg, k) and k not in {"output_dir", "data_dir"}:
            setattr(base_cfg, k, v)

    base_cfg.src_lang, base_cfg.tgt_lang = "en", "vi"
    tok_src, tok_tgt = load_tokenizers(base_cfg.tokenizer_dir, "en", "vi")
    base_cfg.src_vocab_size = tok_src.vocab_size
    base_cfg.tgt_vocab_size = tok_tgt.vocab_size

    model = build_model(base_cfg).to(device)
    model.load_state_dict(ckpt["model"])
    print(f"Loaded final model: epoch={ckpt.get('epoch')}, dev_bleu={ckpt.get('dev_bleu')}")

    if args.test_input and Path(args.test_input).exists():
        input_path = Path(args.test_input)
    else:
        input_path = Path(args.data_dir) / "tst2013.en"
        print("⚠ Dùng tst2013.en làm test input.")

    output_tmp = Path(args.output_dir) / "test_hyp.vi"
    translate_file(model, tok_src, tok_tgt, input_path, output_tmp, device,
                   beam_size=args.submit_beam, batch_size=args.translate_batch)

    # Compute BLEU
    ref_path = Path(args.data_dir) / "tst2013.vi"
    if ref_path.exists():
        hyps = output_tmp.read_text(encoding="utf-8").strip().splitlines()
        refs = ref_path.read_text(encoding="utf-8").strip().splitlines()
        if len(hyps) == len(refs):
            bleu = corpus_bleu(hyps, refs)
            print(f"\n  BLEU tokenized : {bleu['bleu_tokenized']}  (so với paper)")
            print(f"  BLEU detokenized: {bleu['bleu_detok']}  (chuẩn sacreBLEU 13a)")
            print(f"  chrF2          : {bleu['chrf2']}")

    # Generate results.csv
    hyps = output_tmp.read_text(encoding="utf-8").strip().splitlines()
    csv_path = Path(args.submit_file)
    with open(csv_path, "w", encoding="utf-8") as fh:
        fh.write("Vietnamese\n")
        for hyp in hyps:
            escaped = hyp.replace('"', '""')
            fh.write(f'"{escaped}"\n')

    print(f"\n✅ results.csv saved: {csv_path}  ({len(hyps):,} dòng)")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Kaggle NMT Back-Translation Pipeline EN→VI (Transformer)",
    )
    ap.add_argument("--stage", default="all",
                    choices=["all", "forward", "reverse", "backtrans", "final", "submit"])
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--tokenizer-dir", default=DEFAULT_TOKENIZER_DIR)
    ap.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--epochs-reverse", type=int, default=25)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--translate-beam", type=int, default=5)
    ap.add_argument("--submit-beam", type=int, default=5)
    ap.add_argument("--translate-batch", type=int, default=64)
    ap.add_argument("--tag-synthetic", action="store_true", default=True)
    ap.add_argument("--no-tag-synthetic", action="store_false", dest="tag_synthetic")
    ap.add_argument("--test-input", default=None)
    ap.add_argument("--submit-file", default="results.csv")
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", action="store_false", dest="resume")

    # parse_known_args: bỏ qua arg lạ từ Jupyter launcher (-f kernel.json)
    args, _ = ap.parse_known_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    base_cfg = TransformerConfig()
    base_cfg.data_dir = args.data_dir
    base_cfg.tokenizer_dir = args.tokenizer_dir
    base_cfg.patience = args.patience

    stages = {
        "all":       [stage_forward, stage_reverse, stage_backtrans, stage_final, stage_submit],
        "forward":   [stage_forward],
        "reverse":   [stage_reverse],
        "backtrans": [stage_backtrans],
        "final":     [stage_final],
        "submit":    [stage_submit],
    }

    for fn in stages[args.stage]:
        fn(args, base_cfg)

    print("\n✅ Pipeline hoàn thành!")


if __name__ == "__main__":
    main()
