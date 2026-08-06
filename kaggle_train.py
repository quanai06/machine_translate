"""Script train toàn diện cho Kaggle GPU — Back-Translation Pipeline EN→VI (Transformer).

Gọi trực tiếp Python functions, không dùng subprocess.
"""

from __future__ import annotations

import argparse
import math
import random
import shutil
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
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
    DEFAULT_DATA_DIR = "/kaggle/input/iwslt15-envi-data/iwslt15_en_vi"
    DEFAULT_TOKENIZER_DIR = "/kaggle/input/iwslt15-envi-data/tokenizer"
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

    args = ap.parse_args()
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
