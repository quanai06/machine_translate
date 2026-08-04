#!/usr/bin/env python3
"""Average n checkpoint gần nhất (hoặc checkpoint tốt nhất ±k epoch).

Chạy:
    python scripts/average_checkpoints.py --model transformer --last 5
    python scripts/average_checkpoints.py --model transformer --epochs 25 26 27 28 29
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from common.cli import get_device, set_seed
from common.tokenizer import load_tokenizers
from translate_transformers.config import TransformerConfig
from translate_transformers.model import build_model


def average_state_dicts(paths: list[Path]) -> dict:
    """Trung bình cộng state_dict từ nhiều checkpoint."""
    avg = None
    for p in paths:
        state = torch.load(p, map_location="cpu", weights_only=False)["model"]
        if avg is None:
            avg = {k: v.float() for k, v in state.items()}
        else:
            for k in avg:
                avg[k] += state[k].float()
    for k in avg:
        avg[k] /= len(paths)
    return avg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["transformer", "seq2seq"])
    ap.add_argument("--last", type=int, default=5, help="Average N checkpoint cuối")
    ap.add_argument("--epochs", type=int, nargs="+", default=None, help="Chỉ định epoch cụ thể")
    args = ap.parse_args()

    if args.model == "transformer":
        cfg, build_model = TransformerConfig(), build_model
        cfg = TransformerConfig()
    else:
        raise NotImplementedError

    set_seed(cfg.seed)
    device = get_device()

    tok_src, tok_tgt = load_tokenizers(cfg.tokenizer_dir, cfg.src_lang, cfg.tgt_lang)
    cfg.src_vocab_size = tok_src.vocab_size
    cfg.tgt_vocab_size = tok_tgt.vocab_size

    out_dir = Path(cfg.output_dir)
    if args.epochs:
        paths = [out_dir / f"epoch_{e:03d}.pt" for e in args.epochs]
    else:
        # Lấy N checkpoint cuối cùng
        all_ckpts = sorted(out_dir.glob("epoch_*.pt"))
        paths = all_ckpts[-args.last:]

    print(f"Average {len(paths)} checkpoint:")
    for p in paths:
        print(f"  {p.name}")

    avg_state = average_state_dicts(paths)
    model = build_model(cfg)
    model.load_state_dict(avg_state)
    torch.save({
        "model": model.state_dict(),
        "epoch": "averaged",
        "config": vars(cfg),
    }, out_dir / "averaged.pt")
    print(f"Đã lưu: {out_dir / 'averaged.pt'}")

    # Chấm thử trên dev
    from common.data import TranslationDataset, load_split
    from common.engine import score_split
    src, tgt = load_split(cfg.data_dir, "dev", cfg.src_lang, cfg.tgt_lang)
    ds = TranslationDataset(src, tgt, tok_src, tok_tgt, max_len=cfg.max_len, filter_long=False)
    scores, _ = score_split(model.to(device), ds, tok_tgt, device, beam_size=5)
    print(f"Dev BLEU (averaged): {scores['bleu_tokenized']:.2f}")


if __name__ == "__main__":
    main()