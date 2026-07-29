"""Train Case 1 — Transformer trên IWSLT'15 English -> Vietnamese.

    python translate_transformers/train.py                 # train đầy đủ
    python translate_transformers/train.py --smoke         # kiểm tra pipeline
    python translate_transformers/train.py --epochs 20 --dropout 0.2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.cli import (  # noqa: E402
    add_config_args, apply_overrides, dump_json, get_device, print_header, set_seed,
)
from common.bench import measure_latency  # noqa: E402
from common.data import TranslationDataset, build_dataloader, load_split  # noqa: E402
from common.engine import InverseSqrtSchedule, load_best, run_training, score_split  # noqa: E402
from common.tokenizer import load_tokenizers  # noqa: E402
from translate_transformers.config import SMOKE, TransformerConfig  # noqa: E402
from translate_transformers.model import build_model  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Transformer NMT En->Vi (IWSLT15)")
    add_config_args(parser, TransformerConfig)
    args = parser.parse_args()

    cfg = apply_overrides(TransformerConfig(), args, SMOKE)
    set_seed(cfg.seed)
    device = get_device()

    tok_src, tok_tgt = load_tokenizers(cfg.tokenizer_dir, cfg.src_lang, cfg.tgt_lang)
    cfg.src_vocab_size = tok_src.vocab_size
    cfg.tgt_vocab_size = tok_tgt.vocab_size
    print_header("CASE 1 — TRANSFORMER (Vaswani et al., 2017)", cfg, device)

    splits = {}
    for name, filter_long in (("train", True), ("dev", False), ("test", False)):
        src, tgt = load_split(cfg.data_dir, name, cfg.src_lang, cfg.tgt_lang)
        if args.smoke:
            n = 4000 if name == "train" else 200
            src, tgt = src[:n], tgt[:n]
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
        output_dir=Path(cfg.output_dir), run_name="transformer",
    )

    # ---- đánh giá cuối trên tst2013 với checkpoint tốt nhất ----
    print("\nNạp lại checkpoint tốt nhất và chấm trên tst2013 ...")
    ckpt = load_best(model, Path(cfg.output_dir), device)
    print(f"  checkpoint từ epoch {ckpt['epoch']} (dev BLEU {ckpt['dev_bleu']})")

    results = {}
    for beam in sorted({1, cfg.beam_size}):
        scores, hyps = score_split(
            model, splits["test"], tok_tgt, device,
            beam_size=beam, max_new_tokens=cfg.max_new_tokens,
            length_penalty=cfg.length_penalty,
        )
        key = "greedy" if beam == 1 else f"beam{beam}"
        results[key] = scores
        out_file = Path(cfg.output_dir) / f"tst2013.hyp.{key}.vi"
        out_file.write_text("\n".join(hyps) + "\n", encoding="utf-8")
        print(f"  [{key:>7}] BLEU(tok)={scores['bleu_tokenized']:>5.2f}  "
              f"BLEU(detok)={scores['bleu_detok']:>5.2f}  chrF2={scores['chrf2']:>5.2f}")

    lat = measure_latency(
        model,
        build_dataloader(splits["test"], max_tokens=cfg.max_tokens,
                         shuffle=False, num_workers=0),
        device, max_new_tokens=cfg.max_new_tokens,
    )
    rec.finish(test=results, **lat)
    dump_json(Path(cfg.output_dir) / "test_results.json", results)
    print(f"\nXong. Kết quả nằm ở: {cfg.output_dir}")


if __name__ == "__main__":
    main()
