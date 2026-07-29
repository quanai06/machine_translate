"""Train Case 2 — Seq2Seq LSTM + Luong Attention trên IWSLT'15 English -> Vietnamese.

    python translate_seq2seq/train.py                      # train đầy đủ (Adam)
    python translate_seq2seq/train.py --smoke              # kiểm tra pipeline
    python translate_seq2seq/train.py --optimizer sgd --lr 1.0   # bám tensorflow/nmt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.bench import measure_latency  # noqa: E402
from common.cli import (  # noqa: E402
    add_config_args, apply_overrides, dump_json, get_device, print_header, set_seed,
)
from common.data import TranslationDataset, build_dataloader, load_split  # noqa: E402
from common.engine import load_best, run_training, score_split  # noqa: E402
from common.tokenizer import load_tokenizers  # noqa: E402
from translate_seq2seq.config import SMOKE, Seq2SeqConfig  # noqa: E402
from translate_seq2seq.model import build_model  # noqa: E402


class StepDecaySchedule:
    """Giữ lr cố định rồi giảm một nửa mỗi epoch ở giai đoạn cuối.

    Bắt chước lịch của tensorflow/nmt ("sau 8K bước, halve lr mỗi 1K bước"),
    nhưng tính theo epoch để không phụ thuộc vào việc chia batch theo token
    làm số bước mỗi epoch thay đổi.
    """

    def __init__(self, optimizer, base_lr, total_epochs, start_ratio, factor):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.start_epoch = max(1, int(total_epochs * start_ratio))
        self.factor = factor
        self.epoch = 0

    def step_epoch(self, epoch: int) -> float:
        self.epoch = epoch
        n_decay = max(0, epoch - self.start_epoch)
        lr = self.base_lr * (self.factor ** n_decay)
        for g in self.optimizer.param_groups:
            g["lr"] = lr
        return lr


def main() -> None:
    parser = argparse.ArgumentParser(description="Seq2Seq+Attention NMT En->Vi (IWSLT15)")
    add_config_args(parser, Seq2SeqConfig)
    args = parser.parse_args()

    cfg = apply_overrides(Seq2SeqConfig(), args, SMOKE)
    set_seed(cfg.seed)
    device = get_device()

    tok_src, tok_tgt = load_tokenizers(cfg.tokenizer_dir, cfg.src_lang, cfg.tgt_lang)
    cfg.src_vocab_size = tok_src.vocab_size
    cfg.tgt_vocab_size = tok_tgt.vocab_size
    print_header("CASE 2 — SEQ2SEQ + LUONG ATTENTION (tensorflow/nmt)", cfg, device)

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

    if cfg.optimizer.lower() == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr)
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
    scheduler = StepDecaySchedule(
        optimizer, cfg.lr, cfg.epochs, cfg.lr_decay_start_ratio, cfg.lr_decay_factor
    )
    scheduler.step_epoch(1)

    rec = run_training(
        model, cfg, splits["train"], splits["dev"], tok_tgt, device,
        # scheduler=None: lr ở đây đổi theo EPOCH chứ không theo từng step,
        # nên ta cập nhật qua hook `on_epoch_start` thay vì trong vòng batch.
        optimizer=optimizer, scheduler=None,
        on_epoch_start=scheduler.step_epoch,
        output_dir=Path(cfg.output_dir), run_name="seq2seq",
    )

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
