"""Kiểm tra tính đúng đắn của cả hai bản cài đặt bằng phép thử OVERFIT.

    python scripts/sanity_check.py                # cả hai model
    python scripts/sanity_check.py --model seq2seq

Ý tưởng: cho model học thuộc lòng một tập rất nhỏ (mặc định 200 câu) rồi chấm
BLEU trên chính tập đó. Một cài đặt ĐÚNG phải đạt BLEU rất cao (>80). Nếu không:

  - BLEU cao ở greedy nhưng thấp ở beam  -> lỗi trong beam search (thường là
    quên hoán vị state theo beam, hoặc length penalty sai dấu).
  - BLEU thấp ở cả hai                   -> lỗi ở mask, ở teacher forcing,
    hoặc lệch một bước giữa tgt_in và tgt_out.
  - Loss không giảm                      -> lỗi optimizer/khởi tạo.

Nên chạy cái này (mất vài phút trên CPU) TRƯỚC khi tốn hàng giờ GPU trên Colab.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from common.cli import get_device, set_seed  # noqa: E402
from common.data import TranslationDataset, build_dataloader, load_split  # noqa: E402
from common.engine import score_split, train_one_epoch  # noqa: E402
from common.tokenizer import load_tokenizers  # noqa: E402

PASS_THRESHOLD = 80.0

# Số epoch cần để overfit, đo thực nghiệm trên tập 120 câu ngắn.
# Chênh lệch này KHÔNG phải lỗi: Transformer cập nhật toàn bộ chuỗi target song
# song và hội tụ nhanh hơn hẳn mỗi bước, còn LSTM có input feeding phải lan
# gradient tuần tự qua từng timestep. Đây chính là hiện tượng mà bài so sánh
# muốn đo — ở đây nó chỉ xuất hiện dưới dạng "cần nhiều epoch hơn để PASS".
DEFAULT_STEPS = {"transformer": 150, "seq2seq": 500}


def make(model_name: str, tok_src, tok_tgt):
    if model_name == "transformer":
        from translate_transformers.config import TransformerConfig
        from translate_transformers.model import build_model
        cfg = TransformerConfig()
        cfg.d_model, cfg.n_heads = 128, 4
        cfg.n_enc_layers = cfg.n_dec_layers = 2
        cfg.d_ff, cfg.dropout = 256, 0.0
    else:
        from translate_seq2seq.config import Seq2SeqConfig
        from translate_seq2seq.model import build_model
        cfg = Seq2SeqConfig()
        cfg.emb_dim = cfg.hidden_dim = 128
        cfg.n_dec_layers, cfg.dropout = 1, 0.0
    cfg.src_vocab_size, cfg.tgt_vocab_size = tok_src.vocab_size, tok_tgt.vocab_size
    return cfg, build_model(cfg)


def run_one(model_name: str, n_sents: int, steps: int, device) -> bool:
    print("=" * 70)
    print(f"  OVERFIT TEST — {model_name}  ({n_sents} câu, {steps} epoch)")
    print("=" * 70)
    set_seed(0)

    tok_src, tok_tgt = load_tokenizers(ROOT / "data" / "tokenizer")
    src, tgt = load_split(ROOT / "data" / "iwslt15_en_vi", "test")
    # câu ngắn để học thuộc nhanh
    pairs = sorted(zip(src, tgt), key=lambda p: len(p[0].split()))[:n_sents]
    src, tgt = [p[0] for p in pairs], [p[1] for p in pairs]

    ds = TranslationDataset(src, tgt, tok_src, tok_tgt, max_len=100, filter_long=False)
    loader = build_dataloader(ds, max_tokens=4096, shuffle=True, num_workers=0)

    cfg, model = make(model_name, tok_src, tok_tgt)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler(device.type, enabled=False)

    for ep in range(1, steps + 1):
        st = train_one_epoch(
            model, loader, opt, None, scaler, device,
            # Dùng đúng clip_norm của config thật (Transformer 1.0, LSTM 5.0).
            # Clip quá chặt cho LSTM sẽ bóp gradient và làm phép thử này báo
            # FAIL oan cho một cài đặt vốn không sai.
            label_smoothing=0.0, clip_norm=cfg.clip_norm, amp_dtype=None,
            desc=f"{model_name} overfit ep{ep}",
        )
        if ep % 10 == 0 or ep == steps:
            print(f"    ep{ep:>3}  train_ppl={st['train_ppl']:.2f}")

    ok = True
    for beam in (1, 5):
        scores, hyps = score_split(model, ds, tok_tgt, device,
                                   beam_size=beam, max_new_tokens=60)
        bleu = scores["bleu_tokenized"]
        tag = "greedy" if beam == 1 else f"beam{beam}"
        verdict = "PASS" if bleu >= PASS_THRESHOLD else "FAIL"
        if bleu < PASS_THRESHOLD:
            ok = False
        print(f"    [{verdict}] {tag:<7} BLEU={bleu:6.2f}  BP={scores['brevity_penalty']}")

    print(f"    ví dụ  REF: {ds.raw_tgt[0]}")
    print(f"           HYP: {hyps[0]}")
    print()
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="both", choices=["both", "transformer", "seq2seq"])
    ap.add_argument("--n-sents", type=int, default=120)
    ap.add_argument("--steps", type=int, default=None,
                    help="mặc định tuỳ model: transformer 150, seq2seq 500")
    args = ap.parse_args()

    device = get_device()
    print(f"thiết bị: {device}")
    if device.type == "cpu":
        print("(trên CPU phép thử này mất khoảng 20-40 phút; trên GPU vài phút)")
    print()
    names = ["transformer", "seq2seq"] if args.model == "both" else [args.model]
    results = {
        n: run_one(n, args.n_sents, args.steps or DEFAULT_STEPS[n], device)
        for n in names
    }

    print("=" * 70)
    for n, ok in results.items():
        print(f"  {n:<14} {'PASS' if ok else 'FAIL'}")
    print("=" * 70)
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
