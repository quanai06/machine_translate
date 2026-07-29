"""Đánh giá lại một checkpoint đã train, hoặc dịch thử vài câu.

    python evaluate.py --model transformer --beam 5
    python evaluate.py --model seq2seq --beam 10 --show 10
    python evaluate.py --model transformer --text "I love machine translation ."

Tách riêng khỏi train.py để có thể chấm lại với beam khác, hoặc xem câu dịch
mẫu, mà không phải train lại từ đầu.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from common.cli import dump_json, get_device, set_seed  # noqa: E402
from common.data import EOS_ID, TranslationDataset, load_split  # noqa: E402
from common.engine import score_split  # noqa: E402
from common.metrics import detokenize  # noqa: E402
from common.tokenizer import load_tokenizers  # noqa: E402

# Đường dẫn phụ thuộc máy — không lấy từ checkpoint. Train trên Colab rồi chấm
# lại ở máy khác thì các path tuyệt đối trong checkpoint sẽ trỏ vào hư không.
_PATH_FIELDS = {"output_dir", "data_dir", "tokenizer_dir"}


def build(model_name: str):
    if model_name == "transformer":
        from translate_transformers.config import TransformerConfig
        from translate_transformers.model import build_model
        return TransformerConfig(), build_model
    if model_name == "seq2seq":
        from translate_seq2seq.config import Seq2SeqConfig
        from translate_seq2seq.model import build_model
        return Seq2SeqConfig(), build_model
    raise ValueError("--model phải là 'transformer' hoặc 'seq2seq'")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["transformer", "seq2seq"])
    ap.add_argument("--split", default="test", choices=["dev", "test"])
    ap.add_argument("--beam", type=int, default=5)
    ap.add_argument("--length-penalty", type=float, default=1.0)
    ap.add_argument("--show", type=int, default=5, help="in bao nhiêu câu dịch mẫu")
    ap.add_argument("--text", default=None, help="dịch một câu tiếng Anh rồi thoát")
    ap.add_argument("--checkpoint-dir", default=None)
    args = ap.parse_args()

    cfg, build_model = build(args.model)
    if args.checkpoint_dir:
        cfg.output_dir = args.checkpoint_dir
    set_seed(cfg.seed)
    device = get_device()

    # Kiến trúc phải lấy từ checkpoint, không phải từ config mặc định: nếu train
    # bằng cờ khác mặc định (--d-model, --n-enc-layers, ...) thì dựng theo mặc
    # định sẽ lệch shape và load_state_dict báo lỗi.
    ckpt_path = Path(cfg.output_dir) / "best.pt"
    if not ckpt_path.exists():
        raise SystemExit(
            f"Chưa có checkpoint tại {ckpt_path}.\n"
            f"Train trước: python translate_{args.model}/train.py"
        )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    for k, v in (ckpt.get("config") or {}).items():
        if hasattr(cfg, k) and k not in _PATH_FIELDS:
            setattr(cfg, k, v)

    tok_src, tok_tgt = load_tokenizers(cfg.tokenizer_dir, cfg.src_lang, cfg.tgt_lang)
    if (cfg.src_vocab_size, cfg.tgt_vocab_size) != (tok_src.vocab_size, tok_tgt.vocab_size):
        raise SystemExit(
            f"Tokenizer không khớp checkpoint: checkpoint dùng vocab "
            f"{cfg.src_vocab_size}/{cfg.tgt_vocab_size}, tokenizer hiện tại là "
            f"{tok_src.vocab_size}/{tok_tgt.vocab_size}.\n"
            f"Dùng đúng bộ tokenizer đã train cùng model."
        )

    model = build_model(cfg)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    print(f"Checkpoint: epoch {ckpt['epoch']}, dev BLEU {ckpt['dev_bleu']}")

    # ---- chế độ dịch một câu ----
    if args.text:
        ids = torch.tensor([tok_src.encode(args.text) + [EOS_ID]], device=device)
        src_len = torch.tensor([ids.size(1)], device=device)
        with torch.no_grad():
            out = (model.beam_search(ids, src_len, beam_size=args.beam,
                                     max_new_tokens=cfg.max_new_tokens)
                   if args.beam > 1 else
                   model.greedy_decode(ids, src_len, max_new_tokens=cfg.max_new_tokens))
        toks = [t for t in out[0].tolist() if t > EOS_ID]
        print(f"  EN : {args.text}")
        print(f"  VI : {detokenize(tok_tgt.decode(toks))}")
        return

    # ---- chấm cả split ----
    src, tgt = load_split(cfg.data_dir, args.split, cfg.src_lang, cfg.tgt_lang)
    ds = TranslationDataset(src, tgt, tok_src, tok_tgt,
                            max_len=cfg.max_len, filter_long=False)

    scores, hyps = score_split(
        model, ds, tok_tgt, device,
        beam_size=args.beam, max_new_tokens=cfg.max_new_tokens,
        length_penalty=args.length_penalty,
    )

    print(f"\n{args.model} — {args.split} (beam={args.beam})")
    print(f"  BLEU tokenized  : {scores['bleu_tokenized']:.2f}   "
          f"<- con số so được với paper (tương đương multi-bleu.perl)")
    print(f"  BLEU detokenized: {scores['bleu_detok']:.2f}   <- chuẩn sacreBLEU 13a")
    print(f"  chrF2           : {scores['chrf2']:.2f}")
    print(f"  n-gram precision: {scores['precisions']}")
    print(f"  brevity penalty : {scores['brevity_penalty']}  "
          f"(độ dài hệ thống {scores['sys_len']} / tham chiếu {scores['ref_len']})")
    if scores["brevity_penalty"] < 0.95:
        print("    ! BP thấp nghĩa là model dịch NGẮN hơn tham chiếu — thường do "
              "beam thiếu length penalty hoặc EOS được sinh quá sớm.")

    for i in range(min(args.show, len(hyps))):
        print(f"\n  [{i}] EN  : {detokenize(ds.raw_src[i])}")
        print(f"      REF : {detokenize(ds.raw_tgt[i])}")
        print(f"      HYP : {detokenize(hyps[i])}")

    out_dir = Path(cfg.output_dir)
    (out_dir / f"{args.split}.hyp.beam{args.beam}.vi").write_text(
        "\n".join(hyps) + "\n", encoding="utf-8")
    dump_json(out_dir / f"eval_{args.split}_beam{args.beam}.json", scores)


if __name__ == "__main__":
    main()
