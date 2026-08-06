"""Dịch một file văn bản bằng model đã train — dùng cho back-translation.

    python translate.py \
        --model transformer \
        --checkpoint runs/transformer_reverse/best.pt \
        --input mono.vi \
        --output synthetic.en \
        --beam 5 --batch 64

Khác với evaluate.py (dịch split test có sẵn), script này nhận FILE tuỳ ý,
dùng cho bước back-translate mono.tgt -> synthetic.src.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from common.cli import get_device, set_seed       # noqa: E402
from common.data import EOS_ID, PAD_ID, BOS_ID, collate_batch  # noqa: E402
from common.tokenizer import load_tokenizers       # noqa: E402

_PATH_FIELDS = {"output_dir", "data_dir"}  # tokenizer_dir dùng từ checkpoint


def build_model(model_name: str, cfg):
    if model_name == "transformer":
        from translate_transformers.config import TransformerConfig
        from translate_transformers.model import build_model as _build
    elif model_name == "seq2seq":
        from translate_seq2seq.config import Seq2SeqConfig
        from translate_seq2seq.model import build_model as _build
    else:
        raise ValueError("--model phải là 'transformer' hoặc 'seq2seq'")
    return _build(cfg)


def translate_file(
    model,
    tok_src,
    tok_tgt,
    input_path: Path,
    output_path: Path,
    device,
    *,
    beam_size: int = 5,
    max_new_tokens: int = 120,
    length_penalty: float = 1.0,
    batch_size: int = 32,
) -> None:
    """Dịch từng dòng trong input_path, ghi ra output_path."""
    lines = input_path.read_text(encoding="utf-8").strip().splitlines()
    if not lines:
        print("File input rỗng, không có gì để dịch.")
        return

    # Mã hoá trước tất cả dòng
    encoded = [tok_src.encode(line) + [EOS_ID] for line in lines]
    lengths = [len(e) for e in encoded]

    # Gom batch theo số câu (đơn giản, đủ dùng)
    order = sorted(range(len(encoded)), key=lambda i: lengths[i])
    results: dict[int, str] = {}

    model.eval()
    pbar = tqdm(range(0, len(order), batch_size), desc="dịch", unit="batch")
    for start in pbar:
        chunk_indices = order[start: start + batch_size]
        batch_lines = [encoded[i] for i in chunk_indices]

        # Pad thành tensor
        max_len = max(len(s) for s in batch_lines)
        src = torch.full((len(batch_lines), max_len), PAD_ID, dtype=torch.long, device=device)
        src_len = torch.tensor([len(s) for s in batch_lines], dtype=torch.long, device=device)
        for i, s in enumerate(batch_lines):
            src[i, :len(s)] = torch.tensor(s, dtype=torch.long)

        with torch.no_grad():
            hyp = (
                model.beam_search(src, src_len, beam_size=beam_size,
                                  max_new_tokens=max_new_tokens,
                                  length_penalty=length_penalty)
                if beam_size > 1 else
                model.greedy_decode(src, src_len, max_new_tokens=max_new_tokens)
            )

        for row, orig_idx in zip(hyp.tolist(), chunk_indices):
            ids = []
            for t in row:
                if t == EOS_ID:
                    break
                if t not in (PAD_ID, BOS_ID):
                    ids.append(t)
            results[orig_idx] = tok_tgt.decode(ids)

    # Ghi đúng thứ tự gốc
    with open(output_path, "w", encoding="utf-8") as fh:
        for i in range(len(lines)):
            fh.write(results[i] + "\n")

    print(f"Đã dịch {len(lines):,} câu -> {output_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Dịch file văn bản bằng model NMT")
    ap.add_argument("--model", required=True, choices=["transformer", "seq2seq"])
    ap.add_argument("--checkpoint", required=True, help="đường dẫn tới best.pt")
    ap.add_argument("--input", required=True, help="file input (mỗi dòng 1 câu)")
    ap.add_argument("--output", required=True, help="file output")
    ap.add_argument("--beam", type=int, default=5)
    ap.add_argument("--length-penalty", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=64, help="số câu mỗi batch decode")
    ap.add_argument("--max-new-tokens", type=int, default=120)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device()

    # Nạp checkpoint
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise SystemExit(f"Không tìm thấy checkpoint: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ckpt_cfg = ckpt.get("config", {})

    # Dựng config và model
    if args.model == "transformer":
        from translate_transformers.config import TransformerConfig
        cfg = TransformerConfig()
    else:
        from translate_seq2seq.config import Seq2SeqConfig
        cfg = Seq2SeqConfig()

    # Ghi đè config từ checkpoint (giữ path local)
    for k, v in ckpt_cfg.items():
        if hasattr(cfg, k) and k not in _PATH_FIELDS:
            setattr(cfg, k, v)

    tok_src, tok_tgt = load_tokenizers(cfg.tokenizer_dir, cfg.src_lang, cfg.tgt_lang)

    model = build_model(args.model, cfg)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    print(f"Nạp checkpoint epoch={ckpt.get('epoch', '?')} dev_bleu={ckpt.get('dev_bleu', '?')}")
    print(f"  SRC={cfg.src_lang} -> TGT={cfg.tgt_lang}")

    translate_file(
        model, tok_src, tok_tgt,
        Path(args.input), Path(args.output),
        device,
        beam_size=args.beam,
        max_new_tokens=args.max_new_tokens,
        length_penalty=args.length_penalty,
        batch_size=args.batch,
    )


if __name__ == "__main__":
    main()
