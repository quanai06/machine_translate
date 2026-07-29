"""Train tokenizer SentencePiece dùng chung cho cả hai case, rồi in thống kê corpus.

    python scripts/prepare.py                     # vocab 8000 mỗi phía
    python scripts/prepare.py --vocab-size 16000

Chỉ chạy MỘT LẦN. Cả `translate_transformers` lẫn `translate_seq2seq` đều nạp
đúng bộ tokenizer này — đó là điều kiện để so sánh hai model được công bằng.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from common.data import load_split  # noqa: E402
from common.tokenizer import train_sentencepiece  # noqa: E402


def corpus_stats(lines: list[str], name: str) -> dict:
    tokens = [t for line in lines for t in line.split()]
    counts = Counter(tokens)
    lengths = [len(line.split()) for line in lines]
    lengths_sorted = sorted(lengths)
    return {
        "split": name,
        "sentences": len(lines),
        "tokens": len(tokens),
        "vocab_whitespace": len(counts),
        "avg_len": round(len(tokens) / max(1, len(lines)), 2),
        "p95_len": lengths_sorted[int(0.95 * len(lengths_sorted))] if lengths else 0,
        "max_len": max(lengths) if lengths else 0,
        "singletons": sum(1 for c in counts.values() if c == 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(ROOT / "data" / "iwslt15_en_vi"))
    ap.add_argument("--out-dir", default=str(ROOT / "data" / "tokenizer"))
    ap.add_argument("--vocab-size", type=int, default=8000,
                    help="Kích thước vocab MỖI PHÍA. 8000 là điểm hợp lý cho "
                         "133k câu; lớn hơn sẽ có nhiều subword hiếm học không tới.")
    ap.add_argument("--model-type", default="bpe", choices=["bpe", "unigram"])
    args = ap.parse_args()

    data_dir, out_dir = Path(args.data_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("  CHUẨN BỊ DỮ LIỆU — IWSLT'15 English-Vietnamese")
    print("=" * 78)

    stats = {}
    for split in ("train", "dev", "test"):
        src, tgt = load_split(data_dir, split)
        stats[f"{split}_en"] = corpus_stats(src, f"{split}.en")
        stats[f"{split}_vi"] = corpus_stats(tgt, f"{split}.vi")

    hdr = f"{'split':<12}{'câu':>9}{'token':>12}{'vocab':>10}{'dài TB':>9}{'p95':>7}{'hapax':>9}"
    print(hdr)
    print("-" * len(hdr))
    for s in stats.values():
        print(f"{s['split']:<12}{s['sentences']:>9,}{s['tokens']:>12,}"
              f"{s['vocab_whitespace']:>10,}{s['avg_len']:>9.1f}"
              f"{s['p95_len']:>7}{s['singletons']:>9,}")

    print("\nGhi chú đọc bảng trên:")
    tr_en, tr_vi = stats["train_en"], stats["train_vi"]
    print(f"  - Vocab thô phía en là {tr_en['vocab_whitespace']:,} từ, trong đó "
          f"{tr_en['singletons']:,} từ chỉ xuất hiện ĐÚNG 1 LẦN "
          f"({100 * tr_en['singletons'] / tr_en['vocab_whitespace']:.0f}%).")
    print(f"    Với vocab word-level cố định, gần như toàn bộ số đó thành <unk>.")
    print(f"    Đây chính là 'rare word problem' mà BPE sinh ra để giải quyết.")
    print(f"  - Câu tiếng Việt dài hơn tiếng Anh ({tr_vi['avg_len']} vs "
          f"{tr_en['avg_len']} token) vì tiếng Việt viết rời từng âm tiết.")

    print("\nTrain SentencePiece ...")
    for lang, key in (("en", "train_en"), ("vi", "train_vi")):
        corpus_file = data_dir / f"train.{lang}"
        model_path = train_sentencepiece(
            corpus_file, out_dir / f"spm_{lang}", args.vocab_size,
            model_type=args.model_type,
        )
        print(f"  {lang}: {model_path}")

    (out_dir / "prepare_stats.json").write_text(
        json.dumps({"corpus": stats, "vocab_size": args.vocab_size,
                    "model_type": args.model_type}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Kiểm tra nhanh: encode/decode một câu phải khớp lại
    from common.tokenizer import load_tokenizers  # noqa: E402

    tok_en, tok_vi = load_tokenizers(out_dir)
    src, tgt = load_split(data_dir, "test")
    print("\nKiểm tra vòng encode -> decode:")
    for text, tok in ((src[0], tok_en), (tgt[0], tok_vi)):
        ids = tok.encode(text)
        back = tok.decode(ids)
        flag = "OK " if back.split() == text.split() else "LỆCH"
        print(f"  [{flag}] {tok.lang}: {len(text.split())} từ -> {len(ids)} subword")
        if flag == "LỆCH":
            print(f"        gốc : {text[:90]}")
            print(f"        về  : {back[:90]}")

    print(f"\nXong. Giờ có thể train:")
    print(f"    python translate_transformers/train.py")
    print(f"    python translate_seq2seq/train.py")


if __name__ == "__main__":
    main()
