"""Tạo dữ liệu back-translation: mono.tgt -> synthetic.src dùng mô hình ngược.

Pipeline:
    1. Train mô hình ngược target→source (swap src/tgt config)
    2. Dịch mono.tgt (tiếng Việt) -> synthetic.src (tiếng Anh) bằng mô hình ngược
    3. Ghép: backtrain.src = synthetic.src, backtrain.tgt = mono.tgt (giữ nguyên)
    4. Trộn với dữ liệu gốc và shuffle

Cách chạy:
    # Bước 1: Sinh dữ liệu back-translated
    python back_translate.py generate \
        --reverse-ckpt runs/transformer_reverse/best.pt \
        --mono data/mono.vi \
        --out-dir data/backtrain

    # Bước 2: Trộn dữ liệu
    python back_translate.py combine \
        --original-src data/iwslt15_en_vi/train.en \
        --original-tgt data/iwslt15_en_vi/train.vi \
        --back-src data/backtrain/synthetic.en \
        --back-tgt data/backtrain/mono.vi \
        --out-dir data/combined \
        --tag-synthetic   # đánh dấu câu tổng hợp bằng token đặc biệt
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def cmd_generate(args) -> None:
    """Dịch mono.tgt -> synthetic.src bằng mô hình ngược."""
    # Gọi translate.py để làm việc dịch
    import subprocess
    script = ROOT / "translate.py"

    mono_path = Path(args.mono)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    synthetic_src = out_dir / "synthetic.en"

    # Xác định model type từ checkpoint
    import torch
    ckpt = torch.load(args.reverse_ckpt, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    model_type = "transformer"
    if "hidden_dim" in cfg:
        model_type = "seq2seq"

    cmd = [
        sys.executable, str(script),
        "--model", model_type,
        "--checkpoint", args.reverse_ckpt,
        "--input", str(mono_path),
        "--output", str(synthetic_src),
        "--beam", str(args.beam),
        "--batch", str(args.batch),
        "--max-new-tokens", str(args.max_new_tokens),
    ]
    print(f"Chạy: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    # Copy mono.tgt làm backtrain.tgt
    import shutil
    backtrain_tgt = out_dir / "mono.vi"
    shutil.copy(mono_path, backtrain_tgt)

    print(f"\nKết quả:")
    print(f"  synthetic.en  : {synthetic_src}  ({_count_lines(synthetic_src)} dòng)")
    print(f"  mono.vi       : {backtrain_tgt}  ({_count_lines(backtrain_tgt)} dòng)")


def cmd_combine(args) -> None:
    """Trộn dữ liệu gốc + back-translated, shuffle, output combined.*"""
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    original_src = Path(args.original_src).read_text(encoding="utf-8").strip().splitlines()
    original_tgt = Path(args.original_tgt).read_text(encoding="utf-8").strip().splitlines()
    back_src = Path(args.back_src).read_text(encoding="utf-8").strip().splitlines()
    back_tgt = Path(args.back_tgt).read_text(encoding="utf-8").strip().splitlines()

    assert len(original_src) == len(original_tgt), "Lệch số dòng original src/tgt"
    assert len(back_src) == len(back_tgt), "Lệch số dòng back src/tgt"

    if args.tag_synthetic:
        # Đánh dấu câu tổng hợp bằng tag đặc biệt ở đầu câu nguồn
        TAG = "<bt> "
        back_src = [TAG + s for s in back_src]

    all_src = original_src + back_src
    all_tgt = original_tgt + back_tgt

    # Shuffle cùng seed để tái hiện được
    rng = random.Random(args.seed)
    pairs = list(zip(all_src, all_tgt))
    rng.shuffle(pairs)
    all_src, all_tgt = zip(*pairs) if pairs else ([], [])

    combined_src = out_dir / "combined.en"
    combined_tgt = out_dir / "combined.vi"

    combined_src.write_text("\n".join(all_src) + "\n", encoding="utf-8")
    combined_tgt.write_text("\n".join(all_tgt) + "\n", encoding="utf-8")

    print(f"Trộn dữ liệu hoàn tất:")
    print(f"  Gốc     : {len(original_src):,} câu")
    print(f"  Back-BT : {len(back_src):,} câu")
    print(f"  Tổng    : {len(all_src):,} câu")
    if args.tag_synthetic:
        print(f"  Tag     : <bt> được thêm vào đầu câu nguồn tổng hợp")
    print(f"  Output  : {combined_src}, {combined_tgt}")


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path, encoding="utf-8") as fh:
        return sum(1 for _ in fh)


def main() -> None:
    ap = argparse.ArgumentParser(description="Back-translation data pipeline")
    sub = ap.add_subparsers(dest="command", required=True)

    # ---- generate ----
    gen = sub.add_parser("generate", help="Sinh synthetic.src từ mono.tgt")
    gen.add_argument("--reverse-ckpt", required=True, help="Checkpoint mô hình target→source")
    gen.add_argument("--mono", required=True, help="File đơn ngữ tiếng đích (mono.tgt)")
    gen.add_argument("--out-dir", required=True, help="Thư mục output")
    gen.add_argument("--beam", type=int, default=5)
    gen.add_argument("--batch", type=int, default=64)
    gen.add_argument("--max-new-tokens", type=int, default=120)

    # ---- combine ----
    comb = sub.add_parser("combine", help="Trộn dữ liệu gốc + back-translated")
    comb.add_argument("--original-src", required=True)
    comb.add_argument("--original-tgt", required=True)
    comb.add_argument("--back-src", required=True)
    comb.add_argument("--back-tgt", required=True)
    comb.add_argument("--out-dir", required=True)
    comb.add_argument("--tag-synthetic", action="store_true", default=True,
                      help="Thêm tag <bt> vào câu nguồn tổng hợp")
    comb.add_argument("--no-tag-synthetic", action="store_false", dest="tag_synthetic")
    comb.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()

    if args.command == "generate":
        cmd_generate(args)
    elif args.command == "combine":
        cmd_combine(args)


if __name__ == "__main__":
    main()
