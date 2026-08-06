"""Script train toàn diện cho Kaggle GPU — Back-Translation Pipeline EN→VI (Transformer).

Pipeline đầy đủ:
    python kaggle_train.py --stage all

Các stage riêng lẻ:
    python kaggle_train.py --stage forward     # Train EN→VI baseline
    python kaggle_train.py --stage reverse     # Train VI→EN (mô hình ngược)
    python kaggle_train.py --stage backtrans   # Sinh back-translation data
    python kaggle_train.py --stage final       # Train EN→VI trên combined data
    python kaggle_train.py --stage submit      # Sinh file results.csv

Chạy trên Kaggle (dataset tại /kaggle/input/...):
    python kaggle_train.py --stage all \
        --data-dir /kaggle/input/iwslt15-envi-data/iwslt15_en_vi \
        --tokenizer-dir /kaggle/input/iwslt15-envi-data/tokenizer \
        --output-dir /kaggle/working/runs

Kiến trúc: Transformer encoder-decoder (d_model=512, 6 lớp, pre-norm)
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# ── Auto-detect Kaggle vs local ────────────────────────────────────
_KAGGLE_INPUT = Path("/kaggle/input/iwslt15-envi-data")
if _KAGGLE_INPUT.exists():
    _DEFAULT_DATA_DIR = str(_KAGGLE_INPUT / "iwslt15_en_vi")
    _DEFAULT_TOKENIZER_DIR = str(_KAGGLE_INPUT / "tokenizer")
else:
    _DEFAULT_DATA_DIR = str(ROOT / "data" / "iwslt15_en_vi")
    _DEFAULT_TOKENIZER_DIR = str(ROOT / "data" / "tokenizer")


def run(cmd: list[str], desc: str = "") -> None:
    """Chạy lệnh và báo lỗi nếu fail."""
    print(f"\n{'=' * 70}")
    print(f">>> {desc or ' '.join(cmd)}")
    print(f"{'=' * 70}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"LỖI: lệnh thất bại (exit {result.returncode})")
        sys.exit(result.returncode)


def _path_flags(args) -> list[str]:
    return [
        "--data-dir", args.data_dir,
        "--tokenizer-dir", args.tokenizer_dir,
    ]


def _resume_flag(args) -> list[str]:
    return ["--resume", "true"] if args.resume else []


def _train_transformer(
    args,
    src: str,
    tgt: str,
    output_dir: Path,
    epochs: int,
    data_dir: str | None = None,
    desc: str = "",
) -> None:
    """Gọi translate_transformers/train.py với tham số chuẩn."""
    cmd = [
        sys.executable, "translate_transformers/train.py",
        "--epochs", str(epochs),
        "--src-lang", src, "--tgt-lang", tgt,
        "--output-dir", str(output_dir),
        "--data-dir", data_dir or args.data_dir,
        "--tokenizer-dir", args.tokenizer_dir,
        *_resume_flag(args),
    ]
    if args.patience is not None:
        cmd += ["--patience", str(args.patience)]
    run(cmd, desc)


# ═══════════════════════════════════════════════════════════════════
# Stages
# ═══════════════════════════════════════════════════════════════════

def cmd_forward(args) -> None:
    """Train EN→VI baseline (Transformer, 40 epochs)."""
    _train_transformer(
        args, src="en", tgt="vi",
        output_dir=args.output_dir / "forward",
        epochs=args.epochs,
        desc="Stage 1/5: Train EN→VI baseline",
    )


def cmd_reverse(args) -> None:
    """Train VI→EN reverse model (Transformer, 25 epochs)."""
    _train_transformer(
        args, src="vi", tgt="en",
        output_dir=args.output_dir / "reverse",
        epochs=args.epochs_reverse,
        desc="Stage 2/5: Train VI→EN reverse (để back-translate)",
    )


def cmd_backtrans(args) -> None:
    """Sinh dữ liệu back-translation: mono.vi -> synthetic.en."""
    reverse_ckpt = args.output_dir / "reverse" / "best.pt"
    if not reverse_ckpt.exists():
        raise SystemExit(
            f"Chưa có reverse checkpoint: {reverse_ckpt}\n"
            f"Chạy trước: python kaggle_train.py --stage reverse"
        )

    # B1: mono.vi từ train.vi
    mono_path = args.output_dir / "mono.vi"
    if not mono_path.exists():
        shutil.copy(Path(args.data_dir) / "train.vi", mono_path)
        print(f"mono data: {mono_path} ({_count_lines(mono_path)} câu)")

    # B2: Dịch ngược mono.vi -> synthetic.en
    back_dir = args.output_dir / "backtrain"
    back_dir.mkdir(parents=True, exist_ok=True)

    run([
        sys.executable, "translate.py",
        "--model", "transformer",
        "--checkpoint", str(reverse_ckpt),
        "--input", str(mono_path),
        "--output", str(back_dir / "synthetic.en"),
        "--beam", str(args.translate_beam),
        "--batch", str(args.translate_batch),
    ], f"Stage 3/5: Back-translate mono.vi -> synthetic.en (beam={args.translate_beam})")

    shutil.copy(mono_path, back_dir / "mono.vi")

    # B3: Trộn dữ liệu
    tag_flag = ["--tag-synthetic"] if args.tag_synthetic else ["--no-tag-synthetic"]
    run([
        sys.executable, "back_translate.py", "combine",
        "--original-src", str(Path(args.data_dir) / "train.en"),
        "--original-tgt", str(Path(args.data_dir) / "train.vi"),
        "--back-src", str(back_dir / "synthetic.en"),
        "--back-tgt", str(back_dir / "mono.vi"),
        "--out-dir", str(args.output_dir / "combined"),
    ] + tag_flag, "Stage 3/5: Trộn dữ liệu gốc + back-translated")


def cmd_final(args) -> None:
    """Train EN→VI cuối cùng trên combined data."""
    combined_dir = args.output_dir / "combined"
    if not (combined_dir / "combined.en").exists():
        raise SystemExit(
            f"Chưa có combined data: {combined_dir}\n"
            f"Chạy trước: python kaggle_train.py --stage backtrans"
        )

    final_data_dir = args.output_dir / "final_data"
    final_data_dir.mkdir(parents=True, exist_ok=True)

    # Copy dev/test từ dữ liệu gốc
    data_path = Path(args.data_dir)
    for sf in ["tst2012.en", "tst2012.vi", "tst2013.en", "tst2013.vi"]:
        src_f = data_path / sf
        dst_f = final_data_dir / sf
        if src_f.exists() and not dst_f.exists():
            shutil.copy(src_f, dst_f)

    # combined -> train.* trong final_data_dir
    shutil.copy(combined_dir / "combined.en", final_data_dir / "train.en")
    shutil.copy(combined_dir / "combined.vi", final_data_dir / "train.vi")

    _train_transformer(
        args, src="en", tgt="vi",
        output_dir=args.output_dir / "final",
        epochs=args.epochs,
        data_dir=str(final_data_dir),
        desc="Stage 4/5: Train EN→VI final trên combined data",
    )


def cmd_submit(args) -> None:
    """Sinh file results.csv cho competition."""
    final_ckpt = args.output_dir / "final" / "best.pt"
    if not final_ckpt.exists():
        raise SystemExit(
            f"Chưa có final checkpoint: {final_ckpt}\n"
            f"Chạy trước: python kaggle_train.py --stage final"
        )

    # Test input
    if args.test_input and Path(args.test_input).exists():
        input_path = Path(args.test_input)
    else:
        input_path = Path(args.data_dir) / "tst2013.en"
        print("⚠ Không có --test-input, dùng tst2013.en làm test.")

    output_tmp = args.output_dir / "test_hyp.vi"

    run([
        sys.executable, "translate.py",
        "--model", "transformer",
        "--checkpoint", str(final_ckpt),
        "--input", str(input_path),
        "--output", str(output_tmp),
        "--beam", str(args.submit_beam),
        "--batch", str(args.translate_batch),
    ], f"Stage 5/5: Dịch test set (beam={args.submit_beam})")

    # Tạo results.csv — UTF-8, cột "Vietnamese"
    hyps = output_tmp.read_text(encoding="utf-8").strip().splitlines()
    csv_path = Path(args.submit_file)

    with open(csv_path, "w", encoding="utf-8") as fh:
        fh.write("Vietnamese\n")
        for hyp in hyps:
            escaped = hyp.replace('"', '""')
            fh.write(f'"{escaped}"\n')

    print(f"\n✅ results.csv saved: {csv_path}  ({len(hyps):,} dòng)")


def cmd_all(args) -> None:
    """Chạy toàn bộ pipeline tuần tự."""
    for stage_fn in [cmd_forward, cmd_reverse, cmd_backtrans, cmd_final, cmd_submit]:
        stage_fn(args)


def _count_lines(path: Path) -> int:
    with open(path, encoding="utf-8") as fh:
        return sum(1 for _ in fh)


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Kaggle NMT Back-Translation Pipeline EN→VI (Transformer)",
    )
    ap.add_argument("--stage", default="all",
                    choices=["all", "forward", "reverse", "backtrans", "final", "submit"])
    ap.add_argument("--data-dir", default=_DEFAULT_DATA_DIR,
                    help="Thư mục train.*, tst2012.*, tst2013.*")
    ap.add_argument("--tokenizer-dir", default=_DEFAULT_TOKENIZER_DIR,
                    help="Thư mục spm_en.model, spm_vi.model")
    ap.add_argument("--output-dir", default="runs/kaggle",
                    help="Thư mục output cho tất cả stages")
    ap.add_argument("--epochs", type=int, default=40,
                    help="Số epoch forward và final")
    ap.add_argument("--epochs-reverse", type=int, default=25,
                    help="Số epoch reverse model")
    ap.add_argument("--patience", type=int, default=8,
                    help="Early stop sau N epoch không cải thiện")
    ap.add_argument("--translate-beam", type=int, default=5)
    ap.add_argument("--submit-beam", type=int, default=5)
    ap.add_argument("--translate-batch", type=int, default=64)
    ap.add_argument("--tag-synthetic", action="store_true", default=True)
    ap.add_argument("--no-tag-synthetic", action="store_false", dest="tag_synthetic")
    ap.add_argument("--test-input", default=None,
                    help="File test.en của competition")
    ap.add_argument("--submit-file", default="results.csv",
                    help="Đường dẫn file submission (mặc định: results.csv)")
    ap.add_argument("--resume", action="store_true", default=True,
                    help="Tự động resume từ checkpoint nếu có (mặc định: True)")
    ap.add_argument("--no-resume", action="store_false", dest="resume")

    args = ap.parse_args()
    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    {
        "all": cmd_all,
        "forward": cmd_forward,
        "reverse": cmd_reverse,
        "backtrans": cmd_backtrans,
        "final": cmd_final,
        "submit": cmd_submit,
    }[args.stage](args)

    print("\n✅ Hoàn thành!")


if __name__ == "__main__":
    main()
