"""So sánh Case 1 (Transformer) và Case 2 (Seq2Seq+Attention).

    python compare.py                  # in bảng + lưu Markdown
    python compare.py --plots          # kèm biểu đồ learning curve

Đọc `runs/*/benchmark.json` do hai script train sinh ra, dựng bảng so sánh, và
tuỳ chọn vẽ biểu đồ. Đây là đầu ra chính để đưa vào báo cáo.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Con số tham chiếu công bố trên CÙNG bộ tst2013, để đối chiếu xem bản cài đặt
# của ta có nằm trong vùng hợp lý không. Nguồn ghi trong README.
PUBLISHED = {
    "tensorflow/nmt (LSTM+Luong attn, greedy)": 25.5,
    "tensorflow/nmt (LSTM+Luong attn, beam=10)": 26.1,
    "Luong & Manning 2015 (hệ thống IWSLT'15)": 23.3,
    "Transformers without Tears 2019 (SOTA thời điểm đó)": 32.8,
}


def load_run(name: str) -> dict | None:
    path = ROOT / "runs" / name / "benchmark.json"
    if not path.exists():
        print(f"  (chưa có {path} — hãy train {name} trước)")
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(v, nd=2):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    if isinstance(v, int) and v > 9999:
        return f"{v:,}"
    return str(v)


def build_table(runs: dict[str, dict]) -> str:
    names = list(runs)
    rows: list[tuple[str, list]] = []

    def add(label, getter, nd=2):
        rows.append((label, [getter(runs[n]) for n in names]))

    s = lambda r: r.get("summary", {})
    t = lambda r, k, f: (s(r).get("test", {}).get(k) or {}).get(f)

    rows.append(("**Chất lượng dịch (tst2013)**", ["", ""][: len(names)]))
    add("BLEU tokenized — greedy", lambda r: t(r, "greedy", "bleu_tokenized"))
    add("BLEU tokenized — beam", lambda r: next(
        (v.get("bleu_tokenized") for k, v in s(r).get("test", {}).items()
         if k.startswith("beam")), None))
    add("BLEU detokenized (sacreBLEU 13a)", lambda r: next(
        (v.get("bleu_detok") for k, v in s(r).get("test", {}).items()
         if k.startswith("beam")), None))
    add("chrF2", lambda r: next(
        (v.get("chrf2") for k, v in s(r).get("test", {}).items()
         if k.startswith("beam")), None))
    add("BLEU tốt nhất trên dev (tst2012)", lambda r: s(r).get("best_dev_bleu"))

    rows.append(("**Kích thước mô hình**", ["", ""][: len(names)]))
    add("Tổng tham số", lambda r: s(r).get("params_total"), 0)
    add("Tham số ngoài embedding", lambda r: s(r).get("params_non_embedding"), 0)
    add("Tham số embedding", lambda r: s(r).get("params_embedding"), 0)

    rows.append(("**Chi phí huấn luyện**", ["", ""][: len(names)]))
    add("Thời gian train (giây)", lambda r: s(r).get("train_wallclock_sec"), 1)
    add("Trung vị thời gian/epoch (giây)", lambda r: s(r).get("median_epoch_time_sec"), 1)
    add("Throughput (token/giây)", lambda r: (r.get("epochs") or [{}])[0].get("tokens_per_sec"), 0)
    add("Epoch tốt nhất", lambda r: s(r).get("best_epoch"), 0)
    add("Đỉnh bộ nhớ GPU (GB)", lambda r: s(r).get("peak_gpu_mem_gb"))

    rows.append(("**Tốc độ suy luận**", ["", ""][: len(names)]))
    add("Câu/giây (greedy)", lambda r: s(r).get("decode_sentences_per_sec"))
    add("ms mỗi câu (greedy)", lambda r: s(r).get("decode_ms_per_sentence"))

    header = "| Tiêu chí | " + " | ".join(n.capitalize() for n in names) + " |"
    sep = "|---|" + "---|" * len(names)
    lines = [header, sep]
    for label, vals in rows:
        lines.append(f"| {label} | " + " | ".join(fmt(v) for v in vals) + " |")
    return "\n".join(lines)


def speed_ratio(runs: dict[str, dict]) -> str:
    """Câu kết luận về tốc độ — phần người đọc báo cáo quan tâm nhất."""
    if len(runs) < 2:
        return ""
    tf, s2s = runs.get("transformer"), runs.get("seq2seq")
    if not (tf and s2s):
        return ""
    out = []
    a = tf["summary"].get("median_epoch_time_sec")
    b = s2s["summary"].get("median_epoch_time_sec")
    pa = tf["summary"].get("params_total")
    pb = s2s["summary"].get("params_total")
    if a and b:
        ratio = b / a
        if ratio > 1.3:
            out.append(
                f"- Mỗi epoch, Transformer nhanh hơn {ratio:.1f}× so với Seq2Seq "
                f"({a:.0f}s vs {b:.0f}s). Teacher forcing của Transformer chạy song "
                f"song toàn bộ chuỗi target trong một lần, còn LSTM có input feeding "
                f"buộc phải lặp tuần tự qua từng timestep."
            )
        elif ratio < 0.77:
            out.append(
                f"- Mỗi epoch, Seq2Seq nhanh hơn {1 / ratio:.1f}× so với Transformer "
                f"({b:.0f}s vs {a:.0f}s), dù chạy tuần tự — nhiều khả năng do chênh "
                f"lệch số tham số và chi phí attention O(T²)."
            )
        else:
            extra = ""
            if pa and pb:
                extra = (f" Seq2Seq chỉ có {pb / 1e6:.1f}M tham số so với "
                         f"{pa / 1e6:.1f}M, tức khoảng {pa / pb:.1f}× ít phép tính "
                         f"mỗi token, bù lại phần thiệt do chạy tuần tự;")
            out.append(
                f"- Thời gian mỗi epoch gần như bằng nhau ({a:.0f}s vs {b:.0f}s, lệch "
                f"{abs(ratio - 1) * 100:.0f}%), ngược với kỳ vọng thông thường rằng "
                f"Transformer phải nhanh hơn hẳn nhờ song song hoá.{extra} thêm nữa "
                f"attention của Transformer tốn O(T²) theo độ dài câu. Ở quy mô model "
                f"này, hai yếu tố triệt tiêu nhau."
            )
    a = tf["summary"].get("decode_ms_per_sentence")
    b = s2s["summary"].get("decode_ms_per_sentence")
    if a and b:
        if a > b:
            out.append(
                f"- Khi SINH câu, Transformer chậm hơn {a / b:.1f}× ({a:.1f} vs "
                f"{b:.1f} ms/câu). Đây là hạn chế của BẢN CÀI ĐẶT chứ không phải của "
                f"kiến trúc: `greedy_decode` ở đây không cache key/value, nên mỗi bước "
                f"sinh phải chạy lại decoder trên toàn bộ tiền tố, tổng chi phí O(T²). "
                f"LSTM không gặp vấn đề này vì state đã tóm tắt sẵn quá khứ trong một "
                f"vector. Thêm KV cache sẽ thu hẹp khoảng cách này."
            )
        else:
            out.append(
                f"- Khi SINH câu, Transformer nhanh hơn {b / a:.1f}× ({a:.1f} vs "
                f"{b:.1f} ms/câu)."
            )
    return "\n".join(out)


def make_plots(runs: dict[str, dict], out_dir: Path) -> Path | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (không có matplotlib — bỏ qua biểu đồ)")
        return None

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    colors = {"transformer": "#2563eb", "seq2seq": "#dc2626"}

    for name, run in runs.items():
        ep = run.get("epochs", [])
        if not ep:
            continue
        x = [e["epoch"] for e in ep]
        c = colors.get(name, None)
        axes[0].plot(x, [e.get("dev_bleu_greedy") for e in ep], marker="o", label=name, color=c)
        axes[1].plot(x, [e.get("dev_ppl") for e in ep], marker="o", label=name, color=c)
        axes[1].plot(x, [e.get("train_ppl") for e in ep], marker=".", linestyle="--",
                     label=f"{name} (train)", color=c, alpha=0.5)
        cum = []
        acc = 0.0
        for e in ep:
            acc += e.get("epoch_time_sec", 0) / 60
            cum.append(acc)
        axes[2].plot(cum, [e.get("dev_bleu_greedy") for e in ep], marker="o", label=name, color=c)

    axes[0].set(xlabel="epoch", ylabel="BLEU trên dev (greedy)", title="Chất lượng theo epoch")
    axes[1].set(xlabel="epoch", ylabel="perplexity", title="Perplexity (nét liền = dev)")
    axes[1].set_yscale("log")
    axes[2].set(xlabel="thời gian train tích luỹ (phút)", ylabel="BLEU trên dev",
                title="Chất lượng đổi lấy thời gian")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()

    path = out_dir / "comparison.png"
    fig.savefig(path, dpi=140)
    print(f"  biểu đồ -> {path}")
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plots", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "runs" / "COMPARISON.md"))
    args = ap.parse_args()

    runs = {}
    for name in ("transformer", "seq2seq"):
        r = load_run(name)
        if r:
            runs[name] = r
    if not runs:
        print("Chưa có kết quả nào. Train ít nhất một model trước.")
        return

    table = build_table(runs)
    print("\n" + table + "\n")

    parts = [
        "# So sánh Case 1 (Transformer) vs Case 2 (Seq2Seq + Attention)",
        "",
        "Bộ dữ liệu: IWSLT'15 English-Vietnamese. "
        "Train `train.{en,vi}` (133,317 câu), dev `tst2012`, test `tst2013`.",
        "Cả hai model dùng CHUNG tokenizer, chung cách chia batch, chung công thức "
        "BLEU — nên chênh lệch dưới đây phản ánh kiến trúc, không phải kỹ thuật train.",
        "",
        table,
        "",
        "## Nhận xét về tốc độ",
        "",
        speed_ratio(runs) or "_(cần cả hai model để so sánh)_",
        "",
        "## Đối chiếu với số liệu đã công bố (cùng tst2013)",
        "",
        "| Hệ thống | BLEU tokenized |",
        "|---|---|",
    ]
    for k, v in PUBLISHED.items():
        parts.append(f"| {k} | {v} |")
    parts += [
        "",
        "> Lưu ý khi đối chiếu: các con số trên là *tokenized BLEU* "
        "(`multi-bleu.perl`) trên văn bản đã tokenize theo Moses. Dòng "
        "'BLEU tokenized' trong bảng của ta được tính bằng "
        "`sacrebleu(tokenize='none')`, tương đương. Đừng so nhầm với dòng "
        "'BLEU detokenized'.",
    ]

    if args.plots:
        p = make_plots(runs, Path(args.out).parent)
        if p:
            parts += ["", "## Biểu đồ", "", f"![so sánh]({p.name})"]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(parts), encoding="utf-8")
    print(f"Đã lưu: {args.out}")


if __name__ == "__main__":
    main()
