"""Đo benchmark theo cùng một cách cho cả hai model.

Bài yêu cầu "so sánh kết quả", mà so sánh chỉ có giá trị khi mọi con số được đo
bằng cùng một thước. Module này là cái thước đó: cùng cách đếm tham số, cùng
cách đo throughput, cùng cách đo latency inference, cùng định dạng file kết quả
để `compare.py` đọc lên vẽ bảng.
"""

from __future__ import annotations

import json
import platform
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import torch


def count_parameters(model: torch.nn.Module) -> dict[str, int]:
    """Đếm tham số, tách riêng phần embedding.

    Tách embedding ra là cần thiết: với vocab 8k x 2 phía, embedding chiếm
    phần đáng kể tổng tham số và KHÔNG phản ánh độ sâu tính toán của kiến
    trúc. So sánh "số tham số" mà gộp cả embedding dễ dẫn tới kết luận sai.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    emb = sum(
        m.weight.numel()
        for m in model.modules()
        if isinstance(m, torch.nn.Embedding)
    )
    return {
        "params_total": total,
        "params_trainable": trainable,
        "params_embedding": emb,
        "params_non_embedding": total - emb,
    }


def gpu_info() -> dict[str, str | int]:
    if not torch.cuda.is_available():
        return {"device": "cpu", "device_name": platform.processor() or "cpu"}
    props = torch.cuda.get_device_properties(0)
    return {
        "device": "cuda",
        "device_name": props.name,
        "device_total_mem_gb": round(props.total_memory / 1024**3, 2),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda or "n/a",
    }


@torch.no_grad()
def measure_latency(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    max_new_tokens: int = 100,
    n_batches: int = 20,
    warmup: int = 3,
) -> dict[str, float]:
    """Đo tốc độ sinh câu (greedy) — số câu/giây và mili-giây mỗi câu.

    Đây là chỉ số mà so sánh Transformer vs LSTM mới lộ ra sự thật thú vị:
    lúc TRAIN, Transformer song song hoá được toàn bộ chuỗi target nên nhanh
    hơn hẳn; nhưng lúc SINH thì cả hai đều autoregressive từng token một, nên
    khoảng cách thu hẹp lại nhiều (thậm chí Transformer có thể chậm hơn nếu
    không cache, vì mỗi bước phải attend lại toàn bộ tiền tố).
    """
    model.eval()
    times: list[float] = []
    n_sents = 0

    for i, batch in enumerate(dataloader):
        if i >= warmup + n_batches:
            break
        src = batch["src"].to(device)
        src_len = batch["src_len"].to(device)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        model.greedy_decode(src, src_len, max_new_tokens=max_new_tokens)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0

        if i >= warmup:  # bỏ vài batch đầu cho GPU "nóng máy"
            times.append(dt)
            n_sents += src.size(0)

    if not times:
        return {}
    total = sum(times)
    return {
        "decode_sentences_per_sec": round(n_sents / total, 2),
        "decode_ms_per_sentence": round(1000 * total / n_sents, 2),
        "decode_batches_measured": len(times),
    }


@dataclass
class BenchmarkRecorder:
    """Gom mọi số đo của một lần chạy rồi ghi ra JSON + CSV.

    `compare.py` đọc đúng các file này, nên đừng đổi tên khoá tuỳ tiện.
    """

    name: str
    output_dir: Path
    config: dict = field(default_factory=dict)
    epochs: list[dict] = field(default_factory=list)
    summary: dict = field(default_factory=dict)
    _t_start: float = field(default=0.0, repr=False)

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.summary.update(gpu_info())

    def start(self) -> None:
        self._t_start = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def log_epoch(self, **kw) -> None:
        self.epochs.append(kw)
        self.flush()

    def finish(self, **kw) -> None:
        self.summary["train_wallclock_sec"] = round(time.perf_counter() - self._t_start, 1)
        if torch.cuda.is_available():
            self.summary["peak_gpu_mem_gb"] = round(
                torch.cuda.max_memory_allocated() / 1024**3, 2
            )
        if self.epochs:
            per_epoch = [e["epoch_time_sec"] for e in self.epochs if "epoch_time_sec" in e]
            if per_epoch:
                self.summary["median_epoch_time_sec"] = round(
                    sorted(per_epoch)[len(per_epoch) // 2], 1
                )
        self.summary.update(kw)
        self.flush()

    def flush(self) -> None:
        payload = {
            "name": self.name,
            "config": self.config,
            "summary": self.summary,
            "epochs": self.epochs,
        }
        (self.output_dir / "benchmark.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        if self.epochs:
            keys = sorted({k for e in self.epochs for k in e})
            lines = [",".join(keys)]
            for e in self.epochs:
                lines.append(",".join(str(e.get(k, "")) for k in keys))
            (self.output_dir / "history.csv").write_text(
                "\n".join(lines), encoding="utf-8"
            )

    def as_dict(self) -> dict:
        return asdict(self)
