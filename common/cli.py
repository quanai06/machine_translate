"""Tiện ích dùng chung cho hai script train: seed, argparse từ dataclass, in báo cáo."""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch

# Cho phép `python translate_transformers/train.py` chạy từ thư mục gốc repo
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def set_seed(seed: int) -> None:
    """Cố định seed. Lưu ý: cuDNN vẫn có thể phi tất định ở một vài kernel;
    muốn tất định tuyệt đối phải bật `torch.use_deterministic_algorithms(True)`
    và chấp nhận chậm hơn. Ta không bật, nhưng ghi lại seed vào benchmark.json
    để lần chạy sau còn tái hiện gần đúng."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def add_config_args(parser: argparse.ArgumentParser, cfg_cls) -> argparse.ArgumentParser:
    """Sinh cờ dòng lệnh `--ten-truong` cho mọi field của dataclass config."""
    for f in fields(cfg_cls):
        if f.type in (tuple, "tuple[float, float]"):
            continue
        flag = "--" + f.name.replace("_", "-")
        if f.type in (bool, "bool"):
            parser.add_argument(flag, type=lambda x: x.lower() in ("1", "true", "yes"),
                                default=None, help=f"(bool) mặc định: {getattr(cfg_cls(), f.name)}")
        elif f.type in (int, "int"):
            parser.add_argument(flag, type=int, default=None)
        elif f.type in (float, "float"):
            parser.add_argument(flag, type=float, default=None)
        else:
            parser.add_argument(flag, type=str, default=None)
    parser.add_argument("--smoke", action="store_true",
                        help="Chạy thử cực nhanh để kiểm tra pipeline không lỗi")
    return parser


def apply_overrides(cfg, args, smoke_preset: dict | None = None):
    if getattr(args, "smoke", False) and smoke_preset:
        for k, v in smoke_preset.items():
            setattr(cfg, k, v)
    for k, v in vars(args).items():
        if k == "smoke" or v is None:
            continue
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def print_header(title: str, cfg, device) -> None:
    print("=" * 78)
    print(f"  {title}")
    print("=" * 78)
    print(f"  thiết bị : {device} "
          f"({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    for k, v in vars(cfg).items():
        print(f"  {k:<22} {v}")
    print("-" * 78)


def dump_json(path: Path, payload: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
