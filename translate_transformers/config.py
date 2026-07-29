"""Cấu hình cho Case 1 — Transformer.

Preset mặc định là `transformer_iwslt` (tương đương `transformer_iwslt_de_en`
của fairseq). Cần nhấn mạnh: KHÔNG dùng "Transformer base" của Vaswani et al.
(d_model=512, ffn=2048, 6 lớp, ~65M tham số) cho bộ này. IWSLT15 En-Vi chỉ có
133k câu — base sẽ overfit, và trong thực tế các paper báo BLEU cao trên bộ này
đều dùng cấu hình nhỏ hơn với dropout cao (0.3). Chi tiết lý do xem README,
mục "Case 1 — rút ra được gì".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class TransformerConfig:
    # ---- dữ liệu ----
    data_dir: str = str(ROOT / "data" / "iwslt15_en_vi")
    tokenizer_dir: str = str(ROOT / "data" / "tokenizer")
    src_lang: str = "en"
    tgt_lang: str = "vi"
    max_len: int = 100          # lọc câu train dài hơn mức này (theo subword)

    # ---- kiến trúc ----
    d_model: int = 512
    n_heads: int = 4            # 4 head cho d_model=512 -> head_dim=128, hợp low-resource
    n_enc_layers: int = 6
    n_dec_layers: int = 6
    d_ff: int = 1024            # base dùng 2048; 1024 giảm overfit trên 133k câu
    dropout: float = 0.3        # cao hơn base (0.1) — bắt buộc với dữ liệu nhỏ
    attn_dropout: float = 0.1
    activation: str = "relu"
    norm_first: bool = True     # pre-norm: ổn định, ít phụ thuộc warmup
    tie_embeddings: bool = True  # buộc embedding decoder với ma trận output

    # ---- tối ưu ----
    epochs: int = 30
    max_tokens: int = 8192      # số token target mỗi batch (T4 16GB chịu được)
    accum_steps: int = 1
    lr_scale: float = 2.0       # hệ số nhân cho lịch inverse-sqrt
    warmup_steps: int = 4000
    betas: tuple[float, float] = (0.9, 0.98)
    eps: float = 1e-9
    weight_decay: float = 1e-4
    label_smoothing: float = 0.1
    clip_norm: float = 1.0
    patience: int = 5           # early stop nếu BLEU dev đứng yên 5 epoch

    # ---- decode ----
    beam_size: int = 5
    length_penalty: float = 1.0
    max_new_tokens: int = 120

    # ---- hệ thống ----
    amp: bool = True            # fp16 trên T4 (Turing có tensor core fp16)
    resume: bool = False        # train tiếp từ runs/*/last.pt nếu Colab ngắt phiên
    seed: int = 42
    num_workers: int = 2
    output_dir: str = str(ROOT / "runs" / "transformer")

    # điền lúc runtime
    src_vocab_size: int = field(default=0)
    tgt_vocab_size: int = field(default=0)


# Preset nhỏ để chạy thử nhanh (smoke test) trước khi train thật
SMOKE = dict(
    d_model=128, n_heads=4, n_enc_layers=2, n_dec_layers=2, d_ff=256,
    epochs=1, max_tokens=2048, warmup_steps=100, patience=0, beam_size=2,
)
