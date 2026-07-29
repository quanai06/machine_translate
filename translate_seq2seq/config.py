"""Cấu hình cho Case 2 — Seq2Seq + Attention (bám theo tensorflow/nmt).

Mặc định ở đây tái hiện đúng cấu hình mà `tensorflow/nmt` công bố cho benchmark
IWSLT'15 En-Vi (README của repo đó):

    "2-layer LSTMs of 512 units with bidirectional encoder (i.e., 1
     bidirectional layer for the encoder), embedding dim is 512.
     LuongAttention (scale=True) with dropout keep_prob of 0.8."
    "SGD with learning rate 1.0 ... train 12K steps (~12 epochs); after 8K
     steps, halve the learning rate every 1K steps."

Kết quả họ báo: tst2013 BLEU 25.5 (greedy) / 26.1 (beam=10).

Khác biệt duy nhất mà ta cố ý giữ: optimizer mặc định là Adam thay vì SGD 1.0.
Lý do — model Case 1 dùng Adam, và nếu để hai model chạy hai họ optimizer khác
nhau thì chênh lệch BLEU quan sát được lẫn lộn giữa "do kiến trúc" và "do cách
tối ưu". Muốn tái hiện đúng số của tensorflow/nmt thì đặt `optimizer="sgd"`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Seq2SeqConfig:
    # ---- dữ liệu ----
    data_dir: str = str(ROOT / "data" / "iwslt15_en_vi")
    tokenizer_dir: str = str(ROOT / "data" / "tokenizer")
    src_lang: str = "en"
    tgt_lang: str = "vi"
    max_len: int = 100

    # ---- kiến trúc ----
    emb_dim: int = 512
    hidden_dim: int = 512
    n_enc_layers: int = 1        # 1 lớp bi-LSTM = 2 "lớp" theo cách đếm của tf/nmt
    n_dec_layers: int = 2
    bidirectional: bool = True
    attention: str = "general"   # {"general", "dot", "concat"} — Luong et al. 2015
    scale_attention: bool = True  # tương ứng scale=True của LuongAttention
    input_feeding: bool = True   # đóng góp chính của Luong et al. 2015
    dropout: float = 0.2         # = 1 - keep_prob(0.8)
    tie_embeddings: bool = True

    # ---- tối ưu ----
    optimizer: str = "adam"      # "adam" (mặc định) hoặc "sgd" (bám tf/nmt)
    lr: float = 1e-3             # dùng 1.0 nếu optimizer="sgd"
    epochs: int = 30
    max_tokens: int = 8192
    accum_steps: int = 1
    warmup_steps: int = 0        # LSTM không cần warmup như Transformer
    label_smoothing: float = 0.1
    clip_norm: float = 5.0       # tf/nmt dùng max_gradient_norm=5.0
    weight_decay: float = 0.0
    patience: int = 5
    lr_decay_start_ratio: float = 2 / 3  # bắt đầu giảm lr sau 2/3 số epoch
    lr_decay_factor: float = 0.5

    # ---- decode ----
    beam_size: int = 10          # tf/nmt báo số tốt nhất ở beam=10
    length_penalty: float = 1.0
    max_new_tokens: int = 120

    # ---- hệ thống ----
    amp: bool = True
    resume: bool = False        # train tiếp từ runs/*/last.pt nếu Colab ngắt phiên
    seed: int = 42
    num_workers: int = 2
    output_dir: str = str(ROOT / "runs" / "seq2seq")

    src_vocab_size: int = field(default=0)
    tgt_vocab_size: int = field(default=0)


SMOKE = dict(
    emb_dim=128, hidden_dim=128, n_dec_layers=1, epochs=1,
    max_tokens=2048, patience=0, beam_size=2,
)
