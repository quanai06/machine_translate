"""Shared building blocks for both NMT case studies.

Cả `translate_transformers` và `translate_seq2seq` đều import từ đây, nên hai
model dùng CHUNG: tokenizer, cách chia batch, cách tính BLEU và cách đo
benchmark. Đó là điều kiện bắt buộc để phép so sánh giữa hai kiến trúc là
công bằng (apples-to-apples).
"""

from .data import (
    PAD_ID,
    UNK_ID,
    BOS_ID,
    EOS_ID,
    TranslationDataset,
    build_dataloader,
    load_split,
)
from .tokenizer import SentencePieceTokenizer, load_tokenizers
from .metrics import corpus_bleu, moses_unescape
from .bench import BenchmarkRecorder, count_parameters, measure_latency

__all__ = [
    "PAD_ID",
    "UNK_ID",
    "BOS_ID",
    "EOS_ID",
    "TranslationDataset",
    "build_dataloader",
    "load_split",
    "SentencePieceTokenizer",
    "load_tokenizers",
    "corpus_bleu",
    "moses_unescape",
    "BenchmarkRecorder",
    "count_parameters",
    "measure_latency",
]
