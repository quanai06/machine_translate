"""SentencePiece BPE tokenizer dùng chung cho cả hai model.

Vì sao không dùng vocab word-level 17k/7.7k gốc của Stanford?
  - Vocab word-level cố định gây lỗi OOV nặng: mọi từ lạ thành <unk>, và với
    tiếng Việt (âm tiết rời) thì tỉ lệ <unk> ở phía target khá cao.
  - Các paper hiện đại trên chính bộ IWSLT15 En-Vi (xem README, mục Case 1)
    đều dùng subword/BPE và đó là nguồn gốc của phần lớn khoảng cách BLEU so
    với baseline word-level.
  - Quan trọng nhất: hai model phải dùng CÙNG bộ tokenizer thì so sánh mới
    hợp lệ. Nên ta train một lần rồi cả hai case cùng nạp.

Corpus IWSLT15 đã được tokenize sẵn theo Moses (dấu câu tách rời, thực thể
`&apos;` `&quot;` ...). Ta giữ nguyên dạng đó khi train SentencePiece để BLEU
tính ra so sánh được với các con số công bố trong paper.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import sentencepiece as spm

# Thứ tự id đặc biệt phải khớp với common/data.py
PIECE_PAD, PIECE_UNK, PIECE_BOS, PIECE_EOS = "<pad>", "<unk>", "<s>", "</s>"


@dataclass
class SentencePieceTokenizer:
    """Bọc mỏng quanh `spm.SentencePieceProcessor` với id đặc biệt cố định."""

    sp: spm.SentencePieceProcessor
    lang: str

    @property
    def vocab_size(self) -> int:
        return self.sp.get_piece_size()

    def encode(self, text: str) -> list[int]:
        """Chuỗi -> list id, KHÔNG kèm <s> </s> (dataset tự thêm)."""
        return self.sp.encode(text, out_type=int)

    def decode(self, ids: list[int]) -> str:
        """List id -> chuỗi; tự bỏ mọi token đặc biệt."""
        clean = [i for i in ids if i > EOS_PIECE_ID]
        return self.sp.decode(clean)

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        return self.sp.encode(texts, out_type=int)


# id đặc biệt lớn nhất; token thật bắt đầu từ id 4
EOS_PIECE_ID = 3


def train_sentencepiece(
    input_file: str | os.PathLike,
    model_prefix: str | os.PathLike,
    vocab_size: int,
    character_coverage: float = 1.0,
    model_type: str = "bpe",
) -> Path:
    """Train một model SentencePiece và trả về đường dẫn file .model.

    `character_coverage=1.0` là đúng cho cả tiếng Anh và tiếng Việt (bảng chữ
    cái nhỏ, khác với tiếng Nhật/Trung nơi người ta dùng 0.9995).
    """
    model_prefix = Path(model_prefix)
    model_prefix.parent.mkdir(parents=True, exist_ok=True)
    spm.SentencePieceTrainer.train(
        input=str(input_file),
        model_prefix=str(model_prefix),
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=character_coverage,
        # Ép thứ tự id đặc biệt: pad=0, unk=1, bos=2, eos=3
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        pad_piece=PIECE_PAD,
        unk_piece=PIECE_UNK,
        bos_piece=PIECE_BOS,
        eos_piece=PIECE_EOS,
        # Corpus đã tokenize sẵn -> không cho SP nuốt/di chuyển khoảng trắng
        normalization_rule_name="nmt_nfkc",
        input_sentence_size=0,
        shuffle_input_sentence=False,
        num_threads=os.cpu_count() or 4,
    )
    return model_prefix.with_suffix(".model")


def load_tokenizers(
    tokenizer_dir: str | os.PathLike,
    src_lang: str = "en",
    tgt_lang: str = "vi",
) -> tuple[SentencePieceTokenizer, SentencePieceTokenizer]:
    """Nạp cặp tokenizer (source, target) đã train sẵn."""
    tokenizer_dir = Path(tokenizer_dir)
    out = []
    for lang in (src_lang, tgt_lang):
        model_path = tokenizer_dir / f"spm_{lang}.model"
        if not model_path.exists():
            raise FileNotFoundError(
                f"Chưa có {model_path}. Chạy trước:\n"
                f"    python scripts/prepare.py"
            )
        sp = spm.SentencePieceProcessor()
        sp.load(str(model_path))
        out.append(SentencePieceTokenizer(sp=sp, lang=lang))
    return out[0], out[1]
