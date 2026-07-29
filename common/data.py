"""Nạp dữ liệu IWSLT15 En-Vi và gom batch.

Điểm cần nhớ về chuẩn đánh giá của bộ này (theo Luong & Manning, IWSLT 2015):
    train  = train.{en,vi}    133,317 câu
    dev    = tst2012.{en,vi}    1,553 câu
    test   = tst2013.{en,vi}    1,268 câu
Mọi con số BLEU trong các paper đều báo trên tst2013. Ta giữ nguyên quy ước.
"""

from __future__ import annotations

import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Sampler

PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3

SPLIT_FILES = {
    "train": "train",
    "dev": "tst2012",
    "test": "tst2013",
}


def load_split(
    data_dir: str | Path,
    split: str,
    src_lang: str = "en",
    tgt_lang: str = "vi",
) -> tuple[list[str], list[str]]:
    """Trả về (src_lines, tgt_lines) đã strip, cho split trong {train,dev,test}."""
    if split not in SPLIT_FILES:
        raise ValueError(f"split phải thuộc {list(SPLIT_FILES)}, nhận '{split}'")
    data_dir = Path(data_dir)
    stem = SPLIT_FILES[split]

    def _read(lang: str) -> list[str]:
        path = data_dir / f"{stem}.{lang}"
        if not path.exists():
            raise FileNotFoundError(
                f"Thiếu {path}. Chạy trước: bash scripts/download_data.sh"
            )
        with open(path, encoding="utf-8") as fh:
            return [line.strip() for line in fh]

    src, tgt = _read(src_lang), _read(tgt_lang)
    if len(src) != len(tgt):
        raise ValueError(f"{split}: lệch số câu {len(src)} vs {len(tgt)}")
    return src, tgt


class TranslationDataset(Dataset):
    """Mã hoá sẵn toàn bộ split thành id ngay lúc khởi tạo.

    133k câu mã hoá bằng SentencePiece mất vài giây và tốn ~100 MB RAM, đổi lại
    vòng train không tốn CPU cho tokenize nữa — đáng, vì trên Colab T4 thì
    dataloader mới là nút thắt chứ không phải GPU.
    """

    def __init__(
        self,
        src_lines: list[str],
        tgt_lines: list[str],
        tok_src,
        tok_tgt,
        max_len: int = 100,
        filter_long: bool = True,
    ) -> None:
        src_ids = tok_src.encode_batch(src_lines)
        tgt_ids = tok_tgt.encode_batch(tgt_lines)

        self.src: list[list[int]] = []
        self.tgt: list[list[int]] = []
        self.raw_src: list[str] = []
        self.raw_tgt: list[str] = []
        n_dropped = 0

        for s_ids, t_ids, s_raw, t_raw in zip(src_ids, tgt_ids, src_lines, tgt_lines):
            # +2 cho <s> </s> ở phía target, +1 cho </s> ở phía source
            too_long = len(s_ids) + 1 > max_len or len(t_ids) + 2 > max_len
            if filter_long and (too_long or not s_ids or not t_ids):
                n_dropped += 1
                continue
            self.src.append(s_ids + [EOS_ID])
            self.tgt.append([BOS_ID] + t_ids + [EOS_ID])
            self.raw_src.append(s_raw)
            self.raw_tgt.append(t_raw)

        self.n_dropped = n_dropped
        self.lengths = [len(s) for s in self.src]

    def __len__(self) -> int:
        return len(self.src)

    def __getitem__(self, idx: int):
        return self.src[idx], self.tgt[idx], idx


def collate_batch(batch):
    """Pad về cùng độ dài; trả về tensor batch-first.

    tgt_in  = tgt[:-1]  (đầu vào decoder, bắt đầu bằng <s>)
    tgt_out = tgt[1:]   (nhãn, kết thúc bằng </s>)  -> teacher forcing
    """
    srcs, tgts, idxs = zip(*batch)
    max_src = max(len(s) for s in srcs)
    max_tgt = max(len(t) for t in tgts)

    src_pad = torch.full((len(srcs), max_src), PAD_ID, dtype=torch.long)
    tgt_pad = torch.full((len(tgts), max_tgt), PAD_ID, dtype=torch.long)
    src_len = torch.zeros(len(srcs), dtype=torch.long)

    for i, (s, t) in enumerate(zip(srcs, tgts)):
        src_pad[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        tgt_pad[i, : len(t)] = torch.tensor(t, dtype=torch.long)
        src_len[i] = len(s)

    return {
        "src": src_pad,
        "src_len": src_len,
        "tgt_in": tgt_pad[:, :-1].contiguous(),
        "tgt_out": tgt_pad[:, 1:].contiguous(),
        "index": torch.tensor(idxs, dtype=torch.long),
    }


class TokenBatchSampler(Sampler):
    """Gom batch theo SỐ TOKEN thay vì số câu.

    Batch cố định theo số câu làm lãng phí lớn: một batch toàn câu ngắn thì GPU
    nhàn, một batch dính câu 100 token thì suýt OOM. Gom theo token giữ khối
    lượng tính toán mỗi bước gần như không đổi, cho throughput đều và cho phép
    đẩy `max_tokens` sát trần bộ nhớ. Đây là cách fairseq/Marian vẫn làm.

    Sắp xếp theo độ dài trước khi cắt batch cũng giảm padding waste; sau đó
    xáo thứ tự các batch để gradient không bị thiên theo độ dài câu.
    """

    def __init__(
        self,
        lengths: list[int],
        max_tokens: int = 4096,
        shuffle: bool = True,
        seed: int = 42,
        bucket_multiplier: int = 100,
    ) -> None:
        self.lengths = lengths
        self.max_tokens = max_tokens
        self.shuffle = shuffle
        self.seed = seed
        self.bucket_multiplier = bucket_multiplier
        self.epoch = 0
        self._batches = self._build(seed)

    def _build(self, seed: int) -> list[list[int]]:
        rng = random.Random(seed)
        indices = list(range(len(self.lengths)))

        if self.shuffle:
            # Xáo trước, rồi sort theo độ dài trong từng "mega-batch". Cách này
            # vừa giữ tính ngẫu nhiên vừa giữ được lợi ích gom câu cùng độ dài.
            rng.shuffle(indices)
            mega = self.max_tokens * self.bucket_multiplier
            chunk_size = max(1, mega // max(1, max(self.lengths)))
            chunks = [
                indices[i : i + chunk_size] for i in range(0, len(indices), chunk_size)
            ]
            indices = [i for c in chunks for i in sorted(c, key=lambda j: self.lengths[j])]
        else:
            indices.sort(key=lambda j: self.lengths[j])

        batches: list[list[int]] = []
        cur: list[int] = []
        cur_max = 0
        for idx in indices:
            new_max = max(cur_max, self.lengths[idx])
            # chi phí ~ (số câu) x (độ dài dài nhất trong batch)
            if cur and new_max * (len(cur) + 1) > self.max_tokens:
                batches.append(cur)
                cur, cur_max = [idx], self.lengths[idx]
            else:
                cur.append(idx)
                cur_max = new_max
        if cur:
            batches.append(cur)

        if self.shuffle:
            rng.shuffle(batches)
        return batches

    def set_epoch(self, epoch: int) -> None:
        """Đổi thứ tự batch mỗi epoch (gọi từ vòng train)."""
        self.epoch = epoch
        if self.shuffle:
            self._batches = self._build(self.seed + epoch)

    def __iter__(self):
        return iter(self._batches)

    def __len__(self) -> int:
        return len(self._batches)


def build_dataloader(
    dataset: TranslationDataset,
    max_tokens: int = 4096,
    shuffle: bool = True,
    num_workers: int = 2,
    seed: int = 42,
) -> DataLoader:
    sampler = TokenBatchSampler(
        dataset.lengths, max_tokens=max_tokens, shuffle=shuffle, seed=seed
    )
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate_batch,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
