"""Case 1 — Transformer encoder-decoder (Vaswani et al., 2017).

Cài đặt bằng `nn.TransformerEncoder`/`nn.TransformerDecoder` với `norm_first=True`.

Ba lựa chọn thiết kế lệch khỏi paper gốc, đều có lý do rút ra từ literature
(xem README để biết paper nào nói gì):

1. **Pre-norm** thay vì post-norm. Nguyen & Salazar (IWSLT 2019) chỉ ra pre-norm
   cho phép train không cần warmup dài, hội tụ ổn định ở learning rate lớn, và
   ở chế độ low-resource thì tốt hơn hẳn — chính họ lập kỷ lục 32.8 BLEU trên
   đúng bộ IWSLT'15 En-Vi này.

2. **Mô hình nhỏ + dropout 0.3**. 133k câu là rất ít. Transformer base overfit
   rõ; cấu hình kiểu `transformer_iwslt` (d_ff=1024, 4 head) mới là baseline
   hợp lý.

3. **Weight tying** giữa embedding decoder và ma trận chiếu output. Giảm ~4M
   tham số và là quy ước chuẩn ở low-resource.

Giao diện bắt buộc (common/engine.py gọi vào):
    forward(src, src_len, tgt_in) -> logits [B, T, V]
    greedy_decode(src, src_len, max_new_tokens) -> ids [B, T]
    beam_search(src, src_len, beam_size, max_new_tokens, length_penalty) -> [B, T]
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from common.data import BOS_ID, EOS_ID, PAD_ID


class SinusoidalPositionalEncoding(nn.Module):
    """Positional encoding dạng sin/cos của paper gốc.

    Không học tham số, và tổng quát hoá được sang câu dài hơn lúc train — hữu
    ích khi decode gặp câu dài bất thường.
    """

    def __init__(self, d_model: int, max_len: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        return self.dropout(x + self.pe[:, offset : offset + x.size(1)])


class TransformerNMT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.d_model = cfg.d_model

        self.src_emb = nn.Embedding(cfg.src_vocab_size, cfg.d_model, padding_idx=PAD_ID)
        self.tgt_emb = nn.Embedding(cfg.tgt_vocab_size, cfg.d_model, padding_idx=PAD_ID)
        self.pos_enc = SinusoidalPositionalEncoding(
            cfg.d_model, max_len=max(1024, cfg.max_len + 64), dropout=cfg.dropout
        )

        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model, nhead=cfg.n_heads, dim_feedforward=cfg.d_ff,
            dropout=cfg.dropout, activation=cfg.activation,
            batch_first=True, norm_first=cfg.norm_first,
        )
        dec_layer = nn.TransformerDecoderLayer(
            d_model=cfg.d_model, nhead=cfg.n_heads, dim_feedforward=cfg.d_ff,
            dropout=cfg.dropout, activation=cfg.activation,
            batch_first=True, norm_first=cfg.norm_first,
        )
        # Với pre-norm, cần một LayerNorm cuối cùng ở đầu ra mỗi stack —
        # nếu thiếu, output không được chuẩn hoá và loss ban đầu rất lớn.
        self.encoder = nn.TransformerEncoder(
            enc_layer, num_layers=cfg.n_enc_layers,
            norm=nn.LayerNorm(cfg.d_model) if cfg.norm_first else None,
        )
        self.decoder = nn.TransformerDecoder(
            dec_layer, num_layers=cfg.n_dec_layers,
            norm=nn.LayerNorm(cfg.d_model) if cfg.norm_first else None,
        )

        self.out_proj = nn.Linear(cfg.d_model, cfg.tgt_vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.out_proj.weight = self.tgt_emb.weight

        self._init_parameters()

    def _init_parameters(self) -> None:
        for name, p in self.named_parameters():
            if p.dim() > 1 and "emb" not in name:
                nn.init.xavier_uniform_(p)
        nn.init.normal_(self.src_emb.weight, mean=0.0, std=self.d_model ** -0.5)
        nn.init.normal_(self.tgt_emb.weight, mean=0.0, std=self.d_model ** -0.5)
        with torch.no_grad():
            self.src_emb.weight[PAD_ID].zero_()
            self.tgt_emb.weight[PAD_ID].zero_()

    # ------------------------------------------------------------------
    def _embed_src(self, src: torch.Tensor) -> torch.Tensor:
        return self.pos_enc(self.src_emb(src) * math.sqrt(self.d_model))

    def _embed_tgt(self, tgt: torch.Tensor, offset: int = 0) -> torch.Tensor:
        return self.pos_enc(self.tgt_emb(tgt) * math.sqrt(self.d_model), offset=offset)

    @staticmethod
    def _causal_mask(size: int, device) -> torch.Tensor:
        """Mask nhân quả dạng bool (True = che).

        Dùng bool chứ không dùng float(-inf): `nn.Transformer` cảnh báo khi
        `attn_mask` và `key_padding_mask` khác kiểu, và bản float còn dễ sinh
        NaN dưới autocast fp16 (-inf + -inf).
        """
        return torch.ones(size, size, dtype=torch.bool, device=device).triu(diagonal=1)

    def encode(self, src: torch.Tensor):
        src_pad_mask = src.eq(PAD_ID)
        memory = self.encoder(self._embed_src(src), src_key_padding_mask=src_pad_mask)
        return memory, src_pad_mask

    def forward(self, src, src_len, tgt_in):
        """src_len không dùng ở đây (mask lấy từ PAD_ID) nhưng giữ để đồng
        giao diện với model seq2seq — engine.py gọi cả hai như nhau."""
        memory, src_pad_mask = self.encode(src)
        tgt_mask = self._causal_mask(tgt_in.size(1), tgt_in.device)
        hidden = self.decoder(
            self._embed_tgt(tgt_in),
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_in.eq(PAD_ID),
            memory_key_padding_mask=src_pad_mask,
        )
        return self.out_proj(hidden)

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------
    @torch.no_grad()
    def greedy_decode(self, src, src_len=None, max_new_tokens: int = 120):
        """Sinh từng token, luôn lấy argmax.

        Lưu ý về hiệu năng: bản này KHÔNG cache key/value, nên mỗi bước chạy
        lại decoder trên toàn bộ tiền tố -> tổng chi phí O(T^2). Với T<=120 và
        1268 câu test thì hoàn toàn chấp nhận được, và nó giữ code đủ đơn giản
        để đọc. Đây cũng là lý do latency inference của Transformer trong bảng
        benchmark không vượt trội so với LSTM như nhiều người kỳ vọng — điểm
        này đáng nêu trong báo cáo.
        """
        self.eval()
        device = src.device
        bsz = src.size(0)
        memory, src_pad_mask = self.encode(src)

        ys = torch.full((bsz, 1), BOS_ID, dtype=torch.long, device=device)
        finished = torch.zeros(bsz, dtype=torch.bool, device=device)

        for _ in range(max_new_tokens):
            tgt_mask = self._causal_mask(ys.size(1), device)
            hidden = self.decoder(
                self._embed_tgt(ys), memory,
                tgt_mask=tgt_mask, memory_key_padding_mask=src_pad_mask,
            )
            next_tok = self.out_proj(hidden[:, -1]).argmax(-1)
            next_tok = next_tok.masked_fill(finished, PAD_ID)
            ys = torch.cat([ys, next_tok.unsqueeze(1)], dim=1)
            finished |= next_tok.eq(EOS_ID)
            if bool(finished.all()):
                break
        return ys[:, 1:]

    @torch.no_grad()
    def beam_search(
        self, src, src_len=None, beam_size: int = 5,
        max_new_tokens: int = 120, length_penalty: float = 1.0,
    ):
        """Beam search theo batch, có length penalty kiểu GNMT.

        Length penalty cần thiết vì log-prob luôn giảm theo độ dài -> beam
        thuần sẽ thiên vị câu ngắn, kéo brevity penalty của BLEU xuống. Công
        thức ((5+|Y|)/6)^alpha là của Wu et al. 2016 (GNMT).
        """
        self.eval()
        device = src.device
        bsz = src.size(0)
        k = beam_size
        vocab = self.out_proj.out_features

        memory, src_pad_mask = self.encode(src)
        # [B, S, D] -> [B*k, S, D]
        memory = memory.repeat_interleave(k, dim=0)
        src_pad_mask = src_pad_mask.repeat_interleave(k, dim=0)

        ys = torch.full((bsz * k, 1), BOS_ID, dtype=torch.long, device=device)
        # chỉ beam 0 của mỗi câu được sống ở bước đầu, tránh k bản sao giống hệt
        scores = torch.full((bsz, k), float("-inf"), device=device)
        scores[:, 0] = 0.0
        scores = scores.view(-1)

        finished_seqs: list[list[tuple[float, torch.Tensor]]] = [[] for _ in range(bsz)]
        alive = torch.ones(bsz * k, dtype=torch.bool, device=device)

        for step in range(max_new_tokens):
            tgt_mask = self._causal_mask(ys.size(1), device)
            hidden = self.decoder(
                self._embed_tgt(ys), memory,
                tgt_mask=tgt_mask, memory_key_padding_mask=src_pad_mask,
            )
            logp = torch.log_softmax(self.out_proj(hidden[:, -1]).float(), dim=-1)
            logp = logp.masked_fill(~alive.unsqueeze(1), float("-inf"))

            cand = (scores.unsqueeze(1) + logp).view(bsz, k * vocab)
            top_scores, top_idx = cand.topk(k, dim=-1)

            beam_idx = top_idx // vocab           # beam nào sinh ra token này
            tok_idx = top_idx % vocab
            flat_beam = (torch.arange(bsz, device=device).unsqueeze(1) * k + beam_idx).view(-1)

            ys = torch.cat([ys[flat_beam], tok_idx.view(-1, 1)], dim=1)
            scores = top_scores.view(-1)
            alive = alive[flat_beam]

            eos_hit = tok_idx.view(-1).eq(EOS_ID) & alive
            if bool(eos_hit.any()):
                for pos in eos_hit.nonzero(as_tuple=False).view(-1).tolist():
                    b = pos // k
                    lp = ((5.0 + ys.size(1)) / 6.0) ** length_penalty
                    finished_seqs[b].append((float(scores[pos]) / lp, ys[pos, 1:].clone()))
                alive = alive & ~eos_hit
                scores = scores.masked_fill(eos_hit, float("-inf"))

            # dừng khi mọi câu đã đủ k giả thuyết hoàn chỉnh
            if all(len(f) >= k for f in finished_seqs):
                break
            if not bool(alive.any()):
                break

        # gom kết quả: ưu tiên câu đã kết thúc bằng </s>, nếu không có thì lấy
        # beam tốt nhất còn sống (câu bị cắt vì chạm max_new_tokens)
        max_t = 1
        picked: list[torch.Tensor] = []
        scores_2d = scores.view(bsz, k)
        ys_3d = ys.view(bsz, k, -1)
        for b in range(bsz):
            if finished_seqs[b]:
                best = max(finished_seqs[b], key=lambda x: x[0])[1]
            else:
                best = ys_3d[b, int(scores_2d[b].argmax())][1:]
            picked.append(best)
            max_t = max(max_t, best.numel())

        out = torch.full((bsz, max_t), PAD_ID, dtype=torch.long, device=device)
        for b, seq in enumerate(picked):
            out[b, : seq.numel()] = seq
        return out


def build_model(cfg) -> TransformerNMT:
    return TransformerNMT(cfg)
