"""Case 2 — Seq2Seq LSTM + Attention (Luong et al., EMNLP 2015).

Đây là kiến trúc đứng sau repo `tensorflow/nmt`. Repo gốc viết bằng
TensorFlow 1.x (`tf.contrib.seq2seq`, `tf.Session`) và đã archive — `tf.contrib`
bị gỡ khỏi TF2 và TensorFlow Addons (nơi `tfa.seq2seq` từng sống) đã hết vòng
đời tháng 5/2024, nên code gốc không còn chạy được trên môi trường Colab hiện
tại. Module này cài đặt lại đúng kiến trúc đó bằng PyTorch.

Các thành phần bám sát Luong et al. 2015:

* **Global attention, score kiểu `general`**: score(h_t, h_s) = h_t^T W_a h_s.
  Paper thử ba biến thể (dot / general / concat); `general` là biến thể tốt
  nhất trong thí nghiệm của họ và cũng là mặc định của tensorflow/nmt.

* **Input feeding**: vector attentional h~_t được nối vào input của bước kế
  tiếp. Đây là đóng góp riêng của Luong et al. (khác Bahdanau et al. 2015),
  giúp decoder "nhớ" mình đã căn chỉnh (align) vào đâu ở bước trước, tránh
  dịch lặp hoặc bỏ sót. Đổi lại, nó ép decoder chạy TUẦN TỰ theo timestep —
  không thể song song hoá teacher forcing như Transformer.

  Về lý thuyết đây là bất lợi tốc độ lớn, nhưng đo thực tế trên T4 thì hai
  case gần bằng nhau (152 vs 144 giây/epoch). Model này chỉ có 20.0M tham số
  so với 39.7M của Transformer, tức nửa khối lượng tính toán mỗi token, vừa
  đủ bù phần thiệt do chạy tuần tự; thêm nữa attention của Transformer tốn
  O(T²). Đừng suy từ kiến trúc ra tốc độ mà không đo.

* **Encoder hai chiều**: cho mỗi vị trí nguồn một biểu diễn thấy được cả ngữ
  cảnh trái lẫn phải.

Giao diện giống hệt model Transformer để `common/engine.py` dùng chung.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from common.data import BOS_ID, EOS_ID, PAD_ID


class LuongAttention(nn.Module):
    """Global attention của Luong et al. 2015."""

    def __init__(self, hidden_dim: int, score: str = "general", scale: bool = True):
        super().__init__()
        self.score_type = score
        if score == "general":
            self.W_a = nn.Linear(hidden_dim, hidden_dim, bias=False)
        elif score == "concat":
            self.W_a = nn.Linear(2 * hidden_dim, hidden_dim, bias=False)
            self.v_a = nn.Linear(hidden_dim, 1, bias=False)
        elif score != "dot":
            raise ValueError(f"score phải là dot/general/concat, nhận '{score}'")
        # Tham số scale học được, tương ứng scale=True của tf LuongAttention:
        # giữ độ lớn logit attention ở mức hợp lý khi hidden_dim lớn.
        self.scale = nn.Parameter(torch.tensor(1.0)) if scale else None

    def precompute(self, enc_out: torch.Tensor) -> torch.Tensor:
        """Áp W_a lên encoder states MỘT LẦN thay vì mỗi timestep.

        Với input feeding, decoder chạy ~25 bước cho mỗi câu; nếu tính lại
        W_a @ enc_out ở từng bước thì lãng phí đúng 25 lần.
        """
        if self.score_type == "general":
            return self.W_a(enc_out)
        return enc_out

    def forward(self, dec_h, enc_keys, enc_out, src_mask):
        """dec_h [B, H] -> (context [B, H], attn_weights [B, S])."""
        if self.score_type == "concat":
            S = enc_out.size(1)
            cat = torch.cat([dec_h.unsqueeze(1).expand(-1, S, -1), enc_out], dim=-1)
            scores = self.v_a(torch.tanh(self.W_a(cat))).squeeze(-1)
        else:  # dot | general (enc_keys đã được nhân W_a nếu general)
            scores = torch.bmm(enc_keys, dec_h.unsqueeze(2)).squeeze(2)

        if self.scale is not None:
            scores = scores * self.scale

        scores = scores.masked_fill(src_mask, float("-inf"))
        attn = torch.softmax(scores.float(), dim=-1).to(enc_out.dtype)
        context = torch.bmm(attn.unsqueeze(1), enc_out).squeeze(1)
        return context, attn


class Seq2SeqAttentionNMT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        H, E = cfg.hidden_dim, cfg.emb_dim

        self.src_emb = nn.Embedding(cfg.src_vocab_size, E, padding_idx=PAD_ID)
        self.tgt_emb = nn.Embedding(cfg.tgt_vocab_size, E, padding_idx=PAD_ID)
        self.emb_dropout = nn.Dropout(cfg.dropout)

        self.encoder = nn.LSTM(
            E, H, num_layers=cfg.n_enc_layers, batch_first=True,
            bidirectional=cfg.bidirectional,
            dropout=cfg.dropout if cfg.n_enc_layers > 1 else 0.0,
        )
        enc_out_dim = H * (2 if cfg.bidirectional else 1)
        # Ép output encoder về đúng H để attention và decoder cùng chiều
        self.enc_proj = nn.Linear(enc_out_dim, H, bias=False) if enc_out_dim != H else nn.Identity()
        # Khởi tạo state của decoder từ state cuối của encoder
        self.bridge_h = nn.Linear(enc_out_dim, H)
        self.bridge_c = nn.Linear(enc_out_dim, H)

        # input feeding: input mỗi bước = [emb(y_{t-1}) ; h~_{t-1}]
        dec_in = E + (H if cfg.input_feeding else 0)
        self.decoder = nn.LSTM(
            dec_in, H, num_layers=cfg.n_dec_layers, batch_first=True,
            dropout=cfg.dropout if cfg.n_dec_layers > 1 else 0.0,
        )
        self.attention = LuongAttention(H, cfg.attention, cfg.scale_attention)
        # h~_t = tanh(W_c [c_t ; h_t])
        self.attn_combine = nn.Linear(2 * H, H, bias=False)
        self.out_dropout = nn.Dropout(cfg.dropout)

        self.out_proj = nn.Linear(H, cfg.tgt_vocab_size, bias=False)
        if cfg.tie_embeddings and H == E:
            self.out_proj.weight = self.tgt_emb.weight

        self._init_parameters()

    def _init_parameters(self) -> None:
        """Khởi tạo tham số theo từng loại lớp.

        Bản đầu tiên của hàm này dùng `uniform(-0.1, 0.1)` cho MỌI ma trận —
        bắt chước `init_weight=0.1` của tensorflow/nmt. Đó là sai lầm khi port
        sang PyTorch: TF khởi tạo như vậy trên đồ thị có scale khác, còn ở đây
        nó làm activation teo dần qua chuỗi enc_proj -> attention -> tanh ->
        out_proj, khiến logit lúc init có std chỉ ~0.005. Softmax gần như đều
        tuyệt đối trên 8000 lớp, gradient cực phẳng, và model mất hàng trăm
        epoch chỉ để thoát khỏi vùng đó (đo được: chậm hơn Transformer ~10 lần
        trên cùng phép thử overfit).

        Cách đúng là init theo fan-in của từng lớp:
          * Linear / weight_ih : Xavier uniform — giữ phương sai qua các lớp.
          * weight_hh          : orthogonal — chuẩn cho ma trận hồi tiếp, giúp
                                 gradient không nổ/tắt khi lan qua nhiều bước.
          * bias cổng quên     : đặt 1.0 (Jozefowicz et al. 2015) để LSTM mặc
                                 định GIỮ trạng thái ở đầu quá trình học.
          * embedding          : N(0, 1/sqrt(d)) — cùng scale với Transformer,
                                 nên hai model khởi đầu ở cùng vạch xuất phát.
        """
        for name, p in self.named_parameters():
            if "weight_hh" in name:
                for i in range(0, p.size(0), p.size(1)):
                    nn.init.orthogonal_(p[i : i + p.size(1)])
            elif "weight_ih" in name or (p.dim() > 1 and "emb" not in name):
                nn.init.xavier_uniform_(p)
            elif "bias" in name:
                nn.init.zeros_(p)
                is_lstm_bias = ("encoder." in name or "decoder." in name) and "bias_" in name
                if is_lstm_bias:
                    # nn.LSTM xếp bias theo thứ tự [input, forget, cell, output]
                    hidden = p.numel() // 4
                    p.data[hidden : 2 * hidden].fill_(1.0)

        std = self.cfg.emb_dim ** -0.5
        nn.init.normal_(self.src_emb.weight, mean=0.0, std=std)
        nn.init.normal_(self.tgt_emb.weight, mean=0.0, std=std)
        with torch.no_grad():
            self.src_emb.weight[PAD_ID].zero_()
            self.tgt_emb.weight[PAD_ID].zero_()

    # ------------------------------------------------------------------
    def encode(self, src, src_len):
        emb = self.emb_dropout(self.src_emb(src))
        packed = pack_padded_sequence(
            emb, src_len.cpu(), batch_first=True, enforce_sorted=False
        )
        # cuDNN không chạy LSTM dưới autocast fp16 với packed sequence một cách
        # ổn định trên mọi phiên bản; ép fp32 cho riêng phần RNN là an toàn và
        # gần như không ảnh hưởng tốc độ (RNN bị chặn bởi tính tuần tự, không
        # phải bởi throughput số học).
        # Toàn bộ phần RNN + khởi tạo state của decoder chạy trong fp32.
        # `state` phải cùng kiểu với input của nn.LSTM; nếu để bridge_h/bridge_c
        # chạy dưới autocast thì state ra fp16 còn input bị ép fp32 -> LSTM báo
        # lỗi. Lỗi này CHỈ xuất hiện khi bật AMP, nên một lần chạy thử trên CPU
        # sẽ không bắt được.
        with torch.autocast(device_type=src.device.type, enabled=False):
            packed_out, (h_n, c_n) = self.encoder(packed)
            enc_out_fp32, _ = pad_packed_sequence(packed_out, batch_first=True)

            # gộp hai chiều của lớp cuối để khởi tạo decoder
            if self.cfg.bidirectional:
                h_last = torch.cat([h_n[-2], h_n[-1]], dim=-1)
                c_last = torch.cat([c_n[-2], c_n[-1]], dim=-1)
            else:
                h_last, c_last = h_n[-1], c_n[-1]

            h0 = torch.tanh(self.bridge_h(h_last)).unsqueeze(0)
            c0 = torch.tanh(self.bridge_c(c_last)).unsqueeze(0)
            h0 = h0.repeat(self.cfg.n_dec_layers, 1, 1).contiguous()
            c0 = c0.repeat(self.cfg.n_dec_layers, 1, 1).contiguous()

        # enc_out thì để autocast lo — attention là bmm/linear, chạy fp16 vừa
        # nhanh vừa an toàn.
        enc_out = self.enc_proj(enc_out_fp32)

        src_mask = src.eq(PAD_ID)
        enc_keys = self.attention.precompute(enc_out)
        return enc_out, enc_keys, src_mask, (h0, c0)

    def _decode_step(self, y_emb, feed, state, enc_out, enc_keys, src_mask):
        """Một bước decoder: trả về (h~_t, state mới)."""
        rnn_in = torch.cat([y_emb, feed], dim=-1) if self.cfg.input_feeding else y_emb
        # Ép cả input LẪN state về fp32: nn.LSTM đòi hai thứ này cùng kiểu, mà
        # `feed` đến từ attention nên có thể là fp16 khi bật AMP.
        h, c = state
        with torch.autocast(device_type=y_emb.device.type, enabled=False):
            out, state = self.decoder(
                rnn_in.unsqueeze(1).float(),
                (h.float().contiguous(), c.float().contiguous()),
            )
        dec_h = out.squeeze(1).to(enc_out.dtype)
        context, _ = self.attention(dec_h, enc_keys, enc_out, src_mask)
        attn_h = torch.tanh(self.attn_combine(torch.cat([context, dec_h], dim=-1)))
        return attn_h, state

    def forward(self, src, src_len, tgt_in):
        enc_out, enc_keys, src_mask, state = self.encode(src, src_len)
        B, T = tgt_in.shape
        H = self.cfg.hidden_dim

        emb = self.emb_dropout(self.tgt_emb(tgt_in))
        feed = torch.zeros(B, H, device=src.device, dtype=enc_out.dtype)

        outs = []
        for t in range(T):
            attn_h, state = self._decode_step(
                emb[:, t], feed, state, enc_out, enc_keys, src_mask
            )
            outs.append(attn_h)
            feed = attn_h            # input feeding
        hidden = self.out_dropout(torch.stack(outs, dim=1))
        return self.out_proj(hidden)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def greedy_decode(self, src, src_len, max_new_tokens: int = 120):
        self.eval()
        enc_out, enc_keys, src_mask, state = self.encode(src, src_len)
        B, H = src.size(0), self.cfg.hidden_dim
        device = src.device

        y = torch.full((B,), BOS_ID, dtype=torch.long, device=device)
        feed = torch.zeros(B, H, device=device, dtype=enc_out.dtype)
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        outputs = []

        for _ in range(max_new_tokens):
            attn_h, state = self._decode_step(
                self.tgt_emb(y), feed, state, enc_out, enc_keys, src_mask
            )
            y = self.out_proj(attn_h).argmax(-1)
            y = y.masked_fill(finished, PAD_ID)
            outputs.append(y)
            finished |= y.eq(EOS_ID)
            feed = attn_h
            if bool(finished.all()):
                break
        return torch.stack(outputs, dim=1)

    @torch.no_grad()
    def beam_search(
        self, src, src_len, beam_size: int = 10,
        max_new_tokens: int = 120, length_penalty: float = 1.0,
    ):
        """Beam search theo batch với length penalty GNMT.

        Khác với Transformer (state chỉ là chuỗi token đã sinh), ở đây phải
        mang theo và hoán vị cả LSTM state (h, c) lẫn vector input-feeding mỗi
        khi beam được chọn lại. Quên hoán vị một trong ba thứ đó là lỗi kinh
        điển và biểu hiện là BLEU tụt nhẹ chứ không crash — rất khó phát hiện.
        """
        self.eval()
        device = src.device
        B, k = src.size(0), beam_size
        H = self.cfg.hidden_dim
        vocab = self.out_proj.out_features

        enc_out, enc_keys, src_mask, (h0, c0) = self.encode(src, src_len)
        enc_out = enc_out.repeat_interleave(k, dim=0)
        enc_keys = enc_keys.repeat_interleave(k, dim=0)
        src_mask = src_mask.repeat_interleave(k, dim=0)
        state = (h0.repeat_interleave(k, dim=1), c0.repeat_interleave(k, dim=1))

        y = torch.full((B * k,), BOS_ID, dtype=torch.long, device=device)
        feed = torch.zeros(B * k, H, device=device, dtype=enc_out.dtype)
        seqs = torch.zeros(B * k, 0, dtype=torch.long, device=device)

        scores = torch.full((B, k), float("-inf"), device=device)
        scores[:, 0] = 0.0
        scores = scores.view(-1)
        alive = torch.ones(B * k, dtype=torch.bool, device=device)
        finished_seqs: list[list[tuple[float, torch.Tensor]]] = [[] for _ in range(B)]

        for _ in range(max_new_tokens):
            attn_h, state = self._decode_step(
                self.tgt_emb(y), feed, state, enc_out, enc_keys, src_mask
            )
            logp = torch.log_softmax(self.out_proj(attn_h).float(), dim=-1)
            logp = logp.masked_fill(~alive.unsqueeze(1), float("-inf"))

            cand = (scores.unsqueeze(1) + logp).view(B, k * vocab)
            top_scores, top_idx = cand.topk(k, dim=-1)
            beam_idx = top_idx // vocab
            tok_idx = top_idx % vocab
            flat = (torch.arange(B, device=device).unsqueeze(1) * k + beam_idx).view(-1)

            seqs = torch.cat([seqs[flat], tok_idx.view(-1, 1)], dim=1)
            scores = top_scores.view(-1)
            alive = alive[flat]
            feed = attn_h[flat]
            state = (state[0][:, flat].contiguous(), state[1][:, flat].contiguous())
            y = tok_idx.view(-1)

            eos_hit = y.eq(EOS_ID) & alive
            if bool(eos_hit.any()):
                for pos in eos_hit.nonzero(as_tuple=False).view(-1).tolist():
                    lp = ((5.0 + seqs.size(1)) / 6.0) ** length_penalty
                    finished_seqs[pos // k].append(
                        (float(scores[pos]) / lp, seqs[pos].clone())
                    )
                alive = alive & ~eos_hit
                scores = scores.masked_fill(eos_hit, float("-inf"))

            if all(len(f) >= k for f in finished_seqs) or not bool(alive.any()):
                break

        max_t, picked = 1, []
        scores_2d, seqs_3d = scores.view(B, k), seqs.view(B, k, -1)
        for b in range(B):
            if finished_seqs[b]:
                best = max(finished_seqs[b], key=lambda x: x[0])[1]
            else:
                best = seqs_3d[b, int(scores_2d[b].argmax())]
            picked.append(best)
            max_t = max(max_t, best.numel())

        out = torch.full((B, max_t), PAD_ID, dtype=torch.long, device=device)
        for b, seq in enumerate(picked):
            out[b, : seq.numel()] = seq
        return out


def build_model(cfg) -> Seq2SeqAttentionNMT:
    return Seq2SeqAttentionNMT(cfg)
