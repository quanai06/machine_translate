"""Kiểm tra hai model chạy được dưới mixed precision.

    python scripts/check_amp.py

Vì sao cần riêng một script cho việc này: mọi lỗi lệch kiểu (fp16 gặp fp32) đều
**vô hình** khi chạy trên CPU không bật AMP — nhưng làm hỏng ngay lần train đầu
tiên trên GPU. Script này ép chạy dưới `torch.autocast` để lôi những lỗi đó ra
trước, và nó chạy được cả trên máy không có GPU (dùng bfloat16 trên CPU làm
proxy cho fp16 trên CUDA — cùng logic ép kiểu, chỉ khác định dạng số).

Bao phủ: forward, backward qua GradScaler, greedy decode, beam search.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from common.data import TranslationDataset, collate_batch, load_split  # noqa: E402
from common.engine import label_smoothed_nll_loss  # noqa: E402
from common.tokenizer import load_tokenizers  # noqa: E402


def make_models(tok_s, tok_t):
    from translate_seq2seq.config import Seq2SeqConfig
    from translate_seq2seq.model import build_model as build_s2s
    from translate_transformers.config import TransformerConfig
    from translate_transformers.model import build_model as build_tf

    out = []
    for name, cfg, builder in (
        ("transformer", TransformerConfig(), build_tf),
        ("seq2seq", Seq2SeqConfig(), build_s2s),
    ):
        cfg.src_vocab_size, cfg.tgt_vocab_size = tok_s.vocab_size, tok_t.vocab_size
        out.append((name, cfg, builder(cfg)))
    return out


def main() -> None:
    if torch.cuda.is_available():
        device, amp_dtype, label = torch.device("cuda"), torch.float16, "CUDA fp16 (thật)"
    else:
        device, amp_dtype, label = torch.device("cpu"), torch.bfloat16, "CPU bf16 (proxy)"
    print(f"Chế độ: {label}\n")

    tok_s, tok_t = load_tokenizers(ROOT / "data" / "tokenizer")
    src, tgt = load_split(ROOT / "data" / "iwslt15_en_vi", "test")
    ds = TranslationDataset(src[:8], tgt[:8], tok_s, tok_t, 100, False)
    batch = collate_batch([ds[i] for i in range(8)])
    batch = {k: v.to(device) for k, v in batch.items()}

    failures = 0
    for name, cfg, model in make_models(tok_s, tok_t):
        model.to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
        print(f"--- {name} ---")

        # 1. forward + backward qua đúng đường mà engine.train_one_epoch đi
        try:
            with torch.autocast(device_type=device.type, dtype=amp_dtype):
                logits = model(batch["src"], batch["src_len"], batch["tgt_in"])
                loss, nll = label_smoothed_nll_loss(logits, batch["tgt_out"], 0.1)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            assert torch.isfinite(loss), "loss là NaN/Inf"
            assert torch.isfinite(gnorm), "gradient norm là NaN/Inf"
            print(f"  [OK] train step   logits={logits.dtype} loss={float(loss):.3f} "
                  f"|grad|={float(gnorm):.2f}")
        except Exception as e:
            failures += 1
            print(f"  [LỖI] train step  {type(e).__name__}: {e}")

        # 2. decode — engine chạy decode NGOÀI autocast, nhưng vẫn thử cả trong
        #    để chắc chắn model không phụ thuộc vào việc autocast bật hay tắt
        for mode, ctx in (("ngoài autocast", torch.autocast(device.type, enabled=False)),
                          ("trong autocast", torch.autocast(device.type, dtype=amp_dtype))):
            for fn, kw in (("greedy", {}), ("beam5", {"beam_size": 5})):
                try:
                    with torch.no_grad(), ctx:
                        if fn == "greedy":
                            o = model.greedy_decode(batch["src"], batch["src_len"], 15)
                        else:
                            o = model.beam_search(batch["src"], batch["src_len"],
                                                  max_new_tokens=15, **kw)
                    assert o.shape[0] == 8, f"batch size sai: {o.shape}"
                    assert o.dtype == torch.long, f"output phải là long, nhận {o.dtype}"
                    print(f"  [OK] {fn:<6} {mode:<15} shape={tuple(o.shape)}")
                except Exception as e:
                    failures += 1
                    print(f"  [LỖI] {fn:<6} {mode:<15} {type(e).__name__}: {e}")
        print()

    if failures:
        print(f"CÓ {failures} LỖI — chưa nên đẩy lên Colab.")
        sys.exit(1)
    print("Tất cả đều qua. Đường AMP an toàn.")


if __name__ == "__main__":
    main()
