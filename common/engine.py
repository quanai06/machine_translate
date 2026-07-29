"""Vòng train / eval / decode dùng chung cho cả hai kiến trúc.

Hai model chỉ khác nhau ở `model.py`. Toàn bộ phần còn lại — loss, optimizer,
mixed precision, gradient accumulation, checkpoint, early stopping, decode,
chấm BLEU — chạy qua đúng một đoạn code ở đây. Nhờ vậy khi so sánh, chênh lệch
quan sát được là do KIẾN TRÚC chứ không phải do một bên được train khéo hơn.

Giao diện mà mọi model phải cài đặt (xem docstring của từng model):
    forward(src, src_len, tgt_in)      -> logits [B, T, V]
    greedy_decode(src, src_len, ...)   -> ids    [B, T]
    beam_search(src, src_len, ...)     -> ids    [B, T]
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from .bench import BenchmarkRecorder, count_parameters, measure_latency
from .data import BOS_ID, EOS_ID, PAD_ID, build_dataloader, collate_batch
from .metrics import corpus_bleu


# --------------------------------------------------------------------------
# Loss
# --------------------------------------------------------------------------
def label_smoothed_nll_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    epsilon: float = 0.1,
    ignore_index: int = PAD_ID,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cross-entropy có label smoothing; trả về (loss_để_backward, nll_để_báo_cáo).

    Label smoothing làm model bớt "tự tin thái quá" vào một token duy nhất. Với
    dịch máy điều này quan trọng vì một câu nguồn có nhiều bản dịch đúng — ép
    xác suất về 1.0 cho đúng một chuỗi là ép sai. Vaswani et al. 2017 dùng
    eps=0.1: nó làm perplexity XẤU đi nhưng BLEU TỐT lên, nên ta báo cáo NLL
    thật (không smoothing) để đọc perplexity cho đúng.
    """
    lprobs = F.log_softmax(logits.float(), dim=-1)
    lprobs = lprobs.view(-1, lprobs.size(-1))
    target = target.reshape(-1, 1)

    pad_mask = target.eq(ignore_index)
    nll = -lprobs.gather(dim=-1, index=target.clamp(min=0))
    smooth = -lprobs.sum(dim=-1, keepdim=True)

    nll = nll.masked_fill(pad_mask, 0.0)
    smooth = smooth.masked_fill(pad_mask, 0.0)

    n_tokens = (~pad_mask).sum().clamp(min=1)
    nll_sum, smooth_sum = nll.sum(), smooth.sum()
    eps_i = epsilon / (lprobs.size(-1) - 1)
    loss = (1.0 - epsilon) * nll_sum + eps_i * smooth_sum
    return loss / n_tokens, nll_sum / n_tokens


# --------------------------------------------------------------------------
# Learning-rate schedule
# --------------------------------------------------------------------------
class InverseSqrtSchedule:
    """Warmup tuyến tính rồi giảm theo 1/sqrt(step) — lịch chuẩn của Transformer.

    Transformer BẮT BUỘC phải warmup nếu dùng post-norm: những bước đầu gradient
    qua LayerNorm rất lớn và làm mô hình phân kỳ. Bản cài đặt ở đây dùng
    pre-norm (norm_first=True) nên bớt nhạy cảm hơn — đúng như "Transformers
    without Tears" chỉ ra — nhưng vẫn giữ warmup vì nó ổn định và rẻ.
    """

    def __init__(self, optimizer, d_model: int, warmup_steps: int = 4000, scale: float = 1.0):
        self.optimizer = optimizer
        self.warmup = max(1, warmup_steps)
        self.factor = scale * (d_model ** -0.5)
        self.step_num = 0

    def step(self) -> float:
        self.step_num += 1
        lr = self.factor * min(
            self.step_num ** -0.5, self.step_num * self.warmup ** -1.5
        )
        for g in self.optimizer.param_groups:
            g["lr"] = lr
        return lr

    def get_last_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]


# --------------------------------------------------------------------------
# Train / eval
# --------------------------------------------------------------------------
def train_one_epoch(
    model,
    loader,
    optimizer,
    scheduler,
    scaler,
    device,
    *,
    label_smoothing: float = 0.1,
    clip_norm: float = 1.0,
    accum_steps: int = 1,
    amp_dtype: torch.dtype | None = torch.float16,
    desc: str = "train",
) -> dict[str, float]:
    model.train()
    total_nll, total_tokens, n_steps = 0.0, 0, 0
    t0 = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(loader, desc=desc, leave=False)
    for step, batch in enumerate(pbar):
        src = batch["src"].to(device, non_blocking=True)
        src_len = batch["src_len"].to(device, non_blocking=True)
        tgt_in = batch["tgt_in"].to(device, non_blocking=True)
        tgt_out = batch["tgt_out"].to(device, non_blocking=True)

        use_amp = amp_dtype is not None and device.type == "cuda"
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            logits = model(src, src_len, tgt_in)
            loss, nll = label_smoothed_nll_loss(logits, tgt_out, label_smoothing)

        scaler.scale(loss / accum_steps).backward()

        if (step + 1) % accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()
            n_steps += 1

        n_tok = int((tgt_out != PAD_ID).sum())
        nll_val = nll.detach().item()
        total_nll += nll_val * n_tok
        total_tokens += n_tok
        if step % 50 == 0:
            lr = optimizer.param_groups[0]["lr"]
            pbar.set_postfix(nll=f"{nll_val:.3f}", lr=f"{lr:.2e}")

    dt = time.perf_counter() - t0
    mean_nll = total_nll / max(1, total_tokens)
    return {
        "train_nll": round(mean_nll, 4),
        "train_ppl": round(math.exp(min(mean_nll, 20)), 2),
        "epoch_time_sec": round(dt, 1),
        "tokens_per_sec": round(total_tokens / dt, 0),
        "optimizer_steps": n_steps,
        "lr": optimizer.param_groups[0]["lr"],
    }


@torch.no_grad()
def evaluate_loss(model, loader, device, amp_dtype=torch.float16) -> dict[str, float]:
    model.eval()
    total_nll, total_tokens = 0.0, 0
    for batch in tqdm(loader, desc="eval-loss", leave=False):
        src = batch["src"].to(device)
        src_len = batch["src_len"].to(device)
        tgt_in = batch["tgt_in"].to(device)
        tgt_out = batch["tgt_out"].to(device)
        use_amp = amp_dtype is not None and device.type == "cuda"
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            logits = model(src, src_len, tgt_in)
        _, nll = label_smoothed_nll_loss(logits, tgt_out, epsilon=0.0)
        n_tok = int((tgt_out != PAD_ID).sum())
        total_nll += nll.detach().item() * n_tok
        total_tokens += n_tok
    mean_nll = total_nll / max(1, total_tokens)
    return {"nll": round(mean_nll, 4), "ppl": round(math.exp(min(mean_nll, 20)), 2)}


# --------------------------------------------------------------------------
# Decode toàn bộ một split
# --------------------------------------------------------------------------
@torch.no_grad()
def translate_dataset(
    model,
    dataset,
    tok_tgt,
    device,
    *,
    beam_size: int = 1,
    max_new_tokens: int = 120,
    length_penalty: float = 1.0,
    batch_sentences: int = 32,
) -> list[str]:
    """Dịch toàn bộ dataset, trả về list câu theo ĐÚNG thứ tự gốc.

    Ta sắp xếp theo độ dài để giảm padding khi decode (nhanh hơn đáng kể), rồi
    khôi phục thứ tự bằng `index` do collate trả về — nếu quên bước khôi phục
    này thì BLEU sẽ ra gần 0 mà rất khó phát hiện nguyên nhân.
    """
    model.eval()
    order = sorted(range(len(dataset)), key=lambda i: dataset.lengths[i])
    outputs: dict[int, str] = {}

    pbar = tqdm(
        range(0, len(order), batch_sentences),
        desc=f"decode(beam={beam_size})",
        leave=False,
    )
    for start in pbar:
        chunk = order[start : start + batch_sentences]
        batch = collate_batch([dataset[i] for i in chunk])
        src = batch["src"].to(device)
        src_len = batch["src_len"].to(device)

        if beam_size > 1:
            hyp = model.beam_search(
                src,
                src_len,
                beam_size=beam_size,
                max_new_tokens=max_new_tokens,
                length_penalty=length_penalty,
            )
        else:
            hyp = model.greedy_decode(src, src_len, max_new_tokens=max_new_tokens)

        for row, idx in zip(hyp.tolist(), batch["index"].tolist()):
            ids = []
            for t in row:
                if t == EOS_ID:
                    break
                if t not in (PAD_ID, BOS_ID):
                    ids.append(t)
            outputs[idx] = tok_tgt.decode(ids)

    return [outputs[i] for i in range(len(dataset))]


def score_split(
    model, dataset, tok_tgt, device, *, beam_size=1, max_new_tokens=120, length_penalty=1.0
) -> tuple[dict, list[str]]:
    hyps = translate_dataset(
        model,
        dataset,
        tok_tgt,
        device,
        beam_size=beam_size,
        max_new_tokens=max_new_tokens,
        length_penalty=length_penalty,
    )
    scores = corpus_bleu(hyps, dataset.raw_tgt)
    return scores, hyps


# --------------------------------------------------------------------------
# Toàn bộ pipeline train
# --------------------------------------------------------------------------
def run_training(
    model,
    cfg,
    train_ds,
    dev_ds,
    tok_tgt,
    device,
    *,
    optimizer,
    scheduler,
    output_dir: Path,
    run_name: str,
    on_epoch_start=None,
) -> BenchmarkRecorder:
    """Train tới `cfg.epochs`, chọn checkpoint tốt nhất theo BLEU trên dev.

    Chọn theo BLEU dev chứ không theo loss dev là có chủ ý: loss và BLEU không
    đồng biến hoàn toàn (label smoothing, exposure bias), và điều ta quan tâm
    cuối cùng là BLEU. Đây cũng là cách tensorflow/nmt và fairseq làm.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader = build_dataloader(
        train_ds, max_tokens=cfg.max_tokens, shuffle=True,
        num_workers=cfg.num_workers, seed=cfg.seed,
    )
    dev_loader = build_dataloader(
        dev_ds, max_tokens=cfg.max_tokens, shuffle=False, num_workers=0
    )

    rec = BenchmarkRecorder(name=run_name, output_dir=output_dir, config=vars(cfg).copy())
    rec.config.pop("data_dir", None)
    rec.summary.update(count_parameters(model))
    rec.summary["train_sentences"] = len(train_ds)
    rec.summary["train_dropped_too_long"] = train_ds.n_dropped
    rec.start()

    amp_dtype = torch.float16 if cfg.amp and device.type == "cuda" else None
    scaler = torch.amp.GradScaler(device.type, enabled=amp_dtype is not None)

    best_bleu, best_epoch, patience = -1.0, -1, 0
    ckpt_path = output_dir / "best.pt"
    last_path = output_dir / "last.pt"
    start_epoch = 1

    # Colab free hay ngắt phiên sau vài giờ. `last.pt` được ghi sau MỖI epoch
    # (kèm optimizer và scheduler) để có thể train tiếp thay vì làm lại từ đầu.
    if getattr(cfg, "resume", False) and last_path.exists():
        state = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        if scheduler is not None and state.get("sched_step") is not None:
            scheduler.step_num = state["sched_step"]
        start_epoch = state["epoch"] + 1
        best_bleu, best_epoch, patience = state["best_bleu"], state["best_epoch"], state["patience"]
        rec.epochs = state.get("history", [])
        print(f"[{run_name}] train tiếp từ epoch {start_epoch} "
              f"(BLEU dev tốt nhất đang là {best_bleu})")
        if start_epoch > cfg.epochs:
            print(f"[{run_name}] đã đủ {cfg.epochs} epoch, không còn gì để train.")
            rec.finish(best_dev_bleu=best_bleu, best_epoch=best_epoch,
                       checkpoint=str(ckpt_path))
            return rec

    for epoch in range(start_epoch, cfg.epochs + 1):
        if hasattr(train_loader.batch_sampler, "set_epoch"):
            train_loader.batch_sampler.set_epoch(epoch)
        # Cho phép model dùng lịch learning-rate theo EPOCH (seq2seq) song song
        # với lịch theo STEP (Transformer) mà không cần hai vòng train riêng.
        if on_epoch_start is not None:
            on_epoch_start(epoch)

        stats = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler, device,
            label_smoothing=cfg.label_smoothing,
            clip_norm=cfg.clip_norm,
            accum_steps=cfg.accum_steps,
            amp_dtype=amp_dtype,
            desc=f"{run_name} ep{epoch}/{cfg.epochs}",
        )
        dev_loss = evaluate_loss(model, dev_loader, device, amp_dtype)

        t0 = time.perf_counter()
        dev_scores, _ = score_split(
            model, dev_ds, tok_tgt, device,
            beam_size=1, max_new_tokens=cfg.max_new_tokens,
        )
        decode_sec = round(time.perf_counter() - t0, 1)

        row = {
            "epoch": epoch,
            **stats,
            "dev_nll": dev_loss["nll"],
            "dev_ppl": dev_loss["ppl"],
            "dev_bleu_greedy": dev_scores["bleu_tokenized"],
            "dev_decode_sec": decode_sec,
        }
        rec.log_epoch(**row)
        print(
            f"[{run_name}] ep{epoch:>2}  "
            f"train_ppl={stats['train_ppl']:>7.2f}  "
            f"dev_ppl={dev_loss['ppl']:>7.2f}  "
            f"dev_BLEU={dev_scores['bleu_tokenized']:>5.2f}  "
            f"{stats['epoch_time_sec']:>6.1f}s  "
            f"{stats['tokens_per_sec']:>8.0f} tok/s"
        )

        stop_now = False
        if dev_scores["bleu_tokenized"] > best_bleu:
            best_bleu, best_epoch, patience = dev_scores["bleu_tokenized"], epoch, 0
            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "dev_bleu": best_bleu,
                 "config": vars(cfg)},
                ckpt_path,
            )
        else:
            patience += 1
            if cfg.patience and patience >= cfg.patience:
                print(f"[{run_name}] early stop ở epoch {epoch} "
                      f"(BLEU dev không cải thiện {patience} epoch)")
                stop_now = True

        # Ghi state đầy đủ để `--resume` train tiếp được sau khi Colab ngắt.
        torch.save(
            {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
             "sched_step": getattr(scheduler, "step_num", None),
             "epoch": epoch, "best_bleu": best_bleu, "best_epoch": best_epoch,
             "patience": patience, "history": rec.epochs, "config": vars(cfg)},
            last_path,
        )
        if stop_now:
            break

    rec.finish(
        best_dev_bleu=best_bleu,
        best_epoch=best_epoch,
        checkpoint=str(ckpt_path),
    )
    return rec


def load_best(model, output_dir: Path, device) -> dict:
    path = Path(output_dir) / "best.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"Chưa có checkpoint tại {path}.\n"
            f"Hãy train model trước, ví dụ:\n"
            f"    python translate_transformers/train.py\n"
            f"    python translate_seq2seq/train.py"
        )
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    return ckpt
