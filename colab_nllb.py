# ============================================================
# NLLB-200-distilled-600M — English → Vietnamese (1 cell Colab)
# TỰ NHẬN DIỆN định dạng file test (CSV có header / text thường)
# ============================================================
# FILE CẦN UPLOAD:
#   File test của competition (tải từ tab Data trên trang Kaggle competition)
#   ~1200 câu. Có thể là CSV (cột id,English) hoặc text thường.
# ============================================================

# ── Bước 1: Cài đặt ────────────────────────────────────────────────
!pip install -q transformers sentencepiece sacrebleu pandas accelerate

import torch, pandas as pd
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# ── Bước 2: Upload file test ────────────────────────────────────────
from google.colab import files
print("📤 Chọn file test của competition (tải từ Kaggle → Data tab):")
uploaded = files.upload()
test_name = list(uploaded.keys())[0]
print(f"✅ Đã upload: {test_name}")

# ── Bước 3: Đọc file — TỰ NHẬN DIỆN định dạng ─────────────────────
texts, ids = [], None
f = open(test_name, encoding="utf-8")
first_line = f.readline().strip()
f.seek(0)

# Chỉ coi là CSV khi: đuôi .csv HOẶC dòng đầu có header quen thuộc
# (KHÔNG dựa vào dấu phẩy — câu tiếng Anh thường có phẩy bên trong!)
fname_lower = test_name.lower()
first_lower = first_line.lower()
looks_like_header = any(k in first_lower for k in
                        ["english", "sentence", "text", "src", "source", "id,", "input", "vietnamese"])
is_csv = fname_lower.endswith(".csv") or (looks_like_header and ("," in first_line or "\t" in first_line))

if is_csv:
    # Có header → đọc bằng pandas, tự tìm cột tiếng Anh
    df = pd.read_csv(test_name, sep=None, engine="python")
    print(f"📋 Header CSV: {list(df.columns)}")
    # Tìm cột tiếng Anh (theo tên phổ biến)
    en_col = None
    for c in df.columns:
        if str(c).strip().lower() in ("english", "en", "src", "source", "sentence", "text", "english_text", "input"):
            en_col = c
            break
    if en_col is None:
        # Không tìm thấy tên quen thuộc → thử cột cuối cùng (thường là văn bản)
        en_col = df.columns[-1]
    print(f"🔍 Cột tiếng Anh: '{en_col}'")
    texts = df[en_col].astype(str).tolist()
    # Tìm cột id nếu có
    for c in df.columns:
        if str(c).strip().lower() in ("id", "index", "no", "stt"):
            ids = df[c].tolist()
            break
    if ids is not None:
        print(f"🔢 Cột id: '{c}' ({len(ids)} dòng)")
else:
    # Text thường — mỗi dòng 1 câu, GIỮ NGUYÊN số dòng
    lines = [l.strip() for l in open(test_name, encoding="utf-8").read().splitlines()]
    texts = lines

texts = [("" if (t.lower() == "nan") else t) for t in texts]
print(f"📄 {len(texts):,} dòng (giữ nguyên số dòng để khớp submission)")

# ── Bước 4: Load NLLB-200 ──────────────────────────────────────────
MODEL = "facebook/nllb-200-distilled-600M"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔧 Device: {device}")

tok = AutoTokenizer.from_pretrained(MODEL, src_lang="eng_Latn", tgt_lang="vie_Latn")
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL).to(device)
if device == "cuda":
    model.half()
print("✅ Model loaded")

# ── Bước 5: Hàm dịch ───────────────────────────────────────────────
def translate_batch(texts, batch_size=16, num_beams=4, max_len=128):
    results = []
    vie_id = tok.convert_tokens_to_ids("vie_Latn")
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        enc = tok(chunk, return_tensors="pt", padding=True,
                  truncation=True, max_length=max_len).to(device)
        with torch.no_grad():
            out = model.generate(
                **enc,
                forced_bos_token_id=vie_id,
                num_beams=num_beams,
                max_new_tokens=max_len,
            )
        results += tok.batch_decode(out, skip_special_tokens=True)
    return results

# ── Bước 6: Dịch ───────────────────────────────────────────────────
print("⏳ Đang dịch (beam=4)...")
hyps = translate_batch(texts)
print("✅ Dịch xong!")

# ── Bước 7: Lưu results.csv ────────────────────────────────────────
with open("results.csv", "w", encoding="utf-8") as f:
    if ids is not None and len(ids) == len(hyps):
        f.write("id,Vietnamese\n")
        for idx, h in zip(ids, hyps):
            f.write(f'{idx},"{h.replace(chr(34), chr(34)*2)}"\n')
    else:
        f.write("Vietnamese\n")
        for h in hyps:
            f.write(f'"{h.replace(chr(34), chr(34)*2)}"\n')
print(f"💾 results.csv: {len(hyps):,} dòng")

# ── Bước 8 (tùy chọn): BLEU nếu upload kèm tst2013 ────────────────
import os
if os.path.exists("tst2013.en") and os.path.exists("tst2013.vi"):
    import sacrebleu
    src = [l.strip() for l in open("tst2013.en", encoding="utf-8").read().splitlines() if l.strip()]
    refs = [l.strip() for l in open("tst2013.vi", encoding="utf-8").read().splitlines() if l.strip()]
    hyps_t = translate_batch(src)
    if len(hyps_t) == len(refs):
        bleu = sacrebleu.corpus_bleu(hyps_t, [refs])
        print(f"🎯 BLEU trên tst2013: {bleu.score:.2f}")

# ── Bước 9: Tải về máy ─────────────────────────────────────────────
from google.colab import files as _f
print("📥 Tải results.csv về máy...")
_f.download("results.csv")
print("🏁 HOÀN THÀNH! Nộp results.csv lên Kaggle.")
