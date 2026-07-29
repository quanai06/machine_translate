"""Tính BLEU sao cho SO SÁNH ĐƯỢC với các con số trong paper.

Đây là chỗ rất dễ báo sai và cũng là chỗ hay bị hỏi khi bảo vệ. Ba lưu ý:

1. Corpus IWSLT15 phát hành ở dạng ĐÃ tokenize theo Moses. Mọi con số kinh
   điển trên bộ này (tensorflow/nmt 26.1; Luong & Manning 2015; Transformers
   without Tears 32.8) đều là *tokenized BLEU*, tính bằng `multi-bleu.perl`
   trên chính văn bản tokenized đó.

2. `sacrebleu` mặc định dùng tokenizer `13a` và sẽ tokenize LẠI văn bản. Nếu
   đưa thẳng văn bản đã tokenize vào sacrebleu mặc định, con số nhận được
   KHÔNG bằng multi-bleu và cũng không bằng BLEU detokenized — nó là con số
   thứ ba, vô nghĩa để so sánh.

3. Cách đúng: `sacrebleu(..., tokenize="none")` trên văn bản đã tokenize thì
   tương đương `multi-bleu.perl`. Đó là con số ta dùng để đối chiếu paper.
   Song song, ta báo thêm BLEU detokenized (`13a`) vì đó mới là chuẩn báo cáo
   hiện đại. Hàm `corpus_bleu` dưới đây trả về CẢ HAI.

Ngoài ra tiếng Việt là ngôn ngữ âm tiết rời: "học sinh" là 2 token khi tách
theo khoảng trắng nhưng là 1 từ. BLEU tính trên âm tiết vì thế thường cao hơn
BLEU tính trên từ đã ghép. Ta báo trên âm tiết — giống toàn bộ literature của
bộ IWSLT15 — nên vẫn nhất quán.
"""

from __future__ import annotations

import re

import sacrebleu

# Thực thể XML mà Moses tokenizer sinh ra trong corpus IWSLT15
_UNESCAPE = {
    "&apos;": "'",
    "&quot;": '"',
    "&amp;": "&",
    "&lt;": "<",
    "&gt;": ">",
    "&#91;": "[",
    "&#93;": "]",
    "&#124;": "|",
}

_LEFT_ATTACH = set(".,!?:;)]}%")
_RIGHT_ATTACH = set("([{$")


def moses_unescape(text: str) -> str:
    """Đổi `&apos;` `&quot;` ... về ký tự gốc."""
    for k, v in _UNESCAPE.items():
        text = text.replace(k, v)
    return text


def detokenize(text: str) -> str:
    """Ghép lại văn bản đã tokenize thành câu người đọc được.

    Đây là detokenizer rút gọn — đủ tốt cho BLEU-13a và cho việc in câu dịch
    mẫu, không nhằm thay thế `detokenizer.perl` của Moses.
    """
    text = moses_unescape(text)
    tokens = text.split()
    out: list[str] = []
    for tok in tokens:
        if out and (tok in _LEFT_ATTACH or (len(tok) > 1 and tok[0] == "'")):
            out[-1] += tok
        elif out and out[-1] and out[-1][-1] in _RIGHT_ATTACH:
            out[-1] += tok
        else:
            out.append(tok)
    joined = " ".join(out)
    return re.sub(r"\s+", " ", joined).strip()


def corpus_bleu(
    hypotheses: list[str],
    references: list[str],
) -> dict[str, float | str]:
    """Trả về cả BLEU tokenized (so paper) lẫn BLEU detokenized (chuẩn hiện đại).

    Cả `hypotheses` và `references` phải ở dạng ĐÃ tokenize kiểu Moses — tức
    đúng dạng model sinh ra và đúng dạng file `tst2013.vi` gốc.
    """
    if len(hypotheses) != len(references):
        raise ValueError(
            f"lệch số câu: {len(hypotheses)} hyp vs {len(references)} ref"
        )

    # (1) Tương đương multi-bleu.perl -> đối chiếu trực tiếp với paper
    tok_bleu = sacrebleu.corpus_bleu(
        hypotheses, [references], tokenize="none", force=True
    )

    # (2) Chuẩn báo cáo hiện đại: detokenize rồi để sacrebleu tự tokenize 13a
    detok_hyp = [detokenize(h) for h in hypotheses]
    detok_ref = [detokenize(r) for r in references]
    detok_bleu = sacrebleu.corpus_bleu(detok_hyp, [detok_ref], tokenize="13a")

    chrf = sacrebleu.corpus_chrf(detok_hyp, [detok_ref])

    return {
        "bleu_tokenized": round(tok_bleu.score, 2),
        "bleu_detok": round(detok_bleu.score, 2),
        "chrf2": round(chrf.score, 2),
        "bleu_signature": str(tok_bleu.format(signature="")).strip(),
        "precisions": [round(p, 2) for p in tok_bleu.precisions],
        "brevity_penalty": round(tok_bleu.bp, 4),
        "sys_len": tok_bleu.sys_len,
        "ref_len": tok_bleu.ref_len,
    }
