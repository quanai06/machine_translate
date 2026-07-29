#!/usr/bin/env bash
# Tải corpus song ngữ IWSLT'15 English-Vietnamese.
#
# Nguồn gốc là Stanford NLP (https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi/)
# nhưng server đó trả 403 với client không phải trình duyệt. Ta dùng mirror trên
# GitHub — bản sao y hệt, cùng số câu, cùng cách tokenize.
#
#   train    133,317 câu
#   tst2012    1,553 câu  (dev)
#   tst2013    1,268 câu  (test — mọi paper báo BLEU trên split này)
#
# Script tự bỏ qua phần đã có, nên chạy lại nhiều lần đều an toàn.

set -euo pipefail

DEST="${1:-$(dirname "$0")/../data/iwslt15_en_vi}"
MIRROR="https://github.com/stefan-it/nmt-en-vi/raw/master/data"

mkdir -p "$DEST"
cd "$DEST"

# Kiểm tra chính file dữ liệu, không dùng file đánh dấu riêng: nếu repo đã commit
# sẵn data (clone về là có), script phải nhận ra và không tải lại.
fetch() {
    archive=$1; shift
    for f in "$@"; do
        if [ ! -s "$f" ]; then
            echo ">> tải ${archive}.tgz"
            curl -sSfL -O "${MIRROR}/${archive}.tgz"
            tar xzf "${archive}.tgz"
            rm -f "${archive}.tgz"
            return
        fi
    done
    echo ">> đã có: $* — bỏ qua"
}

fetch train-en-vi     train.en    train.vi
fetch dev-2012-en-vi  tst2012.en  tst2012.vi
fetch test-2013-en-vi tst2013.en  tst2013.vi

echo
echo "Số câu mỗi file:"
wc -l train.en train.vi tst2012.en tst2012.vi tst2013.en tst2013.vi

echo
echo "Kiểm tra (kỳ vọng 133317 / 1553 / 1268):"
for pair in "train:133317" "tst2012:1553" "tst2013:1268"; do
    stem="${pair%%:*}"; want="${pair##*:}"
    for lang in en vi; do
        got=$(wc -l < "${stem}.${lang}")
        if [ "$got" != "$want" ]; then
            echo "  LỖI ${stem}.${lang}: có $got dòng, cần $want" >&2
            exit 1
        fi
    done
done
echo "  OK — dữ liệu đầy đủ và đúng số câu."
