# So sánh Case 1 (Transformer) vs Case 2 (Seq2Seq + Attention)

Bộ dữ liệu: IWSLT'15 English-Vietnamese. Train `train.{en,vi}` (133,317 câu), dev `tst2012`, test `tst2013`.
Cả hai model dùng CHUNG tokenizer, chung cách chia batch, chung công thức BLEU — nên chênh lệch dưới đây phản ánh kiến trúc, không phải kỹ thuật train.

| Tiêu chí | Transformer | Seq2seq |
|---|---|---|
| **Chất lượng dịch (tst2013)** |  |  |
| BLEU tokenized — greedy | 29.47 | 26.64 |
| BLEU tokenized — beam | 30.35 | 27.81 |
| BLEU detokenized (sacreBLEU 13a) | 30.39 | 27.85 |
| chrF2 | 48.85 | 46.55 |
| BLEU tốt nhất trên dev (tst2012) | 26.42 | 23.96 |
| **Kích thước mô hình** |  |  |
| Tổng tham số | 39,737,344 | 20,005,889 |
| Tham số ngoài embedding | 31,545,344 | 11,813,889 |
| Tham số embedding | 8,192,000 | 8,192,000 |
| **Chi phí huấn luyện** |  |  |
| Thời gian train (giây) | 5727.00 | 2763.40 |
| Trung vị thời gian/epoch (giây) | 143.80 | 151.90 |
| Throughput (token/giây) | 23,600 | 22,564 |
| Epoch tốt nhất | 30 | 12 |
| Đỉnh bộ nhớ GPU (GB) | 8.86 | 4.85 |
| **Tốc độ suy luận** |  |  |
| Câu/giây (greedy) | 18.93 | 716.64 |
| ms mỗi câu (greedy) | 52.84 | 1.40 |

## Nhận xét về tốc độ

- Thời gian mỗi epoch gần như bằng nhau (144s vs 152s, lệch 6%), ngược với kỳ vọng thông thường rằng Transformer phải nhanh hơn hẳn nhờ song song hoá. Seq2Seq chỉ có 20.0M tham số so với 39.7M, tức khoảng 2.0× ít phép tính mỗi token, bù lại phần thiệt do chạy tuần tự; thêm nữa attention của Transformer tốn O(T²) theo độ dài câu. Ở quy mô model này, hai yếu tố triệt tiêu nhau.
- Khi SINH câu, Transformer chậm hơn 37.7× (52.8 vs 1.4 ms/câu). Đây là hạn chế của BẢN CÀI ĐẶT chứ không phải của kiến trúc: `greedy_decode` ở đây không cache key/value, nên mỗi bước sinh phải chạy lại decoder trên toàn bộ tiền tố, tổng chi phí O(T²). LSTM không gặp vấn đề này vì state đã tóm tắt sẵn quá khứ trong một vector. Thêm KV cache sẽ thu hẹp khoảng cách này.

## Đối chiếu với số liệu đã công bố (cùng tst2013)

| Hệ thống | BLEU tokenized |
|---|---|
| tensorflow/nmt (LSTM+Luong attn, greedy) | 25.5 |
| tensorflow/nmt (LSTM+Luong attn, beam=10) | 26.1 |
| Luong & Manning 2015 (hệ thống IWSLT'15) | 23.3 |
| Transformers without Tears 2019 (SOTA thời điểm đó) | 32.8 |

> Lưu ý khi đối chiếu: các con số trên là *tokenized BLEU* (`multi-bleu.perl`) trên văn bản đã tokenize theo Moses. Dòng 'BLEU tokenized' trong bảng của ta được tính bằng `sacrebleu(tokenize='none')`, tương đương. Đừng so nhầm với dòng 'BLEU detokenized'.