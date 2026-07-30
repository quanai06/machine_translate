# NMT English → Vietnamese — IWSLT'15

Cài đặt lại hai kiến trúc dịch máy trên cùng bộ IWSLT'15 En-Vi và so sánh.

Lê Hoàng Quân — MSV 24022433 · https://github.com/quanai06/machine_translate

**[Báo cáo đầy đủ →](report/BAO_CAO.md)** · [Bảng so sánh chi tiết →](results/COMPARISON.md)
· [Nhật ký lần chạy trên Colab →](notebooks/translate_colab_run.ipynb)

| | Case 1 | Case 2 |
|---|---|---|
| Kiến trúc | Transformer encoder-decoder | Seq2Seq LSTM + Luong attention |
| Tham chiếu | `demo_transformer.ipynb` | [`tensorflow/nmt`](https://github.com/tensorflow/nmt) |
| Thư mục | `translate_transformers/` | `translate_seq2seq/` |
| Tham số | 39.7M | 20.0M |

Hai model dùng chung tokenizer, batching và hàm BLEU (`common/`), nên chênh lệch
đo được phản ánh kiến trúc chứ không phản ánh cách train.

## Cài đặt

```bash
pip install -r requirements.txt
bash scripts/download_data.sh     # IWSLT'15 En-Vi, tự verify số câu
python scripts/prepare.py         # SentencePiece BPE 8k/phía, chạy 1 lần
```

## Chạy

```bash
python scripts/check_amp.py                         # kiểm tra đường fp16
python scripts/sanity_check.py                      # overfit test, ~5 phút trên GPU
python translate_transformers/train.py
python translate_seq2seq/train.py
python compare.py --plots                           # -> runs/COMPARISON.md
```

Trên Colab: mở `notebooks/colab_transformer.ipynb` hoặc `colab_seq2seq.ipynb`,
chọn T4 GPU, Run all.

Colab free thường ngắt phiên sau ~4 giờ. `runs/*/last.pt` được ghi sau mỗi epoch
(model + optimizer + scheduler + history), train tiếp bằng:

```bash
python translate_seq2seq/train.py --resume true
```

Kiểm tra pipeline không lỗi (vài phút, chạy được trên CPU):

```bash
python translate_transformers/train.py --smoke
```

Chấm lại checkpoint hoặc dịch câu lẻ:

```bash
python evaluate.py --model transformer --beam 5 --show 8
python evaluate.py --model seq2seq --text "I love machine translation ."
```

## Dữ liệu

Corpus phụ đề TED Talks, phát hành kèm Luong & Manning [9].

| Split | File | Câu |
|---|---|---|
| train | `train.{en,vi}` | 133,317 |
| dev | `tst2012.{en,vi}` | 1,553 |
| test | `tst2013.{en,vi}` | 1,268 |

Literature báo BLEU trên tst2013; dự án giữ nguyên quy ước.

Thống kê từ `scripts/prepare.py`:

| | câu | token | vocab thô | dài TB | p95 | hapax |
|---|---|---|---|---|---|---|
| train.en | 133,317 | 2,706,255 | 54,169 | 20.3 | 47 | 21,912 (40%) |
| train.vi | 133,317 | 3,311,508 | 25,615 | 24.8 | 59 | 11,298 (44%) |

40% vocab tiếng Anh xuất hiện đúng một lần. Với vocab word-level 17k của Stanford
thì phần lớn số đó thành `<unk>`, nên dự án dùng BPE. Câu tiếng Việt dài hơn 22%
do viết rời âm tiết, ảnh hưởng cách đọc BLEU (xem mục BLEU).

Server Stanford trả HTTP 403 cho client không phải trình duyệt, nên
`download_data.sh` lấy từ mirror [14].

## Framework: PyTorch cho cả hai case

- `tensorflow/nmt` dựa vào `tf.contrib.seq2seq`, bị gỡ khỏi TF2. TensorFlow
  Addons (`tfa.seq2seq`) đã EOL 05/2024. Chọn TF vẫn phải viết lại từ đầu.
- Hai case khác framework thì chênh lệch thời gian và bộ nhớ lẫn giữa "khác kiến
  trúc" với "khác framework", bảng so sánh mất giá trị.
- `nn.TransformerEncoder`/`Decoder` và `nn.LSTM` + `pack_padded_sequence` đều là
  API lõi, không phụ thuộc package phụ.
- Vòng train tường minh đo được thời gian/epoch, throughput và bộ nhớ đỉnh chính
  xác hơn `model.fit`.

`demo_transformer.ipynb` dùng làm tài liệu đối chiếu công thức.

## Cấu hình

### Case 1 — Transformer

| Tham số | Giá trị | Căn cứ |
|---|---|---|
| d_model / heads | 512 / 4 | head_dim=128; 8 head quá mảnh ở dữ liệu nhỏ |
| Lớp | 6 enc + 6 dec | như base |
| d_ff | 1024 | base 2048, giảm nửa để bớt overfit |
| dropout | 0.3 | base 0.1, low-resource cần cao hơn |
| norm_first | True | [2] |
| Weight tying | True | −4M tham số |
| Label smoothing | 0.1 | [1] |
| LR | inverse-sqrt, 4000 warmup | [1] |

Không dùng cấu hình base (d_ff 2048, dropout 0.1): base train trên WMT'14 với
4.5M câu, IWSLT'15 chỉ có 133k.

### Case 2 — Seq2Seq + Attention

Bám cấu hình benchmark En-Vi của `tensorflow/nmt`.

| Tham số | Giá trị | Căn cứ |
|---|---|---|
| Encoder | 1 lớp bi-LSTM 512 | [13] |
| Decoder | 2 lớp LSTM 512 | [13] |
| Attention | Luong `general`, scale | [8] |
| Input feeding | có | [8] |
| dropout | 0.2 | = 1 − keep_prob(0.8) [13] |
| clip gradient | 5.0 | `max_gradient_norm` [13] |
| Beam / length penalty | 10 / α=1.0 | [13]; [10] |
| Optimizer | Adam 1e-3 | xem dưới |

tf/nmt dùng SGD lr=1.0. Ở đây mặc định Adam để hai case khác nhau ở kiến trúc
chứ không ở họ optimizer. Tái hiện đúng tf/nmt:

```bash
python translate_seq2seq/train.py --optimizer sgd --lr 1.0 --epochs 12
```

Chung cho cả hai: BPE 8k/phía, không tách từ tiếng Việt, fp16, batch gom theo
token (8192), early stop theo BLEU dev.

## BLEU

Corpus phát hành ở dạng đã tokenize Moses. Các con số kinh điển trên bộ này là
tokenized BLEU (`multi-bleu.perl`) trên chính văn bản đó.

`sacrebleu` mặc định tokenize lại bằng `13a`. Đưa thẳng văn bản đã tokenize vào
sacrebleu mặc định cho ra một con số thứ ba, không so được với gì.

`common/metrics.py` báo cả hai:

| Chỉ số | Cách tính | Dùng để |
|---|---|---|
| `bleu_tokenized` | `sacrebleu(tokenize="none")` trên văn bản tokenized | đối chiếu paper (≡ `multi-bleu.perl`) |
| `bleu_detok` | detokenize rồi `sacrebleu(tokenize="13a")` | chuẩn báo cáo hiện đại |
| `chrf2` | chrF2 trên văn bản detokenized | ít nhạy với cách tokenize |

BLEU tính trên âm tiết ("học sinh" = 2 token), nhất quán với literature của bộ
này, nhưng không so được với hệ thống có tách từ tiếng Việt.

## Kết quả

Đo trên tst2013, Tesla T4, tokenizer BPE 8k dùng chung.

| | Case 1 Transformer | Case 2 Seq2Seq |
|---|---|---|
| BLEU greedy | 29.47 | 26.64 |
| BLEU beam | 30.35 (beam=5) | 27.81 (beam=10) |
| BLEU detok | 30.39 | 27.85 |
| chrF2 | 48.85 | 46.55 |
| Tham số | 39.7M | 20.0M |
| Giây/epoch | 144 | 152 |
| Epoch tốt nhất | 30 (chưa hội tụ) | 12 |
| Kết thúc | hết 30 epoch | early stop ở 17 |

Đối chiếu số đã công bố trên cùng tst2013:

| Hệ thống | BLEU tokenized | Nguồn |
|---|---|---|
| Luong & Manning 2015 | 23.3 | [9] |
| `tensorflow/nmt`, greedy | 25.5 | [13] |
| `tensorflow/nmt`, beam=10 | 26.1 | [13] |
| Transformers without Tears 2019 | 32.8 | [2] |

Case 2 vượt chính bản gốc nó tái hiện (+1.7 ở beam=10), nhiều khả năng nhờ BPE
thay cho vocab word-level 17k/7.7k và Adam thay cho SGD 1.0.

Hai điểm cần nêu khi đọc bảng:

Transformer hơn 2.5 BLEU nhưng dùng gấp đôi tham số. Nó cũng chưa hội tụ: dev
BLEU vẫn tăng ở epoch cuối (26.11 → 26.42) và early stop chưa kích hoạt, nên
train thêm sẽ còn lên. Seq2Seq thì đã hội tụ thật, đỉnh ở epoch 12 và early stop
ở 17, train thêm không giúp gì.

Thời gian mỗi epoch gần như bằng nhau, ngược với kỳ vọng thông thường rằng
Transformer nhanh hơn nhiều nhờ song song hoá. Lý do: Seq2Seq chỉ có nửa số tham
số nên nửa khối lượng tính toán mỗi token, bù lại đúng phần thiệt do input
feeding chạy tuần tự; thêm nữa attention của Transformer tốn O(T²). Ở quy mô
model này trên T4, hai yếu tố triệt tiêu nhau.

Lúc sinh câu thì Transformer chậm hơn 37.7× (52.8 vs 1.4 ms/câu). Đây là hạn chế
của bản cài đặt chứ không phải kiến trúc: `greedy_decode` không cache key/value
nên mỗi bước phải chạy lại decoder trên toàn bộ tiền tố.

![So sánh hai kiến trúc](results/comparison.png)

## Áp dụng từ literature

Số trong ngoặc trỏ tới [Tài liệu tham khảo](#tài-liệu-tham-khảo) ở cuối.

### Case 1 — Transformer

| Nguồn | Áp dụng vào |
|---|---|
| Vaswani et al. 2017 [1] | Kiến trúc, positional encoding sin/cos, label smoothing, lịch inverse-sqrt |
| Nguyen & Salazar 2019 [2] | `norm_first=True` + LayerNorm cuối stack |
| Sennrich et al. 2016 [3] | SentencePiece BPE 8k/phía |
| Phan-Vu et al. 2017 [4] | Bỏ bước tách từ tiếng Việt |
| Ott et al. 2018 [5] | fp16, `TokenBatchSampler`, gradient accumulation |

[2] thí nghiệm trên đúng bộ IWSLT'15 En-Vi và lập mốc 32.8 BLEU. PreNorm lại làm
giảm chất lượng ở high-resource (WMT'14 En-De), tức là đánh đổi phụ thuộc lượng
dữ liệu chứ không phải cải tiến phổ quát. ScaleNorm/FixNorm chưa cài.

[4] kết luận tách từ tiếng Việt không cần thiết cho NMT dùng subword, dù nó giúp
rõ rệt với dịch máy thống kê. [12] là bản mở rộng của cùng nhóm, khảo sát cả hai
chiều En↔Vi và nhiều cấu hình hyper-parameter hơn.

### Case 2 — Seq2Seq + Attention

| Nguồn | Áp dụng vào |
|---|---|
| Sutskever et al. 2014 [6] | Khung encoder-decoder LSTM |
| Bahdanau et al. 2015 [7] | Attention; score MLP cộng = `attention="concat"` |
| Luong et al. 2015 [8] | `LuongAttention` (`dot`/`general`/`concat`), input feeding |
| Luong & Manning 2015 [9] | Quy ước chia split; mốc BLEU 23.3 |
| Wu et al. 2016 [10] | Length penalty `((5+\|Y\|)/6)^α` cho beam search |
| Nguyen et al. 2019 [11] | Xác nhận rare word là đòn bẩy ở cặp En-Vi (giải bằng BPE) |

[8] là nền tảng trực tiếp của `tensorflow/nmt`. Input feeding buộc decoder chạy
tuần tự từng timestep, không song song hoá được như teacher forcing của
Transformer. Về lý thuyết đây là bất lợi tốc độ, nhưng đo thực tế trên T4 thì
hai case gần bằng nhau (152 vs 144 giây/epoch) — xem mục Kết quả.

[9] tạo ra bộ dữ liệu này. Phần thích nghi miền (pre-train WMT → fine-tune TED,
+3.8 BLEU) không áp dụng vì ta chỉ dùng dữ liệu trong miền.

Chuỗi [6] → [7] → [8] → [1] gỡ dần nút thắt cổ chai: vector cố định, nhìn lại
toàn bộ nguồn, nhìn hiệu quả hơn kèm nhớ đã nhìn đâu, bỏ hồi tiếp.

### Nghiên cứu En-Vi trong nước

Nhóm Trần Hồng Việt (ĐH Công nghệ – ĐHQG Hà Nội) làm cùng cặp ngôn ngữ. [15] là
hệ thống dự thi cùng đợt đánh giá IWSLT 2015 đã sinh ra bộ dữ liệu này, in ở
trang 80–83 cùng kỷ yếu với [9] (trang 76–79).

| Nguồn | Nội dung |
|---|---|
| Trần et al. 2015 [15] | Hệ SMT phrase-based (Moses + Phrasal) dự thi IWSLT 2015 En-Vi |
| Trần et al. 2016 [16] | Mô hình đảo trật tự Vi-En dựa trên thông tin phụ thuộc |
| Trần et al. 2022 [17] | Denoising autoencoder cho NMT + pre-ordering cho PBSMT |
| Ngô et al. 2022 [18] | KC4MT, corpus đa ngữ Vi-Trung/Lào/Khmer |

[15] cho baseline SMT trên cùng bài toán: cao nhất En→Vi là 23.15 BLEU (Moses,
LM 4-gram mở rộng bằng 1 GB dữ liệu báo điện tử tiếng Việt), Vi→En là 20.18
(Phrasal). Ba lần chạy RUN01/02/03 chỉ khác nhau ở lượng dữ liệu đơn ngữ và BLEU
tăng đều, tức đóng góp nằm ở language model chứ không ở mô hình dịch.

Không so trực tiếp 23.15 với bảng Mốc đối chiếu được: [15] dùng dev `TED.dev2010`
(745 câu), test `TED.tst2015` (1046 câu), train 122,132 câu sau lọc; dự án này
dùng dev `tst2012` (1553), test `tst2013` (1268), train 133,317 câu.

[15] tách từ tiếng Việt bằng VnTokenizer còn [4] kết luận không cần. Không mâu
thuẫn: [15] là SMT, phrase table khớp theo chuỗi từ nên ghép "học_sinh" giảm số
phrase phải học; [4] là NMT dùng subword nên tự học được ranh giới từ.

[16] và [17] đều xử lý trật tự từ En↔Vi (tiếng Việt đặt tính từ sau danh từ).
SMT cần pre-ordering tường minh, Transformer học qua self-attention.

## Cấu trúc

```
data/
  iwslt15_en_vi/          corpus thô
  tokenizer/              SentencePiece BPE dùng chung
common/
  data.py                 split, TokenBatchSampler, collate
  tokenizer.py            SentencePiece
  metrics.py              BLEU tokenized/detokenized, chrF2
  engine.py               train/eval/decode dùng chung
  bench.py                tham số, throughput, latency, bộ nhớ
  cli.py                  seed, argparse từ dataclass
translate_transformers/   config.py  model.py  train.py
translate_seq2seq/        config.py  model.py  train.py
scripts/
  download_data.sh        tải + verify
  prepare.py              thống kê corpus + train tokenizer
  check_amp.py            kiểm tra fp16 (chạy được cả khi không có GPU)
  sanity_check.py         overfit test
notebooks/                Colab cho từng case
evaluate.py               chấm lại checkpoint, dịch câu lẻ
compare.py                bảng + biểu đồ -> runs/COMPARISON.md
runs/                     benchmark.json, history.csv, best.pt, last.pt
```

## Kiểm tra fp16

`scripts/check_amp.py` chạy forward, backward qua `GradScaler`, greedy và beam
của cả hai model dưới `torch.autocast`. Máy không có GPU thì dùng bfloat16 trên
CPU làm proxy, cùng logic ép kiểu, khác định dạng số.

Lỗi lệch kiểu fp16/fp32 không xuất hiện khi chạy CPU không bật AMP nhưng làm
hỏng lần train đầu trên GPU. Script này đã bắt một lỗi như vậy: `bridge_h`/
`bridge_c` của seq2seq chạy dưới autocast nên sinh state fp16 trong khi input
LSTM bị ép fp32, mà `nn.LSTM` đòi hai thứ cùng kiểu.

## Sanity check

`scripts/sanity_check.py` bắt model học thuộc 120 câu rồi chấm BLEU trên chính
120 câu đó. Cài đặt đúng phải đạt BLEU > 80.

| Model | epoch | greedy | beam=5 |
|---|---|---|---|
| transformer | 150 | 97.4 | 97.4 |
| seq2seq | 500 | 91.7 | 94.7 |

Chênh lệch số epoch không phải lỗi: Transformer cập nhật toàn chuỗi target song
song, LSTM có input feeding phải lan gradient tuần tự.

Đọc kết quả khi fail:

| Triệu chứng | Nguyên nhân thường gặp |
|---|---|
| Greedy cao, beam thấp | beam search: quên hoán vị state theo beam, length penalty sai dấu |
| Cả hai thấp | mask sai, hoặc lệch một bước giữa `tgt_in` và `tgt_out` |
| Loss không giảm | optimizer hoặc khởi tạo tham số |

Test này đã bắt một lỗi: bản đầu port `init_weight=0.1` của `tensorflow/nmt` sang
PyTorch làm logit lúc init có std ~0.005, softmax gần như đều trên 8000 lớp, và
seq2seq chậm hơn Transformer ~10× chỉ để thoát vùng gradient phẳng. Sau khi sửa
init theo fan-in, perplexity sau 200 bước giảm 7.8 → 2.08. Chi tiết trong
`translate_seq2seq/model.py::_init_parameters`.

## Trạng thái

Đã verify trên CPU: tải data, tokenizer, smoke test hai model, sanity check,
check_amp, resume, build cấu hình thật, `compare.py`. Chưa train đầy đủ nên chưa
có BLEU đo được trên tst2013. Các con số ở mục Mốc đối chiếu là từ paper.

---

## Tài liệu tham khảo

### Case 1 — Transformer

[1] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N.,
Kaiser, Ł. & Polosukhin, I. (2017). *Attention Is All You Need.* Advances in
Neural Information Processing Systems 30 (NeurIPS 2017), 5998–6008.
[arXiv:1706.03762](https://arxiv.org/abs/1706.03762)

[2] Nguyen, T. Q. & Salazar, J. (2019). *Transformers without Tears: Improving
the Normalization of Self-Attention.* Proceedings of the 16th International
Conference on Spoken Language Translation (IWSLT 2019), Hong Kong.
[aclanthology.org/2019.iwslt-1.17](https://aclanthology.org/2019.iwslt-1.17/) ·
[arXiv:1910.05895](https://arxiv.org/abs/1910.05895)

[3] Sennrich, R., Haddow, B. & Birch, A. (2016). *Neural Machine Translation of
Rare Words with Subword Units.* Proceedings of the 54th Annual Meeting of the
Association for Computational Linguistics (ACL 2016), Berlin, 1715–1725.
[aclanthology.org/P16-1162](https://aclanthology.org/P16-1162/) ·
[arXiv:1508.07909](https://arxiv.org/abs/1508.07909)

[4] Phan-Vu, H.-H., Tran, V. T., Nguyen, V. N., Dang, H. V. & Do, P. T. (2017).
*Towards State-of-the-art English-Vietnamese Neural Machine Translation.*
Proceedings of the 8th International Symposium on Information and Communication
Technology (SoICT 2017).
[dl.acm.org/doi/10.1145/3155133.3155205](https://dl.acm.org/doi/10.1145/3155133.3155205)

[5] Ott, M., Edunov, S., Grangier, D. & Auli, M. (2018). *Scaling Neural Machine
Translation.* Proceedings of the Third Conference on Machine Translation
(WMT 2018), Brussels, 1–9.
[aclanthology.org/W18-6301](https://aclanthology.org/W18-6301/) ·
[arXiv:1806.00187](https://arxiv.org/abs/1806.00187)

### Case 2 — Seq2Seq + Attention

[6] Sutskever, I., Vinyals, O. & Le, Q. V. (2014). *Sequence to Sequence Learning
with Neural Networks.* Advances in Neural Information Processing Systems 27
(NeurIPS 2014), 3104–3112.
[arXiv:1409.3215](https://arxiv.org/abs/1409.3215)

[7] Bahdanau, D., Cho, K. & Bengio, Y. (2015). *Neural Machine Translation by
Jointly Learning to Align and Translate.* International Conference on Learning
Representations (ICLR 2015), San Diego.
[arXiv:1409.0473](https://arxiv.org/abs/1409.0473)

[8] Luong, M.-T., Pham, H. & Manning, C. D. (2015). *Effective Approaches to
Attention-based Neural Machine Translation.* Proceedings of the 2015 Conference
on Empirical Methods in Natural Language Processing (EMNLP 2015), Lisbon,
1412–1421.
[aclanthology.org/D15-1166](https://aclanthology.org/D15-1166/) ·
[arXiv:1508.04025](https://arxiv.org/abs/1508.04025)

[9] Luong, M.-T. & Manning, C. D. (2015). *Stanford Neural Machine Translation
Systems for Spoken Language Domains.* Proceedings of the 12th International
Workshop on Spoken Language Translation (IWSLT 2015), Đà Nẵng, 76–79.
[aclanthology.org/2015.iwslt-evaluation.11](https://aclanthology.org/2015.iwslt-evaluation.11/) ·
[PDF](https://nlp.stanford.edu/pubs/luong-manning-iwslt15.pdf)

[10] Wu, Y., Schuster, M., Chen, Z., Le, Q. V., Norouzi, M. et al. (2016).
*Google's Neural Machine Translation System: Bridging the Gap between Human and
Machine Translation.*
[arXiv:1609.08144](https://arxiv.org/abs/1609.08144)

[11] Nguyen, T.-V., Nguyen, L.-M., Nguyen, P.-T. et al. (2019). *Overcoming the
Rare Word Problem for Low-Resource Language Pairs in Neural Machine Translation.*
Proceedings of the 6th Workshop on Asian Translation (WAT@ACL 2019), Hong Kong,
207–214.
[aclanthology.org/D19-5228](https://aclanthology.org/D19-5228/) ·
[arXiv:1910.03467](https://arxiv.org/abs/1910.03467)

### Bổ sung — tiếng Việt

[12] Phan-Vu, H.-H., Tran, V. T., Nguyen, V. N., Dang, H. V. & Do, P. T. (2019).
*Neural Machine Translation between Vietnamese and English: an Empirical Study.*
Journal of Computer Science and Cybernetics 35(2), 147–166.
[vjs.ac.vn](https://vjs.ac.vn/index.php/jcc/article/view/13233) ·
[arXiv:1810.12557](https://arxiv.org/abs/1810.12557)

### Nhóm nghiên cứu En-Vi trong nước

[15] Tran, V. H., Vu, H. T., Le, T. T., Pham, L. N. & Nguyen, V. V. (2015).
*The English-Vietnamese Machine Translation System for IWSLT 2015.* Proceedings
of the 12th International Workshop on Spoken Language Translation: Evaluation
Campaign (IWSLT 2015), Đà Nẵng, 80–83.
[aclanthology.org/2015.iwslt-evaluation.12](https://aclanthology.org/2015.iwslt-evaluation.12/) ·
[PDF](https://workshop2015.iwslt.org/downloads/IWSLT_2015_EP_3.pdf)

[16] Tran, H.-V. et al. (2016). *A Reordering Model for Vietnamese-English
Statistical Machine Translation Using Dependency Information.* IEEE Conference
Publication.
[ieeexplore.ieee.org/document/7800281](https://ieeexplore.ieee.org/document/7800281/)

[17] Hong-Viet, T., Van-Vinh, N. & Hoang-Quan, N. (2022). *Improving Machine
Translation Quality with Denoising Autoencoder and Pre-Ordering.* Journal of
Computing and Information Technology (CIT) 29(1), 39–56.
[cit.fer.hr](http://cit.fer.hr/index.php/CIT/article/view/5316)

[18] Ngo, T.-V., Tran, H.-V., Nguyen, V. V. et al. (2022). *KC4MT: A High-Quality
Corpus for Multilingual Machine Translation.* Proceedings of the 13th Language
Resources and Evaluation Conference (LREC 2022).
[aclanthology.org/2022.lrec-1.588](https://aclanthology.org/2022.lrec-1.588/)

### Mã nguồn tham chiếu

[13] TensorFlow. *Neural Machine Translation (seq2seq) Tutorial.*
[github.com/tensorflow/nmt](https://github.com/tensorflow/nmt) — TF 1.x, archived.

[14] Schweter, S. *nmt-en-vi: IWSLT'15 English-Vietnamese data.*
[github.com/stefan-it/nmt-en-vi](https://github.com/stefan-it/nmt-en-vi)
