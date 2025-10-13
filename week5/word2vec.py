import logging
import os
import time
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec

# ==================================
# ⚙️ CẤU HÌNH ĐƯỜNG DẪN & THAM SỐ
# ==================================
OUTPUT_DIR = r"D:\Vietnamese_Word2Vec"
CORPUS_PATH = os.path.join(OUTPUT_DIR, "vietnamese_corpus_tokenized.txt")
MODEL_PATH = os.path.join(OUTPUT_DIR, "word2vec_model_final.bin")
CHECKPOINT_PATH_FORMAT = os.path.join(OUTPUT_DIR, "word2vec_model_epoch_{}.bin")

VECTOR_SIZE = 300
WINDOW = 5
MIN_COUNT = 5
WORKERS = os.cpu_count() - 2 or 4
EPOCHS = 15

# ==================================
# 🧩 CẤU HÌNH LOGGING
# ==================================
logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s",
    level=logging.INFO
)

# ==================================
# ⚡ FAST LINE SENTENCE (ĐỌC NHANH HƠN)
# ==================================
class FastLineSentence:
    """
    Generator đọc file corpus cực nhanh với buffer lớn (128MB).
    Tiết kiệm RAM, tận dụng SSD/OS cache hiệu quả.
    """
    def __init__(self, fname, encoding="utf8", buffer_size=128 * 1024 * 1024):
        self.fname = fname
        self.encoding = encoding
        self.buffer_size = buffer_size

    def __iter__(self):
        total_bytes = os.path.getsize(self.fname)
        read_bytes = 0
        start_time = time.time()

        with open(self.fname, "r", encoding=self.encoding, buffering=self.buffer_size) as f:
            for line in f:
                read_bytes += len(line.encode("utf-8"))
                yield line.strip().split()

                # Log tiến độ đọc (mỗi 1GB)
                if read_bytes % (1_000_000_000) < 100_000:
                    elapsed = time.time() - start_time
                    elapsed = max(elapsed, 1e-9)  # tránh chia cho 0
                    speed = read_bytes / (1024 * 1024 * elapsed)
                    logging.info(f"📖 Đã đọc {(read_bytes / 1_000_000_000):.2f} GB / {(total_bytes / 1_000_000_000):.2f} GB "
                                 f"({speed:.1f} MB/s)")

# ==================================
# 🎓 CALLBACK LƯU CHECKPOINT
# ==================================
class EpochSaver(CallbackAny2Vec):
    def __init__(self, start_time, checkpoint_path_format):
        self.epoch = 0
        self.start_time = start_time
        self.checkpoint_path_format = checkpoint_path_format

    def on_epoch_begin(self, model):
        logging.info(f"🚀 Bắt đầu Epoch {self.epoch + 1}...")

    def on_epoch_end(self, model):
        self.epoch += 1
        elapsed = time.time() - self.start_time
        checkpoint_path = self.checkpoint_path_format.format(self.epoch)
        model.save(checkpoint_path)
        logging.info(f"✅ Hoàn thành Epoch {self.epoch}. Đã lưu checkpoint tại: {checkpoint_path}")
        logging.info(f"⏱️ Thời gian tổng: {elapsed:.2f}s")

# ==================================
# 🧠 HUẤN LUYỆN WORD2VEC
# ==================================
def train_word2vec():
    if not os.path.exists(CORPUS_PATH):
        logging.error(f"❌ Không tìm thấy corpus tại: {CORPUS_PATH}")
        return None

    logging.info(f"📚 Đọc corpus từ: {CORPUS_PATH}")
    sentences = FastLineSentence(CORPUS_PATH)
    start_time = time.time()
    epoch_saver = EpochSaver(start_time, CHECKPOINT_PATH_FORMAT)

    logging.info(f"🧠 Bắt đầu huấn luyện Skip-Gram với {WORKERS} workers...")
    model = Word2Vec(
        sentences=sentences,
        vector_size=VECTOR_SIZE,
        window=WINDOW,
        min_count=MIN_COUNT,
        workers=WORKERS,
        sg=1,  # ✅ Skip-Gram
        negative=10,  # Số từ âm (negative sampling)
        sample=1e-5,  # Ngưỡng downsample
        epochs=EPOCHS,
        compute_loss=True,
        callbacks=[epoch_saver]
    )

    model.save(MODEL_PATH)
    logging.info(f"💾 Hoàn tất toàn bộ {EPOCHS} Epochs. Lưu mô hình cuối cùng tại: {MODEL_PATH}")
    return model

# ==================================
# 🧪 KIỂM THỬ MÔ HÌNH
# ==================================
def test_model(model):
    print("\n" + "="*50)
    print("🌟 KIỂM THỬ MÔ HÌNH WORD2VEC 🌟")
    print("="*50)

    test_words = ["Việt_Nam", "Hà_Nội", "giáo_dục", "Covid-19", "điện_thoại"]
    for word in test_words:
        if word in model.wv:
            print(f"\n👉 {word}:")
            for rank, (w, score) in enumerate(model.wv.most_similar(word, topn=10)):
                print(f"   {rank+1}. {w:<20} ({score:.4f})")
        else:
            print(f"\n⚠️ '{word}' không đủ tần suất (min_count={MIN_COUNT}).")

# ==================================
# ▶️ CHẠY CHƯƠNG TRÌNH
# ==================================
if __name__ == "__main__":
    trained_model = train_word2vec()
    if trained_model:
        test_model(trained_model)
