import re
import os
import unicodedata
import logging
from tqdm import tqdm
from datasets import load_dataset
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
import pandas as pd
from pyvi import ViTokenizer

# ==========================
# ⚙️ CẤU HÌNH CHUNG
# ==========================
OUTPUT_DIR = r"D:\Vietnamese_Word2Vec"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FINAL_MODEL_PATH = os.path.join(OUTPUT_DIR, "w2v_vietnamese.model")
CHECKPOINT_PREFIX = os.path.join(OUTPUT_DIR, "w2v_vietnamese_checkpoint")
WORKERS = 15 # Số luồng CPU để huấn luyện
# ==========================
# 🧩 CẤU HÌNH LOGGING
# ==========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s : %(levelname)s : %(message)s"
)
logger = logging.getLogger(__name__)

# ==========================
# 🔁 CALLBACK LƯU CHECKPOINT
# ==========================
class EpochSaver(CallbackAny2Vec):
    """Callback lưu mô hình sau mỗi epoch."""
    def __init__(self, prefix):
        self.prefix = prefix
        self.epoch = 0

    def on_epoch_end(self, model):
        self.epoch += 1
        path = f"{self.prefix}_epoch_{self.epoch}.model"
        model.save(path)
        logger.info(f"[Checkpoint] ✅ Đã lưu mô hình sau epoch {self.epoch}: {path}")

# ==========================
# 🧹 HÀM TIỀN XỬ LÝ CHUỖI
# ==========================
_re_non_vn = re.compile(r"[^0-9a-zA-ZÀ-Ỵà-ỵđĐ\s]", re.UNICODE)
_re_multi_space = re.compile(r"\s+")

def normalize_text(text):
    """Chuẩn hoá Unicode và loại bỏ ký tự không hợp lệ."""
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFC", text)
    text = _re_non_vn.sub(" ", text)
    text = _re_multi_space.sub(" ", text)
    return text.strip().lower()

_re_valid_vn = re.compile(
    r"^[a-zàáảãạăắằẳẵặâấầẩẫậđèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵ]+$",
    re.UNICODE
)

def is_valid_vietnamese_word(word):
    """Kiểm tra token có hợp lệ tiếng Việt hay không."""
    return bool(_re_valid_vn.match(word))
# ==========================
# 🧱 XÂY TRIE CHO STOPWORD
# ==========================
def build_trie(phrases):
    """Tạo Trie cho stopwords."""
    trie = {}
    max_len = 0
    single_word_set = set()

    for phr in phrases:
        toks = phr.split()
        if not toks:
            continue
        if len(toks) == 1:
            single_word_set.add(toks[0])
            max_len = max(max_len, 1)
            continue
        node = trie
        for tok in toks:
            node = node.setdefault(tok, {})
        node["_end"] = True
        max_len = max(max_len, len(toks))
    return trie, single_word_set, max_len

# ==========================
# 🧠 TOKENIZE + CLEAN
# ==========================
def tokenize_and_clean_vietnamese(text, trie, single_stop, max_phrase_len):
    """Tách từ, loại stopword & từ rác."""
    text = normalize_text(text)
    # ✅ Dùng PyVi để tách từ đúng ngữ cảnh tiếng Việt
    tokenized = ViTokenizer.tokenize(text)
    # PyVi sẽ tách "đi học" -> "đi_học", "thành phố" -> "thành_phố"
    tokens = [t.replace("_", " ") for t in tokenized.split() if is_valid_vietnamese_word(t.replace("_", " "))]

    filtered = []
    i, n = 0, len(tokens)

    while i < n:
        node = trie
        j = i
        steps = 0
        match_len = 0

        while j < n and steps < max_phrase_len:
            tok = tokens[j]
            if not isinstance(node, dict) or tok not in node:
                break
            node = node[tok]
            j += 1
            steps += 1
            if node.get("_end"):
                match_len = j - i

        if match_len == 0 and tokens[i] not in single_stop:
            filtered.append(tokens[i])

        i += max(match_len, 1)

    return filtered

# ==========================
# 📚 STREAM CORPUS
# ==========================
def tokenize_stream(dataset_iter, trie, single_stop, max_len):
    """Sinh dữ liệu tokenized cho Word2Vec."""
    for sample in dataset_iter:
        text = sample.get("text", "")
        toks = tokenize_and_clean_vietnamese(text, trie, single_stop, max_len)
        if toks:
            yield toks

# ==========================
# 🚀 MAIN
# ==========================
def main():
    # ---- 1️⃣ TẢI STOPWORDS ----
    stopword_file = "stopwords.csv"
    df = pd.read_csv(stopword_file)
    stop_phrases = [str(w).strip().replace("_", " ") for w in df["stopwords"].dropna()]
    trie, single_stop, max_phrase_len = build_trie(stop_phrases)
    logger.info(f"✅ Đã tải {len(stop_phrases)} cụm stopword.")

    # ---- 2️⃣ KIỂM TRA CHECKPOINT ----
    latest_ckpt = None
    ckpt_epochs = []
    for f in os.listdir(OUTPUT_DIR):
        if f.startswith("w2v_vietnamese_checkpoint_epoch_") and f.endswith(".model"):
            num = int(re.findall(r"epoch_(\d+)", f)[0])
            ckpt_epochs.append((num, os.path.join(OUTPUT_DIR, f)))
    if ckpt_epochs:
        latest_ckpt = sorted(ckpt_epochs)[-1][1]
        logger.info(f"🔁 Phát hiện checkpoint: {latest_ckpt}")

    # ---- 3️⃣ TẢI DATASET ----
    logger.info("📥 Đang tải dataset tiếng Việt (streaming)...")
    dataset_id = "VTSNLP/vietnamese_curated_dataset"

    # ---- 4️⃣ KHỞI TẠO HOẶC TẢI MÔ HÌNH ----
    if latest_ckpt:
        model = Word2Vec.load(latest_ckpt)
        start_epoch = int(re.findall(r"epoch_(\d+)", latest_ckpt)[0]) + 1
        logger.info(f"🔄 Tiếp tục huấn luyện từ epoch {start_epoch}")
        # === BỔ SUNG: Ghi đè tham số workers sau khi tải checkpoint ===
        if model.workers != WORKERS:
            TARGET_WORKERS = WORKERS
            model.workers = TARGET_WORKERS
            logger.info(f"⚡ Đã ghi đè workers từ {model.workers} cũ sang {TARGET_WORKERS} mới để tối ưu hiệu năng.")
    else:
        model = Word2Vec(vector_size=300, window=5, min_count=10, sg=1, workers=12)
        start_epoch = 1
        logger.info("✨ Khởi tạo mô hình mới.")

        # --- build vocab ---
        ds_stream_vocab = load_dataset(dataset_id, split="train")
        logger.info("🔨 Đang xây vocabulary...")
        model.build_vocab(
            tqdm(tokenize_stream(ds_stream_vocab, trie, single_stop, max_phrase_len),
                 desc="Xây vocab (streaming)")
        )
        logger.info(f"✅ Đã xây vocab: {len(model.wv.key_to_index):,} từ.")

    # ---- 5️⃣ HUẤN LUYỆN THEO EPOCH ----
    num_epochs = 5
    for epoch in range(start_epoch, num_epochs + 1):
        logger.info(f"🔁 Epoch {epoch}/{num_epochs}")
        ds_stream = load_dataset(dataset_id, split="train")
        model.train(
            tqdm(tokenize_stream(ds_stream, trie, single_stop, max_phrase_len),
                 desc=f"Huấn luyện epoch {epoch}"),
            total_examples=model.corpus_count,
            total_words=model.corpus_total_words,
            epochs=1,
            callbacks=[EpochSaver(CHECKPOINT_PREFIX)]
        )

    # ---- 6️⃣ LƯU MÔ HÌNH CUỐI ----
    model.save(FINAL_MODEL_PATH)
    logger.info(f"💾 Mô hình cuối cùng được lưu tại: {FINAL_MODEL_PATH}")

    # ---- 7️⃣ TEST NHANH ----
    for word in ["đẹp", "ăn", "uống", "trường", "yêu"]:
        if word in model.wv:
            sim = model.wv.most_similar(word, topn=5)
            logger.info(f"Từ gần '{word}': {sim}")
        else:
            logger.info(f"Từ '{word}' chưa đủ tần suất xuất hiện.")

# ==========================
# ▶️ ENTRY POINT
# ==========================
if __name__ == "__main__":
    main()
