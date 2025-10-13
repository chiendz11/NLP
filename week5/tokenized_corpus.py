import re
import os
import unicodedata
import logging
import time
from tqdm import tqdm
from datasets import load_dataset
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from underthesea import word_tokenize

# ==========================
# ⚙️ CẤU HÌNH CHUNG
# ==========================
OUTPUT_DIR = r"D:\Vietnamese_Word2Vec"
os.makedirs(OUTPUT_DIR, exist_ok=True)
# File này sẽ chứa các token, với từ ghép được nối bằng "_"
CORPUS_PATH = os.path.join(OUTPUT_DIR, "vietnamese_corpus_tokenized.txt")

# ==========================
# 🧩 CẤU HÌNH LOGGING
# ==========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s : %(levelname)s : %(message)s"
)
logger = logging.getLogger(__name__)

# ==========================
# 🧹 CLEANING & VALIDATION
# ==========================
_re_non_vn = re.compile(r"[^0-9a-zA-ZÀ-Ỵà-ỵđĐ\s]", re.UNICODE)
_re_multi_space = re.compile(r"\s+")
# Cho phép ký tự '_' để khớp với các token đa từ (multi-word tokens)
_re_valid_vn_token = re.compile(
    r"^[a-zàáảãạăắằẳẵặâấầẩẫậđèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵ_]+$",
    re.UNICODE
)

def normalize_text(text):
    """Chuẩn hoá Unicode và loại bỏ ký tự không hợp lệ."""
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFC", text)
    text = _re_non_vn.sub(" ", text)
    text = _re_multi_space.sub(" ", text)
    return text.strip().lower()

def is_valid_vietnamese_token(word):
    """Kiểm tra token có hợp lệ tiếng Việt, chấp nhận dấu gạch dưới."""
    return bool(_re_valid_vn_token.match(word))

# ==========================
# 🧱 TRIE STOPWORDS (GIỮ NGUYÊN)
# ==========================
def build_trie(phrases):
    """Tạo Trie cho stopwords. Giả định input phrases đã có dấu '_' nếu là từ ghép."""
    trie = {}
    max_len = 0
    single_word_set = set()
    for phr in phrases:
        # Tách phrase bằng khoảng trắng. Ví dụ: "bởi_vì thế_nên" -> ['bởi_vì', 'thế_nên']
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
# 🧠 TOKENIZE + STOPWORD FILTER
# ==========================
def tokenize_and_clean(text, trie, single_stop, max_phrase_len):
    """Tách từ, loại stopword & từ rác, làm việc trực tiếp với token có '_'."""
    text = normalize_text(text)
    if not text:
        return ""
    
    # 1. Tokenize (Output: list of tokens with '_')
    tokenized_str = word_tokenize(text, format="text")
    tokens = [t for t in tokenized_str.split() if is_valid_vietnamese_token(t)]

    if not tokens:
        return ""

    # 2. Lọc Stopword bằng Trie
    filtered = []
    i, n = 0, len(tokens)
    
    while i < n:
        node = trie
        j = i
        steps = 0
        match_len = 0
        
        # Tìm match cho cụm stopword
        while j < n and steps < max_phrase_len:
            tok = tokens[j]
            if not isinstance(node, dict) or tok not in node:
                break
            node = node[tok]
            j += 1
            steps += 1
            if node.get("_end"):
                match_len = j - i
        
        # Lọc và ghi kết quả
        is_stop_phrase = match_len > 0
        is_single_stop_word = match_len == 0 and tokens[i] in single_stop
        
        if not is_stop_phrase and not is_single_stop_word:
            filtered.append(tokens[i])

        i += max(match_len, 1)

    return " ".join(filtered)

# ==========================
# 🔀 MULTIPROCESS TOKENIZATION
# ==========================
def process_batch(batch_texts, trie, single_stop, max_phrase_len, batch_index):
    """Hàm chạy trong tiến trình con để xử lý một batch văn bản."""
    start = time.time()
    results = []
    for text in batch_texts:
        # Gọi hàm xử lý (tokenize_and_clean)
        clean_line = tokenize_and_clean(text, trie, single_stop, max_phrase_len)
        if clean_line:
            results.append(clean_line)
    elapsed = time.time() - start
    logger.info(f"🧩 Batch {batch_index}: {len(batch_texts):,} dòng trong {elapsed:.2f}s ({len(batch_texts)/elapsed:.1f} doc/s)")
    return results

# ==========================
# 🚀 MAIN (STREAMING + MANUAL MULTIPROCESS)
# ==========================
def main():
    # ---- 1️⃣ KIỂM TRA FILE VÀ THIẾT LẬP CHẾ ĐỘ NỐI TIẾP ----
    skip_count = 0
    file_mode = "w" # Mặc định là ghi mới

    if os.path.exists(CORPUS_PATH) and os.path.getsize(CORPUS_PATH) > 0:
        # Đếm số dòng đã xử lý trước đó (Cơ chế Continuation)
        try:
            with open(CORPUS_PATH, "r", encoding="utf-8") as f:
                # Cách đếm dòng hiệu quả hơn cho file lớn
                skip_count = sum(1 for line in f) 
        except Exception as e:
            logger.error(f"❌ Lỗi khi đếm dòng: {e}. Bắt đầu ghi mới (w).")
            # Giữ skip_count = 0 và file_mode = "w"
        
        if skip_count > 0:
            file_mode = "a" # Chuyển sang chế độ nối thêm
            logger.warning(f"⚠️ Phát hiện tiến trình bị gián đoạn. {skip_count:,} dòng đã được xử lý.")
            logger.warning(f"Sẽ tiếp tục xử lý từ mẫu thứ {skip_count + 1} và ghi NỐI THÊM (append) vào file.")
        else:
            # File có size > 0 nhưng đếm dòng = 0 (có thể do lỗi ghi), vẫn ghi mới
            logger.info("File corpus tồn tại nhưng trống hoặc lỗi đếm dòng. Bắt đầu ghi mới (w).")


    # ---- 2️⃣ STOPWORDS ----
    stopword_file = "stopwords.csv"
    try:
        df = pd.read_csv(stopword_file)
    except FileNotFoundError:
        logger.error(f"❌ Lỗi: Không tìm thấy file stopword: {stopword_file}")
        logger.error("Vui lòng đảm bảo file 'stopwords.csv' tồn tại trong cùng thư mục.")
        return
    
    stop_phrases = [str(w).strip() for w in df["stopwords"].dropna()]
    trie, single_stop, max_phrase_len = build_trie(stop_phrases)
    logger.info(f"✅ Đã tải {len(stop_phrases)} cụm stopword.")

    # ---- 3️⃣ DATASET STREAMING ----
    dataset_id = "VTSNLP/vietnamese_curated_dataset"
    # Dùng Streaming để giữ RAM an toàn (BẮT BUỘC để tránh crash)
    dataset = load_dataset(dataset_id, split="train", streaming=True)
    logger.info("📡 Đang dùng chế độ STREAMING (RAM an toàn).")

    # BỎ QUA CÁC DÒNG ĐÃ XỬ LÝ NẾU ĐANG TIẾP TỤC
    if skip_count > 0:
        logger.info(f"⏭️ Bỏ qua {skip_count:,} mẫu đã xử lý trước đó.")
        dataset = dataset.skip(skip_count)
        
    # ---- 4️⃣ TOKENIZE SONG SONG ----
    num_workers = max(1, os.cpu_count() - 2) 
    batch_size = 15000
    
    logger.info(f"⚡ Sử dụng {num_workers} tiến trình. Batch size: {batch_size}")

    start_time = time.time()
    processed_count = skip_count # Khởi tạo processed_count bằng số dòng đã skip
    batch_buffer = []
    batch_index = 0 # Không reset batch_index để log vẫn chạy tuần tự
    max_futures_limit = num_workers + 1

    with tqdm(total=None, desc="Tổng tiến độ (dòng/s)", unit=" lines") as pbar: 
        # Cập nhật pbar để phản ánh tiến độ đã xử lý
        pbar.n = processed_count 
        pbar.refresh()
        
        # Mở file ở chế độ đã xác định ('w' hoặc 'a')
        with open(CORPUS_PATH, file_mode, encoding="utf-8") as out_f:
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = set()
                try:
                    # Đọc từng mẫu từ stream và GỬI TÁC VỤ
                    for sample in dataset:
                        text = sample.get("text", "")
                        if not text:
                            continue
                        batch_buffer.append(text)
                        
                        if len(batch_buffer) >= batch_size:
                            batch_index += 1
                            future = executor.submit(process_batch, batch_buffer, trie, single_stop, max_phrase_len, batch_index)
                            futures.add(future)
                            batch_buffer = []

                            # THU THẬP KẾT QUẢ
                            if len(futures) > max_futures_limit:
                                done = next(as_completed(futures))
                                futures.remove(done)
                                
                                results = done.result()
                                # Ghi kết quả vào file đã mở ở chế độ 'a'
                                out_f.write("\n".join(results) + "\n") 
                                count = len(results)
                                processed_count += count
                                pbar.update(count)

                                elapsed = time.time() - start_time
                                logger.info(f"✅ TỔNG: {processed_count:,} dòng đã xử lý ({processed_count/elapsed:.1f} doc/s)")
                                
                except Exception as e:
                    logger.error(f"❌ Lỗi xảy ra trong quá trình đọc stream hoặc xử lý: {e}")
                    
                # ---- 5️⃣ GIAI ĐOẠN HOÀN TẤT ----
                
                # 5.1 Xử lý batch cuối (nếu có)
                if batch_buffer:
                    batch_index += 1
                    future = executor.submit(process_batch, batch_buffer, trie, single_stop, max_phrase_len, batch_index)
                    futures.add(future)

                pbar.set_description(f"Chờ {len(futures)} Worker cuối") 

                # 5.2 Chờ tất cả futures còn lại hoàn thành
                for future in as_completed(futures):
                     try:
                        results = future.result()
                        out_f.write("\n".join(results) + "\n")
                        count = len(results)
                        processed_count += count
                        pbar.update(count)
                     except Exception as worker_e:
                        logger.error(f"❌ Lỗi Worker: {worker_e}")


    elapsed = time.time() - start_time
    logger.info(f"✅ Hoàn tất! Tokenized {processed_count:,} dòng trong {elapsed:.2f}s "
                f"({processed_count/elapsed:.1f} doc/s)")
    logger.info(f"💾 File corpus được lưu tại: {CORPUS_PATH}")

# ==========================
# ▶️ ENTRY POINT
# ==========================
if __name__ == "__main__":
    main()
