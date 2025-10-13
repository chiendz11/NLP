import os
import logging
import time
from gensim.models import Word2Vec
import numpy as np

# ==========================
# ⚙️ CẤU HÌNH
# ==========================
OUTPUT_DIR = r"D:\Vietnamese_Word2Vec"
# Tên file mô hình Gensim Word2Vec đã huấn luyện (BIN file)
MODEL_FILENAME = "word2vec_model_epoch_2.bin" 
MODEL_PATH = os.path.join(OUTPUT_DIR, MODEL_FILENAME)

# File chứa các vector nhúng (NPY file) - Cần thiết nếu mô hình không chứa sẵn WV
VECTORS_FILENAME = "word2vec_model_epoch_2bin.wv.vectors.npy" 
VECTORS_PATH = os.path.join(OUTPUT_DIR, VECTORS_FILENAME)

# ==========================
# 🧩 CẤU HÌNH LOGGING
# ==========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s : %(levelname)s : %(message)s"
)
logger = logging.getLogger(__name__)

# ==========================
# 🚀 HÀM KIỂM TRA
# ==========================
def run_tests(model):
    """Thực hiện các bài kiểm tra cơ bản về chất lượng vector sử dụng Gensim."""
    
    # Lấy KeyedVectors (WV) từ mô hình Gensim để thực hiện các phép toán vector
    wv = model.wv 
    
    # -------------------------------------------
    # 1. TÌM TỪ GẦN NHẤT (Most Similar)
    # Kiểm tra khả năng bắt được mối quan hệ ngữ nghĩa (Semantic)
    # -------------------------------------------
    print("\n" + "="*50)
    print("1. TÌM TỪ GẦN NHẤT (MOST SIMILAR)")
    print("="*50)
    
    test_words = [
        "hà_nội",    # Thủ đô
        "nhà_khoa_học", # Nghề nghiệp
        "vui_vẻ",     # Tính từ, trạng thái
        "bitcoin"     # Thuật ngữ hiện đại
    ]
    
    for word in test_words:
        try:
            # Gensim dùng wv.most_similar
            neighbors = wv.most_similar(word, topn=10)
            print(f"Từ gốc: '{word}'")
            # In các từ gần nhất, làm tròn độ tương đồng 
            results = [f"({score:.4f}) {neighbor}" for neighbor, score in neighbors]
            print(" -> Gần nhất: " + ", ".join(results))
        except KeyError:
            print(f"Lỗi: Từ '{word}' không có trong từ điển (vocabulary) của mô hình.")
        except Exception as e:
             print(f"Lỗi khi kiểm tra từ '{word}': {e}")
             
    # -------------------------------------------
    # 2. PHÉP TOÁN VECTOR (Word Analogy)
    # Kiểm tra khả năng bắt được mối quan hệ logic (Syntactic/Analogical)
    # Dạng: A là B như C là ? (VD: Vua - Đàn_ông + Đàn_bà = Hoàng_hậu)
    # Gensim: most_similar(positive=[C, A], negative=[B]) = D (D ≈ C - B + A)
    # -------------------------------------------
    print("\n" + "="*50)
    print("2. PHÉP TOÁN VECTOR (WORD ANALOGY: A - B + C = ?)")
    print("="*50)

    # Cấu trúc: (A_positive, B_negative, C_positive) => tìm D (D ≈ A - B + C)
    analogy_tests = [
        ("nhật_bản", "hà_nội", "việt_nam"), # Nước - Thủ đô: Nhật_bản - Hà_nội + Việt_Nam = ? (Đáp án mong đợi: Tokyo)
        ("đã_ăn", "ăn", "đi"),              # Quá khứ - Hiện tại: Đã_ăn - Ăn + Đi = ? (Đáp án mong đợi: Đã_đi)
        ("nhanh_chóng", "nhanh", "to_lớn"), # Tính từ - Trạng từ: Nhanh_chóng - Nhanh + To_lớn = ? (Đáp án mong đợi: To_lớn_một_cách_nhanh_chóng)
    ]
    
    for A, B, C in analogy_tests:
        try:
            # Gensim: most_similar(positive=[A, C], negative=[B])
            # Chúng ta đang tìm D ≈ A - B + C
            analogy_result = wv.most_similar(
                positive=[A, C], 
                negative=[B], 
                topn=5
            )
            
            # Làm tròn độ tương đồng
            results = [f"({score:.4f}) {neighbor}" for neighbor, score in analogy_result]
            print(f"Phép toán: '{A}' - '{B}' + '{C}' = ?")
            print(" -> Kết quả: " + ", ".join(results))
        except KeyError as e:
            print(f"Lỗi: Một trong các từ ({A}, {B}, {C}) không có trong từ điển.")
        except Exception as e:
            print(f"Lỗi khi thực hiện phép toán: {e}")

    # -------------------------------------------
    # 3. ĐO KHOẢNG CÁCH (Dissimilarity)
    # Kiểm tra khả năng nhận diện từ khác biệt (Odd one out)
    # -------------------------------------------
    print("\n" + "="*50)
    print("3. TỪ KHÁC BIỆT (ODD ONE OUT)")
    print("="*50)

    test_lists = [
        ["hồ_chí_minh", "đà_nẵng", "phở", "hải_phòng"], # phở là danh từ khác loại
        ["bóng_đá", "bơi", "nhảy_múa", "quần_vợt"],    # nhảy_múa là loại hình nghệ thuật
        ["bàn_phím", "màn_hình", "cái_ghế", "chuột"]     # cái_ghế là vật dụng không phải thiết bị máy tính
    ]
    
    # Gensim có sẵn hàm wv.doesnt_match
    for words in test_lists:
        try:
            result = wv.doesnt_match(words)
            print(f"Dãy từ: {words}")
            print(f" -> Từ khác biệt nhất (Odd One Out): '{result}'")
        except KeyError as e:
            print(f"Lỗi: Một hoặc nhiều từ trong dãy {words} không có trong từ điển.")
        except Exception as e:
            print(f"Lỗi khi kiểm tra Odd One Out: {e}")


    # -------------------------------------------
    # 4. TÌM TỪ TỪ VECTOR (Vector to Word)
    # Lưu ý: Gensim Word2Vec KHÔNG hỗ trợ OOV (Out-Of-Vocabulary) 
    # như FastText (FastText dùng Subword), nên phép thử này chỉ mang tính tham khảo.
    # Ta sẽ test hàm tương đồng vector thô.
    # -------------------------------------------
    print("\n" + "="*50)
    print("4. TÌM TỪ GẦN NHẤT VỚI VECTOR")
    print("="*50)
    
    # Lấy vector của 3 từ và tính vector trung bình
    words_to_average = ["ô_tô", "xe_máy", "xe_đạp"]
    
    try:
        # Kiểm tra xem tất cả các từ có tồn tại không
        if not all(word in wv.key_to_index for word in words_to_average):
            raise KeyError("Một trong các từ không có trong từ điển.")
            
        # Tính vector trung bình (đại diện cho khái niệm 'phương tiện di chuyển')
        average_vector = np.mean([wv.get_vector(word) for word in words_to_average], axis=0)
        
        # Tìm các từ gần nhất với vector trung bình này
        # Gensim: most_similar(positive=[vector])
        nearest_from_vector = wv.most_similar(
            positive=[average_vector], 
            topn=5
        )
        
        results = [f"({score:.4f}) {neighbor}" for neighbor, score in nearest_from_vector]
        print(f"Vector trung bình của: {words_to_average}")
        print(" -> Gần nhất: " + ", ".join(results))

    except KeyError as e:
        print(f"❌ Lỗi: Một trong các từ không có trong từ điển. Không thể tính vector trung bình.")
    except Exception as e:
        print(f"❌ Lỗi khi kiểm tra vector trung bình: {e}")


# ==========================
# ▶️ ENTRY POINT
# ==========================
if __name__ == "__main__":
    
    # 🚨 Đảm bảo bạn đã cài đặt thư viện Gensim: pip install gensim numpy
    
    if not os.path.exists(MODEL_PATH):
        logger.error(f"❌ Lỗi: Không tìm thấy file mô hình (.bin) tại: {MODEL_PATH}")
        logger.error("Vui lòng kiểm tra lại tên file và đường dẫn.")
    else:
        logger.info(f"💾 Đang tải mô hình Gensim Word2Vec từ: {MODEL_PATH}")
        try:
            # Tải mô hình Gensim (mode cũ)
            # Lưu ý: Gensim chỉ có thể tải file .bin nếu nó được lưu theo định dạng Word2Vec cũ (không phải định dạng Gensim mới)
            # Nếu gặp lỗi, hãy thử tải theo định dạng Gensim mới (Word2Vec.load)
            from gensim.models.keyedvectors import KeyedVectors
            
            # Tải mô hình Word2Vec (Dùng KeyedVectors.load_word2vec_format cho file .bin kiểu cũ)
            # Tuy nhiên, nếu bạn lưu bằng Gensim, bạn nên dùng Word2Vec.load()
            
            try:
                # Thử tải theo cách Gensim lưu trữ
                model = Word2Vec.load(MODEL_PATH)
            except Exception:
                # Thử tải theo định dạng nhị phân Word2Vec cũ (nếu đó là định dạng file bạn lưu)
                # Dựa trên tên file, có vẻ bạn lưu theo kiểu Word2Vec.
                logger.warning("Thử tải mô hình theo định dạng nhị phân Word2Vec KeyedVectors...")
                model = KeyedVectors.load_word2vec_format(MODEL_PATH, binary=True)
                
            
            # Nếu mô hình được tải dưới dạng KeyedVectors (chỉ chứa vector),
            # ta wrap nó lại để sử dụng hàm doesnt_match và most_similar đầy đủ.
            class MinimalWord2VecModel:
                def __init__(self, wv):
                    self.wv = wv
            
            if isinstance(model, KeyedVectors):
                 ft_model = MinimalWord2VecModel(model)
            else:
                 ft_model = model
                 
            logger.info(f"✅ Tải mô hình thành công. Tổng số từ: {len(ft_model.wv.key_to_index)}")
            run_tests(ft_model)
        except Exception as e:
            logger.error(f"❌ Lỗi khi tải mô hình Gensim Word2Vec: {e}")
            logger.error("Đảm bảo đã cài đặt Gensim (pip install gensim) và định dạng file mô hình là chính xác.")
