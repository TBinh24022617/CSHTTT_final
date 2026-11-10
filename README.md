```markdown
# 🏀 AI Commissioner: Hệ thống Kiểm duyệt Giao dịch NBA Fantasy

**Đồ án Môn học:** Cơ sở Hệ thống Thông tin  
**Chủ đề:** Áp dụng kiến trúc *Bridging Human and AI Decision-Making với LLMs (RAGADA)* vào hệ thống tự động hóa quản lý giải đấu thể thao.

## 1. Giới thiệu Dự án
Trong các giải đấu NBA Fantasy League, việc kiểm duyệt lệnh giao dịch (Trade) đòi hỏi Ủy viên giải đấu (Commissioner) phải tốn nhiều thời gian phân tích chỉ số và đối chiếu luật lệ để tránh gian lận. 

Dự án này xây dựng một **AI Commissioner** tự động, sử dụng Kiến trúc RAGADA để đánh giá, phê duyệt hoặc phủ quyết (VETO) các lệnh giao dịch một cách minh bạch, công bằng dựa trên luật lệ giải đấu và chỉ số thực tế của cầu thủ.

## 2. Kiến trúc Hệ thống (The RAGADA Approach)
Hệ thống vận hành dựa trên 3 thành phần cốt lõi của RAGADA:
* **Data Catalog:** Dữ liệu chỉ số trung bình (Avg_FPTS) và tình trạng chấn thương của các cầu thủ NBA, trích xuất từ Sleeper API và tiền xử lý lưu tại `nba_players.csv`.
* **Values Catalog:** Bộ luật và hiến chương của giải đấu (Trade Policy), quy định rõ các hành vi bị cấm (như Lopsided Trade, The Ilgauskas Rule), lưu tại `trade_rules.txt`.
* **Decision Agent (LLM):** Sử dụng Google Gemini kết hợp Vector Database để truy xuất luật (RAG), đối chiếu dữ liệu cầu thủ và đưa ra phán quyết cuối cùng kèm lời giải thích chi tiết.

## 3. Công nghệ Sử dụng
* **Ngôn ngữ:** Python 3.12
* **Mô hình Trí tuệ Nhân tạo:** Google Gemini 2.0 Flash
* **Mô hình Nhúng (Embedding):** text-embedding-004
* **Vector Database:** ChromaDB (Local Search)
* **Xử lý Dữ liệu:** Pandas, python-dotenv

## 4. Hướng dẫn Cài đặt
**Bước 1: Clone kho lưu trữ về máy tính**
```bash
git clone [https://github.com/TBinh24022617/CSHTTT_final.git](https://github.com/TBinh24022617/CSHTTT_final.git)
cd CSHTTT_final

```

**Bước 2: Cài đặt các thư viện cần thiết**

```bash
pip install pandas google-genai chromadb python-dotenv

```

**Bước 3: Cấu hình biến môi trường**
Tạo file `.env` tại thư mục gốc và cấu hình API Key của Google Gemini:

```env
GEMINI_API_KEY=your_api_key_here

```

**Bước 4: Khởi chạy hệ thống**

```bash
python main_rag.py

```

## 5. Cấu trúc Thư mục

```text
CSHTTT_final/
├── .env                  # Biến môi trường chứa API Key (bỏ qua khi push)
├── .gitignore            # Cấu hình ẩn file bảo mật
├── get_sleeper_data.py   # Script gọi API Sleeper lấy dữ liệu gốc
├── nba_players.csv       # Data Catalog: Dữ liệu cầu thủ
├── trade_rules.txt       # Values Catalog: Luật lệ giải đấu
├── main_rag.py           # Core Script chạy AI Commissioner
└── README.md             # Tài liệu mô tả dự án

```

## 6. Thông tin Sinh viên

* **Sinh viên thực hiện:** Nguyễn Đình Thanh Bình
* **Khóa:** K69
* **Chuyên ngành:** Hệ thống thông tin (Information Systems)
* **Đơn vị:** Trường Đại học Công nghệ - Đại học Quốc gia Hà Nội (UET - VNU)

