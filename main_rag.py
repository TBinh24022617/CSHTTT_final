import os
import pandas as pd
from dotenv import load_dotenv
from google import genai
import chromadb

# 1. NẠP API KEY TỪ FILE .ENV (Bảo mật khi đẩy lên GitHub)
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("Chưa tìm thấy GEMINI_API_KEY trong file .env!")

client = genai.Client(api_key=api_key)

print("Đang khởi động AI Commissioner (RAGADA)...")

# Tự động dò tìm model khả dụng
embed_model_name = None
gen_model_name = "gemini-2.0-flash"

try:
    for m in client.models.list():
        actions = getattr(m, 'supported_actions', []) or []
        if 'embedContent' in actions and not embed_model_name:
            embed_model_name = m.name
        if 'generateContent' in actions and ('flash' in m.name.lower()):
            gen_model_name = m.name
except Exception:
    pass

print(f"- Model hội thoại sử dụng: {gen_model_name}")

# 2. DATA CATALOG: Nạp toàn bộ dữ liệu cầu thủ
try:
    df_players = pd.read_csv("nba_players.csv")
    print(f"- Đã tải xong Data Catalog ({len(df_players)} cầu thủ NBA).")
except Exception as e:
    print("- Lỗi tải file csv:", e)
    df_players = pd.DataFrame()

def get_relevant_players_data(query_text):
    """Trích xuất cầu thủ được nhắc tên trong câu hỏi kèm Top 15 cầu thủ dẫn đầu"""
    if df_players.empty:
        return "Không có dữ liệu cầu thủ."
    
    query_lower = query_text.lower()
    matched_rows = []
    for _, row in df_players.iterrows():
        player_name = str(row['Player'])
        # Kiểm tra nếu họ hoặc tên cầu thủ xuất hiện trong câu lệnh
        name_parts = [part.lower() for part in player_name.split() if len(part) > 2]
        if any(part in query_lower for part in name_parts):
            matched_rows.append(row)
            
    matched_df = pd.DataFrame(matched_rows) if matched_rows else pd.DataFrame()
    top_df = df_players.head(15)
    combined_df = pd.concat([matched_df, top_df]).drop_duplicates()
    return combined_df.to_string(index=False)

# 3. VALUES CATALOG: Nạp bộ luật Trade vào Vector Database
print("- Đang nhúng bộ luật vào Vector Database...")

with open("trade_rules.txt", "r", encoding="utf-8") as f:
    rules_text = f.read()

chunks = [chunk.strip() for chunk in rules_text.split('\n\n') if chunk.strip()]
doc_ids = [f"rule_{i}" for i in range(len(chunks))]

chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="trade_rules_collection")

use_google_embed = False
if embed_model_name:
    try:
        embeddings_list = []
        for chunk in chunks:
            res = client.models.embed_content(model=embed_model_name, contents=chunk)
            embeddings_list.append(res.embeddings[0].values)
        collection.add(embeddings=embeddings_list, documents=chunks, ids=doc_ids)
        use_google_embed = True
        print(f"- Đã tải xong Values Catalog vào ChromaDB (Dùng {embed_model_name}).")
    except Exception:
        use_google_embed = False

if not use_google_embed:
    collection.add(documents=chunks, ids=doc_ids)
    print("- Đã tải xong Values Catalog (Luật Trade) vào ChromaDB (Local Embeddings).")

# 4. HÀM ĐÁNH GIÁ GIAO DỊCH (RAGADA CORE)
def evaluate_trade(trade_proposal):
    if use_google_embed:
        q_res = client.models.embed_content(model=embed_model_name, contents=trade_proposal)
        results = collection.query(query_embeddings=[q_res.embeddings[0].values], n_results=2)
    else:
        results = collection.query(query_texts=[trade_proposal], n_results=2)
        
    rule_text = "\n\n".join(results['documents'][0]) if results['documents'] else "Không tìm thấy luật liên quan."
    players_context = get_relevant_players_data(trade_proposal)
    
    prompt = f"""
    Nhiệm vụ của bạn là Ủy viên giải đấu NBA Fantasy (Commissioner). 
    Bạn phải đánh giá lệnh giao dịch (Trade) hoặc hành vi nhân sự sau đây có hợp lệ không dựa trên dữ liệu và luật lệ được cung cấp.
    
    1. LỆNH GIAO DỊCH / TÌNH HUỐNG TỪ USER: 
    "{trade_proposal}"
    
    2. DỮ LIỆU CẦU THỦ TRÍCH XUẤT TỪ DATA CATALOG:
    {players_context}
    
    3. LUẬT LỆ GIẢI ĐẤU TRÍCH XUẤT TỪ VALUES CATALOG (VECTOR DB):
    {rule_text}
    
    Hãy trả lời theo cấu trúc:
    - QUYẾT ĐỊNH: (Duyệt / VETO / Vi phạm kỷ luật)
    - TRÍCH DẪN LUẬT: (Trích dẫn điều khoản cụ thể được áp dụng)
    - PHÂN TÍCH DỮ LIỆU & LÝ DO: (Đối chiếu chỉ số Avg_FPTS, tình trạng chấn thương từ Data Catalog và giải thích lý do ra quyết định)
    """
    
    response = client.models.generate_content(
        model=gen_model_name,
        contents=prompt,
    )
    return response.text

# 5. CHẠY THỬ NGHIỆM
print("\n" + "="*55)
print("HỆ THỐNG KIỂM DUYỆT GIAO DỊCH FANTASY (RAGADA FRAMEWORK)")
print("="*55)

while True:
    user_input = input("\nNhập lệnh Trade bạn muốn đệ trình (hoặc gõ 'exit' để thoát): ").strip()
    if not user_input:
        continue
    if user_input.lower() == 'exit':
        break
    
    print("\n[AI Commissioner đang đối chiếu Data Catalog & Values Catalog...]")
    try:
        result = evaluate_trade(user_input)
        print("\n--- KẾT QUẢ ĐÁNH GIÁ ---")
        print(result)
    except Exception as e:
        print("\n[Lỗi xử lý giao dịch]:", e)