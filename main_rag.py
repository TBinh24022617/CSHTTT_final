import os
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv
from google import genai
import chromadb

# 1. NẠP KHÓA XÁC THỰC TỪ TỆP BIẾN MÔI TRƯỜNG (.ENV)
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("Chưa tìm thấy GEMINI_API_KEY trong file .env!")

client = genai.Client(api_key=api_key)
print("Đang khởi động Hệ thống AI Commissioner (Kiến trúc RAGADA)...")

# Tự động dò tìm mô hình khả dụng trên hệ thống
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

print(f"- Model suy luận trung tâm (LLM): {gen_model_name}")

# 2. SSO CATALOG: Định nghĩa phân quyền truy cập theo kiến trúc RAGADA
SSO_CATALOG = {
    "1": {"role": "General Manager (GM)", "permissions": ["propose_trade", "submit_complaint"]},
    "2": {"role": "League Commissioner", "permissions": ["propose_trade", "update_values_catalog", "view_audit_log"]}
}

# 3. DATA CATALOG: Nạp dữ liệu cầu thủ NBA từ tệp CSV
try:
    df_players = pd.read_csv("nba_players.csv")
    print(f"- Đã tải xong Data Catalog ({len(df_players)} cầu thủ NBA).")
except Exception as e:
    print("- Lỗi tải file csv:", e)
    df_players = pd.DataFrame()

# 4. ALGORITHM CATALOG: Thuật toán lọc dữ liệu & tính toán định lượng (Business Logic)
def get_relevant_players_data(query_text):
    """Trích xuất cầu thủ trong lệnh Trade, tính độ lệch Avg_FPTS và ghép Top 15 chuẩn"""
    if df_players.empty:
        return "Không có dữ liệu cầu thủ.", "Không xác định"
    
    query_lower = query_text.lower()
    matched_rows = []
    for _, row in df_players.iterrows():
        player_name = str(row['Player'])
        name_parts = [part.lower() for part in player_name.split() if len(part) > 2]
        if any(part in query_lower for part in name_parts):
            matched_rows.append(row)
            
    matched_df = pd.DataFrame(matched_rows).drop_duplicates() if matched_rows else pd.DataFrame()
    
    # Thuật toán tính toán định lượng tự động (Decision Algorithm Heuristics)
    quant_summary = "Chưa đủ 2 cầu thủ trong danh sách để tính độ lệch trực tiếp."
    if len(matched_df) >= 2:
        max_p = matched_df.iloc[0]
        min_p = matched_df.iloc[-1]
        diff = round(abs(float(max_p['Avg_FPTS']) - float(min_p['Avg_FPTS'])), 1)
        injuries = matched_df[matched_df['Injury_Status'] != 'Healthy']['Player'].tolist()
        inj_note = f"Cảnh báo chấn thương: {', '.join(injuries)}" if injuries else "Tất cả đều Healthy."
        quant_summary = (
            f"Độ lệch điểm số Delta_FPTS = {diff} FPTS/trận "
            f"({max_p['Player']}: {max_p['Avg_FPTS']} vs {min_p['Player']}: {min_p['Avg_FPTS']}). {inj_note}"
        )

    top_df = df_players.head(15)
    combined_df = pd.concat([matched_df, top_df]).drop_duplicates()
    return combined_df.to_string(index=False), quant_summary

# 5. VALUES & BUSINESS OBJECTIVES CATALOG: Nạp bộ luật vào Vector Store (ChromaDB)
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="trade_rules_collection")
use_google_embed = False

def load_values_catalog():
    global use_google_embed
    with open("trade_rules.txt", "r", encoding="utf-8") as f:
        rules_text = f.read()
    chunks = [chunk.strip() for chunk in rules_text.split('\n\n') if chunk.strip()]
    doc_ids = [f"rule_{i}" for i in range(len(chunks))]
    
    if embed_model_name:
        try:
            embeddings_list = []
            for chunk in chunks:
                res = client.models.embed_content(model=embed_model_name, contents=chunk)
                embeddings_list.append(res.embeddings[0].values)
            collection.upsert(embeddings=embeddings_list, documents=chunks, ids=doc_ids)
            use_google_embed = True
            return len(chunks)
        except Exception:
            use_google_embed = False
            
    collection.upsert(documents=chunks, ids=doc_ids)
    return len(chunks)

num_rules = load_values_catalog()
print(f"- Đã nhúng xong Values Catalog ({num_rules} điều khoản) vào ChromaDB.")

# 6. DECISION CATALOG & AUDIT TRAIL: Ghi nhận lịch sử phán quyết và Ticket
AUDIT_FILE = "decision_audit_log.csv"

def log_to_decision_catalog(user_role, action_type, proposal, decision_summary):
    """Lưu vết kiểm toán (Audit Trail) phục vụ truy vết quyết định theo chuẩn RAGADA"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_entry = pd.DataFrame([{
        "Timestamp": timestamp,
        "User_Role": user_role,
        "Action_Type": action_type,
        "Input_Query": proposal,
        "System_Output_Summary": decision_summary.replace("\n", " ")[:180] + "..."
    }])
    if os.path.exists(AUDIT_FILE):
        new_entry.to_csv(AUDIT_FILE, mode='a', header=False, index=False, encoding="utf-8")
    else:
        new_entry.to_csv(AUDIT_FILE, mode='w', header=True, index=False, encoding="utf-8")

# 7. KHỐI RAG TRANSFORM & INTERACTION AGENT: Hàm đánh giá giao dịch trung tâm
def evaluate_trade(trade_proposal, user_role):
    if use_google_embed:
        q_res = client.models.embed_content(model=embed_model_name, contents=trade_proposal)
        results = collection.query(query_embeddings=[q_res.embeddings[0].values], n_results=2)
    else:
        results = collection.query(query_texts=[trade_proposal], n_results=2)
        
    rule_text = "\n\n".join(results['documents'][0]) if results['documents'] else "Không tìm thấy luật liên quan."
    players_context, quant_summary = get_relevant_players_data(trade_proposal)
    
    prompt = f"""
    Nhiệm vụ của bạn là Ủy viên giải đấu NBA Fantasy (AI Commissioner) vận hành theo kiến trúc RAGADA.
    Người dùng đang tương tác có vai trò: {user_role}.
    Bạn phải đánh giá lệnh giao dịch (Trade) hoặc yêu cầu sau đây dựa trên dữ liệu và luật lệ được cung cấp.
    
    1. LỆNH GIAO DỊCH / TÌNH HUỐNG TỪ USER:
    "{trade_proposal}"
    
    2. KẾT QUẢ TÍNH TOÁN TỪ ALGORITHM CATALOG & DATA CATALOG:
    - Tóm tắt định lượng: {quant_summary}
    - Bảng chỉ số cầu thủ:
    {players_context}
    
    3. ĐIỀU KHOẢN TRÍCH XUẤT TỪ VALUES & BUSINESS OBJECTIVES CATALOG (VECTOR DB):
    {rule_text}
    
    Hãy trả lời theo đúng cấu trúc của Decision Catalog:
    - QUYẾT ĐỊNH: (Duyệt / VETO / Tạo Ticket trình Ủy ban giải đấu)
    - TRÍCH DẪN LUẬT: (Trích dẫn điều khoản cụ thể được áp dụng)
    - PHÂN TÍCH DỮ LIỆU & LÝ DO: (Đối chiếu độ lệch Avg_FPTS, tình trạng chấn thương và giải thích lý do)
    """
    
    response = client.models.generate_content(model=gen_model_name, contents=prompt)
    log_to_decision_catalog(user_role, "EVALUATE_TRADE", trade_proposal, response.text)
    return response.text

# 8. AI AGENT CONTROLLER: Vòng lặp điều phối theo phân quyền SSO
print("\n" + "="*60)
print("HỆ THỐNG KIỂM DUYỆT GIAO DỊCH FANTASY (RAGADA ARCHITECTURE)")
print("="*60)

while True:
    print("\n[SSO AUTHENTICATION] Chọn vai trò truy cập:")
    print("  1. Người chơi (General Manager - GM): Đề xuất Trade / Khiếu nại luật")
    print("  2. Quản trị viên (League Commissioner): Cập nhật Values Catalog / Xem Audit Log")
    print("  0. Thoát chương trình (Exit)")
    
    role_choice = input("Chọn quyền (1/2/0): ").strip()
    if role_choice == '0' or role_choice.lower() == 'exit':
        break
    if role_choice not in SSO_CATALOG:
        print("Lựa chọn không hợp lệ, vui lòng chọn lại!")
        continue
        
    current_role = SSO_CATALOG[role_choice]["role"]
    
    if role_choice == "1":
        user_input = input(f"\n[{current_role}] Nhập đề xuất Trade hoặc câu hỏi: ").strip()
        if not user_input:
            continue
        print("\n[AI Commissioner đang chạy Algorithm Catalog & RAG Transform...]")
        try:
            result = evaluate_trade(user_input, current_role)
            print("\n--- BIÊN BẢN PHÁN QUYẾT (DECISION CATALOG) ---")
            print(result)
            print(f"\n[Hệ thống đã tự động ghi vết kiểm toán (Audit Trail) vào tệp {AUDIT_FILE}]")
        except Exception as e:
            print("\n[Lỗi xử lý giao dịch]:", e)
            
    elif role_choice == "2":
        print(f"\n[{current_role}] Chọn tác vụ quản trị:")
        print("  a. Thêm điều khoản mới vào Values Catalog (trade_rules.txt)")
        print("  b. Xem nhật ký kiểm toán quyết định (Audit Trail Log)")
        sub_cmd = input("Chọn (a/b): ").strip().lower()
        
        if sub_cmd == 'a':
            new_rule = input("Nhập nội dung điều khoản mới: ").strip()
            if new_rule:
                with open("trade_rules.txt", "a", encoding="utf-8") as f:
                    f.write(f"\n\n{new_rule}")
                total_chunks = load_values_catalog()
                log_to_decision_catalog(current_role, "UPDATE_POLICY", new_rule, "Đã cập nhật Values Catalog và nhúng lại Vector Store.")
                print(f"-> Đã cập nhật Values Catalog thành công! Tổng số điều khoản hiện tại: {total_chunks}.")
        elif sub_cmd == 'b':
            if os.path.exists(AUDIT_FILE):
                df_audit = pd.read_csv(AUDIT_FILE)
                print("\n--- NHẬT KÝ KIỂM TOÁN HỆ THỐNG (AUDIT TRAIL) ---")
                print(df_audit.to_string(index=False))
            else:
                print("-> Chưa có bản ghi kiểm toán nào.")