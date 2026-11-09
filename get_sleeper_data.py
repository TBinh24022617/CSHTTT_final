import requests
import pandas as pd
import random

print("Đang tải dữ liệu từ Sleeper API...")
# Gọi API public của Sleeper (Không cần xác thực)
url = "https://api.sleeper.app/v1/players/nba"
response = requests.get(url)
players_data = response.json()

nba_players = []

# Duyệt qua danh sách cầu thủ và bóc tách dữ liệu
for player_id, info in players_data.items():
    # Chỉ lấy cầu thủ đang thi đấu và có đội
    if info.get('active') and info.get('team'):
        # Tạo điểm FPTS giả lập (Mock data) để chạy thuật toán Trade
        # Trong thực tế phần này sẽ gọi thêm API league của ông, nhưng để demo KLTN thì giả lập là đủ
        mock_fpts = round(random.uniform(15.0, 60.0), 1) 
        
        nba_players.append({
            'Player': f"{info.get('first_name')} {info.get('last_name')}",
            'Team': info.get('team'),
            'Position': ",".join(info.get('fantasy_positions') or []),
            'Injury_Status': info.get('injury_status') or 'Healthy',
            'Avg_FPTS': mock_fpts
        })

# Chuyển thành DataFrame, sắp xếp theo điểm giảm dần và xuất ra file CSV
df = pd.DataFrame(nba_players)
df = df.sort_values(by='Avg_FPTS', ascending=False)
df.head(100).to_csv('nba_players.csv', index=False, encoding='utf-8-sig')

print("Đã lấy xong dữ liệu! File 'nba_players.csv' đã xuất hiện trong thư mục của bạn.")