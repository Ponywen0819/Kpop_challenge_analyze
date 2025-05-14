import pandas as pd

# 讀取 CSV 文件
df = pd.read_csv('data/collaboration_videos.csv')

# 顯示列名
print("CSV 文件的列名：")
print(df.columns.tolist())

# 顯示前幾行數據
print("\n前 5 行數據：")
print(df.head()) 