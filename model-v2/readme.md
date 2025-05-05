# Model v2

## 主要機制

1. time window - 使用 60 天的時間窗口來分析數據
2. 合作網路 - 基於偶像之間的互動關係建立有向圖
3. 多維度推薦系統 - 結合 Personalized PageRank、Jaccard 係數和最短路徑的綜合推薦算法

## 數據處理流程

1. 數據載入

   - 從 `data/collaboration_videos.json` 載入已清洗的合作視頻數據
   - 數據格式包含：source（發起人）、target（合作人）、timestamp、views、likes、comments 等字段

2. 時間窗口處理

   - 設定起始日期（2025 年 1 月 1 日）
   - 對每個視頻，取前 60 天的數據作為分析窗口
   - 使用 `check_should_be_ignored` 函數過濾超出時間窗口的數據

3. 合作網路構建

   - 使用 NetworkX 建立有向圖
   - 節點表示偶像（格式：團體名\_成員名）
   - 邊的權重基於合作效果（effectiveness）：
     - 觀看數權重：50%
     - 點讚數權重：30%
     - 評論數權重：20%
   - 邊的距離（weight）為效果值的倒數

## 推薦系統

1. 網路分析算法

   - Personalized PageRank：

     - 以目標偶像為中心進行 PageRank 計算
     - 使用 effectiveness 作為邊的權重
     - alpha 參數設為 0.85

   - Jaccard 係數：

     - 計算出邊鄰居的相似度
     - 考慮共同鄰居的數量

   - 最短路徑：
     - 使用 Dijkstra 算法計算節點間的最短路徑
     - 考慮邊的權重（距離）

2. 綜合評分機制

   - 對每個算法取前 N 個候選人（N 由 threshold 參數控制）
   - 取三個算法的交集作為最終候選人
   - 對候選人進行綜合評分：
     - PageRank 分數 \* 100
     - Jaccard 分數 \* 100
     - 最短路徑分數（1/路徑長度）
   - 最終分數為三個分數的平均值

## 輸出結果

- 生成包含以下信息的 CSV 文件：
  - date：視頻發布日期
  - initiator：發起人（團體名\_成員名）
  - collaborator：合作人（團體名\_成員名）
  - recommend_idol：推薦偶像列表（以逗號分隔）
  - video_id：視頻 ID

## 使用說明

1. 確保數據文件位於正確路徑：

   - `/data/collaboration_videos.json`

2. 運行 `SMA_v2_timewindow.py` 腳本

3. 結果將保存在 `./predictions_60_YYYY-MM-DD_HH-MM-SS.csv` 文件中
   - 文件名包含時間窗口大小（60 天）和生成時間戳
